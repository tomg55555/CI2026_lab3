"""Spawn-safe batch execution, live progress, and fingerprinted checkpoints.

Scheduling is kept outside timed search regions. Measured parallel execution is
opt-in: CPU contention changes wall-time comparisons even on separate processes.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from hashlib import sha256
import json
import multiprocessing
import os
import platform
from pathlib import Path
from queue import Empty
import sys
from time import perf_counter

_EVENTS = None
_LOCAL_SINK = None


def _initialize_worker(events):
    """Process-pool initializer: route progress events to the shared queue."""
    global _EVENTS
    _EVENTS = events
    events.cancel_join_thread()  # Progress must not block worker shutdown after an error.


def emit_progress(batch_id, done, total, detail=""):
    """Report batch progress to the in-process sink or the worker queue (no-op otherwise)."""
    event = (batch_id, done, total, detail)
    if _LOCAL_SINK is not None:
        _LOCAL_SINK(event)
    elif _EVENTS is not None:
        _EVENTS.put(event)


def worker_count(requested=0):
    """0 means auto: leave a CPU available and cap diagnostic-memory pressure."""
    if requested < 0:
        raise ValueError("Workers must be nonnegative (0 means auto)")
    cpus = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return requested or min(8, max(1, cpus - 1))


class Progress:
    """Throttled console progress, with ETA, across the batches of one phase."""
    def __init__(self, phase, jobs, *, enabled=True, interval=0.5):
        """Plan `jobs` (dicts with `id` and `units`) for phase `phase`."""
        self.phase = phase
        self.enabled = enabled
        self.interval = interval
        self.planned = {j["id"]: j["units"] for j in jobs}
        self.done = dict.fromkeys(self.planned, 0)
        self.started = perf_counter()
        self.last_print = self.started
        self.resumed = 0
        self.completed = set()

    def event(self, event):
        """Record one (batch, done, total, detail) event; print it if the interval elapsed or it completes."""
        batch, done, total, detail = event
        if batch not in self.done:
            return
        if done >= total and batch in self.completed:
            return
        self.done[batch] = max(self.done[batch], min(done, total))
        if done >= total:
            self.completed.add(batch)
        now = perf_counter()
        if not self.enabled or (done < total and now - self.last_print < self.interval):
            return
        self.last_print = now
        finished = sum(self.done.values())
        planned = sum(self.planned.values())
        elapsed = now - self.started
        observed = finished - self.resumed
        remaining = elapsed * (planned - finished) / observed if observed else 0
        print(f"[{self.phase}] batch {batch}: {done}/{total} ({100 * done / total:.1f}%) "
              f"| overall {finished}/{planned} ({100 * finished / planned:.1f}%) "
              f"| elapsed {elapsed:.1f}s ETA {remaining:.1f}s {detail}", flush=True)


class Checkpoints:
    """Atomic per-batch JSON. Refuse a mixed configuration or implementation."""

    def __init__(self, directory, config, *, resume=False):
        """Create or validate the manifest (config + package source fingerprint) of a checkpoint directory."""
        self.directory = Path(directory) if directory is not None else None
        self.resume = resume
        if self.directory is None:
            return
        source = sha256()
        # The whole package: a change in any algorithm module invalidates checkpoints.
        package = Path(__file__).resolve().parent.parent
        for path in sorted(package.rglob("*.py")):
            source.update(path.relative_to(package).as_posix().encode())
            source.update(path.read_bytes())
        signature = json.loads(json.dumps(dict(config=config, implementation_sha256=source.hexdigest(),
                                               python=sys.version, platform=platform.platform(), schema=1)))
        encoded = json.dumps(signature, sort_keys=True)
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = self.directory / "manifest.json"
        if manifest.exists():
            previous = json.loads(manifest.read_text(encoding="utf-8"))
            if previous != signature:
                raise ValueError("Checkpoint configuration/source mismatch; use a new output directory")
            if not resume:
                raise ValueError("Checkpoints already exist; use --resume or a new output directory")
        else:
            if resume:
                raise ValueError("No checkpoint manifest found to resume")
            temporary = manifest.with_suffix(".tmp")
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(manifest)

    def _path(self, batch):
        """Checkpoint file of one batch (hashed batch id)."""
        return self.directory / (sha256(batch.encode()).hexdigest() + ".json")

    def load(self, batch):
        """Saved result of a batch when resuming, else None."""
        if self.directory is None or not self.resume:
            return None
        path = self._path(batch)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("batch") != batch:
            raise ValueError(f"Invalid checkpoint identity for {batch}")
        return payload["result"]

    def save(self, batch, result):
        """Atomically write one batch result."""
        if self.directory is None:
            return
        path = self._path(batch)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(batch=batch, result=result), allow_nan=False), encoding="utf-8")
        temporary.replace(path)


class BatchExecutor:
    """One lazily created process pool, reused across non-overlapping phases."""

    def __init__(self, workers=0, *, progress=True, interval=0.5, checkpoints=None):
        """Resolve workers, progress settings and checkpoints; the process pool starts lazily."""
        self.workers = worker_count(workers)
        self.progress = progress
        self.interval = interval
        self.checkpoints = checkpoints or Checkpoints(None, {})
        self.pool = None
        self.context = multiprocessing.get_context("spawn")
        self.events = None
        self.phase_timings = {}
        self.phase_resumed = {}
        self.phase_batches = {}
        self.resumed_batches = 0

    def __enter__(self):
        """Enter the context and return the executor."""
        return self

    def __exit__(self, kind, value, traceback):
        """Shut down the process pool and the progress queue."""
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        if self.events is not None:
            self.events.close()
            self.events.join_thread()

    def _drain(self, progress):
        """Forward queued worker progress events to `progress`."""
        if self.events is None:
            return
        while True:
            try:
                progress.event(self.events.get_nowait())
            except Empty:
                break

    def run(self, phase, jobs, function, *, parallel=False):
        """Run one phase's jobs, resuming checkpointed ones; serial or parallel; results in job order."""
        global _LOCAL_SINK
        start = perf_counter()
        resumed_before = self.resumed_batches
        progress = Progress(phase, jobs, enabled=self.progress, interval=self.interval)
        results = {}
        pending = []
        for job in jobs:
            loaded = self.checkpoints.load(job["id"])
            if loaded is None:
                pending.append(job)
            else:
                results[job["id"]] = loaded
                self.resumed_batches += 1
                progress.resumed += job["units"]
                progress.event((job["id"], job["units"], job["units"], "resumed"))

        def complete(job, result):
            """Checkpoint a finished job, store its result and report completion."""
            self.checkpoints.save(job["id"], result)
            results[job["id"]] = result
            progress.event((job["id"], job["units"], job["units"], "complete"))

        if parallel and self.workers > 1 and pending:
            if self.pool is None:
                self.events = self.context.Queue()
                self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=self.context,
                                                initializer=_initialize_worker, initargs=(self.events,))
            # Largest declared cost first shortens the pool's tail; output order is unchanged.
            ordered = sorted(pending, key=lambda job: -job.get("cost", 0))
            futures = {self.pool.submit(function, job): job for job in ordered}
            try:
                while futures:
                    ready, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                    self._drain(progress)
                    for future in ready:
                        job = futures.pop(future)
                        try:
                            result = future.result()
                        except Exception as error:
                            raise RuntimeError(f"Batch {job['id']} failed; completed checkpoints can be resumed") from error
                        complete(job, result)
                self._drain(progress)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        else:
            try:
                _LOCAL_SINK = progress.event
                for job in pending:
                    try:
                        complete(job, function(job))
                    except Exception as error:
                        raise RuntimeError(f"Batch {job['id']} failed; completed checkpoints can be resumed") from error
            finally:
                _LOCAL_SINK = None
        self.phase_timings[phase] = perf_counter() - start
        self.phase_resumed[phase] = self.resumed_batches - resumed_before
        self.phase_batches[phase] = len(jobs)
        # Completion order never defines output order or seeds.
        return [results[job["id"]] for job in jobs]
