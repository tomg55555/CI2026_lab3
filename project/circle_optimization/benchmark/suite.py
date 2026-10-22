"""Nested size/exploration experiments; raw data precede report generation."""

import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from importlib.util import find_spec
from math import factorial
import os
from pathlib import Path
from random import Random
from time import perf_counter

from .metrics import (
    anytime_auc, exploration_effects, exploration_trends, quality_bound, summarize, target_rows,
)
from ..classic.diagnostics import SearchDiagnostics
from ..classic.genetic import run_ga
from ..classic.guidance import make_guided_mutation
from ..classic.hill_climbing import run_guided_hill_climber, run_hill_climber, run_inversion_hill_climber
from ..problem import circle_fitness, distance_matrix, exact_best, random_profiles, valid_circle
from .execution import BatchExecutor, Checkpoints, Progress, emit_progress, worker_count


@dataclass(frozen=True)
class SuiteConfig:
    """Every setting of one suite run: sizes, protocols, budgets, families, eras, execution."""
    sizes: tuple[int, ...] = (10, 20, 40, 80, 160)
    characteristics: tuple[int, ...] = (5,)
    explorations: tuple[float, ...] = (0, 0.2, 0.5, 0.8, 1)
    protocols: tuple[str, ...] = ("checks", "time")
    instances: int = 3
    trials: int = 10
    repeats: int = 3
    budget: int = 50_000
    time_limit: float = 0.1
    seed: int = 2027
    population_size: int = 50
    generations: int = 600
    max_restarts: int = 200
    mutation_rate: float = 0.25
    include_ga: bool = True
    diagnostics: bool = True
    exact_limit: int = 9
    targets: tuple[float, ...] = (0.9, 0.95, 0.99)
    workers: int = 0
    timing_mode: str = "isolated"
    progress: bool = True
    progress_interval: float = 0.5
    hc_restart_policy: str = "productive"
    hc_attempts_per_restart: int = 1000
    hc_stall_limit: int = 200
    include_inversion_hc: bool = True
    inversion_rate: float = 0.8
    check_repeats: int = 1
    stop_at_bound: bool = True
    alias_full_exploration: bool = True
    reference: str = "milp"
    reference_time_limit: float = 60.0
    classic: bool = True
    eras: tuple[str, ...] = ()


def validate_config(config):
    """Raise ValueError for inconsistent or unsupported suite settings."""
    if not config.sizes or min(config.sizes) < 4:
        raise ValueError("All child sizes must be at least 4")
    if not config.characteristics or min(config.characteristics) < 1:
        raise ValueError("Characteristics must be positive")
    if not config.explorations or any(not 0 <= x <= 1 for x in config.explorations):
        raise ValueError("Exploration probabilities must lie in [0, 1]")
    if min(config.instances, config.trials, config.repeats, config.check_repeats, config.budget,
           config.population_size, config.max_restarts) < 1 or config.generations < 0:
        raise ValueError("Experiment sizes/counts must be positive; generations nonnegative")
    if not 0 < config.time_limit < float("inf") or not 0 <= config.mutation_rate <= 1:
        raise ValueError("Use a finite positive time limit and a mutation rate in [0, 1]")
    if not config.protocols or any(p not in ("checks", "time") for p in config.protocols):
        raise ValueError("Protocols must be checks and/or time")
    if not config.targets or any(not 0 < t <= 1 for t in config.targets):
        raise ValueError("Target fractions must lie in (0, 1]")
    if not 0 <= config.exact_limit <= 10:
        raise ValueError("Exact limit must be between 0 and 10 (0 disables enumeration)")
    for n in config.sizes:
        pop = min(config.population_size, factorial(n - 1) // 2)
        if config.classic and config.include_ga and "checks" in config.protocols and config.budget < pop:
            raise ValueError(f"Budget must fit the GA population ({pop} at N={n})")
    worker_count(config.workers)
    if config.timing_mode not in ("isolated", "throughput"):
        raise ValueError("Timing mode must be isolated or throughput")
    if not 0 < config.progress_interval < float("inf"):
        raise ValueError("Progress interval must be finite and positive")
    if config.hc_restart_policy not in ("productive", "legacy"):
        raise ValueError("HC restart policy must be productive or legacy")
    if min(config.hc_attempts_per_restart, config.hc_stall_limit) < 1:
        raise ValueError("HC attempt/stall limits must be positive")
    if not 0.5 <= config.inversion_rate < 1:
        raise ValueError("Inversion rate must be in [0.5, 1)")
    if config.reference not in ("milp", "bound"):
        raise ValueError("Reference must be milp or bound")
    if config.reference == "milp" and (find_spec("numpy") is None or find_spec("highspy") is None):
        raise ValueError("--reference milp needs numpy and highspy "
                         "(pip install -r project/requirements-benchmark.txt); --reference bound needs neither")
    if not 0 < config.reference_time_limit < float("inf"):
        raise ValueError("Reference time limit must be finite and positive")
    if not config.classic and not config.eras:
        raise ValueError("Select the classic grid and/or at least one era")
    from ..eras import ERAS
    for name in config.eras:
        if name not in ERAS:
            raise ValueError(f"Unknown era {name!r}; see python -m circle_optimization list")
        if not ERAS[name].available():
            raise ValueError(f"Era {name!r} needs {', '.join(ERAS[name].requires)}")
    if len(set(config.eras)) != len(config.eras):
        raise ValueError("Each era may be listed once")




CLASSIC_FAMILIES = ("HC", "GA", "HC-I")


def variants(config):
    """(family, variant, exploration) triples measured for every paired trial.

    The classic grid (HC/GA uniform and guided at each exploration, HC-I) is
    followed by one ("<era name>", "era", None) entry per `config.eras`.
    """
    result = []
    if config.classic:
        for family in (["HC", "GA"] if config.include_ga else ["HC"]):
            result.append((family, "uniform", None))
            result.extend((family, "guided", x) for x in config.explorations)
        if config.include_inversion_hc:
            result.append(("HC-I", "uniform", None))
    result.extend((name, "era", None) for name in config.eras)
    return result


def repeats_for(config, protocol):
    """Fixed-count outcomes are deterministic; extra checks repeats only re-time them."""
    return config.check_repeats if protocol == "checks" else config.repeats


def aliases(config, protocol):
    """Map each guided exploration=1 variant to the uniform control it duplicates.

    At a count cap, exploration=1 consumes the same random draws as the uniform
    swap control, so its result, trace and diagnostics are copied rather than
    rerun. Deadline-limited outcomes depend on elapsed time and stay measured.
    """
    if protocol != "checks" or not config.alias_full_exploration:
        return {}
    return {(family, variant, alpha): (family, "uniform", None)
            for family, variant, alpha in variants(config) if variant == "guided" and alpha == 1}


def search_bound(config, problem):
    """Exact optima and Hamming bounds are proven: no circle can exceed them."""
    return problem["reference"] if config.stop_at_bound else None


def problem_reference(config, profiles, distances=None):
    """Exact optimum when proven (enumeration and/or MILP), else the best bound.

    Enumeration (N <= exact_limit) and the MILP must agree when both apply.
    A MILP stopped by its time limit still contributes its proven upper bound.
    Without `distances` (era-only suites at large N) the identical Hamming
    bound is computed on the profile aggregation instead of the N x N matrix.
    """
    n = len(profiles)
    if distances is None and n <= config.exact_limit:
        distances = distance_matrix(profiles)
    if distances is not None:
        bound = quality_bound(profiles, distances)
    else:
        from ..exact.profiles import aggregate_profiles, profile_upper_bound
        bound = profile_upper_bound(aggregate_profiles(profiles))
    optimum = exact_best(distances)[1] if n <= config.exact_limit else None
    source = "enumeration" if optimum is not None else None
    details = dict(upper_bound=bound, milp_status=None, milp_score=None, milp_upper_bound=None,
                   milp_proof=None, milp_seconds=None)
    if config.reference == "milp":
        from ..exact.milp import solve_circle
        solved = solve_circle(profiles, time_limit=config.reference_time_limit, seed=config.seed)
        assert valid_circle(solved["circle"], n)
        assert distances is None or circle_fitness(solved["circle"], distances) == solved["score"]
        details.update(milp_status=solved["status"], milp_score=solved["score"],
                       milp_upper_bound=solved["upper_bound"], milp_proof=solved["proof"],
                       milp_seconds=solved["seconds"]["total"])
        if solved["status"] == "optimal":
            if optimum is not None and optimum != solved["score"]:
                raise AssertionError(f"MILP optimum {solved['score']} differs from enumeration {optimum}")
            optimum, source = solved["score"], source or "milp"
        else:
            bound = min(bound, solved["upper_bound"])
    return dict(details, exact_optimum=optimum, exact_source=source,
                reference=optimum if optimum is not None else bound,
                reference_kind="exact_optimum" if optimum is not None else "upper_bound")


@lru_cache(maxsize=8)
def _era_problem(seed, n, c):
    """Per-process `eras.Problem` for one instance; representations cached on it."""
    from ..eras import Problem
    return Problem(random_profiles(seed, n, c))


def run_search(config, distances, family, variant, exploration, seed, protocol,
               *, on_improvement=None, deadline=None, diagnostics=None, warmup=False,
               score_bound=None, instance=None):
    """Run one classic family/variant or one registered era (`instance` = (seed, n, c)).

    Eras run through `eras.run_era`: the checks protocol passes the suite
    budget in the era's own unit; the time protocol only its deadline, which
    lifts the era's count caps. Diagnostics exist only for classic families.
    """
    if variant == "era":
        from ..eras import run_era
        budget = min(config.budget, 200) if warmup else (config.budget if protocol == "checks" else None)
        return run_era(family, _era_problem(*instance), seed=seed, evaluation_budget=budget,
                       deadline=deadline, score_bound=score_bound, on_improvement=on_improvement)
    population = min(config.population_size, factorial(len(distances) - 1) // 2)
    budget = config.budget if protocol == "checks" else 10**12
    if warmup:
        budget = min(budget, max(population, 200))
    if family.startswith("HC"):
        settings = dict(evaluation_budget=budget, on_improvement=on_improvement,
                        deadline=deadline, diagnostics=diagnostics,
                        max_restarts=(2 if warmup else config.max_restarts)
                        if protocol == "checks" else 10**9,
                        restart_policy=config.hc_restart_policy,
                        attempts_per_restart=config.hc_attempts_per_restart,
                        stall_limit=config.hc_stall_limit, score_bound=score_bound)
        if family == "HC-I":
            return run_inversion_hill_climber(distances, Random(seed),
                                              inversion_rate=config.inversion_rate, **settings)
        if variant == "guided":
            return run_guided_hill_climber(distances, Random(seed),
                                           exploration=exploration, **settings)
        return run_hill_climber(distances, Random(seed), **settings)
    options = {}
    if variant == "guided":
        options["mutation_operator"] = make_guided_mutation(
            distances, exploration=exploration, diagnostics=diagnostics,
        )
    return run_ga(
        distances, Random(seed), population_size=population,
        generations=(min(2, config.generations) if warmup else config.generations)
        if protocol == "checks" else 10**9,
        tournament_size=min(10, population), elites=min(2, population),
        crossover_rate=0.9, mutation_rate=config.mutation_rate,
        evaluation_budget=budget, deadline=deadline, diagnostics=diagnostics,
        on_improvement=on_improvement, score_bound=score_bound, **options,
    )


def measured_run(config, distances, family, variant, alpha, seed, protocol, reference,
                 reference_kind, run_id, identity, instance=None):
    """Run one timed search; return its row, improvement trace, target rows and raw result."""
    points = []

    def improvement(score, evaluations):
        """Record a strict improvement with its elapsed time."""
        points.append(dict(score=score, evaluations=evaluations, seconds=perf_counter() - start))

    start = perf_counter()
    result = run_search(config, distances, family, variant, alpha, seed, protocol,
                        on_improvement=improvement,
                        deadline=start + config.time_limit if protocol == "time" else None,
                        score_bound=reference if config.stop_at_bound else None, instance=instance)
    seconds = perf_counter() - start
    if variant != "era":  # run_era validates eras against the profiles itself.
        assert valid_circle(result["circle"], len(distances))
        assert circle_fitness(result["circle"], distances) == result["score"]
    assert result["score"] <= reference
    horizon = config.time_limit if protocol == "time" else seconds
    # Exclude improvements completed after the common time allowance. Atomic
    # operations may overrun it; both their cost and returned score are saved.
    eligible = [p for p in points if p["seconds"] <= horizon]
    best = eligible[-1] if eligible else None
    score = best["score"] if best else None
    scale = identity["children"] * identity["characteristics"]
    times = [0] + [p["seconds"] for p in eligible] + [horizon]
    row = dict(identity, run_id=run_id, protocol=protocol, family=family,
               variant=variant, exploration=alpha, score=score, returned_score=result["score"],
               reference_score=reference, reference_kind=reference_kind,
               normalized_score=score / scale if score is not None else 0.0,
               reference_ratio=(score / reference if reference else 1.0) if best else 0.0,
               relative_gap=(reference - score) / reference if reference and best else (0 if best else 1.0),
               seconds=seconds, seconds_to_best=best["seconds"] if best else None,
               evaluations=result["evaluations"], evaluations_to_best=best["evaluations"] if best else None,
               stop_reason=result["stop_reason"],
               status="ok" if best else "no_score_before_deadline",
               overshoot_seconds=max(0, seconds - horizon) if protocol == "time" else 0,
               auc_time=anytime_auc(eligible, horizon, scale),
               auc_checks=anytime_auc(eligible, config.budget if protocol == "checks" else result["evaluations"],
                                     scale, axis="evaluations"),
               auc_time_horizon=horizon,
               auc_check_horizon=config.budget if protocol == "checks" else result["evaluations"],
               post_best_fraction=1 - best["seconds"] / horizon if best and horizon else None,
               throughput=result["evaluations"] / seconds,
               improvement_count=max(0, len(eligible) - 1),
               largest_stagnation_fraction=max(b - a for a, b in zip(times, times[1:])) / horizon,
               restarts=result.get("restarts"), generations_completed=result.get("generations_completed"))
    if family.startswith("HC"):
        endings = [r["reason"] for r in result.get("restart_history", [])]
        row.update(hc_restart_policy=config.hc_restart_policy,
                   inversion_rate=config.inversion_rate if family == "HC-I" else 0.5,
                   restart_exit_attempt_cap=endings.count("attempt_cap"),
                   restart_exit_stall=endings.count("stall"),
                   restart_exit_evaluation_budget=endings.count("evaluation_budget"),
                   restart_exit_time_limit=endings.count("time_limit"))
    targets = target_rows(row, eligible, config.targets)
    row["target_success"] = targets[config.targets.index(0.95) if 0.95 in config.targets else 0]["hit"]
    return row, points, targets, result


@lru_cache(maxsize=32)
def _matrix(seed, n, c):
    """Cached N x N matrix of one instance (per process)."""
    return distance_matrix(random_profiles(seed, n, c))


_WARMED = set()


def _variant_key(family, variant, alpha):
    """String key of a (family, variant, exploration) triple."""
    return f"{family}|{variant}|{alpha}"


def _measurement_batch(job):
    """Measure every variant of one paired trial (warm-ups first, rotated order); a pool job."""
    config, problem = job["config"], job["problem"]
    n, c, instance_seed = problem["children"], problem["characteristics"], problem["instance_seed"]
    distances = _matrix(instance_seed, n, c) if config.classic else None
    instance = (instance_seed, n, c)
    configurations = variants(config)
    for family, variant, alpha in configurations:
        key = (repr(config), instance_seed, n, c, family, variant, alpha)
        if key not in _WARMED:
            run_search(config, distances, family, variant, alpha, config.seed, "checks", warmup=True,
                       instance=instance)
            _WARMED.add(key)
    aliased = aliases(config, job["protocol"])
    measured = [spec for spec in configurations if spec not in aliased]
    rows, traces, targets, signatures = [], [], [], {}

    def record(spec, row, points, target, signature):
        """Store a row with its trace and targets; checks-protocol repeats must match exactly."""
        rows.append(row)
        traces.extend(dict(p, run_id=row["run_id"]) for p in points)
        targets.extend(target)
        if job["protocol"] == "checks":
            key = _variant_key(*spec)
            if key in signatures:
                assert signatures[key] == signature
            signatures[key] = signature

    done = 0
    for repeat in range(repeats_for(config, job["protocol"])):
        offset = (job["trial"] + repeat) % len(measured)
        for family, variant, alpha in measured[offset:] + measured[:offset]:
            identity = dict(children=n, characteristics=c, instance_seed=instance_seed,
                            trial_seed=job["seed"], repeat=repeat)
            prefix = f"c{c}-n{n}-i{job['instance']}-t{job['trial']}-r{repeat}-{job['protocol']}"
            run_id = f"{prefix}-{family}-{variant}-{alpha}"
            row, points, target, result = measured_run(
                config, distances, family, variant, alpha, job["seed"], job["protocol"],
                problem["reference"], problem["reference_kind"], run_id, identity, instance=instance,
            )
            row.update(worker_pid=os.getpid(), timing_mode=config.timing_mode, aliased_from=None)
            signature = [list(result["circle"]), result["score"], result["evaluations"]]
            record((family, variant, alpha), row, points, target, signature)
            for spec, source in aliased.items():
                if source == (family, variant, alpha):
                    copy = dict(variant=spec[1], exploration=spec[2], run_id=f"{prefix}-{spec[0]}-{spec[1]}-{spec[2]}")
                    record(spec, dict(row, **copy, aliased_from=run_id), points,
                           [dict(t, **copy) for t in target], signature)
            done += 1
            emit_progress(job["id"], done, job["units"], f"{family} {variant} e={alpha} repeat={repeat + 1}")
    return dict(runs=rows, improvements=traces, targets=targets, signatures=signatures)


def _diagnostic_batch(job):
    """Replay one variant of one paired trial; small jobs balance the pool."""
    config, problem = job["config"], job["problem"]
    n, c, instance_seed = problem["children"], problem["characteristics"], problem["instance_seed"]
    distances = _matrix(instance_seed, n, c)
    family, variant, alpha = job["variant"]
    diagnostics = SearchDiagnostics(n, family=family)
    result = run_search(config, distances, family, variant, alpha, job["seed"],
                        "checks", diagnostics=diagnostics, score_bound=search_bound(config, problem))
    assert [list(result["circle"]), result["score"], result["evaluations"]] == job["signature"]
    identity = dict(children=n, characteristics=c, instance_seed=instance_seed,
                    trial_seed=job["seed"], family=family, variant=variant, exploration=alpha)
    emit_progress(job["id"], 1, 1, f"{family} {variant} e={alpha}")
    return dict(diagnostics=[dict(identity, **diagnostics.summary(), diagnostic_id=job["id"])],
                diagnostic_snapshots=[dict(s, **identity, diagnostic_id=job["id"])
                                      for s in diagnostics.snapshots])


def _statistics_batch(job):
    """Summaries, exploration effects and trends of one (C, N, protocol) slice; a pool job."""
    result = {}
    for done, (name, function) in enumerate((
        ("summaries", summarize), ("exploration_effects", exploration_effects),
        ("exploration_trends", exploration_trends),
    ), 1):
        result[name] = [dict(r, characteristics=job["characteristics"]) for r in function(job["rows"])]
        emit_progress(job["id"], done, 3, name)
    return result


def execute_suite(config: SuiteConfig, *, checkpoints=None):
    """Run measurements, diagnostics and statistics for a configuration; return all tables."""
    validate_config(config)
    configurations = variants(config)
    trials = len(config.sizes) * len(config.characteristics) * config.instances * config.trials
    measured_count = trials * sum(repeats_for(config, p) * (len(configurations) - len(aliases(config, p)))
                                  for p in config.protocols)
    copied_count = trials * sum(repeats_for(config, p) * len(aliases(config, p)) for p in config.protocols)
    print(f"Suite: {measured_count} measured searches (+{copied_count} copied exploration=1 rows); "
          f"workers={worker_count(config.workers)}, timing_mode={config.timing_mode}. "
          "Diagnostics/statistics follow all measured searches.", flush=True)
    if config.timing_mode == "throughput":
        print("Concurrent measured runs: wall times/deadline quality include CPU contention; "
              "do not compare these timings to isolated runs.", flush=True)
    started = perf_counter()
    jobs, problems = [], []
    for c in config.characteristics:
        for instance in range(config.instances):
            instance_seed = config.seed + instance
            master = random_profiles(instance_seed, max(config.sizes), c)
            for n in config.sizes:
                start = perf_counter()
                # Era-only suites never build the O(N^2) matrix (large-N runs).
                distances = distance_matrix(master[:n]) if config.classic else None
                matrix_seconds = perf_counter() - start if config.classic else None
                profiles = master[:n]
                reference = problem_reference(config, profiles, distances)
                problem = dict(children=n, characteristics=c, instance_seed=instance_seed,
                               reference=reference["reference"], reference_kind=reference["reference_kind"])
                problems.append(dict(children=n, characteristics=c, instance_seed=instance_seed,
                                     distinct_profiles=len(set(profiles)),
                                     duplicate_profile_fraction=1 - len(set(profiles)) / n,
                                     matrix_seconds=matrix_seconds,
                                     **{k: v for k, v in reference.items() if k != "reference_kind"}))
                for trial in range(config.trials):
                    for protocol in config.protocols:
                        jobs.append(dict(id=f"measure-c{c}-n{n}-i{instance}-t{trial}-{protocol}",
                                         config=config, problem=problem, instance=instance, trial=trial,
                                         seed=config.seed + 1_000_000 + instance * 1_000_000 + trial,
                                         protocol=protocol,
                                         units=(len(configurations) - len(aliases(config, protocol)))
                                         * repeats_for(config, protocol)))
    preparation_seconds = perf_counter() - started
    data = dict(runs=[], improvements=[], targets=[], diagnostics=[], diagnostic_snapshots=[],
                problems=problems, summaries=[], exploration_effects=[], exploration_trends=[])
    with BatchExecutor(config.workers, progress=config.progress, interval=config.progress_interval,
                       checkpoints=checkpoints) as executor:
        batches = executor.run("measurements", jobs, _measurement_batch,
                               parallel=config.timing_mode == "throughput")
        diagnostic_jobs, plan = [], []
        copied = aliases(config, "checks")
        for job, batch in zip(jobs, batches):
            for name in ("runs", "improvements", "targets"):
                data[name].extend(batch[name])
            if job["protocol"] == "checks" and config.diagnostics:
                base = job["id"].replace("measure-", "diagnose-", 1)
                for spec in configurations:
                    if spec[0] not in CLASSIC_FAMILIES:
                        continue  # Eras have no fixed-check diagnostic replay.
                    source = copied.get(spec, spec)
                    plan.append((spec, f"{base}-{_variant_key(*spec)}", f"{base}-{_variant_key(*source)}"))
                    if source == spec:
                        # Cost only orders submission (largest first); it never changes results.
                        diagnostic_jobs.append(dict(job, id=plan[-1][1], units=1, variant=spec,
                                                    signature=batch["signatures"][_variant_key(*spec)],
                                                    cost=job["problem"]["children"] * (4 if spec[0] == "GA" else 1)))
        replays = dict(zip((j["id"] for j in diagnostic_jobs),
                           executor.run("diagnostics", diagnostic_jobs, _diagnostic_batch, parallel=True)))
        for (_, variant, alpha), diagnostic_id, source_id in plan:
            copy = {} if diagnostic_id == source_id else dict(
                variant=variant, exploration=alpha, diagnostic_id=diagnostic_id, aliased_from=source_id)
            data["diagnostics"].extend(dict(r, **copy) for r in replays[source_id]["diagnostics"])
            data["diagnostic_snapshots"].extend(dict(s, **copy) for s in replays[source_id]["diagnostic_snapshots"])
        identity = ("children", "characteristics", "instance_seed", "trial_seed", "family", "variant", "exploration")
        lookup = {tuple(r[k] for k in identity): r for r in data["diagnostics"]}
        for row in data["runs"]:
            if row["protocol"] == "checks":
                summary = lookup.get(tuple(row[k] for k in identity), {})
                row.update({k: v for k, v in summary.items()
                            if k not in identity and k not in ("diagnostic_id", "aliased_from")})
        statistic_jobs = []
        for c in config.characteristics:
            for protocol in config.protocols:
                for n in config.sizes:
                    statistic_jobs.append(dict(id=f"statistics-c{c}-n{n}-{protocol}", units=3,
                                               characteristics=c, rows=[r for r in data["runs"]
                                               if (r["characteristics"], r["protocol"], r["children"]) == (c, protocol, n)]))
        for batch in executor.run("statistics", statistic_jobs, _statistics_batch, parallel=True):
            for name, rows in batch.items():
                data[name].extend(rows)
        data["execution"] = [dict(phase="preparation", seconds=preparation_seconds,
                                  workers=1, resumed_batches=0)] + [
            dict(phase=phase, seconds=seconds,
                 workers=executor.workers if phase != "measurements" or config.timing_mode == "throughput" else 1,
                 batches=executor.phase_batches[phase],
                 executed_batches=executor.phase_batches[phase] - executor.phase_resumed[phase],
                 resumed_batches=executor.phase_resumed[phase]) for phase, seconds in executor.phase_timings.items()]
    return data


def write_csv(path, rows):
    """Write rows to a CSV file whose header is the union of their keys."""
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_and_save(config, output=None, *, report=True, resume=False):
    """Execute a suite and export CSV/JSON (and the HTML report) to `output`; resumable."""
    if report:
        from .report import require_plotly
        require_plotly()  # Fail before an expensive experiment if dependency is missing.
    output = Path(output or f"benchmark-results/{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}")
    if (output / "results.json").exists() and not resume:
        raise ValueError("Completed results already exist; choose a new output directory or --resume compatible checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    signature = {k: v for k, v in asdict(config).items() if k not in ("progress", "progress_interval")}
    checkpoints = Checkpoints(output / ".checkpoints", signature, resume=resume)
    started = perf_counter()
    data = execute_suite(config, checkpoints=checkpoints)
    manifest = dict(config=asdict(config), python=sys.version, platform=platform.platform(),
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=perf_counter() - started,
                    initialization="shared profiles and numeric seeds; algorithm-specific starts",
                    uncertainty="hierarchical bootstrap over instance families and trials; repeats collapsed",
                    timing="matrix excluded; search validation/setup and lightweight improvement callbacks included",
                    stopping=("searches stop on reaching the proven reference (exact optimum or upper bound); "
                              "scores and improvement traces are unchanged" if config.stop_at_bound
                              else "searches run to their count/time limits"),
                    aliasing=("checks-protocol exploration=1 rows copy the identical uniform run (aliased_from)"
                              if config.alias_full_exploration else "every variant is measured"))
    manifest["reference"] = (
        "exact optimum for every N from the profile-aggregated MILP (cross-checked by enumeration "
        "up to exact_limit); its upper bound if the MILP time limit expires" if config.reference == "milp"
        else "exact optimum by enumeration up to exact_limit, else the Hamming upper bound")
    manifest.update(schema=2, algorithm_revision="phase14", timing_mode=config.timing_mode,
                    resolved_workers=worker_count(config.workers), execution=data["execution"],
                    resumed=resume,
                    timing_comparability="isolated" if config.timing_mode == "isolated" else "CPU-contention-sensitive")
    if "plotly" in sys.modules:
        manifest["plotly_version"] = sys.modules["plotly"].__version__
    export_started = perf_counter()
    export_progress = Progress("export", [dict(id="data-files", units=len(data) + 2)],
                               enabled=config.progress, interval=config.progress_interval)
    for index, (name, values) in enumerate(data.items(), 1):
        write_csv(output / f"{name}.csv", values)
        export_progress.event(("data-files", index, len(data) + 2, f"{name}.csv"))
    (output / "configuration.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    export_progress.event(("data-files", len(data) + 1, len(data) + 2, "configuration.json"))
    (output / "results.json").write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    export_progress.event(("data-files", len(data) + 2, len(data) + 2, "results.json"))
    manifest["export_seconds"] = perf_counter() - export_started
    if report:
        from .report import render_report
        report_progress = None

        def report_event(done, total, detail):
            """Forward report-rendering progress to a Progress printer."""
            nonlocal report_progress
            if report_progress is None:
                report_progress = Progress("report", [dict(id="HTML-charts", units=total)],
                                           enabled=config.progress, interval=config.progress_interval)
            report_progress.event(("HTML-charts", done, total, detail))

        report_started = perf_counter()
        render_report(data, manifest, output / "report.html", on_progress=report_event)
        manifest["report_seconds"] = perf_counter() - report_started
    manifest["total_seconds"] = perf_counter() - started
    (output / "configuration.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Results saved to {output.resolve()}")
    return output, data
