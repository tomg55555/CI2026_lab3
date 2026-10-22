"""Scheduling/checkpoint contracts and productive/inversion HC behavior."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from time import sleep
from unittest import TestCase, main
from unittest.mock import patch

from circle_optimization import (
    circle_fitness, distance_matrix, run_hill_climber, run_inversion_hill_climber,
)
from circle_optimization.benchmark.execution import BatchExecutor, Checkpoints, emit_progress
from circle_optimization.benchmark.metrics import _bootstrap_weights, clustered_ci, percentile
from circle_optimization.benchmark.suite import SuiteConfig, execute_suite, run_and_save
from circle_optimization.problem import random_profiles
from circle_optimization.classic.diagnostics import SearchDiagnostics


def worker_echo(job):
    """Top-level function so tests exercise macOS-compatible spawn pickling."""
    for i in range(job["units"]):
        sleep(0.005)
        emit_progress(job["id"], i + 1, job["units"])
    return {"id": job["id"], "draws": Random(job["seed"]).sample(range(100), 5)}


def worker_error(job):
    raise ValueError("deliberate test error")


class ExecutionTests(TestCase):
    def test_spawn_and_serial_results_match_with_live_progress(self):
        # Rising costs make the pool submit in reverse; output order must not change.
        jobs = [dict(id=f"test-{i}", seed=i, units=2, cost=i) for i in range(4)]
        output = StringIO()
        with redirect_stdout(output):
            with BatchExecutor(1, interval=0.001) as serial:
                expected = serial.run("serial", jobs, worker_echo)
            with BatchExecutor(2, interval=0.001) as parallel:
                actual = parallel.run("parallel", jobs, worker_echo, parallel=True)
        self.assertEqual(actual, expected)
        self.assertIn("100.0%", output.getvalue())
        self.assertIn("batch test-", output.getvalue())

    def test_isolated_measurement_does_not_start_a_pool(self):
        jobs = [dict(id="isolated", seed=7, units=1)]
        with patch("circle_optimization.benchmark.execution.ProcessPoolExecutor",
                   side_effect=AssertionError("pool must not start during isolated measurement")):
            with BatchExecutor(4, progress=False) as executor:
                executor.run("measurements", jobs, worker_echo, parallel=False)
                self.assertIsNone(executor.pool)

    def test_checkpoint_resume_and_configuration_mismatch(self):
        jobs = [dict(id=f"resume-{i}", seed=i, units=1) for i in range(3)]
        with TemporaryDirectory() as directory:
            config = dict(sizes=(4, 8), seed=7)
            store = Checkpoints(directory, config)
            with BatchExecutor(1, progress=False, checkpoints=store) as executor:
                first = executor.run("phase", jobs[:1], worker_echo)
            resumed = Checkpoints(directory, config, resume=True)
            with BatchExecutor(1, progress=False, checkpoints=resumed) as executor:
                actual = executor.run("phase", jobs, worker_echo)
                self.assertEqual(executor.resumed_batches, 1)
            self.assertEqual(actual[0], first[0])
            self.assertEqual(len(actual), 3)
            with self.assertRaises(ValueError):
                Checkpoints(directory, dict(sizes=(4, 8), seed=8), resume=True)
            with self.assertRaises(ValueError):
                Checkpoints(directory, config)

    def test_batch_errors_identify_the_failing_batch(self):
        with BatchExecutor(2, progress=False) as executor:
            with self.assertRaisesRegex(RuntimeError, "broken-batch"):
                executor.run("failure", [dict(id="broken-batch", units=1)], worker_error, parallel=True)

    def test_actual_parallel_suite_has_same_count_trajectories_and_diagnostics(self):
        common = dict(sizes=(6,), explorations=(0, 1), protocols=("checks",),
                      instances=1, trials=2, repeats=1, budget=80, generations=3,
                      exact_limit=6, progress=False, include_inversion_hc=True)
        with redirect_stdout(StringIO()):
            serial = execute_suite(SuiteConfig(**common, workers=1))
            parallel = execute_suite(SuiteConfig(**common, workers=2, timing_mode="throughput"))
        def signatures(data):
            return {r["run_id"]: (r["score"], r["evaluations"], r["evaluations_to_best"],
                                   r["unique_scored"], r["edge_coverage"], r["stop_reason"])
                    for r in data["runs"]}
        self.assertEqual(signatures(serial), signatures(parallel))
        def improvements(data):
            return [(r["run_id"], r["score"], r["evaluations"]) for r in data["improvements"]]
        self.assertEqual(improvements(serial), improvements(parallel))

    def test_completed_output_is_not_silently_replaced(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text('{"historical":true}')
            with self.assertRaises(ValueError):
                run_and_save(SuiteConfig(workers=1), directory, report=False)
            self.assertEqual(path.read_text(), '{"historical":true}')

    def test_bootstrap_cache_preserves_original_resampling(self):
        rows = [dict(instance_seed=i, value=v, other=v + 7)
                for i, values in enumerate(((1, 2), (3, 4, 5), (6,))) for v in values]
        groups = [[1, 2], [3, 4, 5], [6]]
        rng = Random(9017)
        estimates = []
        for _ in range(400):
            drawn = [rng.choice(groups) for _ in groups]
            estimates.append(sum(sum(rng.choice(g) for _ in g) / len(g) for g in drawn) / len(drawn))
        actual = clustered_ci(rows, "value")
        for a, b in zip(actual, (percentile(estimates, 0.025), percentile(estimates, 0.975))):
            self.assertAlmostEqual(a, b, places=12)
        before = _bootstrap_weights.cache_info().hits
        clustered_ci(rows, "other")
        self.assertGreater(_bootstrap_weights.cache_info().hits, before)


class RestartTests(TestCase):
    def test_productive_policy_continues_past_legacy_cap(self):
        distances = distance_matrix(random_profiles(2027, 10, 5))
        current = tuple(range(10))
        proposals = []
        # Independent full-scoring oracle supplies a real improving sequence.
        for _ in range(3):
            candidates = []
            for i in range(1, 10):
                for j in range(i + 1, 10):
                    candidate = current[:i] + tuple(reversed(current[i:j + 1])) + current[j + 1:]
                    if circle_fitness(candidate, distances) > circle_fitness(current, distances):
                        candidates.append((candidate, i, j))
            self.assertTrue(candidates)
            current, i, j = min(candidates, key=lambda p: circle_fitness(p[0], distances))
            proposals.append((i, j))
        results = {}
        for policy in ("legacy", "productive"):
            rng = Random(2)
            with patch("circle_optimization.classic.hill_climbing.random_individual", return_value=tuple(range(10))), \
                    patch("circle_optimization.classic.hill_climbing.uniform_swap_positions", side_effect=proposals), \
                    patch.object(rng, "random", return_value=0.9):
                results[policy] = run_hill_climber(
                    distances, rng, evaluation_budget=4, max_restarts=1,
                    attempts_per_restart=1, stall_limit=5, restart_policy=policy,
                )
        self.assertEqual(results["legacy"]["restart_history"][0]["reason"], "attempt_cap")
        self.assertEqual(results["legacy"]["evaluations"], 2)
        self.assertEqual(results["productive"]["evaluations"], 4)
        self.assertGreater(results["productive"]["score"], results["legacy"]["score"])

    def test_legacy_hc_matches_historical_probe(self):
        result = run_hill_climber(distance_matrix(random_profiles(2027, 160, 5)), Random(1002027),
                                 evaluation_budget=20000, restart_policy="legacy")
        self.assertEqual((result["score"], result["evaluations"], result["restarts"]), (610, 20000, 21))

    def test_inversion_version_keeps_swaps_and_uniform_control(self):
        distances = distance_matrix(random_profiles(2027, 10, 5))
        diagnostics = SearchDiagnostics(10, family="HC-I")
        result = run_inversion_hill_climber(distances, Random(11), evaluation_budget=1000,
                                            diagnostics=diagnostics)
        self.assertEqual(result["score"], circle_fitness(result["circle"], distances))
        self.assertGreater(diagnostics.counts["uniform_swap_proposals"], 0)
        self.assertGreater(diagnostics.counts["inversion_proposals"],
                           diagnostics.counts["uniform_swap_proposals"])
        self.assertEqual(
            run_hill_climber(distances, Random(11), evaluation_budget=1000),
            run_inversion_hill_climber(distances, Random(11), inversion_rate=0.5, evaluation_budget=1000),
        )
        with self.assertRaises(ValueError):
            run_inversion_hill_climber(distances, Random(11), inversion_rate=1, evaluation_budget=10)


if __name__ == "__main__":
    main()
