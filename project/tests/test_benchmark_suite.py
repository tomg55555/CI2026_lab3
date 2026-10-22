"""Check experiment pairing, deadline semantics, statistics and offline export."""

from contextlib import redirect_stdout
from html.parser import HTMLParser
from importlib.util import find_spec
from io import StringIO
from math import log10
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

from circle_optimization import (
    circle_fitness, distance_matrix, exact_best, make_guided_mutation, random_profiles,
    run_ga, run_guided_hill_climber, run_hill_climber, valid_circle,
)
from circle_optimization.benchmark.metrics import (
    anytime_auc, collapse_repeats, exploration_effects, exploration_trends,
    first_hit, quality_bound,
)
from circle_optimization.benchmark.suite import (
    SuiteConfig, execute_suite, measured_run, run_and_save, validate_config,
)
from circle_optimization.classic.diagnostics import SearchDiagnostics


class BenchmarkMetricTests(TestCase):
    def test_random_profiles_nest_and_upper_bound(self):
        for n in range(4, 9):
            profiles = random_profiles(19, 8, 5)[:n]
            self.assertEqual(profiles, random_profiles(19, n, 5))
            distances = distance_matrix(profiles)
            self.assertGreaterEqual(quality_bound(profiles, distances), exact_best(distances)[1])
            self.assertLessEqual(quality_bound(profiles, distances), n * 5)

    def test_first_hit_and_exact_step_auc(self):
        points = [dict(seconds=2, evaluations=3, score=2),
                  dict(seconds=5, evaluations=8, score=4)]
        self.assertAlmostEqual(anytime_auc(points, 10, 4), 0.65)
        self.assertAlmostEqual(anytime_auc(points, 4, 4), 0.25)
        self.assertEqual(first_hit(points, 4), 5)
        self.assertEqual(first_hit(points, 4, "evaluations"), 8)
        self.assertIsNone(first_hit(points, 5))

    def test_paired_exploration_effects_and_trends_do_not_count_repeats_as_trials(self):
        rows = []
        for instance in (1, 2):
            for trial in (10, 20):
                for repeat in range(3):
                    for variant, alpha in [("uniform", None), ("guided", 0), ("guided", 0.5), ("guided", 1)]:
                        rows.append(dict(protocol="checks", children=8, family="HC", variant=variant,
                                         exploration=alpha, instance_seed=instance, trial_seed=trial,
                                         repeat=repeat, normalized_score=0.5 + (alpha or 0) * 0.1,
                                         seconds=1 + repeat))
        self.assertEqual(len(collapse_repeats(rows)), 16)
        effects = exploration_effects(rows)
        effect = next(r for r in effects if r["exploration"] == 1
                      and r["comparison"] == "uniform" and r["metric"] == "normalized_score")
        self.assertEqual(effect["pairs"], 4)
        self.assertAlmostEqual(effect["mean_delta"], 0.1)
        self.assertAlmostEqual(effect["ci_low"], 0.1)
        self.assertAlmostEqual(effect["ci_high"], 0.1)
        self.assertEqual(effect["favorable"], 4)
        previous = next(r for r in effects if r["exploration"] == 1
                        and r["comparison"] == "previous_alpha" and r["metric"] == "normalized_score")
        self.assertAlmostEqual(previous["mean_delta"], 0.05)
        trend = next(r for r in exploration_trends(rows) if r["metric"] == "normalized_score")
        self.assertAlmostEqual(trend["mean_slope_per_0_1"], 0.01)
        self.assertAlmostEqual(trend["mean_spearman"], 1)

    def test_edge_diversity_and_indexed_coverage(self):
        diagnostics = SearchDiagnostics(4, family="GA")
        circles = [(0, 1, 2, 3), (0, 1, 3, 2)]
        for circle in circles:
            diagnostics.evaluation(circle)
        diagnostics.proposal(circles[0])
        diagnostics.proposal(circles[0])
        diagnostics.snapshot(2, 0, circles)
        summary = diagnostics.summary()
        self.assertAlmostEqual(summary["edge_diversity"], 0.5)
        self.assertEqual(summary["edge_coverage"], 1)
        self.assertEqual(summary["novelty_rate"], 0.5)
        self.assertAlmostEqual(summary["log10_circle_coverage"], log10(2 / 3))
        self.assertIsNone(summary["swap_acceptance"])


class BenchmarkExecutionTests(TestCase):
    def test_expired_deadlines_return_a_valid_scored_fallback(self):
        distances = distance_matrix(random_profiles(9, 6, 5))
        for search in (run_hill_climber, run_guided_hill_climber):
            with patch("circle_optimization.classic.hill_climbing.perf_counter", return_value=2):
                result = search(distances, Random(8), evaluation_budget=100, deadline=1)
            self.assertTrue(valid_circle(result["circle"], 6))
            self.assertEqual(result["evaluations"], 1)
            self.assertEqual(result["stop_reason"], "time_limit")
        with patch("circle_optimization.classic.genetic.perf_counter", return_value=2), \
                patch("circle_optimization.classic.operators.perf_counter", return_value=2):
            result = run_ga(distances, Random(8), population_size=10, generations=10,
                            tournament_size=3, elites=2, crossover_rate=0.9,
                            mutation_rate=0.25, evaluation_budget=100, deadline=1)
        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(result["stop_reason"], "time_limit")
        self.assertEqual(result["score"], circle_fitness(result["circle"], distances))

    def test_after_deadline_score_is_not_credited(self):
        distances = distance_matrix(((False,),) * 4)
        config = SuiteConfig(sizes=(4,), time_limit=1)

        def fake_search(*args, on_improvement=None, **kwargs):
            on_improvement(0, 1)
            return dict(circle=(0, 1, 2, 3), score=0, evaluations=1, stop_reason="time_limit")

        with patch("circle_optimization.benchmark.suite.run_search", side_effect=fake_search), \
                patch("circle_optimization.benchmark.suite.perf_counter", side_effect=[0, 2, 3]):
            row, _, targets, _ = measured_run(
                config, distances, "HC", "uniform", None, 8, "time", 0, "upper_bound", "run",
                dict(children=4, characteristics=1, instance_seed=1, trial_seed=8, repeat=0),
            )
        self.assertIsNone(row["score"])
        self.assertEqual(row["returned_score"], 0)
        self.assertEqual(row["overshoot_seconds"], 2)
        self.assertEqual(row["status"], "no_score_before_deadline")
        self.assertTrue(all(not target["hit"] for target in targets))

    def test_diagnostics_preserve_seeded_results_in_both_families(self):
        distances = distance_matrix(random_profiles(10, 6, 5))
        for family in ("HC", "GA"):
            for guided in (False, True):
                diagnostic = SearchDiagnostics(6, family=family)
                settings = dict(evaluation_budget=100)
                if family == "HC":
                    search = run_guided_hill_climber if guided else run_hill_climber
                    actual = search(distances, Random(5), diagnostics=diagnostic, **settings)
                    expected = search(distances, Random(5), **settings)
                else:
                    settings.update(population_size=10, generations=5, elites=2,
                                    tournament_size=3, crossover_rate=0.9, mutation_rate=0.5)
                    observed_operator = dict(mutation_operator=make_guided_mutation(
                        distances, diagnostics=diagnostic)) if guided else {}
                    expected_operator = dict(mutation_operator=make_guided_mutation(distances)) if guided else {}
                    actual = run_ga(distances, Random(5), diagnostics=diagnostic, **settings, **observed_operator)
                    expected = run_ga(distances, Random(5), **settings, **expected_operator)
                self.assertEqual(actual, expected)
                self.assertGreater(len(diagnostic.scored), 0)
                self.assertTrue(0 <= diagnostic.summary()["edge_coverage"] <= 1)

    def test_suite_pairing_replay_and_configuration_validation(self):
        # Aliasing off: measure exploration=1 to prove the equivalence aliasing relies on.
        config = SuiteConfig(sizes=(4, 6), explorations=(0, 1), protocols=("checks",),
                             instances=1, trials=1, repeats=2, check_repeats=2, budget=60,
                             generations=3, exact_limit=6, workers=1, include_inversion_hc=False,
                             alias_full_exploration=False)
        with redirect_stdout(StringIO()):
            data = execute_suite(config)
        self.assertEqual(len(data["runs"]), 24)
        self.assertEqual(len(data["diagnostics"]), 12)
        self.assertTrue(all(r["evaluations"] <= 60 for r in data["runs"]))
        self.assertTrue(all(r["score"] <= r["reference_score"] for r in data["runs"]))
        diagnostic_fields = ("unique_scored", "unique_proposals", "edge_coverage", "novelty_rate",
                             "swap_acceptance", "inversion_acceptance", "cache_hit_rate",
                             "duplicate_rate", "edge_diversity", "actual_uniform_swap_fraction")
        for row in data["runs"]:
            if row["variant"] == "guided" and row["exploration"] == 1:
                base = next(r for r in data["runs"] if r["variant"] == "uniform"
                            and all(r[key] == row[key] for key in ("children", "family", "trial_seed", "repeat")))
                self.assertIsNone(row["aliased_from"])
                for key in ("score", "evaluations", "evaluations_to_best") + diagnostic_fields:
                    self.assertEqual(row.get(key), base.get(key), key)
        with self.assertRaises(ValueError):
            validate_config(SuiteConfig(sizes=(3,)))
        with self.assertRaises(ValueError):
            validate_config(SuiteConfig(explorations=(float("nan"),)))

    def test_checks_exploration_one_is_copied_and_time_protocol_is_measured(self):
        common = dict(sizes=(6, 8), explorations=(0, 1), instances=1, trials=2, repeats=2,
                      check_repeats=1, budget=80, generations=3, exact_limit=6, time_limit=0.01,
                      workers=1, progress=False)
        with redirect_stdout(StringIO()):
            measured = execute_suite(SuiteConfig(**common, alias_full_exploration=False))
            copied = execute_suite(SuiteConfig(**common))
        key = lambda r: (r["protocol"], r["children"], r["family"], r["variant"], r["exploration"],
                         r["trial_seed"], r["repeat"])
        self.assertEqual(sorted(map(key, measured["runs"])), sorted(map(key, copied["runs"])))
        reference = {key(r): r for r in measured["runs"]}
        aliases = [r for r in copied["runs"] if r["aliased_from"] is not None]
        # Two families x 2 sizes x 2 trials x 1 checks repeat; no time-protocol copies.
        self.assertEqual(len(aliases), 8)
        for row in aliases:
            self.assertEqual((row["protocol"], row["variant"], row["exploration"]), ("checks", "guided", 1))
            source = next(r for r in copied["runs"] if r["run_id"] == row["aliased_from"])
            self.assertEqual(source["variant"], "uniform")
            for field in ("score", "evaluations", "evaluations_to_best", "unique_scored", "edge_coverage"):
                self.assertEqual(row[field], reference[key(row)][field], field)
        self.assertTrue(all(r["aliased_from"] is None for r in copied["runs"] if r["protocol"] == "time"))
        traces = {r["run_id"] for r in copied["improvements"]}
        self.assertTrue(all(r["run_id"] in traces for r in copied["runs"]))
        self.assertEqual(len(copied["diagnostics"]), len(measured["diagnostics"]))

    def test_proven_bound_stops_without_changing_best_or_trace(self):
        distances = distance_matrix(random_profiles(2027, 8, 5))
        optimum = exact_best(distances)[1]
        searches = {
            "HC": lambda **kw: run_hill_climber(distances, Random(3), evaluation_budget=5000, **kw),
            "HC guided": lambda **kw: run_guided_hill_climber(distances, Random(3), evaluation_budget=5000, **kw),
            "GA": lambda **kw: run_ga(distances, Random(3), population_size=20, generations=200,
                                      tournament_size=5, elites=2, crossover_rate=0.9,
                                      mutation_rate=0.25, evaluation_budget=5000, **kw),
        }
        for name, search in searches.items():
            traces = {}
            results = {}
            for bound in (None, optimum, optimum + 2):
                trace = traces[bound] = []
                results[bound] = search(score_bound=bound, on_improvement=lambda s, e: trace.append((s, e)))
            plain, stopped, unreachable = results[None], results[optimum], results[optimum + 2]
            self.assertEqual(plain["score"], optimum, name)
            self.assertEqual(stopped["stop_reason"], "bound_reached", name)
            self.assertEqual((stopped["circle"], stopped["score"]), (plain["circle"], plain["score"]), name)
            self.assertEqual(traces[optimum], traces[None], name)
            self.assertEqual(stopped["evaluations"], traces[None][-1][1], name)
            self.assertLess(stopped["evaluations"], plain["evaluations"], name)
            self.assertEqual(unreachable, plain, name)

    def test_hc_incremental_edge_coverage_matches_all_observed_circles(self):
        distances = distance_matrix(random_profiles(2027, 12, 5))
        for search in (run_hill_climber, run_guided_hill_climber):
            diagnostics = SearchDiagnostics(12)
            search(distances, Random(4), evaluation_budget=3000, diagnostics=diagnostics)
            expected = {frozenset((c[k], c[(k + 1) % 12])) for c in diagnostics.proposed | diagnostics.scored
                        for k in range(12)}
            self.assertEqual(len(diagnostics.edges), len(expected))

    @skipUnless(find_spec("plotly"), "optional Plotly dependency not installed")
    def test_offline_report_and_all_data_exports(self):
        config = SuiteConfig(sizes=(4,), explorations=(0, 1), protocols=("checks",),
                             instances=1, trials=1, repeats=1, budget=10, generations=1,
                             exact_limit=4, diagnostics=True, workers=1)
        captured = StringIO()
        with TemporaryDirectory() as directory, redirect_stdout(captured):
            output, _ = run_and_save(config, directory)
            expected = {"report.html", "configuration.json", "results.json", "runs.csv",
                        "improvements.csv", "summaries.csv", "exploration_effects.csv",
                        "exploration_trends.csv", "diagnostics.csv", "diagnostic_snapshots.csv",
                        "targets.csv", "problems.csv"}
            self.assertTrue(expected.issubset({p.name for p in output.iterdir()}))
            html = (output / "report.html").read_text()

            class ScriptParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.external = []

                def handle_starttag(self, tag, attrs):
                    if tag == "script" and "src" in dict(attrs):
                        self.external.append(dict(attrs)["src"])

            parser = ScriptParser()
            parser.feed(html)
            self.assertEqual(parser.external, [])
            self.assertIn("Exploration response", html)
            self.assertIn("common-target attainment", html)
            self.assertIn("Spearman", html)
        reports = [line for line in captured.getvalue().splitlines() if line.startswith("[report]")]
        self.assertTrue(reports)
        self.assertIn("(100.0%)", reports[-1])


if __name__ == "__main__":
    main()
