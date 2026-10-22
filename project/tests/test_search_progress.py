"""Verify first-hit improvement reporting without relying on clock speed."""

from contextlib import redirect_stdout
from io import StringIO
from random import Random
from unittest import TestCase, main
from unittest.mock import patch

from circle_optimization.benchmark.cli import main as benchmark_main
from circle_optimization import (
    distance_matrix, make_guided_mutation, run_ga, run_guided_hill_climber,
    run_hill_climber,
)


class SearchProgressTests(TestCase):
    def test_hc_records_first_hit_and_does_not_reset_on_ties(self):
        distances = distance_matrix((
            (False, False, False, False, False),
            (True, True, True, True, False),
            (True, True, False, False, True),
            (True, True, False, False, True),
        ))
        for search in (run_hill_climber, run_guided_hill_climber):
            points = []
            rng = Random(1)
            # Start at score 10; the first inversion scores 12. Further
            # proposals undo it and are rejected, but search keeps running.
            with patch("circle_optimization.classic.hill_climbing.random_individual",
                       return_value=(0, 1, 2, 3)), \
                    patch("circle_optimization.classic.hill_climbing.uniform_swap_positions",
                          return_value=(1, 2)), \
                    patch.object(rng, "random", return_value=0.9):
                result = search(distances, rng, evaluation_budget=5, max_restarts=1,
                                on_improvement=lambda score, checks: points.append((score, checks)))
            self.assertEqual(points, [(10, 1), (12, 2)])
            self.assertEqual(result["score"], 12)
            self.assertEqual(result["evaluations"], 5)

    def test_ga_reports_within_initial_population_not_only_generation_end(self):
        distances = distance_matrix(((False,),) * 5)
        points = []
        result = run_ga(
            distances, Random(3), population_size=5, generations=3,
            tournament_size=2, elites=1, crossover_rate=0.9, mutation_rate=0.5,
            evaluation_budget=30,
            on_improvement=lambda score, checks: points.append((score, checks)),
        )
        self.assertEqual(points, [(0, 1)])
        self.assertGreaterEqual(result["evaluations"], 5)

    def test_observing_improvements_preserves_search_results(self):
        rng = Random(7)
        distances = distance_matrix(tuple(
            tuple(bool(rng.getrandbits(1)) for _ in range(7)) for _ in range(8)
        ))
        searches = [
            (run_hill_climber, dict(evaluation_budget=400)),
            (run_guided_hill_climber, dict(evaluation_budget=400)),
            (run_ga, dict(population_size=10, generations=10, tournament_size=3,
                          elites=2, crossover_rate=0.9, mutation_rate=0.5,
                          evaluation_budget=100)),
        ]
        guided_ga_settings = dict(searches[-1][1])
        guided_ga_settings["mutation_operator"] = make_guided_mutation(distances)
        searches.append((run_ga, guided_ga_settings))
        for search, settings in searches:
            points = []
            observed = search(
                distances, Random(11), **settings,
                on_improvement=lambda score, checks: points.append((score, checks)),
            )
            self.assertEqual(observed, search(distances, Random(11), **settings))
            self.assertEqual(points[-1][0], observed["score"])
            self.assertEqual(points[0][1], 1)
            self.assertTrue(all(a[0] < b[0] and a[1] < b[1]
                                for a, b in zip(points, points[1:])))
            self.assertLessEqual(points[-1][1], observed["evaluations"])

    def test_benchmark_reports_time_and_checks_to_best(self):
        output = StringIO()
        with patch("sys.argv", ["benchmark_guided_search.py", "--children", "4",
                                "--instances", "1", "--trials", "1", "--repeats", "2",
                                "--budget", "6", "--details", "--include-ga"]), \
                redirect_stdout(output):
            benchmark_main()
        text = output.getvalue()
        self.assertIn("to-best ms", text)
        self.assertIn("best@check", text)
        header = ("algorithm,instance_seed,trial_seed,score,evaluations,seconds,"
                  "evaluations_to_best,seconds_to_best\n")
        rows = text.split(header)[1].strip().splitlines()
        self.assertEqual(len(rows), 4)
        for row in rows:
            _, _, _, _, evaluations, seconds, best_at, time_to_best = row.split(",")
            self.assertTrue(1 <= int(best_at) <= int(evaluations))
            self.assertTrue(0 <= float(time_to_best) <= float(seconds))


if __name__ == "__main__":
    main()
