"""Correctness checks for weak-edge selection and its HC/GA integration.

Runtime/quality measurements live in benchmark_guided_search.py; these tests
avoid assertions about noisy timing or an unproven heuristic advantage.
"""

from random import Random
from unittest import TestCase, main
from unittest.mock import patch

from circle_optimization import (
    canonical_circle, circle_fitness, distance_matrix, exact_best,
    make_guided_mutation, run_ga, run_guided_hill_climber, run_hill_climber,
    valid_circle,
)
from circle_optimization.classic.guidance import WeakEdgeSwapSelector
from circle_optimization.classic.operators import apply_swap


class GuidedSearchTests(TestCase):
    def setUp(self):
        self.distances = distance_matrix((
            (False, False, False), (True, True, True), (True, False, True),
            (False, True, False), (True, True, False), (False, False, True),
        ))

    def test_selector_targets_weak_edges_and_preserves_anchor(self):
        # Only edge 0-1 is poor. Its endpoint should be targeted much more
        # frequently than uniform swap selection (which includes it 50%).
        distances = tuple(tuple(0 if i == j or {i, j} == {0, 1} else 10
                                for j in range(5)) for i in range(5))
        selector = WeakEdgeSwapSelector(distances, exploration=0)
        circle = list(range(5))
        rng = Random(37)
        count = 0
        for _ in range(4000):
            i, j = selector.select(circle, rng)
            self.assertTrue(1 <= i < 5 and 1 <= j < 5 and i != j)
            count += 1 in (i, j)
        self.assertGreater(count / 4000, 0.80)
        self.assertLess(count / 4000, 0.90)

    def test_selected_edge_is_broken_including_wraparound(self):
        # Force the first random integer to land in each edge's weight bin.
        for n in range(4, 8):
            distances = distance_matrix(((False,),) * n)
            selector = WeakEdgeSwapSelector(distances, exploration=0)
            for edge in range(n):
                for endpoint_draw in (0.0, 0.9):
                    circle = list(range(n))
                    rng = Random(1)
                    with patch.object(rng, "randrange", side_effect=[edge, 1]), \
                            patch.object(rng, "random", return_value=endpoint_draw):
                        i, j = selector.select(circle, rng)
                    apply_swap(circle, i, j)
                    neighbors = {frozenset((circle[k], circle[(k + 1) % n]))
                                 for k in range(n)}
                    self.assertNotIn(frozenset((edge, (edge + 1) % n)), neighbors)
                    self.assertEqual(circle[0], 0)

    def test_weights_are_refreshed_for_a_changed_circle(self):
        distances = tuple(tuple(0 if i == j or {i, j} == {0, 1} else 10
                                for j in range(5)) for i in range(5))
        selector = WeakEdgeSwapSelector(distances, exploration=0)
        circle = list(range(5))
        rng = Random(1)
        with patch.object(rng, "randrange", side_effect=[4, 1]):
            self.assertEqual(selector.select(circle, rng), (1, 2))
        apply_swap(circle, 1, 3)  # The weak pair is no longer adjacent.
        selector.invalidate()
        with patch.object(rng, "randrange", side_effect=[4, 1]):
            self.assertEqual(selector.select(circle, rng), (4, 2))

    def test_guided_hc_is_reproducible_and_matches_full_score(self):
        optimum = exact_best(self.distances)[1]
        for exploration in (0, 0.2, 1):
            for budget in (1, 2, 37, 300):
                settings = dict(evaluation_budget=budget, exploration=exploration,
                                attempts_per_restart=10, stall_limit=5)
                result = run_guided_hill_climber(self.distances, Random(8), **settings)
                self.assertEqual(result, run_guided_hill_climber(
                    self.distances, Random(8), **settings,
                ))
                self.assertTrue(valid_circle(result["circle"], 6))
                self.assertEqual(result["circle"], canonical_circle(result["circle"]))
                self.assertEqual(result["score"], circle_fitness(result["circle"], self.distances))
                self.assertLessEqual(result["score"], optimum)
                self.assertEqual(result["evaluations"], budget)

    def test_exploration_one_reproduces_baseline_hc(self):
        settings = dict(evaluation_budget=500, attempts_per_restart=35, stall_limit=10)
        for seed in range(5):
            self.assertEqual(
                run_hill_climber(self.distances, Random(seed), **settings),
                run_guided_hill_climber(self.distances, Random(seed), exploration=1, **settings),
            )

    def test_ga_guided_mutation_preserves_genomes_and_runs(self):
        mutation = make_guided_mutation(self.distances)
        genome = (0, 1, 2, 3, 4, 5)
        rng = Random(9)
        for _ in range(100):
            genome = mutation(genome, rng)
            self.assertTrue(valid_circle(genome, 6))
            self.assertEqual(genome, canonical_circle(genome))
        settings = dict(population_size=12, generations=10, tournament_size=3,
                        elites=2, crossover_rate=0.9, mutation_rate=0.5,
                        evaluation_budget=100)
        result = run_ga(self.distances, Random(5),
                        mutation_operator=make_guided_mutation(self.distances), **settings)
        self.assertEqual(result, run_ga(
            self.distances, Random(5),
            mutation_operator=make_guided_mutation(self.distances), **settings,
        ))
        self.assertEqual(result["score"], circle_fitness(result["circle"], self.distances))
        self.assertLessEqual(result["evaluations"], 100)
        self.assertTrue(all(valid_circle(child, 6) for child in result["population"]))
        self.assertEqual(
            run_ga(self.distances, Random(5), **settings),
            run_ga(self.distances, Random(5),
                   mutation_operator=make_guided_mutation(self.distances, exploration=1),
                   **settings),
        )

    def test_small_circles_and_invalid_configuration(self):
        for n in range(1, 4):
            distances = distance_matrix(((False,),) * n)
            result = run_guided_hill_climber(distances, Random(1), evaluation_budget=10)
            self.assertTrue(valid_circle(result["circle"], n))
            self.assertEqual(result["score"], 0)
        for exploration in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                run_guided_hill_climber(self.distances, Random(1),
                                        evaluation_budget=10, exploration=exploration)
        with self.assertRaises(ValueError):
            run_guided_hill_climber(((0,),), Random(1), evaluation_budget=0)


if __name__ == "__main__":
    main()
