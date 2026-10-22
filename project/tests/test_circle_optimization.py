"""Behavioral checks for the shared circular-arrangement searches."""

import unittest
from random import Random

from circle_optimization import (
    Child,
    FitnessEvaluator,
    canonical_circle,
    circle_fitness,
    distance_matrix,
    exact_best,
    initial_population,
    inversion_mutation,
    mutate,
    order_crossover,
    pairwise_distances,
    run_ga,
    run_hill_climber,
    swap_mutation,
    unique_circles,
    valid_circle,
)
from circle_optimization.classic.operators import inversion_delta, swap_delta


class CircleOptimizationTests(unittest.TestCase):
    def test_exact_circle_and_distinct_child_identities(self):
        children = [Child("A", (False,)), Child("B", (False,)),
                    Child("C", (True,)), Child("D", (True,))]
        distances = pairwise_distances(children, Child.difference)
        self.assertIsNot(children[0], children[1])
        self.assertEqual(distances, distance_matrix(((False,), (False,), (True,), (True,))))
        self.assertEqual(circle_fitness((0, 1, 2, 3), distances), 2)
        self.assertEqual(circle_fitness((0, 2, 1, 3), distances), 4)
        self.assertEqual(len(list(unique_circles(4))), 3)
        circle, score = exact_best(distances)
        self.assertTrue(valid_circle(circle, 4))
        self.assertEqual(score, 4)
        self.assertEqual(canonical_circle((0, 3, 2, 1)), (0, 1, 2, 3))

    def test_operators_and_population_preserve_permutations(self):
        rng = Random(42)
        population = initial_population(7, 15, rng)
        self.assertEqual(len(set(population)), 15)
        for _ in range(100):
            left, right = rng.sample(population, 2)
            for individual in (order_crossover(left, right, rng),
                               swap_mutation(left, rng), inversion_mutation(right, rng),
                               mutate(left, rng)):
                self.assertTrue(valid_circle(individual, 7))
                self.assertEqual(individual, canonical_circle(individual))

    def test_cached_fitness_and_budgeted_reproducible_searches(self):
        profiles = ((False, False, False), (True, True, True),
                    (True, False, True), (False, True, False),
                    (True, True, False), (False, False, True))
        distances = distance_matrix(profiles)
        optimum = exact_best(distances)[1]
        evaluator = FitnessEvaluator(distances)
        self.assertEqual(evaluator((0, 1, 2, 3, 4, 5)),
                         evaluator((0, 5, 4, 3, 2, 1)))
        self.assertEqual(evaluator.evaluations, 1)
        settings = dict(population_size=12, generations=20, tournament_size=3,
                        elites=2, crossover_rate=0.9, mutation_rate=0.5,
                        evaluation_budget=85)
        ga = run_ga(distances, Random(17), **settings)
        self.assertEqual(ga, run_ga(distances, Random(17), **settings))
        self.assertTrue(valid_circle(ga["circle"], 6))
        self.assertEqual(ga["score"], circle_fitness(ga["circle"], distances))
        self.assertLessEqual(ga["score"], optimum)
        self.assertLessEqual(ga["evaluations"], settings["evaluation_budget"])
        self.assertEqual(len(ga["population"]), len(set(ga["population"])))
        self.assertEqual(ga["history"][-1][-1], ga["evaluations"])
        hill = run_hill_climber(distances, Random(17),
                                evaluation_budget=ga["evaluations"])
        self.assertTrue(valid_circle(hill["circle"], 6))
        self.assertEqual(hill["score"], circle_fitness(hill["circle"], distances))
        self.assertLessEqual(hill["evaluations"], ga["evaluations"])

    def test_hill_move_deltas_match_full_scores(self):
        rng = Random(23)
        for n in range(3, 9):
            profiles = tuple(tuple(bool(rng.getrandbits(1)) for _ in range(5))
                             for _ in range(n))
            distances = distance_matrix(profiles)
            for _ in range(5):
                circle = [0, *rng.sample(range(1, n), k=n - 1)]
                original = circle_fitness(circle, distances)
                for i in range(1, n):
                    for j in range(i + 1, n):
                        swapped = circle.copy()
                        swapped[i], swapped[j] = swapped[j], swapped[i]
                        inverted = circle.copy()
                        inverted[i:j + 1] = reversed(inverted[i:j + 1])
                        with self.subTest(n=n, i=i, j=j, circle=circle):
                            self.assertEqual(
                                swap_delta(circle, distances, i, j),
                                circle_fitness(swapped, distances) - original,
                            )
                            self.assertEqual(
                                inversion_delta(circle, distances, i, j),
                                circle_fitness(inverted, distances) - original,
                            )

    def test_hill_budget_counts_proposals_and_restart_scores(self):
        distances = distance_matrix(((False,), (True,), (False,), (True,)))
        hill = run_hill_climber(
            distances, Random(4), evaluation_budget=5,
            max_restarts=1, attempts_per_restart=10, stall_limit=10,
        )
        self.assertEqual(hill["evaluations"], 5)  # One start and four moves.
        self.assertEqual(hill["restarts"], 1)
        self.assertTrue(valid_circle(hill["circle"], 4))
        self.assertEqual(hill["score"], circle_fitness(hill["circle"], distances))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            distance_matrix(((False,), (False, True)))
        with self.assertRaises(ValueError):
            exact_best(((0, 2), (1, 0)))
        with self.assertRaises(ValueError):
            initial_population(4, 4, Random(0))
        with self.assertRaises(ValueError):
            run_hill_climber(((0,),), Random(0), evaluation_budget=0)


if __name__ == "__main__":
    unittest.main()
