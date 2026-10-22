"""Exactness and edge cases of the profile-aggregated heuristic and MILP solver."""

import unittest
from importlib.util import find_spec
from random import Random

from circle_optimization import distance_matrix, exact_best, circle_fitness, valid_circle
from circle_optimization.problem import random_profiles as suite_profiles
from circle_optimization.benchmark.metrics import quality_bound

HAS_SOLVER = find_spec("numpy") is not None and find_spec("highspy") is not None
if HAS_SOLVER:
    import numpy as np

    from circle_optimization.exact.milp import (
        _model, _separate, _set_integrality, _solution, solve_circle,
    )
    from circle_optimization.exact.profiles import (
        aggregate_profiles, components, euler_sequence, multigraph_score, profile_upper_bound,
        sequence_score,
    )
    from circle_optimization.exact.multigraph import connect, greedy_multigraph, improve, multi_start


def random_profiles(rng, n, c, p_true=0.5):
    return [tuple(rng.random() < p_true for _ in range(c)) for _ in range(n)]


def circle_score(circle, profiles):
    """Circular Hamming score without an N x N matrix."""
    rows = np.asarray(profiles, dtype=bool)[list(circle)]
    return int((rows != np.roll(rows, -1, axis=0)).sum())


def plain_dfj(instance):
    """Reference: full MILP, cut every disconnected optimum, no fixing or repair."""
    h, p, q, _, _ = _model(instance, seed=0, threads=None)
    _set_integrality(h, len(p) + instance.m, True)
    while True:
        h.run()
        x = _solution(h, instance, p, q)
        if not _separate(h, x, p, q):
            return multigraph_score(x, instance.distances)


@unittest.skipUnless(HAS_SOLVER, "profile solver needs numpy and highspy")
class ProfileModelTests(unittest.TestCase):
    def test_aggregation_matches_hamming_and_every_code_path(self):
        rng = Random(3)
        for n, c in ((1, 1), (7, 3), (200, 4), (50, 30), (40, 70)):
            profiles = random_profiles(rng, n, c, 0.3)
            instance = aggregate_profiles(profiles)
            self.assertEqual(instance.n, n)
            self.assertEqual(instance.counts.sum(), n)
            self.assertEqual(len(set(profiles)), instance.m)
            for child, profile in enumerate(profiles):
                self.assertEqual(tuple(instance.rows[instance.child_profile[child]]), profile)
            self.assertEqual(instance.distances.tolist(), [list(r) for r in distance_matrix(
                [tuple(r) for r in instance.rows.tolist()])])
            same = aggregate_profiles(np.asarray(profiles, dtype=bool))
            self.assertTrue((same.child_profile == instance.child_profile).all())

    def test_profile_bound_matches_quality_bound(self):
        rng = Random(5)
        for _ in range(60):
            profiles = random_profiles(rng, rng.randint(3, 30), rng.randint(1, 6), rng.random())
            self.assertEqual(profile_upper_bound(aggregate_profiles(profiles)),
                             quality_bound(profiles, distance_matrix(profiles)))

    def test_euler_sequence_visits_each_profile_its_count(self):
        rng = Random(8)
        for _ in range(40):
            profiles = random_profiles(rng, rng.randint(1, 60), rng.randint(1, 5), 0.4)
            instance = aggregate_profiles(profiles)
            x = greedy_multigraph(instance, np.random.default_rng(1), noise=2.0)
            if instance.m > 1:
                connect(x, instance.distances)
            sequence = euler_sequence(x)
            self.assertEqual(np.bincount(sequence, minlength=instance.m).tolist(), instance.counts.tolist())
            self.assertEqual(sequence_score(sequence, instance.distances),
                             multigraph_score(x, instance.distances))

    def test_bulk_two_exchange_keeps_degrees_and_connectivity(self):
        rng = Random(13)
        for _ in range(30):
            profiles = random_profiles(rng, rng.randint(5, 3000), rng.randint(2, 6))
            instance = aggregate_profiles(profiles)
            x = greedy_multigraph(instance, np.random.default_rng(2), noise=3.0)
            connect(x, instance.distances)
            before = multigraph_score(x, instance.distances)
            improve(x, instance.distances)
            self.assertGreaterEqual(multigraph_score(x, instance.distances), before)
            self.assertEqual((x.sum(axis=1) + x.diagonal()).tolist(), (2 * instance.counts).tolist())
            self.assertEqual(len(components(x)), 1)


@unittest.skipUnless(HAS_SOLVER, "profile solver needs numpy and highspy")
class ProfileSolverTests(unittest.TestCase):
    def check(self, profiles, result):
        self.assertTrue(valid_circle(result["circle"], len(profiles)))
        self.assertEqual(circle_score(result["circle"], profiles), result["score"])
        self.assertEqual(result["gap"], result["upper_bound"] - result["score"])
        self.assertEqual(result["score"] % 2, 0)

    def test_matches_exhaustive_enumeration(self):
        rng = Random(1)
        for trial in range(250):
            n, c = rng.randint(1, 9), rng.randint(1, 5)
            profiles = random_profiles(rng, n, c, rng.random())
            optimum = exact_best(distance_matrix(profiles))[1]
            exact = solve_circle(profiles, seed=trial)
            self.check(profiles, exact)
            self.assertEqual((exact["score"], exact["status"]), (optimum, "optimal"))
            self.assertEqual(circle_fitness(exact["circle"], distance_matrix(profiles)), optimum)
            heuristic = solve_circle(profiles, seed=trial, use_milp=False)
            self.check(profiles, heuristic)
            self.assertLessEqual(heuristic["score"], optimum)
            self.assertGreaterEqual(heuristic["upper_bound"], optimum)

    def test_trivial_sizes_and_identical_children(self):
        for profiles, score in (([(True,)], 0), ([(True, False)] * 5, 0),
                                ([(True, False), (False, True)], 4), ([(False,), (True,), (True,)], 2)):
            result = solve_circle(profiles)
            self.check(profiles, result)
            self.assertEqual((result["score"], result["status"]), (score, "optimal"))

    def test_connectivity_cuts_decide_complement_clusters(self):
        # Degree constraints alone pair 000/111 and 001/110 into two 2-cycles (24).
        profiles = [(0, 0, 0), (1, 1, 1), (0, 0, 1), (1, 1, 0)] * 2
        optimum = exact_best(distance_matrix(profiles))[1]
        self.assertLess(optimum, 24)
        result = solve_circle(profiles)
        self.check(profiles, result)
        self.assertEqual((result["score"], result["status"]), (optimum, "optimal"))

    def test_lp_integrality_gap_falls_back_to_full_model(self):
        rng = Random(16325)  # Found by search: even-rounded LP bound 96, optimum 94.
        n, c, p_true = rng.randint(4, 40), rng.randint(2, 7), rng.choice([0.5, 0.2, 0.1])
        profiles = random_profiles(rng, n, c, p_true)
        result = solve_circle(profiles)
        self.check(profiles, result)
        self.assertEqual(result["lp_bound"], 96)
        self.assertEqual((result["score"], result["status"]), (94, "optimal"))
        self.assertEqual(plain_dfj(aggregate_profiles(profiles)), 94)

    def test_large_instance_is_exact_and_bounded(self):
        profiles = np.random.default_rng(4).integers(0, 2, (50_000, 6), dtype=bool)
        result = solve_circle(profiles)
        self.check(profiles, result)
        self.assertEqual(result["status"], "optimal")
        self.assertEqual(result["profiles"], 64)
        self.assertLessEqual(result["score"], profile_upper_bound(aggregate_profiles(profiles)))
        self.assertLess(result["seconds"]["total"], 10)

    def test_expired_time_limit_still_returns_a_valid_circle(self):
        profiles = np.random.default_rng(9).integers(0, 2, (400, 8), dtype=bool)
        result = solve_circle(profiles, time_limit=1e-9)
        self.check(profiles, result)
        self.assertEqual(result["heuristic_starts"], 1)
        self.assertGreaterEqual(result["gap"], 0)
        if result["status"] != "optimal":
            self.assertIsNone(result["proof"])

    def test_any_time_limit_returns_a_valid_bounded_circle(self):
        # Regression: a MIP stopped before its root LP reports an infinite dual bound.
        profiles = suite_profiles(2028, 160, 7)
        optimum = solve_circle(profiles)["score"]
        for limit in [0.005 * k for k in range(1, 60)]:
            result = solve_circle(profiles, time_limit=limit)
            self.check(profiles, result)
            self.assertLessEqual(result["score"], optimum)
            self.assertGreaterEqual(result["upper_bound"], optimum)

    def test_multi_start_is_seeded_and_stops_at_target(self):
        instance = aggregate_profiles(random_profiles(Random(2), 300, 5))
        first, second = multi_start(instance, seed=7), multi_start(instance, seed=7)
        self.assertEqual(first["history"], second["history"])
        stopped = multi_start(instance, seed=7, target=0)
        self.assertEqual((stopped["starts"], stopped["stop_reason"]), (1, "bound_reached"))


@unittest.skipUnless(HAS_SOLVER, "profile solver needs numpy and highspy")
class SuiteReferenceTests(unittest.TestCase):
    def test_milp_reference_is_exact_beyond_enumeration(self):
        from circle_optimization.benchmark.suite import SuiteConfig, problem_reference
        from circle_optimization.problem import random_profiles

        profiles = random_profiles(2027, 40, 5)
        distances = distance_matrix(profiles)
        exact = problem_reference(SuiteConfig(exact_limit=0), profiles, distances)
        self.assertEqual((exact["reference_kind"], exact["exact_source"]), ("exact_optimum", "milp"))
        self.assertEqual(exact["reference"], 176)
        self.assertLess(exact["reference"], exact["upper_bound"])
        legacy = problem_reference(SuiteConfig(reference="bound", exact_limit=0), profiles, distances)
        self.assertEqual((legacy["reference_kind"], legacy["reference"]), ("upper_bound", exact["upper_bound"]))
        small = random_profiles(2027, 8, 5)
        both = problem_reference(SuiteConfig(exact_limit=8), small, distance_matrix(small))
        self.assertEqual((both["exact_source"], both["reference"]), ("enumeration", both["milp_score"]))


if __name__ == "__main__":
    unittest.main()
