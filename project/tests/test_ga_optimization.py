"""Frozen pre-phase-04 GA trajectories and independent OX equivalence checks."""

import hashlib
import json
from random import Random
from unittest import TestCase, main

from circle_optimization import (
    canonical_circle, distance_matrix, order_crossover, run_ga, tournament_selection,
)
from circle_optimization.problem import random_profiles
from circle_optimization.classic.operators import sample_pair


def reference_ox(left, right, rng):
    """Original modulo-based OX retained only as a behavioral test oracle."""
    a, b = left[1:], right[1:]
    if len(a) < 2:
        return canonical_circle(left)
    start, end = sorted(rng.sample(range(len(a)), k=2))
    child = [-1] * len(a)
    child[start:end + 1] = a[start:end + 1]
    used = set(a[start:end + 1])
    fill = (end + 1) % len(a)
    for offset in range(len(a)):
        gene = b[(end + 1 + offset) % len(a)]
        if gene not in used:
            child[fill] = gene
            fill = (fill + 1) % len(a)
    return canonical_circle((0, *child))


class GAOptimizationTests(TestCase):
    def test_frozen_seeded_trajectories(self):
        # Captured before modifying GA. Includes final population, every history
        # entry, strict-improvement callbacks, scores, and distinct evaluations.
        fixtures = {
            (6, 7): "6a1baa1006d9786ac122956601d0e83d6f5dbd62806941d92892510e5e3d0789",
            (6, 23): "dce52a3e7121b0b430706cba85c1210944e301bf98919e3e64aa02d39ee8892d",
            (10, 7): "e6ba0b107358d88885e493e19468debeda1caf52dd26f21d9c39f6565c6a558c",
            (40, 23): "2da0488fe1c678c469a1e1d81e16d2f482d4188a084113c3edbafa8f74dbe847",
            (160, 7): "ba336f92fef832e035e589241c3c33bbf3028bb6b89547373ab8308f64b67598",
        }
        for (n, seed), expected in fixtures.items():
            trace = []
            result = run_ga(
                distance_matrix(random_profiles(2027, n, 5)), Random(seed),
                population_size=12, generations=20, tournament_size=3, elites=2,
                crossover_rate=0.9, mutation_rate=0.25, evaluation_budget=500,
                on_improvement=lambda score, count: trace.append((score, count)),
            )
            payload = {key: result[key] for key in ("circle", "score", "evaluations", "population")}
            payload["history"] = [(g, b, float(m), float(v), e) for g, b, m, v, e in result["history"]]
            payload["trace"] = trace
            actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            self.assertEqual(actual, expected, (n, seed))

    def test_ox_preserves_original_offspring_and_random_draws(self):
        rng = Random(12)
        for n in range(2, 30):
            for seed in range(40):
                left = (0, *rng.sample(range(1, n), k=n - 1))
                right = (0, *rng.sample(range(1, n), k=n - 1))
                a, b = Random(seed), Random(seed)
                self.assertEqual(order_crossover(left, right, a), reference_ox(left, right, b))
                self.assertEqual(a.getstate(), b.getstate())

    def test_fast_sampling_preserves_random_sample_draws(self):
        for size in range(2, 60):
            for seed in range(20):
                a, b = Random(seed), Random(seed)
                for _ in range(10):
                    self.assertEqual(sample_pair(size, a), tuple(b.sample(range(size), k=2)))
                self.assertEqual(a.getstate(), b.getstate())
        # Covers Random.sample's pool branch and, at n=200 with k<=5, its set branch.
        for n, k in ((5, 1), (12, 3), (50, 10), (50, 50), (200, 3), (200, 10)):
            population = [(i, (i * 37) % 11) for i in range(n)]
            score = lambda item: item[1]  # Many ties: the first sampled maximum must win.
            for seed in range(30):
                a, b = Random(seed), Random(seed)
                for _ in range(10):
                    self.assertEqual(tournament_selection(population, score, k, a),
                                     max(b.sample(population, k=k), key=score))
                self.assertEqual(a.getstate(), b.getstate())


if __name__ == "__main__":
    main()
