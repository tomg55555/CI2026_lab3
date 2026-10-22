"""Layer S2: profile-state HC reproduces classic HC; shared S6 constructors."""

import unittest
from importlib.util import find_spec
from random import Random

from circle_optimization import (
    distance_matrix, random_profiles, run_hill_climber, run_inversion_hill_climber, valid_circle,
)

HAS_NUMPY = find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np

    from circle_optimization.layered.hill_climbing import run_layered_hill_climber
    from circle_optimization.layered.restarts import double_bridge, seed_multigraph, seed_sequence
    from circle_optimization.layered.state import ProfileState


def traced(search, *args, **kwargs):
    """Run a search and return (result, strict-improvement trace)."""
    trace = []
    result = search(*args, on_improvement=lambda s, e: trace.append((s, e)), **kwargs)
    return result, trace


@unittest.skipUnless(HAS_NUMPY, "layered eras need numpy")
class ProfileStateTests(unittest.TestCase):
    def test_sequence_score_circle_and_multigraph_agree(self):
        profiles = random_profiles(5, 50, 4)
        state = ProfileState.from_profiles(profiles)
        distances = distance_matrix(profiles)
        rng = Random(3)
        for _ in range(20):
            genome = tuple(rng.sample(range(50), 50))
            sequence = state.sequence(genome)
            score = sum(distances[a][b] for a, b in zip(genome, genome[1:] + genome[:1]))
            self.assertEqual(state.score(sequence), score)
            self.assertEqual(state.genome_score(genome), score)
            circle = state.circle(sequence)
            self.assertTrue(valid_circle(circle, 50))
            self.assertEqual(state.genome_score(circle), score)
            x = state.multigraph(sequence)
            self.assertEqual(int((x * state.instance.distances).sum()) // 2, score)
        with self.assertRaises(ValueError):
            state.circle([0] * 50)

    def test_profile_hc_reproduces_classic_hc_exactly(self):
        for seed, n, c in ((2027, 5, 3), (2027, 10, 5), (11, 23, 5), (2028, 40, 5), (2029, 160, 5)):
            profiles = random_profiles(seed, n, c)
            distances, state = distance_matrix(profiles), ProfileState.from_profiles(profiles)
            for policy, rate, budget in (("productive", 0.5, 3000), ("legacy", 0.5, 4000), ("productive", 0.8, 2500)):
                settings = dict(evaluation_budget=budget, restart_policy=policy, inversion_rate=rate,
                                max_restarts=50, stall_limit=60, attempts_per_restart=300)
                if rate == 0.5:
                    classic_settings = {k: v for k, v in settings.items() if k != "inversion_rate"}
                    classic, classic_trace = traced(run_hill_climber, distances, Random(seed + n), **classic_settings)
                else:
                    classic, classic_trace = traced(run_inversion_hill_climber, distances, Random(seed + n), **settings)
                layered, layered_trace = traced(run_layered_hill_climber, state, Random(seed + n), **settings)
                self.assertEqual(classic_trace, layered_trace)
                for key in ("score", "evaluations", "restarts", "stop_reason"):
                    self.assertEqual(classic[key], layered[key], (n, policy, key))
                # Layered entries add only the start kind (S6); nothing else may differ.
                self.assertTrue(all(entry.pop("start") == "random" for entry in layered["restart_history"]))
                self.assertEqual(classic["restart_history"], layered["restart_history"], (n, policy))
                self.assertTrue(valid_circle(layered["circle"], n))
                self.assertEqual(state.genome_score(layered["circle"]), layered["score"])

    def test_seeded_starts_and_double_bridge_keep_profile_counts(self):
        state = ProfileState.from_profiles(random_profiles(7, 300, 5))
        counts = state.instance.counts.tolist()
        rng = Random(4)
        x = seed_multigraph(state, rng)
        self.assertEqual((x.sum(axis=1) + x.diagonal()).tolist(), [2 * k for k in counts])
        for _ in range(5):
            sequence = seed_sequence(state, rng)
            self.assertEqual(np.bincount(sequence, minlength=state.m).tolist(), counts)
            kicked = double_bridge(sequence, rng)
            self.assertEqual(sorted(kicked), sorted(sequence))
            self.assertNotEqual(kicked, sequence)
        self.assertEqual(seed_sequence(state, Random(9)), seed_sequence(state, Random(9)))



if __name__ == "__main__":
    unittest.main()
