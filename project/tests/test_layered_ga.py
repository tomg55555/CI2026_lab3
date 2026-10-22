"""Layered GA eras: GA-P (S2), edge genomes and GA-E/GA-M (S4), GA seeding (S6)."""

import unittest
from importlib.util import find_spec
from itertools import product
from random import Random
from time import perf_counter
from unittest.mock import patch

from circle_optimization import distance_matrix, random_profiles, run_ga, valid_circle

HAS_NUMPY = find_spec("numpy") is not None
HAS_HIGHS = HAS_NUMPY and find_spec("highspy") is not None
if HAS_NUMPY:
    import numpy as np

    from circle_optimization.exact.multigraph import improve
    from circle_optimization.exact.profiles import components, multigraph_circle, validate_multigraph
    from circle_optimization.layered.edge_genome import (
        edge_crossover, edge_diversity, edge_mutation, genome_fitness, genome_key, key_fitness, random_genome,
    )
    from circle_optimization.layered.genetic import run_edge_ga, run_profile_ga
    from circle_optimization.layered.memetic import MemeticLayer
    from circle_optimization.layered.restarts import seed_multigraph
    from circle_optimization.layered.state import ProfileState

# Exact optima (phase 07 MILP) of the suite instances used below.
OPTIMA = {(2027, 40, 5): 176, (2027, 160, 5): 748}
GA = dict(population_size=20, tournament_size=4, elites=2, crossover_rate=0.9, mutation_rate=0.25)


def instance(seed, n, c):
    """ProfileState of a suite instance."""
    return ProfileState.from_profiles(random_profiles(seed, n, c))


def traced(search, *args, **kwargs):
    """Run a search and return (result, strict-improvement trace)."""
    trace = []
    result = search(*args, on_improvement=lambda s, e: trace.append((s, e)), **kwargs)
    return result, trace


@unittest.skipUnless(HAS_NUMPY, "layered eras need numpy")
class EdgeGenomeTests(unittest.TestCase):
    def states(self):
        """Varied instances: tiny, few profiles, one profile, complements, many profiles."""
        yield instance(2027, 40, 5)
        yield instance(2028, 160, 5)
        yield instance(3, 60, 8)
        yield instance(4, 7, 2)
        yield ProfileState.from_profiles([(True, False)] * 5 + [(False, True)] * 3)
        yield ProfileState.from_profiles([(True, True, False)] * 6)

    def assert_valid(self, x, state):
        validate_multigraph(x, state.instance)
        self.assertEqual(x.dtype, np.int64)
        circle = multigraph_circle(x, state.instance)
        self.assertTrue(valid_circle(circle, state.n))
        self.assertEqual(state.genome_score(circle), genome_fitness(x, state))
        self.assertEqual(key_fitness(genome_key(x), state), genome_fitness(x, state))

    def test_crossover_children_are_valid_connected_circles(self):
        for state, fill in product(self.states(), ("random", "greedy")):
            rng, py = np.random.default_rng(1), Random(2)
            parents = [random_genome(state, rng) for _ in range(6)] + [seed_multigraph(state, py) for _ in range(3)]
            for parent in parents:
                self.assert_valid(parent, state)
            for _ in range(40):
                i, j = rng.integers(len(parents), size=2)
                a, b = parents[i].copy(), parents[j].copy()
                child = edge_crossover(parents[i], parents[j], state, rng, fill=fill)
                self.assert_valid(child, state)
                self.assertTrue((parents[i] == a).all() and (parents[j] == b).all(), "parents untouched")
                parents.append(child)  # Later crossovers also mix children.
            same = edge_crossover(parents[0], parents[0], state, rng, fill=fill)
            self.assertTrue((same == parents[0]).all(), "identical parents reproduce themselves")
        state = instance(2027, 40, 5)
        x = random_genome(state, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            edge_crossover(x, x, state, np.random.default_rng(0), fill="ox")

    def test_greedy_fill_prefers_long_edges(self):
        state = instance(2027, 160, 5)
        rng = np.random.default_rng(5)
        gains = {fill: [] for fill in ("random", "greedy")}
        for _ in range(30):
            a, b = random_genome(state, rng), random_genome(state, rng)
            for fill in gains:
                gains[fill].append(genome_fitness(edge_crossover(a, b, state, rng, fill=fill), state))
        self.assertGreater(np.mean(gains["greedy"]), np.mean(gains["random"]))

    def test_mutation_is_one_connected_two_exchange(self):
        for state in self.states():
            rng = np.random.default_rng(3)
            x = random_genome(state, rng)
            changed = 0
            for _ in range(60):
                before = x.copy()
                mutated = edge_mutation(x, state, rng)
                self.assertTrue((x == before).all(), "the parent is not modified")
                self.assert_valid(mutated, state)
                difference = np.abs(np.triu(mutated) - np.triu(x)).sum()
                self.assertIn(difference, (0, 2, 4))  # Two copies out, two in (types may coincide).
                changed += difference > 0
                x = mutated
            if state.m > 2:
                self.assertGreater(changed, 40)

    def test_keys_and_diversity(self):
        state = instance(2027, 40, 5)
        rng = np.random.default_rng(0)
        a, b = random_genome(state, rng), random_genome(state, rng)
        self.assertEqual(genome_key(a), genome_key(a.copy()))
        self.assertNotEqual(genome_key(a), genome_key(b))
        keys = [genome_key(a), genome_key(b)]
        shared = int(np.minimum(np.triu(a), np.triu(b)).sum())
        self.assertEqual(edge_diversity(keys, 0), (0 + state.n - shared) / 2)


@unittest.skipUnless(HAS_NUMPY, "layered eras need numpy")
class MemeticLayerTests(unittest.TestCase):
    def test_layer_matches_bulk_local_search_and_counts_checks(self):
        state = instance(2027, 160, 5)
        d = state.instance.distances
        rng = np.random.default_rng(7)
        for max_moves in (None, 1, 4):
            x = random_genome(state, rng)
            y = x.copy()
            stats = {}
            moves = improve(y, d, max_moves=max_moves, stats=stats)
            checks = MemeticLayer(max_moves=max_moves).improve(x, d, rng)
            self.assertTrue((x == y).all())
            self.assertEqual(checks, stats["checks"])
            validate_multigraph(x, state.instance)
            if max_moves is not None:
                self.assertEqual(moves, max_moves)

    def test_limit_probability_and_validation(self):
        state = instance(2027, 160, 5)
        d = state.instance.distances
        rng = np.random.default_rng(8)
        x = random_genome(state, rng)
        edges = np.count_nonzero(np.triu(x))
        for limit in (0, edges ** 2 - 1, edges ** 2, 5 * edges ** 2):
            y = x.copy()
            checks = MemeticLayer().improve(y, d, rng, limit=limit)
            self.assertLessEqual(checks, limit)
            self.assertEqual(checks == 0, limit < edges ** 2)
        y = x.copy()
        self.assertEqual(MemeticLayer(probability=0.0).improve(y, d, rng), 0)
        self.assertTrue((x == y).all())
        self.assertEqual(MemeticLayer().improve(y, d, rng, deadline=perf_counter() - 1), 0)
        for bad in (dict(max_moves=0), dict(probability=1.5)):
            with self.assertRaises(ValueError):
                MemeticLayer(**bad)


@unittest.skipUnless(HAS_NUMPY, "layered eras need numpy")
class ProfileGATests(unittest.TestCase):
    def test_reproduces_classic_ga_when_profiles_are_distinct(self):
        # All 32 profiles of C=5 once: the profile key is a relabeling of the genome.
        profiles = [tuple(bool(code >> k & 1) for k in range(5)) for code in range(32)]
        Random(1).shuffle(profiles)
        state = ProfileState.from_profiles(profiles)
        for seed, budget in ((5, 400), (6, 2000)):
            settings = dict(GA, generations=60, evaluation_budget=budget)
            classic, classic_trace = traced(run_ga, distance_matrix(profiles), Random(seed), **settings)
            layered, layered_trace = traced(run_profile_ga, state, Random(seed), **settings)
            self.assertEqual(classic_trace, layered_trace)
            for key in ("circle", "score", "evaluations", "history", "population", "stop_reason",
                        "generations_completed"):
                self.assertEqual(classic[key], layered[key], key)

    def test_valid_circles_below_the_optimum_and_keys_collapse_relabelings(self):
        state = instance(2027, 40, 5)
        if HAS_HIGHS:
            from circle_optimization.exact.milp import solve_circle
            self.assertEqual(solve_circle(random_profiles(2027, 40, 5))["score"], OPTIMA[2027, 40, 5])
        for seed, seeding in product((1, 2, 3), (0.0, 0.2)):
            result, trace = traced(run_profile_ga, state, Random(seed), generations=40,
                                   evaluation_budget=1500, seeding=seeding, **GA)
            self.assertTrue(valid_circle(result["circle"], state.n))
            self.assertEqual(state.genome_score(result["circle"]), result["score"])
            self.assertEqual(state.sequence(result["circle"]), result["sequence"])
            self.assertLessEqual(result["score"], OPTIMA[2027, 40, 5])
            self.assertLessEqual(result["evaluations"], 1500)
            self.assertEqual(trace[-1][0], result["score"])
            self.assertEqual(result["layers"], ("profile-state", "ox") + (("seeded",) if seeding else ()))
            # Duplicate rejection is by profile sequence, not by child genome.
            self.assertEqual(len({tuple(state.sequence(g)) for g in result["population"]}),
                             len(result["population"]))

    def test_seeding_and_degenerate_instances(self):
        state = instance(2027, 160, 5)
        plain = run_profile_ga(state, Random(4), generations=0, evaluation_budget=100, **GA)
        seeded = run_profile_ga(state, Random(4), generations=0, evaluation_budget=100, seeding=0.5, **GA)
        self.assertGreater(seeded["history"][0][1], plain["history"][0][1] + 200)
        self.assertEqual(seeded, run_profile_ga(state, Random(4), generations=0, evaluation_budget=100,
                                                seeding=0.5, **GA))
        # Fewer distinct profile sequences than the population: still terminates.
        for profiles in ([(True,)] * 6, [(True,)] * 3 + [(False,)] * 3):
            tiny = ProfileState.from_profiles(profiles)
            result = run_profile_ga(tiny, Random(0), generations=5, evaluation_budget=500,
                                    population_size=12, tournament_size=3, elites=2,
                                    crossover_rate=0.9, mutation_rate=0.25)
            self.assertTrue(valid_circle(result["circle"], 6))
            self.assertEqual(tiny.genome_score(result["circle"]), result["score"])
            self.assertLessEqual(result["score"], 0 if len(set(profiles)) == 1 else 6)
        with self.assertRaises(ValueError):
            run_profile_ga(state, Random(0), generations=5, evaluation_budget=500, seeding=2, **GA)


@unittest.skipUnless(HAS_NUMPY, "layered eras need numpy")
class EdgeGATests(unittest.TestCase):
    def test_pure_and_memetic_runs_on_the_same_seed(self):
        state = instance(2027, 160, 5)
        if HAS_HIGHS:
            from circle_optimization.exact.milp import solve_circle
            self.assertEqual(solve_circle(random_profiles(2027, 160, 5))["score"], OPTIMA[2027, 160, 5])
        budget = 10**8
        for fill in ("random", "greedy"):
            runs = {}
            for name, layer in (("ga-e", None), ("ga-m", MemeticLayer(max_moves=3))):
                result, trace = traced(run_edge_ga, state, Random(11), generations=10, evaluation_budget=budget,
                                       fill=fill, local_search=layer, **GA)
                runs[name] = result
                validate_multigraph(result["multigraph"], state.instance)
                self.assertTrue(valid_circle(result["circle"], state.n))
                self.assertEqual(state.genome_score(result["circle"]), result["score"])
                self.assertEqual(genome_fitness(result["multigraph"], state), result["score"])
                self.assertLessEqual(result["score"], OPTIMA[2027, 160, 5])
                self.assertEqual(result["evaluations"], result["fitness_evaluations"] + result["memetic_checks"])
                self.assertLessEqual(result["evaluations"], budget)
                self.assertEqual(trace[-1][0], result["score"])
                self.assertEqual([s for s, _ in trace], sorted({s for s, _ in trace}))
                self.assertEqual(result["history"][-1][4], result["evaluations"])
                expected = ("profile-state", "edge-crossover") + (("greedy-fill",) if fill == "greedy" else ())
                self.assertEqual(result["layers"], expected + (("memetic",) if layer else ()))
            self.assertEqual(runs["ga-e"]["memetic_checks"], 0)
            self.assertGreater(runs["ga-m"]["memetic_checks"], 0)
            self.assertGreater(runs["ga-m"]["score"], runs["ga-e"]["score"])

    def test_determinism_seeding_and_budget(self):
        state = instance(2028, 160, 5)
        for settings in (dict(fill="random"), dict(fill="greedy", seeding=0.2),
                         dict(fill="greedy", local_search=MemeticLayer(max_moves=2), seeding=0.1)):
            first = run_edge_ga(state, Random(3), generations=8, evaluation_budget=10**9, **settings, **GA)
            second = run_edge_ga(state, Random(3), generations=8, evaluation_budget=10**9, **settings, **GA)
            self.assertEqual((first["circle"], first["score"], first["history"], first["evaluations"]),
                             (second["circle"], second["score"], second["history"], second["evaluations"]))
            self.assertEqual(first["stop_reason"], "generation_limit")
            self.assertEqual(first["generations_completed"], 8)
        plain = run_edge_ga(state, Random(4), generations=0, evaluation_budget=100, **GA)
        seeded = run_edge_ga(state, Random(4), generations=0, evaluation_budget=100, seeding=0.5, **GA)
        self.assertGreater(seeded["history"][0][1], plain["history"][0][1] + 200)
        self.assertEqual(seeded["layers"][-1], "seeded")
        small = instance(2027, 40, 5)
        for budget, layer in ((20, None), (137, None), (1_000, MemeticLayer()), (30_000, MemeticLayer()),
                              (30_000, MemeticLayer(max_moves=1))):
            result = run_edge_ga(small, Random(5), generations=10**6, evaluation_budget=budget,
                                 local_search=layer, **GA)
            self.assertLessEqual(result["evaluations"], budget)
            self.assertEqual(result["stop_reason"], "evaluation_budget")
        with self.assertRaises(ValueError):
            run_edge_ga(state, Random(0), generations=5, evaluation_budget=10, **GA)  # < population
        with self.assertRaises(ValueError):
            run_edge_ga(state, Random(0), generations=5, evaluation_budget=500, fill="ox", **GA)

    def test_deadline_bound_and_degenerate_instances(self):
        state = instance(2027, 160, 5)
        with patch("circle_optimization.classic.genetic.perf_counter", return_value=2), \
                patch("circle_optimization.layered.genetic.perf_counter", return_value=2):
            for search, extra in ((run_profile_ga, {}), (run_edge_ga, dict(local_search=MemeticLayer()))):
                result = search(state, Random(8), generations=10, evaluation_budget=10**6, deadline=1, **GA, **extra)
                self.assertEqual(result["evaluations"], 1)
                self.assertEqual(result["stop_reason"], "time_limit")
                self.assertEqual(state.genome_score(result["circle"]), result["score"])
        memetic = dict(local_search=MemeticLayer())
        for search, extra in ((run_profile_ga, {}), (run_edge_ga, {}), (run_edge_ga, memetic)):
            started = perf_counter()
            result = search(state, Random(9), generations=10**9, evaluation_budget=10**12,
                            deadline=started + 0.05, **GA, **extra)
            self.assertLess(perf_counter() - started, 0.5)
            self.assertEqual(result["stop_reason"], "time_limit")
            low = search(state, Random(9), generations=10**9, evaluation_budget=10**12, score_bound=400,
                         **GA, **extra)
            self.assertEqual(low["stop_reason"], "bound_reached")
            self.assertGreaterEqual(low["score"], 400)
        seeded = run_edge_ga(state, Random(2), generations=200, evaluation_budget=10**9, fill="greedy",
                             seeding=0.2, local_search=MemeticLayer(), score_bound=OPTIMA[2027, 160, 5], **GA)
        self.assertEqual((seeded["score"], seeded["stop_reason"]), (OPTIMA[2027, 160, 5], "bound_reached"))
        for profiles in ([(True,)] * 6, [(True,)] * 3 + [(False,)] * 3, [(True, False)] * 4 + [(False, True)]):
            tiny = ProfileState.from_profiles(profiles)
            result = run_edge_ga(tiny, Random(0), generations=5, evaluation_budget=10**6,
                                 local_search=MemeticLayer(), seeding=0.5, **GA)
            self.assertTrue(valid_circle(result["circle"], tiny.n))
            self.assertEqual(tiny.genome_score(result["circle"]), result["score"])
            self.assertEqual(len(components(result["multigraph"])), 1)
        with self.assertRaises(ValueError):
            run_edge_ga(ProfileState.from_profiles([(True,), (False,)]), Random(0), generations=1,
                        evaluation_budget=50, **GA)


if __name__ == "__main__":
    unittest.main()
