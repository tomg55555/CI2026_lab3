"""Layers S1, S3, S5 and S6 of the layered HC: brute-force checks and on/off comparisons."""

import unittest
from collections import Counter
from importlib.util import find_spec
from random import Random
from time import perf_counter

from circle_optimization import random_profiles, valid_circle

HAS_NUMPY = find_spec("numpy") is not None
HAS_SOLVER = HAS_NUMPY and find_spec("highspy") is not None
if HAS_NUMPY:
    from circle_optimization.classic.operators import inversion_delta
    from circle_optimization.layered.hill_climbing import MOVE_KINDS, run_layered_hill_climber
    from circle_optimization.layered.propagation import INVERSION, Propagation
    from circle_optimization.layered.splice import apply_splice, splice_delta, splice_loops
    from circle_optimization.layered.state import ProfileState
if HAS_SOLVER:
    from circle_optimization.exact.milp import lp_relaxation, solve_circle
    from circle_optimization.layered.reduced_costs import ReducedCostGuide, gap_certificate

OPTIMA = {(2027, 40, 5): 176, (2027, 160, 5): 748}  # Proven in phase 07.


def brute_score(sequence, table):
    """Circular score without the engine's helpers."""
    return sum(table[a][b] for a, b in zip(sequence, sequence[1:] + sequence[:1])) if len(sequence) > 1 else 0


def edge_multiset(sequence):
    """Unordered profile pairs of all circle edges."""
    return Counter(tuple(sorted(pair)) for pair in zip(sequence, sequence[1:] + sequence[:1]))


def traced(*args, **kwargs):
    trace = []
    result = run_layered_hill_climber(*args, on_improvement=lambda s, e: trace.append((s, e)), **kwargs)
    return result, trace


def cut_pairs(n, rng, count=60):
    """Cut positions 1 <= i < j <= n-1: all of them for small n, else edge cases plus random."""
    every = [(i, j) for i in range(1, n) for j in range(i + 1, n)]
    if len(every) <= count:
        return every
    special = [(1, n - 1), (1, 2), (n - 2, n - 1), (n // 2, n // 2 + 1), (2, n - 1), (1, n - 2)]
    return special + rng.sample(every, count - len(special))


@unittest.skipUnless(HAS_NUMPY, "layered eras need numpy")
class SpliceTests(unittest.TestCase):
    def test_splice_matches_brute_force(self):
        rng = Random(1)
        for seed, n, c in ((1, 4, 2), (2, 7, 2), (3, 12, 3), (4, 30, 3), (5, 60, 5), (6, 9, 4)):
            state = ProfileState.from_profiles(random_profiles(seed, n, c))
            table = state.table
            for _ in range(12):
                sequence = state.random_sequence(rng)
                original, before = sequence.copy(), brute_score(sequence, table)
                for i, j in cut_pairs(n, rng):
                    delta = splice_delta(sequence, table, i, j)
                    inner, outer = splice_loops(sequence, i, j)
                    # Derivation: delta is the score of the two loops minus the circle's.
                    self.assertEqual(brute_score(inner, table) + brute_score(outer, table) - before, delta)
                    spliced = apply_splice(sequence, i, j, Random(i * n + j))
                    if not set(inner) & set(outer):
                        self.assertIsNone(spliced)
                        continue
                    self.assertEqual(sorted(spliced), sorted(sequence))
                    self.assertEqual(brute_score(spliced, table), before + delta, (n, i, j))
                    a, b, cc, d = sequence[i - 1], sequence[i], sequence[j], sequence[(j + 1) % n]
                    expected = edge_multiset(sequence)
                    expected.subtract([tuple(sorted((a, b))), tuple(sorted((cc, d)))])
                    expected.update([tuple(sorted((a, d))), tuple(sorted((b, cc)))])
                    self.assertEqual(+expected, edge_multiset(spliced))
                    self.assertEqual(spliced, apply_splice(sequence, i, j, Random(i * n + j)))
                self.assertEqual(sequence, original)  # Moves are scored and built without mutation.

    def test_splice_needs_a_shared_profile(self):
        # Eight distinct profiles: the two loops never share one.
        state = ProfileState.from_profiles([(x & 1, x & 2, x & 4) for x in range(8)])
        sequence = state.random_sequence(Random(2))
        for i in range(1, 8):
            for j in range(i + 1, 8):
                self.assertIsNone(apply_splice(sequence, i, j, Random(0)))
        # Engine: positive-delta splices are rejected, the circle stays valid.
        result = run_layered_hill_climber(state, Random(3), evaluation_budget=3000,
                                          inversion_rate=0.0, splice_rate=1.0)
        self.assertEqual(result["moves"]["splice"]["accepted"], 0)
        self.assertTrue(valid_circle(result["circle"], 8))

    def test_splice_and_inversion_are_the_two_reconnections(self):
        state = ProfileState.from_profiles(random_profiles(8, 40, 4))
        table, rng = state.table, Random(5)
        sequence = state.random_sequence(rng)
        for i, j in cut_pairs(40, rng):
            a, b, c, d = sequence[i - 1], sequence[i], sequence[j], sequence[(j + 1) % 40]
            removed = table[a][b] + table[c][d]
            self.assertEqual(inversion_delta(sequence, table, i, j), table[a][c] + table[b][d] - removed)
            self.assertEqual(splice_delta(sequence, table, i, j), table[a][d] + table[b][c] - removed)


@unittest.skipUnless(HAS_NUMPY, "layered eras need numpy")
class PropagationTests(unittest.TestCase):
    def brute_edges(self, sequence, x, y):
        n = len(sequence)
        return sorted((k, sequence[k] == x) for k in range(n)
                      if sorted((sequence[k], sequence[(k + 1) % n])) == sorted((x, y)))

    def test_edge_scan_matches_brute_force(self):
        rng = Random(7)
        for m, n in ((3, 5), (4, 30), (32, 200), (300, 400)):  # 300 profiles: two-byte packing.
            for _ in range(10):
                sequence = [rng.randrange(m) for _ in range(n)]
                pattern = Propagation(0, 1, 2, 3, allow_splice=True, profiles=m)
                for _ in range(10):
                    x, y = rng.choice(sequence), rng.choice(sequence)
                    self.assertEqual(sorted(pattern.edges(sequence, x, y)), self.brute_edges(sequence, x, y))
        loops = [5, 5, 5, 1, 5, 5]  # Overlapping loop matches and the wraparound loop.
        self.assertEqual(sorted(Propagation(5, 5, 1, 1, allow_splice=False).edges(loops, 5, 5)),
                         self.brute_edges(loops, 5, 5))

    def test_pattern_moves_repeat_the_accepted_gain(self):
        rng = Random(11)
        state = ProfileState.from_profiles(random_profiles(2027, 160, 5))
        table, checked = state.table, Counter()
        for _ in range(40):
            sequence = state.random_sequence(rng)
            i, j = sorted(rng.sample(range(1, 160), 2))
            crossed = rng.random() < 0.5
            delta = (splice_delta if crossed else inversion_delta)(sequence, table, i, j)
            for allow_splice in (False, True):
                pattern = Propagation.from_move(sequence, i, j, crossed=crossed, allow_splice=allow_splice)
                self.assertEqual(pattern.gain(table), delta)
                for _ in range(5):
                    move = pattern.next_move(sequence, rng)
                    if move is None:
                        break
                    kind, i2, j2 = move
                    self.assertTrue(1 <= i2 < j2 <= 159)
                    self.assertTrue(allow_splice or kind == INVERSION)
                    move_delta = (inversion_delta if kind == INVERSION else splice_delta)(sequence, table, i2, j2)
                    self.assertEqual(move_delta, delta)
                    checked[kind] += 1
                    if kind == INVERSION:
                        after = sequence[:i2] + sequence[i2:j2 + 1][::-1] + sequence[j2 + 1:]
                    else:
                        after = apply_splice(sequence, i2, j2, rng)
                        if after is None:
                            continue
                    self.assertEqual(brute_score(after, table), brute_score(sequence, table) + delta)
        self.assertGreater(checked[INVERSION], 20)
        self.assertGreater(checked["splice"], 20)

    def test_attempt_cap(self):
        sequence = [0, 1, 2, 3] * 10
        pattern = Propagation(0, 1, 2, 3, allow_splice=True, max_attempts=3)
        moves = [pattern.next_move(sequence, Random(0)) for _ in range(5)]
        self.assertTrue(all(move is not None for move in moves[:3]))
        self.assertEqual(moves[3:], [None, None])


@unittest.skipUnless(HAS_SOLVER, "reduced costs need numpy and highspy")
class ReducedCostTests(unittest.TestCase):
    def test_lp_relaxation_bounds_signs_and_certificate(self):
        rng = Random(13)
        cases = [(2027, 40, 5), (2027, 160, 5)] + [(seed, rng.randint(3, 9), rng.randint(1, 4)) for seed in range(25)]
        for seed, n, c in cases:
            profiles = random_profiles(seed, n, c)
            state = ProfileState.from_profiles(profiles)
            exact = solve_circle(profiles)
            lp = lp_relaxation(state.instance)
            rc, values, upper = lp["reduced_costs"], lp["values"], lp["upper"]
            self.assertTrue(lp["connected"])
            self.assertGreaterEqual(lp["bound"] + 1e-6, exact["score"])
            self.assertTrue((rc == rc.T).all() and (values == values.T).all())
            self.assertTrue((abs(values[rc < -1e-7]) < 1e-7).all())
            self.assertTrue((abs(values - upper)[rc > 1e-7] < 1e-7).all())
            guide = ReducedCostGuide.from_state(state)
            sequences = [state.random_sequence(rng) for _ in range(20)] + [state.sequence(exact["circle"])]
            for sequence in sequences:
                certificate = gap_certificate(guide, sequence)
                self.assertGreaterEqual(certificate, -1e-6)
                self.assertLessEqual(state.score(sequence) + certificate, lp["bound"] + 1e-6)

    def test_guide_positions_are_valid_and_target_lp_pairs(self):
        state = ProfileState.from_profiles(random_profiles(2027, 160, 5))
        guide = ReducedCostGuide.from_state(state, exploration=0.0, tournament=8)
        uniform = ReducedCostGuide.from_state(state, exploration=1.0)
        rc = guide.reduced_costs
        rng = Random(17)
        for n in (5, 6, 9, 40, 160):
            sequence = state.random_sequence(rng)[:n]
            for crossed in (False, True):
                for _ in range(300):
                    i, j = guide.select(sequence, rng, crossed=crossed)
                    self.assertTrue(1 <= i < j <= n - 1)
        sequence = state.random_sequence(rng)
        rates = {}
        for name, selector in (("guided", guide), ("uniform", uniform)):
            created = removed = 0
            for crossed in (False, True):
                for _ in range(2000):
                    i, j = selector.select(sequence, rng, crossed=crossed)
                    a, b, c, d = sequence[i - 1], sequence[i], sequence[j], sequence[(j + 1) % 160]
                    new = ((a, d), (b, c)) if crossed else ((a, c), (b, d))
                    created += any(rc[p][q] >= 0 for p, q in new)
                    removed += rc[a][b] < 0 or rc[c][d] < 0
            rates[name] = created / 4000, removed / 4000
        self.assertGreater(rates["guided"][0], 0.9)
        self.assertGreater(rates["guided"][0], rates["uniform"][0] + 0.2)
        self.assertGreater(rates["guided"][1], rates["uniform"][1])

    def test_degenerate_instances_fall_back_to_uniform(self):
        single = ProfileState.from_profiles([(True, False)] * 6)  # m = 1: only loops.
        guide = ReducedCostGuide.from_state(single, exploration=0.0)
        self.assertFalse(guide.active)
        self.assertEqual(guide.certificate([0] * 6), 0)
        result = run_layered_hill_climber(single, Random(1), evaluation_budget=500, guide=guide,
                                          splice_rate=0.3, propagate=True, start="greedy",
                                          perturbation="double-bridge")
        self.assertEqual(result["score"], 0)
        self.assertTrue(valid_circle(result["circle"], 6))
        state = ProfileState.from_profiles(random_profiles(3, 30, 3))
        flat = ReducedCostGuide([[0.0] * state.m] * state.m, [[0.0] * state.m] * state.m,
                                state.table, bound=0.0, exploration=0.0)
        self.assertFalse(flat.active)
        rng = Random(2)
        for _ in range(100):
            i, j = flat.select(state.random_sequence(rng), rng)
            self.assertTrue(1 <= i < j <= 29)
        with self.assertRaises(ValueError):
            ReducedCostGuide.from_state(state, exploration=1.5)
        with self.assertRaises(ValueError):
            ReducedCostGuide.from_state(state, tournament=0)


@unittest.skipUnless(HAS_SOLVER, "layered HC tests use the exact optimum")
class LayeredEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states = {key: ProfileState.from_profiles(random_profiles(*key)) for key in OPTIMA}
        cls.guides = {key: ReducedCostGuide.from_state(state) for key, state in cls.states.items()}

    def configs(self, key):
        guide = self.guides[key]
        return {
            "splice": dict(inversion_rate=0.4, splice_rate=0.4),
            "reduced-cost": dict(guide=guide),
            "propagation": dict(propagate=True),
            "propagation+splice": dict(propagate=True, inversion_rate=0.3, splice_rate=0.5),
            "greedy-start": dict(start="greedy"),
            "double-bridge": dict(perturbation="double-bridge"),
            "all": dict(inversion_rate=0.4, splice_rate=0.4, guide=guide, propagate=True,
                        start="greedy", perturbation="double-bridge"),
        }

    def check_result(self, state, result, trace, budget, optimum):
        self.assertTrue(valid_circle(result["circle"], state.n))
        self.assertEqual(state.genome_score(result["circle"]), result["score"])
        self.assertEqual(state.score(result["sequence"]), result["score"])
        self.assertLessEqual(result["evaluations"], budget)
        self.assertLessEqual(result["score"], optimum)
        moves = result["moves"]
        self.assertEqual(tuple(moves), MOVE_KINDS)
        self.assertEqual(sum(m["proposed"] for m in moves.values()), result["evaluations"] - result["restarts"])
        self.assertTrue(all(0 <= m["accepted"] <= m["proposed"] for m in moves.values()))
        history = result["restart_history"]
        self.assertEqual(len(history), result["restarts"])
        self.assertEqual(sum(h["attempts"] for h in history), result["evaluations"] - result["restarts"])
        self.assertEqual(sum(h["accepted"] for h in history), sum(m["accepted"] for m in moves.values()))
        scores = [score for score, _ in trace]
        self.assertEqual(scores, sorted(set(scores)))
        self.assertEqual(scores[-1], result["score"])
        self.assertEqual([e for _, e in trace], sorted(e for _, e in trace))

    def test_defaults_equal_explicitly_disabled_layers(self):
        state = self.states[(2027, 40, 5)]
        for policy in ("productive", "legacy"):
            base = traced(state, Random(1), evaluation_budget=5000, restart_policy=policy)
            off = traced(state, Random(1), evaluation_budget=5000, restart_policy=policy, splice_rate=0.0,
                         guide=None, propagate=False, start="random", start_noise=1.5, perturbation=None)
            self.assertEqual(base, off)
            self.assertEqual(base[0]["layers"], ("profile-state",))
            self.assertEqual(base[0]["moves"]["splice"], {"proposed": 0, "accepted": 0})
            self.assertEqual(base[0]["moves"]["propagated"], {"proposed": 0, "accepted": 0})

    def test_each_layer_is_valid_bounded_deterministic_and_changes_the_search(self):
        budget = 20_000
        for key, optimum in OPTIMA.items():
            state = self.states[key]
            base = traced(state, Random(5), evaluation_budget=budget)
            for name, settings in self.configs(key).items():
                result, trace = traced(state, Random(5), evaluation_budget=budget, **settings)
                self.check_result(state, result, trace, budget, optimum)
                self.assertEqual((result, trace), traced(state, Random(5), evaluation_budget=budget, **settings))
                self.assertNotEqual(result["moves"], base[0]["moves"], name)
                layers = result["layers"]
                self.assertEqual(layers[0], "profile-state")
                self.assertEqual("splice" in layers, settings.get("splice_rate", 0) > 0)
                self.assertEqual("reduced-cost" in layers, "guide" in settings)
                self.assertEqual("propagation" in layers, "propagate" in settings)
                starts = [h["start"] for h in result["restart_history"]]
                first = "greedy" if settings.get("start") == "greedy" else "random"
                self.assertEqual(starts[0], first)
                later = "perturb" if "perturbation" in settings else first
                self.assertTrue(all(s == later for s in starts[1:]), name)
                if settings.get("propagate") and state.n >= 160:  # Repeated patterns need N >> m.
                    self.assertGreater(result["moves"]["propagated"]["accepted"], 0)

    def test_full_stack_reaches_the_optimum_and_stops_at_the_bound(self):
        key = (2027, 40, 5)
        settings = self.configs(key)["all"]
        result = run_layered_hill_climber(self.states[key], Random(3), evaluation_budget=300_000,
                                          score_bound=OPTIMA[key], **settings)
        self.assertEqual(result["score"], OPTIMA[key])
        self.assertEqual(result["stop_reason"], "bound_reached")
        self.assertEqual(result["restart_history"][-1]["reason"], "bound_reached")

    def test_tiny_budgets_and_limits_are_respected_inside_propagation(self):
        key = (2027, 160, 5)
        state = self.states[key]
        settings = dict(self.configs(key)["all"], start="random")
        for budget in (1, 2, 3, 17, 150, 1000, 4321):
            result, trace = traced(state, Random(budget), evaluation_budget=budget, **settings)
            self.check_result(state, result, trace, budget, OPTIMA[key])
            self.assertEqual(result["evaluations"], budget)
            self.assertEqual(result["stop_reason"], "evaluation_budget")
        expired = run_layered_hill_climber(state, Random(1), evaluation_budget=10**9,
                                           deadline=perf_counter(), **settings)
        self.assertEqual((expired["evaluations"], expired["stop_reason"]), (1, "time_limit"))
        timed = run_layered_hill_climber(state, Random(1), evaluation_budget=10**12, max_restarts=10**9,
                                         deadline=perf_counter() + 0.05, **settings)
        self.assertEqual(timed["stop_reason"], "time_limit")
        legacy = run_layered_hill_climber(state, Random(2), evaluation_budget=6000, restart_policy="legacy",
                                          attempts_per_restart=250, **settings)
        self.assertTrue(all(h["attempts"] <= 250 for h in legacy["restart_history"]))
        self.assertGreater(legacy["restarts"], 1)

    def test_move_kind_mix_and_validation(self):
        state = self.states[(2027, 40, 5)]
        only_splices = run_layered_hill_climber(state, Random(1), evaluation_budget=3000,
                                                inversion_rate=0.0, splice_rate=1.0)
        moves = only_splices["moves"]
        self.assertEqual(moves["splice"]["proposed"], 3000 - only_splices["restarts"])
        self.assertEqual(moves["swap"]["proposed"] + moves["inversion"]["proposed"], 0)
        for bad in (dict(splice_rate=-0.1), dict(inversion_rate=0.6, splice_rate=0.5), dict(start="best"),
                    dict(perturbation="kick"), dict(start_noise=-1)):
            with self.assertRaises(ValueError):
                run_layered_hill_climber(state, Random(1), evaluation_budget=10, **bad)


if __name__ == "__main__":
    unittest.main()
