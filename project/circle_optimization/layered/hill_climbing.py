"""Layered hill climbing: classic HC on profile sequences plus optional layers.

The engine keeps the classic HC's nature (phases 01–06): random or seeded
starts, one delta-scored proposal per check, strict-improvement acceptance,
productive or legacy restarts, global count/deadline/bound limits. What changes
is the state and which layers are enabled:

- S2 (phase 09, always on): the circle is a list of profile ids scored with
  the m x m table (`state.ProfileState`). With every other layer disabled the
  run reproduces `classic.hill_climbing.run_hill_climber` exactly: same random
  draws, scores, counts, restart history and improvement trace.
- S1 (phase 10, `splice_rate`): a share of proposals are split-and-splice
  moves, the second reconnection of the inversion's two cut edges
  (`splice`).
- S3 (phase 11, `guide`): inversion and splice positions come from a
  `reduced_costs.ReducedCostGuide` (worst edge by LP reduced cost, partner
  creating an LP-compatible pair) instead of uniform sampling.
- S5 (phase 13, `propagate`): after an accepted inversion or splice the same
  profile pattern is retried elsewhere (`propagation.Propagation`).
- S6 (phase 14, `start`, `perturbation`): greedy GRASP starts and
  double-bridge iterated local search around the incumbent (`restarts`).

Every check is one full score at a (re)start or one delta for a proposal of
any kind, propagated proposals included; position sampling is free.
"""

from collections.abc import Callable
from random import Random
from time import perf_counter

from ..classic.operators import apply_swap, inversion_delta, swap_delta, uniform_swap_positions
from .propagation import INVERSION as PROPAGATED_INVERSION, Propagation
from .reduced_costs import ReducedCostGuide
from .restarts import double_bridge, seed_sequence
from .splice import apply_splice, splice_delta
from .state import ProfileState

MOVE_KINDS = ("swap", "inversion", "splice", "propagated")
SWAP, INVERSION, SPLICE, PROPAGATED = range(4)


def run_layered_hill_climber(
    state: ProfileState, rng: Random, *, evaluation_budget: int,
    max_restarts: int = 200, attempts_per_restart: int = 1000, stall_limit: int = 200,
    restart_policy: str = "productive", inversion_rate: float = 0.5,
    splice_rate: float = 0.0, guide: ReducedCostGuide | None = None,
    propagate: bool = False, start: str = "random", start_noise: float = 1.5,
    perturbation: str | None = None,
    on_improvement: Callable[[int, int], None] | None = None,
    deadline: float | None = None, score_bound: int | None = None,
) -> dict:
    """Strict-improvement HC over profile sequences with optional layers.

    Parameters and accounting match `classic.hill_climbing.run_hill_climber`:
    `evaluation_budget` counts one full score per (re)start plus one delta
    check per proposed move; `restart_policy` is "productive" (restart after
    `stall_limit` consecutive failures) or "legacy" (also cap each local
    search at `attempts_per_restart` proposals). Each proposal draws
    r = rng.random(): a swap if r < 1 - inversion_rate - splice_rate, an
    inversion if r < 1 - splice_rate, otherwise a splice (S1), so
    splice_rate = 0 keeps the classic draws. `on_improvement(score,
    evaluations)` reports the first scored circle and each strict global
    improvement; `deadline` is an absolute perf_counter time; reaching
    `score_bound` (a proven upper bound) stops the search.

    Layers: `guide` (S3) picks inversion/splice positions; `propagate` (S5)
    retries an accepted inversion/splice pattern (ended by the first failure
    or `propagation.MAX_ATTEMPTS` proposals); `start` "random" or "greedy"
    (S6, `restarts.seed_sequence` with `start_noise`) builds each start;
    `perturbation="double-bridge"` (S6) builds every start after the first
    as `restarts.double_bridge(best sequence)` (iterated local search).

    A splice with positive delta whose loops share no profile is rejected
    (a failed check). Returns the classic keys (`circle`, `score`,
    `evaluations`, `restarts`, `stop_reason`, `restart_policy`,
    `restart_history`, each entry with `start` "random", "greedy" or
    "perturb") plus `sequence` (best profile sequence), `layers` (enabled
    layer names) and `moves` ({kind: {"proposed", "accepted"}} for swap,
    inversion, splice and propagated proposals).
    """
    if min(evaluation_budget, max_restarts, attempts_per_restart, stall_limit) < 1:
        raise ValueError("Search budgets must be positive")
    if restart_policy not in ("productive", "legacy"):
        raise ValueError("Restart policy must be productive or legacy")
    if not (0 <= inversion_rate <= 1 and 0 <= splice_rate <= 1 and inversion_rate + splice_rate <= 1):
        raise ValueError("Inversion and splice rates must be nonnegative and sum to at most 1")
    if start not in ("random", "greedy"):
        raise ValueError("Start must be random or greedy")
    if perturbation not in (None, "double-bridge"):
        raise ValueError("Perturbation must be None or double-bridge")
    if start_noise < 0:
        raise ValueError("Start noise must be nonnegative")
    n, table = state.n, state.table
    swap_cut, splice_cut = 1 - inversion_rate - splice_rate, 1 - splice_rate
    allow_splice = splice_rate > 0
    layers = ("profile-state",) + tuple(name for name, enabled in (
        ("splice", allow_splice), ("reduced-cost", guide is not None), ("propagation", propagate),
        ("greedy-start", start == "greedy"), ("double-bridge", perturbation is not None)) if enabled)
    proposed, accepted_moves = [0] * 4, [0] * 4
    best: list[int] | None = None
    best_score = -1
    evaluations = restarts = 0
    stop_reason = "restart_limit"
    restart_history = []
    for _ in range(max_restarts):
        if score_bound is not None and best_score >= score_bound:
            stop_reason = "bound_reached"
            break
        if evaluations >= evaluation_budget:
            stop_reason = "evaluation_budget"
            break
        if best is not None and deadline is not None and perf_counter() >= deadline:
            stop_reason = "time_limit"
            break
        restarts += 1
        if perturbation is not None and best is not None:
            current, origin = double_bridge(best, rng), "perturb"
        elif start == "greedy":
            current, origin = seed_sequence(state, rng, noise=start_noise), "greedy"
        else:
            current, origin = state.random_sequence(rng), "random"
        current_score = state.score(current)
        evaluations += 1
        if current_score > best_score:
            best, best_score = current.copy(), current_score
            if on_improvement is not None:
                on_improvement(best_score, evaluations)
        if n < 3:
            stop_reason = "small_instance"
            restart_history.append(dict(restart=restarts, attempts=0, accepted=0, final_score=current_score,
                                        reason="small_instance", start=origin))
            break
        stalled = attempts = accepted = 0
        local_reason = "attempt_cap"
        local_limit = attempts_per_restart if restart_policy == "legacy" else evaluation_budget
        pattern: Propagation | None = None
        for _ in range(local_limit):
            if score_bound is not None and best_score >= score_bound:
                stop_reason = local_reason = "bound_reached"
                break
            if evaluations >= evaluation_budget:
                local_reason = "evaluation_budget"
                break
            if deadline is not None and perf_counter() >= deadline:
                stop_reason = local_reason = "time_limit"
                break
            if pattern is not None and (move := pattern.next_move(current, rng)) is not None:
                kind, crossed, i, j = PROPAGATED, move[0] != PROPAGATED_INVERSION, move[1], move[2]
                delta = (splice_delta if crossed else inversion_delta)(current, table, i, j)
            else:
                pattern = None
                r = rng.random()
                if r < swap_cut:
                    kind = SWAP
                    i, j = uniform_swap_positions(n, rng)
                    delta = swap_delta(current, table, i, j)
                elif r < splice_cut:
                    kind, crossed = INVERSION, False
                    i, j = sorted(uniform_swap_positions(n, rng)) if guide is None else guide.select(current, rng)
                    delta = inversion_delta(current, table, i, j)
                else:
                    kind, crossed = SPLICE, True
                    i, j = (sorted(uniform_swap_positions(n, rng)) if guide is None
                            else guide.select(current, rng, crossed=True))
                    delta = splice_delta(current, table, i, j)
            evaluations += 1
            attempts += 1
            proposed[kind] += 1
            if delta > 0:
                if kind == SWAP:
                    apply_swap(current, i, j)
                else:
                    if propagate and pattern is None:  # A new trigger, built before the move.
                        pattern = Propagation.from_move(current, i, j, crossed=crossed,
                                                        allow_splice=allow_splice, profiles=state.m)
                    if not crossed:
                        current[i:j + 1] = reversed(current[i:j + 1])
                    elif (spliced := apply_splice(current, i, j, rng)) is not None:
                        current = spliced
                    else:  # The two loops share no profile: a failed check.
                        delta = 0
            if delta > 0:
                accepted += 1
                accepted_moves[kind] += 1
                current_score += delta
                stalled = 0
                if current_score > best_score:
                    best, best_score = current.copy(), current_score
                    if on_improvement is not None:
                        on_improvement(best_score, evaluations)
            else:
                pattern = None
                stalled += 1
                if stalled >= stall_limit:
                    local_reason = "stall"
                    break
        if evaluations >= evaluation_budget and local_reason != "bound_reached":
            local_reason = "evaluation_budget"
        restart_history.append(dict(restart=restarts, attempts=attempts, accepted=accepted,
                                    final_score=current_score, reason=local_reason, start=origin))
        if stop_reason in ("time_limit", "bound_reached"):
            break
    assert best is not None
    if evaluations >= evaluation_budget and stop_reason != "bound_reached":
        stop_reason = "evaluation_budget"
    return {"circle": state.circle(best), "score": best_score, "sequence": best,
            "evaluations": evaluations, "restarts": restarts, "stop_reason": stop_reason,
            "restart_policy": restart_policy, "restart_history": restart_history,
            "layers": layers,
            "moves": {name: {"proposed": proposed[k], "accepted": accepted_moves[k]}
                      for k, name in enumerate(MOVE_KINDS)}}
