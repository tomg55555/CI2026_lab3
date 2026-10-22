"""Random-restart hill climbing over index genomes (phases 01–06).

One engine, `_run_hill_climber`, implements strict-improvement search with
delta-scored swap/inversion proposals, productive or legacy restarts, deadlines
and proven-bound stops. Public entry points select a proposal policy:

- `run_hill_climber`: uniform 50/50 swaps and inversions (phase 01; productive
  restarts since phase 04, `restart_policy="legacy"` reproduces phase 01–03).
- `run_inversion_hill_climber`: 80% inversions / 20% swaps (phase 04, HC-I).
- `run_guided_hill_climber`: weak-edge-guided swaps (phase 02).
"""

from collections.abc import Callable
from random import Random
from time import perf_counter

from ..problem import DistanceMatrix, Genome, canonical_circle, circle_fitness, validate_distances
from .diagnostics import SearchDiagnostics
from .guidance import WeakEdgeSwapSelector
from .operators import apply_swap, inversion_delta, random_individual, swap_delta, uniform_swap_positions


def run_hill_climber(
    distances: DistanceMatrix, rng: Random,
    *, evaluation_budget: int, max_restarts: int = 200,
    attempts_per_restart: int = 1000, stall_limit: int = 200,
    on_improvement: Callable[[int, int], None] | None = None,
    deadline: float | None = None, diagnostics: SearchDiagnostics | None = None,
    restart_policy: str = "productive", score_bound: int | None = None,
) -> dict:
    """Find a good circle by accepting improving swap/inversion neighbors.

    Restart after `stall_limit` consecutive unsuccessful attempts, resetting
    patience on every local improvement. `restart_policy="legacy"` additionally
    enforces the old hard `attempts_per_restart` cap. Productive mode does not
    use that hard cap; global budget/deadline/restart limits still apply.
    Candidate moves are scored from the matrix
    using only affected edges; rejected moves do not construct a new circle.
    Return the best circle, score, number of score checks, and restarts.
    `evaluation_budget` counts one full score per restart plus one delta check
    per proposed move, including repeated moves. This differs from the GA's
    count of distinct complete-circle fitness evaluations.
    Optional `on_improvement(score, evaluations)` is called at the first scored
    circle and each strict improvement of the global best, not on ties.
    `deadline` is an absolute perf_counter time, checked between operations.
    At least one starting circle is scored, even for an already expired limit.
    Optional diagnostics are intended for untimed fixed-check replays.
    `score_bound` is a proven upper bound on fitness, such as an exact optimum.
    Reaching it stops with `bound_reached`: no strict improvement can follow,
    so the best circle, score and improvement trace are unchanged.
    """
    return _run_hill_climber(
        distances, rng, evaluation_budget=evaluation_budget, max_restarts=max_restarts,
        attempts_per_restart=attempts_per_restart, stall_limit=stall_limit,
        on_improvement=on_improvement,
        deadline=deadline, diagnostics=diagnostics,
        restart_policy=restart_policy, score_bound=score_bound,
    )


def _run_hill_climber(
    distances: DistanceMatrix, rng: Random,
    *, evaluation_budget: int, max_restarts: int, attempts_per_restart: int,
    stall_limit: int, swap_selector: WeakEdgeSwapSelector | None = None,
    on_improvement: Callable[[int, int], None] | None = None,
    deadline: float | None = None, diagnostics: SearchDiagnostics | None = None,
    restart_policy: str = "productive", inversion_rate: float = 0.5,
    score_bound: int | None = None,
) -> dict:
    """Shared acceptance/restart engine; only swap selection is configurable."""
    validate_distances(distances)
    if min(evaluation_budget, max_restarts, attempts_per_restart, stall_limit) < 1:
        raise ValueError("Search budgets must be positive")
    if restart_policy not in ("productive", "legacy"):
        raise ValueError("Restart policy must be productive or legacy")
    if not 0 <= inversion_rate <= 1:
        raise ValueError("Inversion rate must lie between 0 and 1")
    n = len(distances)
    best: Genome | None = None
    best_score = -1
    evaluations = 0
    restarts = 0
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
        current = list(random_individual(n, rng))
        if swap_selector is not None:
            swap_selector.invalidate()
        current_score = circle_fitness(current, distances)
        evaluations += 1
        if diagnostics is not None:
            diagnostics.counts["restarts"] += 1
            diagnostics.evaluation(current)
        if current_score > best_score:
            best, best_score = tuple(current), current_score
            if diagnostics is not None and restarts > 1:
                diagnostics.counts["restart_improvements"] += 1
            if on_improvement is not None:
                on_improvement(best_score, evaluations)
        if n < 3:
            stop_reason = "small_instance"
            restart_history.append(dict(restart=restarts, attempts=0, accepted=0,
                                        final_score=current_score, reason="small_instance"))
            break
        stalled = 0
        attempts = accepted = 0
        local_reason = "attempt_cap"
        local_limit = attempts_per_restart if restart_policy == "legacy" else evaluation_budget
        for _ in range(local_limit):
            if score_bound is not None and best_score >= score_bound:
                stop_reason = local_reason = "bound_reached"
                break
            if evaluations >= evaluation_budget:
                local_reason = "evaluation_budget"
                break
            if deadline is not None and perf_counter() >= deadline:
                stop_reason = "time_limit"
                local_reason = "time_limit"
                break
            is_swap = rng.random() < 1 - inversion_rate
            if is_swap:
                i, j = (uniform_swap_positions(n, rng) if swap_selector is None else
                        swap_selector.select(current, rng))
                delta = swap_delta(current, distances, i, j)
                if diagnostics is not None and swap_selector is None:
                    diagnostics.counts["uniform_swap_proposals"] += 1
            else:
                i, j = sorted(uniform_swap_positions(n, rng))
                delta = inversion_delta(current, distances, i, j)
                if diagnostics is not None:
                    diagnostics.counts["inversion_proposals"] += 1
            evaluations += 1
            attempts += 1
            if diagnostics is not None:
                candidate = current.copy()
                # Current edges are already recorded; only the move's edges can be new.
                if is_swap:
                    apply_swap(candidate, i, j)
                    changed = ((i - 1) % n, i, (j - 1) % n, j)
                else:
                    candidate[i:j + 1] = reversed(candidate[i:j + 1])
                    changed = (i - 1, j)
                diagnostics.proposal(candidate, scored=True, changed_edges=changed)
                diagnostics.move("swap" if is_swap else "inversion", delta)
            if delta > 0:
                accepted += 1
                if is_swap:
                    apply_swap(current, i, j)
                else:
                    current[i:j + 1] = reversed(current[i:j + 1])
                current_score += delta
                if swap_selector is not None:
                    swap_selector.invalidate()
                stalled = 0
                if current_score > best_score:
                    best, best_score = tuple(current), current_score
                    if on_improvement is not None:
                        on_improvement(best_score, evaluations)
            else:
                stalled += 1
            if stalled >= stall_limit:
                local_reason = "stall"
                break
            if diagnostics is not None:
                diagnostics.snapshot(evaluations, best_score)
        if evaluations >= evaluation_budget and local_reason != "bound_reached":
            local_reason = "evaluation_budget"
        restart_history.append(dict(restart=restarts, attempts=attempts, accepted=accepted,
                                    final_score=current_score, reason=local_reason))
        if diagnostics is not None:
            diagnostics.counts[f"restart_exit_{local_reason}"] += 1
        if stop_reason in ("time_limit", "bound_reached"):
            break
    assert best is not None
    if evaluations >= evaluation_budget and stop_reason != "bound_reached":
        stop_reason = "evaluation_budget"
    if diagnostics is not None:
        diagnostics.snapshot(evaluations, best_score, force=True)
    return {"circle": canonical_circle(best), "score": best_score,
            "evaluations": evaluations, "restarts": restarts, "stop_reason": stop_reason,
            "restart_policy": restart_policy, "restart_history": restart_history}


def run_inversion_hill_climber(
    distances: DistanceMatrix, rng: Random,
    *, evaluation_budget: int, inversion_rate: float = 0.8,
    max_restarts: int = 200, attempts_per_restart: int = 1000, stall_limit: int = 200,
    restart_policy: str = "productive",
    on_improvement: Callable[[int, int], None] | None = None,
    deadline: float | None = None, diagnostics: SearchDiagnostics | None = None,
    score_bound: int | None = None,
) -> dict:
    """Bias toward inversions, retaining a nonzero swap probability.

    0.5 reproduces uniform HC; 1 is deliberately excluded so swaps are retained.
    Score checks and restart/deadline/bound semantics match `run_hill_climber`.
    """
    if not 0.5 <= inversion_rate < 1:
        raise ValueError("Inversion-focused rate must be in [0.5, 1); swaps must remain possible")
    return _run_hill_climber(
        distances, rng, evaluation_budget=evaluation_budget, max_restarts=max_restarts,
        attempts_per_restart=attempts_per_restart, stall_limit=stall_limit,
        restart_policy=restart_policy, inversion_rate=inversion_rate,
        on_improvement=on_improvement, deadline=deadline, diagnostics=diagnostics,
        score_bound=score_bound,
    )


def run_guided_hill_climber(
    distances: DistanceMatrix, rng: Random,
    *, evaluation_budget: int, max_restarts: int = 200,
    attempts_per_restart: int = 1000, stall_limit: int = 200,
    exploration: float = 0.2,
    on_improvement: Callable[[int, int], None] | None = None,
    deadline: float | None = None, diagnostics: SearchDiagnostics | None = None,
    restart_policy: str = "productive", score_bound: int | None = None,
) -> dict:
    """Run delta-scored HC with weak-edge-guided swaps.

    `exploration` is the fraction of swap proposals chosen uniformly; 1 gives
    baseline HC behavior. Score checks include restart scores and all proposed
    moves. The returned dictionary matches `run_hill_climber`.
    `on_improvement(score, evaluations)` reports the first scored circle and
    each strict global improvement, with the same semantics as baseline HC.
    `score_bound` stops at a proven upper bound, as in `run_hill_climber`.
    """
    validate_distances(distances)
    selector = WeakEdgeSwapSelector(distances, exploration=exploration, diagnostics=diagnostics)
    return _run_hill_climber(
        distances, rng, evaluation_budget=evaluation_budget, max_restarts=max_restarts,
        attempts_per_restart=attempts_per_restart, stall_limit=stall_limit,
        swap_selector=selector,
        on_improvement=on_improvement,
        deadline=deadline, diagnostics=diagnostics,
        restart_policy=restart_policy, score_bound=score_bound,
    )

