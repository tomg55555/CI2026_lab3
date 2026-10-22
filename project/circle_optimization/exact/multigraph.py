"""Multi-start heuristic on the profile multigraph (see profiles.py).

Each start builds a randomized greedy b-matching (highest distances first),
merges its components, and applies best-improvement 2-exchanges. A 2-exchange
replaces edges (a, b), (c, d) with (a, c), (b, d) or (a, d), (b, c): degrees
are preserved, so only connectivity must be rechecked. Unlike a 2-opt move on a
single permutation, both reconnections are allowed whenever the profile graph
stays connected, since an Euler circuit can re-thread the children.

The state is m x m regardless of N. Starts stop early at a proven bound.
"""

from time import perf_counter

import numpy as np

from .profiles import ProfileInstance, components, multigraph_score, profile_upper_bound


def greedy_multigraph(instance: ProfileInstance, rng: np.random.Generator,
                      *, noise: float = 0.0) -> np.ndarray:
    """Saturate profile pairs by decreasing distance + uniform noise; rest are loops.

    Residual degrees start even and each pair takes the smaller one, so they
    stay even: leftover degree always becomes self-loops. May be disconnected.
    """
    m, d = instance.m, instance.distances
    x = np.zeros((m, m), dtype=np.int64)
    residual = (2 * instance.counts).tolist()
    p, q = np.triu_indices(m, 1)
    keys = d[p, q] + noise * rng.random(len(p))
    order = np.lexsort((rng.random(len(p)), -keys))  # Random tie-breaking.
    for a, b in zip(p[order].tolist(), q[order].tolist()):
        k = min(residual[a], residual[b])
        if k:
            x[a, b] = x[b, a] = k
            residual[a] -= k
            residual[b] -= k
    x[np.diag_indices(m)] = np.array(residual) // 2
    return x


def _edges(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Used edge types (a <= b), loops included."""
    return np.nonzero(np.triu(x))


def _exchange(x, move, units=1) -> None:
    """Remove `units` copies of (a, b) and (c, d); add (a, c), (b, d) or (a, d), (b, c).

    Negative `units` reverts the same move.
    """
    a, b, c, d, crossed = move
    for u, v, k in ((a, b, -1), (c, d, -1)) + (((a, d, 1), (b, c, 1)) if crossed else ((a, c, 1), (b, d, 1))):
        x[u, v] += units * k
        if u != v:
            x[v, u] += units * k


def _deltas(d, a1, b1, a2, b2):
    """Gains of both reconnections for every pair of edges from two lists."""
    removed = d[a1, b1][:, None] + d[a2, b2][None, :]
    straight = d[a1[:, None], a2[None, :]] + d[b1[:, None], b2[None, :]] - removed
    crossed = d[a1[:, None], b2[None, :]] + d[b1[:, None], a2[None, :]] - removed
    return straight, crossed


def connect(x: np.ndarray, distances: np.ndarray) -> None:
    """Merge components in place by cheapest cross-component 2-exchanges.

    A move that removes bridges in both components can leave two components;
    it is reverted and the next-best move tried, so each accepted move merges.
    """
    groups = components(x)
    while len(groups) > 1:
        label = np.empty(len(x), dtype=np.intp)
        for index, group in enumerate(groups):
            label[group] = index
        smallest = min(range(len(groups)), key=lambda g: len(groups[g]))
        a, b = _edges(x)
        inside = label[a] == smallest
        straight, crossed = _deltas(distances, a[inside], b[inside], a[~inside], b[~inside])
        gains = np.concatenate([straight.ravel(), crossed.ravel()])
        for flat in np.argsort(-gains, kind="stable"):
            swap, rest = divmod(int(flat), straight.size)
            i, j = divmod(rest, straight.shape[1])
            move = (a[inside][i], b[inside][i], a[~inside][j], b[~inside][j], bool(swap))
            _exchange(x, move)
            merged = components(x)
            if len(merged) < len(groups):
                groups = merged
                break
            _exchange(x, move, -1)
        else:
            raise RuntimeError("No merging 2-exchange found")  # Unreachable for valid input.


def improve(x: np.ndarray, distances: np.ndarray, *, deadline: float | None = None,
            block: int = 2048, max_moves: int | None = None, stats: dict | None = None) -> int:
    """Best-improvement bulk 2-exchange local search in place; returns moves applied.

    The gain is linear in the copies moved, so each move transfers as many
    copies as both removed edge types hold (one fewer if deleting a type would
    disconnect the graph); moves are ranked by total gain. A positive gain
    implies the added types differ from the removed ones (Hamming distances
    obey the triangle inequality), so bulk moves preserve degrees. The number
    of moves therefore does not grow with N. Gains are computed in row blocks
    of at most `block` edge types, bounding memory at block x E.

    `max_moves` caps the moves applied (None: until a local optimum or the
    deadline). If `stats` is given, `stats["checks"]` is incremented by the
    number of edge-type pairs whose gains were computed, the unit charged to
    budgets by the memetic GA layer (layered/memetic.py).
    """
    moves = 0
    while (deadline is None or perf_counter() < deadline) and (max_moves is None or moves < max_moves):
        a, b = _edges(x)
        copies = x[a, b]
        best = None
        for start in range(0, len(a), block):
            rows = slice(start, start + block)
            straight, crossed = _deltas(distances, a[rows], b[rows], a, b)
            if stats is not None:
                stats["checks"] = stats.get("checks", 0) + straight.size
            gains = np.maximum(straight, crossed)
            # Each unordered pair once; a pair with itself never gains.
            gains[np.arange(gains.shape[0])[:, None] + start >= np.arange(len(a))[None, :]] = 0
            units = np.minimum(copies[rows][:, None], copies[None, :])
            total = gains * units
            positive = np.flatnonzero(total > 0)
            for flat in positive[np.argsort(-total.ravel()[positive], kind="stable")]:
                i, j = divmod(int(flat), total.shape[1])
                if best is not None and total[i, j] <= best[0]:
                    break
                move = (a[start + i], b[start + i], a[j], b[j], bool(crossed[i, j] > straight[i, j]))
                k = int(units[i, j])
                _exchange(x, move, k)
                if (x[move[0], move[1]] == 0 or x[move[2], move[3]] == 0) and len(components(x)) > 1:
                    k -= 1  # Keep one copy of each removed type: still connected.
                _exchange(x, move, -int(units[i, j]))
                if k:
                    best = (int(gains[i, j]) * k, move, k)
                    break
        if best is None:
            return moves
        _exchange(x, best[1], best[2])
        moves += 1
    return moves


def multi_start(instance: ProfileInstance, *, starts: int = 16, seed: int = 0,
                noise: float = 1.5, deadline: float | None = None,
                target: int | None = None, on_improvement=None) -> dict:
    """Run greedy + connect + 2-exchange from several randomized starts.

    Start 0 is pure greedy (random ties); others add `noise` to distances. Stop
    after `starts`, at `deadline` (absolute perf_counter, at least one start),
    or once a start reaches `target` (default: the proven profile bound).
    `on_improvement(score, starts_completed)` fires on each new best start.
    Return the best connected multigraph, its score and the stop reason.
    """
    if starts < 1:
        raise ValueError("At least one start is required")
    target = profile_upper_bound(instance) if target is None else target
    rng = np.random.default_rng(seed)
    best, best_score, history = None, -1, []
    stop_reason = "start_limit"
    for start in range(starts):
        if best is not None and deadline is not None and perf_counter() >= deadline:
            stop_reason = "time_limit"
            break
        x = greedy_multigraph(instance, rng, noise=0.0 if start == 0 else noise)
        if instance.m > 1:
            connect(x, instance.distances)
            moves = improve(x, instance.distances, deadline=deadline)
        else:
            moves = 0
        score = multigraph_score(x, instance.distances)
        history.append(dict(start=start, score=score, moves=moves))
        if score > best_score:
            best, best_score = x, score
            if on_improvement is not None:
                on_improvement(best_score, start + 1)
        if best_score >= target:
            stop_reason = "bound_reached"
            break
    return dict(multiplicity=best, score=best_score, starts=len(history),
                history=history, stop_reason=stop_reason)
