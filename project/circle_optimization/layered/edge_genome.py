"""Layer S4 (phase 12): edge genomes, edge-preserving crossover and 2-exchange mutation.

A circle's score depends only on its adjacencies, and with interchangeable
children only on how many edges join each pair of profiles (phase 07). The
edge GA therefore evolves the edge-count table itself: a genome is a
connected multiplicity matrix X (m x m, symmetric, int64, loops on the
diagonal, degree 2 n_p per profile), the representation `exact.profiles`
validates and turns into a circle by an Euler circuit. Order crossover (OX)
on child permutations inherits relative *order*; the operators here inherit
*edges*, in the tradition of edge recombination (ERX), GPX and EAX for the TSP.

Levels of the crossover (`fill`), see docs/history/phases/12-edge-ga-memetic.md:

- A, "random": common edges, then the parents' other edges in random order,
  then random completion and minimal repair. A pure GA operator.
- B, "greedy": same, but the parents' edges and the completion prefer long
  edges (decreasing distance, random ties). Still a GA operator.
- C (memetic) adds local search per child; it lives in `memetic.py`.

All operators cost O(m^2) numpy work plus Python loops over at most the
edge types involved (<= min(m^2, 2N)); nothing per child is O(N) except
`random_genome`. Randomness comes from a `numpy.random.Generator`.
"""

import numpy as np

from ..exact.multigraph import _exchange, connect
from ..exact.profiles import components, multigraph_score
from .state import ProfileState

FILLS = ("random", "greedy")


def random_genome(state: ProfileState, rng: np.random.Generator) -> np.ndarray:
    """Edge table of a uniformly random circle (random order of the profile multiset).

    A random permutation of the n_p copies of each profile is the profile
    sequence of a uniformly random child circle; O(N) numpy work.
    """
    counts = state.instance.counts
    sequence = rng.permutation(np.repeat(np.arange(state.m), counts))
    return state.multigraph(sequence)


def genome_key(x: np.ndarray) -> bytes:
    """Hashable identity of a genome: sorted flat indices and counts of its used types.

    Only the upper triangle (loops included) is stored, as int32 indices
    followed by int32 counts, so the key has O(min(m^2, N)) size even when
    m is large and the table sparse.
    """
    flat = np.flatnonzero(np.triu(x))
    return np.concatenate((flat, x.ravel()[flat])).astype(np.int32).tobytes()


def _decode(key: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `genome_key`: (flat upper-triangle indices, counts)."""
    values = np.frombuffer(key, dtype=np.int32)
    half = len(values) // 2
    return values[:half], values[half:]


def key_fitness(key: bytes, state: ProfileState) -> int:
    """Score of the genome with this key, O(used types), without rebuilding it.

    Each upper-triangle type is counted once and loops have distance 0, so
    this equals `genome_fitness` of the genome.
    """
    flat, counts = _decode(key)
    return int(counts.astype(np.int64) @ state.instance.distances.ravel()[flat])


def genome_fitness(x: np.ndarray, state: ProfileState) -> int:
    """Circle score of an edge genome (`exact.profiles.multigraph_score`), O(m^2)."""
    return multigraph_score(x, state.instance.distances)


def edge_diversity(keys: list[bytes], winner: int) -> float:
    """Mean number of edge copies (out of N) of each genome absent from the winner.

    The edge analogue of the classic positional diversity: 0 for a converged
    population, N for genomes sharing no edge with the best one.
    """
    best_flat, best_counts = _decode(keys[winner])
    total = int(best_counts.sum())
    missing = 0
    for key in keys:
        flat, counts = _decode(key)
        _, mine, theirs = np.intersect1d(flat, best_flat, assume_unique=True, return_indices=True)
        missing += total - int(np.minimum(counts[mine], best_counts[theirs]).sum())
    return missing / len(keys)


def _fill_order(lengths: np.ndarray, rng: np.random.Generator, fill: str) -> np.ndarray:
    """Processing order of candidate types: uniform random, or longest first with random ties."""
    if fill == "random":
        return rng.permutation(len(lengths))
    return np.lexsort((rng.random(len(lengths)), -lengths))


def _saturate(x: np.ndarray, residual: list[int], p: np.ndarray, q: np.ndarray,
              limit: np.ndarray | None) -> None:
    """Add copies of types (p, q) in order while residual degrees allow; in place.

    Type k receives min(limit[k], residual[p], residual[q]) copies (a loop
    needs two residual units per copy). With `limit=None` (distinct pairs
    only) each pair is saturated and the loop ends once at most one profile
    keeps residual degree. O(len(p)) Python steps; one numpy update.
    """
    rows, cols, takes = [], [], []
    live = sum(1 for r in residual if r)
    bounds = limit.tolist() if limit is not None else None
    for k, (a, b) in enumerate(zip(p.tolist(), q.tolist())):
        if bounds is None:
            if live <= 1:
                break
            take = min(residual[a], residual[b])
        elif a == b:
            take = min(bounds[k], residual[a] // 2)
        else:
            take = min(bounds[k], residual[a], residual[b])
        if not take:
            continue
        if a == b:
            residual[a] -= 2 * take
        else:
            residual[a] -= take
            residual[b] -= take
            live -= (residual[a] == 0) + (residual[b] == 0)
        rows.append(a)
        cols.append(b)
        takes.append(take)
    if takes:
        rows, cols, takes = np.array(rows), np.array(cols), np.array(takes, dtype=np.int64)
        x[rows, cols] += takes  # Each type occurs at most once per call.
        off = rows != cols
        x[cols[off], rows[off]] += takes[off]


def edge_crossover(xa: np.ndarray, xb: np.ndarray, state: ProfileState, rng: np.random.Generator,
                   *, fill: str = "random") -> np.ndarray:
    """Child edge genome built from the parents' edges; valid and connected.

    1. Keep every edge common to both parents: min(xa, xb), loops included.
    2. Add the parents' other edges, |xa - xb| copies per type, in random
       order (`fill="random"`, level A) or by decreasing distance with random
       ties (`fill="greedy"`, level B), as far as residual degrees allow.
    3. Complete residual degree with distinct-profile pairs (random order or
       longest first), saturating each pair; leftover degree becomes loops.
    4. Merge components with `exact.multigraph.connect` (minimal repair:
       cheapest degree-preserving 2-exchanges between components).

    Why the degrees always work out: let r_p = 2 n_p - deg(p) be the residual
    degree (a loop counts twice). Step 1 keeps a sub-multigraph of a parent, so
    r_p >= 0, and every later copy is added only within residuals. The total
    sum of r_p is 2N - 2|E|: even, and every added copy (loop or not) lowers it
    by 2, so it stays even. Step 3 visits each pair of profiles still having
    residual once and adds min(r_p, r_q) copies, zeroing one of them; residuals
    never grow, so afterwards at most one profile p has r_p > 0, and r_p then
    equals the even total: exactly r_p / 2 loops at p close every degree.
    `connect` preserves degrees, so the child passes `validate_multigraph`.

    Cost: O(m^2) numpy, O(T) Python steps for the T differing or completing
    types (T <= min(m^2, 2N)), plus `connect` (one O(E^2) gain matrix per
    merged component). `rng` is a numpy Generator.
    """
    if fill not in FILLS:
        raise ValueError(f"fill must be one of {FILLS}")
    distances = state.instance.distances
    child = np.minimum(xa, xb)
    residual = (2 * state.instance.counts - child.sum(axis=1) - child.diagonal()).tolist()
    p, q = np.nonzero(np.triu(xa != xb))
    if len(p):
        order = _fill_order(distances[p, q], rng, fill)
        p, q = p[order], q[order]
        _saturate(child, residual, p, q, np.abs(xa[p, q] - xb[p, q]))
    alive = np.flatnonzero(residual)
    if len(alive) > 1:
        i, j = np.triu_indices(len(alive), 1)
        p, q = alive[i], alive[j]
        order = _fill_order(distances[p, q], rng, fill)
        _saturate(child, residual, p[order], q[order], None)
    child[np.diag_indices(state.m)] += np.array(residual, dtype=np.int64) // 2
    if state.m > 1:
        connect(child, distances)
    return child


def edge_mutation(x: np.ndarray, state: ProfileState, rng: np.random.Generator,
                  *, attempts: int = 10) -> np.ndarray:
    """Copy of x after one random degree-preserving 2-exchange that keeps it connected.

    Two distinct edge copies (a, b), (c, d) are drawn uniformly among the N
    copies (loops included) and replaced by (a, c), (b, d) or (a, d), (b, c),
    each with probability 1/2: the multigraph analogue of a 2-opt inversion
    (and of a swap, which on a circle is two such exchanges). Moves that
    change nothing, or disconnect the graph by removing a bridge type, are
    retried up to `attempts` times; then x is returned unchanged (as a copy).
    O(m^2) numpy plus one component check when a removed type vanishes.
    """
    child = x.copy()
    a, b = np.nonzero(np.triu(child))
    cumulative = np.cumsum(child[a, b])
    total = int(cumulative[-1])
    if state.m < 2 or total < 2:
        return child  # One profile: every genome is the same set of loops.
    for _ in range(attempts):
        i = int(rng.integers(total))
        j = int(rng.integers(total - 1))
        j += j >= i
        e, f = np.searchsorted(cumulative, (i, j), side="right").tolist()
        u, v, s, t = int(a[e]), int(b[e]), int(a[f]), int(b[f])
        crossed = bool(rng.integers(2))
        added = ((u, t), (v, s)) if crossed else ((u, s), (v, t))
        if sorted(map(sorted, added)) == sorted(map(sorted, ((u, v), (s, t)))):
            continue  # Null move: the same two types come back.
        move = (u, v, s, t, crossed)
        _exchange(child, move)
        if ((u != v and child[u, v] == 0) or (s != t and child[s, t] == 0)) and len(components(child)) > 1:
            _exchange(child, move, -1)
            continue
        return child
    return child
