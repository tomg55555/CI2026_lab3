"""Weak-edge-guided swap proposals for HC and GA mutation (phase 02).

Selection proposes a move; it never decides acceptance. See
docs/GUIDED_SEARCH.md for the distribution, rationale, cost and GA integration.
"""

from bisect import bisect_right
from collections.abc import Callable, Sequence
from itertools import accumulate
from random import Random

from ..problem import DistanceMatrix, Genome, canonical_circle, validate_distances
from .diagnostics import SearchDiagnostics
from .operators import apply_swap, inversion_mutation, uniform_swap_positions


def _excluded_partners(n: int, left: int, right: int, moving: int) -> list[int]:
    """Exclude positions that would keep the selected weak pair adjacent."""
    stationary = right if moving == left else left
    other_neighbor = ((stationary + 1) % n if moving == (stationary - 1) % n
                      else (stationary - 1) % n)
    return sorted({left, right, other_neighbor} - {0})


class WeakEdgeSwapSelector:
    """Sample poor current edges, then move one endpoint elsewhere.

    With probability `exploration`, propose a uniform swap. Otherwise sample
    an edge with weight 1 + max_distance - edge_distance. Swap one movable
    endpoint elsewhere so that the selected pair stops being adjacent. For
    n < 4, use uniform swaps (all pairs remain neighbors in a triangle).

    Cumulative weights are built lazily in O(n); selection is O(log n).
    Call `invalidate` after any change to the circle or at a restart. Rejected
    moves reuse the distribution. The distance matrix must already be valid.
    """

    def __init__(self, distances: DistanceMatrix, *, exploration: float = 0.2,
                 diagnostics: SearchDiagnostics | None = None) -> None:
        """Validate `exploration` and cache the matrix maximum; weights are built lazily."""
        if not 0 <= exploration <= 1:
            raise ValueError("Exploration probability must lie between 0 and 1")
        self.distances = distances
        self.exploration = exploration
        self.max_distance = max(max(row) for row in distances)
        self._cumulative: list[int] | None = None
        self.diagnostics = diagnostics

    def invalidate(self) -> None:
        """Discard weights after a swap, inversion, or change of circle."""
        self._cumulative = None

    def select(self, circle: Sequence[int], rng: Random) -> tuple[int, int]:
        """Return distinct non-anchor positions without modifying the circle."""
        n = len(circle)
        if n < 4 or self.exploration == 1 or rng.random() < self.exploration:
            if self.diagnostics is not None:
                self.diagnostics.counts["uniform_swap_proposals"] += 1
            return uniform_swap_positions(n, rng)
        if self.diagnostics is not None:
            self.diagnostics.counts["guided_swap_proposals"] += 1
        if self._cumulative is None:
            self._cumulative = list(accumulate(
                1 + self.max_distance - self.distances[circle[k]][circle[(k + 1) % n]]
                for k in range(n)
            ))
        edge = bisect_right(self._cumulative, rng.randrange(self._cumulative[-1]))
        left, right = edge, (edge + 1) % n
        if left == 0:
            i = right
        elif right == 0:
            i = left
        else:
            i = left if rng.random() < 0.5 else right

        excluded = _excluded_partners(n, left, right, i)
        if len(excluded) == n - 1:
            # At n=4, one endpoint of a non-anchor edge may have no partner.
            i = right if i == left else left
            excluded = _excluded_partners(n, left, right, i)
        # Sample a uniform rank among the remaining non-anchor positions.
        j = rng.randrange(1, n - len(excluded))
        for position in excluded:
            if j >= position:
                j += 1
        return i, j


def guided_swap_mutation(
    genome: Genome, rng: Random, *, selector: WeakEdgeSwapSelector,
) -> Genome:
    """Propose a guided swap for GA, allowing worsening moves as mutations.

    Reuse a selector across mutations; its weights are invalidated for each
    genome. Unlike HC, GA does not use a local acceptance test here;
    selection/elitism decide which circles survive.
    """
    selector.invalidate()
    i, j = selector.select(genome, rng)
    child = list(genome)
    apply_swap(child, i, j)
    return canonical_circle(tuple(child))


def make_guided_mutation(
    distances: DistanceMatrix, *, exploration: float = 0.2,
    diagnostics: SearchDiagnostics | None = None,
) -> Callable[[Genome, Random], Genome]:
    """Build a GA operator: 50% guided swaps, 50% ordinary inversions.

    Pass the returned callable as `run_ga(..., mutation_operator=...)`.
    Precompute the matrix's maximum only once; rebuild edge weights for each
    genome that undergoes a guided swap. Mutations may worsen fitness.
    """
    validate_distances(distances)
    selector = WeakEdgeSwapSelector(distances, exploration=exploration, diagnostics=diagnostics)

    def mutation(genome: Genome, rng: Random) -> Genome:
        """Guided swap with probability 1/2, otherwise an ordinary inversion."""
        if rng.random() < 0.5:
            return guided_swap_mutation(genome, rng, selector=selector)
        return inversion_mutation(genome, rng, diagnostics=diagnostics)

    return mutation

