"""Layer S4, optional memetic part (phase 12): local search on each GA offspring.

The edge GA (`genetic.run_edge_ga`) is a pure GA without this layer (GA-E).
Passing a `MemeticLayer` as `local_search` improves every offspring, after
crossover and mutation and before duplicate rejection, with the phase-07 bulk
2-exchange local search (`exact.multigraph.improve`); the improved genome
replaces the child (Lamarckian). That is level C, a memetic algorithm (GA-M).

The layer is a separate object so the same GA, seed and settings can run with
and without it. Its work unit is the *check*: one 2-exchange gain computed
for a pair of edge types (`improve`'s `stats["checks"]`). One local-search
step over E used edge types costs exactly E^2 checks, so a step is started
only if it fits the caller's remaining budget: GA-M budgets stay hard limits.
"""

from time import perf_counter

import numpy as np

from ..exact.multigraph import improve


class MemeticLayer:
    """Bulk 2-exchange local search applied to GA offspring (level C).

    `max_moves` caps the improving moves per child (None: until a local
    optimum); `probability` is the chance that a child is improved at all.
    """

    def __init__(self, max_moves: int | None = None, probability: float = 1.0) -> None:
        """Validate and store the per-child move cap and application probability."""
        if max_moves is not None and max_moves < 1:
            raise ValueError("max_moves must be positive or None")
        if not 0 <= probability <= 1:
            raise ValueError("probability must lie between 0 and 1")
        self.max_moves = max_moves
        self.probability = probability

    def __repr__(self) -> str:
        """Show the layer's settings."""
        return f"MemeticLayer(max_moves={self.max_moves}, probability={self.probability})"

    def improve(self, x: np.ndarray, distances: np.ndarray, rng: np.random.Generator,
                *, limit: int | None = None, deadline: float | None = None) -> int:
        """Improve edge genome x in place; return the number of checks used.

        Runs `exact.multigraph.improve` one move at a time (identical to one
        call with `max_moves`) so that a step costing E^2 checks is started
        only while `checks + E^2 <= limit` (None: unlimited). Stops at a local
        optimum, after `max_moves` moves, at the budget limit or at `deadline`
        (absolute perf_counter). Degrees and connectivity are preserved.
        Draws one number from `rng` only when `probability < 1`.
        """
        if self.probability < 1 and rng.random() >= self.probability:
            return 0
        checks = moves = 0
        while self.max_moves is None or moves < self.max_moves:
            if deadline is not None and perf_counter() >= deadline:
                break
            cost = np.count_nonzero(np.triu(x)) ** 2
            if limit is not None and checks + cost > limit:
                break
            stats = {}
            applied = improve(x, distances, deadline=deadline, max_moves=1, stats=stats)
            checks += stats.get("checks", 0)
            if not applied:
                break
            moves += 1
        return checks
