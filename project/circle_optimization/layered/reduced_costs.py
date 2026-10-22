"""Layer S3 (phase 11): LP reduced costs as move guidance for 2-edge moves.

The connected LP relaxation of the phase-07 model (`exact.milp.lp_relaxation`)
prices every profile pair p-q (and every loop p-p) with a reduced cost rc_pq.
For any circle with edge counts x and the LP optimum x*,

    score(x) <= LP + sum_pq rc_pq (x_pq - x*_pq) = LP - certificate(x),

where rc < 0 only at x* = 0 and rc > 0 only at the column's upper bound, so
every term of the certificate is >= 0. Each circle edge on a pair with
rc < 0 therefore costs at least |rc| against the bound, and a circle
reaching the bound uses only pairs with rc >= 0 (up to the integrality
slack). Only 10-27% of the pairs qualify on uniform instances (phase 07).

`ReducedCostGuide.select` proposes the two cut positions of an inversion or
splice in two steps, without any per-move maintenance:

1. Worst edge: an O(k) tournament over k uniform circle edges, ranked by
   penalty -rc (edges on pairs with rc >= 0 rank below every penalized edge,
   then by lower distance, the phase-02 weak-edge idea).
2. Partner: one endpoint of that edge is kept; the second cut is placed so
   the endpoint's new neighbour has a profile q with rc >= 0 for the pair.
   q is drawn from the endpoint's precomputed candidate list and located by
   the first occurrence at or after a uniform offset (C-level `list.index`),
   so the new edge always joins an LP-compatible pair. Position rejection
   sampling would miss often: some profiles have 2 candidates out of 32.
   Draws whose cut edges would be adjacent are retried (bounded).

With probability `exploration` (and whenever no pair is penalized) the
proposal is uniform, as in the base engine. Selection proposes; strict
improvement still decides acceptance. Sampling computes no delta, so it costs
no score check. Construction solves one LP: 3-13 ms at m = 32, about 35 ms
at m = 128 (C = 7).
"""

from collections.abc import Sequence
from random import Random

import numpy as np

from ..classic.operators import uniform_swap_positions
from .state import ProfileState

TOLERANCE = 1e-7


def reduced_cost_table(state: ProfileState, *, seed: int = 0, threads: int | None = None) -> dict:
    """LP bound, m x m reduced costs and values for a profile state.

    Thin wrapper of `exact.milp.lp_relaxation` (requires highspy); see it for
    the keys. Reduced costs with |rc| <= 1e-7 are set to exactly 0.
    """
    from ..exact.milp import lp_relaxation

    lp = lp_relaxation(state.instance, seed=seed, threads=threads)
    lp["reduced_costs"] = np.where(np.abs(lp["reduced_costs"]) <= TOLERANCE, 0.0, lp["reduced_costs"])
    return lp


class ReducedCostGuide:
    """Reduced-cost-guided position selection for inversions and splices.

    `reduced_costs` and `values` are m x m (diagonal: loops) as returned by
    `reduced_cost_table`; `table` is the profile distance table; `bound` the
    LP bound. `exploration` is the probability of a uniform proposal;
    `tournament` (k >= 1) the number of sampled edges competing for "worst";
    `partner_tries` bounds the retries when a partner cut is adjacent.
    """

    def __init__(self, reduced_costs, values, table: Sequence[Sequence[int]], *,
                 bound: float, exploration: float = 0.2, tournament: int = 8,
                 partner_tries: int = 4) -> None:
        """Validate settings and precompute pure-Python tables (costs, badness, partners)."""
        if not 0 <= exploration <= 1:
            raise ValueError("Exploration probability must lie between 0 and 1")
        if tournament < 1 or partner_tries < 1:
            raise ValueError("Tournament size and partner tries must be positive")
        rc = np.asarray(reduced_costs, dtype=np.float64)
        d = np.asarray(table, dtype=np.float64)
        self.bound = float(bound)
        self.exploration = exploration
        self.tournament = tournament
        self.partner_tries = partner_tries
        # Certificate = sum rc x* - sum over circle edges of rc (see module doc).
        self.offset = float(np.triu(rc * np.asarray(values, dtype=np.float64)).sum())
        # Python floats/ints: numpy scalar lookups would dominate the loop.
        self.reduced_costs = tuple(map(tuple, rc.tolist()))
        self.badness = tuple(map(tuple, np.where(rc < 0, -rc, -d / (d.max(initial=0) + 1)).tolist()))
        self.partners = tuple(tuple(np.flatnonzero(row >= 0).tolist()) for row in rc)
        # Nothing to guide toward: one profile, or no edge is ever penalized.
        self.active = len(rc) > 1 and bool((rc < 0).any())

    @classmethod
    def from_state(cls, state: ProfileState, *, exploration: float = 0.2, tournament: int = 8,
                   partner_tries: int = 4, seed: int = 0, threads: int | None = None) -> "ReducedCostGuide":
        """Solve the instance's LP relaxation (highspy) and build the guide."""
        lp = reduced_cost_table(state, seed=seed, threads=threads)
        return cls(lp["reduced_costs"], lp["values"], state.table, bound=lp["bound"],
                   exploration=exploration, tournament=tournament, partner_tries=partner_tries)

    def select(self, sequence: Sequence[int], rng: Random, *, crossed: bool = False) -> tuple[int, int]:
        """Cut positions 1 <= i < j <= N-1 for an inversion (or a splice if `crossed`).

        The move removes edges i-1 and j (edge e joins positions e, e+1 mod N).
        The inversion adds left-left and right-right endpoint pairs of the two
        removed edges, the splice left-right and right-left, whichever edge
        comes first; the partner edge f is placed accordingly. Cost: O(k +
        partner_tries) lookups plus C-level scans; requires N >= 3 and a
        list or tuple `sequence`.
        """
        n = len(sequence)
        if not self.active or n < 5 or rng.random() < self.exploration:
            return tuple(sorted(uniform_swap_positions(n, rng)))
        random = rng.random  # int(random() * n): uniform enough, 4x cheaper than randrange.
        badness = self.badness
        edge = int(random() * n)
        worst = badness[sequence[edge]][sequence[(edge + 1) % n]]
        for _ in range(self.tournament - 1):
            other = int(random() * n)
            value = badness[sequence[other]][sequence[(other + 1) % n]]
            if value > worst:
                edge, worst = other, value
        side = 1 if random() < 0.5 else 0  # 0 keeps the left endpoint seq[e], 1 the right one.
        partners = self.partners[sequence[(edge + side) % n]]
        offset = side ^ crossed  # The new neighbour seq[t] sits at f = t - offset.
        index = sequence.index
        for _ in range(self.partner_tries):
            t = int(random() * n)
            if partners:
                profile = partners[int(random() * len(partners))]
                try:
                    t = index(profile, t)
                except ValueError:
                    try:
                        t = index(profile)
                    except ValueError:  # Absent profile (partial sequences): keep t.
                        pass
            f = (t - offset) % n
            low, high = (edge, f) if edge < f else (f, edge)
            if high - low >= 2 and (crossed or low or high != n - 1):
                return low + 1, high  # Not adjacent, and not a full reversal.
        return tuple(sorted(uniform_swap_positions(n, rng)))

    def certificate(self, sequence: Sequence[int]) -> float:
        """Proven gap lower bound: score(sequence) <= bound - certificate (O(N)).

        Equals sum over circle edges of -rc plus the constant sum rc x*; every
        circle edge on a pair with rc < 0 contributes its |rc|. Zero only if
        the circle could still reach the LP bound.
        """
        rc = self.reduced_costs
        n = len(sequence)
        return self.offset - sum(rc[sequence[k]][sequence[(k + 1) % n]] for k in range(n))


def gap_certificate(guide: ReducedCostGuide, sequence: Sequence[int]) -> float:
    """Function form of `ReducedCostGuide.certificate`."""
    return guide.certificate(sequence)
