"""Optional, deliberately expensive diagnostics for fixed-check replay runs.

Never enabled in the benchmark's timed runs: hashing every proposed circle
would otherwise hide the benefit of constant-time delta scoring in HC.
"""

from collections import Counter
from functools import lru_cache
from itertools import chain
from math import lgamma, log, log10
from operator import getitem

from ..problem import Genome, canonical_circle


@lru_cache(maxsize=16)
def _edge_ids(n: int) -> tuple[tuple[int, ...], ...]:
    """Map both directions of an undirected pair to one integer edge id."""
    return tuple(tuple(min(a, b) * n + max(a, b) for b in range(n)) for a in range(n))


class SearchDiagnostics:
    """Collect proposal novelty, coverage, counters, and population snapshots."""

    def __init__(self, n: int, snapshot_interval: int = 200, *, family="HC") -> None:
        """Empty counters for n children; snapshot at most every `snapshot_interval` checks."""
        self.n = n
        self.family = family
        self.snapshot_interval = snapshot_interval
        self.counts = Counter()
        self.proposed: set[Genome] = set()
        self.scored: set[Genome] = set()
        self.edges: set[int] = set()
        self.snapshots: list[dict] = []
        self._next_snapshot = 0
        self._ids = _edge_ids(n)

    def _edges(self, circle):
        """Edge ids of a circle, invariant to its direction and rotation."""
        return map(getitem, map(self._ids.__getitem__, circle), circle[1:] + circle[:1])

    def proposal(self, circle, *, scored=False, changed_edges=None) -> None:
        """Record a proposed circle.

        `changed_edges` optionally lists positions k whose edge (k, k+1) may be
        new; the caller guarantees every other edge was already observed, as
        for an HC move from a recorded current circle.
        """
        self.counts["proposals"] += 1
        key = canonical_circle(tuple(circle))
        known = len(self.proposed)
        self.proposed.add(key)
        if len(self.proposed) == known:
            pass  # A repeated proposal's edges are already recorded.
        elif changed_edges is None:
            self.edges.update(self._edges(key))
        else:
            n = len(circle)
            self.edges.update(self._ids[circle[k]][circle[(k + 1) % n]] for k in changed_edges)
        if scored:
            self.scored.add(key)

    def evaluation(self, circle) -> None:
        """Record a scored circle; its edges are new only if it was never proposed."""
        key = canonical_circle(tuple(circle))
        known = len(self.scored)
        self.scored.add(key)
        # GA scores offspring proposed a generation earlier; their edges are known.
        if len(self.scored) != known and key not in self.proposed:
            self.edges.update(self._edges(key))

    def move(self, kind: str, delta: int) -> None:
        """Count one move attempt of `kind` with score change `delta`."""
        self.counts[f"{kind}_attempts"] += 1
        self.counts[f"{kind}_net_gain"] += delta
        if delta > 0:
            self.counts[f"{kind}_accepted"] += 1
            self.counts[f"{kind}_gain"] += delta

    def snapshot(self, evaluations: int, score: int, population=None, *, force=False) -> None:
        """Record coverage (and population diversity) at `evaluations` if due or forced."""
        if not force and evaluations < self._next_snapshot:
            return
        self._next_snapshot = evaluations + self.snapshot_interval
        row = dict(evaluations=evaluations, score=score,
                   unique_proposals=len(self.proposed), unique_scored=len(self.scored),
                   edge_coverage=len(self.edges) / (self.n * (self.n - 1) / 2))
        if population:
            p = len(population)
            frequencies = Counter(chain.from_iterable(map(self._edges, population)))
            # Mean fraction of edges not shared between two distinct members.
            row["edge_diversity"] = (1 - sum(c * (c - 1) for c in frequencies.values())
                                     / (p * (p - 1) * self.n)) if p > 1 else 0.0
            total = sum(frequencies.values())
            row["edge_entropy"] = (-sum((c / total) * log(c / total)
                                         for c in frequencies.values())
                                   / log(self.n * (self.n - 1) / 2))
        self.snapshots.append(row)

    def summary(self) -> dict:
        """All counters plus novelty, coverage, acceptance, gain and diversity rates."""
        c = self.counts
        proposals = c["proposals"]
        log_space = (lgamma(self.n) - log(2)) / log(10)
        result = dict(c)
        result.update(
            unique_proposals=len(self.proposed), unique_scored=len(self.scored),
            novelty_rate=len(self.proposed) / proposals if proposals else None,
            edge_coverage=len(self.edges) / (self.n * (self.n - 1) / 2),
            log10_circle_coverage=log10(len(self.scored)) - log_space if self.scored else None,
        )
        for kind in ("swap", "inversion"):
            attempts, accepted = c[f"{kind}_attempts"], c[f"{kind}_accepted"]
            result[f"{kind}_improvement_rate"] = accepted / attempts if attempts else None
            result[f"{kind}_acceptance"] = accepted / attempts if attempts and self.family.startswith("HC") else None
            result[f"{kind}_gain_per_proposal"] = c[f"{kind}_gain"] / attempts if attempts else None
            result[f"{kind}_net_gain_per_proposal"] = c[f"{kind}_net_gain"] / attempts if attempts else None
        calls = c["fitness_calls"]
        result["cache_hit_rate"] = c["cache_hits"] / calls if calls else None
        result["duplicate_rate"] = c["duplicates"] / proposals if proposals else None
        swaps = c["uniform_swap_proposals"] + c["guided_swap_proposals"]
        result["actual_uniform_swap_fraction"] = c["uniform_swap_proposals"] / swaps if swaps else None
        if self.snapshots:
            for key in ("edge_diversity", "edge_entropy"):
                result[key] = self.snapshots[-1].get(key)
        return result
