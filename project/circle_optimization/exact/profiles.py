"""Aggregate children by Boolean profile and map profile multigraphs to circles.

Children with equal profiles are interchangeable: a circle's score depends only
on how many of its edges join each pair of profiles. A circle is therefore a
connected multigraph on the m distinct profiles in which profile p has degree
2 * n_p (self-loops join two children of p and count twice). Every such
multigraph has an Euler circuit visiting p exactly n_p times, so it maps back
to a circle with the same score. See docs/history/phases/07-profile-milp.md.

Multiplicities are stored as a symmetric m x m integer matrix X: X[p, q] edges
between profiles p != q and X[p, p] self-loops. With m <= min(N, 2**C), all
structures here are O(m^2); only aggregation and reconstruction touch all N.
"""

from dataclasses import dataclass

import numpy as np

from ..problem import Genome, canonical_circle


@dataclass(frozen=True, eq=False)
class ProfileInstance:
    """Distinct profiles, their multiplicities and the child -> profile map."""

    rows: np.ndarray           # (m, C) bool: distinct profiles
    counts: np.ndarray         # (m,) int64: children per profile, all >= 1
    child_profile: np.ndarray  # (N,) intp: profile index of each child
    distances: np.ndarray      # (m, m) int64: Hamming distances between profiles

    @property
    def n(self) -> int:
        """Number of children N."""
        return len(self.child_profile)

    @property
    def m(self) -> int:
        """Number of distinct profiles m."""
        return len(self.counts)

    @property
    def characteristics(self) -> int:
        """Number of Boolean characteristics C."""
        return self.rows.shape[1]


def aggregate_profiles(profiles) -> ProfileInstance:
    """Group an N x C Boolean table (sequences or array) by distinct profile.

    Up to 64 characteristics, rows are packed into integer codes; a dense
    histogram replaces sorting when 2**C is small relative to N. Distances use
    one m x m matrix product: d(p, q) = |p| + |q| - 2 p.q.
    """
    table = np.asarray(profiles, dtype=bool)
    if table.ndim != 2 or 0 in table.shape:
        raise ValueError("Profiles must form a nonempty N x C Boolean table")
    n, c = table.shape
    if c <= 64:
        packed = np.packbits(table, axis=1, bitorder="little")
        codes = np.pad(packed, ((0, 0), (0, 8 - packed.shape[1]))).view("<u8").ravel()
        if c <= 24 and 1 << c <= 2 * n:
            histogram = np.bincount(codes.astype(np.intp), minlength=1 << c)
            unique = np.flatnonzero(histogram)
            lookup = np.empty(1 << c, dtype=np.intp)
            lookup[unique] = np.arange(len(unique))
            inverse, counts = lookup[codes.astype(np.intp)], histogram[unique]
            unique = unique.astype(np.uint64)
        else:
            unique, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
        rows = ((unique[:, None] >> np.arange(c, dtype=np.uint64)) & np.uint64(1)).astype(bool)
    else:
        rows, inverse, counts = np.unique(table, axis=0, return_inverse=True, return_counts=True)
    weight = rows.sum(axis=1)
    dense = rows.astype(np.float64)
    # Float64 products of 0/1 rows are exact far beyond any practical C.
    distances = np.rint(weight[:, None] + weight[None, :] - 2 * (dense @ dense.T)).astype(np.int64)
    return ProfileInstance(rows=rows, counts=counts.astype(np.int64),
                           child_profile=inverse.ravel().astype(np.intp), distances=distances)


def profile_upper_bound(instance: ProfileInstance) -> int:
    """Proven bound min(N*C, feature, degree), rounded down to even, in O(m^2 + mC).

    Same relaxations as `benchmark.metrics.quality_bound` without the N x N
    matrix: each feature flips at most 2*min(true, false) times; each child's
    two edges are at most its two largest distances to other children.
    """
    n, c, counts, d = instance.n, instance.characteristics, instance.counts, instance.distances
    true = counts @ instance.rows
    bound = min(n * c, int((2 * np.minimum(true, n - true)).sum()))
    if n >= 3 and instance.m >= 2:
        others = np.where(np.eye(instance.m, dtype=bool), -1, d)
        first = others.max(axis=1)
        at_first = ((others == first[:, None]) * counts[None, :]).sum(axis=1)
        second = np.where(others < first[:, None], others, -1).max(axis=1)
        second = np.where(counts >= 2, np.maximum(second, 0), second)  # a same-profile neighbour
        second = np.where(at_first >= 2, first, second)
        bound = min(bound, int((counts * (first + second)).sum()) // 2)
    elif instance.m == 1:
        bound = 0
    return bound - bound % 2


def multigraph_score(multiplicity: np.ndarray, distances: np.ndarray) -> int:
    """Total distance of a profile multigraph; self-loops score zero."""
    return int((multiplicity * distances).sum()) // 2


def components(multiplicity: np.ndarray) -> list[np.ndarray]:
    """Profile index groups connected by edges (loops ignored), via BFS layers."""
    support = multiplicity > 0
    label = np.full(len(multiplicity), -1)
    groups = []
    for root in range(len(multiplicity)):
        if label[root] >= 0:
            continue
        label[root] = len(groups)
        members, frontier = [root], [root]
        while frontier:
            reached = np.flatnonzero(support[frontier].any(axis=0) & (label < 0))
            label[reached] = len(groups)
            members.extend(reached.tolist())
            frontier = reached.tolist()
        groups.append(np.array(members))
    return groups


def validate_multigraph(multiplicity: np.ndarray, instance: ProfileInstance) -> None:
    """Require symmetric nonnegative counts, degree 2*n_p and one component."""
    x = multiplicity
    if x.shape != (instance.m, instance.m) or (x < 0).any() or (x != x.T).any():
        raise ValueError("Multiplicities must form a symmetric nonnegative m x m matrix")
    if ((x.sum(axis=1) + x.diagonal()) != 2 * instance.counts).any():
        raise ValueError("Every profile p needs degree 2 * n_p")
    if instance.m > 1 and len(components(x)) != 1:
        raise ValueError("The profile multigraph must be connected")


def euler_sequence(multiplicity: np.ndarray) -> np.ndarray:
    """Profile visits of an Euler circuit, one per edge, in O(m^2 + N).

    Hierholzer runs on a reduced multigraph that keeps 1 (odd) or 2 (even)
    copies of each used pair: same support, same degree parity. The removed
    copies return as back-and-forth excursions p, q, p, ... and self-loops as
    repeated visits, spliced in at the profile's first visit.
    """
    x = np.asarray(multiplicity, dtype=np.int64)
    m = len(x)
    if m == 1:
        return np.zeros(int(x[0, 0]), dtype=np.intp)
    upper = np.triu(x, 1)
    keep = np.where(upper > 0, 2 - upper % 2, 0)
    digons = (upper - keep) // 2
    remaining = (keep + keep.T).tolist()
    position = [0] * m
    stack, circuit = [int(np.flatnonzero(upper.any(axis=1) | upper.any(axis=0))[0])], []
    while stack:
        v = stack[-1]
        row, i = remaining[v], position[v]
        while i < m and not row[i]:
            i += 1
        position[v] = i
        if i == m:
            circuit.append(stack.pop())
        else:
            row[i] -= 1
            remaining[i][v] -= 1
            stack.append(i)
    pieces, expanded = [], np.zeros(m, dtype=bool)
    for v in circuit[:-1]:  # The circuit is closed: drop the repeated start.
        pieces.append(np.array([v], dtype=np.intp))
        if not expanded[v]:
            expanded[v] = True
            for q in np.flatnonzero(digons[v]):
                pieces.append(np.tile(np.array([q, v], dtype=np.intp), int(digons[v, q])))
            if x[v, v]:
                pieces.append(np.full(int(x[v, v]), v, dtype=np.intp))
    return np.concatenate(pieces)


def assign_children(sequence: np.ndarray, child_profile: np.ndarray) -> Genome:
    """Give the k-th visit of profile p to p's k-th child; canonical genome.

    Two stable sorts (radix for small integer keys) align visits and children
    profile by profile. The caller guarantees matching profile counts.
    """
    circle = np.empty(len(sequence), dtype=np.intp)
    circle[np.argsort(sequence, kind="stable")] = np.argsort(child_profile, kind="stable")
    circle = np.roll(circle, -int(np.flatnonzero(circle == 0)[0]))
    return canonical_circle(tuple(circle.tolist()))


def sequence_score(sequence: np.ndarray, distances: np.ndarray) -> int:
    """Circular score of a profile visit sequence, wraparound edge included."""
    return int(distances[sequence, np.roll(sequence, -1)].sum()) if len(sequence) > 1 else 0


def multigraph_circle(multiplicity: np.ndarray, instance: ProfileInstance) -> Genome:
    """Validate a multigraph and turn it into a circle of child indices."""
    validate_multigraph(multiplicity, instance)
    sequence = euler_sequence(multiplicity)
    assert len(sequence) == instance.n
    assert sequence_score(sequence, instance.distances) == multigraph_score(multiplicity, instance.distances)
    return assign_children(sequence, instance.child_profile)
