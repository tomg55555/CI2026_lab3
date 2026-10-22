"""Layer S1 (phase 10): split-and-splice, the second 2-exchange reconnection.

Take positions 1 <= i < j <= N-1 of a profile sequence and the two circle
edges they cut, a-b = (seq[i-1], seq[i]) and c-d = (seq[j], seq[(j+1) % N]).
Removing both leaves two paths, b..c = seq[i..j] and d..a (the rest). There
are two ways to reconnect them with two new edges:

- a-c, b-d: one circle, the classic 2-opt reversal of seq[i..j] (inversion).
- a-d, b-c: two loops, L1 = seq[i..j] closed by c-b and L2 = seq[j+1..] +
  seq[..i-1] closed by a-d. A permutation of children cannot use it directly.

Profiles make the second reconnection usable. Suppose some profile p occurs at
u in L1 and at v in L2, with loop successors u' and v'. Exchanging the
successors (u -> v', v -> u') merges both loops into one circle again. It
removes edges (p, u') and (p, v') and adds (p, v') and (p, u'): the same two
profile pairs, so the score does not change. The whole move therefore scores

    delta = D[a][d] + D[b][c] - D[a][b] - D[c][d]          (O(1), no rebuild)

and keeps the profile multiset; it is the "crossed" exchange that the
phase-07 multigraph search (`exact.multigraph._exchange`) already allows when
the profile graph stays connected. In child terms, the children of L1 are
re-threaded into the circle next to a child of the same profile. The merge
is only possible when the loops share a profile; for the uniform instances
of the suite (m <= 2^C << N) that is almost always the case.

`splice_delta` costs O(1); `apply_splice` costs O(N) in C-level scans and
slices and is called only for accepted moves. Edge cases: j = i + 1 makes L1
a 2-cycle (two b-c edges); i = 1, j = N-1 makes L2 the single element seq[0]
closed by a loop (a = d).
"""

from collections.abc import Sequence
from random import Random

from .state import ProfileSequence

Table = Sequence[Sequence[int]]


def splice_delta(sequence: Sequence[int], table: Table, i: int, j: int) -> int:
    """Score change of the a-d, b-c reconnection at positions i < j (O(1)).

    Valid only if `apply_splice` can merge the loops; the engine checks that
    when it accepts the move. Pairs with `classic.operators.inversion_delta`.
    """
    a, b = sequence[i - 1], sequence[i]
    c, d = sequence[j], sequence[(j + 1) % len(sequence)]
    return table[a][d] + table[b][c] - table[a][b] - table[c][d]


def splice_loops(sequence: Sequence[int], i: int, j: int) -> tuple[list[int], list[int]]:
    """The two loops L1 = seq[i..j] and L2 = seq[j+1..] + seq[..i-1].

    Each is listed in circle order; its closing edge joins its last element
    to its first (c-b for L1, a-d for L2).
    """
    return list(sequence[i:j + 1]), list(sequence[j + 1:]) + list(sequence[:i])


def _occurrence(loop: list[int], profile: int, rng: Random) -> int:
    """First index of `profile` at or after a random offset, cyclically (C-level scans)."""
    offset = int(rng.random() * len(loop))
    try:
        return loop.index(profile, offset)
    except ValueError:
        return loop.index(profile)


def shared_profile(inner: list[int], outer: list[int], rng: Random, *, probes: int = 8) -> int | None:
    """A profile occurring in both loops, or None if they share none.

    Probes up to `probes` uniform elements of the shorter loop for membership
    in the longer one (each a C-level scan); on uniform instances the first
    probe almost always succeeds. Otherwise its distinct profiles are tested
    in increasing order, so the result depends only on `rng`.
    """
    shorter, longer = (inner, outer) if len(inner) <= len(outer) else (outer, inner)
    random = rng.random
    for _ in range(probes):
        profile = shorter[int(random() * len(shorter))]
        if profile in longer:
            return profile
    return next((profile for profile in sorted(set(shorter)) if profile in longer), None)


def apply_splice(sequence: Sequence[int], i: int, j: int, rng: Random) -> ProfileSequence | None:
    """New sequence after the split-and-splice move, or None if impossible.

    Merges L1 into L2 at a shared profile p (`shared_profile`): u and v are
    occurrences of p in L1 and L2 found from random offsets, and L1's
    direction is drawn too (both directions give the same edges). All
    choices come from `rng`. The result starts with L2's first element; its
    score is the old score plus `splice_delta`. O(N) with C-level scans and
    slices only; the engine calls it for accepted moves only.
    """
    inner, outer = splice_loops(sequence, i, j)
    profile = shared_profile(inner, outer, rng)
    if profile is None:
        return None
    v, u = _occurrence(outer, profile, rng), _occurrence(inner, profile, rng)
    if rng.random() < 0.5:
        inner.reverse()
        u = len(inner) - 1 - u
    # outer..v, then L1 from u's successor around to u, then v's successor..
    return outer[:v + 1] + inner[u + 1:] + inner[:u + 1] + outer[v + 1:]
