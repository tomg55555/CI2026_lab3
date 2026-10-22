"""Layer S2 (phase 09): profile-id state with an m x m distance table.

Children with equal profiles are interchangeable (phase 07): a circle's score
depends only on the profiles along it. A layered search therefore stores a
circle as its *sequence of profile ids* and looks distances up in the m x m
profile table instead of the N x N child matrix.

- Memory: O(N + m^2) instead of O(N^2). N = 10^5 needs 10^10 matrix cells in
  the classic eras but only a 2^C x 2^C table here.
- Speed: lookups hit a table of at most 2^C rows that stays in CPU cache.
- Semantics: a move's score change is identical, since D[prof[a]][prof[b]]
  equals the child distance. Children are assigned to profile visits only when
  a result is returned (`ProfileState.circle`).

The table and the child -> profile map are stored as tuples of Python ints:
scalar indexing of numpy arrays would make every lookup several times slower.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

import numpy as np

from ..classic.operators import random_individual
from ..exact.profiles import ProfileInstance, aggregate_profiles, assign_children
from ..problem import Genome, circle_fitness

ProfileSequence = list[int]


@dataclass(frozen=True, eq=False)
class ProfileState:
    """Profile aggregation plus pure-Python lookup tables for search loops.

    `instance` keeps the numpy representation used by the exact solver and the
    multigraph layers; `table[p][q]` is the Hamming distance between profiles
    p and q; `profile_of[child]` is the profile id of each child.
    """

    instance: ProfileInstance
    table: tuple[tuple[int, ...], ...]
    profile_of: tuple[int, ...]

    @classmethod
    def from_instance(cls, instance: ProfileInstance) -> "ProfileState":
        """Wrap an existing aggregation (shares its arrays, copies to Python ints)."""
        return cls(instance=instance,
                   table=tuple(map(tuple, instance.distances.tolist())),
                   profile_of=tuple(instance.child_profile.tolist()))

    @classmethod
    def from_profiles(cls, profiles) -> "ProfileState":
        """Aggregate an N x C Boolean table (sequences or array) and wrap it."""
        return cls.from_instance(aggregate_profiles(profiles))

    @property
    def n(self) -> int:
        """Number of children N."""
        return self.instance.n

    @property
    def m(self) -> int:
        """Number of distinct profiles m."""
        return self.instance.m

    @property
    def characteristics(self) -> int:
        """Number of Boolean characteristics C."""
        return self.instance.characteristics

    def sequence(self, genome: Sequence[int]) -> ProfileSequence:
        """Profile ids along a child circle (same order, same length)."""
        profile_of = self.profile_of
        return [profile_of[child] for child in genome]

    def random_sequence(self, rng: Random) -> ProfileSequence:
        """Profiles of `random_individual(N, rng)`: the classic eras' random start.

        Consumes exactly the same random draws, so a layered search seeded like
        a classic one starts from the same circle (up to child relabeling).
        """
        return self.sequence(random_individual(self.n, rng))

    def score(self, sequence: Sequence[int]) -> int:
        """Circular score of a profile sequence, wraparound edge included."""
        return circle_fitness(tuple(sequence), self.table) if len(sequence) > 1 else 0

    def genome_score(self, genome: Sequence[int]) -> int:
        """Score of a child circle computed through the profile table."""
        return self.score(self.sequence(genome))

    def circle(self, sequence: Sequence[int]) -> Genome:
        """Canonical child circle realizing a profile sequence (same score).

        The k-th visit of profile p receives p's k-th child; child 0 is
        rotated to the front. Raises if the sequence's profile counts differ
        from the instance's.
        """
        array = np.asarray(sequence, dtype=np.intp)
        if len(array) != self.n or (np.bincount(array, minlength=self.m) != self.instance.counts).any():
            raise ValueError("A profile sequence must visit each profile exactly n_p times")
        return assign_children(array, self.instance.child_profile)

    def multigraph(self, sequence: Sequence[int]) -> np.ndarray:
        """Edge-count matrix X of a profile sequence (loops on the diagonal).

        X[p, q] counts circle edges between profiles p != q; X[p, p] counts
        edges joining two children of p. This is the phase-07 representation
        used by the exact solver and the edge-genome GA (layer S4).
        """
        array = np.asarray(sequence, dtype=np.intp)
        x = np.zeros((self.m, self.m), dtype=np.int64)
        np.add.at(x, (array, np.roll(array, -1)), 1)
        x = x + x.T
        x[np.diag_indices(self.m)] //= 2
        return x
