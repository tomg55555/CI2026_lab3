"""Represent and score circular permutations with symmetric pairwise weights."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import permutations
from operator import getitem
from random import Random
from typing import TypeVar

Genome = tuple[int, ...]
DistanceMatrix = tuple[tuple[int, ...], ...]
T = TypeVar("T")


@dataclass(eq=False, frozen=True)
class Child:
    """A named child with Boolean characteristics; identity is by instance.

    Algorithms use indices, so equal characteristics never merge two children.
    """

    name: str
    characteristics: tuple[bool, ...]

    def __init__(self, name: str, characteristics: Sequence[bool]) -> None:
        """Store a child's name and an immutable copy of its characteristics."""
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "characteristics", tuple(characteristics))

    def difference(self, other: "Child") -> int:
        """Return the Hamming distance to another child."""
        return hamming_distance(self.characteristics, other.characteristics)

    def __str__(self) -> str:
        """Display the profile with '+' for true and '-' for false."""
        return f"{self.name}:{''.join('+' if c else '-' for c in self.characteristics)}"


def hamming_distance(left: Sequence[bool], right: Sequence[bool]) -> int:
    """Count differing positions in two equally long Boolean profiles.

    Raises ValueError if the profiles have different lengths.
    """
    if len(left) != len(right):
        raise ValueError("Profiles must have the same number of characteristics")
    return sum(a != b for a, b in zip(left, right))


def pairwise_distances(items: Sequence[T], difference: Callable[[T, T], int]) -> DistanceMatrix:
    """Build a pairwise score matrix from indexed items and a distance function.

    The caller must supply a symmetric, nonnegative distance function. Item
    indices, rather than item equality, distinguish otherwise identical items.
    """
    return tuple(tuple(difference(left, right) for right in items) for left in items)


def random_profiles(seed: int, n: int, characteristics: int) -> tuple[tuple[bool, ...], ...]:
    """Reproducible random Boolean profiles, one getrandbits(1) per characteristic.

    Sizes are nested: the first n profiles of a larger instance with the same
    seed equal the instance of size n. The benchmark suite, CLI and guided tour
    all use this generator, so the same (seed, n, C) is the same instance.
    """
    rng = Random(seed)
    return tuple(tuple(bool(rng.getrandbits(1)) for _ in range(characteristics)) for _ in range(n))


def distance_matrix(profiles: Sequence[Sequence[bool]]) -> DistanceMatrix:
    """Build a Hamming-distance matrix from Boolean item profiles."""
    return pairwise_distances(profiles, hamming_distance)


def validate_distances(distances: DistanceMatrix) -> None:
    """Require a nonempty square, symmetric, nonnegative pairwise matrix."""
    n = len(distances)
    if n == 0 or any(len(row) != n for row in distances):
        raise ValueError("Distances must form a nonempty square matrix")
    if any(value < 0 or value != distances[j][i]
           for i, row in enumerate(distances) for j, value in enumerate(row)):
        raise ValueError("Distances must be symmetric and nonnegative")


def canonical_circle(genome: Genome) -> Genome:
    """Choose one direction for a circle whose first item is already fixed.

    For example, (0, 3, 2, 1) and (0, 1, 2, 3) share the canonical
    representation (0, 1, 2, 3). This does not rotate arbitrary genomes.
    """
    genome = tuple(genome)
    # Both directions share genome[0], so the second item usually decides.
    if len(genome) < 3 or genome[1] < genome[-1]:
        return genome
    reverse = (genome[0], *reversed(genome[1:]))
    return reverse if genome[1] > genome[-1] else min(genome, reverse)


def valid_circle(genome: Genome, n: int) -> bool:
    """Check that a genome starts with 0 and contains each of n indices once."""
    return n > 0 and len(genome) == n and genome[0] == 0 and set(genome) == set(range(n))


def circle_fitness(genome: Genome, distances: DistanceMatrix) -> int:
    """Sum neighboring pair scores, including the last-to-first edge.

    The genome must contain valid indices into distances. Higher is better.
    """
    # Pair each item's row with its successor without Python-level indexing.
    return sum(map(getitem, map(distances.__getitem__, genome), genome[1:] + genome[:1]))


def unique_circles(n: int) -> Iterator[Genome]:
    """Yield each indexed circle once, fixing index 0 and removing reversals.

    For n >= 3 there are (n - 1)! / 2 such circles. Enumeration is intended
    only for small instances; n must be positive.
    """
    if n < 1:
        raise ValueError("A circle must contain at least one item")
    for tail in permutations(range(1, n)):
        if len(tail) > 1 and tail[0] > tail[-1]:
            continue
        yield (0, *tail)


def exact_best(distances: DistanceMatrix) -> tuple[Genome, int]:
    """Enumerate unique circles and return a proven best genome and fitness.

    Runtime grows factorially; use this as a benchmark only for small n.
    """
    validate_distances(distances)
    return max(((circle, circle_fitness(circle, distances))
                for circle in unique_circles(len(distances))), key=lambda pair: pair[1])
