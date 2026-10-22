"""Move operators for index genomes: sampling, swaps, inversions and OX.

A genome is a tuple of child indices with child 0 fixed at position 0 (phase 01).
HC uses the O(1) move deltas (`swap_delta`, `inversion_delta`, phase 02) to score
a proposal without building it; GA uses the operators that return new genomes.
`sample_pair` and `tournament_selection` consume exactly the random draws of the
generic `Random.sample` calls they replace (phase 06), so seeded runs are stable.
"""

from collections.abc import Sequence
from math import ceil, factorial, log
from random import Random
from time import perf_counter

from ..problem import DistanceMatrix, Genome, canonical_circle
from .diagnostics import SearchDiagnostics


def sample_pair(size: int, rng: Random) -> tuple[int, int]:
    """Return `tuple(rng.sample(range(size), k=2))` without its generic overhead.

    The same internal draws are consumed in the same order, so seeded searches
    are unchanged. Sizes up to 21 follow `Random.sample`'s pool branch; larger
    sizes follow its rejection-set branch.
    """
    if size < 2:
        return tuple(rng.sample(range(size), k=2))  # Raises the standard error.
    randbelow = rng._randbelow
    first = randbelow(size)
    if size <= 21:
        second = randbelow(size - 1)
        return first, size - 1 if second == first else second
    second = randbelow(size)
    while second == first:
        second = randbelow(size)
    return first, second


def uniform_swap_positions(n: int, rng: Random) -> tuple[int, int]:
    """Choose two distinct non-anchor positions uniformly (n >= 3)."""
    i, j = sample_pair(n - 1, rng)
    return i + 1, j + 1


def apply_swap(circle: list[int], i: int, j: int) -> None:
    """Swap two positions in place; callers keep position 0 anchored."""
    circle[i], circle[j] = circle[j], circle[i]


def swap_delta(circle: Sequence[int], distances: DistanceMatrix, i: int, j: int) -> int:
    """Score change for a swap, counting shared/wraparound edges once."""
    n = len(circle)
    before = after = 0
    for edge in {(i - 1) % n, i, (j - 1) % n, j}:
        next_edge = (edge + 1) % n
        before += distances[circle[edge]][circle[next_edge]]
        left = circle[j] if edge == i else circle[i] if edge == j else circle[edge]
        right = (circle[j] if next_edge == i else
                 circle[i] if next_edge == j else circle[next_edge])
        after += distances[left][right]
    return after - before


def inversion_delta(circle: list[int], distances: DistanceMatrix, i: int, j: int) -> int:
    """Score change when the segment [i, j] is reversed (i < j, 0 stays fixed)."""
    a, b = circle[i - 1], circle[i]
    c, d = circle[j], circle[(j + 1) % len(circle)]
    return distances[a][c] + distances[b][d] - distances[a][b] - distances[c][d]


def random_individual(n: int, rng: Random) -> Genome:
    """Generate a canonical random individual with item 0 fixed in place."""
    if n < 1:
        raise ValueError("A circle must contain at least one item")
    return canonical_circle((0, *rng.sample(range(1, n), k=n - 1)))


def initial_population(n: int, size: int, rng: Random,
                       *, deadline: float | None = None) -> list[Genome]:
    """Generate a population of distinct canonical individuals.

    For n >= 3, size cannot exceed the number of unique circles.
    An optional absolute deadline can return a smaller nonempty population.
    """
    if n < 3 or not 1 <= size <= factorial(n - 1) // 2:
        raise ValueError("Population must fit in the unique circles for n >= 3")
    population: list[Genome] = []
    seen: set[Genome] = set()
    while len(population) < size:
        if population and deadline is not None and perf_counter() >= deadline:
            break
        individual = random_individual(n, rng)
        if individual not in seen:
            population.append(individual)
            seen.add(individual)
    return population


def tournament_selection(
    population: list[Genome], evaluate: Callable[[Genome], int],
    k: int, rng: Random,
) -> Genome:
    """Select the fittest of k distinct sampled individuals as a parent."""
    n = len(population)
    # Random.sample keeps a pool (rather than a set) while n fits this size.
    if not 0 < k <= n or n > 21 + (4 ** ceil(log(k * 3, 4)) if k > 5 else 0):
        return max(rng.sample(population, k=k), key=evaluate)
    # Same draws and sample order as that pool branch; the first maximum wins.
    randbelow = rng._randbelow
    pool = list(population)
    best = best_score = None
    for i in range(k):
        j = randbelow(n - i)
        candidate = pool[j]
        pool[j] = pool[n - i - 1]
        score = evaluate(candidate)
        if best_score is None or score > best_score:
            best, best_score = candidate, score
    return best


def order_crossover(parent_a: Genome, parent_b: Genome, rng: Random) -> Genome:
    """Produce a valid offspring via order crossover (OX) of two parents.

    Copy a tail segment from parent A and fill unoccupied positions with
    parent B's genes in cyclic order. The anchor at index 0 stays fixed.
    """
    a, b = parent_a[1:], parent_b[1:]
    if len(a) < 2:
        return canonical_circle(parent_a)
    start, end = sorted(sample_pair(len(a), rng))
    child = [-1] * len(a)
    child[start:end + 1] = a[start:end + 1]
    used = set(a[start:end + 1])
    # The unoccupied positions are one suffix followed by one prefix. Preserve
    # the original cyclic OX order without modulo/indexing at every gene.
    remaining = [gene for gene in b[end + 1:] + b[:end + 1] if gene not in used]
    suffix_size = len(a) - end - 1
    child[end + 1:] = remaining[:suffix_size]
    child[:start] = remaining[suffix_size:]
    assert -1 not in child  # Both parents are permutations of the same indices.
    return canonical_circle((0, *child))


def swap_mutation(genome: Genome, rng: Random,
                  *, diagnostics: SearchDiagnostics | None = None) -> Genome:
    """Swap two non-anchor genes of an individual and canonicalize it."""
    i, j = uniform_swap_positions(len(genome), rng)
    child = list(genome)
    if diagnostics is not None:
        diagnostics.counts["uniform_swap_proposals"] += 1
    apply_swap(child, i, j)
    return canonical_circle(tuple(child))


def inversion_mutation(genome: Genome, rng: Random,
                       *, diagnostics: SearchDiagnostics | None = None) -> Genome:
    """Reverse a non-anchor segment (a 2-opt move) and canonicalize it."""
    i, j = sorted(uniform_swap_positions(len(genome), rng))
    child = list(genome)
    if diagnostics is not None:
        diagnostics.counts["inversion_proposals"] += 1
    child[i:j + 1] = reversed(child[i:j + 1])
    return canonical_circle(tuple(child))


def mutate(genome: Genome, rng: Random,
           *, diagnostics: SearchDiagnostics | None = None) -> Genome:
    """Apply one equiprobable swap or inversion mutation to an individual."""
    return (swap_mutation if rng.random() < 0.5 else inversion_mutation)(
        genome, rng, diagnostics=diagnostics,
    )

