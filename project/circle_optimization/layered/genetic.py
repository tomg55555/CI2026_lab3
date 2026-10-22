"""Layered GA engines: GA-P (S2, phase 09) and the edge GA, GA-E/GA-M (S4, phase 12).

Both reuse the classic generation loop `classic.genetic.evolve` (tournament
selection on known scores, crossover with `crossover_rate`, one mutation with
`mutation_rate`, elitism, per-generation duplicate rejection, fitness cache,
budget/deadline/bound stops), so they differ from the classic GA only in what
the module documents:

- `run_profile_ga` (GA-P, S2): classic child genomes, OX and swap/inversion,
  but fitness through the m x m profile table (no N x N matrix) and the cache
  and duplicate rejection keyed by the profile sequence of the canonical
  genome, so relabelings of same-profile children collapse. With all profiles
  distinct it reproduces `run_ga` exactly.
- `run_edge_ga` (S4): genomes are connected profile multigraphs X with
  edge-preserving crossover (`edge_genome.edge_crossover`) and a 2-exchange
  mutation. `local_search=None` is GA-E, a pure GA; a `memetic.MemeticLayer`
  makes it GA-M.
- `seeding` (S6, phase 14): the fraction of the initial population built by
  the greedy constructors of `restarts.py` instead of uniformly at random.

Seed construction and the crossover's connectivity repair are not charged to
`evaluation_budget`; equal-time comparisons charge them as wall time.
"""

from collections.abc import Callable
from math import factorial
from random import Random
from time import perf_counter

import numpy as np

from ..classic.genetic import check_ga_settings, evolve
from ..classic.operators import mutate, order_crossover, random_individual
from ..exact.profiles import multigraph_circle
from ..problem import Genome, canonical_circle, circle_fitness
from .edge_genome import FILLS, edge_crossover, edge_diversity, edge_mutation, genome_key, key_fitness, random_genome
from .memetic import MemeticLayer
from .restarts import seed_multigraph, seed_sequence
from .state import ProfileState


def _seed_count(seeding: float, population_size: int) -> int:
    """Number of seeded individuals: round(seeding * population_size), validated."""
    if not 0 <= seeding <= 1:
        raise ValueError("Seeding must lie between 0 and 1")
    return round(seeding * population_size)


def _expired(deadline: float | None) -> bool:
    """True once an absolute perf_counter deadline has passed."""
    return deadline is not None and perf_counter() >= deadline


def _profile_population(state: ProfileState, rng: Random, size: int, seeds: int,
                        deadline: float | None) -> list[Genome]:
    """Distinct canonical child genomes: up to `seeds` greedy ones, the rest random.

    Seeds are `restarts.seed_sequence` circles realized by `state.circle`;
    10 * seeds attempts at most, since small instances may have fewer distinct
    greedy circles. With seeds = 0 this consumes exactly the draws of
    `classic.operators.initial_population`. A deadline can return fewer
    (at least one) individuals.
    """
    n = state.n
    # (n - 1)! / 2 unique circles; only small n can be exceeded (avoids huge factorials).
    if n < 3 or size < 1 or (n <= 20 and size > factorial(n - 1) // 2):
        raise ValueError("Population must fit in the unique circles for n >= 3")
    population: list[Genome] = []
    seen: set[Genome] = set()
    for _ in range(10 * seeds):
        if len(population) >= seeds or (population and _expired(deadline)):
            break
        genome = state.circle(seed_sequence(state, rng))
        if genome not in seen:
            population.append(genome)
            seen.add(genome)
    while len(population) < size:
        if population and _expired(deadline):
            break
        genome = random_individual(n, rng)
        if genome not in seen:
            population.append(genome)
            seen.add(genome)
    return population


def run_profile_ga(
    state: ProfileState, rng: Random, *, population_size: int, generations: int,
    tournament_size: int, elites: int, crossover_rate: float, mutation_rate: float,
    evaluation_budget: int, seeding: float = 0.0,
    on_improvement: Callable[[int, int], None] | None = None,
    deadline: float | None = None, score_bound: int | None = None,
) -> dict:
    """Classic GA scored through the profile table (GA-P, layer S2, phase 09).

    Parameters, operators and stop semantics are those of `classic.genetic.
    run_ga`. Differences: fitness is `circle_fitness` of the profile sequence
    on `state.table`; the cache and duplicate rejection use the profile
    sequence of the canonical genome as key, so `evaluations` counts distinct
    profile sequences; if an instance has fewer distinct keys than
    `population_size`, duplicates are accepted after one failed random
    fallback. `seeding` (S6) is the fraction of the initial population built
    from `restarts.seed_sequence` (greedy GRASP circles, noise 1.5); 0 keeps
    the classic random population and its random draws.

    Memory O(N * population + m^2); per offspring O(N) as in the classic GA.
    Returns `circle` (canonical child genome), `score`, `sequence` (its
    profile ids), `evaluations`, `history`, `population`, `stop_reason`,
    `generations_completed` and `layers`.
    """
    settings = dict(population_size=population_size, generations=generations,
                    tournament_size=tournament_size, elites=elites, crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate, evaluation_budget=evaluation_budget)
    check_ga_settings(**settings)
    seeds = _seed_count(seeding, population_size)
    n, table, profile_of = state.n, state.table, state.profile_of
    population = _profile_population(state, rng, population_size, seeds, deadline)
    cache: dict[tuple[int, ...], int] = {}

    def key(genome: Genome) -> tuple[int, ...]:
        """Profile sequence of a canonical genome."""
        return tuple(map(profile_of.__getitem__, genome))

    def score(sequence: tuple[int, ...]) -> int:
        """Cached circular score of a profile sequence."""
        value = cache.get(sequence)
        if value is None:
            value = cache[sequence] = circle_fitness(sequence, table)
        return value

    result = evolve(
        population, rng, score=score, work=cache.__len__, key=key,
        crossover=order_crossover, mutation=mutate, finish=canonical_circle,
        random_genome=lambda rng: random_individual(n, rng), duplicate_limit=1,
        on_improvement=on_improvement, deadline=deadline, score_bound=score_bound, **settings,
    )
    best = result["best"]
    return {"circle": best, "score": result["score"], "sequence": list(key(best)),
            "evaluations": len(cache), "history": result["history"], "population": result["population"],
            "stop_reason": result["stop_reason"], "generations_completed": result["generations_completed"],
            "layers": ("profile-state", "ox") + (("seeded",) if seeds else ())}


def _edge_population(state: ProfileState, rng: Random, generator: np.random.Generator, size: int,
                     seeds: int, deadline: float | None) -> list[np.ndarray]:
    """Edge genomes with distinct keys: up to `seeds` greedy ones, the rest random circles.

    Seeds come from `restarts.seed_multigraph` (10 * seeds attempts at most),
    the rest from `edge_genome.random_genome`. After 10 * size duplicates in
    total (instances with few distinct tables) duplicates are accepted.
    A deadline can return fewer (at least one) genomes.
    """
    population: list[np.ndarray] = []
    seen: set[bytes] = set()
    for _ in range(10 * seeds):
        if len(population) >= seeds or (population and _expired(deadline)):
            break
        x = seed_multigraph(state, rng)
        if (key := genome_key(x)) not in seen:
            population.append(x)
            seen.add(key)
    duplicates = 0
    while len(population) < size:
        if population and _expired(deadline):
            break
        x = random_genome(state, generator)
        if (key := genome_key(x)) in seen and duplicates < 10 * size:
            duplicates += 1
            continue
        population.append(x)
        seen.add(key)
    return population


def run_edge_ga(
    state: ProfileState, rng: Random, *, population_size: int, generations: int,
    tournament_size: int, elites: int, crossover_rate: float, mutation_rate: float,
    evaluation_budget: int, fill: str = "random", local_search: MemeticLayer | None = None,
    seeding: float = 0.0, on_improvement: Callable[[int, int], None] | None = None,
    deadline: float | None = None, score_bound: int | None = None,
) -> dict:
    """Generational GA on connected profile multigraphs (GA-E / GA-M, layer S4, phase 12).

    Same loop as `run_ga` (via `classic.genetic.evolve`): elitism, tournament
    selection on known scores, `edge_genome.edge_crossover` with
    `crossover_rate` (`fill`: "random" = level A, "greedy" = level B), one
    `edge_mutation` (random connected 2-exchange) with `mutation_rate`,
    duplicate rejection by `genome_key`, cached fitness. `local_search`:
    None (GA-E, pure GA) or a `MemeticLayer` applied to every offspring
    (GA-M, level C). `seeding` (S6): fraction of the initial population
    from `restarts.seed_multigraph`; the rest are random circles' tables.

    Evaluation unit (charged to `evaluation_budget`, reported in `history`,
    `on_improvement` and `evaluations`): distinct genome fitness computations
    plus, with a memetic layer, its checks (one per 2-exchange gain of a pair
    of edge types). The generation-level check reserves `population_size`
    units for the next generation's scores and each local-search step must fit
    in what remains, so `evaluations <= evaluation_budget` always holds.
    `deadline` and `score_bound` behave as in `run_ga`; local search also
    stops at the deadline. Offspring are scored as soon as they are accepted
    (`evolve(score_offspring=True)`), so children improved by local search
    still count if time runs out mid-generation. Operators draw from a numpy Generator seeded by one
    `rng.getrandbits(64)`; selection draws from `rng`.

    Per offspring O(m^2) numpy plus O(T) Python (see `edge_crossover`); the
    Euler circuit (O(m^2 + N)) is built once, for the returned circle.
    Returns `circle`, `score`, `multigraph`, `evaluations`,
    `fitness_evaluations`, `memetic_checks`, `history` (diversity = mean edge
    copies not shared with the generation's best), `population`,
    `stop_reason`, `generations_completed` and `layers`.
    """
    settings = dict(population_size=population_size, generations=generations,
                    tournament_size=tournament_size, elites=elites, crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate, evaluation_budget=evaluation_budget)
    check_ga_settings(**settings)
    if fill not in FILLS:
        raise ValueError(f"fill must be one of {FILLS}")
    if state.n < 3:
        raise ValueError("The edge GA needs at least 3 children")
    seeds = _seed_count(seeding, population_size)
    generator = np.random.default_rng(rng.getrandbits(64))
    distances = state.instance.distances
    population = _edge_population(state, rng, generator, population_size, seeds, deadline)
    cache: dict[bytes, int] = {}
    checks = [0]

    def score(key: bytes) -> int:
        """Cached fitness of the genome with this key."""
        value = cache.get(key)
        if value is None:
            value = cache[key] = key_fitness(key, state)
        return value

    def work() -> int:
        """Budget units used: distinct fitness computations plus memetic checks."""
        return len(cache) + checks[0]

    def finish(child: np.ndarray) -> np.ndarray:
        """Memetic step (copy of the child, improved within the remaining budget)."""
        if local_search is None:
            return child
        child = child.copy()  # The child may still be a parent object.
        limit = max(0, evaluation_budget - population_size - work())
        checks[0] += local_search.improve(child, distances, generator, limit=limit, deadline=deadline)
        return child

    result = evolve(
        population, rng, score=score, work=work, key=genome_key, diversity=edge_diversity,
        crossover=lambda a, b, _: edge_crossover(a, b, state, generator, fill=fill),
        mutation=lambda x, _: edge_mutation(x, state, generator), finish=finish,
        random_genome=lambda _: random_genome(state, generator), duplicate_limit=1, score_offspring=True,
        on_improvement=on_improvement, deadline=deadline, score_bound=score_bound, **settings,
    )
    best = result["best"]
    layers = (("profile-state", "edge-crossover") + (("greedy-fill",) if fill == "greedy" else ())
              + (("memetic",) if local_search is not None else ()) + (("seeded",) if seeds else ()))
    return {"circle": multigraph_circle(best, state.instance), "score": result["score"], "multigraph": best,
            "evaluations": work(), "fitness_evaluations": len(cache), "memetic_checks": checks[0],
            "history": result["history"], "population": result["population"],
            "stop_reason": result["stop_reason"], "generations_completed": result["generations_completed"],
            "layers": layers}
