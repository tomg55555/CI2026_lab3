"""Generational GA over index genomes with cached fitness (phases 01–06).

Tournament selection, order crossover (OX), one swap/inversion mutation per
offspring with `mutation_rate`, elitism and per-generation duplicate rejection.
Phase-04/06 fast paths reuse known scores and consume identical random draws, so
fixed-count trajectories match the original implementation (frozen hashes in
tests/test_ga_optimization.py).

The generation loop itself is `evolve`, a representation-agnostic engine with
injected scoring, keys and operators (extracted in phase 12). `run_ga` wires
it to child genomes and the N x N matrix; the layered GA eras (GA-P, GA-E,
GA-M in `layered/genetic.py`) reuse it with profile keys and edge genomes.
"""

from collections.abc import Callable, Hashable, Sequence
from operator import ne
from random import Random
from time import perf_counter
from typing import Any

from ..problem import DistanceMatrix, Genome, canonical_circle, circle_fitness, validate_distances
from .diagnostics import SearchDiagnostics
from .operators import initial_population, mutate, order_crossover, random_individual, tournament_selection


class FitnessEvaluator:
    """Cache fitness by canonical genome and count distinct evaluations."""

    def __init__(self, distances: DistanceMatrix,
                 diagnostics: SearchDiagnostics | None = None) -> None:
        """Create a fresh cache for this pairwise distance matrix."""
        self.distances = distances
        self.cache: dict[Genome, int] = {}
        self.diagnostics = diagnostics

    def __call__(self, genome: Genome) -> int:
        """Return cached fitness, evaluating a new genome only once."""
        return self.canonical(canonical_circle(genome))

    def canonical(self, key: Genome) -> int:
        """Internal fast path for an already canonical genome; no normalization."""
        if self.diagnostics is not None:
            self.diagnostics.counts["fitness_calls"] += 1
            if key in self.cache:
                self.diagnostics.counts["cache_hits"] += 1
        if key not in self.cache:
            self.cache[key] = circle_fitness(key, self.distances)
            if self.diagnostics is not None:
                self.diagnostics.evaluation(key)
        return self.cache[key]

    @property
    def evaluations(self) -> int:
        """Number of distinct complete-circle fitness computations."""
        return len(self.cache)


def check_ga_settings(*, population_size: int, generations: int, tournament_size: int, elites: int,
                      crossover_rate: float, mutation_rate: float, evaluation_budget: int) -> None:
    """Raise ValueError for GA settings that `evolve` cannot honour."""
    if not 1 <= elites <= population_size or not 1 <= tournament_size <= population_size:
        raise ValueError("Elites and tournament size must fit the population")
    if not 0 <= crossover_rate <= 1 or not 0 <= mutation_rate <= 1:
        raise ValueError("Rates must lie between 0 and 1")
    if generations < 0 or evaluation_budget < population_size:
        raise ValueError("The initial population must fit the evaluation budget")


def positional_diversity(keys: Sequence[Sequence[Hashable]], winner: int) -> float:
    """Mean number of positions at which each key differs from `keys[winner]`."""
    best = keys[winner]
    return sum(sum(map(ne, key, best)) for key in keys) / len(keys)


def evolve(
    population: list, rng: Random, *,
    score: Callable[[Hashable], int], work: Callable[[], int],
    crossover: Callable[[Any, Any, Random], Any], mutation: Callable[[Any, Random], Any],
    finish: Callable[[Any], Any], random_genome: Callable[[Random], Any],
    population_size: int, generations: int, tournament_size: int, elites: int,
    crossover_rate: float, mutation_rate: float, evaluation_budget: int,
    key: Callable[[Any], Hashable] | None = None,
    diversity: Callable[[list, int], float] = positional_diversity,
    on_improvement: Callable[[int, int], None] | None = None,
    deadline: float | None = None, score_bound: int | None = None,
    on_generation: Callable[[int, int, list], None] | None = None,
    on_proposal: Callable[[Any], None] | None = None,
    on_duplicate: Callable[[], None] | None = None,
    duplicate_limit: int | None = None, score_offspring: bool = False,
) -> dict:
    """Generational GA engine shared by every GA era; representation-agnostic.

    Each generation scores the population (`score(key)`, normally cached),
    keeps `elites` best, and fills the rest with offspring: two tournaments on
    the generation's known scores, `crossover(a, b, rng)` with `crossover_rate`
    (else the first parent), `mutation(child, rng)` with `mutation_rate`, then
    `finish(child)` (canonicalization, local search, ...). An offspring whose
    key is already in the next generation is rejected; after 10 *
    `population_size` consecutive rejections one `random_genome(rng)` is
    tried instead. `duplicate_limit` (None = never, the classic behaviour) is
    the number of failed random fallbacks after which duplicates are accepted
    for the rest of the generation; it bounds the loop on instances with fewer
    distinct keys than `population_size`.

    `score_offspring=False` (classic) scores offspring when their generation
    is scored, so a deadline during offspring construction discards them;
    True scores each accepted offspring at once (same cache, same budget
    units; the next generation then only reads cached scores), so expensive
    offspring, e.g. after local search, still count when time runs out.

    `key(genome)` must determine fitness; None means the genome is its own
    key. `work()` is the evaluation count charged to `evaluation_budget`: the
    loop stops once fewer than `population_size` units remain, a conservative
    reserve for the next generation's new scores. `on_improvement(score,
    work())` fires on the first and every strictly better score; `deadline`
    (absolute perf_counter) is checked while scoring and per offspring, but
    at least one genome is scored; reaching `score_bound` stops at once.
    `on_generation(work, best, scored_population)`, `on_proposal(child)` and
    `on_duplicate()` are diagnostics hooks. History rows are (generation,
    best, mean, `diversity(keys, winner_index)`, work). Selection works on
    population indices, which consumes exactly the draws of selecting genomes.

    Returns best genome, score, history, final population, stop reason and
    the last completed generation. Cost per offspring: two tournaments plus
    the injected operators; per generation O(population_size) keys and scores.
    """
    keys = population if key is None else [key(genome) for genome in population]
    best: Any = None
    best_score = -1
    history: list[tuple[int, int, float, float, int]] = []
    stop_reason = "generation_limit"
    completed_generations = 0
    last_scored_population: list = []

    for generation in range(generations + 1):
        scores = []
        for individual, individual_key in zip(population, keys):
            if best is not None and deadline is not None and perf_counter() >= deadline:
                stop_reason = "time_limit"
                break
            value = score(individual_key)
            scores.append(value)
            if value > best_score:
                best, best_score = individual, value
                if on_improvement is not None:
                    on_improvement(best_score, work())
                if score_bound is not None and best_score >= score_bound:
                    stop_reason = "bound_reached"
                    break
        if not scores:
            population = last_scored_population
            stop_reason = "time_limit"
            break
        scored_population = population[:len(scores)]
        last_scored_population = scored_population
        winner = max(range(len(scores)), key=scores.__getitem__)
        history.append((generation, best_score, sum(scores) / len(scores),
                        diversity(keys[:len(scores)], winner), work()))
        if on_generation is not None:
            on_generation(work(), best_score, scored_population)
        if stop_reason == "bound_reached":
            population = scored_population
            break
        if stop_reason == "time_limit" or len(population) < population_size:
            stop_reason = "time_limit"
            population = scored_population
            break
        completed_generations = generation
        if deadline is not None and perf_counter() >= deadline:
            stop_reason = "time_limit"
            break
        if generation == generations:
            break
        if work() > evaluation_budget - population_size:
            stop_reason = "evaluation_budget"
            break

        # Tournaments/ranking only need this generation's already known scores.
        # Indices consume the same draws as genomes; ties stay stable.
        lookup_score = scores.__getitem__
        indices = list(range(len(population)))
        ranked = sorted(indices, key=lookup_score, reverse=True)[:elites]
        following = [population[i] for i in ranked]
        following_keys = [keys[i] for i in ranked]
        seen = set(following_keys)
        duplicate_attempts = failed_fallbacks = 0
        while len(following) < population_size:
            if deadline is not None and perf_counter() >= deadline:
                stop_reason = "time_limit"
                break
            first = population[tournament_selection(indices, lookup_score, tournament_size, rng)]
            second = population[tournament_selection(indices, lookup_score, tournament_size, rng)]
            child = crossover(first, second, rng) if rng.random() < crossover_rate else first
            if rng.random() < mutation_rate:
                child = mutation(child, rng)
            child = finish(child)
            if on_proposal is not None:
                on_proposal(child)
            child_key = child if key is None else key(child)
            accept_duplicates = duplicate_limit is not None and failed_fallbacks >= duplicate_limit
            if child_key in seen and not accept_duplicates:
                if on_duplicate is not None:
                    on_duplicate()
                duplicate_attempts += 1
                if duplicate_attempts < 10 * population_size:
                    continue
                child = random_genome(rng)
                if on_proposal is not None:
                    on_proposal(child)
                child_key = child if key is None else key(child)
                duplicate_attempts = 0
                if child_key in seen:
                    if on_duplicate is not None:
                        on_duplicate()
                    failed_fallbacks += 1
                    continue
            following.append(child)
            following_keys.append(child_key)
            seen.add(child_key)
            duplicate_attempts = 0
            if score_offspring:
                value = score(child_key)
                if value > best_score:
                    best, best_score = child, value
                    if on_improvement is not None:
                        on_improvement(best_score, work())
                    if score_bound is not None and best_score >= score_bound:
                        stop_reason = "bound_reached"
                        break
        if stop_reason in ("time_limit", "bound_reached"):
            break
        population, keys = following, following_keys

    assert best is not None
    return {"best": best, "score": best_score, "history": history, "population": population,
            "stop_reason": stop_reason, "generations_completed": completed_generations}


def run_ga(
    distances: DistanceMatrix, rng: Random,
    *, population_size: int, generations: int, tournament_size: int,
    elites: int, crossover_rate: float, mutation_rate: float,
    evaluation_budget: int,
    mutation_operator: Callable[[Genome, Random], Genome] = mutate,
    on_improvement: Callable[[int, int], None] | None = None,
    deadline: float | None = None, diagnostics: SearchDiagnostics | None = None,
    score_bound: int | None = None,
) -> dict:
    """Evolve a population to maximize circular pairwise fitness.

    Each generation retains `elites` parents, selects the other parents by
    tournament, applies OX with `crossover_rate`, and applies one mutation per
    offspring with `mutation_rate`. Duplicates in a generation are rejected.
    The conservative budget check reserves up to `population_size` new scores
    for the next generation. Returns the best circle, score, actual number of
    distinct evaluations, history tuples (generation, best, mean, positional
    diversity, evaluations), and final population.
    `mutation_operator` defaults to the original uniform swap/inversion mix;
    a custom callable takes (genome, rng) and must return a valid anchored
    permutation. The mutation rate still applies once per offspring.
    Optional `on_improvement(score, evaluations)` reports the first evaluated
    circle and each strict global improvement immediately upon evaluation,
    including within the initial population rather than at generation end.
    `deadline` is an absolute perf_counter time checked during initialization,
    scoring, and offspring construction. At least one circle is scored. A time
    stop can return a partially scored population; its best is still retained.
    `score_bound` is a proven upper bound on fitness, such as an exact optimum.
    Reaching it stops immediately with `bound_reached`: no strict improvement
    can follow, so the best circle, score and improvement trace are unchanged.
    """
    validate_distances(distances)
    n = len(distances)
    settings = dict(population_size=population_size, generations=generations,
                    tournament_size=tournament_size, elites=elites, crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate, evaluation_budget=evaluation_budget)
    check_ga_settings(**settings)
    evaluate = FitnessEvaluator(distances, diagnostics)
    population = initial_population(n, population_size, rng, deadline=deadline)
    hooks = {}
    mutation = mutation_operator
    if diagnostics is not None:
        def swaps() -> int:
            """Swap proposals recorded so far (uniform + guided)."""
            return diagnostics.counts["uniform_swap_proposals"] + diagnostics.counts["guided_swap_proposals"]

        def mutation(child: Genome, rng: Random) -> Genome:
            """The configured mutation, recording its kind and score change."""
            diagnostics.counts["mutations"] += 1
            previous_swaps = swaps()
            mutated = (mutate(child, rng, diagnostics=diagnostics)
                       if mutation_operator is mutate else mutation_operator(child, rng))
            kind = "swap" if swaps() > previous_swaps else "inversion"
            diagnostics.move(kind, circle_fitness(mutated, distances) - circle_fitness(child, distances))
            return mutated

        def on_duplicate() -> None:
            """Count a rejected duplicate offspring."""
            diagnostics.counts["duplicates"] += 1

        hooks = dict(on_proposal=diagnostics.proposal, on_duplicate=on_duplicate,
                     on_generation=lambda work, best, scored: diagnostics.snapshot(work, best, scored, force=True))
    result = evolve(
        population, rng, score=evaluate.canonical, work=evaluate.cache.__len__,
        crossover=order_crossover, mutation=mutation, finish=canonical_circle,
        random_genome=lambda rng: random_individual(n, rng),
        on_improvement=on_improvement, deadline=deadline, score_bound=score_bound,
        **settings, **hooks,
    )
    return {"circle": result["best"], "score": result["score"], "evaluations": evaluate.evaluations,
            "history": result["history"], "population": result["population"],
            "stop_reason": result["stop_reason"], "generations_completed": result["generations_completed"]}
