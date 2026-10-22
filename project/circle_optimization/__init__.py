"""Search for high-scoring circles of children with Boolean profiles.

The package is organized by algorithm era (see docs/ERAS.md):

- `problem`: representation, scoring, exhaustive search, random instances.
- `classic`: index-genome HC and GA over the N x N matrix (phases 01-06).
- `exact`: profile aggregation, multigraph heuristic and exact MILP (phase 07).
- `layered`: profile-aware HC/GA built from optional layers (phases 09-14).
- `benchmark`: experiment suite, reports and run comparison.
- `eras`: registry that runs any era by name; `python -m circle_optimization`.

Classic names are imported eagerly and need only the standard library. Names
backed by numpy/highspy (exact solver, layered eras) load on first access.
"""

from .classic.genetic import FitnessEvaluator, run_ga
from .classic.guidance import WeakEdgeSwapSelector, guided_swap_mutation, make_guided_mutation
from .classic.hill_climbing import run_guided_hill_climber, run_hill_climber, run_inversion_hill_climber
from .classic.operators import (
    initial_population,
    inversion_delta,
    inversion_mutation,
    mutate,
    order_crossover,
    random_individual,
    swap_delta,
    swap_mutation,
    tournament_selection,
)
from .presentation import format_distance_matrix
from .problem import (
    Child,
    DistanceMatrix,
    Genome,
    canonical_circle,
    circle_fitness,
    distance_matrix,
    exact_best,
    hamming_distance,
    pairwise_distances,
    random_profiles,
    unique_circles,
    valid_circle,
)

# Optional-dependency exports, imported on first use by `__getattr__`.
_LAZY_EXPORTS = {
    "ProfileInstance": "exact.profiles", "aggregate_profiles": "exact.profiles",
    "profile_upper_bound": "exact.profiles", "multi_start": "exact.multigraph",
    "solve_circle": "exact.milp", "solve_profile_milp": "exact.milp",
    "Problem": "eras", "ERAS": "eras", "run_era": "eras",
    "ProfileState": "layered.state", "run_layered_hill_climber": "layered.hill_climbing",
    "ReducedCostGuide": "layered.reduced_costs", "run_profile_ga": "layered.genetic",
    "run_edge_ga": "layered.genetic", "MemeticLayer": "layered.memetic",
}


def __getattr__(name):
    """Import numpy/highspy-backed names lazily so classic use stays stdlib-only."""
    if name in _LAZY_EXPORTS:
        from importlib import import_module
        return getattr(import_module(f".{_LAZY_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Child", "DistanceMatrix", "FitnessEvaluator", "Genome", "WeakEdgeSwapSelector",
    "canonical_circle", "circle_fitness", "distance_matrix", "exact_best", "format_distance_matrix",
    "guided_swap_mutation", "hamming_distance", "initial_population", "inversion_delta",
    "inversion_mutation", "make_guided_mutation", "mutate", "order_crossover", "pairwise_distances",
    "random_individual", "random_profiles", "run_ga", "run_guided_hill_climber", "run_hill_climber",
    "run_inversion_hill_climber", "swap_delta", "swap_mutation", "tournament_selection",
    "unique_circles", "valid_circle", *_LAZY_EXPORTS,
]
