"""Registry of algorithm eras: every past and current search, runnable by name.

An *era* is one algorithm version as it stood after a phase of
docs/history (e.g. "hc-legacy" = HC of phase 01, "hc-p-splice" = profile HC
with the phase-10 splice layer). Each entry records its phase, family,
representation, enabled layers, the era it extends and its default
parameters, and knows how to run itself on a `Problem`.

    from circle_optimization.eras import Problem, run_era
    problem = Problem.random(160, 5, seed=2027)
    result = run_era("hc-i", problem, seed=1, time_limit=0.2)
    result["score"], problem.optimum

`run_era` accepts parameter overrides, so any layer can be switched off or
retuned without a new era (ablation): `run_era("hc-p-ils", problem,
perturbation=None)`. The command line exposes the same registry:
`python -m circle_optimization list | describe | run | compare`.

Budgets: with `time_limit` only, count/restart/generation caps are lifted
(the suite's equal-time protocol); with `evaluation_budget`, each era counts
its own work unit (`Era.budget_unit`). Units differ between families, so
prefer equal time for cross-era comparisons.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from importlib.util import find_spec
from math import factorial
from random import Random
from time import perf_counter

from .problem import DistanceMatrix, distance_matrix, random_profiles, valid_circle

UNLIMITED = 10**12
DEFAULT_BUDGET = 20_000


class Problem:
    """One instance shared by all eras; each representation is built on first use.

    `distances` (N x N, classic eras) costs O(N^2 C) time and O(N^2) memory;
    `state` (profile aggregation, layered eras) and `exact` (phase-07 solver)
    need numpy (and highspy). Building them outside a timed run keeps
    comparisons fair, as in the benchmark suite.
    """

    def __init__(self, profiles, *, label: str | None = None) -> None:
        """Store profiles as tuples of bools; `label` names the instance in reports."""
        self.profiles = tuple(tuple(bool(v) for v in row) for row in profiles)
        if not self.profiles or not self.profiles[0]:
            raise ValueError("A problem needs at least one child and one characteristic")
        self.label = label or f"N={len(self.profiles)} C={len(self.profiles[0])}"

    @classmethod
    def random(cls, children: int, characteristics: int, seed: int = 2027) -> "Problem":
        """The suite's reproducible instance (`problem.random_profiles`)."""
        return cls(random_profiles(seed, children, characteristics),
                   label=f"N={children} C={characteristics} seed={seed}")

    @property
    def n(self) -> int:
        """Number of children."""
        return len(self.profiles)

    @property
    def characteristics(self) -> int:
        """Number of Boolean characteristics."""
        return len(self.profiles[0])

    @cached_property
    def distances(self) -> DistanceMatrix:
        """N x N Hamming matrix used by the classic eras."""
        return distance_matrix(self.profiles)

    @cached_property
    def state(self):
        """Layer-S2 `ProfileState` used by the layered eras (needs numpy)."""
        from .layered.state import ProfileState
        return ProfileState.from_profiles(self.profiles)

    @cached_property
    def exact(self) -> dict:
        """Phase-07 `solve_circle` result: proven optimum and its circle."""
        from .exact.milp import solve_circle
        return solve_circle(self.profiles)

    @property
    def optimum(self) -> int:
        """Proven optimal score (solves the MILP on first access)."""
        if self.exact["status"] != "optimal":
            raise RuntimeError("The exact solver did not prove optimality")
        return self.exact["score"]

    def score(self, circle) -> int:
        """Circular Hamming score of a child circle, computed from the profiles.

        Independent of every search's own bookkeeping, so `run_era` uses it to
        validate results. Vectorized with numpy when available, else O(N C) Python.
        """
        if len(circle) < 2:
            return 0
        if find_spec("numpy") is None:
            rows = self.profiles
            return sum(sum(a != b for a, b in zip(rows[u], rows[v]))
                       for u, v in zip(circle, tuple(circle[1:]) + tuple(circle[:1])))
        import numpy as np
        table = self.__dict__.get("_table")
        if table is None:
            table = self.__dict__["_table"] = np.asarray(self.profiles, dtype=bool)
        ordered = table[list(circle)]
        return int((ordered != np.roll(ordered, -1, axis=0)).sum())


@dataclass(frozen=True)
class Era:
    """One registered algorithm version.

    `run(problem, rng, budget, deadline, score_bound, on_improvement, params)`
    returns the algorithm's result dict; `defaults` are its parameters, all
    overridable through `run_era`.
    """

    name: str
    family: str
    phase: str
    title: str
    summary: str
    representation: str
    layers: tuple[str, ...]
    extends: str | None
    budget_unit: str
    requires: tuple[str, ...]
    run: Callable = field(repr=False)
    defaults: Mapping[str, object] = field(default_factory=dict)

    def available(self) -> bool:
        """True if every optional dependency of this era is importable."""
        return all(find_spec(module) is not None for module in self.requires)


def _caps(budget, *, restarts=200, generations=600):
    """Count caps: the given budget with default caps, or unlimited under time only."""
    if budget is None:
        return dict(evaluation_budget=UNLIMITED, max_restarts=10**9, generations=10**9)
    return dict(evaluation_budget=budget, max_restarts=restarts, generations=generations)


def _ga_shape(problem, params):
    """Population, tournament and elites clipped to the number of unique circles."""
    population = min(params.pop("population_size"), max(1, factorial(problem.n - 1) // 2))
    return dict(population_size=population, tournament_size=min(params.pop("tournament_size"), population),
                elites=min(params.pop("elites"), population))


def _classic_hc(kind):
    """Runner for a classic HC entry point (`uniform`, `inversion`, `guided`)."""
    def run(problem, rng, budget, deadline, score_bound, on_improvement, params):
        """Run this classic HC entry point with suite-style count caps."""
        from .classic import hill_climbing
        search = dict(uniform=hill_climbing.run_hill_climber, inversion=hill_climbing.run_inversion_hill_climber,
                      guided=hill_climbing.run_guided_hill_climber)[kind]
        caps = _caps(budget, restarts=params.pop("max_restarts", 200))
        caps.pop("generations")
        return search(problem.distances, rng, deadline=deadline, score_bound=score_bound,
                      on_improvement=on_improvement, **caps, **params)
    return run


def _classic_ga(problem, rng, budget, deadline, score_bound, on_improvement, params):
    """Runner for the classic GA; `exploration` enables the phase-02 guided mutation."""
    from .classic.genetic import run_ga
    from .classic.guidance import make_guided_mutation
    exploration = params.pop("exploration", None)
    if exploration is not None:
        params["mutation_operator"] = make_guided_mutation(problem.distances, exploration=exploration)
    caps = _caps(budget, generations=params.pop("generations", 600))
    caps.pop("max_restarts")
    shape = _ga_shape(problem, params)
    caps["evaluation_budget"] = max(caps["evaluation_budget"], shape["population_size"])
    return run_ga(problem.distances, rng, deadline=deadline, score_bound=score_bound,
                  on_improvement=on_improvement, **caps, **shape, **params)


def _multistart(problem, rng, budget, deadline, score_bound, on_improvement, params):
    """Runner for the phase-07 multi-start heuristic on the profile multigraph."""
    from .exact.multigraph import multi_start
    from .exact.profiles import multigraph_circle
    starts = params.pop("starts", None)
    starts = starts if starts is not None else (UNLIMITED if deadline is not None else 16)
    result = multi_start(problem.state.instance, starts=starts, seed=rng.getrandbits(32), deadline=deadline,
                         target=score_bound, on_improvement=on_improvement, **params)
    return dict(circle=multigraph_circle(result["multiplicity"], problem.state.instance), score=result["score"],
                evaluations=result["starts"], stop_reason=result["stop_reason"], history=result["history"])


def _exact(problem, rng, budget, deadline, score_bound, on_improvement, params):
    """Runner for the phase-07 exact solver (budget ignored; deadline honoured)."""
    from .exact.milp import solve_circle
    time_limit = None if deadline is None else max(0.0, deadline - perf_counter())
    result = solve_circle(problem.profiles, time_limit=time_limit, seed=rng.getrandbits(31),
                          on_improvement=on_improvement, **params)
    return dict(result, evaluations=result["heuristic_starts"] + (result["rounds"] > 0), stop_reason=result["status"])


def _layered_hc(problem, rng, budget, deadline, score_bound, on_improvement, params):
    """Runner for the layered HC engine; `guide_exploration` builds the S3 guide."""
    from .layered.hill_climbing import run_layered_hill_climber
    exploration, tournament = params.pop("guide_exploration", None), params.pop("guide_tournament", 8)
    if exploration is not None:
        # One LP solve, charged to the run (3-35 ms for m = 32-128).
        from .layered.reduced_costs import ReducedCostGuide
        params["guide"] = ReducedCostGuide.from_state(problem.state, exploration=exploration,
                                                      tournament=tournament)
    caps = _caps(budget, restarts=params.pop("max_restarts", 200))
    caps.pop("generations")
    return run_layered_hill_climber(problem.state, rng, deadline=deadline, score_bound=score_bound,
                                    on_improvement=on_improvement, **caps, **params)


def _layered_ga(engine):
    """Runner for a layered GA engine (`profile` = GA-P, `edge` = GA-E/GA-M)."""
    def run(problem, rng, budget, deadline, score_bound, on_improvement, params):
        """Run the GA-P or edge GA engine; `memetic` settings become a MemeticLayer."""
        from .layered import genetic
        if engine == "edge":
            memetic = params.pop("memetic", None)
            if memetic is not None:
                from .layered.memetic import MemeticLayer
                params["local_search"] = MemeticLayer(**memetic)
        caps = _caps(budget, generations=params.pop("generations", 600))
        caps.pop("max_restarts")
        shape = _ga_shape(problem, params)
        caps["evaluation_budget"] = max(caps["evaluation_budget"], shape["population_size"])
        search = genetic.run_profile_ga if engine == "profile" else genetic.run_edge_ga
        return search(problem.state, rng, deadline=deadline, score_bound=score_bound,
                      on_improvement=on_improvement, **caps, **shape, **params)
    return run


_GA = dict(population_size=50, tournament_size=10, elites=2, crossover_rate=0.9, mutation_rate=0.25)


def _layered_defaults(**layers):
    """Every layered-HC parameter with its 'layer off' value, then the era's own.

    All layered HC eras share these keys, so any layer can be switched on or off
    per run (ablation), e.g. `run_era("hc-p-grasp", problem, propagate=False)`.
    """
    return dict(stall_limit=200, attempts_per_restart=1000, restart_policy="productive",
                inversion_rate=0.5, splice_rate=0.0, guide_exploration=None, guide_tournament=8,
                propagate=False, start="random", start_noise=1.5, perturbation=None) | layers
_HC = dict(stall_limit=200, attempts_per_restart=1000)
_NUMPY, _MILP = ("numpy",), ("numpy", "highspy")
_CHECKS = "score checks: one per (re)start plus one per proposed move"
_GA_UNIT = "distinct complete-circle fitness evaluations (cache hits free)"
_GA_MEMETIC_UNIT = ("distinct fitness evaluations plus memetic 2-exchange checks (E^2 per local-search "
                    "step); prefer equal time")

_ENTRIES = [
    Era("hc-legacy", "HC", "01", "Random-restart hill climber",
        "Delta-scored 50/50 swap/inversion proposals, strict improvement, restart after 200 "
        "consecutive failures or 1,000 proposals (the historical hard cap).",
        "index", ("swap+inversion", "legacy-restarts"), None, _CHECKS, (),
        _classic_hc("uniform"), dict(_HC, restart_policy="legacy")),
    Era("ga", "GA", "01", "Generational GA with order crossover",
        "Tournament selection, OX, one swap/inversion mutation per offspring, elitism, cached "
        "fitness; phase-04/06 fast paths keep identical trajectories.",
        "index", ("ox", "uniform-mutation"), None, _GA_UNIT, (), _classic_ga, dict(_GA)),
    Era("hc-guided", "HC", "02", "Weak-edge-guided hill climber",
        "Phase-01 HC whose swaps break poor edges with weight 1 + max - d; 20% uniform swaps.",
        "index", ("swap+inversion", "legacy-restarts", "weak-edge-guidance"), "hc-legacy", _CHECKS, (),
        _classic_hc("guided"), dict(_HC, restart_policy="legacy", exploration=0.2)),
    Era("ga-guided", "GA", "02", "GA with weak-edge-guided mutation",
        "Phase-01 GA whose swap mutations use the weak-edge selector (20% uniform).",
        "index", ("ox", "weak-edge-mutation"), "ga", _GA_UNIT, (), _classic_ga, dict(_GA, exploration=0.2)),
    Era("hc", "HC", "04", "Productive-restart hill climber",
        "Restart only after 200 consecutive failed proposals; improvements reset patience.",
        "index", ("swap+inversion", "productive-restarts"), "hc-legacy", _CHECKS, (),
        _classic_hc("uniform"), dict(_HC, restart_policy="productive")),
    Era("hc-i", "HC", "04", "Inversion-focused hill climber (HC-I)",
        "Productive HC proposing 80% inversions (2-opt) and 20% swaps.",
        "index", ("swap+inversion", "productive-restarts", "inversion-focus"), "hc", _CHECKS, (),
        _classic_hc("inversion"), dict(_HC, restart_policy="productive", inversion_rate=0.8)),
    Era("multistart", "exact", "07", "Multi-start profile-multigraph heuristic",
        "Randomized greedy b-matching on the m x m multigraph, component merging and bulk "
        "2-exchange local search; repeated until the bound, start cap or deadline.",
        "multigraph", ("greedy-b-matching", "connect", "bulk-2-exchange"), None,
        "completed starts", _NUMPY, _multistart, {}),
    Era("exact", "exact", "07", "Profile-aggregated MILP (proven optimum)",
        "Heuristic incumbent, HiGHS LP/MIP with lazy connectivity cuts, reduced-cost fixing and "
        "Euler reconstruction; status 'optimal' only with a proof.",
        "multigraph", ("multistart", "lp-cuts", "reduced-cost-fixing", "mip"), "multistart",
        "heuristic starts + 1 if the MILP ran (budget ignored)", _MILP, _exact, {}),
    Era("hc-p", "HC", "09", "Profile-state hill climber (S2)",
        "HC on profile-id sequences with the m x m table: identical trajectory to 'hc', memory "
        "O(N + m^2) instead of O(N^2).",
        "profile", ("profile-state",), "hc", _CHECKS, _NUMPY, _layered_hc, _layered_defaults()),
    Era("ga-p", "GA", "09", "Profile-state GA (S2)",
        "Classic GA scored through the profile table; cache and duplicate rejection keyed by the "
        "profile sequence, so relabelled circles collapse.",
        "profile", ("profile-state", "ox"), "ga", _GA_UNIT, _NUMPY, _layered_ga("profile"),
        dict(_GA, seeding=0.0)),
    Era("hc-p-splice", "HC", "10", "Splice hill climber (S2 + S1)",
        "Adds the split-and-splice move: the 2-opt reconnection that splits the circle, re-joined "
        "at zero cost through two same-profile children. Mix 10% swaps / 45% inversions / 45% splices.",
        "profile", ("profile-state", "splice"), "hc-p", _CHECKS, _NUMPY, _layered_hc,
        _layered_defaults(inversion_rate=0.45, splice_rate=0.45)),
    Era("hc-p-rc", "HC", "11", "Reduced-cost-guided splice hill climber (S2 + S1 + S3)",
        "Inversion/splice proposals break the worst edge of an 8-edge tournament on LP reduced cost "
        "and re-attach to a zero-reduced-cost partner; 20% uniform proposals. One LP solve per run.",
        "profile", ("profile-state", "splice", "reduced-cost"), "hc-p-splice", _CHECKS, _MILP, _layered_hc,
        _layered_defaults(inversion_rate=0.45, splice_rate=0.45, guide_exploration=0.2)),
    Era("ga-e", "GA", "12", "Edge-genome GA (S4)",
        "Genome = profile edge-count table; crossover keeps the parents' common edges, then fills "
        "from their union and the remaining degree greedily by distance (fill='random' = level A); "
        "mutation = one connected 2-exchange.",
        "multigraph", ("profile-state", "edge-crossover", "greedy-fill"), "ga-p", _GA_UNIT, _NUMPY,
        _layered_ga("edge"), dict(_GA, fill="greedy", memetic=None, seeding=0.0)),
    Era("ga-m", "GA", "12", "Memetic edge-genome GA (S4 + memetic layer)",
        "GA-E whose offspring are improved by bulk 2-exchange local search with probability 0.2 "
        "(`memetic=None` removes the layer).",
        "multigraph", ("profile-state", "edge-crossover", "greedy-fill", "memetic"), "ga-e", _GA_MEMETIC_UNIT,
        _NUMPY, _layered_ga("edge"), dict(_GA, fill="greedy", memetic=dict(max_moves=None, probability=0.2),
                                          seeding=0.0)),
    Era("hc-p-prop", "HC", "13", "Propagating guided splice hill climber (+ S5)",
        "After an accepted inversion/splice, retries the same profile-pair pattern at its other "
        "occurrences (up to 64 proposals per trigger, each a normal check).",
        "profile", ("profile-state", "splice", "reduced-cost", "propagation"), "hc-p-rc", _CHECKS, _MILP,
        _layered_hc, _layered_defaults(inversion_rate=0.45, splice_rate=0.45, guide_exploration=0.2,
                                       propagate=True)),
    Era("hc-p-grasp", "HC", "14", "GRASP-restart layered hill climber (+ S6 greedy starts)",
        "hc-p-prop whose starts and restarts are randomized greedy b-matching seeds (noise 0.5) "
        "realized by an Euler circuit. Recommended S6 variant.",
        "profile", ("profile-state", "splice", "reduced-cost", "propagation", "greedy-start"), "hc-p-prop",
        _CHECKS, _MILP, _layered_hc,
        _layered_defaults(inversion_rate=0.45, splice_rate=0.45, guide_exploration=0.2, propagate=True,
                          start="greedy", start_noise=0.5)),
    Era("hc-p-ils", "HC", "14", "Iterated local search layered hill climber (+ S6 ILS)",
        "hc-p-grasp whose restarts perturb the incumbent with a double-bridge kick instead of a new "
        "seed. Kept for comparison: measured worse than hc-p-grasp on uniform instances.",
        "profile", ("profile-state", "splice", "reduced-cost", "propagation", "greedy-start", "double-bridge"),
        "hc-p-grasp", _CHECKS, _MILP, _layered_hc,
        _layered_defaults(inversion_rate=0.45, splice_rate=0.45, guide_exploration=0.2, propagate=True,
                          start="greedy", start_noise=0.5, perturbation="double-bridge")),
    Era("ga-e-seed", "GA", "14", "Seeded edge-genome GA (S4 + S6)",
        "GA-E with 10% of the initial population built by greedy GRASP constructions.",
        "multigraph", ("profile-state", "edge-crossover", "greedy-fill", "seeded"), "ga-e", _GA_UNIT, _NUMPY,
        _layered_ga("edge"), dict(_GA, fill="greedy", memetic=None, seeding=0.1)),
    Era("ga-m-seed", "GA", "14", "Seeded memetic edge-genome GA (S4 + memetic + S6)",
        "GA-M with 10% greedy GRASP seeds in the initial population.",
        "multigraph", ("profile-state", "edge-crossover", "greedy-fill", "memetic", "seeded"), "ga-m",
        _GA_MEMETIC_UNIT, _NUMPY, _layered_ga("edge"),
        dict(_GA, fill="greedy", memetic=dict(max_moves=None, probability=0.2), seeding=0.1)),
]

ERAS: dict[str, Era] = {era.name: era for era in _ENTRIES}


def run_era(name: str, problem: Problem, *, seed: int = 0, time_limit: float | None = None,
            evaluation_budget: int | None = None, score_bound: int | None = None,
            on_improvement: Callable[[int, int], None] | None = None,
            deadline: float | None = None, **overrides) -> dict:
    """Run era `name` on `problem` and validate the returned circle.

    `seed` seeds a fresh `random.Random`. `time_limit` (seconds from this call)
    or `deadline` (absolute perf_counter time, as the benchmark suite passes)
    bounds wall time; `evaluation_budget` caps work in the era's own unit;
    with none of them, `DEFAULT_BUDGET` applies. `overrides` replace the era's
    default parameters (unknown names raise). The result gains `era`,
    `seconds` (wall time of the run, problem representations excluded if
    already built) and `layers` when the engine does not report them.
    """
    if name not in ERAS:
        raise KeyError(f"Unknown era {name!r}; choose from {', '.join(ERAS)}")
    era = ERAS[name]
    if not era.available():
        raise RuntimeError(f"Era {name!r} needs {', '.join(era.requires)}: pip install -r project/requirements-benchmark.txt")
    unknown = set(overrides) - set(era.defaults) - {"max_restarts", "generations", "starts"}
    if unknown and era.defaults:
        raise TypeError(f"Era {name!r} has no parameter(s) {sorted(unknown)}; defaults: {dict(era.defaults)}")
    params = {**era.defaults, **overrides}
    if time_limit is not None and deadline is not None:
        raise ValueError("Give time_limit or deadline, not both")
    if time_limit is None and deadline is None and evaluation_budget is None:
        evaluation_budget = DEFAULT_BUDGET
    started = perf_counter()
    if time_limit is not None:
        deadline = started + time_limit
    result = era.run(problem, Random(seed), evaluation_budget, deadline, score_bound, on_improvement, params)
    seconds = perf_counter() - started
    if not valid_circle(tuple(result["circle"]), problem.n) or problem.score(result["circle"]) != result["score"]:
        raise AssertionError(f"Era {name!r} returned an invalid circle or a wrong score")
    return dict(result, era=name, seconds=seconds, layers=result.get("layers", era.layers))
