# `circle_optimization` — the package

Everything that searches, solves, registers or measures circles. Subpackages follow the eras of the project ([docs/ERAS.md](../docs/ERAS.md)):

| Path | Era / phases | Representation | Dependencies | README |
| --- | --- | --- | --- | --- |
| `problem.py`, `presentation.py` | 00 | profiles, $N \times N$ matrix | stdlib | below |
| `classic/` | 01–06 | child genome + $N \times N$ matrix | stdlib | [classic/README.md](classic/README.md) |
| `exact/` | 07 | profile edge-count table | numpy, highspy | [exact/README.md](exact/README.md) |
| `layered/` | 09–14 | profile sequence / edge-count table | numpy (+ highspy for S3) | [layered/README.md](layered/README.md) |
| `eras.py`, `__main__.py` | 08 | all | per era | below |
| `benchmark/` | 03–08 | all | stdlib (+ highspy, plotly optional) | [benchmark/README.md](benchmark/README.md) |

Dependencies point one way: `problem` ← `classic`, `exact` ← `layered` ← `eras` ← `benchmark` / CLI. Older code never imports newer code.

## Core model: `problem.py`

| Name | Purpose |
| --- | --- |
| `Child` | A named child with Boolean characteristics; identity is by instance |
| `hamming_distance(a, b)` | Number of differing characteristics |
| `distance_matrix(profiles)`, `pairwise_distances(items, f)` | $N \times N$ symmetric distance table |
| `circle_fitness(genome, distances)` | Sum of neighbour distances, wraparound included |
| `canonical_circle(genome)`, `valid_circle(genome, n)` | One representative per circle and direction (child 0 first); validity check |
| `unique_circles(n)`, `exact_best(distances)` | Exhaustive enumeration of $(N-1)!/2$ circles, small $N$ only |
| `random_profiles(seed, n, c)` | The reproducible instance generator used by the suite, CLI and notebooks; sizes are nested |

`presentation.format_distance_matrix` prints a labelled preview of the matrix.

## Public API (`from circle_optimization import ...`)

Imported eagerly (standard library only): the `problem` names above, `run_hill_climber`, `run_inversion_hill_climber`, `run_guided_hill_climber`, `run_ga`, `FitnessEvaluator`, the operators (`random_individual`, `initial_population`, `order_crossover`, `swap_mutation`, `inversion_mutation`, `mutate`, `tournament_selection`, `swap_delta`, `inversion_delta`), the guidance helpers and `format_distance_matrix`.

Loaded on first access (numpy/highspy): `aggregate_profiles`, `ProfileInstance`, `profile_upper_bound`, `multi_start`, `solve_circle`, `solve_profile_milp`, `ProfileState`, `run_layered_hill_climber`, `ReducedCostGuide`, `run_profile_ga`, `run_edge_ga`, `MemeticLayer`, `Problem`, `ERAS`, `run_era`. Importing the package therefore never requires numpy.

## Era registry: `eras.py`

- **`Problem(profiles)`**, or `Problem.random(n, c, seed)` — one instance with lazily cached representations: `distances` ($N \times N$), `state` (profile aggregation), `exact` / `optimum` (proven optimum). `score(circle)` rescores any circle straight from the profiles; it is used to validate every run.
- **`Era`** — one algorithm version: `name`, `family` (HC / GA / exact), `phase`, `title`, `summary`, `representation`, `layers`, `extends`, `budget_unit`, `requires`, runner and `defaults`.
- **`ERAS`** — 19 entries ordered by phase.
- **`run_era(name, problem, *, seed, time_limit | deadline, evaluation_budget, score_bound, on_improvement, **overrides)`** — runs an era with a fresh `Random(seed)`, validates the result and adds `era`, `seconds`, `layers`. Overrides replace the era's defaults; unknown names raise `TypeError`. With only a time limit, count/restart/generation caps are lifted.

```python
from circle_optimization.eras import ERAS, Problem, run_era

problem = Problem.random(160, 5, seed=2027)
for name in ("hc", "hc-p-rc", "ga-m"):
    print(name, run_era(name, problem, seed=1, time_limit=0.2)["score"], "/", problem.optimum)
run_era("ga-m", problem, seed=1, time_limit=0.2, memetic=None)     # memetic layer removed
```

## Command line: `python -m circle_optimization`

| Command | Does |
| --- | --- |
| `list [--family HC]` | All eras with phase, parent and layers; marks missing dependencies |
| `describe ERA` | Full record of one era, including defaults and budget unit |
| `run ERA [--set k=v ...] [--optimum] [--json]` | One run; `--set` takes JSON values (`null`, `true`, numbers, dicts) |
| `compare [ERA ...] [--all] [--family F] [--repeats K] [--json FILE]` | Same instance, limit and seeds for every era; gap to the proven optimum |

Instance options for `run` and `compare`: `--children`, `--characteristics`, `--instance-seed`, `--seed`, `--time-limit` (default 0.2 s), `--budget`.

The exact solver has its own CLI, `python -m circle_optimization.exact.milp`, and the benchmark suite has `python -m circle_optimization.benchmark.cli` (also available as `benchmark_guided_search.py` at the repository root).

## How a run flows

```
CLI / notebook / suite
      │  run_era("hc-p-rc", problem, seed=1, time_limit=0.2)
      ▼
eras.ERAS["hc-p-rc"]  ── defaults + overrides ──►  runner
      │                                               │ problem.state (built once, untimed if prebuilt)
      ▼                                               ▼
layered.hill_climbing.run_layered_hill_climber(state, Random(1), splice_rate=0.45, guide=ReducedCostGuide, ...)
      │
      ▼
result {circle, score, evaluations, stop_reason, layers, ...}  →  Problem.score(circle) == score ?  →  caller
```
