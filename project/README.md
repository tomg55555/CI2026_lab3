# Project deep dive — classic, exact and layered searches

This folder is the full project behind the repository's [README](../README.md), [history line](../HISTORY.md) and [walkthrough](../walkthrough.ipynb): package, tests, benchmark tools, documentation, engineering records and saved results.

$N$ children each have $C$ Boolean characteristics (tall, thin, …). Two children differ by their **Hamming distance**, the number of characteristics on which they disagree. Seat them in a circle so as to **maximize the sum of the distances between neighbours**, the last child included; over all circular orders $\pi$:

$$
\max_{\pi} \sum_{k=0}^{N-1} d\left(\pi_k, \pi_{(k+1) \bmod N}\right)
$$

It is a maximization variant of the travelling salesman problem with Boolean "cities". The repository contains the course's heuristic searches (hill climbing, genetic algorithm), an **exact solver** that scales to millions of children, and a set of **enhancement layers** that carry the exact solver's insights back into HC and GA. Every version stays runnable and comparable.

## Contents

1. [Approach](#approach)
2. [Results at a glance](#results-at-a-glance)
3. [Development, step by step](#development-step-by-step)
4. [Key ideas](#key-ideas)
5. [Repository structure](#repository-structure)
6. [Getting started](#getting-started)
7. [Reproducing the experiments](#reproducing-the-experiments)
8. [Documentation and conventions](#documentation-and-conventions)

## Approach

1. **Solve the problem exactly at scale.** With $m$ distinct profiles, $m \le \min(N, 2^C)$. For large $N$ and modest $C$, $m \ll N$, so the problem should be solved on the profiles, not on the children: a MILP aggregated by profile, warm-started by a multi-start heuristic. It must be reliable (proven optimum or proven gap) and fast (data structures and sub-algorithms chosen for time), and usable as a callable module and CLI.
2. **Use it to evaluate the algorithms.** The exact optimum replaces loose upper bounds as the benchmark reference, so every gap is a true gap.
3. **Improve HC and GA without changing their nature.** Transfer what the exact model knows into optional, documented layers. Keep every past version ("era") runnable, so each improvement can be checked era against era.

## Results at a glance

Mean gap to the proven optimum, **same instance, same 0.2 s per run** (3 instance families $\times$ 3 trials, `benchmark-results/eras-phase14`):


| Era (phase)       | What it is                                | $C=5$, $N=160$ | $C=5$, $N=1000$ | $C=7$, $N=1000$ |
| ----------------- | ----------------------------------------- | ---------: | ----------: | ----------: |
| `hc-legacy` (01)  | first hill climber                        |     16.40% |      35.12% |      36.48% |
| `ga` (01)         | first GA                                  |     14.72% |      43.39% |      43.65% |
| `hc-i` (04)       | best classic HC                           |      9.42% |      17.20% |      20.15% |
| `hc-p-rc` (11)    | + profile state, splice move, LP guidance |      1.38% |       4.55% |       5.26% |
| `hc-p-grasp` (14) | + greedy GRASP restarts                   |  **0.00%** |   **0.00%** |       0.28% |
| `ga-e-seed` (14)  | edge-genome GA + seeding                  |  **0.00%** |       0.03% |       0.12% |
| `exact` (07)      | MILP (proves the optimum without a limit) |      0.00% |       0.00% |       0.07% |

At $N = 10{,}000$, $C = 8$ (1 s per run; the classic eras cannot run because they would need a $10^8$-cell matrix): `hc-p-grasp` 0.20%, `ga-e-seed` 0.14%; the exact solver proves each optimum in about 0.6 s. Full tables: [docs/ERAS.md](docs/ERAS.md#results-snapshot).

## Development, step by step

Each phase has a record in [docs/history/phases](docs/history/phases) (phases 00–03 predate the records and are summarized in the [phase index](docs/history/PHASES.md)).


| Phase | What                                                                                                                | Why                                                                                 | Outcome                                                                                  |
| ----- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 00    | Problem model: profiles, symmetric distance matrix, circular score, exhaustive search                               | A correct reference for small $N$                                                   | Exact only up to $N \approx 10$                                                              |
| 01    | Shared HC and GA library: swap/inversion moves, OX crossover, canonical genomes, fitness cache                      | The two heuristics of the lab                                                       | Eras `hc-legacy`, `ga`                                                                  |
| 02    | $O(1)$ move deltas; weak-edge-guided swaps                                                                            | Score proposals without rebuilding circles; target poor edges                       | Eras `hc-guided`, `ga-guided`                                                           |
| 03    | Benchmark suite: nested sizes, exploration sweeps, equal-checks and equal-time protocols, diagnostics, HTML reports | Fair, reproducible comparisons                                                      | `benchmark/`                                                                             |
| 04    | Parallel batches, checkpoints; productive HC restarts; inversion-focused HC; GA fast paths                          | The hard 1,000-proposal cap cut productive searches short                           | Eras `hc`, `hc-i`; $N=160$ HC 608.8 → 645.5 at equal checks                              |
| 05    | Full-grid comparison                                                                                                | Validate phase 04                                                                   | 3,900 searches; $2.77\times$ faster grid                                                     |
| 06    | Stop at proven bounds, faster diagnostics, one checks repeat                                                        | Skip work whose outcome is known                                                    | Grid 1,152 s → 499 s, identical scores                                                  |
| 07    | **Exact profile-aggregated MILP** + multi-start heuristic; exact suite reference                                    | Goal 1 and 2                                                                        | Optimum for $N = 10^6$ in $< 1$ s ($C \le 8$); classic eras shown 7–15% below optimum at $N=160$ |
| 08    | Architecture: package by era, era registry, CLI, guided tour, tests/ and docs/                                      | Goal 3: keep every version runnable and comparable                                  | `eras.py`, `python -m circle_optimization`                                               |
| 09    | **S2** profile state: profile-id sequences and an $m \times m$ table                                                      | Remove the $O(N^2)$ memory barrier; prerequisite for the next layers                | `hc-p` = `hc` move for move; `ga-p`                                                      |
| 10    | **S1** split-and-splice move                                                                                        | HC's local optima still admit improving reconnections through same-profile children | `hc-p-splice`                                                                            |
| 11    | **S3** LP reduced-cost guidance                                                                                     | 11–62% of HC/GA edges were on provably sub-optimal profile pairs                   | `hc-p-rc`: largest single HC gain                                                        |
| 12    | **S4** edge-genome GA, optional memetic layer                                                                       | OX keeps order, the score depends on adjacency                                      | `ga-e`, `ga-m`                                                                           |
| 13    | **S5** move propagation                                                                                             | An accepted pattern has the same gain at its other occurrences                      | `hc-p-prop` (helps mid-size, hurts at $N = 10^4$)                                        |
| 14    | **S6** greedy GRASP seeding, double-bridge ILS, seeded GA populations                                               | Random starts begin at about half the optimum                                       | `hc-p-grasp`, `hc-p-ils`, `ga-e-seed`, `ga-m-seed`                                       |
| 15    | Publication layout: README, history line and walkthrough on top, this folder below                                 | Let a reviewer see the work in minutes and deep-dive on demand                       | No algorithm change; all eras still runnable                                             |

Problems found along the way are recorded where they were fixed, e.g. the MILP's behaviour under very short time limits (correction in phase 07).

## Key ideas

- **Profile aggregation.** Children with equal profiles are interchangeable, so a circle's score depends only on how many of its edges join each pair of profiles. That $m \times m$ edge-count table is the exact model's variable; an Euler circuit turns any valid table back into a circle.
- **Exact model.** Integer edge counts with degree $2 n_p$ per profile $p$ and lazy connectivity cuts; an LP phase gives a tight bound, reduced-cost fixing prunes most columns, and the solver reports `optimal` only with a proof.
- **Eras and layers.** An era is an algorithm version (`hc-p-rc`, `ga-m`, …). A layer is one optional module and one engine switch whose off value reproduces the parent era. Any layer can be removed from any era with an override, so ablations need no new code.
- **Fair comparisons.** Eras count work in different units (score checks, distinct fitness evaluations, memetic checks), so rankings use equal wall time; shared instance representations are built outside the timed run, algorithm-own precomputations (LP solve, seeds) inside.

## Repository structure

```
LAB03/
├── README.md, HISTORY.md, walkthrough.ipynb   presentation layer (repository root)
└── project/                   this folder
    ├── README.md              this file
    ├── guided_tour.ipynb      walk-through of every era and layer, ablation included (executed)
    ├── problem.ipynb          course starter notebook
    ├── genetic_algorithm.ipynb original GA walk-through
    ├── circle_optimization/   the Python package            → circle_optimization/README.md
    │   ├── classic/           phases 01–06: HC and GA on the N×N matrix   → classic/README.md
    │   ├── exact/             phase 07: aggregation, heuristic, MILP       → exact/README.md
    │   ├── layered/           phases 09–14: layers S1–S6 and their engines → layered/README.md
    │   ├── benchmark/         experiment suite, reports, comparisons       → benchmark/README.md
    │   ├── eras.py, __main__.py era registry and command line
    │   └── problem.py         core model and scoring
    ├── tests/                 92 unit, regression and integration tests    → tests/README.md
    ├── docs/                  guides and engineering history               → docs/README.md
    ├── benchmark_guided_search.py entry point of the benchmark CLI (historical commands)
    ├── compare_benchmark_runs.py  entry point of the run comparison
    ├── requirements-benchmark.txt numpy, highspy, plotly
    └── benchmark-results/     generated experiment outputs (git-ignored)
```

## Getting started

Requires Python 3.10+ (developed on 3.14).

From the repository root:

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r project/requirements-benchmark.txt notebook   # numpy, highspy (HiGHS), plotly, Jupyter
cd project
python -m unittest discover -s tests -t .                    # 92 tests
```

Every command below runs **from this folder with the virtual environment active**. Where the history records write `.venv/bin/python`, they mean that same interpreter (from here: `../.venv/bin/python`). The package needs no installation: Python finds `circle_optimization/` in the working directory.

The classic eras, the benchmark with `--reference bound` and CSV/JSON export need only the standard library; the exact solver and the layered eras need numpy and highspy; HTML reports need plotly.

Run algorithms by name:

```sh
python -m circle_optimization list                        # every era: phase, parent, layers
python -m circle_optimization describe hc-p-rc            # one era's record
python -m circle_optimization run hc-i --children 160 --optimum
python -m circle_optimization run hc-p-grasp --set propagate=false   # ablate a layer
python -m circle_optimization compare --all --children 160 --time-limit 0.2 --repeats 3
```

From Python:

```python
from circle_optimization.eras import Problem, run_era

problem = Problem.random(1000, 5, seed=2027)          # the benchmark's instance generator
result = run_era("hc-p-grasp", problem, seed=1, time_limit=0.2)
print(result["score"], problem.optimum, result["layers"])

from circle_optimization import solve_circle            # exact solver on any Boolean table
exact = solve_circle(problem.profiles)
print(exact["score"], exact["status"], exact["seconds"]["total"])
```

Then open `guided_tour.ipynb` (`jupyter notebook guided_tour.ipynb`); the shorter reviewer walkthrough is [`../walkthrough.ipynb`](../walkthrough.ipynb).

## Reproducing the experiments

All commands run from this folder; use a new `--output` directory for fresh runs (completed outputs are protected). Timings are machine-dependent (records were measured on an M1 Max, Python 3.14.3).


| Experiment                               | Command                                                                                                                                                                                                                                              | Record                                                                                     |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Exact solver on a random instance        | `python -m circle_optimization.exact.milp --children 1000000 --characteristics 8`                                                                                                                                                          | [07](docs/history/phases/07-profile-milp.md)                                               |
| All eras, equal time, one instance       | `python -m circle_optimization compare --all --children 160 --repeats 3`                                                                                                                                                                   | [ERAS](docs/ERAS.md)                                                                       |
| Era benchmark (0.2 s, $N \in \{160, 1000\}$, $C \in \{5, 7\}$) | see the bash command in [phase 14](docs/history/phases/14-seeding-and-ils.md#era-benchmark-phase-14-integration) → `benchmark-results/eras-phase14`                                                                                                  | [14](docs/history/phases/14-seeding-and-ils.md)                                            |
| Large-$N$ era benchmark ($N=10^4$, $C=8$, 1 s) | same section → `benchmark-results/eras-phase14-large`                                                                                                                                                                                              | [14](docs/history/phases/14-seeding-and-ils.md)                                            |
| Classic full grid with exact references  | `python benchmark_guided_search.py --suite --sizes 10 20 40 80 160 --explorations 0 0.2 0.5 0.8 1 --instances 3 --trials 5 --repeats 2 --protocol both --budget 20000 --time-limit 0.2 --workers auto --output benchmark-results/full-new` | [07](docs/history/phases/07-profile-milp.md), [06](docs/history/phases/06-faster-suite.md) |
| Phase-06 semantics (bound reference)     | add `--reference bound` to the previous command                                                                                                                                                                                                     | [06](docs/history/phases/06-faster-suite.md)                                               |
| Compare two saved grids                  | `python compare_benchmark_runs.py benchmark-results/full-phase06 benchmark-results/full-phase07`                                                                                                                                           | [05](docs/history/phases/05-full-comparison.md)                                            |
| Rebuild an HTML report from saved data   | `python -m circle_optimization.benchmark.report benchmark-results/eras-phase14`                                                                                                                                                            | [suite guide](docs/BENCHMARK_SUITE.md)                                                     |
| Quick suite smoke test                   | `python benchmark_guided_search.py --suite --quick --eras hc-p exact --output benchmark-results/quick`                                                                                                                                     | [suite guide](docs/BENCHMARK_SUITE.md)                                                     |

Exact optima of the benchmark instances (`random_profiles(seed, N, C)`), for spot checks: seed 2027 → 748 ($N=160$, $C=5$), 4,884 ($N=1000$, $C=5$), 6,722 ($N=1000$, $C=7$), 78,850 ($N=10{,}000$, $C=8$).

## Documentation and conventions


| Document                                           | Read it for                                                                |
| -------------------------------------------------- | -------------------------------------------------------------------------- |
| [docs/ERAS.md](docs/ERAS.md)                       | Era catalog, lineage, layer switches, budget units, results, adding an era |
| [docs/BENCHMARK_SUITE.md](docs/BENCHMARK_SUITE.md) | Experimental protocol, metrics, every suite option                         |
| [docs/GUIDED_SEARCH.md](docs/GUIDED_SEARCH.md)     | Phase-02 weak-edge guidance                                                |
| [docs/history/](docs/history/README.md)            | Phase records, decision log, current state of the art                      |

Conventions: write a phase record before changing an algorithm; add new ideas as layers whose default leaves the parent era unchanged (and test that); never edit a completed era in place; measure against the parent with equal time; append corrections to records rather than rewriting them. Committing is an explicit user action.
