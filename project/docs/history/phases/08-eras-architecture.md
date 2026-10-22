# Phase 08 — era architecture: modular layers, registry, CLI and guided tour

Started/completed: 2026-10-21. Status: **completed and verified**.

## Goal

Phases 09–14 add profile-aware layers to HC and GA (S2, S1, S3, S4, S5, S6 from the phase-07 transfer analysis). Every past and new algorithm version ("era") must stay runnable on its own, from Python and from the command line, and be comparable against every other era on the same instance. Each enhancement is a separate, documented module so it can be enabled, disabled and ablated.

## Planned structure

```
circle_optimization/
  problem.py, presentation.py   core representation, scoring, enumeration, random instances
  classic/      phases 01–06: index genomes over the N×N matrix
    operators.py    sampling, swap/inversion/OX operators and O(1) move deltas
    guidance.py     weak-edge swap selection (phase 02), guided GA mutation
    hill_climbing.py  shared HC engine; uniform, legacy, inversion-focused, guided HC
    genetic.py      fitness cache and generational GA
    diagnostics.py  fixed-check replay diagnostics
  exact/        phase 07: profile aggregation, multigraph heuristic, MILP
    profiles.py, multigraph.py, milp.py
  layered/      phases 09–14: profile-aware layers and the engines combining them
    state.py (S2), splice.py (S1), reduced_costs.py (S3), edge_genome.py and
    memetic.py (S4), propagation.py (S5), restarts.py (S6),
    hill_climbing.py, genetic.py (engines)
  benchmark/    suite, execution, metrics, report, compare, cli
  eras.py       registry: every era's name, phase, layers, summary and runner
  __main__.py   `python -m circle_optimization list | run | compare`
tests/          all unit/regression tests
docs/           guides (BENCHMARK_SUITE, GUIDED_SEARCH, ERAS) and history
guided_tour.ipynb   walk-through of the base and each era/layer on one instance
```

Root `benchmark_guided_search.py` and `compare_benchmark_runs.py` stay as thin entry points so every historical command in these records still runs.

## Rules for this and later phases

- Refactoring must not change any classic trajectory: frozen GA hashes, seeded HC results and all existing tests must pass unchanged in meaning.
- Every new layer is optional, documented per function, and covered by tests that also compare it with the layer disabled.
- `hc-p` (S2 only) must reproduce `hc` exactly in score, counts and trace.
- Eras that count work differently (e.g. memetic local-search checks) document their unit; cross-era comparisons should prefer the equal-time protocol.

## Old → new paths

Definitions moved verbatim (AST extraction); only imports and the renamed `_inversion_delta` → public `inversion_delta` changed.

| Phase 00–07 path | Phase 08 path |
| --- | --- |
| `circle_optimization/ga.py` (operators) | `circle_optimization/classic/operators.py` |
| `circle_optimization/ga.py` (`FitnessEvaluator`, `run_ga`) | `circle_optimization/classic/genetic.py` |
| `circle_optimization/swaps.py` (sampling, swap delta) | `circle_optimization/classic/operators.py` |
| `circle_optimization/swaps.py` (weak-edge selector) | `circle_optimization/classic/guidance.py` |
| `circle_optimization/hill_climbing.py`, `inversion_hill_climbing.py`, `guided_hill_climbing.py` (HC) | `circle_optimization/classic/hill_climbing.py` |
| `guided_hill_climbing.make_guided_mutation` | `circle_optimization/classic/guidance.py` |
| `circle_optimization/diagnostics.py` | `circle_optimization/classic/diagnostics.py` |
| `circle_optimization/profile_model.py` | `circle_optimization/exact/profiles.py` |
| `circle_optimization/profile_search.py` | `circle_optimization/exact/multigraph.py` |
| `circle_optimization/profile_milp.py` | `circle_optimization/exact/milp.py` |
| `circle_optimization/benchmark_{suite,execution,metrics,report}.py` | `circle_optimization/benchmark/{suite,execution,metrics,report}.py` |
| `benchmark_guided_search.py` | `circle_optimization/benchmark/cli.py` (root file kept as entry point) |
| `compare_benchmark_runs.py` | `circle_optimization/benchmark/compare.py` (root file kept as entry point) |
| `benchmark_suite.nested_profiles` | `problem.random_profiles` |
| `circle_optimization/{GUIDED_SEARCH,BENCHMARK_SUITE}.md` | `docs/{GUIDED_SEARCH,BENCHMARK_SUITE}.md` |
| `test_*.py` (root) | `tests/test_*.py` (`python -m unittest discover -s tests -t .`) |

Checkpoint fingerprints now hash every `.py` file of the package (recursively) rather than only the directory of `execution.py`.

## What was implemented

- **Layout** as planned above; every classic definition moved verbatim. The root keeps `README.md`, the two notebooks, `guided_tour.ipynb`, `requirements-benchmark.txt` and the two historical entry scripts.
- **`eras.py`**: `Problem` (profiles; lazily cached `distances`, `state`, `exact`, `optimum`; an independent `score` used to validate every run), `Era` records (name, family, phase, title, summary, representation, layers, extends, budget unit, requirements, runner, defaults) and `run_era` (seeded `Random`, `time_limit` or absolute `deadline`, `evaluation_budget`, `score_bound`, `on_improvement`, checked parameter overrides). With only a time limit the count/restart/generation caps are lifted, as in the suite.
- **CLI** `python -m circle_optimization`: `list`, `describe ERA`, `run ERA [--set key=value ...] [--optimum] [--json]`, `compare [ERA ... | --all | --family F] [--repeats K] [--json FILE]`.
- **Suite**: `--eras NAME ...` measures eras as families (`variant=era`); `--no-classic` drops the classic exploration grid and never builds the $N \times N$ matrix (reference bound from the profile aggregation). Eras have no diagnostic replay; the HTML report skips exploration charts without guided rows and labels eras `era NAME`.
- **`exact.multigraph.multi_start` and `exact.milp.solve_circle`** gained an `on_improvement` hook so the `multistart` and `exact` eras report anytime traces (heuristic incumbents, then the MILP result).
- **Docs**: `README.md` (repository map), `circle_optimization/README.md` (module map), `docs/ERAS.md` (catalog, lineage, layers, budgets, how to add an era), suite guide section on eras; phase records 09–14.
- **Guided tour**: `guided_tour.ipynb` runs every era on $N=160$, $C=5$ with equal time, demonstrates each layer (S2 identity, splice moves at an HC local optimum, reduced-cost certificate, edge crossover), an ablation, a ranking and a large-$N$ section ($N=10{,}000$, $C=8$).

## Verification

- All pre-existing tests pass unchanged in meaning after the move (56 → 56 at the refactor checkpoint), including frozen GA trajectory hashes; test files only changed import paths and patch targets.
- `tests/test_eras.py`: registry ordered by phase, every era's phase indexed and recorded, `extends` links valid; every available era returns a valid circle (independently rescored) under a budget and a deadline and never exceeds the exact optimum; seeded repeatability; override checking; CLI `list/describe/run/compare` with JSON output; suite with eras alongside and without the classic grid (S2 row equals classic HC; no $N \times N$ matrix; era-only HTML report renders).
- Suite smoke runs: `--quick --eras hc-p multistart exact`, `--no-classic --sizes 2000`, and the era report.

## Problems found during integration

- The first era benchmark crashed in the `exact` era under a 0.2 s deadline: HiGHS reports an infinite dual bound when stopped before its root LP (fixed, see the correction in phase 07).
- The HTML report assumed a guided classic family (`make_subplots(cols=0)` for era-only suites); exploration charts are now skipped when absent.
- zsh does not word-split unquoted variables; benchmark scripts with era lists must quote-split explicitly or run under bash.

## Addendum (2026-10-22): READMEs and docstrings

- README per main folder: root (goal, steps, structure, results, how to reproduce), `circle_optimization/`, `classic/`, `exact/`, `layered/`, `benchmark/`, `tests/`, `docs/`.
- Docstring pass: 82 of 267 package definitions (mostly the phase-03/04 benchmark modules) had none; every function, method and class of `circle_optimization` now has one. No code changed; all 92 tests pass.
