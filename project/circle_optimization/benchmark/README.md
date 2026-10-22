# `benchmark/` — experiment suite, reports and run comparison

Reproducible, paired experiments: the same instances and seeds for every algorithm, two budget protocols, exact references, statistics with clustered confidence intervals, and an offline HTML report. The full protocol is in [docs/BENCHMARK_SUITE.md](../../docs/BENCHMARK_SUITE.md); this page is a map.

## Modules

| Module | Role |
| --- | --- |
| `suite.py` | `SuiteConfig`, `validate_config`, instance references (`problem_reference`: enumeration and/or MILP optimum, else proven bound), `variants` (classic grid + eras), `run_search`, `measured_run`, process-pool batch jobs, `execute_suite`, `run_and_save` |
| `execution.py` | `BatchExecutor` (spawn-safe process pool reused across phases), `Progress` (percent/ETA printing), `Checkpoints` (per-batch JSON, fingerprinted by config + package source, `--resume`) |
| `metrics.py` | Standard-library metrics: `quality_bound`, `anytime_auc`, `first_hit`, `target_rows`, `collapse_repeats`, `summarize`, `exploration_effects`, `exploration_trends`, `clustered_ci` (hierarchical bootstrap) |
| `report.py` | Offline Plotly report (`render_report`); `python -m circle_optimization.benchmark.report DIR` rebuilds it from saved data |
| `compare.py` | Paired comparison of two saved runs (`compare_benchmark_runs.py OLD NEW`) |
| `cli.py` | Command line (`benchmark_guided_search.py` at the root): the suite, or the original single-size uniform-vs-guided comparison |

## What a suite run does

1. **Instances**: `random_profiles(seed + i, max(sizes), C)`, nested by size. Reference per instance: the exact optimum (MILP for every $N$ by default, cross-checked by enumeration up to `--exact-limit`), else a proven bound.
2. **Measurements** (isolated by default: one search at a time): for every trial, every variant — classic HC/GA uniform and guided at each exploration, HC-I, plus any `--eras` — under the checks protocol (`--budget` in each family's unit) and/or the time protocol (`--time-limit` wall seconds). Every search stops on reaching the reference (`--stop-at-bound`), which cannot change its score or trace.
3. **Diagnostics** (parallel, classic families only): untimed fixed-check replays recording novelty, edge coverage and diversity.
4. **Statistics** (parallel): repeats collapsed per trial, then means, quartiles, paired exploration effects and trends with clustered intervals.
5. **Export**: CSV/JSON tables, `configuration.json` manifest, `report.html`.

## Typical commands

```sh
# Smoke test (small sizes, few trials)
.venv/bin/python benchmark_guided_search.py --suite --quick --output benchmark-results/quick

# Classic grid with exact references
.venv/bin/python benchmark_guided_search.py --suite --sizes 10 20 40 80 160 \
  --explorations 0 0.2 0.5 0.8 1 --instances 3 --trials 5 --repeats 2 \
  --protocol both --budget 20000 --time-limit 0.2 --workers auto --output benchmark-results/full-new

# Eras only, equal time, no N×N matrix (scales to large N)
.venv/bin/python benchmark_guided_search.py --suite --no-classic \
  --eras hc hc-i hc-p-rc hc-p-grasp ga ga-e-seed exact \
  --sizes 160 1000 --protocol time --time-limit 0.2 --output benchmark-results/eras-new

# Resume an interrupted run, rebuild a report, compare two runs
.venv/bin/python benchmark_guided_search.py --suite ... --output benchmark-results/full-new --resume
.venv/bin/python -m circle_optimization.benchmark.report benchmark-results/full-new
.venv/bin/python compare_benchmark_runs.py benchmark-results/full-phase06 benchmark-results/full-phase07
```

## Outputs (`benchmark-results/<run>/`)

| File | Contents |
| --- | --- |
| `runs.csv` | One row per timed search: score, reference, gap, times, counts, AUCs, stop reason, restart exits, diagnostics (checks protocol) |
| `improvements.csv` | Every strict improvement (score, count, seconds) per run |
| `targets.csv` | Hit/miss and first-hit time/count for each target fraction |
| `problems.csv` | Per instance: distinct profiles, bounds, exact optimum and its source, MILP status/time |
| `summaries.csv`, `exploration_effects.csv`, `exploration_trends.csv` | Statistics |
| `diagnostics.csv`, `diagnostic_snapshots.csv` | Replay diagnostics (classic families) |
| `execution.csv`, `configuration.json`, `results.json` | Phase timings, manifest (config, Python, platform, revision), everything as JSON |
| `report.html` | Interactive offline report |

## Fairness rules built in

- Same instances, seeds and allowance for every variant; execution order rotates between trials.
- Shared instance construction (matrix, profile state, references) is outside the timed region; search setup and callbacks are inside.
- HC counts score checks, GA distinct fitness evaluations, memetic eras add local-search checks: equal counts are not equal work, so rankings use the time protocol.
- Checks-protocol repeats are deterministic; one repeat is the default. Time-protocol repeats are kept and collapsed per trial.
- `--timing-mode throughput` parallelizes measured runs too, and labels their timings as contention-sensitive.
