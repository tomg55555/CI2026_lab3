# Phase 04 — execution throughput and productive search

Started/completed: 2026-10-21. Status: **completed and verified**.

## Scope and priority

1. **Priority:** speed up the full benchmark without dropping experimental work. Add process-based batches, percentages, phase timings, checkpoint/resume, efficient bootstrap reuse, and explicit timing-isolation metadata.
2. Fix HC depth/restart balance while retaining a reproducible legacy control.
3. Optimize GA repeated canonicalization/selection and crossover overhead without changing fixed-count search semantics.
4. Add coexisting inversion-focused HC; keep swaps and normal uniform HC.
5. Publish actionable plans for cheaper guidance and productivity-aware stopping.

## Planned implementation decisions

- Independent paired-trial batches; top-level process workers compatible with macOS spawn. Main process owns result merging and progress reporting.
- Default isolated measured runs; parallel diagnostics/statistics. Explicit throughput mode permits concurrent measurements with report warnings/labels.
- Atomic batch checkpoints with config and implementation fingerprints; resume refuses incompatible results instead of silently mixing experiments.
- Cache bootstrap sampling weights, not fewer bootstrap samples.
- A productive HC policy uses an unsuccessful-attempt allowance that resets on local improvements; a hard-cap legacy mode remains selectable.
- Inversion-focused default proposal mix: 80% inversions, 20% swaps; an experiment setting, not a claim of optimal tuning.
- GA reuses canonical keys and current population scores; preserve tournament sampling and stable tie behavior. Verify against the pre-change implementation.

## Acceptance checks

- Serial/process batches give identical fixed-count trajectories and diagnoses.
- Isolated measurement has no concurrent diagnostic/statistics pool activity.
- Progress includes batch and overall percentages; errors identify the batch.
- Resumed work is checked against fingerprints and produces all expected rows.
- No instance/trial/repeat/budget/bootstrap reduction is used as a speedup.
- GA matches baseline seeded outputs/history and new-fitness counts.
- HC rejects non-improving moves and respects budgets; productive improvements extend the local run; inversion-focused mode retains swaps.
- Report generation supports added variants and execution metadata.

## Changes actually implemented

- Added process-based measurement/diagnostic/statistics batch scheduling with deterministic seed/order handling and percentage/ETA reporting.
- Isolated measurements are serial and precede all pool activity. Explicit throughput mode parallelizes measurements with clear contention metadata.
- Added atomic compatible-source/config checkpoints and resume; completed historical outputs are not silently replaced.
- Cached hierarchical bootstrap weights, retaining the same 400 index resamples.
- Cached exact diagnostic edge sets and combined scored HC proposal accounting; snapshots, candidate observations, and coverage definitions are retained.
- Default HC now uses consecutive-failure patience without the legacy hard cap; per-restart histories record attempts, acceptances, scores and exit reasons.
- Added coexisting HC-I (default 80% inversion, 20% swaps) and suite controls.
- GA uses canonical evaluation fast paths, generation score lookups in ranking/ tournaments, slice-based equivalent OX, and cheaper positional-diversity sums.
- Tightened Hamming upper bounds to even values; historical reports are unchanged.
- Published the staged [guidance and stopping plan](../GUIDANCE_AND_RESTART_PLAN.md).

## Verification and preliminary measurements

- 38 tests passed after the implementation, including macOS spawn, resumed checkpoints, serial/parallel count trajectories, failure identification, productive depth, legacy HC, retained swaps, and cached-bootstrap equivalence.
- Five GA pre-change trajectory hashes (including full population/history and improvement trace) still match; OX matches a retained original test oracle.
- Isolated $N=160$ GA, $C=5$, instance 2027, trial 1002027, population 50, 600 generations, 20,000 cap: old seconds 2.28694/2.28143/2.28332; optimized seconds 0.98501/0.93594/0.93434. Median ~$2.44\times$ faster, same score 684 / 19,975 evaluations.
- Across the historical 15 $C=5$ count-budget trials, uniform HC legacy/productive/ inversion-focused mean scores: $N=80$ 315.33/321.87/324.93; $N=160$ 608.80/645.47/654.40. Productive beats legacy in all 15 trials at each size. These are count-budget comparisons, not equal-time proof of superiority.
- Live percentages were verified through measurement, diagnostics, statistics, CSV/JSON export and all 50 charts of a complete quick report. Running the same final-source command with `--resume` reused 6 measured, 3 diagnostic, and 6 statistics batches; no search batch executed again.

## Controlled scheduling workload

The following grid was run sequentially in three execution modes, with the same new algorithm implementation and no reduction in work:

```sh
.venv/bin/python benchmark_guided_search.py --suite \
  --sizes 40 160 --explorations 0 0.5 1 \
  --instances 2 --trials 1 --repeats 2 \
  --protocol both --budget 5000 --time-limit 0.03 \
  --workers <1-or-4> --timing-mode <isolated-or-throughput> \
  --no-progress --output benchmark-results/phase04-final-<mode>
```

Each mode included 144 measured searches and 36 diagnostic replays. Serial and throughput data exports used `--no-report`; the isolated four-worker run also rendered the report. The phase times below exclude export/render, so they are compared on the same basis:

| Phase | Serial, one worker | Isolated, four workers | Throughput, four workers |
| --- | ---: | ---: | ---: |
| Measurements | 8.010 s | 7.993 s | 3.067 s |
| Diagnostics | 8.241 s | 3.309 s | 2.987 s |
| Statistics | 0.443 s | 0.152 s | 0.150 s |
| Sum including preparation | 16.731 s | 11.489 s | 6.246 s |

Parallel diagnostics are ~$2.49\times$ faster in this small four-batch workload. Isolated measurement times remain essentially unchanged, as intended. Throughput improves work completion, but its measured elapsed-time outcomes include CPU contention. These are smoke/workload measurements, not a measured speedup for the historical 3,600-search full grid.

An earlier phase-04 serial diagnostic run on the same grid took 15.232 s before the exact edge-cache optimization; afterward it took 8.241 s. New report example: `benchmark-results/phase04-final-isolated/report.html`. Final source/checkpoint verification example: `benchmark-results/phase04-verified/report.html`.

## Remaining plans

The staged guidance and adaptive-stopping proposals remain planned work in [GUIDANCE_AND_RESTART_PLAN.md](../GUIDANCE_AND_RESTART_PLAN.md); they are not represented as implemented sampling algorithms or proven local-optimum tests.
