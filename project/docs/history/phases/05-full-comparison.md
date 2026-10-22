# Phase 05 — clean artifacts and compare the enhanced full benchmark

Started/completed: 2026-10-21. Status: **completed and analyzed**.

## Preserved historical baseline

`benchmark-results/full`: 3,600 timed executions, 900 diagnostics, sizes 10/20/40/80/160, $C=5$, exploration 0/0.2/0.5/0.8/1, three instance families, five trials each, two repeats, 20,000 count cap, 0.2-second time allowance. Its recorded pre-export wall time is 3,165.43 seconds.

## Cleanup scope

The following generated small smoke/validation runs were inspected and removed; their implementation evidence remains in phase 04's Markdown record:

- `multi-characteristic`, `scaling-exploration`, `smoke`
- `phase04-smoke`, `phase04-verified`
- `phase04-serial`, `phase04-isolated`, `phase04-throughput`
- `phase04-final-serial`, `phase04-final-isolated`, `phase04-final-throughput`

No historical full baseline is selected for deletion.

## New comparison protocol

```sh
.venv/bin/python benchmark_guided_search.py --suite \
  --sizes 10 20 40 80 160 \
  --explorations 0 0.2 0.5 0.8 1 \
  --instances 3 --trials 5 --repeats 2 \
  --protocol both --budget 20000 --time-limit 0.2 \
  --workers 10 --timing-mode isolated \
  --output benchmark-results/full-phase04
```

The machine reports ten physical/logical CPUs and 32 GiB RAM. All ten workers are requested for independent diagnostic/statistical batches. Timed searches remain isolated so comparison with the previous full run's 0.2-second results is valid; fully concurrent wall-time searches would answer a different question. The new HC-I control adds 300 measured executions and 75 diagnostic replays: expected totals are **3,900 measured executions and 975 diagnostics**.

Nested profiles, seeds, budgets, GA settings, repetitions and original variants match the baseline. HC uses the new productive policy; GA is optimized; HC-I uses 80% inversions with swaps retained. The latest even-score bound changes some target thresholds, so cross-version comparisons must use raw/normalized scores or explicitly rebase both runs to common target/reference values.

## Completion and analysis

The run completed with return code 0. All **3,900 timed results** have status `ok`, and all 975 diagnostic replays matched their corresponding count runs. The preserved old run contains 3,600 valid timed results. No sampling settings, repeats, budgets, diagnostics or bootstrap sample counts were reduced.

### Runtime: the full-grid speedup is now measured

| Stage | New run |
| --- | ---: |
| Preparation | 0.070 s |
| Isolated measurements | 975.700 s |
| Diagnostics, 10 workers | 165.360 s |
| Statistics, 10 workers | 1.071 s |
| Export | 3.772 s |
| HTML report | 6.352 s |
| Total, including export/report | **1,152.487 s = 19 min 12 s** |

On the same pre-export basis as the historical manifest, elapsed time fell from **3,165.434 s to 1,142.363 s**, approximately **$2.77\times$ faster / 63.9% less time**. The enhanced grid includes 8.3% more searches and diagnostic replays because of HC-I. This is an observed end-to-end comparison, not a claim that every component sped up by $2.77\times$ or that the grids contain exactly the same amount of work.

The remaining dominant stage is isolated measurement. The 1,950 time-limited executions alone require approximately 390 s at 0.2 s each. Parallelizing that stage on this machine would change the wall-time experiment through CPU contention; the safe parallel work already uses all ten available CPUs.

### GA: performance improved without changing fixed-count search results

The comparison checked **all 900 matched fixed-count GA executions** across uniform/guided variants: final scores, distinct-evaluation counts, and the full strict-improvement `(score, evaluation)` traces have **zero mismatches**. The phase-04 frozen fixtures additionally cover final populations and histories.

| Children | Old uniform GA mean runtime | New runtime | Speed ratio |
| --- | ---: | ---: | ---: |
| 10 | 0.750 s | 0.371 s | $2.02\times$ |
| 20 | 0.838 s | 0.411 s | $2.04\times$ |
| 40 | 1.081 s | 0.499 s | $2.17\times$ |
| 80 | 1.570 s | 0.670 s | $2.34\times$ |
| 160 | 2.252 s | 0.980 s | $2.30\times$ |

These are means of paired-trial timing-repeat medians. Under the unchanged 0.2-second allowance, the speedup becomes additional search: $N=160$ completes 107.2 generations on average instead of 49.4, and its mean score rises from 572.80 to 614.47 (+41.67, bootstrap interval approximately +36.00 to +46.93). Every paired trial improves at $N=80$ and $N=160$.

### HC: deeper local search has a clear large-instance benefit

Mean final scores under the 20,000 count cap:

| $N$ | Old uniform HC | Productive uniform HC | New HC-I | Uniform GA, both versions |
| --- | ---: | ---: | ---: | ---: |
| 10 | 37.33 | 37.33 | 37.33 | 37.33 |
| 20 | 78.13 | 78.13 | 78.27 | 78.53 |
| 40 | 162.80 | 163.33 | 165.20 | 168.27 |
| 80 | 315.33 | 321.87 | 324.93 | 338.53 |
| 160 | 608.80 | 645.47 | 654.40 | 684.93 |

At $N=160$, productive uniform HC gains 36.67 points (interval +32.27 to +41.33), winning all 15 paired trials. Mean restarts drop from 20.27 to 10.40. New count diagnostics record zero `attempt_cap` exits: completed local searches now stop on sampling stalls, with the final run stopped by the global count allowance. The implementation no longer discards a run merely because 1,000 total proposals have elapsed. A 200-failure sampling stall still does not prove a local optimum.

Uniform HC fixed-count runtime rises modestly: approximately 37.98 to 40.13 ms at $N=160$. At $N \in \{10, 20\}$ scores do not improve; at $N=40$ the small difference is inconclusive. The improvement is therefore size-dependent, not universal.

### Isolated 200 ms results: a practical comparison

| $N$ | Old HC | New HC | HC-I | Old GA | New GA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 37.33 | 37.33 | 37.33 | 37.33 | 37.33 |
| 20 | 78.40 | 78.40 | 78.67 | 78.27 | 78.53 |
| 40 | 165.00 | 164.80 | 166.67 | 163.47 | 166.53 |
| 80 | 318.73 | 326.00 | 329.07 | 311.20 | 325.60 |
| 160 | 614.00 | 656.27 | 666.53 | 572.80 | 614.47 |

At $N=160$ the productive policy gains 42.27 points over old HC (+37.46 to +46.94) at the same time allowance. HC-I adds another 10.27 over productive uniform HC (+8.40 to +12.00), winning all 15 paired trials. HC-I also beats new GA by 52.07 points (+46.47 to +58.27), again in all 15 paired trials. At $N=40$ HC-I and new GA are effectively tied in this sample (difference +0.13; interval crosses zero).

Inversion focus is not just a longer run: at $N=160$ its fixed-count mean runtime is 37.72 ms versus uniform HC's 40.13 ms. Its cheap two-boundary-edge delta and preservation of internal segment edges are plausible advantages over four-edge swaps; the timing observations do not by themselves isolate those causes.

GA still finds the best solutions when allowed its much longer count-budget runtime. $N=160$ uniform GA's 684.93 exceeds HC-I's 654.40, but takes about 0.98 s instead of 0.038 s. A blanket ranking of GA versus HC would ignore this trade-off.

### Convergence and common targets

- $N=160$ fixed-count GA reaches its unchanged final score in 0.929 s on average, down from 2.136 s. Its first-hit evaluation count is unchanged.
- Its 90%-of-common-bound target is reached in about 0.598 s versus 1.355 s on the 14 paired trials where both succeed. This is a conditional timing result; the success rate itself is unchanged, 93.3%.
- Under 200 ms, uniform HC's $N=160$ normalized time-AUC rises from 0.7559 to 0.8043; HC-I reaches 0.8174. New GA rises from 0.6464 to 0.6931. The HC gains therefore improve early/anytime quality as well as final scores.
- A run's time to its own final best is not a common-target comparison: a later final improvement may be a better solution, not slower progress to the old one.

### Guidance after fixing depth: less compelling than the HC-I alternative

Guided HC still has a fixed-count signal at $N=160$: exploration 0 scores 651.33 versus productive uniform 645.47, but its +5.87-point interval crosses zero (-0.67 to +11.60) and its fixed-count runtime is about 55% higher. Under 200 ms, the best selected guided mean is 656.67 (exploration 0.5), just +0.40 over uniform HC with interval -2.67 to +3.47. HC-I's 666.53 is substantially stronger.

For new GA, exploration 0.2 remains modestly helpful at $N=40$ under the count cap (+1.87, interval +0.40 to +3.33). Its equal-time gain is only +0.67 with an interval crossing zero; guidance does not have a consistent equal-time benefit at larger $N$. Selecting the largest mean from this same exploration sweep is not held-out tuning.

Likely explanation: cumulative-weight rebuilding and guided-wrapper setup consume time that could otherwise test more moves. The productive policy changes the visited states and reduces the apparent benefit previously attributed to guidance. The exact cause mix requires profiling; these results support the planned cheap edge-tournament/incremental-sampling experiments rather than a universal alpha.

### Comparison safeguards and interpretation limits

- All old target outcomes were recomputed using the **new common references**. The $N=10$ scores did not improve: both versions reach 36/40/36. Correcting odd bounds is a benchmark interpretation fix, not an algorithmic quality gain.
- GA cache-hit rates are not directly comparable as exploration measures: old parent selections queried the fitness cache; new selections use known scores. Removing those queries changes the denominator, not the seeded search path.
- Confidence intervals are paired hierarchical bootstrap intervals over three instance families and five trials. Timing repeats are collapsed. These are exploratory results, without multiple-comparison correction.
- $C$ remains 5, with only 32 profiles and 80% duplicates at $N=160$. Conclusions need independent $C=100$ families before being generalized broadly.
- Reference bounds above small exact-reference sizes remain potentially unattainable. The comparisons use ratios to bounds, not claims of actual optimality gaps.

## Outputs and reproduction

- Preserved baseline: `benchmark-results/full/report.html`
- Enhanced report: `benchmark-results/full-phase04/report.html`
- Cross-version plots: `benchmark-results/full-phase04/comparison.html`
- Full paired metrics: `comparison.csv` and `comparison.json` in the new run.
- Runtime/process provenance: `run.log`, `run-status.json`, `configuration.json`, `execution.csv`, and compatible batch checkpoints.

Rebuild the cross-version analysis without running searches:

```sh
.venv/bin/python compare_benchmark_runs.py \
  benchmark-results/full benchmark-results/full-phase04 --quiet
```

All 40 tests pass, including two new checks that target-bound corrections do not masquerade as search improvements and that contended timing comparisons are rejected. Core benchmark source was not changed during this full execution.

## Conclusions / next priorities

1. Productive HC fixes the demonstrated large-instance depth problem.
2. HC-I is the strongest tested low-latency large-instance default; retain the original uniform mix as a control rather than claim 80/20 is universally tuned.
3. GA optimization is validated on the complete grid and improves equal-time competitiveness without changing fixed-count search quality.
4. Guidance now needs a cheaper implementation or a better demonstrated payoff; tuning its probability is a smaller lever than depth and move composition.
5. Extend to new instance families with $C=100$ and compare restart patience/move mixes on held-out data. Further isolated-suite wall-time reductions must respect the unavoidable time-budget floor rather than silently parallelize timed searches.
