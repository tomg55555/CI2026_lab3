# State of the art — circle optimization

Last reviewed: 2026-10-22. Latest evidence: phases 08–14 and the era benchmarks `benchmark-results/eras-phase14` and `eras-phase14-large`.

## Eras and layers (phases 08–14)

Every algorithm version is a registered era (`python -m circle_optimization list`, [docs/ERAS.md](../ERAS.md)) runnable alone, in `compare`, in the suite (`--eras`) and in `guided_tour.ipynb`. Profile-aware layers, each optional: S2 profile state (09), S1 splice (10), S3 LP reduced-cost guidance (11), S4 edge-genome GA with an optional memetic layer (12), S5 propagation (13), S6 greedy seeding / double-bridge ILS (14).

Equal-time benchmark (0.2 s per run, 3 instances $\times$ 3 trials, exact optimum references; mean gap to optimum):

| Era | $C=5$, $N=160$ | $C=5$, $N=1000$ | $C=7$, $N=160$ | $C=7$, $N=1000$ |
| --- | ---: | ---: | ---: | ---: |
| `hc-legacy` (01) | 16.40% | 35.12% | 18.90% | 36.48% |
| `ga` (01) | 14.72% | 43.39% | 18.14% | 43.65% |
| `hc-i` (04) | 9.42% | 17.20% | 12.03% | 20.15% |
| `hc-p` (09, S2) | 10.84% | 18.15% | 12.85% | 20.86% |
| `hc-p-splice` (10, +S1) | 9.25% | 16.90% | 11.10% | 19.60% |
| `hc-p-rc` (11, +S3) | 1.38% | 4.55% | 3.13% | 5.26% |
| `hc-p-prop` (13, +S5) | 1.50% | 2.36% | 3.26% | 5.02% |
| `hc-p-grasp` (14, +S6) | 0.00% | 0.00% | 1.00% | 0.28% |
| `ga-e` (12, S4) | 9.85% | 26.82% | 6.29% | 17.81% |
| `ga-m` (12, S4 + memetic) | 0.60% | 41.60% | 2.75% | 35.60% |
| `ga-e-seed` (14, +S6) | 0.00% | 0.03% | 0.22% | 0.12% |
| `exact` (07) | 0.00% | 0.00% | 0.30% | 0.07% |

- S3 is the largest single gain for HC; S6 greedy starts close $C=5$ entirely.
- S1 helps modestly on its own; S5 helps at $N=1000$, $C=5$ and is neutral or slightly negative elsewhere at this allowance. At $N=10^4$ it costs throughput during the first descent ($O(N)$ scans per acceptance): a measured limitation, see phase 13.
- The edge crossover beats OX; the memetic layer is strong at $N=160$ but too expensive per child for 0.2 s at $N=1000$ (it wins with 0.5 s in the phase-12 probe). Time allowance changes the GA ranking.
- Within 0.2 s the MILP proves the optimum on 7/9 ($C=7$, $N=160$) and 8/9 ($C=7$, $N=1000$) runs; with no limit it proves every instance in $\le 0.35$ s.
- GRASP restarts beat double-bridge ILS on uniform instances (`hc-p-grasp` vs `hc-p-ils`); ILS stays registered for comparison.

$N = 10{,}000$, $C = 8$ ($m = 256$), 1 s per run, layered eras only (the classic eras would need a $10^8$-cell matrix): `hc-p-grasp` 0.20%, `ga-e-seed` 0.14%, `multistart` 0.19%, `hc-p-rc` 11.57%, `hc-p` 26.95%, `ga-p` 48.12%; `exact` proves all four optima ($\approx 0.6$ s each). Ablation of `hc-p-grasp` on $N=1000$, $C=7$ (0.2 s): removing S6 raises the gap from 0.21% to 4.39%; removing S1, S3 or S5 changes it by $< 0.1$ point (noise level at this allowance).

92 tests pass (`python -m unittest discover -s tests -t .`).

## Earlier state (phase 07)

## Exact optimum for every $N$ (phase 07)

`circle_optimization.exact.milp.solve_circle` solves the problem exactly by aggregating children into the $m \le \min(N, 2^C)$ distinct profiles: multi-start heuristic on the $m \times m$ edge-multiplicity multigraph, then a HiGHS MILP with lazy connectivity cuts, LP cut phase, reduced-cost fixing, and Euler reconstruction. $N=10^6$ with $C=8$ is solved and proven in about 0.7 s; all suite instances ($C=5$, $N \le 160$) in $\le 14$ ms. The suite now uses this exact optimum as its default reference (`--reference milp`; `--reference bound` restores phase 06).

Measured against exact optima, every existing search stays below optimum above $N=20$: at $N=160$ GA (20k evaluations) is 7.15% short and HC-I at 0.2 s 9.31%. Aggregated 2-exchanges alone close 61–98% of those gaps from the final circles; 11–62% of their edges join profile pairs with negative LP reduced cost. Checks-protocol scores and traces are unchanged from phase 06. 56 tests pass.

## Full-grid validation after phase 04

The enhanced 3,900-search / 975-diagnostic grid is complete. Recorded pre-export wall time fell from 3,165.43 to 1,142.36 seconds (~$2.77\times$ faster), despite adding HC-I. The complete new run including export/report took 19 min 12 s. Timed searches remained isolated; all ten CPUs handled independent post-measurement work.

All 900 matched fixed-count GA scores, evaluation counts and strict-improvement score/count traces are identical to baseline. Uniform GA's count-cap runtime is ~$2.0$–$2.34\times$ faster across sizes. Productive HC and HC-I improve large-instance quality; at $N=160$ and 200 ms, means are 656.27 / 666.53 versus old HC 614.00. New GA scores 614.47 at that time allowance, versus old GA 572.80. With the much longer count-cap runtime, GA retains the quality advantage (684.93 vs HC-I 654.40).

Guidance gives little robust equal-time benefit after the depth fix. Cheaper guidance and independent $C=100$ validation remain planned. See phase 05 for paired intervals, common-target rebasing, timing scopes and interpretation limits.

## Current implementation after phase 04

- Uniform HC retains its 50/50 swap/inversion control; productive patience is the default. Explicit legacy mode restores the historical 1,000-proposal hard cap.
- New `run_inversion_hill_climber` and HC-I suite series default to 80% inversions / 20% swaps. All variants record per-restart exits and obey global budgets.
- GA uses known-canonical fitness and per-generation score lookups, equivalent slice-based OX, and cheaper exact-equivalent positional-diversity arithmetic.
- CPU process batches support percentages, approximate ETA, phase timings, fingerprinted checkpoints/resume, and deterministic fixed-count outputs.
- Isolated measurement remains the default. Diagnostics/statistics follow it in parallel. Explicit throughput mode also parallelizes measured work and labels timings/deadline results as contention-sensitive.
- Diagnostic edge caches and shared bootstrap resampling weights retain every observation and all 400 bootstrap resamples; no experiment counts are reduced.
- CSV/JSON/HTML export has progress. Completed historical outputs are protected; current reports include execution and policy metadata.
- 38 checks pass, including frozen pre-optimization GA trajectories, spawn, resume, isolation, restart behavior, and retained swaps.

Follow-ups are planned in [GUIDANCE_AND_RESTART_PLAN.md](GUIDANCE_AND_RESTART_PLAN.md). The full-grid follow-up and measured execution improvement are now documented in [phase 05](phases/05-full-comparison.md).

Phase 06 ([record](phases/06-faster-suite.md)) cut the same full grid from 1,152 s to 499 s. Searches stop at the proven reference score; checks-protocol runs use one repeat and copy exploration $= 1$ from the uniform control; diagnostics use integer edge ids and per-variant jobs; GA/HC hot paths consume identical random draws. Checks-protocol scores and traces match phase 04 exactly. Time-protocol scores rose slightly because the code is faster. 44 tests pass.

## Baseline at the start of phase 04

- Symmetric Hamming-distance matrix; maximize circular adjacent differences.
- HC: constant-time swap/inversion score deltas; random proposals split 50/50; strict improvement; 200-stall restart and hard 1,000-proposal restart cap.
- Guided HC: cached cumulative weak-edge weights, rebuilt lazily after accepted moves; exploration controls the uniform fraction of swaps.
- GA: OX, tournament selection, cached complete-circle fitness, elitism; repeated canonicalization during parent selection is a measured hot path.
- Suite: nested sizes, exploration sweeps, count/time protocols, separate fixed-check diagnostic replays, hierarchical bootstrap, offline Plotly HTML.
- Execution is serial; progress is only printed when starting a problem size.

## Evidence, not universal guarantees

Historical full run (`benchmark-results/full`, 2026-10-21):

```sh
.venv/bin/python benchmark_guided_search.py --suite \
  --sizes 10 20 40 80 160 \
  --explorations 0 0.2 0.5 0.8 1 \
  --instances 3 --trials 5 --repeats 2 \
  --protocol both --budget 20000 --time-limit 0.2 \
  --output benchmark-results/full
```

3,600 measured executions, 900 diagnostic replays, 3 independent nested instance families, $C=5$. Recorded execution/analysis time: 3,165.43 seconds before report export. Generated files are ignored by git and may not exist in another clone.

- $N=160$ uniform HC: score 608.80 at the count cap, about 38 ms; uniform GA: 684.93, about 2,252 ms. At 0.2 s: HC 614.00 versus GA 572.80.
- Guided HC exploration 0 at $N=160$: +5.20 points at the count cap, +4.20 points under the time allowance, but about 73% more time per fixed-count run.
- Controlled follow-up: changing only HC's hard proposal cap from 1,000 to 10,000 at the same 20,000 checks improves uniform $N=160$ HC from 608.80 to 645.47; all 15 paired trials improve. This is count-budget evidence, not an equal-time validation of the proposed restart policy.
- A representative $N=160$ replay ended 19/20 completed local searches at the proposal cap rather than the stall condition.
- GA profiling: ~688k evaluator calls and ~756k canonicalizations for ~20k new fitness computations. Parent selection and OX are substantial costs.
- Exact $N=10$ follow-up confirms optima 36, 40, 36. All historical algorithms found them; odd upper bounds 41/37 made some 99% targets unattainable.

## Interpretation constraints

Count limits are different-cost units in HC and GA. Concurrent wall-time runs are not directly comparable with isolated runs. $C=5$ has only 32 profiles; $N=160$ has 80% duplicate profiles. Whole-circle and edge coverage mean different things. Three instance families are limited evidence for generalizing exploration tuning.

## Requested next steps

1. Prioritize execution throughput, accurate progress, and resumability.
2. Avoid restarting productive HC searches just because 1,000 attempts elapsed.
3. Keep uniform HC and add an inversion-focused alternative with nonzero swaps.
4. Optimize GA without changing fixed-count seeded search decisions.
5. Plan cheaper guided sampling and more principled productivity-based restarts.
