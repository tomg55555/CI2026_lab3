# Phase 14 — layer S6: greedy seeding and iterated local search

Started/completed: 2026-10-21. Status: **completed and verified** (HC part; GA seeding summarized below, full record in phase 12).

## Motivation

Every classic restart begins from a uniformly random circle, which scores about half the optimum (mean 408 vs 748 at $N=160$, 2514 vs 4884 at $N=1000$), and spends most of its checks climbing back to an ordinary local optimum. Phase 07's multigraph construction (greedy b-matching by decreasing distance + component merging) builds circles close to the optimum in milliseconds at $m = 32$. Two standard ways to use such constructions with a local search: GRASP (every restart from a fresh randomized greedy seed) and iterated local search (ILS: restart from a perturbation of the incumbent, here the 4-opt double bridge, which no single 2-opt or splice move can undo).

## Design

Shared constructors in `circle_optimization/layered/restarts.py` (also used by the GA layers): `seed_multigraph(state, rng, noise)` (one GRASP construction: `greedy_multigraph` with uniform noise in $[0, \text{noise})$ added to distances, then `connect`), `seed_sequence(state, rng, noise)` (its Euler circuit, randomly rotated) and `double_bridge(seq, rng)` (cut $A \mid B \mid C \mid D$ at three uniform points, return $A\,C\,B\,D$).

Engine parameters (`run_layered_hill_climber`):

- `start="random"` (default, classic draws) or `"greedy"`: every (re)start uses `seed_sequence(state, rng, noise=start_noise)`; `start_noise=1.5` by default (the phase-07 `multi_start` value);
- `perturbation=None` (default) or `"double-bridge"`: the first start uses `start`; every later restart (after a stall, or the legacy attempt cap) continues from `double_bridge(best sequence, rng)`, i.e. ILS with "better" acceptance around the global incumbent. The kicked circle is scored once (1 check), like any start.
- `restart_history[k]["start"]` records "random", "greedy" or "perturb".

Acceptance, stall-based restarts, budgets, the deadline, `score_bound` and `on_improvement` are unchanged. A greedy seed is built before the deadline check of its local search, so a run may overshoot the deadline by one seed construction (measured below).

Seed probe (20 seeds each, noise 0.5): mean seed score 744.2 (optimum 748) in 2.2 ms at $N=160$, $C=5$; 4876.3 (4884) in 0.5 ms at $N=1000$, $C=5$; 6682.8 (6722) in 7.1 ms at $N=1000$, $C=7$. A greedy seed alone is within 0.16–0.58% of the optimum, before any local search.

## Verification

`tests/test_layered_state.py` (shared constructors, profile counts, seeded reproducibility) and `tests/test_layered_hc.py::LayeredEngineTests`: greedy-start and double-bridge runs (and the full stack) are valid, within budget, deterministic per seed and $\le$ the exact optimum; restart histories start with "greedy"/"random" and continue with "perturb" exactly when ILS is on; with `score_bound` = 176 the full stack reaches the $N=40$ optimum and stops with `bound_reached`; with every layer off results equal the classic HC (`test_profile_hc_reproduces_classic_hc_exactly`, which ignores only the new `start` key).

## Measurement

Protocol: equal time 0.2 s (`deadline = perf_counter() + 0.2`, `evaluation_budget=10**12`, `max_restarts=10**9`), M1 Max with 4 worker processes while other jobs were running, 5 rng seeds (1–5) per instance. $N=160$ = `random_profiles(2027–2029, 160, 5)`, exact optima 748/752/714 (15 runs); $N=1000$ = `random_profiles(2027, 1000, 5)`, optimum 4884 (5 runs). LP guides were built before the timed window. Each stage keeps the best setting of the previous one (scripts kept outside the repository; about 2 minutes of CPU in total for this and the next table).

| Stack (cumulative) | $N=160$ mean score | gap | optimal runs | checks | $N=1000$ mean score | gap | checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S2 base (inversion 0.5) | 662.1 | 10.25% | 0/15 | 160,413 | 3968.8 | 18.74% | 145,724 |
| S2, inversion 0.8 (HC-I mix) | 666.5 | 9.65% | 0/15 | 180,574 | 4043.2 | 17.22% | 161,888 |
| +S1 inversion/splice 0.4/0.4 | 672.8 | 8.81% | 0/15 | 168,643 | 4022.4 | 17.64% | 133,399 |
| +S1 0.3/0.5 | 668.5 | 9.39% | 0/15 | 166,146 | 4028.0 | 17.53% | 127,517 |
| +S1 0.5/0.3 | 669.5 | 9.25% | 0/15 | 171,277 | 4016.0 | 17.77% | 138,473 |
| **+S1 0.45/0.45** | 672.7 | 8.82% | 0/15 | 174,697 | 4049.6 | 17.08% | 134,232 |
| **+S3 exploration 0.2, $k = 8$** | 726.8 | 1.51% | 0/15 | 85,929 | 4572.0 | 6.39% | 65,984 |
| **+S5 propagate** | 728.3 | 1.31% | 0/15 | 76,056 | 4733.6 | 3.08% | 37,001 |
| +S6 greedy start only (noise 1.5) | 738.0 | **0.00%** | **15/15** | 38,381 | 4884.0 | **0.00%** | 23,519 |
| +S6 double bridge only (random start) | 735.3 | 0.36% | 2/15 | 89,337 | 4824.8 | 1.21% | 64,164 |
| +S6 greedy + double bridge | 736.7 | 0.18% | 7/15 | 84,977 | 4829.6 | 1.11% | 48,669 |
| ablation: greedy + DB stack without S1 (splice share → inversions) | 736.1 | 0.25% | 5/15 | 85,267 | 4842.8 | 0.84% | 47,064 |
| ablation: without S3 | 727.7 | 1.39% | 0/15 | 186,852 | 4821.2 | 1.29% | 156,527 |
| ablation: without S5 | 736.7 | 0.18% | 7/15 | 84,899 | 4847.6 | 0.75% | 52,543 |
| ablation: without S6 (same as the +S5 row) | 728.3 | 1.31% | 0/15 | 76,258 | 4733.6 | 3.08% | 36,954 |
| S2 + greedy + DB only | 724.1 | 1.87% | 0/15 | 161,294 | 4737.2 | 3.01% | 139,890 |

All greedy-restart runs reach the optimum, so a second run measured the mean time to the exact optimum (`score_bound` = optimum, 0.2 s cap; the runs that miss count 0.2 s) and added a harder instance, $N=1000$, $C=7$ (`random_profiles(2027, 1000, 7)`, $m = 128$, optimum 6722; 5 runs):

| Stack | $N=160$ hits | mean ms | $N=1000$ $C=5$ hits | mean ms | $N=1000$ $C=7$ gap at 0.2 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| S2 base | 0/15 | 200 | 0/5 | 200 | 20.46% |
| S1 (0.45/0.45) | 0/15 | 200 | 0/5 | 200 | 19.19% |
| S1+S3 | 0/15 | 200 | 0/5 | 200 | 4.71% |
| S1+S3+S5 | 0/15 | 200 | 0/5 | 200 | 4.62% |
| + greedy, noise 0 | 15/15 | 5.2 | 5/5 | 2.7 | 0.24% |
| + greedy, noise 0.25 | 15/15 | 6.2 | 5/5 | 3.5 | 0.23% |
| **+ greedy, noise 0.5** | 15/15 | 5.9 | 5/5 | 3.6 | 0.23% |
| + greedy, noise 1.0 | 15/15 | 5.9 | 5/5 | 3.6 | 0.23% |
| + greedy, noise 1.5 (earlier run) | 15/15 | 28.8 | 5/5 | 74.0 | n/a |
| + greedy, noise 3.0 (earlier run) | 2/15 | 193.4 | 0/5 | 200 | n/a |
| + greedy 0.5 + double bridge | 14/15 | 55.1 | 2/5 | 120.4 | 0.38% |
| + double bridge, random start | 2/15 | 191.4 | 0/5 | 200.9 | 1.89% |
| greedy 0.5 stack without S1 | 15/15 | 6.6 | 5/5 | 4.5 | 0.20% |
| greedy 0.5 stack without S3 | 15/15 | 11.0 | 5/5 | 3.7 | 0.30% |
| greedy 0.5 stack without S5 | 15/15 | 10.4 | 5/5 | 8.0 | 0.27% |
| S2 + greedy 0.5 | 15/15 | 11.3 | 5/5 | 2.5 | 0.27% |

Findings:

- Greedy restarts dominate on uniform $C = 5$ instances: the seed is close to the optimum, and S1/S3/S5 then only polish it. Low noise (0–1.0) reaches the optimum in $\approx 6$ ms; 1.5 is $5$–$20\times$ slower and 3.0 rarely succeeds.
- Double-bridge ILS is worse than plain greedy restarts in every comparison: kicks of four edges are repaired back into the incumbent's basin, while each greedy restart samples a new near-optimal region. With random starts it still beats no S6 at all.
- $C = 7$ ($m = 128$) is the only instance that the stacks do not solve in 0.2 s; there all greedy variants end 0.20–0.30% from the optimum, the ablations are within noise of each other (5 runs), and S1+S3+S5 without S6 is at 4.6%.

Chosen S6 defaults for the HC era: `start="greedy"`, `start_noise=0.5`, `perturbation=None`. Double bridge stays available as an option and for ablations; the engine default remains `start_noise=1.5` (contract), so eras must pass 0.5 explicitly.

## GA seeding

Full record in [phase 12](12-edge-ga-memetic.md#ga-seeding-phase-14-layer-s6); summary here so the S6 layer is described in one place.

Both layered GA engines take $\mathtt{seeding} \in [0, 1]$: $\operatorname{round}(\mathtt{seeding} \cdot \mathtt{population\_size})$ initial individuals come from the S6 constructors in `layered/restarts.py` (randomized greedy b-matching + `connect`, noise 1.5), the rest are uniformly random. GA-P realizes `seed_sequence` as a child genome (`state.circle`); the edge GA uses `seed_multigraph` directly. Seeds must be distinct (at most 10 attempts per seed), seed construction is not charged to `evaluation_budget` (equal-time comparisons charge it as wall time), and `layers` gains `"seeded"`. `seeding=0` draws no extra random numbers, so GA-P still reproduces `run_ga`.

Measured (0.5 s, $C=5$, population 50): GA-P seed 0.1 gap 0.75% / 0.78% at $N=160$ / $N=1000$ (unseeded 9.39% / 39.01%); GA-E greedy-fill seed 0.1 and GA-M seed 0.1 reach the optimum in 12/12 runs. At $C=8$, $N=1000$ (optimum 7,700): GA-P seed 2.01%, GA-E seed 0.29%, GA-M seed 0.69%. A fraction of 0.1 (5 of
50) is the registry default (`ga-e-seed`, `ga-m-seed`).

## Limitations

- Only uniform random instances were measured; on structured instances the greedy construction may be much weaker and ILS more useful.
- The double bridge has fixed strength (four edges); segment-length or multi-kick schedules and non-incumbent acceptance were not tried.
- Greedy seeds cost numpy work per restart (0.5–2.2 ms at $m = 32$, 7 ms at $m = 128$), so restart counts drop and a run may overshoot its deadline by one seed (up to 6.7 ms observed at $C = 7$).
- Timings were taken under concurrent load: compare rows of the same table, not absolute check counts across phases.

## Era benchmark (phase-14 integration)

All registered eras through the suite, equal time, exact MILP references (isolated measurements, M1 Max, 10 workers for post-processing):

```sh
ERAS=$(.venv/bin/python -c "from circle_optimization.eras import ERAS; print(' '.join(ERAS))")
bash -c ".venv/bin/python benchmark_guided_search.py --suite --no-classic --eras $ERAS \
  --sizes 160 1000 --characteristic-counts 5 7 --instances 3 --trials 3 --repeats 1 \
  --protocol time --time-limit 0.2 --workers 10 --output benchmark-results/eras-phase14"
.venv/bin/python benchmark_guided_search.py --suite --no-classic \
  --eras hc-p hc-p-splice hc-p-rc hc-p-prop hc-p-grasp hc-p-ils ga-p ga-e ga-m ga-e-seed ga-m-seed multistart exact \
  --sizes 10000 --characteristic-counts 8 --instances 2 --trials 2 --repeats 1 \
  --protocol time --time-limit 1.0 --workers 10 --output benchmark-results/eras-phase14-large
```

(zsh does not word-split `$ERAS`; run the first command under bash.) $684 + 52$ measured runs; 135 s and 74 s wall time. Their manifests still read `algorithm_revision: phase07` (the label was bumped to `phase14` afterwards); the source fingerprint in their checkpoints identifies the exact code. Results tables are in [docs/ERAS.md](../../ERAS.md#results-snapshot). For S6: `hc-p-grasp` reaches the optimum in 18/18 $C=5$ runs and is within 1.0% / 0.28% at $C=7$ ($N=160$ / $N=1000$); `hc-p-ils` is equal or worse everywhere except $C=7$, $N=160$ (0.95% vs 1.00%, within noise). Ablation of `hc-p-grasp` on $N=1000$, $C=7$ (guided tour, 3 seeds, 0.2 s): removing greedy starts raises the gap from 0.21% to 4.39%.
