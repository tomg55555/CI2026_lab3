# Phase 06 — faster suite without changing scores

Started/completed: 2026-10-21. Status: **completed and verified**.

This record was written during implementation rather than before it, contrary to the update rules. The pre-change state is pinned by the trajectory fingerprints described under verification, and by `benchmark-results/full-phase04`.

## Starting point

Phase-05 full grid (`full-phase04`, 1,152.49 s total, 10 workers, isolated):

| Phase | Seconds | Observation |
| --- | ---: | --- |
| Measurements (serial) | 975.70 | checks 583.5 s (GA 539.5 s), time 390.6 s |
| Diagnostics | 165.36 | 667 CPU-s of replays: ideal 67 s on 10 workers |
| Statistics, export, report | 11.2 | not a bottleneck |

- Every $N=10$ run (all algorithms, both protocols) reached its upper bound, which is also the exact optimum (36/40/36). GA still ran 600 generations: 0.38 s each.
- Checks-protocol repeats were identical by assertion; repeat 2 cost 292 s.
- Checks-protocol guided exploration $= 1$ matched uniform in 300/300 runs (score, count, count-to-best); those runs cost 94 s.
- HC diagnostic replays cost $15\times$ the plain search at $N=160$ (0.61 s vs 0.04 s), mostly building a 160-edge frozenset for each proposal.

## Scope

Tier 1, no score changes:

1. Stop a search once it reaches its proven reference score.
2. Faster diagnostic replays and better diagnostic scheduling.
3. GA/HC hot paths that consume identical random draws.

Tier 2, experimental-design changes approved by the user:

4. One checks-protocol repeat by default (`--check-repeats`).
5. Copy checks-protocol exploration $= 1$ rows from the uniform control.

## Why the reference is a proof of optimality

The score is $\sum_k \mathrm{flips}_k$, where $\mathrm{flips}_k$ (`flips_k`) counts the circle edges whose endpoints differ on feature $k$. The reference is the exact optimum for $N \le$ `exact_limit`. Otherwise it is

$$
\min\left( N C,\ \sum_k 2 \min(\mathrm{true}_k, \mathrm{false}_k),\ \left\lfloor \frac{1}{2} \sum_i \left( d_i^{(1)} + d_i^{(2)} \right) \right\rfloor \right)
$$

rounded down to an even number, where $\mathrm{true}_k$ and $\mathrm{false}_k$ count the children with and without feature $k$, and $d_i^{(1)} \ge d_i^{(2)}$ are the two largest distances from child $i$:

- each edge contributes at most $C$;
- feature $k$ alternates in runs around the circle, so $\mathrm{flips}_k$ is twice the number of true runs and twice the number of false runs; each run holds $\ge 1$ child;
- each child has two neighbours: its two edges are at most its two largest distances, and summing over children counts every edge twice;
- every $\mathrm{flips}_k$ is even, so the total is even.

No circle can exceed the reference. Both algorithms replace their best only on a strict improvement, so stopping on reaching it cannot change the returned circle, score, improvement trace or target hits. Checked against enumeration: bounds 20/30/36, 24/32/40 and 18/26/36 versus optima 18/30/36, 24/32/40 and 18/24/36 (seeds 2027–2029, $N \in \{6, 8, 10\}$). The bound is valid and sometimes loose; a loose bound simply never triggers.

## Changes actually implemented

- `score_bound` on `run_hill_climber`, `run_guided_hill_climber`, `run_inversion_hill_climber` and `run_ga`; stop reason `bound_reached`. The suite passes the problem reference unless `--no-stop-at-bound`.
- `SuiteConfig.check_repeats` (default 1) and `alias_full_exploration` (default on). Copied rows carry `aliased_from`; their traces, targets and diagnostics are copied with the new identity. Time-protocol rows are always measured.
- Diagnostics: integer edge ids from a cached $N \times N$ table instead of frozensets of tuples; HC proposals add only the $\le 4$ edges a move can create; repeated proposals and already-proposed GA evaluations skip edge updates entirely.
- Diagnostic replays are one job per variant (825 instead of 75 batches), submitted largest cost first; results keep their original order.
- `sample_pair` reproduces `Random.sample(range(n), 2)` draws for swap, inversion and OX positions; `tournament_selection` reproduces `Random.sample`'s pool branch inline. `circle_fitness` uses C-level `map`/`getitem`; `canonical_circle` decides direction from the second item when possible.
- `compare_benchmark_runs.py` counts per-protocol repeats and accepts a shorter count for a `bound_reached` GA run when score and trace match.
- Manifest `algorithm_revision` is now `phase06`, with `stopping`/`aliasing` notes.

## Verification

- 44 tests pass (40 before; new: bound stop, aliasing, sampling equivalence, incremental edge coverage, cost-ordered pool submission).
- Frozen GA trajectory hashes and the original-OX oracle still match.
- 180 pre-change fingerprints match exactly, with no bound: result, improvement trace, GA history, diagnostic summary and every diagnostic snapshot. They cover $N \in \{5, 10, 23, 40, 160\}$, 2 seeds, all 9 variants of a 3-exploration grid, with and without diagnostics. Probe script kept outside the repository.

Same-machine probes (M1 Max, Python 3.14.3, median of 3, before → after):

| Search, 20,000 checks | Before | After |
| --- | ---: | ---: |
| GA uniform $N=160$ | 1.024 s | 0.692 s |
| GA guided 0.2 $N=160$ | 1.007 s | 0.779 s |
| GA uniform $N=10$ | 0.352 s | 0.277 s; 0.005 s with bound stop |
| GA uniform $N=160$, diagnostics | 2.331 s | 1.392 s |
| HC uniform $N=160$, diagnostics | 0.613 s | 0.095 s |
| Diagnostic CPU per paired trial, all sizes | 44.5 s | 21.6 s (no bound stop) |

Faster code lets time-protocol searches do more work within the same allowance, so time-protocol outcomes are not comparable to phase 04 as pure speedups.

## Full-grid measurement

```sh
.venv/bin/python benchmark_guided_search.py --suite \
  --sizes 10 20 40 80 160 --explorations 0 0.2 0.5 0.8 1 \
  --instances 3 --trials 5 --repeats 2 --protocol both --budget 20000 --time-limit 0.2 \
  --workers 10 --timing-mode isolated --progress-interval 30 \
  --output benchmark-results/full-phase06
```

Same settings and machine as `full-phase04`, with the phase-06 defaults (one checks repeat, bound stop, copied exploration $= 1$). 2,775 timed searches + 150 copied rows (phase 04: 3,900 timed); 825 diagnostic replays (phase 04: 975).

| Phase | full-phase04 | full-phase06 |
| --- | ---: | ---: |
| Measurements | 975.70 s | 454.37 s |
| ↳ checks-protocol search time | 583.5 s | 153.9 s |
| ↳ time-protocol search time | 390.6 s | 299.4 s |
| Diagnostics (10 workers) | 165.36 s | 36.31 s |
| Statistics | 1.07 s | 0.69 s |
| Before export (`elapsed_seconds`) | 1,142.36 s | 491.64 s |
| Total including export/report | 1,152.49 s | **499.36 s** |

**$2.31\times$ faster / 56.7% shorter** end to end. This is one observed run, not a guaranteed ratio for other machines or grids.

Integrity, from `compare_benchmark_runs.py full-phase04 full-phase06` and a direct row check:

- All 2,925 rows have status `ok`; every diagnostic replay matched its timed run.
- All 975 checks-protocol rows match phase 04 in score and count-to-best, and in count unless stopped at the bound. All 900 GA traces match.
- `bound_reached`: 232 checks and 488 time-protocol executions ($N \in \{10, 20\}$).

Time-protocol outcomes moved, because faster code does more work in 0.2 s. Mean uniform scores at $N=160$: GA 614.47 → 628.67, HC 656.27 → 659.20, HC-I 666.53 → 669.07. Treat these as a new implementation's equal-time measurement, not as a search-quality improvement at fixed work.
