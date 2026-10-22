# Decision log

## 2026-10-21 — establish history before phase 04 code changes

Record the baseline and requested phases first. Keep old experimental results historical; new algorithm semantics and execution modes must be identifiable.

## 2026-10-21 — distinguish isolated timing from parallel throughput

Default timing mode will remain isolated: measured searches run without other benchmark CPU jobs; diagnostics and statistical batches may run in parallel after measurement. An explicit throughput mode may parallelize measured batches too. It must label timings and deadline outcomes as CPU-contention-sensitive, not as an interchangeable replacement for historical isolated measurements.

Processes, not Python threads, will execute CPU-bound search batches. Seeds depend on experiment identity, not worker or completion order. Parallel execution must not reduce instances, repeats, budgets, diagnostics, or bootstrap samples.

## 2026-10-21 — preserve controls while improving algorithms

Uniform HC remains a 50/50 proposal control. Add a separate inversion-focused entry point with a configurable minority of swaps. Expose an explicit legacy hard-cap policy for historical HC comparisons. GA speedups must preserve random draws, ties, offspring order, fixed-count scores and improvement traces.

## 2026-10-21 — completed phase-04 decisions

- Productive patience is now the default HC policy; `legacy` is an explicit historical control. Inversion-focused HC coexists as HC-I with swaps retained.
- Reuse known scores/canonical keys in GA rather than weakening fitness budgets. Frozen trajectory hashes and an independent original-OX oracle verify behavior.
- Cache exact diagnostic edge sets (bounded to 1,024 tours per process) and hierarchical bootstrap sampling weights. Caching reduces duplicate computation, not coverage observations, diagnostic snapshots, or statistical resamples.
- Protect historical outputs and validate checkpoint config/source/Python/platform compatibility. Completed batches survive interruption; source/config changes require a separate experiment rather than a mixed result set.
- Keep report/JSON regeneration available from saved data, independent of searches.
- Small workload speed measurements are recorded as such; do not extrapolate a guaranteed duration for the older 53-minute full suite.

## 2026-10-21 — skip work whose outcome is already known (phase 06)

- Stop searches at the problem reference. It is an exact optimum or a proven upper bound, so the best circle, score and trace cannot change; only elapsed-time/count metadata and duration-normalized metrics do.
- Supersedes the phase-04 rule against repeat reduction for the **checks** protocol only: one repeat by default, since fixed-count repeats are identical by assertion and only re-time the same outcome. Time-protocol repeats are unchanged.
- Copy checks-protocol exploration $= 1$ from the uniform control, which consumes the same random draws. Copied rows are labelled `aliased_from`; time-protocol exploration $= 1$ remains measured.
- Each default has a flag restoring phase-04 semantics. Accepted trade-off: checks-protocol `seconds` is one sample, and exploration $= 1$ shows no separate wrapper overhead in the checks protocol.
- Replace the bounded exact edge-set cache with integer edge ids and incremental HC edges. Coverage definitions, snapshots and observations are unchanged.


## 2026-10-21 — exact profile-aggregated reference (phase 07)

- Solve the problem exactly on the $m$ distinct profiles instead of the $N$ children: edge-multiplicity MILP with lazy connectivity cuts (HiGHS via `highspy`). Size is $O(m^2)$, independent of $N$; $m \le \min(N, 2^C)$.
- Default suite reference becomes this exact optimum for every $N$ (`--reference milp`); enumeration up to `--exact-limit` must agree with it. `--reference bound` restores phase-06 references. Scores and traces cannot change (stopping at an exact optimum only skips work after the final best), but `reference_ratio`, `relative_gap` and targets of $N >$ `exact_limit` do.
- Accepted dependency: numpy + highspy for the default reference, mirroring the plotly report dependency. Searches, `--reference bound` export and package import remain standard-library only (lazy `__getattr__` exports).
- The multi-start heuristic is a warm start and early proof, not the exact method; the MILP only reports `optimal` with a proof (bound, LP or MILP).
- Bounds are rounded down to even scores (every cut of an even-degree multigraph is even). Reduced-cost fixing targets the even LP bound; if the reduced model cannot reach it, the bound drops and full column bounds return.

## 2026-10-21 — eras as first-class, layered enhancements (phases 08–14)

- Organize the package by era (`classic/`, `exact/`, `layered/`, `benchmark/`) and register every algorithm version in `eras.py`; each runs by name from Python, `python -m circle_optimization` and the suite (`--eras`). Old eras are never edited in place: a change is a new layer plus a new era.
- Every layer is a separate module and an engine parameter whose default reproduces the parent era (tested: `hc-p` = `hc` exactly; GA-P = `run_ga` when profiles are distinct). Layered eras share all switches of their family, so ablation is an override, not new code.
- Extract the classic GA generation loop as `classic.genetic.evolve` and run GA-P and the edge GA on it, rather than duplicating it; frozen trajectory hashes guard `run_ga`.
- The memetic local search is an optional `MemeticLayer`, not part of the edge crossover, so GA-E (pure GA) and GA-M (memetic) stay comparable. Its checks count toward the budget; prefer equal time for GA-M.
- Algorithm-own precomputations (S3's LP solve, S6 seeds) are charged to the timed run; shared instance representations are built before timing.
- S6: greedy GRASP restarts (`hc-p-grasp`, noise 0.5) are the recommended HC variant; double-bridge ILS (`hc-p-ils`) measured worse on every uniform instance and is kept only as a comparison era.
- GA defaults from the phase-12 probe: greedy fill, memetic probability 0.2 with unbounded moves, seeding fraction 0.1. The edge GA scores children at birth so local-search work is not lost to a mid-generation deadline.

## 2026-10-22 — publish a reviewer layer over the full project (phase 15)

- The assignment is an HC and a GA; the published top level presents that work and its improvement story briefly (README, HISTORY, walkthrough) and keeps the full engineering work one folder down (`project/`).
- Nothing is removed: every era stays registered so any comparison can be re-run; the walkthrough shows the first versions, the major classic one and the enhancements.
