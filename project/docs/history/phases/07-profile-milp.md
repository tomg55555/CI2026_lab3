# Phase 07 — exact profile-aggregated solver as benchmark reference

Started/completed: 2026-10-21. Status: **completed and verified**.

## Motivation

Above `--exact-limit` ($N \le 9$–$10$) the suite only knew a proven upper bound, so `relative_gap`, `reference_ratio` and targets of $N = 20, \dots, 160$ measured distance to a bound that may be unattainable. The phase-06 bound stop never fired there.

Children with equal profiles are interchangeable. With $m$ distinct profiles, $m \le \min(N, 2^C)$; for large $N$ and modest $C$, $m \ll N$.

## Formulation

A circle is a closed walk on the $m$ profiles visiting profile $p$ exactly $n_p$ times. Let $x_{pq}$ ($p<q$) count circle edges between profiles $p$ and $q$, and $y_p$ count edges between two children of $p$ (Hamming distance 0). Then

$$
\begin{aligned}
\max \quad & \sum_{p<q} d_{pq}\, x_{pq} \\
\text{s.t.} \quad & \sum_{q} x_{pq} + 2 y_p = 2 n_p && \text{for every profile } p \\
& \sum_{p \in S,\, q \notin S} x_{pq} \ge 2 && \text{for every } \emptyset \ne S \subsetneq \text{profiles} \\
& x_{pq} \in [0, 2 \min(n_p, n_q)], \quad y_p \in [0, n_p - 1] \ (m \ge 2), && \text{integer}
\end{aligned}
$$

Every connected multigraph with even degrees $2 n_p$ has an Euler circuit, which visits $p$ exactly $n_p$ times; assigning $p$'s children to those visits gives a circle with the same score. Conversely every circle yields such a multigraph. The optimum therefore equals the circle optimum, for any $N$, with $m(m-1)/2 + m$ columns and $m$ rows independent of $N$. The $y_p \le n_p - 1$ bound is the singleton cut. Every score is even: each feature's cut of an even-degree multigraph is even, so all bounds are rounded down to even.

## Implementation

`circle_optimization/profile_model.py`

- Aggregation packs rows with `np.packbits` into uint64 codes ($C \le 64$), then a dense `bincount` histogram when $2^C \le 2N$ ($C \le 24$), else `np.unique`; above 64 characteristics, row-wise `np.unique`. Distances are one $m \times m$ float64 product: $d = \lvert p \rvert + \lvert q \rvert - 2\, p \cdot q$ (exact for 0/1 rows).
- `profile_upper_bound`: the suite's `quality_bound` ($N C$, feature, degree; even) in $O(m^2 + mC)$ without the $N \times N$ matrix. Equality is tested.
- `euler_sequence`: Hierholzer on a reduced multigraph keeping 1 (odd) or 2 (even) copies per used pair — same support and degree parity — then splices removed copies back as $p, q, p, \dots$ excursions and loops as repeated visits. $O(m^2 + N)$ with the $N$ part in numpy.
- `assign_children`: two stable radix sorts map the $k$-th visit of $p$ to $p$'s $k$-th child; rotate child 0 first; canonical orientation.

`circle_optimization/profile_search.py` (multi-start heuristic, $m \times m$ state)

- Greedy b-matching: pairs by decreasing distance (+ uniform noise after start 0, random ties), each takes $\min(\mathrm{residual}_p, \mathrm{residual}_q)$; residuals stay even, the rest become loops.
- `connect`: merge the smallest component by the cheapest cross-component 2-exchange; moves that remove bridges on both sides are reverted.
- `improve`: best-improvement **bulk** 2-exchange. Both reconnections of two removed edge types are allowed if the profile graph stays connected (an Euler circuit re-threads the children). The gain is linear in copies moved, so a move transfers $\min(x_{ab}, x_{cd})$ copies (one fewer if a type vanishing disconnects). A positive gain implies added and removed types differ (triangle inequality), so degrees are preserved. First version moved one copy per step: $N=10^5$, $C=6$ took 28 s for 16 starts; bulk moves take 0.04 s.

`circle_optimization/profile_milp.py`

- HiGHS model built column-wise in one `addCols` call; complement pairs (distance $C$) receive seed cuts, the typical isolated 2-cycle of Hamming optima.
- LP phase: continuous relaxation; while its fractional support is disconnected, add one cut per component (one for two) and re-solve from the basis. Its even floor is the bound, and on all large probes it equalled the optimum.
- Target phase: reduced-cost fixing toward a target equal to the even LP bound. For any $x$ satisfying the model rows, $c \cdot x \le \mathrm{LP} - \sum_j \lvert \mathrm{rc}_j \rvert \, \lvert x_j - x_j^* \rvert$ (cut duals only lower it), so $\lvert x_j - x_j^* \rvert \le (\mathrm{LP} - \text{target}) / \lvert \mathrm{rc}_j \rvert$. About 94% of columns are fixed at $C=8$. If the reduced MIP cannot reach the target (infeasible or relaxation below it), the bound drops to $\text{target} - 2$ and full bounds return.
- MIP rounds: integer optima are frequently disconnected ties of equal value; each is cut off and repaired (connect + improve) into an incumbent. The loop ends when an incumbent meets the bound. HiGHS `time_limit` applies per `run()` (measured), so each round receives the remaining wall time.
- `solve_circle`: aggregate → bound → 4 heuristic starts (stop at bound) → MILP only if needed → Euler reconstruction; status `optimal` only with a proof (`bound`, `lp` or `milp`). Lazy package exports keep numpy/highspy optional for everything else.

Suite: `SuiteConfig.reference` (`milp` default, `bound` = phase 06) and `reference_time_limit` (60 s); `problem_reference` asserts enumeration and MILP agree when both apply; problems record `exact_source`, `milp_status`, `milp_upper_bound`, `milp_seconds`. Manifest revision `phase07`.

## Verification

- 56 tests pass (44 before; 12 new in `test_profile_solver.py`): 250 random instances $N \le 9$ equal enumeration (plus $\text{heuristic-only} \le \text{optimum} \le \text{bound}$); trivial $N \in \{1, 2, 3\}$ and identical children; complement clusters needing cuts; an LP integrality-gap instance (even LP bound 96, optimum 94) confirmed by an independent plain DFJ loop; $N=50{,}000$ instance; expired time limit; seeded multi-start; aggregation code paths; bound equality with `quality_bound`; bulk-move invariants; suite reference sources.
- Additional probe: 400 random instances $N \le 9$, $C \le 5$, both modes, 0 mismatches.

Same-machine probes (M1 Max, Python 3.14.3, highspy 1.15.1), `solve_circle` defaults, random uniform profiles:

| $N$ | $C$ | $m$ | optimum | proof | total seconds |
| ---: | ---: | ---: | ---: | --- | ---: |
| 160 (suite seed 2027) | 5 | 32 | 748 | bound | 0.002 |
| 10,000 | 5 | 32 | 49,532 | MILP | 0.022 |
| 100,000 | 6 | 64 | 597,834 | MILP | 0.038 |
| 1,000,000 | 6 | 64 | 5,994,322 | MILP | 0.211 |
| 2,000 | 8 | 256 | 15,486 | MILP | 0.707 |
| 1,000,000 | 8 | 256 | 7,989,886 | MILP | 0.667 |
| 500 | 9 | 323 | 4,160 | MILP | 2.563 |
| 3,000 | 10 | 977 | 28,912 | MILP | 27.15 |

Before the LP/target phases the $N=2{,}000$, $C=8$ MILP took 3.3–6.5 s; plain MIP rounds without repair needed up to 47 rounds on degenerate disconnected ties.

## Full-grid measurement

```sh
.venv/bin/python benchmark_guided_search.py --suite \
  --sizes 10 20 40 80 160 --explorations 0 0.2 0.5 0.8 1 \
  --instances 3 --trials 5 --repeats 2 --protocol both --budget 20000 --time-limit 0.2 \
  --workers 10 --timing-mode isolated --progress-interval 60 \
  --output benchmark-results/full-phase07
```

Same grid as `full-phase06`. 433.3 s total (measurements 390.4 s, diagnostics 34.7 s, preparation including all 15 MILP references 0.2 s; each $\le 14$ ms). All 2,925 rows `ok`, every reference `exact_optimum` (MILP proofs: 6 bound, 4 LP, 5 MILP). Enumeration and MILP agree at $N=10$.

Integrity against `full-phase06`: all 975 checks-protocol rows have identical score, count-to-best and strict-improvement trace. 129 runs at $N \in \{20, 40\}$ stop earlier (`bound_reached`), because the exact optimum is below the old bound (e.g. $N=40$ seed 2029: 172 vs 176). Time-protocol scores are a new isolated measurement, as in every phase.

Exact optima vs the old bound: $N=20$ 78/80/78 (bound 80/82/78), $N=40$ 176/170/172 (178/170/176), $N=80$ 346/358/344 (equal), $N=160$ 748/752/714 (748/754/718).

Mean gap to the exact optimum (uniform variants; 15 checks / 30 time runs):

| $N$ | GA checks | HC checks | HC-I checks | GA 0.2 s | HC 0.2 s | HC-I 0.2 s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.17% | 0.68% | 0.51% | 0.17% | 0.17% | 0.00% |
| 40 | 2.56% | 5.41% | 4.33% | 3.32% | 4.02% | 3.32% |
| 80 | 3.07% | 7.85% | 6.98% | 5.46% | 6.27% | 5.41% |
| 160 | 7.15% | 12.51% | 11.29% | 14.85% | 10.64% | 9.31% |

No HC/GA run reaches the optimum above $N=20$; the gap grows with $N$.

## Transfer probe (what HC/GA local optima miss)

Script kept outside the repository. Seeds 2027–2029, HC-I (0.2 s, 2 s) and GA (20,000 evaluations) final circles, converted to profile multigraphs:

| Instance | gap of HC-I 2 s / GA | closed by aggregated bulk 2-exchange alone | circle edges with LP reduced cost $< 0$ |
| --- | --- | --- | --- |
| $N=160$, $C=5$ (3 seeds) | 7.0–9.3% / 6.4–8.8% | 61–84% | 11–19% |
| $N=1000$, $C=5$ | 16.1% / 23.1% | 94% / 98% | 22% / 35% |
| $N=1000$, $C=7$ | 18.1% / 26.2% | 91% / 94% | 46% / 62% |

Only 10–27% of profile pairs have zero reduced cost; any circle reaching the LP bound uses only those (up to the slack). A uniform swap exchanges two same-profile children (a null move) with probability $\approx \sum_p n_p (n_p - 1) / \bigl(N (N - 1)\bigr)$: 2.9–3.3% at $C=5$, 0.8% at $C=7$ — minor.

## Follow-ups (not implemented)

Profile-aware HC/GA variants that keep their search nature: profile-id state with an $m \times m$ table (no $N \times N$ matrix), split-and-splice moves, LP-reduced-cost candidate lists, edge-preserving crossover on profile multigraphs, seeded starts. Each needs its own phase record and controls.

## Correction (2026-10-21, found in phase 14 measurements)

Under a short `time_limit` the MIP stage could stop before its root LP and HiGHS then reports an infinite `mip_dual_bound`; `_even_floor` raised `OverflowError` (5 of 59 limits between 5 and 295 ms on `random_profiles(2028, 160, 7)`). A non-finite dual bound now leaves the current bound unchanged. Regression test: `tests/test_profile_solver.py::test_any_time_limit_returns_a_valid_bounded_circle`. Results without a time limit were never affected. The module now lives at `circle_optimization/exact/milp.py` (phase 08).
