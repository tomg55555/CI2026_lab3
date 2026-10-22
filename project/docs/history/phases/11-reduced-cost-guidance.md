# Phase 11 — layer S3: LP reduced-cost guidance

Started/completed: 2026-10-21. Status: **completed and verified** (HC layer).

## Motivation

The phase-07 transfer probe found that 11–62% of the edges of HC/GA local optima join profile pairs with negative LP reduced cost, while only 10–27% of all pairs have zero reduced cost. Uniform proposals spend most checks on cuts and new edges that the LP already proves useless. The LP relaxation is cheap (one solve per instance, milliseconds at $m = 32$) and stays fixed for the whole search, so it can steer proposals without per-move maintenance.

## Theory

Let $\mathrm{rc}_j$ be the reduced costs of the connected LP relaxation (columns $x_{pq}$, $p < q$, and loop columns $y_p$) at its optimum $x^*$, and $\mathrm{LP}$ its value. Every column satisfies $c_j = \mathrm{rc}_j + \sum_i u_i a_{ij}$ ($u$: row duals). For the edge counts $x$ of any circle, the degree rows hold with equality and the cut rows with $a_i \cdot x \ge 2 = a_i \cdot x^*$ where $u_i \le 0$, so

$$
\begin{aligned}
\mathrm{score}(x) &= \mathrm{LP} + \sum_j \mathrm{rc}_j (x_j - x_j^*) + \sum_{\text{cuts}} u_i (a_i \cdot x - a_i \cdot x^*) \\
&\le \mathrm{LP} - \mathrm{certificate}(x), \qquad
\mathrm{certificate}(x) = \sum_j \mathrm{rc}_j x_j^* - \sum_j \mathrm{rc}_j x_j .
\end{aligned}
$$

$\mathrm{rc}_j < 0$ only at $x_j^* = 0$ and $\mathrm{rc}_j > 0$ only at the upper bound, so every term is $\ge 0$: each circle edge on a pair with $\mathrm{rc} < 0$ costs at least $\lvert \mathrm{rc} \rvert$ against the bound, and a circle reaching the bound uses only pairs with $\mathrm{rc} \ge 0$ (up to the integrality slack). With zero cut duals the inequality is an identity, which matches the observations: on all probes below the certificate equals $\mathrm{LP} - \mathrm{score}$.

## Design

`circle_optimization/exact/milp.py` — new public `lp_relaxation(instance)`: the LP phase of `solve_profile_milp` on its own (continuous model, one cut per fractional-support component, re-solve until connected). Returns the bound and $m \times m$ `values`, `reduced_costs` and column `upper` (diagonal: loops), plus `rounds`, `cuts`, `connected`. $m = 1$ is answered without HiGHS. The MILP code path is untouched. HiGHS sign convention checked empirically: $\mathrm{rc} < 0 \Rightarrow x^* = 0$, $\mathrm{rc} > 0 \Rightarrow x^* = \text{upper}$.

`circle_optimization/layered/reduced_costs.py`:

- `reduced_cost_table(state)`: wrapper zeroing $\lvert \mathrm{rc} \rvert \le 10^{-7}$;
- `ReducedCostGuide(reduced_costs, values, table, bound=..., exploration=0.2, tournament=8, partner_tries=4)` and `ReducedCostGuide.from_state(state, ...)`; tables are converted to tuples of Python floats/ints;
- `select(seq, rng, *, crossed=False) -> (i, j)`, $1 \le i < j \le N-1$:
  1. with probability `exploration` (or if inactive, or $N < 5$): uniform, exactly the base engine's `sorted(uniform_swap_positions(n, rng))`;
  2. worst edge: tournament over $k$ uniform edges by $\text{badness} = -\mathrm{rc}$ for penalized pairs, otherwise $-d / (d_{\max} + 1)$ (below every penalized edge; lower distance is worse, the phase-02 idea);
  3. keep one endpoint (random side) and place the other cut so the endpoint's new neighbour has profile $q$ with $\mathrm{rc} \ge 0$: $q$ is drawn from the endpoint's precomputed candidate list and located as its first occurrence at or after a uniform offset (C-level `list.index`). The inversion joins left–left/right–right endpoints, the splice left–right/right–left, whichever edge comes first, so the partner edge is $f = t - (\text{side} \oplus \text{crossed})$. Adjacent cuts (null moves) are retried up to `partner_tries` times, then uniform.
- `certificate(seq)` / `gap_certificate(guide, seq)`: the proven gap lower bound above, $O(N)$; `bound` is the LP value.
- Degenerate cases: $m = 1$ or no negative reduced cost $\Rightarrow$ `active = False`, every proposal uniform.

The first partner design rejection-sampled uniform positions until one held a candidate profile (8 tries). It missed 26% of the time on a random circle at $N=160$ because some profiles have only 2 candidate partners out of 32; the index-based draw always hits and is cheaper (1.0 vs 0.74 hit rate).

Engine: `guide` replaces the uniform position draw of inversions and splices (swaps stay uniform). Sampling computes no delta, so it costs no check.

## Verification

`tests/test_layered_hc.py::ReducedCostTests`:

- 27 instances ($N = 3, \dots, 9$ random, plus $N \in \{40, 160\}$): LP connected, $\text{bound} \ge$ exact optimum (`solve_circle`), symmetric matrices, $\mathrm{rc} < 0 \Rightarrow x^* = 0$, $\mathrm{rc} > 0 \Rightarrow x^* = \text{upper}$; for 20 random circles and the optimal circle, $\text{certificate} \ge 0$ and $\text{score} + \text{certificate} \le \text{bound}$;
- positions always valid for $N = 5, \dots, 160$ and both move kinds; with exploration 0 every guided proposal creates an LP-compatible edge (rate 1.0 vs 0.50 uniform) and removes a penalized edge more often than uniform;
- $m = 1$ and an all-zero rc table fall back to uniform; the engine runs with every layer on $m = 1$; invalid parameters raise.

Engine runs with the guide: valid, deterministic per seed, within budget, $\le$ the exact optimum (`LayeredEngineTests`).

## Measurement

Equal time 0.2 s, M1 Max under a concurrent load (4 worker processes, other jobs running), 5 rng seeds per instance; $N=160$ = `random_profiles(2027–2029, 160, 5)` (15 runs), N=1000 = `random_profiles(2027, 1000, 5)`. All on top of S1 with inversion/splice 0.45/0.45. Mean gap to the exact optimum and mean checks per run:

| S3 setting | $N=160$ gap | $N=160$ checks | $N=1000$ gap | $N=1000$ checks |
| --- | ---: | ---: | ---: | ---: |
| none (S1 only) | 8.82% | 174,697 | 17.08% | 134,232 |
| exploration 0.1, $k = 4$ | 1.53% | 95,315 | 6.54% | 72,870 |
| exploration 0.2, $k = 4$ | 1.92% | 100,066 | 6.76% | 73,997 |
| exploration 0.5, $k = 4$ | 2.39% | 114,937 | 8.81% | 87,070 |
| exploration 0.2, $k = 2$ | 2.35% | 106,796 | 8.04% | 80,856 |
| **exploration 0.2, $k = 8$** | **1.51%** | 85,929 | **6.39%** | 65,984 |
| exploration 0.1, $k = 8$ (second run) | 1.42% | 80,470 | 6.13% | 60,426 |
| exploration 0.2, $k = 16$ (second run) | 1.49% | 64,788 | 7.31% | 51,737 |

The guide halves throughput yet cuts the gap $5.8\times$ at $N=160$ and $2.7\times$ at $N=1000$; it is the largest single gain of phases 10–14. Default chosen: exploration 0.2, tournament 8 (0.1 vs 0.2 at $k = 8$ is within run-to-run noise).

Certificate probe (seed 2027, 3 rng seeds, 0.2 s, S1 = 0.45/0.45; LP build 3–35 ms; share of pairs with $\mathrm{rc} \ge 0$: 25% / 28% / 10%):

| Instance | stack | mean score | $\mathrm{LP} - \mathrm{score} = \mathrm{certificate}$ | circle edges with $\mathrm{rc} < 0$ |
| --- | --- | ---: | ---: | ---: |
| $N=160$, $C=5$ (LP 748) | S2 / S1 / S1+S3 | 666.7 / 678.7 / 738.0 | 81.3 / 69.3 / 10.0 | 23.1% / 20.8% / 2.9% |
| $N=1000$, $C=5$ (LP 4884) | S2 / S1 / S1+S3 | 3968.7 / 4048.7 / 4572.7 | 915.3 / 835.3 / 311.3 | 26.3% / 23.7% / 2.2% |
| $N=1000$, $C=7$ (LP 6722) | S2 / S1 / S1+S3 | 5341.3 / 5448.7 / 6410.0 | 1380.7 / 1273.3 / 312.0 | 50.3% / 47.8% / 6.6% |

## Limitations

- One LP solve per instance before the search (3–13 ms at $m = 32$, 35 ms at $m = 128$; untested for $m \approx 10^3$). Era wiring must decide whether that time is charged to an equal-time run; the measurement above excludes it.
- The guide steers inversions and splices only; swaps stay uniform.
- Once the certificate is small, the remaining gap sits on few edges and the exploration share matters more; no adaptive schedule was tried.
- `select` uses `int(rng.random() * n)` instead of `randrange` ($4\times$ cheaper); the bias is below $2^{-40}$ for practical $N$.
