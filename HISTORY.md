# History line — how the solutions improved, and why

The assignment: seat N children with C Boolean characteristics in a circle so that the sum of Hamming distances between neighbours is maximal, using a **hill climber (HC)** and a **genetic algorithm (GA)**. This page follows the work from that baseline to the final algorithms, one step at a time: the problem observed, the concept introduced, and the measured effect. Detailed records (design, verification, exact commands) are in [`project/docs/history/`](project/docs/history/PHASES.md); every version named here (an *era*) can still be run, e.g. `python -m circle_optimization run hc-legacy` from `project/`.

Unless stated otherwise, numbers are for the benchmark instance $N = 160$, $C = 5$ (seed 2027, optimum 748) or the equal-time era benchmark (0.2 s per run, mean gap to the proven optimum).

---

## Stage A — the lab's solution (phases 00–02)

### 1. Model the problem (phase 00)

- **Representation.** A circle is an order of child indices. Rotations and reflections are the same circle, so child 0 is fixed first and the smaller of the two directions is kept: one canonical tuple per circle, usable as a cache key.
- **Score.** Precompute the N × N Hamming matrix once $O(N²C)$; a circle then costs $O(N)$ lookups.
- **Reference.** Exhaustive enumeration of the $(N − 1)!/2$ circles gives the true optimum, but only up to $N ≈ 10$.

### 2. Hill climber and GA (phase 01) — eras `hc-legacy`, `ga`

- **Moves.** A *swap* exchanges two children; an *inversion* reverses a segment, which is the classic 2-opt move for tours (it replaces two edges).
- **HC.** Propose a random move, keep it only if it strictly improves, restart from a random circle when stuck (200 failures or 1,000 proposals).
- **GA.** Tournament selection, **order crossover (OX)** (copy a slice of one parent, fill the rest in the other parent's order, so every child appears once), one swap/inversion mutation per child, elitism, duplicate rejection, fitness cache.

### 3. Score moves, not circles (phase 02) — `hc-guided`, `ga-guided`

- **Delta evaluation.** A move changes at most four edges, so its effect is computed in O(1) — for an inversion `D[a][c] + D[b][d] − D[a][b] − D[c][d]` — without building the new circle. HC can afford far more proposals.
- **Weak-edge guidance.** Bias swaps toward breaking short edges. A sensible idea that measured little benefit; kept as an era for comparison.

**Standing at the end of stage A:** `hc-legacy` 16.4% and `ga` 14.7% below the optimum — but at the time the optimum was unknown above N ≈ 10, so this gap was invisible.

---

## Stage B — measure properly, then improve the classics (phases 03–06)

### 4. A fair benchmark (phase 03)

Same instances and seeds for every algorithm; nested sizes; two protocols: **equal checks** (a work-count budget) and **equal time** (a wall-clock allowance). HC counts cheap delta checks and GA counts full evaluations, so equal counts are not equal work; rankings use equal time. Statistics are paired by trial with clustered bootstrap intervals; an HTML report shows scaling, convergence and target attainment.

### 5. Productive restarts and inversion focus (phase 04) — `hc`, `hc-i`

- **Observation.** 19 of 20 local searches at N = 160 ended at the 1,000-proposal cap while still improving.
- **Productive patience.** Restart only after consecutive failures; every improvement resets the patience. N = 160 HC at equal checks: 608.8 → 645.5.
- **Inversion focus.** 80% inversions (2-opt suits tours) — `hc-i`, the best classic HC (9.4% gap).
- **GA fast paths** that consume identical random draws: same results, about 2× faster, guarded by frozen trajectory hashes.

### 6. Faster experiments (phases 05–06)

Parallel post-processing, checkpoints, stopping a search once it reaches a proven bound (it cannot improve further, so its result and trace are unchanged), one repeat for deterministic count runs: the full grid fell from 3,165 s (phase 03 code) to 1,142 s (phase 04) and 499 s (phase 06), the last step reproducing phase 04's fixed-count scores exactly.

---

## Stage C — know the optimum (phase 07)

### 7. The exact profile-aggregated solver — eras `exact`, `multistart`

- **Key observation.** With C characteristics there are at most $2^C$ profiles (32 for C = 5), so for large N most children share a profile and are **interchangeable**: swapping two same-profile children changes nothing.
- **Aggregation.** A circle is then fully described, up to relabelling, by an **m × m edge-count table**: how many neighbouring pairs join profiles p and q. A table is a valid circle iff every profile has degree $2·n_p$ and the table is connected; an **Euler circuit** turns it back into a circle. The problem size becomes $m(m − 1)/2 + m$ variables, *independent of N*.
- **MILP.** Maximize $Σ d_pq·x_pq$ subject to the degree equations, with connectivity cuts added lazily (only for components that actually appear). A multi-start heuristic (greedy matching + local search) provides the warm start; an LP phase gives a tight bound; **reduced-cost fixing** removes about 94% of the variables at $C = 8$; disconnected optima are cut off and repaired. The solver reports "optimal" only with a proof.
- **Effect.** Optimum of $N = 1,000,000$ children (C = 8) proven in 0.7 s; every benchmark instance in milliseconds. The benchmark finally measured true gaps: the classic eras were **7–15% below the optimum at N = 160**.
- **Diagnosis.** From the classic algorithms' final circles, moves on the aggregated table alone closed 61–98% of the gap, and 11–62% of their edges joined profile pairs that the LP proves sub-optimal. The searches were not short of time; they were missing moves and information.

---

## Stage D — bring the insight back into HC and GA (phases 08–14)

### 8. Eras and layers (phase 08)

Every algorithm version is registered as an era with its phase and parent and stays runnable. Each new idea is a **layer**: one module and one switch whose off value reproduces the parent exactly, so the HC remains an HC and the GA a GA, and any layer can be removed again to measure its contribution.

### 9. S2 — profile state (phase 09) — `hc-p`, `ga-p`

Store the circle as a **sequence of profile ids** and read distances from the m × m table. Same moves, same decisions: `hc-p` reproduces `hc` move for move, but memory drops from $O(N²)$ to $O(N + m²)$ ($N = 10⁵$ would otherwise need $10¹⁰$ matrix cells), and later layers can see profile identities.

### 10. S1 — split-and-splice move (phase 10) — `hc-p-splice`

Cutting two edges leaves two reconnections. One is the inversion. The other splits the circle into two loops — useless for a single tour, unless both loops contain a child of the **same profile**: exchanging those two children's successors re-joins the loops at zero cost. The move's score change is still O(1). At HC's local optima about as many improving splices exist as improving inversions. Effect: 10.8% → 9.3%.

### 11. S3 — LP reduced-cost guidance (phase 11) — `hc-p-rc`

The LP relaxation gives every profile pair a **reduced cost** rc ≤ 0, and every circle satisfies `score ≤ LP bound − Σ |rc|` over its edges: an edge on a pair with rc < 0 provably costs points, while optimal circles use only rc = 0 pairs (here 125 of 496). The guide breaks the worst edge of a small random tournament and re-attaches it to an rc = 0 partner; 20% of proposals stay uniform. The certificate Σ |rc| even tells how far a circle is from the bound. Effect: 9.3% → **1.4%**, the largest single HC gain.

### 12. S4 — edge-genome GA and an optional memetic layer (phase 12) — `ga-e`, `ga-m`

OX preserves the *order* of parents, but the score depends on *adjacency*: whenever OX skips a gene, it creates an edge neither parent had. The edge GA's genome is the edge-count table: a child keeps all edges its parents share, adds edges from their union, completes degrees greedily by distance and repairs connectivity. That alone beats OX (`ga-e` 9.9% vs `ga` 14.7%). The **memetic layer** — local search on 20% of the children — is a separate switch, so the pure GA and the hybrid stay comparable: very strong at N = 160 (0.6%), too slow per child at N = 1000 within 0.2 s.

### 13. S5 — move propagation (phase 13) — `hc-p-prop`

A move's gain depends only on the four profiles involved, so after an accepted move the same pattern is retried at its other occurrences. Mixed result: helps at N = 1000, C = 5 (4.6% → 2.4%), neutral at N = 160, harmful at N = 10⁴, where each acceptance pays an O(N) scan while most moves still improve anyway. Recorded as a measured limitation.

### 14. S6 — greedy seeding (phase 14) — `hc-p-grasp`, `ga-e-seed`

Random circles start at about half the optimum. **GRASP** restarts build each start with the randomized greedy construction of stage C; the GA seeds 10% of its population the same way. HC and GA still do the climbing and the recombination. Effect: **0% gap** at C = 5 for both; 0.28% (`hc-p-grasp`) and 0.12% (`ga-e-seed`) at N = 1000, C = 7. The alternative, iterated local search with double-bridge kicks (`hc-p-ils`), was never better and stays for comparison.

---

## Where it ended


| Era           | Step | C=5, N=160 | C=5, N=1000 | C=7, N=1000 |
| ------------- | ---- | ---------: | ----------: | ----------: |
| `hc-legacy`   | 2    |     16.40% |      35.12% |      36.48% |
| `ga`          | 2    |     14.72% |      43.39% |      43.65% |
| `hc-i`        | 5    |      9.42% |      17.20% |      20.15% |
| `hc-p-splice` | 10   |      9.25% |      16.90% |      19.60% |
| `hc-p-rc`     | 11   |      1.38% |       4.55% |       5.26% |
| `ga-e`        | 12   |      9.85% |      26.82% |      17.81% |
| `hc-p-grasp`  | 14   |  **0.00%** |   **0.00%** |       0.28% |
| `ga-e-seed`   | 14   |  **0.00%** |       0.03% |       0.12% |
| `exact`       | 7    |      0.00% |       0.00% |       0.07% |

At N = 10,000, C = 8 (1 s per run), where the classic eras cannot run at all, `hc-p-grasp` is 0.20% and `ga-e-seed` 0.14% from the optimum, which the exact solver proves in about 0.6 s.

## Lessons

1. **Measure against the truth.** Upper bounds hid a 7–15% gap; the exact solver turned "good enough" into a measurable problem.
2. **Representation decides what a search can see.** Profiles, not children, carry the information; every big gain came from using them (splice moves, reduced costs, edge genomes, greedy seeds).
3. **Use what the exact model knows inside the heuristic.** LP reduced costs are cheap and turn blind proposals into targeted ones.
4. **Keep every version runnable.** Equal-time, era-against-era comparisons and one-switch ablations showed which ideas helped (S3, S6), which only helped in some regimes (S5, memetic), and which did not.
