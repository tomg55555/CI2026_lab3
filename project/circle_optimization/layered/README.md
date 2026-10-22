# `layered/` — profile-aware HC and GA built from optional layers (phases 09–14)

The phase-07 analysis showed what the classic searches miss. From their final circles, aggregated 2-exchanges alone closed 61–98% of the gap to the optimum, and 11–62% of their edges joined profile pairs that the LP proves sub-optimal. This package carries those insights back into HC and GA as **layers**: one documented module and one engine switch each. Each switch's off value reproduces the parent era, which keeps the searches HC and GA in nature.

Needs numpy; layer S3 (and the eras using it) also needs highspy.

## Layers

| Layer | Phase | Module | Idea | Engine switch (off value) |
| --- | --- | --- | --- | --- |
| **S2** profile state | 09 | `state.py` | Store a circle as its profile-id sequence and read distances from the $m \times m$ table; memory $O(N + m^2)$ instead of $O(N^2)$ | always on |
| **S1** splice | 10 | `splice.py` | The reversal's second reconnection splits the circle into two loops; re-join them for free through two same-profile children (score change still $O(1)$) | `splice_rate` (0) |
| **S3** reduced costs | 11 | `reduced_costs.py` | LP duals give each profile pair a reduced cost $\mathrm{rc} \le 0$, and $\text{score} \le \text{LP bound} - \sum \lvert \mathrm{rc} \rvert$ over a circle's edges; break the worst edge of a small tournament and re-attach to an $\mathrm{rc} = 0$ partner | `guide` (None) |
| **S4** edge genome | 12 | `edge_genome.py` | Genome = edge-count table; crossover keeps the parents' shared edges, fills from their union, completes degrees (random or greedy) and repairs connectivity; mutation = one connected 2-exchange | GA engine choice, `fill` |
| **S4** memetic | 12 | `memetic.py` | Optional bulk 2-exchange local search on GA children, its checks charged to the budget | `local_search` (None) |
| **S5** propagation | 13 | `propagation.py` | An accepted move's gain depends only on the four profiles involved, so retry the same pattern at its other occurrences | `propagate` (False) |
| **S6** restarts | 14 | `restarts.py` | Greedy GRASP seeds (randomized b-matching + Euler circuit), double-bridge kicks for iterated local search, seeded GA populations | `start` ("random"), `perturbation` (None), `seeding` (0) |

Key functions: `ProfileState.from_profiles / sequence / score / circle / multigraph`; `splice_delta`, `apply_splice`; `reduced_cost_table`, `ReducedCostGuide.from_state(...).select / certificate`; `random_genome`, `edge_crossover`, `edge_mutation`; `MemeticLayer(max_moves, probability).improve`; `Propagation.from_move(...).next_move`; `seed_multigraph`, `seed_sequence`, `double_bridge`.

## Engines

**`hill_climbing.run_layered_hill_climber(state, rng, *, evaluation_budget, ...)`** is the classic HC loop on profile sequences (strict improvement, productive or legacy restarts, budget/deadline/bound stops), plus:

| Parameter | Layer | Meaning |
| --- | --- | --- |
| `inversion_rate`, `splice_rate` | S1 | Proposal mix; swap share $= 1 - \mathtt{inversion\_rate} - \mathtt{splice\_rate}$. `splice_rate=0` keeps the classic random draws |
| `guide` | S3 | `ReducedCostGuide` choosing reversal/splice positions |
| `propagate` | S5 | Retry an accepted reversal/splice pattern ($\le 64$ proposals per trigger) |
| `start`, `start_noise` | S6 | `"random"` or `"greedy"` (GRASP) starts and restarts |
| `perturbation` | S6 | `"double-bridge"`: every restart after the first kicks the incumbent |

With every layer off it reproduces `classic.run_hill_climber` exactly (scores, counts, restart history, improvement trace). Extra result keys: `sequence`, `layers`, `moves` (proposed/accepted per move kind).

**`genetic.run_profile_ga(state, rng, *, ..., seeding=0.0)`** (GA-P) is the classic GA scored through the profile table, with cache and duplicate rejection keyed by profile sequence. **`genetic.run_edge_ga(state, rng, *, ..., fill="random", local_search=None, seeding=0.0)`** (GA-E / GA-M) evolves edge tables. Both run on `classic.genetic.evolve`, so selection, elitism, budgets and stop rules are the classic GA's. GA-M's `evaluations` add the memetic checks to the distinct fitness computations.

## Eras built here

| Era | Phase | Parent | Settings |
| --- | --- | --- | --- |
| `hc-p` | 09 | `hc` | S2 only (identical trajectory to `hc`) |
| `hc-p-splice` | 10 | `hc-p` | `inversion_rate=0.45, splice_rate=0.45` |
| `hc-p-rc` | 11 | `hc-p-splice` | + `guide_exploration=0.2` (tournament 8) |
| `hc-p-prop` | 13 | `hc-p-rc` | + `propagate=True` |
| `hc-p-grasp` | 14 | `hc-p-prop` | + `start="greedy", start_noise=0.5` (recommended) |
| `hc-p-ils` | 14 | `hc-p-grasp` | + `perturbation="double-bridge"` (kept for comparison) |
| `ga-p` | 09 | `ga` | S2 only |
| `ga-e` | 12 | `ga-p` | edge genome, `fill="greedy"` |
| `ga-m` | 12 | `ga-e` | + `memetic={"max_moves": None, "probability": 0.2}` |
| `ga-e-seed`, `ga-m-seed` | 14 | `ga-e`, `ga-m` | + `seeding=0.1` |

Every layered HC era accepts all HC switches and every edge-GA era all GA switches (`guide_exploration`, `guide_tournament`, `propagate`, `start`, `start_noise`, `perturbation`, `splice_rate`, `inversion_rate`; `fill`, `memetic`, `seeding`), so any combination is one override away:

```python
from circle_optimization.eras import Problem, run_era

problem = Problem.random(1000, 7, seed=2027)
full = run_era("hc-p-grasp", problem, seed=1, time_limit=0.2)
no_s6 = run_era("hc-p-grasp", problem, seed=1, time_limit=0.2, start="random")
pure = run_era("ga-m", problem, seed=1, time_limit=0.2, memetic=None)          # = ga-e
level_a = run_era("ga-e", problem, seed=1, time_limit=0.2, fill="random")
```

## What each layer bought (equal time, 0.2 s; mean gap to the optimum)

| Step | $C=5$, $N=160$ | $C=5$, $N=1000$ | $C=7$, $N=1000$ |
| --- | ---: | ---: | ---: |
| `hc-p` (S2) | 10.84% | 18.15% | 20.86% |
| + S1 `hc-p-splice` | 9.25% | 16.90% | 19.60% |
| + S3 `hc-p-rc` | 1.38% | 4.55% | 5.26% |
| + S5 `hc-p-prop` | 1.50% | 2.36% | 5.02% |
| + S6 `hc-p-grasp` | 0.00% | 0.00% | 0.28% |
| `ga-p` (S2) | 16.29% | 42.36% | 42.72% |
| `ga-e` (S4) | 9.85% | 26.82% | 17.81% |
| `ga-m` (S4 + memetic) | 0.60% | 41.60% | 35.60% |
| `ga-e-seed` (S4 + S6) | 0.00% | 0.03% | 0.12% |

Caveats measured in the records: S5 costs throughput while a very large circle is still descending ($N = 10^4$: 20.0% vs 11.6% without it); the memetic layer needs enough time per child (wins at $N = 160$, too slow at $N = 1000$ in 0.2 s); double-bridge ILS was never better than GRASP restarts on these uniform instances. Records: phases [09](../../docs/history/phases/09-profile-state.md), [10](../../docs/history/phases/10-splice.md), [11](../../docs/history/phases/11-reduced-cost-guidance.md), [12](../../docs/history/phases/12-edge-ga-memetic.md), [13](../../docs/history/phases/13-propagation.md), [14](../../docs/history/phases/14-seeding-and-ils.md).

## Adding a layer

1. Write `docs/history/phases/NN-name.md` (plan) first.
2. Put the idea in a new module here, with a docstring on every function.
3. Add an engine parameter whose default leaves the engine unchanged, and a test proving the default reproduces the parent era.
4. Register an era in `eras.py` (add the parameter to `_layered_defaults` or the GA defaults so other eras can toggle it); `tests/test_eras.py` then runs it automatically.
5. Measure against the parent with equal time and update the record.
