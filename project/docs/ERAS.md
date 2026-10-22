# Algorithm eras

An **era** is one algorithm version as it stood after a phase of the [history](history/PHASES.md). Every era stays runnable by name, so past and present versions can be compared on the same instance with the same limits. The registry lives in [`circle_optimization/eras.py`](../circle_optimization/eras.py).

## Lineage

```
hc-legacy (01) ── hc-guided (02)
      └── hc (04) ── hc-i (04)
            └── hc-p (09, S2) ── hc-p-splice (10, S1) ── hc-p-rc (11, S3) ── hc-p-prop (13, S5)
                                                                                └── hc-p-grasp (14, S6) ── hc-p-ils (14, S6 ILS)
ga (01) ── ga-guided (02)
  └── ga-p (09, S2) ── ga-e (12, S4) ── ga-m (12, S4 + memetic) ── ga-m-seed (14, S6)
                            └── ga-e-seed (14, S6)
multistart (07) ── exact (07)
```

Each arrow adds one documented change; `python -m circle_optimization describe NAME` prints an era's record.

| Era | Phase | Representation | Adds | Record |
| --- | --- | --- | --- | --- |
| `hc-legacy` | 01 | child genome, $N \times N$ matrix | Delta-scored 50/50 swap/inversion HC; restart after 200 failures or 1,000 proposals | [phases](history/PHASES.md) |
| `ga` | 01 | child genome, $N \times N$ matrix | Tournament, OX, swap/inversion mutation, elitism, fitness cache | [phases](history/PHASES.md) |
| `hc-guided` | 02 | child genome | Weak-edge-guided swaps ($w = 1 + \max D - d$) | [GUIDED_SEARCH](GUIDED_SEARCH.md) |
| `ga-guided` | 02 | child genome | Weak-edge-guided swap mutation | [GUIDED_SEARCH](GUIDED_SEARCH.md) |
| `hc` | 04 | child genome | Productive restarts (patience reset by improvements) | [04](history/phases/04-throughput-and-search.md) |
| `hc-i` | 04 | child genome | 80% inversions / 20% swaps | [04](history/phases/04-throughput-and-search.md) |
| `multistart` | 07 | profile multigraph | Greedy b-matching, merge, bulk 2-exchange, repeated | [07](history/phases/07-profile-milp.md) |
| `exact` | 07 | profile multigraph | HiGHS MILP with lazy connectivity cuts; proven optimum | [07](history/phases/07-profile-milp.md) |
| `hc-p` | 09 | profile sequence, $m \times m$ table | S2: same trajectory as `hc`, memory $O(N + m^2)$ | [09](history/phases/09-profile-state.md) |
| `ga-p` | 09 | child genome, $m \times m$ table | S2: profile-table fitness, profile-keyed cache | [09](history/phases/09-profile-state.md) |
| `hc-p-splice` | 10 | profile sequence | S1: split-and-splice move (45%) | [10](history/phases/10-splice.md) |
| `hc-p-rc` | 11 | profile sequence | S3: LP reduced-cost guided positions | [11](history/phases/11-reduced-cost-guidance.md) |
| `ga-e` | 12 | edge-count table | S4: edge-preserving crossover, greedy fill | [12](history/phases/12-edge-ga-memetic.md) |
| `ga-m` | 12 | edge-count table | S4 memetic layer: bulk 2-exchange on 20% of children | [12](history/phases/12-edge-ga-memetic.md) |
| `hc-p-prop` | 13 | profile sequence | S5: retry an accepted move's pattern elsewhere | [13](history/phases/13-propagation.md) |
| `hc-p-grasp` | 14 | profile sequence | S6: greedy GRASP starts and restarts (noise 0.5) | [14](history/phases/14-seeding-and-ils.md) |
| `hc-p-ils` | 14 | profile sequence | S6 alternative: double-bridge kicks of the incumbent | [14](history/phases/14-seeding-and-ils.md) |
| `ga-e-seed` | 14 | edge-count table | S6: 10% greedy seeds in the initial population | [14](history/phases/14-seeding-and-ils.md) |
| `ga-m-seed` | 14 | edge-count table | S6 on top of the memetic GA | [14](history/phases/14-seeding-and-ils.md) |

## Layers

Each layer is one module under `circle_optimization/layered/` and one switch of an engine (`layered/hill_climbing.run_layered_hill_climber`, `layered/genetic.run_profile_ga` / `run_edge_ga`). The registry keys below are the override names accepted by `run_era` and `--set`.

| Layer | Module | Registry keys (off value) | Idea |
| --- | --- | --- | --- |
| S2 profile state | `state.py` | always on | Circle = profile-id sequence; distances from the $m \times m$ table |
| S1 splice | `splice.py` | `splice_rate` (0), `inversion_rate` | The inversion's second reconnection splits the circle; re-join it free through two same-profile children |
| S3 reduced costs | `reduced_costs.py` | `guide_exploration` (null), `guide_tournament` | LP duals: $\text{score} \le \text{LP} - \sum \lvert \mathrm{rc} \rvert$; break negative-$\mathrm{rc}$ edges, create $\mathrm{rc} = 0$ pairs |
| S4 edge genome | `edge_genome.py` | `fill` ("greedy"/"random") | Genome = edge-count table; children inherit common edges |
| S4 memetic | `memetic.py` | `memetic` (null or `{"max_moves": .., "probability": ..}`) | Optional local search on children |
| S5 propagation | `propagation.py` | `propagate` (false) | Same profile pattern $\Rightarrow$ same gain at its other occurrences |
| S6 restarts | `restarts.py` | `start` ("random"), `start_noise`, `perturbation` (null), `seeding` (0) | Greedy GRASP starts, double-bridge ILS, seeded GA populations |

Every layered HC era carries all HC layer keys, and every edge-GA era all GA keys, so any layer can be removed or added to any era of its family.

## Running eras

Python:

```python
from circle_optimization.eras import Problem, run_era

problem = Problem.random(160, 5, seed=2027)       # the benchmark suite's instance
result = run_era("hc-p-grasp", problem, seed=1, time_limit=0.2)
print(result["score"], problem.optimum, result["layers"])
ablated = run_era("hc-p-grasp", problem, seed=1, time_limit=0.2, guide_exploration=None)
```

Command line:

```sh
python -m circle_optimization list
python -m circle_optimization describe ga-m
python -m circle_optimization run hc-p-rc --children 1000 --time-limit 0.5 --optimum
python -m circle_optimization run ga-m --set memetic=null          # = ga-e
python -m circle_optimization compare --family HC --children 160 --repeats 5
python -m circle_optimization compare --all --children 1000 --json compare.json
```

Benchmark suite: `--eras NAME ...` adds eras as families; `--no-classic` drops the classic exploration grid (see [BENCHMARK_SUITE.md](BENCHMARK_SUITE.md)). The guided walk-through is [`guided_tour.ipynb`](../guided_tour.ipynb).

## Budgets and fairness

| Era group | `evaluation_budget` unit |
| --- | --- |
| HC eras (classic and layered) | One full score per (re)start + one delta per proposal (propagated proposals included) |
| `ga`, `ga-guided`, `ga-p`, `ga-e`, `ga-e-seed` | Distinct fitness computations (cache hits free) |
| `ga-m`, `ga-m-seed` | Distinct fitness computations + memetic checks ($E^2$ per local-search step) |
| `multistart` | Completed starts |
| `exact` | Ignored (deadline honoured) |

Units differ, so rank eras with equal time (`time_limit`). With only a time limit, restart/generation/count caps are lifted. Precomputations an era needs for itself (S3's LP solve, S6 seeds) run inside the timed region; the shared instance representations (`Problem.distances`, `Problem.state`) do not.

## Results snapshot

Equal time, exact-optimum references, mean gap to the optimum (runs that hit it). Reproduce with the commands in [phase 14](history/phases/14-seeding-and-ils.md#era-benchmark-phase-14-integration).

**0.2 s per run**, 3 instance families $\times$ 3 trials (`benchmark-results/eras-phase14`):

| Era | Phase | $C=5$, $N=160$ | $C=5$, $N=1000$ | $C=7$, $N=160$ | $C=7$, $N=1000$ |
| --- | --- | ---: | ---: | ---: | ---: |
| `hc-legacy` | 01 | 16.40% (0/9) | 35.12% (0/9) | 18.90% (0/9) | 36.48% (0/9) |
| `ga` | 01 | 14.72% (0/9) | 43.39% (0/9) | 18.14% (0/9) | 43.65% (0/9) |
| `hc-guided` | 02 | 16.41% (0/9) | 35.58% (0/9) | 18.84% (0/9) | 44.23% (0/9) |
| `ga-guided` | 02 | 15.36% (0/9) | 45.44% (0/9) | 17.95% (0/9) | 45.75% (0/9) |
| `hc` | 04 | 10.84% (0/9) | 18.60% (0/9) | 12.85% (0/9) | 20.98% (0/9) |
| `hc-i` | 04 | 9.42% (0/9) | 17.20% (0/9) | 12.03% (0/9) | 20.15% (0/9) |
| `multistart` | 07 | 0.00% (9/9) | 0.00% (9/9) | 1.28% (0/9) | 0.38% (0/9) |
| `exact` | 07 | 0.00% (9/9) | 0.00% (9/9) | 0.30% (7/9) | 0.07% (8/9) |
| `hc-p` | 09 | 10.84% (0/9) | 18.15% (0/9) | 12.85% (0/9) | 20.86% (0/9) |
| `ga-p` | 09 | 16.29% (0/9) | 42.36% (0/9) | 19.23% (0/9) | 42.72% (0/9) |
| `hc-p-splice` | 10 | 9.25% (0/9) | 16.90% (0/9) | 11.10% (0/9) | 19.60% (0/9) |
| `hc-p-rc` | 11 | 1.38% (0/9) | 4.55% (0/9) | 3.13% (0/9) | 5.26% (0/9) |
| `ga-e` | 12 | 9.85% (0/9) | 26.82% (0/9) | 6.29% (0/9) | 17.81% (0/9) |
| `ga-m` | 12 | 0.60% (3/9) | 41.60% (0/9) | 2.75% (0/9) | 35.60% (0/9) |
| `hc-p-prop` | 13 | 1.50% (0/9) | 2.36% (0/9) | 3.26% (0/9) | 5.02% (0/9) |
| `hc-p-grasp` | 14 | 0.00% (9/9) | 0.00% (9/9) | 1.00% (0/9) | 0.28% (0/9) |
| `hc-p-ils` | 14 | 0.03% (8/9) | 0.02% (6/9) | 0.95% (0/9) | 0.50% (0/9) |
| `ga-e-seed` | 14 | 0.00% (9/9) | 0.03% (7/9) | 0.22% (1/9) | 0.12% (0/9) |
| `ga-m-seed` | 14 | 0.00% (9/9) | 0.09% (6/9) | 0.41% (3/9) | 0.44% (1/9) |

**$N = 10{,}000$, $C = 8$, 1 s per run**, $2 \times 2$ trials, layered eras only — the classic eras would need a $10^8$-cell matrix (`benchmark-results/eras-phase14-large`):

| Era | Phase | $C=8$, $N=10000$ |
| --- | --- | ---: |
| `multistart` | 07 | 0.19% (0/4) |
| `exact` | 07 | 0.00% (4/4) |
| `hc-p` | 09 | 26.95% (0/4) |
| `ga-p` | 09 | 48.12% (0/4) |
| `hc-p-splice` | 10 | 26.05% (0/4) |
| `hc-p-rc` | 11 | 11.57% (0/4) |
| `ga-e` | 12 | 26.52% (0/4) |
| `ga-m` | 12 | 41.16% (0/4) |
| `hc-p-prop` | 13 | 20.02% (0/4) |
| `hc-p-grasp` | 14 | 0.20% (0/4) |
| `hc-p-ils` | 14 | 0.23% (0/4) |
| `ga-e-seed` | 14 | 0.14% (0/4) |
| `ga-m-seed` | 14 | 0.62% (0/4) |

Reading: S2 alone changes nothing (by design); S1 adds about 1.5 points; S3 is the largest single HC gain; S6 greedy starts close the remaining gap at $C=5$ and dominate the ablation at $C=7$. S5 helps at $N=1000$, $C=5$ but costs throughput while a large circle is still descending ($N=10^4$). In the GA family the edge crossover beats OX, the memetic layer needs enough time per child (strong at $N=160$, too slow at $N=1000$ in 0.2 s), and seeding brings every edge GA to within 0.5%. One classic `hc-guided` run at $N=1000$, $C=7$ produced no score within 0.2 s (selector setup over the $10^6$-cell matrix) and counts as zero.

## Adding an era

1. Write the phase record first (`docs/history/phases/NN-*.md`, plan status).
2. Put the new idea in its own documented module (a layer) and expose it as an optional engine parameter whose default reproduces the previous behaviour; add a test that the default is unchanged.
3. Register an `Era` in `eras.py` (phase, extends, layers, defaults, budget unit). `tests/test_eras.py` then runs it automatically.
4. Measure it against its parent with equal time, update the record, the phase index and [STATE_OF_ART](history/STATE_OF_ART.md).
