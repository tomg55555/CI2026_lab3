# Weak-edge-guided swap selection

For nested size sweeps, exploration response statistics, both budget protocols, and interactive graphs, see [BENCHMARK_SUITE.md](BENCHMARK_SUITE.md).

## Why guide a swap?

The circle's score is the sum of distances between current neighbors. A small edge score is a possible improvement opportunity. Uniform HC spends proposals on every pair of positions equally; guided HC spends more of its swap proposals trying to replace poor neighboring pairs.

This changes **which move is proposed**, not the acceptance rule. A swap affects up to four edges, so breaking a weak edge can still make the total score worse. HC checks the complete score delta and accepts only strictly positive changes. It remains a heuristic and does not prove an optimal circle.

## The probability rule

Let $M$ be the largest entry in the fixed distance matrix $D$. For current edge $k$, between $c_k$ (`circle[k]`) and $c_{(k+1) \bmod N}$ (`circle[(k + 1) % N]`), define:

$$
w_k = 1 + M - D_{c_k,\, c_{(k+1) \bmod N}}, \qquad
P(\text{edge } k \mid \text{guided proposal}) = \frac{w_k}{\sum_{j} w_j}
$$

Poor edges get larger weights. The added 1 keeps every edge eligible, including maximum-distance edges. These are probabilities of selecting **current edges**, not independent probabilities applied to every possible swap. One swap is proposed at a time. Edges with equal distances have equal weights; no new profile comparisons are needed.

Each swap proposal follows this procedure:

1. With probability `exploration` (default **0.2**), choose an ordinary uniform swap of two non-anchor positions.
2. Otherwise sample an edge using the weights above.
3. Select either endpoint with equal probability, unless one is child 0, which must stay anchored. Move the other endpoint in that case.
4. Choose a uniform legal partner elsewhere in the circle. Exclude both selected edge endpoints and the stationary endpoint's other neighbor, so the weak pair actually stops being adjacent. If the chosen endpoint has no legal partner (possible with four children), move the other endpoint instead.
5. Compute the swap delta. HC accepts only if the sum of all affected edges improves; otherwise the circle stays unchanged.

With fewer than four children, the selector uses uniform swaps: for three children every pair is already adjacent. HC returns the single possible score without proposing swaps when $N < 3$.

The uniform fraction provides broader exploration. It also makes every ordinary swap eligible even if the guided branch would not choose it. `exploration=1` reproduces the baseline HC's proposals and seeded results (the suite therefore copies it from the uniform control at a count cap); `exploration=0` uses only guided swaps. The default is an experiment setting, not a tuned optimum.

### Interpreting exploration

`exploration` is a probability **conditional on proposing a swap**, not a probability of accepting worse moves or restarting. At `exploration=0.2`, about 20% of swap proposals are uniform and 80% target weak current edges. The guided branch is still stochastic, even at `exploration=0`.

HC chooses inversions half the time. Consequently, across 1,000 proposed HC moves, the expected mix at 0.2 is 500 inversions, 100 uniform swaps, and 400 guided swaps (not exact quotas). Uniform swaps can explore pairs the guided branch would neglect. Larger values reduce the bias and usually reduce its selection overhead; they do not guarantee better or worse solutions. HC still rejects every non-improving move, so this setting alone cannot escape all local optima; random restarts provide another route.

In GA, this split is conditional on an offspring actually receiving a mutation. The benchmark's current `mutation_rate=0.25` means approximately 25% of offspring receive a mutation; of those, half receive an inversion, and half a swap. Exploration then controls only that swap half. It is separate from the mutation rate and does not affect either uniform algorithm.

## What changed in the code

Module paths follow the phase-08 layout (`circle_optimization/classic/`):

- [`operators.py`](../circle_optimization/classic/operators.py) owns uniform position selection, in-place application and constant-time swap/inversion deltas.
- [`guidance.py`](../circle_optimization/classic/guidance.py) owns the weighted selector, the GA swap adapter and the optional GA mutation factory.
- [`hill_climbing.py`](../circle_optimization/classic/hill_climbing.py) owns the common acceptance/restart loop and the uniform, inversion-focused and `run_guided_hill_climber` entry points.
- [`genetic.py`](../circle_optimization/classic/genetic.py) uses the shared swap operation. Its default mutation remains a 50/50 uniform swap/inversion mix. An optional `mutation_operator` argument allows the guided mix to be tested on the same GA implementation.

Guided HC still proposes swaps and inversions with equal probability. Inversion selection, strict improvement, random starting circles, restart limits, and stall limits match the baseline. A guided sampler is invalidated after **any** accepted move, including an inversion, and at every restart. Both HC versions count one score check per initial circle or proposed move, including repeats.

### Selection cost and caching

Finding $M$ costs $O(N^2)$ once per guided HC run or GA mutation-factory creation. It is not recomputed for each proposal. Cumulative edge weights cost $O(N)$ to build. Sampling from them uses binary search, costing $O(\log N)$, followed by constant-size endpoint/partner selection. The swap delta itself is $O(1)$.

HC builds the cumulative distribution lazily at the next guided swap and reuses it while the circle is unchanged. Rejected moves and uniform proposals do not rebuild it. An accepted swap or inversion invalidates it. This implementation rebuilds all cumulative weights after a change rather than maintaining a tree with incremental updates. That keeps the code small, but means accepted moves can incur an extra $O(N)$ rebuild before a subsequent guided proposal. Accepted inversions also still cost up to $O(N)$ to apply, as in baseline HC.

GA visits different genomes, so a guided swap must build weights for that genome. The factory reuses the matrix maximum, but cannot reuse the previous genome's edge distribution. That overhead is part of the GA benchmark.

## Use guided HC

```python
from random import Random
from circle_optimization import run_hill_climber, run_guided_hill_climber

# distances is a precomputed symmetric matrix shared by both searches.
baseline = run_hill_climber(distances, Random(42), evaluation_budget=10_000)
guided = run_guided_hill_climber(
    distances, Random(42), evaluation_budget=10_000, exploration=0.2,
)
print(baseline["score"], guided["score"])
```

The same numeric seed makes each run reproducible. Guiding proposals consumes random numbers differently, so the two runs do not follow identical moves.

## Use the same guidance in GA

```python
from random import Random
from circle_optimization import make_guided_mutation, run_ga

result = run_ga(
    distances, Random(42), population_size=50, generations=200,
    tournament_size=10, elites=2, crossover_rate=0.9,
    mutation_rate=0.5, evaluation_budget=10_000,
    mutation_operator=make_guided_mutation(distances, exploration=0.2),
)
```

Create the mutation operator once for the corresponding distance matrix. It proposes 50% guided swaps and 50% ordinary inversions. `mutation_rate=0.5` still means that each offspring has a 50% chance of receiving one mutation; it is distinct from the 50/50 choice of mutation type and from `exploration`. GA mutations can worsen fitness: tournament selection and elitism determine survival. There is no HC-style local acceptance test in the GA mutation.

For a swap-only operator, bind `guided_swap_mutation` to a `classic.guidance.WeakEdgeSwapSelector` using `functools.partial`. The same selector must use the GA's matrix; the adapter invalidates weights between genomes.

## Correctness and performance tests

From the repository root:

```sh
python -m unittest -v tests.test_circle_optimization tests.test_guided_search
python -m unittest -v tests.test_search_progress
python benchmark_guided_search.py --instances 3 --trials 10 --repeats 3
python benchmark_guided_search.py --instances 3 --trials 10 --repeats 3 --include-ga
```

Use `.venv/bin/python` instead of `python` when using the local virtual environment. The benchmark is standard-library-only. Useful options:

- `--budget 10000`: the same maximum score-check count for both HC versions.
- `--children 40 --characteristics 100`: problem size, matching the GA notebook.
- `--seed 2026`: deterministic instance and search seeds.
- `--exploration 0.2`: vary the proportion of uniform swaps.
- `--details`: print per-trial seeds, scores, actual counts, and runtime.

The benchmark generates each instance once, shares its matrix across algorithms, warms each implementation, and rotates execution order between trials/repeats. Each repeat starts with the same seed, so score and count must agree; it reports the median runtime across repeats. Matrix construction is excluded; validation and selector setup are included. Result validation runs outside the timed block. It reports mean score and sample standard deviation, best score, median runtime, actual counts, and paired guided-vs-uniform score wins/ties/losses and time ratios.

### What the budget controls

`--budget` applies separately to **each algorithm execution**, not to all instances, trials, or repeats together. It is a count limit, not milliseconds. For HC, a random initial circle consumes one check and each proposed swap or inversion consumes another, including rejected/repeated moves. A restart's initial score also consumes one. With a budget of 100, HC could score one initial circle and try 99 moves, or distribute those checks across restarts.

GA counts distinct complete-circle scores, cached by canonical genome. Its initial population consumes evaluations; cache hits, parent selections, and duplicate-generation attempts consume time but not new evaluations. Before starting a new generation, GA conservatively reserves enough count capacity for a full population, so unused capacity may remain.

Increasing the budget permits more search and may improve the best retained score; it may instead add time after the best score has already been reached. With the same seed and other settings, a larger count cap does not discard an earlier best result. A smaller cap typically saves time and can stop before a useful improvement. Budgets remain upper bounds: HC's restart cap and GA's generation cap can stop searches earlier.

In particular, this benchmark fixes GA at 200 generations. With 50 population members and 2 elites, it evaluates at most $50 + 200 \times 48 = 9650$ distinct circles. A 50,000 budget therefore cannot give that GA additional generations; raising an already non-binding cap has no effect on its search. The actual count can be lower because of caching.

### First time reaching the final best score

The benchmark now measures **first-hit time to each run's own final best score**:

- `median ms`: total search duration, including subsequent unsuccessful search.
- `to-best ms`: median time when the eventual final score was first found.
- `best@check`: median check/evaluation count at that first occurrence.
- `--details` adds `evaluations_to_best` and `seconds_to_best` for each trial.

Each search accepts an optional `on_improvement(score, evaluations)` callback. It fires for the first scored circle and every strict global improvement. Ties do not reset the timestamp, and a restart that merely matches a previous best does not reset it either. GA reports immediately when an individual is scored, including inside the initial population, not just at generation boundaries. The callback does not alter random decisions or stopping rules.

The benchmark timestamps these notifications using the same starting clock as total runtime, so setup/validation is included and matrix construction excluded. It takes median first-hit time across identical seeded repeats and then across trials, just as it does with total runtime. Only strict improvement notifications read the clock, adding small instrumentation overhead to the measured searches.

For example, a run that finds score 180 at check 2,000 / 4 ms, then stops after 10,000 checks / 20 ms without improving it, reports a 4 ms first-hit time and 20 ms total time. It could not know at 4 ms that no later improvement would occur: "final best" is identified retrospectively from the completed run.

This is not time to a proven optimum. Also, each algorithm may finish at a different score: reaching 175 quickly is not necessarily better than reaching 185 later. Read the score alongside first-hit time. A comparison to a common target score would be a separate experiment, with some runs possibly never reaching that target.

The optional GA pair uses identical GA settings and initial-population seeds; only mutation guidance differs. GA counts **distinct full-circle evaluations**, so its counts are not equal work to HC checks. GA's generation limit can also stop a run before its evaluation cap. HC can stop at its restart limit.

An equal-check experiment isolates whether guided proposals improve quality and how much selection overhead they add. It is **not an equal-time experiment**: better scores with longer runtimes do not establish better efficiency. Try several budgets (for example 1,000, 5,000, and 10,000) to inspect score/time trade-offs. Repeating the experiment across instances/seeds matters: weighting can focus on inherently poor edges with no profitable swap, and raw distances may be closely clustered, giving only a mild bias. No automated test asserts a speedup or score advantage. Correctness tests cover deltas, weak-edge bias, anchor/wraparound handling, small circles, reproducibility, budgets, exact small-instance bounds, and GA integration.

### Initial measured results

Command used on the local environment:

```sh
.venv/bin/python benchmark_guided_search.py --instances 3 --trials 10 --repeats 3 --include-ga --characteristics 100 --seed 2026 --budget 10000
```

With 40 children, 100 characteristics, seed 2026, a 10,000-check cap, and `exploration=0.2`, the 30 paired trials produced:

These historical measurements used GA mutation rate 0.5; the benchmark's current rate is 0.25, so rerunning with current settings is a different experiment.

| Algorithm | Mean score | Score standard deviation | Best score | Median search time |
| --- | ---: | ---: | ---: | ---: |
| Uniform HC | 2306.60 | 6.75 | 2322 | 17.697 ms |
| Guided HC | 2309.00 | 9.08 | 2326 | 20.533 ms |
| Uniform GA | 2325.00 | 9.79 | 2348 | 339.178 ms |
| Guided GA | 2323.13 | 8.64 | 2348 | 349.694 ms |

Guided HC won/tied/lost on score in **14/3/13** paired trials, with a mean score change of **+2.40** and a median paired runtime ratio of **$1.16\times$**. Both HC versions used all 10,000 checks in these runs. Guided GA won/tied/lost **13/1/16**, with a mean score change of **-1.87** and runtime ratio **$1.04\times$**. The GA variants averaged 8607.4 and 8602.9 distinct evaluations respectively.

This shows a small HC score gain at an added selection cost, not a demonstrated equal-time advantage. This sample does not support preferring guidance in GA. Times are machine-dependent, and the score spread is across these selected instances/trials rather than a confidence interval for all possible problems.
