# `classic/` — hill climbing and the GA on the $N \times N$ matrix (phases 01–06)

The lab's original heuristics. A genome is a tuple of child indices with child 0 fixed at position 0; a circle and its reverse share one canonical form. Distances come from the $N \times N$ matrix (`problem.distance_matrix`), so memory is $O(N^2)$. Standard library only.

## Modules

| Module | Contents |
| --- | --- |
| `operators.py` | `sample_pair`, `uniform_swap_positions`, `apply_swap`, `swap_delta`, `inversion_delta` ($O(1)$ move scores), `random_individual`, `initial_population`, `tournament_selection`, `order_crossover` (OX), `swap_mutation`, `inversion_mutation`, `mutate` |
| `guidance.py` | `WeakEdgeSwapSelector` (phase 02), `guided_swap_mutation`, `make_guided_mutation` |
| `hill_climbing.py` | `_run_hill_climber` engine; `run_hill_climber`, `run_inversion_hill_climber`, `run_guided_hill_climber` |
| `genetic.py` | `FitnessEvaluator`, `evolve` (generation engine shared with the layered GAs), `run_ga`, `check_ga_settings`, `positional_diversity` |
| `diagnostics.py` | `SearchDiagnostics`: novelty, edge coverage and diversity for untimed replays |

## Hill climbing

One engine, several proposal policies:

1. Start from a random circle (one full score = one check).
2. Propose a swap or a segment reversal (2-opt). Score it from the affected edges only: a reversal replacing $(a, b)$, $(c, d)$ with $(a, c)$, $(b, d)$ changes the score by $\Delta = D_{ac} + D_{bd} - D_{ab} - D_{cd}$; a swap touches at most four edges. Each proposal is one check, accepted only if strictly improving.
3. Restart according to the policy:
   - **productive** (default since phase 04): after `stall_limit` (200) consecutive failed proposals; any improvement resets the patience.
   - **legacy** (phases 01–03): additionally after `attempts_per_restart` (1,000) proposals, even if the search is still improving.
4. Stop at `evaluation_budget`, `max_restarts`, `deadline` (absolute `perf_counter`) or `score_bound` (a proven upper bound, e.g. the optimum).

| Entry point | Proposal mix | Era |
| --- | --- | --- |
| `run_hill_climber(restart_policy="legacy")` | 50% swaps / 50% reversals | `hc-legacy` (01) |
| `run_guided_hill_climber(exploration=0.2, restart_policy="legacy")` | swaps break weak edges (weight $1 + \max D - d$), 20% uniform | `hc-guided` (02) |
| `run_hill_climber()` | 50/50, productive restarts | `hc` (04) |
| `run_inversion_hill_climber(inversion_rate=0.8)` | 80% reversals / 20% swaps | `hc-i` (04) |

Results: `circle`, `score`, `evaluations` (checks), `restarts`, `stop_reason`, `restart_policy`, `restart_history` (per local search: attempts, accepted, final score, exit reason). `on_improvement(score, evaluations)` fires on the first circle and every strict global improvement.

## Genetic algorithm

`run_ga(distances, rng, *, population_size, generations, tournament_size, elites, crossover_rate, mutation_rate, evaluation_budget, mutation_operator, on_improvement, deadline, diagnostics, score_bound)`:

- distinct random initial population; `elites` best survive unchanged;
- parents by tournament of `tournament_size` distinct members;
- order crossover with `crossover_rate`: copy a tail slice of parent A, fill the rest in parent B's cyclic order (child 0 stays anchored);
- one mutation per offspring with `mutation_rate` (swap or reversal, or the phase-02 guided swap via `mutation_operator=make_guided_mutation(...)`);
- duplicates in a generation are rejected (random fallback after many);
- fitness cached by canonical genome; `evaluations` = distinct fitness computations; the budget check reserves a generation's worth of scores.

The generation loop is `evolve(...)`, a representation-agnostic engine with injected scoring, keys, operators and hooks. `run_ga` wires it to child genomes; `layered/genetic.py` reuses it for the profile and edge-genome GAs. Its random draws are identical to the original implementation: frozen trajectory hashes in `tests/test_ga_optimization.py` guard this.

Eras: `ga` (01, uniform mutation), `ga-guided` (02, weak-edge mutation).

## Example

```python
from random import Random
from circle_optimization import distance_matrix, random_profiles, run_ga, run_inversion_hill_climber

distances = distance_matrix(random_profiles(2027, 160, 5))
hc = run_inversion_hill_climber(distances, Random(1), evaluation_budget=20_000)
ga = run_ga(distances, Random(1), population_size=50, generations=600, tournament_size=10,
            elites=2, crossover_rate=0.9, mutation_rate=0.25, evaluation_budget=20_000)
print(hc["score"], ga["score"])   # the optimum of this instance is 748
```

## Measured standing

At $N = 160$, $C = 5$ the best classic eras stay 7–15% below the proven optimum (GA 7.15% at 20,000 evaluations; HC-I 9.42% at 0.2 s), and 17–46% at $N = 1000$ within 0.2 s. Why, and what fixes it: [docs/history/phases/07-profile-milp.md](../../docs/history/phases/07-profile-milp.md) and the layered eras ([layered/README.md](../layered/README.md)).

Records: phases 01–03 in [PHASES.md](../../docs/history/PHASES.md), [GUIDED_SEARCH.md](../../docs/GUIDED_SEARCH.md), phases [04](../../docs/history/phases/04-throughput-and-search.md)–[06](../../docs/history/phases/06-faster-suite.md).
