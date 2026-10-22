# Phase 12 — edge-genome GA and an optional memetic layer (S4)

Started/completed: 2026-10-21. Status: **completed and verified** (implementation, tests, short equal-time probe; not yet run through the full benchmark grid).

This record also documents, in their own sections, the GA engine of phase 09 (GA-P, layer S2) and GA seeding of phase 14 (layer S6), which share the code below; their own phase records may copy those sections.

## Motivation

Fitness depends only on which children are adjacent, and with interchangeable children only on the edge-count table $X$ of the profile multigraph (phase 07). Order crossover (OX) inherits the *relative order* of genes, not adjacencies. Concrete example with the repository's OX (`classic.operators.order_crossover`, draws of `Random(21)`, segment positions 1..3 of the non-anchor genes):

| | circle | edges |
| --- | --- | --- |
| parent A | 0 1 2 3 4 5 6 7 | 01 12 23 34 45 56 67 70 |
| parent B | 0 1 2 6 5 4 3 7 (A after one 2-opt move) | 01 12 26 65 54 43 37 70 |
| OX child | 0 5 2 3 4 7 1 6 | 05 52 23 34 47 71 16 60 |

The parents share 6 of their 8 edges (01 12 34 45 56 70). The child keeps one of them (34), inherits 23 from A, and its other 6 edges are in **neither** parent. Every edge is a score term, so the child is essentially a new random circle near the parents in order space, not in fitness space. The phase-07 transfer probe pointed the same way: GA local optima at $N = 160$ were 6–9% below the optimum, and 61–84% of that gap was closed by 2-exchanges on $X$.

An edge-preserving crossover (the ERX / GPX / EAX tradition for the TSP) fits the problem better. On the example it keeps all 6 common edges and must then choose between $\{23, 67\}$ and $\{26, 37\}$ for the four residual degree-1 vertices: the child is A or B, never a circle with edges from neither parent.

## Design

Genome: the connected multiplicity matrix $X$ ($m \times m$, int64, loops on the diagonal, degree $2 n_p$), validated by `exact.profiles.validate_multigraph` and turned into a circle by an Euler circuit only once, at the end. Relabelings of same-profile children are one genome by construction.

Three levels, by how much problem knowledge the offspring construction uses:

- **A — `fill="random"` (GA-E, pure GA).** Keep the edges common to both parents, $\min(x_a, x_b)$ including loops; add the parents' other edges ($\lvert x_a - x_b \rvert$ copies per type) in random type order as far as residual degrees allow; saturate random distinct-profile pairs with the remaining residual; leftover even residual becomes loops; `connect` merges components (minimal repair).
- **B — `fill="greedy"` (still a GA).** Same steps, but the parents' edges and the completion are taken by decreasing distance with random ties.
- **C — memetic (GA-M).** Level A or B plus the phase-07 bulk 2-exchange local search on offspring (Lamarckian: the improved child replaces it).

Why the degrees always work out (argued in `edge_crossover`'s docstring): the residual $r_p = 2 n_p - \deg(p)$ starts $\ge 0$ since $\min(x_a, x_b)$ is a sub-multigraph of a parent; each added copy lowers $\sum_p r_p$ by 2, so the sum stays even. The completion visits each pair of still-unsaturated profiles once and adds $\min(r_p, r_q)$, zeroing one; hence at most one profile keeps residual, equal to the even total: $r_p / 2$ loops close all degrees.

Mutation: one random degree-preserving 2-exchange of two edge copies drawn uniformly among the $N$ (the multigraph analogue of 2-opt; a swap is two such exchanges), retried up to 10 times if it is null or disconnects the graph.

The memetic part is a **separate object** (`MemeticLayer`), not a flag hidden in the crossover, because it changes the algorithm class (GA → memetic), the work unit and the cost per child by orders of magnitude (a local search from a random $N = 1000$ table takes ~0.5 s; a GA child ~0.2 ms). With the same GA, seed and settings it can be switched on or off (`local_search=None`), so its contribution is measured, not assumed.

## Implementation

- `classic/genetic.py`: the generation loop of `run_ga` was extracted verbatim into `evolve`, an engine with injected `score(key)`, `work()`, `key`, `crossover`, `mutation`, `finish`, `random_genome`, `diversity` and diagnostics hooks; selection runs on population indices, which consumes exactly the draws of selecting genomes. `run_ga` keeps its signature and behaviour (frozen hashes unchanged; same-seed results identical; A/B timing at $N \in \{160, 1000\}$ equal within noise: 0.085/0.568 s vs 0.085/0.581 s for 60 generations). Two opt-in engine options, off for `run_ga`: `duplicate_limit` (accept duplicates for the rest of a generation after that many failed random fallbacks — instances with fewer distinct keys than the population would otherwise loop forever) and `score_offspring` (score each accepted child at birth, so improved children still count if the deadline falls mid-generation; same cache and units).
- `layered/edge_genome.py`: `random_genome` (table of a uniformly random circle), `genome_key` (int32 bytes of the used upper-triangle types and counts, $O(\min(m^2, N))$), `key_fitness`, `genome_fitness`, `edge_diversity`, `edge_crossover(xa, xb, state, rng, fill=...)`, `edge_mutation`. Per child: $O(m^2)$ numpy plus Python steps over at most the differing/completing types. Operators use a numpy Generator seeded from one `rng.getrandbits(64)`.
- `layered/memetic.py`: `MemeticLayer(max_moves=None, probability=1.0)`; `.improve(x, distances, rng, limit=, deadline=)` runs `exact.multigraph.improve` one move at a time (identical to one call with `max_moves`), returns the checks used, and starts a step only if its exact cost $E^2$ ($E$ is the number of used edge types) fits `limit`.
- `layered/genetic.py`: `run_edge_ga(state, rng, *, population_size, generations, tournament_size, elites, crossover_rate, mutation_rate, evaluation_budget, fill="random", local_search=None, seeding=0.0, on_improvement=None, deadline=None, score_bound=None)` on `evolve` with `score_offspring=True`, `duplicate_limit=1`. Returns `circle` (`multigraph_circle`), `score`, `multigraph`, `evaluations`, `fitness_evaluations`, `memetic_checks`, `history` (diversity = mean edge copies not shared with the generation's best), `population`, `stop_reason`, `generations_completed`, `layers` = ("profile-state", "edge-crossover" [, "greedy-fill"] [, "memetic"] [, "seeded"]).

**Evaluation unit.** GA-E: distinct genome fitness computations (cache hits free), like the classic GA. GA-M: those **plus memetic checks** (one 2-exchange gain of a pair of edge types, `improve`'s `stats["checks"]`). The generation-level check reserves `population_size` units; each local-search step must fit in $\mathtt{budget} - \mathtt{population\_size} - \mathtt{used}$, so $\mathtt{evaluations} \le \mathtt{evaluation\_budget}$ always holds. A GA-M budget is therefore dominated by checks (~$10^4$–$10^7$ per generation) and is not comparable with GA-E counts: compare eras at equal time. Seed construction and the `connect` repair are not charged.

## Verification

`tests/test_layered_ga.py` (12 tests, ~2 s):

- Crossover validity, both fills, on 6 instances ($N = 6, \dots, 160$, $m = 1, \dots, 50$ incl. one profile, two profiles, $C = 8$): random, seeded and crossover-of-crossover parents; `validate_multigraph`, int64, Euler circle valid with `state.genome_score(circle) == multigraph_score == key_fitness`; parents not modified; identical parents reproduce themselves; greedy fill scores higher than random fill on average; invalid `fill` rejected.
- Mutation: valid, connected, parent untouched, exactly two copies exchanged ($\lvert \Delta X \rvert \in \{0, 2, 4\}$), and changes the genome in $> 2/3$ of calls for $m > 2$.
- `MemeticLayer` equals `improve(max_moves=...)` move for move, reports the same checks; $\mathtt{limit} \in \{0, E^2 - 1, E^2, 5E^2\}$ respected; probability 0, expired deadline, invalid arguments.
- GA-E vs GA-M on the same seed and settings (both fills): valid circles $\le$ the exact optimum, `evaluations == fitness_evaluations + memetic_checks`, checks 0 vs $> 0$, strictly increasing improvement trace ending at the score, layers; GA-M scores higher at 10 generations.
- Determinism per seed (three settings incl. memetic + seeding); generation limit; seeded first generation $\ge 200$ above random; hard budgets 20, 137, 1,000, 30,000 without and with (uncapped or 1-move) memetic layer → $\mathtt{evaluations} \le \mathtt{budget}$, `evaluation_budget` stop; patched expired deadline → exactly one evaluation, `time_limit`; real 0.05 s deadline honoured; `score_bound` stops (`bound_reached`), seeded GA-M reaches the $N = 160$ optimum 748; degenerate instances ($m = 1$, two profiles, a single valid table) terminate.

Full suite: `.venv/bin/python -m unittest discover -s tests -t .` — 91 tests OK (including the frozen GA hashes in `test_ga_optimization.py` and the era registry tests, which run GA-P through `eras.run_era`).

## Measurement (short probe)

Preliminary equal-time probe, single process, sequential (another agent's tests ran concurrently), Python 3.14.3, numpy 2.5.4, arm64 macOS. Script kept outside the repository. Instances `random_profiles(seed, N, 5)`: $N = 160$ for seeds 2027/2028/2029 (optima 748/752/714) and $N = 1000$ for seed 2027 (optimum 4,884; all optima from `solve_circle`). Each run: `deadline = perf_counter() + 0.5`, `evaluation_budget=10**12`, `generations=10**9`, population 50, tournament 10, elites 2, crossover 0.9, mutation 0.25, rng seeds 1/2/3 (9 runs at $N = 160$, 3 at $N = 1000$). Classic GA gets `distance_matrix(profiles)` built outside timing (0.56 s at $N = 1000$; the profile state takes 0.7 ms). `mm` = `max_moves`, `p` = memetic `probability`; gens = completed generations. Longest run 0.507 s.

| variant | $N=160$ mean | gap | optimal | gens | $N=1000$ mean | gap | optimal | gens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ga (classic OX) | 674.2 | 8.62% | 0/9 | 364 | 2962.0 | 39.35% | 0/3 | 54 |
| ga-p | 668.4 | 9.39% | 0/9 | 300 | 2978.7 | 39.01% | 0/3 | 57 |
| ga-e random (A) | 644.0 | 12.68% | 0/9 | 82 | 3028.7 | 37.99% | 0/3 | 79 |
| ga-e greedy (B) | 696.4 | 5.60% | 0/9 | 65 | 3736.7 | 23.49% | 0/3 | 74 |
| ga-m random mm=3 | 640.9 | 13.11% | 0/9 | 7 | 2680.0 | 45.13% | 0/3 | 0 |
| ga-m greedy mm=1 | 734.7 | 0.45% | 4/9 | 21 | 3214.0 | 34.19% | 0/3 | 1 |
| ga-m greedy mm=3 | 737.3 | 0.09% | 7/9 | 15 | 2964.0 | 39.31% | 0/3 | 0 |
| ga-m greedy mm=10 | 731.6 | 0.86% | 3/9 | 3 | 3075.3 | 37.03% | 0/3 | 0 |
| ga-m greedy mm=None | 732.9 | 0.69% | 0/9 | 0 | 4632.0 | 5.16% | 0/3 | 0 |
| ga-m greedy mm=3 p=0.5 | 738.0 | 0.00% | 9/9 | 28 | 3266.0 | 33.13% | 0/3 | 1 |
| ga-m greedy mm=10 p=0.2 | 738.0 | 0.00% | 9/9 | 37 | 3177.3 | 34.94% | 0/3 | 0 |
| ga-m greedy mm=None p=0.5 | 735.8 | 0.30% | 6/9 | 7 | 4758.7 | 2.57% | 0/3 | 0 |
| **ga-m greedy mm=None p=0.2** | 738.0 | 0.00% | 9/9 | 36 | 4809.3 | 1.53% | 0/3 | 0 |
| ga-p seed=0.1 | 732.4 | 0.75% | 1/9 | 295 | 4846.0 | 0.78% | 0/3 | 55 |
| ga-e greedy seed=0.04 | 736.2 | 0.24% | 5/9 | 60 | 4869.3 | 0.30% | 1/3 | 77 |
| **ga-e greedy seed=0.1** | 738.0 | 0.00% | 9/9 | 63 | 4884.0 | 0.00% | 3/3 | 83 |
| ga-e greedy seed=0.5 | 738.0 | 0.00% | 9/9 | 62 | 4884.0 | 0.00% | 3/3 | 88 |
| ga-e random seed=0.1 | 736.9 | 0.15% | 6/9 | 77 | 4879.3 | 0.10% | 1/3 | 75 |
| ga-m greedy mm=3 seed=0.1 | 738.0 | 0.00% | 9/9 | 31 | 4884.0 | 0.00% | 3/3 | 29 |
| ga-m greedy mm=None seed=0.1 | 738.0 | 0.00% | 9/9 | 25 | 4882.7 | 0.03% | 2/3 | 0 |
| ga-m random mm=None seed=0.1 | 738.0 | 0.00% | 9/9 | 12 | 4878.7 | 0.11% | 1/3 | 0 |
| **ga-m greedy mm=None p=0.2 seed=0.1** | 738.0 | 0.00% | 9/9 | 54 | 4884.0 | 0.00% | 3/3 | 29 |

Because the seeded variants saturate at $C = 5$, they were also run once on a harder instance, `random_profiles(2027, 1000, 8)` ($m = 249$, optimum 7,700), same protocol, 3 seeds:

| variant | mean | gap | gens |
| --- | ---: | ---: | ---: |
| ga-p seed=0.1 | 7545.3 | 2.01% | 28 |
| ga-e greedy seed=0.1 | 7678.0 | 0.29% | 4 |
| ga-m greedy mm=3 seed=0.1 | 7640.0 | 0.78% | 0 |
| ga-m greedy mm=None p=0.2 seed=0.1 | 7646.7 | 0.69% | 0 |

Reading (probe-level evidence only):

- Edge inheritance alone is not enough: level A (random fill) is no better than OX. Level B (greedy fill) is the first real gain of the representation (gap 8.6% → 5.6% at $N = 160$, 39% → 23% at $N = 1000$) and stays a pure GA.
- The memetic layer dominates without seeding, but its setting is size-dependent: shallow capped moves win at $N = 160$, deep searches at $N = 1000$ (a single move there costs ~$2 \cdot 10^5$ checks and children start far from local optima). Applying a full local search to 20% of offspring (`max_moves=None, probability=0.2`) is the best compromise measured at both sizes. Random fill + memetic is poor: children far from both parents make the local search do the whole job.
- With 10% greedy seeds (S6) every edge variant reaches the optimum on all 12 $C = 5$ runs; at $C = 8$ GA-E-seed is best within 0.5 s, while GA-M-seed completes no generation (one local-search move takes ~20 ms at $m = 249$).

Recommended era parameters (`eras.py` style, `_GA = dict(population_size=50, tournament_size=10, elites=2, crossover_rate=0.9, mutation_rate=0.25)`):

| era | defaults |
| --- | --- |
| ga-p | `dict(_GA)` (seeding 0.0) |
| ga-e | `dict(_GA, fill="greedy")`; level A is the override `fill="random"` |
| ga-m | `dict(_GA, fill="greedy", memetic=dict(max_moves=None, probability=0.2))` |
| ga-e-seed | `dict(_GA, fill="greedy", seeding=0.1)` |
| ga-m-seed | `dict(_GA, fill="greedy", seeding=0.1, memetic=dict(max_moves=None, probability=0.2))` |

## Limitations

- The equal-time probe is small (4 instances at $C = 5$ plus one at $C = 8$, 3 seeds, 0.5 s) and was run while another process used the machine; it is a preliminary probe, not a suite measurement. The full grid is a follow-up.
- A local-search move cannot be interrupted: the deadline is checked between moves, so GA-M overshoots by up to one move (runs ended at $\le 0.521$ s at $C = 8$, $m = 249$, where a move costs ~20 ms).
- `edge_crossover` loops in Python over the differing and completing types; when $m$ approaches $N$ (large $C$, small $N$) that is $O(N)$ per child, like OX.
- The completion and `connect` ignore the parents once residuals are filled; EAX-style AB-cycle recombination would inherit more structure but is far more complex on multigraphs.
- GA-M's local search starts from offspring only; the initial population is not improved (seeding is the cheap way to start from good tables).
- Memetic checks and fitness evaluations share one budget; their CPU costs differ (a check is a few ns of numpy work), so count budgets mislead across eras.

## GA-P (phase 09, layer S2)

`layered/genetic.run_profile_ga(state, rng, *, population_size, generations, tournament_size, elites, crossover_rate, mutation_rate, evaluation_budget, seeding=0.0, on_improvement=None, deadline=None, score_bound=None)`.

The classic GA unchanged in nature — child genomes, OX, swap/inversion mutation, tournament, elitism, duplicate rejection — run by the shared engine `classic.genetic.evolve`, with two differences:

- Fitness is `circle_fitness` of the profile sequence on the $m \times m$ table `state.table`; no $N \times N$ matrix is built (memory $O(N \cdot \text{population} + m^2)$).
- The fitness cache and duplicate rejection are keyed by the profile sequence of the canonical genome, so relabelings of same-profile children that keep child 0's position collapse into one key. `evaluations` counts distinct profile sequences. Rotations that move child 0 to another visit of its profile, and mirror images, are not identified (cheap to add; frequency not measured). On instances with fewer distinct sequences than the population, duplicates are accepted after one failed random fallback (`duplicate_limit=1`).

Returns the `run_ga` keys plus `sequence` (profile ids of the best circle) and `layers = ("profile-state", "ox")` (+ `"seeded"`).

Verification (`tests/test_layered_ga.py`, `ProfileGATests`): with all 32 profiles of $C = 5$ distinct, GA-P reproduces `run_ga` exactly (circle, score, evaluations, history, final population, stop reason, generations, improvement trace) for two seeds and budgets — the key is then a bijection of the genome. On `random_profiles(2027, 40, 5)` (optimum 176, checked against `solve_circle`) 3 seeds $\times$ seeding $\in \{0, 0.2\}$: valid circles with `state.genome_score(circle) == score` and $\text{score} \le 176$, `sequence` consistent, budget respected, final population has distinct profile sequences. Degenerate instances (one profile; $3 + 3$ children) terminate. Patched expired deadline → one evaluation; real 0.05 s deadline; `score_bound` stop.

Measurement (same probe as above, 0.5 s): GA-P vs classic GA, gap 9.39% vs 8.62% at $N = 160$ (300 vs 364 generations, presumably the extra $O(N)$ key tuple per child) and 39.01% vs 39.35% at $N = 1000$ (57 vs 54 generations), where the classic run additionally needed a 0.56 s matrix build outside timing against 0.7 ms for the profile state. S2 is a representation and memory change; it does not improve the search by itself.

## GA seeding (phase 14, layer S6)

Both layered GA engines take $\mathtt{seeding} \in [0, 1]$: $\operatorname{round}(\mathtt{seeding} \cdot \mathtt{population\_size})$ initial individuals come from the S6 constructors in `layered/restarts.py` (randomized greedy b-matching + `connect`, noise 1.5, i.e. one GRASP construction each), the rest are uniformly random. GA-P uses `seed_sequence` realized as a child genome by `state.circle`; the edge GA uses `seed_multigraph` directly. Seeds must be distinct (at most 10 attempts per requested seed; small instances may have fewer distinct greedy tables), seed construction is not charged to `evaluation_budget` (equal-time comparisons charge it as wall time), and `layers` gains `"seeded"`. `seeding=0` draws no extra random numbers, so GA-P still reproduces `run_ga`.

Verification (`tests/test_layered_ga.py`): seeded first generations beat random ones by more than 200 at $N = 160$ in both engines; seeded runs are deterministic per seed; layers report `"seeded"`; tiny instances with seeding terminate; seeded GA-M reaches the $N = 160$ optimum 748 and stops with `bound_reached`; `seeding` outside $[0, 1]$ is rejected.

Measurement (same probe, 0.5 s, $C = 5$): GA-P seed 0.1 0.75% / 0.78% gap at $N = 160$ / $N = 1000$ (vs 9.39% / 39.01% unseeded); GA-E greedy seed 0.04 / 0.1 / 0.5: 0.24% / 0.00% / 0.00% at $N = 160$ and 0.30% / 0.00% / 0.00% at $N = 1000$ (12/12 optimal from 0.1 on); GA-M seed 0.1 (`max_moves=None, probability=0.2`) 12/12 optimal. At $C = 8$, $N = 1000$ (optimum 7,700): GA-P seed 2.01%, GA-E seed 0.29%, GA-M seed 0.69%. A fraction of 0.1 (5 of 50) is recommended: 2 seeds are too few for consistent optima, 25 seeds add nothing measurable. Caveat: a single greedy seed (noise 1.5) is already 2.6–3.0% below the optimum on average at $C = 5$ (best of 20: 0.0–0.5%) and 2.2% at $C = 8$, $N = 1000$ (best of 20: 1.6%), so these instances mainly measure how well each GA *recombines* good tables; the seeded GAs do close that remaining gap.
