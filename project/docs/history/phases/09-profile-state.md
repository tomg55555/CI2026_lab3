# Phase 09 — layer S2: profile-id state and the $m \times m$ table

Started/completed: 2026-10-21. Status: **completed and verified** (HC and GA-P).

## Motivation

Classic HC and GA look distances up in the $N \times N$ child matrix. It costs $O(N^2)$ memory and $O(N^2C)$ construction: 25,600 cells at $N=160$, but $10^{10}$ at $N=10^5$, so the classic eras cannot run at the large $N$ this lab targets. Phase 07 showed that children with equal profiles are interchangeable: every distance is `D[profile(a)][profile(b)]` with an $m \times m$ table, $m \le \min(N, 2^C)$.

## Design

`circle_optimization/layered/state.py` — `ProfileState`:

- wraps the phase-07 `ProfileInstance` (numpy) and adds `table` ($m \times m$ tuple of Python ints) and `profile_of` (child → profile id), because numpy scalar indexing inside a Python search loop is several times slower than tuples;
- `sequence(genome)`, `random_sequence(rng)` (profiles of the classic `random_individual`, same random draws), `score(sequence)` (the classic `circle_fitness` applied to the table), `genome_score(genome)`;
- `circle(sequence)`: canonical child circle realizing a sequence (the $k$-th visit of $p$ maps to $p$'s $k$-th child, child 0 first), with a profile-count check;
- `multigraph(sequence)`: the phase-07 edge-count matrix, used by S4.

`circle_optimization/layered/hill_climbing.py` — `run_layered_hill_climber` copies the classic engine loop but keeps the circle as a list of profile ids. A swap or reversal at the same positions has the same delta, so acceptance, restarts and counts are unchanged. Only the returned child labels may differ (children are assigned at the end), never the score.

Memory per run: $O(N)$ for the sequence plus the shared $m \times m$ table. The S2 era `hc-p` never builds the $N \times N$ matrix (`eras.Problem.state`).

## Verification

`tests/test_layered_state.py`:

- sequence/score/circle/multigraph agree with the $N \times N$ matrix on random genomes; an invalid profile sequence is rejected;
- **trajectory identity**: for $N \in \{5, 10, 23, 40, 160\}$ and productive/legacy restarts and 50%/80% inversion mixes, `run_layered_hill_climber` equals the classic `run_hill_climber` / `run_inversion_hill_climber` in score, evaluations, restarts, stop reason, restart history and the full strict-improvement trace (15 configurations).

Suite smoke (`--quick --eras hc-p`): every checks-protocol `hc-p` row equals the classic `HC` row of the same trial (identical seeds).

Equal-time probe ($N=160$, $C=5$, seed 2027, 0.2 s, M1 Max): `hc` 179,146 checks, `hc-p` 181,381 checks, same best score (664). The smaller table is slightly faster; the gain is cache locality, not a different search.

## GA-P (layer S2 for the GA)

`layered/genetic.run_profile_ga(state, rng, *, population_size, generations, tournament_size, elites, crossover_rate, mutation_rate, evaluation_budget, seeding=0.0, on_improvement=None, deadline=None, score_bound=None)`.

The classic GA unchanged in nature — child genomes, OX, swap/inversion mutation, tournament, elitism, duplicate rejection — run by the shared engine `classic.genetic.evolve` (extracted from `run_ga` in phase 12; `run_ga`'s frozen hashes still match), with two differences:

- Fitness is `circle_fitness` of the profile sequence on the $m \times m$ table; no $N \times N$ matrix is built (memory $O(N \cdot \text{population} + m^2)$).
- The fitness cache and duplicate rejection are keyed by the profile sequence of the canonical genome, so relabelings of same-profile children that keep child 0's position collapse into one key; `evaluations` counts distinct profile sequences. Rotations that move child 0 to another visit of its profile, and mirror images, are not identified (cheap to add; frequency not measured). With fewer distinct sequences than the population, duplicates are accepted after one failed random fallback (`duplicate_limit=1`).

Verification (`tests/test_layered_ga.py`, `ProfileGATests`): with all 32 profiles of $C=5$ distinct, GA-P reproduces `run_ga` exactly (circle, score, evaluations, history, final population, stop reason, generations, trace) for two seeds and budgets — the key is then a bijection of the genome. On `random_profiles(2027, 40, 5)` (optimum 176): valid circles with `state.genome_score(circle) == score` and $\text{score} \le 176$, budget respected, distinct final profile sequences; degenerate instances terminate; deadline and bound stops.

Measurement (0.5 s per run, population 50, 3 seeds; $C=5$): GA-P vs classic GA gap 9.39% vs 8.62% at $N=160$ (300 vs 364 generations, presumably the extra $O(N)$ key tuple per child) and 39.01% vs 39.35% at $N=1000$ (57 vs 54 generations), where the classic run also needed a 0.56 s matrix build outside timing against 0.7 ms for the profile state.

## Limitations

S2 alone does not change search quality: it is the foundation for S1, S3, S5 and S6 (phases 10–14), which need profile identities, and it removes the $O(N^2)$ barrier for large $N$.
