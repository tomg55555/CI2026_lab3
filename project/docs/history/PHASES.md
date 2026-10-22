# Phase index

| Phase | Status | Scope |
| --- | --- | --- |
| 00 — initial formulation | Historical | Profiles, symmetric matrix, circular score, exhaustive small-instance reference |
| 01 — heuristic library | Historical | Shared HC/GA, operators, canonical genomes, caching |
| 02 — delta and guidance | Historical | $O(1)$ move deltas, weak-edge-guided swaps, GA mutation injection |
| 03 — experiment suite | Historical | Nested size/exploration sweeps, timing, diagnostics, interactive reports |
| [04 — throughput and search](phases/04-throughput-and-search.md) | Completed, 2026-10-21 | Parallel batches, progress/checkpoints, HC productive depth, inversion focus, GA overhead |
| [05 — full comparison](phases/05-full-comparison.md) | Completed, 2026-10-21 | Clean generated validation runs; execute and compare the enhanced full grid |
| [06 — faster suite](phases/06-faster-suite.md) | Completed, 2026-10-21 | Proven-bound stop, faster diagnostics, draw-identical hot paths, one checks repeat, copied exploration $= 1$ |
| [07 — exact profile MILP](phases/07-profile-milp.md) | Completed, 2026-10-21 | Profile aggregation, multi-start heuristic, HiGHS MILP with lazy cuts, exact suite reference for every $N$ |
| [08 — era architecture](phases/08-eras-architecture.md) | Completed, 2026-10-21/22 | Package by era (classic/exact/layered/benchmark), era registry, CLI, guided tour, tests/ and docs/ moves |
| [09 — profile state (S2)](phases/09-profile-state.md) | Completed, 2026-10-21/22 | Profile-id sequences and $m \times m$ table for HC (exact classic trajectory) and GA-P |
| [10 — splice move (S1)](phases/10-splice.md) | Completed, 2026-10-21/22 | Split-and-splice reconnection through same-profile children |
| [11 — reduced-cost guidance (S3)](phases/11-reduced-cost-guidance.md) | Completed, 2026-10-21/22 | LP reduced costs pick edges to break and candidate partners |
| [12 — edge GA and memetic layer (S4)](phases/12-edge-ga-memetic.md) | Completed, 2026-10-21/22 | Multigraph genome, edge-preserving crossover, optional memetic local search |
| [13 — move propagation (S5)](phases/13-propagation.md) | Completed, 2026-10-21/22 | Retry an accepted move's profile pattern elsewhere |
| [14 — seeding and ILS (S6)](phases/14-seeding-and-ils.md) | Completed, 2026-10-21/22 | Greedy GRASP starts, double-bridge perturbation, seeded GA populations |
| [15 — publication layout](phases/15-publication-layout.md) | Completed, 2026-10-22 | README, HISTORY and walkthrough at the root; the full project in `project/` |

Phases 00–03 summarize work predating this documentation system. Era names in `circle_optimization/eras.py` (`python -m circle_optimization list`) carry the phase that introduced them; see [docs/ERAS.md](../ERAS.md). Their details are in `circle_optimization/README.md`, `docs/GUIDED_SEARCH.md`, and `docs/BENCHMARK_SUITE.md`; they were not retrospectively assigned fabricated implementation dates.
