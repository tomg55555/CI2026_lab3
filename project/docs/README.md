# `docs/` — guides and engineering history

| Document | Contents |
| --- | --- |
| [ERAS.md](ERAS.md) | Catalog of every algorithm era: lineage, what each adds, layer switches, budget units and fairness, results snapshot, how to add an era |
| [BENCHMARK_SUITE.md](BENCHMARK_SUITE.md) | Experimental protocol: instances, references, budget protocols, metrics, statistics, every suite option, outputs |
| [GUIDED_SEARCH.md](GUIDED_SEARCH.md) | Phase-02 weak-edge-guided swaps: probability rule, rationale, cost, GA integration |
| [history/README.md](history/README.md) | How the engineering record is kept (rules below) |
| [history/PHASES.md](history/PHASES.md) | Index of phases 00–14 with status and scope |
| [history/phases/](history/phases) | One record per phase from 04 on: motivation, design, implementation, verification, measurements, limitations |
| [history/STATE_OF_ART.md](history/STATE_OF_ART.md) | Current best evidence and how each era performs |
| [history/DECISIONS.md](history/DECISIONS.md) | Consequential choices and why they were made |
| [history/GUIDANCE_AND_RESTART_PLAN.md](history/GUIDANCE_AND_RESTART_PLAN.md) | The phase-04 follow-up plan, annotated with what later phases did |

## Reading paths

- **Reviewer, first look**: repository [README](../../README.md) → [HISTORY.md](../../HISTORY.md) → [walkthrough.ipynb](../../walkthrough.ipynb).
- **Deep dive**: [project README](../README.md) → [ERAS.md](ERAS.md) → [`guided_tour.ipynb`](../guided_tour.ipynb).
- **Why the exact solver works**: [phase 07](history/phases/07-profile-milp.md) → [exact/README](../circle_optimization/exact/README.md).
- **What each enhancement layer does and gained**: phases [09](history/phases/09-profile-state.md)–[14](history/phases/14-seeding-and-ils.md) → [layered/README](../circle_optimization/layered/README.md).
- **Running experiments**: [BENCHMARK_SUITE.md](BENCHMARK_SUITE.md) → [benchmark/README](../circle_optimization/benchmark/README.md).

## Record-keeping rules (from history/README.md)

- Write a phase record before changing the implementation.
- Keep completed records historical; append corrections instead of rewriting measurements.
- Record exact commands, seeds, settings and execution isolation, and whether numbers are measured, preliminary or expected.
- Separate behaviour-changing improvements from semantics-preserving speedups.
- A phase is complete only after its checks pass.

Phase records 00–07 cite the module paths of their time; phase 08 contains the old → new path table.
