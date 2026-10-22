# Implementation history

This directory is the durable engineering record for the circle optimizers. Benchmark data are generated artifacts; the reasoning, implementation phases, and verification evidence are kept here as Markdown.

## Reading order

1. [Current state](STATE_OF_ART.md): current algorithms, evidence, limitations.
2. [Phase index](PHASES.md): status and links to implementation records.
3. [Decision log](DECISIONS.md): consequential choices and their rationale.
4. Individual phase records: scope, actual changes, tests, measurements, follow-ups.

## Update rules

- Create a phase record **before** changing the corresponding implementation.
- Keep completed records historical; append corrections rather than silently replacing old measurements or interpreting historical settings as current.
- Update the state-of-the-art summary and phase index after verification.
- Record exact benchmark commands, seed/settings, execution isolation, and whether numbers are measured results, preliminary probes, or untested expectations.
- Separate behavior-changing search improvements from semantics-preserving speedups.
- A phase is complete only after required checks, not merely after its code exists.

No history entry implies a git commit; committing remains an explicit user action.

Phase records 00–07 cite the flat module layout of their time (for example `circle_optimization/ga.py`, `test_*.py` at the root). Phase 08 moved every module; its record contains the [old → new path table](phases/08-eras-architecture.md#old--new-paths).
