"""Layered eras (phases 09-14): profile-aware HC and GA built from optional layers.

Each layer is one documented module that can be enabled or disabled:

- S2 `state` (phase 09): profile-id sequences and the m x m table.
- S1 `splice` (phase 10): split-and-splice move between same-profile children.
- S3 `reduced_costs` (phase 11): LP reduced-cost guidance and candidate lists.
- S4 `edge_genome`, `memetic` (phase 12): edge-preserving crossover on profile
  multigraphs and an optional memetic local-search layer.
- S5 `propagation` (phase 13): retry an accepted move's pattern elsewhere.
- S6 `restarts` (phase 14): greedy seeding and double-bridge perturbation.

Engines combining them: `hill_climbing` (HC-P eras) and `genetic` (GA-P/E/M
eras). Requires numpy; S3 also requires highspy. See docs/ERAS.md.
"""
