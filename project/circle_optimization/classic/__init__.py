"""Classic eras (phases 01-06): HC and GA over index genomes and the N x N matrix.

- `operators`: sampling, swap/inversion/OX operators, O(1) move deltas.
- `guidance`: weak-edge-guided swap proposals (phase 02).
- `hill_climbing`: uniform, legacy-restart, inversion-focused and guided HC.
- `genetic`: cached-fitness generational GA.
- `diagnostics`: expensive coverage/diversity replays for the benchmark suite.

These modules need only the standard library. Memory is O(N^2) for the matrix.
"""
