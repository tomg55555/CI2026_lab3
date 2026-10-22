"""Exact era (phase 07): solve the problem on the m distinct profiles.

- `profiles`: aggregation, Hamming bound, multigraph checks, Euler reconstruction.
- `multigraph`: multi-start heuristic (greedy b-matching, merge, bulk 2-exchange).
- `milp`: HiGHS MILP with lazy connectivity cuts; `solve_circle` entry point.

Requires numpy (all) and highspy (`milp`). See docs/history/phases/07-profile-milp.md.
"""
