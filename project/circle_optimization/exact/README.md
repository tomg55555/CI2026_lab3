# `exact/` — profile aggregation and the exact MILP (phase 07)

Solves the problem **exactly for any $N$** by working on the $m$ distinct profiles ($m \le \min(N, 2^C)$) instead of the $N$ children. Needs numpy; the MILP needs highspy (HiGHS).

## Why aggregation is exact

Children with the same profile are interchangeable, so a circle's score depends only on the **edge-count table** $X$ ($m \times m$, symmetric): $X_{pq}$ edges between profiles $p \ne q$, $X_{pp}$ edges joining two children of $p$ (distance 0). A table is a circle iff

- every profile $p$ has degree $2 n_p$ (each child has two neighbours), and
- its edges connect all profiles (otherwise it describes several circles).

Every such table has an Euler circuit that visits $p$ exactly $n_p$ times; giving $p$'s children to those visits yields a circle with the same score. Conversely every circle yields such a table. The model therefore has $m(m-1)/2 + m$ integer variables, **independent of $N$**. With $x_{pq} = X_{pq}$ for $p < q$ and $y_p = X_{pp}$:

$$
\begin{aligned}
\max \quad & \sum_{p<q} d_{pq}\, x_{pq} \\
\text{s.t.} \quad & \sum_{q} x_{pq} + 2 y_p = 2 n_p && \text{for every profile } p \\
& \sum_{p \in S,\, q \notin S} x_{pq} \ge 2 && \text{for every proper subset } S \text{ (added lazily)} \\
& x_{pq} \in [0, 2 \min(n_p, n_q)], \quad y_p \in [0, n_p - 1], && \text{integer}
\end{aligned}
$$

Every circle score is even (each feature flips an even number of times), so all bounds are rounded down to even.

## Modules

| Module | Contents |
| --- | --- |
| `profiles.py` | `aggregate_profiles` (bit packing, histogram or `np.unique`, one $m \times m$ matrix product for distances), `ProfileInstance` (`rows`, `counts`, `child_profile`, `distances`), `profile_upper_bound`, `multigraph_score`, `components`, `validate_multigraph`, `euler_sequence` ($O(m^2 + N)$), `assign_children`, `sequence_score`, `multigraph_circle` |
| `multigraph.py` | Multi-start heuristic on edge tables: `greedy_multigraph` (randomized greedy b-matching), `connect` (merge components by cheapest 2-exchange), `improve` (best-improvement **bulk** 2-exchange: moves as many edge copies as possible, so the number of moves does not grow with $N$), `multi_start` |
| `milp.py` | `solve_circle` (entry point), `solve_profile_milp`, `lp_relaxation` (bound and reduced costs, used by layer S3), CLI `main` |

## Solver pipeline (`solve_circle`)

1. **Aggregate** the $N \times C$ Boolean table ($O(N C)$, numpy).
2. **Bound** with the Hamming relaxations ($O(m^2 + mC)$).
3. **Heuristic**: 4 randomized greedy starts + merge + bulk 2-exchange. If one meets the bound, the answer is proven optimal without any MILP.
4. **LP phase**: solve the continuous relaxation, add a cut for every component of its fractional support, re-solve from the basis until connected. Its even floor bounds the optimum (tight on every probe).
5. **Target phase**: reduced-cost fixing toward that bound leaves only a few percent of the columns ($\approx 6\%$ at $C = 8$). If the reduced model cannot reach the target, the bound drops and the full column bounds return.
6. **MIP rounds**: integer optima are often disconnected ties; each is cut off and also repaired (merge + 2-exchange) into an incumbent, until an incumbent meets the bound.
7. **Rebuild** a child circle from the table by an Euler circuit.

The result reports `status="optimal"` only with a proof (`proof` = "bound", "lp" or "milp"); with `time_limit` it returns the best circle, its proven `upper_bound` and `gap`. Also: `profiles` ($m$), `heuristic_score`, `lp_bound`, round/cut/node counts, the multiplicity table and per-stage `seconds`. `on_improvement(score, step)` reports incumbents as they appear.

## Usage

```python
from circle_optimization import random_profiles, solve_circle

result = solve_circle(random_profiles(2027, 10_000, 8))
print(result["score"], result["status"], result["proof"], result["seconds"]["total"])
```

```sh
.venv/bin/python -m circle_optimization.exact.milp --children 1000000 --characteristics 8
.venv/bin/python -m circle_optimization.exact.milp --children 3000 --characteristics 10 --time-limit 5
```

Eras: `multistart` (heuristic only) and `exact` (full solver).

## Performance (M1 Max, defaults, random uniform profiles)

| $N$ | $C$ | $m$ | proof | seconds |
| ---: | ---: | ---: | --- | ---: |
| 160 | 5 | 32 | bound | 0.002 |
| 100,000 | 6 | 64 | MILP | 0.04 |
| 1,000,000 | 6 | 64 | MILP | 0.21 |
| 1,000,000 | 8 | 256 | MILP | 0.67 |
| 500 | 9 | 323 | MILP | 2.6 |
| 3,000 | 10 | 977 | MILP | 27 |

Cost grows with $m$ ($\approx 2^C$), not $N$; $C \ge 10$ is where it becomes slow.

## Verification

Matches exhaustive enumeration on hundreds of random instances ($N \le 9$), including duplicate-heavy cases, instances that need connectivity cuts and an LP integrality gap confirmed by an independent plain cutting-plane loop; any time limit returns a valid bounded circle (`tests/test_profile_solver.py`). Record: [docs/history/phases/07-profile-milp.md](../../docs/history/phases/07-profile-milp.md).
