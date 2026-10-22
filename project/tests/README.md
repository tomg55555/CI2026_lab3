# `tests/` — unit, regression and integration tests

```sh
.venv/bin/python -m unittest discover -s tests -t .          # all 92 tests (~16 s)
.venv/bin/python -m unittest tests.test_layered_hc -v        # one file
```

Run from the repository root. Tests that need numpy, highspy or plotly skip themselves when the dependency is missing, so the classic tests also run on a bare standard-library Python.

| File | Tests | What it guarantees |
| --- | ---: | --- |
| `test_circle_optimization.py` | 6 | Core model: exact search, canonical circles, operators keep permutations, $O(1)$ deltas equal full rescoring, HC budget accounting, reproducible HC/GA, invalid inputs rejected |
| `test_ga_optimization.py` | 3 | **Frozen GA trajectories**: hashes of pre-optimization runs, an independent OX oracle, identical sampling draws |
| `test_guided_search.py` | 7 | Weak-edge selector targets and breaks weak edges (wraparound included), refreshes weights; guided HC reproducible; exploration $= 1$ equals baseline HC; guided GA mutation |
| `test_search_progress.py` | 4 | First-hit reporting for HC and GA (ties do not reset it), observing improvements never changes results, single-size benchmark CLI |
| `test_execution_and_restart.py` | 10 | Spawned and serial batches give identical results; isolated mode starts no pool; checkpoints/resume and mismatch detection; protected outputs; bootstrap cache; productive vs legacy restarts; HC-I keeps swaps |
| `test_benchmark_suite.py` | 12 | Bounds, AUC, paired statistics, deadlines, suite pairing, aliasing, bound stops, HTML export |
| `test_benchmark_comparison.py` | 2 | Cross-run comparison rebases targets instead of crediting bound changes |
| `test_profile_solver.py` | 13 | Exact solver = enumeration on hundreds of instances; connectivity cuts; an LP integrality gap; large $N$; any time limit returns a valid bounded circle; suite references |
| `test_layered_state.py` | 3 | **S2 identity**: profile-state HC reproduces classic HC exactly; S6 constructors keep profile counts |
| `test_layered_hc.py` | 14 | S1 splice vs brute force, S3 guide and certificate, S5 propagation, S6 starts/ILS; layer on/off comparisons; never above the optimum |
| `test_layered_ga.py` | 12 | Edge crossover/mutation validity, memetic layer accounting, GA-P = classic GA when profiles are distinct, GA-E vs GA-M, seeding, budgets and deadlines |
| `test_eras.py` | 6 | Registry consistency (phases, records, lineage); **every era runs and is rescored independently**; CLI; suite with eras and era-only HTML report |

Regression anchors: `test_ga_optimization` (classic GA must not drift), `test_layered_state` (S2 must equal classic HC), `test_profile_solver` (exactness against enumeration). A new era is covered automatically by `test_eras.py` once registered; add a layer test that its default leaves the parent era unchanged.
