# Phase 15 — publication layout: walkthrough on top, full project below

Started/completed: 2026-10-22. Status: **completed and verified**.

## Goal

The lab's assignment is an HC and a GA for the children's circle. The published repository should let a reviewer understand the work in minutes and deep-dive only if they want to:

- top level: a short `README.md` (presentation, results, how to run, use of AI), `HISTORY.md` (the improvement history line with the concepts behind each step) and `walkthrough.ipynb` (the lab's HC/GA, the major versions and the enhancements, executed);
- `project/`: everything else, unchanged inside — package, tests, docs and history records, benchmark entry points, the course and GA notebooks, the full guided tour, generated results.

## Rules

- No algorithm or API change. Every era stays registered and runnable; the walkthrough only selects the first versions, the major ones and the enhancements.
- Everything inside `project/` keeps its relative paths, so its links, notebooks (`cwd = project/`) and commands keep working when run from `project/`. Code that depended on the working directory is made location-independent where cheap (tests).
- The top-level notebook adds `project/` to `sys.path`; nothing is installed.

## Acceptance

All tests pass from `project/` and from the repository root; the CLI, the suite smoke run, the comparison script, the exact-solver CLI and every notebook execute; no broken relative link in any Markdown file.

## What was done

- Moved, unchanged: `circle_optimization/`, `tests/`, `docs/`, `benchmark-results/`, both entry scripts, `requirements-benchmark.txt`, `problem.ipynb`, `genetic_algorithm.ipynb`, `guided_tour.ipynb` → `project/`. The former root README became `project/README.md` (deep-dive overview; only framing, structure tree and run commands changed).
- New at the root: `README.md` (presentation, results, how to run, use of AI), `HISTORY.md` (history line: problem, concept, effect per step) and `walkthrough.ipynb` (lab HC/GA → `hc-i` → exact optimum → S2, S1, S3, S5, S6 → edge GA ± memetic, seeded → ranking → N = 10,000; executed).
- `tests/test_eras.py` locates `docs/history` from its own file, so the suite passes from any working directory; dependency hints name `project/requirements-benchmark.txt`.

## Verification

- 92 tests pass from `project/` and from the repository root (`python -m unittest discover -s project/tests -t project`).
- From `project/`: `python -m circle_optimization list | run | compare`, `python -m circle_optimization.exact.milp`, the suite smoke run with eras, the single-size benchmark, `compare_benchmark_runs.py` and the report rebuild all run.
- `walkthrough.ipynb` (root), `problem.ipynb`, `genetic_algorithm.ipynb` and `guided_tour.ipynb` (`project/`) execute without errors.
- No broken relative link in any Markdown file or notebook.
