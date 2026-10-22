"""Historical entry point for the benchmark CLI; see circle_optimization/benchmark/cli.py.

Every command recorded in docs/history keeps working, e.g.
    python benchmark_guided_search.py --suite --sizes 10 20 40 --output benchmark-results/run
"""

from circle_optimization.benchmark.cli import main

if __name__ == "__main__":
    main()
