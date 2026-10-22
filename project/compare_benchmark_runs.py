"""Historical entry point for run comparison; see circle_optimization/benchmark/compare.py.

    python compare_benchmark_runs.py benchmark-results/full-phase06 benchmark-results/full-phase07
"""

from circle_optimization.benchmark.compare import main

if __name__ == "__main__":
    main()
