"""Benchmark command line: the suite (`--suite`) or the single-size comparison.

Run: python benchmark_guided_search.py --instances 3 --trials 10 --repeats 3
(equivalently `python -m circle_optimization.benchmark.cli ...`).
Suite options (`--suite`, `--sizes`, ...) run the size/exploration grid; see
docs/BENCHMARK_SUITE.md. Without them, the original single-size uniform versus
guided comparison runs; see docs/GUIDED_SEARCH.md.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from math import factorial
from random import Random
from statistics import mean, median, stdev
from time import perf_counter

from ..classic.genetic import run_ga
from ..classic.guidance import make_guided_mutation
from ..classic.hill_climbing import run_guided_hill_climber, run_hill_climber
from ..problem import circle_fitness, distance_matrix, valid_circle


@dataclass
class Measurement:
    """One single-size comparison run: score, counts, runtime and first-hit statistics."""
    algorithm: str
    instance_seed: int
    trial_seed: int
    score: int
    evaluations: int
    seconds: float
    evaluations_to_best: int
    seconds_to_best: float


def main() -> None:
    """Parse arguments; run the suite (suite options) or the single-size uniform vs guided comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--children", type=int, default=40)
    parser.add_argument("--characteristics", type=int, default=5)
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3,
                        help="Median runtime across identical seeded runs (suite: time protocol)")
    parser.add_argument("--check-repeats", type=int, default=1,
                        help="Suite checks-protocol repeats; fixed-count outcomes are identical, only timing differs")
    parser.add_argument("--budget", type=int, default=50_000,
                        help="Per-run cap: HC starts + proposals; GA distinct full-circle scores")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--exploration", type=float, default=0.1,
                        help="Uniform fraction of guided SWAP proposals (0=guided, 1=uniform)")
    parser.add_argument("--include-ga", action="store_true", default=True)
    parser.add_argument("--details", action="store_true", help="Print per-trial measurements")
    parser.add_argument("--suite", action="store_true", help="Run the size/exploration reporting suite")
    parser.add_argument("--sizes", type=int, nargs="+", help="Nested child-size sweep")
    parser.add_argument("--characteristic-counts", type=int, nargs="+", help="Separate characteristic-count suites")
    parser.add_argument("--explorations", type=float, nargs="+", help="Guided exploration sweep")
    parser.add_argument("--protocol", choices=("checks", "time", "both"))
    parser.add_argument("--time-limit", type=float, help="Seconds per equal-time execution (suite default 0.1)")
    parser.add_argument("--output", help="Suite output directory; defaults to a timestamped benchmark-results folder")
    parser.add_argument("--no-report", action="store_true", help="Export data without the optional Plotly report")
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True,
                        help="Separate fixed-check coverage/diversity replays in suite mode")
    parser.add_argument("--quick", action="store_true", help="Small complete suite for validating the pipeline")
    parser.add_argument("--generations", type=int, default=600, help="GA generation cap in check-budget runs")
    parser.add_argument("--max-restarts", type=int, default=200, help="HC restart cap in check-budget runs")
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--mutation-rate", type=float, default=0.25)
    parser.add_argument("--no-ga", dest="include_ga", action="store_false")
    parser.add_argument("--target-fractions", type=float, nargs="+", default=[0.9, 0.95, 0.99])
    parser.add_argument("--exact-limit", type=int, default=9, help="Exact reference up to this N; 0 disables")
    parser.add_argument("--workers", type=lambda value: 0 if value == "auto" else int(value), default=0,
                        help="Process workers for independent batches: auto (default), or a positive count")
    parser.add_argument("--timing-mode", choices=("isolated", "throughput"), default="isolated",
                        help="Isolated measurements + parallel diagnostics/stats; throughput parallelizes measurements too")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-interval", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true", help="Reuse compatible completed batch checkpoints in --output")
    parser.add_argument("--hc-restart-policy", choices=("productive", "legacy"), default="productive")
    parser.add_argument("--hc-attempts-per-restart", type=int, default=1000,
                        help="Hard local cap only in legacy HC restart mode")
    parser.add_argument("--hc-stall-limit", type=int, default=200)
    parser.add_argument("--inversion-hc", action=argparse.BooleanOptionalAction, default=True,
                        help="Include the coexisting inversion-focused HC in suite mode")
    parser.add_argument("--inversion-rate", type=float, default=0.8)
    parser.add_argument("--stop-at-bound", action=argparse.BooleanOptionalAction, default=True,
                        help="Suite searches stop on reaching the proven reference score")
    parser.add_argument("--alias-full-exploration", action=argparse.BooleanOptionalAction, default=True,
                        help="Copy checks-protocol exploration=1 runs from their identical uniform control")
    parser.add_argument("--reference", choices=("milp", "bound"), default="milp",
                        help="Suite reference: exact MILP optimum for every N, or phase-06 enumeration/bound")
    parser.add_argument("--reference-time-limit", type=float, default=60.0,
                        help="Seconds per instance for the MILP reference; on expiry its upper bound is used")
    parser.add_argument("--eras", nargs="+", default=[],
                        help="Registered eras to measure as extra families (python -m circle_optimization list)")
    parser.add_argument("--classic", action=argparse.BooleanOptionalAction, default=True,
                        help="Include the classic HC/GA/HC-I exploration grid (--no-classic: eras only)")
    args = parser.parse_args()
    if (args.suite or args.eras or not args.classic or args.sizes or args.explorations or args.characteristic_counts
            or args.protocol or args.time_limit is not None or args.output or args.quick or args.no_report or args.resume):
        from .suite import SuiteConfig, run_and_save, validate_config
        config = SuiteConfig(
            sizes=tuple(sorted(set(args.sizes or ([8, 16, 40] if args.quick else [10, 20, 40, 80, 160])))),
            characteristics=tuple(sorted(set(args.characteristic_counts or [args.characteristics]))),
            explorations=tuple(sorted(set(args.explorations or ([0, 0.5, 1] if args.quick else [0, 0.2, 0.5, 0.8, 1])))),
            protocols=("checks", "time") if args.protocol in (None, "both") else (args.protocol,),
            instances=min(args.instances, 1) if args.quick else args.instances,
            trials=min(args.trials, 2) if args.quick else args.trials,
            repeats=min(args.repeats, 1) if args.quick else args.repeats,
            check_repeats=min(args.check_repeats, 1) if args.quick else args.check_repeats,
            budget=min(args.budget, 1000) if args.quick else args.budget,
            time_limit=args.time_limit if args.time_limit is not None else (0.03 if args.quick else 0.1),
            seed=args.seed, population_size=args.population_size, generations=args.generations,
            max_restarts=args.max_restarts, mutation_rate=args.mutation_rate,
            include_ga=args.include_ga, diagnostics=args.diagnostics,
            exact_limit=args.exact_limit, targets=tuple(sorted(set(args.target_fractions))),
            workers=args.workers, timing_mode=args.timing_mode,
            progress=args.progress, progress_interval=args.progress_interval,
            hc_restart_policy=args.hc_restart_policy,
            hc_attempts_per_restart=args.hc_attempts_per_restart, hc_stall_limit=args.hc_stall_limit,
            include_inversion_hc=args.inversion_hc, inversion_rate=args.inversion_rate,
            stop_at_bound=args.stop_at_bound, alias_full_exploration=args.alias_full_exploration,
            reference=args.reference, reference_time_limit=args.reference_time_limit,
            eras=tuple(args.eras), classic=args.classic,
        )
        try:
            validate_config(config)
        except ValueError as error:
            parser.error(str(error))
        if args.resume and not args.output:
            parser.error("--resume requires the original --output directory")
        try:
            run_and_save(config, args.output, report=not args.no_report, resume=args.resume)
        except ValueError as error:
            parser.error(str(error))
        return
    if args.children < 4 or min(args.characteristics, args.instances, args.trials,
                                args.repeats, args.budget) < 1:
        parser.error("Use at least 4 children and positive sizes/budgets")
    if not 0 <= args.exploration <= 1:
        parser.error("Exploration must lie between 0 and 1")
    population = min(args.population_size, factorial(args.children - 1) // 2)
    if args.include_ga and args.budget < population:
        parser.error("The GA initial population must fit the evaluation budget")

    labels = ["HC uniform", "HC guided"]
    if args.include_ga:
        labels += ["GA uniform", "GA guided"]
    records: list[Measurement] = []
    for instance in range(args.instances):
        instance_seed = args.seed + instance
        rng = Random(instance_seed)
        distances = distance_matrix(tuple(
            tuple(bool(rng.getrandbits(1)) for _ in range(args.characteristics))
            for _ in range(args.children)
        ))

        def search(
            label: str, seed: int,
            on_improvement: Callable[[int, int], None] | None = None,
        ) -> dict:
            """Run one labelled algorithm (HC/GA, uniform or guided) on this instance."""
            if label == "HC uniform":
                return run_hill_climber(distances, Random(seed), evaluation_budget=args.budget,
                                        on_improvement=on_improvement, max_restarts=args.max_restarts,
                                        restart_policy=args.hc_restart_policy,
                                        attempts_per_restart=args.hc_attempts_per_restart,
                                        stall_limit=args.hc_stall_limit)
            if label == "HC guided":
                return run_guided_hill_climber(
                    distances, Random(seed), evaluation_budget=args.budget,
                    exploration=args.exploration,
                    on_improvement=on_improvement,
                    max_restarts=args.max_restarts,
                    restart_policy=args.hc_restart_policy,
                    attempts_per_restart=args.hc_attempts_per_restart, stall_limit=args.hc_stall_limit,
                )
            options = {}
            if label == "GA guided":
                options["mutation_operator"] = make_guided_mutation(
                    distances, exploration=args.exploration,
                )
            return run_ga(
                distances, Random(seed), population_size=population, generations=args.generations,
                tournament_size=min(10, population), elites=min(2, population),
                crossover_rate=0.9, mutation_rate=args.mutation_rate,
                evaluation_budget=args.budget, on_improvement=on_improvement, **options,
            )

        # Warm each algorithm before measurement; no warm-up result is reported.
        for label in labels:
            search(label, args.seed)
        for trial in range(args.trials):
            trial_seed = args.seed + 1_000_000 + instance * args.trials + trial
            results = {label: [] for label in labels}
            timings = {label: [] for label in labels}
            times_to_best = {label: [] for label in labels}
            for repeat in range(args.repeats):
                offset = (trial + repeat) % len(labels)
                for label in labels[offset:] + labels[:offset]:
                    improvements = []

                    def record_improvement(score: int, evaluations: int) -> None:
                        """Store (score, checks, elapsed seconds) of a strict improvement."""
                        improvements.append((score, evaluations, perf_counter() - start))

                    start = perf_counter()
                    result = search(label, trial_seed, record_improvement)
                    seconds = perf_counter() - start
                    assert valid_circle(result["circle"], args.children)
                    assert result["score"] == circle_fitness(result["circle"], distances)
                    assert result["evaluations"] <= args.budget
                    best_score, best_at, time_to_best = improvements[-1]
                    assert best_score == result["score"]
                    assert 1 <= best_at <= result["evaluations"]
                    assert 0 <= time_to_best <= seconds
                    results[label].append((result["circle"], result["score"],
                                           result["evaluations"], best_at))
                    timings[label].append(seconds)
                    times_to_best[label].append(time_to_best)
            for label in labels:
                assert all(result == results[label][0] for result in results[label])
                _, score, evaluations, best_at = results[label][0]
                records.append(Measurement(label, instance_seed, trial_seed, score,
                                           evaluations, median(timings[label]), best_at,
                                           median(times_to_best[label])))

    print(f"N={args.children}, C={args.characteristics}, instances={args.instances}, "
          f"trials/instance={args.trials}, budget={args.budget}, exploration={args.exploration}")
    print(f"Median of {args.repeats} repeats/trial; matrix construction excluded; selector setup included.")
    print("HC counts starts + proposed moves; GA counts distinct full-circle evaluations.")
    print("Equal count limits are not equal runtime budgets. Higher score and lower time are better.\n")
    print("To-best = first time reaching that run's final best score, not a proven optimum or shared target.")
    print(f"{'algorithm':<12} {'mean score ± sd':>21} {'best':>7} {'median ms':>11} "
          f"{'to-best ms':>11} {'best@check':>11} {'mean checks':>12}")
    for label in labels:
        group = [record for record in records if record.algorithm == label]
        scores = [record.score for record in group]
        spread = stdev(scores) if len(scores) > 1 else 0
        print(f"{label:<12} {mean(scores):>11.2f} ± {spread:<6.2f} {max(scores):>7} "
               f"{1000 * median(record.seconds for record in group):>11.3f} "
              f"{1000 * median(record.seconds_to_best for record in group):>11.3f} "
              f"{median(record.evaluations_to_best for record in group):>11.1f} "
              f"{mean(record.evaluations for record in group):>12.1f}")
    for family in (["HC", "GA"] if args.include_ga else ["HC"]):
        baseline = [record for record in records if record.algorithm == f"{family} uniform"]
        guided = [record for record in records if record.algorithm == f"{family} guided"]
        differences = [b.score - a.score for a, b in zip(baseline, guided)]
        print(f"\n{family} guided vs uniform paired score wins/ties/losses: "
              f"{sum(d > 0 for d in differences)}/{sum(d == 0 for d in differences)}/"
              f"{sum(d < 0 for d in differences)}; mean score change {mean(differences):+.2f}; "
              f"median time ratio {median(b.seconds / a.seconds for a, b in zip(baseline, guided)):.2f}x")
    if args.details:
        print("\nalgorithm,instance_seed,trial_seed,score,evaluations,seconds,"
              "evaluations_to_best,seconds_to_best")
        for record in records:
            print(f"{record.algorithm},{record.instance_seed},{record.trial_seed},"
                  f"{record.score},{record.evaluations},{record.seconds:.6f},"
                  f"{record.evaluations_to_best},{record.seconds_to_best:.6f}")


if __name__ == "__main__":
    main()
