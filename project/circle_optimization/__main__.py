"""Command line for the era registry: list, describe, run or compare eras.

    python -m circle_optimization list
    python -m circle_optimization describe hc-p-splice
    python -m circle_optimization run hc-i --children 160 --time-limit 0.2
    python -m circle_optimization run hc-p-ils --set perturbation=null --set splice_rate=0.3
    python -m circle_optimization compare hc hc-i hc-p-splice --children 160 --repeats 5
    python -m circle_optimization compare --all --children 1000 --time-limit 0.5

Instances come from `problem.random_profiles(--instance-seed, --children,
--characteristics)`, the benchmark suite's generator. `--set key=value`
overrides an era parameter (JSON values: numbers, true/false, null, strings).
`compare` gives every era the same instance, time limit and seeds, and reports
the gap to the proven optimum when the exact solver is available.
"""

import argparse
import json
from statistics import mean
from time import perf_counter

from .eras import ERAS, Problem, run_era


def _value(text: str):
    """Parse a `--set` value as JSON, falling back to the raw string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _overrides(pairs: list[str]) -> dict:
    """Turn ["key=value", ...] into keyword overrides."""
    result = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        result[key.replace("-", "_")] = _value(value)
    return result


def _problem(args) -> Problem:
    """Build the instance named by the instance options."""
    return Problem.random(args.children, args.characteristics, seed=args.instance_seed)


def _optimum(problem: Problem, wanted: bool) -> int | None:
    """Proven optimum if requested and the exact era's dependencies are installed."""
    if not wanted or not ERAS["exact"].available():
        return None
    started = perf_counter()
    optimum = problem.optimum
    print(f"{problem.label}: optimum {optimum} (exact solver {1000 * (perf_counter() - started):.1f} ms)")
    return optimum


def _limits(args) -> dict:
    """Time limit and/or evaluation budget for a run."""
    return dict(time_limit=args.time_limit, evaluation_budget=args.budget)


def command_list(args) -> None:
    """Print the registry, grouped by phase, with availability."""
    print(f"{'era':<12} {'family':<6} {'phase':<5} {'extends':<12} {'layers'}")
    for era in ERAS.values():
        if args.family and era.family != args.family:
            continue
        missing = "" if era.available() else f"  [needs {', '.join(era.requires)}]"
        print(f"{era.name:<12} {era.family:<6} {era.phase:<5} {era.extends or '-':<12} "
              f"{', '.join(era.layers)}{missing}")


def command_describe(args) -> None:
    """Print one era's full record."""
    era = ERAS[args.era]
    print(f"{era.name} — {era.title} (phase {era.phase}, family {era.family})")
    print(era.summary)
    print(f"representation: {era.representation}; extends: {era.extends or '-'}")
    print(f"layers: {', '.join(era.layers)}")
    print(f"budget unit: {era.budget_unit}")
    print(f"requires: {', '.join(era.requires) or 'standard library'}")
    print(f"defaults: {json.dumps(dict(era.defaults), default=str)}")


def command_run(args) -> None:
    """Run one era once and print its result summary (or JSON)."""
    problem = _problem(args)
    optimum = _optimum(problem, args.optimum)
    result = run_era(args.era, problem, seed=args.seed, **_limits(args), **_overrides(args.set))
    summary = dict(era=args.era, score=result["score"], optimum=optimum,
                   gap=None if optimum is None else optimum - result["score"],
                   seconds=round(result["seconds"], 4), evaluations=result.get("evaluations"),
                   stop_reason=result.get("stop_reason"), layers=list(result["layers"]))
    if args.json:
        print(json.dumps(dict(summary, circle=list(result["circle"])), default=str))
    else:
        for key, value in summary.items():
            print(f"{key:>12}: {value}")


def command_compare(args) -> None:
    """Run several eras with identical instance, limits and seeds; print a table."""
    names = list(ERAS) if args.all else args.eras
    if args.family:
        names = [name for name in names if ERAS[name].family == args.family]
    names = [name for name in names if ERAS[name].available() or print(f"skip {name}: missing dependency")]
    if not names:
        raise SystemExit("No eras selected: name some, or use --all / --family")
    problem = _problem(args)
    optimum = _optimum(problem, True)
    problem.distances, problem.state  # Build representations outside the timed runs.
    rows = []
    for name in names:
        runs = [run_era(name, problem, seed=args.seed + r, **_limits(args)) for r in range(args.repeats)]
        scores = [r["score"] for r in runs]
        rows.append(dict(era=name, phase=ERAS[name].phase, mean=mean(scores), best=max(scores),
                         gap=None if optimum is None else 100 * (optimum - mean(scores)) / optimum,
                         seconds=mean(r["seconds"] for r in runs)))
    print(f"{'era':<12} {'phase':<5} {'mean':>9} {'best':>6} {'gap %':>7} {'seconds':>8}")
    for row in rows:
        gap = "-" if row["gap"] is None else f"{row['gap']:.2f}"
        print(f"{row['era']:<12} {row['phase']:<5} {row['mean']:>9.2f} {row['best']:>6} {gap:>7} {row['seconds']:>8.3f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(dict(problem=problem.label, optimum=optimum, time_limit=args.time_limit,
                           budget=args.budget, repeats=args.repeats, rows=rows), file, indent=2)


def main(argv=None) -> None:
    """Parse arguments and dispatch to a subcommand."""
    parser = argparse.ArgumentParser(prog="python -m circle_optimization", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List registered eras")
    listing.add_argument("--family", choices=("HC", "GA", "exact"))
    describe = commands.add_parser("describe", help="Show one era's record")
    describe.add_argument("era", choices=list(ERAS))
    for name, help_text in (("run", "Run one era"), ("compare", "Compare eras on one instance")):
        sub = commands.add_parser(name, help=help_text)
        if name == "run":
            sub.add_argument("era", choices=list(ERAS))
            sub.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                             help="Override an era parameter (repeatable)")
            sub.add_argument("--optimum", action="store_true", help="Also solve exactly and report the gap")
            sub.add_argument("--json", action="store_true", help="Print the result as JSON (with circle)")
        else:
            sub.add_argument("eras", nargs="*", help="Era names (default: --all)")
            sub.add_argument("--all", action="store_true", help="Every available era")
            sub.add_argument("--family", choices=("HC", "GA", "exact"))
            sub.add_argument("--repeats", type=int, default=3, help="Seeds per era (seed, seed+1, ...)")
            sub.add_argument("--json", help="Also write the table to this JSON file")
        sub.add_argument("--children", type=int, default=160)
        sub.add_argument("--characteristics", type=int, default=5)
        sub.add_argument("--instance-seed", type=int, default=2027)
        sub.add_argument("--seed", type=int, default=1, help="Search seed")
        sub.add_argument("--time-limit", type=float, default=0.2, help="Seconds per run (equal-time protocol)")
        sub.add_argument("--budget", type=int, help="Evaluation budget in each era's own unit")
    args = parser.parse_args(argv)
    if args.command == "compare" and not args.eras:
        args.all = True
    dict(list=command_list, describe=command_describe, run=command_run, compare=command_compare)[args.command](args)


if __name__ == "__main__":
    main()
