"""Paired comparison of saved full grids, with targets rebased to common bounds.

Run after both suites finish:
    python compare_benchmark_runs.py benchmark-results/full benchmark-results/full-phase04
(equivalently `python -m circle_optimization.benchmark.compare OLD NEW`).

Searches are never rerun. Timing repeats are collapsed before clustered CIs.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from math import ceil
from pathlib import Path
from statistics import mean, median

from .metrics import clustered_ci, first_hit


CONDITION = ("protocol", "characteristics", "children", "family", "variant", "exploration")
PAIR = CONDITION + ("instance_seed", "trial_seed")
METRICS = ("score", "normalized_score", "seconds", "seconds_to_best", "evaluations_to_best",
           "evaluations", "auc_time", "common_reference_ratio", "post_best_fraction",
           "target90_success", "target95_success", "target99_success", "restarts",
           "target90_seconds", "target95_seconds", "target99_seconds",
           "generations_completed", "swap_improvement_rate", "inversion_improvement_rate",
           "novelty_rate", "edge_diversity", "cache_hit_rate")


def load_csv(path):
    """Read a suite CSV; empty cells become None and numeric/JSON cells are decoded."""
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key, value in row.items():
            if value == "":
                row[key] = None
            elif key not in ("run_id", "protocol", "family", "variant", "reference_kind", "status", "stop_reason"):
                try:
                    row[key] = json.loads(value)
                except (ValueError, TypeError):
                    pass
    return rows


def group(rows, fields):
    """Group rows by the values of `fields` (missing fields count as None)."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in fields)].append(row)
    return groups


def read_run(path):
    """Load a saved suite directory and check its row count against its configuration."""
    manifest = json.loads((path / "configuration.json").read_text())
    rows = load_csv(path / "runs.csv")
    points = group(load_csv(path / "improvements.csv"), ("run_id",))
    problems = load_csv(path / "problems.csv")
    config = manifest["config"]
    # Older runs used one repeat count for both protocols.
    repeats = {p: config.get("check_repeats", config["repeats"]) if p == "checks" else config["repeats"]
               for p in config["protocols"]}
    count = (len(config["sizes"]) * len(config["characteristics"]) * config["instances"]
             * config["trials"] * sum(repeats.values()))
    variants = len({(r["family"], r["variant"], r["exploration"]) for r in rows})
    assert len(rows) == count * variants, (path, len(rows), count, variants)
    assert len({r["run_id"] for r in rows}) == len(rows)
    assert all((r["run_id"],) in points for r in rows)
    return dict(manifest=manifest, rows=rows, points=points, problems=problems)


def collapse(rows):
    """One row per paired trial: mean of target successes, median of other metrics over repeats."""
    output = {}
    for key, repeated in group(rows, PAIR).items():
        row = dict(zip(PAIR, key))
        for metric in METRICS:
            values = [r[metric] for r in repeated if r.get(metric) is not None]
            row[metric] = ((mean(values) if metric.endswith("_success") else median(values))
                           if values else None)
        output[key] = row
    return output


def paired_statistics(pairs, identity, comparison):
    """Paired old/new differences, clustered bootstrap interval and change counts per metric."""
    rows = []
    for metric in METRICS:
        usable = [(a, b) for a, b in pairs if a.get(metric) is not None and b.get(metric) is not None]
        if not usable:
            continue
        deltas = [dict(instance_seed=a["instance_seed"], delta=b[metric] - a[metric]) for a, b in usable]
        lo, hi = clustered_ci(deltas, "delta")
        ratios = [b[metric] / a[metric] for a, b in usable if a[metric] > 0]
        rows.append(dict(identity, comparison=comparison, metric=metric,
                         old_mean=mean(a[metric] for a, _ in usable),
                         new_mean=mean(b[metric] for _, b in usable),
                         mean_delta=mean(r["delta"] for r in deltas), ci_low=lo, ci_high=hi,
                         median_paired_ratio=median(ratios) if ratios else None,
                         increases=sum(r["delta"] > 0 for r in deltas),
                         ties=sum(r["delta"] == 0 for r in deltas),
                         decreases=sum(r["delta"] < 0 for r in deltas), pairs=len(usable)))
    return rows


def compare(old, new):
    """Pair two runs trial by trial, rebase targets to the new references, check GA traces."""
    settings = ("sizes", "characteristics", "explorations", "instances", "trials", "repeats",
                "budget", "time_limit", "seed", "population_size", "generations", "mutation_rate")
    for key in settings:
        assert old["manifest"]["config"][key] == new["manifest"]["config"][key], key
    if new["manifest"].get("timing_mode", "isolated") != "isolated":
        raise ValueError("A direct latency comparison requires isolated new measurements")
    references = {(p["characteristics"], p["children"], p["instance_seed"]):
                  p["exact_optimum"] if p["exact_optimum"] is not None else p["upper_bound"]
                  for p in new["problems"]}
    for version in (old, new):
        for row in version["rows"]:
            reference = references[row["characteristics"], row["children"], row["instance_seed"]]
            row["common_reference_ratio"] = ((row["score"] / reference if reference else 1)
                                              if row["score"] is not None else 0)
            points = version["points"][row["run_id"],]
            if row["protocol"] == "time":
                points = [p for p in points if p["seconds"] <= version["manifest"]["config"]["time_limit"]]
            for fraction, metric in ((0.9, "target90_success"), (0.95, "target95_success"), (0.99, "target99_success")):
                hit_time = first_hit(points, ceil(fraction * reference))
                row[metric] = int(hit_time is not None)
                row[metric.replace("success", "seconds")] = hit_time
    a, b = collapse(old["rows"]), collapse(new["rows"])
    stats = []
    conditions = group(list(a.values()), CONDITION)
    for key, old_rows in conditions.items():
        pairs = [(r, b[tuple(r[k] for k in PAIR)]) for r in old_rows]
        stats.extend(paired_statistics(pairs, dict(zip(CONDITION, key)), "old_vs_new"))
    # HC-I is new: compare it against productive uniform HC and optimized GA.
    for key, inversion_rows in group([r for r in b.values() if r["family"] == "HC-I"], CONDITION).items():
        for baseline_family in ("HC", "GA"):
            pairs = []
            for row in inversion_rows:
                base_key = tuple(baseline_family if field == "family" else row[field] for field in PAIR)
                pairs.append((b[base_key], row))
            stats.extend(paired_statistics(pairs, dict(zip(CONDITION, key)), f"HC-I_vs_new_{baseline_family}"))
    regressions = []
    checked = 0
    raw_new = group(new["rows"], PAIR)
    for row in old["rows"]:
        if row["protocol"] != "checks" or row["family"] != "GA":
            continue
        key = tuple(row[k] for k in PAIR)
        # Fixed-count repeats are identical, so a run with fewer checks repeats pairs with repeat 0.
        partner = next((r for r in raw_new[key] if r["repeat"] == row["repeat"]),
                       next(r for r in raw_new[key] if r["repeat"] == 0))
        old_trace = [(p["score"], p["evaluations"]) for p in old["points"][row["run_id"],]]
        new_trace = [(p["score"], p["evaluations"]) for p in new["points"][partner["run_id"],]]
        checked += 1
        # Stopping at a proven bound shortens the count but cannot change score or trace.
        same_count = (partner.get("stop_reason") == "bound_reached"
                      or row["evaluations"] == partner["evaluations"])
        if (row["score"], old_trace) != (partner["score"], new_trace) or not same_count:
            regressions.append(row["run_id"])
    return dict(statistics=stats, ga_checked=checked, ga_mismatches=regressions,
                old_runs=len(old["rows"]), new_runs=len(new["rows"]),
                old_status=dict(Counter(r["status"] for r in old["rows"])),
                new_status=dict(Counter(r["status"] for r in new["rows"])),
                old_elapsed=old["manifest"]["elapsed_seconds"],
                new_elapsed=new["manifest"]["elapsed_seconds"],
                new_total=new["manifest"].get("total_seconds"),
                execution=new.get("execution", []))


def render(result, old, new, directory):
    """Write comparison.html: old versus new HC/GA (and HC-I) curves per protocol and metric."""
    import plotly.graph_objects as go
    chunks = []
    first = True
    for protocol in ("checks", "time"):
        for metric in ("score", "normalized_score", "seconds", "seconds_to_best", "auc_time"):
            fig = go.Figure()
            for family in ("HC", "GA"):
                selected = [s for s in result["statistics"] if s["comparison"] == "old_vs_new"
                            and s["variant"] == "uniform" and s["family"] == family
                            and s["protocol"] == protocol and s["metric"] == metric]
                selected.sort(key=lambda r: r["children"])
                for version in ("old", "new"):
                    fig.add_trace(go.Scatter(x=[r["children"] for r in selected],
                                             y=[r[version + "_mean"] for r in selected],
                                             name=f"{version} {family}", mode="lines+markers",
                                             line=dict(dash="dash" if version == "old" else "solid")))
            inversion = [s for s in result["statistics"] if s["comparison"] == "HC-I_vs_new_HC"
                         and s["protocol"] == protocol and s["metric"] == metric]
            inversion.sort(key=lambda r: r["children"])
            fig.add_trace(go.Scatter(x=[r["children"] for r in inversion],
                                     y=[r["new_mean"] for r in inversion], name="new HC-I", mode="lines+markers"))
            fig.update_layout(title=f"{protocol}: {metric}", template="plotly_white",
                              xaxis_title="Children", yaxis_title=metric)
            chunks.append(fig.to_html(full_html=False, include_plotlyjs=first))
            first = False
    html = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Historical comparison</title>'
            '<body style="font-family:system-ui;max-width:1300px;margin:30px auto">'
            '<h1>Historical full run versus phase 04</h1>'
            '<p>Solid lines: enhanced implementation. Dashed lines: historical run. '
            'All measured searches are isolated. Targets in the comparison data are rebased '
            'to the new common references; raw scores do not depend on reference changes. '
            'For anytime-efficiency comparisons use the time protocol: count-protocol time AUC '
            'is normalized over each run’s different elapsed duration. Target-hit times are '
            'conditional on successful attainment and must be read alongside success rates.</p>'
            f'<p>Verified GA fixed-count traces: {result["ga_checked"]}; mismatches: {len(result["ga_mismatches"])}.</p>'
            + ''.join(chunks) + '</body></html>')
    (directory / "comparison.html").write_text(html, encoding="utf-8")


def main():
    """Command line: compare two saved runs; write comparison.json/csv/html into the newer one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--quiet", action="store_true", help="Write all results, suppress per-metric console output")
    args = parser.parse_args()
    old, new = read_run(args.old), read_run(args.new)
    result = compare(old, new)
    if (args.new / "execution.csv").exists():
        result["execution"] = load_csv(args.new / "execution.csv")
    (args.new / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with (args.new / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result["statistics"][0]))
        writer.writeheader()
        writer.writerows(result["statistics"])
    render(result, old, new, args.new)
    if not args.quiet:
        print("INTEGRITY", {k: v for k, v in result.items() if k != "statistics"})
        for row in result["statistics"]:
            if row["metric"] in ("score", "seconds") \
                    and row["variant"] == "uniform":
                print(row)
    print(f"Comparison saved to {args.new / 'comparison.html'}")


if __name__ == "__main__":
    main()
