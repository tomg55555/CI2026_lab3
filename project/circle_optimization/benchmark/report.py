"""Offline interactive Plotly report. Imported only when HTML is requested."""

from collections import defaultdict
from html import escape
import json
from statistics import mean, median

from .metrics import collapse_repeats, step_value


def require_plotly():
    """Import plotly.graph_objects, or explain how to install it."""
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise RuntimeError(
            "Interactive reports require Plotly. Install project/requirements-benchmark.txt "
            "or use --no-report for CSV/JSON only."
        ) from error
    return go


def label(row):
    """Legend name of a run/summary row: classic family + variant, or the era name."""
    if row["variant"] == "era":
        return f"era {row['family']}"
    if row["family"] == "HC-I":
        return "HC inversion-focused"
    return (f"{row['family']} uniform" if row["variant"] == "uniform" else
            f"{row['family']} guided e={row['exploration']:g}")


def groups(rows, key):
    """Group rows by `key(row)`."""
    result = defaultdict(list)
    for row in rows:
        result[key(row)].append(row)
    return result


def line_summary(go, rows, metric, title):
    """Mean of `metric` versus N per algorithm, with clustered-interval error bars."""
    fig = go.Figure()
    for name, series in groups([r for r in rows if r["metric"] == metric], label).items():
        series.sort(key=lambda r: r["children"])
        values = [r["mean"] for r in series]
        fig.add_trace(go.Scatter(
            x=[r["children"] for r in series], y=values, mode="lines+markers", name=name,
            error_y=dict(type="data", symmetric=False,
                         array=[max(0, r["ci_high"] - r["mean"]) if r["ci_high"] is not None else 0 for r in series],
                         arrayminus=[max(0, r["mean"] - r["ci_low"]) if r["ci_low"] is not None else 0 for r in series]),
            customdata=[[r["sd"], r["trials"], r["instances"]] for r in series],
            hovertemplate="N=%{x}<br>Mean=%{y:.5g}<br>SD=%{customdata[0]:.4g}"
                          "<br>Trials=%{customdata[1]}, instances=%{customdata[2]}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(title=title, xaxis_title="Children N", yaxis_title=metric)
    return fig


EXPLORATION_METRICS = (
    "normalized_score", "reference_ratio", "relative_gap", "seconds", "seconds_to_best",
    "evaluations_to_best", "auc_time", "auc_checks", "post_best_fraction", "target_success",
    "throughput", "improvement_count", "largest_stagnation_fraction", "swap_improvement_rate",
    "inversion_improvement_rate", "swap_net_gain_per_proposal", "novelty_rate", "edge_coverage",
    "edge_diversity", "edge_entropy", "cache_hit_rate", "duplicate_rate",
    "actual_uniform_swap_fraction", "log10_circle_coverage", "restart_improvements",
    "restart_exit_attempt_cap", "restart_exit_stall",
)


def exploration_response(go, rows, *, differences=False):
    """A metric (or its paired change vs uniform) against exploration, with a metric dropdown."""
    fig = go.Figure()
    menus = []
    supported = []
    for metric in EXPLORATION_METRICS:
        chosen = [r for r in rows if r["metric"] == metric
                  and (r.get("comparison") == "uniform" if differences else r["variant"] == "guided")]
        if not chosen:
            continue
        start = len(fig.data)
        for (family, n), series in groups(chosen, lambda r: (r["family"], r["children"])).items():
            series.sort(key=lambda r: r["exploration"])
            field = "mean_delta" if differences else "mean"
            fig.add_trace(go.Scatter(
                x=[r["exploration"] for r in series], y=[r[field] for r in series],
                mode="lines+markers", name=f"{family}, N={n}", visible=not supported,
                error_y=dict(type="data", symmetric=False,
                             array=[max(0, r["ci_high"] - r[field]) if r["ci_high"] is not None else 0 for r in series],
                             arrayminus=[max(0, r[field] - r["ci_low"]) if r["ci_low"] is not None else 0 for r in series]),
            ))
        supported.append((metric, start, len(fig.data)))
    for metric, start, end in supported:
        menus.append(dict(label=metric, method="update", args=[
            dict(visible=[start <= i < end for i in range(len(fig.data))]),
            dict(yaxis=dict(title=metric)),
        ]))
    fig.update_layout(
        title="Paired change versus uniform search" if differences else "Exploration response (select a metric)",
        xaxis_title="Exploration: fraction of swaps chosen uniformly",
        yaxis_title=supported[0][0] if supported else "No data",
        updatemenus=[dict(buttons=menus, x=0, y=1.18, xanchor="left")], margin=dict(t=115),
    )
    if differences:
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    return fig


def heatmaps(go, summaries):
    """Heatmaps of a metric over N x exploration, one per guided family."""
    from plotly.subplots import make_subplots
    families = sorted({r["family"] for r in summaries if r["variant"] == "guided"})
    fig = make_subplots(rows=1, cols=len(families), subplot_titles=families)
    supported = []
    for metric in EXPLORATION_METRICS:
        subset = [r for r in summaries if r["variant"] == "guided" and r["metric"] == metric]
        if not subset:
            continue
        start = len(fig.data)
        minimum, maximum = min(r["mean"] for r in subset), max(r["mean"] for r in subset)
        for col, family in enumerate(families, 1):
            series = [r for r in subset if r["family"] == family]
            xs = sorted({r["exploration"] for r in series})
            ys = sorted({r["children"] for r in series})
            lookup = {(r["children"], r["exploration"]): r["mean"] for r in series}
            fig.add_trace(go.Heatmap(x=xs, y=ys,
                                    z=[[lookup.get((n, a)) for a in xs] for n in ys],
                                    colorscale="Viridis", zmin=minimum, zmax=maximum,
                                    showscale=col == len(families),
                                    visible=not supported,
                                    hovertemplate="Exploration=%{x}<br>N=%{y}<br>Mean=%{z:.5g}<extra></extra>"),
                          row=1, col=col)
        supported.append((metric, start, len(fig.data)))
    buttons = [dict(label=metric, method="update", args=[
        dict(visible=[start <= i < end for i in range(len(fig.data))])])
        for metric, start, end in supported]
    fig.update_xaxes(title_text="Exploration")
    fig.update_yaxes(title_text="Children N")
    fig.update_layout(title="Exploration × size heatmaps", margin=dict(t=115),
                      updatemenus=[dict(buttons=buttons, x=0, y=1.2)])
    return fig


def convergence(go, runs, trace_lookup, *, axis="seconds"):
    """Mean best-so-far normalized score over time or checks per algorithm (61-point grid)."""
    fig = go.Figure()
    horizon = max(r["seconds"] if axis == "seconds" else r["evaluations"] for r in runs)
    if runs[0]["protocol"] == "time" and axis == "seconds":
        # First-hit histories used below are truncated to the intended allowance.
        horizon = runs[0]["time_allowance"]
    if runs[0]["protocol"] == "checks" and axis == "evaluations":
        horizon = runs[0]["auc_check_horizon"]
    grid = [horizon * i / 60 for i in range(61)]
    for name, series in groups(runs, label).items():
        # Each trial contributes one curve: median across its timing repeats.
        trials = groups(series, lambda r: (r["instance_seed"], r["trial_seed"]))
        curves = []
        for repeated in trials.values():
            curves.append([median(step_value(trace_lookup[r["run_id"]], x, axis=axis,
                                              scale=r["children"] * r["characteristics"])
                                  for r in repeated) for x in grid])
        fig.add_trace(go.Scatter(x=grid, y=[mean(c[i] for c in curves) for i in range(len(grid))],
                                 mode="lines", line_shape="hv", name=name))
    fig.update_layout(title=f"N={runs[0]['children']}: mean best-so-far quality versus {axis}",
                      xaxis_title="Elapsed seconds" if axis == "seconds" else "Checks (HC and GA units differ)",
                      yaxis_title="Score / (N × C)")
    return fig


def target_attainment(go, runs, targets):
    """Fraction of runs reaching each common target over time; misses stay in the denominator."""
    fig = go.Figure()
    fractions = sorted({r["fraction"] for r in targets})
    sections = []
    horizon = max(r["seconds"] for r in runs)
    for fraction in fractions:
        start = len(fig.data)
        for name, series in groups([r for r in targets if r["fraction"] == fraction], label).items():
            hits = sorted(r["seconds"] for r in series if r["hit"])
            xs = [0] + hits + [max(horizon, hits[-1] if hits else 0)]
            ys = [0] + [(i + 1) / len(series) for i in range(len(hits))] + [len(hits) / len(series)]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line_shape="hv",
                                     name=f"{name} ({len(hits)}/{len(series)} hits)",
                                     visible=not sections))
        sections.append((fraction, start, len(fig.data)))
    buttons = [dict(label=f"{fraction:.0%} of reference", method="update", args=[
        dict(visible=[start <= i < end for i in range(len(fig.data))])])
        for fraction, start, end in sections]
    fig.update_layout(title=f"N={runs[0]['children']}: common-target attainment (failures stay in denominator)",
                      xaxis_title="Elapsed seconds", yaxis_title="Fraction of runs reaching target",
                      yaxis_range=[0, 1.02], margin=dict(t=110),
                      updatemenus=[dict(buttons=buttons, x=0, y=1.18)])
    return fig


def diagnostic_curve(go, snapshots, metric, n):
    """A diagnostic snapshot metric (coverage, diversity) against checks per algorithm."""
    fig = go.Figure()
    for name, series in groups(snapshots, label).items():
        trials = groups(series, lambda r: r["diagnostic_id"])
        horizon = max(r["evaluations"] for r in series)
        grid = [horizon * i / 40 for i in range(41)]
        curves = []
        for points in trials.values():
            points.sort(key=lambda r: r["evaluations"])
            curve = []
            for x in grid:
                seen = [r for r in points if r["evaluations"] <= x and r.get(metric) is not None]
                curve.append(seen[-1][metric] if seen else None)
            curves.append(curve)
        ys = [mean(c[i] for c in curves if c[i] is not None)
              if any(c[i] is not None for c in curves) else None for i in range(len(grid))]
        fig.add_trace(go.Scatter(x=grid, y=ys, mode="lines", line_shape="hv", name=name))
    fig.update_layout(title=f"N={n}: {metric} in untimed diagnostic replays",
                      xaxis_title="Checks (HC and GA units differ)", yaxis_title=metric)
    return fig


def table(rows, fields):
    """HTML table of `fields` for `rows` (None shown as an em dash)."""
    def cell(value):
        """Format one table value."""
        if value is None:
            return "—"
        return f"{value:.5g}" if isinstance(value, float) else str(value)
    return ('<div class="table-scroll"><table><thead><tr>'
            + "".join(f"<th>{escape(k)}</th>" for k in fields) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{escape(cell(r.get(k)))}</td>" for k in fields) + "</tr>"
                      for r in rows) + "</tbody></table></div>")


def render_report(data, manifest, path, *, on_progress=None):
    """Write the offline interactive HTML report for saved suite data."""
    go = require_plotly()
    config = manifest["config"]
    first = True
    graph_count = 0
    graph_total = len(config["characteristics"]) * len(config["protocols"]) * (11 + 3 * len(config["sizes"]))
    if "checks" in config["protocols"]:
        graph_total += 3 * len({(r["characteristics"], r["children"])
                                for r in data["diagnostic_snapshots"]})
    graph_total += bool(data.get("execution"))

    def chart(fig):
        """Embed one figure, including plotly.js only once."""
        nonlocal first, graph_count
        graph_count += 1
        fig.update_layout(template="plotly_white", font=dict(family="system-ui, sans-serif"),
                          height=450, legend=dict(orientation="h", y=-0.22),
                          margin=dict(b=140), hovermode="closest")
        html = fig.to_html(full_html=False, include_plotlyjs=first,
                           div_id=f"chart-{graph_count}", config=dict(responsive=True, displaylogo=False))
        first = False
        if on_progress is not None:
            on_progress(graph_count, graph_total, str(fig.layout.title.text or "chart"))
        return '<div class="chart">' + html + "</div>"

    trace_lookup = defaultdict(list)
    raw_by_id = {r["run_id"]: r for r in data["runs"]}
    for point in data["improvements"]:
        row = raw_by_id[point["run_id"]]
        if row["protocol"] != "time" or point["seconds"] <= config["time_limit"]:
            trace_lookup[point["run_id"]].append(point)
    sections, options = [], []
    for c in config["characteristics"]:
        for protocol in config["protocols"]:
            context = f"c{c}-{protocol}"
            options.append(f'<option value="{context}">C={c} · {protocol} budget</option>')
            runs = [dict(r, time_allowance=config["time_limit"]) for r in data["runs"]
                    if r["characteristics"] == c and r["protocol"] == protocol]
            summaries = [r for r in data["summaries"] if r["characteristics"] == c and r["protocol"] == protocol]
            effects = [r for r in data["exploration_effects"] if r["characteristics"] == c and r["protocol"] == protocol]
            trends = [r for r in data["exploration_trends"] if r["characteristics"] == c and r["protocol"] == protocol]
            content = [f'<section class="context" data-context="{context}"><h2>C={c} · {protocol} protocol</h2>']
            if "hc_restart_policy" in config:
                content.append(f"<p>HC restart policy: <b>{escape(config['hc_restart_policy'])}</b>. "
                               f"Inversion-focused HC uses {config['inversion_rate']:.0%} inversions "
                               f"and {1 - config['inversion_rate']:.0%} swaps.</p>")
            content.append('<p>Scaling plots show means and 95% hierarchical bootstrap intervals. '
                           'Intervals are omitted with one instance family; timing repeats are not independent trials.</p>')
            for metric, title in (
                ("normalized_score", "Quality as children grow"),
                ("relative_gap", "Gap to exact optimum / upper bound (see problems.csv)"),
                ("seconds", "Total runtime scaling"),
                ("seconds_to_best", "First time reaching each run’s own final best"),
                ("post_best_fraction", "Fraction of search after the last score improvement"),
                ("auc_time", "Anytime quality: normalized area under the convergence curve"),
            ):
                content.append(chart(line_summary(go, summaries, metric, title)))
            # Era-only suites (--no-classic) have no guided exploration grid to chart.
            if any(r["variant"] == "guided" for r in summaries):
                content.append('<h3>How exploration changes the metrics</h3>'
                               '<p>The metric dropdowns include quality, runtime, first-hit time, AUC, '
                               'target success and, in the checks protocol, optional search diagnostics. '
                               'target_success uses 95% of the reference, or the first configured fraction '
                               'when 95% is absent. Missing solutions before a deadline contribute zero '
                               'quality rather than being dropped.</p>')
                content.append(chart(exploration_response(go, summaries)))
                content.append(chart(heatmaps(go, summaries)))
                content.append(chart(exploration_response(go, effects, differences=True)))
            scatter = go.Figure()
            for name, series in groups(collapse_repeats(runs), label).items():
                scatter.add_trace(go.Scatter(x=[r["seconds"] for r in series],
                                            y=[r["normalized_score"] for r in series], mode="markers", name=name,
                                            text=[f"N={r['children']}, seed={r['trial_seed']}" for r in series]))
            for n in config["sizes"]:
                candidates = []
                selected = [r for r in collapse_repeats(runs)
                            if r["children"] == n and r["normalized_score"] is not None]
                for name, series in groups(selected, label).items():
                    candidates.append((mean(r["seconds"] for r in series),
                                       mean(r["normalized_score"] for r in series), name))
                frontier = sorted(p for p in candidates if not any(
                    other[0] <= p[0] and other[1] >= p[1]
                    and (other[0] < p[0] or other[1] > p[1]) for other in candidates))
                scatter.add_trace(go.Scatter(x=[p[0] for p in frontier], y=[p[1] for p in frontier],
                                            text=[p[2] for p in frontier], mode="lines+markers",
                                            line=dict(dash="dash"), name=f"Mean Pareto frontier N={n}"))
            scatter.update_layout(title="Quality–runtime trade-off (one point per paired trial)",
                                  xaxis_title="Seconds", yaxis_title="Score / (N × C)")
            content.append(chart(scatter))
            variability = go.Figure()
            for name, series in groups(collapse_repeats(runs), label).items():
                variability.add_trace(go.Box(x=[r["children"] for r in series],
                                              y=[r["normalized_score"] for r in series], name=name,
                                              boxpoints=False))
            variability.update_layout(title="Quality variability across instances and search seeds",
                                       xaxis_title="Children N", yaxis_title="Score / (N × C)", boxmode="group")
            content.append(chart(variability))
            for n in config["sizes"]:
                selected = [r for r in runs if r["children"] == n]
                ids = {r["run_id"] for r in selected}
                content.append(f"<details><summary>Convergence, targets and diagnostics · N={n}</summary>")
                content.append(chart(convergence(go, selected, trace_lookup)))
                content.append(chart(convergence(go, selected, trace_lookup, axis="evaluations")))
                content.append(chart(target_attainment(go, selected,
                                                      [r for r in data["targets"] if r["run_id"] in ids])))
                if protocol == "checks":
                    snapshots = [r for r in data["diagnostic_snapshots"]
                                 if r["characteristics"] == c and r["children"] == n]
                    if snapshots:
                        for metric in ("edge_coverage", "edge_diversity", "unique_scored"):
                            content.append(chart(diagnostic_curve(go, snapshots, metric, n)))
                content.append("</details>")
            key_metrics = {"normalized_score", "seconds", "seconds_to_best", "auc_time", "target_success",
                           "swap_improvement_rate", "novelty_rate", "edge_diversity"}
            content.append('<details><summary>Paired exploration changes: uniform, adjacent α and first α</summary>')
            content.append(table([r for r in effects if r["metric"] in key_metrics],
                                 ["children", "family", "exploration", "comparison", "reference_exploration",
                                  "metric", "mean_delta", "ci_low", "ci_high", "mean_ratio", "paired_dz",
                                  "mean_percent_change",
                                  "increases", "ties", "decreases", "pairs"]))
            content.append("</details><details><summary>Exploration trends: slope per +0.1 and Spearman correlation</summary>")
            content.append(table(trends, ["children", "family", "metric", "mean_slope_per_0_1",
                                          "ci_low", "ci_high", "mean_spearman", "paired_trials"]))
            content.append("</details><h3>Stopping reasons and deadline overshoot</h3>")
            stops = []
            for (n, name, reason, status), group in groups(
                    runs, lambda r: (r["children"], label(r), r["stop_reason"], r["status"])).items():
                stops.append(dict(children=n, algorithm=name, reason=reason, status=status, executions=len(group),
                                  median_overshoot_ms=1000 * median(r["overshoot_seconds"] for r in group)))
            content.append(table(stops, ["children", "algorithm", "reason", "status", "executions", "median_overshoot_ms"]))
            content.append("</section>")
            sections.append("".join(content))

    problem_html = table(data["problems"], ["children", "characteristics", "instance_seed", "distinct_profiles",
                                           "duplicate_profile_fraction", "matrix_seconds", "upper_bound", "exact_optimum",
                                           "exact_source", "milp_status", "milp_upper_bound", "milp_seconds"])
    execution_html = ""
    if data.get("execution"):
        fig = go.Figure(go.Bar(x=[r["phase"] for r in data["execution"]],
                               y=[r["seconds"] for r in data["execution"]]))
        fig.update_layout(title="Suite wall time by phase", yaxis_title="Wall seconds")
        execution_html = chart(fig) + table(data["execution"], ["phase", "seconds", "workers", "resumed_batches"])
    execution_note = ("<div class='notice'><b>Execution mode: throughput.</b> Measured searches ran "
                      "concurrently. Their elapsed times and deadline outcomes include CPU contention "
                      "and must not be compared as isolated algorithm performance.</div>"
                      if manifest.get("timing_mode") == "throughput" else
                      "<p>Execution mode: isolated measurements. Diagnostic/statistical parallelism, "
                      "when enabled, occurs after all measured searches.</p>")
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Guided search benchmark</title>
<style>
body{margin:0;background:#f3f5f8;color:#16243a;font:15px/1.65 system-ui,sans-serif}
main{max-width:1320px;margin:auto;padding:32px 24px}h1{font-size:36px;letter-spacing:-1px;margin:0}
h2{margin-top:34px}header{border-bottom:3px solid #127b81;padding-bottom:22px}.subtitle{color:#53647b}
.chart{background:white;margin:20px 0;border:1px solid #dde4ec;border-radius:8px;overflow:hidden}
.notice{padding:18px 22px;background:#e5f3f1;border-left:4px solid #127b81;margin:24px 0}
select{font:inherit;padding:8px 12px;border:1px solid #aebdcc;border-radius:5px;background:white}
details{background:white;border:1px solid #dde4ec;border-radius:6px;margin:14px 0;padding:12px}
summary{cursor:pointer;font-weight:650}table{border-collapse:collapse;font-size:12px;width:100%}
td,th{padding:8px 10px;text-align:left;border-bottom:1px solid #dde4ec;white-space:nowrap}
th{background:#edf2f7;position:sticky;top:0}.table-scroll{max-height:540px;overflow:auto;margin:12px 0}
pre{white-space:pre-wrap;font-size:12px}.context{display:none}.context.active{display:block}a{color:#087b82}
</style></head><body><main><header><p class="subtitle">COMPUTATIONAL INTELLIGENCE · REPRODUCIBLE EXPERIMENTS</p>
<h1>Guided search as the circle grows</h1><p>Size scaling, exploration response, convergence and search diagnostics.</p></header>
<div class="notice"><b>Reading the results.</b> Count limits have different costs in HC and GA. Equal-time runs
include setup and use only scores reached before the common deadline; overshoot and returned scores are saved.
“Time to best” means each run’s own final score, not a shared target or proof of optimality.
Targets use an exact optimum for small N and a potentially unattainable upper bound otherwise.
Coverage/diversity come from separate untimed fixed-check replays. Correlation and slopes are descriptive,
not significance tests or evidence that one exploration setting generalizes best.</div>
EXECUTION_NOTE
<label for="context">Experiment slice: </label><select id="context">OPTIONS</select>
SECTIONS
<details><summary>Problem instances, duplicate profiles and matrix construction</summary>PROBLEMS</details>
<details><summary>Configuration and environment</summary><pre>MANIFEST</pre></details>
<details><summary>Execution phases and worker counts</summary>EXECUTION_PHASES</details>
<p>Raw files: <a href="runs.csv">runs</a> · <a href="improvements.csv">improvements</a> ·
<a href="targets.csv">targets</a> · <a href="summaries.csv">summaries</a> ·
<a href="exploration_effects.csv">paired effects</a> · <a href="exploration_trends.csv">exploration trends</a> ·
<a href="diagnostics.csv">diagnostics</a> · <a href="results.json">all data</a>.</p>
<script>
const picker=document.getElementById('context');
function selectContext(){document.querySelectorAll('.context').forEach(s=>s.classList.toggle('active',s.dataset.context===picker.value));window.dispatchEvent(new Event('resize'));}
picker.addEventListener('change',selectContext);document.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',()=>window.dispatchEvent(new Event('resize'))));selectContext();
</script></main></body></html>"""
    html = html.replace("PROBLEMS", problem_html).replace("MANIFEST", escape(json.dumps(manifest, indent=2)))
    html = html.replace("EXECUTION_NOTE", execution_note).replace("EXECUTION_PHASES", execution_html)
    html = html.replace("OPTIONS", "".join(options)).replace("SECTIONS", "".join(sections))
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Rebuild an interactive report from saved suite data, without rerunning searches")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    data = json.loads((args.directory / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.directory / "configuration.json").read_text(encoding="utf-8"))
    render_report(data, manifest, args.directory / "report.html")
    print(f"Report rebuilt: {args.directory / 'report.html'}")
