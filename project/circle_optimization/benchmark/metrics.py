"""Pure, standard-library metrics and paired exploration statistics.

Repeats are collapsed within a trial before inference. Confidence intervals
resample instance families, then trials within each sampled family. They are
omitted when only one family is available, rather than treating timing repeats
or correlated nested sizes as independent problems.
"""

from collections import defaultdict
from functools import lru_cache
from math import ceil, log, sqrt
from random import Random
from statistics import mean, median, stdev


def quality_bound(profiles, distances) -> int:
    """Upper bound, not necessarily attainable: feature and degree relaxations."""
    n, c = len(profiles), len(profiles[0])
    feature = sum(2 * min(sum(p[k] for p in profiles),
                          n - sum(p[k] for p in profiles)) for k in range(c))
    degree = sum(sum(sorted((d for j, d in enumerate(row) if j != i),
                           reverse=True)[:2]) for i, row in enumerate(distances)) // 2
    bound = min(n * c, feature, degree)
    # Each Boolean feature flips an even number of times around a circle.
    return bound - bound % 2


def first_hit(points: list[dict], target: float, axis="seconds"):
    """First time (or count, with `axis`) at which a trace reaches `target`, else None."""
    return next((p[axis] for p in points if p["score"] >= target), None)


def step_value(points, x, *, axis="seconds", scale=1):
    """Best normalized score reached by `x` on a step trace (0 before the first point)."""
    value = 0
    for point in points:
        if point[axis] > x:
            break
        value = point["score"] / scale
    return value


def anytime_auc(points, horizon: float, scale: float, *, axis="seconds") -> float:
    """Time-average normalized best-so-far score; setup before first score is 0."""
    if horizon <= 0:
        return 0.0
    area = previous = value = 0.0
    for point in points:
        x = min(float(point[axis]), horizon)
        area += max(0, x - previous) * value
        if point[axis] > horizon:
            return area / horizon
        value = point["score"] / scale
        previous = x
    return (area + (horizon - previous) * value) / horizon


def percentile(values, q):
    """Linearly interpolated percentile of `values` for q in [0, 1]."""
    values = sorted(values)
    at = (len(values) - 1) * q
    lo = int(at)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (at - lo)


@lru_cache(maxsize=64)
def _bootstrap_weights(sizes: tuple[int, ...], samples: int):
    """Reuse identical hierarchical resamples across metrics, without fewer draws.

    Sampling indices consumes the same RNG draws as sampling the old lists of
    values. Only linear weights are cached; data/estimates are never cached.
    """
    rng = Random(9017)
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    matrix = []
    for _ in range(samples):
        drawn = [rng.randrange(len(sizes)) for _ in sizes]
        weights = [0.0] * offsets[-1]
        for group in drawn:
            size = sizes[group]
            weight = 1 / (len(sizes) * size)
            for _ in range(size):
                weights[offsets[group] + rng.randrange(size)] += weight
        matrix.append(tuple(weights))
    return tuple(matrix)


def clustered_ci(rows, metric, *, samples=400):
    """95% hierarchical bootstrap interval of a metric: resample instance families, then trials."""
    clusters = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            clusters[row["instance_seed"]].append(row[metric])
    if len(clusters) < 2:
        return None, None
    groups = list(clusters.values())
    values = [value for group in groups for value in group]
    estimates = [sum(value * weight for value, weight in zip(values, weights))
                 for weights in _bootstrap_weights(tuple(map(len, groups)), samples)]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


CONDITION = ("protocol", "children", "family", "variant", "exploration")
IDENTITY = CONDITION + ("instance_seed", "trial_seed")
METRICS = (
    "normalized_score", "reference_ratio", "relative_gap", "seconds", "seconds_to_best",
    "evaluations_to_best", "auc_time", "auc_checks", "post_best_fraction", "throughput",
    "target_success", "swap_acceptance", "inversion_acceptance", "swap_gain_per_proposal",
    "swap_improvement_rate", "inversion_improvement_rate", "swap_net_gain_per_proposal",
    "inversion_net_gain_per_proposal",
    "novelty_rate", "edge_coverage", "edge_diversity", "edge_entropy", "cache_hit_rate",
    "duplicate_rate", "actual_uniform_swap_fraction", "log10_circle_coverage",
    "improvement_count", "largest_stagnation_fraction", "restart_improvements",
    "restart_exit_attempt_cap", "restart_exit_stall", "restart_exit_evaluation_budget",
    "restart_exit_time_limit",
)


def collapse_repeats(rows):
    """One row per paired trial: mean target success, median of other metrics over repeats."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in IDENTITY)].append(row)
    collapsed = []
    for group in groups.values():
        row = {key: group[0].get(key) for key in IDENTITY}
        for metric in METRICS:
            values = [r[metric] for r in group if r.get(metric) is not None]
            row[metric] = ((mean(values) if metric == "target_success" else median(values))
                           if values else None)
        collapsed.append(row)
    return collapsed


def summarize(rows):
    """Per-condition mean, median, SD, quartiles and clustered interval of every metric."""
    groups = defaultdict(list)
    for row in collapse_repeats(rows):
        groups[tuple(row.get(k) for k in CONDITION)].append(row)
    output = []
    for key, group in groups.items():
        for metric in METRICS:
            values = [r[metric] for r in group if r.get(metric) is not None]
            if not values:
                continue
            lo, hi = clustered_ci(group, metric)
            output.append(dict(zip(CONDITION, key), metric=metric, mean=mean(values),
                               median=median(values), sd=stdev(values) if len(values) > 1 else 0,
                               q25=percentile(values, 0.25), q75=percentile(values, 0.75),
                               ci_low=lo, ci_high=hi, trials=len(values),
                               instances=len({r["instance_seed"] for r in group})))
    return output


LOWER_IS_BETTER = {"relative_gap", "seconds", "seconds_to_best", "evaluations_to_best",
                   "post_best_fraction", "cache_hit_rate", "duplicate_rate"}


def exploration_effects(rows):
    """Paired differences vs uniform, the previous alpha, and the first alpha."""
    grouped = defaultdict(list)
    for row in collapse_repeats(rows):
        grouped[(row["protocol"], row["children"], row["family"])].append(row)
    output = []
    for (protocol, n, family), group in grouped.items():
        alphas = sorted({r["exploration"] for r in group if r["variant"] == "guided"})
        lookup = {(r["variant"], r["exploration"], r["instance_seed"], r["trial_seed"]): r
                  for r in group}
        for index, alpha in enumerate(alphas):
            references = [("uniform", None)]
            if index:
                references.append(("previous_alpha", alphas[index - 1]))
                if index > 1:
                    references.append(("first_alpha", alphas[0]))
            for comparison, reference in references:
                selected = [r for r in group if r["variant"] == "guided" and r["exploration"] == alpha]
                for metric in METRICS:
                    pairs = []
                    ratios = []
                    for row in selected:
                        base = lookup.get(("uniform" if comparison == "uniform" else "guided",
                                           reference, row["instance_seed"], row["trial_seed"]))
                        if base is None or row.get(metric) is None or base.get(metric) is None:
                            continue
                        pairs.append(dict(instance_seed=row["instance_seed"],
                                          delta=row[metric] - base[metric]))
                        if base[metric] > 0:
                            ratios.append(row[metric] / base[metric])
                    if not pairs:
                        continue
                    changes = [p["delta"] for p in pairs]
                    lo, hi = clustered_ci(pairs, "delta")
                    direction = -1 if metric in LOWER_IS_BETTER else 1
                    spread = stdev(changes) if len(changes) > 1 else 0
                    output.append(dict(protocol=protocol, children=n, family=family,
                                       exploration=alpha, comparison=comparison,
                                       reference_exploration=reference, metric=metric,
                                       mean_delta=mean(changes), median_delta=median(changes),
                                       ci_low=lo, ci_high=hi,
                                       mean_ratio=mean(ratios) if ratios else None,
                                       mean_percent_change=100 * (mean(ratios) - 1) if ratios else None,
                                       paired_dz=mean(changes) / spread if spread else None,
                                       increases=sum(d > 0 for d in changes),
                                       ties=sum(d == 0 for d in changes),
                                       decreases=sum(d < 0 for d in changes),
                                       favorable=sum(d * direction > 0 for d in changes),
                                       pairs=len(changes)))
    return output


def _ranks(values):
    """Average ranks (ties share their mean rank), for Spearman correlation."""
    ordered = sorted(values)
    return [mean(i for i, x in enumerate(ordered) if x == value) for value in values]


def exploration_trends(rows):
    """Within-paired-trial slope and Spearman rho across exploration settings."""
    groups = defaultdict(list)
    for row in collapse_repeats(rows):
        if row["variant"] == "guided":
            groups[(row["protocol"], row["children"], row["family"],
                    row["instance_seed"], row["trial_seed"])].append(row)
    trends = defaultdict(list)
    for (protocol, n, family, instance, trial), group in groups.items():
        for metric in METRICS:
            points = [(r["exploration"], r[metric]) for r in group if r.get(metric) is not None]
            if len(points) < 2:
                continue
            xs, ys = map(list, zip(*points))
            xmean, ymean = mean(xs), mean(ys)
            slope = sum((x - xmean) * (y - ymean) for x, y in points) / sum((x - xmean) ** 2 for x in xs)
            rx, ry = _ranks(xs), _ranks(ys)
            mx, my = mean(rx), mean(ry)
            denom = sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
            rho = sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / denom if denom else None
            trends[(protocol, n, family, metric)].append(dict(
                instance_seed=instance, trial_seed=trial, slope_per_0_1=slope * 0.1, rho=rho,
            ))
    output = []
    for (protocol, n, family, metric), group in trends.items():
        lo, hi = clustered_ci(group, "slope_per_0_1")
        rhos = [r["rho"] for r in group if r["rho"] is not None]
        output.append(dict(protocol=protocol, children=n, family=family, metric=metric,
                           mean_slope_per_0_1=mean(r["slope_per_0_1"] for r in group),
                           ci_low=lo, ci_high=hi,
                           mean_spearman=mean(rhos) if rhos else None, paired_trials=len(group)))
    return output


def target_rows(row, points, fractions):
    """Hit/miss, first-hit time and count for each target fraction of the reference."""
    output = []
    for fraction in fractions:
        threshold = ceil(fraction * row["reference_score"])
        seconds = first_hit(points, threshold)
        checks = first_hit(points, threshold, "evaluations")
        output.append(dict(run_id=row["run_id"], protocol=row["protocol"],
                           children=row["children"], characteristics=row["characteristics"],
                           family=row["family"], variant=row["variant"],
                           exploration=row["exploration"], instance_seed=row["instance_seed"],
                           trial_seed=row["trial_seed"], repeat=row["repeat"], fraction=fraction,
                           threshold=threshold, reference_kind=row["reference_kind"],
                           hit=int(seconds is not None), seconds=seconds, evaluations=checks,
                           observed_seconds=row["seconds"], observed_evaluations=row["evaluations"]))
    return output
