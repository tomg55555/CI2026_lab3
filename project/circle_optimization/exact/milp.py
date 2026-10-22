"""Exact circle optimum by a profile-aggregated MILP with lazy connectivity cuts.

    max  sum d_pq x_pq
    s.t. sum_q x_pq + 2 y_p = 2 n_p              for each profile p
         sum_{p in S, q not in S} x_pq >= 2      for each proper profile subset S
         x_pq in [0, 2 min(n_p, n_q)], y_p in [0, n_p - 1] (m >= 2), integer

x_pq counts circle edges between profiles p < q and y_p edges inside p. The
model has m(m-1)/2 + m columns and m rows whatever N is. Subset cuts are added
only for components of an integer solution, then the model is re-solved with
the best connected solution as a MIP start. Each relaxation bounds the optimum;
a connected relaxation optimum, or an incumbent equal to a bound, proves it.

Requires `highspy` (HiGHS). Run `python -m circle_optimization.exact.milp -h`.
"""

import argparse
from time import perf_counter

import numpy as np

from .profiles import (
    ProfileInstance, aggregate_profiles, components, multigraph_circle,
    multigraph_score, profile_upper_bound,
)
from .multigraph import connect, improve, multi_start


def _model(instance: ProfileInstance, *, seed: int, threads: int | None):
    """Degree rows, columns and complement-pair seed cuts (integrality set later)."""
    import highspy

    m, counts, d = instance.m, instance.counts, instance.distances
    p, q = np.triu_indices(m, 1)
    columns = len(p) + m
    h = highspy.Highs()
    for name, value in (("output_flag", False), ("mip_rel_gap", 0.0),
                        ("mip_abs_gap", 1 - 1e-6), ("random_seed", seed)):
        h.setOptionValue(name, value)
    if threads is not None:
        h.setOptionValue("threads", threads)
    degree = (2 * counts).astype(np.float64)
    h.addRows(m, degree, degree, 0, np.zeros(m, np.int32), np.zeros(0, np.int32), np.zeros(0))
    upper = np.concatenate([2 * np.minimum(counts[p], counts[q]), counts - (m >= 2)])
    h.addCols(columns, np.concatenate([d[p, q], np.zeros(m)]).astype(np.float64),
              np.zeros(columns), upper.astype(np.float64), 2 * len(p) + m,
              np.concatenate([np.arange(0, 2 * len(p), 2), 2 * len(p) + np.arange(m)]).astype(np.int32),
              np.concatenate([np.column_stack([p, q]).ravel(), np.arange(m)]).astype(np.int32),
              np.concatenate([np.ones(2 * len(p)), np.full(m, 2.0)]))
    h.changeObjectiveSense(highspy.ObjSense.kMaximize)
    # Hamming optima favour complementary pairs (distance C): forbid isolated 2-cycles.
    complements = np.argwhere(np.triu(d == instance.characteristics, 1)) if m > 2 else []
    for pair in complements:
        _add_cut(h, np.isin(np.arange(m), pair), p, q)
    return h, p, q, len(complements), upper.astype(np.float64)


def _add_cut(h, inside: np.ndarray, p: np.ndarray, q: np.ndarray) -> None:
    """At least two circle edges leave the profile subset `inside`."""
    import highspy

    crossing = np.flatnonzero(inside[p] != inside[q]).astype(np.int32)
    h.addRow(2.0, highspy.kHighsInf, len(crossing), crossing, np.ones(len(crossing)))


def _start(h, x: np.ndarray, p: np.ndarray, q: np.ndarray) -> None:
    """Pass a connected multiplicity table to HiGHS as a MIP start."""
    values = np.concatenate([x[p, q], x.diagonal()]).astype(np.float64)
    h.setSolution(len(values), np.arange(len(values), dtype=np.int32), values)


def _solution(h, instance: ProfileInstance, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Round the MILP columns to a multiplicity matrix and recheck degrees exactly."""
    values = np.rint(np.asarray(h.getSolution().col_value)).astype(np.int64)
    x = np.zeros((instance.m, instance.m), dtype=np.int64)
    x[p, q] = x[q, p] = values[:len(p)]
    x[np.diag_indices(instance.m)] = values[len(p):]
    if ((x.sum(axis=1) + x.diagonal()) != 2 * instance.counts).any():
        raise RuntimeError("HiGHS returned a solution violating the degree equations")
    return x


def _separate(h, support: np.ndarray, p: np.ndarray, q: np.ndarray) -> int:
    """Cut off a disconnected support: one cut per component (one for two)."""
    groups = components(support)
    if len(groups) == 1:
        return 0
    for group in groups[:1] if len(groups) == 2 else groups:
        _add_cut(h, np.isin(np.arange(len(support)), group), p, q)
    return 1 if len(groups) == 2 else len(groups)


def _set_integrality(h, columns: int, integer: bool) -> None:
    """Mark every column integer or continuous."""
    import highspy

    kind = highspy.HighsVarType.kInteger if integer else highspy.HighsVarType.kContinuous
    h.changeColsIntegrality(columns, np.arange(columns, dtype=np.int32),
                            np.full(columns, int(kind), dtype=np.uint8))


def _remaining(h, deadline: float | None) -> bool:
    """Give HiGHS the time left (its limit applies per run); False if none."""
    if deadline is None:
        return True
    remaining = deadline - perf_counter()
    if remaining > 0:
        h.setOptionValue("time_limit", remaining)
    return remaining > 0


def _even_floor(value: float) -> int:
    """Largest even integer <= value: every feature flips an even number of times
    around a circle (each cut of an even-degree multigraph is even)."""
    bound = int(np.floor(value + 1e-6))
    return bound - bound % 2


def _fix_by_reduced_cost(values, reduced_costs, lower, upper, slack):
    """Column bounds every solution within `slack` of the LP optimum satisfies.

    For a maximization LP optimum, c.x <= LP - sum_j |rc_j| * |x_j - x*_j|
    for every x satisfying its rows (cut duals only lower it further), so
    |x_j - x*_j| <= slack / |rc_j|. Nonbasic x*_j sit at integer bounds.
    """
    reach = np.floor(slack / np.maximum(np.abs(reduced_costs), 1e-12) + 1e-9)
    anchor = np.rint(values)
    lower = np.where(reduced_costs > 1e-7, np.maximum(lower, anchor - reach), lower)
    upper = np.where(reduced_costs < -1e-7, np.minimum(upper, anchor + reach), upper)
    return lower, upper


def lp_relaxation(instance: ProfileInstance, *, seed: int = 0, threads: int | None = None,
                  max_rounds: int = 1000) -> dict:
    """Connected LP relaxation: bound, values and reduced costs per profile pair.

    Runs the LP phase of `solve_profile_milp` on its own: the continuous model
    is re-solved while its fractional support is disconnected, one cut per
    component (one for two). Returns m x m symmetric float arrays `values`
    (LP edge counts x*; diagonal: loops y*) and `reduced_costs` (column duals;
    diagonal: loop columns), plus `bound` (LP objective), `upper` (column upper
    bounds as m x m), `rounds`, `cuts` and `connected` (False only if
    `max_rounds` ran out; the bound is valid either way).

    For any circle with edge counts x, score <= bound + sum_j rc_j (x_j - x*_j):
    equality-row duals cancel and cut duals only lower the right-hand side.
    At the optimum rc_j < 0 only at x*_j = 0 and rc_j > 0 only at the upper
    bound. Used by layer S3 (`layered/reduced_costs.py`, phase 11).
    """
    m = instance.m
    p, q = np.triu_indices(m, 1)
    if m == 1:  # Every edge is a loop: score 0, nothing to price.
        zero = np.zeros((1, 1))
        loops = np.full((1, 1), float(instance.counts[0]))
        return dict(bound=0.0, values=loops, reduced_costs=zero, upper=loops.copy(),
                    rounds=0, cuts=0, connected=True)
    import highspy

    h, p, q, cuts, column_upper = _model(instance, seed=seed, threads=threads)
    _set_integrality(h, len(p) + m, False)
    rounds, connected = 0, False
    while rounds < max_rounds:
        h.run()
        rounds += 1
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"HiGHS LP stopped with {h.modelStatusToString(h.getModelStatus())}")
        solution = h.getSolution()
        support = np.zeros((m, m), dtype=bool)
        support[p, q] = support[q, p] = np.asarray(solution.col_value)[:len(p)] > 1e-9
        added = _separate(h, support, p, q)
        cuts += added
        if not added:
            connected = True
            break

    def square(columns: np.ndarray) -> np.ndarray:
        """Column vector (pair columns, then loops) as a symmetric m x m matrix."""
        matrix = np.zeros((m, m))
        matrix[p, q] = matrix[q, p] = columns[:len(p)]
        matrix[np.diag_indices(m)] = columns[len(p):]
        return matrix

    return dict(bound=float(h.getInfo().objective_function_value),
                values=square(np.asarray(solution.col_value)),
                reduced_costs=square(np.asarray(solution.col_dual)),
                upper=square(column_upper), rounds=rounds, cuts=cuts, connected=connected)


def solve_profile_milp(
    instance: ProfileInstance, *, incumbent: np.ndarray | None = None,
    upper_bound: int | None = None, deadline: float | None = None,
    max_rounds: int = 1000, seed: int = 0, threads: int | None = None,
) -> dict:
    """Close the gap between a connected incumbent and the optimum.

    `incumbent` is a connected multiplicity matrix (warm start). `upper_bound`
    is any proven bound (default: `profile_upper_bound`). `deadline` is an
    absolute perf_counter time; on expiry the best incumbent and best proven
    bound are returned with status "feasible".

    1. LP phase: the continuous relaxation re-solves from its basis while
       components of its fractional support receive cuts. Its value bounds
       the optimum and is usually tight; it may already prove the incumbent.
    2. Target phase: aim for target = floor(LP). Reduced-cost fixing keeps
       only columns compatible with reaching it (typically a few percent).
       If the reduced MIP cannot reach the target, the optimum is below it:
       the bound drops to the next even score and the original bounds return.
    3. MIP rounds: integer optima are often disconnected ties. Each one is
       cut off and also repaired (merge + 2-exchange) into an incumbent; the
       search ends once an incumbent meets the bound.
    """
    import highspy

    statuses = highspy.HighsModelStatus
    upper_bound = profile_upper_bound(instance) if upper_bound is None else upper_bound
    best = incumbent
    best_score = -1 if incumbent is None else multigraph_score(incumbent, instance.distances)
    result = dict(rounds=0, lp_rounds=0, cuts=0, nodes=0, lp_bound=None, fixed_columns=0)
    if best_score >= upper_bound:
        return dict(result, multiplicity=best, score=best_score, upper_bound=upper_bound,
                    status="optimal", proof="bound")
    h, p, q, result["cuts"], column_upper = _model(instance, seed=seed, threads=threads)
    columns, m = len(p) + instance.m, instance.m
    column_lower, every = np.zeros(columns), np.arange(columns, dtype=np.int32)
    _set_integrality(h, columns, False)
    lp_solution = None
    while _remaining(h, deadline):
        h.run()
        result["lp_rounds"] += 1
        if h.getModelStatus() != statuses.kOptimal:
            break
        lp_solution = h.getSolution()
        support = np.zeros((m, m), dtype=bool)
        support[p, q] = support[q, p] = np.asarray(lp_solution.col_value)[:len(p)] > 1e-9
        added = _separate(h, support, p, q)
        result["cuts"] += added
        if not added:
            result["lp_bound"] = h.getInfo().objective_function_value
            upper_bound = min(upper_bound, _even_floor(result["lp_bound"]))
            break
    if best_score >= upper_bound:
        return dict(result, multiplicity=best, score=best_score, upper_bound=upper_bound,
                    status="optimal", proof="lp")
    _set_integrality(h, columns, True)
    target = None
    if result["lp_bound"] is not None:
        target = upper_bound
        lower, upper = _fix_by_reduced_cost(
            np.asarray(lp_solution.col_value), np.asarray(lp_solution.col_dual),
            column_lower, column_upper, result["lp_bound"] - target + 1e-6)
        result["fixed_columns"] = int((lower == upper).sum())
        h.changeColsBounds(columns, every, lower, upper)
    status, proof = "feasible", None
    while result["rounds"] < max_rounds and _remaining(h, deadline):
        if best is not None and target is None:
            _start(h, best, p, q)  # The target model may exclude the incumbent.
        h.run()
        result["rounds"] += 1
        model_status, info = h.getModelStatus(), h.getInfo()
        result["nodes"] += int(info.mip_node_count)
        if model_status == statuses.kOptimal:
            relaxation = _even_floor(info.objective_function_value)
        elif model_status == statuses.kTimeLimit:
            # A run stopped before its root LP has no finite dual bound yet.
            relaxation = (_even_floor(info.mip_dual_bound) if np.isfinite(info.mip_dual_bound)
                          else upper_bound)
        elif model_status == statuses.kInfeasible and target is not None:
            relaxation = -1
        else:
            raise RuntimeError(f"HiGHS stopped with {h.modelStatusToString(model_status)}")
        if target is not None:
            # Fixed columns exclude only solutions below target.
            relaxation = max(relaxation, target - 2)
            if relaxation < target:
                target = None
                h.changeColsBounds(columns, every, column_lower, column_upper)
        upper_bound = min(upper_bound, relaxation)
        if info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible \
                and model_status != statuses.kInfeasible:
            x = _solution(h, instance, p, q)
            added = _separate(h, x, p, q)
            result["cuts"] += added
            if added:
                connect(x, instance.distances)
                improve(x, instance.distances, deadline=deadline)
            score = multigraph_score(x, instance.distances)
            if score > best_score:
                best, best_score = x, score
        if best_score >= upper_bound:
            status, proof = "optimal", "milp"
            break
        if model_status == statuses.kTimeLimit:
            break
    return dict(result, multiplicity=best, score=best_score, upper_bound=upper_bound,
                status=status, proof=proof)


def solve_circle(
    profiles, *, time_limit: float | None = None, starts: int = 4, seed: int = 0,
    use_milp: bool = True, threads: int | None = None,
    on_improvement=None,
) -> dict:
    """Best circle for Boolean child profiles: multi-start heuristic, then MILP.

    `profiles` is an N x C Boolean table (sequence of sequences or array);
    child i is row i. The heuristic yields a connected incumbent; if it does not
    already meet the combinatorial bound, the MILP proves or improves it.
    `use_milp=False` returns the heuristic result with its bound and gap; more
    `starts` then help, whereas the MILP rarely gains from more than a few.
    `time_limit` (seconds, from the call) bounds the heuristic and the MILP;
    on expiry the best circle so far is returned with status "feasible".
    `on_improvement(score, step)` reports each improving heuristic start
    (step = starts completed) and a better MILP result (step = starts + 1).

    Returns `circle` (canonical genome, child 0 first), `score`, `upper_bound`,
    `gap`, `status` ("optimal" only with a proof), `proof` ("bound", "lp" or
    "milp"), `profiles` (m), `heuristic_score`, `lp_bound`, `lp_rounds`, MIP
    `rounds`/`cuts`/`nodes`, the multiplicity matrix, and per-stage `seconds`.
    """
    started = perf_counter()
    deadline = None if time_limit is None else started + time_limit
    instance = aggregate_profiles(profiles)
    aggregated = perf_counter()
    bound = profile_upper_bound(instance)
    heuristic = multi_start(instance, starts=starts, seed=seed, deadline=deadline, target=bound,
                            on_improvement=on_improvement)
    searched = perf_counter()
    result = dict(multiplicity=heuristic["multiplicity"], score=heuristic["score"], upper_bound=bound,
                  status="optimal" if heuristic["score"] >= bound else "feasible",
                  proof="bound" if heuristic["score"] >= bound else None,
                  rounds=0, lp_rounds=0, cuts=0, nodes=0, lp_bound=None, fixed_columns=0)
    if use_milp and result["status"] != "optimal":
        result = solve_profile_milp(instance, incumbent=heuristic["multiplicity"], upper_bound=bound,
                                    deadline=deadline, seed=seed, threads=threads)
        if on_improvement is not None and result["score"] > heuristic["score"]:
            on_improvement(result["score"], heuristic["starts"] + 1)
    solved = perf_counter()
    circle = multigraph_circle(result["multiplicity"], instance)
    finished = perf_counter()
    return dict(result, circle=circle, gap=result["upper_bound"] - result["score"],
                children=instance.n, characteristics=instance.characteristics, profiles=instance.m,
                combinatorial_bound=bound, heuristic_score=heuristic["score"],
                heuristic_starts=heuristic["starts"],
                seconds=dict(aggregate=aggregated - started, heuristic=searched - aggregated,
                             milp=solved - searched, reconstruct=finished - solved, total=finished - started))


def main() -> None:
    """Command line: solve a random instance and print the result summary."""
    parser = argparse.ArgumentParser(description="Solve a random instance exactly via profile aggregation.")
    parser.add_argument("--children", type=int, default=100_000)
    parser.add_argument("--characteristics", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--heuristic-only", action="store_true")
    args = parser.parse_args()
    if args.children < 1 or args.characteristics < 1:
        parser.error("Use at least one child and one characteristic")
    profiles = np.random.default_rng(args.seed).integers(0, 2, (args.children, args.characteristics), dtype=bool)
    result = solve_circle(profiles, time_limit=args.time_limit, starts=args.starts,
                          seed=args.seed, use_milp=not args.heuristic_only)
    print(f"N={result['children']} C={result['characteristics']} profiles={result['profiles']}")
    print(f"score={result['score']} upper_bound={result['upper_bound']} gap={result['gap']} "
          f"status={result['status']} proof={result['proof']}")
    print(f"heuristic={result['heuristic_score']} starts={result['heuristic_starts']} "
          f"lp_rounds={result['lp_rounds']} mip_rounds={result['rounds']} "
          f"cuts={result['cuts']} nodes={result['nodes']}")
    print("seconds: " + ", ".join(f"{k}={v:.4f}" for k, v in result["seconds"].items()))


if __name__ == "__main__":
    main()
