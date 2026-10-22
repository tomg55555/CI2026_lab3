"""Cross-version comparisons must not mistake bound fixes for search gains."""

from copy import deepcopy
from unittest import TestCase, main

from circle_optimization.benchmark.compare import compare


def synthetic_run(*, enhanced):
    config = dict(sizes=[10], characteristics=[5], explorations=[], instances=1,
                  trials=1, repeats=2, budget=100, time_limit=10, seed=7,
                  population_size=12, generations=2, mutation_rate=0.25)
    data = dict(manifest=dict(config=config, elapsed_seconds=50 if enhanced else 100,
                              timing_mode="isolated"), rows=[], points={},
                problems=[dict(characteristics=5, children=10, instance_seed=7,
                               exact_optimum=None, upper_bound=40 if enhanced else 41)])
    for family in (["HC", "GA", "HC-I"] if enhanced else ["HC", "GA"]):
        for protocol in ("checks", "time"):
            for repeat in range(2):
                run_id = f"{family}-{protocol}-{repeat}"
                data["rows"].append(dict(protocol=protocol, characteristics=5, children=10,
                                         family=family, variant="uniform", exploration=None,
                                         instance_seed=7, trial_seed=123, repeat=repeat,
                                         run_id=run_id, score=40, evaluations=2, status="ok"))
                data["points"][(run_id,)] = [dict(score=38, evaluations=1, seconds=0.5 if enhanced else 1),
                                              dict(score=40, evaluations=2, seconds=2.5 if enhanced else 5)]
    return data


class ComparisonTests(TestCase):
    def test_rebases_old_targets_and_preserves_pairing(self):
        result = compare(synthetic_run(enhanced=False), synthetic_run(enhanced=True))
        self.assertEqual(result["ga_checked"], 2)
        self.assertEqual(result["ga_mismatches"], [])
        rows = [r for r in result["statistics"] if r["comparison"] == "old_vs_new"
                and r["metric"] == "target99_success"]
        self.assertTrue(rows)
        self.assertTrue(all(r["old_mean"] == r["new_mean"] == 1 for r in rows))
        self.assertTrue(all(r["pairs"] == 1 for r in rows))  # Two repeats, one trial.
        times = [r for r in result["statistics"] if r["metric"] == "target99_seconds"
                 and r["comparison"] == "old_vs_new"]
        self.assertTrue(all(r["median_paired_ratio"] == 0.5 for r in times))

    def test_detects_count_trace_regressions_and_rejects_contended_timing(self):
        old, new = synthetic_run(enhanced=False), synthetic_run(enhanced=True)
        new["points"][("GA-checks-0",)][0]["evaluations"] = 2
        self.assertEqual(compare(old, new)["ga_mismatches"], ["GA-checks-0"])
        contended = deepcopy(new)
        contended["manifest"]["timing_mode"] = "throughput"
        with self.assertRaises(ValueError):
            compare(old, contended)


if __name__ == "__main__":
    main()
