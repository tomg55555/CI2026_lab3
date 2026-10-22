"""Era registry, command line and suite integration: every era runs by name."""

import json
import unittest
from importlib.util import find_spec
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from circle_optimization.eras import ERAS, Problem, run_era

HISTORY = Path(__file__).resolve().parents[1] / "docs" / "history"  # independent of the working directory


class EraRegistryTests(unittest.TestCase):
    def test_registry_records_are_complete_and_linked(self):
        index = (HISTORY / "PHASES.md").read_text(encoding="utf-8")
        phases = [era.phase for era in ERAS.values()]
        self.assertEqual(phases, sorted(phases), "registry is ordered by phase")
        for era in ERAS.values():
            self.assertIn(era.family, ("HC", "GA", "exact"))
            self.assertTrue(era.title and era.summary and era.layers and era.budget_unit)
            if era.extends is not None:
                self.assertIn(era.extends, ERAS)
                self.assertLessEqual(ERAS[era.extends].phase, era.phase)
            # Phases 00-03 predate per-phase records and are summarized in the index.
            self.assertIn(f"{era.phase} — ", index, f"phase {era.phase} of {era.name} is indexed")
            if era.phase >= "04":
                self.assertTrue(list((HISTORY / "phases").glob(f"{era.phase}-*.md")), era.name)

    def test_every_available_era_returns_a_valid_circle(self):
        problem = Problem.random(40, 5, seed=2027)
        optimum = problem.optimum if ERAS["exact"].available() else None
        for era in ERAS.values():
            if not era.available():
                continue
            for limits in (dict(evaluation_budget=600), dict(time_limit=0.05)):
                result = run_era(era.name, problem, seed=3, **limits)
                self.assertEqual(problem.score(result["circle"]), result["score"], era.name)
                if optimum is not None:
                    self.assertLessEqual(result["score"], optimum, era.name)
                self.assertTrue(result["layers"])

    def test_seeded_runs_repeat_and_overrides_are_checked(self):
        problem = Problem.random(30, 4, seed=5)
        for name in ("hc", "ga") + (("hc-p",) if ERAS["hc-p"].available() else ()):
            first = run_era(name, problem, seed=7, evaluation_budget=800)
            second = run_era(name, problem, seed=7, evaluation_budget=800)
            self.assertEqual((first["score"], first["circle"]), (second["score"], second["circle"]))
        changed = run_era("hc", problem, seed=7, evaluation_budget=800, stall_limit=5)
        self.assertGreaterEqual(changed["restarts"], 2)
        with self.assertRaises(TypeError):
            run_era("hc", problem, seed=7, not_a_parameter=1)
        with self.assertRaises(KeyError):
            run_era("no-such-era", problem)
        with self.assertRaises(ValueError):
            run_era("hc", problem, time_limit=1, deadline=1)

    def test_problem_score_matches_classic_fitness(self):
        from circle_optimization import circle_fitness
        problem = Problem.random(25, 6, seed=9)
        circle = tuple(range(25))
        self.assertEqual(problem.score(circle), circle_fitness(circle, problem.distances))


class EraCommandLineTests(unittest.TestCase):
    def run_cli(self, *argv):
        from circle_optimization.__main__ import main
        output = StringIO()
        with redirect_stdout(output):
            main(list(argv))
        return output.getvalue()

    def test_list_describe_run_and_compare(self):
        listing = self.run_cli("list")
        for name in ERAS:
            self.assertIn(name, listing)
        self.assertIn("phase", self.run_cli("describe", "hc-i"))
        run = json.loads(self.run_cli("run", "hc", "--children", "12", "--time-limit", "0.02", "--json",
                                      "--set", "stall_limit=10"))
        self.assertEqual(sorted(run["circle"]), list(range(12)))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "compare.json"
            table = self.run_cli("compare", "hc", "ga", "--children", "12", "--time-limit", "0.02",
                                 "--repeats", "2", "--json", str(path))
            self.assertIn("hc", table)
            saved = json.loads(path.read_text())
            self.assertEqual([row["era"] for row in saved["rows"]], ["hc", "ga"])


@unittest.skipUnless(ERAS["hc-p"].available() and ERAS["exact"].available(), "needs numpy and highspy")
class SuiteEraTests(unittest.TestCase):
    def test_suite_measures_eras_alongside_or_without_the_classic_grid(self):
        from circle_optimization.benchmark.suite import SuiteConfig, execute_suite
        base = dict(sizes=(8, 20), explorations=(0,), instances=1, trials=1, repeats=1, budget=300,
                    time_limit=0.02, generations=3, exact_limit=8, workers=1, progress=False,
                    include_inversion_hc=False, diagnostics=True)
        data = execute_suite(SuiteConfig(eras=("hc-p", "multistart"), **base))
        rows = {(r["protocol"], r["children"], r["family"], r["variant"]): r for r in data["runs"]}
        for protocol in ("checks", "time"):
            for n in (8, 20):
                self.assertIn((protocol, n, "hc-p", "era"), rows)
        # S2 reproduces classic HC at a fixed count.
        self.assertEqual(rows["checks", 20, "hc-p", "era"]["score"], rows["checks", 20, "HC", "uniform"]["score"])
        self.assertFalse(any(d["family"] in ("hc-p", "multistart") for d in data["diagnostics"]))
        only_config = SuiteConfig(eras=("hc-p",), classic=False, **base)
        only = execute_suite(only_config)
        self.assertEqual({r["family"] for r in only["runs"]}, {"hc-p"})
        self.assertTrue(all(p["matrix_seconds"] is None for p in only["problems"]))
        if find_spec("plotly") is not None:
            from circle_optimization.benchmark.report import render_report
            with TemporaryDirectory() as directory:
                from dataclasses import asdict
                render_report(only, dict(config=asdict(only_config)), Path(directory) / "report.html")
                self.assertIn("era hc-p", (Path(directory) / "report.html").read_text())


if __name__ == "__main__":
    unittest.main()
