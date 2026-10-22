# Lab 3: A max TSP like Problem

$N$ children each have $C$ Boolean characteristics. Two children differ by their Hamming distance (the number of characteristics on which they disagree). Seat them in a circle so that the **sum of the distances between neighbours is maximal**:

$$
\max_{\pi} \sum_{k=0}^{N-1} d\left(\pi_k, \pi_{(k+1) \bmod N}\right)
$$

The assignment is solved using a **hill climber (HC)** and a **genetic algorithm (GA)**. This repository contains both, and how they were improved until they reach the proven optimum.

## (Big) Disclaimer: Use of AI

AI was used throughout the project, deliberately and massively, for three things:

- **Validating and refining ideas**: challenging and sharpening a hypothesis before building it, understanding its theory and choosing the experiment that would confirm or reject it.
- **Building and testing fast**: implementing modules, tests and benchmark runs quickly, so that every idea was turned into measured evidence before the next step and from it assume what was working and what wasn't.
- **Writing documentation**: engineering records, READMEs and the notebook narrative.

This is not a "vibe coded" solution. The ideas and the engineering, the approach, the project structure, the modules, the solutions adopted or rejected, came from me, either directly or after deep dives in numerous exploratory sessions on GA and, more broadly, CI topics, mostly conducted with AI. Each step is recorded with its motivation and measured evidence in [`HISTORY.md`](HISTORY.md) and [`project/docs/history/`](project/docs/history/PHASES.md). This is not a "one shot" prompt, it took over ~20ish ours of work.

- Models: GPT-Sol 6.1 (x-high) and Claude Opus 5.5 (x-high).
- Coding harness: Opencode, Claude Code.

## How the project evolved

It started where the lab asks: a hill climber developed during friday lab and later a GA working on orders of children, with distances read from an $N \times N$ matrix. After implementing the first working solution I wanted to check my results, why GA performed better on scale (relative small scale tho)? Where HC is "failing"? For large N how far are we from a global optimum? Those questions were the foundation of days spent trying to reach a better solution by collecting metrics and tweaking the algorithms from what I could understand from data (this is where AI really helped, running thousands of instances of the same problem with all proposed variants, analyze their result and find patterns to formulate a tweak of the og version). To tell whether a change really helped, the next step was a benchmark that gives every algorithm the same instances, seeds and time allowance. That benchmark drove the first round of improvements: it showed, for example, that the HC was restarting searches that were still improving, which led to productive restarts and an inversion-focused HC.

One question remained open: how far from the optimum were the results? Exhaustive search stops at about ten children, and the upper bounds were loose. The turning point was noticing that with $C$ characteristics there are at most $2^C$ distinct profiles, so most children are interchangeable. Rewriting the problem on the table of profile pairs gave an exact solver whose size does not depend on $N$ (I'm not claiming to have developed the solver, I wondered how solvers approached the problem and studied to possibly take some ideas to improve my GA and HC). It revealed that even the best classic versions were still 7–15% below the optimum. It also showed why: their final circles could be improved by moves they did not have, and many of their edges joined profile pairs that the LP proves sub-optimal. Something that I was suspecting but partially failed to prove earlier, at this point I had something more reasonable to try.

That diagnosis set the agenda for the rest. Each idea taken from the exact model became a separate layer on top of HC or GA, with a switch whose off position reproduces the previous version exactly. Every iteration followed the same loop: write down the idea and its expected effect, implement it as a layer, test that switching it off changes nothing, measure it against its parent with equal time, then keep it or record why not. Every version stays runnable, so any two can still be compared and any layer removed again (details per step in [`HISTORY.md`](HISTORY.md)).

## Key findings

- The lab's HC and GA stop about 15–16% below the optimum at $N=160$ (35–44% at $N=1000$); the gap was invisible until the exact solver existed.
- The exact solver works on the profile table rather than on children: one million children are solved to proven optimality in under a second.
- The biggest single gains came from the exact model's own information: LP reduced-cost guidance (9.3% → 1.4%) and greedy seeding (→ 0%).
- Did not help: iterated local search (never better than greedy restarts), and move propagation at very large $N$ (it costs more than it finds).

## Start here


| File or doc                              | What it is                                                                                                                                                 | Time to read |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| [`walkthrough.ipynb`](walkthrough.ipynb) | Runs the lab's HC and GA, the best classic version, the exact solver and each enhancement on one instance, with the same time allowance (already executed) | 10 min       |
| [`HISTORY.md`](HISTORY.md)               | The history line: each improvement, the concept behind it, and its measured effect                                                                         | 10 min       |
| [`project/`](project/README.md)          | The full work: package, tests, benchmark suite, engineering records, saved results                                                                         | optional     |

## Results

Mean gap to the proven optimum; same instances, 0.2 s per run for every algorithm (3 instance families × 3 trials each).


| Version      | What it adds                              | $N=160$, $C=5$ | $N=1000$, $C=5$ | $N=1000$, $C=7$ |
| ------------ | ----------------------------------------- | -------------: | --------------: | --------------: |
| `hc-legacy`  | the lab's HC                              |         16.40% |          35.12% |          36.48% |
| `ga`         | the lab's GA                              |         14.72% |          43.39% |          43.65% |
| `hc-i`       | productive restarts, 80% inversions       |          9.42% |          17.20% |          20.15% |
| `hc-p-rc`    | profile state + splice move + LP guidance |          1.38% |           4.55% |           5.26% |
| `hc-p-grasp` | + greedy (GRASP) restarts                 |      **0.00%** |       **0.00%** |           0.28% |
| `ga-e-seed`  | edge-preserving crossover + seeding       |      **0.00%** |           0.03% |           0.12% |
| `exact`      | MILP on the profile table                 |          0.00% |           0.00% |           0.07% |

At $N = 10{,}000$, $C = 8$ the classic versions cannot run (they would need a $10^8$-cell distance matrix); `hc-p-grasp` and `ga-e-seed` end within 0.2% of the optimum in one second, and the exact solver proves it in about 0.6 s.

## Next steps: larger $C$

All results so far are for modest $C$ ($5 \le C \le 8$), where $m \le 2^C \ll N$ and most children share a profile. The key ingredients rely on that: the exact solver's size grows with $m$, the splice move needs two children with the same profile, and the profile table only replaces the $N \times N$ matrix while $m \ll N$. As $C$ grows the picture changes. At $C = 10$ the exact solver already needs about 27 s ($m = 977$), and with $C = 20$, $N = 1000$ almost every profile is unique ($m \approx N$). With more time, the work would continue along four lines:

1. **Map the regime.** Sweep $C$ (from 5 to 100) and $N$ with the existing suite (`--characteristic-counts`), recording the duplicate fraction $1 - m/N$, the exact solver's time and every version's gap, to locate where aggregation stops paying off. For large $C$ the distances concentrate around $C/2$, so every circle scores close to $N C / 2$; gaps should then also be reported relative to a random circle, not only to the optimum.
2. **Revise the exact algorithm for $m \approx N$.** The problem then becomes a genuine maximum TSP, and a model with all $m(m-1)/2$ pair columns stops scaling.
3. **Test HC and GA in that regime.** Re-run the versions to see which layers survive: S1 needs repeated profiles and becomes inactive, S2 brings no memory gain, and S4's edge table degenerates into the children's adjacency matrix, while S3 (reduced costs) and S6 (greedy seeds) should still apply as long as their LP and construction scale.
4. **Improve where the findings point.**

## Run it

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r project/requirements-benchmark.txt notebook   # numpy, highspy, plotly, Jupyter
jupyter notebook walkthrough.ipynb
```

Every version is runnable by name from `project/`:

```sh
cd project
python -m circle_optimization list                               # all versions, their phase and layers
python -m circle_optimization run hc-legacy --children 160 --optimum
python -m circle_optimization compare hc-legacy ga hc-i hc-p-grasp ga-e-seed --children 1000
python -m unittest discover -s tests -t .                        # 92 tests
```

## Repository layout

```
README.md            this page
HISTORY.md           history line of the improvements
walkthrough.ipynb    executed walkthrough
project/
├── circle_optimization/   package: classic/ (lab HC and GA), exact/ (MILP), layered/ (enhancements),
│                          benchmark/ (experiment suite), eras.py (every version, runnable by name)
├── tests/                 unit, regression and integration tests
├── docs/                  protocol, era catalog, one engineering record per phase
├── problem.ipynb          course starter notebook
├── genetic_algorithm.ipynb, guided_tour.ipynb   original GA notebook; tour of every version
└── benchmark-results/     saved experiment outputs (not versioned)
```
