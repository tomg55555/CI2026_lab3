# Size scaling and exploration experiments

`benchmark_guided_search.py --suite` compares uniform HC, guided HC/GA, and inversion-focused HC across nested child sets, exploration probabilities, and equal-check/equal-time protocols. Commands without suite options retain the original single-size console experiment.

## Completed full comparison: where to look

The generated-artifact cleanup retained the historical baseline and the enhanced full run. Smaller smoke/validation runs were removed; their evidence remains in the implementation history.

| Resource | Location |
| --- | --- |
| Historical report | [`benchmark-results/full/report.html`](../benchmark-results/full/report.html) |
| Enhanced report | [`benchmark-results/full-phase04/report.html`](../benchmark-results/full-phase04/report.html) |
| Old-versus-new plots | [`benchmark-results/full-phase04/comparison.html`](../benchmark-results/full-phase04/comparison.html) |
| Phase-06 faster-suite run and its phase-04 comparison | [`benchmark-results/full-phase06/report.html`](../benchmark-results/full-phase06/report.html), [`comparison.html`](../benchmark-results/full-phase06/comparison.html), [phase 06](history/phases/06-faster-suite.md) |
| Detailed findings and exact settings | [Phase 05 full comparison](history/phases/05-full-comparison.md) |
| Exact-reference grid (phase 07) | [`benchmark-results/full-phase07/report.html`](../benchmark-results/full-phase07/report.html), [phase 07](history/phases/07-profile-milp.md) |
| Every era, equal time (phase 14) | [`benchmark-results/eras-phase14/report.html`](../benchmark-results/eras-phase14/report.html), [`eras-phase14-large`](../benchmark-results/eras-phase14-large/report.html), [ERAS.md](ERAS.md#results-snapshot) |
| Historical state, phases and decisions | [History index](history/README.md) |

Both full runs use $C=5$, sizes 10/20/40/80/160, exploration 0/0.2/0.5/0.8/1, three instance families, five trials, two repeats, a 20,000 count cap and a 0.2-second time allowance. The enhanced run used 10 workers for diagnostics and statistics, with measured searches isolated.

### Measured execution improvement

| Metric | Historical run | Enhanced run |
| --- | ---: | ---: |
| Timed searches | 3,600 | 3,900 |
| Diagnostic replays | 900 | 975 |
| Execution/analysis before export | 3,165.434 s | 1,142.363 s |
| Total including export/report | Not recorded separately | 1,152.487 s |

The comparable pre-export duration is **$2.77\times$ faster / 63.9% shorter**, despite 8.3% more searches from the added HC-I series. This is an observed full-grid result, not a guaranteed duration or speed ratio for another machine or setting. All 3,900 results were valid, and every diagnostic replay matched its count run.

### Algorithm findings

Mean final scores under the same isolated **200 ms** allowance:

| Children | Old uniform HC | Productive HC | HC-I | Old GA | Optimized GA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40 | 165.00 | 164.80 | 166.67 | 163.47 | 166.53 |
| 80 | 318.73 | 326.00 | 329.07 | 311.20 | 325.60 |
| 160 | 614.00 | 656.27 | 666.53 | 572.80 | 614.47 |

- **Productive restarts:** at $N=160$ and the count cap, uniform HC improves from 608.80 to 645.47; mean restarts drop from 20.27 to 10.40. All 15 paired trials improve. Benefits at small sizes are limited or inconclusive.
- **Inversion focus:** at $N=160$ and 200 ms, HC-I gains another 10.27 points over productive uniform HC, winning all 15 paired trials. At $N=40$, HC-I and optimized GA are effectively tied in this sample.
- **GA optimization:** all **900 matched fixed-count GA executions** preserve final scores, distinct-evaluation counts and strict-improvement score/count traces. Uniform GA runs approximately $2.0$–$2.34\times$ faster across sizes. At $N=160$, its count-cap runtime falls from 2.252 s to 0.980 s with the same score 684.93.
- **Quality versus latency:** GA still exceeds HC-I's count-cap score (684.93 versus 654.40 at $N=160$), but uses much more elapsed time. HC-I is the strongest tested low-latency large-instance option; there is no universal winner.
- **Guidance:** after fixing depth, weighted swaps have little robust equal-time benefit. At $N=160$, the best selected guided-HC mean is 656.67 versus uniform 656.27, while HC-I reaches 666.53. Exploration tuning is a smaller lever than depth and move composition in this grid.

These are exploratory results from three nested instance families. $C=5$ permits only 32 profiles, so $N=160$ has 80% duplicate profiles. Validate on independent families and $C=100$ before generalizing the ranking or selecting an exploration value. The phase-05 record contains paired intervals and interpretation details.

## Current algorithms and execution controls

Implementation history starts at [`docs/history/README.md`](history/README.md). The current algorithms differ from historical phase-03 results: HC now defaults to productive patience rather than a hard local cap, GA uses semantics-preserving fast paths, and the suite includes inversion-focused HC (80% inversions / 20% swaps). Use `--hc-restart-policy legacy --no-inversion-hc` for the old HC policy and variant grid; GA's fixed-count search decisions are regression-checked against pre-optimization fixtures, but its elapsed times are intentionally different.

### Launch a fresh full experiment

Use a new output name: `full-phase04` and `full-phase06` are completed retained runs.

```sh
.venv/bin/python benchmark_guided_search.py --suite \
  --sizes 10 20 40 80 160 \
  --explorations 0 0.2 0.5 0.8 1 \
  --instances 3 --trials 5 --repeats 2 \
  --protocol both --budget 20000 --time-limit 0.2 \
  --workers auto --timing-mode isolated \
  --output benchmark-results/full-next
```

Since phase 07 every problem's reference is its **exact optimum**, computed by the profile-aggregated MILP (`circle_optimization/exact/milp.py`) for every $N$ and cross-checked against enumeration up to `--exact-limit`. Gaps, reference ratios and targets above $N=9$ therefore measure distance to the true optimum, not to a possibly unattainable bound. `--reference bound` restores the phase-06 reference; `--reference-time-limit 60` caps the MILP per instance, after which its proven upper bound is used instead (`reference_kind=upper_bound`). The MILP runs during preparation, outside every timed region. See [phase 07](history/phases/07-profile-milp.md).

### Measuring eras (phase 08)

Any registered era (`python -m circle_optimization list`, [ERAS.md](ERAS.md)) can join the grid as an extra family with `--eras NAME ...`; its rows have `family=<era name>`, `variant=era`, `exploration` empty. `--no-classic` drops the classic HC/GA/HC-I exploration grid, so an era-only suite never builds the $N \times N$ matrix and can run at large $N$ (the reference then uses the identical Hamming bound computed on the profile aggregation, plus the MILP optimum):

```sh
.venv/bin/python benchmark_guided_search.py --suite --no-classic \
  --eras hc hc-i hc-p hc-p-ils ga ga-m exact \
  --sizes 160 1000 --protocol time --time-limit 0.2 --output benchmark-results/eras
```

In the checks protocol each era spends `--budget` in its own unit (`python -m circle_optimization describe NAME`); prefer the time protocol for cross-era rankings. Eras have no fixed-check diagnostic replay; their diagnostic columns stay empty. Era runs go through `eras.run_era`, which validates every returned circle against the profiles.

Since phase 06 the suite skips work whose outcome is already known (see [phase 06](history/phases/06-faster-suite.md)):

- **Proven-bound stop** (`--stop-at-bound`, default): a search stops as soon as it reaches its problem's reference, an exact optimum or a proven upper bound. Nothing can beat that score, so scores and improvement traces are unchanged.
- **One checks-protocol repeat** (`--check-repeats 1`, default): fixed-count repeats always reproduce the same circle, score and count; another repeat would only time it again. `--repeats` still sets the time-protocol repeats.
- **Copied exploration $= 1$ rows** (`--alias-full-exploration`, default): in the checks protocol, guided exploration $= 1$ uses exactly the uniform control's random draws, so its row, trace, targets and diagnostics are copied from that control and marked with `aliased_from`. Time-protocol rows remain measured.

For phase-04 semantics, add `--check-repeats 2 --no-stop-at-bound --no-alias-full-exploration` (with `--repeats 2`, as in the retained runs).

- **Isolated (default):** measured runs execute serially, without an active diagnostic/statistical pool. All measurements finish first; process workers then run independent diagnostic batches and statistical batches. GA/metric optimizations speed this up without changing count trajectories or reducing instance/trial/repeat counts, diagnostics, or the 400 bootstrap resamples.
- **Throughput:** `--timing-mode throughput` parallelizes measured batches too. Fixed-count outcomes remain reproducible; wall times and deadline-limited outcomes include CPU contention. Reports/CSV/manifest label that distinction. Use this for sweep throughput or a deliberately concurrent-workload study, not as interchangeable isolated latency results.
- `--workers auto` leaves one available CPU and caps the default at 8 processes to limit diagnostic-memory pressure; use an explicit positive count to override. Python processes use the macOS-compatible spawn context, not CPU-bound threads.
- Each paired-trial batch reports its own percentage plus overall phase percentage, elapsed time, and approximate ETA. `--no-progress` disables output; `--progress-interval` controls intermediate-update frequency.
- Phase timings and worker counts appear in `execution.csv`, the manifest, and the HTML report, so a remaining bottleneck can be located instead of guessed. CSV/JSON export and HTML chart generation also report progress; the final manifest separately records export, report, and total invocation seconds.

### Resume rather than lose completed work

Completed measurement, diagnostic, and statistics batches are atomically saved under `<output>/.checkpoints/`. Repeat the same command with `--resume` to reuse completed batches after an interruption. Keep the same workers/timing settings and algorithm configuration; verbosity may change. Config and Python-source fingerprints prevent mixing incompatible results or implementations. Python/platform compatibility is checked too. `--resume` requires the original explicit `--output`; it restores completed batches rather than creating extra independent observations. Phase times on a resumed invocation describe loading and remaining work, while `runs.csv` retains the original measured runtimes.

An existing completed output is not silently overwritten. Use a new output name for a new experiment. Historical `benchmark-results/full` can still be viewed or re-rendered with the standalone report command; it is not a compatible source of new phase-04 checkpoints.

### Search controls and interpretation

`--hc-restart-policy productive` resets unsuccessful-proposal patience on every accepted local move and does not enforce the old total local-attempt cap. `--hc-stall-limit` defaults to 200. Global budget/deadline/restart caps remain. `--hc-restart-policy legacy` additionally enforces `--hc-attempts-per-restart` (default 1,000). Per-restart exit reasons distinguish stalls from forced caps.

`--inversion-hc` / `--no-inversion-hc` controls the additional HC-I variant. `--inversion-rate 0.8` keeps 20% swaps; rates must be in $[0.5, 1)$. At 0.5 it reproduces the uniform HC mix. Both variants coexist; the original uniform mix still proposes half swaps and half inversions.

Hamming reference bounds are now rounded down to even scores, since every Boolean feature flips an even number of times around a circle. This avoids impossible odd targets that appeared in historical reports. The bound may still be unattainable for other reasons and is not relabeled as an exact optimum.

Cheaper-guidance and future productive-stopping work is planned in [`GUIDANCE_AND_RESTART_PLAN.md`](history/GUIDANCE_AND_RESTART_PLAN.md).

---

## Start here

The searches and CSV/JSON exports use the standard library. Interactive HTML reports use the optional Plotly dependency:

```sh
.venv/bin/python -m pip install -r requirements-benchmark.txt
.venv/bin/python benchmark_guided_search.py --quick --output benchmark-results/quick
```

Open `benchmark-results/quick/report.html` in a browser. The plots include their JavaScript and work offline. They support zoom, hover values, legend toggles, metric dropdowns, and a characteristic-count/protocol selector. Data links refer to adjacent CSV files; retain the directory when sharing those links.

To regenerate figures from saved results without repeating searches:

```sh
.venv/bin/python -m circle_optimization.benchmark.report benchmark-results/quick
```

`--quick` uses sizes 8/16/40, exploration 0/0.5/1, at most one instance, two trials, one repeat, 1,000 checks, and a default time allowance of 0.03 s. Explicit size/exploration/time arguments override those defaults. It validates the whole pipeline; it is not enough evidence to recommend a particular algorithm.

A more useful paired comparison:

```sh
.venv/bin/python benchmark_guided_search.py --suite \
  --sizes 10 20 40 80 160 \
  --explorations 0 0.2 0.5 0.8 1 \
  --instances 5 --trials 10 --repeats 3 \
  --protocol both --budget 10000 --time-limit 0.1 \
  --output benchmark-results/scaling
```

The number of runs grows multiplicatively. The runner prints the planned measured-search count before starting. Warm-ups, exact reference calculations, statistical summaries, rendering, and optional diagnostic replays add work. Begin with a quick run or fewer sizes/trials before a large sweep.

To calculate the workload, let $S$ be the number of sizes, $C_s$ the number of characteristic-count settings, $I$ instances, $T$ trials, $R$ time-protocol repeats, $R_c$ checks-protocol repeats, $E$ exploration values, and $F$ the number of HC/GA families (2 normally, 1 with `--no-ga`). The variant count is $V = F(1+E) + 1$ with HC-I enabled, or $V = F(1+E)$ with `--no-inversion-hc`. Let $A = F$ when exploration 1 is in the grid and copying is enabled, otherwise $A = 0$:

$$
\begin{aligned}
\text{timed searches} &= S \cdot C_s \cdot I \cdot T \cdot \bigl(R V + R_c (V - A)\bigr) \\
\text{diagnostic replays} &= S \cdot C_s \cdot I \cdot T \cdot (V - A)
\end{aligned}
$$

Drop the $R$ term for a checks-only suite and the $R_c$ term (and the replays) for a time-only suite. Diagnostics run once per paired trial/variant, not once per timing repeat. The phase-06 grid has $V=13$, $A=2$, $R=2$, $R_c=1$: 2,775 timed searches, 150 copied rows and 825 replays (phase 04: 3,900 and 975).

There is a time-budget floor: each time-protocol execution lasts the full allowance unless it reaches the proven reference first. That happens only for small $N$, where the bounds are often tight. More diagnostic workers cannot shorten that measured stage. Throughput mode changes the timing conditions rather than providing an interchangeable isolated-latency speedup.

Useful controls:

| Option | Meaning |
| --- | --- |
| `--sizes ...` | Child counts; sorted and deduplicated |
| `--characteristics 5` | Fixed characteristic count within a sweep |
| `--characteristic-counts 5 100` | Separate suites, never pooled statistically |
| `--explorations ...` | Uniform fraction of the guided swap branch |
| `--instances 3 --trials 10 --repeats 3` | Independent instance families, search seeds per family, and repeated time-protocol executions per trial |
| `--check-repeats 1` | Repeated checks-protocol executions per trial; they only re-time an identical outcome |
| `--stop-at-bound` / `--no-stop-at-bound` | Stop when the proven reference score is reached; enabled by default |
| `--alias-full-exploration` / `--no-alias-full-exploration` | Copy checks-protocol exploration $= 1$ from the uniform control, or measure it |
| `--seed 2027` | Reproducible instance/search seeds |
| `--protocol checks/time/both` | Which comparisons to run; suite default both |
| `--budget 50000` | Per-execution count cap in the checks protocol |
| `--time-limit 0.1` | Seconds per execution in the time protocol |
| `--generations 600` | GA cap in check-budget runs |
| `--max-restarts 200` | HC cap in check-budget runs |
| `--population-size 50` | GA population, capped by the number of unique circles |
| `--mutation-rate 0.25` | GA mutation probability, separate from exploration |
| `--target-fractions 0.9 0.95 0.99` | Shared quality targets relative to each instance's reference |
| `--exact-limit 9` | Enumerate an exact reference through this size; 0 disables, maximum 10 |
| `--reference milp/bound` | Exact MILP optimum for every $N$ (default; needs `highspy`), or the phase-06 enumeration/upper-bound reference |
| `--reference-time-limit 60` | MILP seconds per instance; on expiry its proven upper bound is the reference |
| `--eras NAME ...` | Registered eras measured as extra families (`variant=era`) |
| `--classic` / `--no-classic` | Include or drop the classic HC/GA/HC-I exploration grid |
| `--workers auto` | Process count; auto leaves one CPU available and caps at 8; an explicit count overrides |
| `--timing-mode isolated/throughput` | Isolate measurements, or also parallelize measured batches |
| `--progress` / `--no-progress` | Enable/disable batch and phase progress; enabled by default |
| `--progress-interval 0.5` | Minimum interval between intermediate progress updates, in seconds |
| `--resume --output ...` | Restore compatible completed batches from the same experiment |
| `--hc-restart-policy productive/legacy` | Consecutive-failure patience, or the historical additional hard cap |
| `--hc-stall-limit 200` | Consecutive failed HC proposals before restarting; improvements reset patience |
| `--hc-attempts-per-restart 1000` | Total local-attempt cap, enforced only by legacy restart policy |
| `--inversion-hc` / `--no-inversion-hc` | Include/omit HC-I; enabled by default |
| `--inversion-rate 0.8` | HC-I inversion probability in $[0.5, 1)$, retaining a nonzero swap fraction |
| `--no-diagnostics` | Skip expensive fixed-check diagnostic replays |
| `--no-ga` | HC only |
| `--no-report` | Export data without installing/importing Plotly |

Specifying sizes, exploration sweeps, protocol, time limit, characteristic-count sweeps, an output directory, or `--quick` also activates suite mode.

## Compare saved implementations without rerunning searches

```sh
.venv/bin/python compare_benchmark_runs.py \
  benchmark-results/full benchmark-results/full-phase04 --quiet
```

This produces `comparison.html`, `comparison.csv`, and `comparison.json` in the new run directory. It pairs matching instance/trial seeds, collapses timing repeats, computes clustered intervals, and compares HC-I with the new uniform controls. It verifies fixed-count GA scores/evaluation counts/improvement traces and rejects a direct comparison against contention-sensitive new measurements.

Old target outcomes are recomputed from improvement traces using the new run's common references. This matters because rounding Hamming bounds down to even scores fixes historical unattainable odd targets: $N=10$ scores 36/40/36 are unchanged, even though some target success rates change. Target-hit times are conditional on successful attainment and must be read alongside success rates.

The required raw data are `configuration.json`, `runs.csv`, `improvements.csv`, and `problems.csv`; `execution.csv`, when present, adds phase timing information. Generated-artifact cleanup should retain these files for any historical run that needs to remain comparable, not just its HTML report.

## What stays the same as $N$ grows?

Each instance family generates `max(sizes)` profiles once. The $N$-child instance uses the first $N$ profiles. Existing children retain their identities and characteristics, and smaller matrices match the corresponding larger prefixes. Every algorithm receives the same matrix at each $N$. Instance/trial seeds are stable when the exploration or size grid changes. Uniform baselines are run once per condition, not redundantly for each exploration value.

Parallel completion order never changes seeds or output ordering. Matrices are cached per worker outside measured regions. Bootstrap index/weight resamples are cached and reused across metrics; confidence levels and sample counts are unchanged. The cache stores sampling weights, not estimates or observations.

Initialization remains algorithm-specific, as requested: HC draws one starting circle, while GA draws a population. Equal numeric seeds provide reproducibility but not identical states, random decisions, or restart schedules. Guided/uniform GA variants begin with the same population in check-budget runs; tiny time limits can truncate initialization differently. Exploration $= 1$ reproduces baseline seeded behavior at a fixed count cap, so the checks protocol copies it from the uniform control by default; wrapper overhead and time-limited stopping can still give different elapsed-time outcomes, so time-protocol exploration $= 1$ is measured.

Only 32 Boolean profiles exist when $C=5$. Increasing $N$ beyond 32 necessarily adds duplicate profiles. Children remain distinct, but this changes the optimization landscape. `problems.csv` and the report show distinct-profile counts, duplicate fractions, and matrix-construction time. Use $C=100$ as a separate sensitivity analysis rather than silently changing $C$ with $N$.

## Budget and timing semantics

**Checks protocol:** HC counts starting-circle scores and all proposed moves; GA counts distinct cached complete-circle evaluations. These are different-cost units. Generation/restart caps can stop a run before its budget. The stopping reason and actual count are always retained.

**Proven-bound stop:** every problem has a reference score: the exact optimum ($N$ up to `--exact-limit`) or the minimum of three proven upper bounds ($N C$, the feature bound and the degree bound, rounded down to an even score; see below). No circle can score above it, so a search that reaches it can never improve again. Since both algorithms only replace their best on a strict improvement, stopping there leaves the returned circle, score, improvement trace and target hits unchanged; the stop reason is `bound_reached`. Elapsed time, counts, throughput and generations completed do change. So do checks-protocol `auc_time`, `post_best_fraction` and stagnation (horizon = the run's own duration) and time-protocol `auc_checks` (horizon = the actual count). Time-protocol quality, time AUC and stagnation use the common allowance and are unchanged. A loose bound is never reached and costs nothing. Use `--no-stop-at-bound` to run every search to its limit.

**Time protocol:** both searches receive the same absolute `perf_counter` deadline, measured from before search/selector setup. Count, generation, and restart caps are set very high so they do not normally bind. Deadline checks occur during GA initialization, evaluation, and offspring construction, and between HC operations. No algorithm is interrupted in the middle of an atomic operation. At least one valid circle is scored even if setup already overran.

The report's equal-time quality credits only improvements reported **before the common deadline**. `returned_score` separately preserves the algorithm's final return value, and `overshoot_seconds` records runtime beyond the allowance. If no valid score was reported in time, `score` is empty, `status` identifies the failure, and normalized-quality utility is zero; the run is not dropped from quality or target-success comparisons.

Shared profile/matrix construction and result validation are outside timed regions. Search validation, population/selector setup, and lightweight strict improvement callbacks are inside. Warm-ups are short and excluded. Execution order rotates between trials/repeats.

At a count cap, identical repeats must produce the same circle, score, and count, so the default runs one (`--check-repeats`). Time-limited repeats can stop at different points and produce different results. Both are stored individually; repeats are collapsed within a paired trial for statistical summaries, rather than treated as independent problems.

## Quality, convergence and targets

- **Normalized score:** $\text{score} / (N C)$, removing the trivial growth in raw score as children or characteristics increase.
- **Reference ratio / relative gap:** use the exact optimum: MILP for every $N$ by default, enumeration for small $N$. With `--reference bound`, or if the MILP time limit expires, the reference is instead the smaller of two valid upper bounds: the sum over features $f$ of $2 \min(T_f, F_f)$, with $T_f$ and $F_f$ the numbers of children having and lacking $f$, and half the sum of each child's two largest incident distances (rounded down). The minimum bound is then rounded down to an even score. Such a bound may be unattainable; its gap is **not** a claimed optimality gap.
- **First-hit time/check:** first occurrence of the run's eventual final score. Ties do not reset it. In equal-time runs, this is the best reached before the deadline. Different algorithms may have different final scores.
- **Post-best fraction:** fraction of the measured search horizon after that final improvement. This is retrospective, not a proof that the extra search could safely have been skipped.
- **AUC:** average normalized best-so-far score over a horizon, using exact step integration. Time before the first scored circle contributes zero. The time protocol uses the common time allowance. The checks protocol's time AUC uses each execution's duration, so use equal-time AUC for direct efficiency rankings. Check AUC uses the common count allowance in checks runs and the actual count in time runs. Returned best scores are held after an early cap. Raw rows include both AUC horizons.
- **Common targets:** integer thresholds $\lceil \text{fraction} \cdot \text{reference} \rceil$. Missing targets remain misses with an observation horizon; they are not removed to make time-to-target averages look better. The attainment plots are empirical hit fractions, not Kaplan–Meier extrapolations. An upper-bound-relative target may be unreachable. `target_success` in the exploration summaries uses 95%, or the first configured target if 95% is absent; all targets are in targets.csv.
- **Largest stagnation fraction:** longest interval without a strict global improvement, including initial setup and the final plateau, divided by the search horizon. Improvement count and throughput are also retained.

The report includes quality/gap/runtime scaling, first-hit/post-best scaling, convergence versus time and checks, target attainment, quality distributions, and score–runtime scatter plots. Pareto frontiers use condition means separately for each $N$, not pooled across problem sizes. Convergence plots sample step histories on a common grid; underlying AUC/target calculations use full histories.

## How statistics vary with exploration

Every exploration setting has the same instance/trial seeds. The report's metric dropdowns and size $\times$ exploration heatmaps show quality, total and first-hit time, AUC, stagnation, target success, throughput, and available diagnostics.

`exploration_effects.csv` pairs each guided result with:

1. its uniform baseline;
2. the previous exploration value in the sorted grid;
3. the first exploration value (when different from the previous value).

It reports absolute mean/median changes, positive-baseline ratios and percent changes, increase/tie/decrease counts, paired standardized change $d_z$ (`paired_dz`, undefined when changes have zero spread), and 95% confidence intervals for the mean change.

`exploration_trends.csv` fits a within-trial linear slope, expressed per **+0.1 exploration**, and computes tied-rank Spearman correlation. It then summarizes those trial trends. This reveals whether a metric tends to rise or fall without pretending every response is linear. A U-shaped response may have near-zero slope/correlation, so inspect the response plots and heatmaps too. These are descriptive comparisons, not a multiple-comparison-corrected significance test or a guarantee that an exploration setting generalizes to unseen problems.

Summary error bars are 95% hierarchical bootstrap intervals: sample instance families with replacement, then paired trials within a family. Repeats are first collapsed to medians; target success uses the repeat mean. Intervals are omitted with fewer than two instance families. Standard deviations and quartiles describe the selected instance/trial sample. No inference pools characteristic counts, sizes, algorithms, protocols, or exploration settings. Nested sizes remain correlated; the suite does not claim independent observations across those sizes.

## Exploration/exploitation diagnostics without corrupting timings

With diagnostics enabled, each **checks-protocol** trial is rerun once with the same seed and count settings. Its final circle/score/count must match the timed execution. Each variant replay is a separate job, submitted largest first, so the pool does not wait on a few large batches at the end. Copied exploration $= 1$ rows reuse the uniform replay. Expensive diagnostics never run inside equal-time searches, and their runtime is never used as search runtime. A time-only suite therefore has no detailed coverage/diversity results. These fixed-check diagnoses explain search behavior, not the exact path of a deadline-truncated execution.

- **Actual proposal mix:** guided swaps, uniform swaps, inversions. Exploration is conditional on the swap branch, not the fraction of all moves.
- **HC acceptance / improvement rates:** positive-delta fraction by move type. GA has no HC acceptance rule; it instead records the fraction of mutations improving the pre-mutation child, plus positive and signed gain per proposal. Diagnostic fitness comparisons do not consume the algorithm's evaluation budget/cache. Custom mutation callbacks without built-in counters are not supported for move-type attribution by this suite.
- **Proposal novelty:** unique canonical proposed circles divided by proposals. HC includes rejected moves; GA includes offspring attempts and fallback random injections. Initialization is not a proposal. Scored circles are counted separately (a delta-checked HC candidate has a known score too).
- **Circle coverage:** unique scored indexed circles relative to $(N-1)!/2$, stored as `log10_circle_coverage` to avoid numerical underflow. Equal profiles do not merge distinct children. Almost-zero coverage is normal for large $N$.
- **Edge coverage:** distinct observed neighbor pairs divided by $N(N-1)/2$. High edge coverage does not imply high coverage of whole circles.
- **GA edge diversity:** average fraction of edges not shared by two distinct population members. Unlike positional distance, this respects the circular neighbor structure and reversal symmetry.
- **GA edge entropy:** normalized entropy of observed population edge frequencies. Even an identical population has entropy over its $N$ edges; use edge diversity alongside entropy instead of treating entropy alone as population uniqueness.
- **Cache-hit/duplicate rates:** all fitness-cache queries and duplicate offspring attempts, respectively. A high rate can mean wasted work, beneficial reuse, or convergence; it is not automatically bad. Across implementation versions, removing parent-selection cache queries changes the cache-hit denominator; a lower rate is not evidence of greater search-space exploration by itself.
- **Restart improvements:** strict records found immediately at a new random starting circle after the first start. The global stagnation histories include the subsequent local searches as well.
- **Restart exit counts:** distinguish sampling stalls, legacy attempt caps, count-budget exhaustion and time limits. Search functions return per-restart histories; exported HC rows/summaries expose exit counts. A sampling stall is not a certificate that no improving neighbor exists.

Coverage tracking stores canonical circles and costs memory/time proportional to the number and length of observed candidates. Use `--no-diagnostics` for large experiments where only the primary performance comparisons are needed.

## Files and reproducibility

| File | Contents |
| --- | --- |
| `configuration.json` | Settings, seeds, Python/platform and protocol descriptions |
| `execution.csv` | Current invocation's phase wall times, worker counts, and executed/resumed batch counts |
| `runs.csv` | One row per timed execution, including repeats and stop reasons; copied exploration $= 1$ rows name their source in `aliased_from` |
| `improvements.csv` | Full strict-improvement traces, joined through run_id |
| `targets.csv` | Hit/miss, first-hit time/count and observation horizon for every target |
| `problems.csv` | Distinct profiles, bounds, exact optimum and its source (enumeration/MILP), MILP status/bound/seconds, matrix timing |
| `summaries.csv` | Per-condition metrics, variability and clustered intervals |
| `exploration_effects.csv` | Paired exploration changes and ratios |
| `exploration_trends.csv` | Per-+0.1 slopes and Spearman summaries |
| `diagnostics.csv` | Fixed-check replay diagnostics associated with timed trials |
| `diagnostic_snapshots.csv` | Coverage/diversity trajectories from those replays |
| `results.json` | All result tables together |
| `report.html` | Self-contained interactive figures and interpretation notes |
| `.checkpoints/` | Atomic batch results and config/source/environment compatibility manifest |

The comparison command additionally creates `comparison.html`, `comparison.csv`, and `comparison.json`. The retained enhanced full run also has supervisor-created `run.log` and `run-status.json`; ordinary suite invocations print progress to standard output rather than creating these two files automatically.

In `configuration.json`, `elapsed_seconds` covers preparation, searches, diagnostics and analysis before export. `export_seconds` and `report_seconds` cover output generation, and `total_seconds` covers the full invocation. Use matching timing scopes when comparing historical runtimes. On a resumed run, phase wall times describe that invocation's work, not the cost of producing all restored observations from scratch.

Generated `benchmark-results/` directories are ignored by git. Output directories are created as needed; completed results require compatible `--resume` checkpoints or a different output directory. Distinct timestamped directories are the default.

## Verification

```sh
.venv/bin/python -m unittest discover -v
```

Tests cover bounds against exact optima, nested profiles, known first-hit/AUC values, paired exploration changes/trends, repeat handling, diagnostics replay invariance, expired deadlines, uncredited after-deadline scores, coverage/diversity, spawn/serial equivalence, timing isolation, checkpoint resume/mismatch rejection, productive and legacy restarts, retained swaps in HC-I, frozen pre-optimization GA trajectories and original OX behavior, cross-version target rebasing, CSV/JSON output, and offline HTML rendering when Plotly is installed. Tests do not assert a speedup or preferred exploration value from noisy runtime samples.
