# Next-stage plan: cheap guidance and productive stopping

> **Status update (2026-10-21, phases 11 and 14).** The $O(1)$ tournament idea of step A2 is used by the reduced-cost guide (S3, tournament of 8 edges ranked by LP reduced cost instead of distance; [phase 11](phases/11-reduced-cost-guidance.md)). Step B4's "perturb a strong incumbent" was measured as double-bridge ILS against greedy GRASP restarts; GRASP won on every uniform instance ([phase 14](phases/14-seeding-and-ils.md)). Steps A1, A3 and B1–B3 remain open for the classic eras.


Date: 2026-10-21. This is a plan, not a claim that these optimizations have been implemented or measured. The phase-04 productive restart fix is implemented; the more adaptive policies below remain follow-ups.

## A. Make guided swap selection cheaper

### Current bottleneck

The cumulative distribution is cached on an unchanged circle, but any accepted swap/inversion invalidates it. The next guided swap rebuilds $N$ weights in $O(N)$. Historical $N=160$ guided HC costs ~73% more per fixed-count execution than uniform HC. More accepted moves can paradoxically increase rebuilding overhead.

### Step A1 — measure rebuilding, not just final score

In untimed diagnostic replays, count invalidations, rebuilds, their causes (swap/inversion), reversed-segment lengths, and proposals per rebuild. Profile the existing selector across $N$, exploration, and the new inversion-heavy mix. Do not add expensive clocks to every timed proposal.

Acceptance: identify where rebuilds dominate and retain a baseline response curve for fixed-count and isolated equal-time protocols.

### Step A2 — cheap approximate guidance as a separate strategy

Add a weak-edge tournament sampler: draw $k$ current edges uniformly and choose the lowest-scoring one, then sample an endpoint/partner as today. For fixed small $k \in \{2, 4\}$, selection is $O(1)$, with no probability table or invalidation cost.

This changes the probability distribution and random-draw sequence: label it a new guidance strategy, not a bit-for-bit optimization of the weighted sampler. Keep an independent uniform-exploration fraction and compare quality/time/AUC, acceptance, and actual proposal bias against both existing controls.

### Step A3 — exact incremental weighted sampling

Prototype a Fenwick sum tree over current edge weights. Preserve the same total integer weights, ticket draw, and prefix-selection order. A swap changes at most four edge slots; update them in $O(\log N)$. Rejected moves require no updates.

Inversions complicate this: many edge *positions* reverse even though internal edge scores are preserved. A tree indexed by positions cannot update just two boundary weights. Compare $O(k \log N)$ updates for short segments with an $O(N)$ rebuild for long segments. Do not claim $O(\log N)$ inversion maintenance without also solving that ordering problem.

If long inversions dominate (likely in inversion-focused HC), assess a stable edge-ID representation or a sequence tree with lazy reversal/sum maintenance. That is a larger tour-representation change and should be justified by profiling.

Acceptance: exact sampler matches a linear reference for every integer ticket after random adjacent/nonadjacent swaps and reversals, including wraparound; seeded proposal trajectories match when promising unchanged semantics. Adopt only if isolated wall-time/quality trade-offs improve over the simpler sampler.

## B. Stop cutting productive local searches short

### Implemented first fix: productive patience

Default HC now restarts after `stall_limit` consecutive failed proposals, resetting the counter on every accepted local improvement. The legacy total-attempt cap is active only with `restart_policy="legacy"`. Uniform HC's 50/50 move mix and the separate inversion-focused mix share this policy. Global count/deadline and restart limits still bound work. Exit reasons are recorded per restart.

The first fix addresses the demonstrated arbitrary 1,000-proposal cutoff. A random sampling stall still does **not** prove a local optimum.

### Step B1 — validate policy across size and budgets

Compare legacy/productive at identical seeds and both budget protocols. Use restart exit counts, attempts and accepted moves per restart, gain per check, first-hit time, AUC, and final quality. Run additional $C=100$ instance families; do not tune and declare victory on the historical three $C=5$ families alone.

### Step B2 — bounded adaptive patience

Estimate a recent local improvement rate and extend patience when a search has recently been productive; shrink it only when prolonged failure is observed. Bound patience above and below and charge all proposals to the same budget.

The heuristic $(1-p)^L$ can motivate a failure allowance, but proposal outcomes are dependent and $p$ changes as the circle improves. It is not a calibrated certificate that no improving move exists. Record policy parameters and the reason for every extension/restart; preserve a fixed-patience control.

### Step B3 — optional neighborhood confirmation before restart

After a random stall, spend a bounded remaining allowance on inversion moves, using delta checks. A complete 2-opt scan can establish a 2-opt local optimum; a sampled or candidate-list scan cannot. Report these statuses distinctly. At $N=160$ a complete scan costs $O(N^2)$ checks and can consume a large fraction of the whole budget, so validate the quality/time trade-off before making it default.

### Step B4 — evidence-based allocation between depth and restarts

Use histories to compare marginal gains from extending a current search against gains from new starts. Consider perturbing a strong incumbent as another explicit variant, rather than always discarding it for a fully random start. Compare success/target attainment across independent held-out families.

Global budget exhaustion must always be logged as budget exhaustion, not as an inferred local optimum or an unproductive run.
