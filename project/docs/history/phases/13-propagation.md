# Phase 13 — layer S5: move propagation

Started/completed: 2026-10-21. Status: **completed and verified** (HC layer).

## Motivation

On profile sequences, the gain of a 2-edge move depends only on the four profiles it touches. When HC accepts an inversion that replaces edge types $\{a, b\}$, $\{c, d\}$ by $\{a, c\}$, $\{b, d\}$ with gain $g > 0$, every other pair of circle edges of types $\{a, b\}$ and $\{c, d\}$ offers exactly the same $g$, but uniform proposals only find them by chance. The phase-07 bulk 2-exchange, which moves all copies of a profitable pattern at once, closed 61–98% of the gap left by HC/GA local optima. Propagation brings that idea into the HC without changing its acceptance rule.

## Design

Normalize an accepted move to the pattern "remove $(p_1, p_2)$ and $(q_1, q_2)$, add $(p_1, q_1)$ and $(p_2, q_2)$": for an inversion at $i < j$, $(p_1, p_2, q_1, q_2) = (a, b, c, d)$; for a splice ($(a, d)$ and $(b, c)$ added), $(a, b, d, c)$.

Which reconnection realizes the pattern on another pair of edges $e$, $f$ depends on their orientation along the circle (edge $e$ joins positions $e$ and $(e+1) \bmod N$): an inversion joins left–left and right–right endpoints, a splice left–right and right–left, independently of which edge comes first. So the pattern needs an inversion when both edges are oriented like $(p_1, p_2)$, $(q_1, q_2)$ or both reversed, or when either type is a same-profile loop; otherwise a splice (only offered when S1 is enabled, `splice_rate > 0`). A full reversal (cuts $0$ and $N-1$) is skipped as a null inversion.

Engine protocol (`layered/hill_climbing.py`, `propagate=True`):

1. after an accepted *non-propagated* inversion or splice, a `Propagation` is built from the sequence before the move;
2. while it is active, each loop iteration asks `next_move(current, rng)` for a pair of edges of the two types (rescanned every time, so positions are always current) and proposes it as an ordinary move: one check, the $O(1)$ delta, strict-improvement acceptance, the usual budget, deadline and `score_bound` checks, and `on_improvement` on global improvements;
3. the trigger ends at the first failure ($\Delta \le 0$, or a splice whose loops share no profile), when no pair remains, or after `MAX_ATTEMPTS = 64` proposals; then random proposals resume.

Propagated proposals count in `moves["propagated"]`, in the restart's `attempts`/`accepted`, and in the stall counter like every check.

## Implementation

`circle_optimization/layered/propagation.py`: `Propagation(p1, p2, q1, q2, *, allow_splice, max_attempts=64, profiles=None)`, `Propagation.from_move(seq, i, j, *, crossed, allow_splice, ...)`, `gain(table)`, `edges(seq, x, y)`, `next_move(seq, rng)` → ("inversion" | "splice", i, j) or None.

Scanning speed decided the design. A Python loop over all $N$ positions, then a scan of the occurrences of $p_1$ with C-level `list.index` (17 µs per type at $N=1000$, about half of the run time), were replaced by packing the sequence into bytes once per scan (`bytearray(seq)`, 1.2 µs per 1000 profiles; `array` with two or four bytes per id when $m > 256$) and searching each edge type as a two-element substring with `bytes.find`. The Python work is then proportional to the number of matching edges, not to $N$ or $N/m$. The wraparound edge is checked separately; overlapping matches (runs of loops) are kept; misaligned matches of wide encodings are skipped.

## Verification

`tests/test_layered_hc.py::PropagationTests`:

- `edges` equals brute-force enumeration for $m = 3, \dots, 300$ (one- and two-byte packing), including overlapping loops and the wraparound loop;
- from 40 random sequences and moves (both kinds, splices on and off), every proposed move has delta exactly equal to the pattern's gain, applying it (inversion or splice) raises the brute-force score by that gain, and no splice is proposed when splices are disabled;
- the attempt cap ends a trigger.

Engine (`LayeredEngineTests`): with propagation, budgets of 1, 2, 3, 17, 150, 1000 and 4321 checks are met exactly (limits are checked inside the propagation loop), an expired deadline scores only the start, a 50 ms deadline stops with `time_limit`, legacy attempt caps hold, results are valid, deterministic, and $\le$ the exact optimum; propagated acceptances occur at $N=160$.

Probe (smoke runs, seed 2027, rng seed 1, 0.2 s, S1+S3 stack): every propagated proposal was accepted (519/519 at $N=160$, 4007/4007 at $N=1000$), as expected from the constant gain.

## Measurement

Same protocol as [phase 11](11-reduced-cost-guidance.md#measurement) (0.2 s, 5 rng seeds; $N=160$ = seeds 2027–2029; $C=7$ row from a second run with `score_bound` = optimum, 5 runs). Stack S1 (0.45/0.45) + S3 (0.2, $k = 8$):

| | $N=160$, $C=5$ gap | checks | $N=1000$, $C=5$ gap | checks | $N=1000$, $C=7$ gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| without S5 | 1.51% | 85,929 | 6.39% | 65,984 | 4.71% |
| **with S5** | **1.31%** | 76,056 | **3.08%** | 37,001 | **4.62%** |

Propagation halves the gap at $N=1000$, where each profile pair has many copies ($N/m \approx 31$), and helps little at $N=160$ ($N/m = 5$). It costs throughput (scans after every acceptance) but each propagated check is an accepted move. With greedy starts (phase 14) removing S5 doubled the time to the optimum (5.9 → 10.4 ms at $N=160$, 3.6 → 8.0 ms at $N=1000$); in the ILS stack its effect was within noise.

## Limitations

- Only the most recent accepted pattern is propagated, greedily and first-found; the bulk 2-exchange ranks all patterns by total gain.
- Each trigger costs at least one scan even when the pattern occurs once; for $N \approx m$ (no repeated pairs) propagation is pure overhead.
- Swaps are not propagated (they change up to four edges).

## Integration measurement (phase-14 era benchmark)

Equal time 0.2 s (`benchmark-results/eras-phase14`): `hc-p-prop` vs `hc-p-rc` mean gap 1.50% vs 1.38% ($C=5$, $N=160$), **2.36% vs 4.55%** ($C=5$, $N=1000$), 3.26% vs 3.13% ($C=7$, $N=160$), 5.02% vs 5.26% ($C=7$, $N=1000$). At $N=10{,}000$, $C=8$ and 1 s (`eras-phase14-large`) propagation is harmful: 20.02% vs 11.57%. A profile of one run shows why: the search is still in its first descent, where about 25% of ordinary proposals already improve, yet each acceptance pays an $O(N)$ edge scan (27% of run time) and accepted splices an $O(N)$ rebuild; checks fall from 40.6k to 17.8k per second. Follow-up: trigger propagation only once the acceptance rate has dropped (late search), and measure again at large $N$.
