# Phase 10 — layer S1: split-and-splice moves

Started/completed: 2026-10-21. Status: **completed and verified** (HC layer; era wiring is done by the registry, `circle_optimization/eras.py`).

## Motivation

A 2-exchange removes two circle edges $(a, b)$ and $(c, d)$ and adds two new ones. On a permutation of children only one reconnection keeps a single circle: $(a, c)$, $(b, d)$, the 2-opt reversal (classic inversion). The other one, $(a, d)$, $(b, c)$, splits the circle in two. The phase-07 multigraph search (`exact.multigraph`) uses *both* reconnections because an Euler circuit can re-thread the children, and its aggregated 2-exchanges closed 61–98% of the gap left by HC/GA local optima. The classic HC can never propose the crossed reconnection.

## Derivation

Positions $1 \le i < j \le N-1$ of a profile sequence; $a = \mathrm{seq}[i-1]$, $b = \mathrm{seq}[i]$, $c = \mathrm{seq}[j]$, $d = \mathrm{seq}[(j+1) \bmod N]$. The crossed reconnection gives two loops:

- $L_1 = \mathrm{seq}[i \ldots j]$, closed by the new edge $(c, b)$;
- $L_2 = \mathrm{seq}[j+1 \ldots N-1] + \mathrm{seq}[0 \ldots i-1]$, closed by the new edge $(a, d)$.

If a profile $p$ occurs at $u$ in $L_1$ and at $v$ in $L_2$, with loop successors $u'$ and $v'$, exchanging the successors ($u \to v'$, $v \to u'$) merges the loops into one circle. It removes $(p, u')$, $(p, v')$ and adds $(p, v')$, $(p, u')$: the same two profile pairs, so the merge is free. Hence

$$
\Delta = D_{ad} + D_{bc} - D_{ab} - D_{cd}
$$

in $O(1)$, exactly like `inversion_delta`, and the profile multiset is kept. In child terms, $L_1$'s children are re-threaded next to a child of the same profile somewhere in $L_2$. The merge needs a shared profile; with $m \le 2^C \ll N$ that is almost always true. Edge cases: $j = i+1$ makes $L_1$ a 2-cycle; $i = 1$, $j = N-1$ makes $L_2$ the single element $\mathrm{seq}[0]$ closed by a loop ($a = d$), i.e. the move relocates $\mathrm{seq}[0]$.

## Implementation

`circle_optimization/layered/splice.py`:

- `splice_delta(seq, table, i, j)`: the $O(1)$ delta above;
- `splice_loops(seq, i, j)`: $L_1$ and $L_2$ in circle order;
- `shared_profile(inner, outer, rng)`: probes up to 8 uniform elements of the shorter loop for membership in the longer (C-level `in`), then falls back to testing its distinct profiles in increasing order; None if none;
- `apply_splice(seq, i, j, rng)`: picks occurrences $u$, $v$ of the shared profile from random offsets (C-level `list.index`) and $L_1$'s direction (both give the same edges), returns `L2[..v] + L1[u+1..] + L1[..u] + L2[v+1..]`, or None.

A first version built a Python-level list of all shared positions: 40 µs per accepted splice at $N=1000$, which halved throughput (78k vs 155k checks in 0.2 s). The probing version costs a few µs.

Engine (`layered/hill_climbing.py`): `splice_rate` adds a third proposal kind. One draw $r$ = `rng.random()` picks a swap if $r < 1 - \mathtt{inversion\_rate} - \mathtt{splice\_rate}$, an inversion if $r < 1 - \mathtt{splice\_rate}$, otherwise a splice, so `splice_rate=0` consumes the classic draws exactly. Splice positions are `sorted(uniform_swap_positions(n, rng))` (or the S3 guide). A splice costs one check; if $\Delta > 0$ but the loops share no profile it is rejected (counted as a failed check, `moves["splice"]["accepted"]` unchanged). The rebuild ($O(N)$) happens only for accepted moves.

## Verification

`tests/test_layered_hc.py::SpliceTests`:

- brute force on 6 instance shapes ($N = 4, \dots, 60$, $C = 2, \dots, 5$), 12 random sequences each, all $(i, j)$ for small $N$ or 60 pairs including $(1, N-1)$, $(1, 2)$, $(N-2, N-1)$: $\Delta = \mathrm{score}(L_1) + \mathrm{score}(L_2) - \mathrm{score}(\text{circle})$ (the derivation); the spliced sequence keeps the profile multiset, scores $\text{old} + \Delta$, and its edge multiset is exactly $\text{old} - \{(a, b), (c, d)\} + \{(a, d), (b, c)\}$; seeded results repeat; inputs are never mutated;
- None exactly when the loops share no profile (8 distinct profiles), and the engine then accepts no splice;
- inversion and splice deltas are the two reconnections of the same cut.

Engine: with `splice_rate=0` results equal the base engine and classic HC (`tests/test_layered_state.py`); with splices, results are valid, $\text{score} =$ `genome_score(circle)`, counts within budget, deterministic per seed, $\le$ the exact optimum (`LayeredEngineTests`).

## Measurement

All stacks were measured in one run; protocol and the full table are in [phase 14](14-seeding-and-ils.md#measurement): equal time 0.2 s, 5 rng seeds, $N=160$ = `random_profiles(2027–2029, 160, 5)` (15 runs), $N=1000$ = `random_profiles(2027, 1000, 5)` (5 runs), mean gap to the exact optimum.

| Proposal mix (swap / inversion / splice) | $N=160$ gap | checks | $N=1000$ gap | checks |
| --- | ---: | ---: | ---: | ---: |
| 0.5 / 0.5 / 0 (S2 base) | 10.25% | 160,413 | 18.74% | 145,724 |
| 0.2 / 0.8 / 0 (HC-I mix) | 9.65% | 180,574 | 17.22% | 161,888 |
| 0.2 / 0.4 / 0.4 | 8.81% | 168,643 | 17.64% | 133,399 |
| 0.2 / 0.3 / 0.5 | 9.39% | 166,146 | 17.53% | 127,517 |
| 0.2 / 0.5 / 0.3 | 9.25% | 171,277 | 17.77% | 138,473 |
| **0.1 / 0.45 / 0.45** | **8.82%** | 174,697 | **17.08%** | 134,232 |

Alone, S1 is a modest gain: 1.4 points over the base and 0.8 points over the 80%-inversion mix at $N=160$, and within noise of that mix at $N=1000$, where accepted splices ($O(N)$ rebuild) lower throughput. Chosen: inversion_rate 0.45, splice_rate 0.45. In the ablations, giving the splice share to inversions made the greedy + double-bridge stack 0.07 points worse at $N=160$ and 0.27 points better at $N=1000$; with greedy restarts the difference is within noise (phase 14). S1 matters mostly as an enabler: S3 places splice cuts too, and S5 can repeat reversed-orientation patterns only through splices.

## Limitations

- The engine draws the move kind before the positions, so a cut whose best reconnection is the other one is wasted; evaluating both reconnections per cut (2 checks) was not tried.
- On instances with $m$ close to $N$ ($C$ large, few repeated profiles) the merge is often impossible; splices then only cost checks.
- Accepted splices rotate the sequence (the result starts with $L_2$); harmless, since profile sequences have no anchor.
