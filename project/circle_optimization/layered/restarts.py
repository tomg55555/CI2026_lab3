"""Layer S6 (phase 14): seeded starts and incumbent perturbation.

Shared constructors for both layered engines:

- `seed_multigraph`: randomized greedy b-matching on the profile multigraph
  (phase 07 `greedy_multigraph` + `connect`), i.e. one GRASP construction.
  The edge-genome GA (S4) seeds its population with it.
- `seed_sequence`: the same construction realized as a profile sequence by an
  Euler circuit, randomly rotated; the layered HC starts or restarts from it.
- `double_bridge`: the classic 4-opt "double bridge" kick for iterated local
  search: cut the circle into A B C D and reconnect as A C B D. A 2-opt or
  splice move cannot undo it in one step, so the search leaves its basin.

All randomness comes from the caller's `random.Random`, so runs are seeded.
The noise level controls diversity: 0 is pure greedy (random ties only).
"""

from random import Random

import numpy as np

from ..exact.multigraph import connect, greedy_multigraph
from ..exact.profiles import euler_sequence
from .state import ProfileSequence, ProfileState


def seed_multigraph(state: ProfileState, rng: Random, *, noise: float = 1.5) -> np.ndarray:
    """Connected greedy multigraph whose degrees match the instance.

    Profile pairs are saturated by decreasing distance plus uniform noise in
    [0, noise); remaining degree becomes same-profile loops; components are
    then merged by the cheapest 2-exchanges. A numpy generator is seeded from
    one 64-bit draw of `rng`.
    """
    x = greedy_multigraph(state.instance, np.random.default_rng(rng.getrandbits(64)), noise=noise)
    if state.m > 1:
        connect(x, state.instance.distances)
    return x


def seed_sequence(state: ProfileState, rng: Random, *, noise: float = 1.5) -> ProfileSequence:
    """Profile sequence of a greedy seed (`seed_multigraph`), randomly rotated."""
    sequence = euler_sequence(seed_multigraph(state, rng, noise=noise)).tolist()
    shift = rng.randrange(len(sequence))
    return sequence[shift:] + sequence[:shift]


def double_bridge(sequence: ProfileSequence, rng: Random) -> ProfileSequence:
    """Return A C B D for random cuts A|B|C|D (needs at least 4 elements).

    Three distinct cut points 0 < a < b < c < N are drawn uniformly. The
    circle keeps its length and profile counts; four edges change.
    """
    n = len(sequence)
    if n < 4:
        return list(sequence)
    a, b, c = sorted(rng.sample(range(1, n), 3))
    return sequence[:a] + sequence[b:c] + sequence[a:b] + sequence[c:]
