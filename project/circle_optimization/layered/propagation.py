"""Layer S5 (phase 13): repeat an accepted 2-edge move's pattern elsewhere.

On profile sequences the gain of an inversion or splice depends only on the
four profiles involved: removing edge types {p1, p2} and {q1, q2} and adding
p1-q1, p2-q2 always gains

    g = D[p1][q1] + D[p2][q2] - D[p1][p2] - D[q1][q2].

So after a move with gain g > 0 is accepted, every other pair of circle
edges of the same two types offers the same g, and the classic HC would only
find them one random proposal at a time. This is the per-circle analogue of
the phase-07 bulk 2-exchange, which moves all copies of a profitable pattern
at once and closed 61-98% of the gap left by HC/GA local optima.

Which reconnection realizes the pattern depends on the edges' orientation
along the circle (edge e joins positions e, e+1 mod N). An inversion joins
left-left and right-right endpoints of the two cut edges, a splice (layer S1)
left-right and right-left, whichever edge comes first. Hence the pattern
needs an inversion if both edges are oriented like the original (or both
reversed, or either is a same-profile loop), otherwise a splice.

`Propagation` finds such a pair by one scan per proposal; the engine
(`hill_climbing.run_layered_hill_climber`) scores it as an ordinary
proposal (one check, strict improvement, global limits, `on_improvement`),
rescans after each acceptance and ends the trigger at the first failure, at
no remaining pair, or after `max_attempts` proposals. A scan packs the
sequence into bytes (one C-level pass) and searches the edge types as
two-element substrings, so its Python work is proportional to the matching
edges, not to N. Scans run only after accepted moves, which already cost
O(N) to apply.
"""

from array import array
from collections.abc import Sequence
from random import Random

INVERSION, SPLICE = "inversion", "splice"
MAX_ATTEMPTS = 64


def _typecode(profiles: int) -> str:
    """Smallest array typecode holding profile ids below `profiles`."""
    return "B" if profiles <= 1 << 8 else "H" if profiles <= 1 << 16 else "I"


def pack(sequence: Sequence[int], code: str) -> bytes | bytearray:
    """The sequence as fixed-width bytes (`bytearray` for one byte: ~1 µs per 1000)."""
    return bytearray(sequence) if code == "B" else array(code, sequence).tobytes()


class Propagation:
    """An accepted pattern: remove (p1, p2) and (q1, q2), add p1-q1 and p2-q2.

    Build it with `from_move` from the sequence *before* the move was applied.
    `allow_splice` permits pairs that need a splice (layer S1 enabled);
    `max_attempts` caps the proposals of this trigger; `profiles` is m (any
    bound above the largest profile id; None derives it from the sequence
    at the first scan).
    """

    __slots__ = ("p1", "p2", "q1", "q2", "allow_splice", "attempts_left", "profiles")

    def __init__(self, p1: int, p2: int, q1: int, q2: int, *, allow_splice: bool,
                 max_attempts: int = MAX_ATTEMPTS, profiles: int | None = None) -> None:
        """Remember the pattern's four profiles and its remaining attempt allowance."""
        self.p1, self.p2, self.q1, self.q2 = p1, p2, q1, q2
        self.allow_splice = allow_splice
        self.attempts_left = max_attempts
        self.profiles = profiles

    @classmethod
    def from_move(cls, sequence: Sequence[int], i: int, j: int, *, crossed: bool, allow_splice: bool,
                  max_attempts: int = MAX_ATTEMPTS, profiles: int | None = None) -> "Propagation":
        """Pattern of an inversion (or splice if `crossed`) at positions i < j.

        Inversion: removes (a, b), (c, d) and adds a-c, b-d. Splice: adds a-d,
        b-c, i.e. the same form with the second edge read as (d, c).
        """
        a, b = sequence[i - 1], sequence[i]
        c, d = sequence[j], sequence[(j + 1) % len(sequence)]
        if crossed:
            c, d = d, c
        return cls(a, b, c, d, allow_splice=allow_splice, max_attempts=max_attempts, profiles=profiles)

    def gain(self, table: Sequence[Sequence[int]]) -> int:
        """The pattern's score change g (O(1))."""
        p1, p2, q1, q2 = self.p1, self.p2, self.q1, self.q2
        return table[p1][q1] + table[p2][q2] - table[p1][p2] - table[q1][q2]

    def edges(self, sequence: Sequence[int], x: int, y: int,
              packed: bytes | bytearray | None = None) -> list[tuple[int, bool]]:
        """Circle edges of type {x, y} as (edge, starts with x); loops once each.

        `packed` is `pack(sequence, typecode)` (built here if omitted).
        Overlapping matches count (runs of loops).
        """
        if self.profiles is None:
            self.profiles = max(sequence) + 1
        code = _typecode(self.profiles)
        width = array(code).itemsize
        if packed is None:
            packed = pack(sequence, code)
        found = []
        find = packed.find
        for first, second, forward in ((x, y, True),) + (((y, x, False),) if x != y else ()):
            target = array(code, (first, second)).tobytes()
            k = find(target)
            while k >= 0:
                if k % width == 0:
                    found.append((k // width, forward))
                k = find(target, k + 1)
        n = len(sequence)
        last, head = sequence[n - 1], sequence[0]  # The wraparound edge n-1 -> 0.
        if last == x and head == y:
            found.append((n - 1, True))
        elif last == y and head == x:
            found.append((n - 1, False))
        return found

    def next_move(self, sequence: Sequence[int], rng: Random,
                  *, tries: int = 8) -> tuple[str, int, int] | None:
        """(kind, i, j) of another occurrence of the pattern, or None.

        Finds the edges of both types with their orientation (`edges`), then
        draws up to `tries` random (first-type, second-type) pairs until one
        is non-adjacent and realizable by an allowed move. Consumes one
        attempt; None once attempts or occurrences run out.
        """
        if self.attempts_left <= 0:
            return None
        self.attempts_left -= 1
        p1, p2, q1, q2 = self.p1, self.p2, self.q1, self.q2
        if self.profiles is None:
            self.profiles = max(sequence) + 1
        packed = pack(sequence, _typecode(self.profiles))
        first = self.edges(sequence, p1, p2, packed)
        second = self.edges(sequence, q1, q2, packed) if first else None
        if not second:
            return None
        n = len(sequence)
        loop = p1 == p2 or q1 == q2
        random = rng.random
        for _ in range(tries):
            e, e_forward = first[int(random() * len(first))]
            f, f_forward = second[int(random() * len(second))]
            low, high = (e, f) if e < f else (f, e)
            if high - low < 2:
                continue
            if loop or e_forward == f_forward:
                if low == 0 and high == n - 1:
                    continue  # Reversing all but position 0 changes nothing.
                return INVERSION, low + 1, high
            if self.allow_splice:
                return SPLICE, low + 1, high
        return None
