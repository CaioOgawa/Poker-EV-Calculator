"""Preflop GTO chart loader — reads solver-exported range charts (CSV/JSON)
and exposes per-hand action frequencies.

Expected format (both CSV and JSON): one entry per canonical hand (e.g.
"AKs", "77", "T9o") mapping to a dict of action -> frequency, matching how
solvers like PioSolver/GTO+ export mixed-strategy ranges.

CSV layout: header `hand,<action1>,<action2>,...`, one row per hand,
frequencies as floats; blank cells are treated as 0 and omitted. JSON layout:
`{"AKs": {"raise": 1.0}, "77": {"raise": 0.6, "call": 0.4}, ...}`.

Hands missing from the chart raise KeyError on lookup unless a
`default_action` is supplied, in which case they're treated as 100% that
action (typically "fold").
"""

from __future__ import annotations
import csv
import json
import random
from pathlib import Path

from core.ranges.range_parser import RANKS
from engine.gto.mixed_strategy import MixedStrategy

Frequencies = dict[str, float]

_RANK_IDX = {r: i for i, r in enumerate(RANKS)}


class PreflopChart:
    def __init__(self, ranges: dict[str, Frequencies], *, default_action: str | None = None):
        self._ranges = {_normalize_hand(h): freqs for h, freqs in ranges.items()}
        self._default_action = default_action
        self._sampler = MixedStrategy()

    @classmethod
    def from_csv(cls, path: str | Path, *, default_action: str | None = None) -> PreflopChart:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "hand" not in reader.fieldnames:
                raise ValueError(f"{path}: CSV must have a 'hand' column")
            actions = [c for c in reader.fieldnames if c != "hand"]
            ranges = {
                row["hand"]: {a: float(row[a]) for a in actions if row[a] not in (None, "")}
                for row in reader
            }
        return cls(ranges, default_action=default_action)

    @classmethod
    def from_json(cls, path: str | Path, *, default_action: str | None = None) -> PreflopChart:
        with open(path) as f:
            data = json.load(f)
        ranges = {hand: {a: float(v) for a, v in freqs.items()} for hand, freqs in data.items()}
        return cls(ranges, default_action=default_action)

    def frequencies(self, hand: str) -> Frequencies:
        """Return the action -> frequency mapping for a canonical hand."""
        hand = _normalize_hand(hand)
        if hand in self._ranges:
            return self._ranges[hand]
        if self._default_action is not None:
            return {self._default_action: 1.0}
        raise KeyError(f"No chart entry for hand '{hand}'")

    def action(self, hand: str, *, rng: random.Random | None = None) -> str:
        """Sample a single action for `hand` according to its mixed strategy."""
        return self._sampler.sample(self.frequencies(hand), rng=rng)

    def range_for_action(self, action: str, min_frequency: float = 0.5) -> list[str]:
        """Canonical hands where `action`'s frequency is at least `min_frequency`."""
        return [
            hand for hand, freqs in self._ranges.items() if freqs.get(action, 0.0) >= min_frequency
        ]

    def combo_weighted_range(self, action: str) -> list[tuple[str, float]]:
        """(hand, frequency) pairs for every hand with a nonzero frequency on `action`.

        Unlike `range_for_action`'s hard cutoff, this preserves the mix — for
        consumers that weight equity/EV contributions by how often a hand
        actually takes the action rather than treating the range as binary.
        """
        return [
            (hand, freqs[action])
            for hand, freqs in self._ranges.items()
            if freqs.get(action, 0.0) > 0.0
        ]


def _normalize_hand(hand: str) -> str:
    """Normalize to canonical form: higher rank first, 's'/'o' suffix for non-pairs."""
    hand = hand.strip()
    if len(hand) == 2:
        r1, r2 = hand[0].upper(), hand[1].upper()
        if r1 not in _RANK_IDX or r2 not in _RANK_IDX:
            raise ValueError(f"Invalid rank in hand '{hand}'")
        if r1 != r2:
            raise ValueError(f"Two-character hand '{hand}' must be a pocket pair")
        return r1 + r2
    if len(hand) == 3:
        r1, r2, suit = hand[0].upper(), hand[1].upper(), hand[2].lower()
        if suit not in ("s", "o"):
            raise ValueError(f"Hand '{hand}' must end in 's' or 'o'")
        if r1 not in _RANK_IDX or r2 not in _RANK_IDX:
            raise ValueError(f"Invalid rank in hand '{hand}'")
        if _RANK_IDX[r1] < _RANK_IDX[r2]:
            r1, r2 = r2, r1
        return r1 + r2 + suit
    raise ValueError(f"Invalid hand string '{hand}'")
