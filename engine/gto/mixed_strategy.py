"""Sampler for mixed-strategy frequency distributions (e.g. GTO solver outputs).

A mixed strategy is a decision point where several actions each carry a
frequency in [0, 1] rather than a single deterministic choice — the hallmark
of a Nash equilibrium at a node where the player is indifferent between
actions. This module samples from and evaluates such distributions; it does
not compute them (see `engine.gto.chart.PreflopChart` for loading them from
solver exports).
"""

from __future__ import annotations
import random


class MixedStrategy:
    """Samples an action from a mixed-strategy frequency distribution."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def sample(self, frequencies: dict[str, float], *, rng: random.Random | None = None) -> str:
        """Sample one action, weighted by its frequency.

        Frequencies need not sum to exactly 1.0 — solver exports often carry
        floating-point drift — they're normalized against their own total.
        """
        total = _validated_total(frequencies)
        r = (rng or self._rng).random() * total
        cumulative = 0.0
        for action, freq in frequencies.items():
            cumulative += freq
            if r < cumulative:
                return action
        # Floating-point drift: r landed past the last cumulative bucket.
        return next(reversed(frequencies))

    def expected_value(self, frequencies: dict[str, float], values: dict[str, float]) -> float:
        """Weighted-average EV of a mixed strategy given per-action EVs.

        Actions present in `frequencies` but missing from `values` are ignored
        rather than erroring, since a chart's action set may be a superset of
        the actions a particular EV model prices.
        """
        total = _validated_total(frequencies)
        return sum(freq / total * values[a] for a, freq in frequencies.items() if a in values)


def _validated_total(frequencies: dict[str, float]) -> float:
    if not frequencies:
        raise ValueError("frequencies must not be empty")
    if any(f < 0 for f in frequencies.values()):
        raise ValueError(f"frequencies must be non-negative, got {frequencies}")
    total = sum(frequencies.values())
    if total <= 0:
        raise ValueError(f"frequencies must sum to > 0, got {frequencies}")
    return total
