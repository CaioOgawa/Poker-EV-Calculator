"""Implied odds — pot odds extended to account for expected future winnings.

Standard pot odds (`EVCalculator.pot_odds`) only look at chips already in the
pot. A drawing hand that doesn't have the equity to profitably call on pot
odds alone can still be a call if it expects to win additional chips on later
streets when it hits — that's what this module quantifies.
"""

from __future__ import annotations


class ImpliedOdds:
    def effective_pot_odds(
        self, call_size: float, pot: float, expected_future_winnings: float = 0.0
    ) -> float:
        """Minimum equity needed to break even on a call, given expected future
        winnings on later streets when the hand hits.

        Reduces to plain pot odds (`call_size / (pot + call_size)`) when
        `expected_future_winnings` is 0.
        """
        if call_size < 0 or pot < 0 or expected_future_winnings < 0:
            raise ValueError("call_size, pot, and expected_future_winnings must be non-negative")
        return call_size / (pot + call_size + expected_future_winnings)

    def ev(
        self, equity: float, pot: float, call_size: float, expected_future_winnings: float = 0.0
    ) -> float:
        """EV of a call, crediting `expected_future_winnings` only on the branch
        where the hand wins (i.e. the draw completes and villain pays it off).
        """
        if not 0.0 <= equity <= 1.0:
            raise ValueError(f"equity must be in [0, 1], got {equity}")
        if expected_future_winnings < 0:
            raise ValueError("expected_future_winnings must be non-negative")
        return equity * (pot + call_size + expected_future_winnings) - call_size

    def required_future_winnings(self, equity: float, pot: float, call_size: float) -> float:
        """Minimum additional winnings needed on future streets for a call with
        this `equity` to break even, when pot odds alone aren't enough.

        Returns 0.0 if the call is already break-even or better on pot odds
        alone — there's nothing further required.
        """
        if not 0.0 < equity <= 1.0:
            raise ValueError(f"equity must be in (0, 1], got {equity}")
        breakeven_total = call_size / equity
        needed = breakeven_total - pot - call_size
        return max(needed, 0.0)
