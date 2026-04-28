"""Kelly Criterion for optimal bet sizing."""
from __future__ import annotations


class KellyCriterion:
    def fraction(self, win_prob: float, win_odds: float, fraction: float = 1.0) -> float:
        """
        Kelly fraction of bankroll to wager.
        win_prob: probability of winning [0,1]
        win_odds: net odds (e.g. 2.0 means win 2x the stake)
        fraction: fractional Kelly multiplier (default 1 = full Kelly)
        """
        kelly = (win_prob * win_odds - (1 - win_prob)) / win_odds
        return max(0.0, kelly * fraction)

    def units(self, bankroll: float, win_prob: float, win_odds: float, fraction: float = 0.25) -> float:
        """Return stake size in currency units."""
        return bankroll * self.fraction(win_prob, win_odds, fraction)
