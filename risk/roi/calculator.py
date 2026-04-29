"""ROI, BB/100, and variance confidence interval calculators."""
from __future__ import annotations
import math
from scipy.stats import norm


class ROICalculator:
    """Tournament ROI and in-the-money rate calculator."""

    def roi(self, buyins: list[float], cashes: list[float]) -> float:
        """Return ROI as a decimal (e.g. 0.40 = 40% ROI).

        buyins: buy-in cost per tournament (including rake).
        cashes: prize won per tournament (0 if no cash).
        """
        if not buyins:
            raise ValueError("buyins must not be empty")
        if len(buyins) != len(cashes):
            raise ValueError("buyins and cashes must have the same length")
        total_invested = sum(buyins)
        if total_invested == 0:
            raise ValueError("total buy-in cannot be zero")
        return (sum(cashes) - total_invested) / total_invested

    def itm_rate(self, cashes: list[float]) -> float:
        """Return fraction of tournaments where player cashed (result > 0)."""
        if not cashes:
            raise ValueError("cashes must not be empty")
        return sum(1 for c in cashes if c > 0) / len(cashes)


class CashGameStats:
    """BB/100 and win-rate statistics for cash game sessions."""

    def bb_per_100(
        self,
        winnings_bb: float | None = None,
        hands: int = 0,
        *,
        winnings_dollars: float | None = None,
        big_blind_dollars: float | None = None,
    ) -> float:
        """Return win rate in BB/100 hands.

        Pass either:
          - winnings_bb + hands  (already in big blinds)
          - winnings_dollars + hands + big_blind_dollars  (auto-converts)
        """
        if hands <= 0:
            raise ValueError("hands must be positive")

        if winnings_bb is not None:
            bb = winnings_bb
        elif winnings_dollars is not None and big_blind_dollars is not None:
            if big_blind_dollars <= 0:
                raise ValueError("big_blind_dollars must be positive")
            bb = winnings_dollars / big_blind_dollars
        else:
            raise ValueError(
                "Provide either winnings_bb or both winnings_dollars and big_blind_dollars"
            )

        return bb / hands * 100


class VarianceCI:
    """Confidence interval for true win rate given observed sample."""

    def interval(
        self,
        winrate_per_100: float,
        std_per_100: float,
        hands: int,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Return (lower, upper) confidence interval for true win rate.

        winrate_per_100: observed win rate in BB/100.
        std_per_100: standard deviation in BB/100 (as reported by trackers).
        hands: sample size in hands.
        confidence: confidence level, default 0.95 (two-sided).

        Both winrate and std must be in the same unit (BB/100).
        """
        if hands <= 0:
            raise ValueError("hands must be positive")
        if std_per_100 < 0:
            raise ValueError("std_per_100 must be non-negative")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")

        z = norm.ppf(1 - (1 - confidence) / 2)
        # std_per_100 is per 100 hands; convert sample size to units of 100 hands
        standard_error = std_per_100 / math.sqrt(hands / 100)
        margin = z * standard_error
        return (winrate_per_100 - margin, winrate_per_100 + margin)

    def hands_for_margin(
        self,
        target_margin: float,
        std_per_100: float,
        confidence: float = 0.95,
    ) -> int:
        """Return hands needed to achieve a given CI half-width (BB/100)."""
        if target_margin <= 0:
            raise ValueError("target_margin must be positive")
        z = norm.ppf(1 - (1 - confidence) / 2)
        # margin = z * std / sqrt(n/100)  =>  n = (z * std / margin)^2 * 100
        return math.ceil((z * std_per_100 / target_margin) ** 2 * 100)
