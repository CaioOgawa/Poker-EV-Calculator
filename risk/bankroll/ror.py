"""Risk of Ruin calculator."""

from __future__ import annotations
import math


class RiskOfRuin:
    def continuous(self, win_rate: float, std_dev: float, bankroll: float) -> float:
        """
        Continuous RoR formula: e^(-2 * win_rate * bankroll / std_dev^2)
        win_rate and std_dev must be in same units (BB/100 or $/hr etc.)
        """
        if std_dev <= 0 or win_rate <= 0:
            return 1.0
        return math.exp(-2 * win_rate * bankroll / (std_dev**2))

    def required_bankroll(self, win_rate: float, std_dev: float, target_ror: float = 0.01) -> float:
        """Bankroll needed to achieve a target risk of ruin."""
        if win_rate <= 0:
            return float("inf")
        return (-math.log(target_ror) * std_dev**2) / (2 * win_rate)
