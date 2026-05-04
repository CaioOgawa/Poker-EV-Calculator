"""Stake recommender based on Risk of Ruin at each stake level.

All bankroll inputs in dollars; winrate and std in BB/100.
RoR is computed by converting dollar bankroll → BBs at each stake.
"""
from __future__ import annotations
from dataclasses import dataclass
from risk.bankroll.ror import RiskOfRuin

STANDARD_STAKES: list[tuple[str, float]] = [
    ("NL2",   0.02),
    ("NL5",   0.05),
    ("NL10",  0.10),
    ("NL25",  0.25),
    ("NL50",  0.50),
    ("NL100", 1.00),
    ("NL200", 2.00),
    ("NL500", 5.00),
    ("NL1000", 10.00),
]


@dataclass
class StakeRecommendation:
    stake_name: str       # e.g. "NL50"
    bb_size: float        # big blind in dollars
    bankroll_in_bb: float # bankroll expressed in BBs at this stake
    ror: float            # risk of ruin at this stake (0–1)
    buy_ins: float        # bankroll / (buy_in_bbs * bb_size)
    ev_per_hour: float    # expected $/hour at hands_per_hour


class StakeRecommender:
    """Recommend the highest stake where Risk of Ruin ≤ max_ror.

    If no stake meets the threshold, returns the lowest available stake with
    its actual RoR so the caller can make an informed decision.
    """

    def __init__(self, ror_model: RiskOfRuin | None = None):
        self._ror = ror_model or RiskOfRuin()

    def recommend(
        self,
        bankroll_dollars: float,
        winrate_per_100: float,
        std_per_100: float,
        max_ror: float = 0.05,
        buy_in_bbs: int = 100,
        hands_per_hour: int = 70,
        stakes: list[tuple[str, float]] | None = None,
    ) -> StakeRecommendation:
        """Return the highest safe stake (RoR ≤ max_ror).

        Raises ValueError if winrate_per_100 ≤ 0 (ruin is guaranteed long-term).
        If no stake meets max_ror, returns the lowest stake with its actual RoR.
        """
        if winrate_per_100 <= 0:
            raise ValueError("winrate_per_100 must be positive to recommend a stake")

        available = stakes or STANDARD_STAKES
        best: StakeRecommendation | None = None

        for name, bb_size in available:
            rec = self._build(name, bb_size, bankroll_dollars, winrate_per_100,
                              std_per_100, buy_in_bbs, hands_per_hour)
            if rec.ror <= max_ror:
                best = rec  # keep iterating — higher stakes may still qualify

        if best is None:
            name, bb_size = available[0]
            best = self._build(name, bb_size, bankroll_dollars, winrate_per_100,
                               std_per_100, buy_in_bbs, hands_per_hour)

        return best

    def all_stakes_analysis(
        self,
        bankroll_dollars: float,
        winrate_per_100: float,
        std_per_100: float,
        buy_in_bbs: int = 100,
        hands_per_hour: int = 70,
        stakes: list[tuple[str, float]] | None = None,
    ) -> list[StakeRecommendation]:
        """Return RoR analysis for every available stake."""
        available = stakes or STANDARD_STAKES
        return [
            self._build(name, bb_size, bankroll_dollars, winrate_per_100,
                        std_per_100, buy_in_bbs, hands_per_hour)
            for name, bb_size in available
        ]

    def _build(
        self,
        name: str,
        bb_size: float,
        bankroll_dollars: float,
        winrate_per_100: float,
        std_per_100: float,
        buy_in_bbs: int,
        hands_per_hour: int,
    ) -> StakeRecommendation:
        bankroll_bb = bankroll_dollars / bb_size
        ror = (
            self._ror.continuous(winrate_per_100, std_per_100, bankroll_bb)
            if winrate_per_100 > 0
            else 1.0
        )
        ev_per_hour = winrate_per_100 / 100 * hands_per_hour * bb_size
        return StakeRecommendation(
            stake_name=name,
            bb_size=bb_size,
            bankroll_in_bb=bankroll_bb,
            ror=ror,
            buy_ins=bankroll_bb / buy_in_bbs,
            ev_per_hour=ev_per_hour,
        )
