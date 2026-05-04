"""Bankroll career simulator using Monte Carlo path simulation.

All monetary values are in big blinds (BBs). Dollars appear only when
converting in StakeRecommender. winrate and std must share the same unit
base: BB/100 hands.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SimResult:
    hands: np.ndarray   # x-axis: cumulative hand counts (length num_sessions + 1)
    p5: np.ndarray      # 5th-percentile bankroll curve
    p50: np.ndarray     # median bankroll curve
    p95: np.ndarray     # 95th-percentile bankroll curve
    ruin_rate: float    # fraction of careers that hit ruin_threshold
    mean_final: float   # mean final bankroll across all careers


class BankrollSimulator:
    """Simulate many poker careers and return percentile bankroll curves.

    Each session is modelled as a draw from N(mu, sigma²) where mu and sigma
    are derived from winrate_per_100 and std_per_100 scaled to hands_per_session.
    Ruin is an absorbing barrier: once bankroll ≤ ruin_threshold it stays at 0.
    """

    def simulate(
        self,
        winrate_per_100: float,
        std_per_100: float,
        starting_bankroll: float,
        hands_per_session: int = 100,
        num_sessions: int = 500,
        num_careers: int = 10_000,
        ruin_threshold: float = 0.0,
        seed: int | None = 42,
    ) -> SimResult:
        """Run the simulation and return percentile curves.

        winrate_per_100: expected win rate in BB/100 hands.
        std_per_100: standard deviation in BB/100 hands (from a tracker).
        starting_bankroll: initial bankroll in BBs.
        hands_per_session: hands per simulated session.
        num_sessions: number of sessions per career.
        num_careers: number of independent career paths.
        ruin_threshold: bankroll at or below this is considered ruin (in BBs).
        seed: random seed for reproducibility (None = non-deterministic).
        """
        rng = np.random.default_rng(seed)

        mu = winrate_per_100 * hands_per_session / 100
        sigma = std_per_100 * np.sqrt(hands_per_session / 100)

        sessions = rng.normal(mu, sigma, size=(num_careers, num_sessions))

        # cumulative[:, 0] = starting_bankroll; rest = running sum of sessions
        cumulative = np.empty((num_careers, num_sessions + 1))
        cumulative[:, 0] = starting_bankroll
        cumulative[:, 1:] = starting_bankroll + np.cumsum(sessions, axis=1)

        # Absorbing ruin barrier: once the running minimum drops ≤ threshold,
        # that career is ruined and all subsequent values become 0.
        running_min = np.minimum.accumulate(cumulative, axis=1)
        ruined = running_min <= ruin_threshold
        cumulative = np.where(ruined, 0.0, cumulative)

        hands = np.arange(num_sessions + 1) * hands_per_session

        return SimResult(
            hands=hands,
            p5=np.percentile(cumulative, 5, axis=0),
            p50=np.percentile(cumulative, 50, axis=0),
            p95=np.percentile(cumulative, 95, axis=0),
            ruin_rate=float(np.mean(cumulative[:, -1] <= ruin_threshold)),
            mean_final=float(np.mean(cumulative[:, -1])),
        )

    def simulate_multitable(
        self,
        num_tables: int,
        winrate_per_100: float,
        std_per_100: float,
        starting_bankroll: float,
        hands_per_session: int = 100,
        num_sessions: int = 500,
        num_careers: int = 10_000,
        ruin_threshold: float = 0.0,
        seed: int | None = 42,
    ) -> SimResult:
        """Simulate multitabling by scaling winrate and variance.

        num_tables tables played simultaneously → combined winrate scales
        linearly; combined std scales by sqrt(num_tables) (independent tables).
        """
        import math
        combined_wr = winrate_per_100 * num_tables
        combined_std = std_per_100 * math.sqrt(num_tables)
        return self.simulate(
            winrate_per_100=combined_wr,
            std_per_100=combined_std,
            starting_bankroll=starting_bankroll,
            hands_per_session=hands_per_session,
            num_sessions=num_sessions,
            num_careers=num_careers,
            ruin_threshold=ruin_threshold,
            seed=seed,
        )
