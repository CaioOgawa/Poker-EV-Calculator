"""Multitabling simulator with correlated tables.

`BankrollSimulator.simulate_multitable` already scales winrate/std for N
simultaneous tables, but it assumes the tables are fully *independent*
(combined std scales by sqrt(N)). Real multitabling isn't independent —
the same player's tilt, fatigue, or table selection affects every table
they're playing at once. This module models that with a correlation
coefficient `rho` between any two simultaneous tables, so the combined
variance sits somewhere between the independent case (rho=0) and the fully
correlated case (rho=1, no diversification benefit at all — equivalent to
one table run N times bigger).

Tables are modelled as equicorrelated (every pair of tables shares the same
`rho`) via the standard common-factor construction:
    X_i = sqrt(rho) * Z_common + sqrt(1 - rho) * Z_i
with Z_common and each Z_i drawn iid N(0, 1) and scaled by sigma. This
gives each table the right marginal variance and every pair the right
covariance by construction — at rho=0 combined std reduces exactly to
BankrollSimulator's sqrt(N) formula, and at rho=1 it reduces exactly to N
(both checked in tests, since there's no external reference solver for
this).
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class MultitableResult:
    trials: int
    combined_mean: float
    combined_std: float
    p5: float
    p50: float
    p95: float


class MultitableSim:
    def simulate(
        self,
        num_tables: int,
        winrate_per_100: float,
        std_per_100: float,
        correlation: float,
        hands_per_session: int = 100,
        num_trials: int = 10_000,
        seed: int | None = 42,
    ) -> MultitableResult:
        """Simulate one session's combined result across `num_tables`
        simultaneous, equicorrelated tables.

        winrate_per_100, std_per_100: per-table stats in BB/100 hands.
        correlation: pairwise correlation between any two tables, in [0, 1].
        hands_per_session: hands played per table in this session.
        """
        if num_tables < 1:
            raise ValueError("num_tables must be at least 1")
        if not 0.0 <= correlation <= 1.0:
            raise ValueError(f"correlation must be in [0, 1], got {correlation}")
        if num_trials <= 0:
            raise ValueError("num_trials must be positive")

        rng = np.random.default_rng(seed)
        mu = winrate_per_100 * hands_per_session / 100
        sigma = std_per_100 * math.sqrt(hands_per_session / 100)

        z_common = rng.normal(0.0, 1.0, size=num_trials)
        z_indep = rng.normal(0.0, 1.0, size=(num_trials, num_tables))
        table_results = mu + sigma * (
            math.sqrt(correlation) * z_common[:, None] + math.sqrt(1 - correlation) * z_indep
        )
        combined = table_results.sum(axis=1)

        return MultitableResult(
            trials=num_trials,
            combined_mean=float(np.mean(combined)),
            combined_std=float(np.std(combined)),
            p5=float(np.percentile(combined, 5)),
            p50=float(np.percentile(combined, 50)),
            p95=float(np.percentile(combined, 95)),
        )
