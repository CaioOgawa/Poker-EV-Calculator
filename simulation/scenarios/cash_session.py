"""Single cash-game session simulator, hand-by-hand, with optional
stop-loss / stop-win quit rules.

This sits between two things the project already has: `VarianceCI` (an
analytic confidence interval for the *true* win rate given an *observed*
sample — inference, not simulation) and `BankrollSimulator` (a career-level
walk across many sessions with an absorbing ruin barrier). Neither answers
"if I sit down for N hands today, what does the result distribution for
*this session* look like, especially if I quit early on a stop-loss or
stop-win?" — a stop rule breaks the plain Gaussian closed form (VarianceCI
can't express it), and it's a within-session question BankrollSimulator's
session-level ruin model doesn't address.

All monetary values are in big blinds (BBs), matching the winrate/std
convention used across `risk/`. Each hand's result is modelled as an iid
N(mu_per_hand, sigma_per_hand) draw — the same Gaussian-per-unit assumption
`BankrollSimulator` makes at the session level, just applied one level
finer. Real single-hand results are nowhere near Gaussian (mostly zero,
occasional large swings), so treat this as a coarse approximation, most
trustworthy for the aggregate (session-level) statistics it reports rather
than any individual simulated hand.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class CashSessionResult:
    trials: int
    mean_result: float
    std_result: float
    p5: float
    p50: float
    p95: float
    losing_session_rate: float
    mean_hands_played: float
    stopped_early_rate: float


class CashSessionSim:
    def simulate(
        self,
        winrate_per_100: float,
        std_per_100: float,
        hands: int,
        num_trials: int = 10_000,
        stop_loss_bb: float | None = None,
        stop_win_bb: float | None = None,
        seed: int | None = 42,
    ) -> CashSessionResult:
        """Simulate `num_trials` independent sessions of up to `hands` hands.

        winrate_per_100: expected win rate in BB/100 hands.
        std_per_100: standard deviation in BB/100 hands (from a tracker).
        hands: hand cap per session.
        stop_loss_bb: if set, a session quits the first time it's down this
            many BBs (a positive number, e.g. 50 means quit at -50bb).
        stop_win_bb: if set, a session quits the first time it's up this
            many BBs.
        seed: random seed for reproducibility (None = non-deterministic).
        """
        if hands <= 0:
            raise ValueError("hands must be positive")
        if num_trials <= 0:
            raise ValueError("num_trials must be positive")
        if stop_loss_bb is not None and stop_loss_bb <= 0:
            raise ValueError("stop_loss_bb must be positive (a loss magnitude)")
        if stop_win_bb is not None and stop_win_bb <= 0:
            raise ValueError("stop_win_bb must be positive")

        rng = np.random.default_rng(seed)
        mu = winrate_per_100 / 100
        sigma = std_per_100 / 10  # std over 100 hands -> per-hand: /sqrt(100)

        increments = rng.normal(mu, sigma, size=(num_trials, hands))
        cumulative = np.cumsum(increments, axis=1)

        hit_stop = np.zeros_like(cumulative, dtype=bool)
        if stop_loss_bb is not None:
            hit_stop |= cumulative <= -stop_loss_bb
        if stop_win_bb is not None:
            hit_stop |= cumulative >= stop_win_bb

        any_hit = hit_stop.any(axis=1)
        first_hit_idx = np.argmax(hit_stop, axis=1)  # 0 when a row has no True
        final_idx = np.where(any_hit, first_hit_idx, hands - 1)
        final_results = cumulative[np.arange(num_trials), final_idx]
        hands_played = final_idx + 1

        return CashSessionResult(
            trials=num_trials,
            mean_result=float(np.mean(final_results)),
            std_result=float(np.std(final_results)),
            p5=float(np.percentile(final_results, 5)),
            p50=float(np.percentile(final_results, 50)),
            p95=float(np.percentile(final_results, 95)),
            losing_session_rate=float(np.mean(final_results < 0)),
            mean_hands_played=float(np.mean(hands_played)),
            stopped_early_rate=float(np.mean(any_hit)),
        )
