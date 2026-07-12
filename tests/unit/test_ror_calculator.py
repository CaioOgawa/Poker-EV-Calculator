"""Tests for RiskOfRuin.

Validation strategy: `docs/ROADMAP.md` calls for checking RoR "against
tabulated values (Chen, Mason Malmuth)". Rather than hardcode specific
numbers recalled from those books — which risks enshrining a misremembered
constant as if it were verified — this file validates the closed-form
`continuous()` formula two ways that don't depend on that recall:

1. Exact algebraic self-consistency (`required_bankroll` is the closed-form
   formula's own inverse — this is deterministic, not an approximation).
2. Cross-validation against `BankrollSimulator`, an independently coded
   Monte Carlo path simulator already in this repo (numpy cumulative sums +
   an absorbing floor) that models the same underlying process (a random
   walk with drift, absorbed at 0) a different way. Both the closed form
   and the simulator rest on the same random-walk-converges-to-Brownian-
   motion argument, so agreement is meaningful, not circular — and the
   ~5-10% gap actually observed is explained by discrete monitoring: the
   simulator only checks for ruin at session boundaries, so a path that
   dips below zero and recovers mid-session isn't caught, which understates
   its ruin rate relative to the continuously-monitored closed form. That's
   why the tolerance below is a generous one-sided allowance, not a tight
   symmetric band.
"""

import math

import pytest

from risk.bankroll.ror import RiskOfRuin
from risk.bankroll.simulator import BankrollSimulator


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


def test_continuous_zero_or_negative_winrate_is_certain_ruin():
    ror = RiskOfRuin()
    assert ror.continuous(win_rate=0.0, std_dev=80, bankroll=1000) == 1.0
    assert ror.continuous(win_rate=-1.0, std_dev=80, bankroll=1000) == 1.0


def test_continuous_zero_std_is_certain_ruin():
    ror = RiskOfRuin()
    assert ror.continuous(win_rate=5.0, std_dev=0, bankroll=1000) == 1.0


def test_continuous_decreases_with_bankroll():
    ror = RiskOfRuin()
    small = ror.continuous(win_rate=5.0, std_dev=80.0, bankroll=500)
    large = ror.continuous(win_rate=5.0, std_dev=80.0, bankroll=2000)
    assert large < small


def test_continuous_decreases_with_winrate():
    ror = RiskOfRuin()
    low_wr = ror.continuous(win_rate=2.0, std_dev=80.0, bankroll=1000)
    high_wr = ror.continuous(win_rate=10.0, std_dev=80.0, bankroll=1000)
    assert high_wr < low_wr


def test_continuous_increases_with_std():
    ror = RiskOfRuin()
    low_std = ror.continuous(win_rate=5.0, std_dev=40.0, bankroll=1000)
    high_std = ror.continuous(win_rate=5.0, std_dev=120.0, bankroll=1000)
    assert high_std > low_std


def test_continuous_bounded_in_zero_one():
    ror = RiskOfRuin()
    for bankroll in [1, 100, 10_000]:
        r = ror.continuous(win_rate=5.0, std_dev=80.0, bankroll=bankroll)
        assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# required_bankroll — exact algebraic inverse of continuous()
# ---------------------------------------------------------------------------


def test_required_bankroll_is_exact_inverse_of_continuous():
    ror = RiskOfRuin()
    for target in [0.01, 0.05, 0.1, 0.3]:
        bankroll = ror.required_bankroll(win_rate=5.0, std_dev=80.0, target_ror=target)
        recovered = ror.continuous(win_rate=5.0, std_dev=80.0, bankroll=bankroll)
        assert abs(recovered - target) < 1e-9


def test_required_bankroll_matches_closed_form_algebra():
    # bankroll = -ln(target) * std^2 / (2 * win_rate), independently recomputed.
    ror = RiskOfRuin()
    wr, std, target = 5.0, 80.0, 0.05
    expected = -math.log(target) * std**2 / (2 * wr)
    assert abs(ror.required_bankroll(wr, std, target) - expected) < 1e-9


def test_required_bankroll_infinite_for_nonpositive_winrate():
    ror = RiskOfRuin()
    assert ror.required_bankroll(win_rate=0.0, std_dev=80.0, target_ror=0.05) == float("inf")


def test_required_bankroll_higher_for_tighter_target():
    ror = RiskOfRuin()
    loose = ror.required_bankroll(win_rate=5.0, std_dev=80.0, target_ror=0.10)
    tight = ror.required_bankroll(win_rate=5.0, std_dev=80.0, target_ror=0.01)
    assert tight > loose


# ---------------------------------------------------------------------------
# Cross-validation against BankrollSimulator's Monte Carlo path simulation
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("bankroll", [700.0, 1030.0, 1470.0])
def test_continuous_matches_bankroll_simulator_ruin_rate(bankroll):
    wr, std = 5.0, 80.0
    closed_form = RiskOfRuin().continuous(wr, std, bankroll)

    mc = BankrollSimulator().simulate(
        winrate_per_100=wr,
        std_per_100=std,
        starting_bankroll=bankroll,
        hands_per_session=15,
        num_sessions=12_000,
        num_careers=8_000,
        seed=1,
    )

    # One-sided: discrete session-boundary monitoring only ever *understates*
    # ruin relative to the continuously-monitored closed form, so the MC
    # estimate should sit at or somewhat below it — not equal, not above.
    assert mc.ruin_rate <= closed_form + 0.01
    assert mc.ruin_rate >= closed_form * 0.75
