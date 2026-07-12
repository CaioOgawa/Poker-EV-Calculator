import math

import pytest

from risk.bankroll.kelly import KellyCriterion


def test_kelly_fraction_positive():
    k = KellyCriterion()
    f = k.fraction(win_prob=0.55, win_odds=1.0)
    assert 0 < f < 1


def test_kelly_fraction_zero_when_negative_ev():
    k = KellyCriterion()
    f = k.fraction(win_prob=0.3, win_odds=1.0)
    assert f == 0.0


# ---------------------------------------------------------------------------
# Validation against the Kelly criterion's defining property.
#
# `docs/ROADMAP.md` calls for validating bankroll math "against tabulated
# values". For Kelly, rather than trust a specific worked example recalled
# from a book, this checks the formula against what Kelly betting is
# *defined* to do: maximize the expected per-bet log-growth rate
#   g(f) = p*ln(1 + f*b) + (1-p)*ln(1 - f)
# which is an exact, independently-derivable closed form (Kelly, 1956) —
# not a simulation, so there's no sampling noise to tolerate. Any correct
# implementation of f* = (p*b - q)/b must sit exactly at g's maximum.
# ---------------------------------------------------------------------------


def _expected_log_growth(f: float, win_prob: float, win_odds: float) -> float:
    return win_prob * math.log(1 + f * win_odds) + (1 - win_prob) * math.log(1 - f)


@pytest.mark.parametrize(
    "win_prob,win_odds",
    [(0.55, 1.0), (0.6, 1.0), (0.4, 2.0), (0.7, 0.5), (0.52, 3.0)],
)
def test_kelly_fraction_is_local_maximum_of_log_growth(win_prob, win_odds):
    k = KellyCriterion()
    f_star = k.fraction(win_prob, win_odds)
    g_star = _expected_log_growth(f_star, win_prob, win_odds)
    for eps in (1e-4, 5e-4, 2e-3):
        assert g_star >= _expected_log_growth(f_star - eps, win_prob, win_odds)
        assert g_star >= _expected_log_growth(f_star + eps, win_prob, win_odds)


@pytest.mark.parametrize(
    "win_prob,win_odds",
    [(0.55, 1.0), (0.4, 2.0), (0.7, 0.5)],
)
def test_kelly_fraction_is_global_maximum_over_fraction_grid(win_prob, win_odds):
    k = KellyCriterion()
    f_star = k.fraction(win_prob, win_odds)
    g_star = _expected_log_growth(f_star, win_prob, win_odds)
    grid_best = max(_expected_log_growth(f / 1000, win_prob, win_odds) for f in range(0, 999))
    assert g_star >= grid_best - 1e-9


def test_kelly_even_money_matches_2p_minus_1_identity():
    # At win_odds=1 (even money), f* = (p - (1-p))/1 = 2p - 1 — a standard
    # simplification of the Kelly formula, safe to assert as exact algebra.
    k = KellyCriterion()
    for p in (0.52, 0.55, 0.6, 0.75):
        assert abs(k.fraction(p, 1.0) - (2 * p - 1)) < 1e-12


def test_kelly_fractional_scales_linearly():
    k = KellyCriterion()
    full = k.fraction(0.6, 1.5, fraction=1.0)
    half = k.fraction(0.6, 1.5, fraction=0.5)
    assert abs(half - full / 2) < 1e-12


def test_kelly_units_matches_fraction_times_bankroll():
    k = KellyCriterion()
    bankroll = 10_000.0
    f = k.fraction(0.6, 1.5, fraction=0.25)
    assert abs(k.units(bankroll, 0.6, 1.5, fraction=0.25) - bankroll * f) < 1e-9
