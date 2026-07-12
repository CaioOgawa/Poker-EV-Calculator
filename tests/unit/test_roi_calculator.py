"""Tests for ROICalculator, CashGameStats, VarianceCI.

All reference values computed by hand and verified before being hardcoded.
"""

import math
import pytest
from risk.roi.calculator import ROICalculator, CashGameStats, VarianceCI

roi_calc = ROICalculator()
cash = CashGameStats()
vci = VarianceCI()


# ------------------------------------------------------------------
# ROICalculator
# ------------------------------------------------------------------


def test_roi_positive():
    # Invested 500, won 700 → ROI = (700-500)/500 = 0.40
    buyins = [100, 100, 100, 100, 100]
    cashes = [0, 0, 200, 0, 500]
    assert abs(roi_calc.roi(buyins, cashes) - 0.40) < 1e-9


def test_roi_negative():
    buyins = [100, 100, 100]
    cashes = [0, 0, 0]
    assert roi_calc.roi(buyins, cashes) == -1.0


def test_roi_breakeven():
    buyins = [100, 100]
    cashes = [150, 50]
    assert abs(roi_calc.roi(buyins, cashes)) < 1e-9


def test_roi_length_mismatch_raises():
    with pytest.raises(ValueError):
        roi_calc.roi([100, 100], [0])


def test_roi_empty_raises():
    with pytest.raises(ValueError):
        roi_calc.roi([], [])


def test_itm_rate():
    cashes = [0, 0, 200, 0, 500]
    assert abs(roi_calc.itm_rate(cashes) - 0.4) < 1e-9


def test_itm_all_miss():
    assert roi_calc.itm_rate([0, 0, 0]) == 0.0


def test_itm_all_cash():
    assert roi_calc.itm_rate([100, 200, 50]) == 1.0


# ------------------------------------------------------------------
# CashGameStats — BB/100
# ------------------------------------------------------------------


def test_bb_per_100_from_bb():
    # 750 bb won over 100k hands = 0.75 bb/100
    result = cash.bb_per_100(winnings_bb=750.0, hands=100_000)
    assert abs(result - 0.75) < 1e-9


def test_bb_per_100_from_dollars():
    # $150 at NL2 (bb=$2) over 10k hands
    # 150/2 = 75 bb; 75/10000*100 = 0.75
    result = cash.bb_per_100(winnings_dollars=150.0, hands=10_000, big_blind_dollars=2.0)
    assert abs(result - 0.75) < 1e-9


def test_bb_per_100_negative_winrate():
    result = cash.bb_per_100(winnings_bb=-200.0, hands=10_000)
    assert result < 0


def test_bb_per_100_missing_args_raises():
    with pytest.raises(ValueError):
        cash.bb_per_100(hands=1000)


def test_bb_per_100_zero_hands_raises():
    with pytest.raises(ValueError):
        cash.bb_per_100(winnings_bb=100.0, hands=0)


# ------------------------------------------------------------------
# VarianceCI
# ------------------------------------------------------------------


def test_ci_95_wide_sample():
    # wr=6, std=100, 100k hands
    # margin = 1.96 * 100 / sqrt(1000) = 196 / 31.623 ≈ 6.198
    low, high = vci.interval(6.0, 100.0, 100_000)
    assert abs(low - -0.198) < 0.01
    assert abs(high - 12.198) < 0.01


def test_ci_narrows_with_more_hands():
    lo1, hi1 = vci.interval(5.0, 80.0, 10_000)
    lo2, hi2 = vci.interval(5.0, 80.0, 100_000)
    assert (hi1 - lo1) > (hi2 - lo2)


def test_ci_90_narrower_than_95():
    lo95, hi95 = vci.interval(5.0, 80.0, 50_000, confidence=0.95)
    lo90, hi90 = vci.interval(5.0, 80.0, 50_000, confidence=0.90)
    assert (hi95 - lo95) > (hi90 - lo90)


def test_ci_symmetric_around_winrate():
    wr = 3.5
    lo, hi = vci.interval(wr, 90.0, 30_000)
    assert abs((lo + hi) / 2 - wr) < 1e-9


def test_ci_invalid_confidence_raises():
    with pytest.raises(ValueError):
        vci.interval(5.0, 80.0, 10_000, confidence=1.5)


def test_hands_for_margin():
    from scipy.stats import norm as _norm

    # std=100, target margin=1 bb/100 at 95%
    n = vci.hands_for_margin(target_margin=1.0, std_per_100=100.0)
    z = _norm.ppf(0.975)
    expected = math.ceil((z * 100 / 1.0) ** 2 * 100)
    assert n == expected
    assert n > 1_000_000  # sanity: need millions of hands for ±1 bb/100


def test_hands_for_margin_tighter_needs_more():
    n1 = vci.hands_for_margin(target_margin=2.0, std_per_100=100.0)
    n2 = vci.hands_for_margin(target_margin=1.0, std_per_100=100.0)
    assert n2 > n1
