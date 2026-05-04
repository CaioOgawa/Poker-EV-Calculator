"""Tests for BankrollSimulator and StakeRecommender."""
import pytest
import numpy as np
from risk.bankroll import (
    BankrollSimulator,
    SimResult,
    StakeRecommender,
    StakeRecommendation,
)

# Small params so tests stay fast (mirrors equity_iterations=1500 pattern)
SIM = BankrollSimulator()
CAREERS = 500
SESSIONS = 50
START = 100.0  # BBs


# ------------------------------------------------------------------
# BankrollSimulator
# ------------------------------------------------------------------

def _sim(**kwargs):
    defaults = dict(
        winrate_per_100=5.0,
        std_per_100=80.0,
        starting_bankroll=START,
        num_sessions=SESSIONS,
        num_careers=CAREERS,
        seed=42,
    )
    defaults.update(kwargs)
    return SIM.simulate(**defaults)


def test_sim_returns_simresult():
    assert isinstance(_sim(), SimResult)


def test_sim_hands_array_length():
    result = _sim()
    assert len(result.hands) == SESSIONS + 1


def test_sim_hands_starts_at_zero():
    result = _sim()
    assert result.hands[0] == 0


def test_sim_hands_ends_at_sessions_times_hands_per_session():
    result = _sim(hands_per_session=100)
    assert result.hands[-1] == SESSIONS * 100


def test_sim_starts_at_starting_bankroll():
    result = _sim()
    assert result.p5[0] == START
    assert result.p50[0] == START
    assert result.p95[0] == START


def test_sim_percentile_ordering():
    result = _sim()
    assert np.all(result.p5 <= result.p50)
    assert np.all(result.p50 <= result.p95)


def test_sim_positive_winrate_median_grows():
    result = _sim(winrate_per_100=10.0, std_per_100=40.0, num_careers=1000)
    assert result.p50[-1] > START


def test_sim_ruin_rate_in_range():
    result = _sim()
    assert 0.0 <= result.ruin_rate <= 1.0


def test_sim_high_winrate_low_ruin():
    result = _sim(winrate_per_100=20.0, std_per_100=30.0, num_careers=1000)
    assert result.ruin_rate < 0.10


def test_sim_strongly_negative_winrate_high_ruin():
    result = _sim(winrate_per_100=-20.0, std_per_100=80.0, num_careers=1000, seed=0)
    assert result.ruin_rate > 0.5


def test_sim_reproducible_with_seed():
    r1 = _sim(seed=7)
    r2 = _sim(seed=7)
    np.testing.assert_array_equal(r1.p50, r2.p50)


def test_sim_different_seeds_differ():
    r1 = _sim(seed=1)
    r2 = _sim(seed=2)
    assert not np.array_equal(r1.p50, r2.p50)


def test_sim_mean_final_is_float():
    result = _sim()
    assert isinstance(result.mean_final, float)


# ------------------------------------------------------------------
# StakeRecommender
# ------------------------------------------------------------------

REC = StakeRecommender()
BANKROLL = 500.0   # dollars
WR = 5.0           # BB/100
STD = 80.0         # BB/100


def test_recommend_returns_recommendation():
    r = REC.recommend(BANKROLL, WR, STD)
    assert isinstance(r, StakeRecommendation)


def test_recommend_ror_within_limit():
    r = REC.recommend(BANKROLL, WR, STD, max_ror=0.05)
    assert r.ror <= 0.05


def test_recommend_higher_bankroll_higher_or_equal_stake():
    r_low = REC.recommend(100.0, WR, STD)
    r_high = REC.recommend(5000.0, WR, STD)
    assert r_high.bb_size >= r_low.bb_size


def test_recommend_negative_winrate_raises():
    with pytest.raises(ValueError):
        REC.recommend(BANKROLL, -1.0, STD)


def test_recommend_zero_winrate_raises():
    with pytest.raises(ValueError):
        REC.recommend(BANKROLL, 0.0, STD)


def test_recommend_ev_per_hour_positive():
    r = REC.recommend(BANKROLL, WR, STD)
    assert r.ev_per_hour > 0


def test_recommend_buy_ins_positive():
    r = REC.recommend(BANKROLL, WR, STD)
    assert r.buy_ins > 0


def test_all_stakes_length():
    from risk.bankroll import STANDARD_STAKES
    analysis = REC.all_stakes_analysis(BANKROLL, WR, STD)
    assert len(analysis) == len(STANDARD_STAKES)


def test_all_stakes_ror_increases_with_stake():
    analysis = REC.all_stakes_analysis(BANKROLL, WR, STD)
    rors = [r.ror for r in analysis]
    # Higher stakes → larger BB → smaller bankroll_in_bb → higher RoR
    assert rors == sorted(rors)


def test_all_stakes_returns_list_of_recommendations():
    analysis = REC.all_stakes_analysis(BANKROLL, WR, STD)
    assert all(isinstance(r, StakeRecommendation) for r in analysis)


def test_recommend_no_safe_stake_returns_lowest():
    # Tiny bankroll, high variance → nothing safe → falls back to NL2
    r = REC.recommend(1.0, WR, STD, max_ror=0.0001)
    assert r.stake_name == "NL2"
