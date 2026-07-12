import pytest

from simulation.scenarios.cash_session import CashSessionSim, CashSessionResult


def test_simulate_returns_result():
    sim = CashSessionSim()
    result = sim.simulate(winrate_per_100=5, std_per_100=80, hands=200, num_trials=1000, seed=0)
    assert isinstance(result, CashSessionResult)
    assert result.trials == 1000


def test_no_stop_rule_plays_full_session():
    sim = CashSessionSim()
    result = sim.simulate(winrate_per_100=5, std_per_100=80, hands=200, num_trials=1000, seed=0)
    assert result.mean_hands_played == 200.0
    assert result.stopped_early_rate == 0.0


def test_mean_result_matches_winrate():
    sim = CashSessionSim()
    result = sim.simulate(winrate_per_100=5, std_per_100=80, hands=1000, num_trials=20_000, seed=0)
    expected = 5 * 1000 / 100
    assert abs(result.mean_result - expected) < 1.5


def test_std_matches_variance_scaling():
    sim = CashSessionSim()
    result = sim.simulate(winrate_per_100=0, std_per_100=80, hands=400, num_trials=50_000, seed=0)
    expected_std = (80 / 10) * (400**0.5)
    assert abs(result.std_result - expected_std) / expected_std < 0.03


def test_percentiles_are_ordered():
    sim = CashSessionSim()
    result = sim.simulate(winrate_per_100=2, std_per_100=80, hands=300, num_trials=2000, seed=0)
    assert result.p5 <= result.p50 <= result.p95


def test_positive_winrate_reduces_losing_session_rate():
    sim = CashSessionSim()
    winner = sim.simulate(winrate_per_100=15, std_per_100=80, hands=500, num_trials=5000, seed=0)
    breakeven = sim.simulate(winrate_per_100=0, std_per_100=80, hands=500, num_trials=5000, seed=0)
    assert winner.losing_session_rate < breakeven.losing_session_rate


def test_stop_loss_reduces_downside_tail_vs_no_stop_rule():
    # A stop-loss can still overshoot the threshold on the triggering hand
    # (it's a continuous per-hand step, not checked mid-hand), so it doesn't
    # give an exact cap — but it should be far less extreme than letting a
    # 2000-hand session run completely unconstrained.
    sim = CashSessionSim()
    with_stop = sim.simulate(
        winrate_per_100=0, std_per_100=80, hands=2000, num_trials=5000, stop_loss_bb=30, seed=0
    )
    without_stop = sim.simulate(
        winrate_per_100=0, std_per_100=80, hands=2000, num_trials=5000, seed=0
    )
    assert with_stop.p5 > without_stop.p5


def test_stop_win_reduces_upside_tail_vs_no_stop_rule():
    sim = CashSessionSim()
    with_stop = sim.simulate(
        winrate_per_100=0, std_per_100=80, hands=2000, num_trials=5000, stop_win_bb=30, seed=0
    )
    without_stop = sim.simulate(
        winrate_per_100=0, std_per_100=80, hands=2000, num_trials=5000, seed=0
    )
    assert with_stop.p95 < without_stop.p95


def test_stop_rules_trigger_before_hand_cap():
    sim = CashSessionSim()
    result = sim.simulate(
        winrate_per_100=0,
        std_per_100=80,
        hands=5000,
        num_trials=2000,
        stop_loss_bb=20,
        stop_win_bb=20,
        seed=0,
    )
    assert result.stopped_early_rate > 0.9
    assert result.mean_hands_played < 5000


def test_rejects_nonpositive_hands():
    sim = CashSessionSim()
    with pytest.raises(ValueError):
        sim.simulate(winrate_per_100=5, std_per_100=80, hands=0)


def test_rejects_nonpositive_num_trials():
    sim = CashSessionSim()
    with pytest.raises(ValueError):
        sim.simulate(winrate_per_100=5, std_per_100=80, hands=100, num_trials=0)


def test_rejects_nonpositive_stop_loss():
    sim = CashSessionSim()
    with pytest.raises(ValueError):
        sim.simulate(winrate_per_100=5, std_per_100=80, hands=100, stop_loss_bb=-10)


def test_rejects_nonpositive_stop_win():
    sim = CashSessionSim()
    with pytest.raises(ValueError):
        sim.simulate(winrate_per_100=5, std_per_100=80, hands=100, stop_win_bb=0)


def test_reproducible_with_seed():
    sim = CashSessionSim()
    r1 = sim.simulate(winrate_per_100=5, std_per_100=80, hands=100, num_trials=500, seed=42)
    r2 = sim.simulate(winrate_per_100=5, std_per_100=80, hands=100, num_trials=500, seed=42)
    assert r1 == r2
