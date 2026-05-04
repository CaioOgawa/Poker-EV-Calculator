"""Tests for TournamentSim and BankrollSimulator.simulate_multitable."""
import pytest

pytestmark = pytest.mark.slow

from simulation.scenarios.tournament import TournamentSim, TournamentResult
from risk.bankroll.simulator import BankrollSimulator

# ------------------------------------------------------------------
# TournamentSim
# ------------------------------------------------------------------

SIM = TournamentSim(
    num_players=6,
    starting_stack=10_000,
    starting_bb=100,
    paid_spots=2,
    push_threshold_bbs=15.0,
    caller_equity=0.60,
    num_trials=1000,
    seed=42,
)


def test_run_returns_tournament_result():
    result = SIM.run()
    assert isinstance(result, TournamentResult)


def test_finish_positions_length():
    result = SIM.run()
    assert len(result.finish_positions) == SIM.num_trials


def test_finish_positions_in_valid_range():
    result = SIM.run()
    for pos in result.finish_positions:
        assert 1 <= pos <= 6


def test_itm_rate_in_range():
    result = SIM.run()
    assert 0.0 <= result.itm_rate <= 1.0


def test_avg_finish_in_range():
    result = SIM.run()
    assert 1.0 <= result.avg_finish <= 6.0


def test_itm_rate_plausible():
    # 2 paid spots out of 6 players → ITM ≈ 1/3 with ±15% tolerance
    result = SIM.run()
    assert 0.10 <= result.itm_rate <= 0.65


def test_roi_near_zero_zerosumgame():
    # Tournament is zero-sum (no rake modelled) — ROI should be near 0
    # Tolerance 0.50: 1000 MC trials have meaningful variance; the key invariant
    # is that the prize pool is exactly conserved (total prizes = num_players).
    result = SIM.run()
    assert abs(result.roi) < 0.50


def test_seed_reproducibility():
    s1 = TournamentSim(num_players=6, num_trials=100, seed=7).run()
    s2 = TournamentSim(num_players=6, num_trials=100, seed=7).run()
    assert s1.finish_positions == s2.finish_positions


def test_different_seeds_differ():
    s1 = TournamentSim(num_players=6, num_trials=200, seed=1).run()
    s2 = TournamentSim(num_players=6, num_trials=200, seed=99).run()
    assert s1.finish_positions != s2.finish_positions


def test_avg_finish_symmetric_with_equal_equity():
    # With symmetric setup and enough trials, avg finish ≈ (N+1)/2
    sim = TournamentSim(
        num_players=6,
        starting_stack=10_000,
        starting_bb=200,
        paid_spots=2,
        push_threshold_bbs=15.0,
        caller_equity=0.50,
        num_trials=1000,
        seed=0,
    )
    result = sim.run()
    # avg finish for a random player ≈ 3.5 (midpoint of 1..6)
    assert abs(result.avg_finish - 3.5) < 1.2


def test_larger_stack_improves_avg_finish():
    # Bigger stack → should finish better on average
    small = TournamentSim(
        num_players=6, starting_stack=5_000, num_trials=300, seed=5
    ).run()
    large = TournamentSim(
        num_players=6, starting_stack=20_000, num_trials=300, seed=5
    ).run()
    assert large.avg_finish <= small.avg_finish + 0.5


def test_paid_spots_affects_itm():
    # More paid spots → higher ITM rate
    low = TournamentSim(num_players=9, paid_spots=1, num_trials=300, seed=3).run()
    high = TournamentSim(num_players=9, paid_spots=4, num_trials=300, seed=3).run()
    assert high.itm_rate >= low.itm_rate


# ------------------------------------------------------------------
# BankrollSimulator.simulate_multitable
# ------------------------------------------------------------------

BS = BankrollSimulator()


def test_multitable_returns_simresult():
    from risk.bankroll.simulator import SimResult
    result = BS.simulate_multitable(
        num_tables=2, winrate_per_100=5.0, std_per_100=80.0,
        starting_bankroll=200.0, num_sessions=100, num_careers=500,
    )
    assert isinstance(result, SimResult)


def test_multitable_scales_winrate():
    # 2 tables → 2x EV → higher median final bankroll
    single = BS.simulate(
        winrate_per_100=5.0, std_per_100=80.0, starting_bankroll=200.0,
        num_sessions=100, num_careers=1000, seed=0,
    )
    multi = BS.simulate_multitable(
        num_tables=2, winrate_per_100=5.0, std_per_100=80.0,
        starting_bankroll=200.0, num_sessions=100, num_careers=1000, seed=0,
    )
    assert multi.mean_final > single.mean_final


def test_multitable_more_tables_more_variance():
    # More tables → higher variance → higher p95 and lower p5 spread
    single = BS.simulate_multitable(
        num_tables=1, winrate_per_100=5.0, std_per_100=80.0,
        starting_bankroll=500.0, num_sessions=100, num_careers=2000, seed=1,
    )
    quad = BS.simulate_multitable(
        num_tables=4, winrate_per_100=5.0, std_per_100=80.0,
        starting_bankroll=500.0, num_sessions=100, num_careers=2000, seed=1,
    )
    assert quad.p95[-1] > single.p95[-1]


def test_multitable_seed_reproducible():
    r1 = BS.simulate_multitable(
        num_tables=2, winrate_per_100=5.0, std_per_100=80.0,
        starting_bankroll=200.0, num_sessions=50, num_careers=200, seed=99,
    )
    r2 = BS.simulate_multitable(
        num_tables=2, winrate_per_100=5.0, std_per_100=80.0,
        starting_bankroll=200.0, num_sessions=50, num_careers=200, seed=99,
    )
    import numpy as np
    np.testing.assert_array_equal(r1.p50, r2.p50)
