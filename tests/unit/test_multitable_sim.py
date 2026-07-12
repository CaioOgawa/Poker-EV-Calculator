import math

import pytest

from simulation.scenarios.multitable import MultitableSim, MultitableResult


def test_simulate_returns_result():
    sim = MultitableSim()
    result = sim.simulate(
        num_tables=3, winrate_per_100=5, std_per_100=80, correlation=0.3, num_trials=2000, seed=0
    )
    assert isinstance(result, MultitableResult)
    assert result.trials == 2000


def test_mean_independent_of_correlation():
    # Correlation only reshapes variance, not the expected combined winrate —
    # but a fully-correlated table has ~2x the combined std of an
    # independent one, so its sample mean is noisier for the same trial
    # count; tolerance is set relative to that expected standard error
    # (combined_std / sqrt(num_trials) ~= 2.25 here) rather than an
    # arbitrarily tight absolute bound.
    sim = MultitableSim()
    low = sim.simulate(
        num_tables=4, winrate_per_100=5, std_per_100=80, correlation=0.0, num_trials=20_000, seed=0
    )
    high = sim.simulate(
        num_tables=4, winrate_per_100=5, std_per_100=80, correlation=1.0, num_trials=20_000, seed=0
    )
    assert abs(low.combined_mean - high.combined_mean) < 8.0


def test_higher_correlation_increases_combined_std():
    sim = MultitableSim()
    low = sim.simulate(
        num_tables=4, winrate_per_100=5, std_per_100=80, correlation=0.0, num_trials=5000, seed=0
    )
    high = sim.simulate(
        num_tables=4, winrate_per_100=5, std_per_100=80, correlation=0.9, num_trials=5000, seed=0
    )
    assert high.combined_std > low.combined_std


def test_percentiles_are_ordered():
    sim = MultitableSim()
    result = sim.simulate(
        num_tables=3, winrate_per_100=5, std_per_100=80, correlation=0.4, num_trials=2000, seed=0
    )
    assert result.p5 <= result.p50 <= result.p95


def test_single_table_ignores_correlation():
    sim = MultitableSim()
    a = sim.simulate(
        num_tables=1, winrate_per_100=5, std_per_100=80, correlation=0.0, num_trials=2000, seed=1
    )
    b = sim.simulate(
        num_tables=1, winrate_per_100=5, std_per_100=80, correlation=1.0, num_trials=2000, seed=1
    )
    assert abs(a.combined_std - b.combined_std) < 1.0


def test_rejects_num_tables_below_one():
    sim = MultitableSim()
    with pytest.raises(ValueError):
        sim.simulate(num_tables=0, winrate_per_100=5, std_per_100=80, correlation=0.0)


def test_rejects_correlation_out_of_range():
    sim = MultitableSim()
    with pytest.raises(ValueError):
        sim.simulate(num_tables=2, winrate_per_100=5, std_per_100=80, correlation=1.5)
    with pytest.raises(ValueError):
        sim.simulate(num_tables=2, winrate_per_100=5, std_per_100=80, correlation=-0.1)


def test_rejects_nonpositive_num_trials():
    sim = MultitableSim()
    with pytest.raises(ValueError):
        sim.simulate(num_tables=2, winrate_per_100=5, std_per_100=80, correlation=0.0, num_trials=0)


def test_reproducible_with_seed():
    sim = MultitableSim()
    r1 = sim.simulate(
        num_tables=3, winrate_per_100=5, std_per_100=80, correlation=0.3, num_trials=500, seed=9
    )
    r2 = sim.simulate(
        num_tables=3, winrate_per_100=5, std_per_100=80, correlation=0.3, num_trials=500, seed=9
    )
    assert r1 == r2


# ------------------------------------------------------------------
# Closed-form validation — no external reference solver exists for this
# model, so its boundary cases (full independence, full correlation) are
# checked against the exact analytic variance formula instead.
# ------------------------------------------------------------------


@pytest.mark.slow
def test_rho_zero_matches_independent_variance_formula():
    sim = MultitableSim()
    num_tables, wr, std, hands = 4, 5.0, 80.0, 100
    sigma = std * math.sqrt(hands / 100)
    result = sim.simulate(
        num_tables=num_tables,
        winrate_per_100=wr,
        std_per_100=std,
        correlation=0.0,
        hands_per_session=hands,
        num_trials=200_000,
        seed=1,
    )
    expected_std = sigma * math.sqrt(num_tables)
    assert abs(result.combined_std - expected_std) / expected_std < 0.02


@pytest.mark.slow
def test_rho_one_matches_fully_correlated_variance_formula():
    sim = MultitableSim()
    num_tables, wr, std, hands = 4, 5.0, 80.0, 100
    sigma = std * math.sqrt(hands / 100)
    result = sim.simulate(
        num_tables=num_tables,
        winrate_per_100=wr,
        std_per_100=std,
        correlation=1.0,
        hands_per_session=hands,
        num_trials=200_000,
        seed=1,
    )
    expected_std = sigma * num_tables
    assert abs(result.combined_std - expected_std) / expected_std < 0.02


@pytest.mark.slow
def test_rho_half_matches_equicorrelated_variance_formula():
    sim = MultitableSim()
    num_tables, wr, std, hands, rho = 4, 5.0, 80.0, 100, 0.5
    sigma = std * math.sqrt(hands / 100)
    result = sim.simulate(
        num_tables=num_tables,
        winrate_per_100=wr,
        std_per_100=std,
        correlation=rho,
        hands_per_session=hands,
        num_trials=200_000,
        seed=1,
    )
    # Var(sum of n equicorrelated X) = n*sigma^2*(1 + (n-1)*rho)
    expected_var = num_tables * sigma**2 * (1 + (num_tables - 1) * rho)
    expected_std = math.sqrt(expected_var)
    assert abs(result.combined_std - expected_std) / expected_std < 0.02
