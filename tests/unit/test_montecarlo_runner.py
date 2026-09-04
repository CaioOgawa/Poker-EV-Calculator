import random
import statistics

import numpy as np
import pytest

from simulation.montecarlo.runner import MonteCarloSim, MonteCarloResult


def _ev_fn(x: float) -> float:
    return x + random.random()


def test_run_ev_returns_montecarlo_result():
    sim = MonteCarloSim(iterations=500, seed=0)
    result = sim.run_ev(_ev_fn, x=10.0)
    assert isinstance(result, MonteCarloResult)
    assert result.iterations == 500


def test_run_ev_mean_matches_expected():
    # x + uniform(0,1) -> mean ~= x + 0.5
    sim = MonteCarloSim(iterations=20_000, seed=0)
    result = sim.run_ev(_ev_fn, x=10.0)
    assert abs(result.mean_ev - 10.5) < 0.05


def test_run_ev_sequential_reproducible_with_seed():
    r1 = MonteCarloSim(iterations=500, seed=7).run_ev(_ev_fn, x=1.0)
    r2 = MonteCarloSim(iterations=500, seed=7).run_ev(_ev_fn, x=1.0)
    assert r1.raw == r2.raw


def test_run_ev_sequential_different_seeds_differ():
    r1 = MonteCarloSim(iterations=500, seed=1).run_ev(_ev_fn, x=1.0)
    r2 = MonteCarloSim(iterations=500, seed=2).run_ev(_ev_fn, x=1.0)
    assert r1.raw != r2.raw


def test_run_ev_does_not_leak_global_random_state():
    prior = random.getstate()
    MonteCarloSim(iterations=200, seed=123).run_ev(_ev_fn, x=1.0)
    assert random.getstate() == prior


def test_run_ev_keep_raw_false_drops_raw_but_keeps_stats():
    sim = MonteCarloSim(iterations=500, seed=0)
    result = sim.run_ev(_ev_fn, keep_raw=False, x=10.0)
    assert result.raw == []
    assert result.iterations == 500
    assert abs(result.mean_ev - 10.5) < 0.1


def test_run_ev_p5_p95_bracket_mean():
    sim = MonteCarloSim(iterations=5_000, seed=0)
    result = sim.run_ev(_ev_fn, x=0.0)
    assert result.p5 <= result.mean_ev <= result.p95


# ------------------------------------------------------------------
# sample stats, not population (docs/AUDITORIA-2026-08-26.md item E8)
# ------------------------------------------------------------------


def test_run_ev_std_dev_is_sample_not_population():
    # /n underestimates spread vs. the unbiased /(n-1) estimator — the two
    # definitions diverge visibly at small n, so this is a real behavior
    # change to the reported number, not noise.
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    it = iter(values)
    sim = MonteCarloSim(iterations=len(values))
    result = sim.run_ev(lambda: next(it))
    assert result.std_dev == pytest.approx(statistics.stdev(values))
    assert result.std_dev != pytest.approx(statistics.pstdev(values))


def test_run_ev_std_dev_zero_for_single_iteration():
    # ddof=1 is undefined at n<=1 (no spread to estimate from one point) —
    # falls back to 0.0 rather than propagating NaN.
    sim = MonteCarloSim(iterations=1)
    result = sim.run_ev(lambda: 5.0)
    assert result.std_dev == 0.0


def test_run_ev_percentiles_use_interpolation():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    it = iter(values)
    sim = MonteCarloSim(iterations=len(values))
    result = sim.run_ev(lambda: next(it))
    assert result.p5 == pytest.approx(np.percentile(values, 5))
    assert result.p95 == pytest.approx(np.percentile(values, 95))
    # The old truncated-index lookup gave sorted[int(10*0.05)] == sorted[0]
    # == 1.0 exactly; interpolation lands strictly above the minimum.
    assert result.p5 > 1.0


# ------------------------------------------------------------------
# n_jobs parallel path
# ------------------------------------------------------------------


def test_run_ev_parallel_returns_same_iteration_count():
    sim = MonteCarloSim(iterations=300, seed=0)
    result = sim.run_ev(_ev_fn, n_jobs=2, x=5.0)
    assert result.iterations == 300


def test_run_ev_parallel_reproducible_with_seed():
    r1 = MonteCarloSim(iterations=300, seed=11).run_ev(_ev_fn, n_jobs=2, x=1.0)
    r2 = MonteCarloSim(iterations=300, seed=11).run_ev(_ev_fn, n_jobs=2, x=1.0)
    assert r1.raw == r2.raw


def test_run_ev_parallel_handles_iterations_not_divisible_by_n_jobs():
    # 301 iterations over 4 jobs doesn't divide evenly — every iteration
    # must still land in exactly one chunk.
    sim = MonteCarloSim(iterations=301, seed=3)
    result = sim.run_ev(_ev_fn, n_jobs=4, x=1.0)
    assert result.iterations == 301
    assert len(result.raw) == 301


def test_run_ev_parallel_mean_close_to_sequential():
    # Different sampling pattern (per-call sub-seeds vs one evolving stream),
    # so results won't match exactly — but should agree on the underlying
    # distribution's mean within Monte Carlo noise.
    seq = MonteCarloSim(iterations=10_000, seed=0).run_ev(_ev_fn, x=10.0)
    par = MonteCarloSim(iterations=10_000, seed=0).run_ev(_ev_fn, n_jobs=2, x=10.0)
    assert abs(seq.mean_ev - par.mean_ev) < 0.1
