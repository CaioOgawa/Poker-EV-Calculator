import random

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


def test_run_ev_p5_p95_bracket_mean():
    sim = MonteCarloSim(iterations=5_000, seed=0)
    result = sim.run_ev(_ev_fn, x=0.0)
    assert result.p5 <= result.mean_ev <= result.p95


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


def test_run_ev_parallel_mean_close_to_sequential():
    # Different sampling pattern (per-call sub-seeds vs one evolving stream),
    # so results won't match exactly — but should agree on the underlying
    # distribution's mean within Monte Carlo noise.
    seq = MonteCarloSim(iterations=10_000, seed=0).run_ev(_ev_fn, x=10.0)
    par = MonteCarloSim(iterations=10_000, seed=0).run_ev(_ev_fn, n_jobs=2, x=10.0)
    assert abs(seq.mean_ev - par.mean_ev) < 0.1
