"""Monte Carlo runner for EV distributions and variance analysis."""

from __future__ import annotations
import random
from dataclasses import dataclass, field

from joblib import Parallel, delayed


@dataclass
class MonteCarloResult:
    iterations: int
    mean_ev: float
    std_dev: float
    p5: float
    p95: float
    raw: list[float] = field(default_factory=list)


def _run_one_seeded(ev_fn, seed: int | None, kwargs: dict) -> float:
    """Run one `ev_fn(**kwargs)` call under an independent seed of the
    global `random` module. Module-level (not a closure) so it can be
    pickled and sent to joblib's worker processes.
    """
    if seed is None:
        return ev_fn(**kwargs)
    prior_state = random.getstate()
    try:
        random.seed(seed)
        return ev_fn(**kwargs)
    finally:
        random.setstate(prior_state)


class MonteCarloSim:
    def __init__(self, iterations: int = 50_000, seed: int | None = None):
        self.iterations = iterations
        self.seed = seed

    def run_ev(self, ev_fn, n_jobs: int = 1, **kwargs) -> MonteCarloResult:
        """
        Run ev_fn(**kwargs) for N iterations and return distribution stats.
        ev_fn must return a float (single trial EV).

        n_jobs=1 (default) runs sequentially in-process: if a seed was
        provided at construction, the module-level `random` state is seeded
        once for the whole run and restored afterward, so ev_fn can use
        random.random()/random.choice()/etc. reproducibly without this
        instance permanently mutating global random state.

        n_jobs != 1 distributes the `iterations` calls across worker
        processes via joblib (n_jobs=-1 uses all cores). Workers don't share
        memory, so a single evolving random stream isn't possible — instead
        each call gets its own independent sub-seed (`seed + i`) when a seed
        was provided. This means parallel and sequential runs are each
        reproducible on their own, but a given seed does *not* reproduce the
        same `raw` values across different n_jobs settings.
        """
        if n_jobs == 1:
            prior_state = random.getstate()
            try:
                if self.seed is not None:
                    random.seed(self.seed)
                results = [ev_fn(**kwargs) for _ in range(self.iterations)]
            finally:
                random.setstate(prior_state)
        else:
            seeds = [None if self.seed is None else self.seed + i for i in range(self.iterations)]
            results = Parallel(n_jobs=n_jobs)(
                delayed(_run_one_seeded)(ev_fn, seed, kwargs) for seed in seeds
            )
        results_sorted = sorted(results)
        n = len(results_sorted)
        mean = sum(results) / n
        variance = sum((x - mean) ** 2 for x in results) / n
        std = variance**0.5
        return MonteCarloResult(
            iterations=n,
            mean_ev=mean,
            std_dev=std,
            p5=results_sorted[int(n * 0.05)],
            p95=results_sorted[int(n * 0.95)],
            raw=results,
        )
