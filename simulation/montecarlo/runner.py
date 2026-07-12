"""Monte Carlo runner for EV distributions and variance analysis."""

from __future__ import annotations
import random
from dataclasses import dataclass, field


@dataclass
class MonteCarloResult:
    iterations: int
    mean_ev: float
    std_dev: float
    p5: float
    p95: float
    raw: list[float] = field(default_factory=list)


class MonteCarloSim:
    def __init__(self, iterations: int = 50_000, seed: int | None = None):
        self.iterations = iterations
        self.seed = seed

    def run_ev(self, ev_fn, **kwargs) -> MonteCarloResult:
        """
        Run ev_fn(**kwargs) for N iterations and return distribution stats.
        ev_fn must return a float (single trial EV).

        If a seed was provided at construction, the module-level `random`
        state is seeded for the duration of this call only and restored
        afterward, so ev_fn can use random.random()/random.choice()/etc.
        reproducibly without this instance permanently mutating global
        random state or interfering with other code in the same process.
        """
        prior_state = random.getstate()
        try:
            if self.seed is not None:
                random.seed(self.seed)
            results = [ev_fn(**kwargs) for _ in range(self.iterations)]
        finally:
            random.setstate(prior_state)
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
