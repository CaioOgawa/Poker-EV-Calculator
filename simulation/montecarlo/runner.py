"""Monte Carlo runner for EV distributions and variance analysis."""

from __future__ import annotations
import os
import random
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed


@dataclass
class MonteCarloResult:
    iterations: int
    mean_ev: float
    std_dev: float
    p5: float
    p95: float
    raw: list[float] = field(default_factory=list)


def _run_chunk(ev_fn, kwargs: dict, seeds: list[int | None]) -> list[float]:
    """Run one `ev_fn(**kwargs)` call per seed in `seeds`, sequentially,
    inside a single joblib task. Module-level (not a closure) so it can be
    pickled and sent to worker processes.

    A chunk, not one task per iteration: serializing `ev_fn`/kwargs and
    saving/restoring the global `random` state per task dominates the
    runtime for cheap `ev_fn`s when there's one task per iteration.
    Batching `iterations` into `n_jobs` chunks pays that overhead `n_jobs`
    times instead of `iterations` times — each call still gets its own
    independent seed, so per-iteration reproducibility is unchanged from
    before this was chunked.
    """
    prior_state = random.getstate()
    try:
        results = []
        for seed in seeds:
            if seed is not None:
                random.seed(seed)
            results.append(ev_fn(**kwargs))
        return results
    finally:
        random.setstate(prior_state)


class MonteCarloSim:
    def __init__(self, iterations: int = 50_000, seed: int | None = None):
        self.iterations = iterations
        self.seed = seed

    def run_ev(self, ev_fn, n_jobs: int = 1, keep_raw: bool = True, **kwargs) -> MonteCarloResult:
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

        keep_raw=False drops the per-trial results from the returned
        `MonteCarloResult.raw` (still used internally for mean/std/p5/p95)
        — worth it at large `iterations` where materializing every trial
        isn't needed, just the summary stats.
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
            n_chunks = n_jobs if n_jobs > 0 else (os.cpu_count() or 4)
            chunk_size = max(1, -(-self.iterations // n_chunks))  # ceil division
            chunks = [seeds[i : i + chunk_size] for i in range(0, self.iterations, chunk_size)]
            chunked_results = Parallel(n_jobs=n_jobs)(
                delayed(_run_chunk)(ev_fn, kwargs, chunk) for chunk in chunks
            )
            results = [r for chunk in chunked_results for r in chunk]
        results_arr = np.asarray(results)
        n = len(results_arr)
        # Sample stats, not population (docs/AUDITORIA-2026-08-26.md item
        # E8): `results` is a sample of the EV distribution, not the whole
        # population, so std_dev needs Bessel's correction (ddof=1) to be
        # unbiased — /n underestimates spread, more so at small n. ddof=1 is
        # undefined at n<=1 (no spread to estimate from one point), so that
        # case falls back to 0.0 rather than propagating NaN. p5/p95 switch
        # from a truncated-index lookup to linear interpolation between
        # order statistics (numpy's default, and what the rest of the repo's
        # percentile calls already use in cash_session.py/simulator.py) —
        # both are real behavior changes to MonteCarloResult's reported
        # numbers, not bug fixes with an obviously-right answer, so this was
        # held for an explicit decision rather than folded into I5's chunking
        # rewrite.
        mean = float(np.mean(results_arr))
        std = float(np.std(results_arr, ddof=1)) if n > 1 else 0.0
        p5 = float(np.percentile(results_arr, 5))
        p95 = float(np.percentile(results_arr, 95))
        return MonteCarloResult(
            iterations=n,
            mean_ev=mean,
            std_dev=std,
            p5=p5,
            p95=p95,
            raw=results if keep_raw else [],
        )
