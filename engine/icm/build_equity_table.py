"""Offline generator for the 169x169 canonical hand-vs-hand equity table.

Run once (or whenever the equity model changes) to (re)populate
`engine/icm/data/hand_vs_hand_equity.npy`, which `ICMNash` loads at import
time instead of running Monte Carlo per query during `solve_hu`. See
docs/AUDITORIA-2026-08-26.md item I3 for the motivation: `ICMNash._equity`
used to run a fresh 2,000-iteration Monte Carlo estimate (~1.1pp sampling
error) for every (hand, range) pair it needed, live inside the
best-response loop -- noise that fed back into threshold instability right
on the decision margin. Precomputing every pairwise matchup once, at much
higher precision, removes both the noise and the per-call cost.

Method: reuses `EquityCalculator.range_vs_range` -- the repo's existing,
tested, card-removal-correct combo sampler -- called once per unordered
pair of canonical hands (i <= j), each treated as a one-hand "range" so
both sides are combo-weighted with mutual card removal exactly as the
audit specifies. This is high-precision, fixed-seed Monte Carlo, not
exhaustive board enumeration: exact board enumeration (~1.7M runouts) is
cheap for one *specific pair of hole-card combos* (~25s measured with
`treys`), but the number of distinct hole-card combo pairs needed for
full combo-level exactness across all 169x169 canonical matchups (up to
~70 valid combo pairs for some canonical pairs) is not — it would take
this same job from tens of minutes to many CPU-days. At the default
`iterations=20_000`, the standard error per matchup is ~0.3pp -- roughly a
4x tightening over ICMNash's previous per-call 2,000-iteration budget --
computed once and reused forever instead of recomputed (with fresh noise)
on every solve.

Usage:
    python -m engine.icm.build_equity_table
    python -m engine.icm.build_equity_table --iterations 50000 --jobs 4
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from core.equity.calculator import EquityCalculator
from engine.icm.nash import HAND_RANK

_DATA_DIR = Path(__file__).parent / "data"
_TABLE_PATH = _DATA_DIR / "hand_vs_hand_equity.npy"


def _pair_equity(
    i: int, j: int, hands: list[str], iterations: int, seed: int
) -> tuple[int, int, float]:
    calc = EquityCalculator(iterations=iterations, seed=seed)
    eq_i, _eq_j = calc.range_vs_range(hands[i], hands[j], seed=seed)
    return i, j, eq_i


def build_table(
    hands: list[str] = HAND_RANK,
    iterations: int = 20_000,
    seed: int = 0,
    n_jobs: int = -1,
    verbose: int = 5,
) -> tuple[np.ndarray, float]:
    """Return (table, elapsed_seconds). table[i][j] = equity of hands[i] vs hands[j]."""
    n = len(hands)
    # i == j is a mirror matchup: true equity is exactly 0.5, not an MC
    # estimate, so it's excluded here and filled in directly below.
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_pair_equity)(i, j, hands, iterations, seed) for i, j in pairs
    )
    elapsed = time.time() - t0

    table = np.empty((n, n), dtype=np.float64)
    np.fill_diagonal(table, 0.5)
    for i, j, eq_i in results:
        table[i, j] = eq_i
        table[j, i] = 1.0 - eq_i

    return table, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--out", type=Path, default=_TABLE_PATH)
    args = parser.parse_args()

    table, elapsed = build_table(iterations=args.iterations, seed=args.seed, n_jobs=args.jobs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, table)

    meta = {
        "hand_order": HAND_RANK,
        "iterations": args.iterations,
        "seed": args.seed,
        "method": "EquityCalculator.range_vs_range (combo-weighted MC, card-removal correct)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "shape": list(table.shape),
    }
    meta_path = args.out.parent / (args.out.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Wrote {args.out} {table.shape} in {elapsed:.1f}s -> {meta_path}")


if __name__ == "__main__":
    main()
