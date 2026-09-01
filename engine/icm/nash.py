"""Heads-up ICM Nash push/fold equilibrium solver.

Algorithm:
  1. Pre-rank the 169 canonical hands by preflop all-in equity vs random.
  2. Iterate:
     a. Given villain's calling range, find hero's optimal push threshold.
     b. Given hero's push range, find villain's optimal calling threshold.
  3. Repeat until ranges stabilize (< max_iter rounds).

"Threshold" = the weakest hand worth pushing/calling, expressed as an index
into HAND_RANK (higher index = wider range).

Hand-vs-hand equities come from a precomputed 169x169 table (see
`build_equity_table.py` and docs/AUDITORIA-2026-08-26.md item I3), not a
live Monte Carlo call per query: with equity thresholds decided on the
margin (`ev_push >= ev_hero_fold`), the ~1% sampling error of an on-demand
2,000-iteration estimate was itself a source of threshold instability, on
top of the genuine best-response cycling documented below. The table is
generated once offline at much higher precision and loaded here as a
constant, making `_equity` an O(1) lookup instead of an O(iterations) sim.

Stack conventions: all stacks in chips. Blinds in chips.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from engine.icm.model import ICMModel

# 169 canonical preflop hands ordered from strongest to weakest by actual
# all-in equity vs. a uniformly random opponent hand (card-removal aware).
#
# Computed once via Monte Carlo (EquityCalculator, 15,000 iterations/hand,
# seed=0/12345) and baked in here — regenerating at import time would make
# every import pay a ~60-90s Monte Carlo cost. Regenerate with the script
# used to produce this list if the equity model ever changes.
#
# This replaces an earlier hand-shape heuristic (pairs, then suited by
# broadway/Ax/Kx/..., then offsuit) that mis-ranked several hands relative
# to their real strength — e.g. it placed K9o-K2o below every weak suited
# hand (42s included), when K9o (~57% vs random) is in fact stronger than
# most of them. That heuristic ordering fed directly into ICMNash's
# threshold-based push/call range search, so a wrong order produced wrong
# ranges, not just a cosmetic mis-sort.
HAND_RANK: list[str] = [
    "AA",
    "KK",
    "QQ",
    "JJ",
    "TT",
    "99",
    "88",
    "AKs",
    "77",
    "AQs",
    "AKo",
    "AJs",
    "AQo",
    "ATs",
    "KQs",
    "AJo",
    "A9s",
    "ATo",
    "66",
    "KJs",
    "A7s",
    "KTs",
    "A8s",
    "KQo",
    "KJo",
    "A9o",
    "55",
    "K9s",
    "A5s",
    "A8o",
    "QJs",
    "A6s",
    "KTo",
    "QTs",
    "A4s",
    "A7o",
    "A3s",
    "A5o",
    "Q9s",
    "A6o",
    "K8s",
    "QJo",
    "K9o",
    "A2s",
    "K7s",
    "JTs",
    "K6s",
    "QTo",
    "44",
    "Q8s",
    "A4o",
    "K8o",
    "K4s",
    "Q9o",
    "JTo",
    "A3o",
    "K5s",
    "J9s",
    "Q7s",
    "K7o",
    "A2o",
    "J8s",
    "Q6s",
    "K6o",
    "Q8o",
    "T9s",
    "K5o",
    "K3s",
    "33",
    "K2s",
    "J9o",
    "Q5s",
    "K4o",
    "Q4s",
    "T8s",
    "Q7o",
    "J8o",
    "J7s",
    "98s",
    "T9o",
    "K3o",
    "Q6o",
    "Q3s",
    "22",
    "K2o",
    "J5s",
    "T7s",
    "Q5o",
    "Q2s",
    "J7o",
    "J6s",
    "T8o",
    "97s",
    "J4s",
    "T6s",
    "Q4o",
    "98o",
    "J3s",
    "Q3o",
    "87s",
    "T5s",
    "T7o",
    "J6o",
    "Q2o",
    "J5o",
    "96s",
    "J4o",
    "T3s",
    "T4s",
    "J2s",
    "86s",
    "97o",
    "95s",
    "T6o",
    "87o",
    "96o",
    "76s",
    "85s",
    "94s",
    "J3o",
    "T2s",
    "T5o",
    "J2o",
    "75s",
    "65s",
    "T4o",
    "T3o",
    "93s",
    "86o",
    "84s",
    "92s",
    "95o",
    "64s",
    "76o",
    "74s",
    "85o",
    "T2o",
    "54s",
    "94o",
    "82s",
    "75o",
    "73s",
    "53s",
    "83s",
    "65o",
    "93o",
    "84o",
    "63s",
    "43s",
    "74o",
    "92o",
    "72s",
    "52s",
    "62s",
    "54o",
    "64o",
    "83o",
    "42s",
    "82o",
    "73o",
    "53o",
    "63o",
    "43o",
    "32s",
    "72o",
    "52o",
    "62o",
    "42o",
    "32o",
]

# Verify no duplicates
assert len(HAND_RANK) == len(set(HAND_RANK)), "Duplicate hands in HAND_RANK"
assert len(HAND_RANK) == 169, "HAND_RANK must contain all 169 canonical hands"

_HAND_INDEX: dict[str, int] = {hand: idx for idx, hand in enumerate(HAND_RANK)}

_DATA_DIR = Path(__file__).parent / "data"
_EQUITY_TABLE_PATH = _DATA_DIR / "hand_vs_hand_equity.npy"

_equity_table_cache: np.ndarray | None = None


def _load_equity_table() -> np.ndarray:
    global _equity_table_cache
    if _equity_table_cache is not None:
        return _equity_table_cache
    if not _EQUITY_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Precomputed hand-vs-hand equity table not found at {_EQUITY_TABLE_PATH}. "
            "Generate it once with:\n"
            "    python -m engine.icm.build_equity_table\n"
            "(~40 min on 8 cores at the default 20,000 iterations/pair; see that module's "
            "docstring for the tradeoffs)."
        )
    table = np.load(_EQUITY_TABLE_PATH)
    expected_shape = (len(HAND_RANK), len(HAND_RANK))
    if table.shape != expected_shape:
        raise ValueError(
            f"{_EQUITY_TABLE_PATH} has shape {table.shape}, expected {expected_shape} — "
            "regenerate it with engine.icm.build_equity_table."
        )
    _equity_table_cache = table
    return table


@dataclass
class NashResult:
    push_range: list[str]
    call_range: list[str]
    push_threshold: int
    call_threshold: int


class ICMNash:
    """Heads-up ICM Nash push/fold solver.

    Finds approximate Nash equilibrium push/call ranges for a heads-up
    all-in-or-fold scenario (e.g. HU at a final table or short-stack HU).
    """

    def __init__(self, model: ICMModel | None = None):
        self._model = model or ICMModel()
        self._table = _load_equity_table()

    def _equity(self, hand: str, vs_range: str) -> float:
        """Hero `hand`'s equity vs. a comma-separated list of canonical hands.

        Looks up the precomputed table and averages uniformly over the
        opponent hands named in `vs_range` (one weight per canonical hand,
        not per combo — matching `_range_str`'s uncombo-weighted range
        construction; see docs/AUDITORIA-2026-08-26.md item E2, still open).
        """
        row = self._table[_HAND_INDEX[hand]]
        cols = [_HAND_INDEX[h] for h in vs_range.split(",")]
        return float(row[cols].mean())

    def _range_str(self, threshold: int) -> str:
        """Return comma-separated range string for hands up to threshold index."""
        if threshold < 0:
            return ""
        hands = HAND_RANK[: threshold + 1]
        return ",".join(hands)

    def solve_hu(
        self,
        stacks: list[int],
        payouts: list[float],
        sb: int = 50,
        bb: int = 100,
        max_iter: int = 12,
    ) -> NashResult:
        """Find Nash push/call ranges for a heads-up push-or-fold spot.

        stacks: [hero_chips, villain_chips]  (hero = index 0)
        payouts: prize fractions e.g. [0.65, 0.35]
        sb, bb: blind sizes in chips
        max_iter: iterations for range convergence

        Hero (stacks[0]) is always the small blind and acts first: push or
        fold. Villain (stacks[1]) is the big blind and responds to a push
        with call or fold. This is the only role assignment push_range/
        call_range are solved for — push_range is always hero's shoving
        range, call_range is always villain's calling range. To get the
        equivalent solve from the big blind's perspective, call solve_hu
        with the stacks list reversed and re-read the result with the roles
        swapped (returned push_range becomes "the BB's shoving range" etc.).

        Stack conventions:
          - EV(hero fold)        → [hero - sb, villain + sb]
          - EV(push, v folds)    → [hero + bb, villain - bb]
          - EV(push, v calls, hero wins)  → [total, 0]
          - EV(push, v calls, hero loses) → [0, total]
          - EV(villain fold push) → [hero + bb, villain - bb]  (same as hero fold wins bb)

        Returns NashResult with push_range (hero) and call_range (villain).
        """
        assert len(stacks) == 2, "solve_hu requires exactly 2 players"

        hero_stack, villain_stack = stacks

        # --- Pre-compute static ICM scenarios ---
        # Hero folds their small blind
        eq_hero_fold = self._model.equity([max(0, hero_stack - sb), villain_stack + sb], payouts)
        ev_hero_fold = eq_hero_fold[0]

        # Villain folds to hero's push (hero wins the big blind)
        eq_villain_fold = self._model.equity([hero_stack + bb, max(0, villain_stack - bb)], payouts)

        # HU all-in outcomes: winner takes 1st-place payout, loser gets 2nd-place payout.
        # ICMModel.equity([total, 0]) returns [p1, 0] — it excludes the busted player and
        # doesn't assign their consolation prize. For a 2-player game the loser always
        # secures payouts[1] (or 0 if there's only one payout).
        p1 = payouts[0]
        p2 = payouts[1] if len(payouts) >= 2 else 0.0
        eq_hero_wins = [p1, p2]  # hero 1st, villain 2nd
        eq_hero_loses = [p2, p1]  # villain 1st, hero 2nd

        # --- Iterate push/call ranges ---
        # Start: villain calls top 30% by default
        call_threshold = int(len(HAND_RANK) * 0.30) - 1
        history: list[tuple[int, int]] = []

        for _ in range(max_iter):
            # Step 1: given villain's call range, find all hands where push > fold
            call_range_str = self._range_str(call_threshold)
            call_freq = (call_threshold + 1) / len(HAND_RANK) if call_threshold >= 0 else 0.0

            new_push_threshold = -1
            for idx in range(len(HAND_RANK)):
                hand = HAND_RANK[idx]
                if call_range_str:
                    eq_h = self._equity(hand, call_range_str)
                else:
                    eq_h = 1.0  # villain never calls

                ev_push = (1.0 - call_freq) * eq_villain_fold[0] + call_freq * (
                    eq_h * eq_hero_wins[0] + (1.0 - eq_h) * eq_hero_loses[0]
                )
                if ev_push >= ev_hero_fold:
                    new_push_threshold = idx
                else:
                    # HAND_RANK is ordered by real equity vs random, so EV is
                    # expected to be (near-)monotonically decreasing down the
                    # list. Stop at the first unprofitable hand instead of
                    # scanning the rest — this guarantees push_range is always
                    # a genuine contiguous prefix, not just the last hand that
                    # happened to clear the bar despite weaker hands in between
                    # failing (which produced non-monotonic, gappy ranges).
                    break

            push_threshold = new_push_threshold

            # Step 2: given hero's push range, find all hands where villain call > fold
            push_range_str = self._range_str(push_threshold)
            if not push_range_str:
                call_threshold = -1
                break

            # Villain fold EV (loses blind when hero pushes)
            ev_villain_fold = eq_villain_fold[1]

            new_call_threshold = -1
            for idx in range(len(HAND_RANK)):
                hand = HAND_RANK[idx]
                eq_v = self._equity(hand, push_range_str)

                # Villain wins → hero loses (stacks [0, total])
                # Villain loses → hero wins (stacks [total, 0])
                ev_call = (
                    eq_v * eq_hero_loses[1]  # villain wins → eq_hero_loses[1]
                    + (1.0 - eq_v) * eq_hero_wins[1]  # villain loses → eq_hero_wins[1]
                )
                if ev_call >= ev_villain_fold:
                    new_call_threshold = idx
                else:
                    # Same contiguity argument as the push loop above.
                    break

            call_threshold = new_call_threshold

            # Best-response iteration on a nonlinear ICM landscape doesn't
            # always settle at a fixed point — it can orbit a cycle of two or
            # more states indefinitely (verified empirically: e.g. stacks
            # [6000, 4000] cycles through 3 distinct (push, call) pairs
            # ranging from a near-empty push range to "shove everything").
            # Stopping at whatever state max_iter happens to land on would
            # report an arbitrary corner of that cycle. Instead, once a state
            # repeats one we've already seen, treat everything from its first
            # occurrence onward as one full cycle and report the *average*
            # threshold across it — the standard fictitious-play remedy for
            # best-response oscillation, and a stable summary regardless of
            # which point in the cycle max_iter would otherwise have cut at.
            state = (push_threshold, call_threshold)
            if state in history:
                cycle = history[history.index(state) :]
                push_threshold = round(sum(s[0] for s in cycle) / len(cycle))
                call_threshold = round(sum(s[1] for s in cycle) / len(cycle))
                break
            history.append(state)

        push_range = HAND_RANK[: push_threshold + 1] if push_threshold >= 0 else []
        call_range = HAND_RANK[: call_threshold + 1] if call_threshold >= 0 else []

        return NashResult(
            push_range=push_range,
            call_range=call_range,
            push_threshold=push_threshold,
            call_threshold=call_threshold,
        )
