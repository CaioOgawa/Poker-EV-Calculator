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

All-in confrontations are contested for the *effective* stack — min(hero,
villain), not the full stack on both sides — and range EVs (`_equity`,
`_call_freq`) are combo-weighted and card-removal-aware against the fixed
hand's own two cards, not a flat 1/169 per canonical hand
(docs/AUDITORIA-2026-08-26.md items E1 and E2).

Stack conventions: all stacks in chips. Blinds in chips.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.ranges.range_parser import RANKS, SUITS, RangeParser
from engine.icm.model import ICMModel

_PARSER = RangeParser()

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

# Combo-count and card-removal bookkeeping for E2 (docs/AUDITORIA-2026-08-26.md):
# treating every canonical hand as equally likely (1/169) ignores that AA is 6
# combos, AKs is 4, AKo is 12, and ignores blockers from hero's own hole cards.
# _COMBO_WEIGHT[h] = combo count of canonical hand h (6, 4, or 12).
# _BLOCK_MATRIX[h, c] = how many of hand h's combos contain card c.
# Both are computed once at import time from RangeParser's own combo generator
# (the same one EquityCalculator uses), not re-derived ad hoc.
_ALL_CARDS: list[str] = [r + s for r in RANKS for s in SUITS]
_CARD_INDEX: dict[str, int] = {c: i for i, c in enumerate(_ALL_CARDS)}
# C(50, 2): villain's combo universe given hero's 2 known cards
_TOTAL_COMBOS_EXCLUDING_KNOWN = 1225

_COMBO_WEIGHT = np.zeros(len(HAND_RANK), dtype=np.float64)
_BLOCK_MATRIX = np.zeros((len(HAND_RANK), len(_ALL_CARDS)), dtype=np.float64)
for _h_idx, _hand in enumerate(HAND_RANK):
    _combos = _PARSER.parse(_hand)
    _COMBO_WEIGHT[_h_idx] = len(_combos)
    for _c1, _c2 in _combos:
        _BLOCK_MATRIX[_h_idx, _CARD_INDEX[_c1]] += 1
        _BLOCK_MATRIX[_h_idx, _CARD_INDEX[_c2]] += 1
del _h_idx, _hand, _combos, _c1, _c2

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

    def _range_weights(self, fixed_hand: str, target_hands: list[str]) -> np.ndarray:
        """Combo-weighted, card-removal-aware weight for each hand in `target_hands`,
        given that `fixed_hand` blocks some of their combos (docs/AUDITORIA-2026-08-26.md
        item E2).

        `fixed_hand` is canonical (e.g. "AKs" covers 4 specific-card combos, each
        blocking a different subset of the deck), so the weight is averaged over
        its own combos. For one specific fixed combo (a, b), the number of a
        target hand's combos left unblocked is, by inclusion-exclusion:
            combo_count − (combos containing a) − (combos containing b)
                        + (combos containing both a and b)
        The last term is 1 only for the fixed hand's own canonical bucket (two
        specific cards form exactly one combo, and it belongs to exactly one
        canonical hand) and 0 everywhere else.

        Weights sum to the expected number of unblocked combos across
        `target_hands`; divide by 1225 (= C(50,2), the villain's combo universe
        given hero's 2 known cards) to turn the total into a probability.
        """
        fixed_idx = _HAND_INDEX[fixed_hand]
        fixed_combos = _PARSER.parse(fixed_hand)
        total = np.zeros(len(HAND_RANK))
        for a, b in fixed_combos:
            blocked = _BLOCK_MATRIX[:, _CARD_INDEX[a]] + _BLOCK_MATRIX[:, _CARD_INDEX[b]]
            unblocked = _COMBO_WEIGHT - blocked
            unblocked[fixed_idx] += 1.0
            total += unblocked
        total /= len(fixed_combos)
        cols = [_HAND_INDEX[h] for h in target_hands]
        return total[cols]

    def _equity(self, hand: str, vs_range: str) -> float:
        """Hero `hand`'s equity vs. a comma-separated list of canonical hands.

        Looks up the precomputed table and averages over the opponent hands
        named in `vs_range`, weighted by each hand's available combo count
        given hero's own two cards as blockers (see `_range_weights`) —
        not one uniform weight per canonical hand regardless of combo count.

        Residual approximation: the table itself (`build_equity_table.py`) is
        canonical-hand-level, i.e. `table[i][j]` is already marginalized over
        both sides' combos with mutual card removal between i and j alone. It
        doesn't know about hero's blockers against a *third* hand in `vs_range`,
        so this is a weighted average of pairwise-exact equities, not a fully
        joint combo-vs-combo equity (average-of-products, not product-of-
        averages). Closing that gap needs a full combo-level equity table —
        out of scope here.
        """
        hands = vs_range.split(",")
        row = self._table[_HAND_INDEX[hand]]
        cols = [_HAND_INDEX[h] for h in hands]
        weights = self._range_weights(hand, hands)
        return float(np.average(row[cols], weights=weights))

    def _call_freq(self, hero_hand: str, call_range_hands: list[str]) -> float:
        """P(villain holds a hand in `call_range_hands`), combo-weighted and
        card-removal-aware against hero's own two cards (docs/AUDITORIA-2026-08-26.md
        item E2 — this replaces treating every canonical hand as equally likely).
        """
        if not call_range_hands:
            return 0.0
        weights = self._range_weights(hero_hand, call_range_hands)
        return float(weights.sum() / _TOTAL_COMBOS_EXCLUDING_KNOWN)

    def _all_in_equity(self, s_hero: int, s_villain: int, payouts: list[float]) -> list[float]:
        """ICM equity split [hero, villain] for stacks (s_hero, s_villain) right
        after an all-in confrontation (docs/AUDITORIA-2026-08-26.md item E1).

        `ICMModel.equity` assigns 0.0 — not the consolation payout — to a
        zero-chip entry (verified: `equity([10000, 0], payouts) == [p1, 0.0]`,
        not `[p1, p2]`), so the two-player elimination case is handled
        explicitly here instead of routed through the model.
        """
        p1 = payouts[0]
        p2 = payouts[1] if len(payouts) >= 2 else 0.0
        if s_villain == 0:
            return [p1, p2]
        if s_hero == 0:
            return [p2, p1]
        return self._model.equity([s_hero, s_villain], payouts)

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
          - EV(push, v calls, hero wins)  → ICM([hero+eff, villain-eff]), eff=min(hero,villain)
          - EV(push, v calls, hero loses) → ICM([hero-eff, villain+eff])
          - EV(villain fold push) → [hero + bb, villain - bb]  (same as hero fold wins bb)
          An all-in is contested for the *effective* stack (the smaller of the
          two), not the full stack on both sides (docs/AUDITORIA-2026-08-26.md
          item E1) — the bigger stack survives a loss with its excess chips.

        Returns NashResult with push_range (hero) and call_range (villain).
        """
        if len(stacks) != 2:
            raise ValueError(f"solve_hu requires exactly 2 players, got {len(stacks)}")

        hero_stack, villain_stack = stacks

        # --- Pre-compute static ICM scenarios ---
        # Hero folds their small blind
        eq_hero_fold = self._model.equity([max(0, hero_stack - sb), villain_stack + sb], payouts)
        ev_hero_fold = eq_hero_fold[0]

        # Villain folds to hero's push (hero wins the big blind)
        eq_villain_fold = self._model.equity([hero_stack + bb, max(0, villain_stack - bb)], payouts)

        # HU all-in outcomes, contested for the effective stack (E1): the
        # winner gains min(hero, villain) chips from the loser, not the
        # loser's entire stack — with unequal stacks the loser only busts if
        # they had the shorter stack.
        eff = min(hero_stack, villain_stack)
        eq_hero_wins = self._all_in_equity(hero_stack + eff, villain_stack - eff, payouts)
        eq_hero_loses = self._all_in_equity(hero_stack - eff, villain_stack + eff, payouts)

        # --- Iterate push/call ranges ---
        # Start: villain calls top 30% by default
        call_threshold = int(len(HAND_RANK) * 0.30) - 1
        history: list[tuple[int, int]] = []

        for _ in range(max_iter):
            # Step 1: given villain's call range, find all hands where push > fold
            call_range_str = self._range_str(call_threshold)
            call_range_hands = HAND_RANK[: call_threshold + 1] if call_threshold >= 0 else []

            new_push_threshold = -1
            for idx in range(len(HAND_RANK)):
                hand = HAND_RANK[idx]
                if call_range_str:
                    eq_h = self._equity(hand, call_range_str)
                    # Combo-weighted, card-removal-aware fold equity (E2) — how
                    # likely villain's random hand actually falls in their call
                    # range given hero's own two cards as blockers, not a flat
                    # (call_threshold + 1) / 169.
                    call_freq = self._call_freq(hand, call_range_hands)
                else:
                    eq_h = 1.0  # villain never calls
                    call_freq = 0.0

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

                # Villain wins → hero loses the effective stack
                # Villain loses → hero wins the effective stack
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
            # more states indefinitely. Fixing E1 (effective-stack showdown)
            # and E2 (combo-weighted, blocker-aware fold equity/range
            # averaging) narrowed this, but didn't eliminate it: stacks
            # [6000, 4000] used to cycle through 3 states swinging from
            # near-empty to "shove everything" (push_threshold 5 <-> 168);
            # post-fix it settles into a tighter 2-state cycle instead
            # (verified empirically: push_threshold 32 <-> 135). Stopping at
            # whatever state max_iter happens to land on would
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
