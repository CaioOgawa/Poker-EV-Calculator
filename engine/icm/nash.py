"""Heads-up ICM Nash push/fold equilibrium solver.

Algorithm:
  1. Pre-rank the 169 canonical hands by preflop all-in equity vs random.
  2. Iterate:
     a. Given villain's calling range, find hero's optimal push threshold.
     b. Given hero's push range, find villain's optimal calling threshold.
  3. Repeat until ranges stabilize (< max_iter rounds).

"Threshold" = the weakest hand worth pushing/calling, expressed as an index
into HAND_RANK (higher index = wider range).

Equity approximations use the EquityCalculator (Monte Carlo), which is slow
for full 169-hand sweeps. To keep runtime practical, equity is computed
on-demand and cached per (hand, range_str) pair within a session.

Stack conventions: all stacks in chips. Blinds in chips.
"""
from __future__ import annotations
from dataclasses import dataclass
from engine.icm.model import ICMModel
from core.equity.calculator import EquityCalculator

# 169 canonical preflop hands ordered from strongest to weakest
# (pairs first, then suited, then offsuit — standard ranking)
_PAIRS = ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22"]
_BROADWAYS_S = ["AKs", "AQs", "AJs", "ATs", "KQs", "KJs", "KTs", "QJs", "QTs", "JTs"]
_BROADWAYS_O = ["AKo", "AQo", "AJo", "ATo", "KQo", "KJo", "KTo", "QJo", "QTo", "JTo"]
_CONNECTORS_S = ["T9s", "98s", "87s", "76s", "65s", "54s", "43s", "32s"]
_CONNECTORS_O = ["T9o", "98o", "87o", "76o", "65o", "54o", "43o", "32o"]
_AX_S = ["A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s"]
_AX_O = ["A9o", "A8o", "A7o", "A6o", "A5o", "A4o", "A3o", "A2o"]
_KX_S = ["K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s"]
_KX_O = ["K9o", "K8o", "K7o", "K6o", "K5o", "K4o", "K3o", "K2o"]
_QX_S = ["Q9s", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s", "Q3s", "Q2s"]
_QX_O = ["Q9o", "Q8o", "Q7o", "Q6o", "Q5o", "Q4o", "Q3o", "Q2o"]
_JX_S = ["J9s", "J8s", "J7s", "J6s", "J5s", "J4s", "J3s", "J2s"]
_JX_O = ["J9o", "J8o", "J7o", "J6o", "J5o", "J4o", "J3o", "J2o"]
_TX_S = ["T8s", "T7s", "T6s", "T5s", "T4s", "T3s", "T2s"]
_TX_O = ["T8o", "T7o", "T6o", "T5o", "T4o", "T3o", "T2o"]
_SMALL_S = [
    "97s", "96s", "95s", "94s", "93s", "92s",
    "86s", "85s", "84s", "83s", "82s",
    "75s", "74s", "73s", "72s",
    "64s", "63s", "62s",
    "53s", "52s",
    "42s",
]
_SMALL_O = [
    "97o", "96o", "95o", "94o", "93o", "92o",
    "86o", "85o", "84o", "83o", "82o",
    "75o", "74o", "73o", "72o",
    "64o", "63o", "62o",
    "53o", "52o",
    "42o",
]

HAND_RANK: list[str] = (
    _PAIRS
    + _BROADWAYS_S
    + _AX_S
    + _KX_S
    + _BROADWAYS_O
    + _AX_O
    + _QX_S
    + _JX_S
    + _TX_S
    + _CONNECTORS_S
    + _QX_O
    + _JX_O
    + _TX_O
    + _CONNECTORS_O
    + _SMALL_S
    + _KX_O
    + _SMALL_O
)

# Verify no duplicates
assert len(HAND_RANK) == len(set(HAND_RANK)), "Duplicate hands in HAND_RANK"


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

    def __init__(
        self,
        model: ICMModel | None = None,
        equity_calc: EquityCalculator | None = None,
        equity_iterations: int = 2_000,
    ):
        self._model = model or ICMModel()
        self._calc = equity_calc or EquityCalculator(iterations=equity_iterations, seed=0)
        self._equity_cache: dict[tuple[str, str], float] = {}

    def _equity(self, hand: str, vs_range: str) -> float:
        key = (hand, vs_range)
        if key not in self._equity_cache:
            # Convert canonical hand to specific cards for calculation
            cards = self._hand_to_cards(hand)
            self._equity_cache[key] = self._calc.vs_range(cards, vs_range)
        return self._equity_cache[key]

    @staticmethod
    def _hand_to_cards(hand: str) -> list[str]:
        """Convert canonical hand string to two specific cards (worst blockers)."""
        if len(hand) == 2:  # pocket pair e.g. "AA"
            return [hand[0] + "s", hand[0] + "h"]
        suited = hand.endswith("s")
        r1, r2 = hand[0], hand[1]
        if suited:
            return [r1 + "s", r2 + "s"]
        return [r1 + "s", r2 + "h"]

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
        hero_is_sb: bool = True,
        max_iter: int = 5,
    ) -> NashResult:
        """Find Nash push/call ranges for a heads-up push-or-fold spot.

        stacks: [hero_chips, villain_chips]  (hero = index 0)
        payouts: prize fractions e.g. [0.65, 0.35]
        sb, bb: blind sizes in chips
        hero_is_sb: True if hero posts the small blind (typical HU button = SB)
        max_iter: iterations for range convergence

        Stack conventions (hero_is_sb=True):
          - EV(hero fold)        → [hero - sb, villain + sb]
          - EV(push, v folds)    → [hero + bb, villain - bb]
          - EV(push, v calls, hero wins)  → [total, 0]
          - EV(push, v calls, hero loses) → [0, total]
          - EV(villain fold push) → [hero + bb, villain - bb]  (same as hero fold wins bb)

        Returns NashResult with push_range (hero) and call_range (villain).
        """
        assert len(stacks) == 2, "solve_hu requires exactly 2 players"

        hero_stack, villain_stack = stacks
        total = hero_stack + villain_stack

        # --- Pre-compute static ICM scenarios ---
        # Hero folds their blind
        if hero_is_sb:
            eq_hero_fold = self._model.equity(
                [max(0, hero_stack - sb), villain_stack + sb], payouts
            )
        else:
            eq_hero_fold = self._model.equity(
                [hero_stack, villain_stack], payouts
            )
        ev_hero_fold = eq_hero_fold[0]

        # Villain folds to hero's push (hero wins the blind)
        if hero_is_sb:
            eq_villain_fold = self._model.equity(
                [hero_stack + bb, max(0, villain_stack - bb)], payouts
            )
        else:
            eq_villain_fold = self._model.equity(
                [hero_stack + sb, max(0, villain_stack - sb)], payouts
            )

        # HU all-in outcomes: winner takes 1st-place payout, loser gets 2nd-place payout.
        # ICMModel.equity([total, 0]) returns [p1, 0] — it excludes the busted player and
        # doesn't assign their consolation prize. For a 2-player game the loser always
        # secures payouts[1] (or 0 if there's only one payout).
        p1 = payouts[0]
        p2 = payouts[1] if len(payouts) >= 2 else 0.0
        eq_hero_wins = [p1, p2]   # hero 1st, villain 2nd
        eq_hero_loses = [p2, p1]  # villain 1st, hero 2nd

        # --- Iterate push/call ranges ---
        # Start: villain calls top 30% by default
        call_threshold = int(len(HAND_RANK) * 0.30) - 1

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

                ev_push = (
                    (1.0 - call_freq) * eq_villain_fold[0]
                    + call_freq * (
                        eq_h * eq_hero_wins[0]
                        + (1.0 - eq_h) * eq_hero_loses[0]
                    )
                )
                if ev_push >= ev_hero_fold:
                    new_push_threshold = idx

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
                    eq_v * eq_hero_loses[1]          # villain wins → eq_hero_loses[1]
                    + (1.0 - eq_v) * eq_hero_wins[1] # villain loses → eq_hero_wins[1]
                )
                if ev_call >= ev_villain_fold:
                    new_call_threshold = idx

            call_threshold = new_call_threshold

        push_range = HAND_RANK[: push_threshold + 1] if push_threshold >= 0 else []
        call_range = HAND_RANK[: call_threshold + 1] if call_threshold >= 0 else []

        return NashResult(
            push_range=push_range,
            call_range=call_range,
            push_threshold=push_threshold,
            call_threshold=call_threshold,
        )
