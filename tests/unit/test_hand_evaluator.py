"""HandEvaluator tests — hand_class, outs, draw_strength."""
from core.hand_evaluator import HandEvaluator

ev = HandEvaluator()


# ------------------------------------------------------------------
# hand_class
# ------------------------------------------------------------------

def test_hand_class_royal_flush():
    assert ev.hand_class(["As", "Ks"], ["Qs", "Js", "Ts"]) == "Royal Flush"


def test_hand_class_straight_flush():
    # 9-high straight flush (not royal)
    assert ev.hand_class(["9s", "8s"], ["7s", "6s", "5s"]) == "Straight Flush"


def test_hand_class_four_of_a_kind():
    assert ev.hand_class(["Ah", "Ad"], ["As", "Ac", "2h"]) == "Four of a Kind"


def test_hand_class_full_house():
    assert ev.hand_class(["Ah", "Ad"], ["As", "Kh", "Kd"]) == "Full House"


def test_hand_class_flush():
    assert ev.hand_class(["2h", "5h"], ["7h", "Jh", "Qh"]) == "Flush"


def test_hand_class_straight():
    assert ev.hand_class(["9h", "8d"], ["7s", "6c", "5h"]) == "Straight"


def test_hand_class_three_of_a_kind():
    assert ev.hand_class(["Ah", "Ad"], ["As", "2h", "3d"]) == "Three of a Kind"


def test_hand_class_two_pair():
    assert ev.hand_class(["Ah", "Kh"], ["As", "Kd", "2c"]) == "Two Pair"


def test_hand_class_pair():
    assert ev.hand_class(["Ah", "Ad"], ["2s", "3h", "7c"]) == "Pair"


def test_hand_class_high_card():
    assert ev.hand_class(["Ah", "3d"], ["2s", "5c", "7h"]) == "High Card"


# ------------------------------------------------------------------
# outs — class-based: a card is an out only if it moves to a better hand CLASS
# ------------------------------------------------------------------

def test_outs_flush_draw_hearts_all_present():
    # 4 hearts on flop — all 9 remaining hearts complete the flush (Flush > High Card)
    # (Ah, 2h in hole, 3h 7h on board = 4 hearts used; 13-4=9 remain)
    outs = ev.outs(["Ah", "2h"], ["3h", "7h", "Kd"])
    hearts = [c for c in outs if c.endswith("h")]
    assert len(hearts) == 9


def test_outs_flush_draw_pairs_also_count():
    # From High Card, pairing hole or board cards also improves the class (Pair > High Card)
    # 9 flush outs + 14 pair-only outs = 23 (Kh is counted once, as both flush and pair)
    outs = ev.outs(["Ah", "2h"], ["3h", "7h", "Kd"])
    assert len(outs) == 23


def test_outs_oesd_straight_completers_present():
    # JTo on 9s 8c 2h — open-ended straight draw (Q or 7 complete the straight)
    # Note: board 9s 8c 2h — no straight yet (gap: 8-2)
    outs = ev.outs(["Jh", "Td"], ["9s", "8c", "2h"])
    queens = [c for c in outs if c[0] == "Q"]
    sevens = [c for c in outs if c[0] == "7"]
    assert len(queens) == 4
    assert len(sevens) == 4


def test_outs_made_flush_only_sf_upgrade():
    # Made flush A-K-Q-J-2 of hearts. Only Th upgrades to Royal Flush (SF class).
    # Other hearts give a better-ranked flush but same class — NOT outs.
    outs = ev.outs(["Ah", "Kh"], ["Qh", "Jh", "2h"])
    assert len(outs) == 1
    assert outs[0] == "Th"


def test_outs_made_straight_very_few():
    # Made J-high straight (JT on 987 board) — almost nothing upgrades to Flush/SF
    outs = ev.outs(["Jh", "Td"], ["9s", "8c", "7h"])
    # No flush possible with 1 extra card (only 2 hearts: Jh+7h)
    # No SF possible; all classes below Straight require flush/quads/boat
    assert len(outs) == 0


def test_outs_are_valid_card_strings():
    outs = ev.outs(["Ah", "2h"], ["3h", "7h", "Kd"])
    for c in outs:
        assert len(c) == 2
        assert c[0] in "23456789TJQKA"
        assert c[1] in "shdc"


def test_outs_no_dead_cards_in_outs():
    hole = ["Ah", "2h"]
    board = ["3h", "7h", "Kd"]
    outs = ev.outs(hole, board)
    dead = set(hole + board)
    assert not any(c in dead for c in outs)


# ------------------------------------------------------------------
# draw_strength
# ------------------------------------------------------------------

def test_draw_strength_flush_draw():
    # Ah2h on 3h7hKd: 23 class-improving outs out of 47 unseen cards
    ds = ev.draw_strength(["Ah", "2h"], ["3h", "7h", "Kd"])
    assert abs(ds - 23 / 47) < 1e-9


def test_draw_strength_made_hand_minimal():
    # Made flush — only 1 out (Th → Royal Flush) out of 47 unseen
    ds = ev.draw_strength(["Ah", "Kh"], ["Qh", "Jh", "2h"])
    assert abs(ds - 1 / 47) < 1e-9


def test_draw_strength_made_straight_zero():
    # Made straight, no class-improving outs with 1 card
    ds = ev.draw_strength(["Jh", "Td"], ["9s", "8c", "7h"])
    assert ds == 0.0


def test_draw_strength_in_unit_interval():
    ds = ev.draw_strength(["9h", "8d"], ["7s", "6c", "2h"])
    assert 0.0 <= ds <= 1.0
