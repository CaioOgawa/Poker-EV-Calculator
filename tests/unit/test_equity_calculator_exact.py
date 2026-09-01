"""Exact-enumeration paths (docs/AUDITORIA-2026-08-26.md item I1): 0 or 1
cards to come should give a deterministic result, not Monte Carlo noise —
so these run fast and unmarked (not `slow`), unlike test_equity_calculator.py.
"""

import pytest
from treys import Card, Deck, Evaluator

from core.equity.calculator import EquityCalculator
from core.ranges.range_parser import RangeParser

_EVAL = Evaluator()
calc = EquityCalculator(iterations=200, seed=0)


# ------------------------------------------------------------------
# heads_up
# ------------------------------------------------------------------


def test_heads_up_need0_is_exact_river_showdown():
    # Nut flush vs top pair on a complete board — a single deterministic
    # showdown, not a sampled estimate.
    eq = calc.heads_up(["9s", "8s"], ["Ah", "Kd"], board=["2s", "5s", "Qs", "3d", "Tc"])
    assert eq == 1.0


def test_heads_up_need0_ignores_iterations_and_seed():
    kwargs = dict(hero=["9s", "8s"], villain=["Ah", "Kd"], board=["2s", "5s", "Qs", "3d", "Tc"])
    fast = EquityCalculator(iterations=1)
    slow = EquityCalculator(iterations=99_999)
    assert fast.heads_up(**kwargs) == slow.heads_up(**kwargs) == 1.0
    assert fast.heads_up(**kwargs, seed=1) == fast.heads_up(**kwargs, seed=2)


def test_heads_up_need1_matches_manual_enumeration():
    hero, villain, board = ["As", "Ad"], ["Ks", "Kd"], ["2c", "7d", "Th", "Qs"]
    eq = calc.heads_up(hero, villain, board=board)

    known = {Card.new(c) for c in hero + villain + board}
    deck = [c for c in Deck().cards if c not in known]
    board_c = [Card.new(c) for c in board]
    hero_c = [Card.new(c) for c in hero]
    villain_c = [Card.new(c) for c in villain]
    wins = 0.0
    for card in deck:
        run_board = board_c + [card]
        h = _EVAL.evaluate(run_board, hero_c)
        v = _EVAL.evaluate(run_board, villain_c)
        wins += 1.0 if h < v else (0.5 if h == v else 0.0)
    expected = wins / len(deck)

    assert eq == expected


def test_heads_up_need1_ignores_seed():
    kwargs = dict(hero=["As", "Ad"], villain=["Ks", "Kd"], board=["2c", "7d", "Th", "Qs"])
    assert calc.heads_up(**kwargs, seed=1) == calc.heads_up(**kwargs, seed=2)


# ------------------------------------------------------------------
# vs_range
# ------------------------------------------------------------------


def test_vs_range_need0_matches_manual_enumeration():
    hero, board = ["As", "Ad"], ["2s", "5s", "Qs", "3d", "Tc"]
    eq = calc.vs_range(hero, "KK,QQ", board=board)

    hero_c = [Card.new(c) for c in hero]
    board_c = [Card.new(c) for c in board]
    hero_set = set(hero_c + board_c)
    combos = [
        [Card.new(a), Card.new(b)]
        for a, b in RangeParser().parse("KK,QQ")
        if Card.new(a) not in hero_set and Card.new(b) not in hero_set
    ]
    hero_rank = _EVAL.evaluate(board_c, hero_c)
    wins = 0.0
    for c in combos:
        v = _EVAL.evaluate(board_c, c)
        wins += 1.0 if hero_rank < v else (0.5 if hero_rank == v else 0.0)
    assert eq == wins / len(combos)


def test_vs_range_need1_matches_manual_enumeration():
    # This is the most intricate path added (nested combo x runout), so it
    # gets the same treatment as heads_up's need==1 test: checked against an
    # independent manual enumeration, not just sanity bounds. The expected
    # value is an average-of-averages (one villain combo, uniform; one
    # runout given that combo, uniform) — a flat win/total ratio across the
    # whole nested loop would only coincide with this if every combo's own
    # deck size happened to match, so this also pins that weighting down.
    hero, board = ["As", "Ad"], ["2s", "5s", "Qs", "3d"]
    eq = calc.vs_range(hero, "KK,QQ", board=board)

    hero_c = [Card.new(c) for c in hero]
    board_c = [Card.new(c) for c in board]
    hero_set = set(hero_c + board_c)
    combos = [
        [Card.new(a), Card.new(b)]
        for a, b in RangeParser().parse("KK,QQ")
        if Card.new(a) not in hero_set and Card.new(b) not in hero_set
    ]
    per_combo_equity = []
    for combo in combos:
        villain_set = set(combo)
        deck = [c for c in Deck().cards if c not in hero_set and c not in villain_set]
        wins = 0.0
        for card in deck:
            run_board = board_c + [card]
            h = _EVAL.evaluate(run_board, hero_c)
            v = _EVAL.evaluate(run_board, combo)
            wins += 1.0 if h < v else (0.5 if h == v else 0.0)
        per_combo_equity.append(wins / len(deck))
    expected = sum(per_combo_equity) / len(per_combo_equity)

    assert eq == expected
    assert eq == calc.vs_range(hero, "KK,QQ", board=board, seed=123)


# ------------------------------------------------------------------
# multi_way
# ------------------------------------------------------------------


def test_multi_way_need0_is_exact_and_sums_to_one():
    hands = [["As", "Ad"], ["Ks", "Kd"], ["Qh", "Qd"]]
    board = ["2s", "5s", "Qs", "3d", "Tc"]
    equities = calc.multi_way(hands, board=board)
    assert sum(equities) == 1.0
    # The board's Qs turns QQ into trip queens — deterministic winner over
    # AA/KK's mere overpairs, not just "AA is best preflop".
    assert equities == [0.0, 0.0, 1.0]


def test_multi_way_need1_matches_manual_enumeration():
    hands = [["As", "Ad"], ["Ks", "Kd"], ["Qh", "Qd"]]
    board = ["2s", "5s", "Qs", "3d"]
    equities = calc.multi_way(hands, board=board)
    assert sum(equities) == pytest.approx(1.0)

    known = {Card.new(c) for h in hands for c in h} | {Card.new(c) for c in board}
    deck = [c for c in Deck().cards if c not in known]
    hands_c = [[Card.new(c) for c in h] for h in hands]
    board_c = [Card.new(c) for c in board]
    expected = [0.0, 0.0, 0.0]
    for card in deck:
        run_board = board_c + [card]
        ranks = [_EVAL.evaluate(run_board, h) for h in hands_c]
        best = min(ranks)
        winners = [i for i, r in enumerate(ranks) if r == best]
        for i in winners:
            expected[i] += 1.0 / len(winners)
    expected = [e / len(deck) for e in expected]

    assert equities == pytest.approx(expected)


# ------------------------------------------------------------------
# need==1 exact result should sit inside Monte Carlo's tolerance for the
# same spot, cross-checking the two code paths against each other.
# ------------------------------------------------------------------


def test_need1_exact_matches_mc_within_tolerance():
    hero, villain, board = ["As", "Ad"], ["Ks", "Kd"], ["2c", "7d", "Th", "Qs"]
    exact = calc.heads_up(hero, villain, board=board)
    mc = EquityCalculator(iterations=20_000, seed=7).heads_up(hero, villain, board=board)
    assert abs(exact - mc) < 0.02
