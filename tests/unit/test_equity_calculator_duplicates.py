"""Duplicate-card validation for EquityCalculator (audit E5).

Kept in its own fast, unmarked module: these are input-validation checks that
raise before any Monte Carlo runs, so they don't need the `slow` marker (or
iteration counts) that the rest of test_equity_calculator.py carries.
"""

import pytest

from core.equity.calculator import EquityCalculator

calc = EquityCalculator(iterations=100, seed=0)


def test_heads_up_rejects_card_shared_by_hero_and_villain():
    with pytest.raises(ValueError, match="Duplicate card"):
        calc.heads_up(["As", "Ks"], ["As", "Qd"])


def test_heads_up_rejects_duplicate_within_hero():
    with pytest.raises(ValueError, match="Duplicate card"):
        calc.heads_up(["As", "As"], ["Kd", "Qd"])


def test_heads_up_rejects_card_shared_with_board():
    with pytest.raises(ValueError, match="Duplicate card"):
        calc.heads_up(["As", "Ks"], ["Qd", "Jd"], board=["As", "7c", "2h"])


def test_vs_range_rejects_card_shared_with_board():
    with pytest.raises(ValueError, match="Duplicate card"):
        calc.vs_range(["As", "Ks"], "QQ+", board=["As", "7c", "2h"])


def test_multi_way_rejects_card_shared_between_hands():
    with pytest.raises(ValueError, match="Duplicate card"):
        calc.multi_way([["As", "Kd"], ["As", "Qc"], ["7s", "8s"]])


def test_range_vs_range_rejects_duplicate_board_card():
    with pytest.raises(ValueError, match="Duplicate card"):
        calc.range_vs_range("QQ+", "22-99", board=["As", "As", "2h"])


def test_heads_up_accepts_disjoint_cards():
    # Sanity check: the validation doesn't false-positive on a clean input.
    eq = calc.heads_up(["As", "Ks"], ["Qd", "Jd"], board=["2c", "7d", "9h"])
    assert 0.0 <= eq <= 1.0
