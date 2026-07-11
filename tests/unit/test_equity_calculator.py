"""Equity calculator tests.

Reference equities from Equilab / PokerStove (preflop, all-in, random runout).
Monte Carlo tolerance: ±2% at 50k iterations with fixed seed.
"""
import pytest
from core.equity.calculator import EquityCalculator

pytestmark = pytest.mark.slow

ITERS = 50_000
SEED = 42
TOL = 0.02  # ±2% Monte Carlo tolerance

calc = EquityCalculator(iterations=ITERS, seed=SEED)


# ------------------------------------------------------------------
# heads_up — known hand vs known hand
# ------------------------------------------------------------------

def test_aa_vs_kk():
    # Reference: ~81.9% (Equilab)
    eq = calc.heads_up(["As", "Ad"], ["Ks", "Kd"], seed=SEED)
    assert abs(eq - 0.819) < TOL


def test_aks_vs_qq():
    # Reference: ~46.3% (Equilab)
    eq = calc.heads_up(["As", "Ks"], ["Qh", "Qd"], seed=SEED)
    assert abs(eq - 0.463) < TOL


def test_equity_sums_to_one():
    hero_eq = calc.heads_up(["Jh", "Js"], ["Ah", "Kd"], seed=SEED)
    villain_eq = calc.heads_up(["Ah", "Kd"], ["Jh", "Js"], seed=SEED)
    # hero + villain should sum to ~1 (ties split 0.5 each)
    assert abs(hero_eq + villain_eq - 1.0) < 0.01


def test_heads_up_on_board():
    # River: hero flush vs villain top pair — hero should be well above 95%
    eq = calc.heads_up(
        ["9s", "8s"],
        ["Ah", "Kd"],
        board=["2s", "5s", "Qs", "3d", "Tc"],
        seed=SEED,
    )
    assert eq > 0.95


def test_heads_up_dominated_hand():
    # 72o vs AA — villain should be ~88%
    eq = calc.heads_up(["7h", "2d"], ["Ac", "Ad"], seed=SEED)
    assert eq < 0.15


# ------------------------------------------------------------------
# vs_range — hero hand vs villain range string
# ------------------------------------------------------------------

def test_vs_range_aa_vs_top_hands():
    # AA vs JJ+,AKs — AA should be around 80%
    eq = calc.vs_range(["As", "Ad"], "JJ+,AKs", seed=SEED)
    assert 0.75 < eq < 0.90


def test_vs_range_blockers_respected():
    # Hero has As Ad — AKs combos are reduced from 4 to 2 (Ks and Kd only)
    # Still should return a valid float without error
    eq = calc.vs_range(["As", "Ad"], "AKs,KK", seed=SEED)
    assert 0.0 < eq < 1.0


def test_vs_range_fully_blocked_raises():
    # Hero As+Ah + board Ad+Ac uses all 4 aces — no AA combo remains
    with pytest.raises(ValueError):
        calc.vs_range(["As", "Ah"], "AA", board=["Ad", "Ac", "2h"], seed=SEED)


# ------------------------------------------------------------------
# multi_way
# ------------------------------------------------------------------

def test_multi_way_equities_sum_to_one():
    equities = calc.multi_way(
        [["As", "Ad"], ["Ks", "Kd"], ["Qh", "Qd"]],
        seed=SEED,
    )
    assert len(equities) == 3
    assert abs(sum(equities) - 1.0) < 0.01


def test_multi_way_aa_leads():
    equities = calc.multi_way(
        [["As", "Ad"], ["Ks", "Kd"], ["Qh", "Qd"]],
        seed=SEED,
    )
    # AA should be clear leader
    assert equities[0] > equities[1]
    assert equities[0] > equities[2]
    assert equities[0] > 0.60


def test_multi_way_two_players_matches_heads_up():
    # 2-player multi_way should match heads_up within tolerance
    mw = calc.multi_way([["As", "Kd"], ["Qh", "Qd"]], seed=SEED)
    hu = calc.heads_up(["As", "Kd"], ["Qh", "Qd"], seed=SEED)
    assert abs(mw[0] - hu) < TOL


# ------------------------------------------------------------------
# range_vs_range
# ------------------------------------------------------------------

def test_range_vs_range_sums_to_one():
    eq_a, eq_b = calc.range_vs_range("AA,KK", "QQ,JJ", seed=SEED)
    assert abs(eq_a + eq_b - 1.0) < 0.01


def test_range_vs_range_aa_kk_beats_qq_jj():
    # AA/KK should be heavy favourite over QQ/JJ
    eq_a, eq_b = calc.range_vs_range("AA,KK", "QQ,JJ", seed=SEED)
    assert eq_a > 0.70


def test_range_vs_range_symmetric_ranges():
    # Identical ranges — each side should be close to 50% (slight variance due to blockers)
    eq_a, eq_b = calc.range_vs_range("AKs,AKo", "AKs,AKo", seed=SEED)
    assert abs(eq_a - 0.5) < 0.05
    assert abs(eq_b - 0.5) < 0.05


def test_range_vs_range_returns_tuple():
    result = calc.range_vs_range("AA", "KK", seed=SEED)
    assert isinstance(result, tuple)
    assert len(result) == 2
