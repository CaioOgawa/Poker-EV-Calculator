"""Tests for ICMPressure (bubble factor) and ICMNash (HU push/fold)."""

import pytest

from engine.icm import ICMPressure, ICMNash, HAND_RANK

# No `slow` marker: ICMNash now looks equity up in a precomputed table
# (docs/AUDITORIA-2026-08-26.md item I3) instead of running Monte Carlo per
# query, so solve_hu is milliseconds, not seconds — this whole module is
# fast enough for the default suite.

# ------------------------------------------------------------------
# ICMPressure — bubble factor
# ------------------------------------------------------------------

pressure = ICMPressure()
STACKS_3 = [5000, 3000, 2000]
PAYOUTS_3 = [0.5, 0.3, 0.2]


def test_bf_always_positive():
    bf = pressure.bubble_factor(STACKS_3, PAYOUTS_3, hero=2, villain=0)
    assert bf > 0


def test_bf_chip_leader_vs_shortstack_higher_than_reverse():
    # Chip leader has more to lose → higher BF when they risk chips vs the short stack
    bf_cl_vs_ss = pressure.bubble_factor(STACKS_3, PAYOUTS_3, hero=0, villain=2)
    bf_ss_vs_cl = pressure.bubble_factor(STACKS_3, PAYOUTS_3, hero=2, villain=0)
    assert bf_cl_vs_ss > bf_ss_vs_cl


def test_bf_equal_stacks_symmetric():
    stacks = [5000, 5000, 5000]
    payouts = [0.5, 0.3, 0.2]
    bf_01 = pressure.bubble_factor(stacks, payouts, 0, 1)
    bf_10 = pressure.bubble_factor(stacks, payouts, 1, 0)
    assert abs(bf_01 - bf_10) < 1e-9


def test_bf_eq_base_hero_override_matches_computed():
    # docs/AUDITORIA-2026-08-26.md item I2: all_bubble_factors passes a
    # precomputed eq_base_hero to skip redundant baseline solves — must not
    # change the result vs letting bubble_factor compute it itself.
    from engine.icm.model import ICMModel

    eq_base_hero = ICMModel().equity_of(0, STACKS_3, PAYOUTS_3)
    without_override = pressure.bubble_factor(STACKS_3, PAYOUTS_3, hero=0, villain=2)
    with_override = pressure.bubble_factor(
        STACKS_3, PAYOUTS_3, hero=0, villain=2, eq_base_hero=eq_base_hero
    )
    assert without_override == with_override


def test_bf_diagonal_is_one_in_matrix():
    matrix = pressure.all_bubble_factors(STACKS_3, PAYOUTS_3)
    for i in range(3):
        assert matrix[i][i] == 1.0


def test_bf_matrix_shape():
    matrix = pressure.all_bubble_factors(STACKS_3, PAYOUTS_3)
    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)


def test_bf_matrix_off_diagonal_positive():
    matrix = pressure.all_bubble_factors(STACKS_3, PAYOUTS_3)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert matrix[i][j] > 0


def test_pressure_ranking_length():
    ranking = pressure.pressure_ranking(STACKS_3, PAYOUTS_3)
    assert len(ranking) == 3


def test_pressure_ranking_all_players_present():
    ranking = pressure.pressure_ranking(STACKS_3, PAYOUTS_3)
    indices = [idx for idx, _ in ranking]
    assert sorted(indices) == [0, 1, 2]


def test_pressure_ranking_sorted_descending():
    ranking = pressure.pressure_ranking(STACKS_3, PAYOUTS_3)
    values = [v for _, v in ranking]
    assert values == sorted(values, reverse=True)


# ------------------------------------------------------------------
# ICMNash — HU push/fold
# ------------------------------------------------------------------

nash = ICMNash()


def test_hand_rank_has_169_hands():
    assert len(HAND_RANK) == 169


def test_hand_rank_no_duplicates():
    assert len(HAND_RANK) == len(set(HAND_RANK))


def test_hand_rank_starts_with_aa():
    assert HAND_RANK[0] == "AA"


def test_all_in_equity_uses_effective_stack_not_full_stack():
    # docs/AUDITORIA-2026-08-26.md item E1, second worked example: hero wins
    # an all-in for the effective stack (4000) at [4000, 6000], ending at
    # [8000, 2000] — villain survives with 2000, doesn't bust. The old,
    # unfixed code hardcoded this branch to the winner-takes-all payout
    # ([0.65, 0.35]); the correct ICM split for [8000, 2000] is [0.59, 0.41].
    result = nash._all_in_equity(8000, 2000, [0.65, 0.35])
    assert result == pytest.approx([0.59, 0.41], abs=1e-4)


def test_all_in_equity_zero_stack_hits_elimination_guard():
    # ICMModel.equity assigns 0.0 (not the consolation payout) to a
    # zero-chip entry — this is the branch that keeps the guard from
    # silently dropping the loser's second-place equity in the equal-stack
    # and short-stack-busts cases.
    assert nash._all_in_equity(10000, 0, [0.65, 0.35]) == [0.65, 0.35]
    assert nash._all_in_equity(0, 10000, [0.65, 0.35]) == [0.35, 0.65]


def test_solve_hu_returns_nashresult():
    from engine.icm.nash import NashResult

    result = nash.solve_hu([2000, 8000], [0.65, 0.35], sb=50, bb=100, max_iter=2)
    assert isinstance(result, NashResult)


def test_solve_hu_shortstack_pushes_wide():
    # 20BB short stack should push near 100% of hands
    result = nash.solve_hu([2000, 8000], [0.65, 0.35], sb=50, bb=100, max_iter=2)
    assert len(result.push_range) > 100


def test_solve_hu_push_range_starts_with_aa():
    result = nash.solve_hu([2000, 8000], [0.65, 0.35], sb=50, bb=100, max_iter=2)
    if result.push_range:
        assert result.push_range[0] == "AA"


def test_solve_hu_push_range_is_contiguous_from_top():
    # Push range should be a prefix of HAND_RANK (strongest hands first)
    result = nash.solve_hu([2000, 8000], [0.65, 0.35], sb=50, bb=100, max_iter=2)
    if result.push_range:
        expected = HAND_RANK[: result.push_threshold + 1]
        assert result.push_range == expected


def test_solve_hu_call_range_is_contiguous_from_top():
    result = nash.solve_hu([2000, 8000], [0.65, 0.35], sb=50, bb=100, max_iter=2)
    if result.call_range:
        expected = HAND_RANK[: result.call_threshold + 1]
        assert result.call_range == expected


def test_solve_hu_equal_stacks_moderate_ranges():
    # Equal stacks: neither player should push/call 0% or 100%
    result = nash.solve_hu([5000, 5000], [0.65, 0.35], sb=50, bb=100, max_iter=3)
    assert 10 < len(result.push_range) < 169
    assert len(result.call_range) >= 0  # at least 0 (can fold everything if ICM dictates)


def test_solve_hu_push_threshold_matches_range_length():
    result = nash.solve_hu([5000, 5000], [0.65, 0.35], sb=50, bb=100, max_iter=2)
    if result.push_range:
        assert result.push_threshold == len(result.push_range) - 1
    else:
        assert result.push_threshold == -1


def test_range_weights_sum_to_full_combo_universe():
    # Card-removal self-check (E2): for any fixed hand, its available combos
    # summed across all 169 opponent canonical hands must equal exactly
    # C(50, 2) = 1225 — the deck minus the fixed hand's own two known cards.
    # A wrong entry in the block matrix would silently under/over-count
    # blocked combos without this catching it.
    for hand in ["AA", "AKs", "AKo", "72o", "T9s"]:
        assert nash._range_weights(hand, HAND_RANK).sum() == pytest.approx(1225)


def test_solve_hu_deep_stacks_converges_to_stable_range():
    # Regression: best-response iteration at [6000, 4000] (60bb/40bb) doesn't
    # settle at a fixed point. Before E1 (effective-stack showdown) + E2
    # (combo-weighted, blocker-aware fold equity) it orbited a 3-state cycle
    # ranging from a near-empty push range to "shove every hand" (threshold 5
    # <-> 168), averaging to push_threshold=64/call_threshold=53 (push len
    # 65, call len 54). After the fix it settles into a tighter 2-state cycle
    # (threshold 32 <-> 135), averaging to push_threshold=84/call_threshold=42
    # (push len 85, call len 43). The bounds below bracket the post-fix
    # values on both sides and, unlike the old `20 < push < 150`, exclude the
    # pre-fix result on both push and call — so this actually regresses
    # against E1/E2 breaking, not just against a gross algorithm failure.
    result = nash.solve_hu([6000, 4000], [0.65, 0.35], sb=50, bb=100)
    assert 75 < len(result.push_range) < 95
    assert 35 < len(result.call_range) < 50
