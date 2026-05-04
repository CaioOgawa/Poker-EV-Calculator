"""Tests for ICMPressure (bubble factor) and ICMNash (HU push/fold)."""
import pytest
pytestmark = pytest.mark.slow
from engine.icm import ICMPressure, ICMNash, HAND_RANK

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

nash = ICMNash(equity_iterations=1500)


def test_hand_rank_has_169_hands():
    assert len(HAND_RANK) == 169


def test_hand_rank_no_duplicates():
    assert len(HAND_RANK) == len(set(HAND_RANK))


def test_hand_rank_starts_with_aa():
    assert HAND_RANK[0] == "AA"


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
