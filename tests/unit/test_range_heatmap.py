import os

import matplotlib

matplotlib.use("Agg")

from output.charts.range_heatmap import RangeHeatmap, _cell_label, GRID_RANKS


def test_grid_ranks_descending_and_complete():
    assert GRID_RANKS == "AKQJT98765432"
    assert len(GRID_RANKS) == 13


def test_cell_label_diagonal_is_pair():
    assert _cell_label(0, 0) == "AA"
    assert _cell_label(12, 12) == "22"


def test_cell_label_upper_triangle_is_suited_higher_rank_first():
    assert _cell_label(0, 1) == "AKs"


def test_cell_label_lower_triangle_is_offsuit_higher_rank_first():
    assert _cell_label(1, 0) == "AKo"


def test_matrix_from_range_marks_included_hands():
    rh = RangeHeatmap()
    matrix = rh.matrix_from_range("AA,AKs,72o")
    assert matrix[0][0] == 1.0  # AA
    assert matrix[0][1] == 1.0  # AKs
    r7, r2 = GRID_RANKS.index("7"), GRID_RANKS.index("2")
    assert matrix[max(r7, r2)][min(r7, r2)] == 1.0  # 72o


def test_matrix_from_range_excludes_hands_not_present():
    rh = RangeHeatmap()
    matrix = rh.matrix_from_range("AA")
    assert matrix[1][0] == 0.0  # AKo not in range
    assert matrix[0][1] == 0.0  # AKs not in range


def test_matrix_from_range_total_matches_number_of_canonical_hands():
    rh = RangeHeatmap()
    matrix = rh.matrix_from_range("QQ+,AKs")
    total_on = sum(sum(row) for row in matrix)
    # QQ, KK, AA (3 pairs) + AKs = 4 canonical cells
    assert total_on == 4.0


def test_matrix_from_range_plus_notation_expands_correctly():
    rh = RangeHeatmap()
    matrix = rh.matrix_from_range("77+")
    # 77, 88, 99, TT, JJ, QQ, KK, AA = 8 pairs
    total_on = sum(sum(row) for row in matrix)
    assert total_on == 8.0


def test_matrix_from_chart_uses_frequencies():
    class FakeChart:
        def frequencies(self, hand):
            if hand == "AA":
                return {"push": 1.0}
            if hand == "72o":
                return {"push": 0.2, "fold": 0.8}
            raise KeyError(hand)

    rh = RangeHeatmap()
    matrix = rh.matrix_from_chart(FakeChart(), "push")
    assert matrix[0][0] == 1.0  # AA
    r7, r2 = GRID_RANKS.index("7"), GRID_RANKS.index("2")
    assert matrix[max(r7, r2)][min(r7, r2)] == 0.2  # 72o
    assert matrix[0][1] == 0.0  # AKs not in fake chart -> KeyError -> 0.0


def test_plot_saves_file(tmp_path):
    rh = RangeHeatmap()
    matrix = rh.matrix_from_range("QQ+,AKs")
    out = tmp_path / "heatmap.png"
    rh.plot(matrix, save_path=str(out))
    assert out.exists()
    assert os.path.getsize(out) > 0
