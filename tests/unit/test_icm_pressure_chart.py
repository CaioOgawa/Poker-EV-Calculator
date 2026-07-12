import os

import matplotlib

matplotlib.use("Agg")

from output.charts.icm_pressure_chart import ICMPressureChart


def test_matrix_matches_diagonal_ones():
    chart = ICMPressureChart()
    matrix = chart.matrix([5000, 3000, 2000], [0.5, 0.3, 0.2])
    assert matrix[0][0] == 1.0
    assert matrix[1][1] == 1.0
    assert matrix[2][2] == 1.0


def test_matrix_off_diagonal_is_bubble_factor():
    chart = ICMPressureChart()
    matrix = chart.matrix([5000, 3000, 2000], [0.5, 0.3, 0.2])
    assert matrix[2][0] > 1.0  # short stack facing chip leader — real pressure


def test_plot_saves_file(tmp_path):
    chart = ICMPressureChart()
    out = tmp_path / "icm.png"
    chart.plot([5000, 3000, 2000], [0.5, 0.3, 0.2], save_path=str(out))
    assert out.exists()
    assert os.path.getsize(out) > 0


def test_plot_with_custom_labels(tmp_path):
    chart = ICMPressureChart()
    out = tmp_path / "icm_labels.png"
    chart.plot(
        [5000, 3000, 2000],
        [0.5, 0.3, 0.2],
        labels=["Hero", "Villain1", "Villain2"],
        save_path=str(out),
    )
    assert out.exists()


def test_plot_handles_very_large_bubble_factor(tmp_path):
    # Satellite (top 3 of 4 get an equal share): an overwhelming chip leader
    # has almost no marginal ICM upside left, driving BF very large — this
    # exercises the vmax-clipping path without needing a literal inf.
    chart = ICMPressureChart()
    out = tmp_path / "icm_large.png"
    chart.plot([9000, 300, 300, 400], [1 / 3, 1 / 3, 1 / 3, 0], save_path=str(out))
    assert out.exists()
    assert os.path.getsize(out) > 0
