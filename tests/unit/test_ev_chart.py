"""EVChart tests — use the non-interactive Agg backend so plotting never
blocks or requires a display, and assert on the saved file instead of pixels.
"""

import os

import matplotlib

matplotlib.use("Agg")

import pytest

from output.charts.ev_chart import EVChart


def test_plot_bankroll_saves_file(tmp_path):
    chart = EVChart()
    out = tmp_path / "bankroll.png"
    chart.plot_bankroll([10.0, -5.0, 20.0, 3.0], save_path=str(out))
    assert out.exists()
    assert os.path.getsize(out) > 0


def test_plot_bankroll_with_ev_line_saves_file(tmp_path):
    chart = EVChart()
    out = tmp_path / "bankroll_with_ev.png"
    chart.plot_bankroll(
        [10.0, -5.0, 20.0, 3.0],
        ev_line=[2.0, 4.0, 6.0, 8.0],
        title="Custom Title",
        save_path=str(out),
    )
    assert out.exists()


def test_plot_bankroll_empty_results_does_not_raise(tmp_path):
    chart = EVChart()
    out = tmp_path / "empty.png"
    chart.plot_bankroll([], save_path=str(out))
    assert out.exists()


def test_matplotlib_available_flag_is_true_in_this_environment():
    # If this fails, MATPLOTLIB_AVAILABLE's ImportError guard needs testing
    # via a separate process with matplotlib uninstalled — not worth the
    # complexity here since the guard itself is a one-line try/except.
    from output.charts.ev_chart import MATPLOTLIB_AVAILABLE

    assert MATPLOTLIB_AVAILABLE is True


def test_plot_bankroll_raises_without_matplotlib(monkeypatch):
    import output.charts.ev_chart as mod

    monkeypatch.setattr(mod, "MATPLOTLIB_AVAILABLE", False)
    with pytest.raises(ImportError):
        EVChart().plot_bankroll([1.0, 2.0])
