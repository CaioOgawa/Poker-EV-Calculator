import os

import matplotlib
import pytest

matplotlib.use("Agg")

from output.reports.ev_distribution_report import EVDistributionReport, DistributionStats


def test_stats_returns_distribution_stats():
    report = EVDistributionReport()
    stats = report.stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert isinstance(stats, DistributionStats)
    assert stats.n == 5
    assert stats.mean == 3.0


def test_stats_percentiles_ordered():
    report = EVDistributionReport()
    stats = report.stats([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
    assert stats.p5 <= stats.p50 <= stats.p95


def test_stats_empty_raises():
    report = EVDistributionReport()
    with pytest.raises(ValueError):
        report.stats([])


def test_plot_histogram_saves_file(tmp_path):
    report = EVDistributionReport()
    out = tmp_path / "hist.png"
    report.plot_histogram([1.0, -2.0, 3.0, -1.0, 5.0] * 10, save_path=str(out))
    assert out.exists()
    assert os.path.getsize(out) > 0


def test_plot_histogram_with_ev_overlay(tmp_path):
    report = EVDistributionReport()
    out = tmp_path / "hist_ev.png"
    results = [1.0, -2.0, 3.0, -1.0, 5.0] * 10
    ev = [0.5, -0.5, 1.5, 0.0, 2.0] * 10
    report.plot_histogram(results, ev=ev, save_path=str(out))
    assert out.exists()


def test_plot_violin_saves_file(tmp_path):
    report = EVDistributionReport()
    out = tmp_path / "violin.png"
    results = [1.0, -2.0, 3.0, -1.0, 5.0, 2.0, -3.0] * 5
    report.plot_violin(results, save_path=str(out))
    assert out.exists()
    assert os.path.getsize(out) > 0


def test_plot_violin_with_ev_saves_file(tmp_path):
    report = EVDistributionReport()
    out = tmp_path / "violin_ev.png"
    results = [1.0, -2.0, 3.0, -1.0, 5.0, 2.0, -3.0] * 5
    ev = [0.5, -0.5, 1.5, 0.0, 2.0, 1.0, -1.0] * 5
    report.plot_violin(results, ev=ev, save_path=str(out))
    assert out.exists()
