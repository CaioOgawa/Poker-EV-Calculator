import json
import random

import pytest

from engine.gto.chart import PreflopChart
from engine.gto.mixed_strategy import MixedStrategy
from engine.gto.solver_bridge import NotImplementedBridge


# ---------------------------------------------------------------------------
# MixedStrategy
# ---------------------------------------------------------------------------


def test_sample_deterministic_single_action():
    strat = MixedStrategy()
    assert strat.sample({"push": 1.0}) == "push"


def test_sample_respects_seed_reproducibility():
    freqs = {"push": 0.4, "fold": 0.6}
    a = MixedStrategy(seed=42).sample(freqs)
    b = MixedStrategy(seed=42).sample(freqs)
    assert a == b


def test_sample_distribution_matches_frequency():
    freqs = {"push": 0.8, "fold": 0.2}
    rng = random.Random(0)
    counts = {"push": 0, "fold": 0}
    n = 5000
    for _ in range(n):
        counts[MixedStrategy().sample(freqs, rng=rng)] += 1
    assert abs(counts["push"] / n - 0.8) < 0.03


def test_sample_normalizes_totals_not_equal_to_one():
    # 2:1 ratio should behave like {push: 2/3, fold: 1/3} even though it sums to 3.
    freqs = {"push": 2.0, "fold": 1.0}
    rng = random.Random(1)
    counts = {"push": 0, "fold": 0}
    n = 5000
    for _ in range(n):
        counts[MixedStrategy().sample(freqs, rng=rng)] += 1
    assert abs(counts["push"] / n - 2 / 3) < 0.03


def test_sample_empty_frequencies_raises():
    with pytest.raises(ValueError):
        MixedStrategy().sample({})


def test_sample_negative_frequency_raises():
    with pytest.raises(ValueError):
        MixedStrategy().sample({"push": -0.1, "fold": 1.1})


def test_sample_all_zero_frequencies_raises():
    with pytest.raises(ValueError):
        MixedStrategy().sample({"push": 0.0, "fold": 0.0})


def test_expected_value_weighted_average():
    strat = MixedStrategy()
    ev = strat.expected_value({"push": 0.6, "fold": 0.4}, {"push": 10.0, "fold": 0.0})
    assert abs(ev - 6.0) < 1e-9


def test_expected_value_ignores_actions_missing_from_values():
    strat = MixedStrategy()
    ev = strat.expected_value({"push": 0.5, "limp": 0.5}, {"push": 10.0})
    assert abs(ev - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# PreflopChart — construction & normalization
# ---------------------------------------------------------------------------


def test_frequencies_lookup_pair():
    chart = PreflopChart({"AA": {"push": 1.0}})
    assert chart.frequencies("AA") == {"push": 1.0}


def test_hand_normalization_reorders_ranks():
    chart = PreflopChart({"AKs": {"push": 1.0}})
    assert chart.frequencies("KAs") == {"push": 1.0}


def test_hand_normalization_lowercase_suit():
    chart = PreflopChart({"AKo": {"push": 1.0}})
    assert chart.frequencies("AkO") == {"push": 1.0}


def test_pair_with_mismatched_ranks_raises():
    with pytest.raises(ValueError):
        PreflopChart({"AK": {"push": 1.0}})


def test_invalid_suit_suffix_raises():
    with pytest.raises(ValueError):
        PreflopChart({"AKx": {"push": 1.0}})


def test_invalid_length_raises():
    with pytest.raises(ValueError):
        PreflopChart({"AKQ": {"push": 1.0}})


def test_missing_hand_without_default_raises_key_error():
    chart = PreflopChart({"AA": {"push": 1.0}})
    with pytest.raises(KeyError):
        chart.frequencies("72o")


def test_missing_hand_with_default_action():
    chart = PreflopChart({"AA": {"push": 1.0}}, default_action="fold")
    assert chart.frequencies("72o") == {"fold": 1.0}


# ---------------------------------------------------------------------------
# PreflopChart — sampling, ranges
# ---------------------------------------------------------------------------


def test_action_samples_from_frequencies():
    chart = PreflopChart({"AA": {"push": 1.0}})
    assert chart.action("AA") == "push"


def test_range_for_action_default_threshold():
    chart = PreflopChart(
        {
            "AA": {"push": 1.0},
            "72o": {"push": 0.2, "fold": 0.8},
            "KK": {"push": 0.5, "fold": 0.5},
        }
    )
    result = chart.range_for_action("push")
    assert "AA" in result
    assert "KK" in result
    assert "72o" not in result


def test_range_for_action_custom_threshold():
    chart = PreflopChart({"72o": {"push": 0.2, "fold": 0.8}})
    assert chart.range_for_action("push", min_frequency=0.1) == ["72o"]
    assert chart.range_for_action("push", min_frequency=0.3) == []


def test_combo_weighted_range_includes_partial_frequencies():
    chart = PreflopChart({"KK": {"push": 0.5, "fold": 0.5}, "22": {"fold": 1.0}})
    weighted = dict(chart.combo_weighted_range("push"))
    assert weighted == {"KK": 0.5}


# ---------------------------------------------------------------------------
# PreflopChart — CSV / JSON loading
# ---------------------------------------------------------------------------


def test_from_csv_loads_chart(tmp_path):
    path = tmp_path / "chart.csv"
    path.write_text("hand,push,fold\nAA,1.0,0.0\n72o,0.1,0.9\n")
    chart = PreflopChart.from_csv(path)
    assert chart.frequencies("AA") == {"push": 1.0, "fold": 0.0}
    assert chart.frequencies("72o") == {"push": 0.1, "fold": 0.9}


def test_from_csv_blank_cells_omitted(tmp_path):
    path = tmp_path / "chart.csv"
    path.write_text("hand,push,fold\nAA,1.0,\n")
    chart = PreflopChart.from_csv(path)
    assert chart.frequencies("AA") == {"push": 1.0}


def test_from_csv_missing_hand_column_raises(tmp_path):
    path = tmp_path / "chart.csv"
    path.write_text("push,fold\n1.0,0.0\n")
    with pytest.raises(ValueError):
        PreflopChart.from_csv(path)


def test_from_json_loads_chart(tmp_path):
    path = tmp_path / "chart.json"
    path.write_text(json.dumps({"AKs": {"push": 1.0}, "72o": {"fold": 1.0}}))
    chart = PreflopChart.from_json(path)
    assert chart.frequencies("AKs") == {"push": 1.0}
    assert chart.frequencies("72o") == {"fold": 1.0}


def test_from_csv_with_default_action(tmp_path):
    path = tmp_path / "chart.csv"
    path.write_text("hand,push\nAA,1.0\n")
    chart = PreflopChart.from_csv(path, default_action="fold")
    assert chart.frequencies("72o") == {"fold": 1.0}


# ---------------------------------------------------------------------------
# Solver bridge stub
# ---------------------------------------------------------------------------


def test_not_implemented_bridge_raises():
    bridge = NotImplementedBridge()
    with pytest.raises(NotImplementedError):
        bridge.load_chart("some_solver_export.txt")
