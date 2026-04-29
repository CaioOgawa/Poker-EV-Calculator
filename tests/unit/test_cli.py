"""CLI integration tests using Typer's CliRunner."""
import json
import pytest
from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


# ── eval ──────────────────────────────────────────────────────────────────────

def test_eval_straight_flush():
    result = runner.invoke(app, ["eval", "--hero", "AsKs", "--board", "QsJsTs"])
    assert result.exit_code == 0, result.output
    assert "Straight Flush" in result.output or "Royal Flush" in result.output


def test_eval_json_keys():
    result = runner.invoke(app, ["eval", "--hero", "AsKs", "--board", "QsJsTs", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert set(data.keys()) >= {"hand", "board", "rank", "class", "percentile"}


def test_eval_json_percentile_range():
    result = runner.invoke(app, ["eval", "--hero", "2h2d", "--board", "7c8c9c", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert 0.0 <= data["percentile"] <= 100.0


def test_eval_json_rank_positive():
    result = runner.invoke(app, ["eval", "--hero", "AsKs", "--board", "QsJsTs", "--json"])
    data = json.loads(result.output)
    assert data["rank"] > 0


# ── equity ────────────────────────────────────────────────────────────────────

def test_equity_vs_range():
    result = runner.invoke(app, ["equity", "--hero", "AsKs", "--villain", "QQ+,AKs"])
    assert result.exit_code == 0, result.output
    assert "%" in result.output


def test_equity_vs_hand_json():
    result = runner.invoke(app, ["equity", "--hero", "AsKs", "--villain", "QhQd", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "equity" in data
    assert 0.0 <= data["equity"] <= 100.0
    assert data["mode"] == "vs hand"


def test_equity_vs_range_json():
    result = runner.invoke(app, ["equity", "--hero", "AsKs", "--villain", "22+", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["mode"] == "vs range"
    assert 0.0 <= data["equity"] <= 100.0


def test_equity_with_board():
    result = runner.invoke(
        app,
        ["equity", "--hero", "AsKs", "--villain", "QQ+", "--board", "QhJsTs", "--json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["board"] == "QhJsTs"


# ── icm ───────────────────────────────────────────────────────────────────────

def test_icm_output_has_table():
    result = runner.invoke(app, ["icm", "--stacks", "5000,3000,2000", "--payouts", "0.5,0.3,0.2"])
    assert result.exit_code == 0, result.output
    assert "ICM" in result.output


def test_icm_json_length():
    result = runner.invoke(
        app, ["icm", "--stacks", "5000,3000,2000", "--payouts", "0.5,0.3,0.2", "--json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 3


def test_icm_json_fractions_sum_to_one():
    result = runner.invoke(
        app, ["icm", "--stacks", "5000,3000,2000", "--payouts", "0.5,0.3,0.2", "--json"]
    )
    data = json.loads(result.output)
    total = sum(r["icm_fraction"] for r in data)
    assert abs(total - 1.0) < 1e-4


def test_icm_json_value_uses_total():
    result = runner.invoke(
        app,
        [
            "icm",
            "--stacks", "5000,5000",
            "--payouts", "0.6,0.4",
            "--total", "2000",
            "--json",
        ],
    )
    data = json.loads(result.output)
    # Equal stacks → equal ICM fractions → each worth 0.5 × 2000 = 1000
    assert abs(data[0]["icm_value"] - 1000.0) < 1.0
    assert abs(data[1]["icm_value"] - 1000.0) < 1.0


# ── roi ───────────────────────────────────────────────────────────────────────

def test_roi_output_contains_roi():
    result = runner.invoke(app, ["roi", "--buyins", "100,100,100", "--cashes", "0,200,500"])
    assert result.exit_code == 0, result.output
    assert "ROI" in result.output


def test_roi_json_values():
    result = runner.invoke(
        app, ["roi", "--buyins", "100,100,100", "--cashes", "0,200,500", "--json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    # invested=300, cashed=700, roi=(700-300)/300 = 133.3%
    assert data["roi"] == pytest.approx(133.3, abs=0.1)
    # itm_rate=2/3=66.7%
    assert data["itm_rate"] == pytest.approx(66.7, abs=0.1)
    assert data["tournaments"] == 3
    assert data["profit"] == pytest.approx(400.0)


def test_roi_json_negative_roi():
    result = runner.invoke(
        app, ["roi", "--buyins", "100,100,100", "--cashes", "0,0,50", "--json"]
    )
    data = json.loads(result.output)
    assert data["roi"] < 0
