"""CLI integration tests using Typer's CliRunner."""

import json
from pathlib import Path

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


def test_eval_table_top_percent_is_not_inverted():
    # Royal flush: percentile() ~= 1.0 (nuts), so the displayed "Top X%"
    # label must read close to 0%, not close to 100%.
    result = runner.invoke(app, ["eval", "--hero", "AsKs", "--board", "QsJsTs"])
    assert result.exit_code == 0, result.output
    assert "Top 0.0%" in result.output
    assert "Top 99.9%" not in result.output


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
            "--stacks",
            "5000,5000",
            "--payouts",
            "0.6,0.4",
            "--total",
            "2000",
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
    result = runner.invoke(app, ["roi", "--buyins", "100,100,100", "--cashes", "0,0,50", "--json"])
    data = json.loads(result.output)
    assert data["roi"] < 0


# ── nash ──────────────────────────────────────────────────────────────────────


def test_nash_output_has_table():
    result = runner.invoke(
        app, ["nash", "--stacks", "3000,7000", "--payouts", "0.65,0.35", "--json"]
    )
    assert result.exit_code == 0, result.output


def test_nash_json_keys():
    result = runner.invoke(
        app, ["nash", "--stacks", "3000,7000", "--payouts", "0.65,0.35", "--json"]
    )
    data = json.loads(result.output)
    assert set(data.keys()) == {
        "hero_stack",
        "villain_stack",
        "push_range",
        "push_pct",
        "call_range",
        "call_pct",
    }


def test_nash_short_stack_pushes_wide():
    # 3bb effective stack (sb=50, bb=100, stack=300) — hero should be
    # shoving the large majority of hands.
    result = runner.invoke(
        app,
        [
            "nash",
            "--stacks",
            "300,300",
            "--sb",
            "50",
            "--bb",
            "100",
            "--payouts",
            "0.65,0.35",
            "--json",
        ],
    )
    data = json.loads(result.output)
    assert data["push_pct"] > 70.0


def test_nash_rejects_wrong_stack_count():
    result = runner.invoke(app, ["nash", "--stacks", "3000,7000,1000", "--payouts", "0.5,0.3,0.2"])
    assert result.exit_code != 0
    assert "exactly 2 values" in result.output


# ── bankroll ──────────────────────────────────────────────────────────────────


def test_bankroll_output_has_table():
    result = runner.invoke(app, ["bankroll", "--bankroll", "5000", "--winrate", "5", "--std", "90"])
    assert result.exit_code == 0, result.output
    assert "RoR" in result.output


def test_bankroll_json_single_row_by_default():
    result = runner.invoke(
        app, ["bankroll", "--bankroll", "5000", "--winrate", "5", "--std", "90", "--json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["ror"] <= 5.0


def test_bankroll_all_returns_every_standard_stake():
    result = runner.invoke(
        app,
        ["bankroll", "--bankroll", "5000", "--winrate", "5", "--std", "90", "--all", "--json"],
    )
    data = json.loads(result.output)
    assert len(data) == 9  # STANDARD_STAKES
    # Fixed $ bankroll buys fewer BBs at a bigger stake, so RoR only rises.
    rors = [r["ror"] for r in data]
    assert rors == sorted(rors)


def test_bankroll_nonpositive_winrate_errors_cleanly():
    result = runner.invoke(
        app, ["bankroll", "--bankroll", "5000", "--winrate", "-1", "--std", "90"]
    )
    assert result.exit_code != 0
    assert "positive" in result.output.lower()


# ── session ───────────────────────────────────────────────────────────────────

_WALK_HAND = """\
PokerStars Hand #200000001: Hold'em No Limit ($0.05/$0.10 USD) - 2024/01/01 12:05:00 ET
Table 'Test I' 6-max Seat #1 is the button
Seat 1: Hero ($10.00 in chips)
Seat 2: Player2 ($10.00 in chips)
Hero: posts small blind $0.05
Player2: posts big blind $0.10
*** HOLE CARDS ***
Dealt to Hero [7c 2d]
Hero: folds
Player2 collected $0.15 from pot
*** SUMMARY ***
Total pot $0.15 | Rake $0.00
Seat 1: Hero (button) (small blind) folded before Flop
Seat 2: Player2 (big blind) collected ($0.15)
"""


def test_session_missing_file_errors_cleanly():
    result = runner.invoke(app, ["session", "--import", "/no/such/file.txt"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_session_json_summary(tmp_path):
    hh = tmp_path / "history.txt"
    hh.write_text(_WALK_HAND)
    result = runner.invoke(app, ["session", "--import", str(hh), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["hands_played"] == 1
    assert data["total_result"] == pytest.approx(-0.5)  # -0.05 / 0.10 bb


def test_session_report_flag_shows_leak_table(tmp_path):
    hh = tmp_path / "history.txt"
    hh.write_text(_WALK_HAND)
    result = runner.invoke(app, ["session", "--import", str(hh), "--report"])
    assert result.exit_code == 0, result.output
    assert "uncategorized" in result.output


# ── icm-pressure ─────────────────────────────────────────────────────────────


def test_icm_pressure_output_has_table():
    result = runner.invoke(
        app, ["icm-pressure", "--stacks", "5000,3000,2000", "--payouts", "0.5,0.3,0.2"]
    )
    assert result.exit_code == 0, result.output
    assert "Bubble Factor" in result.output


def test_icm_pressure_json_diagonal_is_one():
    result = runner.invoke(
        app, ["icm-pressure", "--stacks", "5000,3000,2000", "--payouts", "0.5,0.3,0.2", "--json"]
    )
    data = json.loads(result.output)
    matrix = data["matrix"]
    assert [matrix[i][i] for i in range(3)] == [1.0, 1.0, 1.0]
    assert len(data["ranking"]) == 3


def test_icm_pressure_chip_leader_is_most_pressured():
    # Chip leader has the most ICM equity at stake and the least to gain
    # proportionally from busting a micro stack — highest average BF, not
    # the short stack (who has little left to lose).
    result = runner.invoke(
        app, ["icm-pressure", "--stacks", "9000,900,100", "--payouts", "0.5,0.3,0.2", "--json"]
    )
    data = json.loads(result.output)
    assert data["ranking"][0]["player"] == 1


# ── range-heatmap / icm-chart ───────────────────────────────────────────────


def test_range_heatmap_saves_file(tmp_path):
    out = tmp_path / "heatmap.png"
    result = runner.invoke(app, ["range-heatmap", "--range", "22+,ATs+", "--save", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.stat().st_size > 0


def test_range_heatmap_invalid_range_errors_cleanly():
    result = runner.invoke(
        app, ["range-heatmap", "--range", "not-a-range!!", "--save", "/tmp/x.png"]
    )
    assert result.exit_code != 0


def test_icm_chart_saves_file(tmp_path):
    out = tmp_path / "bubble.png"
    result = runner.invoke(
        app,
        ["icm-chart", "--stacks", "5000,3000,2000", "--payouts", "0.5,0.3,0.2", "--save", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.stat().st_size > 0


# ── sim ───────────────────────────────────────────────────────────────────────

_SIM_ARGS = [
    "sim",
    "--winrate",
    "5",
    "--std",
    "90",
    "--bankroll",
    "2000",
    "--num-careers",
    "500",
    "--num-sessions",
    "50",
]


def test_sim_output_has_summary():
    result = runner.invoke(app, _SIM_ARGS)
    assert result.exit_code == 0, result.output
    assert "Ruin rate" in result.output


def test_sim_json_keys():
    result = runner.invoke(app, [*_SIM_ARGS, "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert set(data.keys()) >= {
        "ruin_rate",
        "mean_final_bb",
        "final_p5_bb",
        "final_p50_bb",
        "final_p95_bb",
        "hands",
        "p5",
        "p50",
        "p95",
    }
    assert len(data["hands"]) == 51  # num_sessions + 1
    assert data["hands"][0] == 0
    assert data["hands"][-1] == 50 * 100  # num_sessions * hands_per_session


def test_sim_same_seed_reproduces_same_result():
    r1 = runner.invoke(app, [*_SIM_ARGS, "--seed", "7", "--json"])
    r2 = runner.invoke(app, [*_SIM_ARGS, "--seed", "7", "--json"])
    assert json.loads(r1.output) == json.loads(r2.output)


def test_sim_multitable_beats_single_table_mean():
    single = runner.invoke(app, [*_SIM_ARGS, "--seed", "1", "--json"])
    multi = runner.invoke(app, [*_SIM_ARGS, "--multitable", "4", "--seed", "1", "--json"])
    single_mean = json.loads(single.output)["mean_final_bb"]
    multi_mean = json.loads(multi.output)["mean_final_bb"]
    assert multi_mean > single_mean


# ── solve ─────────────────────────────────────────────────────────────────────

# River-only (4-card board) is the fast solve path — a 3-card/flop board
# takes minutes even with small ranges (see docs/ROADMAP.md), so no CLI
# test exercises that path.
_SOLVE_ARGS = [
    "solve",
    "--oop",
    "AKs",
    "--ip",
    "QJs",
    "--board",
    "2c7dThKs",
    "--pot",
    "100",
    "--iterations",
    "300",
    "--seed",
    "1",
]


def test_solve_output_has_summary():
    result = runner.invoke(app, _SOLVE_ARGS)
    assert result.exit_code == 0, result.output
    assert "Exploitability" in result.output


def test_solve_json_ev_sums_to_pot():
    result = runner.invoke(app, [*_SOLVE_ARGS, "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["oop_ev"] + data["ip_ev"] == pytest.approx(data["pot"], abs=1e-3)


def test_solve_rejects_bad_board_length():
    result = runner.invoke(
        app, ["solve", "--oop", "AKs", "--ip", "QJs", "--board", "2c7dTh9h5c", "--pot", "100"]
    )
    assert result.exit_code != 0


# ── vision detect ────────────────────────────────────────────────────────────
#
# Trained weights (*.pt) and raw screenshots are gitignored (see docs/
# AUDITORIA-2026-08-26.md item F8 — not backed up anywhere yet), so these
# only run where they happen to exist on disk (this machine); everywhere
# else they skip instead of failing.

try:
    import cv2 as _cv2  # noqa: F401

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

_WEIGHTS_PATH = Path("vision/models/table_detector/weights/best.pt")
_IMAGES_DIR = Path("vision/data/raw/pokerstars/valid/images")
_SAMPLE_IMAGE = next(_IMAGES_DIR.glob("*.jpg"), None) if _IMAGES_DIR.exists() else None

requires_cv2 = pytest.mark.skipif(not _CV2_AVAILABLE, reason="opencv-python not installed")
requires_trained_model = pytest.mark.skipif(
    not (_CV2_AVAILABLE and _WEIGHTS_PATH.exists() and _SAMPLE_IMAGE is not None),
    reason="trained weights or sample screenshot not present on this machine",
)


@requires_trained_model
def test_vision_detect_output_has_seats_table():
    result = runner.invoke(app, ["vision", "detect", str(_SAMPLE_IMAGE)])
    assert result.exit_code == 0, result.output
    assert "Seats" in result.output


@requires_trained_model
def test_vision_detect_raw_flag_returns_bbox_json():
    result = runner.invoke(app, ["vision", "detect", str(_SAMPLE_IMAGE), "--raw"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "cards" in data and "players" in data


@requires_cv2
def test_vision_detect_missing_file_errors_cleanly():
    # cv2.imread returns None for a missing path before any model gets
    # loaded, so this doesn't need trained weights to exercise the error path.
    result = runner.invoke(app, ["vision", "detect", "/no/such/image.jpg"])
    assert result.exit_code != 0
