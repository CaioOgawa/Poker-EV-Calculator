"""Tests for vision/loop.py.

The throttle-logic tests use a fake detector/reader (duck-typed the same
way build_game_state consumes a real TableDetector/CardReader) so they run
everywhere, no trained model needed. The directory-replay test is gated
behind real weights + real screenshots like the rest of the vision tests.
"""

from pathlib import Path

import numpy as np
import pytest

from vision.loop import frames_from_directory, run

_TINY_FRAME = np.zeros((10, 10, 3), dtype=np.uint8)
_EMPTY_PLAYERS = {
    s: {"bet": None, "dealer": False, "stack": None, "in_hand": False} for s in range(6)
}


def _card(card: str, x: float) -> dict:
    return {"card": card, "bbox": [x, 100, x + 20, 140], "conf": 0.9}


def _raw(board_cards: list[str]) -> dict:
    return {
        "cards": [_card(c, i * 25) for i, c in enumerate(board_cards)],
        "pot": None,
        "table_bet": None,
        "board_stage": None,
        "players": _EMPTY_PLAYERS,
    }


class _FakeDetector:
    def __init__(self, raw_per_frame: list[dict]):
        self._queue = list(raw_per_frame)

    def detect(self, arr):
        return self._queue.pop(0)


class _FakeReader:
    def read_number(self, crop):
        return None


def _run_fake(raw_per_frame: list[dict], **kwargs):
    seen = []
    frames = [(float(i), _TINY_FRAME) for i in range(len(raw_per_frame))]
    n = run(
        frames,
        lambda ts, state: seen.append((ts, state.board)),
        detector=_FakeDetector(raw_per_frame),
        reader=_FakeReader(),
        calibration=None,
        **kwargs,
    )
    return n, seen


def test_run_returns_total_frame_count():
    n, _ = _run_fake([_raw(["As", "Kd", "Qc"])] * 3)
    assert n == 3


def test_run_skips_unchanged_state_by_default():
    n, seen = _run_fake(
        [_raw(["As", "Kd", "Qc"]), _raw(["As", "Kd", "Qc"]), _raw(["As", "Kd", "Qc", "Th"])]
    )
    assert n == 3
    assert len(seen) == 2  # frame 1 duplicates frame 0's state and is skipped
    assert [ts for ts, _ in seen] == [0.0, 2.0]


def test_run_fires_every_frame_when_skip_unchanged_is_false():
    n, seen = _run_fake([_raw(["As", "Kd", "Qc"]), _raw(["As", "Kd", "Qc"])], skip_unchanged=False)
    assert len(seen) == 2


def test_run_calls_on_state_zero_times_for_empty_source():
    n = run([], lambda ts, state: pytest.fail("should not be called"))
    assert n == 0


# ── directory replay (needs real weights + real screenshots) ────────────────

_WEIGHTS_PATH = Path("vision/models/table_detector/weights/best.pt")
_IMAGES_DIR = Path("vision/data/raw/pokerstars/valid/images")

requires_trained_model = pytest.mark.skipif(
    not (_WEIGHTS_PATH.exists() and _IMAGES_DIR.exists() and any(_IMAGES_DIR.glob("*.jpg"))),
    reason="trained weights or sample screenshots not present on this machine",
)


@requires_trained_model
def test_frames_from_directory_and_run_end_to_end():
    frames = frames_from_directory(_IMAGES_DIR)
    seen = []
    n = run(frames, lambda ts, state: seen.append(state))
    assert n > 0
    assert len(seen) <= n  # throttling never fires more than it reads
    assert all(s.street in {"preflop", "flop", "turn", "river", "unknown"} for s in seen)
