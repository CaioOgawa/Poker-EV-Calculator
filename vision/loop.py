"""Live capture loop: frame source -> detect -> state -> callback.

Takes any iterable of `(timestamp, frame)` pairs rather than depending on
`ScreenCapture` directly, so `run()` is testable against a directory of real
screenshots (this repo has 447 of them under vision/data/raw/pokerstars/)
without a live PokerStars table, and swappable for `ScreenCapture.stream()`
in production — see `frames_from_directory` for the offline source, and
`poker vision watch` (cli/main.py) for the live one.

There is no "is it hero's turn" signal to gate on: the trained model's 81
classes (52 cards + flop/turn/river/pot/table_bet + 18 p{N}{b,d,s} + 6
{N}in) don't include action buttons — detecting "fold/call/raise are on
screen" needs new annotations and a retrain, not code (see docs/ROADMAP.md).
So `run()` processes every frame it's given rather than only frames where
it's hero's turn.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np

from vision.calibration import POKERSTARS_6MAX as _DEFAULT_CALIBRATION
from vision.calibration import TableCalibration
from vision.detector.table_detector import TableDetector
from vision.ocr.card_reader import CardReader
from vision.state.game_state import GameState, build_game_state


def frames_from_directory(
    directory: str | Path, pattern: str = "*.jpg"
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield `(index, frame)` for every image in `directory`, sorted by
    filename — a deterministic, no-camera-needed frame source for tests and
    offline replay (e.g. against vision/data/raw/pokerstars/valid/images)."""
    import cv2

    for i, path in enumerate(sorted(Path(directory).glob(pattern))):
        arr = cv2.imread(str(path))
        if arr is not None:
            yield float(i), arr


def run(
    frames: Iterable[tuple[float, np.ndarray]],
    on_state: Callable[[float, GameState], None],
    detector: TableDetector | None = None,
    reader: CardReader | None = None,
    calibration: TableCalibration | None = _DEFAULT_CALIBRATION,
    skip_unchanged: bool = True,
) -> int:
    """Run `build_game_state` over every frame from `frames`, calling
    `on_state(timestamp, state)` for each one.

    `skip_unchanged` (default True) throttles redundant analysis on a live
    feed where most consecutive frames look identical: a frame is skipped
    unless (board, street, hero_hand) differs from the last one that fired.
    Pass False to get every frame's state regardless.

    Returns the number of frames read from `frames` (processed or skipped).
    """
    detector = detector or TableDetector()
    reader = reader or CardReader()

    last_signature = None
    n = 0
    for ts, frame in frames:
        n += 1
        state = build_game_state(frame, detector, reader, calibration)
        signature = (tuple(state.board), state.street, tuple(state.hero_hand or ()))
        if skip_unchanged and signature == last_signature:
            continue
        last_signature = signature
        on_state(ts, state)
    return n
