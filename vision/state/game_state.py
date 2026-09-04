"""Group TableDetector's raw detections into a structured GameState.

TableDetector.detect() returns ungrouped card detections with no hero/board
distinction (see vision/detector/table_detector.py). The only grouping rule
validated against real data so far: community cards cluster tightly around a
shared y-coordinate on the felt (checked against ground-truth labels — a
5-card river board sat within ~1% of image height of each other, while a
stray hole card 13% away in x and y was correctly excluded). Board size is
always 3, 4, or 5; anything else means "no confident board found".

Hole-card ownership (hero vs. an opponent's hand exposed at showdown) is
resolved for cards that fall in the calibrated hero region (see
`vision/calibration.py`) — cards outside it stay in `exposed_cards` with
their bbox rather than being guessed at (no seat-level attribution for
opponents yet).

`street` comes from `len(board)`, not the raw `board_stage` detection: the
"flop"/"turn"/"river" watermark classes are among the model's weaker ones
(mAP50 0.33-0.61, see docs/ROADMAP.md), while board cards themselves are
its strongest (mAP50 ~0.90-0.995) — `street` uses the reliable signal.
`board_stage` is kept as-is for whatever secondary use it has, not relied on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vision.calibration import TableCalibration, attribute_hero_hand
from vision.calibration import POKERSTARS_6MAX as _DEFAULT_CALIBRATION
from vision.detector.table_detector import TableDetector
from vision.ocr.card_reader import CardReader

_BOARD_SIZES = (5, 4, 3)  # try river, then turn, then flop
_Y_TOLERANCE_FRAC = 0.03  # fraction of image height cards must share to count as "the board"
_STREET_BY_BOARD_LEN = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


@dataclass
class SeatState:
    index: int
    in_hand: bool = False
    is_dealer: bool = False
    stack_bbox: list[float] | None = None
    bet_bbox: list[float] | None = None


@dataclass
class GameState:
    board: list[str] = field(default_factory=list)
    street: str = "unknown"
    board_stage: str | None = None
    pot: float | None = None
    table_bet: float | None = None
    hero_hand: list[str] | None = None
    seats: dict[int, SeatState] = field(default_factory=dict)
    exposed_cards: list[dict] = field(default_factory=list)


def _cluster_board(cards: list[dict], img_height: float) -> tuple[list[str], list[dict]]:
    """Split detected cards into (board, leftover) by y-alignment. Picks the
    largest cluster whose size is a valid board size (3/4/5); ties favor more cards."""
    if not cards:
        return [], []

    tol = _Y_TOLERANCE_FRAC * img_height
    ys = [(c["bbox"][1] + c["bbox"][3]) / 2 for c in cards]

    best_cluster: list[dict] = []
    for anchor_y in ys:
        cluster = [c for c, y in zip(cards, ys) if abs(y - anchor_y) <= tol]
        if len(cluster) in _BOARD_SIZES and len(cluster) > len(best_cluster):
            best_cluster = cluster

    if not best_cluster:
        return [], cards

    board_sorted = sorted(best_cluster, key=lambda c: (c["bbox"][0] + c["bbox"][2]) / 2)
    board_ids = {id(c) for c in best_cluster}
    leftover = [c for c in cards if id(c) not in board_ids]
    return [c["card"] for c in board_sorted], leftover


def _read_bbox_number(img, entry: dict | None, reader: CardReader) -> float | None:
    if entry is None:
        return None
    x1, y1, x2, y2 = entry["bbox"]
    crop = img.crop((x1, y1, x2, y2))
    return reader.read_number(crop)


def build_game_state(
    image: str | Path | np.ndarray,
    detector: TableDetector | None = None,
    reader: CardReader | None = None,
    calibration: TableCalibration | None = _DEFAULT_CALIBRATION,
) -> GameState:
    """Run TableDetector + CardReader's OCR on `image` and group the results
    into a GameState. `image` is a file path or an already-decoded BGR array
    (e.g. from `ScreenCapture.grab()`) — the frame is read/decoded once,
    passed to the detector as-is, and reused for the OCR crops below, instead
    of round-tripping through disk a second and third time.

    `calibration` picks which screen region counts as hero's hole cards (see
    vision/calibration.py); pass None to skip hero attribution entirely and
    leave every exposed card unresolved.

    See module docstring for what's NOT resolved yet."""
    from PIL import Image

    detector = detector or TableDetector()
    reader = reader or CardReader()

    if isinstance(image, np.ndarray):
        arr = image
    else:
        import cv2

        arr = cv2.imread(str(image))
        if arr is None:
            raise FileNotFoundError(f"Cannot read image: {image}")

    raw = detector.detect(arr)
    img = Image.fromarray(arr[:, :, ::-1])  # BGR -> RGB, for the PIL-based OCR crops below
    img_width, img_height = arr.shape[1], arr.shape[0]

    board, leftover = _cluster_board(raw["cards"], img_height)

    hero_hand = None
    if calibration is not None:
        hero_hand = attribute_hero_hand(leftover, img_width, img_height, calibration)
        if hero_hand is not None:
            hero_ids = {id(c) for c in leftover if c["card"] in hero_hand}
            leftover = [c for c in leftover if id(c) not in hero_ids]

    seats = {
        seat: SeatState(
            index=seat,
            in_hand=info["in_hand"],
            is_dealer=info["dealer"],
            stack_bbox=info["stack"]["bbox"] if info["stack"] else None,
            bet_bbox=info["bet"]["bbox"] if info["bet"] else None,
        )
        for seat, info in raw["players"].items()
    }

    return GameState(
        board=board,
        street=_STREET_BY_BOARD_LEN.get(len(board), "unknown"),
        board_stage=raw["board_stage"],
        pot=_read_bbox_number(img, raw["pot"], reader),
        table_bet=_read_bbox_number(img, raw["table_bet"], reader),
        hero_hand=hero_hand,
        seats=seats,
        exposed_cards=leftover,
    )
