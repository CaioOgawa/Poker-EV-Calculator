"""Group TableDetector's raw detections into a structured GameState.

TableDetector.detect() returns ungrouped card detections with no hero/board
distinction (see vision/detector/table_detector.py). The only grouping rule
validated against real data so far: community cards cluster tightly around a
shared y-coordinate on the felt (checked against ground-truth labels — a
5-card river board sat within ~1% of image height of each other, while a
stray hole card 13% away in x and y was correctly excluded). Board size is
always 3, 4, or 5; anything else means "no confident board found".

Hole-card ownership (hero vs. an opponent's hand exposed at showdown) is
NOT resolved here — that needs a per-room seat calibration (ROADMAP Fase 5,
"Calibração por sala") that doesn't exist yet. Cards outside the board
cluster are returned as `exposed_cards` with their bbox so a future
calibration layer can attribute them to a seat instead of guessing wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vision.detector.table_detector import TableDetector
from vision.ocr.card_reader import CardReader

_BOARD_SIZES = (5, 4, 3)  # try river, then turn, then flop
_Y_TOLERANCE_FRAC = 0.03  # fraction of image height cards must share to count as "the board"


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
    board_stage: str | None = None
    pot: float | None = None
    table_bet: float | None = None
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
    image_path: str | Path,
    detector: TableDetector | None = None,
    reader: CardReader | None = None,
) -> GameState:
    """Run TableDetector + CardReader's OCR on `image_path` and group the
    results into a GameState. See module docstring for what's NOT resolved yet."""
    from PIL import Image

    detector = detector or TableDetector()
    reader = reader or CardReader()

    raw = detector.detect(image_path)
    img = Image.open(str(image_path))

    board, leftover = _cluster_board(raw["cards"], img.height)

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
        board_stage=raw["board_stage"],
        pot=_read_bbox_number(img, raw["pot"], reader),
        table_bet=_read_bbox_number(img, raw["table_bet"], reader),
        seats=seats,
        exposed_cards=leftover,
    )
