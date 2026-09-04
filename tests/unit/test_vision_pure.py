"""Pure-function tests for vision/ — no trained model or real image needed.

`_to_treys` and `_cluster_board` are the two testable-without-a-model pieces
of the vision pipeline (docs/AUDITORIA-2026-08-26.md item F2: vision had
zero tests). Everything else here needs the YOLO weights (gitignored, not
backed up yet — F8) and is covered instead in test_cli.py's `@requires_*`
skip-guarded tests.
"""

from vision.detector.table_detector import _to_treys as _detector_to_treys
from vision.ocr.card_reader import _to_treys as _reader_to_treys
from vision.state.game_state import _cluster_board


def _card(card: str, x1: float, y1: float, x2: float, y2: float) -> dict:
    return {"card": card, "bbox": [x1, y1, x2, y2], "conf": 0.9}


# ── _to_treys (duplicated in table_detector.py and card_reader.py — both
# tested since they're separate functions that could drift apart) ──────────


def test_detector_to_treys_maps_ten_alias():
    assert _detector_to_treys("10c") == "Tc"


def test_detector_to_treys_uppercases_rank():
    assert _detector_to_treys("as") == "As"
    assert _detector_to_treys("2h") == "2h"


def test_detector_to_treys_rejects_non_card_classes():
    assert _detector_to_treys("pot") is None
    assert _detector_to_treys("p0b") is None
    assert _detector_to_treys("flop") is None


def test_reader_to_treys_maps_ten_alias():
    assert _reader_to_treys("10s") == "Ts"


def test_reader_to_treys_rejects_non_card_classes():
    assert _reader_to_treys("table_bet") is None
    assert _reader_to_treys("3in") is None


# ── _cluster_board ───────────────────────────────────────────────────────


def test_cluster_board_empty_input():
    assert _cluster_board([], img_height=500) == ([], [])


def test_cluster_board_five_cards_sorted_by_x():
    # Given out of x-order on purpose — output must be left-to-right.
    cards = [
        _card("Kd", 300, 240, 320, 280),
        _card("As", 100, 242, 120, 282),
        _card("2c", 500, 238, 520, 278),
        _card("7h", 200, 241, 220, 281),
        _card("Ts", 400, 239, 420, 279),
    ]
    board, leftover = _cluster_board(cards, img_height=500)
    assert board == ["As", "7h", "Kd", "Ts", "2c"]
    assert leftover == []


def test_cluster_board_two_aligned_cards_is_not_a_board():
    # The real case this pins: TableDetector missed one flop card on a real
    # screenshot, leaving 2 y-aligned cards — 2 isn't a valid board size
    # (3/4/5), so both must fall through to exposed_cards rather than being
    # reported as a (wrong) 2-card board.
    cards = [_card("8h", 446, 214, 467, 260), _card("2h", 381, 214, 404, 259)]
    board, leftover = _cluster_board(cards, img_height=519)
    assert board == []
    assert leftover == cards


def test_cluster_board_excludes_stray_card_outside_y_tolerance():
    board_cards = [
        _card("As", 100, 250, 120, 290),
        _card("Kd", 200, 251, 220, 291),
        _card("Qc", 300, 249, 320, 289),
    ]
    stray = _card("7h", 150, 50, 170, 90)  # far above the board row
    board, leftover = _cluster_board([*board_cards, stray], img_height=500)
    assert board == ["As", "Kd", "Qc"]
    assert leftover == [stray]


def test_cluster_board_picks_the_larger_valid_cluster():
    # 3 tightly-aligned board cards plus 2 more cards aligned with each other
    # at a different y (e.g. two exposed hole cards) — the board-sized
    # cluster must win even though it isn't the first one found.
    board_cards = [
        _card("As", 100, 250, 120, 290),
        _card("Kd", 200, 251, 220, 291),
        _card("Qc", 300, 249, 320, 289),
    ]
    other_pair = [
        _card("2h", 150, 400, 170, 440),
        _card("2d", 250, 401, 270, 441),
    ]
    board, leftover = _cluster_board([*other_pair, *board_cards], img_height=500)
    assert board == ["As", "Kd", "Qc"]
    assert leftover == other_pair
