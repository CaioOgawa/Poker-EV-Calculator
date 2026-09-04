from vision.calibration import POKERSTARS_6MAX, TableCalibration, attribute_hero_hand

IMG_W, IMG_H = 949, 519


def _card(card: str, cx_frac: float, cy_frac: float, w_frac: float, h_frac: float) -> dict:
    """Build a bbox dict from YOLO-style normalized (center, width, height),
    matching how vision/data/raw/pokerstars labels are stored."""
    cx, cy = cx_frac * IMG_W, cy_frac * IMG_H
    w, h = w_frac * IMG_W, h_frac * IMG_H
    return {"card": card, "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], "conf": 0.9}


def test_attribute_hero_hand_finds_real_ground_truth_pair():
    # Exact normalized bboxes from vision/data/raw/pokerstars/train/labels/
    # table_6max_1684665487_jpg...txt for classes '5c' and '4s' — a real
    # image where both of the bottom-seat's hole cards are labeled.
    exposed = [
        _card("5c", 0.4495, 0.8196, 0.0219, 0.0860),
        _card("4s", 0.5127, 0.8196, 0.0219, 0.0860),
    ]
    assert attribute_hero_hand(exposed, IMG_W, IMG_H) == ["5c", "4s"]


def test_attribute_hero_hand_none_when_only_one_card_present():
    # What actually happened running the trained model on that same image:
    # it missed '5c' (imperfect recall, already documented), leaving only
    # one card in the hero region — must not guess a hand from one card.
    exposed = [_card("4s", 0.5127, 0.8196, 0.0219, 0.0860)]
    assert attribute_hero_hand(exposed, IMG_W, IMG_H) is None


def test_attribute_hero_hand_ignores_cards_outside_region():
    # Board cards from the same real image (flop position, y ~= 0.46).
    exposed = [
        _card("Ts", 0.4823, 0.4595, 0.023, 0.09),
        _card("Kc", 0.4142, 0.4550, 0.023, 0.09),
    ]
    assert attribute_hero_hand(exposed, IMG_W, IMG_H) is None


def test_attribute_hero_hand_picks_the_two_in_region_among_more():
    in_region = [
        _card("5c", 0.4495, 0.8196, 0.0219, 0.0860),
        _card("4s", 0.5127, 0.8196, 0.0219, 0.0860),
    ]
    board_card = _card("Ts", 0.4823, 0.4595, 0.023, 0.09)
    result = attribute_hero_hand([board_card, *in_region], IMG_W, IMG_H)
    assert result == ["5c", "4s"]


def test_attribute_hero_hand_uses_custom_calibration():
    custom = TableCalibration(name="test", hero_region=(0.0, 0.0, 0.2, 0.2), hero_seat=2)
    top_left_cards = [
        _card("As", 0.05, 0.05, 0.02, 0.08),
        _card("Ks", 0.1, 0.05, 0.02, 0.08),
    ]
    assert attribute_hero_hand(top_left_cards, IMG_W, IMG_H, custom) == ["As", "Ks"]
    assert attribute_hero_hand(top_left_cards, IMG_W, IMG_H, POKERSTARS_6MAX) is None
