"""Detect and extract poker table regions from screenshots using OpenCV/YOLO."""

from __future__ import annotations

from pathlib import Path

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

_DEFAULT_WEIGHTS = (
    Path(__file__).parent.parent / "models" / "table_detector" / "weights" / "best.pt"
)

# dataset class names use "10c" for ten instead of treys' "Tc"
_RANK_ALIASES = {"10": "T"}
_CARD_SUITS = "cdhs"
_CARD_RANKS = "23456789TJQKA"


def _to_treys(class_name: str) -> str | None:
    """Map a dataset class name (e.g. '10c', 'as') to treys notation ('Tc', 'As').

    Returns None if `class_name` isn't one of the 52 card classes (e.g. 'pot', 'p0b').
    """
    for suit in _CARD_SUITS:
        if class_name.endswith(suit):
            rank = _RANK_ALIASES.get(class_name[:-1], class_name[:-1]).upper()
            if rank in _CARD_RANKS:
                return f"{rank}{suit}"
    return None


class TableDetector:
    def __init__(self, weights: str | Path = _DEFAULT_WEIGHTS, confidence: float = 0.4):
        self.weights = Path(weights)
        self.confidence = confidence
        self._model = None

    def _load_model(self) -> "YOLO":
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics required — pip install -e '.[vision]'")
        if self._model is None:
            if not self.weights.exists():
                raise FileNotFoundError(
                    f"No trained detector at {self.weights} — "
                    "run `python -m vision.train_detector` first"
                )
            self._model = YOLO(str(self.weights))
        return self._model

    def detect(self, image_path: str | Path) -> dict:
        """
        Detect table elements from a screenshot: cards, pot, board stage, per-seat markers.

        Returns a dict with raw detections (bbox + confidence), not a semantic hero/board
        split — that classification belongs to a future vision/state/ layer (see ROADMAP).
        {
            "cards": [{"card": "As", "bbox": [x1,y1,x2,y2], "conf": 0.9}, ...],
            "pot": {"bbox": [...], "conf": ...} | None,
            "table_bet": {...} | None,
            "board_stage": "flop" | "turn" | "river" | None,
            "players": {0: {"bet": {...}|None, "dealer": bool, "stack": {...}|None,
                             "in_hand": bool}, ..., 5: {...}},
        }
        """
        if not CV2_AVAILABLE:
            raise ImportError("opencv-python required for vision module")
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        model = self._load_model()
        result = model(str(image_path), conf=self.confidence, verbose=False)[0]

        cards: list[dict] = []
        pot: dict | None = None
        table_bet: dict | None = None
        board_stage: str | None = None
        players: dict[int, dict] = {
            seat: {"bet": None, "dealer": False, "stack": None, "in_hand": False}
            for seat in range(6)
        }

        for box in result.boxes:
            class_name = result.names[int(box.cls)]
            entry = {"bbox": box.xyxy[0].tolist(), "conf": float(box.conf)}

            card = _to_treys(class_name)
            if card:
                cards.append({"card": card, **entry})
            elif class_name == "pot":
                pot = entry
            elif class_name == "table_bet":
                table_bet = entry
            elif class_name in ("flop", "turn", "river"):
                board_stage = class_name
            elif len(class_name) == 3 and class_name[0] == "p" and class_name[1].isdigit():
                seat, kind = int(class_name[1]), class_name[2]
                if kind == "b":
                    players[seat]["bet"] = entry
                elif kind == "d":
                    players[seat]["dealer"] = True
                elif kind == "s":
                    players[seat]["stack"] = entry
            elif class_name.endswith("in") and class_name[:-2].isdigit():
                players[int(class_name[:-2])]["in_hand"] = True

        return {
            "cards": cards,
            "pot": pot,
            "table_bet": table_bet,
            "board_stage": board_stage,
            "players": players,
        }
