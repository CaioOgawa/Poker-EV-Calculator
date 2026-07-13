"""Card and number reader: trained YOLOv8-cls classifier for cards, Tesseract for numbers."""

from __future__ import annotations

from pathlib import Path

try:
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

CARD_RANKS = "23456789TJQKA"
CARD_SUITS = "shdc"

_DEFAULT_WEIGHTS = (
    Path(__file__).parent.parent / "models" / "card_classifier" / "weights" / "best.pt"
)

# dataset class names use "10c" for ten instead of treys' "Tc"
_RANK_ALIASES = {"10": "T"}


def _to_treys(class_name: str) -> str | None:
    """Map a dataset class name (e.g. '10c', 'as') to treys notation ('Tc', 'As').

    Returns None if `class_name` isn't one of the 52 card classes (e.g. 'pot', 'p0b') —
    the classifier was trained on all 81 dataset classes, not just cards.
    """
    for suit in "cdhs":
        if class_name.endswith(suit):
            rank = _RANK_ALIASES.get(class_name[:-1], class_name[:-1]).upper()
            if rank in CARD_RANKS:
                return f"{rank}{suit}"
    return None


class CardReader:
    def __init__(self, weights: str | Path = _DEFAULT_WEIGHTS, confidence: float = 0.5):
        self.weights = Path(weights)
        self.confidence = confidence
        self._model = None

    def _load_model(self) -> "YOLO":
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics required — pip install -e '.[vision]'")
        if self._model is None:
            if not self.weights.exists():
                raise FileNotFoundError(
                    f"No trained classifier at {self.weights} — "
                    "run `python -m vision.train_card_classifier` first"
                )
            self._model = YOLO(str(self.weights))
        return self._model

    def read_card(self, image_path: str | Path) -> str | None:
        """Return card string (e.g. 'As') from cropped card image, or None."""
        model = self._load_model()
        result = model(str(image_path), verbose=False)[0]
        top1_conf = result.probs.top1conf.item()
        if top1_conf < self.confidence:
            return None
        class_name = result.names[result.probs.top1]
        return _to_treys(class_name)

    def read_number(self, image: str | Path | Image.Image) -> float | None:
        """Read a numeric value (pot size, stack) from a cropped region or an
        already-loaded PIL image (e.g. a crop from a full table screenshot)."""
        if not OCR_AVAILABLE:
            raise ImportError("pytesseract and Pillow required for vision module")
        img = image if isinstance(image, Image.Image) else Image.open(str(image))
        text = pytesseract.image_to_string(img, config="--psm 7 digits")
        try:
            return float(text.strip().replace(",", ""))
        except ValueError:
            return None
