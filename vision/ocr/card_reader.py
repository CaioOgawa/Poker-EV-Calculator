"""OCR-based card and number reader using Tesseract + template matching."""
from __future__ import annotations
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

CARD_RANKS = "23456789TJQKA"
CARD_SUITS = "shdc"


class CardReader:
    def read_card(self, image_path: str | Path) -> str | None:
        """Return card string (e.g. 'As') from cropped card image, or None."""
        if not OCR_AVAILABLE:
            raise ImportError("pytesseract and Pillow required for vision module")
        # Stub — template matching + OCR pipeline via poker-vision skill
        return None

    def read_number(self, image_path: str | Path) -> float | None:
        """Read a numeric value (pot size, stack) from a cropped region."""
        if not OCR_AVAILABLE:
            raise ImportError("pytesseract and Pillow required for vision module")
        img = Image.open(str(image_path))
        text = pytesseract.image_to_string(img, config="--psm 7 digits")
        try:
            return float(text.strip().replace(",", ""))
        except ValueError:
            return None
