"""Computer vision for poker table/card recognition. Skill: poker-vision."""

from vision.detector.table_detector import TableDetector
from vision.ocr.card_reader import CardReader

__all__ = ["TableDetector", "CardReader"]
