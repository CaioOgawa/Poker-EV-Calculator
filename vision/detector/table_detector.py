"""Detect and extract poker table regions from screenshots using OpenCV."""
from __future__ import annotations
from pathlib import Path

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class TableDetector:
    def __init__(self, confidence: float = 0.7):
        self.confidence = confidence

    def detect(self, image_path: str | Path) -> dict:
        """
        Detect table, pot, player stacks, and community cards from a screenshot.
        Returns dict with regions: {table, pot, board, players}.
        """
        if not CV2_AVAILABLE:
            raise ImportError("opencv-python required for vision module")
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        # Stub — full YOLO detection via poker-vision skill
        return {"table": None, "pot": None, "board": [], "players": []}
