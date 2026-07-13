"""Train the poker table YOLOv8 detector on an annotated dataset.

Usage:
    python -m vision.train_detector
    python -m vision.train_detector --data vision/data/raw/cards/data.yaml --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

DEFAULT_DATA = Path(__file__).parent / "data" / "raw" / "pokerstars" / "data.yaml"
DEFAULT_PROJECT = Path(__file__).parent / "models"


def train(
    data_yaml: str | Path = DEFAULT_DATA,
    epochs: int = 100,
    imgsz: int = 640,
    base_model: str = "yolov8n.pt",
    project: str | Path = DEFAULT_PROJECT,
    name: str = "table_detector",
) -> Path:
    """Fine-tune a YOLOv8n model on `data_yaml`. Returns the path to best.pt."""
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("ultralytics required — pip install -e '.[vision]'")
    model = YOLO(base_model)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, project=str(project), name=name)
    return Path(project) / name / "weights" / "best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA, help="path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", default="yolov8n.pt", help="base checkpoint to fine-tune")
    parser.add_argument("--name", default="table_detector", help="run name under vision/models/")
    args = parser.parse_args()

    best = train(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        base_model=args.model,
        name=args.name,
    )
    print(f"Best weights: {best}")
