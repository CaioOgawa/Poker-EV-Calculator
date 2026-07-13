"""Train a card classifier on the reorganized ImageFolder card crops.

Run `python -m vision.prepare_card_classifier_data` first to build
`vision/data/processed/cards/{train,val,test}/<class_name>/*.jpg`.

Usage:
    python -m vision.train_card_classifier
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

DEFAULT_DATA = Path(__file__).parent / "data" / "processed" / "cards"
DEFAULT_PROJECT = Path(__file__).parent / "models"


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def train(
    data_dir: str | Path = DEFAULT_DATA,
    epochs: int = 50,
    imgsz: int = 64,
    base_model: str = "yolov8n-cls.pt",
    project: str | Path = DEFAULT_PROJECT,
    name: str = "card_classifier",
    device: str | None = None,
) -> Path:
    """Fine-tune a YOLOv8n classifier on `data_dir`. Returns the path to best.pt."""
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("ultralytics required — pip install -e '.[vision]'")
    model = YOLO(base_model)
    model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        project=str(project),
        name=name,
        device=device or _default_device(),
    )
    return Path(project) / name / "weights" / "best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA, help="path to ImageFolder root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=64, help="crops are tiny, no need for 640")
    parser.add_argument("--model", default="yolov8n-cls.pt", help="base checkpoint to fine-tune")
    parser.add_argument("--name", default="card_classifier", help="run name under vision/models/")
    parser.add_argument("--device", default=None, help="mps/cpu/0 — auto-detected if omitted")
    args = parser.parse_args()

    best = train(
        data_dir=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        base_model=args.model,
        name=args.name,
        device=args.device,
    )
    print(f"Best weights: {best}")
