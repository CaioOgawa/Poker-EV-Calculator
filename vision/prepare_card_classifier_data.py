"""Reorganize the pre-cropped card detection dataset into an ImageFolder layout.

`vision/data/raw/cards/` ships each element (card/marker) as its own tiny
image with a single full-frame YOLO label (class, 0.5, 0.5, 1, 1) — that's a
classification dataset wearing a detection dataset's clothes. This script
copies each image into `vision/data/processed/cards/<split>/<class_name>/`
so it can be trained with `yolov8n-cls` or any ImageFolder-based classifier.

Usage:
    python -m vision.prepare_card_classifier_data
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

RAW_DIR = Path(__file__).parent / "data" / "raw" / "cards"
OUT_DIR = Path(__file__).parent / "data" / "processed" / "cards"
# raw split dir -> ImageFolder split name (ultralytics classification expects
# exactly train/val/test on disk, not "valid")
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def reorganize(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> dict[str, int]:
    """Copy each crop into out_dir/<split>/<class_name>/. Returns count per split."""
    names = yaml.safe_load((raw_dir / "data.yaml").read_text())["names"]

    counts: dict[str, int] = {}
    for raw_split, out_split in SPLITS.items():
        images_dir = raw_dir / raw_split / "images"
        labels_dir = raw_dir / raw_split / "labels"
        if not images_dir.exists():
            continue

        n = 0
        for label_path in labels_dir.glob("*.txt"):
            class_idx = int(label_path.read_text().split()[0])
            class_name = names[class_idx]
            image_path = images_dir / f"{label_path.stem}.jpg"
            if not image_path.exists():
                continue

            dest_dir = out_dir / out_split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, dest_dir / image_path.name)
            n += 1
        counts[out_split] = n

    return counts


if __name__ == "__main__":
    counts = reorganize()
    for split, n in counts.items():
        print(f"{split}: {n} images")
