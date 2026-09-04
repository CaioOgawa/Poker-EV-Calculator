"""Continuous screen capture for feeding live poker table frames to the vision pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

try:
    import mss
    import numpy as np

    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False


class ScreenCapture:
    def __init__(self, monitor: int = 1, region: dict[str, int] | None = None):
        """`region` (screen pixels: {"top", "left", "width", "height"}) overrides
        `monitor` when given — use it to capture just the poker table window."""
        if not MSS_AVAILABLE:
            raise ImportError("mss required — pip install -e '.[vision]'")
        self.monitor = monitor
        self.region = region

    def grab(self) -> "np.ndarray":
        """Single frame as a BGR numpy array (cv2-compatible, alpha dropped)."""
        with mss.MSS() as sct:
            target = self.region or sct.monitors[self.monitor]
            shot = sct.grab(target)
            return np.array(shot)[:, :, :3]

    def save(self, path: str | Path) -> Path:
        """Grab one frame and write it to `path`."""
        import cv2

        path = Path(path)
        cv2.imwrite(str(path), self.grab())
        return path

    def stream(
        self,
        interval: float = 2.0,
        out_dir: str | Path | None = None,
        max_frames: int | None = None,
    ) -> Iterator[tuple[float, "np.ndarray"]]:
        """Yield (timestamp, frame) every `interval` seconds. If `out_dir` is
        given, also writes each frame there as `frame_<ms>.jpg` for later
        TableDetector/CardReader runs or dataset collection."""
        out_dir = Path(out_dir) if out_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        while max_frames is None or n < max_frames:
            ts = time.time()
            frame = self.grab()
            if out_dir:
                import cv2

                cv2.imwrite(str(out_dir / f"frame_{int(ts * 1000)}.jpg"), frame)
            yield ts, frame
            n += 1
            time.sleep(interval)
