"""Heatmap of ICM bubble factors between every pair of players."""

from __future__ import annotations

import numpy as np

from engine.icm.pressure import ICMPressure

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ICMPressureChart:
    def __init__(self, pressure: ICMPressure | None = None):
        self._pressure = pressure or ICMPressure()

    def matrix(self, stacks: list[int], payouts: list[float]) -> list[list[float]]:
        """BF[i][j] matrix — see `ICMPressure.all_bubble_factors` for semantics."""
        return self._pressure.all_bubble_factors(stacks, payouts)

    def plot(
        self,
        stacks: list[int],
        payouts: list[float],
        labels: list[str] | None = None,
        title: str = "ICM Bubble Factor",
        cmap: str = "RdYlGn_r",
        save_path: str | None = None,
    ) -> None:
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for output module")
        matrix = self.matrix(stacks, payouts)
        n = len(stacks)
        labels = labels or [f"P{i}" for i in range(n)]

        # Bubble factor is unbounded above (inf when a hero can't gain ICM
        # equity at all) — clip only for the colormap scale, not the
        # annotated values, so a chart with an infinite BF still renders
        # instead of blowing up vmax.
        finite = [v for row in matrix for v in row if np.isfinite(v)]
        vmax = max(finite) if finite else 1.0
        arr = np.array([[min(v, vmax) for v in row] for row in matrix])

        fig, ax = plt.subplots(figsize=(1.2 * n + 2, 1.2 * n + 2))
        im = ax.imshow(arr, cmap=cmap, vmin=1.0, vmax=vmax)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels)
        ax.set_xlabel("vs villain")
        ax.set_ylabel("hero")
        for i in range(n):
            for j in range(n):
                value = matrix[i][j]
                text = "inf" if not np.isfinite(value) else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
