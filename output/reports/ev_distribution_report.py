"""Distribution view of a session's results vs. their EV — histogram and
violin plots, plus the summary stats behind them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class DistributionStats:
    n: int
    mean: float
    std_dev: float
    p5: float
    p50: float
    p95: float


class EVDistributionReport:
    def stats(self, values: list[float]) -> DistributionStats:
        if not values:
            raise ValueError("values must not be empty")
        arr = np.array(values, dtype=float)
        return DistributionStats(
            n=len(values),
            mean=float(np.mean(arr)),
            std_dev=float(np.std(arr)),
            p5=float(np.percentile(arr, 5)),
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
        )

    def plot_histogram(
        self,
        results: list[float],
        ev: list[float] | None = None,
        bins: int = 30,
        title: str = "EV Distribution",
        save_path: str | None = None,
    ) -> None:
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for output module")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(results, bins=bins, alpha=0.6, label="Result", color="steelblue")
        if ev:
            ax.hist(ev, bins=bins, alpha=0.6, label="EV", color="orange")
        ax.set_xlabel("BB")
        ax.set_ylabel("Hands")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        self._finish(fig, save_path)

    def plot_violin(
        self,
        results: list[float],
        ev: list[float] | None = None,
        title: str = "EV Distribution",
        save_path: str | None = None,
    ) -> None:
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for output module")
        datasets = [results] if ev is None else [results, ev]
        labels = ["Result"] if ev is None else ["Result", "EV"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.violinplot(datasets, showmedians=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel("BB")
        ax.set_title(title)
        plt.tight_layout()
        self._finish(fig, save_path)

    def _finish(self, fig, save_path: str | None) -> None:
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
