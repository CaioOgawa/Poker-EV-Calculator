"""13x13 canonical starting-hand grid — the standard poker range-chart
layout: pairs on the diagonal, suited combos in the upper-right triangle,
offsuit combos in the lower-left triangle, ranks descending A..2 on both
axes.

Grid construction is kept separate from rendering (`matrix_from_*` return
plain nested lists) so the layout logic is unit-testable without needing
pixel-level assertions on a rendered figure.
"""

from __future__ import annotations

import numpy as np

from core.ranges.range_parser import RangeParser

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

_PARSER = RangeParser()
GRID_RANKS = "AKQJT98765432"  # descending — standard chart display order


def _cell_label(row: int, col: int) -> str:
    r_row, r_col = GRID_RANKS[row], GRID_RANKS[col]
    if row == col:
        return r_row + r_col
    if row < col:
        return r_row + r_col + "s"
    return r_col + r_row + "o"


class RangeHeatmap:
    def matrix_from_range(self, range_str: str) -> list[list[float]]:
        """13x13 grid: 1.0 where the cell's canonical hand is in `range_str`."""
        range_combos = {frozenset(c) for c in _PARSER.parse(range_str)}
        grid = [[0.0] * 13 for _ in range(13)]
        for row in range(13):
            for col in range(13):
                hand_combos = {frozenset(c) for c in _PARSER.parse(_cell_label(row, col))}
                grid[row][col] = 1.0 if hand_combos & range_combos else 0.0
        return grid

    def matrix_from_chart(self, chart, action: str) -> list[list[float]]:
        """13x13 grid of `chart`'s frequency for `action` at each hand — for a
        `engine.gto.chart.PreflopChart` (or anything with the same
        `.frequencies(hand) -> dict[str, float]` interface). Hands missing
        from the chart (and no `default_action` set) are treated as 0.0.
        """
        grid = [[0.0] * 13 for _ in range(13)]
        for row in range(13):
            for col in range(13):
                label = _cell_label(row, col)
                try:
                    freqs = chart.frequencies(label)
                except KeyError:
                    continue
                grid[row][col] = freqs.get(action, 0.0)
        return grid

    def plot(
        self,
        matrix: list[list[float]],
        title: str = "Range",
        cmap: str = "YlOrRd",
        save_path: str | None = None,
    ) -> None:
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for output module")
        arr = np.array(matrix)
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(13))
        ax.set_xticklabels(list(GRID_RANKS))
        ax.set_yticks(range(13))
        ax.set_yticklabels(list(GRID_RANKS))
        for row in range(13):
            for col in range(13):
                ax.text(col, row, _cell_label(row, col), ha="center", va="center", fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
