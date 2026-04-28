"""EV and bankroll evolution charts using matplotlib/plotly."""
from __future__ import annotations

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class EVChart:
    def plot_bankroll(self, results: list[float], ev_line: list[float] | None = None, title: str = "Bankroll Evolution") -> None:
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for output module")
        cumulative = [sum(results[: i + 1]) for i in range(len(results))]
        plt.figure(figsize=(12, 5))
        plt.plot(cumulative, label="Actual", color="steelblue")
        if ev_line:
            plt.plot(ev_line, label="EV Line", color="orange", linestyle="--")
        plt.title(title)
        plt.xlabel("Hands")
        plt.ylabel("BB")
        plt.legend()
        plt.tight_layout()
        plt.show()
