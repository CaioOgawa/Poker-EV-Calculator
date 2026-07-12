"""Stub interface for future GTO solver integration (OpenSpiel CFR, PioSolver API).

No solver is wired in yet. This defines the contract a real integration
should satisfy — anything that can turn solver output into a `PreflopChart`
— so callers written against `SolverBridge` don't need to change when one
lands. Until then, `PreflopChart.from_csv`/`from_json` cover manually
exported charts.
"""

from __future__ import annotations
from typing import Protocol

from engine.gto.chart import PreflopChart


class SolverBridge(Protocol):
    """Anything that can produce a PreflopChart from solver output."""

    def load_chart(self, source: str) -> PreflopChart:
        """Load solver output (a file path, URL, or solver-specific handle)
        into a PreflopChart."""
        ...


class NotImplementedBridge:
    """Placeholder bridge — raises until a real solver integration exists."""

    def load_chart(self, source: str) -> PreflopChart:
        raise NotImplementedError(
            "No GTO solver integration is wired in yet (OpenSpiel CFR / "
            "PioSolver API). Use PreflopChart.from_csv/from_json to load an "
            "exported chart directly."
        )
