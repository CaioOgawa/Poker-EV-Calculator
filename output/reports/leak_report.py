"""Aggregate a session's hands by category and surface the worst-EV spots.

Requires `Hand.category` to be set (e.g. "3bet pot", "river bluff", "vs
short stack") — hands without one are grouped under "uncategorized" rather
than dropped, so a partially-tagged session still produces a usable report.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from output.reports.session_report import Hand


@dataclass
class CategoryLeak:
    category: str
    hands: int
    total_result: float
    total_ev: float
    luck_adjusted: float
    avg_ev: float


class LeakReport:
    def by_category(self, hands: list[Hand]) -> list[CategoryLeak]:
        """One `CategoryLeak` per category, sorted worst total EV first."""
        groups: dict[str, list[Hand]] = defaultdict(list)
        for h in hands:
            groups[h.category or "uncategorized"].append(h)

        leaks = []
        for category, group in groups.items():
            total_result = sum(h.result for h in group)
            total_ev = sum(h.ev for h in group)
            leaks.append(
                CategoryLeak(
                    category=category,
                    hands=len(group),
                    total_result=round(total_result, 2),
                    total_ev=round(total_ev, 2),
                    luck_adjusted=round(total_result - total_ev, 2),
                    avg_ev=round(total_ev / len(group), 4),
                )
            )
        return sorted(leaks, key=lambda c: c.total_ev)

    def top_leaks(self, hands: list[Hand], n: int = 5) -> list[CategoryLeak]:
        """The `n` worst categories by total EV (most negative first)."""
        return self.by_category(hands)[:n]
