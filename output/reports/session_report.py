"""Generate session EV and ROI reports."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Hand:
    hero_cards: list[str]
    board: list[str]
    pot: float
    result: float
    ev: float


@dataclass
class SessionReport:
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M"))
    hands: list[Hand] = field(default_factory=list)

    def add_hand(self, hand: Hand) -> None:
        self.hands.append(hand)

    def summary(self) -> dict:
        if not self.hands:
            return {}
        total_result = sum(h.result for h in self.hands)
        total_ev = sum(h.ev for h in self.hands)
        return {
            "session_id": self.session_id,
            "hands_played": len(self.hands),
            "total_result": round(total_result, 2),
            "total_ev": round(total_ev, 2),
            "luck_adjusted": round(total_result - total_ev, 2),
        }
