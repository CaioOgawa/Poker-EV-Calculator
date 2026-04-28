"""Parse poker range notation (e.g. 'AA,KK,AKs,AQo+') into hand combos."""
from __future__ import annotations

RANKS = "23456789TJQKA"
SUITED_COMBOS = 4
OFFSUIT_COMBOS = 12
PAIR_COMBOS = 6


class RangeParser:
    def parse(self, range_str: str) -> list[tuple[str, str]]:
        """Return list of (card1, card2) string tuples representing all combos in range."""
        combos: list[tuple[str, str]] = []
        for token in range_str.replace(" ", "").split(","):
            combos.extend(self._parse_token(token))
        return combos

    def combo_count(self, range_str: str) -> int:
        return len(self.parse(range_str))

    def _parse_token(self, token: str) -> list[tuple[str, str]]:
        # Stub — full implementation iterates over rank/suit combos
        return []
