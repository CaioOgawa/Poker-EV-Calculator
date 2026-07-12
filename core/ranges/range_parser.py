"""Parse poker range notation (e.g. 'AA,KK,AKs,AQo+') into hand combos."""

from __future__ import annotations
from itertools import product

RANKS = "23456789TJQKA"
SUITS = "shdc"
_RANK_IDX = {r: i for i, r in enumerate(RANKS)}

SUITED_COMBOS = 4
OFFSUIT_COMBOS = 12
PAIR_COMBOS = 6


def _suited_combos(r1: str, r2: str) -> list[tuple[str, str]]:
    return [(r1 + s, r2 + s) for s in SUITS]


def _offsuit_combos(r1: str, r2: str) -> list[tuple[str, str]]:
    return [(r1 + s1, r2 + s2) for s1, s2 in product(SUITS, SUITS) if s1 != s2]


def _pair_combos(r: str) -> list[tuple[str, str]]:
    suits = list(SUITS)
    return [(r + suits[i], r + suits[j]) for i in range(4) for j in range(i + 1, 4)]


class RangeParser:
    def parse(self, range_str: str) -> list[tuple[str, str]]:
        """Return list of (card1, card2) string tuples for all combos in range.

        Supports: AA, AKs, AKo, AK (both), 77+, AKs+, JTs-87s, 22-66.
        """
        combos: list[tuple[str, str]] = []
        for token in range_str.replace(" ", "").split(","):
            if token:
                combos.extend(self._parse_token(token))
        return combos

    def combo_count(self, range_str: str) -> int:
        return len(self.parse(range_str))

    # ------------------------------------------------------------------
    def _validate_ranks(self, ranks: str, original: str) -> None:
        for ch in ranks:
            if ch not in _RANK_IDX:
                raise ValueError(f"Invalid rank {ch!r} in range token: {original!r}")

    def _parse_token(self, token: str) -> list[tuple[str, str]]:
        original = token
        plus = token.endswith("+")
        if plus:
            token = token[:-1]
        if not token:
            raise ValueError(f"Invalid range token: {original!r}")

        dash_pos = self._find_dash(token)
        if dash_pos != -1:
            return self._parse_dash_range(token, dash_pos, original)

        suited = token.endswith("s")
        offsuit = token.endswith("o")
        if suited or offsuit:
            token = token[:-1]

        if len(token) != 2:
            raise ValueError(f"Invalid range token: {original!r}")
        self._validate_ranks(token, original)

        if token[0] == token[1]:
            if suited or offsuit:
                raise ValueError(f"Pocket pairs cannot use a suited/offsuit suffix: {original!r}")
            # Pocket pair: AA or AA+
            return self._pair_range(token[0], plus)

        r1, r2 = token[0], token[1]
        if _RANK_IDX[r1] < _RANK_IDX[r2]:
            r1, r2 = r2, r1  # ensure r1 >= r2
        if suited:
            return self._suited_range(r1, r2, plus)
        if offsuit:
            return self._offsuit_range(r1, r2, plus)
        # no suffix — both suited and offsuit
        return self._suited_range(r1, r2, plus) + self._offsuit_range(r1, r2, plus)

    # ------------------------------------------------------------------
    def _pair_range(self, rank: str, plus: bool) -> list[tuple[str, str]]:
        if not plus:
            return _pair_combos(rank)
        low = _RANK_IDX[rank]
        return [c for r in RANKS[low:] for c in _pair_combos(r)]

    def _suited_range(self, high: str, low: str, plus: bool) -> list[tuple[str, str]]:
        """AKs or AKs+ (kicker moves up toward high)."""
        if not plus:
            return _suited_combos(high, low)
        lo_idx = _RANK_IDX[low]
        hi_idx = _RANK_IDX[high]
        # kicker goes from low up to high-1
        return [c for ki in range(lo_idx, hi_idx) for c in _suited_combos(high, RANKS[ki])]

    def _offsuit_range(self, high: str, low: str, plus: bool) -> list[tuple[str, str]]:
        if not plus:
            return _offsuit_combos(high, low)
        lo_idx = _RANK_IDX[low]
        hi_idx = _RANK_IDX[high]
        return [c for ki in range(lo_idx, hi_idx) for c in _offsuit_combos(high, RANKS[ki])]

    # ------------------------------------------------------------------
    def _find_dash(self, token: str) -> int:
        """Find dash separating two hands (e.g. JTs-87s), ignoring suit suffixes."""
        for i, ch in enumerate(token):
            if ch == "-" and i > 0:
                return i
        return -1

    def _parse_dash_range(
        self, token: str, dash: int, original: str | None = None
    ) -> list[tuple[str, str]]:
        """Parse JTs-87s or 22-66 style ranges."""
        original = token if original is None else original
        left, right = token[:dash], token[dash + 1 :]
        if not left or not right:
            raise ValueError(f"Invalid range token: {original!r}")

        # Pocket pair range: 22-66
        if len(left) == 2 and left[0] == left[1] and len(right) == 2 and right[0] == right[1]:
            self._validate_ranks(left[0] + right[0], original)
            lo = min(_RANK_IDX[left[0]], _RANK_IDX[right[0]])
            hi = max(_RANK_IDX[left[0]], _RANK_IDX[right[0]])
            return [c for r in RANKS[lo : hi + 1] for c in _pair_combos(r)]

        # Connector range: JTs-87s or JTo-87o
        suited = left.endswith("s") and right.endswith("s")
        offsuit = left.endswith("o") and right.endswith("o")
        if suited or offsuit:
            l_ranks = left[:-1]
            r_ranks = right[:-1]
        else:
            l_ranks = left
            r_ranks = right

        if len(l_ranks) != 2 or len(r_ranks) != 2:
            raise ValueError(f"Invalid range token: {original!r}")
        self._validate_ranks(l_ranks + r_ranks, original)
        if l_ranks[0] == l_ranks[1] or r_ranks[0] == r_ranks[1]:
            raise ValueError(f"Use pair range syntax (e.g. 22-66) for pairs: {original!r}")

        hi_l, lo_l = sorted([_RANK_IDX[l_ranks[0]], _RANK_IDX[l_ranks[1]]], reverse=True)
        hi_r, lo_r = sorted([_RANK_IDX[r_ranks[0]], _RANK_IDX[r_ranks[1]]], reverse=True)
        gap = hi_l - lo_l  # gap should be same for all connectors in range

        if gap != hi_r - lo_r:
            raise ValueError(f"Mismatched gap in connector range: {original!r}")

        lo_start = min(lo_l, lo_r)
        lo_end = max(lo_l, lo_r)
        combos: list[tuple[str, str]] = []
        for lo_idx in range(lo_start, lo_end + 1):
            hi_idx = lo_idx + gap
            if hi_idx >= len(RANKS):
                continue
            r1, r2 = RANKS[hi_idx], RANKS[lo_idx]
            if suited:
                combos.extend(_suited_combos(r1, r2))
            elif offsuit:
                combos.extend(_offsuit_combos(r1, r2))
            else:
                combos.extend(_suited_combos(r1, r2))
                combos.extend(_offsuit_combos(r1, r2))
        return combos
