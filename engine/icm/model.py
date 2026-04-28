"""Independent Chip Model (ICM) calculator for tournament spots."""
from __future__ import annotations
from itertools import permutations


class ICMModel:
    def equity(self, stacks: list[int], payouts: list[float]) -> list[float]:
        """
        Return each player's ICM equity as fraction of total prize pool.
        stacks: chip counts; payouts: prize fractions (must sum to 1).
        """
        total = sum(stacks)
        n = len(stacks)
        equities = [0.0] * n

        for perm in permutations(range(n)):
            prob = 1.0
            remaining = total
            for place, player in enumerate(perm):
                prob *= stacks[player] / remaining
                remaining -= stacks[player]
                if place < len(payouts):
                    equities[player] += prob * payouts[place]
        return equities
