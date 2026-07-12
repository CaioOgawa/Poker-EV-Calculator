"""Independent Chip Model (ICM) calculator — Malmuth-Harville algorithm.

Complexity: O(n * 2^n) with memoization, viable through ~15 players.
The old permutations-based approach was O(n!) and also had a correctness bug
(joint probability overcounting for earlier finishers).
"""

from __future__ import annotations
from functools import lru_cache


class ICMModel:
    def equity(self, stacks: list[int], payouts: list[float]) -> list[float]:
        """Return each player's ICM equity as a fraction of total prize pool.

        stacks: chip counts (0 for busted players).
        payouts: prize fractions in finishing order (must sum to ≤ 1).
        """
        n = len(stacks)
        stacks_t = tuple(stacks)
        payouts_t = tuple(payouts)

        # Only players with chips can finish — busted players always get 0.
        active = frozenset(i for i in range(n) if stacks_t[i] > 0)
        n_active = len(active)

        @lru_cache(maxsize=None)
        def _ev(player: int, remaining: frozenset[int]) -> float:
            # How many places have already been assigned in this branch?
            place = n_active - len(remaining)
            if place >= len(payouts_t):
                return 0.0

            total = sum(stacks_t[j] for j in remaining)
            p_player_first = stacks_t[player] / total
            ev = p_player_first * payouts_t[place]

            for j in remaining:
                if j == player:
                    continue
                p_j_first = stacks_t[j] / total
                ev += p_j_first * _ev(player, remaining - {j})

            return ev

        result = []
        for i in range(n):
            if stacks_t[i] == 0:
                result.append(0.0)
            else:
                result.append(_ev(i, active))

        _ev.cache_clear()
        return result
