"""ICM Pressure (Bubble Factor) calculator.

Bubble Factor measures how much riskier a chip-for-chip confrontation is
for player i vs player j compared to a chip-EV game.

BF(i vs j) = ICM_loss_if_i_busts_to_j / ICM_gain_if_i_wins_j_chips

BF > 1 always (ICM punishes busting more than it rewards winning).
BF near 1 = chip leader, little pressure.
BF >> 1 = short stack near the bubble.
"""
from __future__ import annotations
from engine.icm.model import ICMModel


class ICMPressure:
    def __init__(self, model: ICMModel | None = None):
        self._model = model or ICMModel()

    def bubble_factor(
        self,
        stacks: list[int],
        payouts: list[float],
        hero: int,
        villain: int,
    ) -> float:
        """Return the bubble factor for hero risking chips against villain.

        stacks: chip counts for all players.
        payouts: prize fractions in finishing order.
        hero: index of the player who might push/call.
        villain: index of the opponent.

        Returns BF = ICM_loss / ICM_gain (always ≥ 1 with >2 players).
        """
        eq_base = self._model.equity(stacks, payouts)

        # Hero wins all of villain's chips
        stacks_win = list(stacks)
        stacks_win[hero] += stacks[villain]
        stacks_win[villain] = 0
        eq_win = self._model.equity(stacks_win, payouts)
        icm_gain = eq_win[hero] - eq_base[hero]

        # Hero loses all chips to villain
        stacks_lose = list(stacks)
        stacks_lose[villain] += stacks[hero]
        stacks_lose[hero] = 0
        eq_lose = self._model.equity(stacks_lose, payouts)
        icm_loss = eq_base[hero] - eq_lose[hero]

        if icm_gain <= 0:
            return float("inf")

        return icm_loss / icm_gain

    def all_bubble_factors(
        self,
        stacks: list[int],
        payouts: list[float],
    ) -> list[list[float]]:
        """Return matrix BF[i][j] = bubble factor for player i vs player j.

        Diagonal (i == j) is set to 1.0 (meaningless confrontation).
        """
        n = len(stacks)
        return [
            [
                1.0 if i == j else self.bubble_factor(stacks, payouts, i, j)
                for j in range(n)
            ]
            for i in range(n)
        ]

    def pressure_ranking(
        self,
        stacks: list[int],
        payouts: list[float],
    ) -> list[tuple[int, float]]:
        """Return players sorted by their average bubble factor (most pressured first).

        Each player's pressure = mean BF vs all other players.
        """
        n = len(stacks)
        matrix = self.all_bubble_factors(stacks, payouts)
        avg = []
        for i in range(n):
            others = [matrix[i][j] for j in range(n) if j != i]
            avg.append((i, sum(others) / len(others) if others else 1.0))
        return sorted(avg, key=lambda x: x[1], reverse=True)
