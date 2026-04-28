"""Monte Carlo equity calculator for heads-up and multi-way spots."""
from __future__ import annotations
import random
from treys import Evaluator, Card, Deck


class EquityCalculator:
    def __init__(self, iterations: int = 10_000):
        self.iterations = iterations
        self._eval = Evaluator()

    def heads_up(self, hero: list[str], villain: list[str], board: list[str] | None = None) -> float:
        """Return hero equity [0, 1] vs a known villain hand."""
        board = board or []
        wins = 0
        hero_c = [Card.new(c) for c in hero]
        villain_c = [Card.new(c) for c in villain]
        known = set(hero_c + villain_c + [Card.new(c) for c in board])

        for _ in range(self.iterations):
            deck = [c for c in Deck().cards if c not in known]
            random.shuffle(deck)
            run_board = [Card.new(c) for c in board] + deck[: 5 - len(board)]
            hero_rank = self._eval.evaluate(run_board, hero_c)
            villain_rank = self._eval.evaluate(run_board, villain_c)
            if hero_rank < villain_rank:
                wins += 1
            elif hero_rank == villain_rank:
                wins += 0.5
        return wins / self.iterations
