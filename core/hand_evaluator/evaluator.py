"""Hand strength evaluator using treys."""
from treys import Evaluator, Card


class HandEvaluator:
    def __init__(self):
        self._eval = Evaluator()

    def rank(self, hole: list[str], board: list[str]) -> int:
        """Return hand rank (lower = stronger). hole/board are card strings like 'As', 'Kh'."""
        h = [Card.new(c) for c in hole]
        b = [Card.new(c) for c in board]
        return self._eval.evaluate(b, h)

    def percentile(self, hole: list[str], board: list[str]) -> float:
        """Return hand percentile [0, 1] where 1 = nuts."""
        rank = self.rank(hole, board)
        return 1 - self._eval.get_five_card_rank_percentage(rank)
