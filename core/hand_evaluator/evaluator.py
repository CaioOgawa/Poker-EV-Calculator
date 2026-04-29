"""Hand strength evaluator using treys."""
from __future__ import annotations
from treys import Evaluator, Card, Deck


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

    def hand_class(self, hole: list[str], board: list[str]) -> str:
        """Return human-readable hand class, e.g. 'Flush', 'Two Pair', 'High Card'."""
        h = [Card.new(c) for c in hole]
        b = [Card.new(c) for c in board]
        r = self._eval.evaluate(b, h)
        return self._eval.class_to_string(self._eval.get_rank_class(r))

    def outs(self, hole: list[str], board: list[str]) -> list[str]:
        """Return list of card strings that move the hand to a strictly better class.

        Only meaningful on flop (3 board cards) or turn (4 board cards).
        An "out" is any unseen card that raises the hand to a better category
        (e.g. High Card → Pair, Flush Draw → Flush). Within-class rank improvements
        (e.g. better kicker) do not count.
        """
        h_ints = [Card.new(c) for c in hole]
        b_ints = [Card.new(c) for c in board]
        current_class = self._eval.get_rank_class(self._eval.evaluate(b_ints, h_ints))

        known = set(h_ints + b_ints)
        result: list[str] = []

        for card in Deck().cards:
            if card in known:
                continue
            new_class = self._eval.get_rank_class(self._eval.evaluate(b_ints + [card], h_ints))
            if new_class < current_class:
                result.append(Card.int_to_str(card))

        return result

    def draw_strength(self, hole: list[str], board: list[str]) -> float:
        """Return fraction of unseen cards that move the hand to a better class [0, 1].

        Uses the same class-based definition as outs(). 0 = drawing dead, 1 = every
        unseen card upgrades the hand category.
        """
        h_ints = [Card.new(c) for c in hole]
        b_ints = [Card.new(c) for c in board]
        current_class = self._eval.get_rank_class(self._eval.evaluate(b_ints, h_ints))

        known = set(h_ints + b_ints)
        unseen = [c for c in Deck().cards if c not in known]

        if not unseen:
            return 0.0

        improving = sum(
            1 for c in unseen
            if self._eval.get_rank_class(self._eval.evaluate(b_ints + [c], h_ints)) < current_class
        )
        return improving / len(unseen)
