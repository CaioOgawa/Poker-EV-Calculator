"""Monte Carlo equity calculator for heads-up and multi-way spots."""
from __future__ import annotations
import random as _random_mod
from treys import Evaluator, Card, Deck

from core.ranges.range_parser import RangeParser

_PARSER = RangeParser()


def _to_cards(strs: list[str]) -> list[int]:
    return [Card.new(c) for c in strs]


def _full_deck() -> list[int]:
    return list(Deck().cards)


class EquityCalculator:
    def __init__(self, iterations: int = 10_000, seed: int | None = None):
        self.iterations = iterations
        self._eval = Evaluator()
        self._rng = _random_mod.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def heads_up(
        self,
        hero: list[str],
        villain: list[str],
        board: list[str] | None = None,
        *,
        seed: int | None = None,
    ) -> float:
        """Hero equity [0, 1] vs a single known villain hand."""
        rng = _random_mod.Random(seed) if seed is not None else self._rng
        board = board or []
        hero_c = _to_cards(hero)
        villain_c = _to_cards(villain)
        board_c = _to_cards(board)
        known = set(hero_c + villain_c + board_c)

        deck = [c for c in _full_deck() if c not in known]
        need = 5 - len(board_c)
        wins = 0.0

        for _ in range(self.iterations):
            rng.shuffle(deck)
            run_board = board_c + deck[:need]
            h = self._eval.evaluate(run_board, hero_c)
            v = self._eval.evaluate(run_board, villain_c)
            if h < v:
                wins += 1.0
            elif h == v:
                wins += 0.5
        return wins / self.iterations

    def vs_range(
        self,
        hero: list[str],
        villain_range: str,
        board: list[str] | None = None,
        *,
        seed: int | None = None,
    ) -> float:
        """Hero equity [0, 1] vs a villain range string (e.g. 'JJ+,AKs').

        Each iteration samples one combo from the villain range after removing
        combos blocked by hero cards or board cards.
        """
        rng = _random_mod.Random(seed) if seed is not None else self._rng
        board = board or []
        hero_c = _to_cards(hero)
        board_c = _to_cards(board)
        hero_set = set(hero_c + board_c)

        all_combos = _PARSER.parse(villain_range)
        # Filter combos blocked by hero/board
        available = [
            (_to_cards(list(combo)), combo)
            for combo in all_combos
        ]
        available = [
            (cards, combo)
            for cards, combo in available
            if not any(c in hero_set for c in cards)
        ]
        if not available:
            raise ValueError(
                f"No unblocked combos in range '{villain_range}' given hero {hero} board {board}"
            )

        deck_base = [c for c in _full_deck() if c not in hero_set]
        need = 5 - len(board_c)
        wins = 0.0

        for _ in range(self.iterations):
            villain_c, _ = rng.choice(available)
            villain_set = set(villain_c)
            deck = [c for c in deck_base if c not in villain_set]
            rng.shuffle(deck)
            run_board = board_c + deck[:need]
            h = self._eval.evaluate(run_board, hero_c)
            v = self._eval.evaluate(run_board, villain_c)
            if h < v:
                wins += 1.0
            elif h == v:
                wins += 0.5
        return wins / self.iterations

    def multi_way(
        self,
        hands: list[list[str]],
        board: list[str] | None = None,
        *,
        seed: int | None = None,
    ) -> list[float]:
        """Return equity [0, 1] for each player in a multi-way pot.

        hands: list of hole-card lists, e.g. [['As','Kd'], ['Qh','Qc'], ['7s','8s']]
        Ties split equity equally among all tied players.
        """
        rng = _random_mod.Random(seed) if seed is not None else self._rng
        board = board or []
        n = len(hands)
        hands_c = [_to_cards(h) for h in hands]
        board_c = _to_cards(board)
        known = set(c for h in hands_c for c in h) | set(board_c)

        deck = [c for c in _full_deck() if c not in known]
        need = 5 - len(board_c)
        equity = [0.0] * n

        for _ in range(self.iterations):
            rng.shuffle(deck)
            run_board = board_c + deck[:need]
            ranks = [self._eval.evaluate(run_board, h) for h in hands_c]
            best = min(ranks)
            winners = [i for i, r in enumerate(ranks) if r == best]
            share = 1.0 / len(winners)
            for i in winners:
                equity[i] += share

        return [e / self.iterations for e in equity]
