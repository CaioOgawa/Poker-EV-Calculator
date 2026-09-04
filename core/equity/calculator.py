"""Equity calculator for heads-up and multi-way spots.

Exact enumeration when 0 or 1 cards remain to be dealt (a complete or
turn-complete board), Monte Carlo sampling otherwise
(docs/AUDITORIA-2026-08-26.md item I1) — see each method's docstring.
`range_vs_range` stays Monte Carlo-only regardless of `need`: it's built for
range-vs-range spots (e.g. the 169x169 preflop equity table in
`engine/icm/build_equity_table.py`), where exact combo-pair enumeration
would multiply, not just add, the per-side combo counts.
"""

from __future__ import annotations
import random as _random_mod
from itertools import accumulate
from treys import Evaluator, Card, Deck

from core.ranges.range_parser import RangeParser

_PARSER = RangeParser()


def _to_cards(strs: list[str]) -> list[int]:
    return [Card.new(c) for c in strs]


def _full_deck() -> list[int]:
    return list(Deck().cards)


def _win_share(rank_a: int, rank_b: int) -> float:
    """1.0/0.5/0.0 for a's share of a heads-up showdown (treys: lower rank wins)."""
    if rank_a < rank_b:
        return 1.0
    if rank_a == rank_b:
        return 0.5
    return 0.0


def _check_no_duplicates(groups: dict[str, list[str]]) -> None:
    """Raise ValueError naming the card if the same card appears twice,
    whether within one group (e.g. hero=['As','As']) or across groups
    (e.g. the same card in both hero and board) — a silently-collapsed
    duplicate feeds the same card to both hands and returns a plausible,
    wrong equity instead of failing loudly.
    """
    seen: dict[str, str] = {}
    for group_name, cards in groups.items():
        for card in cards:
            if card in seen:
                raise ValueError(
                    f"Duplicate card '{card}' in both {seen[card]!r} and {group_name!r}"
                )
            seen[card] = group_name


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
        """Hero equity [0, 1] vs a single known villain hand.

        Exact, not Monte Carlo, whenever the board leaves 0 or 1 cards to
        come (docs/AUDITORIA-2026-08-26.md item I1): a complete board is one
        deterministic showdown, and a turn-complete board has at most 46
        possible river cards — both cheap to enumerate outright, so there's
        no sampling noise to pay for. `seed` is unused in both cases (no
        randomness to seed). Falls back to Monte Carlo sampling only when 2+
        cards remain (preflop/flop).
        """
        board = board or []
        _check_no_duplicates({"hero": hero, "villain": villain, "board": board})
        hero_c = _to_cards(hero)
        villain_c = _to_cards(villain)
        board_c = _to_cards(board)
        known = set(hero_c + villain_c + board_c)

        deck = [c for c in _full_deck() if c not in known]
        need = 5 - len(board_c)

        if need == 0:
            return _win_share(
                self._eval.evaluate(board_c, hero_c), self._eval.evaluate(board_c, villain_c)
            )
        if need == 1:
            wins = sum(
                _win_share(
                    self._eval.evaluate(board_c + [card], hero_c),
                    self._eval.evaluate(board_c + [card], villain_c),
                )
                for card in deck
            )
            return wins / len(deck)

        rng = _random_mod.Random(seed) if seed is not None else self._rng
        wins = 0.0
        for _ in range(self.iterations):
            rng.shuffle(deck)
            run_board = board_c + deck[:need]
            h = self._eval.evaluate(run_board, hero_c)
            v = self._eval.evaluate(run_board, villain_c)
            wins += _win_share(h, v)
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

        Each combo in the range is equally likely a priori (`RangeParser`
        already lists one entry per combo, so no extra weighting is needed),
        after removing combos blocked by hero/board cards. With 2+ cards
        still to come, one villain combo is sampled per Monte Carlo
        iteration, then a random runout. With 0 or 1 cards to come
        (docs/AUDITORIA-2026-08-26.md item I1), both the villain's combo and
        the runout are enumerated exactly instead — a river-complete board
        has only the combo itself to average over (≤ a few hundred, cheap),
        and a turn-complete board adds at most 46 river cards per combo.
        `seed` is unused in both exact cases (no randomness to seed).
        """
        board = board or []
        _check_no_duplicates({"hero": hero, "board": board})
        hero_c = _to_cards(hero)
        board_c = _to_cards(board)
        hero_set = set(hero_c + board_c)

        all_combos = _PARSER.parse(villain_range)
        available = [
            _to_cards(list(combo))
            for combo in all_combos
            if not any(c in hero_set for c in _to_cards(list(combo)))
        ]
        if not available:
            raise ValueError(
                f"No unblocked combos in range '{villain_range}' given hero {hero} board {board}"
            )

        return self._vs_weighted(hero_c, board_c, [(cards, 1.0) for cards in available], seed=seed)

    def vs_weighted_range(
        self,
        hero: list[str],
        weighted_hands: list[tuple[str, float]],
        board: list[str] | None = None,
        *,
        seed: int | None = None,
    ) -> float:
        """Hero equity [0, 1] vs a range where each canonical hand carries its
        own weight — e.g. `PreflopChart.combo_weighted_range('push')` fed in
        directly (docs/AUDITORIA-2026-08-26.md item F5), so a solver's mixed
        strategy (AKs pushed 60% of the time) shapes equity instead of
        collapsing to `vs_range`'s binary in/out range.

        Every combo of a given canonical hand shares that hand's weight — a
        mixed strategy is a frequency over hand *classes*, not individual
        combos, so the 4 AKs combos are indistinguishable to a chart. Same
        need==0/1/2+ exact-vs-Monte-Carlo dispatch as `vs_range`
        (docs/AUDITORIA-2026-08-26.md item I1).
        """
        board = board or []
        _check_no_duplicates({"hero": hero, "board": board})
        hero_c = _to_cards(hero)
        board_c = _to_cards(board)
        hero_set = set(hero_c + board_c)

        weighted_combos = [
            (cards, weight)
            for hand, weight in weighted_hands
            if weight > 0.0
            for combo in _PARSER.expand_hand(hand)
            for cards in [_to_cards(list(combo))]
            if not any(c in hero_set for c in cards)
        ]
        if not weighted_combos:
            raise ValueError(
                f"No unblocked, positively-weighted combos given hero {hero} board {board}"
            )

        return self._vs_weighted(hero_c, board_c, weighted_combos, seed=seed)

    def _vs_weighted(
        self,
        hero_c: list[int],
        board_c: list[int],
        weighted_combos: list[tuple[list[int], float]],
        *,
        seed: int | None,
    ) -> float:
        """Shared engine behind `vs_range`/`vs_weighted_range`: `vs_range`
        calls this with every combo weighted 1.0, which is exactly its old
        flat-average behavior since a weighted average over equal weights is
        the flat average.
        """
        hero_set = set(hero_c) | set(board_c)
        need = 5 - len(board_c)
        total_weight = sum(w for _, w in weighted_combos)

        if need == 0:
            hero_rank = self._eval.evaluate(board_c, hero_c)
            wins = sum(
                w * _win_share(hero_rank, self._eval.evaluate(board_c, cards))
                for cards, w in weighted_combos
            )
            return wins / total_weight

        if need == 1:
            # Weighted average-of-averages, not a flat win/total ratio: the
            # villain combo is drawn proportional to its weight (matching the
            # MC branch's `rng.choices(..., weights=...)`), the runout is
            # uniform *given* that combo. Each combo's own runout count
            # varies with what its two cards block, so weighting by combo
            # first and runout second inside that combo is what reproduces
            # the MC estimator's weighting exactly.
            weighted_wins = 0.0
            for cards, w in weighted_combos:
                villain_set = set(cards)
                deck = [c for c in _full_deck() if c not in hero_set and c not in villain_set]
                wins = 0.0
                for card in deck:
                    run_board = board_c + [card]
                    h = self._eval.evaluate(run_board, hero_c)
                    v = self._eval.evaluate(run_board, cards)
                    wins += _win_share(h, v)
                weighted_wins += w * (wins / len(deck))
            return weighted_wins / total_weight

        deck_base = [c for c in _full_deck() if c not in hero_set]
        rng = _random_mod.Random(seed) if seed is not None else self._rng
        combos_only = [cards for cards, _ in weighted_combos]
        cum_weights = list(accumulate(w for _, w in weighted_combos))
        wins = 0.0
        for _ in range(self.iterations):
            (villain_c,) = rng.choices(combos_only, cum_weights=cum_weights, k=1)
            villain_set = set(villain_c)
            deck = [c for c in deck_base if c not in villain_set]
            rng.shuffle(deck)
            run_board = board_c + deck[:need]
            h = self._eval.evaluate(run_board, hero_c)
            v = self._eval.evaluate(run_board, villain_c)
            wins += _win_share(h, v)
        return wins / self.iterations

    def range_vs_range(
        self,
        range_a: str,
        range_b: str,
        board: list[str] | None = None,
        *,
        seed: int | None = None,
    ) -> tuple[float, float]:
        """Return (equity_a, equity_b) for two ranges colliding heads-up.

        Each iteration samples one unblocked combo from each range, then runs
        a heads-up Monte Carlo runout. Combos that block each other are removed
        before sampling. Iterations where the sampled combo from range_a blocks
        every combo in range_b are skipped and excluded from the denominator
        (they carry no information about either range's equity).
        """
        board = board or []
        _check_no_duplicates({"board": board})
        rng = _random_mod.Random(seed) if seed is not None else self._rng
        board_c = _to_cards(board)
        board_set = set(board_c)

        combos_a = [_to_cards(list(c)) for c in _PARSER.parse(range_a)]
        combos_b = [_to_cards(list(c)) for c in _PARSER.parse(range_b)]

        need = 5 - len(board_c)
        wins_a = 0.0
        valid = 0

        for _ in range(self.iterations):
            # Sample combo A first, then pick an unblocked combo B
            a_cards = rng.choice(combos_a)
            a_set = set(a_cards) | board_set

            available_b = [cb for cb in combos_b if not any(c in a_set for c in cb)]
            if not available_b:
                continue

            valid += 1
            b_cards = rng.choice(available_b)
            known = a_set | set(b_cards)
            deck = [c for c in _full_deck() if c not in known]
            rng.shuffle(deck)
            run_board = board_c + deck[:need]

            ra = self._eval.evaluate(run_board, a_cards)
            rb = self._eval.evaluate(run_board, b_cards)
            if ra < rb:
                wins_a += 1.0
            elif ra == rb:
                wins_a += 0.5

        if valid == 0:
            raise ValueError(
                f"Ranges '{range_a}' and '{range_b}' block each other completely "
                f"given board {board}"
            )

        eq_a = wins_a / valid
        return eq_a, 1.0 - eq_a

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

        Exact, not Monte Carlo, with 0 or 1 cards to come
        (docs/AUDITORIA-2026-08-26.md item I1) — same reasoning as `heads_up`.
        `seed` is unused in both exact cases (no randomness to seed).
        """
        board = board or []
        _check_no_duplicates({f"hand {i}": h for i, h in enumerate(hands)} | {"board": board})
        n = len(hands)
        hands_c = [_to_cards(h) for h in hands]
        board_c = _to_cards(board)
        known = set(c for h in hands_c for c in h) | set(board_c)

        deck = [c for c in _full_deck() if c not in known]
        need = 5 - len(board_c)

        def _shares(run_board: list[int]) -> list[float]:
            ranks = [self._eval.evaluate(run_board, h) for h in hands_c]
            best = min(ranks)
            winners = [i for i, r in enumerate(ranks) if r == best]
            share = 1.0 / len(winners)
            return [share if r == best else 0.0 for r in ranks]

        if need == 0:
            return _shares(board_c)
        if need == 1:
            equity = [0.0] * n
            for card in deck:
                for i, s in enumerate(_shares(board_c + [card])):
                    equity[i] += s
            return [e / len(deck) for e in equity]

        rng = _random_mod.Random(seed) if seed is not None else self._rng
        equity = [0.0] * n
        for _ in range(self.iterations):
            rng.shuffle(deck)
            run_board = board_c + deck[:need]
            for i, s in enumerate(_shares(run_board)):
                equity[i] += s

        return [e / self.iterations for e in equity]
