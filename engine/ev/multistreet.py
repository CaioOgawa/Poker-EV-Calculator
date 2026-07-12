"""Multi-street (turn/river) EV solver — chance-sampled vanilla CFR over a
heads-up betting subgame with weighted ranges.

Scope, deliberately bounded ("CFR-lite", per the project roadmap):
  - Heads-up only, OOP acts first on every street.
  - `board` must have 3 cards (flop — solves turn AND river) or 4 cards
    (turn — solves river only). There's no street left to solve from a
    5-card board.
  - Each betting round allows at most one bet and a fold/call response —
    no raises. This bounds the tree; it's the "lite" in CFR-lite.
  - Card removal is always exact: every (OOP combo, IP combo, run-out)
    triple is evaluated individually via `treys`, weighted by its natural
    probability given the cards already in play. Nothing about *payoffs*
    is abstracted.
  - `bucket_report()` groups a range into human-readable equity tiers for
    reporting. It's a read-only aggregation over the exact per-combo
    solve — every combo still gets its own solved strategy — not a
    solver-side abstraction that shares strategies across hands the way
    production solvers bucket ranges for tractability.

Algorithm: "chance-sampled" CFR (an established, simple CFR variant — see
Zinkevich et al.'s MCCFR). Each training iteration samples one (OOP combo,
IP combo, run-out) uniformly and walks a single path through the betting
tree, updating regret-matching tables for the info sets it touches. This
avoids ever materializing the full combo x run-out state space during
training, at the cost of needing enough iterations for the sampling noise
to average out.

After training, `solve()` runs one exact, unsampled pass over every legal
(combo, run-out) pair to compute the trained average strategy's expected
value and its exploitability (the total gain either player could get by
best-responding instead of playing their average strategy — the standard
CFR convergence diagnostic; it should shrink towards 0 as `iterations`
grows, and by construction is never negative in a correct implementation).

Because nothing is abstracted, the number of (combo-pair x run-out) terminal
evaluations the final pass has to do grows fast — a few dozen combos per
side times ~2,000 turn/river run-outs is still fine, full unpruned ranges
are not. `max_combo_pairs` guards against silently running something that
would take far too long; narrow the ranges or raise it deliberately.

Performance note: a `board` with 4 cards (river only — one street, ~44
run-outs per combo pair) solves in well under a second for a handful of
combos per side. A `board` with 3 cards (flop — two streets, turn AND
river, ~2,000 run-outs per combo pair) is far more expensive — both the
per-solve cost and the number of iterations needed to converge grow a lot
— and can take minutes even for small ranges. Prefer river-only solves
unless you specifically need the turn decision too.
"""

from __future__ import annotations
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from treys import Card, Deck, Evaluator

from core.ranges.range_parser import RangeParser

_PARSER = RangeParser()
_EVAL = Evaluator()
_FULL_DECK: tuple[int, ...] = tuple(Deck().cards)

Combo = tuple[int, int]


@dataclass
class MultiStreetResult:
    pot: float
    oop_ev: float
    ip_ev: float
    exploitability: float


def _regret_match(regrets: Sequence[float]) -> list[float]:
    positive = [max(r, 0.0) for r in regrets]
    total = sum(positive)
    n = len(regrets)
    if total <= 0:
        return [1.0 / n] * n
    return [r / total for r in positive]


def _canon(combo: Combo) -> Combo:
    a, b = sorted(combo)
    return (a, b)


def _combo_str(combo: Combo) -> str:
    return "".join(Card.int_to_str(c) for c in combo)


def _action_labels(history: str, bet_sizes: tuple[float, ...]) -> list[str]:
    if history in ("", "x"):
        return ["check"] + [f"bet_{b}" for b in bet_sizes]
    return ["fold", "call"]


def _combos(range_str: str, blocked: set[int]) -> list[Combo]:
    combos: list[Combo] = []
    for c1, c2 in _PARSER.parse(range_str):
        cards = (Card.new(c1), Card.new(c2))
        if cards[0] in blocked or cards[1] in blocked:
            continue
        combos.append(cards)
    return combos


def _legal_pairs(
    oop_combos: list[Combo], ip_combos: list[Combo], max_pairs: int
) -> list[tuple[Combo, Combo]]:
    pairs = [(oop, ip) for oop in oop_combos for ip in ip_combos if not set(oop) & set(ip)]
    if not pairs:
        raise ValueError(
            "No legal (OOP, IP) combo pairs given board/ranges — ranges fully block each other"
        )
    if len(pairs) > max_pairs:
        raise ValueError(
            f"{len(pairs)} legal combo pairs exceeds max_combo_pairs={max_pairs}; narrow the "
            "ranges or raise max_combo_pairs (solve time grows with combo pairs)."
        )
    return pairs


def _enumerate_deals(deck: list[int], n_deal: int) -> Iterator[tuple[int, ...]]:
    """Every possible ordered n_deal-card sequence dealt from `deck`."""
    if n_deal == 0:
        yield ()
    elif n_deal == 1:
        for c in deck:
            yield (c,)
    elif n_deal == 2:
        for i, c1 in enumerate(deck):
            for c2 in deck[:i] + deck[i + 1 :]:
                yield (c1, c2)
    else:
        raise ValueError(f"n_deal must be 0, 1, or 2, got {n_deal}")


class MultiStreetEV:
    def __init__(self, iterations: int = 400, seed: int | None = None, max_combo_pairs: int = 30):
        self.iterations = iterations
        self.max_combo_pairs = max_combo_pairs
        self._rng = random.Random(seed)

        self._regret_sum: dict[tuple, list[float]] = {}
        self._strategy_sum: dict[tuple, list[float]] = {}
        self._bet_sizes: tuple[float, ...] | None = None
        self._board_start: tuple[int, ...] | None = None
        self._oop_combos: list[Combo] | None = None
        self._ip_combos: list[Combo] | None = None
        self._pairs: list[tuple[Combo, Combo]] | None = None
        self._n_deal: int | None = None
        self._pot: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        oop_range: str,
        ip_range: str,
        board: list[str],
        pot: float,
        bet_sizes: Sequence[float] = (0.5, 1.0),
    ) -> MultiStreetResult:
        if pot <= 0:
            raise ValueError(f"pot must be positive, got {pot}")
        if not bet_sizes or any(b <= 0 for b in bet_sizes):
            raise ValueError("bet_sizes must be a non-empty sequence of positive pot fractions")
        board_cards = tuple(Card.new(c) for c in board)
        if len(board_cards) not in (3, 4):
            raise ValueError(f"board must have 3 (flop) or 4 (turn) cards, got {len(board_cards)}")
        n_deal = 5 - len(board_cards)

        blocked = set(board_cards)
        oop_combos = _combos(oop_range, blocked)
        ip_combos = _combos(ip_range, blocked)
        if not oop_combos or not ip_combos:
            raise ValueError("oop_range/ip_range produced no combos after removing board cards")

        pairs = _legal_pairs(oop_combos, ip_combos, self.max_combo_pairs)

        self._regret_sum = {}
        self._strategy_sum = {}
        self._bet_sizes = tuple(bet_sizes)
        self._board_start = board_cards
        self._oop_combos = oop_combos
        self._ip_combos = ip_combos
        self._pairs = pairs
        self._n_deal = n_deal
        self._pot = pot

        full_deck = [c for c in _FULL_DECK if c not in blocked]

        for _ in range(self.iterations):
            oop_combo, ip_combo = self._rng.choice(pairs)
            pair_blocked = set(oop_combo) | set(ip_combo)
            deck = [c for c in full_deck if c not in pair_blocked]
            self._rng.shuffle(deck)
            deal_queue = deck[:n_deal]
            self._root(
                oop_combo,
                ip_combo,
                board_cards,
                deal_queue,
                self._bet_sizes,
                pot,
                0.0,
                0.0,
                1.0,
                1.0,
                "train",
            )

        oop_ev, exploitability = self._evaluate()
        return MultiStreetResult(
            pot=pot, oop_ev=oop_ev, ip_ev=pot - oop_ev, exploitability=exploitability
        )

    def strategy(
        self, player: str, combo: list[str], board: list[str], history: str = ""
    ) -> dict[str, float]:
        """Trained average strategy at one information set, after `solve()`."""
        if self._bet_sizes is None:
            raise RuntimeError("call solve() before strategy()")
        if player not in ("oop", "ip"):
            raise ValueError("player must be 'oop' or 'ip'")
        combo_ints = _canon((Card.new(combo[0]), Card.new(combo[1])))
        board_ints = tuple(Card.new(c) for c in board)
        key = (player, combo_ints, board_ints, history)
        labels = _action_labels(history, self._bet_sizes)
        freqs = self._get_strategy(key, len(labels), "avg")
        return dict(zip(labels, freqs))

    def bucket_report(self, player: str, n_buckets: int = 3) -> dict[str, list[str]]:
        """Group `player`'s range into `n_buckets` equity tiers vs the opponent's
        full range on the starting board (exact enumeration, not a solver-side
        abstraction — see module docstring).
        """
        if self._oop_combos is None or self._ip_combos is None:
            raise RuntimeError("call solve() before bucket_report()")
        if player not in ("oop", "ip"):
            raise ValueError("player must be 'oop' or 'ip'")
        own_combos = self._oop_combos if player == "oop" else self._ip_combos
        opp_combos = self._ip_combos if player == "oop" else self._oop_combos
        assert self._board_start is not None and self._n_deal is not None

        full_deck = [c for c in _FULL_DECK if c not in set(self._board_start)]
        equities: list[tuple[Combo, float]] = []
        for own in own_combos:
            wins = 0.0
            total = 0.0
            for opp in opp_combos:
                if set(own) & set(opp):
                    continue
                pair_blocked = set(own) | set(opp)
                deck = [c for c in full_deck if c not in pair_blocked]
                for deal in _enumerate_deals(deck, self._n_deal):
                    full_board = self._board_start + deal
                    own_rank = _EVAL.evaluate(list(full_board), list(own))
                    opp_rank = _EVAL.evaluate(list(full_board), list(opp))
                    if own_rank < opp_rank:
                        wins += 1.0
                    elif own_rank == opp_rank:
                        wins += 0.5
                    total += 1.0
            equities.append((own, wins / total if total else 0.0))

        equities.sort(key=lambda item: item[1], reverse=True)
        n = len(equities)
        n_buckets = max(1, min(n_buckets, n))
        size = -(-n // n_buckets)  # ceil, so tiers stay non-empty while there's range left
        buckets: dict[str, list[str]] = {}
        for b in range(n_buckets):
            chunk = equities[b * size : (b + 1) * size]
            if not chunk:
                continue
            buckets[f"tier_{b + 1}"] = [_combo_str(c) for c, _ in chunk]
        return buckets

    # ------------------------------------------------------------------
    # Tree traversal — one implementation, four modes, so training and
    # evaluation can never silently diverge from each other.
    #   "train"  — chance/combo-sampled, regret-matched strategy, updates
    #              regret_sum/strategy_sum for the info sets it visits.
    #   "avg"    — both players play their trained average strategy.
    #   "br_oop" — OOP best-responds (maximizes); IP plays its average
    #              strategy. Used for OOP's exploitability term.
    #   "br_ip"  — IP best-responds (minimizes OOP's value, since payoffs
    #              are pot-sum); OOP plays its average strategy.
    # ------------------------------------------------------------------

    def _get_strategy(self, key: tuple, n_actions: int, kind: str) -> list[float]:
        if kind == "train":
            regrets = self._regret_sum.get(key)
            return _regret_match(regrets) if regrets else [1.0 / n_actions] * n_actions
        strat_sum = self._strategy_sum.get(key)
        if not strat_sum:
            return [1.0 / n_actions] * n_actions
        total = sum(strat_sum)
        if total <= 0:
            return [1.0 / n_actions] * n_actions
        return [s / total for s in strat_sum]

    def _root(
        self,
        oop_combo: Combo,
        ip_combo: Combo,
        board: tuple[int, ...],
        deal_queue: list[int],
        bet_sizes: tuple[float, ...],
        pot: float,
        oop_c: float,
        ip_c: float,
        reach_oop: float,
        reach_ip: float,
        mode: str,
    ) -> float:
        n_actions = 1 + len(bet_sizes)
        key = ("oop", _canon(oop_combo), board, "")

        if mode == "train":
            strategy: list[float] | None = self._get_strategy(key, n_actions, "train")
        elif mode == "br_oop":
            strategy = None
        else:
            strategy = self._get_strategy(key, n_actions, "avg")

        action_values = [0.0] * n_actions
        r_oop = (
            (lambda i: reach_oop * strategy[i]) if strategy is not None else (lambda i: reach_oop)
        )
        action_values[0] = self._after_check(
            oop_combo,
            ip_combo,
            board,
            deal_queue,
            bet_sizes,
            pot,
            oop_c,
            ip_c,
            r_oop(0),
            reach_ip,
            mode,
        )
        for i in range(len(bet_sizes)):
            action_values[i + 1] = self._facing_oop_bet(
                oop_combo,
                ip_combo,
                board,
                deal_queue,
                bet_sizes,
                i,
                pot,
                oop_c,
                ip_c,
                r_oop(i + 1),
                reach_ip,
                mode,
            )

        if mode == "br_oop":
            node_value = max(action_values)
        else:
            assert strategy is not None
            node_value = sum(s * v for s, v in zip(strategy, action_values))

        if mode == "train":
            assert strategy is not None
            regrets = self._regret_sum.setdefault(key, [0.0] * n_actions)
            strat_sum = self._strategy_sum.setdefault(key, [0.0] * n_actions)
            for i in range(n_actions):
                regrets[i] += reach_ip * (action_values[i] - node_value)
                strat_sum[i] += reach_oop * strategy[i]

        return node_value

    def _after_check(
        self,
        oop_combo: Combo,
        ip_combo: Combo,
        board: tuple[int, ...],
        deal_queue: list[int],
        bet_sizes: tuple[float, ...],
        pot: float,
        oop_c: float,
        ip_c: float,
        reach_oop: float,
        reach_ip: float,
        mode: str,
    ) -> float:
        n_actions = 1 + len(bet_sizes)
        key = ("ip", _canon(ip_combo), board, "x")

        if mode == "train":
            strategy: list[float] | None = self._get_strategy(key, n_actions, "train")
        elif mode == "br_ip":
            strategy = None
        else:
            strategy = self._get_strategy(key, n_actions, "avg")

        action_values = [0.0] * n_actions
        r_ip = (lambda i: reach_ip * strategy[i]) if strategy is not None else (lambda i: reach_ip)
        action_values[0] = self._advance_or_showdown(
            oop_combo,
            ip_combo,
            board,
            deal_queue,
            bet_sizes,
            pot,
            oop_c,
            ip_c,
            reach_oop,
            r_ip(0),
            mode,
        )
        for i in range(len(bet_sizes)):
            action_values[i + 1] = self._facing_ip_bet_after_check(
                oop_combo,
                ip_combo,
                board,
                deal_queue,
                bet_sizes,
                i,
                pot,
                oop_c,
                ip_c,
                reach_oop,
                r_ip(i + 1),
                mode,
            )

        if mode == "br_ip":
            node_value = min(action_values)
        else:
            assert strategy is not None
            node_value = sum(s * v for s, v in zip(strategy, action_values))

        if mode == "train":
            assert strategy is not None
            regrets = self._regret_sum.setdefault(key, [0.0] * n_actions)
            strat_sum = self._strategy_sum.setdefault(key, [0.0] * n_actions)
            for i in range(n_actions):
                regrets[i] += reach_oop * (node_value - action_values[i])
                strat_sum[i] += reach_ip * strategy[i]

        return node_value

    def _facing_oop_bet(
        self,
        oop_combo: Combo,
        ip_combo: Combo,
        board: tuple[int, ...],
        deal_queue: list[int],
        bet_sizes: tuple[float, ...],
        i: int,
        pot: float,
        oop_c: float,
        ip_c: float,
        reach_oop: float,
        reach_ip: float,
        mode: str,
    ) -> float:
        key = ("ip", _canon(ip_combo), board, f"b{i}")
        bet = bet_sizes[i] * pot

        if mode == "train":
            strategy: list[float] | None = self._get_strategy(key, 2, "train")
        elif mode == "br_ip":
            strategy = None
        else:
            strategy = self._get_strategy(key, 2, "avg")

        fold_value = pot - oop_c
        next_reach_ip = reach_ip * strategy[1] if strategy is not None else reach_ip
        call_value = self._advance_or_showdown(
            oop_combo,
            ip_combo,
            board,
            deal_queue,
            bet_sizes,
            pot + 2 * bet,
            oop_c + bet,
            ip_c + bet,
            reach_oop,
            next_reach_ip,
            mode,
        )

        action_values = [fold_value, call_value]
        if mode == "br_ip":
            node_value = min(action_values)
        else:
            assert strategy is not None
            node_value = strategy[0] * fold_value + strategy[1] * call_value

        if mode == "train":
            assert strategy is not None
            regrets = self._regret_sum.setdefault(key, [0.0, 0.0])
            strat_sum = self._strategy_sum.setdefault(key, [0.0, 0.0])
            for idx in range(2):
                regrets[idx] += reach_oop * (node_value - action_values[idx])
                strat_sum[idx] += reach_ip * strategy[idx]

        return node_value

    def _facing_ip_bet_after_check(
        self,
        oop_combo: Combo,
        ip_combo: Combo,
        board: tuple[int, ...],
        deal_queue: list[int],
        bet_sizes: tuple[float, ...],
        i: int,
        pot: float,
        oop_c: float,
        ip_c: float,
        reach_oop: float,
        reach_ip: float,
        mode: str,
    ) -> float:
        key = ("oop", _canon(oop_combo), board, f"xb{i}")
        bet = bet_sizes[i] * pot

        if mode == "train":
            strategy: list[float] | None = self._get_strategy(key, 2, "train")
        elif mode == "br_oop":
            strategy = None
        else:
            strategy = self._get_strategy(key, 2, "avg")

        fold_value = -oop_c
        next_reach_oop = reach_oop * strategy[1] if strategy is not None else reach_oop
        call_value = self._advance_or_showdown(
            oop_combo,
            ip_combo,
            board,
            deal_queue,
            bet_sizes,
            pot + 2 * bet,
            oop_c + bet,
            ip_c + bet,
            next_reach_oop,
            reach_ip,
            mode,
        )

        action_values = [fold_value, call_value]
        if mode == "br_oop":
            node_value = max(action_values)
        else:
            assert strategy is not None
            node_value = strategy[0] * fold_value + strategy[1] * call_value

        if mode == "train":
            assert strategy is not None
            regrets = self._regret_sum.setdefault(key, [0.0, 0.0])
            strat_sum = self._strategy_sum.setdefault(key, [0.0, 0.0])
            for idx in range(2):
                regrets[idx] += reach_ip * (action_values[idx] - node_value)
                strat_sum[idx] += reach_oop * strategy[idx]

        return node_value

    def _advance_or_showdown(
        self,
        oop_combo: Combo,
        ip_combo: Combo,
        board: tuple[int, ...],
        deal_queue: list[int],
        bet_sizes: tuple[float, ...],
        pot: float,
        oop_c: float,
        ip_c: float,
        reach_oop: float,
        reach_ip: float,
        mode: str,
    ) -> float:
        if deal_queue:
            new_board = board + (deal_queue[0],)
            return self._root(
                oop_combo,
                ip_combo,
                new_board,
                deal_queue[1:],
                bet_sizes,
                pot,
                oop_c,
                ip_c,
                reach_oop,
                reach_ip,
                mode,
            )
        return self._showdown(oop_combo, ip_combo, board, pot, oop_c, ip_c)

    def _showdown(
        self,
        oop_combo: Combo,
        ip_combo: Combo,
        board: tuple[int, ...],
        pot: float,
        oop_c: float,
        ip_c: float,
    ) -> float:
        oop_rank = _EVAL.evaluate(list(board), list(oop_combo))
        ip_rank = _EVAL.evaluate(list(board), list(ip_combo))
        if oop_rank < ip_rank:  # treys: lower rank int = stronger hand
            return pot - oop_c
        if ip_rank < oop_rank:
            return -oop_c
        return pot / 2 - oop_c

    # ------------------------------------------------------------------
    # Post-training exact evaluation
    # ------------------------------------------------------------------

    def _evaluate(self) -> tuple[float, float]:
        assert self._pairs is not None and self._board_start is not None
        assert self._n_deal is not None and self._bet_sizes is not None and self._pot is not None

        full_deck = [c for c in _FULL_DECK if c not in set(self._board_start)]
        n_pairs = len(self._pairs)
        avg_ev = 0.0
        br_oop_ev = 0.0
        br_ip_ev = 0.0

        for oop_combo, ip_combo in self._pairs:
            pair_blocked = set(oop_combo) | set(ip_combo)
            pair_deck = [c for c in full_deck if c not in pair_blocked]
            deals = list(_enumerate_deals(pair_deck, self._n_deal))
            n_runouts = len(deals)
            pair_weight = 1.0 / n_pairs
            for deal in deals:
                w = pair_weight / n_runouts
                args = (
                    oop_combo,
                    ip_combo,
                    self._board_start,
                    list(deal),
                    self._bet_sizes,
                    self._pot,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                )
                avg_ev += w * self._root(*args, "avg")
                br_oop_ev += w * self._root(*args, "br_oop")
                br_ip_ev += w * self._root(*args, "br_ip")

        exploitability = ((br_oop_ev - avg_ev) + (avg_ev - br_ip_ev)) / 2.0
        return avg_ev, exploitability
