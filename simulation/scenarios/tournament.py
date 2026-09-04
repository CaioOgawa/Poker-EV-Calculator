"""Push/fold MTT simulator with ICM-aware calling decisions."""

from __future__ import annotations

import random
from dataclasses import dataclass

from engine.icm.model import ICMModel
from engine.icm.pressure import ICMPressure


@dataclass
class TournamentResult:
    finish_positions: list[int]  # finish position per trial (1 = winner)
    itm_rate: float  # fraction of trials that finished in the money
    avg_finish: float
    roi: float  # (avg_prize / buy_in) - 1; 0-sum ≈ 0


def _round_stack(s: int, granularity: int = 100) -> int:
    return max(granularity, round(s / granularity) * granularity)


class TournamentSim:
    """Push-or-fold MTT simulator.

    Each hand a rotating big blind posts chips (guaranteeing chip drainage and
    eventual termination). The short stack shoves once their stack-to-BB ratio
    drops below push_threshold_bbs. Calling decisions use ICM bubble factor
    when ≤ 4 players remain; otherwise chip-EV pot odds.

    Parameters
    ----------
    num_players : starting table size
    starting_stack : chip count per player at start
    starting_bb : initial big blind size
    paid_spots : how many places pay a prize
    push_threshold_bbs : stack-to-BB ratio below which the short stack shoves
    caller_equity : assumed equity of the caller when called
    num_trials : Monte Carlo iterations
    seed : RNG seed for reproducibility
    """

    def __init__(
        self,
        num_players: int = 9,
        starting_stack: int = 10_000,
        starting_bb: int = 100,
        paid_spots: int = 3,
        push_threshold_bbs: float = 15.0,
        caller_equity: float = 0.60,
        num_trials: int = 1_000,
        seed: int | None = 42,
    ):
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.starting_bb = starting_bb
        self.paid_spots = paid_spots
        self.push_threshold_bbs = push_threshold_bbs
        self.caller_equity = caller_equity
        self.num_trials = num_trials
        self._rng = random.Random(seed)
        self._icm = ICMModel()
        self._pressure = ICMPressure(self._icm)
        # Threshold scaled to hundredths and kept as an int (Python ints are
        # arbitrary-precision, so this stays exact for `bb` of any size) —
        # `int(push_threshold_bbs)` used to truncate 12.5 to 12, so a 12.9bb
        # stack (above the 12.5 threshold) wrongly qualified as push/fold
        # territory (docs/AUDITORIA-2026-08-26.md item E8).
        self._threshold_scaled = round(push_threshold_bbs * 100)
        self._bf_cache: dict[tuple[tuple[int, ...], tuple[float, ...], int, int], float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, payouts: list[float] | None = None) -> TournamentResult:
        """Run num_trials tournaments and return aggregate statistics.

        payouts: prize fractions in finishing order (length == num_players).
                 Defaults to a flat equal-split structure for paid_spots places.
        """
        if payouts is None:
            payouts = self._default_payouts()

        finish_positions: list[int] = []
        prizes: list[float] = []

        for _ in range(self.num_trials):
            pos, prize = self._run_one(payouts)
            finish_positions.append(pos)
            prizes.append(prize)

        n = len(finish_positions)
        itm = sum(1 for p in prizes if p > 0) / n
        avg_finish = sum(finish_positions) / n
        avg_prize = sum(prizes) / n
        roi = avg_prize - 1.0

        return TournamentResult(
            finish_positions=finish_positions,
            itm_rate=itm,
            avg_finish=avg_finish,
            roi=roi,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _default_payouts(self) -> list[float]:
        n = self.paid_spots
        total = self.num_players
        payouts = [0.0] * total
        share = 1.0 / n
        for i in range(n):
            payouts[i] = share
        return payouts

    @staticmethod
    def _next_seat(current: int, alive: list[int]) -> int:
        """Return the next alive seat after `current` in circular seat order.

        Operates on seat identities, not list position, so it stays correct
        no matter how `alive` has been mutated by prior remove() calls.
        """
        for s in sorted(alive):
            if s > current:
                return s
        return min(alive)

    def _run_one(self, payouts: list[float]) -> tuple[int, float]:
        """Simulate one tournament. Returns (finish_position, prize_fraction)."""
        stacks = [self.starting_stack] * self.num_players
        bb = self.starting_bb
        alive = list(range(self.num_players))
        finish_order: list[int] = []

        hand = 0
        # Randomise starting BB seat so no seat has a systematic advantage.
        # bb_seat tracks a player identity (not a list index), because `alive`
        # shrinks via remove() as players bust — an index-based pointer would
        # silently skip or repeat a seat whenever a removal shifted the list.
        bb_seat = self._rng.choice(alive)

        while len(alive) > 1:
            hand += 1
            # Blind level-up: pure integer arithmetic to avoid float overflow
            if hand % 15 == 0:
                bb = bb + bb // 2

            # Post big blind — ensures chips drain every hand (termination guarantee)
            blind = min(bb, stacks[bb_seat])
            stacks[bb_seat] -= blind

            if stacks[bb_seat] == 0:
                alive.remove(bb_seat)
                finish_order.append(bb_seat)
                if len(alive) <= 1:
                    break
                bb_seat = self._next_seat(bb_seat, alive)
                continue

            if len(alive) <= 1:
                break

            # Pick the short stack with random tie-breaking to avoid seat-0 bias
            min_stack = min(stacks[s] for s in alive)
            candidates = [s for s in alive if stacks[s] == min_stack]
            pusher_seat = self._rng.choice(candidates)
            push_size = stacks[pusher_seat]

            # Integer comparison, exact to hundredths of a bb: avoids both a
            # float * huge_int precision loss and the truncated-threshold bug
            # the old `push_size // bb > int(threshold)` had.
            if bb > 0 and push_size * 100 > bb * self._threshold_scaled:
                # Not in push/fold territory yet; blinds keep draining stacks,
                # but the button still advances to the next hand.
                bb_seat = self._next_seat(bb_seat, alive)
                continue

            caller_seat = self._find_caller(
                alive, pusher_seat, stacks, payouts, bb, blind, push_size
            )

            if caller_seat is None:
                # Steal: pusher wins the dead blind
                stacks[pusher_seat] += blind
            else:
                eq = self.caller_equity
                if self._rng.random() < eq:
                    # caller wins
                    stacks[caller_seat] += push_size + blind
                    stacks[pusher_seat] = 0
                    alive.remove(pusher_seat)
                    finish_order.append(pusher_seat)
                else:
                    # pusher wins
                    stacks[pusher_seat] += push_size + blind
                    stacks[caller_seat] = max(0, stacks[caller_seat] - push_size)
                    if stacks[caller_seat] == 0:
                        alive.remove(caller_seat)
                        finish_order.append(caller_seat)

            if len(alive) <= 1:
                break
            bb_seat = self._next_seat(bb_seat, alive)

        # Last player standing wins
        winner_seat = alive[0]
        finish_order.append(winner_seat)

        total = self.num_players
        position_of = {}
        for rank, seat in enumerate(reversed(finish_order)):
            position_of[seat] = rank + 1

        hero_pos = position_of.get(0, total)
        prize = payouts[hero_pos - 1] * total if hero_pos <= len(payouts) else 0.0

        return hero_pos, prize

    def _find_caller(
        self,
        alive: list[int],
        pusher_seat: int,
        stacks: list[int],
        payouts: list[float],
        bb: int,
        blind: int,
        push_size: int,
    ) -> int | None:
        """Return the seat of the first player who would call, or None."""
        for seat in alive:
            if seat == pusher_seat:
                continue
            if stacks[seat] < push_size:
                continue  # can't cover (simplified — ignores side pots)
            if self._should_call(alive, seat, pusher_seat, stacks, payouts, bb, blind, push_size):
                return seat
        return None

    def _should_call(
        self,
        alive: list[int],
        caller_seat: int,
        pusher_seat: int,
        stacks: list[int],
        payouts: list[float],
        bb: int,
        blind: int,
        push_size: int,
    ) -> bool:
        """Decide whether caller_seat should call the shove from pusher_seat."""
        n = len(alive)

        if n <= 4:
            # ICM-aware: caller calls only if equity exceeds ICM-adjusted threshold
            rounded = tuple(_round_stack(stacks[s]) for s in alive)
            caller_idx = alive.index(caller_seat)
            pusher_idx = alive.index(pusher_seat)
            alive_payouts = tuple(payouts[:n])

            bf = self._bubble_factor_cached(rounded, alive_payouts, caller_idx, pusher_idx)
            required_eq = bf / (1.0 + bf)
            return self.caller_equity > required_eq
        else:
            # Chip-EV pot odds: required_eq = call / (call + gain_if_win)
            # Caller risks push_size; stands to gain push_size + blind
            gain = push_size + blind
            required_eq = push_size / (push_size + gain)
            return self.caller_equity > required_eq

    def _bubble_factor_cached(
        self,
        rounded_stacks: tuple[int, ...],
        alive_payouts: tuple[float, ...],
        caller_idx: int,
        pusher_idx: int,
    ) -> float:
        key = (rounded_stacks, alive_payouts, caller_idx, pusher_idx)
        cached = self._bf_cache.get(key)
        if cached is not None:
            return cached
        result = self._pressure.bubble_factor(
            list(rounded_stacks), list(alive_payouts), caller_idx, pusher_idx
        )
        self._bf_cache[key] = result
        return result
