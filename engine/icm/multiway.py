"""3-way ICM Nash push/fold: hero shoves, two independent villains (A and B)
each decide call/fold in sequence (docs/AUDITORIA-2026-08-26.md item F6).

Why this needs new machinery beyond `nash.py`, not just "more players":
ICM is nonlinear in stacks, so a win *share* is not a usable input —
`E[ICM(stacks)] != ICM(E[stacks])`. `solve_hu` never blends equity shares
directly; it computes the *discrete showdown outcome* (hero wins the
effective stack, or loses it), applies ICM to the resulting stack vector for
each outcome, and only then averages. `EquityCalculator.multi_way()` returns
win-shares and discards the ordering of the losers — the wrong *shape* for
this (side pots need to know who finished 2nd, not just who won), not merely
too slow. Plugging its shares into `ICMModel` would reproduce E1's original
error (a plausible number from a structurally wrong showdown model) in a new
place.

Design:
- The "one villain calls, the other folds" branches reuse `ICMNash`'s own
  exact machinery unchanged (the 169x169 hand-vs-hand table, combo-weighted
  and card-removal-aware call frequencies via `_equity`/`_call_freq`) — no
  new approximation there. This is also what makes the degenerate-case test
  in the test suite meaningful: it isn't fit to match `solve_hu`, it's
  built from the same exact functions, so it matches by construction.
- The "both villains call" branch is genuinely new and MC-based: exact
  enumeration is infeasible for 3-way (a preflop 3-way all-in has up to
  C(46,5) ~= 1.37M runouts per hand triple, and there are ~169^3 ~= 4.8M
  distinct triples — no offline table saves you the way the pairwise table
  did for `solve_hu`). One joint Monte Carlo sample is drawn per
  `solve_3way()` call (fixed seed) and *reweighted* every fixed-point
  iteration by re-masking on the current thresholds, rather than resampled
  — resampling per iteration would reintroduce exactly the threshold
  instability that the precomputed table removed from the HU solver (I3).
- This is an honest regression, not a free upgrade: the 3-way branch carries
  real Monte Carlo sampling error that the HU solver no longer has. Fixing
  the seed makes one `solve_3way()` call reproducible; it does not make it
  exact. Measured directly (stacks=[2000,4000,4000], payouts=[.5,.3,.2],
  mc_samples=100_000, seeds 0-2): `push_threshold` and
  `call_threshold_b_if_a_called` vary by 1 rank across seeds,
  `call_threshold_a` by up to 3 ranks (out of 169) — small but real, and
  `call_threshold_b_if_a_folded` is exact (no MC in that branch) so it
  doesn't move at all. Treat any single-seed threshold from the "both call"
  branch as having a few ranks of uncertainty, not as exact.

Scope, deliberately: 3 players only. A general N-way solver needs
2^(N-1) villain response combinations and an N-dimensional fixed point per
hero hand — that doesn't survive much past N=3. Final-table (4+) push/fold
is not attempted here.

Strategy shape: like `solve_hu`, ranges are threshold sets (a contiguous
prefix of `HAND_RANK`). For heads-up this is a reasonable simplification;
for multiway it's a tractability *restriction*, not a proven-optimal
structure — a caller's presence correlates with their strength, which can
make true best responses non-monotonic in hand rank. Treat push/call ranges
here as "the best strategy in this restricted family", not as an
equilibrium in the unrestricted game.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from treys import Card, Deck, Evaluator

from engine.icm.model import ICMModel
from engine.icm.nash import HAND_RANK, ICMNash, _HAND_INDEX

_STR_RANKS = Card.STR_RANKS  # "23456789TJQKA", low to high — matches RANKS in range_parser


def resolve_allin_showdown(stacks: Sequence[float], ranks: Sequence[int]) -> list[float]:
    """Side-pot-resolved final stacks for players all-in against each other.

    stacks: each player's chips going into the pot (their full stack for a
    push/fold all-in) — differing stacks are the whole point, this is what
    makes 3-way correct where a flat "effective stack" (E1's HU shortcut)
    would not be. ranks: a treys-style evaluator rank per player, LOWER =
    stronger; ties split their shared layer evenly. Chips are conserved:
    sum(result) == sum(stacks).

    Standard side-pot algorithm: process players from shortest stack to
    longest. Each stack level closes a "layer" contested by every player
    whose stack still reaches it; that layer goes to the best hand(s) among
    that layer's contestants only — a player already exhausted at a lower
    level can't win chips from a layer they didn't reach.
    """
    n = len(stacks)
    order = sorted(range(n), key=lambda i: stacks[i])
    payouts = [0.0] * n
    remaining = set(range(n))
    prev_level = 0.0
    for i in order:
        level = stacks[i]
        if level > prev_level and remaining:
            layer = (level - prev_level) * len(remaining)
            best = min(ranks[j] for j in remaining)
            winners = [j for j in remaining if ranks[j] == best]
            share = layer / len(winners)
            for w in winners:
                payouts[w] += share
        remaining.discard(i)
        prev_level = level
    return payouts


def _win_share(rank_a: int, rank_b: int) -> float:
    """1.0/0.5/0.0 for a's share of a heads-up showdown (treys: lower rank wins)."""
    if rank_a < rank_b:
        return 1.0
    if rank_a == rank_b:
        return 0.5
    return 0.0


def _combo_to_canonical(c1: int, c2: int) -> str:
    """Two treys card ints -> canonical hand string ('AKs', '77', 'T9o'),
    matching `HAND_RANK`'s naming exactly."""
    r1, r2 = Card.get_rank_int(c1), Card.get_rank_int(c2)
    if r1 == r2:
        return _STR_RANKS[r1] * 2
    hi, lo = (r1, r2) if r1 > r2 else (r2, r1)
    suited = "s" if Card.get_suit_int(c1) == Card.get_suit_int(c2) else "o"
    return _STR_RANKS[hi] + _STR_RANKS[lo] + suited


def resolve_equity_with_busts_settled(
    model: ICMModel, stacks: Sequence[float], payouts: Sequence[float]
) -> list[float]:
    """Like `ICMModel.equity`, but a player at exactly 0 stack gets their
    locked-in finishing-position payout instead of the model's own 0.0 for
    an eliminated player.

    `ICMModel.equity_of` returns 0.0 for a 0-stack player by design — correct
    when 0 chips means "no further equity to project" for someone who busted
    *earlier* and already banked whatever they were owed elsewhere. It is
    the wrong number for *this* call site, which is evaluating the outcome
    of a specific showdown and needs the busted player's actual payout
    folded into the same result (this is exactly why `ICMNash._all_in_equity`
    special-cases `s_villain == 0` / `s_hero == 0` instead of calling
    `ICMModel.equity` directly). This generalizes that special case past
    exactly 2 players — proven to reduce to it exactly, checked in the test
    suite. Players who bust in the *same* resolution (e.g. hero holds the
    nuts and both villains are drawing dead in a 3-way pot) are treated as
    tied for their shared worst finishing place(s), splitting that payout
    tail evenly — a same-hand simultaneous elimination is a genuine tie,
    not something this model can resolve further.

    IMPORTANT: `stacks`/`payouts` here must describe only players who are
    *live going into this specific showdown* — a player already eliminated
    in some earlier hand is not a "simultaneous" bust with anyone in this
    one and must not be passed in at all (see `_LiveContext`, which is what
    every caller in this module actually uses). Passed a stale pre-existing
    0 alongside a stack that newly reaches 0 in the same call, this function
    has no way to tell the two apart and would incorrectly tie them.
    """
    n = len(stacks)
    busted = [i for i in range(n) if stacks[i] <= 0]
    if not busted:
        return model.equity(list(stacks), list(payouts))

    n_active = n - len(busted)
    tail = list(payouts[n_active:]) if n_active < len(payouts) else []
    busted_payout = sum(tail) / len(busted) if tail else 0.0

    result = [0.0] * n
    for i in busted:
        result[i] = busted_payout

    if n_active > 0:
        active_indices = [i for i in range(n) if i not in busted]
        active_stacks = [stacks[i] for i in active_indices]
        active_eq = model.equity(active_stacks, list(payouts[:n_active]))
        for idx, i in enumerate(active_indices):
            result[i] = active_eq[idx]
    return result


@dataclass
class _LiveContext:
    """Which of (hero=0, A=1, B=2) are live players in this ICM prize
    picture, and the payout slots that implies for the players who are.

    A villain fixed at 0 stack from the *start* — already eliminated before
    this hand — is not "newly busting" the way a player who loses this
    hand's all-in is; they locked in their finishing payout in some earlier
    hand already. This context excludes them from every stack/rank
    computation and just returns their fixed payout back, so the rest of
    the module never has to special-case "a stack that happens to be 0"
    versus "a stack that just became 0" — that distinction is exactly what
    `resolve_equity_with_busts_settled` can't make on its own (see its
    docstring), so it's resolved once here instead.
    """

    live: tuple[int, ...]
    live_payouts: list[float]
    dead_payout: dict[int, float]

    @classmethod
    def build(cls, a_live: bool, b_live: bool, payouts: list[float]) -> "_LiveContext":
        live = (0,) + ((1,) if a_live else ()) + ((2,) if b_live else ())
        dead = [i for i in (1, 2) if i not in live]
        n_live = len(live)
        dead_payout = {i: payouts[n_live + k] for k, i in enumerate(dead)}
        return cls(live=live, live_payouts=list(payouts[:n_live]), dead_payout=dead_payout)

    def settle(self, model: ICMModel, stacks: Sequence[float]) -> list[float]:
        """Full [hero, A, B] equity for `stacks` (only the live entries of
        which are meaningful — a dead player's stack value is ignored)."""
        live_stacks = [stacks[i] for i in self.live]
        live_eq = resolve_equity_with_busts_settled(model, live_stacks, self.live_payouts)
        result = [0.0, 0.0, 0.0]
        for idx, i in enumerate(self.live):
            result[i] = live_eq[idx]
        for i, p in self.dead_payout.items():
            result[i] = p
        return result


@dataclass
class _ThreeWaySamples:
    """One joint Monte Carlo sample per row: a hero/A/B hole-card deal plus a
    board. Bucketed by canonical hand class per player so the fixed-point
    loop can re-mask by the *current* call thresholds every iteration
    without re-sampling. Two showdown resolutions are precomputed per row:

    - `*_eq3`: all three players all-in (the "both villains call" branch).
    - `b_bystander_eq`: only hero and A go to showdown (at their effective
      stack), B's own stack untouched — what B gets by folding *after A has
      already called*, needed because B's fold EV in that branch isn't a
      fixed baseline the way `solve_hu`'s is: it depends on the same hero/A
      matchup B would be joining if B called instead.
    """

    hero_class: np.ndarray
    a_class: np.ndarray
    b_class: np.ndarray
    hero_eq3: np.ndarray
    a_eq3: np.ndarray
    b_eq3: np.ndarray
    b_bystander_eq: np.ndarray


def _sample_threeway_outcomes(
    stacks: tuple[float, float, float],
    ctx: "_LiveContext",
    model: ICMModel,
    n_samples: int,
    seed: int,
) -> _ThreeWaySamples:
    if len(ctx.live) < 3:
        # A villain who can never call contributes no "all three call"
        # branch at all — that weight is always 0 in the fixed-point loop,
        # so there's nothing for this sample set to be used for.
        empty_i = np.empty(0, dtype=np.int16)
        empty_f = np.empty(0, dtype=np.float64)
        return _ThreeWaySamples(empty_i, empty_i, empty_i, empty_f, empty_f, empty_f, empty_f)

    rng = random.Random(seed)
    evaluator = Evaluator()
    full_deck = list(Deck.GetFullDeck())
    hero_stack, a_stack, b_stack = stacks
    eff_hero_a = min(hero_stack, a_stack)

    hero_class = np.empty(n_samples, dtype=np.int16)
    a_class = np.empty(n_samples, dtype=np.int16)
    b_class = np.empty(n_samples, dtype=np.int16)
    hero_eq3 = np.empty(n_samples, dtype=np.float64)
    a_eq3 = np.empty(n_samples, dtype=np.float64)
    b_eq3 = np.empty(n_samples, dtype=np.float64)
    b_bystander_eq = np.empty(n_samples, dtype=np.float64)

    deck = full_deck[:]
    for i in range(n_samples):
        rng.shuffle(deck)
        hero_hole = deck[0:2]
        a_hole = deck[2:4]
        b_hole = deck[4:6]
        board = deck[6:11]

        hero_rank = evaluator.evaluate(board, hero_hole)
        a_rank = evaluator.evaluate(board, a_hole)
        b_rank = evaluator.evaluate(board, b_hole)

        new_stacks = resolve_allin_showdown(list(stacks), [hero_rank, a_rank, b_rank])
        eq = ctx.settle(model, new_stacks)

        hero_share = _win_share(hero_rank, a_rank)
        hero_delta = eff_hero_a * (2.0 * hero_share - 1.0)
        bystander_stacks = [hero_stack + hero_delta, a_stack - hero_delta, b_stack]
        bystander_eq = ctx.settle(model, bystander_stacks)

        hero_class[i] = _HAND_INDEX[_combo_to_canonical(*hero_hole)]
        a_class[i] = _HAND_INDEX[_combo_to_canonical(*a_hole)]
        b_class[i] = _HAND_INDEX[_combo_to_canonical(*b_hole)]
        hero_eq3[i] = eq[0]
        a_eq3[i] = eq[1]
        b_eq3[i] = eq[2]
        b_bystander_eq[i] = bystander_eq[2]

    return _ThreeWaySamples(hero_class, a_class, b_class, hero_eq3, a_eq3, b_eq3, b_bystander_eq)


def _embedded_hu_equity(
    model: ICMModel,
    stacks: list[float],
    ctx: "_LiveContext",
    i: int,
    j: int,
    i_wins: bool,
) -> list[float]:
    """ICM equity for all players after a HU all-in between stacks[i] and
    stacks[j] at their effective stack (E1's convention), every other
    player's stack untouched."""
    eff = min(stacks[i], stacks[j])
    new_stacks = list(stacks)
    if i_wins:
        new_stacks[i] += eff
        new_stacks[j] -= eff
    else:
        new_stacks[i] -= eff
        new_stacks[j] += eff
    return ctx.settle(model, new_stacks)


@dataclass
class ThreeWayNashResult:
    push_range: list[str]
    call_range_a: list[str]
    call_range_b_if_a_folded: list[str]
    call_range_b_if_a_called: list[str]
    push_threshold: int
    call_threshold_a: int
    call_threshold_b_if_a_folded: int
    call_threshold_b_if_a_called: int


class ICMNashMultiway:
    """3-way ICM Nash push/fold solver. See module docstring for scope and
    what's exact vs. Monte-Carlo-approximate."""

    def __init__(self, model: ICMModel | None = None):
        self._model = model or ICMModel()
        self._nash = ICMNash(self._model)

    def solve_3way(
        self,
        stacks: list[float],
        payouts: list[float],
        sb: int = 50,
        bb: int = 100,
        max_iter: int = 12,
        mc_samples: int = 100_000,
        seed: int = 0,
    ) -> ThreeWayNashResult:
        """Find push/call ranges for hero shoving 3-handed against two
        independent villains. A acts first and does not observe B. B acts
        second and *does* observe whether A called or folded — so B is
        modeled with two separate thresholds, not one: `call_threshold_b_
        if_a_folded` (a pure heads-up decision vs. hero, exact) and
        `call_threshold_b_if_a_called` (a genuine 3-way decision, Monte
        Carlo — see module docstring).

        stacks: [hero, villain_a, villain_b] in chips.
        payouts: prize fractions, e.g. [0.5, 0.3, 0.2].
        sb/bb: A is modeled as the big blind — already committed `bb`,
        forfeited to hero if A folds, exactly `solve_hu`'s villain
        convention. B is a third player who hasn't committed anything
        toward this pot, so B folding transfers nothing (a simplification —
        real 3-handed blind structures vary; this is the one `solve_hu`'s
        own 2-player convention generalizes to most directly). Hero risks
        `sb` folding, matching `solve_hu`.
        mc_samples/seed: size and seed of the one joint Monte Carlo sample
        used for the "both villains call" branch and for B's decision given
        A called (see module docstring — this is the source of
        approximation here; everything else is exact).

        A villain whose stack is exactly 0 is treated as already eliminated
        — locked to folding forever, never sampled or iterated on. If A is
        the one dead, B transparently inherits the bb-poster role (labels
        are swapped internally and swapped back on the result) so hero
        still collects the bb on an uncalled push regardless of which slot
        the one live villain happens to occupy. This is also how the test
        suite checks correctness: with either villain dead, every branch
        touching the other one has zero weight and every remaining
        computation routes through the same `_equity`/`_call_freq` calls
        `solve_hu` itself uses, so the two solvers must agree exactly.
        """
        if len(stacks) != 3:
            raise ValueError(f"solve_3way requires exactly 3 players, got {len(stacks)}")
        hero_stack, a_stack, b_stack = stacks

        # A is the modeled bb-poster (see docstring). If A is dead and B is
        # the sole live villain, B needs to inherit that role — otherwise
        # hero would never collect the bb from anyone on an uncalled push,
        # which would disagree with `solve_hu`'s own single-villain
        # convention for no reason but which internal slot the live villain
        # happens to occupy. Swap labels for the solve, swap the result back
        # at the end.
        swapped = a_stack <= 0 and b_stack > 0
        if swapped:
            a_stack, b_stack = b_stack, a_stack

        a_live = a_stack > 0
        b_live = b_stack > 0
        ctx = _LiveContext.build(a_live, b_live, payouts)

        samples = _sample_threeway_outcomes(
            (hero_stack, a_stack, b_stack), ctx, self._model, mc_samples, seed
        )

        push_threshold = len(HAND_RANK) - 1  # start: hero shoves everything
        call_threshold_a = int(len(HAND_RANK) * 0.30) - 1 if a_live else -1
        call_threshold_b_fold = int(len(HAND_RANK) * 0.30) - 1 if b_live else -1
        call_threshold_b_call = int(len(HAND_RANK) * 0.20) - 1 if b_live else -1
        history: list[tuple[int, int, int, int]] = []

        for _ in range(max_iter):
            push_threshold = self._solve_push(
                hero_stack,
                a_stack,
                b_stack,
                ctx,
                sb,
                bb,
                call_threshold_a,
                call_threshold_b_fold,
                call_threshold_b_call,
                samples,
            )
            if a_live:
                call_threshold_a = self._solve_call_a(
                    hero_stack,
                    a_stack,
                    b_stack,
                    ctx,
                    bb,
                    push_threshold,
                    call_threshold_b_call,
                    samples,
                )
            if b_live:
                call_threshold_b_fold = self._solve_call_b_if_a_folded(
                    hero_stack, a_stack, b_stack, ctx, bb, push_threshold
                )
                call_threshold_b_call = self._solve_call_b_if_a_called(
                    push_threshold, call_threshold_a, samples
                )

            state = (push_threshold, call_threshold_a, call_threshold_b_fold, call_threshold_b_call)
            if state in history:
                cycle = history[history.index(state) :]
                push_threshold = round(sum(s[0] for s in cycle) / len(cycle))
                call_threshold_a = round(sum(s[1] for s in cycle) / len(cycle))
                call_threshold_b_fold = round(sum(s[2] for s in cycle) / len(cycle))
                call_threshold_b_call = round(sum(s[3] for s in cycle) / len(cycle))
                break
            history.append(state)

        def _range(threshold: int) -> list[str]:
            return HAND_RANK[: threshold + 1] if threshold >= 0 else []

        if swapped:
            # Original A is the dead one (that's what triggered the swap):
            # it never had a decision. Original B's decision is whatever the
            # internal "A" slot (the live villain) computed — a single
            # unconditional threshold, since original A never calls for
            # original B to have observed.
            call_threshold_a, call_threshold_b_fold, call_threshold_b_call = (
                -1,
                call_threshold_a,
                -1,
            )

        return ThreeWayNashResult(
            push_range=_range(push_threshold),
            call_range_a=_range(call_threshold_a),
            call_range_b_if_a_folded=_range(call_threshold_b_fold),
            call_range_b_if_a_called=_range(call_threshold_b_call),
            push_threshold=push_threshold,
            call_threshold_a=call_threshold_a,
            call_threshold_b_if_a_folded=call_threshold_b_fold,
            call_threshold_b_if_a_called=call_threshold_b_call,
        )

    # ------------------------------------------------------------------
    # Per-hand EV, one player at a time
    # ------------------------------------------------------------------

    def _both_call_bucket(
        self, samples: _ThreeWaySamples, bucket: str, class_idx: int, other_max: dict[str, int]
    ) -> np.ndarray:
        """Row indices of `samples` matching `bucket`'s class == class_idx
        and every other named player's class within `other_max`'s threshold
        (a threshold of -1 matches nothing, correctly modeling "never calls").
        """
        classes = {"hero": samples.hero_class, "a": samples.a_class, "b": samples.b_class}
        mask = classes[bucket] == class_idx
        for who, thresh in other_max.items():
            mask &= classes[who] <= thresh
        return np.nonzero(mask)[0]

    def _solve_push(
        self,
        hero_stack,
        a_stack,
        b_stack,
        ctx,
        sb,
        bb,
        call_threshold_a,
        call_threshold_b_fold,
        call_threshold_b_call,
        samples,
    ) -> int:
        # A is the big blind (forfeits `bb` to hero on fold, matching
        # `solve_hu`'s villain convention exactly). B has posted nothing, so
        # B folding transfers nothing. A calling, or both calling, put each
        # player's *entire* stack at risk (the posted bb is folded into
        # that, same as `solve_hu` never separately deducts it once a call
        # happens) — the bb transfer below applies only where A folds.
        stacks = [hero_stack, a_stack, b_stack]
        stacks_a_folds = [hero_stack + bb, max(0, a_stack - bb), b_stack]
        ev_fold = ctx.settle(self._model, [hero_stack - sb, a_stack + sb, b_stack])[0]
        ev_uncalled = ctx.settle(self._model, stacks_a_folds)[0]
        call_range_a = HAND_RANK[: call_threshold_a + 1] if call_threshold_a >= 0 else []
        call_range_b_fold = (
            HAND_RANK[: call_threshold_b_fold + 1] if call_threshold_b_fold >= 0 else []
        )
        call_range_b_call = (
            HAND_RANK[: call_threshold_b_call + 1] if call_threshold_b_call >= 0 else []
        )
        # Showdown resolution depends only on stacks/ctx, not on `hand` --
        # hoisted out of the per-hand loop (I4 precedent).
        win_only_b = _embedded_hu_equity(self._model, stacks_a_folds, ctx, 0, 2, True)[0]
        lose_only_b = _embedded_hu_equity(self._model, stacks_a_folds, ctx, 0, 2, False)[0]
        win_only_a = _embedded_hu_equity(self._model, stacks, ctx, 0, 1, True)[0]
        lose_only_a = _embedded_hu_equity(self._model, stacks, ctx, 0, 1, False)[0]

        new_push = -1
        for idx, hand in enumerate(HAND_RANK):
            call_freq_a = self._nash._call_freq(hand, call_range_a) if call_range_a else 0.0
            call_freq_b_fold = (
                self._nash._call_freq(hand, call_range_b_fold) if call_range_b_fold else 0.0
            )
            call_freq_b_call = (
                self._nash._call_freq(hand, call_range_b_call) if call_range_b_call else 0.0
            )

            ev_uncalled_branch = ev_uncalled
            ev_only_b = 0.0
            if call_freq_b_fold > 0:
                eq_h_vs_b = self._nash._equity(hand, ",".join(call_range_b_fold))
                ev_only_b = eq_h_vs_b * win_only_b + (1.0 - eq_h_vs_b) * lose_only_b

            ev_only_a = 0.0
            if call_freq_a > 0:
                eq_h_vs_a = self._nash._equity(hand, ",".join(call_range_a))
                ev_only_a = eq_h_vs_a * win_only_a + (1.0 - eq_h_vs_a) * lose_only_a

            ev_both = 0.0
            if call_freq_a > 0 and call_freq_b_call > 0:
                rows = self._both_call_bucket(
                    samples, "hero", idx, {"a": call_threshold_a, "b": call_threshold_b_call}
                )
                if len(rows) > 0:
                    ev_both = float(np.mean(samples.hero_eq3[rows]))

            ev_push = (
                (1.0 - call_freq_a) * (1.0 - call_freq_b_fold) * ev_uncalled_branch
                + (1.0 - call_freq_a) * call_freq_b_fold * ev_only_b
                + call_freq_a * (1.0 - call_freq_b_call) * ev_only_a
                + call_freq_a * call_freq_b_call * ev_both
            )
            if ev_push >= ev_fold:
                new_push = idx
            else:
                break
        return new_push

    def _solve_call_a(
        self, hero_stack, a_stack, b_stack, ctx, bb, push_threshold, call_threshold_b_call, samples
    ) -> int:
        """A doesn't observe B, but reasons about what B will do *if A
        calls* — B's "after A called" threshold (a fixed point on
        `call_threshold_b_call`, updated alongside this every outer
        iteration). A is the big blind (`solve_hu` convention): folding
        forfeits the already-committed `bb` to hero."""
        stacks = [hero_stack, a_stack, b_stack]
        push_range = HAND_RANK[: push_threshold + 1] if push_threshold >= 0 else []
        if not push_range:
            return -1
        push_range_str = ",".join(push_range)
        call_range_b_call = (
            HAND_RANK[: call_threshold_b_call + 1] if call_threshold_b_call >= 0 else []
        )
        ev_fold = ctx.settle(self._model, [hero_stack + bb, max(0, a_stack - bb), b_stack])[1]
        win_hu = _embedded_hu_equity(self._model, stacks, ctx, 1, 0, True)[1]
        lose_hu = _embedded_hu_equity(self._model, stacks, ctx, 1, 0, False)[1]

        new_call = -1
        for idx, hand in enumerate(HAND_RANK):
            call_freq_b = (
                self._nash._call_freq(hand, call_range_b_call) if call_range_b_call else 0.0
            )

            eq_a_vs_hero = self._nash._equity(hand, push_range_str)
            ev_hu = eq_a_vs_hero * win_hu + (1.0 - eq_a_vs_hero) * lose_hu

            ev_3way = 0.0
            if call_freq_b > 0:
                rows = self._both_call_bucket(
                    samples, "a", idx, {"hero": push_threshold, "b": call_threshold_b_call}
                )
                if len(rows) > 0:
                    ev_3way = float(np.mean(samples.a_eq3[rows]))

            ev_call = (1.0 - call_freq_b) * ev_hu + call_freq_b * ev_3way
            if ev_call >= ev_fold:
                new_call = idx
            else:
                break
        return new_call

    def _solve_call_b_if_a_folded(
        self, hero_stack, a_stack, b_stack, ctx, bb, push_threshold
    ) -> int:
        """Pure heads-up decision: B vs. hero's range, A already folded (so
        A's posted bb has already transferred to hero) — exact, no MC."""
        stacks_a_folded = [hero_stack + bb, max(0, a_stack - bb), b_stack]
        push_range = HAND_RANK[: push_threshold + 1] if push_threshold >= 0 else []
        if not push_range:
            return -1
        push_range_str = ",".join(push_range)
        ev_fold = ctx.settle(self._model, stacks_a_folded)[2]
        win_hu = _embedded_hu_equity(self._model, stacks_a_folded, ctx, 2, 0, True)[2]
        lose_hu = _embedded_hu_equity(self._model, stacks_a_folded, ctx, 2, 0, False)[2]

        new_call = -1
        for idx, hand in enumerate(HAND_RANK):
            eq_b_vs_hero = self._nash._equity(hand, push_range_str)
            ev_call = eq_b_vs_hero * win_hu + (1.0 - eq_b_vs_hero) * lose_hu
            if ev_call >= ev_fold:
                new_call = idx
            else:
                break
        return new_call

    def _solve_call_b_if_a_called(
        self, push_threshold: int, call_threshold_a: int, samples: _ThreeWaySamples
    ) -> int:
        """Genuine 3-way decision: B vs. hero's range AND A's actual calling
        hand — Monte Carlo, no exact shortcut available (module docstring).

        B's fold EV here is not `solve_hu`'s kind of fixed baseline: it's
        "hero and A go to showdown without me, my own stack untouched" —
        `b_bystander_eq`, sampled jointly with the same hero/A/B deal as the
        call branch so both sides of the comparison share the same
        conditioning (hero in `push_range`, A in `call_range_a`), not two
        independently-noisy MC estimates.
        """
        push_range = HAND_RANK[: push_threshold + 1] if push_threshold >= 0 else []
        call_range_a = HAND_RANK[: call_threshold_a + 1] if call_threshold_a >= 0 else []
        if not push_range or not call_range_a:
            return -1

        # b_bystander_eq depends only on hero-vs-A (B's own stack is
        # untouched), so it is mathematically constant across B's hand class
        # for a fixed (hero, A) conditioning. Estimating it once here, over
        # every B-class row that shares that conditioning, avoids splitting
        # an already-noisy MC quantity 169 further ways for no reason -- a
        # per-idx re-slice on `b_class` would just be resampling noise on a
        # quantity idx doesn't affect.
        baseline_rows = np.nonzero(
            (samples.hero_class <= push_threshold) & (samples.a_class <= call_threshold_a)
        )[0]
        if len(baseline_rows) == 0:
            return -1
        ev_fold = float(np.mean(samples.b_bystander_eq[baseline_rows]))

        # NOTE: an empty bucket at idx `continue`s rather than `break`s
        # (unlike the other three solve methods), since low sample counts
        # are common per-B-class and a gap here shouldn't truncate the
        # range early. This means a weaker hand past an empty bucket can
        # still set `new_call` if a stronger hand's bucket was empty --
        # accepted as a lesser evil than a spuriously short call range.
        new_call = -1
        for idx in range(len(HAND_RANK)):
            rows = self._both_call_bucket(
                samples, "b", idx, {"hero": push_threshold, "a": call_threshold_a}
            )
            if len(rows) == 0:
                continue
            ev_call = float(np.mean(samples.b_eq3[rows]))
            if ev_call >= ev_fold:
                new_call = idx
            else:
                break
        return new_call
