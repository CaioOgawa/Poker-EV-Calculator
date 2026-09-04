"""Tests for engine/icm/multiway.py (docs/AUDITORIA-2026-08-26.md item F6).

The discriminating check (docs/AUDITORIA-2026-08-26.md item F6, following the
advisor's own framing): with one villain already at 0 stack, `solve_3way`
must reduce EXACTLY (not approximately) to `ICMNash.solve_hu`, in both
directions (villain A dead, villain B dead) — this is what proves the "one
caller" branches are the same exact machinery as `solve_hu`, not a fit.
"""

import numpy as np
import pytest

from engine.icm.model import ICMModel
from engine.icm.nash import ICMNash, HAND_RANK
from engine.icm.multiway import (
    ICMNashMultiway,
    _LiveContext,
    _sample_threeway_outcomes,
    resolve_allin_showdown,
    resolve_equity_with_busts_settled,
)

pytestmark = pytest.mark.slow

MODEL = ICMModel()

# ------------------------------------------------------------------
# resolve_allin_showdown — side-pot arithmetic, no equity involved
# ------------------------------------------------------------------


def test_resolve_allin_showdown_conserves_chips():
    stacks = [1000, 3000, 6000]
    ranks = [1, 2, 3]  # A best, B second, C worst
    result = resolve_allin_showdown(stacks, ranks)
    assert sum(result) == pytest.approx(sum(stacks))


def test_resolve_allin_showdown_side_pot_layers_match_manual_calculation():
    # A(1000, best hand) can only win the layer everyone covers: 1000*3=3000.
    # B(3000, 2nd best) wins the next layer alone: (3000-1000)*2=4000.
    # C(6000, worst) wins the uncontested excess: (6000-3000)*1=3000.
    result = resolve_allin_showdown([1000, 3000, 6000], [1, 2, 3])
    assert result == [3000.0, 4000.0, 3000.0]


def test_resolve_allin_showdown_all_tied_still_respects_stack_eligibility():
    # A three-way tie doesn't mean an equal 3-way split of the whole pot --
    # each player can still only win layers their own stack reaches: A(1000)
    # only ever contests layer 1, capping them at 1000/3*3=1000; the extra
    # money the bigger stacks put in above 1000 was never something the
    # short stack could win a share of regardless of hand strength.
    result = resolve_allin_showdown([1000, 3000, 6000], [1, 1, 1])
    assert sum(result) == pytest.approx(10000.0)
    assert result == pytest.approx([1000.0, 3000.0, 6000.0])


def test_resolve_allin_showdown_shortest_stack_wins_only_its_own_layer():
    # Shortest stack has the best hand but can only win what everyone put in.
    result = resolve_allin_showdown([1000, 3000, 6000], [1, 3, 2])
    assert result[0] == pytest.approx(3000.0)  # covers all 3, wins layer 1
    assert sum(result) == pytest.approx(10000.0)


def test_resolve_allin_showdown_equal_stacks_is_plain_split():
    result = resolve_allin_showdown([1000, 1000, 1000], [1, 2, 3])
    assert result == [3000.0, 0.0, 0.0]


def test_resolve_allin_showdown_tie_splits_layer_evenly():
    result = resolve_allin_showdown([1000, 1000], [1, 1])
    assert result == [1000.0, 1000.0]


# ------------------------------------------------------------------
# resolve_equity_with_busts_settled — reduces exactly to ICMNash._all_in_equity
# ------------------------------------------------------------------


def test_busts_settled_matches_all_in_equity_villain_busts():
    nash = ICMNash(MODEL)
    payouts = [0.6, 0.4]
    expected = nash._all_in_equity(10000, 0, payouts)
    got = resolve_equity_with_busts_settled(MODEL, [10000, 0], payouts)
    assert got == pytest.approx(expected)


def test_busts_settled_matches_all_in_equity_hero_busts():
    nash = ICMNash(MODEL)
    payouts = [0.6, 0.4]
    expected = nash._all_in_equity(0, 10000, payouts)
    got = resolve_equity_with_busts_settled(MODEL, [0, 10000], payouts)
    assert got == pytest.approx(expected)


def test_busts_settled_no_one_busted_matches_plain_equity():
    payouts = [0.5, 0.3, 0.2]
    stacks = [3000, 4000, 5000]
    assert resolve_equity_with_busts_settled(MODEL, stacks, payouts) == pytest.approx(
        MODEL.equity(stacks, payouts)
    )


def test_busts_settled_simultaneous_busts_split_the_tail_evenly():
    # Hero has everything, both others are drawing dead: hero locks 1st,
    # the other two split (2nd+3rd) evenly -- a same-hand tie, not resolvable
    # further by this model.
    payouts = [0.5, 0.3, 0.2]
    result = resolve_equity_with_busts_settled(MODEL, [12000, 0, 0], payouts)
    assert result[0] == pytest.approx(0.5)
    assert result[1] == pytest.approx(0.25)
    assert result[2] == pytest.approx(0.25)
    assert sum(result) == pytest.approx(1.0)


# ------------------------------------------------------------------
# solve_3way degenerate case (one villain dead) must match solve_hu exactly
# ------------------------------------------------------------------

_DEGENERATE_CASES = [
    ([2000, 8000], [0.5, 0.3, 0.2]),
    ([6000, 4000], [0.6, 0.25, 0.15]),  # solve_hu's own documented cycling case
    ([1000, 3000], [0.5, 0.3, 0.2]),
    ([9000, 1000], [0.5, 0.3, 0.2]),
]


@pytest.mark.parametrize("stacks,payouts", _DEGENERATE_CASES)
def test_solve_3way_matches_solve_hu_when_b_is_dead(stacks, payouts):
    hu = ICMNash(MODEL).solve_hu(stacks, payouts[:2], sb=50, bb=100, max_iter=12)
    mw = ICMNashMultiway(MODEL).solve_3way(
        [stacks[0], stacks[1], 0], payouts, sb=50, bb=100, max_iter=12, mc_samples=2000, seed=0
    )
    assert mw.push_threshold == hu.push_threshold
    assert mw.call_threshold_a == hu.call_threshold
    assert mw.call_threshold_b_if_a_folded == -1
    assert mw.call_threshold_b_if_a_called == -1


@pytest.mark.parametrize("stacks,payouts", _DEGENERATE_CASES)
def test_solve_3way_matches_solve_hu_when_a_is_dead(stacks, payouts):
    # A dead, B is the sole live villain -- B must inherit the bb-poster
    # role (module docstring) for this to line up with solve_hu at all.
    hu = ICMNash(MODEL).solve_hu(stacks, payouts[:2], sb=50, bb=100, max_iter=12)
    mw = ICMNashMultiway(MODEL).solve_3way(
        [stacks[0], 0, stacks[1]], payouts, sb=50, bb=100, max_iter=12, mc_samples=2000, seed=0
    )
    assert mw.push_threshold == hu.push_threshold
    assert mw.call_threshold_b_if_a_folded == hu.call_threshold
    assert mw.call_threshold_a == -1
    assert mw.call_threshold_b_if_a_called == -1


# ------------------------------------------------------------------
# Genuine 3-way sanity (both villains live) — qualitative, not exact
# ------------------------------------------------------------------


def test_solve_3way_both_live_calling_range_tighter_when_both_call():
    # Calling a 3-way pot needs to beat two hands, not one -- the "both
    # already call" threshold should be strictly tighter than the "only me"
    # threshold, for equal-stack villains.
    result = ICMNashMultiway(MODEL).solve_3way(
        [2000, 4000, 4000], [0.5, 0.3, 0.2], sb=50, bb=100, max_iter=12, mc_samples=40_000, seed=0
    )
    assert result.call_threshold_b_if_a_called < result.call_threshold_b_if_a_folded
    assert result.push_threshold >= 0
    assert result.call_threshold_a >= 0


def test_solve_3way_short_stack_pushes_wide():
    result = ICMNashMultiway(MODEL).solve_3way(
        [1500, 5000, 5000], [0.5, 0.3, 0.2], sb=50, bb=100, max_iter=12, mc_samples=20_000, seed=0
    )
    # A ~15bb effective stack 3-handed should be well into push/fold territory.
    assert len(result.push_range) >= len(HAND_RANK) // 2


def test_solve_3way_rejects_wrong_player_count():
    with pytest.raises(ValueError):
        ICMNashMultiway(MODEL).solve_3way([1000, 2000], [0.5, 0.3, 0.2])


# ------------------------------------------------------------------
# The module's core premise: E[ICM(stacks)] != ICM(E[stacks]).
# If this ever came out equal, `resolve_allin_showdown`/`ctx.settle` would
# have collapsed into linearity and the whole design (discrete outcome +
# ICM, then average -- never blend shares first) would be untested.
# ------------------------------------------------------------------


def test_icm_equity_is_not_linear_in_stacks():
    payouts = [0.5, 0.3, 0.2]
    hero_wins_all = [10000, 0, 0]
    a_wins_all = [0, 10000, 0]
    mean_of_two_outcomes = [(a + b) / 2 for a, b in zip(hero_wins_all, a_wins_all)]

    icm_of_mean_stacks = MODEL.equity(mean_of_two_outcomes, payouts)[0]
    mean_of_icm_outcomes = (
        0.5 * MODEL.equity(hero_wins_all, payouts)[0] + 0.5 * MODEL.equity(a_wins_all, payouts)[0]
    )

    assert mean_of_icm_outcomes != pytest.approx(icm_of_mean_stacks, abs=1e-6)


# ------------------------------------------------------------------
# Regression: B's fold-EV baseline in the "A called" branch must not
# depend on B's own hand class (b_bystander_eq is purely hero-vs-A; see
# the advisor-flagged conditioning bug fixed in _solve_call_b_if_a_called).
# ------------------------------------------------------------------


def test_b_bystander_eq_pooled_baseline_is_stable_across_b_class_slices():
    stacks = [2000, 4000, 4000]
    payouts = [0.5, 0.3, 0.2]
    ctx = _LiveContext.build(True, True, payouts)
    samples = _sample_threeway_outcomes(stacks, ctx, MODEL, n_samples=60_000, seed=0)

    push_threshold, call_threshold_a = 90, 40
    base_mask = (samples.hero_class <= push_threshold) & (samples.a_class <= call_threshold_a)
    pooled = float(np.mean(samples.b_bystander_eq[np.nonzero(base_mask)[0]]))

    # Slice the same conditioning further by a few B classes: since
    # b_bystander_eq doesn't depend on B's cards, each slice's mean should
    # land close to the pooled estimate (small per-slice samples are noisy,
    # so this is a loose bound -- it would fail hard if slicing on B's class
    # introduced a systematic, not just noisy, difference).
    for b_idx in (0, 40, 80, 120):
        rows = np.nonzero(base_mask & (samples.b_class == b_idx))[0]
        if len(rows) < 5:
            continue
        slice_mean = float(np.mean(samples.b_bystander_eq[rows]))
        assert slice_mean == pytest.approx(pooled, abs=0.05)
