"""Tests for the precomputed hand-vs-hand equity table (audit item I3).

Not marked `slow`: these only load the committed .npy and do array
arithmetic — no Monte Carlo runs in this module.
"""

import time

import numpy as np
import pytest

from engine.icm.nash import HAND_RANK, ICMNash, _HAND_INDEX, _load_equity_table


@pytest.fixture(scope="module")
def table() -> np.ndarray:
    return _load_equity_table()


def test_table_shape_matches_hand_rank(table):
    assert table.shape == (len(HAND_RANK), len(HAND_RANK))


def test_table_values_in_unit_range(table):
    assert table.min() >= 0.0
    assert table.max() <= 1.0


def test_table_is_complementary_across_the_diagonal(table):
    # range_vs_range always returns (eq_a, 1 - eq_a), so this holds exactly,
    # not just approximately.
    n = table.shape[0]
    assert np.allclose(table + table.T, np.ones((n, n)), atol=1e-9)


def test_table_diagonal_is_close_to_half(table):
    # Same canonical hand on both sides is a mirror matchup — true equity is
    # exactly 0.5; the stored value carries only Monte Carlo sampling noise
    # around it (generated at high iteration count, so noise is small).
    diag = np.diag(table)
    assert np.allclose(diag, 0.5, atol=0.03)


def test_table_strong_hand_dominates_weak_hand(table):
    aa = _HAND_INDEX["AA"]
    weak = _HAND_INDEX["72o"]
    assert table[aa, weak] > 0.80
    assert table[weak, aa] < 0.20


def test_table_close_hands_are_near_even(table):
    aks = _HAND_INDEX["AKs"]
    ako = _HAND_INDEX["AKo"]
    assert 0.40 < table[aks, ako] < 0.60


def test_equity_matches_manual_combo_weighted_mean_over_range(table):
    # docs/AUDITORIA-2026-08-26.md item E2: _equity averages over the range
    # weighted by each hand's available combo count given hero's own two
    # cards as blockers, not a flat 1/N per canonical hand — so the expected
    # value here is a combo-weighted (not uniform) mean.
    nash = ICMNash()
    row = table[_HAND_INDEX["QQ"]]
    hands = ["AA", "KK", "AKs"]
    cols = [_HAND_INDEX[h] for h in hands]
    weights = nash._range_weights("QQ", hands)
    expected = np.average(row[cols], weights=weights)
    assert nash._equity("QQ", ",".join(hands)) == pytest.approx(expected)


def test_equity_range_weights_differ_from_uniform_for_mixed_combo_counts():
    # AA (6 combos) should be weighted differently from AKs (4 combos) once
    # QQ's own two cards are removed — sanity check that E2 actually changed
    # behavior rather than the weights collapsing back to uniform.
    nash = ICMNash()
    weights = nash._range_weights("QQ", ["AA", "AKs"])
    assert weights[0] != pytest.approx(weights[1])


def test_two_instances_share_the_same_cached_table():
    # The table is a module-level singleton — constructing ICMNash twice
    # must not reload/reallocate it.
    a, b = ICMNash(), ICMNash()
    assert a._table is b._table


def test_solve_hu_is_fast_now_that_equity_is_a_lookup():
    nash = ICMNash()
    t0 = time.time()
    nash.solve_hu([5000, 5000], [0.65, 0.35], sb=50, bb=100, max_iter=12)
    elapsed = time.time() - t0
    # Was ~25s worth of live Monte Carlo before I3; a generous 2s ceiling
    # still catches an accidental regression back to per-call simulation.
    assert elapsed < 2.0


def test_solve_hu_is_deterministic_across_instances():
    inputs = dict(stacks=[2000, 8000], payouts=[0.65, 0.35], sb=50, bb=100, max_iter=12)
    r1 = ICMNash().solve_hu(**inputs)
    r2 = ICMNash().solve_hu(**inputs)
    assert r1.push_range == r2.push_range
    assert r1.call_range == r2.call_range
