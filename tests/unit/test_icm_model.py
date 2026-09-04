"""ICM model tests.

Reference values computed with the correct O(n!) formula:
    for perm in permutations: compute full perm_prob, then assign payouts.
All sums verified to equal 1.0 (or < 1 when busted players exhaust payouts early).
"""

import time
import random
from engine.icm.model import ICMModel

m = ICMModel()
TOL = 1e-9


def test_equal_stacks_symmetric():
    # 3 players equal stacks: each gets exactly 1/3 of total payouts
    eq = m.equity([1000, 1000, 1000], [0.5, 0.3, 0.2])
    assert len(eq) == 3
    for e in eq:
        assert abs(e - 1 / 3) < TOL


def test_chip_leader_gets_more():
    # Reference: [0.383929, 0.3275, 0.288571]
    eq = m.equity([5000, 3000, 2000], [0.5, 0.3, 0.2])
    assert abs(eq[0] - 0.383929) < 1e-5
    assert abs(eq[1] - 0.327500) < 1e-5
    assert abs(eq[2] - 0.288571) < 1e-5


def test_equities_sum_to_one():
    eq = m.equity([5000, 3000, 2000], [0.5, 0.3, 0.2])
    assert abs(sum(eq) - 1.0) < TOL


def test_four_player_two_paid():
    # Reference: [0.36634921, 0.30333333, 0.21650794, 0.11380952]
    eq = m.equity([4000, 3000, 2000, 1000], [0.6, 0.4])
    assert abs(eq[0] - 0.36634921) < 1e-6
    assert abs(eq[1] - 0.30333333) < 1e-6
    assert abs(eq[2] - 0.21650794) < 1e-6
    assert abs(eq[3] - 0.11380952) < 1e-6
    assert abs(sum(eq) - 1.0) < TOL


def test_busted_player_gets_zero():
    eq = m.equity([6000, 4000, 0], [0.5, 0.3, 0.2])
    assert eq[2] == 0.0
    assert abs(eq[0] - 0.42) < TOL
    assert abs(eq[1] - 0.38) < TOL


def test_stack_proportional_to_equity_with_one_payout():
    # If there's only one payout (winner-take-all), equity = stack / total
    stacks = [3000, 2000, 5000]
    total = sum(stacks)
    eq = m.equity(stacks, [1.0])
    for i, s in enumerate(stacks):
        assert abs(eq[i] - s / total) < TOL


def test_chip_leader_ordering():
    # Largest stack must have highest ICM equity
    eq = m.equity([8000, 5000, 3000, 2000, 1000, 1000], [0.40, 0.25, 0.15, 0.10, 0.07, 0.03])
    assert eq[0] > eq[1] > eq[2]
    assert abs(sum(eq) - 1.0) < TOL


def test_equity_of_matches_full_equity():
    stacks, payouts = [5000, 3000, 2000], [0.5, 0.3, 0.2]
    full = m.equity(stacks, payouts)
    for i in range(len(stacks)):
        assert abs(m.equity_of(i, stacks, payouts) - full[i]) < TOL


def test_equity_of_busted_player_is_zero():
    assert m.equity_of(2, [6000, 4000, 0], [0.5, 0.3, 0.2]) == 0.0


def test_performance_ten_players():
    # 10-player final table must complete in under 1 second
    rng = random.Random(0)
    stacks = [rng.randint(500, 5000) for _ in range(10)]
    payouts = [0.30, 0.20, 0.13, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.01]
    t0 = time.time()
    eq = m.equity(stacks, payouts)
    elapsed = time.time() - t0
    assert elapsed < 1.0, f"10-player ICM took {elapsed:.2f}s (limit: 1s)"
    assert abs(sum(eq) - 1.0) < TOL
