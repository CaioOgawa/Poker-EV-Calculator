import pytest

from engine.ev.multistreet import MultiStreetEV, _regret_match

RIVER_BOARD = ["2c", "7d", "Th", "Ks"]


# ---------------------------------------------------------------------------
# _regret_match — pure function, exact expected outputs
# ---------------------------------------------------------------------------


def test_regret_match_proportional_to_positive_regret():
    assert _regret_match([3.0, 1.0]) == [0.75, 0.25]


def test_regret_match_ignores_negative_regret():
    assert _regret_match([5.0, -5.0, 0.0]) == [1.0, 0.0, 0.0]


def test_regret_match_uniform_when_all_nonpositive():
    assert _regret_match([-1.0, -2.0, 0.0]) == [1 / 3, 1 / 3, 1 / 3]


def test_regret_match_sums_to_one():
    probs = _regret_match([2.0, 5.0, 0.0, 1.0])
    assert abs(sum(probs) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# solve() — invariants that hold structurally, regardless of convergence
# ---------------------------------------------------------------------------


def test_oop_ev_plus_ip_ev_equals_pot():
    solver = MultiStreetEV(iterations=200, seed=0, max_combo_pairs=40)
    res = solver.solve(
        oop_range="AKs", ip_range="QJs", board=RIVER_BOARD, pot=100, bet_sizes=[0.5, 1.0]
    )
    assert abs((res.oop_ev + res.ip_ev) - res.pot) < 1e-6


def test_exploitability_is_never_negative():
    # By construction, a best response can never do worse than the average
    # strategy it's responding to — this must hold even with very few
    # training iterations, since it's a property of the evaluation pass,
    # not of convergence.
    solver = MultiStreetEV(iterations=5, seed=0, max_combo_pairs=40)
    res = solver.solve(
        oop_range="AKs", ip_range="QJs", board=RIVER_BOARD, pot=100, bet_sizes=[0.5, 1.0]
    )
    assert res.exploitability >= -1e-6


def test_dominant_range_wins_most_of_the_pot():
    # OOP has top set on this board, IP has bottom pair — OOP should end up
    # with substantially more than half the pot.
    solver = MultiStreetEV(iterations=1500, seed=0, max_combo_pairs=40)
    res = solver.solve(
        oop_range="TT", ip_range="22", board=RIVER_BOARD, pot=100, bet_sizes=[0.5, 1.0]
    )
    assert res.oop_ev > res.pot * 0.9


# ---------------------------------------------------------------------------
# CFR convergence — exploitability must shrink with more iterations
# ---------------------------------------------------------------------------


def test_exploitability_shrinks_with_more_iterations():
    low = MultiStreetEV(iterations=50, seed=11, max_combo_pairs=20).solve(
        oop_range="AKs", ip_range="QJs", board=RIVER_BOARD, pot=100, bet_sizes=[0.5, 1.0]
    )
    high = MultiStreetEV(iterations=3000, seed=11, max_combo_pairs=20).solve(
        oop_range="AKs", ip_range="QJs", board=RIVER_BOARD, pot=100, bet_sizes=[0.5, 1.0]
    )
    assert high.exploitability < low.exploitability
    assert high.exploitability < low.exploitability * 0.5


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_5_card_board():
    solver = MultiStreetEV(iterations=10, seed=0)
    with pytest.raises(ValueError):
        solver.solve(oop_range="AA", ip_range="KK", board=["2c", "7d", "Th", "Ks", "9h"], pot=100)


def test_rejects_2_card_board():
    solver = MultiStreetEV(iterations=10, seed=0)
    with pytest.raises(ValueError):
        solver.solve(oop_range="AA", ip_range="KK", board=["2c", "7d"], pot=100)


def test_rejects_nonpositive_pot():
    solver = MultiStreetEV(iterations=10, seed=0)
    with pytest.raises(ValueError):
        solver.solve(oop_range="AKs", ip_range="QJs", board=RIVER_BOARD, pot=0)


def test_rejects_empty_bet_sizes():
    solver = MultiStreetEV(iterations=10, seed=0)
    with pytest.raises(ValueError):
        solver.solve(oop_range="AKs", ip_range="QJs", board=RIVER_BOARD, pot=100, bet_sizes=[])


def test_rejects_negative_bet_size():
    solver = MultiStreetEV(iterations=10, seed=0)
    with pytest.raises(ValueError):
        solver.solve(oop_range="AKs", ip_range="QJs", board=RIVER_BOARD, pot=100, bet_sizes=[-0.5])


def test_rejects_combo_pairs_over_cap():
    solver = MultiStreetEV(iterations=10, seed=0, max_combo_pairs=5)
    with pytest.raises(ValueError):
        solver.solve(oop_range="AA,KK", ip_range="QQ,JJ", board=RIVER_BOARD, pot=100)


def test_rejects_fully_blocked_ranges():
    solver = MultiStreetEV(iterations=10, seed=0)
    with pytest.raises(ValueError):
        # Board already holds 2 of the 4 aces, so both AA ranges collapse to
        # the same single remaining combo (Ad-Ac) — the only "pair" is a
        # combo against itself, which always shares cards.
        solver.solve(oop_range="AA", ip_range="AA", board=["As", "Ah", "7d", "Ks"], pot=100)


# ---------------------------------------------------------------------------
# strategy() query
# ---------------------------------------------------------------------------


def test_strategy_requires_solve_first():
    solver = MultiStreetEV(iterations=10, seed=0)
    with pytest.raises(RuntimeError):
        solver.strategy("oop", ["As", "Ks"], RIVER_BOARD)


def test_strategy_rejects_invalid_player():
    solver = MultiStreetEV(iterations=10, seed=0, max_combo_pairs=20)
    solver.solve(oop_range="AKs", ip_range="QJs", board=RIVER_BOARD, pot=100)
    with pytest.raises(ValueError):
        solver.strategy("dealer", ["As", "Ks"], RIVER_BOARD)


def test_strategy_frequencies_sum_to_one():
    solver = MultiStreetEV(iterations=500, seed=0, max_combo_pairs=20)
    solver.solve(oop_range="AKs", ip_range="QJs", board=RIVER_BOARD, pot=100, bet_sizes=[0.5, 1.0])
    strat = solver.strategy("oop", ["Ah", "Kh"], RIVER_BOARD)
    assert set(strat.keys()) == {"check", "bet_0.5", "bet_1.0"}
    assert abs(sum(strat.values()) - 1.0) < 1e-9


def test_strategy_facing_bet_has_fold_call_actions():
    solver = MultiStreetEV(iterations=500, seed=0, max_combo_pairs=20)
    solver.solve(oop_range="AKs", ip_range="QJs", board=RIVER_BOARD, pot=100, bet_sizes=[0.5, 1.0])
    strat = solver.strategy("oop", ["Ah", "Kh"], RIVER_BOARD, history="xb0")
    assert set(strat.keys()) == {"fold", "call"}
    assert abs(sum(strat.values()) - 1.0) < 1e-9


def test_strong_hand_bets_more_than_it_checks():
    # AhKh is top pair/top kicker on a Ks-high river vs a weaker range —
    # the trained strategy should favor betting over checking.
    solver = MultiStreetEV(iterations=3000, seed=3, max_combo_pairs=20)
    solver.solve(oop_range="AKs", ip_range="QJs", board=RIVER_BOARD, pot=100, bet_sizes=[0.5, 1.0])
    strat = solver.strategy("oop", ["Ah", "Kh"], RIVER_BOARD)
    assert (strat["bet_0.5"] + strat["bet_1.0"]) > strat["check"]


# ---------------------------------------------------------------------------
# bucket_report()
# ---------------------------------------------------------------------------


def test_bucket_report_requires_solve_first():
    solver = MultiStreetEV(iterations=10, seed=0)
    with pytest.raises(RuntimeError):
        solver.bucket_report("oop")


def test_bucket_report_covers_every_combo_exactly_once():
    solver = MultiStreetEV(iterations=10, seed=0, max_combo_pairs=40)
    solver.solve(oop_range="AKs", ip_range="QJs", board=RIVER_BOARD, pot=100)
    buckets = solver.bucket_report("oop", n_buckets=2)
    all_combos = [c for tier in buckets.values() for c in tier]
    assert len(all_combos) == len(set(all_combos))
    # AKs has 4 combos; the board's Ks blocks AsKs, leaving 3.
    assert len(all_combos) == 3


def test_bucket_report_top_tier_has_higher_equity_than_bottom_tier():
    # AA crushes 72o vs a fixed pair of 55 on a dry, disconnected board — the
    # 6 AA combos should land in tier_1 and all 12 72o combos below it.
    solver = MultiStreetEV(iterations=10, seed=0, max_combo_pairs=120)
    solver.solve(oop_range="AA,72o", ip_range="55", board=["3c", "4d", "Qh", "9s"], pot=100)
    buckets = solver.bucket_report("oop", n_buckets=3)
    assert buckets["tier_1"] == ["AsAh", "AsAd", "AsAc", "AhAd", "AhAc", "AdAc"]
    assert all(c.startswith("7") for tier in ("tier_2", "tier_3") for c in buckets[tier])


def test_bucket_report_clips_n_buckets_to_combo_count():
    solver = MultiStreetEV(iterations=10, seed=0, max_combo_pairs=40)
    solver.solve(oop_range="AKs", ip_range="QJs", board=RIVER_BOARD, pot=100)
    buckets = solver.bucket_report("oop", n_buckets=100)
    assert len(buckets) <= 3  # AKs only has 3 legal combos on this board
