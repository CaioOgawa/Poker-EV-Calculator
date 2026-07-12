import pytest

from engine.ev.implied_odds import ImpliedOdds


def test_effective_pot_odds_matches_plain_pot_odds_with_no_implied_winnings():
    io = ImpliedOdds()
    assert abs(io.effective_pot_odds(50, 100) - 1 / 3) < 1e-9


def test_effective_pot_odds_decreases_with_future_winnings():
    io = ImpliedOdds()
    plain = io.effective_pot_odds(50, 100)
    with_implied = io.effective_pot_odds(50, 100, expected_future_winnings=200)
    assert with_implied < plain


def test_effective_pot_odds_rejects_negative_inputs():
    io = ImpliedOdds()
    with pytest.raises(ValueError):
        io.effective_pot_odds(-1, 100)
    with pytest.raises(ValueError):
        io.effective_pot_odds(50, -1)
    with pytest.raises(ValueError):
        io.effective_pot_odds(50, 100, expected_future_winnings=-1)


def test_ev_matches_call_ev_with_no_implied_winnings():
    io = ImpliedOdds()
    ev = io.ev(equity=0.5, pot=100, call_size=40)
    assert abs(ev - (0.5 * (100 + 40) - 40)) < 1e-9


def test_ev_increases_with_future_winnings():
    io = ImpliedOdds()
    plain = io.ev(equity=0.3, pot=100, call_size=40)
    boosted = io.ev(equity=0.3, pot=100, call_size=40, expected_future_winnings=100)
    assert boosted > plain


def test_ev_rejects_invalid_equity():
    io = ImpliedOdds()
    with pytest.raises(ValueError):
        io.ev(equity=1.5, pot=100, call_size=40)


def test_required_future_winnings_zero_when_pot_odds_already_sufficient():
    io = ImpliedOdds()
    # equity 0.5 already clears the 1/3 pot-odds threshold on this call.
    assert io.required_future_winnings(equity=0.5, pot=100, call_size=50) == 0.0


def test_required_future_winnings_positive_when_pot_odds_insufficient():
    io = ImpliedOdds()
    needed = io.required_future_winnings(equity=0.2, pot=100, call_size=50)
    assert needed > 0.0
    # At the computed X, the implied-odds EV should be ~breakeven.
    ev = io.ev(equity=0.2, pot=100, call_size=50, expected_future_winnings=needed)
    assert abs(ev) < 1e-6


def test_required_future_winnings_rejects_zero_equity():
    io = ImpliedOdds()
    with pytest.raises(ValueError):
        io.required_future_winnings(equity=0.0, pot=100, call_size=50)
