import pytest

from engine.ev.preflop import PreflopRaiseEV


def test_raise_ev_matches_manual_calculation():
    calc = PreflopRaiseEV()
    ev = calc.raise_ev(fold_equity=0.6, pot=10, raise_size=20, equity_if_called=0.4)
    ev_fold = 0.6 * 10
    ev_called = 0.4 * (0.4 * (10 + 40) - 20)
    assert abs(ev - (ev_fold + ev_called)) < 1e-9


def test_raise_ev_high_fold_equity_is_positive():
    calc = PreflopRaiseEV()
    ev = calc.raise_ev(fold_equity=0.9, pot=10, raise_size=20, equity_if_called=0.1)
    assert ev > 0


def test_raise_ev_rejects_invalid_fold_equity():
    calc = PreflopRaiseEV()
    with pytest.raises(ValueError):
        calc.raise_ev(fold_equity=1.5, pot=10, raise_size=20, equity_if_called=0.4)


def test_raise_ev_rejects_invalid_equity_if_called():
    calc = PreflopRaiseEV()
    with pytest.raises(ValueError):
        calc.raise_ev(fold_equity=0.5, pot=10, raise_size=20, equity_if_called=-0.1)


def test_rfi_ev_delegates_to_raise_ev():
    calc = PreflopRaiseEV()
    assert calc.rfi_ev(0.5, 1.5, 2.5, 0.55) == calc.raise_ev(0.5, 1.5, 2.5, 0.55)


def test_three_bet_ev_delegates_to_raise_ev():
    calc = PreflopRaiseEV()
    assert calc.three_bet_ev(0.5, 8.5, 9.0, 0.5) == calc.raise_ev(0.5, 8.5, 9.0, 0.5)


def test_four_bet_ev_delegates_to_raise_ev():
    calc = PreflopRaiseEV()
    assert calc.four_bet_ev(0.4, 21.0, 22.0, 0.55) == calc.raise_ev(0.4, 21.0, 22.0, 0.55)


def test_three_bet_ev_pot_must_include_openers_raise():
    # Sanity check on the documented convention: a 3-bet's `pot` includes the
    # open-raise, so it should be strictly larger than the RFI pot alone.
    calc = PreflopRaiseEV()
    rfi_pot = 1.5  # blinds only
    open_size = 2.5
    three_bet_pot = rfi_pot + open_size
    ev_low_pot = calc.three_bet_ev(0.5, rfi_pot, 9.0, 0.5)
    ev_correct_pot = calc.three_bet_ev(0.5, three_bet_pot, 9.0, 0.5)
    assert ev_correct_pot > ev_low_pot
