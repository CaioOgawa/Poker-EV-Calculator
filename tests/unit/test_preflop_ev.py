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


def test_four_bet_ev_does_not_double_count_prior_investment():
    # Hand-arithmetic ground truth (blinds 0.5/1, hero opens 2.5, villain
    # 3-bets to 9, hero 4-bets to 22, villain calls): real final pot is
    # 0.5 (BB dead money the SB never matched) + 22 + 22 = 44.5, and hero's
    # incremental cost is 22 - 2.5 = 19.5 — not the naive `four_bet_size`,
    # since hero's own open is already sitting in `pot`.
    calc = PreflopRaiseEV()
    pot = 0.5 + 2.5 + 9.0  # BB dead + hero's open + villain's 3-bet
    ev = calc.four_bet_ev(
        fold_equity=0.0,
        pot=pot,
        four_bet_size=22.0,
        equity_if_called=0.5,
        hero_invested=2.5,
        villain_invested=9.0,
    )
    pot_if_called = 44.5
    hero_cost = 22.0 - 2.5
    expected = 0.5 * pot_if_called - hero_cost
    assert abs(ev - expected) < 1e-9

    # The audit's naive replacement formula (pot + 2*raise - both invested)
    # gives a different, wrong number for this same spot — pin it down so a
    # regression back to that formula is caught.
    wrong_pot_if_called = pot + 22.0 + (22.0 - 9.0)
    wrong_ev = 0.5 * wrong_pot_if_called - (22.0 - 2.5)
    assert abs(wrong_ev - 4.0) < 1e-9
    assert abs(ev - wrong_ev) > 1.0
