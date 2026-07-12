from engine.ev.calculator import EVCalculator


def test_pot_odds():
    calc = EVCalculator()
    assert abs(calc.pot_odds(50, 100) - 1 / 3) < 1e-9


def test_call_ev_positive():
    calc = EVCalculator()
    ev = calc.call_ev(equity=0.5, pot=100, call_size=40)
    assert ev > 0


def test_call_ev_negative():
    calc = EVCalculator()
    ev = calc.call_ev(equity=0.1, pot=100, call_size=90)
    assert ev < 0
