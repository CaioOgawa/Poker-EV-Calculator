from risk.bankroll.kelly import KellyCriterion


def test_kelly_fraction_positive():
    k = KellyCriterion()
    f = k.fraction(win_prob=0.55, win_odds=1.0)
    assert 0 < f < 1


def test_kelly_fraction_zero_when_negative_ev():
    k = KellyCriterion()
    f = k.fraction(win_prob=0.3, win_odds=1.0)
    assert f == 0.0
