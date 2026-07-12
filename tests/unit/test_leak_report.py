from output.reports.session_report import Hand
from output.reports.leak_report import LeakReport, CategoryLeak


def _hand(result, ev, category=None):
    return Hand(
        hero_cards=["As", "Ks"],
        board=["Qh", "Js", "Ts"],
        pot=10.0,
        result=result,
        ev=ev,
        category=category,
    )


def test_by_category_returns_category_leaks():
    hands = [_hand(5, 8, "value bet"), _hand(-15, -2, "river bluff")]
    leaks = LeakReport().by_category(hands)
    assert all(isinstance(leak, CategoryLeak) for leak in leaks)


def test_by_category_groups_matching_categories():
    hands = [
        _hand(-15, -2, "river bluff"),
        _hand(-20, -3, "river bluff"),
        _hand(5, 8, "value bet"),
    ]
    leaks = LeakReport().by_category(hands)
    bluff = next(leak for leak in leaks if leak.category == "river bluff")
    assert bluff.hands == 2
    assert bluff.total_ev == -5.0
    assert bluff.total_result == -35.0
    assert bluff.luck_adjusted == -30.0


def test_by_category_sorted_worst_ev_first():
    hands = [
        _hand(5, 8, "value bet"),
        _hand(-15, -2, "river bluff"),
        _hand(3, 3, "cbet"),
    ]
    leaks = LeakReport().by_category(hands)
    assert leaks[0].category == "river bluff"
    assert leaks[-1].category == "value bet"


def test_by_category_groups_none_as_uncategorized():
    hands = [_hand(1, 1), _hand(2, 2)]
    leaks = LeakReport().by_category(hands)
    assert len(leaks) == 1
    assert leaks[0].category == "uncategorized"
    assert leaks[0].hands == 2


def test_by_category_empty_hands_returns_empty():
    assert LeakReport().by_category([]) == []


def test_top_leaks_limits_count():
    hands = [_hand(-1, -1, f"cat{i}") for i in range(10)]
    leaks = LeakReport().top_leaks(hands, n=3)
    assert len(leaks) == 3


def test_top_leaks_defaults_to_five():
    hands = [_hand(-1, -1, f"cat{i}") for i in range(10)]
    leaks = LeakReport().top_leaks(hands)
    assert len(leaks) == 5


def test_avg_ev_computed_correctly():
    hands = [_hand(0, 4, "spot"), _hand(0, 2, "spot")]
    leaks = LeakReport().by_category(hands)
    assert leaks[0].avg_ev == 3.0
