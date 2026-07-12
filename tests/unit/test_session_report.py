"""SessionReport tests."""

from output.reports.session_report import Hand, SessionReport


def _hand(result: float, ev: float) -> Hand:
    return Hand(hero_cards=["As", "Ks"], board=["Qh", "Js", "Ts"], pot=10.0, result=result, ev=ev)


def test_add_hand_appends():
    report = SessionReport()
    report.add_hand(_hand(5.0, 2.0))
    assert len(report.hands) == 1


def test_summary_empty_session_has_zeroed_fields():
    report = SessionReport()
    summary = report.summary()
    assert summary["hands_played"] == 0
    assert summary["total_result"] == 0
    assert summary["total_ev"] == 0
    assert summary["luck_adjusted"] == 0
    assert "session_id" in summary


def test_summary_sums_results_and_ev():
    report = SessionReport()
    report.add_hand(_hand(result=10.0, ev=4.0))
    report.add_hand(_hand(result=-3.0, ev=1.0))
    summary = report.summary()
    assert summary["hands_played"] == 2
    assert summary["total_result"] == 7.0
    assert summary["total_ev"] == 5.0
    assert summary["luck_adjusted"] == 2.0


def test_summary_luck_adjusted_is_result_minus_ev():
    report = SessionReport()
    report.add_hand(_hand(result=8.0, ev=8.0))
    summary = report.summary()
    assert summary["luck_adjusted"] == 0.0


def test_session_id_defaults_to_timestamp_format():
    report = SessionReport()
    # "YYYYmmdd_HHMM" — 13 chars, digits with one underscore
    assert len(report.session_id) == 13
    assert report.session_id[8] == "_"


def test_custom_session_id():
    report = SessionReport(session_id="my-session")
    assert report.summary()["session_id"] == "my-session"
