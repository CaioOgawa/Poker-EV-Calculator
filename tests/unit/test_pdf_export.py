import os

from output.reports.session_report import SessionReport, Hand
from output.reports.pdf_export import PDFExporter


def _hand(result, ev, category=None):
    return Hand(
        hero_cards=["As", "Ks"],
        board=["Qh", "Js", "Ts"],
        pot=10.0,
        result=result,
        ev=ev,
        category=category,
    )


def test_export_session_creates_pdf(tmp_path):
    report = SessionReport(session_id="test-session")
    report.add_hand(_hand(5, 8, "value bet"))
    report.add_hand(_hand(-15, -2, "river bluff"))

    out = tmp_path / "session.pdf"
    PDFExporter().export_session(report, str(out))
    assert out.exists()
    assert os.path.getsize(out) > 0


def test_export_session_starts_with_pdf_magic_bytes(tmp_path):
    report = SessionReport(session_id="test-session")
    report.add_hand(_hand(5, 8))
    out = tmp_path / "session.pdf"
    PDFExporter().export_session(report, str(out))
    assert out.read_bytes()[:5] == b"%PDF-"


def test_export_session_without_hands(tmp_path):
    report = SessionReport(session_id="empty-session")
    out = tmp_path / "empty.pdf"
    PDFExporter().export_session(report, str(out))
    assert out.exists()


def test_export_session_without_leaks(tmp_path):
    report = SessionReport(session_id="test-session")
    report.add_hand(_hand(5, 8, "value bet"))
    out = tmp_path / "no_leaks.pdf"
    PDFExporter().export_session(report, str(out), include_leaks=False)
    assert out.exists()
