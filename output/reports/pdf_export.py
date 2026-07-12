"""PDF export for a `SessionReport` — summary stats plus a per-category
leak breakdown (when hands are tagged via `Hand.category`).
"""

from __future__ import annotations

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from output.reports.leak_report import LeakReport
from output.reports.session_report import SessionReport

_TABLE_STYLE = None
if REPORTLAB_AVAILABLE:
    _TABLE_STYLE = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
    )


class PDFExporter:
    def export_session(self, report: SessionReport, path: str, include_leaks: bool = True) -> None:
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab required for PDF export")

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(path, pagesize=letter)
        story = [Paragraph(f"Session Report — {report.session_id}", styles["Title"]), Spacer(1, 12)]

        summary = report.summary()
        summary_rows = [["Metric", "Value"]] + [[k, str(v)] for k, v in summary.items()]
        story.append(Table(summary_rows, style=_TABLE_STYLE))
        story.append(Spacer(1, 20))

        if include_leaks and report.hands:
            leaks = LeakReport().by_category(report.hands)
            story.append(Paragraph("Leaks by Category", styles["Heading2"]))
            story.append(Spacer(1, 8))
            leak_rows = [["Category", "Hands", "Total EV", "Luck Adj."]] + [
                [leak.category, str(leak.hands), str(leak.total_ev), str(leak.luck_adjusted)]
                for leak in leaks
            ]
            story.append(Table(leak_rows, style=_TABLE_STYLE))

        doc.build(story)
