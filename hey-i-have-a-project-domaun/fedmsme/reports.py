from __future__ import annotations

import json
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def prediction_pdf(machine, prediction) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("FedMSME-PdM Machine Health Report", styles["Title"]))
    story.append(Paragraph("Software-only federated predictive maintenance dashboard", styles["Normal"]))
    story.append(Spacer(1, 0.22 * inch))

    summary = [
        ["Machine", machine["name"]],
        ["Machine Type", machine["machine_type"]],
        ["Sensor Schema", machine["sensor_schema"]],
        ["Status", prediction["status"]],
        ["Critical Area", _primary_zone(prediction)],
        ["Failure Risk", f"{prediction['risk'] * 100:.1f}%"],
        ["Health Score", f"{prediction['health']:.1f}/100"],
        ["Estimated RUL", f"{prediction['rul']:.1f} operating hours"],
        ["Generated At", prediction["created_at"]],
    ]
    table = Table(summary, colWidths=[1.8 * inch, 4.6 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F7")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#172033")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8DEE9")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Explainable Risk Factors", styles["Heading2"]))
    reasons = json.loads(prediction["reasons_json"])
    if reasons:
        rows = [["Factor", "Value", "Impact"]]
        for reason in reasons:
            rows.append(
                [
                    f"{reason['label']} - {reason.get('zone', 'machine-health zone')}",
                    str(reason["value"]),
                    reason["impact"],
                ]
            )
        reason_table = Table(rows, colWidths=[2.6 * inch, 1.1 * inch, 2.7 * inch])
        reason_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172033")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8DEE9")),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(reason_table)
    else:
        story.append(Paragraph("No explanations were generated for this prediction.", styles["Normal"]))

    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Recommended Action", styles["Heading2"]))
    default_action = {
        "Safe": "Continue normal monitoring and run the next prediction after the current shift.",
        "Warning": "Schedule inspection during the next maintenance window and check highlighted sensors.",
        "Critical": "Prioritize inspection before the next production cycle and prepare spare parts.",
    }.get(prediction["status"], "Review the machine health trend and sensor values.")
    action = reasons[0].get("action", default_action) if reasons else default_action
    story.append(Paragraph(action, styles["Normal"]))
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        Paragraph(
            "Note: This capstone demo uses benchmark-style and simulated sensor data. In production, "
            "the same software layer can receive IoT/PLC streams from real shop-floor machines.",
            styles["Italic"],
        )
    )

    doc.build(story)
    return buffer.getvalue()


def _primary_zone(prediction) -> str:
    try:
        reasons = json.loads(prediction["reasons_json"])
    except Exception:
        reasons = []
    if reasons:
        return reasons[0].get("zone", "General machine-health zone")
    return "General machine-health zone"
