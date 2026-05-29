import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER


def export_pdf(
    question: str,
    final_report: str,
    confidence: float,
    rows_validated: int,
    chart_paths: list = None,
    output_dir: str = "outputs"
) -> str:

    os.makedirs(output_dir, exist_ok=True)

    # clean filename from question
    safe_q = "".join(c for c in question[:30] if c.isalnum() or c == " ").strip()
    safe_q = safe_q.replace(" ", "_")
    output_path = os.path.join(output_dir, f"InsightFlow_{safe_q}.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    # colors
    dark_bg   = HexColor("#0a0e0a")
    lime      = HexColor("#c5f432")
    dark_green= HexColor("#0c110c")
    light_text= HexColor("#e8f0e8")
    mid_green = HexColor("#6a8a6a")
    border    = HexColor("#1e2a1e")

    styles = getSampleStyleSheet()

    # custom styles
    title_style = ParagraphStyle(
        "InsightTitle",
        fontSize=22,
        textColor=lime,
        spaceAfter=6,
        fontName="Helvetica-Bold",
        alignment=TA_LEFT,
    )
    sub_style = ParagraphStyle(
        "InsightSub",
        fontSize=9,
        textColor=mid_green,
        spaceAfter=20,
        fontName="Helvetica",
    )
    label_style = ParagraphStyle(
        "InsightLabel",
        fontSize=8,
        textColor=mid_green,
        spaceAfter=4,
        fontName="Helvetica",
        textTransform="uppercase",
    )
    question_style = ParagraphStyle(
        "InsightQuestion",
        fontSize=13,
        textColor=light_text,
        spaceAfter=16,
        fontName="Helvetica",
        leading=18,
    )
    finding_style = ParagraphStyle(
        "InsightFinding",
        fontSize=12,
        textColor=light_text,
        spaceAfter=8,
        fontName="Helvetica",
        leading=18,
    )
    body_style = ParagraphStyle(
        "InsightBody",
        fontSize=11,
        textColor=HexColor("#8aaa8a"),
        spaceAfter=6,
        fontName="Helvetica",
        leading=16,
    )

    story = []

    # header
    story.append(Paragraph("InsightFlow", title_style))
    story.append(Paragraph("Plan · Execute · Critique — Agentic AI Data Analyst", sub_style))

    # metadata table
    conf_color = "#6ab46a" if confidence >= 90 else "#e2a84a" if confidence >= 75 else "#d45a5a"
    meta_data = [
        ["QUESTION", question],
        ["CONFIDENCE", f"{confidence:.0f}%"],
        ["ROWS VALIDATED", str(rows_validated)],
        ["VERIFIED BY", "Critic Agent"],
        ["MODEL", "gpt-4o-mini (routing) · gpt-4o (reasoning)"],
    ]
    meta_table = Table(meta_data, colWidths=[4*cm, 13*cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), dark_bg),
        ("TEXTCOLOR",     (0,0), (0,-1), HexColor("#3a5a3a")),
        ("TEXTCOLOR",     (1,0), (1,-1), light_text),
        ("FONTNAME",      (0,0), (0,-1), "Helvetica"),
        ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [dark_bg, dark_green]),
        ("GRID",          (0,0), (-1,-1), 0.5, border),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 16))

    # finding section
    story.append(Paragraph("EXECUTIVE FINDING", label_style))

    # parse markdown sections from report
    lines = final_report.strip().split("\n")
    in_finding = False
    in_why = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("## Finding") or line.startswith("## finding"):
            in_finding = True
            in_why = False
            continue
        elif line.startswith("## Why it matters") or line.startswith("## Why It Matters"):
            in_finding = False
            in_why = True
            story.append(Spacer(1, 10))
            story.append(Paragraph("WHY IT MATTERS", label_style))
            continue
        elif line.startswith("##"):
            continue

        # clean markdown
        line = line.replace("**", "").replace("*", "").replace("#", "")

        if in_finding:
            story.append(Paragraph(line, finding_style))
        elif in_why:
            story.append(Paragraph(line, body_style))
        else:
            story.append(Paragraph(line, body_style))

    story.append(Spacer(1, 16))

    # charts
    if chart_paths:
        story.append(Paragraph("CHARTS", label_style))
        story.append(Spacer(1, 8))
        for chart_path in chart_paths:
            if chart_path and os.path.exists(chart_path):
                try:
                    img = Image(chart_path, width=16*cm, height=8*cm)
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception:
                    pass

    # footer
    story.append(Spacer(1, 20))
    footer_style = ParagraphStyle(
        "Footer",
        fontSize=8,
        textColor=HexColor("#3a5a3a"),
        fontName="Helvetica",
        alignment=TA_CENTER,
    )
    story.append(Paragraph(
        "Generated by InsightFlow · Plan · Execute · Critique · Agentic AI Data Analyst",
        footer_style
    ))

    doc.build(story)
    return output_path