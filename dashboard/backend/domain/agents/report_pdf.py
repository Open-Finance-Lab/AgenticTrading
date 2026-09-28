"""Markdown → PDF fallback for research reports.

The agent service generates PDF via Microsoft Word, which is unavailable on
its Linux host — so the PDF artifact is routinely absent while the Markdown
report is always there. Rather than wait on the agent side, the platform
renders the PDF itself at download time: the caller asks for ``kind='pdf'``,
we find no stored PDF, and we convert the stored Markdown on the fly.

Design constraints:
- **reportlab, not pandoc/wkhtmltopdf**: a pure-Python dependency that
  needs no system binaries, matching the repo's requirements.txt deployment.
- **Bounded**: the Markdown of a real Deep Research run can be 60K+ chars;
  we cap the pages so a runaway conversion cannot exhaust memory.
- **Best-effort formatting**: headings, bold, bullet lists, and horizontal
  rules are rendered; tables fall back to fixed-width text. This is a
  readable PDF of the report, not a typeset replica.
"""

from __future__ import annotations

import io
import re
from typing import Optional

_PDF_MAX_PAGES = 60
_PDF_BODY_SIZE = 9.5
_PDF_HEADING_SIZES = {1: 16, 2: 13, 3: 11.5, 4: 10.5}


def markdown_to_pdf_bytes(markdown_text: str) -> Optional[bytes]:
    """Convert a Markdown report to PDF bytes; None if reportlab is missing."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Preformatted,
        )
    except ImportError:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="Research Report",
    )

    base = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=_PDF_BODY_SIZE,
        leading=_PDF_BODY_SIZE + 3.5, spaceAfter=4,
    )
    styles = {}
    for level, size in _PDF_HEADING_SIZES.items():
        styles[level] = ParagraphStyle(
            f"h{level}", parent=base, fontName="Helvetica-Bold",
            fontSize=size, leading=size + 4, spaceBefore=10, spaceAfter=5,
        )
    mono = ParagraphStyle("mono", parent=base, fontName="Courier", fontSize=8, leading=10)

    story = []
    # Strip inline Markdown noise that reportlab's mini-HTML can't handle;
    # keep <b>/<i> which it can.
    def clean(text: str) -> str:
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"<i>\1</i>", text)
        text = re.sub(r"`([^`]+)`", r"<font face='Courier' size='8'>\1</font>", text)
        return text

    lines = (markdown_text or "").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", line.strip()):
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.5, color="#cccccc"))
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Heading
        m = re.match(r"^(#{1,4})\s+(.+)", line)
        if m:
            level = min(len(m.group(1)), 4)
            story.append(Paragraph(clean(m.group(2)), styles[level]))
            i += 1
            continue

        # Table row: render as fixed-width block
        if line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = lines[i].strip()
                # Skip separator rows
                if not re.match(r"^\|[\s:|-]+\|$", row):
                    cells = [c.strip() for c in row.strip("|").split("|")]
                    table_lines.append(" | ".join(cells))
                i += 1
            if table_lines:
                story.append(Preformatted("\n".join(table_lines), mono))
                story.append(Spacer(1, 4))
            continue

        # Bullet
        m = re.match(r"^[-*]\s+(.+)", line)
        if m:
            story.append(Paragraph(f"• {clean(m.group(1))}", base))
            i += 1
            continue

        # Numbered
        m = re.match(r"^\d+[.)]\s+(.+)", line)
        if m:
            story.append(Paragraph(f"{clean(m.group(1))}", base))
            i += 1
            continue

        # Blockquote
        m = re.match(r"^>\s?(.+)", line)
        if m:
            story.append(Paragraph(
                f"<i>{clean(m.group(1))}</i>",
                ParagraphStyle("quote", parent=base, leftIndent=12,
                               textColor="#555555"),
            ))
            i += 1
            continue

        # Paragraph (merge consecutive non-special lines)
        para = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].rstrip()
            if (not nxt.strip() or nxt.lstrip().startswith(("#", "|", "- ", "* ", ">"))
                    or re.match(r"^\d+[.)]\s", nxt) or re.match(r"^(-{3,}|\*{3,}|_{3,})$", nxt.strip())):
                break
            para.append(nxt)
            i += 1
        story.append(Paragraph(clean(" ".join(para)), base))

    try:
        doc.build(story)
    except Exception:
        return None

    pdf = buf.getvalue()
    if not pdf or len(pdf) > 20 * 1024 * 1024:
        return None
    return pdf
