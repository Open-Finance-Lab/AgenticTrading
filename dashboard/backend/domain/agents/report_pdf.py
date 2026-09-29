"""Markdown → PDF for research reports (formatted edition).

Replaces the "readable but plain" v1 converter with one that produces a
professional document:

- **CJK support** via reportlab's built-in CID fonts (STSong-Light) — no
  external font files needed, works identically on macOS and Render Linux.
- **Real tables** — header band, alternating row tint, column borders, and
  cell-level Paragraph wrapping for long text.
- **Page furniture** — report title in the header, page numbers in the
  footer.
- **Page cap** — enforced during rendering via handle_pageBegin.

reportlab is imported behind a guard: environments that don't install it
still import this module (and the research router) fine — PDF generation
degrades to "no PDF", exactly like v1.
"""

from __future__ import annotations

import io
import re

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable,
    )
    _HAS_REPORTLAB = True
except Exception:  # reportlab is optional at runtime — best-effort PDF
    _HAS_REPORTLAB = False

_PDF_MAX_PAGES = 60
_BODY = 9
_LEADING = _BODY + 4
_H_SIZES = {1: 15, 2: 12.5, 3: 11, 4: 10}


class _PageCapExceeded(Exception):
    pass


def _clean(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<font face='Courier' size='7.5'>\1</font>", text)
    return text


def _split_table_block(lines, start):
    rows = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = lines[i].strip()
        if not re.match(r"^\|[\s:|-]+\|$", row):
            cells = [c.strip() for c in row.strip("|").split("|")]
            rows.append(cells)
        i += 1
    return rows, i


if _HAS_REPORTLAB:
    _ACCENT = colors.HexColor("#0e7490")
    _ROW_ALT = colors.HexColor("#f8fafc")
    _BORDER = colors.HexColor("#cbd5e1")
    _TEXT_DARK = colors.HexColor("#1e293b")
    _TEXT_MUTED = colors.HexColor("#64748b")

    class _CappedDoc(BaseDocTemplate):
        def __init__(self, *args, title="", **kw):
            super().__init__(*args, **kw)
            self._report_title = title
            self._pages = 0
            frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
            self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self._furniture)])

        def _furniture(self, canvas, doc):
            canvas.saveState()
            y = A4[1] - 10 * mm
            canvas.setStrokeColor(_BORDER)
            canvas.setLineWidth(0.5)
            canvas.line(18 * mm, y, A4[0] - 18 * mm, y)
            canvas.setFont("STSong-Light", 7)
            canvas.setFillColor(_TEXT_MUTED)
            canvas.drawString(18 * mm, y + 3, self._report_title[:60])
            canvas.drawRightString(A4[0] - 18 * mm, y + 3, "Agentic Trading Lab")
            canvas.setFont("STSong-Light", 8)
            canvas.drawCentredString(A4[0] / 2, 10 * mm, "- %d -" % doc.page)
            canvas.restoreState()

        def handle_pageBegin(self):
            self._pages += 1
            if self._pages > _PDF_MAX_PAGES:
                raise _PageCapExceeded()
            super().handle_pageBegin()

    def _ensure_fonts():
        if "STSong-Light" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def _build_table(rows):
    if not rows:
        return Spacer(1, 1)
    ncols = max(len(r) for r in rows)
    padded = [r + [""] * (ncols - len(r)) for r in rows]
    cell_style = ParagraphStyle(
        "tcell", fontName="STSong-Light", fontSize=7.5,
        leading=10, textColor=_TEXT_DARK, wordWrap="CJK",
    )
    head_style = ParagraphStyle("thead", parent=cell_style, textColor=colors.white)
    data = []
    for ri, row in enumerate(padded):
        st = head_style if ri == 0 else cell_style
        data.append([Paragraph(_clean(c), st) for c in row])
    avail_width = A4[0] - 36 * mm
    if ncols > 0:
        w0 = min(avail_width * 0.28, avail_width / ncols * 1.4)
        wn = (avail_width - w0) / max(ncols - 1, 1) if ncols > 1 else avail_width
        col_widths = [w0] + [wn] * (ncols - 1)
    else:
        col_widths = [avail_width]
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _ROW_ALT]),
        ("GRID", (0, 0), (-1, -1), 0.4, _BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def markdown_to_pdf_bytes(markdown_text):
    if not _HAS_REPORTLAB:
        return None
    _ensure_fonts()
    buf = io.BytesIO()
    title_match = re.match(r"^#\s+(.+)", (markdown_text or "").strip())
    report_title = title_match.group(1).strip() if title_match else "Research Report"
    doc = _CappedDoc(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title=report_title,
    )
    base = ParagraphStyle(
        "body", fontName="STSong-Light", fontSize=_BODY,
        leading=_LEADING, spaceAfter=4, textColor=_TEXT_DARK, wordWrap="CJK",
    )
    styles = {}
    for level, size in _H_SIZES.items():
        styles[level] = ParagraphStyle(
            f"h{level}", parent=base, fontSize=size, leading=size + 5,
            spaceBefore=10, spaceAfter=5,
            textColor=_ACCENT if level <= 2 else _TEXT_DARK,
        )
    quote_style = ParagraphStyle("quote", parent=base, leftIndent=12, textColor=_TEXT_MUTED)
    story = []
    lines = (markdown_text or "").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", line.strip()):
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER))
            story.append(Spacer(1, 4))
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.+)", line)
        if m:
            level = min(len(m.group(1)), 4)
            story.append(Paragraph(_clean(m.group(2)), styles[level]))
            i += 1
            continue
        if line.strip().startswith("|"):
            rows, next_i = _split_table_block(lines, i)
            if rows:
                story.append(Spacer(1, 3))
                story.append(_build_table(rows))
                story.append(Spacer(1, 6))
            i = next_i
            continue
        m = re.match(r"^[-*]\s+(.+)", line)
        if m:
            bullet = ParagraphStyle("bullet", parent=base, leftIndent=14, bulletIndent=4)
            story.append(Paragraph(_clean(m.group(1)), bullet, bulletText="•"))
            i += 1
            continue
        m = re.match(r"^(\d+)[.)]\s+(.+)", line)
        if m:
            numbered = ParagraphStyle("num", parent=base, leftIndent=18, firstLineIndent=-12)
            story.append(Paragraph(f"{m.group(1)}. {_clean(m.group(2))}", numbered))
            i += 1
            continue
        m = re.match(r"^>\s?(.+)", line)
        if m:
            story.append(Paragraph(f"<i>{_clean(m.group(1))}</i>", quote_style))
            i += 1
            continue
        para = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].rstrip()
            if (not nxt.strip() or nxt.lstrip().startswith(("#", "|", "- ", "* ", ">"))
                    or re.match(r"^\d+[.)]\s", nxt)
                    or re.match(r"^(-{3,}|\*{3,}|_{3,})$", nxt.strip())):
                break
            para.append(nxt)
            i += 1
        story.append(Paragraph(_clean(" ".join(para)), base))
    try:
        doc.build(story)
    except _PageCapExceeded:
        pass
    except Exception:
        return None
    pdf = buf.getvalue()
    if not pdf or len(pdf) > 20 * 1024 * 1024:
        return None
    return pdf
