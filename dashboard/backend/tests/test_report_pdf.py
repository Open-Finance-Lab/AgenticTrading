"""PDF fallback: the platform renders a PDF from the stored Markdown report
when the agent service's own PDF is absent (its Word-based generator doesn't
run on Linux). Covers the converter's formatting contracts."""

from dashboard.backend.domain.agents.report_pdf import markdown_to_pdf_bytes


SAMPLE = """# Shell Company Screening Report

**Client:** Huatai Capital
**Target market:** HKEX Main Board

---

## 1. Executive Summary

This screening reviewed **HKEX Main Board** listed-platform candidates that
could serve as reverse-merger targets. Two candidates passed with caveats.

- ABC Holdings — clean shell, minor warrant overhang
- DEF Industries — pending litigation, low impact

### Candidate Comparison

| Candidate | Result | Key reason |
|---|---|---|
| ABC Holdings | PASS WITH CAVEAT | Clean shell |
| DEF Industries | PASS WITH CAVEAT | Litigation pending |
| GHI Group | FAIL | Controlling shareholder unwilling |

> Note: market capitalization is a reference, not a hard filter.

1. ABC Holdings — best platform suitability
2. DEF Industries — acceptable if litigation clears

`inline code` should survive.
"""


def test_produces_valid_pdf():
    result = markdown_to_pdf_bytes(SAMPLE)
    assert result is not None
    assert result[:5] == b"%PDF-"
    assert len(result) > 1000


def test_empty_and_none_inputs():
    assert markdown_to_pdf_bytes("") is not None  # valid empty PDF
    assert markdown_to_pdf_bytes(None) is not None  # treated as empty


def test_large_document_is_bounded():
    # A realistic Deep Research run can be 60K+ chars; the converter must
    # finish without exhausting memory or producing a gigabyte blob.
    big = SAMPLE * 100  # ~200K chars
    result = markdown_to_pdf_bytes(big)
    assert result is not None
    assert len(result) < 20 * 1024 * 1024  # 20 MB cap from the module


def test_unicode_content():
    text = "# 中文标题\n\n这是中文段落，包含**加粗**内容。\n"
    result = markdown_to_pdf_bytes(text)
    # Helvetica can't encode CJK; reportlab substitutes or the paragraph
    # is dropped — the PDF itself must still be valid, never a crash.
    assert result is not None
    assert result[:5] == b"%PDF-"


def test_no_stored_artifact_still_404s():
    """The endpoint must 404 when BOTH pdf and markdown are missing."""
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)
    # Login first
    resp = client.post("/api/auth/signup", json={
        "email": "pdf.fallback@example.test",
        "display_name": "PDF Fallback",
        "password": "SecurePass1!",
    })
    assert resp.status_code in (200, 201, 409)  # 409 if already exists

    resp = client.post("/api/auth/login", json={
        "email": "pdf.fallback@example.test",
        "password": "SecurePass1!",
    })
    assert resp.status_code == 200
    csrf = resp.cookies.get("atl_csrf") or resp.headers.get("set-cookie", "")

    # Request a PDF for a run that doesn't exist → 404
    resp = client.get(
        "/api/v1/research/runs/rr_nonexistent/artifacts/pdf",
        headers={"X-CSRF-Token": csrf.split("atl_csrf=")[-1].split(";")[0]} if csrf else {},
    )
    assert resp.status_code == 404


def test_crossing_inline_markup_degrades_to_plain_text_not_a_crash():
    # _clean's regexes do not nest, so these produce crossed tags that
    # reportlab's Paragraph parser rejects at construction time.
    for line in (
        "*a `b* c`",
        "**bold `code** tail`",
        "| *x `y* z` | ok |\n|---|---|\n| 1 | 2 |",
        "- *a `b* c`",
        "1. *a `b* c`",
        "> *a `b* c`",
        "## *a `b* c`",
    ):
        result = markdown_to_pdf_bytes(f"# Title\n\n{line}\n")
        assert result is not None, line
        assert result[:5] == b"%PDF-", line


def test_page_cap_truncates_instead_of_dropping_the_pdf():
    import re

    result = markdown_to_pdf_bytes(SAMPLE * 400)
    assert result is not None
    assert result[:5] == b"%PDF-"
    pages = len(re.findall(rb"/Type /Page\b", result))
    # 60 capped pages plus the one carrying the truncation notice.
    assert 60 < pages <= 62


def test_report_lists_available_artifacts_without_rendering_a_pdf(monkeypatch):
    """The download buttons read this list; it must never run the converter."""
    from dashboard.backend.api.routers import research

    stored = {"markdown": {"content_base64": "# r", "filename": "r.md"}, "evidence_json": {"x": 1}}
    monkeypatch.setattr(research, "_run_and_template_or_404", lambda run_id, user: ({"template_id": "t"}, None))
    monkeypatch.setattr(research.research_store, "get_artifact", lambda run_id, kind: stored.get(kind))

    def forbidden(_markdown):
        raise AssertionError("listing artifacts rendered a PDF")

    monkeypatch.setattr(research.report_pdf, "markdown_to_pdf_bytes", forbidden)

    payload = research.get_run_report("rr_1", current_user={"id": 1})
    # No stored docx; the PDF is offered because the Markdown can render it.
    assert payload["available_artifacts"] == ["markdown", "pdf", "evidence_json"]

    monkeypatch.setattr(research.report_pdf, "_HAS_REPORTLAB", False)
    assert research.get_run_report("rr_1", current_user={"id": 1})["available_artifacts"] == [
        "markdown", "evidence_json",
    ]


def test_report_view_does_not_probe_downloads():
    from pathlib import Path

    app_js = (Path(__file__).resolve().parents[2] / "frontend" / "app.js").read_text(encoding="utf-8")
    start = app_js.index("async function showCompletedResearchReport(")
    body = app_js[start : app_js.index("\nasync function ", start + 1)]
    assert "available_artifacts" in body
    assert "Range" not in body and "fetch(" not in body
