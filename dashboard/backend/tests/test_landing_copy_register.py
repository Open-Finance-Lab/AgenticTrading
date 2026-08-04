"""Guards the serious-register landing copy rewrite (2026-08-04 audience-language plan).

The landing page's original copy read as flippant ("it costs one sentence and a
few minutes", "different brains") and leaned on jargon a non-technical, older,
wealthy audience does not use ("token cost"). This suite pins the shipped bundle
— not the TSX source — so a copy edit that lands in ``landing/src`` but is never
rebuilt into ``dashboard/frontend/`` (see test_frontend_bundle_integrity.py) is
caught here too: these assertions read exactly what prod serves.

Block comments are stripped before the presence/absence checks so a stray
``/*! ... */`` license banner esbuild sometimes preserves can't accidentally
satisfy (or defeat) a substring check; ``//`` is deliberately left alone since
naive line-comment stripping corrupts minified JS (``//`` also appears inside
string literals like URLs).
"""

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_INDEX_HTML = _FRONTEND / "index.html"

_LOCAL_REF = re.compile(r'(?:src|href)="(/(?:assets|images)/[^"?#]+)')
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

_NO_REAL_MONEY_SENTENCE = (
    "Every test here uses simulated money. Real money is involved only if you "
    "explicitly connect a brokerage account and turn on live trading."
)


def _index_html() -> str:
    return _INDEX_HTML.read_text(encoding="utf-8")


def _referenced_js() -> list[Path]:
    html = _index_html()
    return [
        _FRONTEND / ref.lstrip("/")
        for ref in _LOCAL_REF.findall(html)
        if ref.endswith(".js") and (_FRONTEND / ref.lstrip("/")).is_file()
    ]


def _shipped_text() -> str:
    """``index.html`` plus every JS entry bundle it loads, block comments stripped."""
    parts = [_index_html()]
    for p in _referenced_js():
        parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return _BLOCK_COMMENT.sub("", "\n".join(parts))


def test_shipped_bundle_has_a_js_entry():
    """Guards the rest of this module against passing vacuously on an empty read."""
    assert _referenced_js(), "index.html references no /assets/*.js entry bundle"


def test_flippant_speed_claim_is_gone():
    assert "one sentence and a few minutes" not in _shipped_text()


def test_brains_metaphor_is_gone():
    assert "different brains" not in _shipped_text()


def test_token_cost_wording_is_gone():
    assert "Est. token cost" not in _shipped_text()


def test_illustrative_example_label_appears_at_least_twice():
    """Hero's metric row and Race's chart/standings block both need the label —
    the sample curves elsewhere on the page are not backed by a live account and
    must not read as a real result."""
    text = _shipped_text()
    count = text.count("Illustrative example")
    assert count >= 2, f"expected 'Illustrative example' at least twice, found {count}"


def test_no_real_money_sentence_is_present_verbatim():
    assert _NO_REAL_MONEY_SENTENCE in _shipped_text()


def test_auth_error_gives_a_next_step():
    """The sign-up/sign-in modal's generic failure fallback used to read as a dead
    end ("Something went wrong.") with nothing telling this audience what to do
    next. Pins the follow-up sentence and forbids the bare version it replaces —
    this string lives only in index.html's hand-written end-of-body auth <script>
    (see dashboard/landing/README.md), not in landing/src, so no TSX source or
    rebuild is involved."""
    text = _shipped_text()
    assert "Something went wrong. Please try again." in text
    assert not re.search(r"Something went wrong\.(?! Please try again\.)", text)
