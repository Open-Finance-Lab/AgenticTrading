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


def test_footer_names_the_operating_entity():
    """The footer previously read only "© 2026 Agentic Trading Lab. All rights
    reserved." with no indication of who runs the platform -- all three personas
    ranked "who runs this?" the #1 trust-killer. Pins the operating entity and
    the open-source framing in the shipped bundle."""
    text = _shipped_text()
    assert "© 2026 SecureFinAI Lab" in text
    assert "open-source research platform" in text


_BANNED_FRAGMENTS = (
    "different brains",
    "one sentence and a few minutes",
    "Est. token cost",
    "All rights reserved",
    "just chat",
    "Talk to it on Discord",
    "412k in",
    "Test AI trading agents",
    "strategy prompt",
    "Strategy prompt",
    "Pick a model",
)


def test_final_review_fix_wave_fragments_are_gone():
    """Curated exact fragments only — no stemmed/root matching (e.g. never ban
    "play"; the deferred agent-playground.exe string would false-positive)."""
    text = _shipped_text()
    for fragment in _BANNED_FRAGMENTS:
        assert fragment not in text, f"banned fragment still present: {fragment!r}"


def test_discord_first_mention_uses_the_community_phrase():
    assert "our Discord community" in _shipped_text()


_META_DESCRIPTION = (
    "Talk to agents. Test trading ideas. Try AI trading agents on real market "
    "data — no code required."
)


def test_meta_description_reads_cleanly_in_all_three_tags():
    """The previous description opened two consecutive sentences with "Test"
    ("Test trading ideas. Test AI trading agents…") — and this is the page's most
    externally visible copy (search snippets, og/twitter cards). Tag-scoped on
    purpose: a raw ``count(...) == 3`` can be satisfied by the right string in
    the wrong tags (shown by fault injection during review), so each of the
    three description tags is located and compared individually."""
    html = _index_html()
    for pattern in (
        r'<meta name="description" content="([^"]*)"',
        r'<meta property="og:description" content="([^"]*)"',
        r'<meta name="twitter:description" content="([^"]*)"',
    ):
        m = re.search(pattern, html)
        assert m, f"missing description tag: {pattern}"
        assert m.group(1) == _META_DESCRIPTION, f"{pattern} carries {m.group(1)!r}"


def test_settings_label_uses_the_ai_model_glossary_term():
    """The experiment-settings panel said bare "Model"; the plan's glossary maps
    bare "model" -> "AI model" for every user-facing label. Asserting the compiled
    ``label:"..."`` form is minifier-stable: esbuild never rewrites string
    literals and leaves identifier-valid object keys unquoted."""
    text = _shipped_text()
    assert 'label:"AI model"' in text
    assert 'label:"Model"' not in text


def test_race_sample_cards_have_no_live_pulse():
    """Race's Standings/Leaderboard cards carry "Illustrative example" tags, yet a
    pulsing green "Live" badge sat beside both — animating exactly the claim the
    label disclaims. The badge's ping dot was the landing's only use of Tailwind's
    ``animate-ping``, so its absence from the shipped text means the badge (not
    merely its caption) is gone. The surrounding prose may still say "live" —
    live *market prices* are a real product property; the badge on sample data
    was the contradiction. The positive assertions pin that the cards themselves
    still ship AND that the bundle text was actually read: "Standings" and
    "Leaderboard" live only in the JS bundle, so a broken entry-bundle reference
    cannot turn the negative check vacuous (shown by fault injection during
    review)."""
    text = _shipped_text()
    assert "Standings" in text and "Leaderboard" in text
    assert "animate-ping" not in text


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
