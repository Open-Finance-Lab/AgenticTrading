"""The landing states what the product is for, above the three-act narrative.

A tester could not tell what the platform's core advantage was without clicking
in and exploring. The narrative sections (Talk/Test/Race) each describe an act
but never state the problem being solved or who it is for.

Also pins the two claims that must never appear. Both contradict the code: no
order-submission route exists on any surface, and ROBINHOOD_EXECUTE defaults to
false. docs/source/lab/operating_modes.rst says the same. Copy that promises
either would be the fabricated-Performance-Drivers failure again.
"""

from pathlib import Path

_LANDING_SRC = Path(__file__).resolve().parents[2] / "landing" / "src"
_BAND = _LANDING_SRC / "components" / "home" / "WhyCare.tsx"
_PAGE = _LANDING_SRC / "pages" / "landing-page.tsx"


def test_band_component_exists():
    assert _BAND.is_file()


def test_band_is_rendered_between_hero_and_talk():
    page = _PAGE.read_text(encoding="utf-8")
    assert "WhyCare" in page
    assert page.index("<Hero />") < page.index("<WhyCare />") < page.index("<Talk />")


def test_band_states_the_problem_before_the_features():
    body = _BAND.read_text(encoding="utf-8")
    assert "Testing it properly is the expensive part" in body


def test_band_covers_the_three_acts():
    body = _BAND.read_text(encoding="utf-8")
    for heading in ("Describe it in plain English", "Prove it on real market data", "See how it ranks"):
        assert heading in body


def test_band_names_the_uncovered_capabilities():
    """Model choice and external agents are real and were absent from the landing."""
    body = _BAND.read_text(encoding="utf-8")
    assert "Pick the model" in body
    assert "Bring your own agent" in body


def test_band_makes_no_paper_trading_claim():
    body = _BAND.read_text(encoding="utf-8").lower()
    assert "paper trading" not in body
    assert "paper-trade" not in body


def test_band_makes_no_real_capital_claim():
    body = _BAND.read_text(encoding="utf-8").lower()
    for phrase in ("real capital", "real money", "go live", "trade live"):
        assert phrase not in body


def test_hero_scroll_anchor_still_resolves():
    """Hero.tsx scrolls to #landing-stats. If the band takes that anchor, Talk
    must give it up -- two elements with one id is a silent mis-scroll."""
    sources = [p.read_text(encoding="utf-8") for p in _LANDING_SRC.rglob("*.tsx")]
    total = sum(s.count('id="landing-stats"') for s in sources)
    assert total == 1, f"expected exactly one #landing-stats anchor, found {total}"
    assert 'id="landing-stats"' in _BAND.read_text(encoding="utf-8")


_TALK = _LANDING_SRC / "components" / "home" / "Talk.tsx"


def test_talk_leads_with_the_on_site_path():
    """The heading no longer sells Discord as the way in. On-site plain-English
    authoring has existed since the agent editor shipped (app.html:972)."""
    body = _TALK.read_text(encoding="utf-8")
    assert "Talk to agents on Discord" not in body
    assert "Describe your idea" in body


def test_talk_keeps_discord_as_an_alternative():
    """Reframed, not removed -- the Discord path works and some users prefer it."""
    assert "Discord" in _TALK.read_text(encoding="utf-8")


def test_talk_keeps_its_anchor_and_visual():
    body = _TALK.read_text(encoding="utf-8")
    assert 'id="talk"' in body
    assert "<DiscordMock />" in body


def test_talk_has_exactly_one_section_label():
    """Step 3's replacement block *re-includes* the `01 — Talk` mono-label, so
    pasting it below the existing one stacks two identical labels. Every other
    assertion in this file is a substring check and would stay green."""
    assert _TALK.read_text(encoding="utf-8").count("01 — Talk") == 1


_FOOTER = _LANDING_SRC / "components" / "home" / "FooterCTA.tsx"


def test_footer_has_no_dead_links():
    """Three href="#" anchors shipped since the 2026-07-25 audit. A link that
    goes nowhere costs more trust than an absent one."""
    assert 'href="#"' not in _FOOTER.read_text(encoding="utf-8")


def test_footer_documentation_points_at_the_docs_site():
    body = _FOOTER.read_text(encoding="utf-8")
    assert "finagent-orchestration.readthedocs.io" in body


def test_footer_external_link_is_safe():
    """target=_blank without rel=noopener hands the opener window to the target."""
    body = _FOOTER.read_text(encoding="utf-8")
    if 'target="_blank"' in body:
        assert "noopener" in body
