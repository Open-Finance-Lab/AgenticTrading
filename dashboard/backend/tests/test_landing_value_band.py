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
