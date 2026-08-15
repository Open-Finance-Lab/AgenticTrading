"""The landing states what the product is for, above the three-act narrative.

A tester could not tell what the platform's core advantage was without clicking
in and exploring. The narrative sections (Talk/Test/Race) each describe an act
but never state the problem being solved or who it is for.

Also pins the two claims that must never appear here. Corrected 2026-08-15: the
old rationale said "no order-submission route exists on any surface", which is
stale — `execution/robinhood_live_service.py` is the live-money path and
`api/routers/robinhood_live.py` mounts it. It is armed only by ROBINHOOD_EXECUTE,
which defaults false, and it is a separate opt-in per-user path, not something
this band's subject (the Lab's simulated Talk/Test/Race flow) reaches. So the ban
is on implying *these* flows trade real money, not on the platform having a
brokered path at all. docs/source/lab/operating_modes.rst draws the same line.
Copy that blurs it would be the fabricated-Performance-Drivers failure again.
"""

import re
from pathlib import Path

_LANDING_SRC = Path(__file__).resolve().parents[2] / "landing" / "src"
_BAND = _LANDING_SRC / "components" / "home" / "WhyCare.tsx"
_PAGE = _LANDING_SRC / "pages" / "landing-page.tsx"
_INDEX_CSS = _LANDING_SRC / "index.css"
_NAVBAR = _LANDING_SRC / "components" / "home" / "Navbar.tsx"

#: Tailwind's spacing scale is 0.25rem per unit, so `scroll-mt-40` == 160px.
_TAILWIND_UNIT_PX = 4


def test_band_component_exists():
    assert _BAND.is_file()


def test_band_is_rendered_between_hero_and_talk():
    page = _PAGE.read_text(encoding="utf-8")
    assert "WhyCare" in page
    assert page.index("<Hero />") < page.index("<WhyCare />") < page.index("<Talk />")


def test_band_states_the_problem_before_the_features():
    """Ordering, not merely presence -- stating the problem first is the band's
    whole job, and a presence check would stay green if it were moved below the
    feature grid.

    Compared inside the component body: ACTS/EXTRAS are declared above the JSX,
    so a whole-file index comparison would measure declaration order rather
    than render order and could never fail.
    """
    body = _BAND.read_text(encoding="utf-8")
    jsx = body[body.index("export function WhyCare()") :]
    assert "Testing it properly is the expensive part" in jsx
    assert jsx.index("Testing it properly is the expensive part") < jsx.index("ACTS.map(")


def test_band_covers_the_three_acts():
    body = _BAND.read_text(encoding="utf-8")
    for heading in ("Describe it in plain English", "Prove it on real market data", "See how it ranks"):
        assert heading in body


def test_band_names_the_uncovered_capabilities():
    """Model choice and external agents are real and were absent from the landing."""
    body = _BAND.read_text(encoding="utf-8")
    assert "Pick the AI model" in body
    assert "For developers: bring your own agent" in body


def test_band_makes_no_paper_trading_claim():
    """Matched as a pattern, not a literal list.

    The first cut pinned "paper trading" and "paper-trade", which let the most
    natural singular -- "paper trade" -- straight through. A guard whose only
    job is intercepting one phrase has to cover the phrase's whole inflection,
    or it reads as coverage it does not have.
    """
    body = _BAND.read_text(encoding="utf-8").lower()
    hit = re.search(r"paper[\s\-]?trad", body)
    assert hit is None, f"band claims paper trading: {hit.group(0)!r}"


def test_band_makes_no_real_capital_claim():
    """Same reasoning as the paper-trading guard: shapes, not a phrase list.

    The bare phrase "live trading" used to be on this list. It came off on
    2026-08-15: "Live Trading Leaderboard" is now a board name, so banning the
    words would make *naming the product here* a test failure while the claim
    itself walked in through any other wording. What is banned is the claim —
    turning live trading on, connecting a broker — not the noun.
    """
    body = _BAND.read_text(encoding="utf-8").lower()
    for pattern in (
        r"real (capital|money|cash|funds|dollars)",
        r"go live",
        r"trade live",
        r"turn on live trading",
        r"connect (a|an|your) brokerage",
    ):
        hit = re.search(pattern, body)
        assert hit is None, f"band claims real capital: {hit.group(0)!r}"


def test_hero_scroll_anchor_still_resolves():
    """Hero.tsx scrolls to #landing-stats. If the band takes that anchor, Talk
    must give it up -- two elements with one id is a silent mis-scroll."""
    sources = [p.read_text(encoding="utf-8") for p in _LANDING_SRC.rglob("*.tsx")]
    total = sum(s.count('id="landing-stats"') for s in sources)
    assert total == 1, f"expected exactly one #landing-stats anchor, found {total}"
    assert 'id="landing-stats"' in _BAND.read_text(encoding="utf-8")


def test_hero_scroll_anchor_clears_the_fixed_chrome():
    """Resolving to one element is not the same as landing somewhere readable.

    `scroll-margin` applies to the element `scrollIntoView()` targets, not to
    its ancestors -- it is not inherited. The section's own `scroll-mt-40` does
    nothing for Hero's gesture, which targets the zero-height #landing-stats
    div *inside* it; a bare div's scroll-margin-top is 0, so the band parks at
    viewport y=0, under the fixed `.landing-chrome`. Measured before the fix:
    80 of 80 headline px hidden at 1440x900, 123 of 144 at 390x844.

    Strictly greater, not >=: --landing-chrome-height is a rounded proxy (its
    own comment says the real stack is ~118px) and the chrome measures 122.5px
    with its border, so a scroll margin equal to the declared value still
    clips.

    Anchored on the id rather than pinned as a whole tag, so reordering classes
    or attributes cannot silently pass this.
    """
    body = _BAND.read_text(encoding="utf-8")
    tag = body[body.index('<div id="landing-stats"') :].split(">")[0]
    scroll_mt = re.search(r"scroll-mt-(\d+)", tag)
    assert scroll_mt, f"#landing-stats needs a scroll-mt-* class or Hero's scroll hides the band: {tag}"

    declared = re.search(r"--landing-chrome-height:\s*(\d+)px", _INDEX_CSS.read_text(encoding="utf-8"))
    assert declared, "--landing-chrome-height moved; this guard no longer measures anything"
    margin_px = int(scroll_mt.group(1)) * _TAILWIND_UNIT_PX
    chrome_px = int(declared.group(1))
    assert margin_px > chrome_px, f"anchor clears only {margin_px}px of {chrome_px}px chrome"


def test_band_is_reachable_from_the_navbar():
    """An id nothing links to is decorative. Talk/Test/Race each have a nav
    entry; the band that introduces them shipped without one."""
    body = _NAVBAR.read_text(encoding="utf-8")
    assert 'href: "#why"' in body, "the band's id has no referrer"
    assert body.index('href: "#why"') < body.index('href: "#talk"'), "nav order should match page order"


def test_band_runs_no_second_step_sequence():
    """Talk/Test/Race own the 01/02/03 mono-labels. The band summarising them
    shipped its own 01/02/03, so one page ran two numbered sequences -- the
    summary competing with the narrative instead of leading into it."""
    stray = re.findall(r'"0[1-9]"', _BAND.read_text(encoding="utf-8"))
    assert not stray, f"band declares its own step numbers: {stray}"


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
    """Pins the whole href, not the bare host.

    A bare-host substring is both a weaker assertion -- it would accept any URL
    merely containing the host, including a path this link must not use -- and
    the shape CodeQL flags as py/incomplete-url-substring-sanitization. The
    /en/latest/ form is deliberate: it is what the README's docs badge
    publishes, so the link does not depend on a redirect.
    """
    body = _FOOTER.read_text(encoding="utf-8")
    assert 'href="https://finagent-orchestration.readthedocs.io/en/latest/"' in body


def test_footer_external_link_is_safe():
    """target=_blank without rel=noopener hands the opener window to the target."""
    body = _FOOTER.read_text(encoding="utf-8")
    if 'target="_blank"' in body:
        assert "noopener" in body
