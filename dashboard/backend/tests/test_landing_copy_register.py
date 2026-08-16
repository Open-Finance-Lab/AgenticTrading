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

_NO_REAL_MONEY_SENTENCE = "No real money. Simulated money only."


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
    # Race's pre-2026-08-15 pitch. Each of these described a product that does
    # not exist: entries come from the curated `config/leaderboard.json` roster,
    # so no user agent is on any board and nothing "climbs"; the Competition
    # board is a fixed historical window, so its prices are not live and its
    # rankings do not update; and "paper trading on live markets" reads as
    # brokered realtime execution, which is an explicit non-goal (PR #328) with
    # `execution/paper_backend.py` still a stub.
    "Race on the live leaderboard",
    "Paper trading on live markets",
    "Rankings update as agents trade",
    "Live market prices — no real money at risk",
    "climb against the community",
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


_RACE_TSX = (
    Path(__file__).resolve().parents[2] / "landing" / "src" / "components" / "home" / "Race.tsx"
)


def test_race_names_the_two_boards_the_app_actually_serves():
    """The landing sold one "live leaderboard"; the app serves two boards with
    different contracts, and conflating them is what let the old copy promise a
    live race over a fixed historical window. Both names ship."""
    text = _shipped_text()
    assert "Live Trading Leaderboard" in text
    assert "Competition" in text


def test_race_discloses_that_the_live_board_is_not_ranking_yet():
    """Naming the board on the acquisition page while its nightly advance is
    undeployed is the landing-side version of the preview banner. Without this
    sentence the bullet above it ("runs forward in two-week seasons") reads as a
    board that is running now — the same over-claim the banner exists to stop,
    just moved one page upstream where nothing renders the banner."""
    text = _shipped_text()
    assert "in preview for Season 0" in text
    assert "Season 1 is the first that counts" in text


_LANDING_HOME = _RACE_TSX.parent

# Sentences that contain a banned phrase because they *deny* the claim, listed
# verbatim and stripped before the scan. An allowlist rather than a narrower
# scope: the WhyCare guard in test_landing_value_band.py banned
# `paper[\s-]?trad` inside WhyCare.tsx alone, and the claim simply moved to the
# unguarded neighbour — Race shipped "Paper trading on live markets" for the
# whole time the band was pinned clean. Scoping the replacement to Race would
# have moved it one door further. Adding an entry here is a visible decision;
# leaving a file unguarded is not.
_CLAIM_DISCLAIMERS = (
    # Hero's standing safety line, pinned verbatim by this suite (above) and by
    # test_landing_chart_first. It denies the claim outright rather than stating
    # the brokered path's condition -- accurate for what the landing sells, which
    # is simulated end to end (`execution/paper_backend.py` is still a stub). The
    # opt-in brokered path does exist (api/routers/robinhood_live.py, mounted at
    # api/router.py; ROBINHOOD_EXECUTE defaults false), and the /app surface still
    # carries the longer conditional sentence that states its condition -- see
    # test_frontend_shelves, which pins it there. Banning the phrase this line
    # needs would delete the disclaimer in order to satisfy the guard against the
    # claim.
    "No real money. Simulated money only.",
    # Hero's gloss on what a Lab paper-trading run is. Accurate as written: the
    # prices are real, the money is not, and the sentence says exactly that.
    '<span className="text-primary font-semibold">paper trading</span> '
    '{" "}— practice trading with simulated money at live market prices —',
)

# Claim shapes, not vocabulary. "live trading" is deliberately absent: it is now
# a board name ("Live Trading Leaderboard"), so banning the bare phrase would
# make naming the product a test failure — the mirror image of the scoping bug
# above, and the reason the sibling guard in test_landing_value_band.py was
# changed to match the claim instead.
_BROKERED_CLAIM_PATTERNS = (
    r"paper[\s\-]?trad",
    r"real (capital|money|cash|funds|dollars)",
    r"go live",
    r"trade live",
    r"turn on live trading",
    r"connect (a|an|your) brokerage",
)


def test_no_landing_component_claims_brokered_or_real_capital_trading():
    """Nothing on the narrative path puts real capital at risk — say nothing else.

    Not because brokered execution does not exist: it does
    (`api/routers/robinhood_live.py`), behind `ROBINHOOD_EXECUTE`, which defaults
    false. It is a separate, opt-in, per-user path. What the landing sells — Talk
    → Test → Race, the boards, the playground — is simulated throughout, and
    `execution/paper_backend.py` is still a stub, so copy implying that running an
    agent here trades real money describes something these flows do not do.

    Hero's conditional sentence is the correct way to say it, which is why it is
    allowlisted above rather than banned. Every home component is scanned because
    the one thing this class of copy reliably does is relocate: the WhyCare-scoped
    guard in test_landing_value_band.py was clean the whole time Race shipped
    "Paper trading on live markets" next door.
    """
    components = sorted(_LANDING_HOME.glob("*.tsx"))
    assert components, "no landing components found — the glob is wrong"
    for component in components:
        body = " ".join(component.read_text(encoding="utf-8").split())
        for disclaimer in _CLAIM_DISCLAIMERS:
            body = body.replace(disclaimer, " ")
        lowered = body.lower()
        for pattern in _BROKERED_CLAIM_PATTERNS:
            hit = re.search(pattern, lowered)
            assert hit is None, (
                f"{component.name} claims brokered/real-capital trading: "
                f"{hit.group(0)!r}. If the phrase is part of a disclaimer, add "
                f"the sentence to _CLAIM_DISCLAIMERS rather than narrowing the scan."
            )


def test_the_disclaimer_allowlist_is_not_stale():
    """Non-vacuity: an allowlist entry that no longer matches silently re-arms.

    If the wording is edited, the stale entry stops stripping anything and the
    test above starts failing on a sentence that was always fine — which reads as
    the guard being broken and invites deleting it.

    Scanned across every component rather than Hero.tsx alone. Both sentences
    started in Hero; the second travelled to ChatSimulation.tsx when the board
    took the hero's right column and the conversation demo moved down to the
    Talk act. A file-scoped freshness check turns any such relocation into a
    failure that looks like a deleted disclaimer, when the disclaimer is right
    there one file over — and the pressure then is to drop the allowlist entry,
    which re-arms the ban on a sentence that must keep shipping.
    """
    bodies = {
        path.name: " ".join(path.read_text(encoding="utf-8").split())
        for path in sorted(_LANDING_HOME.glob("*.tsx"))
    }
    assert bodies, "no landing components found — the glob is wrong"
    for disclaimer in _CLAIM_DISCLAIMERS:
        assert any(disclaimer in body for body in bodies.values()), (
            f"allowlisted disclaimer no longer ships verbatim: {disclaimer[:60]!r}…"
        )


def test_no_landing_component_puts_a_user_agent_on_the_board():
    """Board entries come from the curated `config/leaderboard.json` roster.

    The prose was corrected to drop "race your agent", but the illustration kept
    the story agent highlighted at rank 2 and drawn as the thickest curve — a
    picture makes the entry-flow promise more vividly than the sentence that was
    removed, and the fragment ban above only reads prose.

    Scanned across every component, not Race.tsx alone. The chart and its sample
    rows moved to BoardPreview.tsx when the board was promoted into the hero,
    and a guard pinned to the file the drawing *used* to live in is the same
    defect as the WhyCare-scoped paper-trading ban: it stays green while the
    claim redraws itself next door. The whole point is that no component may
    draw a user curve, wherever the drawing lives.
    """
    # Comments stripped first. The source explains at length *why* there is no
    # "yours" curve, and a guard that reads its own rationale as a violation is
    # the same defect one level up — it fails on the fix and passes on silence.
    components = sorted(_LANDING_HOME.glob("*.tsx"))
    assert components, "no landing components found — the glob is wrong"
    bodies = {
        component.name: _BLOCK_COMMENT.sub("", component.read_text(encoding="utf-8"))
        for component in components
    }

    # The story agent is not banned page-wide — Test.tsx is its home, and a
    # backtest run report is exactly where a named user agent belongs. What is
    # banned is its arrival anywhere else, which is how it would reach a board.
    # Asserting the *set* rather than per-file absence is what makes this survive
    # the components being reorganised: a new file that names it changes the set.
    naming_story_agent = {name for name, body in bodies.items() if "STORY_AGENT_NAME" in body}
    assert naming_story_agent == {"Test.tsx"}, (
        f"the storyline agent belongs to the Test run report only; found in "
        f"{sorted(naming_story_agent)}"
    )

    for name, body in bodies.items():
        assert "yours" not in body, f"{name}: no user curve on a board no user agent is on"

    # Non-vacuity, scoped to whatever actually draws the board today. Membership
    # is derived from the shared sample rows rather than hardcoded to Race.tsx:
    # the chart moved to BoardPreview.tsx when the board was promoted into the
    # hero, and a filename-pinned check would have passed on the file that no
    # longer draws anything.
    board = {name: body for name, body in bodies.items() if "SAMPLE_STANDINGS" in body}
    assert board, "no component renders the board sample rows"
    corpus = "".join(board.values())
    assert "DeepSeek V4 Pro" in corpus and "dataKey=" in corpus


def test_race_source_and_shipped_bundle_agree():
    """The register's thesis applied to this section specifically: every other
    assertion here reads the bundle, so a Race.tsx edit that was never rebuilt
    into ../frontend/ would leave them green against stale text. Anchoring one
    string on both sides makes the missing `npm run build` the failure."""
    source = _RACE_TSX.read_text(encoding="utf-8")
    assert "Live Trading Leaderboard" in source, "Race.tsx no longer names the board"
    assert "Live Trading Leaderboard" in _shipped_text(), (
        "Race.tsx names the board but the shipped bundle does not — "
        "rebuild per dashboard/landing/README.md"
    )
