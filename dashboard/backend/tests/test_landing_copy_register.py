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

import json
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


def test_the_disclaimer_survives_where_the_data_is_still_invented():
    """The label was on three cards: the hero board, the Race standings, and the
    chat mock. Two of the three now draw the LIVE Competition board, and the
    label on real numbers is its own false claim -- so the count is 1, not 3,
    and asserting >= 2 would force the disclaimer back onto real data.

    The chat mock is still a mock, so it keeps it. Pinning that specifically,
    rather than dropping the guard, is what stops the next edit from removing the
    one place it is still true."""
    text = _shipped_text()
    assert text.count("Illustrative example") == 1, (
        "exactly one card is still illustrative — the chat simulation"
    )
    chat = (
        Path(__file__).resolve().parents[2]
        / "landing" / "src" / "components" / "home" / "ChatSimulation.tsx"
    ).read_text(encoding="utf-8")
    assert "Illustrative example" in chat


def test_the_board_cards_name_the_window_they_draw():
    """What replaced the disclaimer on the two board cards. They now draw real
    entries over a real window, and the window is the one detail that must not be
    left implicit -- the forward arrow under the chart otherwise reads as a claim
    that this window is still running, when it closed on 2026-05-15.

    SCOPED PER COMPONENT, because counting occurrences in the merged bundle
    cannot attribute them. EACH card emits the literal TWICE -- once in the
    interpolated branch, once in the fallback branch, and neither branch is
    dead-code-eliminable -- so the bundle count is 4 and the old `>= 2` was met
    by either card ALONE. Verified by mutation: dropping the words from Race's
    chip, so the standings card renders the bare window label and no
    provenance, left the whole register green at 43 passed. "Competition
    Standings" would then publish real returns from a fixed historical window
    with nothing on the card dating them.

    The bundle assertion stays as this file's usual rebuild check -- source that
    was edited but never built into ../frontend/ is the register's own thesis --
    but it is no longer the claim. Comments are stripped on both files first:
    BoardPreview.tsx's chip carries a five-line comment explaining what the
    provenance chip replaced, and a guard that reads its own rationale as
    coverage passes on the file that deleted the chip.
    """
    assert "Competition window" in _shipped_text(), (
        "the chip is in neither component's built output — rebuild per "
        "dashboard/landing/README.md"
    )
    for name in ("BoardPreview.tsx", "Race.tsx"):
        body = _BLOCK_COMMENT.sub(
            "", (_LANDING_HOME / name).read_text(encoding="utf-8")
        )
        assert "Competition window" in body, (
            f"{name} draws real entries over a fixed historical window and no "
            f"longer states which one"
        )


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
    """Race's Standings card carried "Illustrative example" yet a pulsing green
    "Live" badge sat beside it — animating exactly the claim the label
    disclaimed. The badge's ping dot was the landing's only use of Tailwind's
    ``animate-ping``, so its absence from the shipped text means the badge (not
    merely its caption) is gone.

    The card draws the live board now, which removes the contradiction but not
    the ban: a green pulse beside a FIXED historical window would be a fresh
    claim of its own, and a worse one for being on real numbers. The surrounding
    prose may still say "live" — the Live Trading Leaderboard is a real board
    with a real name, and live *market prices* are a real product property.

    The positive assertions pin that the cards themselves still ship AND that the
    bundle text was actually read: these strings live only in the JS bundle, so a
    broken entry-bundle reference cannot turn the negative check vacuous (shown
    by fault injection during review).

    THE ANCHOR MOVED FROM "Standings" TO "Contender" because the card did. Race's
    "Competition Standings" card was retired when its table moved into the hero
    card, and the hero table heads that column "Contender" -- it ranks benchmarks
    beside models, so it may not call them all AI models (see
    test_the_standings_table_does_not_present_a_benchmark_as_an_ai_model). The
    ban itself is unchanged and is still global over the bundle: an animated
    pulse beside a FIXED historical window is a fresh claim wherever it is
    rendered, and the hero card is a worse place for it than Race was."""
    text = _shipped_text()
    assert "Contender" in text and "Leaderboard" in text
    assert "animate-ping" not in text
    assert "animate-pulse" not in text or "Competition window" in text, (
        "a pulse on this card must not outlive the window label that dates it"
    )


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
    # is derived rather than hardcoded to a filename: the chart moved to
    # BoardPreview.tsx when the board was promoted into the hero, and the sample
    # rows it was previously keyed on were deleted when the board went live. The
    # anchor is now the hook both board components read, which is the strongest
    # version of this check yet -- a component drawing a curve it did NOT get
    # from the API is exactly the thing being banned.
    # Keyed on the CALL, not the import: a component that imports the hook and
    # then renders a literal is the thing being banned, and it would otherwise
    # stay in this set and keep the ban vacuous against itself.
    board = {name: body for name, body in bodies.items() if "useLeaderboard()" in body}
    assert board, "no component reads the live board"
    corpus = "".join(board.values())
    assert "dataKey=" in corpus, "one of them must actually draw curves"
    assert "SAMPLE_STANDINGS" not in corpus and "SAMPLE_CURVES" not in corpus, (
        "a board component that carries its own rows is drawing something the "
        "API did not send"
    )


def test_the_hero_standings_render_the_full_selection_baselines_included():
    """ADJUDICATED BY THE CONTROLLER, pinned rather than left to drift back: the
    standings table includes buy_hold_djia and djia_index alongside the models,
    unlike /app's home CHART rank list, which is models-only. Three reasons,
    restated in the BoardPreview.tsx comment beside the table: (1) the
    dashboard's own Competition Leaderboard tab ranks all twelve entries
    including the baselines -- it is the home CHART rank list that is
    models-only, and this card is a board, not that list; (2) the chart on this
    page already draws both baselines as dashed curves, so a row-less curve
    would be a dangling reference; (3) most of the models lost to buy-and-hold,
    and a models-only table would silently make the page more flattering than
    the truth -- the exact failure the copy guards in this file exist to
    prevent.

    THE TABLE MOVED FROM Race.tsx INTO THE HERO CARD and this guard moved with
    it. It was written against Race because Race was where the ranking lived;
    the hero card carried a chip strip and Race carried the table. The hero card
    carries the table now and Race carries none, so scanning Race for
    ``standings.map`` would pin an empty invariant on a file that no longer
    renders a row -- green forever, guarding nothing. The ruling is unchanged;
    only the file that has to obey it moved.

    `selectBoardEntries` (pinned separately in
    test_select_board_entries_returns_nine_of_twelve_models_first_then_baselines,
    test_landing_live_board.py) already seeds `board.data.standings` with both
    baselines; what this guard pins is that the render does not narrow that list
    back down -- e.g. a `standings.filter((s) => ...)` inserted ahead of the
    `.map` would compile cleanly and pass every other guard in this file while
    silently dropping the baseline rows.

    THE BAN IS ON THE OPERATION, NOT ON A RECEIVER NAMED `standings`. Requiring
    `standings` to be the immediate receiver pinned one spelling of the edit and
    left at least three others: filtering the TERNARY instead --
    `const standings = (board.status === "ready" ? board.data.standings :
    []).filter(...)`, whose receiver is `)` -- rebinding first
    (`standings0.filter(...)`), and `board.data.standings.slice(0, 7)`. The
    ternary form was reproduced end to end: `npm run typecheck` clean and the
    five landing suites green, with the models-only table the controller ruled
    against.

    WHOLE-FILE, AND AN EARLIER DRAFT OF THIS CASE SCOPED IT TO THE TABLE AND
    FAILED OPEN. The reasoning for narrowing was that Race.tsx rendered exactly
    one collection so a file-wide ban cost nothing there, while BoardPreview.tsx
    also owns the chart and might legitimately need those methods on `series` or
    `times`. Both halves were true and the conclusion was still wrong: the scan
    started at the table header, and `standings` is DERIVED about 370 lines
    above it. The one edit this case exists to catch --
    `const standings = (data?.standings ?? []).filter((s) => s.isModel)` --
    sits outside the scanned region. Reproduced end to end: that one line drops
    both baselines from the table, leaves the chart's two dashed curves unnamed,
    keeps `tableCoverage` self-consistent because it reads the same narrowed
    array, and left the three landing modules at 107 passed.

    The chart's need was hypothetical and the file still contains zero
    `.filter(`/`.slice(` calls, so the whole-file ban costs nothing today. If a
    chart-side call is ever genuinely needed, exclude that call site by name --
    do not re-narrow the window and reopen 370 lines. Comments are stripped so
    neither this prose nor the component's own "do not add a filter here" note
    can trip it."""
    source = _BLOCK_COMMENT.sub(
        "", (_LANDING_HOME / "BoardPreview.tsx").read_text(encoding="utf-8")
    )
    assert 'data-testid="board-rank-head"' in source, (
        "the standings table header is gone from BoardPreview.tsx — the ranking "
        "moved again, and this guard has to move with it"
    )
    assert "standings.map(" in source, (
        "BoardPreview.tsx no longer maps the full standings array"
    )
    narrowing = re.search(r"\.\s*(filter|slice)\(", source)
    assert not narrowing, (
        f"BoardPreview.tsx narrows a collection before rendering it "
        f"({narrowing.group(0)!r} at offset {narrowing.start()}); the baseline "
        f"rows the controller ruled must stay on the board are the only thing "
        f"this component can narrow away"
    )


def test_the_race_headline_is_derived_from_the_board_it_sits_beside():
    """The section opened with "Seven leading AI models traded the same days with
    simulated money, ranked against buy-and-hold and the index. Only one finished
    ahead of both." -- a count and an outcome, typed into the page, directly above
    a table that is now live off the same payload.

    Both halves were falsifiable by ordinary changes. An eighth `llm_agent` entry
    in dashboard/config/leaderboard.json is the documented way the roster reached
    seven, and `test_the_illustrative_run_report_names_no_real_roster_model` was
    written to absorb exactly that automatically -- it would have left this
    sentence saying "Seven" beside eight rows. A re-run that put a second model
    ahead of buy-and-hold falsifies the second half with the counter-evidence
    rendered beside it. On the page's most checkable claim, on the highest-traffic
    anonymous surface, with no guard anywhere.

    The number-word list is not banned -- the sentence still spells its counts --
    so this pins the DERIVATION: the counts must come from the payload, and the
    two retired literals must not come back.

    COMMENTS ARE STRIPPED, like the standings guard above. Race.tsx's own note
    quotes the retired sentence verbatim to say what was wrong with it, and a raw
    substring scan reddens on that -- which reads as the ban working and invites
    deleting the explanation that is the whole reason the ban exists. The bundle
    half needs no stripping: esbuild drops comments, so a retired string surviving
    there is genuinely rendered copy."""
    source = _BLOCK_COMMENT.sub("", _RACE_TSX.read_text(encoding="utf-8"))
    assert "boardHeadlineCounts" in source, (
        "the headline's counts must come from the board, not from the source text"
    )
    assert "headlineSentence(standings)" in source, (
        "the rendered sentence must be built from the standings actually shown"
    )
    for retired in ("Seven leading AI models", "Only one finished ahead of both"):
        assert retired not in source, (
            f"{retired!r} is a hardcoded claim about live data; derive it"
        )
        assert retired not in _shipped_text(), (
            f"{retired!r} still ships in the bundle — rebuild per "
            f"dashboard/landing/README.md"
        )


def test_the_standings_table_does_not_present_a_benchmark_as_an_ai_model():
    """The table deliberately ranks buy_hold_djia and djia_index alongside the
    models (pinned above), and most of the models lost to buy-and-hold -- so on
    the live board the `#1` row IS a benchmark. In Race it was rendered in the
    brand accent, under a column headed "AI model", beneath a heading reading
    "What the AI models actually returned", with nothing marking it as a
    reference curve. The chart distinguishes those two with a dash pattern; the
    table had no equivalent, so three signals at once told a visitor the passive
    index was the leading AI model.

    What is banned is the column header calling every row an AI model, with no
    per-row tag to say otherwise.

    THE PRESSURE WENT UP WHEN THE TABLE MOVED INTO THE HERO CARD, which is why
    this guard follows it rather than being retired with Race's table. /app's
    rank list -- the visual this one is ported from -- heads that column "AI
    Model", and it is right to: `isHomeModelEntry` filters its rows to models.
    Copying the markup without copying that filter is the single most likely
    way this regresses, and it would land the false header on the page's
    highest-traffic surface, above the fold, in the first thing a visitor sees.

    The `#1` accent is NOT what this bans, in either card. That buy-and-hold
    came first is the honest, unflattering fact the board exists to show, and
    moving the highlight to the best model would be the flattery every guard in
    this file is against."""
    source = _BLOCK_COMMENT.sub(
        "", (_LANDING_HOME / "BoardPreview.tsx").read_text(encoding="utf-8")
    )
    head = re.search(
        r'data-testid="board-rank-head"(.*?)</div>', source, re.S
    )
    assert head, "the standings header row is gone"
    labels = [
        text.strip()
        for text in re.findall(r">([^<>{}]+)<", head.group(1))
        if text.strip()
    ]
    assert labels, "the standings header row has no column labels"
    assert not any(label.lower() == "ai model" for label in labels), (
        f"the column holds benchmarks too; heading it 'AI model' publishes "
        f"buy-and-hold as the leading model. Found {labels}"
    )
    assert "Contender" in labels, (
        f"the mixed field needs a word that covers both; found {labels}"
    )
    # POLARITY, NOT PRESENCE. `"item.isModel" in source` and
    # `"Benchmark" in source` are both satisfied by the inverted ternary
    # `{!item.isModel ? null : <span>Benchmark</span>}`, which tags all seven
    # models and leaves buy_hold_djia -- the actual #1 row -- untagged under the
    # medal accent. That is this case's own failure, wearing its own assertions.
    # The tag belongs on the FALSE arm: a row is tagged when it is NOT a model.
    assert re.search(r"item\.isModel\s*\?\s*null\s*:", source), (
        "the Benchmark tag must hang off the false arm of `item.isModel` — an "
        "inverted ternary tags every AI model and leaves the benchmarks bare, "
        "which is this guard's own failure mode with its assertions still green"
    )
    assert "Benchmark" in source, "the per-row benchmark tag is gone"
    assert "Benchmark" in _shipped_text(), (
        "the benchmark tag is in the source but not in the bundle — rebuild per "
        "dashboard/landing/README.md"
    )


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


# ---------------------------------------------------------------------------
# §02's illustrative run report vs. the live board four screens up.
#
# CONTROLLER RULING, minimal form. Test.tsx/storyline.ts predate the live-board
# change; what that change did was make them FALSIFIABLE -- before it, nothing
# on this page could be checked against anything. The page is what ships, and as
# it stood it stated two different results for the SAME named model over the
# SAME stated window and the SAME base: the hero said Claude Sonnet 4.6 returned
# +0.11% over 2026-04-15 → 2026-05-15 from $10,000, and §02 said +14.2% over
# "Apr 15 – May 15, 2026" from $10,000, on a dollar axis the hero itself refuses
# because a dollar level names an account that never existed.
#
# The fix is narrow and deliberately reversible: §02 keeps its numbers, its
# dollar axis, its layout and its "Illustrative" chip, and stops naming a real
# roster model and the real contest window. That removes the comparison a
# visitor can make; it does not pass judgement on the section.
#
# Derived from dashboard/config/leaderboard.json rather than hardcoded, so an
# eighth LLM entry -- the documented way the roster grew to seven -- extends the
# ban without anyone remembering to.
# ---------------------------------------------------------------------------

_LANDING_HOME = (
    Path(__file__).resolve().parents[2] / "landing" / "src" / "components" / "home"
)
_LEADERBOARD_CONFIG = Path(__file__).resolve().parents[2] / "config" / "leaderboard.json"
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _story_sources() -> dict[str, str]:
    """The three files that render the Talk → Test storyline."""
    return {
        name: (_LANDING_HOME / name).read_text(encoding="utf-8")
        for name in ("storyline.ts", "Test.tsx", "DiscordMock.tsx")
    }


def _leaderboard_config() -> dict:
    return json.loads(_LEADERBOARD_CONFIG.read_text(encoding="utf-8"))


def test_the_illustrative_run_report_names_no_real_roster_model():
    """A fabricated +14.2% under a real model's name, four screens below that
    model's real return on the live board, is a claim a visitor can falsify by
    scrolling. Baseline names (DJIA, Buy & Hold, S&P 500) are NOT banned -- the
    storyline legitimately names its benchmarks, and they are not the entries
    whose results are being contradicted."""
    roster = sorted(
        {
            (s.get("model") or s.get("name") or "").strip()
            for s in _leaderboard_config().get("strategies", [])
            if s.get("strategy") == "llm_agent"
        }
        - {""}
    )
    assert len(roster) >= 5, (
        f"derived from the live roster and it came back nearly empty ({roster!r}) "
        f"— this guard would be vacuous"
    )
    for name, source in _story_sources().items():
        for model in roster:
            assert model not in source, (
                f"{name} names the roster model {model!r}; the live board four "
                f"screens up publishes that model's real return, so a fabricated "
                f"one here is falsifiable by scrolling"
            )


def test_the_illustrative_run_report_does_not_restate_the_contest_window():
    """The other half of the comparison. Same model AND same window is what makes
    the two numbers commensurable; breaking either breaks the comparison, and the
    window is the cheaper one to break.

    The CHART's own window counts, not only the stated one: Test.tsx builds its
    x-axis day labels from a hardcoded `Date.UTC(...)` pair, so leaving that on
    the contest window while restating the window elsewhere would put "Apr 15 …
    May 15" back under the figure -- and would also make the settings card and
    the chart beneath it name different months, which is a fresh contradiction
    rather than a fix."""
    config = _leaderboard_config()
    start, end = config["start_date"], config["end_date"]
    y0, m0, d0 = (int(p) for p in start.split("-"))
    y1, m1, d1 = (int(p) for p in end.split("-"))
    human_forms = {
        f"{_MONTHS[m0 - 1]} {d0} {dash} {_MONTHS[m1 - 1]} {d1}, {y1}"
        for dash in ("–", "—", "-", "to")
    }
    for name, source in _story_sources().items():
        assert "STORY" in source or "storyline" in source, f"{name} read empty?"
        assert start not in source, f"{name} restates the contest start date {start}"
        assert end not in source, f"{name} restates the contest end date {end}"
        for form in human_forms:
            assert form not in source, f"{name} restates the contest window as {form!r}"
        assert not re.search(
            rf"Date\.UTC\(\s*{y0}\s*,\s*{m0 - 1}\s*,\s*{d0}\s*\)", source
        ), f"{name} draws its chart over the contest window's start date"
        assert not re.search(
            rf"Date\.UTC\(\s*{y1}\s*,\s*{m1 - 1}\s*,\s*{d1}\s*\)", source
        ), f"{name} draws its chart over the contest window's end date"


def test_the_illustrative_run_report_still_labels_itself_illustrative():
    """The counterweight. Nothing above says §02 must be deleted or defanged --
    it keeps its numbers and its dollar axis. What it must keep is the label that
    says they are not a result, so a later edit cannot close this by quietly
    dropping the chip instead of the impersonation."""
    assert "Illustrative" in _story_sources()["Test.tsx"]


def test_the_illustrative_placeholders_source_and_shipped_bundle_agree():
    """The three cases above read ``landing/src``; prod reads the bundle.

    So all three stay green against a bundle built before the placeholders
    landed -- and that bundle still ships "Claude Sonnet 4.6" over the real
    contest window, which is the whole defect they were written to close.
    Measured: the full landing suite (156 cases) passed on the pre-rebuild
    bundle, so nothing at all made the missing ``npm run build`` red.

    Same shape as ``test_race_source_and_shipped_bundle_agree`` and for the same
    reason -- anchor one string on BOTH sides. Both values are read out of
    ``storyline.ts`` rather than hardcoded, so changing the placeholder to some
    other non-roster model or window keeps this honest by itself; a *bare*
    absence check could not, because the bundle legitimately carries every roster
    model name for the live board four screens up.
    """
    source = _BLOCK_COMMENT.sub("", _story_sources()["storyline.ts"])
    stated = {
        field: re.search(rf'\b{field}:\s*"([^"]+)"', source)
        for field in ("model", "timePeriod")
    }
    missing_in_source = sorted(f for f, m in stated.items() if not m)
    assert not missing_in_source, (
        f"storyline.ts no longer declares {missing_in_source} as a plain string "
        "literal, so this guard can no longer see which placeholders ship — "
        "re-point it at however STORY_SPECS now states the model and the window."
    )

    shipped = _shipped_text()
    absent = sorted(m.group(1) for m in stated.values() if m.group(1) not in shipped)
    assert not absent, (
        f"storyline.ts states {absent} but the shipped bundle does not carry "
        "them, so prod still renders the previous illustrative run report — "
        "rebuild per dashboard/landing/README.md"
    )
