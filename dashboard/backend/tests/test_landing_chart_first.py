"""Guards for the chart-first rebuild of / (2026-08-15 spec).

These read the TSX SOURCE, not the shipped bundle. The bundle-reading guards in
test_landing_copy_register.py already catch "edited but never rebuilt"; what
they cannot catch is a layout constant, because minified Tailwind classes and
Recharts props survive the build as opaque strings that no copy guard inspects.

Heights are asserted per surface and never shared with /app: the two surfaces
have different vertical envelopes and therefore different formulas (spec §2).
"""

import re
from pathlib import Path

_HOME = (
    Path(__file__).resolve().parents[2] / "landing" / "src" / "components" / "home"
)


def _strip_comments(source: str) -> str:
    """TSX with its comments removed, so a scan reads code and never prose.

    NOT optional here, in both directions. A comment explaining *why*
    `max-w-2xl` was removed contains the string `max-w-2xl`, which trips a
    `not in` guard on a correct file; and a comment naming a class that has been
    deleted satisfies an `in` guard on a broken one. The second is the one that
    ships a regression -- and it is exactly how PR #357's claim scans went green
    against the wrong file. `<BoardPreview/>` named in a comment above the copy
    column likewise inverts the source-order check below.

    Whole-line `//` only: an inline `//` would eat the tail of any line holding
    a URL.
    """
    source = re.sub(r"\{/\*.*?\*/\}", "", source, flags=re.S)  # JSX {/* ... */}
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)  # block comment
    source = re.sub(r"(?m)^\s*//.*$", "", source)  # whole-line //
    return source


_HERO = _strip_comments((_HOME / "Hero.tsx").read_text(encoding="utf-8"))
_BOARD = _strip_comments((_HOME / "BoardPreview.tsx").read_text(encoding="utf-8"))

_BOARD_CHALLENGE = "Can you beat the strategies and baselines on the left?"
_NO_REAL_MONEY = "No real money. Simulated money only."


def _collapse(source: str) -> str:
    """JSX text with its line breaks and indentation collapsed, so a sentence
    split across lines by the formatter still matches as one string."""
    return re.sub(r"\s+", " ", source)


def test_the_hero_lede_is_one_line_and_still_glosses_agent():
    """/ is the acquisition page: the headline uses "agent" before anything else
    defines it, and the board beside it is the only other thing above the fold.
    The gloss has to land here or not at all -- unlike /app, where the reader is
    already inside the product. So this trims to one line; it does not drop.
    """
    hero = _collapse(_HERO)
    assert "Agents here are AI trading assistants that follow your written instruction" in hero
    assert "it trades the idea hour by hour, measured against buy-and-hold and the index" not in hero, (
        "the second clause is what makes this two lines at 1/3 column width"
    )


def test_the_board_challenge_and_its_small_print_both_ship():
    """The challenge points at the board; the line under it is what stops that
    from reading as an invitation to risk anything, so the two travel together.

    The second line is pinned three times -- here, by
    test_no_real_money_sentence_is_present_verbatim (which reads the shipped
    bundle, not this source), and by the _CLAIM_DISCLAIMERS allowlist, whose
    staleness check fails if the wording drifts. It is on that allowlist because
    it contains the exact phrase the brokered-claim scan bans, in order to deny
    it: reword it without updating the allowlist and the ban re-arms on the
    disclaimer itself. Moving either line between components is fine; rewording
    is not.
    """
    hero = _collapse(_HERO)
    assert _BOARD_CHALLENGE in hero
    assert _NO_REAL_MONEY in hero


def test_the_board_column_is_two_thirds_and_uncapped():
    hero = _HERO
    assert "max-w-2xl" not in hero, (
        "672px is card width; two-thirds of a 1280px container is 853px, so this "
        "cap silently reverts the layout to what PR #357 already shipped"
    )
    assert "lg:basis-2/3" in hero
    assert "lg:basis-1/3" in hero


def test_the_columns_are_ordered_with_utilities_not_by_source_order():
    """The visual ask is chart-left / hero-right at lg:, chart first when
    stacked -- which reads as "move <BoardPreview/> above the copy in source".
    Doing that puts BoardPreview's <h2> ahead of the page's only <h1>.
    """
    hero = _HERO
    assert hero.index("<h1") < hero.index("<BoardPreview"), (
        "the h1 block must stay first in source"
    )
    # Unprefixed only. `order-first` is unconditional, so a `lg:order-first`
    # beside it restates the base class and does nothing -- and asserting both,
    # as this did, pinned the dead prefix in place: deleting it reddened CI.
    assert "order-first" in hero and "order-last" in hero
    assert "lg:order-first" not in hero and "lg:order-last" not in hero, (
        "a responsive prefix that repeats the unconditional base is dead weight; "
        "add one back only if the two orders actually differ by breakpoint"
    )


def test_the_hero_row_leaves_no_unclaimed_width():
    """The board's negative inline-start margin turns the container's left
    gutter into flex FREE SPACE -- ~152px at 1920 -- and free space in a row
    where every item is `grow-0` simply sits at the end. The copy column
    stopped short of the container's right edge with nothing able to absorb it.

    The board keeps `lg:grow-0`, so the 2/3 split above stays exactly what it
    declares; the copy column takes the slack.
    """
    assert "lg:basis-1/3 lg:grow " in _HERO or "lg:basis-1/3 lg:grow\"" in _HERO, (
        "the copy column must absorb the width the negative margin frees"
    )
    board = _HERO[_HERO.index("<motion.div") :]
    assert "lg:basis-2/3 lg:grow-0" in board, (
        "the board must not grow, or the declared two-thirds is not what renders"
    )


def test_the_chart_column_escapes_the_container_on_its_left_edge_only():
    """Both columns live inside one `container mx-auto px-6` div that also owns
    the hero's min-height contract, so this is a negative inline-start margin at
    lg: and above -- not a class removal. It is a >=1300px effect: the container
    gutter is 0px at 1280 and below.
    """
    assert "lg:ms-[calc((100%-100vw)/2)]" in _HERO


def test_the_landing_chart_uses_its_own_measured_clamp():
    """`clamp(320px, 56vh, 520px)` -- the first draft's number, shared with /app
    -- puts the card 25-46px BELOW the fold at 1440x768, 1366x768, 1280x800 and
    1280x720. All four are ordinary laptop heights. The replacement is the
    largest formula with non-negative fold slack at every tested viewport.

    Both reserves are derived, not taste, and there are TWO because the card's
    non-chart height is not one number: 218-241px beside the copy at >=lg, but
    443px stacked at 390px wide, where the title, the "Illustrative example"
    chip and the caption all wrap and the chip strip runs to five rows.

        lg+     390 = ~218-241 non-chart + 120 chrome + ~14-43 fold margin
        below   590 = 443 non-chart + 132 section padding + ~9 margin

    Measured, not derived: the single desktop constant put the card 77px past
    the fold at 390x844. RE-DERIVE BOTH if the caption, title or chip strip
    changes height -- the failure mode is a silently half-visible card, not a
    broken build. That is not hypothetical: unclipping the chip strip took it
    from 24px to 152px at 390px wide, the stacked reserve from 480 to 590, and
    then the FLOOR from 300 to 260 -- because at 844px tall the card had 269px
    left for a chart whose floor was 300, so the floor, not the reserve, was
    what put it 95px past the fold on the second pass.

    The var() indirection is load-bearing and not a tidy-up: the formula's
    commas defeat Tailwind's arbitrary-VALUE parser, so the breakpoint-dependent
    number rides an arbitrary PROPERTY instead, which does take a prefix.
    """
    board = _BOARD.replace(" ", "")
    assert "clamp(260px,calc(100dvh-var(--board-chart-reserve)),520px)" in board
    assert "[--board-chart-reserve:590px]" in board, "the stacked-phone reserve"
    assert "lg:[--board-chart-reserve:390px]" in board, "the side-by-side reserve"
    assert "56vh" not in _BOARD, "the first draft's clamp fails at four viewports"
    assert "h-[210px]" not in _BOARD and "md:h-[240px]" not in _BOARD


def test_landing_chart_axis_ticks_are_14px():
    assert _BOARD.count("fontSize={14}") == 2, "both XAxis and YAxis"
    assert "fontSize={11}" not in _BOARD, (
        "11px belongs to the gutter labels, which live in EndpointRail.tsx"
    )


def test_the_y_axis_reserve_is_measured_rather_than_guessed():
    """`width={56}` was measured against `$1030` at 11px; the tick font later
    moved to 14px and four of five labels lost their leading `$` with nothing
    failing. The axis is percent now, so the number would have to be re-measured
    anyway -- measuring it at render removes the whole class."""
    assert "width={56}" not in _BOARD
    assert "domain={[960, 1240]}" not in _BOARD, "a hardcoded dollar domain"
    # NOT a bare `"measureTextWidth" in _BOARD`. That string also appears in the
    # file's import line, so replacing the whole computation with
    # `const yAxisWidth = 60;` -- a guessed reserve, the exact regression this
    # case is named for -- left the import behind and this case GREEN (verified
    # by mutation; `noUnusedLocals` is off, so that mutant typechecks too).
    # The measurement has to reach the axis, so pin the binding AND the fact
    # that what is measured is the rendered tick text.
    assert "width={yAxisWidth}" in _BOARD, "the YAxis must take the measured width"
    assert "measureTextWidth(axisTick(" in _BOARD, (
        "the reserve must be measured from the tick text this axis actually "
        "renders, not guessed"
    )


def test_the_panel_title_is_text_xl():
    """Spec §2. The card is now two-thirds of the hero; a text-lg title reads as
    a widget label on it."""
    assert 'className="text-xl font-bold flex items-center gap-2 min-w-0"' in _BOARD


def test_the_standings_table_becomes_a_chip_strip_that_can_show_every_chip():
    """Demotion, not deletion: the chart ships no <Legend>, so the chips are the
    only thing linking a curve colour to a model name -- and they are now also
    the fallback when the endpoint rail declines to draw (a narrow card, a
    Recharts internal that moved). The full table lives in Race.tsx.

    THE STRIP MUST WRAP, and the pressure just went up: it went from five
    hardcoded entries to nine from the payload. `flex-nowrap` with
    `overflow-hidden` cut entries off the end wherever the strip was narrower
    than its content -- measured scrollWidth 910 against clientWidth 285 at 390
    (one chip survives, keying five drawn curves), 663 at 768, 895 at 1024, so
    the whole lg band and every phone, silently, because the only live-browser
    guard on it ran at 1440.
    """
    board = _BOARD
    assert "grid-cols-12" not in board, "the 5-row table is what the chart needs the height of"
    assert "flex-wrap" in board and "flex-nowrap" not in board, (
        "a legend that cannot show its entries is not a legend"
    )
    # Anchored on the JSX render site (`{standings.map`), NOT on the first
    # `standings.map` in the file. The old card mapped its rows exactly once, so
    # a bare `.index("SAMPLE_STANDINGS.map")` was the strip; the live card maps
    # `standings` three times, and the two earlier call sites (the frameLayout
    # labels and valueByKey) sit ~4.5KB above the strip. Anchoring on the first
    # put this 400-char window over the ResizeObserver effect, where
    # `overflow-hidden` can never appear -- verified by mutation: adding
    # `overflow-hidden` to the chip strip's own className left this case GREEN.
    strip = board[board.index("{standings.map") - 400 : board.index("{standings.map")]
    assert "overflow-hidden" not in strip, (
        "clipping the strip is the same failure by another route -- no scrollbar, "
        "no ellipsis, and nothing fails"
    )
    assert "text-base" in board, "text-sm rows were one of the three reported problems"
    # The identity link. `swatch` is gone with the sample rows; the colour now
    # comes off the same BoardSeries the curve is drawn from, which is stronger:
    # a row and its curve cannot disagree because there is one value.
    assert "item.color" in board
    assert "dataKey=" in board


def test_the_hero_draws_the_board_the_signed_in_home_draws():
    """The whole point of the change. No component may reintroduce a curve that
    is not on the board, and the only way to be sure of that is for the data to
    come from the API rather than from a literal."""
    assert "useLeaderboard" in _BOARD
    assert "SAMPLE_CURVES" not in _BOARD and "SAMPLE_STANDINGS" not in _BOARD


def test_the_hero_reports_a_failed_load_instead_of_shimmering_forever():
    """Three states, and they must be distinguishable. A permanent skeleton and
    a silent fallback are the same defect: "the backend is down" and "the backend
    is fine" would render near-identically."""
    board = _collapse(_BOARD)
    assert 'status === "error"' in board or "status === 'error'" in board
    assert 'status === "loading"' in board or "status === 'loading'" in board
    assert "state.message" in board or "board.message" in board, (
        "the failed card must name the failure, not print a dead end"
    )


def test_talk_drops_the_three_step_list_but_keeps_its_pinned_strings():
    """The <ol> restates WhyCare's three acts one screen later. Everything the
    existing suite pins about this section survives -- listed here so the trim
    does not discover them by reddening CI.

    Comment-stripped, like the scans above: these are claims about what the
    component RENDERS, and a comment explaining the deleted list would otherwise
    keep `<ol` "present" forever.
    """
    talk = _strip_comments((_HOME / "Talk.tsx").read_text(encoding="utf-8"))
    assert "<ol" not in talk
    assert 'id="talk"' in talk
    assert "Describe your idea" in talk
    assert "Discord" in talk
    assert "<DiscordMock />" in talk
    assert talk.count("01 — Talk") == 1


def test_whycare_headings_are_untouched():
    """Headings and the step-number ban, which are checked against DIFFERENT
    texts on purpose.

    The headings are a render claim, so they read the stripped source. The
    quoted-step-number ban is not: `test_band_runs_no_second_step_sequence` greps
    the raw file, and the file's own header comment tells editors the ban covers
    the whole file precisely so nobody writes the number in a comment and then
    copies it into JSX. Stripping here would quietly hold this copy of the rule
    to a weaker standard than the guard it backs up.
    """
    raw = (_HOME / "WhyCare.tsx").read_text(encoding="utf-8")
    whycare = _collapse(_strip_comments(raw))
    for heading in (
        "Describe it in plain English",
        "Prove it on real market data",
        "See how it ranks",
        "Pick the AI model",
        "For developers: bring your own agent",
    ):
        assert heading in whycare
    assert not re.search(r'"0[1-9]"', raw), "quoted step numbers are banned here"


def test_the_two_surfaces_agree_on_the_numbers_that_must_agree():
    """There is no shared code and no shared token between / and /app, so after
    this change there are two chart implementations with two axis-tick
    declarations and two legend treatments. That duplication is forced by the
    stacks and accepted; leaving it UNGUARDED is not. Pin the values that must
    match so the pair drifts loudly or not at all.

    Heights are deliberately absent: the surfaces have different vertical
    envelopes and therefore different clamps (spec §2). A shared height
    assertion here would be the bug it looks like a guard against. Units are
    the same kind of case and are asserted per-surface below, not shared.
    """
    home_js = (
        Path(__file__).resolve().parents[2] / "frontend" / "home-page.js"
    ).read_text(encoding="utf-8")

    # Axis ticks: 14px on both.
    assert "fontSize={14}" in _BOARD
    assert re.search(r"font:\s*\{\s*size:\s*14\s*\}", home_js)

    # The key's type scale: text-base on /, and /app's rows inherit the panel's
    # base size rather than the old 11px table register.
    assert "text-base" in _BOARD
    assert "hm-rank-swatch" in home_js

    # Neither surface draws a built-in legend: the standings/chips are the key.
    assert "<Legend" not in _BOARD
    assert re.search(r"legend:\s*\{\s*display:\s*false\s*\}", home_js)

    # UNITS: percent on BOTH, and this is the assertion that inverted.
    #
    # It used to pin an ASYMMETRY -- /app percent, / dollars -- and the
    # justification was precise: / plotted fabricated curves that all shared a
    # base of 1000, so `$1210` was unambiguous and read as SAMPLE_STANDINGS'
    # +21.0%. That premise is gone. / now plots the same LIVE entries screen 0
    # does, and every dollar level in that payload is a x0.1 rescale of a
    # $100,000 backtest onto the config's $10,000 display base (leaderboard
    # service.py), so a `$10,749` tick names an account that never existed while
    # the percent is what actually ran.
    #
    # NOT the reason, though an earlier draft of the chart-first plan said so:
    # issue #365 does NOT make a dollar axis draw a 10x break here.
    # get_leaderboard normalises every entry to one display base before serving
    # -- measured against a hand-built mixed-capital database -- so on this
    # payload dollars and percent are an affine transform. Do not re-derive the
    # scale argument and then "discover" it is false; the label argument above
    # is the one that holds.
    assert "(v * 100).toFixed(1)}%" in home_js
    assert "toFixed(1)" in _BOARD, "the landing axis is percent to one decimal too"
    assert not re.search(r"tickFormatter=\{\(v\) => `\$", _BOARD), (
        "a dollar tick on this card names an account that never existed"
    )
