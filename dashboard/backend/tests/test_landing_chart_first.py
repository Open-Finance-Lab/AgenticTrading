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
# The SECOND board card. There are two of them and they took the same window
# chip in the same commit, so a layout guard scoped to one of them is scoped to
# half the change -- see test_the_race_window_chip_cannot_out_size_its_header_row.
_RACE = _strip_comments((_HOME / "Race.tsx").read_text(encoding="utf-8"))

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
    """Both reserves are derived, not taste, and there are TWO because the card's
    non-chart height is not one number: one thing beside the copy at >=lg, and
    another stacked at phone widths where the title, the window chip and the
    caption all wrap and the chip strip runs to eight rows.

    RE-DERIVED when the board went live -- the strip went from five hardcoded
    entries to nine from the payload and the "Illustrative example" chip became
    a longer window label -- and BOTH NUMBERS MOVED. Measured against the BUILT
    card with the board READY, at the NARROWEST width of each band:

        lg+     460 = ceil10(136 cardTop + 313.75 non-chart @1024x768) + 10
        below   730 = ceil10(132 cardTop + 583.25 non-chart @360x800)  + 10

    The trailing +10 is measured too: rounding alone left 0.25px of fold slack
    at 1024, a margin that survives exactly one browser.

    THE OLD lg VALUE WAS MEASURED AT THE WRONG WIDTH, which is why it is pinned
    here with the band spelled out. `lg:` binds from 1024, but 390 was derived
    at 1440 where non-chart is 249.75; from 1024 to 1279 the strip takes a fifth
    row and the card hung 55.75px BELOW THE FOLD across that entire band while
    every viewport it had been checked at (1280+) passed with 4.25px to spare.
    Re-derive at 1024 and 360, never at 1440 and 390.

    The 260px FLOOR is what binds on a phone, not either reserve: at 390x844 the
    card needs 920.5px against 844, so the strip's tail is below the fold at any
    reserve. That is deliberate -- see the component comment. The floor is
    pinned here so a future "fix" that shrinks it to chase the fold has to argue
    with this docstring first: the chart already ends above the fold there, and
    lowering the floor trades the chart for its own fallback key.

    RE-DERIVE BOTH AGAIN if the caption, title or chip strip changes height. The
    failure mode is a silently half-visible card, not a broken build.

    The var() indirection is load-bearing and not a tidy-up: the formula's
    commas defeat Tailwind's arbitrary-VALUE parser, so the breakpoint-dependent
    number rides an arbitrary PROPERTY instead, which does take a prefix.
    """
    board = _BOARD.replace(" ", "")
    assert "clamp(260px,calc(100dvh-var(--board-chart-reserve)),520px)" in board
    # UNPREFIXED, and this is the severe one. As a bare substring check this
    # assertion was satisfied by `md:[--board-chart-reserve:730px]`: below `md`
    # the custom property is then undefined, `clamp(260px, calc(100dvh -
    # var(--board-chart-reserve)), 520px)` is invalid at computed-value time, the
    # `height` declaration is DROPPED, the container computes to `auto`, and
    # <ResponsiveContainer height="100%"> resolves against zero. Measured in
    # headless Chromium at 390x844: chart region 260px with the variable
    # defined, 0px without -- the hero chart does not render at all on any
    # phone, with this file at 19 passed and `npm run typecheck` clean.
    #
    # Read off _BOARD with its spaces INTACT rather than the collapsed `board`
    # above, so the class can be anchored on the whitespace that separates
    # Tailwind classes. A negative lookbehind for `<letter>:` was the first
    # draft and has a hole: an arbitrary variant (`min-[390px]:`) ends `]:`,
    # which no `[a-z]:` lookbehind rejects. Anchoring on the separator rejects
    # every variant form, present and future, because a variant by definition
    # occupies the characters between the separator and the class.
    assert re.search(r'(?:^|\s)\[--board-chart-reserve:730px\](?=\s|"|$)', _BOARD), (
        "the base reserve must be unprefixed, or the clamp is invalid below "
        "that breakpoint and the chart region computes to 0"
    )
    assert "lg:[--board-chart-reserve:460px]" in board, "the side-by-side reserve"
    assert "56vh" not in _BOARD, "the first draft's clamp fails at four viewports"
    assert "h-[210px]" not in _BOARD and "md:h-[240px]" not in _BOARD


def test_the_window_chip_cannot_out_size_the_header_row():
    """The chip states the window the chart draws, and at 390px it used to run
    38.8px past the card's right edge -- which is `overflow-hidden`, so the end
    date was cut off -- while squeezing the <h2> beside it to width ZERO that
    still rendered 112px tall.

    Both symptoms came from one class: `shrink-0` on a chip whose text went from
    19 characters ("Illustrative example") to 44 ("Competition window ·
    2026-04-15 → 2026-05-15") when the board went live. Nothing failed: no
    scrollbar, no ellipsis, no console error -- the same silent clipping the chip
    strip below shipped once already.

    Pinned as the two classes that fix it because the measurement that caught it
    only exists in a browser, and nothing in CI opens one. `max-w-full` is what
    lets the chip wrap instead of overflowing; `flex-wrap` is what lets it take
    its own row instead of collapsing the title to reach it.

    BOTH ARE REQUIRED UNPREFIXED, and as bare substring checks neither was:
    `lg:flex-wrap` and `lg:max-w-full` satisfy `in`, bind only from 1024px, and
    therefore restore the measured defect across the entire sub-1024 band --
    including the 390px the measurement above was taken at -- with this file at
    19 passed and `npm run typecheck` clean.

    The OTHER board card has its own case,
    test_the_race_window_chip_cannot_out_size_its_header_row, deliberately
    separate rather than merged into this one: an edit aimed at one card must
    not be able to delete the other card's pin. If you add a third board card,
    add a third case -- do not widen either of these to scan several files.
    """
    row = re.search(r'<div className="([^"]*)">\s*<h2', _BOARD)
    assert row, "could not find the header row that wraps the <h2>"
    assert re.search(r"(?:^|\s)flex-wrap(?:\s|$)", row.group(1)), (
        f"the title/chip row must wrap at every width, or the chip collapses "
        f"the title to width 0; found {row.group(1)!r}"
    )

    chip = re.search(r'<span className="([^"]*)">\s*\{data\?\.windowLabel', _BOARD)
    assert chip, "could not find the window chip — did the label move?"
    assert "shrink-0" not in chip.group(1), (
        f"shrink-0 on the window chip is what pushed it 38.8px past the card's "
        f"edge; found {chip.group(1)!r}"
    )
    assert re.search(r"(?:^|\s)max-w-full(?:\s|$)", chip.group(1)), (
        f"the window chip must be capped at the row width at every width; "
        f"found {chip.group(1)!r}"
    )


def test_the_race_window_chip_cannot_out_size_its_header_row():
    """The same 44-character chip, in the OTHER board card, which never got the
    fix the hero card got.

    "Illustrative example" (19 chars) became "Competition window ·
    2026-04-15 → 2026-05-15" (44) in BOTH cards in the same commit. Only
    BoardPreview.tsx was repaired, and the case above is scoped to `_BOARD`, so
    nothing could see the other half. Measured in headless Chromium, base vs
    HEAD in the same browser: at 390x844 Race's `<h3>Competition Standings</h3>`
    goes 109px -> 0px wide while still rendering 56px tall, so its text
    overflows under the chip's own `bg-muted` and the heading reads "Standings"
    with "Competition" painted over; the chip's right edge lands 58.2px past the
    card's inner right of 327; and at 360x800 `documentElement.scrollWidth`
    becomes 385 against an `innerWidth` of 360 -- 25px of horizontal page scroll
    on a 360px phone, on the highest-traffic anonymous surface in the product.

    Deliberately a SEPARATE case per component rather than one merged scan: an
    edit aimed at one card cannot then delete the other card's pin. Both cards
    are guarded; neither guard is the other's.

    The two classes are required UNPREFIXED. `lg:flex-wrap` and `lg:max-w-full`
    satisfy a bare substring check and bind only from 1024px -- which restores
    the measured defect across the entire sub-1024 band the measurements above
    were taken in, with the guard green.
    """
    row = re.search(r'<div className="([^"]*)">\s*<h3', _RACE)
    assert row, "could not find the Race header row that wraps the <h3>"
    assert re.search(r"(?:^|\s)flex-wrap(?:\s|$)", row.group(1)), (
        f"the title/chip row must wrap at every width, or the chip collapses "
        f"the title to width 0; found {row.group(1)!r}"
    )

    chip = re.search(r'<span className="([^"]*)">\s*\{board\.status', _RACE)
    assert chip, "could not find Race's window chip — did the label move?"
    assert "shrink-0" not in chip.group(1), (
        f"shrink-0 on the window chip is what pushed it 58.2px past the card's "
        f"edge and put 25px of horizontal scroll on a 360px phone; found "
        f"{chip.group(1)!r}"
    )
    assert re.search(r"(?:^|\s)max-w-full(?:\s|$)", chip.group(1)), (
        f"the window chip must be capped at the row width at every width; "
        f"found {chip.group(1)!r}"
    )


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
    come from the API rather than from a literal.

    THE PARENS ARE THE ASSERTION. A bare `"useLeaderboard" in _BOARD` holds
    against the IMPORT line whether or not the hook is ever called, and
    `noUnusedLocals` is off. Verified by mutation: keeping the import, adding a
    module-level `function fabricatedBoard(): BoardState { ... }` of hardcoded
    curves, returns and window label, and calling it instead left `npm run
    typecheck` clean and the five landing suites at 113 passed -- with the hero
    drawing an entirely invented board, which is precisely the state this change
    exists to remove. The return-type annotation defeats TS literal narrowing,
    so no branch below goes unreachable and nothing else notices either.

    The companion assertion below is not a backstop: it bans the two retired
    sample-data symbols by name and says nothing about where the data comes
    from."""
    assert "useLeaderboard()" in _BOARD, (
        "the hero must CALL the hook, not merely import it — an unused import "
        "type-checks clean and leaves the board free to be a literal"
    )
    assert "SAMPLE_CURVES" not in _BOARD and "SAMPLE_STANDINGS" not in _BOARD


def test_the_hero_mounts_the_frame_it_reserves_room_for():
    """The rail is only ever reached through this element, and nothing else on
    the branch checks that anyone renders it.

    `test_the_rail_*` cases in test_landing_live_board.py read EndpointRail.tsx's
    OWN source, so they keep passing when the component becomes dead code.
    Verified by mutation: deleting the `<Customized>` element, its two imports
    and the reserved gutter -- the landing half of the frame removed wholesale --
    left the eight-file focused suite at its usual 1 failed / 128 passed AND
    typechecked clean, because an unmounted component still compiles.

    The gutter and the three props are asserted separately rather than as one
    blob: they fail independently in the browser. `right: frame.gutter` is the
    reserved column the labels are drawn into -- lose it and the rail paints
    over the plot area. `gap` is the one the rail cannot recover on its own: it
    consumes the value and never calls frameLayout, so a dropped prop is silent
    label collision, not an error. Three of the four (`gutter`, `drawLabels`,
    `gap`) come off the ONE frameLayout call above, which is what keeps this
    card from growing a second geometry; `valueByKey` is the endpoint values the
    rail labels, and without it the rail has nothing to draw.
    """
    assert "component={EndpointRail}" in _BOARD, (
        "the hero must actually mount the rail, not merely coexist with it"
    )
    assert "right: frame.gutter" in _BOARD, (
        "the gutter is reserved by the one frameLayout call, not by a literal"
    )
    assert "valueByKey={valueByKey}" in _BOARD
    assert "drawLabels={frame.drawLabels}" in _BOARD
    assert "gap={frame.gap}" in _BOARD, (
        "the rail never computes the gap -- this prop is where it comes from"
    )


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
    # The line above only sees an INLINE arrow formatter, and this file does not
    # use one -- it binds the named `axisTick`, so the most natural way to put
    # dollars back is to edit that function, where the regex cannot reach.
    # `toFixed(1)` on its own does not close it either: a dollar tick has one
    # decimal too. Verified by mutation: rewriting `axisTick` to return
    # `$${(10000 * (1 + v)).toFixed(1)}` -- the exact $10,749 display-base tick
    # the comment above forbids -- left this whole file GREEN. So pin the landing
    # formatter's BODY the same way home_js's is pinned two lines above, which
    # makes both surfaces fail on the same edit.
    assert "(v * 100).toFixed(1)}%" in _BOARD, (
        "the landing axis renders the percent that actually ran, not a level"
    )
    # ...AND pin what the axis is BOUND to, because the body pin above closes
    # only half of it. `axisTick` is also referenced by
    # `measureTextWidth(axisTick(domain[0]), ...)`, so it can be left byte-
    # identical -- keeping the body assertion AND the y-axis-reserve guard green
    # -- while a SECOND named formatter is declared beside it and bound to the
    # axis instead. Verified by mutation: adding
    # `function dollarTick(v) { return `$${(10000 * (1 + v)).toFixed(1)}`; }`
    # and binding it left this file at 19 passed with `npm run typecheck` clean,
    # and the hero rendering the $10,749-style ticks the comment above calls the
    # one hard "must never" of this change. The inline-arrow ban is likewise
    # blind to it: a named binding carries no arrow.
    #
    # Scoped to the <YAxis> ELEMENT, not the file, for the same reason the
    # y-axis-reserve guard is: a substring that may live in an import line, a
    # helper or a dead function is not a claim about what the axis renders.
    # A prop containing `>` (an inline arrow formatter) makes the element regex
    # miss and fires the first assertion -- fail-closed, which is the direction
    # this guard has to fail in.
    yaxis = re.search(r"<YAxis\b[^>]*?/>", _BOARD, re.S)
    assert yaxis, "could not find the <YAxis> element"
    assert "tickFormatter={axisTick}" in yaxis.group(0), (
        "the y-axis must bind the percent formatter itself, not a second "
        "formatter that walks around the body pin above; found "
        f"{yaxis.group(0)!r}"
    )
