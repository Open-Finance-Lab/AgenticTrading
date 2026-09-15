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
    caption all wrap.

    RE-DERIVED when the chip strip became the scrollable ranking table, which is
    the third time these two numbers have moved and the first time the thing
    that moved them stopped being a row count. The rule is unchanged --
    ``reserve = ceil10(cardTop + nonChart) + 10`` at the NARROWEST width of each
    band -- and the arithmetic is spelled out in the component beside the
    constants:

        lg+     520 = ceil10(136 cardTop + 364.15 non-chart @1024x768) + 10
        below   650 = ceil10(132 cardTop + 505.65 non-chart @360x800)  + 10

    Each starts from the measured figure it replaces (313.75 and 583.25), takes
    out the chip strip's height at that width (four rows / 120px at 1024, eight
    rows / 248px at 360, plus a 32px caption block in both) and puts back the
    table's, which does not vary by width: 20px caption + 26.4px head + 156px
    list. THAT INVARIANCE IS THE POINT. The strip's height was a function of how
    many entries the roster had and how wide the card was, so every roster
    change and every re-measurement at a new width moved these constants --
    which is exactly how an lg value derived at 1440 came to govern a band
    starting at 1024. A fixed-height scrolling list has no row count: the board
    can grow to twenty entries and the non-chart height does not move.

    THE ARITHMETIC HALF IS NOT A MEASUREMENT, and this docstring should not
    pretend otherwise. cardTop and the header block carry over from the browser
    measurements above; the strip heights removed and the table height added are
    computed from pitches stated in the source (24px + 8px for the strip, and
    a 28px row + 8px gap for the list). THE ROW PITCH IS THE BADGE, NOT THE
    TEXT, and an earlier draft of this docstring got that wrong -- it said
    "13.5px/1.35 + 3px padding-block = 24.2px", which is neither the text box
    (18.2 + 6 = 24.2 is arithmetically fine) nor the row: the <li> is a grid
    with ``items-center``, so its height is its TALLEST cell, and the 22px rank
    badge beats the text at 13.5px and still beats it at 16px (21.6). The row
    is 22 + 3 + 3 = 28px, and was 28px before the type bump too -- which is why
    raising the rows cost the list nothing. Re-measure at 1024x768 and 360x800
    when a browser is available and correct both numbers if they disagree.

    THE TYPE BUMP MOVED EXACTLY ONE CONSTANT, and only through the head. Rows
    13.5 -> 16 changed no height at all (badge dominance, above). The head
    12 -> 14.4 -- ``text-[11px]`` -> ``text-[12px]`` against a pinned
    ``leading-[1.2]`` -- added 1.2px, enough to carry the lg figure across a
    ceil10 boundary (500.15 rounds up to 510, +10 = 520) and not enough to
    carry the base one (637.65 rounds up to 640 either way).

    THE FIT IS NO LONGER A SINGLE TIGHT POINT. Pairing the list's own clamp
    with the chart's -- ``clamp(96, 100dvh - 632, 148)`` against
    ``clamp(260, 100dvh - 520, 520)`` -- leaves exactly one of the two in its
    linear middle at any viewport from 728 to 1040 tall, so the card's bottom
    tracks the viewport 1:1 and the slack is a CONSTANT 19.85px across that
    whole band, 1024x768 included (list 136, chart floored at 260, card
    748.15 against 768). Raising the reserve alone could not have produced
    that: at 768 the chart is already on its floor, so the reserve is not in
    the expression there and the list clamp is the only lever -- which is why
    ``calc(100dvh-620px)`` became ``calc(100dvh-632px)`` while the base
    reserve did not move.

    The 260px FLOOR is what binds on a phone, not either reserve -- unchanged,
    and now less severe: 505.65 of non-chart against the old 583.25 pulls ~78px
    of the card back above the fold, because a table that scrolls does not wrap
    the way a strip did. Still below the fold, still deliberate. The floor is
    pinned here so a future "fix" that shrinks it to chase the fold has to argue
    with this docstring first: the chart already ends above the fold there, and
    lowering the floor trades the chart for its own fallback key.

    RE-DERIVE BOTH AGAIN if the caption, the title, the table head or the list's
    ``max-h`` changes height. The failure mode is a silently half-visible card,
    not a broken build.

    The var() indirection is load-bearing and not a tidy-up: the formula's
    commas defeat Tailwind's arbitrary-VALUE parser, so the breakpoint-dependent
    number rides an arbitrary PROPERTY instead, which does take a prefix.
    """
    board = _BOARD.replace(" ", "")
    assert "clamp(260px,calc(100dvh-var(--board-chart-reserve)),520px)" in board
    # UNPREFIXED, and this is the severe one. As a bare substring check this
    # assertion was satisfied by `md:[--board-chart-reserve:650px]`: below `md`
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
    assert re.search(r'(?:^|\s)\[--board-chart-reserve:650px\](?=\s|"|$)', _BOARD), (
        "the base reserve must be unprefixed, or the clamp is invalid below "
        "that breakpoint and the chart region computes to 0"
    )
    assert "lg:[--board-chart-reserve:520px]" in board, "the side-by-side reserve"
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


def test_the_hero_card_carries_the_full_ranking_table_and_scrolls_it():
    """THIS CASE REPLACES ONE THAT ASSERTED THE OPPOSITE, and the reversal was
    adjudicated rather than drifted into.

    It used to be ``test_the_standings_table_becomes_a_chip_strip_that_can_show
    _every_chip``, and its argument was sound for what the card then was: a
    table costs vertical height the chart needs, so the standings were demoted
    to a wrapping chip strip and the full ranking lived in Race.tsx four screens
    down. It banned ``grid-cols-12`` for that reason and required ``flex-wrap``,
    because a strip that cannot wrap silently truncates -- measured scrollWidth
    910 against clientWidth 285 at 390px, so four of five chips simply vanished.

    What changed is the premise, not the measurement. The strip's height was
    unbounded in the one direction that mattered: it grew with the roster and
    with every narrowing of the card, which is why its height had to be
    re-measured into the chart reserve three separate times. A table whose rows
    live in a fixed-height scrolling viewport is bounded in both -- 148px
    whatever the roster does -- so the card can carry ranks, ending values and
    Sharpe for every contender AND keep a chart the strip was protecting. Race
    no longer holds the detail; this is now the only ranking on the page, which
    is also why the columns below are pinned by name.

    The strip's two jobs both had to survive the swap, and both are asserted
    here: it was the only thing linking a curve's colour to a name (the chart
    ships no Recharts <Legend>), and the fallback whenever EndpointRail declines
    to draw. A table without the per-row swatch would silently drop both.
    """
    board = _BOARD

    # ONE TEMPLATE FOR HEAD AND ROWS. The /app original cannot desync because
    # styles.css joins the two selectors into a single rule; here the shared
    # constant is that rule, so what is pinned is that BOTH consumers use it and
    # that no third grid template exists to drift from.
    assert "const RANK_GRID" in board, "the head/row grid template must be one constant"
    assert board.count("${RANK_GRID}") == 2, (
        "the header row and the data rows must both be laid out by RANK_GRID; "
        "a second literal template is how a table unaligns from its own header"
    )
    literal_grids = re.findall(r'"[^"]*grid-cols-\[[^"]*"', board)
    assert len(literal_grids) == 1, (
        f"grid templates must live only in RANK_GRID; found {literal_grids}"
    )

    # BOUNDED HEIGHT AND A SCROLLBAR, which is what buys the chart its size
    # back. `max-h` without `overflow-y-auto` clips rows with no scrollbar and
    # nothing failing -- the same shape as the chip strip's silent truncation,
    # which is the failure this card has now shipped twice and must not a third
    # time.
    # READ OFF THE <ol> ITSELF, not a character window around it. The case this
    # replaces had to take `board[index - 400 : index]` because the chip strip
    # was one unremarkable <div> among many and the file maps `standings` three
    # times; that window was mis-anchored once already and passed a mutation
    # that added `overflow-hidden` to the strip. The scroll container is the
    # file's only <ol>, so the element can be matched exactly and the mutation
    # has nowhere to hide.
    ol = re.search(r'<ol\s+className="([^"]*)"', board)
    assert ol, "the standings list is no longer an <ol> — re-anchor this guard"
    classes = ol.group(1)
    assert "overflow-y-auto" in classes, "the standings list must scroll, not clip"
    # PINNED WHOLE, like the two reserves, because all three numbers are derived
    # and derived together: 148 is the height the lg reserve was computed
    # against, and 632 is chosen so that the list is ALREADY giving height back
    # at 768 -- `100dvh - 632` is 136 there, one row and change short of the
    # cap. 620 put the crossover exactly at 768 (100dvh - 620 == 148), which
    # meant the tightest supported viewport was also the one viewport where
    # neither clamp had any give: the card cleared the fold by 9px there and by
    # more everywhere else. Staggering the two clamps by 12px is what flattens
    # that into a constant 19.85px of slack from 728 to 1040 -- see the
    # derivation in BoardPreview.tsx and in this module's clamp case. A bare
    # "has some max-h" check would pass on any of them being edited alone,
    # which is the only way they can go wrong.
    assert "max-h-[clamp(96px,calc(100dvh-632px),148px)]" in classes, (
        f"the list must keep its derived height clamp — see the derivation in "
        f"BoardPreview.tsx and in this module's clamp case; found {classes!r}"
    )
    assert "overflow-hidden" not in classes, (
        "clipping the list is the same failure by another route -- no scrollbar, "
        "no ellipsis, and nothing fails. (`overflow-x-hidden` is fine and is a "
        "different class; this bans the unaxed form on the scrolling element.)"
    )

    # THE COLUMNS, BY NAME. This is the only ranking on the page now.
    # \s* around the label: the formatter breaks a <span> that carries more than
    # one attribute across lines, so the Sharpe cell (which also carries the
    # tooltip `title`) collapses to "> Sharpe <" while the bare ones collapse to
    # ">Contender<". A tight match passes on four columns and fails on the fifth
    # for a reason that has nothing to do with the column.
    head_text = _collapse(board)
    for column in ("Ending value", "Return", "Sharpe"):
        assert re.search(rf">\s*{re.escape(column)}\s*<", head_text), (
            f"the {column!r} column is gone"
        )
    assert re.search(r">\s*Contender\s*<", head_text), (
        "the column holds benchmarks too -- see "
        "test_the_standings_table_does_not_present_a_benchmark_as_an_ai_model"
    )

    # THE PHONE TABLE IS THREE COLUMNS, and the tracks drop with the cells: a
    # `hidden` cell stops occupying its column but its TRACK survives, so a
    # narrow template is required alongside, not instead.
    assert board.count("hidden sm:block") == 4, (
        "Ending value and Sharpe each drop below sm in BOTH the head and the "
        "rows -- four cells, or the table and its header disagree at one width"
    )
    assert "sm:grid-cols-[" in board and "grid-cols-[26px_minmax(0,1fr)_78px]" in board, (
        "a narrow three-track template must accompany the hidden cells"
    )

    # THE STRIP'S TWO JOBS. The colour comes off the same BoardStanding the
    # curve is drawn from, so a row and its curve cannot disagree.
    assert "item.color" in board
    assert "dataKey=" in board


def test_the_standings_table_waits_for_the_fetch_before_claiming_the_board_is_empty():
    """The table's four states, and the one that was missing.

    `standingsCoverage([])` is `"empty"`, and `standings` is `[]` in all three
    of `loading`, `error` and a genuinely empty 200 -- so a table that branches
    on coverage alone renders "The standings came back empty. The request
    succeeded and carried no entries" while the request is still in flight, and
    again when it failed outright. Both are false sentences, and the first one
    is on screen for the whole of a free-tier cold start: 30-60 seconds, for the
    first visitor of the day, above the fold.

    It is the fail-closed-is-not-fail-visible shape inverted. The usual version
    makes a failure look like success; this one makes a pending request look
    like a settled, successful, empty one -- and it asserts the success
    explicitly, in the copy, which is worse than saying nothing.

    The chart above never had this: its branches start from `board.status` and
    only reach coverage on `ready`. Race.tsx's table, which this one replaced,
    started from `board.status` too. The gating was the one thing that did not
    travel with the rows, and nothing in the suite noticed, because every other
    guard reads the source for structure rather than for order.

    PINNED AS ORDER, NOT PRESENCE. `board.status` appearing somewhere in the file
    is satisfied by the chart's own branches 200 lines above; what makes the
    table correct is that its status checks come BEFORE its coverage check
    inside the list.
    """
    board = _BOARD
    ol = board.index('data-testid="board-rank-list"')
    table = board[ol:]

    loading = table.find('board.status === "loading"')
    error = table.find('board.status === "error"')
    empty = table.find('tableCoverage === "empty"')

    assert loading != -1, (
        "the standings list must render a loading state — until the fetch "
        "settles there is no board to call empty"
    )
    assert error != -1, (
        "the standings list must name a failed fetch rather than reporting it "
        "as a successful empty response"
    )
    assert empty != -1, "the genuinely-empty branch is gone"
    # ALL THREE DEAD-END BRANCHES END THE SAME WAY. The loading line is the one
    # state that resolves itself; the other two do not, and a reader given no
    # next step on two of three is being told the page is broken in a way they
    # cannot act on. The error branch always carried "Reload to try again." and
    # the empty one did not, for no reason beyond the order they were written.
    assert table.count("Reload to try again.") == 2, (
        "the error and empty branches must both name the recovery step; only "
        "the loading state resolves on its own"
    )
    assert loading < empty and error < empty, (
        "the status branches must be tested BEFORE the coverage branch, or an "
        "in-flight request renders the copy that claims it already succeeded"
    )

    # The baselines-only notice is a claim about a SUCCESSFUL response too, and
    # reaches its own branch by the same route.
    assert 'board.status === "ready" && tableCoverage === "baselines-only"' in board, (
        "the 'no AI model results came back' line asserts a completed request; "
        "it must not render for one that has not completed"
    )

    # THE COLUMN HEADER IS A CLAIM TOO, and it sat OUTSIDE every branch above --
    # which is the same defect one element higher up, and the one this test's
    # own <ol>-scoped search could not see. "# | Contender | Ending value |
    # Return | Sharpe" painted over the loading shimmer, over the error line,
    # and over "the standings came back empty": a five-column frame asserting a
    # ranking that is not under it.
    #
    # lib/leaderboard.ts bans this by name in its own docstring ("Race drew its
    # Rank/AI model/Return header over zero rows"), so it is a re-run of a
    # documented defect, not a new judgement call.
    #
    # `tableCoverage`, NOT `coverage`: the two answer different questions (the
    # chart's caption uses `coverage`), and gating the head on the chart's rule
    # would put it back over a list printing its own empty message.
    head = board.index('data-testid="board-rank-head"')
    gate = board.rfind('board.status === "ready" && tableCoverage !== "empty"', 0, head)
    assert gate != -1, (
        "the standings column header must be gated on a ready, non-empty "
        "board — unbranched, it draws a five-column frame over the loading "
        "shimmer, over the error message and over the empty-board copy"
    )
    # Nothing between the gate and the head but the opening <div>: a gate that
    # governs some ancestor several branches away is not this element's gate.
    assert head - gate < 400, (
        "the gate must be the header's own conditional, not a distant ancestor's"
    )


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

    # THE ROW TYPE SCALE, AND IT IS NOW ONE NUMBER ON BOTH SURFACES -- which is
    # a stronger guard than the one it replaces, not a weaker one.
    #
    # This used to assert `text-base` on / against `hm-rank-swatch` on /app:
    # two different facts about two different treatments, agreeing on nothing.
    # It could not do better, because / had a chip strip and /app had a table;
    # there was no shared number to pin. / now draws the same table, so the
    # shared number exists and is 16px -- styles.css sets it for the promoted
    # hero card specifically (`.home-landing-board .home-module-rank-list li`),
    # and the React rows take it as `text-[16px]` rather than `text-base`.
    #
    # `text-[16px]` AND NOT `text-base`, WHICH IS THE SAME FONT SIZE. The two
    # differ in what else they set: `text-base` also sets
    # `line-height: 1.5rem` = 24px, which is past the 22px rank badge that
    # currently decides the row height, so it would grow every row from 28px to
    # 30px and silently invalidate the 156px list figure both reserves are
    # derived from. The arbitrary value sets font-size and nothing else, and
    # `leading-[1.35]` beside it keeps the line box at 21.6 -- under the badge,
    # which is why this bump cost no height at all.
    #
    # THE BUMP IS SCOPED ON THE /app SIDE and unscoped here, which is not an
    # asymmetry: every selector in that styles.css block is prefixed
    # `.home-landing-board` because the same markup is still a one-third-width
    # dashboard tile elsewhere, where 12.5px is right for the column. This file
    # IS the promoted card, so it has no narrow twin to protect.
    #
    # The swatch assertion stays and gains its counterpart: the colour-to-name
    # link is what both surfaces would silently lose if a row stopped carrying
    # it, and neither chart draws a legend to fall back on.
    styles_css = (
        Path(__file__).resolve().parents[2] / "frontend" / "styles.css"
    ).read_text(encoding="utf-8")
    assert "text-[16px]" in _BOARD, "the landing rows must keep the /app row size"
    assert "text-base" not in _BOARD, (
        "`text-base` is the same font-size with a 24px line-height, which "
        "exceeds the 22px rank badge and grows every row from 28px to 30px -- "
        "invalidating the 156px list height both reserves are derived from"
    )
    assert ".home-landing-board .home-module-rank-list li { font-size: 16px; }" in styles_css, (
        "the /app hero card's row size moved; the landing's text-[16px] now "
        "disagrees with it"
    )
    # THE HEAD AND THE BADGE TOO, because a row raised without them re-opens
    # exactly the readability gap the bump closed -- 11px column labels over
    # 16px rows read as a different component. Both are pinned on both surfaces
    # for the same reason the row size is: the two tables are one design.
    assert "text-[12px] leading-[1.2]" in _BOARD, (
        "the standings head must keep its size AND its pinned line-height -- "
        "`text-[12px]` sets font-size only, so without `leading-[1.2]` the row "
        "inherits preflight's 1.5 and stands 18px instead of 14.4"
    )
    assert ".home-landing-board .hm-rank-table-head { font-size: 12px; }" in styles_css
    assert ".home-landing-board .home-module-rank { font-size: 12px; }" in styles_css
    assert ".home-landing-board .hm-rank-value { font-size: 14px; }" in styles_css
    assert "hm-rank-swatch" in home_js
    assert "backgroundColor: item.color" in _BOARD, (
        "the landing rows must carry the curve's own colour, as /app's do"
    )

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
