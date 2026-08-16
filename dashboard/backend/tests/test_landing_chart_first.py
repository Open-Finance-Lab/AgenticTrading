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

_NO_REAL_MONEY = (
    "Every test here uses simulated money. Real money is involved only if you "
    "explicitly connect a brokerage account and turn on live trading."
)


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
    assert "An agent is an AI trading assistant that follows your written instruction" in hero
    assert "it trades the idea hour by hour, measured against buy-and-hold and the index" not in hero, (
        "the second clause is what makes this two lines at 1/3 column width"
    )


def test_the_simulated_money_sentence_survives_verbatim_as_small_print():
    """Pinned twice -- by test_no_real_money_sentence_is_present_verbatim and by
    the _CLAIM_DISCLAIMERS allowlist, whose staleness check fails if the wording
    drifts. Moving it between components is fine; rewording it is not.
    """
    assert _NO_REAL_MONEY in _collapse(_HERO)


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
    assert "order-first" in hero and "lg:order-first" in hero


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

    The 390 is derived, not taste: the card's own non-chart height (~227px:
    caption bar, chip strip, detail line, padding) + 120px
    --landing-chrome-height + ~43px fold margin. RE-DERIVE IT if the caption or
    chip strip changes height -- the failure mode is a silently half-visible
    card, not a broken build.
    """
    assert "clamp(300px,calc(100dvh-390px),520px)" in _BOARD.replace(" ", "")
    assert "56vh" not in _BOARD, "the first draft's clamp fails at four viewports"
    assert "h-[210px]" not in _BOARD and "md:h-[240px]" not in _BOARD


def test_landing_chart_axis_ticks_are_14px():
    assert _BOARD.count("fontSize={14}") == 2, "both XAxis and YAxis"
    assert "fontSize={11}" not in _BOARD


def test_the_panel_title_is_text_xl():
    """Spec §2. The card is now two-thirds of the hero; a text-lg title reads as
    a widget label on it."""
    assert 'className="text-xl font-bold flex items-center gap-2 min-w-0"' in _BOARD


def test_the_standings_table_becomes_a_one_row_chip_strip():
    """Demotion, not deletion: the chart ships no <Legend> (a five-item legend
    wraps to two rows at this width), so the table is the ONLY thing linking a
    curve colour to a model name. The chips preserve that swatch-to-curve link
    at a fraction of the height. The full table already lives in Race.tsx.
    """
    board = _BOARD
    assert "grid-cols-12" not in board, "the 5-row table is what the chart needs the height of"
    assert "flex-nowrap" in board, "five chips, one row"
    assert "text-base" in board, "text-sm rows were one of the three reported problems"
    # The identity link and the guard corpus both depend on these staying here.
    assert "SAMPLE_STANDINGS" in board
    assert "dataKey=" in board
    assert "item.swatch" in board
