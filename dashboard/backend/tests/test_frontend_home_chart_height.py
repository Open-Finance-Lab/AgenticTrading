"""Screen 0's chart floor, and the label verdict that rides on it.

The 2026-09-03 user feedback opened with "Lines on the leader-board are too
compact to see the trend", screenshotting the /app home panel. That panel's
chart is `flex: 0 1 auto` and deliberately yields height before the standings
list does, so at a short viewport it is not the `clamp()` that binds -- it is
`.hm-rank-chart`'s `min-height`, which is the number the reader was complaining
about.

Raising it is not a free CSS tweak, because the same height decides whether the
chart draws endpoint labels. `boardFrameLayout` refuses them outright below
`BOARD_MIN_LABEL_HEIGHT` (178px), so a floor raised past that would answer "the
lines are too compact" by adding nine labels to the same panel -- which is the
*other* complaint in the same document ("the labeling is quite messy"), one
screen over.

THIS GUARD USED TO DERIVE THE 178 instead of reading it, as
`(BOARD_PILL_HEIGHT + 1) * 9 + BOARD_XAXIS_ALLOWANCE`, and that derivation was
wrong in the direction that matters. `BOARD_XAXIS_ALLOWANCE` is only the
FIRST-FRAME estimate: from the second update onwards `boardFrameLayout` divides
by the real axis height off `chart.scales.x`, ~24px on this panel at its 14px
ticks. The true geometric flip was therefore ~168px, not 178 -- exactly the
floor this file was clearing -- so nine pills would have appeared on the first
re-layout after paint (a resize into the 508-646px band is one) with the guard
still green, because a test that calls the function once only ever sees the
estimate. The threshold is now an explicit constant on the JS side and this
reads it.

Hence two guards: the floor is high enough to be worth the change, and still low
enough that the label verdict at short viewports is exactly what it was.
"""

import re
from pathlib import Path

from ._frontend_source import css_blocks

_LEADERBOARD_JS = (
    Path(__file__).resolve().parents[2] / "frontend" / "js" / "leaderboard.js"
).read_text(encoding="utf-8")

# Curves screen 0 draws: seven competition models plus the two reference curves
# it keeps (`test_chart_draws_the_baselines_the_rank_list_filters_out` pins that
# it carries two, not five). The count is what the pitch is divided by, so a
# board that adds an eighth model lowers the flip threshold and this guard has
# to be recomputed rather than re-baselined.
_SCREEN_ZERO_SERIES = 9

# Floor the feedback asked for.
#
# Scope it honestly: `clamp(140px, 26vh, 280px)` crosses 168px at a 646px-tall
# viewport, so this floor is inert above that and the change reaches only the
# 508-646px band (+12px at a 600px viewport, +36px at the short end). That is
# where the report came from -- its screenshot has no endpoint pills, and pills
# start at 178px. The cost lands in the same band: the chart is the flex item
# that yields, so those px come out of the standings list, about one row at the
# short end.
_MIN_USEFUL_FLOOR_PX = 168


def _js_int(name: str) -> int:
    match = re.search(rf"const {re.escape(name)} = (\d+)", _LEADERBOARD_JS)
    assert match, f"{name} is no longer a plain integer const in leaderboard.js"
    return int(match.group(1))


def _chart_floor_px() -> int:
    blocks = css_blocks(".hm-rank-chart")
    assert blocks, ".hm-rank-chart was renamed or deleted"
    match = re.search(r"min-height:\s*(\d+)px", blocks[0])
    assert match, "the chart floor is no longer a px min-height"
    return int(match.group(1))


def _label_flip_threshold_px() -> int:
    """Smallest chart height at which `boardFrameLayout` will draw labels.

    Read from the shipped constant rather than restated, so lowering
    `BOARD_MIN_LABEL_HEIGHT` under an unchanged floor reddens this. It is NOT
    re-derived from the pill/axis geometry any more -- see the module docstring
    for why that derivation read ~10px high and hid the bug it existed to catch.
    """
    return _js_int("BOARD_MIN_LABEL_HEIGHT")


def test_the_home_chart_floor_is_tall_enough_to_show_a_trend():
    """The feedback's actual ask, as a number."""
    assert _chart_floor_px() >= _MIN_USEFUL_FLOOR_PX, (
        f"the home chart floor is {_chart_floor_px()}px; at nine curves that is a "
        f"{(_chart_floor_px() - _js_int('BOARD_XAXIS_ALLOWANCE')) / _SCREEN_ZERO_SERIES:.1f}px "
        "band per curve, which is the 'too compact to see the trend' report"
    )


def test_the_home_chart_floor_stays_below_the_endpoint_label_threshold():
    """The constraint that makes the guard above safe to satisfy.

    This is the one that would have caught the mistake: raising the floor to
    "make the chart bigger" reads as pure improvement right up until nine
    endpoint pills appear in a panel whose sibling complaint is that labelling
    is messy. The threshold is read off the shipped constant, so this fails if
    someone raises the floor OR lowers `BOARD_MIN_LABEL_HEIGHT` underneath it.
    """
    flip = _label_flip_threshold_px()
    floor = _chart_floor_px()
    assert floor < flip, (
        f"the home chart floor ({floor}px) is at or above the {flip}px at which "
        f"{_SCREEN_ZERO_SERIES} endpoint labels start drawing. Short viewports "
        "would gain a label stack the panel has never shown -- decide that "
        "deliberately, do not acquire it by raising a min-height"
    )


def test_the_label_threshold_does_not_depend_on_a_measurement_taken_once():
    """`BOARD_MIN_LABEL_HEIGHT` has to gate on `chart.height` and nothing else.

    The gate this replaced was `(height - boardXAxisHeight(chart)) / labels`,
    and `boardXAxisHeight` deliberately returns two different numbers over a
    chart's life: `BOARD_XAXIS_ALLOWANCE` before the first layout, the scale's
    real height after. A verdict built on it therefore changes on re-layout
    while nothing about the page changed, which is unobservable to every test
    in this repo -- all of them call the layout function once. Pinning the
    gate's SHAPE is the only thing that can catch a revert to the geometry.
    """
    match = re.search(
        r"if \(chart\.height < BOARD_MIN_LABEL_HEIGHT\) return none;", _LEADERBOARD_JS
    )
    assert match, (
        "boardFrameLayout no longer refuses labels on chart.height alone; if the "
        "gate moved back onto boardXAxisHeight, the label verdict is frame-"
        "dependent again and the home chart floor above is no longer safe"
    )
