"""Screen 0's chart floor, and the label verdict that rides on it.

The 2026-09-03 user feedback opened with "Lines on the leader-board are too
compact to see the trend", screenshotting the /app home panel. That panel's
chart is `flex: 0 1 auto` and deliberately yields height before the standings
list does, so at a short viewport it is not the `clamp()` that binds -- it is
`.hm-rank-chart`'s `min-height`, which is the number the reader was complaining
about.

Raising it is not a free CSS tweak, because the same height decides whether the
chart draws endpoint labels. `boardFrameLayout` gives up and reserves the arrow
alone once the per-label pitch falls under `BOARD_LABEL_GAP_MIN`, and the pitch
is `(height - BOARD_XAXIS_ALLOWANCE) / labels`. With today's constants and screen
0's nine curves that flips at **178px**: below it the panel is label-free, at or
above it nine pills appear. So a floor raised past 178 would answer "the lines
are too compact" by adding nine labels to the same panel -- which is the *other*
complaint in the same document ("the labeling is quite messy"), one screen over.

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


def _label_flip_threshold_px(series: int) -> int:
    """Smallest chart height at which `boardFrameLayout` starts drawing labels.

    Derived from the shipped constants rather than restated: a guard that hard-
    codes 178 keeps passing after someone edits `BOARD_PILL_HEIGHT`, which is
    precisely when it needed to fail.
    """
    gap_min = _js_int("BOARD_PILL_HEIGHT") + 1  # BOARD_LABEL_GAP_MIN, derived
    return gap_min * series + _js_int("BOARD_XAXIS_ALLOWANCE")


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
    is messy. The threshold is computed from the shipped constants, so this
    fails if someone raises the floor OR lowers `BOARD_PILL_HEIGHT` underneath it.
    """
    flip = _label_flip_threshold_px(_SCREEN_ZERO_SERIES)
    floor = _chart_floor_px()
    assert floor < flip, (
        f"the home chart floor ({floor}px) is at or above the {flip}px at which "
        f"{_SCREEN_ZERO_SERIES} endpoint labels start drawing. Short viewports "
        "would gain a label stack the panel has never shown -- decide that "
        "deliberately, do not acquire it by raising a min-height"
    )
