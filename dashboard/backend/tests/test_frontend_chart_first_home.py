"""Guards for the chart-first rebuild of /app screen 0 (2026-08-15 spec).

Screen 0 lives inside `.home-pager-screen`, which is `height:100%;
overflow:hidden` in a scroll-snap pager: it CLIPS rather than scrolls, with no
scrollbar and no error. Every constraint here exists because the failure mode is
silent -- rows vanish, the chart is a blank box, and nothing logs.

The behavioural cases run the real extracted functions under node, following
test_frontend_leaderboard_hover.py. The source-shape cases guard the seams that
node cannot see (CSS, DOM insertion points, cross-file globals).
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import STYLES, css_blocks

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_HOME_JS = (_FRONTEND / "home-page.js").read_text(encoding="utf-8")
_LEADERBOARD_JS = (_FRONTEND / "js" / "leaderboard.js").read_text(encoding="utf-8")

_PANEL_SELECTOR = (
    'html[data-nav-page="home"] #homeView .home-landing-board .home-module'
)


def _panel_block() -> str:
    """The unscoped (>1200px) rule for the board panel.

    `css_blocks` returns every block with this prelude; the <=1200px media query
    re-declares the same selector, so taking [0] rather than the whole list is
    what makes "the cap is gone" mean the desktop cap and not the stacked one.
    """
    blocks = css_blocks(_PANEL_SELECTOR)
    assert blocks, "the board panel rule was renamed or deleted"
    return blocks[0]


def test_board_panel_is_not_capped_at_a_fixed_height():
    """Measured at 1440x900, the panel's own chrome (head, meta, table head,
    Season-0 note, footer button, padding) consumes 253px of a 520px cap, and
    seven standings rows need 202px -- leaving ~0px for a chart, and a negative
    budget at 1366x768. The cap was a card-proportion choice from when the panel
    held only a table; the board is the screen's subject now and takes the row.
    """
    block = _panel_block()
    assert "height: 100%" in block
    assert "min-height: 0" in block
    assert "max-height: none" in block
    assert "520px" not in block, (
        "the 520px cap leaves ~0px for the chart at 1440x900 and is negative at 1366x768"
    )
