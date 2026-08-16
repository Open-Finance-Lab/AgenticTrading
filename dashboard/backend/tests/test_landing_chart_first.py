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
_HERO = (_HOME / "Hero.tsx").read_text(encoding="utf-8")
_BOARD = (_HOME / "BoardPreview.tsx").read_text(encoding="utf-8")

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
