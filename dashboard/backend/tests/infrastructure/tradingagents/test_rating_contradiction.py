"""Refuse a decision whose stated rating upstream's parser could not use.

Upstream's ``parse_rating`` runs two passes: an explicit ``Rating: X`` label,
then the first five-tier word appearing *anywhere* in the text. The passes are
not exclusive -- a label whose value is off the scale matches the regex but
fails the set test, so pass one falls through silently and pass two answers
from the surrounding prose.

Measured against the pinned revision ``a33fd4c`` on 2026-10-27, by executing
``tradingagents/agents/utils/rating.py`` directly:

    "The bear case recommends Sell.\\nFinal Rating: Strong Buy"  ->  'Sell'
    "We should Buy here.\\nRating: Strong Sell"                  ->  'Buy'
    "Final Rating: Strong Buy"                                   ->  'Buy'
    "Rating: Buy"                                                ->  'Buy'
    "Rating: Accumulate"                                         ->  ''

The direction reverses both ways, and the third row shows why it is hard to
catch: with no competing rating word in the prose, the same unsupported label
parses correctly by accident, so the fault is intermittent.

Nothing downstream can detect it. ATL cross-checks the rating against the
action, but both come from that one parse, so they always agree; the record
validates cleanly and becomes an order in the wrong direction.

The real fix is upstream -- pass one should stop once it has found a label --
which is a change to a shared module and to trade direction, so it is escalated
rather than made here. What this guard does is narrower and defensible: when
the model *states* a rating ATL does not support, refuse rather than trade on a
value read from elsewhere in the text.
"""

import importlib.util

import pytest

from dashboard.backend.infrastructure.tradingagents.bridge import (
    BridgeInputError,
    BridgeOutputError,
    unsupported_rating_label,
)


# The measured table above, as data. Each entry is
# (raw text, the rating upstream returns, the label ATL should refuse).
_MEASURED = (
    (
        "The bear case recommends Sell.\nFinal Rating: Strong Buy",
        "Sell",
        "Strong Buy",
    ),
    ("We should Buy here.\nRating: Strong Sell", "Buy", "Strong Sell"),
    ("Final Rating: Strong Buy", "Buy", "Strong Buy"),
    ("Rating: Accumulate", "", "Accumulate"),
)

# Transcripts the guard must leave alone, with what upstream really answers.
# These are the regression half: the first version of the guard refused every
# one of them. It returned on the first out-of-scale label it saw, but
# upstream's pass one *keeps scanning* for a label that is in scale -- and the
# hyphen in upstream's own ``[:\-]`` makes any hyphenated word on a line
# containing "rating" look like a label. Each false positive discarded a fully
# billed decision (11 sequential model calls) and held the step; the third such
# day would abort the run. Found by independent review and reproduced before
# fixing.
_LEGITIMATE = (
    # A hyphenated word after "rating" on a later line.
    ("Rating: Buy\nThe rating reflects a balanced, risk-adjusted view.", "Buy"),
    ("Rating: Sell\nA bottom-up rating review supports this.", "Sell"),
    # A secondary rating-ish label, which risk syntheses routinely carry.
    ("Final Rating: Buy\nRisk rating: Elevated", "Buy"),
    ("Rating: Overweight\nRating confidence: 8/10", "Overweight"),
    # A rating stated in prose rather than as a label: upstream resolves it
    # from its second pass, and the loose pattern's capture here would be the
    # word after the dash.
    ("Our rating is Buy - a medium-term call.", "Buy"),
    # No label at all. Pass two decides, and refusing every one of these would
    # refuse most legitimate output.
    ("The committee leans toward a Buy with reservations.", "Buy"),
    # Upstream's own tolerated spellings.
    ("Rating: **Sell**", "Sell"),
    ("rating - Hold", "Hold"),
)


def upstream_parse_rating(text, default="Hold"):
    """A faithful copy of upstream's ``parse_rating`` at revision ``a33fd4c``.

    Vendored so the backend suite can exercise the real algorithm. The bridge's
    own tests otherwise hand the generator a ``lambda`` that returns a fixed
    rating, which means no test in this repository ever ran the code whose
    behaviour every mapped order depends on.

    A copy is only worth having if it is held to the original, and it is held
    twice: ``test_the_vendored_copy_reproduces_the_measured_behaviour`` runs it
    against the measured table on every run, and
    ``test_the_measured_parser_behaviour_still_holds`` runs *upstream* against
    the same table wherever upstream is importable. If the two ever disagree,
    one of those two fails.
    """
    import re

    label_re = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)
    scale = {"buy", "overweight", "hold", "underweight", "sell"}
    for line in text.splitlines():
        match = label_re.search(line)
        if match and match.group(1).lower() in scale:
            return match.group(1).capitalize()
    for line in text.splitlines():
        for word in line.lower().split():
            clean = word.strip("*:.,")
            if clean in scale:
                return clean.capitalize()
    return default


@pytest.mark.parametrize("raw,parsed,_label", _MEASURED)
def test_the_vendored_copy_reproduces_the_measured_behaviour(raw, parsed, _label):
    assert upstream_parse_rating(raw, default="") == parsed


@pytest.mark.parametrize("raw,_parsed,label", _MEASURED)
def test_a_stated_but_unsupported_rating_is_detected(raw, _parsed, label):
    assert unsupported_rating_label(raw) == label


@pytest.mark.parametrize(
    "raw",
    [
        "Rating: Buy",
        "Final Rating: **Sell**",
        "rating - Hold",
        "Rating: Overweight",
        "RATING: UNDERWEIGHT",
        # No label at all. This also goes through pass two, but refusing every
        # unlabelled decision would refuse most legitimate output, so the rule
        # is deliberately narrower: "the model named a rating we do not
        # support".
        "The committee leans toward a Buy with reservations.",
        "",
    ],
)
def test_a_supported_or_absent_label_is_not_flagged(raw):
    assert unsupported_rating_label(raw) == ""


def test_the_first_unsupported_label_is_the_one_reported():
    raw = "Rating: Strong Buy\nRevised Rating: Accumulate"
    assert unsupported_rating_label(raw) == "Strong Buy"


@pytest.mark.parametrize("raw,_parsed", _LEGITIMATE)
def test_a_decision_upstream_resolved_correctly_is_never_refused(raw, _parsed):
    """The regression half. Each of these was refused by the first version.

    A false positive here is not harmless: it discards a decision the platform
    has already paid for and holds the step, and three of them abort the run.
    """
    assert unsupported_rating_label(raw) == ""


@pytest.mark.parametrize("raw,parsed", _LEGITIMATE)
def test_the_vendored_parser_resolves_the_legitimate_cases(raw, parsed):
    """And upstream really does resolve them, so the cases above are not merely
    asserting the guard's own opinion."""
    assert upstream_parse_rating(raw, default="") == parsed


def test_the_guard_mirrors_upstreams_first_pass_before_complaining():
    """The shape of the fix, as an invariant rather than a case list.

    If upstream's explicit-label pass resolves, the rating came from a label
    the model stated and ATL supports -- so there is nothing to refuse, however
    many other lines happen to mention a rating.
    """
    from dashboard.backend.infrastructure.tradingagents.bridge import (
        _RATING_LABEL_RE,
        _RATING_SCALE,
        _STATED_RATING_RE,
    )

    corpus = list(_LEGITIMATE) + [(raw, parsed) for raw, parsed, _ in _MEASURED]
    checked = 0
    for raw, _parsed in corpus:
        resolved = False
        for line in raw.splitlines():
            match = _RATING_LABEL_RE.search(line)
            if match and match.group(1).lower() in _RATING_SCALE:
                resolved = True
                break
        if resolved:
            checked += 1
            assert unsupported_rating_label(raw) == "", raw
    assert checked >= 4, "the invariant must actually be exercised"
    # The two patterns are deliberately different: the mirror has to be
    # upstream's exact one so it predicts upstream, and the detector a stricter
    # one so prose that merely mentions a rating is not read as stating one.
    assert _RATING_LABEL_RE.pattern != _STATED_RATING_RE.pattern
    assert _RATING_LABEL_RE.pattern == r"rating.*?[:\-][\s*]*(\w+)"


def test_detection_reads_the_raw_text_not_the_parsed_rating():
    # The point of the guard: the parsed rating is self-consistent with the
    # action, so a check built on it can never fire. This one is built on the
    # text, so it fires regardless of what the parse produced.
    raw = "The bear case recommends Sell.\nFinal Rating: Strong Buy"
    assert unsupported_rating_label(raw)


def test_the_refusal_is_transient_rather_than_fatal():
    """One bad generation must cost the step, not the run.

    A payload ATL got wrong (`BridgeInputError`) is identical on every step
    and aborts. A rating the model worded oddly on one day is a property of
    that generation, so it holds the step and the run continues -- which is
    what the separate class buys.
    """
    from dashboard.backend.infrastructure.tradingagents.adapter import (
        TradingAgentsConfigurationError,
        TradingAgentsRuntimeError,
        classify_runtime_failure,
    )

    assert not issubclass(BridgeOutputError, BridgeInputError)
    # The parent classifies by text across the process boundary; the bridge
    # prefixes its exception class, so these are the two strings it sees.
    assert (
        classify_runtime_failure("BridgeOutputError: TradingAgents stated an ...")
        is TradingAgentsRuntimeError
    )
    assert (
        classify_runtime_failure("BridgeInputError: unsupported analysts [...]")
        is TradingAgentsConfigurationError
    )


@pytest.mark.skipif(
    importlib.util.find_spec("tradingagents") is None,
    reason="upstream is installed only in the isolated venv",
)
@pytest.mark.parametrize("raw,parsed,_label", _MEASURED)
def test_the_measured_parser_behaviour_still_holds(raw, parsed, _label):
    """Re-measure the table above against whatever upstream is installed.

    Skipped wherever upstream is absent -- which is everywhere except the
    deploy image and a machine with the isolated venv on `sys.path` -- so it
    is not load-bearing. Its job is to fail loudly if a future upstream bump
    fixes the parser, because then this guard is refusing decisions that would
    now parse correctly and should be removed rather than kept.
    """
    from tradingagents.agents.utils.rating import parse_rating

    assert parse_rating(raw, default="") == parsed
