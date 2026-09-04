"""Escalating the output ceiling on the first retry instead of the fifth.

An empty reply is what a reply looks like when reasoning ran out the output
ceiling before any text was emitted. Re-asking at the ceiling that just failed
asks for the same failure, and the single-prompt loop does that four times
before raising it. ``pipeline_runner`` already escalates on its first retry;
these tests pin the two paths to the same policy.

The first request is deliberately untouched, so a decision that succeeds
first time issues exactly the request it does today. That is what keeps this a
cost change rather than a behaviour change.
"""

from datetime import datetime

import pytest

from dashboard.backend.domain.backtesting import portfolio_manager as pm
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.infrastructure.llm.pipeline_runner import (
    RECOVERY_MAX_OUTPUT_TOKENS,
    escalate_ceiling_on_retry,
)

DECISION_JSON = '{"actions": []}'


class _Usage:
    input_tokens = 100
    output_tokens = 2000


class _Resp:
    """`text=None` is the thinking-only reply the retry loop exists for."""

    def __init__(self, text=None):
        self._text = text
        self.usage = _Usage()
        self.content = []


def _state():
    return {
        "timestamp": datetime(2026, 4, 1, 15, 30),
        "cash": 1000.0,
        "positions": [],
        "positions_value": 0.0,
        "total_equity": 1000.0,
        "market_signals": {
            "AAPL": {"price": 250.0, "rsi": 50.0, "macd": 0.0,
                     "macd_signal": 0.0, "sma_20": 250.0, "sma_50": 250.0,
                     "bb_upper": 260.0, "bb_lower": 240.0, "volume": 1000},
        },
    }


@pytest.fixture
def spy(monkeypatch):
    """Record the max_tokens of every request the loop issues."""
    calls = []
    plan = {"texts": []}

    _SENTINEL = object()

    def fake_request(client, *, prompt, model=None, max_tokens=_SENTINEL,
                     temperature=None, market_context=None):
        # Records the sentinel when the kwarg was not passed at all, so the
        # "unescalated calls are identical" claim is tested on call shape and
        # not just on the resolved value.
        calls.append(None if max_tokens is _SENTINEL else max_tokens)
        i = len(calls) - 1
        texts = plan["texts"]
        return _Resp(texts[i] if i < len(texts) else DECISION_JSON)

    def fake_extract(response):
        if response._text is None:
            raise AttributeError(
                "No text content block in LLM response "
                "(content types: ['thinking'])")
        return response._text

    monkeypatch.setattr(pm, "_request_trading_decision", fake_request)
    monkeypatch.setattr(pm, "_extract_response_text", fake_extract)
    monkeypatch.setattr(pm, "_extract_token_usage", lambda r: (100, 2000))
    return calls, plan


def _run(manager, monkeypatch, on):
    if on:
        monkeypatch.setenv("LLM_ESCALATE_CEILING_ON_RETRY", "1")
    else:
        monkeypatch.delenv("LLM_ESCALATE_CEILING_ON_RETRY", raising=False)
    return manager.make_trading_decision_with_llm(
        _state(), llm_client=object(), model="m")


def _manager():
    return PortfolioManager(initial_capital=1000.0, allowed_symbols=["AAPL"])


# ------------------------------------------------------------------- flag ---

def test_the_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("LLM_ESCALATE_CEILING_ON_RETRY", raising=False)
    assert escalate_ceiling_on_retry() is False


@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("", False), ("nope", False),
])
def test_the_flag_parses_the_usual_spellings(monkeypatch, raw, expected):
    monkeypatch.setenv("LLM_ESCALATE_CEILING_ON_RETRY", raw)
    assert escalate_ceiling_on_retry() is expected


# --------------------------------------------------- the first request ------

def test_a_first_time_success_is_one_unchanged_request(spy, monkeypatch):
    calls, plan = spy
    plan["texts"] = [DECISION_JSON]
    _run(_manager(), monkeypatch, on=True)
    assert calls == [None], "the flag must not touch a request that succeeds"


def test_the_first_request_is_never_escalated(spy, monkeypatch):
    calls, plan = spy
    plan["texts"] = [None, DECISION_JSON]
    _run(_manager(), monkeypatch, on=True)
    assert calls[0] is None


# ------------------------------------------------------- off vs on ----------

def test_off_retries_at_the_ceiling_that_just_failed(spy, monkeypatch):
    """Current behaviour: four attempts at the default, then the rescue."""
    calls, plan = spy
    plan["texts"] = [None, None, None, None, DECISION_JSON]
    _run(_manager(), monkeypatch, on=False)
    assert calls == [None, None, None, None, RECOVERY_MAX_OUTPUT_TOKENS]


def test_on_escalates_from_the_first_retry(spy, monkeypatch):
    calls, plan = spy
    plan["texts"] = [None, None, None, None, DECISION_JSON]
    _run(_manager(), monkeypatch, on=True)
    assert calls == [None] + [RECOVERY_MAX_OUTPUT_TOKENS] * 4


def test_on_reaches_the_working_ceiling_four_calls_sooner(spy, monkeypatch):
    """The saving, stated as a number: the ceiling that eventually works is
    reached on call 2 rather than call 5."""
    calls_off, plan = spy
    plan["texts"] = [None, None, None, None, DECISION_JSON]
    _run(_manager(), monkeypatch, on=False)
    first_off = calls_off.index(RECOVERY_MAX_OUTPUT_TOKENS)
    calls_off.clear()
    _run(_manager(), monkeypatch, on=True)
    first_on = calls_off.index(RECOVERY_MAX_OUTPUT_TOKENS)
    assert first_off == 4 and first_on == 1


def test_a_retry_that_succeeds_stops_the_loop(spy, monkeypatch):
    calls, plan = spy
    plan["texts"] = [None, DECISION_JSON]
    _run(_manager(), monkeypatch, on=True)
    assert calls == [None, RECOVERY_MAX_OUTPUT_TOKENS]


# ------------------------------------------- interaction with recovery ------

def test_a_reply_won_at_the_recovery_ceiling_does_not_retry_again(
        spy, monkeypatch):
    """`recovery_spent` must be set when the escalated retry is what produced
    the text; otherwise the post-parse truncation retry re-issues an identical
    request, which upstream's own comment says has nothing left to ask for."""
    calls, plan = spy
    # Unparseable text won at the escalated ceiling.
    plan["texts"] = [None, "not json at all"]
    _run(_manager(), monkeypatch, on=True)
    assert calls == [None, RECOVERY_MAX_OUTPUT_TOKENS], (
        "a third request would be the same request again")


def test_with_the_flag_off_the_truncation_retry_still_runs(spy, monkeypatch):
    calls, plan = spy
    plan["texts"] = ["not json at all"]
    _run(_manager(), monkeypatch, on=False)
    assert RECOVERY_MAX_OUTPUT_TOKENS in calls[1:], (
        "the existing post-parse recovery must be untouched when off")


def test_billing_counts_every_attempt_either_way(spy, monkeypatch):
    calls, plan = spy
    plan["texts"] = [None, None, DECISION_JSON]
    m = _manager()
    _run(m, monkeypatch, on=True)
    assert m.llm_calls == len(calls) == 3
