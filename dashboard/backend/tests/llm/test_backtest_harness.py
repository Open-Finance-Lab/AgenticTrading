"""Characterization tests for the extracted LLM backtest harness (Phase 2C2).

Locks in the behavior of
``dashboard.backend.infrastructure.llm.backtest_harness`` and the legacy
``PortfolioManager.make_trading_decision_with_llm`` that delegates its
infrastructure steps to it. A fake Anthropic-shaped client is used; no real
external service is ever called.
"""

import json
from datetime import datetime

from dashboard.backend.domain.backtesting import (
    portfolio_manager as portfolio_manager_module,
)
import dashboard.backend.infrastructure.llm.backtest_harness as harness
from dashboard.backend.infrastructure.llm.backtest_harness import request_trading_decision
from dashboard.scripts import backtest_hourly_agent as bha


# ---------------------------------------------------------------------------
# Fakes that reproduce the Anthropic response object shape
# ---------------------------------------------------------------------------

class _FakeUsage:
    def __init__(self, input_tokens=0, output_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeBlock:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text, usage=None):
        self.content = [_FakeBlock(text)]
        self.usage = usage


class _FakeMessages:
    def __init__(self, response, recorder):
        self._response = response
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder.update(kwargs)
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.captured = {}
        self.messages = _FakeMessages(response, self.captured)


# ---------------------------------------------------------------------------
# Import / compatibility
# ---------------------------------------------------------------------------

def test_harness_imports_without_api_key():
    # Importing the module above already succeeded without an API key.
    assert harness is not None
    assert hasattr(harness, "request_trading_decision")


def test_harness_forwards_reasoning_effort_to_provider_factory(monkeypatch):
    captured = {}

    def _fake_factory(integration=None, *, reasoning_effort=None):
        captured["integration"] = integration
        captured["reasoning_effort"] = reasoning_effort
        return "client"

    monkeypatch.setattr(harness, "_providers_make_llm_client", _fake_factory)

    assert harness.make_llm_client("openrouter", reasoning_effort="none") == "client"
    assert captured == {
        "integration": "openrouter",
        "reasoning_effort": "none",
    }


def test_script_reexports_symbols():
    assert hasattr(bha, "Anthropic")
    assert hasattr(bha, "HAS_ANTHROPIC")
    assert bha.LLM_MODEL_NAME == "claude-haiku-4-5-20251001"
    assert bha.LLM_MODEL_NAME == harness.LLM_MODEL_NAME
    assert bha.Anthropic is harness.Anthropic
    assert bha.HAS_ANTHROPIC == harness.HAS_ANTHROPIC


def test_legacy_method_still_defined():
    assert callable(bha.PortfolioManager.make_trading_decision_with_llm)


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------

def test_request_uses_default_model_and_params():
    client = _FakeClient(_FakeResponse('{"actions": []}'))
    harness.request_trading_decision(client, prompt="HELLO")
    cap = client.captured
    assert cap["model"] == harness.LLM_MODEL_NAME
    assert cap["max_tokens"] == 2000
    assert cap["system"] == harness.SYSTEM_PROMPT
    assert cap["messages"] == [{"role": "user", "content": "HELLO"}]


def test_request_uses_market_aware_a_share_system_prompt():
    client = _FakeClient(_FakeResponse('{"actions": []}'))
    market_context = {
        "market": "CN",
        "timezone": "Asia/Shanghai",
        "timeframe": "60m",
        "symbols": ["600519.SH", "601318.SH"],
        "paper_backtest": True,
    }

    request_trading_decision(
        client,
        prompt="HELLO",
        market_context=market_context,
    )

    system_prompt = client.captured["system"]
    assert "Chinese A-share" in system_prompt
    assert "Asia/Shanghai" in system_prompt
    assert "60m" in system_prompt
    assert "historical paper backtest" in system_prompt
    assert "DJIA" not in system_prompt
    assert "600519.SH" not in system_prompt


def test_request_model_override():
    client = _FakeClient(_FakeResponse("{}"))
    harness.request_trading_decision(client, prompt="P", model="custom-model")
    assert client.captured["model"] == "custom-model"


def test_request_includes_explicit_zero_temperature():
    client = _FakeClient(_FakeResponse("{}"))
    harness.request_trading_decision(client, prompt="P", temperature=0)
    assert client.captured["temperature"] == 0


def test_request_omits_temperature_when_unset():
    client = _FakeClient(_FakeResponse("{}"))
    harness.request_trading_decision(client, prompt="P")
    assert "temperature" not in client.captured


def test_system_prompt_required_fragments():
    # Assert stable required fragments of the current (unchanged) prompt.
    assert "expert quantitative trading advisor" in harness.SYSTEM_PROMPT
    assert '"actions" array' in harness.SYSTEM_PROMPT
    assert "ONLY valid JSON" in harness.SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Response text + token usage extraction
# ---------------------------------------------------------------------------

def test_extract_response_text():
    resp = _FakeResponse("hello world")
    assert harness.extract_response_text(resp) == "hello world"


def test_extract_response_text_skips_thinking_block():
    """Reasoning models (Nemotron via OpenRouter) put ThinkingBlock first."""
    class _Thinking:
        type = "thinking"
        thinking = "internal chain of thought"

    class _Text:
        type = "text"
        text = '{"actions": []}'

    class _Resp:
        content = [_Thinking(), _Text()]

    assert harness.extract_response_text(_Resp()) == '{"actions": []}'


def test_extract_response_text_joins_multiple_text_blocks():
    class _Text:
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class _Resp:
        content = [_Text('{"actions":'), _Text(" []}")]

    assert harness.extract_response_text(_Resp()) == '{"actions":\n []}'


def test_extract_response_text_raises_when_only_thinking():
    class _Thinking:
        type = "thinking"

    class _Resp:
        content = [_Thinking()]

    try:
        harness.extract_response_text(_Resp())
        assert False, "expected AttributeError"
    except AttributeError as exc:
        assert "No text content block" in str(exc)


def test_extract_token_usage_present():
    resp = _FakeResponse("{}", usage=_FakeUsage(123, 45))
    assert harness.extract_token_usage(resp) == (123, 45)


def test_extract_token_usage_missing():
    resp = _FakeResponse("{}", usage=None)
    assert harness.extract_token_usage(resp) == (0, 0)


def test_extract_token_usage_none_fields_coerced_zero():
    resp = _FakeResponse("{}", usage=_FakeUsage(None, None))
    assert harness.extract_token_usage(resp) == (0, 0)


# ---------------------------------------------------------------------------
# parse_llm_response
# ---------------------------------------------------------------------------

def test_parse_valid_json_object():
    out = harness.parse_llm_response('{"actions": [{"symbol": "AAPL", "action": "buy"}]}')
    assert out == {"actions": [{"symbol": "AAPL", "action": "buy"}]}


def test_parse_fenced_json():
    out = harness.parse_llm_response('```json\n{"actions": []}\n```')
    assert out == {"actions": []}


def test_parse_plain_fence():
    out = harness.parse_llm_response('```\n{"actions": [1]}\n```')
    assert out == {"actions": [1]}


def test_parse_surrounding_text():
    out = harness.parse_llm_response('Here is my answer: {"actions": []} thanks')
    assert out == {"actions": []}


def test_parse_no_json_returns_none():
    assert harness.parse_llm_response("no json here") is None


def test_parse_empty_string_returns_none():
    assert harness.parse_llm_response("") is None


def test_parse_trailing_comma_fixed():
    # fix_json_formatting repairs trailing commas
    out = harness.parse_llm_response('{"actions": [1, 2,]}')
    assert out == {"actions": [1, 2]}


def test_parse_unrecoverable_returns_none():
    assert harness.parse_llm_response("{ this : is : not json ]") is None


def test_parse_non_dict_json_returns_none():
    # "[1,2]" has no braces -> no JSON found -> None
    assert harness.parse_llm_response("[1, 2, 3]") is None


# ---------------------------------------------------------------------------
# Legacy method: failure / fallback behavior
# ---------------------------------------------------------------------------

def _portfolio_state():
    return {
        "timestamp": datetime(2026, 1, 1),
        "cash": 100000,
        "positions": [],
        "positions_value": 0,
        "total_equity": 100000,
        "market_signals": {
            "AAPL": {"price": 100.0, "rsi": 25.0, "macd": 1.0, "macd_signal": 0.5,
                     "sma20": 110.0, "sma50": 120.0, "bb_upper": 130.0, "bb_lower": 90.0},
        },
    }


def test_no_client_falls_back_to_rule_based():
    pm = bha.PortfolioManager(100000)
    out = pm.make_trading_decision_with_llm(_portfolio_state(), None)
    # rule-based path: AAPL rsi 25 < 30 and price < sma20 -> buy
    assert out == pm.make_trading_decision(_portfolio_state())
    assert pm.llm_calls == 0


def test_api_exception_falls_back_to_rule_based():
    class _BoomClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("boom")

    pm = bha.PortfolioManager(100000)
    out = pm.make_trading_decision_with_llm(_portfolio_state(), _BoomClient())
    assert out == pm.make_trading_decision(_portfolio_state())


def test_portfolio_forwards_zero_temperature_to_request(monkeypatch):
    captured = []
    response_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 0.9,
         "reasoning": "strong", "position_size": 1},
    ]})

    def _fake_request(client, **kwargs):
        captured.append(kwargs)
        return _FakeResponse(response_text, usage=_FakeUsage(10, 5))

    monkeypatch.setattr(
        portfolio_manager_module,
        "_request_trading_decision",
        _fake_request,
    )

    pm = bha.PortfolioManager(100000)
    pm.make_trading_decision_with_llm(
        _portfolio_state(), object(), temperature=0
    )

    assert len(captured) == 1
    assert captured[0]["temperature"] == 0


def test_portfolio_forwards_temperature_to_no_text_retry(monkeypatch):
    captured = []
    extract_attempts = 0
    response_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 0.9,
         "reasoning": "strong", "position_size": 1},
    ]})

    def _fake_request(client, **kwargs):
        captured.append(kwargs)
        return _FakeResponse(response_text, usage=_FakeUsage(10, 5))

    def _fake_extract(response):
        nonlocal extract_attempts
        extract_attempts += 1
        if extract_attempts == 1:
            raise AttributeError("No text content block in response")
        return response_text

    monkeypatch.setattr(
        portfolio_manager_module,
        "_request_trading_decision",
        _fake_request,
    )
    monkeypatch.setattr(
        portfolio_manager_module,
        "_extract_response_text",
        _fake_extract,
    )

    pm = bha.PortfolioManager(100000)
    pm.make_trading_decision_with_llm(
        _portfolio_state(), object(), temperature=0
    )

    assert len(captured) == 2
    assert [call["temperature"] for call in captured] == [0, 0]


def test_no_json_response_returns_empty_actions():
    pm = bha.PortfolioManager(100000)
    client = _FakeClient(_FakeResponse("totally not json", usage=_FakeUsage(10, 5)))
    out = pm.make_trading_decision_with_llm(_portfolio_state(), client)
    assert out == {"actions": []}
    # token usage still recorded before the parse failure
    assert pm.input_tokens == 10
    assert pm.output_tokens == 5
    assert pm.llm_calls == 1


def test_empty_actions_list_falls_back_to_rule_based():
    pm = bha.PortfolioManager(100000)
    client = _FakeClient(_FakeResponse('{"actions": []}', usage=_FakeUsage(7, 3)))
    out = pm.make_trading_decision_with_llm(_portfolio_state(), client)
    assert out == pm.make_trading_decision(_portfolio_state())
    assert pm.input_tokens == 7
    assert pm.llm_calls == 1


# ---------------------------------------------------------------------------
# llm_decisions: the H6 coverage numerator (steps the model actually drove).
# Distinct from llm_calls (billed API calls): a call whose response is empty or
# unparseable is billed but produced no usable decision, so it must NOT count
# toward model coverage — else a run that returns garbage every step would show
# 100% coverage and slip past the H6 integrity guard while trading rule-based.
# ---------------------------------------------------------------------------

def test_fresh_manager_has_zero_llm_decisions():
    assert bha.PortfolioManager(100000).llm_decisions == 0


def test_usable_decision_counts_as_llm_decision():
    pm = bha.PortfolioManager(100000)
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 0.9,
         "reasoning": "strong", "position_size": 10},
    ]})
    client = _FakeClient(_FakeResponse(resp_text, usage=_FakeUsage(100, 50)))
    pm.make_trading_decision_with_llm(_portfolio_state(), client)
    assert pm.llm_calls == 1
    assert pm.llm_decisions == 1


def test_malformed_response_is_billed_but_not_a_decision():
    # Unparseable output (e.g. JSON truncated by an output-token cap): billed,
    # but no usable model decision → counts for cost, not for coverage.
    pm = bha.PortfolioManager(100000)
    client = _FakeClient(_FakeResponse("totally not json", usage=_FakeUsage(10, 5)))
    pm.make_trading_decision_with_llm(_portfolio_state(), client)
    assert pm.llm_calls == 1
    assert pm.llm_decisions == 0


def test_empty_actions_is_billed_but_not_a_decision():
    # An empty actions list explicitly falls back to rule-based, so that step is
    # rule-based-driven and must not count toward model coverage.
    pm = bha.PortfolioManager(100000)
    client = _FakeClient(_FakeResponse('{"actions": []}', usage=_FakeUsage(7, 3)))
    pm.make_trading_decision_with_llm(_portfolio_state(), client)
    assert pm.llm_calls == 1
    assert pm.llm_decisions == 0


def test_filtered_actions_still_count_as_a_decision():
    # The model produced a non-empty decision that our confidence policy then
    # filtered out. The model still drove the step, so it counts toward coverage.
    pm = bha.PortfolioManager(100000)
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 0.1, "reasoning": "meh"},
    ]})
    client = _FakeClient(_FakeResponse(resp_text, usage=_FakeUsage(1, 1)))
    pm.make_trading_decision_with_llm(_portfolio_state(), client)
    assert pm.llm_calls == 1
    assert pm.llm_decisions == 1


def test_post_parse_exception_is_billed_but_not_a_decision():
    # Valid JSON with a non-empty actions list, but a field has the wrong type
    # (confidence as a string "0.9"), so the per-action processing loop raises.
    # The blanket except converts the whole step to a pure rule-based fallback —
    # the returned actions are the rule-based engine's, not the model's — so it
    # must NOT count toward model coverage even though parsing "succeeded".
    # (Otherwise a model that reliably emits subtly-malformed actions would show
    # 100% coverage while trading fully rule-based, defeating the H6 guard.)
    pm = bha.PortfolioManager(100000)
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": "0.9",
         "reasoning": "strong", "position_size": 10},
    ]})
    client = _FakeClient(_FakeResponse(resp_text, usage=_FakeUsage(10, 5)))
    out = pm.make_trading_decision_with_llm(_portfolio_state(), client)
    assert out == pm.make_trading_decision(_portfolio_state())  # rule-based result
    assert pm.llm_calls == 1        # billed
    assert pm.llm_decisions == 0    # but not a usable model decision


def test_malformed_action_item_is_billed_but_not_a_decision():
    # A non-empty actions list whose items are the wrong shape (strings, not
    # dicts) also throws inside the processing loop → rule-based fallback.
    pm = bha.PortfolioManager(100000)
    client = _FakeClient(_FakeResponse('{"actions": ["buy AAPL now"]}',
                                       usage=_FakeUsage(10, 5)))
    out = pm.make_trading_decision_with_llm(_portfolio_state(), client)
    assert out == pm.make_trading_decision(_portfolio_state())
    assert pm.llm_calls == 1
    assert pm.llm_decisions == 0


# ---------------------------------------------------------------------------
# Legacy method: successful BUY / SELL conversion + token accounting
# ---------------------------------------------------------------------------

def test_llm_buy_action_converted():
    pm = bha.PortfolioManager(100000)
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 0.9,
         "reasoning": "strong", "position_size": 10},
    ]})
    client = _FakeClient(_FakeResponse(resp_text, usage=_FakeUsage(100, 50)))
    out = pm.make_trading_decision_with_llm(_portfolio_state(), client)
    assert out["actions"] == [{
        "symbol": "AAPL",
        "action": "buy",
        "shares": 10,
        "reason": "[LLM] strong (confidence: 90%)",
        "confidence": 0.9,
    }]
    assert pm.input_tokens == 100
    assert pm.output_tokens == 50
    assert pm.llm_calls == 1


def test_llm_sell_action_requires_position():
    pm = bha.PortfolioManager(100000)
    pm.positions = {"AAPL": 8}
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "sell", "confidence": 0.8, "reasoning": "weak"},
    ]})
    client = _FakeClient(_FakeResponse(resp_text, usage=_FakeUsage(1, 1)))
    out = pm.make_trading_decision_with_llm(_portfolio_state(), client)
    assert out["actions"] == [{
        "symbol": "AAPL",
        "action": "sell",
        "shares": 8,
        "reason": "[LLM] weak (confidence: 80%)",
        "confidence": 0.8,
    }]


def test_llm_low_confidence_skipped_then_rule_fallback():
    # All actions skipped (low confidence) -> actions empty -> but llm_actions
    # was non-empty, so STEP 4 runs and produces []; returns {"actions": []}.
    pm = bha.PortfolioManager(100000)
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 0.1, "reasoning": "meh"},
    ]})
    client = _FakeClient(_FakeResponse(resp_text, usage=_FakeUsage(1, 1)))
    out = pm.make_trading_decision_with_llm(_portfolio_state(), client)
    assert out == {"actions": []}


def test_llm_invalid_symbol_skipped():
    pm = bha.PortfolioManager(100000)
    resp_text = json.dumps({"actions": [
        {"symbol": "NOTREAL", "action": "buy", "confidence": 0.9, "reasoning": "x",
         "position_size": 5},
    ]})
    client = _FakeClient(_FakeResponse(resp_text, usage=_FakeUsage(1, 1)))
    out = pm.make_trading_decision_with_llm(_portfolio_state(), client)
    assert out == {"actions": []}


# ---------------------------------------------------------------------------
# Legacy equivalence + subclass compatibility
# ---------------------------------------------------------------------------

def test_legacy_equivalence_full_workflow():
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 0.7,
         "reasoning": "trend", "position_size": 3},
    ]})

    # Two managers, identical fake clients -> identical results & token counts
    pm1 = bha.PortfolioManager(100000)
    pm2 = bha.PortfolioManager(100000)
    out1 = pm1.make_trading_decision_with_llm(
        _portfolio_state(), _FakeClient(_FakeResponse(resp_text, _FakeUsage(11, 22))))
    out2 = pm2.make_trading_decision_with_llm(
        _portfolio_state(), _FakeClient(_FakeResponse(resp_text, _FakeUsage(11, 22))))
    assert out1 == out2
    assert (pm1.input_tokens, pm1.output_tokens, pm1.llm_calls) == \
           (pm2.input_tokens, pm2.output_tokens, pm2.llm_calls)


def test_subclass_inherits_llm_method():
    class MyPM(bha.PortfolioManager):
        def custom_method(self):
            return "ok"

    pm = MyPM(100000)
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 0.9,
         "reasoning": "r", "position_size": 2},
    ]})
    out = pm.make_trading_decision_with_llm(
        _portfolio_state(), _FakeClient(_FakeResponse(resp_text, _FakeUsage(1, 1))))
    assert out["actions"][0]["action"] == "buy"
    assert pm.custom_method() == "ok"
    assert MyPM.make_trading_decision_with_llm is bha.PortfolioManager.make_trading_decision_with_llm


# ---------------------------------------------------------------------------
# LOW #5 — LLM_MAX_OUTPUT_TOKENS must be parsed defensively at import time
# ---------------------------------------------------------------------------

def _reload_harness_with_env(monkeypatch, value):
    import importlib
    if value is None:
        monkeypatch.delenv("LLM_MAX_OUTPUT_TOKENS", raising=False)
    else:
        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", value)
    return importlib.reload(harness)


def _restore_harness(monkeypatch):
    import importlib
    monkeypatch.delenv("LLM_MAX_OUTPUT_TOKENS", raising=False)
    importlib.reload(harness)


def test_malformed_max_output_tokens_falls_back_to_default(monkeypatch):
    """A malformed env value must not crash the module import — it falls back
    to the 2000 default (with a warning) instead of raising ValueError."""
    try:
        mod = _reload_harness_with_env(monkeypatch, "twenty")
        assert mod.DEFAULT_MAX_OUTPUT_TOKENS == 2000
    finally:
        _restore_harness(monkeypatch)


def test_nonpositive_max_output_tokens_falls_back_to_default(monkeypatch):
    """0/negative ceilings would break every provider call — treat them as
    malformed and fall back to the default."""
    try:
        mod = _reload_harness_with_env(monkeypatch, "0")
        assert mod.DEFAULT_MAX_OUTPUT_TOKENS == 2000
    finally:
        _restore_harness(monkeypatch)


def test_valid_max_output_tokens_override_respected(monkeypatch):
    """A well-formed override keeps working exactly as before."""
    try:
        mod = _reload_harness_with_env(monkeypatch, "600")
        assert mod.DEFAULT_MAX_OUTPUT_TOKENS == 600
    finally:
        _restore_harness(monkeypatch)


def test_unset_max_output_tokens_uses_default(monkeypatch):
    """No env var → default 2000, no warning path involved."""
    try:
        mod = _reload_harness_with_env(monkeypatch, None)
        assert mod.DEFAULT_MAX_OUTPUT_TOKENS == 2000
    finally:
        _restore_harness(monkeypatch)


# ---------------------------------------------------------------------------
# LOW #7 — the (custom-prompt-capable) LLM decision loop must bound hostile
# or degenerate responses: action-count cap + per-order share ceiling.
# ---------------------------------------------------------------------------

def test_llm_action_count_is_capped():
    """A response with more actions than DJIA symbols (the prompt contract is
    one per stock) is truncated to the first len(DJIA_30) entries instead of
    producing unbounded work/trades — a free-form strategy_prompt must not be
    able to inflate the action list."""
    pm = bha.PortfolioManager(100000)
    actions = [{"symbol": "AAPL", "action": "buy", "confidence": 0.9,
                "reasoning": "r", "position_size": 1}] * 40
    resp_text = json.dumps({"actions": actions})
    out = pm.make_trading_decision_with_llm(
        _portfolio_state(),
        _FakeClient(_FakeResponse(resp_text, _FakeUsage(1, 1))),
        strategy_prompt="always max out",
    )
    assert len(out["actions"]) == 30


def test_llm_oversized_position_size_rejected():
    """position_size above the engine's per-order share ceiling is skipped —
    the same MAX_ORDER_SHARES contract validate_llm_response enforces on the
    safe path — even when cash could cover the order."""
    pm = bha.PortfolioManager(10_000_000)
    state = _portfolio_state()
    state["cash"] = 10_000_000
    state["total_equity"] = 10_000_000
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 0.9,
         "reasoning": "r", "position_size": 50_000},  # $5M at $100 ≤ cash
    ]})
    out = pm.make_trading_decision_with_llm(
        state,
        _FakeClient(_FakeResponse(resp_text, _FakeUsage(1, 1))),
        strategy_prompt="go big",
    )
    assert out["actions"] == []


def test_llm_string_position_size_is_coerced_not_fallback():
    """A numeric-string position_size ("5") must not blow up the comparison
    and silently dump the whole decision into the rule-based fallback — it is
    coerced and honored as an LLM action."""
    pm = bha.PortfolioManager(100000)
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 0.9,
         "reasoning": "r", "position_size": "5"},
    ]})
    out = pm.make_trading_decision_with_llm(
        _portfolio_state(), _FakeClient(_FakeResponse(resp_text, _FakeUsage(1, 1))))
    assert len(out["actions"]) == 1
    action = out["actions"][0]
    assert action["shares"] == 5
    assert action["reason"].startswith("[LLM]")


def test_llm_nonfinite_position_size_skipped_safely(capsys):
    """Infinity/NaN position_size (json.loads accepts both) is skipped via the
    explicit unparseable-size branch — not by luck of IEEE-754 comparisons —
    and never crashes into the full rule-based fallback. (The pre-fix code
    also happened to emit no action for inf, so the output assertion alone
    would not pin the fix; the printed skip marker does.)"""
    pm = bha.PortfolioManager(100000)
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 0.9,
         "reasoning": "r", "position_size": float("inf")},
    ]})
    out = pm.make_trading_decision_with_llm(
        _portfolio_state(), _FakeClient(_FakeResponse(resp_text, _FakeUsage(1, 1))))
    assert out["actions"] == []
    assert "unparseable position_size" in capsys.readouterr().out
