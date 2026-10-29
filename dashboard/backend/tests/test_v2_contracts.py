

import pytest
from pydantic import ValidationError

from dashboard.backend.api.v2.models import (
    SCHEMA_VERSION, UNIVERSE, ActionItem, ContextEnvelope,
    DecisionRequest, ErrorEnvelope, NewsSentimentEntry, SubmitAck,
)


def test_schema_version_and_universe():
    assert SCHEMA_VERSION == "2.0"
    assert "AAPL" in UNIVERSE and len(UNIVERSE) == 30


def test_news_sentiment_entry_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        NewsSentimentEntry(sentiment="bullish", score=1.5, headline="x",
                           source="Reuters", url="http://x", age_hours=1.0, n_articles=1)


def test_action_item_rejects_off_universe_symbol():
    with pytest.raises(ValidationError):
        ActionItem(action="buy", symbol="ZZZ", confidence=0.5,
                   reasoning="valid reason", position_size=10)


def test_action_item_rejects_bad_confidence():
    with pytest.raises(ValidationError):
        ActionItem(action="buy", symbol="AAPL", confidence=2.0,
                   reasoning="valid reason", position_size=10)


def test_decision_request_requires_idempotency_key():
    with pytest.raises(ValidationError):
        DecisionRequest(actions=[])


def test_validate_actions_splits_valid_and_invalid():
    from dashboard.backend.api.v2.models import validate_actions
    valid, rejected = validate_actions([
        {"action": "buy", "symbol": "AAPL", "confidence": 0.7,
         "reasoning": "valid reason", "position_size": 3},
        {"action": "buy", "symbol": "ZZZ", "confidence": 0.7,
         "reasoning": "bad symbol here", "position_size": 3},
    ])
    assert len(valid) == 1 and valid[0]["symbol"] == "AAPL"
    assert rejected == [{"symbol": "ZZZ", "reason": "universe_violation"}]


def test_decision_request_accepts_raw_actions_for_partial_rejection():
    # actions are raw dicts so the boundary can drop invalid ones with reasons
    # (spec §5.3) instead of 422-ing the whole submission.
    req = DecisionRequest(idempotency_key="k", actions=[{"action": "buy", "symbol": "ZZZ"}])
    assert req.actions == [{"action": "buy", "symbol": "ZZZ"}]


def test_context_envelope_defaults_news_slot_present():
    env = ContextEnvelope(
        schema_version=SCHEMA_VERSION, run_id="run_1", mode="backtest",
        step_index=0, total_steps=10, loop="lockstep", status="loading",
        universe=UNIVERSE,
    )
    assert env.news_sentiment == {}
    assert env.news_overview is None


def test_submit_ack_minimal_valid():
    ack = SubmitAck(accepted=True, decision_source="external_agent",
                    status="waiting_decision", run_id="run_1")
    assert ack.executed == [] and ack.rejected == []


def test_error_envelope_shape():
    err = ErrorEnvelope.model_validate({"error": {
        "code": "validation_failed", "message": "bad", "retryable": False}})
    assert err.error.code == "validation_failed"


def test_news_sentiment_entry_accepts_rationale():
    entry = NewsSentimentEntry(
        sentiment="bullish", score=0.5, headline="h", source="s",
        url="https://example.com", age_hours=1.0, n_articles=2,
        rationale="Two outlets report upbeat guidance.",
    )
    assert entry.rationale == "Two outlets report upbeat guidance."


@pytest.mark.parametrize("symbol", ["aapl", "AAPL ", " Aapl", "\taapl\n"])
def test_validate_actions_normalizes_symbol_case_and_whitespace(symbol):
    """v1 (LLMTradingDecision) strips and uppercases the ticker; v2 compared it
    raw, so the same decision executed on v1 and was held on v2 as a
    "universe_violation" for a DJIA-30 name."""
    from dashboard.backend.api.v2.models import validate_actions
    valid, rejected = validate_actions([
        {"action": "buy", "symbol": symbol, "confidence": 0.8,
         "reasoning": "trend is up", "position_size": 5},
    ])
    assert rejected == []
    assert [a["symbol"] for a in valid] == ["AAPL"]


@pytest.mark.parametrize("symbol", ["zzz", " ZZZ "])
def test_validate_actions_still_rejects_off_universe_after_normalizing(symbol):
    from dashboard.backend.api.v2.models import validate_actions
    valid, rejected = validate_actions([
        {"action": "buy", "symbol": symbol, "confidence": 0.8,
         "reasoning": "trend is up", "position_size": 5},
    ])
    assert valid == []
    assert [r["reason"] for r in rejected] == ["universe_violation"]


@pytest.mark.parametrize("symbol", ["", "   "])
def test_validate_actions_rejects_blank_symbol(symbol):
    # Only that it is rejected: which reason a blank symbol earns is not part
    # of this contract.
    from dashboard.backend.api.v2.models import validate_actions
    valid, rejected = validate_actions([
        {"action": "buy", "symbol": symbol, "confidence": 0.8,
         "reasoning": "trend is up", "position_size": 5},
    ])
    assert valid == []
    assert len(rejected) == 1


@pytest.mark.parametrize("symbol", ["aapl", "AAPL ", " Aapl", "\taapl\n"])
def test_every_decision_surface_normalizes_symbols_identically(symbol):
    """All three step-driven schemas: the legacy /api/v1/backtest decision
    (LLMTradingDecision), the v1 Protocol Run API order (OrderIn, behind
    /api/v1/runs) and v2's ActionItem."""
    from dashboard.backend.domain.runs.protocol import OrderIn
    from dashboard.backend.infrastructure.llm.validator import LLMTradingDecision
    fields = {"action": "buy", "symbol": symbol, "confidence": 0.8,
              "reasoning": "trend is up", "position_size": 5}
    order = OrderIn(symbol=symbol, side="buy", quantity=5)
    assert (LLMTradingDecision(**fields).symbol == ActionItem(**fields).symbol
            == order.symbol == "AAPL")


# Unicode case mapping folds these onto DJIA-30 tickers ('ıbm'.upper() ==
# 'IBM'), and str.strip() removes the exotic padding; neither is a ticker.
_NOT_TICKERS = ["ıbm", "cſco", "mſft", "aapl\x1f", "aapl ", " aapl"]


@pytest.mark.parametrize("symbol", _NOT_TICKERS)
def test_normalize_symbol_never_folds_non_ascii_into_a_ticker(symbol):
    from dashboard.backend.infrastructure.llm.validator import (
        DJIA_30, normalize_symbol,
    )
    assert normalize_symbol(symbol) not in DJIA_30


@pytest.mark.parametrize("symbol", _NOT_TICKERS)
def test_every_decision_surface_refuses_unicode_look_alikes(symbol):
    """v2's validate_actions is also the Robinhood live path's validator, so a
    look-alike accepted here would be a real-money order there."""
    from dashboard.backend.api.v2.models import validate_actions
    from dashboard.backend.domain.runs.protocol import OrderIn
    from dashboard.backend.infrastructure.llm.validator import (
        DJIA_30, LLMTradingDecision,
    )
    fields = {"action": "buy", "symbol": symbol, "confidence": 0.8,
              "reasoning": "trend is up", "position_size": 5}
    valid, rejected = validate_actions([fields])
    assert valid == [] and [r["reason"] for r in rejected] == ["universe_violation"]
    with pytest.raises(ValidationError):
        LLMTradingDecision(**fields)
    # OrderIn only normalizes; the protocol's per-order invalid_symbol check
    # refuses anything that is not on the run's allow-list.
    assert OrderIn(symbol=symbol, side="buy", quantity=5).symbol not in DJIA_30


def test_ack_reports_executed_and_rejected_symbols_by_one_rule():
    """A client keying its submitted actions by normalize_symbol(sent) must find
    every one of them in the ack, whichever list it landed in."""
    from dashboard.backend.api.v2.models import validate_actions
    base = {"action": "buy", "confidence": 0.8, "reasoning": "trend is up",
            "position_size": 5}
    valid, rejected = validate_actions([{**base, "symbol": " aapl "},
                                        {**base, "symbol": " zzz "}])
    assert [a["symbol"] for a in valid] == ["AAPL"]
    assert rejected == [{"symbol": "ZZZ", "reason": "universe_violation"}]


def test_overlong_symbol_is_refused_and_its_echo_is_bounded():
    from dashboard.backend.api.v2.models import SYMBOL_MAX_LENGTH, validate_actions
    valid, rejected = validate_actions([
        {"action": "buy", "symbol": "x" * 100_000, "confidence": 0.8,
         "reasoning": "trend is up", "position_size": 5},
    ])
    assert valid == []
    assert rejected == [{"symbol": "X" * SYMBOL_MAX_LENGTH, "reason": "validation_failed"}]


def test_padded_symbol_within_the_length_bound_is_accepted():
    from dashboard.backend.api.v2.models import SYMBOL_MAX_LENGTH
    padded = "aapl".center(SYMBOL_MAX_LENGTH)
    assert ActionItem(action="buy", symbol=padded, confidence=0.8,
                      reasoning="trend is up", position_size=5).symbol == "AAPL"


def test_schema_documents_symbol_normalization():
    properties = ActionItem.model_json_schema()["properties"]
    description = properties["symbol"]["description"]
    assert "trimmed" in description and "uppercased" in description
    assert properties["symbol"]["maxLength"] == 32


def test_news_sentiment_entry_rationale_is_optional():
    entry = NewsSentimentEntry(
        sentiment="neutral", score=0.0, headline="h", source="s",
        url="https://example.com", age_hours=0.0, n_articles=1,
    )
    assert entry.rationale is None
