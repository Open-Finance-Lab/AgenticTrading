"""
Unit tests for LLM Response Validator

Tests:
- Valid JSON responses accepted
- Invalid JSON rejected
- Tool calling attempts rejected
- Invalid symbols rejected
- Portfolio constraint violations rejected
- Valid trading decisions executed
"""

import pytest
import json

from dashboard.backend.infrastructure.llm.validator import (
    TRADING_ACTIONS,
    validate_llm_response,
    TradingAction,
    create_safe_prompt,
    model_number,
    normalize_action,
    read_confidence,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def portfolio_state():
    """Sample portfolio state for validation"""
    return {
        "cash": 50000,
        "positions": [
            {"symbol": "AAPL", "shares": 100},
            {"symbol": "MSFT", "shares": 50}
        ],
        "constraints": {
            "cash_available": 50000,
            "max_position_size": 5000,
            "max_daily_trades": 20,
            "max_position_value": 50000,
            "min_confidence": 0.3
        }
    }


@pytest.fixture
def current_prices():
    """Sample current prices"""
    return {
        "AAPL": 150.00,
        "MSFT": 420.00,
        "JPM": 180.00,
        "V": 290.00,
        "JNJ": 160.00
    }


# ============================================================================
# VALID RESPONSES (should be accepted)
# ============================================================================

class TestValidResponses:
    """Test that valid LLM responses are accepted"""
    
    def test_valid_buy_decision(self, portfolio_state, current_prices):
        """Valid BUY decision should be accepted"""
        response = json.dumps({
            "action": "buy",
            "symbol": "JPM",
            "confidence": 0.75,
            "reasoning": "RSI oversold, technical setup favorable",
            "position_size": 100
        })
        
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is not None
        assert decision.action == TradingAction.BUY
        assert decision.symbol == "JPM"
        assert decision.confidence == 0.75
        assert decision.position_size == 100
    
    def test_valid_sell_decision(self, portfolio_state, current_prices):
        """Valid SELL decision should be accepted"""
        response = json.dumps({
            "action": "sell",
            "symbol": "AAPL",
            "confidence": 0.85,
            "reasoning": "Profit target reached, taking gains",
            "position_size": 50
        })
        
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is not None
        assert decision.action == TradingAction.SELL
        assert decision.symbol == "AAPL"
    
    def test_valid_hold_decision(self, portfolio_state, current_prices):
        """Valid HOLD decision should be accepted"""
        response = json.dumps({
            "action": "hold",
            "symbol": "V",
            "confidence": 0.5,
            "reasoning": "Waiting for better setup",
            "position_size": 0
        })
        
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is not None
        assert decision.action == TradingAction.HOLD
    
    def test_decision_with_stop_loss(self, portfolio_state, current_prices):
        """Decision with stop loss and take profit accepted"""
        response = json.dumps({
            "action": "buy",
            "symbol": "V",
            "confidence": 0.8,
            "reasoning": "Buying with risk management",
            "position_size": 50,
            "stop_loss_price": 280.00,
            "take_profit_price": 305.00
        })
        
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is not None
        assert decision.stop_loss_price == 280.00
        assert decision.take_profit_price == 305.00


# ============================================================================
# INVALID JSON (should be rejected)
# ============================================================================

class TestInvalidJSON:
    """Test that invalid JSON is rejected"""
    
    def test_malformed_json(self, portfolio_state, current_prices):
        """Malformed JSON should be rejected"""
        response = '{"action": "buy", "symbol": "AAPL"'  # Missing closing brace
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None
    
    def test_empty_json(self, portfolio_state, current_prices):
        """Empty JSON should be rejected"""
        response = "{}"
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None
    
    def test_non_json_response(self, portfolio_state, current_prices):
        """Non-JSON text should be rejected"""
        response = "I think we should buy AAPL because the price is low"
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None


# ============================================================================
# TOOL CALLING ATTEMPTS (security critical - must reject)
# ============================================================================

class TestToolCallingRejection:
    """Test that tool calling attempts are REJECTED (security critical)"""
    
    def test_tool_use_block_rejected(self, portfolio_state, current_prices):
        """LLM tool_use block should be rejected"""
        response = '''
        {
          "action": "buy",
          "symbol": "AAPL",
          "confidence": 0.8,
          "reasoning": "Based on analysis",
          "position_size": 100
        }
        <tool_use>
            get_latest_news("AAPL")
        </tool_use>
        '''
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None  # MUST be rejected
    
    def test_tool_calls_attribute_rejected(self, portfolio_state, current_prices):
        """LLM tool_calls attribute should be rejected"""
        response = '''{
          "action": "buy",
          "symbol": "AAPL",
          "confidence": 0.8,
          "reasoning": "Based on websearch",
          "position_size": 100,
          "tool_calls": [
            {
              "name": "web_search",
              "arguments": {"query": "AAPL earnings"}
            }
          ]
        }'''
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None  # MUST be rejected
    
    def test_function_calls_rejected(self, portfolio_state, current_prices):
        """LLM function_calls should be rejected"""
        response = '''{
          "action": "buy",
          "symbol": "AAPL",
          "function_calls": [
            {
              "function": "call_alpaca_api",
              "args": {"action": "buy", "symbol": "AAPL"}
            }
          ]
        }'''
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None  # MUST be rejected
    
    def test_api_access_attempt_rejected(self, portfolio_state, current_prices):
        """Any mention of 'invoke' or API access rejected"""
        response = '''{
          "action": "buy",
          "symbol": "AAPL",
          "reasoning": "Let me invoke the Alpaca API",
          "position_size": 100
        }'''
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None  # MUST be rejected


# ============================================================================
# INVALID SCHEMA (field violations)
# ============================================================================

class TestInvalidSchema:
    """Test that invalid schema is rejected"""
    
    def test_invalid_symbol(self, portfolio_state, current_prices):
        """Invalid symbol should be rejected"""
        response = json.dumps({
            "action": "buy",
            "symbol": "XYZ123",  # Not in DJIA
            "confidence": 0.8,
            "reasoning": "Buying fake stock",
            "position_size": 100
        })
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None
    
    def test_confidence_out_of_range(self, portfolio_state, current_prices):
        """Confidence > 1.0 should be rejected"""
        response = json.dumps({
            "action": "buy",
            "symbol": "AAPL",
            "confidence": 1.5,  # Out of range
            "reasoning": "Very confident",
            "position_size": 100
        })
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None
    
    def test_negative_position_size(self, portfolio_state, current_prices):
        """Negative position size should be rejected"""
        response = json.dumps({
            "action": "buy",
            "symbol": "AAPL",
            "confidence": 0.8,
            "reasoning": "Buying negative shares",
            "position_size": -100  # Invalid
        })
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None
    
    def test_position_size_too_large(self, portfolio_state, current_prices):
        """Position size > 10,000 should be rejected"""
        response = json.dumps({
            "action": "buy",
            "symbol": "AAPL",
            "confidence": 0.8,
            "reasoning": "Buying too much",
            "position_size": 50000  # Way too large
        })
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None
    
    def test_missing_required_field(self, portfolio_state, current_prices):
        """Missing required fields should be rejected"""
        response = json.dumps({
            "action": "buy",
            "symbol": "AAPL",
            # Missing confidence, reasoning, position_size
        })
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None
    
    def test_invalid_action(self, portfolio_state, current_prices):
        """Invalid action value should be rejected"""
        response = json.dumps({
            "action": "short",  # Invalid (must be buy/sell/hold)
            "symbol": "AAPL",
            "confidence": 0.8,
            "reasoning": "Shorting",
            "position_size": 100
        })
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None


# ============================================================================
# PORTFOLIO CONSTRAINT VIOLATIONS
# ============================================================================

class TestPortfolioConstraints:
    """Test that portfolio constraints are enforced"""
    
    def test_insufficient_cash(self, portfolio_state, current_prices):
        """BUY when insufficient cash should be rejected"""
        response = json.dumps({
            "action": "buy",
            "symbol": "AAPL",
            "confidence": 0.8,
            "reasoning": "Buying despite no cash",
            "position_size": 500  # 500 * $150 = $75,000 > $50,000 cash
        })
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None  # Should be rejected
    
    def test_position_exceeds_max(self, portfolio_state, current_prices):
        """Position size > max should be rejected"""
        response = json.dumps({
            "action": "buy",
            "symbol": "V",
            "confidence": 0.8,
            "reasoning": "Buying too much",
            "position_size": 6000  # Exceeds max of 5000
        })
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None
    
    def test_sell_nonexistent_position(self, portfolio_state, current_prices):
        """SELL of non-held position should be rejected"""
        response = json.dumps({
            "action": "sell",
            "symbol": "JPM",  # Not in portfolio
            "confidence": 0.8,
            "reasoning": "Selling position we don't have",
            "position_size": 100
        })
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None
    
    def test_sell_more_than_held(self, portfolio_state, current_prices):
        """SELL more shares than held should be rejected"""
        response = json.dumps({
            "action": "sell",
            "symbol": "AAPL",
            "confidence": 0.8,
            "reasoning": "Selling more than we have",
            "position_size": 200  # Have 100, trying to sell 200
        })
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is None
    
    def test_low_confidence_becomes_hold(self, portfolio_state, current_prices):
        """Low confidence decision becomes HOLD"""
        response = json.dumps({
            "action": "buy",
            "symbol": "V",
            "confidence": 0.25,  # Below min of 0.3
            "reasoning": "Not very confident",
            "position_size": 100
        })
        decision = validate_llm_response(response, portfolio_state, current_prices)
        assert decision is not None
        assert decision.action == TradingAction.HOLD  # Downgraded to HOLD


# ============================================================================
# SAFE PROMPT GENERATION
# ============================================================================

class TestSafePromptGeneration:
    """Test that safe prompts are generated correctly"""
    
    def test_prompt_contains_constraints(self):
        """Prompt should contain the no-tools + JSON-only constraints.

        The wording tracks the current active prompt (the strategy body was
        rewritten; the old CANNOT-phrased prompt is retired). These lines are
        defense-in-depth only — the hard boundary is response-side
        enforcement in validate_llm_response (tool_use rejected, JSON-only),
        pinned by the TestToolCallPrevention tests."""
        snapshot = {"test": "data"}
        prompt = create_safe_prompt(snapshot)

        assert "No internet, tools, APIs, or code" in prompt
        assert "Use ONLY the provided market_snapshot" in prompt
        assert "ONLY valid JSON" in prompt
    
    def test_prompt_contains_schema(self):
        """Prompt should contain response schema"""
        snapshot = {"test": "data"}
        prompt = create_safe_prompt(snapshot)
        
        assert '"action"' in prompt
        assert '"symbol"' in prompt
        assert '"confidence"' in prompt
        assert '"reasoning"' in prompt
        assert '"position_size"' in prompt
    
    def test_prompt_contains_symbols(self):
        """Prompt should contain valid symbols"""
        snapshot = {"test": "data"}
        prompt = create_safe_prompt(snapshot)
        
        for symbol in ["AAPL", "MSFT", "JPM"]:
            assert symbol in prompt


# ============================================================================
# Integration: Full E2E flow
# ============================================================================

class TestE2EFlow:
    """Test complete end-to-end trading decision flow"""
    
    def test_e2e_valid_buy_to_sell(self, portfolio_state, current_prices):
        """E2E: Valid BUY followed by SELL"""
        
        # Step 1: BUY decision
        buy_response = json.dumps({
            "action": "buy",
            "symbol": "V",
            "confidence": 0.75,
            "reasoning": "Strong technical setup",
            "position_size": 100
        })
        buy_decision = validate_llm_response(buy_response, portfolio_state, current_prices)
        assert buy_decision is not None
        
        # Step 2: Update portfolio with new position
        updated_portfolio = portfolio_state.copy()
        updated_portfolio["positions"].append({
            "symbol": "V",
            "shares": 100
        })
        updated_portfolio["cash"] = 50000 - (100 * 290)
        
        # Step 3: SELL decision
        sell_response = json.dumps({
            "action": "sell",
            "symbol": "V",
            "confidence": 0.8,
            "reasoning": "Target reached",
            "position_size": 50
        })
        sell_decision = validate_llm_response(sell_response, updated_portfolio, current_prices)
        assert sell_decision is not None
        assert sell_decision.action == TradingAction.SELL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestReadConfidence:
    """read_confidence: one confidence rule for the backtest gate and every
    pipeline envelope it reads."""

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_missing_confidence_takes_the_default(self, value):
        assert read_confidence(value, 0.75) == 0.75

    @pytest.mark.parametrize(
        "value, expected",
        [(0, 0.0), (0.0, 0.0), ("0", 0.0), (0.3, 0.3), (" 0.4 ", 0.4), (1, 1.0)],
    )
    def test_a_number_from_0_to_1_is_kept(self, value, expected):
        assert read_confidence(value, 0.75) == expected

    @pytest.mark.parametrize(
        "value",
        [False, True, "high", "50%", [], {}, [0.5], float("nan"), "NaN",
         float("inf"), float("-inf"), 25, 1.5, -0.1,
         pytest.param(10**400, id="int-past-float")],
    )
    def test_anything_else_skips_the_action(self, value):
        assert read_confidence(value, 0.75) is None

    def test_model_number_reads_an_oversized_int_as_infinite(self):
        assert model_number(10**400) == float("inf")
        assert model_number(-(10**400)) == float("-inf")
        assert model_number(True) is None
        assert model_number(None, 1.0) == 1.0


class TestNormalizeAction:
    """normalize_action: one reading of a model's action for every path that
    matches it against TRADING_ACTIONS (#658)."""

    @pytest.mark.parametrize(
        "value, expected",
        [("buy", "buy"), ("buy ", "buy"), ("  BUY\n", "buy"), ("\tSell\r", "sell"),
         ("Hold", "hold"), (" stop_loss\n", "stop_loss")],
    )
    def test_ascii_whitespace_is_trimmed_and_case_folded(self, value, expected):
        assert normalize_action(value) == expected

    def test_a_missing_action_is_empty_for_the_caller_to_read(self):
        assert normalize_action(None) == ""

    @pytest.mark.parametrize(
        "value", [1, ["buy"], {"a": "buy"}, True, "Buy.", "buy_more", "long"]
    )
    def test_anything_else_never_equals_an_action(self, value):
        assert normalize_action(value) not in TRADING_ACTIONS

    @pytest.mark.parametrize("value", ["buy ", " sell", "buy\x1c"])
    def test_unicode_whitespace_is_not_trimmed(self, value):
        # The same set normalize_symbol trims, so one order's symbol and
        # action are never held to different rules.
        assert normalize_action(value) not in TRADING_ACTIONS

    def test_trading_actions_is_the_enum(self):
        assert TRADING_ACTIONS == {action.value for action in TradingAction}
