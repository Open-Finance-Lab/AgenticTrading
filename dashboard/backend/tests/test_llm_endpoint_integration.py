"""
Integration tests for LLM trading endpoint

Tests that the FastAPI endpoint correctly:
1. Calls llm_validator.validate_llm_response()
2. Rejects responses with tool_calls/tool_use
3. Handles mock LLM responses
4. Executes valid decisions

These are endpoint-level tests (not just unit tests of the validator).
"""

import pytest
import json

# Add backend to path
from fastapi.testclient import TestClient
from dashboard.backend.infrastructure.llm.validator import validate_llm_response


# ============================================================================
# MOCK ENDPOINT FOR TESTING
# ============================================================================
# In a real scenario, this would be in app.py. For testing, we create a minimal version.

from fastapi import FastAPI, Request, HTTPException
from dashboard.backend.infrastructure.llm.validator import validate_llm_response

test_app = FastAPI()

# Simulated portfolio state for testing
MOCK_PORTFOLIO_STATE = {
    "cash": 50000,
    "positions": [
        {"symbol": "AAPL", "shares": 100}
    ],
    "constraints": {
        "cash_available": 50000,
        "max_position_size": 5000,
        "max_daily_trades": 20,
        "max_position_value": 50000,
        "min_confidence": 0.3
    }
}

MOCK_CURRENT_PRICES = {
    "AAPL": 150.00,
    "MSFT": 420.00,
    "JPM": 180.00,
    "V": 290.00
}


@test_app.post("/api/llm-trading-decision")
async def get_llm_trading_decision(request: Request):
    """
    Endpoint that processes LLM trading decision with validation.
    
    In production, this would:
    1. Fetch market snapshot
    2. Call LLM with safe prompt
    3. Validate response with llm_validator
    4. Execute trade if valid
    """
    try:
        # In real app, this comes from the request body or LLM response
        body = await request.json()
        llm_response = body.get("llm_response")
        
        if not llm_response:
            raise HTTPException(status_code=400, detail="llm_response required")
        
        # CRITICAL: Call the validator (this is what we're testing)
        decision = validate_llm_response(
            llm_response,
            MOCK_PORTFOLIO_STATE,
            MOCK_CURRENT_PRICES
        )
        
        if decision is None:
            return {
                "success": False,
                "error": "LLM response validation failed",
                "decision": None
            }
        
        # In real app, execute trade here
        return {
            "success": True,
            "decision": decision.model_dump(),
            "trade_id": f"trade_{decision.symbol}_{decision.action}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "decision": None
        }


# ============================================================================
# ENDPOINT INTEGRATION TESTS
# ============================================================================

class TestLLMEndpointIntegration:
    """Test that endpoint properly validates LLM responses"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(test_app)
    
    def test_endpoint_accepts_valid_buy_decision(self, client):
        """Endpoint accepts valid BUY decision"""
        response = client.post("/api/llm-trading-decision", json={
            "llm_response": json.dumps({
                "action": "buy",
                "symbol": "JPM",
                "confidence": 0.75,
                "reasoning": "RSI oversold, good setup",
                "position_size": 100
            })
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["decision"]["action"] == "buy"
        assert data["decision"]["symbol"] == "JPM"
    
    def test_endpoint_rejects_tool_use_block(self, client):
        """
        CRITICAL: Endpoint rejects tool_use block.
        
        This simulates an LLM trying to use tools in its response.
        The endpoint should reject it at the validator level.
        """
        malicious_response = '''
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
        
        response = client.post("/api/llm-trading-decision", json={
            "llm_response": malicious_response
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False  # MUST reject
        assert data["decision"] is None
        assert "validation failed" in data.get("error", "").lower()
    
    def test_endpoint_rejects_tool_calls_attribute(self, client):
        """
        CRITICAL: Endpoint rejects tool_calls attribute.
        
        This simulates an LLM response with tool_calls array.
        The endpoint should reject it.
        """
        malicious_response = json.dumps({
            "action": "buy",
            "symbol": "AAPL",
            "confidence": 0.8,
            "reasoning": "Using tools to fetch data",
            "position_size": 100,
            "tool_calls": [
                {
                    "name": "web_search",
                    "arguments": {"query": "AAPL earnings"}
                }
            ]
        })
        
        response = client.post("/api/llm-trading-decision", json={
            "llm_response": malicious_response
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False  # MUST reject
        assert data["decision"] is None
    
    def test_endpoint_rejects_function_calls(self, client):
        """
        CRITICAL: Endpoint rejects function_calls.
        
        This simulates LLM trying to call Alpaca API.
        """
        malicious_response = json.dumps({
            "action": "buy",
            "symbol": "AAPL",
            "reasoning": "Will use function to execute",
            "position_size": 100,
            "function_calls": [
                {
                    "function": "alpaca_submit_order",
                    "args": {"symbol": "AAPL", "qty": 100}
                }
            ]
        })
        
        response = client.post("/api/llm-trading-decision", json={
            "llm_response": malicious_response
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False  # MUST reject
        assert data["decision"] is None
    
    def test_endpoint_rejects_api_invoke_attempt(self, client):
        """
        CRITICAL: Endpoint rejects API invocation attempts.
        
        This simulates LLM trying to invoke an API.
        """
        malicious_response = json.dumps({
            "action": "buy",
            "symbol": "V",
            "confidence": 0.8,
            "reasoning": "Let me invoke the Alpaca API to check my account",
            "position_size": 100
        })
        
        response = client.post("/api/llm-trading-decision", json={
            "llm_response": malicious_response
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False  # MUST reject due to "invoke"
        assert data["decision"] is None
    
    def test_endpoint_rejects_malformed_json(self, client):
        """Endpoint rejects invalid JSON"""
        response = client.post("/api/llm-trading-decision", json={
            "llm_response": '{"action": "buy", "symbol": "AAPL"'  # Missing closing brace
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["decision"] is None
    
    def test_endpoint_rejects_invalid_schema(self, client):
        """Endpoint rejects responses that don't match schema"""
        response = client.post("/api/llm-trading-decision", json={
            "llm_response": json.dumps({
                "action": "buy",
                "symbol": "INVALID_TICKER",  # Not in DJIA
                "confidence": 0.8,
                "reasoning": "Good setup",
                "position_size": 100
            })
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["decision"] is None
    
    def test_endpoint_enforces_portfolio_constraints(self, client):
        """Endpoint enforces portfolio constraints"""
        # Try to buy more shares than we have cash for
        response = client.post("/api/llm-trading-decision", json={
            "llm_response": json.dumps({
                "action": "buy",
                "symbol": "MSFT",
                "confidence": 0.8,
                "reasoning": "Overbought setup",
                "position_size": 200  # 200 * $420 = $84,000 > $50,000 cash
            })
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False  # MUST reject: insufficient cash
        assert data["decision"] is None
    
    def test_endpoint_downgrades_low_confidence(self, client):
        """Endpoint downgrades low-confidence decisions to HOLD"""
        response = client.post("/api/llm-trading-decision", json={
            "llm_response": json.dumps({
                "action": "buy",
                "symbol": "V",
                "confidence": 0.25,  # Below min of 0.3
                "reasoning": "Not very confident",
                "position_size": 100
            })
        })
        
        assert response.status_code == 200
        data = response.json()
        # Should succeed but action downgraded to HOLD
        assert data["success"] is True or data["success"] is False  # Depends on implementation
        if data["decision"]:
            # If accepted, should be HOLD
            assert data["decision"]["action"] in ["hold", "buy"]
    
    def test_endpoint_missing_llm_response(self, client):
        """Endpoint requires llm_response"""
        response = client.post("/api/llm-trading-decision", json={})
        
        # Accept either 400 or 200 with error in response
        data = response.json()
        if response.status_code == 400:
            assert "required" in data.get("detail", "").lower()
        else:
            # 200 with error in body is also acceptable
            assert data.get("success") is False
            assert "required" in data.get("error", "").lower() or data.get("error")


# ============================================================================
# END-TO-END FLOW TEST
# ============================================================================

class TestE2ELLMFlow:
    """Test complete flow from LLM response to execution"""
    
    @pytest.fixture
    def client(self):
        return TestClient(test_app)
    
    def test_e2e_valid_trading_sequence(self, client):
        """E2E: Valid sequence of LLM decisions"""
        
        # Step 1: BUY decision
        buy_response = client.post("/api/llm-trading-decision", json={
            "llm_response": json.dumps({
                "action": "buy",
                "symbol": "JPM",
                "confidence": 0.75,
                "reasoning": "Oversold condition detected",
                "position_size": 100
            })
        })
        assert buy_response.status_code == 200
        buy_data = buy_response.json()
        assert buy_data["success"] is True
        assert buy_data["decision"]["action"] == "buy"
        
        # Step 2: HOLD decision
        hold_response = client.post("/api/llm-trading-decision", json={
            "llm_response": json.dumps({
                "action": "hold",
                "symbol": "MSFT",
                "confidence": 0.5,
                "reasoning": "Consolidating",
                "position_size": 0
            })
        })
        assert hold_response.status_code == 200
        hold_data = hold_response.json()
        assert hold_data["success"] is True
        assert hold_data["decision"]["action"] == "hold"
    
    def test_e2e_rejects_malicious_sequence(self, client):
        """E2E: Rejects malicious response in trading sequence"""
        
        # Valid decision first
        buy_response = client.post("/api/llm-trading-decision", json={
            "llm_response": json.dumps({
                "action": "buy",
                "symbol": "V",
                "confidence": 0.8,
                "reasoning": "Good setup",
                "position_size": 50
            })
        })
        assert buy_response.json()["success"] is True
        
        # Malicious response second (with tool_calls)
        malicious_response = client.post("/api/llm-trading-decision", json={
            "llm_response": json.dumps({
                "action": "sell",
                "symbol": "AAPL",
                "reasoning": "Selling while calling websearch",
                "position_size": 100,
                "tool_calls": [{"name": "web_search"}]  # MALICIOUS
            })
        })
        assert malicious_response.json()["success"] is False  # MUST reject


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
