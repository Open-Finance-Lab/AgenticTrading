"""
Shared test utilities for FinAgent integration tests

This module provides common utilities and synthetic agent implementations
for testing the FinAgent orchestration system without requiring external dependencies.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Any


class SyntheticDataAgent:
    """Synthetic data agent for testing purposes"""
    
    def __init__(self):
        self.name = "synthetic_data_agent"
        
    def get_tool(self, tool_name):
        if tool_name == "get_historical_data":
            return self._get_historical_data
        return None
        
    async def _get_historical_data(self, request):
        """Generate synthetic historical data"""
        symbol = request.get("symbol", "AAPL")
        start_date = request.get("start_date", "2024-01-01")
        end_date = request.get("end_date", "2024-01-31")
        
        # Generate realistic synthetic data
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days + 1
        
        random.seed(hash(symbol) % 1000)  # Consistent data per symbol
        base_price = random.uniform(90, 150)
        
        data = []
        for i in range(days):
            current_date = start_dt + timedelta(days=i)
            daily_change = random.normalvariate(0.001, 0.02)  # 0.1% mean, 2% volatility
            base_price *= (1 + daily_change)
            base_price = max(base_price, 1.0)
            
            data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "timestamp": current_date.strftime("%Y-%m-%d"),
                "open": round(base_price * 0.99, 2),
                "high": round(base_price * 1.02, 2),
                "low": round(base_price * 0.98, 2),
                "close": round(base_price, 2),
                "volume": random.randint(1000000, 10000000)
            })
        
        return {
            "status": "success",
            "symbol": symbol,
            "data": data,
            "source": "synthetic"
        }


class SyntheticAlphaAgent:
    """Synthetic alpha agent for testing purposes"""
    
    def __init__(self):
        self.name = "synthetic_alpha_agent"
        
    def get_tool(self, tool_name):
        if tool_name == "generate_alpha_signals":
            return self._generate_alpha_signals
        return None
        
    async def _generate_alpha_signals(self, request):
        """Generate synthetic alpha signals"""
        symbols = request.get("symbols", [])
        date = request.get("date", "2024-01-15")
        current_prices = request.get("current_prices", {})
        
        signals = {}
        for symbol in symbols:
            # Simple synthetic signal logic
            random.seed(hash(symbol + date) % 1000)
            momentum = random.uniform(-0.1, 0.1)
            
            if momentum > 0.03:
                direction = "buy"
                strength = min(momentum * 10, 0.8)
                predicted_return = momentum * 0.5
                execution_weight = min(momentum * 5, 0.4)
            elif momentum < -0.03:
                direction = "sell"
                strength = min(abs(momentum) * 10, 0.8)
                predicted_return = momentum * 0.5
                execution_weight = max(momentum * 5, -0.4)
            else:
                direction = "hold"
                strength = 0.1
                predicted_return = 0.0
                execution_weight = 0.0
            
            signals[symbol] = {
                "direction": direction,  # Expected by orchestrator
                "strength": strength,    # Expected by orchestrator
                "signal": direction,     # Legacy compatibility 
                "confidence": strength,  # Legacy compatibility
                "predicted_return": predicted_return,
                "risk_estimate": 0.02,
                "execution_weight": execution_weight,
                "current_price": current_prices.get(symbol, 100.0)
            }
        
        return {
            "status": "success",
            "signals": signals,
            "date": date,
            "generated_by": "synthetic_alpha_agent"
        }


def create_synthetic_sandbox() -> Dict[str, Any]:
    """Create a synthetic sandbox environment for testing"""
    return {
        "data_agent_pool": SyntheticDataAgent(),
        "alpha_agent_pool": SyntheticAlphaAgent()
    }


def create_cache_sandbox(symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
    """Create a sandbox with cached data for testing"""
    sandbox = {}
    
    # Generate synthetic data for each symbol and cache it
    data_agent = SyntheticDataAgent()
    for symbol in symbols:
        data_request = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date
        }
        # Use sync version since we're in sync context
        import asyncio
        data_result = asyncio.run(data_agent._get_historical_data(data_request))
        sandbox[f"data_{symbol}"] = data_result
    
    return sandbox
