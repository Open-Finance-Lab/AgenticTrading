"""
Simplified Agent Tests for Risk Agent Pool.

Author: Jifeng Li
License: openMDW
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from FinAgents.agent_pools.risk_agent_pool.registry import (
    BaseRiskAgent, MarketRiskAgent, VolatilityAgent, VaRAgent, 
    CreditRiskAgent, LiquidityRiskAgent, CorrelationAgent,
    OperationalRiskAgent, StressTestingAgent, ModelRiskAgent
)


class TestAgentBasics:
    """Test basic functionality of all risk agents."""
    
    def test_base_agent_abstract(self):
        """Test that BaseRiskAgent is abstract."""
        with pytest.raises(TypeError):
            BaseRiskAgent()
    
    def test_market_risk_agent_creation(self):
        """Test MarketRiskAgent creation."""
        agent = MarketRiskAgent()
        assert agent.name == "MarketRiskAgent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'config')
    
    def test_volatility_agent_creation(self):
        """Test VolatilityAgent creation."""
        agent = VolatilityAgent()
        assert agent.name == "VolatilityAgent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'config')
    
    def test_var_agent_creation(self):
        """Test VaRAgent creation."""
        agent = VaRAgent()
        assert agent.name == "VaRAgent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'config')
    
    def test_credit_risk_agent_creation(self):
        """Test CreditRiskAgent creation."""
        agent = CreditRiskAgent()
        assert agent.name == "CreditRiskAgent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'config')
    
    def test_liquidity_risk_agent_creation(self):
        """Test LiquidityRiskAgent creation."""
        agent = LiquidityRiskAgent()
        assert agent.name == "LiquidityRiskAgent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'config')
    
    def test_correlation_agent_creation(self):
        """Test CorrelationAgent creation."""
        agent = CorrelationAgent()
        assert agent.name == "CorrelationAgent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'config')
    
    def test_operational_risk_agent_creation(self):
        """Test OperationalRiskAgent creation."""
        agent = OperationalRiskAgent()
        assert agent.name == "OperationalRiskAgent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'config')
    
    def test_stress_testing_agent_creation(self):
        """Test StressTestingAgent creation."""
        agent = StressTestingAgent()
        assert agent.name == "StressTestingAgent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'config')
    
    def test_model_risk_agent_creation(self):
        """Test ModelRiskAgent creation."""
        agent = ModelRiskAgent()
        assert agent.name == "ModelRiskAgent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'config')


class TestAgentAnalysis:
    """Test agent analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_market_risk_analysis(self):
        """Test MarketRiskAgent analysis."""
        agent = MarketRiskAgent()
        
        # Test with sample request
        request = {
            "portfolio_data": {
                "positions": [
                    {"symbol": "AAPL", "quantity": 100, "price": 150.0}
                ]
            },
            "risk_measures": ["volatility", "var", "beta"],
            "time_horizon": "daily"
        }
        
        result = await agent.analyze(request)
        
        # Verify result structure
        assert "agent" in result
        assert "risk_type" in result
        assert "status" in result
        assert result["agent"] == "MarketRiskAgent"
        assert result["risk_type"] == "market"
    
    @pytest.mark.asyncio
    async def test_volatility_analysis(self):
        """Test VolatilityAgent analysis."""
        agent = VolatilityAgent()
        
        request = {
            "portfolio_data": {
                "price_data": [100, 102, 99, 103, 101],
                "symbols": ["AAPL"]
            },
            "time_horizon": "daily"
        }
        
        result = await agent.analyze(request)
        
        # Verify result structure
        assert "agent" in result
        assert "risk_type" in result
        assert "analysis_type" in result
        assert result["agent"] == "VolatilityAgent"
        assert result["analysis_type"] == "volatility"
    
    @pytest.mark.asyncio
    async def test_var_analysis(self):
        """Test VaRAgent analysis."""
        agent = VaRAgent()
        
        request = {
            "portfolio_data": {
                "positions": [
                    {"symbol": "AAPL", "quantity": 100, "price": 150.0}
                ]
            },
            "time_horizon": "daily",
            "confidence_levels": [95, 99]
        }
        
        result = await agent.analyze(request)
        
        # Verify result structure
        assert "agent" in result
        assert "risk_type" in result
        assert "analysis_type" in result
        assert result["agent"] == "VaRAgent"
        assert result["analysis_type"] == "var"
    
    @pytest.mark.asyncio
    async def test_credit_risk_analysis(self):
        """Test CreditRiskAgent analysis."""
        agent = CreditRiskAgent()
        
        request = {
            "portfolio_data": {
                "credit_exposures": [
                    {"counterparty": "Bank A", "exposure": 1000000, "rating": "AA"}
                ]
            }
        }
        
        result = await agent.analyze(request)
        
        # Verify result structure
        assert "agent" in result
        assert "risk_type" in result
        assert result["agent"] == "CreditRiskAgent"
        assert result["risk_type"] == "credit"
    
    @pytest.mark.asyncio
    async def test_liquidity_risk_analysis(self):
        """Test LiquidityRiskAgent analysis."""
        agent = LiquidityRiskAgent()
        
        request = {
            "portfolio_data": {
                "positions": [
                    {"symbol": "AAPL", "quantity": 1000, "avg_daily_volume": 50000000}
                ]
            },
            "market_conditions": "normal"
        }
        
        result = await agent.analyze(request)
        
        # Verify result structure
        assert "agent" in result
        assert "risk_type" in result
        assert result["agent"] == "LiquidityRiskAgent"
        assert result["risk_type"] == "liquidity"
    
    @pytest.mark.asyncio
    async def test_correlation_analysis(self):
        """Test CorrelationAgent analysis."""
        agent = CorrelationAgent()
        
        request = {
            "portfolio_data": {
                "positions": [
                    {"symbol": "AAPL", "returns": [0.01, -0.02, 0.03]},
                    {"symbol": "GOOGL", "returns": [0.02, -0.01, 0.02]}
                ]
            }
        }
        
        result = await agent.analyze(request)
        
        # Verify result structure
        assert "agent" in result
        assert "risk_type" in result
        assert "analysis_type" in result
        assert result["agent"] == "CorrelationAgent"
        assert result["analysis_type"] == "correlation"
    
    @pytest.mark.asyncio
    async def test_operational_risk_analysis(self):
        """Test OperationalRiskAgent analysis."""
        agent = OperationalRiskAgent()
        
        request = {
            "analysis_type": "metrics",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        }
        
        result = await agent.analyze(request)
        
        # Verify result structure
        assert "agent" in result
        assert "risk_type" in result
        assert result["agent"] == "OperationalRiskAgent"
        assert result["risk_type"] == "operational"
    
    @pytest.mark.asyncio
    async def test_stress_testing_analysis(self):
        """Test StressTestingAgent analysis."""
        agent = StressTestingAgent()
        
        request = {
            "test_type": "scenario",
            "portfolio": [
                {"symbol": "AAPL", "quantity": 100, "price": 150.0}
            ],
            "scenario_id": "market_crash"
        }
        
        result = await agent.analyze(request)
        
        # Verify result structure
        assert "agent" in result
        assert "risk_type" in result
        assert result["agent"] == "StressTestingAgent"
        assert result["risk_type"] == "stress_testing"
    
    @pytest.mark.asyncio
    async def test_model_risk_analysis(self):
        """Test ModelRiskAgent analysis."""
        agent = ModelRiskAgent()
        
        request = {
            "action": "inventory_report",
            "filters": {"model_type": "risk"}
        }
        
        result = await agent.analyze(request)
        
        # Verify result structure
        assert "agent" in result
        assert "risk_type" in result
        assert result["agent"] == "ModelRiskAgent"
        assert result["risk_type"] == "model_risk"


class TestAgentErrorHandling:
    """Test agent error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_request_handling(self):
        """Test agents handle empty requests gracefully."""
        agents = [
            MarketRiskAgent(),
            VolatilityAgent(),
            VaRAgent(),
            CreditRiskAgent(),
            LiquidityRiskAgent()
        ]
        
        for agent in agents:
            result = await agent.analyze({})
            
            # Should return error status or minimal result
            assert "agent" in result
            assert result["agent"] == agent.name
    
    @pytest.mark.asyncio
    async def test_invalid_data_handling(self):
        """Test agents handle invalid data gracefully."""
        agent = MarketRiskAgent()
        
        # Test with invalid portfolio data
        request = {
            "portfolio_data": "invalid_data",
            "risk_measures": ["volatility"]
        }
        
        result = await agent.analyze(request)
        
        # Should handle gracefully
        assert "agent" in result
        assert result["agent"] == "MarketRiskAgent"
    
    @pytest.mark.asyncio
    async def test_agent_cleanup(self):
        """Test agent cleanup functionality."""
        agents = [
            MarketRiskAgent(),
            VolatilityAgent(),
            VaRAgent()
        ]
        
        for agent in agents:
            # Cleanup should not raise exceptions
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_analysis(self):
        """Test concurrent analysis by multiple agents."""
        agents = [
            MarketRiskAgent(),
            VolatilityAgent(),
            VaRAgent()
        ]
        
        request = {
            "portfolio_data": {
                "positions": [
                    {"symbol": "AAPL", "quantity": 100, "price": 150.0}
                ]
            }
        }
        
        # Run analyses concurrently
        tasks = [agent.analyze(request) for agent in agents]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert "agent" in result
            assert "status" in result
