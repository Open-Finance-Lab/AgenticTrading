"""
Comprehensive test suite for individual Risk Agent implementations.

Author: Jifeng Li
License: openMDW
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from FinAgents.agent_pools.risk_agent_pool.registry import (
    MarketRiskAgent, VolatilityAgent, VaRAgent, CreditRiskAgent,
    LiquidityRiskAgent, CorrelationAgent, OperationalRiskAgent, 
    StressTestingAgent, ModelRiskAgent
)

from .fixtures import (
    sample_market_data, sample_portfolio_data, mock_openai_client,
    sample_risk_context, sample_volatility_data
)


class TestMarketRiskAgent:
    """Test MarketRiskAgent functionality."""
    
    @pytest.fixture
    def agent(self):
        """Create MarketRiskAgent for testing."""
        return MarketRiskAgent(config={"test_mode": True})
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.name == "MarketRiskAgent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'config')
    
    @pytest.mark.asyncio
    async def test_analyze_market_risk(self, agent, sample_market_data):
        """Test market risk analysis."""
        context = {
            "market_data": sample_market_data,
            "portfolio": {"positions": [{"symbol": "AAPL", "quantity": 100}]}
        }
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "risk_level": "Medium",
            "market_beta": 1.2,
            "sector_exposure": {"technology": 0.6, "finance": 0.4},
            "key_risks": ["Interest rate sensitivity", "Market volatility"],
            "recommendations": ["Diversify across sectors", "Consider hedging"]
        }
        """
        
        agent.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Perform analysis
        result = await agent.analyze(context)
        
        # Verify result structure
        assert "risk_level" in result
        assert "market_beta" in result
        assert "sector_exposure" in result
        assert "key_risks" in result
        assert "recommendations" in result
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_beta(self, agent):
        """Test portfolio beta calculation."""
        portfolio_data = {
            "positions": [
                {"symbol": "AAPL", "quantity": 100, "beta": 1.2},
                {"symbol": "GOOGL", "quantity": 50, "beta": 1.1},
                {"symbol": "MSFT", "quantity": 75, "beta": 0.9}
            ]
        }
        
        beta = agent._calculate_portfolio_beta(portfolio_data)
        
        # Verify beta calculation
        assert isinstance(beta, float)
        assert 0.5 <= beta <= 2.0  # Reasonable range for portfolio beta
    
    def test_sector_analysis(self, agent):
        """Test sector exposure analysis."""
        portfolio_data = {
            "positions": [
                {"symbol": "AAPL", "sector": "Technology", "market_value": 10000},
                {"symbol": "JPM", "sector": "Finance", "market_value": 5000},
                {"symbol": "MSFT", "sector": "Technology", "market_value": 8000}
            ]
        }
        
        sector_exposure = agent._analyze_sector_exposure(portfolio_data)
        
        # Verify sector analysis
        assert "Technology" in sector_exposure
        assert "Finance" in sector_exposure
        assert abs(sector_exposure["Technology"] - 0.78) < 0.01  # 18000/23000
        assert abs(sector_exposure["Finance"] - 0.22) < 0.01    # 5000/23000


class TestVolatilityAgent:
    """Test VolatilityAgent functionality."""
    
    @pytest.fixture
    def agent(self):
        """Create VolatilityAgent for testing."""
        return VolatilityAgent(config={"test_mode": True})
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.name == "VolatilityAgent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'config')
    
    @pytest.mark.asyncio
    async def test_analyze_volatility(self, agent, sample_volatility_data):
        """Test volatility analysis."""
        context = {
            "price_data": sample_volatility_data,
            "portfolio": {"symbols": ["AAPL", "GOOGL"]}
        }
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "portfolio_volatility": 0.18,
            "individual_volatilities": {"AAPL": 0.22, "GOOGL": 0.25},
            "volatility_trend": "Increasing",
            "risk_level": "Medium-High",
            "recommendations": ["Monitor volatility spikes", "Consider volatility hedging"]
        }
        """
        
        agent.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Perform analysis
        result = await agent.analyze(context)
        
        # Verify result structure
        assert "portfolio_volatility" in result
        assert "individual_volatilities" in result
        assert "volatility_trend" in result
        assert "risk_level" in result
    
    def test_calculate_historical_volatility(self, agent):
        """Test historical volatility calculation."""
        # Generate sample price data
        prices = [100, 102, 98, 105, 103, 107, 104, 109, 106, 112]
        
        volatility = agent._calculate_historical_volatility(prices)
        
        # Verify volatility calculation
        assert isinstance(volatility, float)
        assert volatility > 0
        assert volatility < 1  # Should be less than 100%
    
    def test_garch_volatility(self, agent):
        """Test GARCH volatility modeling."""
        # Generate sample return data
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        
        garch_vol = agent._estimate_garch_volatility(returns)
        
        # Verify GARCH estimation
        assert isinstance(garch_vol, float)
        assert garch_vol > 0
        assert garch_vol < 1


class TestVaRAgent:
    """Test VaRAgent functionality."""
    
    @pytest.fixture
    def agent(self):
        """Create VaRAgent for testing."""
        return VaRAgent(config={"test_mode": True})
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.name == "VaRAgent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'config')
    
    @pytest.mark.asyncio
    async def test_calculate_var(self, agent, sample_portfolio_data):
        """Test VaR calculation."""
        context = {
            "portfolio": sample_portfolio_data,
            "confidence_level": 0.95,
            "time_horizon": 1
        }
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "var_95": 25000,
            "var_99": 35000,
            "expected_shortfall": 42000,
            "method": "Historical Simulation",
            "portfolio_value": 1000000,
            "var_percentage": 2.5,
            "risk_level": "Moderate"
        }
        """
        
        agent.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Perform analysis
        result = await agent.analyze(context)
        
        # Verify result structure
        assert "var_95" in result
        assert "var_99" in result
        assert "expected_shortfall" in result
        assert "method" in result
    
    def test_historical_var(self, agent):
        """Test historical VaR calculation."""
        # Generate sample return data
        returns = np.random.normal(-0.001, 0.02, 1000)
        portfolio_value = 1000000
        confidence_level = 0.95
        
        var = agent._calculate_historical_var(returns, portfolio_value, confidence_level)
        
        # Verify VaR calculation
        assert isinstance(var, float)
        assert var > 0  # VaR should be positive (loss amount)
        assert var < portfolio_value  # VaR shouldn't exceed portfolio value
    
    def test_parametric_var(self, agent):
        """Test parametric VaR calculation."""
        portfolio_value = 1000000
        expected_return = 0.08 / 252  # Daily return
        volatility = 0.20 / np.sqrt(252)  # Daily volatility
        confidence_level = 0.95
        
        var = agent._calculate_parametric_var(
            portfolio_value, expected_return, volatility, confidence_level
        )
        
        # Verify parametric VaR
        assert isinstance(var, float)
        assert var > 0
        assert var < portfolio_value


class TestCreditRiskAgent:
    """Test CreditRiskAgent functionality."""
    
    @pytest.fixture
    def agent(self):
        """Create CreditRiskAgent for testing."""
        return CreditRiskAgent(config={"test_mode": True})
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.name == "CreditRiskAgent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'config')
    
    @pytest.mark.asyncio
    async def test_analyze_credit_risk(self, agent):
        """Test credit risk analysis."""
        context = {
            "credit_exposures": [
                {"counterparty": "Bank A", "exposure": 1000000, "rating": "AA"},
                {"counterparty": "Corp B", "exposure": 500000, "rating": "BBB"}
            ],
            "portfolio": {"total_value": 10000000}
        }
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "overall_credit_risk": "Low-Medium",
            "total_exposure": 1500000,
            "concentration_risk": "Low",
            "avg_credit_rating": "A",
            "expected_loss": 15000,
            "recommendations": ["Monitor BBB exposures", "Diversify counterparties"]
        }
        """
        
        agent.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Perform analysis
        result = await agent.analyze(context)
        
        # Verify result structure
        assert "overall_credit_risk" in result
        assert "total_exposure" in result
        assert "concentration_risk" in result
        assert "expected_loss" in result
    
    def test_calculate_default_probability(self, agent):
        """Test default probability calculation."""
        credit_ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
        
        for rating in credit_ratings:
            prob = agent._calculate_default_probability(rating)
            assert isinstance(prob, float)
            assert 0 <= prob <= 1


class TestLiquidityRiskAgent:
    """Test LiquidityRiskAgent functionality."""
    
    @pytest.fixture
    def agent(self):
        """Create LiquidityRiskAgent for testing."""
        return LiquidityRiskAgent(config={"test_mode": True})
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.name == "LiquidityRiskAgent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'config')
    
    @pytest.mark.asyncio
    async def test_analyze_liquidity_risk(self, agent):
        """Test liquidity risk analysis."""
        context = {
            "portfolio": {
                "positions": [
                    {"symbol": "AAPL", "quantity": 1000, "avg_daily_volume": 50000000},
                    {"symbol": "SMALLCAP", "quantity": 10000, "avg_daily_volume": 100000}
                ]
            },
            "market_conditions": "normal"
        }
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "liquidity_risk_level": "Low-Medium",
            "illiquid_positions": ["SMALLCAP"],
            "liquidation_timeframe": "3-5 days",
            "bid_ask_impact": 0.02,
            "recommendations": ["Monitor small cap positions", "Maintain cash reserves"]
        }
        """
        
        agent.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Perform analysis
        result = await agent.analyze(context)
        
        # Verify result structure
        assert "liquidity_risk_level" in result
        assert "illiquid_positions" in result
        assert "liquidation_timeframe" in result
        assert "bid_ask_impact" in result


class TestOperationalRiskAgent:
    """Test OperationalRiskAgent functionality."""
    
    @pytest.fixture
    def agent(self, mock_openai_client):
        """Create OperationalRiskAgent for testing."""
        return OperationalRiskAgent(openai_client=mock_openai_client)
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.name == "operational_risk"
        assert agent.agent_type == "operational"
        assert "operational risk" in agent.description.lower()
    
    @pytest.mark.asyncio
    async def test_analyze_operational_risk(self, agent):
        """Test operational risk analysis."""
        context = {
            "systems": ["trading_system", "risk_system", "settlement_system"],
            "processes": ["trade_execution", "risk_monitoring", "compliance"],
            "recent_incidents": []
        }
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "operational_risk_level": "Low",
            "key_risk_areas": ["System reliability", "Process automation"],
            "risk_score": 2.5,
            "mitigation_effectiveness": "High",
            "recommendations": ["Regular system backups", "Process documentation"]
        }
        """
        
        agent.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Perform analysis
        result = await agent.analyze(context)
        
        # Verify result structure
        assert "operational_risk_level" in result
        assert "key_risk_areas" in result
        assert "risk_score" in result
        assert "mitigation_effectiveness" in result


class TestStressTestingAgent:
    """Test StressTestingAgent functionality."""
    
    @pytest.fixture
    def agent(self, mock_openai_client):
        """Create StressTestingAgent for testing."""
        return StressTestingAgent(openai_client=mock_openai_client)
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.name == "stress_testing"
        assert agent.agent_type == "stress"
        assert "stress test" in agent.description.lower()
    
    @pytest.mark.asyncio
    async def test_run_stress_test(self, agent, sample_portfolio_data):
        """Test stress testing analysis."""
        context = {
            "portfolio": sample_portfolio_data,
            "stress_scenarios": [
                {"name": "Market Crash", "market_shock": -0.30},
                {"name": "Interest Rate Spike", "rate_shock": 0.02}
            ]
        }
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "stress_test_results": {
                "Market Crash": {"portfolio_loss": 300000, "loss_percentage": 30.0},
                "Interest Rate Spike": {"portfolio_loss": 50000, "loss_percentage": 5.0}
            },
            "worst_case_scenario": "Market Crash",
            "survival_probability": 0.85,
            "recommendations": ["Increase hedging", "Reduce leverage"]
        }
        """
        
        agent.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Perform analysis
        result = await agent.analyze(context)
        
        # Verify result structure
        assert "stress_test_results" in result
        assert "worst_case_scenario" in result
        assert "survival_probability" in result


class TestModelRiskAgent:
    """Test ModelRiskAgent functionality."""
    
    @pytest.fixture
    def agent(self, mock_openai_client):
        """Create ModelRiskAgent for testing."""
        return ModelRiskAgent(openai_client=mock_openai_client)
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.name == "model_risk"
        assert agent.agent_type == "model"
        assert "model risk" in agent.description.lower()
    
    @pytest.mark.asyncio
    async def test_analyze_model_risk(self, agent):
        """Test model risk analysis."""
        context = {
            "models": [
                {"name": "VaR Model", "type": "risk", "last_validation": "2024-01-01"},
                {"name": "Credit Scoring", "type": "credit", "last_validation": "2023-12-01"}
            ],
            "model_performance": {"accuracy": 0.85, "backtesting_results": "pass"}
        }
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "model_risk_level": "Medium",
            "validation_status": "Current",
            "performance_issues": ["Credit model outdated"],
            "model_uncertainty": 0.15,
            "recommendations": ["Update credit model", "Increase validation frequency"]
        }
        """
        
        agent.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Perform analysis
        result = await agent.analyze(context)
        
        # Verify result structure
        assert "model_risk_level" in result
        assert "validation_status" in result
        assert "performance_issues" in result
        assert "model_uncertainty" in result


class TestAgentIntegration:
    """Test agent integration and coordination."""
    
    @pytest.mark.asyncio
    async def test_multi_agent_analysis(self, mock_openai_client, sample_risk_context):
        """Test coordinated analysis across multiple agents."""
        # Create multiple agents
        market_agent = MarketRiskAgent(openai_client=mock_openai_client)
        var_agent = VaRAgent(openai_client=mock_openai_client)
        credit_agent = CreditRiskAgent(openai_client=mock_openai_client)
        
        # Mock OpenAI responses for each agent
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"risk_level": "Medium", "analysis": "Test analysis"}'
        
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Run analyses concurrently
        tasks = [
            market_agent.analyze(sample_risk_context),
            var_agent.analyze(sample_risk_context),
            credit_agent.analyze(sample_risk_context)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all analyses completed
        assert len(results) == 3
        for result in results:
            assert "risk_level" in result
            assert "analysis" in result
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, mock_openai_client):
        """Test agent error handling."""
        agent = MarketRiskAgent(openai_client=mock_openai_client)
        
        # Mock OpenAI to raise an exception
        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        # Analysis should handle error gracefully
        result = await agent.analyze({"portfolio": "test"})
        
        # Should return error information
        assert "error" in result
        assert "API Error" in result["error"]
