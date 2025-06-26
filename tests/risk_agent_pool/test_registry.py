"""
Unit Tests for Risk Agent Registry

Author: Jifeng Li
License: openMDW

This module contains unit tests for the risk agent registry functionality
including agent registration, discovery, and lifecycle management.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any
from unittest.mock import Mock, patch

from FinAgents.agent_pools.risk_agent_pool.registry import (
    BaseRiskAgent, MarketRiskAgent, VolatilityAgent, VaRAgent,
    CreditRiskAgent, LiquidityRiskAgent, CorrelationAgent,
    OperationalRiskAgent, StressTestingAgent, ModelRiskAgent,
    register_agent, unregister_agent, get_agent_class, list_agents,
    AGENT_REGISTRY
)
from .fixtures import sample_portfolio_data, sample_credit_data
from .utils import TestValidator, test_timer


logger = logging.getLogger(__name__)


class TestBaseRiskAgent:
    """Test the base risk agent interface"""
    
    def test_base_agent_initialization(self):
        """Test base agent initialization"""
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            BaseRiskAgent()
    
    def test_base_agent_with_config(self):
        """Test base agent with configuration"""
        class TestAgent(BaseRiskAgent):
            async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
                return {"test": "result"}
        
        config = {"param1": "value1", "param2": 42}
        agent = TestAgent(config)
        
        assert agent.config == config
        assert agent.name == "TestAgent"
        assert hasattr(agent, 'logger')
    
    @pytest.mark.asyncio
    async def test_base_agent_calculate_alias(self):
        """Test that calculate method works as alias for analyze"""
        class TestAgent(BaseRiskAgent):
            async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
                return {"analyzed": True, "request": request}
        
        agent = TestAgent()
        request = {"test": "data"}
        
        analyze_result = await agent.analyze(request)
        calculate_result = await agent.calculate(request)
        
        assert analyze_result == calculate_result
        assert analyze_result["analyzed"] is True
        assert analyze_result["request"] == request
    
    @pytest.mark.asyncio
    async def test_base_agent_cleanup(self):
        """Test agent cleanup method"""
        class TestAgent(BaseRiskAgent):
            async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
                return {}
        
        agent = TestAgent()
        # Should not raise exception
        await agent.cleanup()


class TestAgentRegistry:
    """Test agent registry functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Save original registry state
        self.original_registry = AGENT_REGISTRY.copy()
    
    def teardown_method(self):
        """Cleanup after each test method"""
        # Restore original registry state
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(self.original_registry)
    
    def test_register_valid_agent(self):
        """Test registering a valid agent"""
        class TestRiskAgent(BaseRiskAgent):
            async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
                return {"test": "result"}
        
        register_agent("test_agent", TestRiskAgent)
        
        assert "test_agent" in AGENT_REGISTRY
        assert AGENT_REGISTRY["test_agent"] == TestRiskAgent
    
    def test_register_invalid_agent(self):
        """Test registering invalid agent raises error"""
        class InvalidAgent:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseRiskAgent"):
            register_agent("invalid_agent", InvalidAgent)
    
    def test_unregister_agent(self):
        """Test unregistering an agent"""
        class TestRiskAgent(BaseRiskAgent):
            async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
                return {}
        
        register_agent("test_agent", TestRiskAgent)
        assert "test_agent" in AGENT_REGISTRY
        
        unregister_agent("test_agent")
        assert "test_agent" not in AGENT_REGISTRY
    
    def test_unregister_nonexistent_agent(self):
        """Test unregistering non-existent agent doesn't raise error"""
        # Should not raise exception
        unregister_agent("nonexistent_agent")
    
    def test_get_agent_class(self):
        """Test getting agent class by name"""
        class TestRiskAgent(BaseRiskAgent):
            async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
                return {}
        
        register_agent("test_agent", TestRiskAgent)
        
        retrieved_class = get_agent_class("test_agent")
        assert retrieved_class == TestRiskAgent
        
        # Test non-existent agent
        assert get_agent_class("nonexistent") is None
    
    def test_list_agents(self):
        """Test listing all registered agents"""
        original_count = len(list_agents())
        
        class TestAgent1(BaseRiskAgent):
            async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
                return {}
        
        class TestAgent2(BaseRiskAgent):
            async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
                return {}
        
        register_agent("test_agent_1", TestAgent1)
        register_agent("test_agent_2", TestAgent2)
        
        agents = list_agents()
        assert "test_agent_1" in agents
        assert "test_agent_2" in agents
        assert len(agents) == original_count + 2


class TestMarketRiskAgent:
    """Test market risk agent functionality"""
    
    @pytest.mark.asyncio
    async def test_market_risk_agent_initialization(self):
        """Test market risk agent initialization"""
        agent = MarketRiskAgent()
        assert agent.name == "MarketRiskAgent"
        assert hasattr(agent, 'config')
        assert hasattr(agent, 'logger')
    
    @pytest.mark.asyncio
    async def test_market_risk_volatility_calculation(self, sample_portfolio_data):
        """Test volatility calculation"""
        agent = MarketRiskAgent()
        
        request = {
            "portfolio_data": sample_portfolio_data,
            "risk_measures": ["volatility"],
            "time_horizon": "daily"
        }
        
        with test_timer:
            result = await agent.analyze(request)
        
        assert TestValidator.validate_risk_analysis_result(result)
        assert result["status"] == "success"
        assert "volatility" in result["results"]
        
        vol_result = result["results"]["volatility"]
        assert "daily_volatility" in vol_result
        assert "annualized_volatility" in vol_result
        assert isinstance(vol_result["daily_volatility"], (int, float))
        assert isinstance(vol_result["annualized_volatility"], (int, float))
    
    @pytest.mark.asyncio
    async def test_market_risk_var_calculation(self, sample_portfolio_data):
        """Test VaR calculation"""
        agent = MarketRiskAgent()
        
        request = {
            "portfolio_data": sample_portfolio_data,
            "risk_measures": ["var"],
            "time_horizon": "daily"
        }
        
        result = await agent.analyze(request)
        
        assert result["status"] == "success"
        assert "var" in result["results"]
        
        var_result = result["results"]["var"]
        assert TestValidator.validate_var_result(var_result)
    
    @pytest.mark.asyncio
    async def test_market_risk_beta_calculation(self, sample_portfolio_data):
        """Test beta calculation"""
        agent = MarketRiskAgent()
        
        request = {
            "portfolio_data": sample_portfolio_data,
            "risk_measures": ["beta"],
            "time_horizon": "daily"
        }
        
        result = await agent.analyze(request)
        
        assert result["status"] == "success"
        assert "beta" in result["results"]
        
        beta_result = result["results"]["beta"]
        assert "portfolio_beta" in beta_result
        assert isinstance(beta_result["portfolio_beta"], (int, float))
    
    @pytest.mark.asyncio
    async def test_market_risk_multiple_measures(self, sample_portfolio_data):
        """Test calculating multiple risk measures"""
        agent = MarketRiskAgent()
        
        request = {
            "portfolio_data": sample_portfolio_data,
            "risk_measures": ["volatility", "var", "beta"],
            "time_horizon": "daily"
        }
        
        result = await agent.analyze(request)
        
        assert result["status"] == "success"
        assert "volatility" in result["results"]
        assert "var" in result["results"]
        assert "beta" in result["results"]
    
    @pytest.mark.asyncio
    async def test_market_risk_error_handling(self):
        """Test error handling in market risk agent"""
        agent = MarketRiskAgent()
        
        # Test with invalid portfolio data
        request = {
            "portfolio_data": None,
            "risk_measures": ["volatility"]
        }
        
        result = await agent.analyze(request)
        
        # Should handle gracefully and return error status
        assert "status" in result
        # Error handling may vary, but should not raise exception


class TestVolatilityAgent:
    """Test volatility agent functionality"""
    
    @pytest.mark.asyncio
    async def test_volatility_agent_analysis(self, sample_portfolio_data):
        """Test comprehensive volatility analysis"""
        agent = VolatilityAgent()
        
        request = {
            "portfolio_data": sample_portfolio_data,
            "time_horizon": "daily"
        }
        
        result = await agent.analyze(request)
        
        assert TestValidator.validate_risk_analysis_result(result)
        assert result["status"] == "success"
        assert result["analysis_type"] == "volatility"
        
        results = result["results"]
        assert "historical_volatility" in results
        assert "implied_volatility" in results
        assert "volatility_forecast" in results
        assert "volatility_clustering" in results
    
    @pytest.mark.asyncio
    async def test_volatility_historical_calculation(self, sample_portfolio_data):
        """Test historical volatility calculation"""
        agent = VolatilityAgent()
        
        request = {
            "portfolio_data": sample_portfolio_data,
            "time_horizon": "daily"
        }
        
        result = await agent.analyze(request)
        
        hist_vol = result["results"]["historical_volatility"]
        assert "30_day" in hist_vol
        assert "60_day" in hist_vol
        assert "252_day" in hist_vol
        assert "method" in hist_vol
        
        # Validate volatility values are reasonable
        for key in ["30_day", "60_day", "252_day"]:
            assert isinstance(hist_vol[key], (int, float))
            assert 0 <= hist_vol[key] <= 1  # Volatility should be between 0 and 100%


class TestVaRAgent:
    """Test VaR agent functionality"""
    
    @pytest.mark.asyncio
    async def test_var_agent_multiple_methods(self, sample_portfolio_data):
        """Test VaR calculation with multiple methods"""
        agent = VaRAgent()
        
        request = {
            "portfolio_data": sample_portfolio_data,
            "time_horizon": "daily",
            "confidence_levels": [95, 99]
        }
        
        result = await agent.analyze(request)
        
        assert result["status"] == "success"
        assert result["analysis_type"] == "var"
        
        results = result["results"]
        assert "var_parametric" in results
        assert "var_historical" in results
        assert "var_monte_carlo" in results
        assert "expected_shortfall" in results
    
    @pytest.mark.asyncio
    async def test_var_confidence_levels(self, sample_portfolio_data):
        """Test VaR calculation with different confidence levels"""
        agent = VaRAgent()
        
        request = {
            "portfolio_data": sample_portfolio_data,
            "confidence_levels": [90, 95, 99, 99.9]
        }
        
        result = await agent.analyze(request)
        
        # Check that VaR increases with confidence level
        var_parametric = result["results"]["var_parametric"]["values"]
        
        assert "var_90" in var_parametric
        assert "var_95" in var_parametric
        assert "var_99" in var_parametric
        assert "var_99.9" in var_parametric
        
        # VaR should increase with confidence level
        assert var_parametric["var_90"] <= var_parametric["var_95"]
        assert var_parametric["var_95"] <= var_parametric["var_99"]


class TestCreditRiskAgent:
    """Test credit risk agent functionality"""
    
    @pytest.mark.asyncio
    async def test_credit_risk_analysis(self, sample_portfolio_data):
        """Test comprehensive credit risk analysis"""
        agent = CreditRiskAgent()
        
        request = {
            "portfolio_data": sample_portfolio_data
        }
        
        result = await agent.analyze(request)
        
        assert result["status"] == "success"
        assert result["risk_type"] == "credit"
        
        results = result["results"]
        assert "default_probability" in results
        assert "credit_spreads" in results
        assert "credit_var" in results
        assert "recovery_rates" in results
    
    @pytest.mark.asyncio
    async def test_credit_default_probability(self, sample_portfolio_data):
        """Test default probability calculation"""
        agent = CreditRiskAgent()
        
        request = {"portfolio_data": sample_portfolio_data}
        result = await agent.analyze(request)
        
        pd_result = result["results"]["default_probability"]
        assert "1_year_pd" in pd_result
        assert "3_year_pd" in pd_result
        assert "5_year_pd" in pd_result
        
        # PD should increase with time horizon
        assert pd_result["1_year_pd"] <= pd_result["3_year_pd"]
        assert pd_result["3_year_pd"] <= pd_result["5_year_pd"]


class TestLiquidityRiskAgent:
    """Test liquidity risk agent functionality"""
    
    @pytest.mark.asyncio
    async def test_liquidity_risk_analysis(self, sample_portfolio_data):
        """Test liquidity risk analysis"""
        agent = LiquidityRiskAgent()
        
        request = {
            "portfolio_data": sample_portfolio_data
        }
        
        result = await agent.analyze(request)
        
        assert result["status"] == "success"
        assert result["risk_type"] == "liquidity"
        
        results = result["results"]
        assert "liquidity_ratios" in results
        assert "bid_ask_spreads" in results
        assert "market_impact" in results
        assert "funding_liquidity" in results
    
    @pytest.mark.asyncio
    async def test_liquidity_ratios(self, sample_portfolio_data):
        """Test liquidity ratio calculations"""
        agent = LiquidityRiskAgent()
        
        request = {"portfolio_data": sample_portfolio_data}
        result = await agent.analyze(request)
        
        ratios = result["results"]["liquidity_ratios"]
        assert "current_ratio" in ratios
        assert "quick_ratio" in ratios
        assert "cash_ratio" in ratios
        
        # All ratios should be positive
        for ratio_name, ratio_value in ratios.items():
            assert isinstance(ratio_value, (int, float))
            assert ratio_value >= 0


class TestCorrelationAgent:
    """Test correlation agent functionality"""
    
    @pytest.mark.asyncio
    async def test_correlation_analysis(self, sample_portfolio_data):
        """Test correlation and dependency analysis"""
        agent = CorrelationAgent()
        
        request = {
            "portfolio_data": sample_portfolio_data
        }
        
        result = await agent.analyze(request)
        
        assert result["status"] == "success"
        assert result["risk_type"] == "systemic"
        assert result["analysis_type"] == "correlation"
        
        results = result["results"]
        assert "correlation_matrix" in results
        assert "tail_dependencies" in results
        assert "dynamic_correlations" in results
        assert "copula_analysis" in results
    
    @pytest.mark.asyncio
    async def test_correlation_matrix(self, sample_portfolio_data):
        """Test correlation matrix calculation"""
        agent = CorrelationAgent()
        
        request = {"portfolio_data": sample_portfolio_data}
        result = await agent.analyze(request)
        
        corr_matrix = result["results"]["correlation_matrix"]
        assert "average_correlation" in corr_matrix
        assert "max_correlation" in corr_matrix
        assert "min_correlation" in corr_matrix
        
        # Correlation bounds should be valid
        assert -1 <= corr_matrix["min_correlation"] <= 1
        assert -1 <= corr_matrix["max_correlation"] <= 1
        assert -1 <= corr_matrix["average_correlation"] <= 1


class TestOperationalRiskAgent:
    """Test operational risk agent functionality"""
    
    @pytest.mark.asyncio
    async def test_operational_risk_metrics(self):
        """Test operational risk metrics calculation"""
        agent = OperationalRiskAgent()
        
        request = {
            "analysis_type": "metrics",
            "start_date": None,
            "end_date": None
        }
        
        result = await agent.analyze(request)
        
        assert result["status"] == "success"
        assert result["risk_type"] == "operational"
        assert "metrics" in result["results"]
    
    @pytest.mark.asyncio
    async def test_fraud_assessment(self, sample_transaction_data):
        """Test fraud risk assessment"""
        agent = OperationalRiskAgent()
        
        request = {
            "analysis_type": "fraud_assessment",
            "transaction_data": sample_transaction_data
        }
        
        result = await agent.analyze(request)
        
        assert result["status"] == "success"
        assert "fraud_risk" in result["results"]
        
        fraud_result = result["results"]["fraud_risk"]
        assert "risk_score" in fraud_result
        assert "risk_level" in fraud_result
        assert "recommendation" in fraud_result
        
        # Risk score should be between 0 and 1
        assert 0 <= fraud_result["risk_score"] <= 1


class TestStressTestingAgent:
    """Test stress testing agent functionality"""
    
    @pytest.mark.asyncio
    async def test_scenario_stress_test(self, sample_stress_portfolio):
        """Test scenario-based stress testing"""
        agent = StressTestingAgent()
        
        request = {
            "test_type": "scenario",
            "scenario_id": "2008_financial_crisis",
            "portfolio": sample_stress_portfolio
        }
        
        result = await agent.analyze(request)
        
        assert result["status"] == "success"
        assert result["risk_type"] == "stress_testing"
        assert "stress_test" in result["results"]
    
    @pytest.mark.asyncio
    async def test_sensitivity_analysis(self, sample_stress_portfolio):
        """Test sensitivity analysis"""
        agent = StressTestingAgent()
        
        request = {
            "test_type": "sensitivity",
            "risk_factor": "equity_market",
            "shock_range": (-0.2, 0.2),
            "portfolio": sample_stress_portfolio
        }
        
        result = await agent.analyze(request)
        
        assert result["status"] == "success"
        assert "sensitivity" in result["results"]
        
        sensitivity_result = result["results"]["sensitivity"]
        assert "risk_factor" in sensitivity_result
        assert "shock_values" in sensitivity_result
        assert "portfolio_values" in sensitivity_result
    
    @pytest.mark.asyncio
    async def test_scenario_library(self):
        """Test scenario library access"""
        agent = StressTestingAgent()
        
        request = {
            "test_type": "scenario_library"
        }
        
        result = await agent.analyze(request)
        
        assert result["status"] == "success"
        assert "scenario_library" in result["results"]


class TestModelRiskAgent:
    """Test model risk agent functionality"""
    
    @pytest.mark.asyncio
    async def test_model_registration(self, sample_model_metadata):
        """Test model registration"""
        agent = ModelRiskAgent()
        
        request = {
            "action": "register_model",
            "model_metadata": sample_model_metadata
        }
        
        result = await agent.analyze(request)
        
        assert result["status"] == "success"
        assert result["risk_type"] == "model_risk"
        assert "model_id" in result["results"]
    
    @pytest.mark.asyncio
    async def test_model_validation(self, sample_model_metadata, sample_validation_config):
        """Test model validation"""
        agent = ModelRiskAgent()
        
        # First register a model
        register_request = {
            "action": "register_model",
            "model_metadata": sample_model_metadata
        }
        
        register_result = await agent.analyze(register_request)
        assert register_result["status"] == "success"
        model_id = register_result["results"]["model_id"]
        
        # Now validate the registered model
        validate_request = {
            "action": "validate_model",
            "model_id": model_id,
            "validator": "Risk Team",
            "validation_config": sample_validation_config
        }
        
        result = await agent.analyze(validate_request)
        
        assert result["status"] == "success"
        assert "validation_report" in result["results"]
    
    @pytest.mark.asyncio
    async def test_inventory_report(self):
        """Test model inventory report generation"""
        agent = ModelRiskAgent()
        
        request = {
            "action": "inventory_report",
            "filters": None
        }
        
        result = await agent.analyze(request)
        
        assert result["status"] == "success"
        assert "inventory_report" in result["results"]


class TestAgentIntegration:
    """Test agent integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_all_agents_registered(self):
        """Test that all expected agents are registered"""
        expected_agents = [
            "market_risk_agent",
            "volatility_agent",
            "var_agent",
            "credit_risk_agent",
            "liquidity_risk_agent",
            "correlation_agent",
            "operational_risk_agent",
            "stress_testing_agent",
            "model_risk_agent"
        ]
        
        registered_agents = list_agents()
        
        for agent_name in expected_agents:
            assert agent_name in registered_agents, f"Agent {agent_name} not registered"
    
    @pytest.mark.asyncio
    async def test_agent_instantiation(self):
        """Test that all registered agents can be instantiated"""
        registered_agents = list_agents()
        
        for agent_name in registered_agents:
            agent_class = get_agent_class(agent_name)
            assert agent_class is not None
            
            # Should be able to instantiate
            agent = agent_class()
            assert isinstance(agent, BaseRiskAgent)
            assert hasattr(agent, 'analyze')
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_usage(self, sample_portfolio_data):
        """Test concurrent usage of multiple agents"""
        import asyncio
        
        # Create multiple agent instances
        market_agent = MarketRiskAgent()
        volatility_agent = VolatilityAgent()
        var_agent = VaRAgent()
        
        # Create concurrent requests
        requests = [
            market_agent.analyze({
                "portfolio_data": sample_portfolio_data,
                "risk_measures": ["volatility"]
            }),
            volatility_agent.analyze({
                "portfolio_data": sample_portfolio_data
            }),
            var_agent.analyze({
                "portfolio_data": sample_portfolio_data,
                "confidence_levels": [95, 99]
            })
        ]
        
        # Run concurrently
        results = await asyncio.gather(*requests)
        
        # All should succeed
        for result in results:
            assert result["status"] == "success"
    
    @pytest.mark.asyncio 
    async def test_agent_performance(self, sample_portfolio_data):
        """Test agent performance under load"""
        agent = MarketRiskAgent()
        
        request = {
            "portfolio_data": sample_portfolio_data,
            "risk_measures": ["volatility", "var", "beta"]
        }
        
        # Run multiple times to test performance
        num_iterations = 10
        durations = []
        
        for _ in range(num_iterations):
            with test_timer:
                result = await agent.analyze(request)
            
            assert result["status"] == "success"
            durations.append(test_timer.get_duration())
        
        # Calculate performance statistics
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        
        # Performance assertions (adjust thresholds as needed)
        assert avg_duration < 1.0, f"Average duration too high: {avg_duration:.3f}s"
        assert max_duration < 2.0, f"Max duration too high: {max_duration:.3f}s"
        
        logger.info(f"Performance test - Avg: {avg_duration:.3f}s, Max: {max_duration:.3f}s")
