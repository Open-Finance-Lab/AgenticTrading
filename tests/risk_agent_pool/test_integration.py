"""
End-to-end integration tests for the Risk Agent Pool.

Author: Jifeng Li
License: openMDW
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from FinAgents.agent_pools.risk_agent_pool.core import RiskAgentPool
from FinAgents.agent_pools.risk_agent_pool.memory_bridge import RiskMemoryBridge
from FinAgents.agent_pools.risk_agent_pool.registry import preload_default_agents

from .fixtures import (
    sample_market_data, sample_portfolio_data, mock_openai_client,
    mock_memory_bridge, sample_risk_context, sample_volatility_data
)


class TestRiskAgentPoolIntegration:
    """End-to-end integration tests for the Risk Agent Pool."""
    
    @pytest.fixture
    async def full_pool(self, mock_openai_client, mock_memory_bridge):
        """Create a fully configured RiskAgentPool for testing."""
        pool = RiskAgentPool(
            openai_client=mock_openai_client,
            memory_bridge=mock_memory_bridge
        )
        
        # Load default agents
        preload_default_agents(pool)
        
        return pool
    
    @pytest.mark.asyncio
    async def test_complete_risk_analysis_workflow(self, full_pool, sample_risk_context):
        """Test complete end-to-end risk analysis workflow."""
        # Mock OpenAI responses for different agents
        def mock_openai_response(content):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps(content)
            return mock_response
        
        # Mock responses for individual agents
        agent_responses = {
            "market": {"risk_level": "Medium", "beta": 1.2, "volatility": 0.18},
            "credit": {"risk_level": "Low", "exposure": 100000, "default_prob": 0.02},
            "liquidity": {"risk_level": "Low", "liquidation_time": "1-2 days"},
            "var": {"var_95": 25000, "var_99": 35000, "expected_shortfall": 40000}
        }
        
        # Mock final synthesis response
        synthesis_response = {
            "overall_risk_assessment": "Medium Risk",
            "key_risk_factors": [
                "Market volatility at 18%",
                "Portfolio beta of 1.2 indicates market sensitivity",
                "Credit exposure manageable at current levels"
            ],
            "risk_metrics": {
                "total_var_95": 25000,
                "portfolio_beta": 1.2,
                "credit_exposure": 100000,
                "liquidity_horizon": "1-2 days"
            },
            "recommendations": [
                "Monitor market volatility closely",
                "Consider hedging strategies for high beta exposure",
                "Maintain current diversification levels",
                "Review credit exposures quarterly"
            ],
            "confidence_level": 0.85,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Set up mock to return different responses based on call order
        call_count = 0
        
        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # First few calls are for individual agents
            if call_count <= len(agent_responses):
                agent_keys = list(agent_responses.keys())
                return mock_openai_response(agent_responses[agent_keys[call_count - 1]])
            else:
                # Final call is for synthesis
                return mock_openai_response(synthesis_response)
        
        full_pool.openai_client.chat.completions.create = mock_create
        
        # Mock memory operations
        full_pool.memory_bridge.store_analysis_result = AsyncMock(return_value="analysis_123")
        full_pool.memory_bridge.get_historical_analysis = AsyncMock(return_value=[])
        full_pool.memory_bridge.log_event = AsyncMock(return_value="event_123")
        
        # Perform comprehensive risk analysis
        result = await full_pool.analyze_risk(
            context=sample_risk_context,
            risk_types=["market", "credit", "liquidity", "var"],
            agents=["market_risk", "credit_risk", "liquidity_risk", "var_calculator"],
            store_results=True
        )
        
        # Verify comprehensive result structure
        assert "overall_risk_assessment" in result
        assert "key_risk_factors" in result
        assert "risk_metrics" in result
        assert "recommendations" in result
        assert "confidence_level" in result
        assert "analysis_metadata" in result
        
        # Verify specific metrics
        assert result["risk_metrics"]["total_var_95"] == 25000
        assert result["risk_metrics"]["portfolio_beta"] == 1.2
        assert result["confidence_level"] == 0.85
        
        # Verify memory operations were called
        full_pool.memory_bridge.store_analysis_result.assert_called_once()
        full_pool.memory_bridge.log_event.assert_called()
    
    @pytest.mark.asyncio
    async def test_portfolio_stress_testing_workflow(self, full_pool, sample_portfolio_data):
        """Test complete portfolio stress testing workflow."""
        stress_context = {
            "portfolio": sample_portfolio_data,
            "stress_scenarios": [
                {
                    "name": "2008 Financial Crisis",
                    "market_shock": -0.40,
                    "credit_spread_widening": 0.05,
                    "liquidity_drying": 0.70
                },
                {
                    "name": "COVID-19 Pandemic",
                    "market_shock": -0.30,
                    "volatility_spike": 2.0,
                    "sector_rotation": True
                },
                {
                    "name": "Interest Rate Shock",
                    "rate_increase": 0.03,
                    "duration_impact": -0.15,
                    "credit_tightening": 0.02
                }
            ]
        }
        
        # Mock stress testing response
        stress_response = {
            "stress_test_summary": {
                "worst_case_scenario": "2008 Financial Crisis",
                "maximum_loss": 400000,
                "maximum_loss_percentage": 40.0,
                "scenarios_passed": 1,
                "scenarios_failed": 2
            },
            "scenario_results": {
                "2008 Financial Crisis": {
                    "portfolio_loss": 400000,
                    "loss_percentage": 40.0,
                    "survival_probability": 0.60,
                    "recovery_time_months": 18
                },
                "COVID-19 Pandemic": {
                    "portfolio_loss": 300000,
                    "loss_percentage": 30.0,
                    "survival_probability": 0.75,
                    "recovery_time_months": 12
                },
                "Interest Rate Shock": {
                    "portfolio_loss": 150000,
                    "loss_percentage": 15.0,
                    "survival_probability": 0.90,
                    "recovery_time_months": 6
                }
            },
            "risk_recommendations": [
                "Increase cash reserves to 10% of portfolio",
                "Implement systematic hedging program",
                "Diversify across asset classes and geographies",
                "Establish contingency funding sources"
            ]
        }
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(stress_response)
        
        full_pool.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Perform stress testing
        result = await full_pool.analyze_risk(
            context=stress_context,
            risk_types=["stress"],
            agents=["stress_testing"]
        )
        
        # Verify stress test results
        assert "stress_test_summary" in result
        assert "scenario_results" in result
        assert "risk_recommendations" in result
        
        assert result["stress_test_summary"]["worst_case_scenario"] == "2008 Financial Crisis"
        assert result["stress_test_summary"]["maximum_loss"] == 400000
        assert len(result["scenario_results"]) == 3
    
    @pytest.mark.asyncio
    async def test_real_time_risk_monitoring(self, full_pool):
        """Test real-time risk monitoring scenario."""
        # Simulate market data updates
        market_updates = [
            {
                "timestamp": datetime.now().isoformat(),
                "market_data": {
                    "VIX": 25.5,
                    "SP500": -2.5,
                    "credit_spreads": {"IG": 120, "HY": 350},
                    "rates": {"10Y": 4.2, "2Y": 4.8}
                }
            },
            {
                "timestamp": (datetime.now() + timedelta(minutes=15)).isoformat(),
                "market_data": {
                    "VIX": 28.2,
                    "SP500": -3.8,
                    "credit_spreads": {"IG": 135, "HY": 380},
                    "rates": {"10Y": 4.3, "2Y": 4.9}
                }
            }
        ]
        
        # Mock real-time analysis responses
        monitoring_responses = [
            {
                "alert_level": "Medium",
                "triggered_thresholds": ["VIX > 25"],
                "risk_changes": {"market_risk": "increased", "volatility": "elevated"},
                "immediate_actions": ["Monitor positions closely", "Prepare hedging"]
            },
            {
                "alert_level": "High",
                "triggered_thresholds": ["VIX > 28", "SP500 decline > 3%"],
                "risk_changes": {"market_risk": "significantly_increased", "volatility": "high"},
                "immediate_actions": ["Activate hedging", "Reduce leverage", "Alert management"]
            }
        ]
        
        # Test each market update
        for i, update in enumerate(market_updates):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps(monitoring_responses[i])
            
            full_pool.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
            
            # Analyze updated market conditions
            result = await full_pool.analyze_risk(
                context=update,
                risk_types=["market", "volatility"],
                agents=["market_risk", "volatility"]
            )
            
            # Verify monitoring results
            assert "alert_level" in result
            assert "triggered_thresholds" in result
            assert "risk_changes" in result
            assert "immediate_actions" in result
            
            # Verify escalation in second update
            if i == 1:
                assert result["alert_level"] == "High"
                assert len(result["triggered_thresholds"]) >= 2
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_load(self, full_pool, sample_risk_context):
        """Test system under concurrent analysis load."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "risk_assessment": "Medium",
            "analysis_time": datetime.now().isoformat()
        })
        
        full_pool.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Create multiple concurrent analysis tasks
        concurrent_requests = 10
        tasks = []
        
        for i in range(concurrent_requests):
            # Vary the analysis slightly for each request
            context = {
                **sample_risk_context,
                "analysis_id": f"concurrent_test_{i}",
                "timestamp": datetime.now().isoformat()
            }
            
            task = full_pool.analyze_risk(
                context=context,
                risk_types=["market", "credit"],
                agents=["market_risk", "credit_risk"]
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = datetime.now()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.now()
        
        # Verify all requests completed
        assert len(results) == concurrent_requests
        
        # Check for errors
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Found {len(errors)} errors in concurrent execution"
        
        # Verify reasonable performance (should complete within reasonable time)
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 30, f"Concurrent execution took too long: {execution_time}s"
        
        # Verify all results have expected structure
        for result in results:
            assert "risk_assessment" in result
            assert "analysis_metadata" in result
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self, full_pool, sample_risk_context):
        """Test system error recovery and resilience."""
        # Test OpenAI API failure
        full_pool.openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("OpenAI API temporarily unavailable")
        )
        
        result = await full_pool.analyze_risk(
            context=sample_risk_context,
            risk_types=["market"]
        )
        
        # Should return error but not crash
        assert "error" in result
        assert "OpenAI API temporarily unavailable" in result["error"]
        
        # Test memory bridge failure
        full_pool.memory_bridge.store_analysis_result = AsyncMock(
            side_effect=Exception("Memory storage error")
        )
        
        # Reset OpenAI to work
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"risk_assessment": "Medium"})
        full_pool.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Analysis should complete despite memory error
        result = await full_pool.analyze_risk(
            context=sample_risk_context,
            risk_types=["market"],
            store_results=True
        )
        
        # Should have analysis result but note storage failure
        assert "risk_assessment" in result
        assert "storage_error" in result.get("warnings", []) or "error" in result
    
    @pytest.mark.asyncio
    async def test_historical_analysis_comparison(self, full_pool, sample_portfolio_data):
        """Test historical analysis comparison functionality."""
        # Mock historical data
        historical_analyses = [
            {
                "timestamp": (datetime.now() - timedelta(days=30)).isoformat(),
                "risk_assessment": "Low",
                "var_95": 15000,
                "portfolio_beta": 0.9
            },
            {
                "timestamp": (datetime.now() - timedelta(days=15)).isoformat(),
                "risk_assessment": "Medium",
                "var_95": 20000,
                "portfolio_beta": 1.1
            }
        ]
        
        full_pool.memory_bridge.get_historical_analysis = AsyncMock(
            return_value=historical_analyses
        )
        
        # Mock current analysis response
        current_analysis = {
            "risk_assessment": "Medium-High",
            "var_95": 30000,
            "portfolio_beta": 1.3,
            "trend_analysis": {
                "risk_trend": "Increasing",
                "var_trend": "Increasing significantly",
                "beta_trend": "Increasing",
                "risk_velocity": "Accelerating"
            },
            "historical_comparison": {
                "vs_30_days_ago": {"risk": "+2 levels", "var": "+100%", "beta": "+44%"},
                "vs_15_days_ago": {"risk": "+1 level", "var": "+50%", "beta": "+18%"}
            }
        }
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(current_analysis)
        
        full_pool.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Perform analysis with historical comparison
        result = await full_pool.analyze_risk(
            context={"portfolio": sample_portfolio_data, "include_historical": True},
            risk_types=["market", "var"]
        )
        
        # Verify historical comparison results
        assert "trend_analysis" in result
        assert "historical_comparison" in result
        assert result["trend_analysis"]["risk_trend"] == "Increasing"
        assert "vs_30_days_ago" in result["historical_comparison"]
    
    @pytest.mark.asyncio
    async def test_custom_agent_integration(self, full_pool, mock_openai_client):
        """Test integration of custom risk agents."""
        from FinAgents.agent_pools.risk_agent_pool.registry import BaseRiskAgent
        
        # Create custom risk agent
        class CustomESGRiskAgent(BaseRiskAgent):
            def __init__(self, openai_client):
                super().__init__(
                    name="esg_risk",
                    agent_type="esg",
                    description="ESG risk analysis agent",
                    openai_client=openai_client
                )
            
            async def analyze(self, context):
                # Mock custom analysis
                return {
                    "esg_score": 7.5,
                    "environmental_risk": "Medium",
                    "social_risk": "Low",
                    "governance_risk": "Low-Medium",
                    "esg_trend": "Improving"
                }
        
        # Register custom agent
        custom_agent = CustomESGRiskAgent(mock_openai_client)
        full_pool.register_agent(custom_agent)
        
        # Verify agent registration
        assert "esg_risk" in full_pool.agents
        assert full_pool.get_agent("esg_risk") == custom_agent
        
        # Mock OpenAI response for combined analysis
        combined_response = {
            "overall_risk_assessment": "Medium",
            "traditional_risk": "Medium",
            "esg_risk_impact": "Positive",
            "esg_score": 7.5,
            "recommendations": [
                "Maintain ESG improvements",
                "Monitor traditional risk factors"
            ]
        }
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(combined_response)
        
        full_pool.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Perform analysis including custom agent
        result = await full_pool.analyze_risk(
            context={"portfolio": sample_portfolio_data, "esg_data": {"score": 7.5}},
            risk_types=["market", "esg"],
            agents=["market_risk", "esg_risk"]
        )
        
        # Verify custom agent integration
        assert "esg_score" in result
        assert "esg_risk_impact" in result
        assert result["esg_score"] == 7.5
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, full_pool):
        """Test comprehensive system health monitoring."""
        # Mock memory bridge health
        full_pool.memory_bridge.health_check = AsyncMock(return_value={
            "status": "healthy",
            "memory_status": "operational",
            "cache_status": "operational",
            "latency_ms": 15
        })
        
        # Perform health check
        health = await full_pool.health_check()
        
        # Verify health check results
        assert "status" in health
        assert "agents_count" in health
        assert "server_running" in health
        assert "memory_bridge_status" in health
        assert "timestamp" in health
        
        # Verify specific health metrics
        assert health["status"] == "healthy"
        assert health["agents_count"] > 0
        assert health["server_running"] is False  # Server not started in tests
        assert health["memory_bridge_status"]["status"] == "healthy"
