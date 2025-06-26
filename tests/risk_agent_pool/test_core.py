"""
Comprehensive test suite for Risk Agent Pool core functionality.

Author: Jifeng Li
License: openMDW
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import json

from FinAgents.agent_pools.risk_agent_pool.core import RiskAgentPool
from FinAgents.agent_pools.risk_agent_pool.registry import BaseRiskAgent
from .fixtures import (
    sample_market_data, mock_openai_client, mock_memory_bridge,
    sample_portfolio_data, sample_risk_context
)


class TestRiskAgentPool:
    """Test the main RiskAgentPool class functionality."""
    
    @pytest.fixture
    def pool(self, mock_openai_client, mock_memory_bridge):
        """Create a RiskAgentPool instance for testing."""
        config = {
            "openai_client": mock_openai_client,
            "memory_bridge": mock_memory_bridge,
            "openai": {
                "api_key": "test-key",
                "model": "gpt-4"
            }
        }
        return RiskAgentPool(config=config)
    
    def test_pool_initialization(self, pool):
        """Test that RiskAgentPool initializes correctly."""
        assert pool.openai_client is not None
        assert pool.memory_bridge is not None
        assert pool.agents == {}
        assert pool.server is None
        assert pool.is_running is False
        assert len(pool.agent_registry) > 0  # Should have default agents
    
    def test_register_agent(self, pool):
        """Test agent registration functionality."""
        # Create a mock agent
        mock_agent = MagicMock(spec=BaseRiskAgent)
        mock_agent.name = "test_agent"
        mock_agent.agent_type = "test"
        
        # Register the agent
        pool.register_agent(mock_agent)
        
        # Verify registration
        assert "test_agent" in pool.agents
        assert pool.agents["test_agent"] == mock_agent
    
    def test_register_duplicate_agent(self, pool):
        """Test that registering duplicate agents raises an error."""
        # Create two agents with the same name
        agent1 = MagicMock(spec=BaseRiskAgent)
        agent1.name = "duplicate_agent"
        agent1.agent_type = "test"
        
        agent2 = MagicMock(spec=BaseRiskAgent)
        agent2.name = "duplicate_agent"
        agent2.agent_type = "test"
        
        # Register first agent
        pool.register_agent(agent1)
        
        # Attempt to register duplicate should raise error
        with pytest.raises(ValueError, match="Agent with name 'duplicate_agent' already registered"):
            pool.register_agent(agent2)
    
    def test_get_agent(self, pool):
        """Test agent retrieval functionality."""
        # Register a test agent
        mock_agent = MagicMock(spec=BaseRiskAgent)
        mock_agent.name = "test_agent"
        mock_agent.agent_type = "test"
        pool.register_agent(mock_agent)
        
        # Test successful retrieval
        retrieved_agent = pool.get_agent("test_agent")
        assert retrieved_agent == mock_agent
        
        # Test retrieval of non-existent agent
        assert pool.get_agent("non_existent") is None
    
    def test_list_agents(self, pool):
        """Test listing all registered agents."""
        # Initially should have default agents
        initial_agents = pool.list_agents()
        assert len(initial_agents) > 0
        
        # Register additional test agent
        mock_agent = MagicMock(spec=BaseRiskAgent)
        mock_agent.name = "test_agent"
        mock_agent.agent_type = "test"
        mock_agent.description = "Test agent"
        pool.register_agent(mock_agent)
        
        # Should now include the new agent
        all_agents = pool.list_agents()
        assert len(all_agents) == len(initial_agents) + 1
        assert any(agent["name"] == "test_agent" for agent in all_agents)
    
    def test_decompress_context(self, pool, sample_risk_context):
        """Test context decompression functionality."""
        # Test with string context
        string_context = "Simple risk analysis context"
        result = pool._decompress_context(string_context)
        assert result == string_context
        
        # Test with dict context
        dict_context = {"portfolio": sample_risk_context["portfolio"]}
        result = pool._decompress_context(dict_context)
        assert isinstance(result, str)
        assert "portfolio" in result.lower()
        
        # Test with list context
        list_context = ["item1", "item2", "item3"]
        result = pool._decompress_context(list_context)
        assert isinstance(result, str)
        assert "item1" in result
    
    @pytest.mark.asyncio
    async def test_analyze_risk_basic(self, pool, sample_risk_context):
        """Test basic risk analysis functionality."""
        # Mock the OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "risk_assessment": "Medium risk",
            "key_factors": ["Market volatility", "Credit exposure"],
            "recommendations": ["Diversify portfolio", "Monitor exposures"]
        })
        
        pool.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Perform risk analysis
        result = await pool.analyze_risk(
            context=sample_risk_context,
            risk_types=["market", "credit"]
        )
        
        # Verify result structure
        assert "risk_assessment" in result
        assert "key_factors" in result
        assert "recommendations" in result
        assert "analysis_metadata" in result
    
    @pytest.mark.asyncio
    async def test_analyze_risk_with_agents(self, pool, sample_risk_context):
        """Test risk analysis with specific agents."""
        # Mock agent responses
        mock_agent = AsyncMock(spec=BaseRiskAgent)
        mock_agent.name = "market_risk"
        mock_agent.agent_type = "market"
        mock_agent.analyze = AsyncMock(return_value={
            "risk_level": "High",
            "volatility": 0.25,
            "beta": 1.2
        })
        
        pool.register_agent(mock_agent)
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "risk_assessment": "High risk due to market volatility",
            "agent_results": {"market_risk": {"risk_level": "High"}},
            "recommendations": ["Reduce exposure", "Hedge positions"]
        })
        
        pool.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Perform analysis with specific agents
        result = await pool.analyze_risk(
            context=sample_risk_context,
            risk_types=["market"],
            agents=["market_risk"]
        )
        
        # Verify agent was called
        mock_agent.analyze.assert_called_once()
        assert "agent_results" in result
    
    @pytest.mark.asyncio
    async def test_analyze_risk_error_handling(self, pool, sample_risk_context):
        """Test error handling in risk analysis."""
        # Mock OpenAI to raise an exception
        pool.openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("OpenAI API error")
        )
        
        # Analysis should handle the error gracefully
        result = await pool.analyze_risk(
            context=sample_risk_context,
            risk_types=["market"]
        )
        
        # Should return error information
        assert "error" in result
        assert "OpenAI API error" in result["error"]
    
    @pytest.mark.asyncio
    async def test_start_stop_server(self, pool):
        """Test MCP server start/stop functionality."""
        # Mock server
        mock_server = MagicMock()
        mock_server.serve = AsyncMock()
        
        with patch('FinAgents.agent_pools.risk.core.Server', return_value=mock_server):
            # Test start server
            await pool.start_server(host="localhost", port=8080)
            assert pool.server == mock_server
            assert pool.is_running is True
            
            # Test stop server
            await pool.stop_server()
            assert pool.is_running is False
    
    def test_create_analysis_prompt(self, pool, sample_risk_context):
        """Test analysis prompt creation."""
        prompt = pool._create_analysis_prompt(
            context=sample_risk_context,
            risk_types=["market", "credit"],
            agent_results={"market": {"volatility": 0.2}}
        )
        
        assert isinstance(prompt, str)
        assert "risk analysis" in prompt.lower()
        assert "market" in prompt.lower()
        assert "credit" in prompt.lower()
        assert "volatility" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, pool, sample_risk_context):
        """Test concurrent risk analysis requests."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "risk_assessment": "Medium risk",
            "recommendations": ["Monitor closely"]
        })
        
        pool.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Run multiple concurrent analyses
        tasks = [
            pool.analyze_risk(
                context=sample_risk_context,
                risk_types=["market"]
            )
            for _ in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert "risk_assessment" in result
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, pool, sample_risk_context):
        """Test integration with memory bridge."""
        # Mock memory bridge methods
        pool.memory_bridge.store_analysis = AsyncMock()
        pool.memory_bridge.get_historical_data = AsyncMock(return_value=[])
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "risk_assessment": "Low risk",
            "recommendations": ["Maintain current positions"]
        })
        
        pool.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Perform analysis
        result = await pool.analyze_risk(
            context=sample_risk_context,
            risk_types=["market"],
            store_results=True
        )
        
        # Verify memory operations were called
        pool.memory_bridge.store_analysis.assert_called_once()
        
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test with valid config
        config = {
            "openai_client": MagicMock(),
            "memory_bridge": MagicMock()
        }
        pool = RiskAgentPool(config=config)
        assert pool.openai_client is not None
        assert pool.memory_bridge is not None
        
        # Test with empty config
        empty_pool = RiskAgentPool(config={})
        assert empty_pool.config == {}
    
    @pytest.mark.asyncio
    async def test_health_check(self, pool):
        """Test system health check functionality."""
        health = await pool.health_check()
        
        assert "status" in health
        assert "agents_count" in health
        assert "server_running" in health
        assert "timestamp" in health
        
        # Should be healthy by default
        assert health["status"] == "healthy"
        assert health["agents_count"] > 0
        assert health["server_running"] is False
