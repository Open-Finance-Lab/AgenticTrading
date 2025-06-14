import pytest
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime
from agent_pools.data_agent_pool.core import DataAgentPool

@pytest.fixture
def mock_agent():
    agent = Mock()
    agent.fetch.return_value = pd.DataFrame({
        'open': [1.0],
        'high': [2.0],
        'low': [0.5],
        'close': [1.5],
        'volume': [1000]
    })
    return agent

@pytest.fixture
def pool():
    with patch('agent_pools.data_agent_pool.core.preload_default_agents'):
        pool = DataAgentPool("test-pool")
        return pool

class TestDataAgentPool:
    """Unit tests for DataAgentPool core functionality"""

    def test_init_agents(self, pool):
        """Test agent initialization and categorization"""
        assert "crypto" in pool.agents
        assert "equity" in pool.agents
        assert "news" in pool.agents

    def test_determine_agent_type(self, pool, mock_agent):
        """Test agent type determination logic"""
        from agent_pools.data_agent_pool.agents.crypto.binance_agent import BinanceAgent
        
        # Test crypto agent detection
        mock_agent.__class__ = BinanceAgent
        assert pool._determine_agent_type(mock_agent) == "crypto"
        
        # Test default type
        mock_agent.__class__ = Mock
        assert pool._determine_agent_type(mock_agent) == "equity"

    @pytest.mark.asyncio
    async def test_fetch_success(self, pool, mock_agent):
        """Test successful data fetch operation"""
        pool.agents["crypto"]["test"] = mock_agent
        
        df = await pool.fetch(
            symbol="BTC/USDT",
            start="2024-01-01",
            end="2024-01-02",
            interval="1h",
            agent_type="crypto"
        )
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        mock_agent.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_all_agents_fail(self, pool):
        """Test behavior when all agents fail"""
        failing_agent = Mock()
        failing_agent.fetch.side_effect = Exception("API Error")
        pool.agents["crypto"]["test"] = failing_agent
        
        with pytest.raises(RuntimeError):
            await pool.fetch(
                symbol="BTC/USDT",
                start="2024-01-01",
                end="2024-01-02"
            )