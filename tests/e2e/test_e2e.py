import pytest
import asyncio
import pandas as pd
from agent_pools.data_agent_pool.core import DataAgentPool

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

@pytest.mark.e2e
class TestEndToEnd:
    """End-to-end tests for DataAgentPool system"""

    @pytest.mark.asyncio
    async def test_full_data_pipeline(self):
        """Test complete data pipeline from request to response"""
        pool = DataAgentPool("e2e-test")
        
        # Test crypto market data fetch
        df = await pool.fetch(
            symbol="BTC/USDT",
            start="2024-01-01",
            end="2024-01-02",
            interval="1h",
            agent_type="crypto"
        )
        
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"open", "high", "low", "close", "volume"}
        assert len(df) > 0

    @pytest.mark.asyncio
    async def test_failover_scenario(self):
        """Test system resilience in failure scenarios"""
        pool = DataAgentPool("e2e-test-failover")
        
        # Force primary agent to fail
        pool.agents["crypto"]["primary"].fetch = lambda *args: exec('raise Exception("Simulated failure")')
        
        # Should failover to backup agent
        df = await pool.fetch(
            symbol="BTC/USDT",
            start="2024-01-01",
            end="2024-01-02",
            agent_type="crypto"
        )
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty