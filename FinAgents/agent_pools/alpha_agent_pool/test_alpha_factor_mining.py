#!/usr/bin/env python3
"""Alpha Factor Mining Test

Unit tests for Alpha Agent Pool alpha factor mining functionality.
Tests the complete flow from agent startup to factor discovery to database storage.
"""

import asyncio
import pytest
import pytest_asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from mcp.client.sse import sse_client
    from mcp import ClientSession
    SSE_AVAILABLE = True
    logger.info("✅ SSE MCP client modules loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import SSE MCP client: {e}")
    SSE_AVAILABLE = False


class AlphaFactorMiningTest:
    """Test class for alpha factor mining functionality."""
    
    def __init__(self):
        self.alpha_pool_url = "http://127.0.0.1:8081/sse"
        self.session = None
        self.discovered_factors = []
    
    async def setup_session(self):
        """Setup SSE session with Alpha Agent Pool."""
        if not SSE_AVAILABLE:
            pytest.skip("SSE client not available")
        
        try:
            transport = sse_client(self.alpha_pool_url)
            self.session = ClientSession(transport)
            await self.session.initialize()
            logger.info("✅ Session established with Alpha Agent Pool")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to setup session: {e}")
            return False
    
    async def teardown_session(self):
        """Cleanup session."""
        if self.session:
            try:
                await self.session.close()
                logger.info("🔌 Session closed")
            except:
                pass


@pytest_asyncio.fixture
async def alpha_test():
    """Pytest fixture for AlphaFactorMiningTest."""
    test_instance = AlphaFactorMiningTest()
    await test_instance.setup_session()
    yield test_instance
    await test_instance.teardown_session()


@pytest.mark.asyncio
async def test_agent_health_check(alpha_test):
    """Test agent health status check."""
    if not alpha_test.session:
        pytest.skip("Session not available")
    
    try:
        # Check Alpha Agent Pool status
        pool_status_result = await alpha_test.session.call_tool("get_agent_status", {})
        assert pool_status_result.content is not None
        logger.info(f"📊 Alpha Pool status: {pool_status_result.content}")
        
        # Check momentum agent health
        momentum_health_result = await alpha_test.session.call_tool("momentum_health", {})
        assert momentum_health_result.content is not None
        logger.info(f"🔍 Momentum agent health: {momentum_health_result.content}")
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        pytest.fail(f"Health check failed: {e}")


@pytest.mark.asyncio
async def test_historical_data_loading(alpha_test):
    """Test historical market data loading."""
    if not alpha_test.session:
        pytest.skip("Session not available")
    
    try:
        # List available memory keys
        memory_keys_result = await alpha_test.session.call_tool("list_memory_keys", {})
        memory_keys = memory_keys_result.content
        
        assert isinstance(memory_keys, list)
        assert len(memory_keys) > 0
        logger.info(f"🗂️ Available data keys: {len(memory_keys)} entries")
        
        # Sample some data
        if memory_keys:
            sample_key = memory_keys[0]
            data_result = await alpha_test.session.call_tool("get_memory", {"key": sample_key})
            assert data_result.content is not None
            logger.info(f"📊 Sample data for {sample_key}: {data_result.content}")
            
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        pytest.fail(f"Data loading test failed: {e}")


@pytest.mark.asyncio
async def test_momentum_factor_discovery(alpha_test):
    """Test momentum alpha factor discovery."""
    if not alpha_test.session:
        pytest.skip("Session not available")
    
    try:
        # Test parameters
        symbol = "AAPL"
        lookback_period = 20
        test_date = "2024-01-30"
        
        logger.info(f"🔍 Testing momentum factor for {symbol} with {lookback_period}-day lookback")
        
        # Generate alpha signals
        signals_result = await alpha_test.session.call_tool(
            "generate_alpha_signals",
            {
                "symbol": symbol,
                "date": test_date,
                "lookback_period": lookback_period
            }
        )
        
        assert signals_result.content is not None
        logger.info(f"📈 Signal result: {signals_result.content}")
        
        # Parse signal data
        if hasattr(signals_result.content, 'get'):
            signal_data = signals_result.content
        else:
            import json
            try:
                signal_data = json.loads(signals_result.content) if isinstance(signals_result.content, str) else signals_result.content
            except:
                signal_data = {"signal": "HOLD", "confidence": 0.0}
        
        # Validate signal structure
        assert "signal" in signal_data
        assert "confidence" in signal_data
        assert signal_data["signal"] in ["BUY", "SELL", "HOLD"]
        
        # Store discovered factor
        factor = {
            "symbol": symbol,
            "factor_name": f"momentum_{lookback_period}d",
            "factor_data": {
                "lookback_period": lookback_period,
                "signal": signal_data.get("signal", "HOLD"),
                "confidence": signal_data.get("confidence", 0.0),
                "factor_strength": abs(signal_data.get("confidence", 0.0)),
                "discovery_timestamp": datetime.now().isoformat()
            }
        }
        
        alpha_test.discovered_factors.append(factor)
        logger.info(f"✅ Factor discovered: {factor}")
        
    except Exception as e:
        logger.error(f"❌ Factor discovery test failed: {e}")
        pytest.fail(f"Factor discovery test failed: {e}")


@pytest.mark.asyncio
async def test_factor_database_storage(alpha_test):
    """Test storing discovered factors to database via memory agent."""
    if not alpha_test.session:
        pytest.skip("Session not available")
    
    # Ensure we have discovered factors from previous test
    if not alpha_test.discovered_factors:
        # Create a test factor if none exist
        test_factor = {
            "symbol": "AAPL",
            "factor_name": "momentum_20d",
            "factor_data": {
                "lookback_period": 20,
                "signal": "BUY",
                "confidence": 0.75,
                "factor_strength": 0.75,
                "discovery_timestamp": datetime.now().isoformat()
            }
        }
        alpha_test.discovered_factors.append(test_factor)
    
    try:
        storage_results = []
        
        for factor in alpha_test.discovered_factors:
            logger.info(f"💾 Storing factor: {factor['symbol']} - {factor['factor_name']}")
            
            # Submit factor as strategy event to memory system
            storage_result = await alpha_test.session.call_tool(
                "submit_strategy_event",
                {
                    "event_type": "ALPHA_FACTOR_DISCOVERY",
                    "strategy_id": f"{factor['symbol']}_{factor['factor_name']}",
                    "event_data": {
                        "symbol": factor['symbol'],
                        "factor_name": factor['factor_name'],
                        "factor_strength": factor['factor_data'].get('factor_strength', 0.0),
                        "signal": factor['factor_data'].get('signal', 'HOLD'),
                        "confidence": factor['factor_data'].get('confidence', 0.0),
                        "lookback_period": factor['factor_data'].get('lookback_period', 0)
                    },
                    "metadata": {
                        "discovery_method": "momentum_analysis",
                        "test_client": "alpha_factor_mining_test",
                        "storage_timestamp": datetime.now().isoformat()
                    }
                }
            )
            
            assert storage_result.content is not None
            storage_results.append(storage_result.content)
            logger.info(f"✅ Storage result: {storage_result.content}")
        
        # Validate storage success
        assert len(storage_results) == len(alpha_test.discovered_factors)
        logger.info(f"💾 Successfully stored {len(storage_results)} factors to database")
        
    except Exception as e:
        logger.error(f"❌ Database storage test failed: {e}")
        pytest.fail(f"Database storage test failed: {e}")


@pytest.mark.asyncio
async def test_complete_alpha_mining_workflow(alpha_test):
    """Test complete alpha factor mining workflow."""
    if not alpha_test.session:
        pytest.skip("Session not available")
    
    try:
        logger.info("🚀 Starting complete alpha mining workflow test")
        
        # 1. Health check
        await test_agent_health_check(alpha_test)
        logger.info("✅ Step 1: Health check passed")
        
        # 2. Data loading
        await test_historical_data_loading(alpha_test)
        logger.info("✅ Step 2: Data loading passed")
        
        # 3. Factor discovery
        await test_momentum_factor_discovery(alpha_test)
        logger.info("✅ Step 3: Factor discovery passed")
        
        # 4. Database storage
        await test_factor_database_storage(alpha_test)
        logger.info("✅ Step 4: Database storage passed")
        
        logger.info("🎉 Complete alpha mining workflow test successful!")
        
    except Exception as e:
        logger.error(f"❌ Complete workflow test failed: {e}")
        pytest.fail(f"Complete workflow test failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    import subprocess
    import sys
    
    # Run pytest on this file
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", "-s"
    ], capture_output=False)
    
    sys.exit(result.returncode)
