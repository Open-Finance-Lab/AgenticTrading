"""
Comprehensive test suite for Risk Agent Pool memory bridge functionality.

Author: Jifeng Li
License: openMDW
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import json

from FinAgents.agent_pools.risk_agent_pool.memory_bridge import RiskMemoryBridge
from .fixtures import sample_market_data, sample_portfolio_data


class TestRiskMemoryBridge:
    """Test the RiskMemoryBridge class functionality."""
    
    @pytest.fixture
    def memory_bridge(self):
        """Create a RiskMemoryBridge instance for testing."""
        mock_memory_client = AsyncMock()
        return RiskMemoryBridge(memory_client=mock_memory_client)
    
    def test_bridge_initialization(self, memory_bridge):
        """Test that RiskMemoryBridge initializes correctly."""
        assert memory_bridge.memory_client is not None
        assert hasattr(memory_bridge, 'event_log')
        assert hasattr(memory_bridge, 'data_cache')
    
    @pytest.mark.asyncio
    async def test_store_analysis_result(self, memory_bridge):
        """Test storing analysis results in memory."""
        analysis_result = {
            "risk_assessment": "Medium risk",
            "key_factors": ["Market volatility", "Credit exposure"],
            "recommendations": ["Diversify portfolio"],
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat()
        }
        
        context = {
            "portfolio_id": "PORTFOLIO_001",
            "analysis_type": "comprehensive"
        }
        
        # Mock memory client store method
        memory_bridge.memory_client.store = AsyncMock(return_value={"id": "analysis_123"})
        
        # Store the analysis
        result_id = await memory_bridge.store_analysis_result(analysis_result, context)
        
        # Verify storage was called correctly
        memory_bridge.memory_client.store.assert_called_once()
        assert result_id == "analysis_123"
    
    @pytest.mark.asyncio
    async def test_get_historical_analysis(self, memory_bridge):
        """Test retrieving historical analysis data."""
        # Mock historical data
        mock_data = [
            {
                "id": "analysis_001",
                "risk_assessment": "Low risk",
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat()
            },
            {
                "id": "analysis_002", 
                "risk_assessment": "Medium risk",
                "timestamp": (datetime.now() - timedelta(days=2)).isoformat()
            }
        ]
        
        memory_bridge.memory_client.query = AsyncMock(return_value=mock_data)
        
        # Query historical data
        historical_data = await memory_bridge.get_historical_analysis(
            portfolio_id="PORTFOLIO_001",
            days_back=7
        )
        
        # Verify query was called and data returned
        memory_bridge.memory_client.query.assert_called_once()
        assert len(historical_data) == 2
        assert historical_data[0]["risk_assessment"] == "Low risk"
    
    @pytest.mark.asyncio
    async def test_store_market_data(self, memory_bridge, sample_market_data):
        """Test storing market data in memory."""
        memory_bridge.memory_client.store = AsyncMock(return_value={"id": "market_data_123"})
        
        # Store market data
        data_id = await memory_bridge.store_market_data(sample_market_data)
        
        # Verify storage
        memory_bridge.memory_client.store.assert_called_once()
        assert data_id == "market_data_123"
    
    @pytest.mark.asyncio
    async def test_get_market_data(self, memory_bridge):
        """Test retrieving market data from memory."""
        mock_market_data = {
            "symbol": "AAPL",
            "price": 150.00,
            "volatility": 0.25,
            "timestamp": datetime.now().isoformat()
        }
        
        memory_bridge.memory_client.query = AsyncMock(return_value=[mock_market_data])
        
        # Retrieve market data
        market_data = await memory_bridge.get_market_data(
            symbols=["AAPL"],
            start_date=datetime.now() - timedelta(days=30)
        )
        
        # Verify retrieval
        memory_bridge.memory_client.query.assert_called_once()
        assert len(market_data) == 1
        assert market_data[0]["symbol"] == "AAPL"
    
    @pytest.mark.asyncio
    async def test_log_event(self, memory_bridge):
        """Test event logging functionality."""
        event_data = {
            "event_type": "risk_analysis_started",
            "portfolio_id": "PORTFOLIO_001",
            "agent_name": "market_risk",
            "details": {"risk_types": ["market", "credit"]}
        }
        
        memory_bridge.memory_client.store = AsyncMock(return_value={"id": "event_123"})
        
        # Log the event
        event_id = await memory_bridge.log_event(event_data)
        
        # Verify logging
        memory_bridge.memory_client.store.assert_called_once()
        assert event_id == "event_123"
    
    @pytest.mark.asyncio
    async def test_get_event_history(self, memory_bridge):
        """Test retrieving event history."""
        mock_events = [
            {
                "id": "event_001",
                "event_type": "risk_analysis_completed",
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": "event_002",
                "event_type": "portfolio_update",
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat()
            }
        ]
        
        memory_bridge.memory_client.query = AsyncMock(return_value=mock_events)
        
        # Get event history
        events = await memory_bridge.get_event_history(
            event_types=["risk_analysis_completed", "portfolio_update"],
            hours_back=24
        )
        
        # Verify retrieval
        memory_bridge.memory_client.query.assert_called_once()
        assert len(events) == 2
        assert events[0]["event_type"] == "risk_analysis_completed"
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, memory_bridge):
        """Test data caching functionality."""
        cache_key = "portfolio_risk_PORTFOLIO_001"
        cache_data = {
            "portfolio_id": "PORTFOLIO_001",
            "total_risk": 0.15,
            "last_updated": datetime.now().isoformat()
        }
        
        # Test cache set
        await memory_bridge.cache_set(cache_key, cache_data, ttl=3600)
        
        # Test cache get
        cached_data = await memory_bridge.cache_get(cache_key)
        assert cached_data == cache_data
        
        # Test cache delete
        await memory_bridge.cache_delete(cache_key)
        cached_data = await memory_bridge.cache_get(cache_key)
        assert cached_data is None
    
    @pytest.mark.asyncio
    async def test_store_portfolio_snapshot(self, memory_bridge, sample_portfolio_data):
        """Test storing portfolio snapshots."""
        memory_bridge.memory_client.store = AsyncMock(return_value={"id": "snapshot_123"})
        
        # Store portfolio snapshot
        snapshot_id = await memory_bridge.store_portfolio_snapshot(sample_portfolio_data)
        
        # Verify storage
        memory_bridge.memory_client.store.assert_called_once()
        assert snapshot_id == "snapshot_123"
    
    @pytest.mark.asyncio
    async def test_get_portfolio_history(self, memory_bridge):
        """Test retrieving portfolio history."""
        mock_snapshots = [
            {
                "id": "snapshot_001",
                "portfolio_id": "PORTFOLIO_001",
                "total_value": 1000000,
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": "snapshot_002",
                "portfolio_id": "PORTFOLIO_001", 
                "total_value": 950000,
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat()
            }
        ]
        
        memory_bridge.memory_client.query = AsyncMock(return_value=mock_snapshots)
        
        # Get portfolio history
        history = await memory_bridge.get_portfolio_history(
            portfolio_id="PORTFOLIO_001",
            days_back=30
        )
        
        # Verify retrieval
        memory_bridge.memory_client.query.assert_called_once()
        assert len(history) == 2
        assert history[0]["total_value"] == 1000000
    
    @pytest.mark.asyncio
    async def test_error_handling(self, memory_bridge):
        """Test error handling in memory operations."""
        # Mock memory client to raise an exception
        memory_bridge.memory_client.store = AsyncMock(
            side_effect=Exception("Memory storage error")
        )
        
        # Storage should handle error gracefully
        with pytest.raises(Exception, match="Memory storage error"):
            await memory_bridge.store_analysis_result(
                {"risk_assessment": "test"}, 
                {"portfolio_id": "TEST"}
            )
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, memory_bridge):
        """Test batch storage and retrieval operations."""
        # Mock batch data
        batch_data = [
            {"id": "item_001", "data": "test_data_1"},
            {"id": "item_002", "data": "test_data_2"},
            {"id": "item_003", "data": "test_data_3"}
        ]
        
        memory_bridge.memory_client.batch_store = AsyncMock(
            return_value={"stored_count": 3}
        )
        
        # Batch store
        result = await memory_bridge.batch_store(batch_data)
        
        # Verify batch storage
        memory_bridge.memory_client.batch_store.assert_called_once()
        assert result["stored_count"] == 3
    
    @pytest.mark.asyncio
    async def test_search_functionality(self, memory_bridge):
        """Test search functionality in memory."""
        search_query = "risk analysis portfolio PORTFOLIO_001"
        mock_results = [
            {
                "id": "result_001",
                "content": "Risk analysis for PORTFOLIO_001",
                "relevance": 0.95
            },
            {
                "id": "result_002", 
                "content": "Historical analysis PORTFOLIO_001",
                "relevance": 0.87
            }
        ]
        
        memory_bridge.memory_client.search = AsyncMock(return_value=mock_results)
        
        # Perform search
        results = await memory_bridge.search(search_query, limit=10)
        
        # Verify search
        memory_bridge.memory_client.search.assert_called_once()
        assert len(results) == 2
        assert results[0]["relevance"] == 0.95
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self, memory_bridge):
        """Test memory cleanup operations."""
        cutoff_date = datetime.now() - timedelta(days=90)
        
        memory_bridge.memory_client.delete = AsyncMock(
            return_value={"deleted_count": 25}
        )
        
        # Perform cleanup
        result = await memory_bridge.cleanup_old_data(cutoff_date)
        
        # Verify cleanup
        memory_bridge.memory_client.delete.assert_called_once()
        assert result["deleted_count"] == 25
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, memory_bridge):
        """Test concurrent memory operations."""
        # Mock concurrent store operations
        memory_bridge.memory_client.store = AsyncMock(
            side_effect=lambda x: {"id": f"concurrent_{hash(str(x)) % 1000}"}
        )
        
        # Create multiple concurrent store tasks
        tasks = [
            memory_bridge.store_analysis_result(
                {"analysis_id": i, "risk_level": "medium"},
                {"portfolio_id": f"PORTFOLIO_{i:03d}"}
            )
            for i in range(5)
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify all completed successfully
        assert len(results) == 5
        for result in results:
            assert result.startswith("concurrent_")
    
    def test_configuration_validation(self):
        """Test memory bridge configuration validation."""
        # Test invalid memory client
        with pytest.raises(ValueError, match="Memory client is required"):
            RiskMemoryBridge(memory_client=None)
    
    @pytest.mark.asyncio
    async def test_health_check(self, memory_bridge):
        """Test memory bridge health check."""
        # Mock health check response
        memory_bridge.memory_client.health_check = AsyncMock(
            return_value={"status": "healthy", "latency_ms": 15}
        )
        
        # Perform health check
        health = await memory_bridge.health_check()
        
        # Verify health check
        assert "status" in health
        assert "memory_status" in health
        assert "cache_status" in health
        assert health["status"] == "healthy"
