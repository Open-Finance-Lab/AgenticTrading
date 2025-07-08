"""
Comprehensive Integration Test for Portfolio Construction Agent Pool

This test suite validates the Portfolio Construction Agent Pool's ability to
integrate multi-agent inputs (alpha, risk, transaction costs) and generate
optimal portfolios using various optimization techniques.

Author: Jifeng Li
Created: 2025-06-30
License: openMDW
"""

import asyncio
import pytest
import pytest_asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
import json
import tempfile
import uuid

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockExternalMemory:
    """Mock external memory agent for testing multi-agent integration"""
    
    def __init__(self):
        self.storage = {}
        self.events = []
        self._setup_mock_data()
    
    def _setup_mock_data(self):
        """Setup mock multi-agent data"""
        # Mock alpha signals
        self.alpha_signals = [
            {
                "event_id": "alpha_001",
                "namespace": "alpha_agent_pool",
                "event_type": "SIGNAL_GENERATED",
                "content": {
                    "symbol": "AAPL",
                    "signal_type": "BUY",
                    "confidence_score": 0.85,
                    "predicted_return": 0.12,
                    "strength": 1.2
                },
                "timestamp": datetime.now(timezone.utc)
            },
            {
                "event_id": "alpha_002", 
                "namespace": "alpha_agent_pool",
                "event_type": "SIGNAL_GENERATED",
                "content": {
                    "symbol": "GOOGL",
                    "signal_type": "BUY",
                    "confidence_score": 0.78,
                    "predicted_return": 0.10,
                    "strength": 0.9
                },
                "timestamp": datetime.now(timezone.utc)
            }
        ]
        
        # Mock risk metrics
        self.risk_metrics = [
            {
                "event_id": "risk_001",
                "namespace": "risk_agent_pool",
                "event_type": "ANALYSIS_COMPLETED",
                "content": {
                    "asset_id": "AAPL",
                    "volatility": 0.28,
                    "var_95": 0.045,
                    "beta": 1.15,
                    "correlation": {"SPY": 0.85}
                },
                "timestamp": datetime.now(timezone.utc)
            },
            {
                "event_id": "risk_002",
                "namespace": "risk_agent_pool", 
                "event_type": "ANALYSIS_COMPLETED",
                "content": {
                    "asset_id": "GOOGL",
                    "volatility": 0.32,
                    "var_95": 0.052,
                    "beta": 1.05,
                    "correlation": {"SPY": 0.78}
                },
                "timestamp": datetime.now(timezone.utc)
            }
        ]
        
        # Mock transaction costs
        self.transaction_costs = [
            {
                "event_id": "cost_001",
                "namespace": "transaction_cost_agent_pool",
                "event_type": "COST_ANALYSIS_COMPLETED",
                "content": {
                    "asset_id": "AAPL",
                    "bid_ask_spread": 0.0008,
                    "market_impact": 0.0012,
                    "commission": 0.0005,
                    "total_cost": 0.0025
                },
                "timestamp": datetime.now(timezone.utc)
            },
            {
                "event_id": "cost_002",
                "namespace": "transaction_cost_agent_pool",
                "event_type": "COST_ANALYSIS_COMPLETED", 
                "content": {
                    "asset_id": "GOOGL",
                    "bid_ask_spread": 0.0010,
                    "market_impact": 0.0015,
                    "commission": 0.0005,
                    "total_cost": 0.0030
                },
                "timestamp": datetime.now(timezone.utc)
            }
        ]
    
    async def store_data(self, key: str, data: Dict[str, Any], **kwargs) -> bool:
        self.storage[key] = data
        return True
    
    async def retrieve_data(self, key: str) -> Dict[str, Any]:
        return self.storage.get(key, {})
    
    async def query_events(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock event querying based on namespace"""
        namespace = filters.get("namespace")
        
        if namespace == "alpha_agent_pool":
            return self.alpha_signals
        elif namespace == "risk_agent_pool":
            return self.risk_metrics
        elif namespace == "transaction_cost_agent_pool":
            return self.transaction_costs
        else:
            return []
    
    async def log_event(self, **kwargs):
        event_id = f"event_{len(self.events)}"
        event_data = {
            "event_id": event_id,
            "timestamp": datetime.now(timezone.utc),
            **kwargs
        }
        self.events.append(event_data)
        return event_id
    
    async def store_event(self, event) -> bool:
        """Store a MemoryEvent object"""
        event_data = {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "event_type": event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
            "log_level": event.log_level.value if hasattr(event.log_level, 'value') else str(event.log_level),
            "source_agent_pool": event.source_agent_pool,
            "source_agent_id": event.source_agent_id,
            "title": event.title,
            "content": event.content,
            "tags": list(event.tags) if event.tags else [],
            "metadata": event.metadata or {},
            "session_id": event.session_id,
            "correlation_id": event.correlation_id
        }
        self.events.append(event_data)
        return True
    
    async def close(self):
        pass


class MockOpenAIClient:
    """Mock OpenAI client for natural language processing"""
    
    async def chat_completions_create(self, **kwargs):
        # Mock response for portfolio construction
        mock_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "task_type": "portfolio_optimization",
                        "optimization_type": "mean_variance",
                        "investment_universe": ["AAPL", "GOOGL", "MSFT"],
                        "objective": "maximize_sharpe",
                        "constraints": {
                            "max_weight": 0.40,
                            "min_weight": 0.05,
                            "risk_budget": 0.20
                        },
                        "benchmark": "SPY",
                        "time_horizon": "daily",
                        "rebalancing_frequency": "monthly"
                    })
                }
            }]
        }
        
        class MockResponse:
            def __init__(self, data):
                self.choices = [type('Choice', (), {
                    'message': type('Message', (), {'content': data["choices"][0]["message"]["content"]})()
                })()]
        
        return MockResponse(mock_response)


@pytest.fixture
def sample_investment_universe():
    """Sample investment universe for testing"""
    return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]


@pytest.fixture
def sample_portfolio_constraints():
    """Sample portfolio constraints for testing"""
    return {
        "max_weight": 0.25,
        "min_weight": 0.02,
        "sector_limits": {
            "technology": 0.60,
            "consumer_discretionary": 0.30
        },
        "max_concentration": 0.40,
        "risk_budget": 0.18
    }


class TestPortfolioConstructionIntegration:
    """Integration tests for Portfolio Construction Agent Pool"""
    
    async def _create_test_pool(self):
        """Create a portfolio construction agent pool for testing"""
        from FinAgents.agent_pools.portfolio_construction_agent_pool.core import PortfolioConstructionAgentPool
        
        # Create mock external memory
        mock_memory = MockExternalMemory()
        
        # Create portfolio construction pool with mocked dependencies
        pool = PortfolioConstructionAgentPool(
            openai_api_key="test_key",
            external_memory_config={
                "host": "localhost",
                "port": 8000
            }
        )
        
        # Initialize first
        await pool.initialize()
        
        # Replace with mocks after initialization
        pool.memory_bridge.external_memory_agent = mock_memory
        pool.openai_client = MockOpenAIClient()
        
        return pool
    
    @pytest.mark.asyncio
    async def test_pool_initialization(self):
        """Test portfolio construction agent pool initialization"""
        pool = await self._create_test_pool()
        
        # Check that agents are registered
        agent_names = pool.agent_registry.list_agents()
        
        expected_agents = [
            "mean_variance_optimizer",
            "black_litterman_optimizer",
            "risk_parity_optimizer",
            "factor_optimizer",
            "robust_optimizer",
            "rebalancing_agent",
            "constraint_manager",
            "performance_analyzer"
        ]
        
        for expected_agent in expected_agents:
            assert expected_agent in agent_names, f"Agent {expected_agent} not registered"
        
        # Check memory bridge initialization
        assert pool.memory_bridge is not None
        assert pool.memory_unit is not None
        
        logger.info(f"Successfully initialized portfolio pool with {len(agent_names)} agents")
    
    @pytest.mark.asyncio
    async def test_natural_language_processing(self):
        """Test natural language portfolio construction requests"""
        pool = await self._create_test_pool()
        
        test_requests = [
            "Create a conservative portfolio with low risk for retirement",
            "Build an aggressive growth portfolio maximizing returns",
            "Construct a market-neutral portfolio with equal risk allocation",
            "Design a factor-based momentum portfolio for tech stocks"
        ]
        
        for request in test_requests:
            try:
                result = await pool.process_natural_language_input(request)
                
                assert result is not None
                assert "status" in result
                
                if result["status"] == "success":
                    structured_request = result["structured_request"]
                    assert "task_type" in structured_request
                    assert "optimization_type" in structured_request
                    
                logger.info(f"Successfully processed: {request[:50]}...")
                
            except Exception as e:
                logger.error(f"Failed to process request: {request}, Error: {e}")
                raise
    
    @pytest.mark.asyncio
    async def test_multi_agent_signal_retrieval(self, sample_investment_universe):
        """Test retrieval and integration of multi-agent signals"""
        pool = await self._create_test_pool()
        
        # Test multi-agent signal retrieval
        signals = await pool.retrieve_multi_agent_signals(
            investment_universe=sample_investment_universe,
            time_horizon="daily"
        )
        
        assert "investment_universe" in signals
        assert "alpha_signals" in signals
        assert "risk_metrics" in signals
        assert "transaction_costs" in signals
        assert "data_quality" in signals
        
        # Check data quality metrics
        data_quality = signals["data_quality"]
        assert "alpha_signals_available" in data_quality
        assert "risk_analyses_available" in data_quality
        assert "cost_analyses_available" in data_quality
        assert "coverage_ratio" in data_quality
        
        # Verify signal processing
        alpha_signals = signals["alpha_signals"]
        assert isinstance(alpha_signals, dict)
        
        risk_metrics = signals["risk_metrics"]
        assert isinstance(risk_metrics, dict)
        
        transaction_costs = signals["transaction_costs"]
        assert isinstance(transaction_costs, dict)
        
        logger.info(f"Multi-agent signals retrieved: Alpha={len(alpha_signals)}, "
                   f"Risk={len(risk_metrics)}, Costs={len(transaction_costs)}")
    
    @pytest.mark.asyncio
    async def test_mean_variance_optimization(self, sample_investment_universe):
        """Test mean-variance portfolio optimization"""
        pool = await self._create_test_pool()
        
        # Execute mean-variance optimization
        task = {
            "task_type": "portfolio_optimization",
            "agent_type": "mean_variance_optimizer",
            "parameters": {
                "investment_universe": sample_investment_universe[:3],  # Use subset for testing
                "optimization_type": "mean_variance",
                "objective": "maximize_sharpe",
                "constraints": {
                    "max_weight": 0.50,
                    "min_weight": 0.10,
                    "risk_budget": 0.20
                }
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        assert result["status"] == "success"
        assert "optimization_type" in result
        assert result["optimization_type"] == "mean_variance"
        assert "result" in result
        
        optimization_result = result["result"]
        assert hasattr(optimization_result, 'optimal_weights')
        assert hasattr(optimization_result, 'expected_metrics')
        assert hasattr(optimization_result, 'optimization_status')
        
        # Verify weights sum to approximately 1
        total_weight = sum(optimization_result.optimal_weights.values())
        assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, not 1.0"
        
        logger.info("Mean-variance optimization completed successfully")
    
    @pytest.mark.asyncio
    async def test_black_litterman_optimization(self, sample_investment_universe):
        """Test Black-Litterman portfolio optimization"""
        pool = await self._create_test_pool()
        
        task = {
            "task_type": "portfolio_optimization",
            "agent_type": "black_litterman_optimizer",
            "parameters": {
                "investment_universe": sample_investment_universe[:3],
                "optimization_type": "black_litterman",
                "investor_views": {
                    "AAPL": {"expected_return": 0.15, "confidence": 0.8},
                    "GOOGL": {"expected_return": 0.12, "confidence": 0.6}
                }
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        assert result["status"] == "success"
        assert result["optimization_type"] == "black_litterman"
        
        logger.info("Black-Litterman optimization completed successfully")
    
    @pytest.mark.asyncio
    async def test_risk_parity_optimization(self, sample_investment_universe):
        """Test risk parity portfolio optimization"""
        pool = await self._create_test_pool()
        
        task = {
            "task_type": "portfolio_optimization",
            "agent_type": "risk_parity_optimizer",
            "parameters": {
                "investment_universe": sample_investment_universe[:4],
                "optimization_type": "risk_parity",
                "risk_budget": "equal"
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        assert result["status"] == "success"
        assert result["optimization_type"] == "risk_parity"
        
        logger.info("Risk parity optimization completed successfully")
    
    @pytest.mark.asyncio
    async def test_factor_based_optimization(self, sample_investment_universe):
        """Test factor-based portfolio optimization"""
        pool = await self._create_test_pool()
        
        task = {
            "task_type": "portfolio_optimization",
            "agent_type": "factor_optimizer",
            "parameters": {
                "investment_universe": sample_investment_universe,
                "optimization_type": "factor_based",
                "factor_exposures": {
                    "momentum": 1.0,
                    "quality": 0.5,
                    "low_volatility": -0.3
                }
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        assert result["status"] == "success"
        assert result["optimization_type"] == "factor_based"
        
        logger.info("Factor-based optimization completed successfully")
    
    @pytest.mark.asyncio
    async def test_portfolio_analysis(self):
        """Test portfolio performance analysis"""
        pool = await self._create_test_pool()
        
        # First create a portfolio (mock)
        portfolio_id = str(uuid.uuid4())
        
        # Store mock portfolio in memory
        from FinAgents.agent_pools.portfolio_construction_agent_pool.memory_bridge import (
            PortfolioRecord, PortfolioPosition, OptimizationType, PortfolioStatus
        )
        
        mock_positions = [
            PortfolioPosition(
                asset_id="AAPL",
                asset_name="Apple Inc.",
                asset_type="equity",
                target_weight=0.4,
                current_weight=0.38,
                target_quantity=100,
                current_quantity=95,
                target_value=40000,
                current_value=38000,
                expected_return=0.12,
                volatility=0.28
            ),
            PortfolioPosition(
                asset_id="GOOGL",
                asset_name="Alphabet Inc.",
                asset_type="equity",
                target_weight=0.6,
                current_weight=0.62,
                target_quantity=50,
                current_quantity=52,
                target_value=60000,
                current_value=62000,
                expected_return=0.10,
                volatility=0.32
            )
        ]
        
        mock_portfolio = PortfolioRecord(
            portfolio_id=portfolio_id,
            portfolio_name="Test Portfolio",
            portfolio_type=OptimizationType.MEAN_VARIANCE,
            status=PortfolioStatus.ACTIVE,
            positions=mock_positions,
            total_value=100000,
            benchmark="SPY",
            objective="Maximize risk-adjusted returns",
            constraints={},
            risk_budget=0.20,
            expected_return=0.11,
            expected_volatility=0.18,
            sharpe_ratio=0.5
        )
        
        await pool.memory_bridge.store_portfolio_record(mock_portfolio)
        
        # Test portfolio analysis
        task = {
            "task_type": "portfolio_analysis",
            "agent_type": "performance_analyzer",
            "parameters": {
                "portfolio_id": portfolio_id,
                "analysis_type": "comprehensive"
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        assert result["status"] == "success"
        assert "portfolio_id" in result
        assert "analytics" in result
        
        logger.info("Portfolio analysis completed successfully")
    
    @pytest.mark.asyncio
    async def test_portfolio_rebalancing(self):
        """Test portfolio rebalancing functionality"""
        pool = await self._create_test_pool()
        
        portfolio_id = str(uuid.uuid4())
        
        task = {
            "task_type": "portfolio_rebalancing",
            "agent_type": "rebalancing_agent",
            "parameters": {
                "portfolio_id": portfolio_id,
                "rebalancing_threshold": 0.05,
                "transaction_cost_model": "linear"
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        assert result["status"] == "success"
        assert "result" in result
        
        rebalancing_result = result["result"]
        assert "portfolio_id" in rebalancing_result
        assert "rebalancing_threshold" in rebalancing_result
        assert "trades_required" in rebalancing_result
        
        logger.info("Portfolio rebalancing test completed successfully")
    
    @pytest.mark.asyncio
    async def test_performance_evaluation(self):
        """Test portfolio performance evaluation"""
        pool = await self._create_test_pool()
        
        portfolio_id = str(uuid.uuid4())
        
        task = {
            "task_type": "performance_evaluation",
            "agent_type": "performance_analyzer",
            "parameters": {
                "portfolio_id": portfolio_id,
                "evaluation_days": 30
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        assert result["status"] == "success"
        assert "portfolio_id" in result
        assert "performance_analytics" in result
        
        logger.info("Performance evaluation completed successfully")
    
    @pytest.mark.asyncio
    async def test_memory_integration(self):
        """Test external memory integration"""
        pool = await self._create_test_pool()
        memory_bridge = pool.memory_bridge
        
        # Test storing optimization result
        from FinAgents.agent_pools.portfolio_construction_agent_pool.memory_bridge import (
            create_optimization_result, OptimizationType
        )
        
        optimization_result = create_optimization_result(
            optimization_type=OptimizationType.MEAN_VARIANCE,
            portfolio_id=str(uuid.uuid4()),
            optimal_weights={"AAPL": 0.6, "GOOGL": 0.4},
            input_signals={"test": "data"}
        )
        
        storage_key = await memory_bridge.store_optimization_result(optimization_result)
        assert storage_key is not None
        assert "optimization:" in storage_key
        
        # Test storing portfolio metrics
        from FinAgents.agent_pools.portfolio_construction_agent_pool.memory_bridge import (
            create_portfolio_metrics_record
        )
        
        portfolio_metrics = create_portfolio_metrics_record(
            portfolio_id=optimization_result.portfolio_id,
            total_return=0.08,
            benchmark_return=0.06
        )
        
        metrics_key = await memory_bridge.store_portfolio_metrics(portfolio_metrics)
        assert metrics_key is not None
        assert "metrics:" in metrics_key
        
        logger.info("Memory integration test completed successfully")
    
    @pytest.mark.asyncio
    async def test_mcp_server_integration(self):
        """Test MCP server integration"""
        pool = await self._create_test_pool()
        mcp_server = pool.mcp_server
        
        # Test portfolio optimization request
        optimization_request = {
            "type": "portfolio_optimization",
            "request_id": str(uuid.uuid4()),
            "parameters": {
                "investment_universe": ["AAPL", "GOOGL"],
                "optimization_type": "mean_variance",
                "constraints": {"max_weight": 0.6},
                "objective": "maximize_sharpe"
            }
        }
        
        result = await mcp_server.handle_portfolio_construction_request(optimization_request)
        
        assert result["status"] == "success"
        assert "optimization_result" in result
        assert "server_id" in result
        
        # Test portfolio analysis request
        analysis_request = {
            "type": "portfolio_analysis",
            "request_id": str(uuid.uuid4()),
            "parameters": {
                "portfolio_id": str(uuid.uuid4()),
                "analysis_type": "comprehensive"
            }
        }
        
        result = await mcp_server.handle_portfolio_construction_request(analysis_request)
        
        assert result["status"] == "success"
        assert "analysis_result" in result
        
        logger.info("MCP server integration test completed successfully")
    
    @pytest.mark.asyncio
    async def test_concurrent_optimizations(self, sample_investment_universe):
        """Test concurrent portfolio optimizations"""
        pool = await self._create_test_pool()
        
        # Create multiple concurrent optimization tasks
        tasks = []
        optimization_types = ["mean_variance", "black_litterman", "risk_parity"]
        
        for opt_type in optimization_types:
            task = {
                "task_type": "portfolio_optimization",
                "agent_type": f"{opt_type}_optimizer",
                "parameters": {
                    "investment_universe": sample_investment_universe[:3],
                    "optimization_type": opt_type
                }
            }
            tasks.append(pool.execute_structured_task(task))
        
        # Execute all optimizations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all optimizations completed successfully
        successful_results = 0
        for result in results:
            if isinstance(result, dict) and result.get("status") == "success":
                successful_results += 1
            elif isinstance(result, Exception):
                logger.error(f"Optimization failed with exception: {result}")
        
        assert successful_results >= 2, f"Only {successful_results}/3 concurrent optimizations succeeded"
        logger.info(f"Concurrent optimization test completed: {successful_results}/3 successful")
    
    @pytest.mark.asyncio
    async def test_constraint_validation(self, sample_portfolio_constraints):
        """Test portfolio constraint validation"""
        pool = await self._create_test_pool()
        
        # Test optimization with strict constraints
        task = {
            "task_type": "portfolio_optimization",
            "agent_type": "mean_variance_optimizer",
            "parameters": {
                "investment_universe": ["AAPL", "GOOGL", "MSFT"],
                "optimization_type": "mean_variance",
                "constraints": sample_portfolio_constraints
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        assert result["status"] == "success"
        
        # Verify constraint satisfaction
        optimization_result = result["result"]
        optimal_weights = optimization_result.optimal_weights
        
        # Check individual weight constraints
        max_weight = sample_portfolio_constraints["max_weight"]
        min_weight = sample_portfolio_constraints["min_weight"]
        
        for asset, weight in optimal_weights.items():
            assert weight <= max_weight, f"{asset} weight {weight} exceeds max {max_weight}"
            assert weight >= min_weight, f"{asset} weight {weight} below min {min_weight}"
        
        logger.info("Constraint validation test completed successfully")
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and recovery"""
        pool = await self._create_test_pool()
        
        # Test with invalid optimization type
        invalid_task = {
            "task_type": "portfolio_optimization",
            "agent_type": "nonexistent_optimizer",
            "parameters": {
                "investment_universe": ["AAPL"],
                "optimization_type": "invalid_type"
            }
        }
        
        result = await pool.execute_structured_task(invalid_task)
        
        assert result["status"] == "error"
        assert "error" in result
        
        # Test with empty investment universe
        empty_universe_task = {
            "task_type": "portfolio_optimization",
            "agent_type": "mean_variance_optimizer",
            "parameters": {
                "investment_universe": [],
                "optimization_type": "mean_variance"
            }
        }
        
        result = await pool.execute_structured_task(empty_universe_task)
        
        assert result["status"] == "error"
        assert "Investment universe required" in result["error"]
        
        logger.info("Error handling test completed successfully")


class TestPortfolioPerformanceMetrics:
    """Performance and benchmarking tests for portfolio construction"""
    
    async def _create_test_pool(self):
        """Create a portfolio construction agent pool for testing"""
        from FinAgents.agent_pools.portfolio_construction_agent_pool.core import PortfolioConstructionAgentPool
        
        # Create mock external memory
        mock_memory = MockExternalMemory()
        
        # Create portfolio construction pool with mocked dependencies
        pool = PortfolioConstructionAgentPool(
            openai_api_key="test_key",
            external_memory_config={
                "host": "localhost",
                "port": 8000
            }
        )
        
        # Initialize first
        await pool.initialize()
        
        # Replace with mocks after initialization
        pool.memory_bridge.external_memory_agent = mock_memory
        pool.openai_client = MockOpenAIClient()
        
        return pool
    
    @pytest.mark.asyncio
    async def test_optimization_performance(self, sample_investment_universe):
        """Test portfolio optimization performance metrics"""
        pool = await self._create_test_pool()
        
        start_time = datetime.now()
        
        task = {
            "task_type": "portfolio_optimization",
            "agent_type": "mean_variance_optimizer",
            "parameters": {
                "investment_universe": sample_investment_universe,
                "optimization_type": "mean_variance",
                "objective": "maximize_sharpe"
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        assert result["status"] == "success"
        assert execution_time < 10.0, f"Optimization took too long: {execution_time}s"
        
        logger.info(f"Portfolio optimization completed in {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_multi_agent_integration_performance(self, sample_investment_universe):
        """Test multi-agent signal integration performance"""
        pool = await self._create_test_pool()
        
        start_time = datetime.now()
        
        signals = await pool.retrieve_multi_agent_signals(
            investment_universe=sample_investment_universe,
            time_horizon="daily"
        )
        
        end_time = datetime.now()
        integration_time = (end_time - start_time).total_seconds()
        
        assert "alpha_signals" in signals
        assert "risk_metrics" in signals
        assert "transaction_costs" in signals
        assert integration_time < 5.0, f"Multi-agent integration took too long: {integration_time}s"
        
        logger.info(f"Multi-agent integration completed in {integration_time:.3f}s")


if __name__ == "__main__":
    """Run integration tests manually"""
    import sys
    
    async def run_tests():
        # Create test fixtures
        mock_memory = MockExternalMemory()
        
        from FinAgents.agent_pools.portfolio_construction_agent_pool.core import PortfolioConstructionAgentPool
        
        pool = PortfolioConstructionAgentPool(
            openai_api_key="test_key",
            external_memory_config={"host": "localhost", "port": 8000}
        )
        
        pool.memory_bridge.external_memory_agent = mock_memory
        pool.openai_client = MockOpenAIClient()
        
        await pool.initialize()
        
        sample_universe = ["AAPL", "GOOGL", "MSFT"]
        
        # Run basic tests
        print("Running Portfolio Construction Agent Pool Integration Tests...")
        
        try:
            # Test 1: Initialization
            agent_names = pool.agent_registry.list_agents()
            print(f"✓ Initialized with {len(agent_names)} agents")
            
            # Test 2: Natural language processing
            result = await pool.process_natural_language_input(
                "Create a balanced portfolio with moderate risk"
            )
            print(f"✓ Natural language processing: {result.get('status')}")
            
            # Test 3: Multi-agent signal retrieval
            signals = await pool.retrieve_multi_agent_signals(sample_universe)
            print(f"✓ Multi-agent signal retrieval: {len(signals)} signal types")
            
            # Test 4: Portfolio optimization
            task = {
                "task_type": "portfolio_optimization",
                "agent_type": "mean_variance_optimizer",
                "parameters": {
                    "investment_universe": sample_universe,
                    "optimization_type": "mean_variance"
                }
            }
            
            result = await pool.execute_structured_task(task)
            print(f"✓ Portfolio optimization: {result.get('status')}")
            
            await pool.close()
            print("\n✅ All integration tests passed!")
            
        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
            sys.exit(1)
    
    # Run the tests
    asyncio.run(run_tests())
