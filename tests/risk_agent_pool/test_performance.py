"""
Performance and load testing for Risk Agent Pool.

Author: Jifeng Li
License: openMDW
"""

import pytest
import asyncio
import time
import psutil
import gc
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
import json
import statistics

from FinAgents.agent_pools.risk_agent_pool.core import RiskAgentPool
from FinAgents.agent_pools.risk_agent_pool.registry import preload_default_agents
from .fixtures import (
    sample_market_data, sample_portfolio_data, mock_openai_client,
    mock_memory_bridge, sample_risk_context
)


class TestRiskAgentPoolPerformance:
    """Performance and load testing for Risk Agent Pool."""
    
    @pytest.fixture
    async def performance_pool(self, mock_openai_client, mock_memory_bridge):
        """Create optimized pool for performance testing."""
        pool = RiskAgentPool(
            openai_client=mock_openai_client,
            memory_bridge=mock_memory_bridge
        )
        preload_default_agents(pool)
        return pool
    
    def setup_fast_mock_responses(self, pool):
        """Setup fast mock responses for performance testing."""
        # Mock fast OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "risk_assessment": "Medium",
            "risk_score": 5.5,
            "analysis_time": datetime.now().isoformat()
        })
        
        # Make OpenAI calls return quickly
        async def fast_create(*args, **kwargs):
            await asyncio.sleep(0.01)  # Simulate minimal latency
            return mock_response
        
        pool.openai_client.chat.completions.create = fast_create
        
        # Fast memory operations
        pool.memory_bridge.store_analysis_result = AsyncMock(return_value="test_id")
        pool.memory_bridge.log_event = AsyncMock(return_value="event_id")
    
    @pytest.mark.asyncio
    async def test_single_analysis_performance(self, performance_pool, sample_risk_context):
        """Test performance of single risk analysis."""
        self.setup_fast_mock_responses(performance_pool)
        
        # Measure single analysis performance
        start_time = time.time()
        
        result = await performance_pool.analyze_risk(
            context=sample_risk_context,
            risk_types=["market", "credit"],
            agents=["market_risk", "credit_risk"]
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify performance (should complete quickly with mocks)
        assert execution_time < 1.0, f"Single analysis took too long: {execution_time:.3f}s"
        assert "risk_assessment" in result
        
        print(f"Single analysis execution time: {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_performance(self, performance_pool, sample_risk_context):
        """Test performance under concurrent load."""
        self.setup_fast_mock_responses(performance_pool)
        
        # Test various concurrency levels
        concurrency_levels = [5, 10, 20, 50]
        performance_results = {}
        
        for concurrency in concurrency_levels:
            # Create concurrent tasks
            tasks = []
            for i in range(concurrency):
                context = {
                    **sample_risk_context,
                    "analysis_id": f"perf_test_{i}"
                }
                task = performance_pool.analyze_risk(
                    context=context,
                    risk_types=["market"],
                    agents=["market_risk"]
                )
                tasks.append(task)
            
            # Measure concurrent execution
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            execution_time = end_time - start_time
            errors = [r for r in results if isinstance(r, Exception)]
            
            performance_results[concurrency] = {
                "execution_time": execution_time,
                "requests_per_second": concurrency / execution_time,
                "error_count": len(errors),
                "success_rate": (concurrency - len(errors)) / concurrency
            }
            
            # Verify no errors and reasonable performance
            assert len(errors) == 0, f"Found {len(errors)} errors at concurrency {concurrency}"
            assert execution_time < 10.0, f"Concurrent execution too slow: {execution_time:.3f}s"
            
            print(f"Concurrency {concurrency}: {execution_time:.3f}s, "
                  f"{performance_results[concurrency]['requests_per_second']:.1f} req/s")
        
        # Verify performance scales reasonably
        baseline_rps = performance_results[5]["requests_per_second"]
        high_load_rps = performance_results[50]["requests_per_second"]
        
        # Should maintain at least 50% of baseline performance at high load
        assert high_load_rps >= baseline_rps * 0.5, "Performance degraded too much under load"
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, performance_pool, sample_risk_context):
        """Test memory usage remains stable under load."""
        self.setup_fast_mock_responses(performance_pool)
        
        # Record initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple analysis cycles
        cycles = 10
        analyses_per_cycle = 20
        memory_measurements = []
        
        for cycle in range(cycles):
            # Run batch of analyses
            tasks = []
            for i in range(analyses_per_cycle):
                context = {
                    **sample_risk_context,
                    "cycle": cycle,
                    "analysis": i
                }
                task = performance_pool.analyze_risk(
                    context=context,
                    risk_types=["market", "credit"],
                    agents=["market_risk", "credit_risk"]
                )
                tasks.append(task)
            
            # Execute batch
            await asyncio.gather(*tasks)
            
            # Force garbage collection
            gc.collect()
            
            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append(current_memory)
            
            print(f"Cycle {cycle}: Memory usage: {current_memory:.1f}MB")
        
        # Analyze memory stability
        final_memory = memory_measurements[-1]
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_measurements)
        
        # Memory growth should be reasonable (less than 50MB)
        assert memory_growth < 50, f"Excessive memory growth: {memory_growth:.1f}MB"
        assert max_memory < initial_memory + 100, f"Peak memory too high: {max_memory:.1f}MB"
        
        print(f"Memory stability test: Initial: {initial_memory:.1f}MB, "
              f"Final: {final_memory:.1f}MB, Growth: {memory_growth:.1f}MB")
    
    @pytest.mark.asyncio
    async def test_agent_instantiation_performance(self, mock_openai_client):
        """Test performance of agent instantiation and pool creation."""
        start_time = time.time()
        
        # Create multiple pools (simulating multiple instances)
        pools = []
        for i in range(10):
            pool = RiskAgentPool(
                openai_client=mock_openai_client,
                memory_bridge=AsyncMock()
            )
            preload_default_agents(pool)
            pools.append(pool)
        
        end_time = time.time()
        instantiation_time = end_time - start_time
        
        # Verify reasonable instantiation time
        assert instantiation_time < 5.0, f"Pool instantiation too slow: {instantiation_time:.3f}s"
        
        # Verify all pools have agents
        for pool in pools:
            assert len(pool.agents) > 0
            assert len(pool.agent_registry) > 0
        
        print(f"Pool instantiation time (10 pools): {instantiation_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_large_context_handling(self, performance_pool):
        """Test performance with large context data."""
        self.setup_fast_mock_responses(performance_pool)
        
        # Create large context data
        large_portfolio = {
            "positions": [
                {
                    "symbol": f"STOCK_{i:04d}",
                    "quantity": 100 + i,
                    "price": 50.0 + (i % 100),
                    "sector": f"Sector_{i % 10}",
                    "market_value": (100 + i) * (50.0 + (i % 100))
                }
                for i in range(1000)  # 1000 positions
            ],
            "metadata": {
                "total_positions": 1000,
                "last_updated": datetime.now().isoformat(),
                "data_quality": "high"
            }
        }
        
        large_market_data = {
            f"symbol_{i}": {
                "price": 100.0 + i,
                "volume": 1000000 + (i * 1000),
                "volatility": 0.15 + (i % 50) / 1000,
                "beta": 0.8 + (i % 30) / 100
            }
            for i in range(500)  # 500 market data points
        }
        
        large_context = {
            "portfolio": large_portfolio,
            "market_data": large_market_data,
            "analysis_request": "comprehensive_risk_analysis",
            "timestamp": datetime.now().isoformat()
        }
        
        # Measure performance with large context
        start_time = time.time()
        
        result = await performance_pool.analyze_risk(
            context=large_context,
            risk_types=["market", "liquidity"],
            agents=["market_risk", "liquidity_risk"]
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify reasonable performance even with large context
        assert execution_time < 5.0, f"Large context analysis too slow: {execution_time:.3f}s"
        assert "risk_assessment" in result
        
        print(f"Large context analysis time: {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, performance_pool, sample_risk_context):
        """Test performance impact of error recovery."""
        # Setup mock that fails intermittently
        call_count = 0
        
        async def intermittent_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Simulated API failure")
            
            # Successful response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps({
                "risk_assessment": "Medium",
                "analysis_successful": True
            })
            return mock_response
        
        performance_pool.openai_client.chat.completions.create = intermittent_failure
        
        # Run multiple analyses with intermittent failures
        num_requests = 30
        start_time = time.time()
        
        tasks = []
        for i in range(num_requests):
            context = {**sample_risk_context, "request_id": i}
            task = performance_pool.analyze_risk(
                context=context,
                risk_types=["market"],
                agents=["market_risk"]
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Count successes and failures
        successes = [r for r in results if not isinstance(r, Exception) and "error" not in r]
        failures = [r for r in results if isinstance(r, Exception) or "error" in r]
        
        # Verify error recovery doesn't severely impact performance
        assert execution_time < 15.0, f"Error recovery too slow: {execution_time:.3f}s"
        assert len(successes) > 0, "No successful analyses"
        
        success_rate = len(successes) / num_requests
        print(f"Error recovery test: {execution_time:.3f}s, "
              f"Success rate: {success_rate:.2%}")
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, performance_pool, sample_risk_context):
        """Test performance under sustained load over time."""
        self.setup_fast_mock_responses(performance_pool)
        
        # Run sustained load test
        duration_minutes = 2  # 2 minute test
        requests_per_minute = 60  # 1 per second
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        request_times = []
        error_count = 0
        
        while time.time() < end_time:
            batch_start = time.time()
            
            # Submit batch of requests
            batch_size = 10
            tasks = []
            
            for i in range(batch_size):
                context = {
                    **sample_risk_context,
                    "timestamp": datetime.now().isoformat(),
                    "batch_id": i
                }
                task = performance_pool.analyze_risk(
                    context=context,
                    risk_types=["market"],
                    agents=["market_risk"]
                )
                tasks.append(task)
            
            # Execute batch and measure
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_end = time.time()
            
            batch_time = batch_end - batch_start
            request_times.append(batch_time / batch_size)  # Per-request time
            
            # Count errors
            batch_errors = [r for r in batch_results if isinstance(r, Exception)]
            error_count += len(batch_errors)
            
            # Wait to maintain rate limit
            sleep_time = 60 / requests_per_minute - batch_time
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        total_duration = time.time() - start_time
        total_requests = len(request_times) * batch_size
        
        # Analyze performance metrics
        avg_request_time = statistics.mean(request_times)
        p95_request_time = statistics.quantiles(request_times, n=20)[18]  # 95th percentile
        requests_per_second = total_requests / total_duration
        error_rate = error_count / total_requests
        
        # Verify sustained performance
        assert avg_request_time < 1.0, f"Average request time too high: {avg_request_time:.3f}s"
        assert p95_request_time < 2.0, f"P95 request time too high: {p95_request_time:.3f}s"
        assert error_rate < 0.01, f"Error rate too high: {error_rate:.2%}"
        
        print(f"Sustained load test ({duration_minutes}m): "
              f"Avg: {avg_request_time:.3f}s, P95: {p95_request_time:.3f}s, "
              f"RPS: {requests_per_second:.1f}, Errors: {error_rate:.2%}")
    
    @pytest.mark.asyncio
    async def test_agent_switching_performance(self, performance_pool, sample_risk_context):
        """Test performance of dynamic agent switching."""
        self.setup_fast_mock_responses(performance_pool)
        
        # List of available agents
        available_agents = [
            "market_risk", "credit_risk", "liquidity_risk", 
            "var_calculator", "volatility", "operational_risk"
        ]
        
        # Test rapid agent switching
        num_switches = 100
        start_time = time.time()
        
        for i in range(num_switches):
            # Randomly select agents for this analysis
            selected_agents = available_agents[i % len(available_agents)]
            
            context = {
                **sample_risk_context,
                "switch_test": i,
                "selected_agent": selected_agents
            }
            
            await performance_pool.analyze_risk(
                context=context,
                risk_types=["market"],
                agents=[selected_agents]
            )
        
        end_time = time.time()
        switching_time = end_time - start_time
        
        # Verify agent switching performance
        avg_switch_time = switching_time / num_switches
        assert avg_switch_time < 0.1, f"Agent switching too slow: {avg_switch_time:.3f}s"
        
        print(f"Agent switching test: {switching_time:.3f}s for {num_switches} switches, "
              f"Avg: {avg_switch_time:.3f}s per switch")
    
    def test_memory_efficiency(self, mock_openai_client, mock_memory_bridge):
        """Test memory efficiency of pool operations."""
        import tracemalloc
        
        # Start memory tracing
        tracemalloc.start()
        
        # Create and use pool
        pool = RiskAgentPool(
            openai_client=mock_openai_client,
            memory_bridge=mock_memory_bridge
        )
        preload_default_agents(pool)
        
        # Take snapshot after initialization
        snapshot1 = tracemalloc.take_snapshot()
        
        # Simulate some operations
        for i in range(100):
            # Add and remove agents to test cleanup
            from FinAgents.agent_pools.risk_agent_pool.registry import BaseRiskAgent
            
            class TempAgent(BaseRiskAgent):
                def __init__(self):
                    super().__init__(f"temp_{i}", "temp", "Temporary agent", mock_openai_client)
                
                async def analyze(self, context):
                    return {"temp_result": i}
            
            temp_agent = TempAgent()
            pool.register_agent(temp_agent)
            
            # Remove agent to test cleanup
            if f"temp_{i}" in pool.agents:
                del pool.agents[f"temp_{i}"]
        
        # Force garbage collection
        gc.collect()
        
        # Take final snapshot
        snapshot2 = tracemalloc.take_snapshot()
        
        # Analyze memory usage
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_size_diff = sum(stat.size_diff for stat in top_stats)
        
        # Memory growth should be minimal
        memory_growth_mb = total_size_diff / 1024 / 1024
        assert memory_growth_mb < 10, f"Excessive memory growth: {memory_growth_mb:.1f}MB"
        
        print(f"Memory efficiency test: Growth: {memory_growth_mb:.1f}MB")
        
        tracemalloc.stop()
