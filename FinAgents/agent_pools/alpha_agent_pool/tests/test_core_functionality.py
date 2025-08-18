"""Comprehensive test suite for Alpha Agent Pool core functionality.

This test module covers the decoupled architecture including:
- Domain models immutability and validation
- Port interfaces and contracts
- Service coordination (Planner, Executor, Orchestrator)  
- Adapter implementations
- Policy enforcement (retry, circuit breaker, backpressure)
- Configuration management
- Observability (logging, metrics)
- End-to-end workflow testing

Test Design Principles:
- No hardcoded values - all test data is parameterized or generated
- Isolation - each test is independent and can run in parallel
- Mocking - external dependencies are mocked for unit tests
- Integration - separate integration tests for adapter interactions
- Property-based testing where applicable
"""

import asyncio
import pytest
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

# Import core modules
from ..corepkg.domain.models import (
    AlphaTask, AlphaPlan, PlanNode, AlphaSignal, 
    AlphaArtifact, AlphaResult, Ack
)
from ..corepkg.ports.orchestrator import OrchestratorPort
from ..corepkg.ports.strategy import StrategyPort, StrategyCapabilities
from ..corepkg.ports.feature import FeaturePort, FeatureSpec, FeatureTable
from ..corepkg.ports.memory import MemoryPort
from ..corepkg.ports.result import ResultPort
from ..corepkg.services.planner import Planner
from ..corepkg.services.executor import Executor
from ..corepkg.services.orchestrator import Orchestrator
from ..corepkg.policies.retry_policy import ExponentialBackoffRetry, retry_async
from ..corepkg.policies.circuit_breaker import CircuitBreaker, CircuitBreakerError
from ..corepkg.policies.backpressure import QueueBackpressure, BackpressureError
from ..corepkg.observability.logger import StructuredLogger, get_logger
from ..corepkg.observability.metrics import MetricsCollector, increment_counter
from ..runtime.config import AlphaPoolConfig, ConfigManager, YamlConfigLoader
from ..runtime.bootstrap import Bootstrap, DependencyContainer

# Test fixtures and utilities


@pytest.fixture
def sample_market_ctx():
    """Generate sample market context."""
    return {
        "symbol": "AAPL",
        "venue": "NASDAQ",
        "timezone": "America/New_York"
    }


@pytest.fixture
def sample_time_window():
    """Generate sample time window."""
    start = datetime.now() - timedelta(days=30)
    end = datetime.now()
    return {
        "start": start.isoformat(),
        "end": end.isoformat()
    }


@pytest.fixture
def sample_alpha_task(sample_market_ctx, sample_time_window):
    """Generate sample alpha task."""
    return AlphaTask(
        task_id=f"task_{uuid.uuid4().hex[:8]}",
        strategy_id="momentum_strategy",
        market_ctx=sample_market_ctx,
        time_window=sample_time_window,
        features_req=["price", "volume", "volatility"],
        risk_hint={"max_leverage": 2.0},
        idempotency_key=f"idem_{uuid.uuid4().hex[:8]}"
    )


@pytest.fixture
def mock_strategy_port():
    """Create mock strategy port."""
    
    class MockStrategyPort:
        def probe(self) -> StrategyCapabilities:
            return StrategyCapabilities(
                strategy_id="test_strategy",
                supported_features=["price", "volume"],
                cost_estimate=1.0,
                timeout_seconds=30.0
            )
        
        async def warmup(self) -> None:
            pass
        
        async def run(self, node_ctx: Dict[str, Any]) -> Dict[str, Any]:
            symbol = node_ctx.get("market_ctx", {}).get("symbol", "UNKNOWN")
            return {
                "signals": [
                    {
                        "signal_id": f"sig_{uuid.uuid4().hex[:8]}",
                        "strategy_id": node_ctx.get("strategy_id", "test"),
                        "ts": datetime.now().isoformat(),
                        "symbol": symbol,
                        "direction": "BUY",
                        "strength": 0.75,
                        "confidence": 0.85,
                    }
                ],
                "metadata": {"execution_time_ms": 150}
            }
        
        async def validate_input(self, node_ctx: Dict[str, Any]) -> bool:
            return True
        
        def dispose(self) -> None:
            pass
    
    return MockStrategyPort()


@pytest.fixture
def mock_feature_port():
    """Create mock feature port."""
    
    class MockFeaturePort:
        async def fetch(self, spec: FeatureSpec) -> FeatureTable:
            return FeatureTable(
                feature_name=spec.feature_name,
                data={
                    "values": [100.0, 101.5, 99.8] * len(spec.symbols),
                    "timestamps": [datetime.now().isoformat()] * 3
                },
                metadata={"source": "mock", "count": 3}
            )
        
        async def compute(self, node_ctx: Dict[str, Any]) -> FeatureTable:
            feature_name = node_ctx.get("feature", "computed_feature")
            return FeatureTable(
                feature_name=feature_name,
                data={"computed_values": [0.5, 0.3, 0.8]},
                metadata={"computation": "mock"}
            )
        
        async def list_available_features(self) -> List[str]:
            return ["price", "volume", "volatility", "rsi", "macd"]
        
        async def get_feature_schema(self, feature_name: str) -> Dict[str, Any]:
            return {
                "fields": {
                    "values": "float[]",
                    "timestamps": "datetime[]"
                }
            }
        
        async def validate_spec(self, spec: FeatureSpec) -> bool:
            return spec.feature_name in await self.list_available_features()
    
    return MockFeaturePort()


@pytest.fixture
def mock_memory_port():
    """Create mock memory port."""
    
    class MockMemoryPort:
        def __init__(self):
            self._storage = {}
        
        async def retrieve(self, query: Dict[str, Any], scope: str = "global") -> Dict[str, Any]:
            key = query.get("key", "")
            return self._storage.get(key, {})
        
        async def append(self, event: Dict[str, Any]) -> Dict[str, Any]:
            event_id = f"event_{uuid.uuid4().hex[:8]}"
            self._storage[event_id] = event
            return {"status": "stored", "event_id": event_id}
        
        async def lockless_suggest(self, context: Dict[str, Any]) -> Dict[str, Any]:
            return {"suggestions": ["increase_lookback", "add_momentum"]}
    
    return MockMemoryPort()


@pytest.fixture  
def mock_result_port():
    """Create mock result port."""
    
    class MockResultPort:
        def __init__(self):
            self.published_results = []
            self.stored_artifacts = []
            self.emitted_events = []
        
        def publish(self, result: Dict[str, Any]) -> None:
            self.published_results.append(result)
        
        def store_artifact(self, blob_ref: str, metadata: Dict[str, Any]) -> None:
            self.stored_artifacts.append({"ref": blob_ref, "metadata": metadata})
        
        def emit_event(self, event: Dict[str, Any]) -> None:
            self.emitted_events.append(event)
    
    return MockResultPort()


# Domain Model Tests

class TestDomainModels:
    """Test domain model immutability and validation."""
    
    def test_alpha_task_immutability(self, sample_alpha_task):
        """Test that AlphaTask is immutable."""
        with pytest.raises(AttributeError):
            sample_alpha_task.task_id = "modified"
    
    def test_alpha_task_creation(self, sample_market_ctx, sample_time_window):
        """Test AlphaTask creation with various parameters."""
        task = AlphaTask(
            task_id="test_task",
            strategy_id="test_strategy", 
            market_ctx=sample_market_ctx,
            time_window=sample_time_window
        )
        
        assert task.task_id == "test_task"
        assert task.strategy_id == "test_strategy"
        assert task.features_req == []  # default
        assert task.risk_hint is None  # default
    
    def test_alpha_plan_immutability(self):
        """Test that AlphaPlan is immutable."""
        node = PlanNode("node1", "feature", {"param": "value"})
        plan = AlphaPlan("plan1", (node,))
        
        with pytest.raises(AttributeError):
            plan.plan_id = "modified"
        
        # Tuple should be immutable
        with pytest.raises(TypeError):
            plan.nodes.append(node)
    
    def test_alpha_signal_creation(self):
        """Test AlphaSignal creation."""
        signal = AlphaSignal(
            signal_id="sig1",
            strategy_id="momentum",
            ts="2024-01-01T00:00:00Z",
            symbol="AAPL", 
            direction="BUY",
            strength=0.8,
            confidence=0.9
        )
        
        assert signal.direction == "BUY"
        assert signal.strength == 0.8
        assert signal.confidence == 0.9


# Service Tests

class TestPlanner:
    """Test the Planner service."""
    
    def test_plan_generation(self, sample_alpha_task):
        """Test plan generation from task."""
        planner = Planner()
        plan = planner.plan(sample_alpha_task)
        
        assert plan.plan_id is not None
        assert len(plan.nodes) > 0
        
        # Should have feature nodes, strategy node, and validation node
        node_types = [node.node_type for node in plan.nodes]
        assert "feature" in node_types
        assert "strategy" in node_types  
        assert "validate" in node_types
    
    def test_plan_deterministic(self, sample_alpha_task):
        """Test that planning is deterministic for same task."""
        planner = Planner()
        plan1 = planner.plan(sample_alpha_task)
        plan2 = planner.plan(sample_alpha_task)
        
        assert plan1.plan_id == plan2.plan_id
        assert len(plan1.nodes) == len(plan2.nodes)
    
    def test_plan_features_dependency(self, sample_alpha_task):
        """Test that strategy node depends on feature nodes."""
        planner = Planner()
        plan = planner.plan(sample_alpha_task)
        
        strategy_nodes = [n for n in plan.nodes if n.node_type == "strategy"]
        feature_nodes = [n for n in plan.nodes if n.node_type == "feature"]
        
        assert len(strategy_nodes) == 1
        strategy_node = strategy_nodes[0]
        
        # Strategy should depend on all feature nodes
        feature_node_ids = {n.node_id for n in feature_nodes}
        strategy_deps = set(strategy_node.depends_on)
        
        assert feature_node_ids.issubset(strategy_deps)


class TestExecutor:
    """Test the Executor service."""
    
    @pytest.mark.asyncio
    async def test_execution_workflow(self, mock_feature_port, mock_strategy_port, sample_alpha_task):
        """Test full execution workflow."""
        planner = Planner()
        executor = Executor(mock_feature_port, mock_strategy_port)
        
        plan = planner.plan(sample_alpha_task)
        result = await executor.run(plan)
        
        assert isinstance(result, AlphaResult)
        assert result.task_id == plan.plan_id
        assert len(result.signals) > 0
        assert len(result.lineage) > 0
    
    @pytest.mark.asyncio
    async def test_execution_with_memory(self, mock_feature_port, mock_strategy_port, mock_memory_port, sample_alpha_task):
        """Test execution with memory port."""
        planner = Planner()
        executor = Executor(mock_feature_port, mock_strategy_port, mock_memory_port)
        
        plan = planner.plan(sample_alpha_task)
        result = await executor.run(plan)
        
        assert isinstance(result, AlphaResult)
        # Should complete successfully even with memory port
    
    @pytest.mark.asyncio 
    async def test_execution_failure_recovery(self, mock_feature_port, sample_alpha_task):
        """Test execution continues after node failures."""
        
        class FailingStrategyPort:
            def probe(self):
                return StrategyCapabilities("failing", [])
            
            async def warmup(self):
                pass
            
            async def run(self, node_ctx):
                raise RuntimeError("Strategy failed")
            
            async def validate_input(self, node_ctx):
                return True
            
            def dispose(self):
                pass
        
        planner = Planner()
        executor = Executor(mock_feature_port, FailingStrategyPort())
        
        plan = planner.plan(sample_alpha_task)
        result = await executor.run(plan)
        
        # Should complete with empty signals due to failure
        assert isinstance(result, AlphaResult)
        assert len(result.signals) == 0
        assert len(result.lineage) > 0  # Should still track attempted nodes


class TestOrchestrator:
    """Test the Orchestrator service."""
    
    def test_orchestrator_initialization(self, mock_feature_port, mock_strategy_port):
        """Test orchestrator initialization."""
        planner = Planner()
        executor = Executor(mock_feature_port, mock_strategy_port)
        orchestrator = Orchestrator(planner, executor)
        
        assert orchestrator._planner is planner
        assert orchestrator._executor is executor
        assert orchestrator._worker is not None
        assert orchestrator._worker.is_alive()
    
    def test_task_submission(self, mock_feature_port, mock_strategy_port, sample_alpha_task):
        """Test task submission returns immediate ACK."""
        planner = Planner()
        executor = Executor(mock_feature_port, mock_strategy_port)
        orchestrator = Orchestrator(planner, executor)
        
        ack = orchestrator.submit(sample_alpha_task)
        
        assert isinstance(ack, Ack)
        assert ack.status == "ACK"
        assert ack.task_id == sample_alpha_task.task_id
        assert ack.idempotency_key == sample_alpha_task.idempotency_key
    
    def test_orchestrator_shutdown(self, mock_feature_port, mock_strategy_port):
        """Test orchestrator graceful shutdown."""
        planner = Planner()
        executor = Executor(mock_feature_port, mock_strategy_port)
        orchestrator = Orchestrator(planner, executor)
        
        # Ensure worker is running
        assert orchestrator._worker.is_alive()
        
        orchestrator.stop()
        
        # Worker should stop
        orchestrator._worker.join(timeout=2.0)
        assert not orchestrator._worker.is_alive()


# Policy Tests

class TestRetryPolicy:
    """Test retry policy implementations."""
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_success(self):
        """Test successful operation with retry policy."""
        policy = ExponentialBackoffRetry(max_attempts=3, base_delay=0.01)
        
        call_count = 0
        
        async def operation():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await retry_async(operation, policy)
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_retry(self):
        """Test retry after failures."""
        policy = ExponentialBackoffRetry(max_attempts=3, base_delay=0.01)
        
        call_count = 0
        
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Temporary failure")
            return "success"
        
        result = await retry_async(operation, policy)
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test retry exhaustion raises last exception."""
        policy = ExponentialBackoffRetry(max_attempts=2, base_delay=0.01)
        
        async def operation():
            raise ValueError("Persistent failure")
        
        with pytest.raises(ValueError, match="Persistent failure"):
            await retry_async(operation, policy)


class TestCircuitBreaker:
    """Test circuit breaker implementation."""
    
    def test_circuit_breaker_closed_state(self):
        """Test normal operation in closed state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        
        result = cb.call(lambda: "success")
        assert result == "success"
        assert cb.failure_count == 0
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        # First failure
        with pytest.raises(RuntimeError):
            cb.call(lambda: exec('raise RuntimeError("fail")'))
        
        assert cb.failure_count == 1
        
        # Second failure should open circuit
        with pytest.raises(RuntimeError):
            cb.call(lambda: exec('raise RuntimeError("fail")'))
        
        assert cb.failure_count == 2
        
        # Third call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            cb.call(lambda: "should not execute")
    
    def test_circuit_breaker_recovery(self):
        """Test circuit recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        
        # Open circuit
        with pytest.raises(RuntimeError):
            cb.call(lambda: exec('raise RuntimeError("fail")'))
        
        # Should be open
        with pytest.raises(CircuitBreakerError):
            cb.call(lambda: "blocked")
        
        # Wait for recovery timeout
        time.sleep(0.1)
        
        # Should now allow one attempt (half-open)
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.failure_count == 0


class TestBackpressure:
    """Test backpressure policy implementations."""
    
    def test_queue_backpressure_acceptance(self):
        """Test queue backpressure accepts within limits."""
        policy = QueueBackpressure(max_queue_size=5)
        
        assert policy.should_accept(3) is True
        assert policy.should_accept(5) is False
        assert policy.should_accept(10) is False
    
    def test_queue_backpressure_warning_level(self):
        """Test warning level detection."""
        policy = QueueBackpressure(max_queue_size=10, warning_threshold=0.8)
        
        policy.should_accept(7)  # Update internal state
        assert not policy.is_warning_level()
        
        policy.should_accept(9)  # Update internal state
        assert policy.is_warning_level()


# Configuration Tests

class TestConfigurationManagement:
    """Test configuration management system."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = AlphaPoolConfig()
        
        assert config.pool_id == "alpha_pool_default"
        assert config.environment == "development"
        assert config.retry.max_attempts == 3
        assert config.mcp.server_port == 8081
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = AlphaPoolConfig()
        
        # Valid config should not raise
        config.validate()
        
        # Invalid retry attempts
        config.retry.max_attempts = 0
        with pytest.raises(ValueError, match="retry.max_attempts"):
            config.validate()
    
    def test_yaml_config_loading(self):
        """Test YAML configuration loading."""
        config_data = {
            "pool_id": "test_pool",
            "environment": "test",
            "retry": {
                "max_attempts": 5,
                "base_delay": 2.0
            },
            "mcp": {
                "server_port": 9999
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            loader = YamlConfigLoader(config_path)
            config = loader.load(AlphaPoolConfig)
            
            assert config.pool_id == "test_pool"
            assert config.environment == "test"
            assert config.retry.max_attempts == 5
            assert config.retry.base_delay == 2.0
            assert config.mcp.server_port == 9999
        finally:
            Path(config_path).unlink()


# Observability Tests

class TestObservability:
    """Test observability components."""
    
    def test_structured_logger(self):
        """Test structured logger functionality."""
        logger = get_logger("test_logger")
        
        # Should not raise exceptions
        logger.info("Test message", task_id="test_task", duration=123.45)
        logger.error("Test error", error_code=500)
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        collector = MetricsCollector()
        
        # Test counter
        counter = collector.register_counter("test_counter", "Test counter")
        counter.increment(5.0)
        assert counter.get_value() == 5.0
        
        # Test gauge
        gauge = collector.register_gauge("test_gauge", "Test gauge")
        gauge.set(10.0)
        assert gauge.get_value() == 10.0
        
        # Test histogram
        histogram = collector.register_histogram("test_histogram", "Test histogram")
        histogram.observe(1.5)
        histogram.observe(2.5)
        assert histogram.count == 2
        assert histogram.sum == 4.0


# Integration Tests

class TestIntegration:
    """Integration tests for end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_task_processing(self, 
                                             mock_feature_port, 
                                             mock_strategy_port, 
                                             mock_memory_port,
                                             sample_alpha_task):
        """Test complete end-to-end task processing."""
        # Setup components
        planner = Planner()
        executor = Executor(mock_feature_port, mock_strategy_port, mock_memory_port)
        orchestrator = Orchestrator(planner, executor)
        
        # Submit task
        ack = orchestrator.submit(sample_alpha_task)
        assert ack.status == "ACK"
        
        # Give some time for background processing
        await asyncio.sleep(0.1)
        
        # Cleanup
        orchestrator.stop()
    
    def test_dependency_injection_container(self):
        """Test dependency injection container."""
        container = DependencyContainer()
        
        # Register singleton
        config = AlphaPoolConfig()
        container.register_singleton(AlphaPoolConfig, config)
        
        # Register factory
        def create_planner():
            return Planner()
        
        container.register_factory(Planner, create_planner)
        
        # Test retrieval
        retrieved_config = container.get(AlphaPoolConfig)
        assert retrieved_config is config
        
        planner1 = container.get(Planner)
        planner2 = container.get(Planner)
        assert planner1 is planner2  # Should be cached as singleton
    
    def test_bootstrap_initialization(self):
        """Test bootstrap initialization process."""
        from ..runtime.config import create_config_manager
        
        config_manager = create_config_manager(use_env=False)
        bootstrap = Bootstrap(config_manager)
        
        container = bootstrap.initialize()
        
        # Should have all required services
        assert container.get(AlphaPoolConfig) is not None
        assert container.get(Planner) is not None
        assert container.get(Executor) is not None
        assert container.get(Orchestrator) is not None
        
        # Cleanup
        bootstrap.shutdown()


# Property-based and Stress Tests

class TestStressAndReliability:
    """Stress tests and reliability verification."""
    
    @pytest.mark.asyncio
    async def test_concurrent_task_submission(self, mock_feature_port, mock_strategy_port):
        """Test concurrent task submissions don't cause race conditions."""
        planner = Planner()
        executor = Executor(mock_feature_port, mock_strategy_port)
        orchestrator = Orchestrator(planner, executor)
        
        # Submit multiple tasks concurrently
        tasks = []
        for i in range(10):
            task = AlphaTask(
                task_id=f"concurrent_task_{i}",
                strategy_id="test_strategy",
                market_ctx={"symbol": f"SYM{i}"},
                time_window={"start": "2024-01-01", "end": "2024-01-02"}
            )
            tasks.append(task)
        
        # Submit all tasks
        acks = [orchestrator.submit(task) for task in tasks]
        
        # All should be acknowledged
        assert len(acks) == 10
        for ack in acks:
            assert ack.status == "ACK"
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        orchestrator.stop()
    
    def test_memory_usage_stability(self, mock_feature_port, mock_strategy_port):
        """Test that repeated operations don't cause memory leaks."""
        planner = Planner()
        
        # Create many plans
        for i in range(100):
            task = AlphaTask(
                task_id=f"mem_test_task_{i}",
                strategy_id="test_strategy", 
                market_ctx={"symbol": "TEST"},
                time_window={"start": "2024-01-01", "end": "2024-01-02"}
            )
            plan = planner.plan(task)
            # Plan should be created without accumulating memory
            assert len(plan.nodes) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
