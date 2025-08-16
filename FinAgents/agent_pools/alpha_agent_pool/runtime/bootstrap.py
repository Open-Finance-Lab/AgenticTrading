"""Bootstrap and dependency injection for Alpha Agent Pool."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Type, TypeVar, Callable

from corepkg.domain.models import AlphaTask, Ack
from corepkg.ports.orchestrator import OrchestratorPort
from corepkg.ports.strategy import StrategyPort
from corepkg.ports.feature import FeaturePort
from corepkg.ports.memory import MemoryPort
from corepkg.ports.result import ResultPort
from corepkg.services.planner import Planner
from corepkg.services.executor import Executor
from corepkg.services.orchestrator import Orchestrator
from corepkg.observability.logger import get_logger
from corepkg.observability.metrics import get_metrics_collector
from runtime.config import AlphaPoolConfig, ConfigManager

T = TypeVar('T')


class DependencyContainer:
    """Dependency injection container."""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[[], Any]] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = threading.RLock()
    
    def register_singleton(self, interface: Type[T], instance: T) -> None:
        """Register a singleton instance."""
        with self._lock:
            self._singletons[interface] = instance
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function."""
        with self._lock:
            self._factories[interface] = factory
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register a specific instance."""
        with self._lock:
            self._services[interface] = instance
    
    def get(self, interface: Type[T]) -> T:
        """Get instance of specified interface."""
        with self._lock:
            # Check singletons first
            if interface in self._singletons:
                return self._singletons[interface]
            
            # Check registered instances
            if interface in self._services:
                return self._services[interface]
            
            # Check factories
            if interface in self._factories:
                instance = self._factories[interface]()
                self._singletons[interface] = instance  # Cache as singleton
                return instance
            
            raise ValueError(f"No registration found for {interface}")
    
    def get_optional(self, interface: Type[T]) -> Optional[T]:
        """Get instance if registered, None otherwise."""
        try:
            return self.get(interface)
        except ValueError:
            return None


class Bootstrap:
    """Bootstrap the Alpha Agent Pool with all dependencies."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.container = DependencyContainer()
        self.logger = get_logger("bootstrap")
        self._orchestrator: Optional[Orchestrator] = None
    
    def initialize(self) -> DependencyContainer:
        """Initialize all dependencies and return container."""
        config = self.config_manager.config
        
        self.logger.info("Initializing Alpha Agent Pool", pool_id=config.pool_id)
        
        # Register configuration
        self.container.register_singleton(AlphaPoolConfig, config)
        self.container.register_singleton(ConfigManager, self.config_manager)
        
        # Register observability components
        self._register_observability(config)
        
        # Register ports with appropriate adapters
        self._register_ports(config)
        
        # Register core services
        self._register_services(config)
        
        # Register orchestrator
        self._register_orchestrator(config)
        
        self.logger.info("Alpha Agent Pool initialized successfully")
        
        return self.container
    
    def _register_observability(self, config: AlphaPoolConfig) -> None:
        """Register observability components."""
        # Metrics collector is global singleton
        metrics_collector = get_metrics_collector()
        self.container.register_singleton(type(metrics_collector), metrics_collector)
        
        # Register loggers
        self.container.register_factory(
            type(self.logger),
            lambda: get_logger("alpha_pool", config.observability.log_level)
        )
    
    def _register_ports(self, config: AlphaPoolConfig) -> None:
        """Register port implementations."""
        # Feature port
        def create_feature_port() -> FeaturePort:
            from adapters.feature.simple_feature_adapter import SimpleFeatureAdapter
            return SimpleFeatureAdapter()
        
        self.container.register_factory(FeaturePort, create_feature_port)
        
        # Memory port (if A2A coordinator is available)
        def create_memory_port() -> Optional[MemoryPort]:
            try:
                # This would be injected from the existing core.py
                from adapters.a2a_client.adapter import A2AClientAdapter
                # Would get coordinator from existing system
                return None  # Placeholder
            except Exception:
                return None
        
        self.container.register_factory(MemoryPort, create_memory_port)
        
        # Strategy port
        def create_strategy_port() -> StrategyPort:
            from adapters.mcp_client.strategy_adapter import MCPStrategyClientAdapter
            
            async def mock_strategy_call(tool_name: str, params: dict):
                # Mock implementation for demonstration
                return {
                    "signals": [
                        {
                            "signal_id": f"sig_{params.get('strategy_id','unknown')}",
                            "strategy_id": params.get("strategy_id", "unknown"),
                            "ts": "2024-01-01T00:00:00Z",
                            "symbol": params.get("market_ctx", {}).get("symbol", "UNKNOWN"),
                            "direction": "HOLD",
                            "strength": 0.0,
                            "confidence": 0.0,
                        }
                    ]
                }
            
            return MCPStrategyClientAdapter(mock_strategy_call)
        
        self.container.register_factory(StrategyPort, create_strategy_port)
        
        # Result port
        def create_result_port() -> ResultPort:
            from adapters.storage.outbox_adapter import FileOutboxAdapter
            return FileOutboxAdapter("./outbox")
        
        self.container.register_factory(ResultPort, create_result_port)
    
    def _register_services(self, config: AlphaPoolConfig) -> None:
        """Register core services."""
        # Planner
        def create_planner() -> Planner:
            return Planner()
        
        self.container.register_factory(Planner, create_planner)
        
        # Executor
        def create_executor() -> Executor:
            feature_port = self.container.get(FeaturePort)
            strategy_port = self.container.get(StrategyPort)
            memory_port = self.container.get_optional(MemoryPort)
            return Executor(feature_port, strategy_port, memory_port)
        
        self.container.register_factory(Executor, create_executor)
    
    def _register_orchestrator(self, config: AlphaPoolConfig) -> None:
        """Register orchestrator."""
        def create_orchestrator() -> Orchestrator:
            planner = self.container.get(Planner)
            executor = self.container.get(Executor)
            return Orchestrator(planner, executor)
        
        self.container.register_factory(Orchestrator, create_orchestrator)
        
        # Also register as OrchestratorPort
        def get_orchestrator_port() -> OrchestratorPort:
            return self.container.get(Orchestrator)
        
        self.container.register_factory(OrchestratorPort, get_orchestrator_port)
    
    def get_orchestrator(self) -> Orchestrator:
        """Get the main orchestrator instance."""
        if self._orchestrator is None:
            self._orchestrator = self.container.get(Orchestrator)
        return self._orchestrator
    
    def shutdown(self) -> None:
        """Shutdown all components gracefully."""
        self.logger.info("Shutting down Alpha Agent Pool")
        
        if self._orchestrator:
            self._orchestrator.stop()
        
        self.logger.info("Alpha Agent Pool shutdown complete")


def create_bootstrap(config_manager: ConfigManager) -> Bootstrap:
    """Create and initialize bootstrap instance.
    
    Args:
        config_manager: Configuration manager
        
    Returns:
        Initialized Bootstrap instance
    """
    bootstrap = Bootstrap(config_manager)
    bootstrap.initialize()
    return bootstrap
