from __future__ import annotations

from typing import Protocol, Dict, Any, List, Optional


class StrategyCapabilities:
    """Strategy capability metadata."""
    
    def __init__(self, 
                 strategy_id: str,
                 supported_features: List[str],
                 cost_estimate: float = 1.0,
                 timeout_seconds: float = 30.0,
                 requires_warmup: bool = False):
        self.strategy_id = strategy_id
        self.supported_features = supported_features
        self.cost_estimate = cost_estimate
        self.timeout_seconds = timeout_seconds
        self.requires_warmup = requires_warmup


class StrategyPort(Protocol):
    """Port for strategy plugins (local or via MCP).
    
    This port defines the interface for alpha factor generation strategies.
    Implementations can be local plugins or remote services accessed via MCP.
    """

    def probe(self) -> StrategyCapabilities:
        """Probe strategy capabilities and metadata.
        
        Returns:
            Strategy capability information including supported features
            and performance characteristics
        """
        ...

    async def warmup(self) -> None:
        """Initialize strategy if warmup is required.
        
        This method should be called once before the first run() call
        if the strategy requires initialization.
        """
        ...

    async def run(self, node_ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strategy with given context.
        
        Args:
            node_ctx: Execution context containing:
                - strategy_id: Strategy identifier
                - market_ctx: Market context (symbol, venue, etc.)
                - features: Input features from previous nodes
                - parameters: Strategy-specific parameters
                
        Returns:
            Strategy output containing:
                - signals: List of alpha signals generated
                - metadata: Execution metadata (duration, confidence, etc.)
                - artifacts: Optional artifacts (charts, logs, etc.)
                
        Raises:
            TimeoutError: If execution exceeds timeout
            ValueError: If context is invalid
            RuntimeError: If strategy execution fails
        """
        ...

    async def validate_input(self, node_ctx: Dict[str, Any]) -> bool:
        """Validate input context before execution.
        
        Args:
            node_ctx: Execution context to validate
            
        Returns:
            True if context is valid for this strategy
        """
        ...

    def dispose(self) -> None:
        """Clean up strategy resources.
        
        This method should be called when the strategy is no longer needed
        to properly release any allocated resources.
        """
        ...

