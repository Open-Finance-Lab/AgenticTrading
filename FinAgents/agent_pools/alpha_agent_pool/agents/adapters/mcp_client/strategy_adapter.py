from __future__ import annotations

import asyncio
from typing import Any, Dict

from corepkg.ports.strategy import StrategyPort, StrategyCapabilities
from corepkg.policies.retry_policy import ExponentialBackoffRetry, retry_async
from corepkg.policies.circuit_breaker import CircuitBreaker, CircuitBreakerError
from corepkg.policies.timeout_policy import TimeoutPolicy
from corepkg.observability.logger import get_logger
from corepkg.observability.metrics import increment_counter, observe_histogram


class MCPStrategyClientAdapter(StrategyPort):
    """Adapter that proxies strategy execution to MCP agents with full resilience.

    Features:
    - Retry policy for transient failures
    - Circuit breaker for persistent failures  
    - Timeout protection
    - Comprehensive observability
    """

    def __init__(self, 
                 call_tool_async,
                 strategy_id: str = "mcp_strategy",
                 timeout_seconds: float = 30.0,
                 max_retries: int = 3):
        self._call = call_tool_async
        self._strategy_id = strategy_id
        self._timeout_seconds = timeout_seconds
        
        # Initialize policies
        self._retry_policy = ExponentialBackoffRetry(
            max_attempts=max_retries,
            base_delay=1.0,
            max_delay=30.0
        )
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0
        )
        self._timeout_policy = TimeoutPolicy(timeout_seconds)
        
        # Initialize observability
        self._logger = get_logger(f"strategy_adapter.{strategy_id}")
        self._is_warmed_up = False

    def probe(self) -> StrategyCapabilities:
        """Probe strategy capabilities."""
        return StrategyCapabilities(
            strategy_id=self._strategy_id,
            supported_features=["price", "volume", "volatility"],
            cost_estimate=1.0,
            timeout_seconds=self._timeout_seconds,
            requires_warmup=True
        )

    async def warmup(self) -> None:
        """Initialize strategy if needed."""
        if self._is_warmed_up:
            return
            
        self._logger.info("Warming up MCP strategy adapter", strategy_id=self._strategy_id)
        
        try:
            # Attempt a simple probe call to verify connectivity
            await self._timeout_policy.call_async(
                self._call,
                5.0,  # Short timeout for warmup
                "probe",
                {"strategy_id": self._strategy_id}
            )
            self._is_warmed_up = True
            increment_counter("strategy_warmup_success", labels={"strategy_id": self._strategy_id})
            self._logger.info("Strategy warmup completed", strategy_id=self._strategy_id)
            
        except Exception as e:
            increment_counter("strategy_warmup_failure", labels={"strategy_id": self._strategy_id})
            self._logger.error("Strategy warmup failed", 
                             strategy_id=self._strategy_id, 
                             error=str(e))
            # Don't raise - continue with cold start

    async def run(self, node_ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strategy with full error handling and observability."""
        start_time = asyncio.get_event_loop().time()
        
        self._logger.info("Executing strategy", 
                         strategy_id=self._strategy_id,
                         symbol=node_ctx.get("market_ctx", {}).get("symbol"))
        
        try:
            # Prepare parameters
            params = {
                "strategy_id": node_ctx.get("strategy_id", self._strategy_id),
                "market_ctx": node_ctx.get("market_ctx", {}),
                "features": node_ctx.get("features", {}),
            }
            
            # Execute with circuit breaker and retry
            result = await self._circuit_breaker.call_async(
                self._execute_with_retry,
                params
            )
            
            # Record success metrics
            duration = asyncio.get_event_loop().time() - start_time
            observe_histogram("strategy_execution_duration", 
                            duration, 
                            labels={"strategy_id": self._strategy_id, "status": "success"})
            increment_counter("strategy_execution_success", 
                            labels={"strategy_id": self._strategy_id})
            
            self._logger.info("Strategy execution completed", 
                            strategy_id=self._strategy_id,
                            duration_ms=duration * 1000,
                            signals_count=len(result.get("signals", [])))
            
            return result
            
        except CircuitBreakerError as e:
            increment_counter("strategy_execution_circuit_breaker", 
                            labels={"strategy_id": self._strategy_id})
            self._logger.error("Strategy execution blocked by circuit breaker",
                             strategy_id=self._strategy_id,
                             error=str(e))
            
            # Return empty result with error indication
            return {
                "signals": [],
                "metadata": {
                    "error": "circuit_breaker_open",
                    "message": str(e)
                }
            }
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            observe_histogram("strategy_execution_duration",
                            duration,
                            labels={"strategy_id": self._strategy_id, "status": "error"})
            increment_counter("strategy_execution_failure",
                            labels={"strategy_id": self._strategy_id})
            
            self._logger.error("Strategy execution failed",
                             strategy_id=self._strategy_id,
                             error=str(e),
                             duration_ms=duration * 1000)
            
            # Return empty result with error indication
            return {
                "signals": [],
                "metadata": {
                    "error": "execution_failed",
                    "message": str(e)
                }
            }

    async def _execute_with_retry(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strategy call with retry policy."""
        
        async def strategy_call():
            return await self._timeout_policy.call_async(
                self._call,
                self._timeout_seconds,
                "generate_signal",
                params
            )
        
        return await retry_async(strategy_call, self._retry_policy)

    async def validate_input(self, node_ctx: Dict[str, Any]) -> bool:
        """Validate input context."""
        required_fields = ["strategy_id", "market_ctx"]
        
        for field in required_fields:
            if field not in node_ctx:
                self._logger.warning("Missing required field in context",
                                   field=field,
                                   strategy_id=self._strategy_id)
                return False
        
        market_ctx = node_ctx.get("market_ctx", {})
        if not market_ctx.get("symbol"):
            self._logger.warning("Missing symbol in market context",
                               strategy_id=self._strategy_id)
            return False
        
        return True

    def dispose(self) -> None:
        """Clean up adapter resources."""
        self._logger.info("Disposing strategy adapter", strategy_id=self._strategy_id)
        
        # Reset circuit breaker
        self._circuit_breaker.reset()
        
        increment_counter("strategy_adapter_disposed", 
                        labels={"strategy_id": self._strategy_id})

