from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable

from corepkg.ports.feature import FeaturePort, FeatureSpec, FeatureTable
from corepkg.policies.retry_policy import ExponentialBackoffRetry, retry_async
from corepkg.policies.timeout_policy import TimeoutPolicy
from corepkg.observability.logger import get_logger
from corepkg.observability.metrics import increment_counter, observe_histogram


class SimpleFeatureAdapter(FeaturePort):
    """Enhanced feature adapter with caching, retry, and observability.
    
    Features:
    - In-memory caching with TTL
    - Retry policy for transient failures
    - Timeout protection
    - Comprehensive observability
    - Mock data generation for testing
    """

    def __init__(self, 
                 fetch_callable: Optional[Callable] = None,
                 compute_callable: Optional[Callable] = None,
                 cache_ttl_seconds: int = 300,
                 fetch_timeout: float = 15.0):
        self._fetch = fetch_callable
        self._compute = compute_callable
        self._cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, tuple] = {}  # key -> (data, timestamp)
        
        # Initialize policies
        self._retry_policy = ExponentialBackoffRetry(
            max_attempts=3,
            base_delay=0.5,
            max_delay=10.0
        )
        self._timeout_policy = TimeoutPolicy(fetch_timeout)
        
        # Initialize observability
        self._logger = get_logger("feature_adapter")

    async def fetch(self, spec: FeatureSpec) -> FeatureTable:
        """Fetch feature data with caching and error handling."""
        start_time = asyncio.get_event_loop().time()
        cache_key = self._build_cache_key(spec)
        
        self._logger.info("Fetching feature data",
                         feature_name=spec.feature_name,
                         symbols=spec.symbols,
                         start_date=spec.start_date.isoformat(),
                         end_date=spec.end_date.isoformat())
        
        try:
            # Check cache first
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                increment_counter("feature_cache_hit", 
                                labels={"feature": spec.feature_name})
                self._logger.info("Feature data found in cache",
                                feature_name=spec.feature_name)
                return cached_data
            
            increment_counter("feature_cache_miss", 
                            labels={"feature": spec.feature_name})
            
            # Fetch data with retry and timeout
            if callable(self._fetch):
                data = await retry_async(
                    lambda: self._timeout_policy.call_async(
                        self._fetch, 
                        None,  # Use default timeout
                        spec
                    ),
                    self._retry_policy
                )
            else:
                # Generate mock data
                data = self._generate_mock_data(spec)
            
            # Create feature table
            feature_table = FeatureTable(
                feature_name=spec.feature_name,
                data=data,
                metadata={
                    "source": "adapter",
                    "symbols": spec.symbols,
                    "frequency": spec.frequency,
                    "transformations": spec.transformations,
                    "fetch_time": datetime.now().isoformat()
                }
            )
            
            # Cache the result
            self._cache_data(cache_key, feature_table)
            
            # Record metrics
            duration = asyncio.get_event_loop().time() - start_time
            observe_histogram("feature_fetch_duration",
                            duration,
                            labels={"feature": spec.feature_name, "status": "success"})
            increment_counter("feature_fetch_success",
                            labels={"feature": spec.feature_name})
            
            self._logger.info("Feature data fetched successfully",
                            feature_name=spec.feature_name,
                            duration_ms=duration * 1000,
                            data_points=len(data.get("values", [])))
            
            return feature_table
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            observe_histogram("feature_fetch_duration",
                            duration,
                            labels={"feature": spec.feature_name, "status": "error"})
            increment_counter("feature_fetch_failure",
                            labels={"feature": spec.feature_name})
            
            self._logger.error("Feature fetch failed",
                             feature_name=spec.feature_name,
                             error=str(e),
                             duration_ms=duration * 1000)
            
            # Return empty feature table on error
            return FeatureTable(
                feature_name=spec.feature_name,
                data={"values": [], "error": str(e)},
                metadata={"error": "fetch_failed", "message": str(e)}
            )

    async def compute(self, node_ctx: Dict[str, Any]) -> FeatureTable:
        """Compute derived features with error handling."""
        start_time = asyncio.get_event_loop().time()
        feature_name = node_ctx.get("feature", "computed_feature")
        
        self._logger.info("Computing derived feature",
                         feature_name=feature_name,
                         inputs=list(node_ctx.get("inputs", {}).keys()))
        
        try:
            if callable(self._compute):
                data = await retry_async(
                    lambda: self._timeout_policy.call_async(
                        self._compute,
                        None,
                        node_ctx
                    ),
                    self._retry_policy
                )
            else:
                # Generate mock computed data
                data = self._generate_mock_computed_data(node_ctx)
            
            feature_table = FeatureTable(
                feature_name=feature_name,
                data=data,
                metadata={
                    "computation": "adapter",
                    "inputs": list(node_ctx.get("inputs", {}).keys()),
                    "parameters": node_ctx.get("parameters", {}),
                    "compute_time": datetime.now().isoformat()
                }
            )
            
            # Record metrics
            duration = asyncio.get_event_loop().time() - start_time
            observe_histogram("feature_compute_duration",
                            duration,
                            labels={"feature": feature_name, "status": "success"})
            increment_counter("feature_compute_success",
                            labels={"feature": feature_name})
            
            self._logger.info("Feature computation completed",
                            feature_name=feature_name,
                            duration_ms=duration * 1000)
            
            return feature_table
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            observe_histogram("feature_compute_duration",
                            duration,
                            labels={"feature": feature_name, "status": "error"})
            increment_counter("feature_compute_failure",
                            labels={"feature": feature_name})
            
            self._logger.error("Feature computation failed",
                             feature_name=feature_name,
                             error=str(e),
                             duration_ms=duration * 1000)
            
            return FeatureTable(
                feature_name=feature_name,
                data={"values": [], "error": str(e)},
                metadata={"error": "compute_failed", "message": str(e)}
            )

    async def list_available_features(self) -> List[str]:
        """List all available feature names."""
        return [
            "price", "volume", "volatility", "returns", "sma_20", "sma_50",
            "rsi", "macd", "bollinger_bands", "atr", "momentum", "mean_reversion"
        ]

    async def get_feature_schema(self, feature_name: str) -> Dict[str, Any]:
        """Get schema information for a specific feature."""
        schemas = {
            "price": {
                "fields": {
                    "values": {"type": "float[]", "description": "Price values"},
                    "timestamps": {"type": "datetime[]", "description": "Timestamps"}
                },
                "frequency": "1D",
                "unit": "USD"
            },
            "volume": {
                "fields": {
                    "values": {"type": "int[]", "description": "Volume values"},
                    "timestamps": {"type": "datetime[]", "description": "Timestamps"}
                },
                "frequency": "1D",
                "unit": "shares"
            },
            "volatility": {
                "fields": {
                    "values": {"type": "float[]", "description": "Volatility values"},
                    "timestamps": {"type": "datetime[]", "description": "Timestamps"}
                },
                "frequency": "1D", 
                "unit": "decimal"
            }
        }
        
        return schemas.get(feature_name, {
            "fields": {
                "values": {"type": "float[]", "description": "Feature values"},
                "timestamps": {"type": "datetime[]", "description": "Timestamps"}
            }
        })

    async def validate_spec(self, spec: FeatureSpec) -> bool:
        """Validate feature specification."""
        available_features = await self.list_available_features()
        
        if spec.feature_name not in available_features:
            self._logger.warning("Unknown feature requested",
                               feature_name=spec.feature_name)
            return False
        
        if not spec.symbols:
            self._logger.warning("No symbols specified in feature spec")
            return False
        
        if spec.start_date >= spec.end_date:
            self._logger.warning("Invalid date range in feature spec",
                               start_date=spec.start_date.isoformat(),
                               end_date=spec.end_date.isoformat())
            return False
        
        return True

    def _build_cache_key(self, spec: FeatureSpec) -> str:
        """Build cache key from feature specification."""
        return f"{spec.feature_name}:{':'.join(spec.symbols)}:{spec.start_date}:{spec.end_date}:{spec.frequency}"

    def _get_cached_data(self, cache_key: str) -> Optional[FeatureTable]:
        """Get data from cache if still valid."""
        if cache_key not in self._cache:
            return None
        
        data, timestamp = self._cache[cache_key]
        
        # Check TTL
        if (datetime.now().timestamp() - timestamp) > self._cache_ttl:
            del self._cache[cache_key]
            return None
        
        return data

    def _cache_data(self, cache_key: str, data: FeatureTable) -> None:
        """Cache feature data with timestamp."""
        self._cache[cache_key] = (data, datetime.now().timestamp())
        
        # Simple cache size management
        if len(self._cache) > 1000:
            # Remove oldest entries
            sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_items[:100]:
                del self._cache[key]

    def _generate_mock_data(self, spec: FeatureSpec) -> Dict[str, Any]:
        """Generate realistic mock data for testing."""
        import random
        
        days = (spec.end_date - spec.start_date).days
        timestamps = [
            (spec.start_date + timedelta(days=i)).isoformat()
            for i in range(min(days, 100))  # Limit to 100 data points
        ]
        
        if spec.feature_name == "price":
            base_price = 100.0
            values = []
            for i in range(len(timestamps)):
                # Simulate price movement
                change = random.uniform(-0.05, 0.05)
                base_price *= (1 + change)
                values.append(round(base_price, 2))
        
        elif spec.feature_name == "volume":
            values = [random.randint(1000000, 10000000) for _ in timestamps]
        
        elif spec.feature_name == "volatility":
            values = [random.uniform(0.1, 0.8) for _ in timestamps]
        
        else:
            # Generic numeric feature
            values = [random.uniform(0, 100) for _ in timestamps]
        
        return {
            "values": values,
            "timestamps": timestamps,
            "symbols": spec.symbols
        }

    def _generate_mock_computed_data(self, node_ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock computed feature data."""
        import random
        
        feature_name = node_ctx.get("feature", "computed")
        inputs = node_ctx.get("inputs", {})
        
        # Generate based on inputs if available
        if inputs:
            input_values = []
            for input_data in inputs.values():
                if isinstance(input_data, dict) and "values" in input_data:
                    input_values.extend(input_data["values"])
            
            if input_values:
                # Simple transformation of input data
                computed_values = [v * random.uniform(0.8, 1.2) for v in input_values[:10]]
            else:
                computed_values = [random.uniform(0, 1) for _ in range(10)]
        else:
            computed_values = [random.uniform(0, 1) for _ in range(10)]
        
        return {
            "computed_values": computed_values,
            "feature_type": feature_name,
            "input_count": len(inputs)
        }

