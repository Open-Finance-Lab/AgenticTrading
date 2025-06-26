"""
Risk Memory Bridge - Integration with External Memory Agent

This module provides integration with the External Memory Agent for risk analysis data,
enabling storage, retrieval, and management of risk calculations, model parameters,
and historical analysis results.

The memory bridge supports:
- Risk calculation result persistence
- Model parameter versioning
- Historical risk data storage
- Cross-analysis correlation tracking
- Performance metrics storage
- Unified logging through External Memory Agent

Author: Jifeng Li
License: openMDW
"""

import logging
import asyncio
import json
import aiohttp
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import uuid

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class RiskAnalysisRecord:
    """Data class for risk analysis records."""
    id: str
    agent_name: str
    risk_type: str
    analysis_type: str
    input_parameters: Dict[str, Any]
    results: Dict[str, Any]
    timestamp: str
    execution_time_ms: float
    status: str
    error_message: Optional[str] = None


@dataclass
class RiskModelParameters:
    """Data class for risk model parameters."""
    model_id: str
    model_name: str
    parameters: Dict[str, Any]
    version: str
    created_at: str
    performance_metrics: Dict[str, Any]


class RiskMemoryBridge:
    """
    Memory bridge for risk analysis agents to interact with external memory systems.
    
    This class provides methods to store and retrieve risk analysis results,
    model parameters, and historical data through the External Memory Agent.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Risk Memory Bridge.
        
        Args:
            config: Configuration dictionary containing memory agent settings
        """
        self.config = config
        self.memory_agent_url = config.get('memory_agent_url', 'http://localhost:8001')
        self.enable_external_memory = config.get('enable_external_memory', True)
        self.session = None
        self.logger = logging.getLogger("RiskMemoryBridge")
        
        # Local cache for frequently accessed data
        self.local_cache = {}
        self.cache_ttl = config.get('cache_ttl_seconds', 300)  # 5 minutes default
    
    async def initialize(self):
        """Initialize HTTP session and test connection to memory agent."""
        if self.enable_external_memory:
            try:
                self.session = aiohttp.ClientSession()
                # Test connection
                await self._test_memory_agent_connection()
                self.logger.info("Risk Memory Bridge initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize memory bridge: {e}")
                self.enable_external_memory = False
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
    
    async def _test_memory_agent_connection(self):
        """Test connection to the External Memory Agent."""
        if not self.session:
            return
        
        try:
            async with self.session.get(
                f"{self.memory_agent_url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    self.logger.info("Memory agent connection test successful")
                else:
                    self.logger.warning(f"Memory agent health check returned status {response.status}")
        except Exception as e:
            self.logger.error(f"Memory agent connection test failed: {e}")
            raise
    
    async def record_event(self, agent_name: str, task: str, input_data: Dict[str, Any], 
                          summary: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Record a risk analysis event in the memory system.
        
        Args:
            agent_name: Name of the executing agent
            task: Task identifier
            input_data: Input parameters used
            summary: Summary of execution results
            metadata: Optional additional metadata
        """
        try:
            event_data = {
                "id": str(uuid.uuid4()),
                "agent_name": agent_name,
                "task": task,
                "input_data": input_data,
                "summary": summary,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat(),
                "component": "RiskAgentPool"
            }
            
            # Log locally first
            self.logger.info(f"[RiskMemoryBridge] {event_data}")
            
            # Send to external memory agent if enabled
            if self.enable_external_memory and self.session:
                await self._send_to_memory_agent("record_event", event_data)
            
        except Exception as e:
            self.logger.error(f"Failed to record event: {e}")
    
    async def store_risk_analysis_result(self, record: RiskAnalysisRecord) -> str:
        """
        Store a risk analysis result in the memory system.
        
        Args:
            record: Risk analysis record to store
            
        Returns:
            Storage ID for the record
        """
        try:
            record_data = asdict(record)
            
            # Store in local cache
            cache_key = f"risk_analysis_{record.id}"
            self.local_cache[cache_key] = {
                "data": record_data,
                "timestamp": datetime.utcnow()
            }
            
            # Store in external memory if enabled
            if self.enable_external_memory and self.session:
                storage_id = await self._send_to_memory_agent("store_risk_analysis", record_data)
                self.logger.info(f"Stored risk analysis result with ID: {storage_id}")
                return storage_id
            
            return record.id
            
        except Exception as e:
            self.logger.error(f"Failed to store risk analysis result: {e}")
            return record.id
    
    async def retrieve_risk_analysis_results(self, filters: Dict[str, Any]) -> List[RiskAnalysisRecord]:
        """
        Retrieve risk analysis results based on filters.
        
        Args:
            filters: Filter criteria (agent_name, risk_type, date_range, etc.)
            
        Returns:
            List of matching risk analysis records
        """
        try:
            # Check local cache first
            cached_results = self._search_local_cache(filters)
            if cached_results:
                return cached_results
            
            # Query external memory if enabled
            if self.enable_external_memory and self.session:
                results_data = await self._send_to_memory_agent("retrieve_risk_analyses", filters)
                
                # Convert to RiskAnalysisRecord objects
                results = []
                for data in results_data:
                    try:
                        record = RiskAnalysisRecord(**data)
                        results.append(record)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse risk analysis record: {e}")
                
                return results
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve risk analysis results: {e}")
            return []
    
    async def store_model_parameters(self, params: RiskModelParameters) -> str:
        """
        Store risk model parameters.
        
        Args:
            params: Risk model parameters to store
            
        Returns:
            Storage ID for the parameters
        """
        try:
            params_data = asdict(params)
            
            # Store in local cache
            cache_key = f"model_params_{params.model_id}_{params.version}"
            self.local_cache[cache_key] = {
                "data": params_data,
                "timestamp": datetime.utcnow()
            }
            
            # Store in external memory if enabled
            if self.enable_external_memory and self.session:
                storage_id = await self._send_to_memory_agent("store_model_parameters", params_data)
                self.logger.info(f"Stored model parameters with ID: {storage_id}")
                return storage_id
            
            return params.model_id
            
        except Exception as e:
            self.logger.error(f"Failed to store model parameters: {e}")
            return params.model_id
    
    async def retrieve_model_parameters(self, model_id: str, version: Optional[str] = None) -> Optional[RiskModelParameters]:
        """
        Retrieve risk model parameters.
        
        Args:
            model_id: Model identifier
            version: Specific version (latest if None)
            
        Returns:
            Risk model parameters or None if not found
        """
        try:
            # Check local cache first
            cache_key = f"model_params_{model_id}_{version or 'latest'}"
            if cache_key in self.local_cache:
                cached_data = self.local_cache[cache_key]
                if self._is_cache_valid(cached_data["timestamp"]):
                    return RiskModelParameters(**cached_data["data"])
            
            # Query external memory if enabled
            if self.enable_external_memory and self.session:
                query = {"model_id": model_id}
                if version:
                    query["version"] = version
                
                params_data = await self._send_to_memory_agent("retrieve_model_parameters", query)
                if params_data:
                    return RiskModelParameters(**params_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve model parameters: {e}")
            return None
    
    async def store_historical_risk_data(self, data: Dict[str, Any]) -> str:
        """
        Store historical risk data for trend analysis.
        
        Args:
            data: Historical risk data dictionary
            
        Returns:
            Storage ID for the data
        """
        try:
            data_with_id = {
                "id": str(uuid.uuid4()),
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
                "data_type": "historical_risk"
            }
            
            # Store in external memory if enabled
            if self.enable_external_memory and self.session:
                storage_id = await self._send_to_memory_agent("store_historical_data", data_with_id)
                self.logger.info(f"Stored historical risk data with ID: {storage_id}")
                return storage_id
            
            return data_with_id["id"]
            
        except Exception as e:
            self.logger.error(f"Failed to store historical risk data: {e}")
            return data_with_id["id"]
    
    async def retrieve_historical_risk_data(self, date_range: Dict[str, str], 
                                           risk_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve historical risk data for analysis.
        
        Args:
            date_range: Dictionary with 'start_date' and 'end_date'
            risk_type: Optional risk type filter
            
        Returns:
            List of historical risk data records
        """
        try:
            query = {
                "data_type": "historical_risk",
                "date_range": date_range
            }
            if risk_type:
                query["risk_type"] = risk_type
            
            # Query external memory if enabled
            if self.enable_external_memory and self.session:
                results = await self._send_to_memory_agent("retrieve_historical_data", query)
                return results or []
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve historical risk data: {e}")
            return []
    
    async def get_agent_performance_metrics(self, agent_name: str, 
                                          time_period: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a specific risk agent.
        
        Args:
            agent_name: Name of the agent
            time_period: Time period for metrics (last 24 hours if None)
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            if time_period is None:
                time_period = timedelta(hours=24)
            
            end_time = datetime.utcnow()
            start_time = end_time - time_period
            
            query = {
                "agent_name": agent_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            # Query external memory if enabled
            if self.enable_external_memory and self.session:
                metrics = await self._send_to_memory_agent("get_agent_metrics", query)
                return metrics or {}
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to get agent performance metrics: {e}")
            return {}
    
    async def _send_to_memory_agent(self, endpoint: str, data: Dict[str, Any]) -> Any:
        """
        Send data to the External Memory Agent.
        
        Args:
            endpoint: API endpoint
            data: Data to send
            
        Returns:
            Response from memory agent
        """
        if not self.session:
            return None
        
        try:
            url = f"{self.memory_agent_url}/api/v1/{endpoint}"
            
            async with self.session.post(
                url,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("data")
                else:
                    error_text = await response.text()
                    self.logger.error(f"Memory agent request failed: {response.status} - {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout sending data to memory agent endpoint: {endpoint}")
            return None
        except Exception as e:
            self.logger.error(f"Error sending data to memory agent: {e}")
            return None
    
    def _search_local_cache(self, filters: Dict[str, Any]) -> List[RiskAnalysisRecord]:
        """Search local cache for matching records."""
        results = []
        
        for key, cached_item in self.local_cache.items():
            if not key.startswith("risk_analysis_"):
                continue
            
            if not self._is_cache_valid(cached_item["timestamp"]):
                continue
            
            record_data = cached_item["data"]
            
            # Apply filters
            matches = True
            for filter_key, filter_value in filters.items():
                if filter_key in record_data and record_data[filter_key] != filter_value:
                    matches = False
                    break
            
            if matches:
                try:
                    record = RiskAnalysisRecord(**record_data)
                    results.append(record)
                except Exception as e:
                    self.logger.warning(f"Failed to parse cached record: {e}")
        
        return results
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached data is still valid."""
        return (datetime.utcnow() - timestamp).total_seconds() < self.cache_ttl
    
    def clear_cache(self):
        """Clear the local cache."""
        self.local_cache.clear()
        self.logger.info("Local cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        valid_items = 0
        expired_items = 0
        
        for cached_item in self.local_cache.values():
            if self._is_cache_valid(cached_item["timestamp"]):
                valid_items += 1
            else:
                expired_items += 1
        
        return {
            "total_items": len(self.local_cache),
            "valid_items": valid_items,
            "expired_items": expired_items,
            "cache_ttl_seconds": self.cache_ttl
        }


# Convenience function for backward compatibility
async def record_event(agent_name: str, task: str, input_data: Dict, summary: str):
    """
    Records a memory log entry for task execution tracking and retrospective analysis.
    
    This is a simplified interface for backward compatibility.
    """
    entry = {
        "agent": agent_name,
        "task": task,
        "input": input_data,
        "summary": summary,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # This should be connected to the actual memory system in production
    logger.info(f"[RiskMemoryBridge] {entry}")
    # Future work: integrate with RiskMemoryBridge instance
