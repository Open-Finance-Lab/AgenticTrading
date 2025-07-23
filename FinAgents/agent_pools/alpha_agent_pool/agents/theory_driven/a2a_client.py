"""
A2A (Agent-to-Agent) Protocol Client for Alpha Agent Pool

This module implements the official A2A protocol integration for the Alpha Agent Pool,
enabling direct communication between alpha agents and memory agents using the
official A2A Python toolkit. This architecture provides loosely-coupled memory
integration at the agent pool level.

The A2A protocol enables:
- Official A2A protocol compliance
- Agent pool level memory coordination
- Direct alpha agent communication
- Standardized task-based memory operations

Author: FinAgent Team
License: Open Source
"""

import json
import asyncio
import httpx
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid
import logging
from pathlib import Path

# Official A2A toolkit imports
try:
    from a2a.client.client import A2AClient, A2ACardResolver
    from a2a.types import (
        SendMessageRequest, 
        SendMessageResponse,
        SendStreamingMessageRequest,
        GetTaskRequest,
        GetTaskResponse,
        CancelTaskRequest
    )
    from a2a.client.errors import (
        A2AClientHTTPError,
        A2AClientJSONError,
        A2AClientTimeoutError
    )
    A2A_AVAILABLE = True
except ImportError:
    # Fallback for development/testing when A2A toolkit is not available
    A2A_AVAILABLE = False
    A2AClient = None
    logger = logging.getLogger(__name__)
    logger.warning("Official A2A toolkit not available, using fallback implementation")

logger = logging.getLogger(__name__)


class A2AProtocolError(Exception):
    """Exception raised for A2A protocol communication errors"""
    pass


class AlphaAgentA2AClient:
    """
    Alpha Agent Pool A2A Protocol Client based on official toolkit.
    
    This client implements the official A2A protocol for communication between
    the Alpha Agent Pool and memory agents. It provides a pool-level memory
    interface that coordinates with individual alpha agents.
    """
    
    def __init__(self, 
                 agent_pool_id: str,
                 memory_agent_base_url: str = "http://127.0.0.1:8010",
                 timeout: float = 30.0,
                 max_retries: int = 3):
        """
        Initialize Alpha Agent Pool A2A client.
        
        Args:
            agent_pool_id: Unique identifier for the alpha agent pool
            memory_agent_base_url: Base URL of the memory agent
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.agent_pool_id = agent_pool_id
        self.memory_agent_base_url = memory_agent_base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session_id = str(uuid.uuid4())
        
        # Track connection status
        self.is_connected = False
        self.last_heartbeat = None
        
        # Initialize HTTP client
        self.http_client = None
        self.a2a_client = None
        
        logger.info(f"Alpha Agent Pool A2A client initialized for pool {agent_pool_id}")
        
        # Initialize HTTP client for immediate use
        asyncio.create_task(self._init_http_client())
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def _init_http_client(self):
        """Initialize HTTP client for fallback communication"""
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=self.timeout)
    
    async def connect(self) -> bool:
        """
        Establish connection to memory agent using official A2A protocol.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Ensure HTTP client is initialized
            if not self.http_client:
                self.http_client = httpx.AsyncClient(timeout=self.timeout)
            
            if A2A_AVAILABLE:
                # Use official A2A toolkit
                self.a2a_client = await A2AClient.get_client_from_agent_card_url(
                    httpx_client=self.http_client,
                    base_url=self.memory_agent_base_url
                )
                logger.info(f"Connected using official A2A protocol for pool {self.agent_pool_id}")
            else:
                # Fallback to direct HTTP communication
                logger.warning(f"Using fallback HTTP communication for pool {self.agent_pool_id}")
            
            # Test connection
            await self._test_connection()
            
            self.is_connected = True
            self.last_heartbeat = datetime.utcnow()
            logger.info(f"A2A connection established for alpha agent pool {self.agent_pool_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish A2A connection: {e}")
            self.is_connected = False
            if self.http_client:
                await self.http_client.aclose()
            return False
    
    async def disconnect(self):
        """Gracefully disconnect from memory agent"""
        self.is_connected = False
        if self.http_client:
            await self.http_client.aclose()
        logger.info(f"A2A connection closed for alpha agent pool {self.agent_pool_id}")
    
    async def _test_connection(self):
        """Test connection to memory agent"""
        if A2A_AVAILABLE and self.a2a_client:
            # Use official A2A protocol for testing
            request = SendMessageRequest(
                messages=[{
                    "role": "user", 
                    "content": "Health check from alpha agent pool"
                }],
                context={"test": True, "agent_pool_id": self.agent_pool_id}
            )
            response = await self.a2a_client.send_message(request)
            logger.debug(f"A2A health check response: {response}")
        else:
            # Fallback HTTP health check using A2A endpoint
            response = await self.http_client.get(f"{self.memory_agent_base_url}/a2a/health")
            response.raise_for_status()
    
    async def store_alpha_signal_event(self, 
                                     agent_id: str,
                                     signal: str,
                                     confidence: float,
                                     symbol: str,
                                     reasoning: str,
                                     market_context: Dict[str, Any],
                                     correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Store alpha signal event using A2A protocol.
        
        Args:
            agent_id: ID of the specific alpha agent
            signal: Trading signal (BUY/SELL/HOLD)
            confidence: Signal confidence score (0.0-1.0)
            symbol: Trading symbol
            reasoning: Human-readable reasoning
            market_context: Market context and analysis data
            correlation_id: Optional correlation ID for linking events
            
        Returns:
            Dict containing storage confirmation and response
        """
        event_data = {
            "agent_pool_id": self.agent_pool_id,
            "agent_id": agent_id,
            "signal": signal,
            "confidence": confidence,
            "symbol": symbol,
            "reasoning": reasoning,
            "market_context": market_context,
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "ALPHA_SIGNAL"
        }
        
        task_content = {
            "action": "store_memory",
            "data": event_data,
            "correlation_id": correlation_id or str(uuid.uuid4())
        }
        
        if A2A_AVAILABLE and self.a2a_client:
            # Use official A2A protocol
            request = SendMessageRequest(
                messages=[{
                    "role": "assistant",
                    "content": json.dumps(task_content)
                }],
                context={
                    "agent_pool_id": self.agent_pool_id,
                    "task_type": "memory_storage",
                    "session_id": self.session_id
                }
            )
            
            response = await self.a2a_client.send_message(request)
            return self._parse_a2a_response(response)
        else:
            # Fallback to direct HTTP
            return await self._fallback_http_request("store_alpha_signal", task_content)
    
    async def store_strategy_performance(self,
                                       agent_id: str,
                                       strategy_id: str,
                                       performance_metrics: Dict[str, Any],
                                       correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Store strategy performance metrics using A2A protocol.
        
        Args:
            agent_id: ID of the alpha agent
            strategy_id: Unique strategy identifier
            performance_metrics: Dictionary of performance metrics
            correlation_id: Optional correlation ID
            
        Returns:
            Dict containing storage confirmation
        """
        performance_data = {
            "agent_pool_id": self.agent_pool_id,
            "agent_id": agent_id,
            "strategy_id": strategy_id,
            "performance_metrics": performance_metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "STRATEGY_PERFORMANCE"
        }
        
        task_content = {
            "action": "store_performance",
            "data": performance_data,
            "correlation_id": correlation_id or str(uuid.uuid4())
        }
        
        if A2A_AVAILABLE and self.a2a_client:
            request = SendMessageRequest(
                messages=[{
                    "role": "assistant",
                    "content": json.dumps(task_content)
                }],
                context={
                    "agent_pool_id": self.agent_pool_id,
                    "task_type": "performance_storage",
                    "session_id": self.session_id
                }
            )
            
            response = await self.a2a_client.send_message(request)
            return self._parse_a2a_response(response)
        else:
            return await self._fallback_http_request("store_performance", task_content)
    
    async def retrieve_strategy_insights(self, 
                                       search_query: str,
                                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve strategy insights using A2A protocol.
        
        Args:
            search_query: Natural language search query
            limit: Maximum number of results to return
            
        Returns:
            List of strategy insights and similar patterns
        """
        query_data = {
            "search_query": search_query,
            "limit": limit,
            "agent_pool_id": self.agent_pool_id,
            "session_id": self.session_id
        }
        
        task_content = {
            "action": "retrieve_insights",
            "data": query_data,
            "correlation_id": str(uuid.uuid4())
        }
        
        if A2A_AVAILABLE and self.a2a_client:
            request = SendMessageRequest(
                messages=[{
                    "role": "user",
                    "content": json.dumps(task_content)
                }],
                context={
                    "agent_pool_id": self.agent_pool_id,
                    "task_type": "memory_retrieval",
                    "session_id": self.session_id
                }
            )
            
            response = await self.a2a_client.send_message(request)
            parsed_response = self._parse_a2a_response(response)
            return parsed_response.get("insights", [])
        else:
            response = await self._fallback_http_request("retrieve_insights", task_content)
            return response.get("insights", [])
    
    async def store_learning_feedback(self,
                                    agent_id: str,
                                    feedback_type: str,
                                    feedback_data: Dict[str, Any],
                                    correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Store learning and adaptation feedback using A2A protocol.
        
        Args:
            agent_id: ID of the alpha agent
            feedback_type: Type of feedback (RL_UPDATE, WINDOW_SELECTION, etc.)
            feedback_data: Detailed feedback information
            correlation_id: Optional correlation ID
            
        Returns:
            Dict containing storage confirmation
        """
        learning_data = {
            "agent_pool_id": self.agent_pool_id,
            "agent_id": agent_id,
            "feedback_type": feedback_type,
            "feedback_data": feedback_data,
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "LEARNING_FEEDBACK"
        }
        
        task_content = {
            "action": "store_learning",
            "data": learning_data,
            "correlation_id": correlation_id or str(uuid.uuid4())
        }
        
        if A2A_AVAILABLE and self.a2a_client:
            request = SendMessageRequest(
                messages=[{
                    "role": "assistant",
                    "content": json.dumps(task_content)
                }],
                context={
                    "agent_pool_id": self.agent_pool_id,
                    "task_type": "learning_storage",
                    "session_id": self.session_id
                }
            )
            
            response = await self.a2a_client.send_message(request)
            return self._parse_a2a_response(response)
        else:
            return await self._fallback_http_request("store_learning", task_content)
    
    def _parse_a2a_response(self, response) -> Dict[str, Any]:
        """Parse A2A protocol response"""
        if hasattr(response, 'model_dump'):
            response_data = response.model_dump()
        elif isinstance(response, dict):
            response_data = response
        else:
            response_data = {"raw_response": str(response)}
        
        return response_data
    
    async def _fallback_http_request(self, 
                                   action: str, 
                                   data: Dict[str, Any],
                                   retry_count: int = 0) -> Dict[str, Any]:
        """
        Fallback HTTP request when A2A toolkit is not available.
        
        Args:
            action: Action type
            data: Request data
            retry_count: Current retry attempt number
            
        Returns:
            Dict containing response
        """
        # Map actions to A2A endpoints
        endpoint_mapping = {
            "store_alpha_signal": "/a2a/signals/transmit",
            "store_performance": "/a2a/strategies/share", 
            "retrieve_insights": "/a2a/data/query",
            "store_learning": "/a2a/signals/transmit"
        }
        
        endpoint = endpoint_mapping.get(action, "/a2a/signals/transmit")
        
        # Convert data to A2A format based on action
        if action == "store_alpha_signal":
            a2a_payload = {
                "sender_agent_id": data.get("data", {}).get("agent_id", "unknown"),
                "signal_type": "alpha",
                "signal_data": data.get("data", {}),
                "timestamp": data.get("timestamp"),
                "priority": "normal",
                "metadata": {
                    "agent_pool_id": self.agent_pool_id,
                    "correlation_id": data.get("correlation_id")
                }
            }
        elif action == "store_performance":
            a2a_payload = {
                "sender_agent_id": data.get("data", {}).get("agent_id", "unknown"),
                "strategy_type": "performance",
                "strategy_data": data.get("data", {}),
                "performance_metrics": data.get("data", {}).get("performance_metrics", {}),
                "sharing_permission": "read"
            }
        elif action == "retrieve_insights":
            a2a_payload = {
                "requesting_agent_id": self.agent_pool_id,
                "query_type": "strategies",
                "filters": data.get("data", {}),
                "limit": 10
            }
        else:
            # Default to signal format
            a2a_payload = {
                "sender_agent_id": data.get("data", {}).get("agent_id", "unknown"),
                "signal_type": "unknown",
                "signal_data": data.get("data", {}),
                "priority": "normal"
            }
        
        try:
            response = await self.http_client.post(
                f"{self.memory_agent_base_url}{endpoint}",
                json=a2a_payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Agent-Pool-ID": self.agent_pool_id,
                    "X-Session-ID": self.session_id,
                    "X-Protocol": "A2A-Fallback"
                }
            )
            
            if response.status_code == 200:
                self.last_heartbeat = datetime.utcnow()
                return response.json()
            else:
                raise A2AProtocolError(f"HTTP {response.status_code}: {response.text}")
                
        except httpx.TimeoutException:
            if retry_count < self.max_retries:
                logger.warning(f"Fallback request timeout, retrying ({retry_count + 1}/{self.max_retries})")
                await asyncio.sleep(1.0 * (retry_count + 1))
                return await self._fallback_http_request(action, data, retry_count + 1)
            else:
                raise A2AProtocolError(f"Request timeout after {self.max_retries} retries")
                
        except Exception as e:
            if retry_count < self.max_retries:
                logger.warning(f"Fallback request failed, retrying ({retry_count + 1}/{self.max_retries}): {e}")
                await asyncio.sleep(1.0 * (retry_count + 1))
                return await self._fallback_http_request(action, data, retry_count + 1)
            else:
                raise A2AProtocolError(f"Request failed after {self.max_retries} retries: {e}")
    
    async def healthcheck(self) -> bool:
        """
        Perform health check on A2A connection.
        
        Returns:
            bool: True if connection is healthy
        """
        try:
            await self._test_connection()
            self.last_heartbeat = datetime.utcnow()
            return True
        except Exception as e:
            logger.warning(f"A2A healthcheck failed: {e}")
            self.is_connected = False
            return False


# Factory function for creating Alpha Agent Pool A2A clients
async def create_alpha_pool_a2a_client(agent_pool_id: str, 
                                memory_url: str = "http://127.0.0.1:8010") -> AlphaAgentA2AClient:
    """
    Factory function to create Alpha Agent Pool A2A client.
    
    Args:
        agent_pool_id: Unique alpha agent pool identifier
        memory_url: Memory agent base URL
        
    Returns:
        Configured AlphaAgentA2AClient instance
    """
    client = AlphaAgentA2AClient(agent_pool_id=agent_pool_id, memory_agent_base_url=memory_url)
    # Ensure HTTP client is initialized for immediate use
    await client._init_http_client()
    return client
