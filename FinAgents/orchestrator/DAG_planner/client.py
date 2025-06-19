import asyncio
import logging
from typing import Dict, Any, Optional
import json
from datetime import datetime, UTC

from .protocols.planner_protocol import (
    PlannerMessage, PlannerMessageType,
    QueryMessage, MemoryUpdateMessage, DAGResponseMessage
)

logger = logging.getLogger(__name__)

class DAGPlannerClient:
    """Client for interacting with DAG Planner MCP Server"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        
    async def connect(self):
        """Connect to the DAG Planner server"""
        self.reader, self.writer = await asyncio.open_connection(
            self.host,
            self.port
        )
        logger.info(f"Connected to DAG Planner server at {self.host}:{self.port}")
        
    async def disconnect(self):
        """Disconnect from the server"""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            self.reader = None
            self.writer = None
            
    async def send_message(self, message: PlannerMessage) -> Optional[PlannerMessage]:
        """Send a message to the server and wait for response"""
        if not self.writer:
            logger.debug("No active connection, attempting to connect...")
            await self.connect()
            
        try:
            # Convert message to bytes
            message_bytes = message.to_json().encode()
            logger.debug(f"Sending message of type: {message.message_type}")
            
            # Send message length and content
            self.writer.write(len(message_bytes).to_bytes(4, 'big'))
            self.writer.write(message_bytes)
            await self.writer.drain()
            
            # Read response length
            length_bytes = await self.reader.read(4)
            if not length_bytes:
                logger.error("Connection closed by server while reading response length")
                await self.disconnect()
                return None
                
            length = int.from_bytes(length_bytes, 'big')
            
            # Read response content
            response_bytes = await self.reader.read(length)
            if not response_bytes:
                logger.error("Connection closed by server while reading response content")
                await self.disconnect()
                return None
                
            # Parse response
            response = PlannerMessage.from_json(response_bytes.decode())
            logger.debug(f"Received response of type: {response.message_type}")
            return response
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            await self.disconnect()
            return None
            
    async def plan_dag(self, query: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a query to plan a DAG"""
        logger.info(f"Planning DAG for query: {query}")
        message = QueryMessage(query=query, context=context)
        response = await self.send_message(message)
        
        if response is None:
            logger.error("No response received from server")
            return None
            
        if not isinstance(response, DAGResponseMessage):
            logger.error(f"Unexpected response type: {type(response)}")
            return None
            
        logger.info("Successfully received DAG from server")
        return response.payload["dag"]
        
    async def update_memory(self, memory_data: Dict[str, Any]) -> None:
        """Send memory update to the server"""
        logger.info("Sending memory update to server")
        message = MemoryUpdateMessage(memory_data=memory_data)
        response = await self.send_message(message)
        
        if response is None:
            logger.error("Failed to send memory update")
        else:
            logger.info("Successfully sent memory update")
        
    async def request_dag(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """Request a previously planned DAG"""
        logger.info(f"Requesting DAG with correlation_id: {correlation_id}")
        message = PlannerMessage(
            message_type=PlannerMessageType.DAG_REQUEST,
            timestamp=datetime.now(UTC),
            payload={},
            source="client",
            correlation_id=correlation_id
        )
        
        response = await self.send_message(message)
        if response is None:
            logger.error("No response received for DAG request")
            return None
            
        if not isinstance(response, DAGResponseMessage):
            logger.error(f"Unexpected response type: {type(response)}")
            return None
            
        logger.info("Successfully received requested DAG")
        return response.payload["dag"] 