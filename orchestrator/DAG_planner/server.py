import asyncio
import logging
from typing import Dict, Any, Optional, Set
import uuid
from datetime import datetime, UTC
import networkx as nx

from .planner import DAGPlannerAgent
from .task import TaskDefinition, TaskStatus
from .protocols.planner_protocol import (
    PlannerMessage, PlannerMessageType,
    QueryMessage, MemoryUpdateMessage, DAGResponseMessage
)

logger = logging.getLogger(__name__)

class DAGPlannerServer:
    """MCP Server implementation for DAG Planner
    
    Handles communication with user queries, memory agent, and orchestrator.
    Manages the lifecycle of DAG planning requests.
    """
    
    def __init__(self, planner: DAGPlannerAgent, host: str = "localhost", port: int = 8000):
        self.planner = planner
        self.host = host
        self.port = port
        self.active_requests: Dict[str, asyncio.Task] = {}
        self.completed_requests: Dict[str, tuple[Dict[str, Any], Dict[str, Any]]] = {}
        self.memory_context: Dict[str, Any] = {}
        self.server = None
        
    async def start(self):
        """Start the MCP server"""
        self.server = await asyncio.start_server(
            self.handle_connection,
            self.host,
            self.port
        )
        logger.info(f"DAG Planner MCP Server started on {self.host}:{self.port}")
        
        async with self.server:
            await self.server.serve_forever()
            
    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming connections and messages"""
        try:
            while True:
                # Read message length (4 bytes)
                length_bytes = await reader.read(4)
                if not length_bytes:
                    break
                    
                length = int.from_bytes(length_bytes, 'big')
                
                # Read message content
                message_bytes = await reader.read(length)
                if not message_bytes:
                    break
                    
                # Parse and process message
                message = PlannerMessage.from_json(message_bytes.decode())
                response = await self.process_message(message)
                
                if response:
                    # Send response
                    response_bytes = response.to_json().encode()
                    writer.write(len(response_bytes).to_bytes(4, 'big'))
                    writer.write(response_bytes)
                    await writer.drain()
                    
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            
    async def process_message(self, message: PlannerMessage) -> Optional[PlannerMessage]:
        """Process incoming messages based on their type"""
        try:
            if message.message_type == PlannerMessageType.QUERY:
                response = await self.handle_query(message)
                if response is None:
                    return DAGResponseMessage(
                        dag=None,
                        metadata={"error": "Failed to process query"},
                        correlation_id=message.correlation_id or str(uuid.uuid4())
                    )
                return response
            elif message.message_type == PlannerMessageType.MEMORY_UPDATE:
                await self.handle_memory_update(message)
                return DAGResponseMessage(
                    dag=None,
                    metadata={"status": "Memory updated successfully"},
                    correlation_id=message.correlation_id or str(uuid.uuid4())
                )
            elif message.message_type == PlannerMessageType.DAG_REQUEST:
                response = await self.handle_dag_request(message)
                if response is None:
                    return DAGResponseMessage(
                        dag=None,
                        metadata={"error": "DAG request not found"},
                        correlation_id=message.correlation_id or str(uuid.uuid4())
                    )
                return response
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                return DAGResponseMessage(
                    dag=None,
                    metadata={"error": f"Unknown message type: {message.message_type}"},
                    correlation_id=message.correlation_id or str(uuid.uuid4())
                )
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return DAGResponseMessage(
                dag=None,
                metadata={"error": str(e)},
                correlation_id=message.correlation_id or str(uuid.uuid4())
            )
            
    async def handle_query(self, message: QueryMessage) -> Optional[DAGResponseMessage]:
        """Handle user query for DAG planning"""
        correlation_id = message.correlation_id or str(uuid.uuid4())
        logger.info(f"Handling query with correlation_id: {correlation_id}")
        
        try:
            # Create planning task
            planning_task = asyncio.create_task(
                self._plan_dag(
                    message.payload["query"],
                    message.payload["context"],
                    correlation_id
                )
            )
            
            self.active_requests[correlation_id] = planning_task
            
            # Wait for planning task to complete
            dag, metadata = await planning_task
            
            if dag is None:
                logger.error(f"Planning failed for correlation_id: {correlation_id}")
                return DAGResponseMessage(
                    dag=None,
                    metadata={"error": "Planning failed"},
                    correlation_id=correlation_id
                )
                
            # Store the completed DAG
            self.completed_requests[correlation_id] = (dag, metadata)
            logger.info(f"Successfully planned DAG for correlation_id: {correlation_id}")
            return DAGResponseMessage(
                dag=dag,
                metadata=metadata,
                correlation_id=correlation_id
            )
            
        except Exception as e:
            logger.error(f"Error planning DAG for correlation_id {correlation_id}: {e}")
            return DAGResponseMessage(
                dag=None,
                metadata={"error": str(e)},
                correlation_id=correlation_id
            )
        finally:
            self.active_requests.pop(correlation_id, None)
            
    async def handle_memory_update(self, message: MemoryUpdateMessage) -> None:
        """Handle memory agent updates"""
        self.memory_context.update(message.payload["memory_data"])
        
    async def handle_dag_request(self, message: PlannerMessage) -> Optional[DAGResponseMessage]:
        """Handle DAG execution request from orchestrator"""
        correlation_id = message.correlation_id
        logger.info(f"Handling DAG request for correlation_id: {correlation_id}")
        
        # First check active requests
        if correlation_id in self.active_requests:
            task = self.active_requests[correlation_id]
            try:
                dag, metadata = await task
                if dag is None:
                    logger.error(f"No DAG found for correlation_id: {correlation_id}")
                    return DAGResponseMessage(
                        dag=None,
                        metadata={"error": "DAG not found"},
                        correlation_id=correlation_id
                    )
                # Store the completed DAG
                self.completed_requests[correlation_id] = (dag, metadata)
                return DAGResponseMessage(
                    dag=dag,
                    metadata=metadata,
                    correlation_id=correlation_id
                )
            except Exception as e:
                logger.error(f"Error retrieving DAG for correlation_id {correlation_id}: {e}")
                return DAGResponseMessage(
                    dag=None,
                    metadata={"error": str(e)},
                    correlation_id=correlation_id
                )
        # Then check completed requests
        elif correlation_id in self.completed_requests:
            dag, metadata = self.completed_requests[correlation_id]
            logger.info(f"Retrieved completed DAG for correlation_id: {correlation_id}")
            return DAGResponseMessage(
                dag=dag,
                metadata=metadata,
                correlation_id=correlation_id
            )
        else:
            logger.warning(f"No request found for correlation_id: {correlation_id}")
            return DAGResponseMessage(
                dag=None,
                metadata={"error": "No request found"},
                correlation_id=correlation_id
            )
        
    async def _plan_dag(
        self,
        query: str,
        context: Dict[str, Any],
        correlation_id: str
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Internal method to plan DAG from query"""
        try:
            # Merge context with memory context
            full_context = {**self.memory_context, **context}
            logger.debug(f"Planning DAG with context: {full_context}")
            
            # Plan DAG using the planner
            dag = await self.planner.plan(query)
            
            if dag is None or not isinstance(dag, nx.DiGraph):
                logger.error(f"Planner returned invalid DAG for query: {query}")
                return None, None
                
            # Convert DAG to serializable format
            dag_dict = {
                "nodes": [
                    {
                        "id": node,
                        "type": self.planner.task_registry[node].agent_type,
                        "parameters": self.planner.task_registry[node].parameters,
                        "dependencies": self.planner.task_registry[node].dependencies
                    }
                    for node in dag.nodes()
                ],
                "edges": [
                    {"source": u, "target": v}
                    for u, v in dag.edges()
                ]
            }
            
            metadata = {
                "correlation_id": correlation_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "context": full_context
            }
            
            logger.debug(f"Successfully created DAG with {len(dag_dict['nodes'])} nodes")
            return dag_dict, metadata
            
        except Exception as e:
            logger.error(f"Error in _plan_dag: {e}")
            return None, None 