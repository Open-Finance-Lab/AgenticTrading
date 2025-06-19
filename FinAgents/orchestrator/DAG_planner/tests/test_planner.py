import asyncio
import pytest
import json
from datetime import datetime
from typing import Dict, Any
import networkx as nx

from orchestrator.DAG_planner.planner import DAGPlannerAgent, TaskNode
from orchestrator.DAG_planner.task import TaskDefinition, TaskStatus, AgentType
from orchestrator.DAG_planner.protocols.planner_protocol import (
    PlannerMessage, PlannerMessageType,
    QueryMessage, MemoryUpdateMessage, DAGResponseMessage
)
from orchestrator.DAG_planner.server import DAGPlannerServer
from orchestrator.DAG_planner.client import DAGPlannerClient

# Mock DAG Planner implementation for testing
class MockDAGPlanner(DAGPlannerAgent):
    """Mock implementation of DAGPlannerAgent for testing"""
    
    async def plan(self, query: str) -> nx.DiGraph:
        """Mock implementation of plan method"""
        # Create a simple DAG for testing
        dag = nx.DiGraph()
        
        # Add test nodes
        nodes = [
            TaskNode(
                task_id="data_task",
                agent_type=AgentType.DATA.value,
                parameters={"symbol": "AAPL", "timeframe": "1d"},
                dependencies=[]
            ),
            TaskNode(
                task_id="alpha_task",
                agent_type=AgentType.ALPHA.value,
                parameters={"model": "momentum"},
                dependencies=["data_task"]
            ),
            TaskNode(
                task_id="risk_task",
                agent_type=AgentType.RISK.value,
                parameters={"max_drawdown": 0.1},
                dependencies=["alpha_task"]
            )
        ]
        
        # Add nodes to DAG
        for node in nodes:
            self.add_task(node)
            
        return self.dag

@pytest.fixture
async def planner_server():
    """Fixture to create and start a test server"""
    planner = MockDAGPlanner()
    server = DAGPlannerServer(planner, port=8001)  # Use different port for testing
    server_task = asyncio.create_task(server.start())
    
    # Give the server time to start
    await asyncio.sleep(0.1)
    
    yield server
    
    # Cleanup
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

@pytest.fixture
async def planner_client():
    """Fixture to create a test client"""
    client = DAGPlannerClient(port=8001)  # Use same port as server
    await client.connect()
    yield client
    await client.disconnect()

class TestDAGPlannerProtocol:
    """Test suite for protocol messages"""
    
    def test_query_message_serialization(self):
        """Test serialization and deserialization of QueryMessage"""
        original = QueryMessage(
            query="Test query",
            context={"test": "context"},
            correlation_id="test-123"
        )
        
        # Convert to JSON and back
        json_str = original.to_json()
        restored = PlannerMessage.from_json(json_str)
        
        assert isinstance(restored, PlannerMessage)
        assert restored.message_type == PlannerMessageType.QUERY
        assert restored.payload["query"] == "Test query"
        assert restored.payload["context"] == {"test": "context"}
        assert restored.correlation_id == "test-123"
        
    def test_memory_update_message(self):
        """Test MemoryUpdateMessage creation and serialization"""
        memory_data = {
            "market_data": {"AAPL": 150.0},
            "risk_metrics": {"volatility": 0.2}
        }
        
        message = MemoryUpdateMessage(memory_data=memory_data)
        assert message.message_type == PlannerMessageType.MEMORY_UPDATE
        assert message.payload["memory_data"] == memory_data

class TestDAGPlannerServer:
    """Test suite for DAG Planner Server"""
    
    @pytest.mark.asyncio
    async def test_server_handles_query(self, planner_server, planner_client):
        """Test server handling of query messages"""
        # Send a test query
        dag = await planner_client.plan_dag(
            query="Test trading strategy",
            context={"market": "US"}
        )
        
        assert dag is not None
        assert "nodes" in dag
        assert "edges" in dag
        
        # Verify DAG structure
        nodes = dag["nodes"]
        assert len(nodes) == 3
        assert any(node["type"] == AgentType.DATA.value for node in nodes)
        assert any(node["type"] == AgentType.ALPHA.value for node in nodes)
        assert any(node["type"] == AgentType.RISK.value for node in nodes)
        
    @pytest.mark.asyncio
    async def test_server_handles_memory_update(self, planner_server, planner_client):
        """Test server handling of memory updates"""
        memory_data = {
            "market_data": {"AAPL": 150.0},
            "risk_metrics": {"volatility": 0.2}
        }
        
        # Send memory update
        await planner_client.update_memory(memory_data)
        
        # Send a query to verify memory context is used
        dag = await planner_client.plan_dag(
            query="Test with memory",
            context={}
        )
        
        assert dag is not None
        # Note: In a real implementation, we would verify that the memory
        # context influenced the DAG planning

class TestDAGPlannerClient:
    """Test suite for DAG Planner Client"""
    
    @pytest.mark.asyncio
    async def test_client_connection(self, planner_server):
        """Test client connection and disconnection"""
        client = DAGPlannerClient(port=8001)
        
        # Test connection
        await client.connect()
        assert client.writer is not None
        assert client.reader is not None
        
        # Test disconnection
        await client.disconnect()
        assert client.writer is None
        assert client.reader is None
        
    @pytest.mark.asyncio
    async def test_client_request_dag(self, planner_server, planner_client):
        """Test client requesting a previously planned DAG"""
        # First plan a DAG
        message = QueryMessage(
            query="Test strategy",
            context={},
            correlation_id="test-request-123"  # Use a known correlation ID
        )
        
        # Send the query directly to get the correlation ID
        response = await planner_client.send_message(message)
        assert response is not None
        assert isinstance(response, DAGResponseMessage)
        correlation_id = response.correlation_id
        
        # Verify we got a valid DAG
        dag = response.payload["dag"]
        assert dag is not None
        
        # Request the same DAG using the correlation ID
        requested_dag = await planner_client.request_dag(correlation_id)
        assert requested_dag is not None
        assert requested_dag == dag  # Verify we got the same DAG

@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    # Create and start server
    planner = MockDAGPlanner()
    server = DAGPlannerServer(planner, port=8002)
    server_task = asyncio.create_task(server.start())
    await asyncio.sleep(0.1)
    
    try:
        # Create client
        client = DAGPlannerClient(port=8002)
        await client.connect()
        
        # Update memory
        await client.update_memory({
            "market_data": {"AAPL": 150.0},
            "risk_metrics": {"volatility": 0.2}
        })
        
        # Plan DAG
        dag = await client.plan_dag(
            query="End-to-end test strategy",
            context={"market": "US", "timeframe": "1d"}
        )
        
        assert dag is not None
        assert "nodes" in dag
        assert "edges" in dag
        
        # Verify DAG structure
        nodes = dag["nodes"]
        edges = dag["edges"]
        
        # Check node types
        node_types = {node["type"] for node in nodes}
        assert AgentType.DATA.value in node_types
        assert AgentType.ALPHA.value in node_types
        assert AgentType.RISK.value in node_types
        
        # Check dependencies
        assert len(edges) == 2  # Should have two edges in our mock DAG
        edge_sources = {edge["source"] for edge in edges}
        edge_targets = {edge["target"] for edge in edges}
        
        assert "data_task" in edge_sources
        assert "alpha_task" in edge_targets
        assert "risk_task" in edge_targets
        
    finally:
        # Cleanup
        await client.disconnect()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 