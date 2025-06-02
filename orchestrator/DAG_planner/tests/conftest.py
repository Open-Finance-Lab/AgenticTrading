import os
import sys
import pytest
import pytest_asyncio
import asyncio
from pathlib import Path

# Add project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure pytest
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as async"
    )
    # Set pytest-asyncio to auto mode
    config.option.asyncio_mode = "auto"

@pytest_asyncio.fixture
async def planner_server():
    """Fixture to create and start a test server"""
    from orchestrator.DAG_planner.tests.test_planner import MockDAGPlanner
    from orchestrator.DAG_planner.server import DAGPlannerServer
    
    planner = MockDAGPlanner()
    server = DAGPlannerServer(planner, port=8001)
    server_task = asyncio.create_task(server.start())
    
    # Give the server some time to start
    await asyncio.sleep(0.1)
    
    yield server
    
    # Cleanup
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

@pytest_asyncio.fixture
async def planner_client():
    """Fixture to create a test client"""
    from orchestrator.DAG_planner.client import DAGPlannerClient
    
    client = DAGPlannerClient(port=8001)
    await client.connect()
    yield client
    await client.disconnect() 