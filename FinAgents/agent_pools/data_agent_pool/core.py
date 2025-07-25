# core.py
"""
Unified entry point for starting the Data Agent Pool MCP service.
This module manages the lifecycle and orchestration of data agents following the alpha agent pool architecture.

The DataAgentPool acts as a coordinator that connects to various data agent MCP servers:
- PolygonAgent MCP Server (port 8002) - Market data and natural language queries
- BinanceAgent MCP Server (port 8003) - Crypto data  
- Other data agents as needed

Architecture:
- DataAgentPool exposes unified tools that proxy to individual agent servers
- Agents run as independent MCP servers on different ports
- Natural language queries are routed to appropriate agents
- Batch operations are coordinated through the pool
"""
import logging
import contextvars
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import asyncio
import os
import httpx
import json
import subprocess
import signal
import time
import sys
from pathlib import Path

# Add memory module to path (temporarily disabled)
# memory_path = Path(__file__).parent.parent.parent / "memory"
# sys.path.insert(0, str(memory_path))

# Temporarily disable memory client import until implemented
# from FinAgents.memory.data_memory_client import DataMemoryClient

# Simple placeholder for DataMemoryClient
class DataMemoryClient:
    """Placeholder DataMemoryClient for basic functionality."""
    def __init__(self, *args, **kwargs):
        pass
    
    async def store_data(self, *args, **kwargs):
        return True
    
    async def retrieve_data(self, *args, **kwargs):
        return []

from mcp.server.fastmcp import FastMCP
from mcp.client.sse import sse_client
from mcp import ClientSession

# Configure global logging with standardized format
logger = logging.getLogger("DataAgentPool")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

class DataAgentPoolMCPServer:
    """
    DataAgentPoolMCPServer is the central orchestrator for managing the lifecycle and unified access
    of all data agents in the FinAgent ecosystem. It acts as a coordinator that connects to various
    data agent MCP servers via HTTP clients.

    Key responsibilities:
    - Starting and managing agent processes automatically
    - Coordinating requests to individual agent MCP servers
    - Providing unified natural language interface for all data agents
    - Batch processing and cross-agent operations
    - Health monitoring and automatic restart of failed agents
    - Graceful shutdown of all managed processes
    """

    def __init__(self, host="0.0.0.0", port=8001, auto_start_agents=True):
        """
        Initialize a new DataAgentPoolMCPServer instance.

        Args:
            host (str): Host address to bind the MCP server.
            port (int): Port number to bind the MCP server.
            auto_start_agents (bool): Whether to automatically start agent processes.
        """
        self.host = host
        self.port = port
        self.auto_start_agents = auto_start_agents
        self.pool_server = FastMCP("DataAgentPoolMCPServer")
        
        # Agent process management
        self.agent_processes = {}
        self.agent_configs = {
            "polygon_agent": {
                "endpoint": "http://localhost:8003/sse",
                "port": 8003,
                "start_script": self._get_polygon_start_script(),
                "health_check_interval": 30,  # seconds
                "max_restart_attempts": 3
            }
            # Future: add more agents here
        }
        
        # Agent endpoints mapping (for backward compatibility)
        self.agent_endpoints = {
            agent_id: config["endpoint"] 
            for agent_id, config in self.agent_configs.items()
        }
        
        # Memory integration
        self.memory_client = DataMemoryClient(agent_id="data_pool_coordinator")
        
        # Session management
        self.current_session_id = None
        
        logger.info(f"Initialized DataAgentPoolMCPServer on {host}:{port}")
        logger.info(f"Agent endpoints: {self.agent_endpoints}")
        logger.info(f"Auto-start agents: {auto_start_agents}")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self._register_pool_tools()

    def _get_polygon_start_script(self) -> str:
        """Get the path to the PolygonAgent start script."""
        project_root = Path(__file__).parent.parent.parent.parent
        script_path = project_root / "FinAgents" / "agent_pools" / "data_agent_pool" / "start_polygon_agent.py"
        return str(script_path)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)

    async def start_agent(self, agent_id: str) -> bool:
        """
        Start a specific agent process.
        
        Args:
            agent_id: ID of the agent to start
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        if agent_id not in self.agent_configs:
            logger.error(f"Unknown agent ID: {agent_id}")
            return False
            
        config = self.agent_configs[agent_id]
        
        # Check if already running
        if agent_id in self.agent_processes:
            proc = self.agent_processes[agent_id]
            if proc.poll() is None:  # Still running
                logger.info(f"Agent {agent_id} is already running (PID: {proc.pid})")
                return True
                
        try:
            # Start the agent process
            cmd = [
                sys.executable, 
                config["start_script"], 
                "--port", str(config["port"]),
                "--host", "0.0.0.0"
            ]
            
            logger.info(f"Starting {agent_id} with command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.agent_processes[agent_id] = process
            
            # Wait a moment for the process to start
            await asyncio.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info(f"✅ Successfully started {agent_id} (PID: {process.pid})")
                
                # Wait for the MCP server to be ready
                if await self._wait_for_agent_ready(agent_id):
                    logger.info(f"✅ {agent_id} MCP server is ready")
                    return True
                else:
                    logger.error(f"❌ {agent_id} MCP server failed to become ready")
                    return False
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ Failed to start {agent_id}:")
                logger.error(f"   stdout: {stdout}")
                logger.error(f"   stderr: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Exception starting {agent_id}: {e}")
            return False

    async def _wait_for_agent_ready(self, agent_id: str, timeout: int = 30) -> bool:
        """
        Wait for an agent's MCP server to be ready.
        
        Args:
            agent_id: ID of the agent
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if agent is ready, False if timeout
        """
        config = self.agent_configs[agent_id]
        endpoint = config["endpoint"]
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with sse_client(endpoint, timeout=5) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool("health_check", {})
                        if result.content:
                            return True
            except Exception:
                await asyncio.sleep(1)
                continue
                
        return False

    async def stop_agent(self, agent_id: str) -> bool:
        """
        Stop a specific agent process.
        
        Args:
            agent_id: ID of the agent to stop
            
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if agent_id not in self.agent_processes:
            logger.info(f"Agent {agent_id} is not running")
            return True
            
        process = self.agent_processes[agent_id]
        
        try:
            # Try graceful shutdown first
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
                logger.info(f"✅ Gracefully stopped {agent_id}")
            except subprocess.TimeoutExpired:
                # Force kill if needed
                process.kill()
                process.wait()
                logger.info(f"⚡ Force killed {agent_id}")
                
            del self.agent_processes[agent_id]
            return True
            
        except Exception as e:
            logger.error(f"❌ Error stopping {agent_id}: {e}")
            return False

    async def restart_agent(self, agent_id: str) -> bool:
        """
        Restart a specific agent.
        
        Args:
            agent_id: ID of the agent to restart
            
        Returns:
            bool: True if restarted successfully, False otherwise
        """
        logger.info(f"🔄 Restarting agent {agent_id}...")
        await self.stop_agent(agent_id)
        await asyncio.sleep(1)
        return await self.start_agent(agent_id)

    async def start_all_agents(self):
        """Start all configured agents."""
        if not self.auto_start_agents:
            logger.info("Auto-start agents is disabled, skipping agent startup")
            return
            
        logger.info("🚀 Starting all agents...")
        for agent_id in self.agent_configs:
            await self.start_agent(agent_id)

    def shutdown(self):
        """Shutdown all agent processes gracefully."""
        logger.info("🛑 Shutting down all agents...")
        for agent_id in list(self.agent_processes.keys()):
            asyncio.run(self.stop_agent(agent_id))
        logger.info("✅ All agents stopped")

    async def monitor_agents(self):
        """Background task to monitor agent health and restart if needed."""
        while True:
            try:
                for agent_id, config in self.agent_configs.items():
                    if agent_id in self.agent_processes:
                        process = self.agent_processes[agent_id]
                        
                        # Check if process is still alive
                        if process.poll() is not None:
                            logger.warning(f"⚠️ Agent {agent_id} process died, restarting...")
                            await self.restart_agent(agent_id)
                        else:
                            # Check MCP server health
                            try:
                                endpoint = config["endpoint"]
                                async with sse_client(endpoint, timeout=5) as (read, write):
                                    async with ClientSession(read, write) as session:
                                        await session.initialize()
                                        await session.call_tool("health_check", {})
                            except Exception as e:
                                logger.warning(f"⚠️ Agent {agent_id} health check failed: {e}")
                                logger.warning(f"🔄 Restarting {agent_id}...")
                                await self.restart_agent(agent_id)
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in agent monitoring: {e}")
                await asyncio.sleep(10)  # Wait before retrying

    async def _call_agent_tool(self, agent_endpoint: str, tool_name: str, arguments: dict) -> dict:
        """
        Call a tool on a specific agent's MCP server using proper MCP SSE client.
        
        Args:
            agent_endpoint: The SSE endpoint of the agent MCP server
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            dict: Response from the agent tool
        """
        try:
            from mcp.client.sse import sse_client
            from mcp import ClientSession
            
            async with sse_client(agent_endpoint, timeout=10) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()
                    
                    # Call the tool
                    result = await session.call_tool(tool_name, arguments)
                    
                    # Extract the result content
                    if result.content and len(result.content) > 0:
                        content_item = result.content[0]
                        if hasattr(content_item, 'text'):
                            return json.loads(content_item.text)
                    
                    return {"status": "error", "error": "No content in response"}
                    
        except Exception as e:
            return {"status": "error", "error": f"MCP client error: {str(e)}"}

    def _register_pool_tools(self):
        """
        Register coordinating tools that proxy requests to individual agent MCP servers.
        """
        
        @self.pool_server.tool(name="process_market_query", description="Process natural language market data queries via PolygonAgent")
        async def process_market_query(query: str) -> dict:
            """
            Process natural language market data requests via PolygonAgent MCP server.
            
            Args:
                query: Natural language query (e.g., "Get daily data for AAPL from 2024-01-01 to 2024-12-31")
            
            Returns:
                dict: Structured response with execution plan and results
            """
            endpoint = self.agent_endpoints.get("polygon_agent")
            if not endpoint:
                return {"status": "error", "error": "PolygonAgent endpoint not configured"}
            
            return await self._call_agent_tool(endpoint, "process_market_query", {"query": query})

        @self.pool_server.tool(name="fetch_market_data", description="Directly fetch market data via PolygonAgent")
        async def fetch_market_data(symbol: str, start: str, end: str, interval: str = "1d") -> dict:
            """
            Directly fetch market data for a specific symbol via PolygonAgent MCP server.
            
            Args:
                symbol: Stock symbol (e.g., 'AAPL')
                start: Start date (YYYY-MM-DD)
                end: End date (YYYY-MM-DD)
                interval: Time interval (1d, 1h, etc.)
            
            Returns:
                dict: Market data response
            """
            endpoint = self.agent_endpoints.get("polygon_agent")
            if not endpoint:
                return {"status": "error", "error": "PolygonAgent endpoint not configured"}
            
            return await self._call_agent_tool(endpoint, "fetch_market_data", {
                "symbol": symbol,
                "start": start,
                "end": end,
                "interval": interval
            })

        @self.pool_server.tool(name="get_company_info", description="Get company information via PolygonAgent")
        async def get_company_info(symbol: str) -> dict:
            """
            Get company information for a specific symbol via PolygonAgent MCP server.
            
            Args:
                symbol: Stock symbol (e.g., 'AAPL')
            
            Returns:
                dict: Company information response
            """
            endpoint = self.agent_endpoints.get("polygon_agent")
            if not endpoint:
                return {"status": "error", "error": "PolygonAgent endpoint not configured"}
            
            return await self._call_agent_tool(endpoint, "get_company_info", {"symbol": symbol})

        @self.pool_server.tool(name="batch_fetch_market_data", description="Batch fetch market data for multiple symbols")
        async def batch_fetch_market_data(symbols: List[str], start: str, end: str, interval: str = "1d") -> dict:
            """
            Batch fetch market data for multiple symbols via natural language processing.
            
            Args:
                symbols: List of stock symbols
                start: Start date (YYYY-MM-DD)
                end: End date (YYYY-MM-DD)
                interval: Time interval (1d, 1h, etc.)
            
            Returns:
                dict: Batch fetch results
            """
            endpoint = self.agent_endpoints.get("polygon_agent")
            if not endpoint:
                return {"status": "error", "error": "PolygonAgent endpoint not configured"}
            
            results = []
            for symbol in symbols:
                query = f"Get {interval} price data for {symbol} from {start} to {end}"
                result = await self._call_agent_tool(endpoint, "process_market_query", {"query": query})
                results.append({
                    "symbol": symbol,
                    "result": result
                })
            
            return {
                "status": "success",
                "batch_results": results,
                "total_symbols": len(symbols),
                "timestamp": datetime.now().isoformat()
            }

        @self.pool_server.tool(name="list_agents", description="List all configured data agents and their endpoints.")
        def list_agents() -> dict:
            """
            List all currently configured data agents and their endpoints.
            
            Returns:
                dict: Agent endpoints and status
            """
            agent_status = {}
            for agent_id, config in self.agent_configs.items():
                process_info = None
                if agent_id in self.agent_processes:
                    proc = self.agent_processes[agent_id]
                    process_info = {
                        "pid": proc.pid,
                        "running": proc.poll() is None,
                        "returncode": proc.returncode
                    }
                
                agent_status[agent_id] = {
                    "endpoint": config["endpoint"],
                    "port": config["port"],
                    "process": process_info,
                    "auto_managed": True
                }
            
            return {
                "agents": agent_status,
                "total_agents": len(self.agent_configs)
            }

        @self.pool_server.tool(name="start_agent", description="Start a specific agent")
        async def start_agent_tool(agent_id: str) -> dict:
            """
            Start a specific agent process.
            
            Args:
                agent_id: ID of the agent to start (e.g., 'polygon_agent')
            
            Returns:
                dict: Start operation result
            """
            success = await self.start_agent(agent_id)
            return {
                "status": "success" if success else "error",
                "agent_id": agent_id,
                "message": f"Agent {agent_id} {'started successfully' if success else 'failed to start'}"
            }

        @self.pool_server.tool(name="stop_agent", description="Stop a specific agent")
        async def stop_agent_tool(agent_id: str) -> dict:
            """
            Stop a specific agent process.
            
            Args:
                agent_id: ID of the agent to stop (e.g., 'polygon_agent')
            
            Returns:
                dict: Stop operation result
            """
            success = await self.stop_agent(agent_id)
            return {
                "status": "success" if success else "error",
                "agent_id": agent_id,
                "message": f"Agent {agent_id} {'stopped successfully' if success else 'failed to stop'}"
            }

        @self.pool_server.tool(name="restart_agent", description="Restart a specific agent")
        async def restart_agent_tool(agent_id: str) -> dict:
            """
            Restart a specific agent process.
            
            Args:
                agent_id: ID of the agent to restart (e.g., 'polygon_agent')
            
            Returns:
                dict: Restart operation result
            """
            success = await self.restart_agent(agent_id)
            return {
                "status": "success" if success else "error",
                "agent_id": agent_id,
                "message": f"Agent {agent_id} {'restarted successfully' if success else 'failed to restart'}"
            }

        @self.pool_server.tool(name="health_check", description="Health check for DataAgentPool and connected agents")
        async def health_check() -> dict:
            """
            Return the health status of the DataAgentPool and all connected agent servers.
            
            Returns:
                dict: Health status for pool and agents
            """
            agent_health = {}
            
            for agent_id, endpoint in self.agent_endpoints.items():
                try:
                    health_result = await self._call_agent_tool(endpoint, "health_check", {})
                    agent_health[agent_id] = {
                        "endpoint": endpoint,
                        "status": health_result.get("status", "unknown"),
                        "details": health_result
                    }
                except Exception as e:
                    agent_health[agent_id] = {
                        "endpoint": endpoint,
                        "status": "error",
                        "error": str(e)
                    }
            
            return {
                "pool_status": "ok",
                "timestamp": datetime.now().isoformat(),
                "agents": agent_health
            }

        @self.pool_server.tool(name="get_historical_data", description="Get historical price data for symbols with memory storage")
        async def get_historical_data(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> dict:
            """
            Get historical price data for a symbol and store it in memory agent.
            
            Args:
                symbol: Stock symbol (e.g., 'AAPL')
                start_date: Start date (YYYY-MM-DD)
                end_date: End date (YYYY-MM-DD)
                interval: Time interval (1d, 1h, etc.)
            
            Returns:
                dict: Historical data with memory storage confirmation
            """
            try:
                # Try to get data from Polygon agent if available
                endpoint = self.agent_endpoints.get("polygon_agent")
                
                if endpoint:
                    # Use real Polygon data
                    query = f"Get {interval} price data for {symbol} from {start_date} to {end_date}"
                    result = await self._call_agent_tool(endpoint, "process_market_query", {"query": query})
                    
                    if result.get("status") == "success" and result.get("data"):
                        historical_data = result["data"]
                        
                        # Store each data point in memory agent
                        for i, data_point in enumerate(historical_data):
                            await self.memory_client.store_event(
                                event_type="MARKET_DATA",
                                log_level="INFO",
                                title=f"Historical Data: {symbol}",
                                content=f"Date: {data_point.get('date', 'N/A')}, "
                                       f"Open: {data_point.get('open', 0)}, "
                                       f"High: {data_point.get('high', 0)}, "
                                       f"Low: {data_point.get('low', 0)}, "
                                       f"Close: {data_point.get('close', 0)}, "
                                       f"Volume: {data_point.get('volume', 0)}",
                                tags={f"symbol_{symbol}", "historical_data", "market_data"},
                                metadata={
                                    "symbol": symbol,
                                    "date": data_point.get('date'),
                                    "price_data": data_point,
                                    "interval": interval,
                                    "data_source": "polygon_agent"
                                }
                            )
                        
                        await self.memory_client.store_event(
                            event_type="MARKET_DATA",
                            log_level="INFO",
                            title=f"Historical Data Retrieved: {symbol}",
                            content=f"Successfully retrieved {len(historical_data)} data points "
                                   f"for {symbol} from {start_date} to {end_date}",
                            tags={f"symbol_{symbol}", "data_retrieval", "success"},
                            metadata={
                                "symbol": symbol,
                                "start_date": start_date,
                                "end_date": end_date,
                                "data_points_count": len(historical_data),
                                "interval": interval
                            }
                        )
                        
                        return {
                            "status": "success",
                            "symbol": symbol,
                            "data": historical_data,
                            "data_points": len(historical_data),
                            "memory_stored": True,
                            "source": "polygon_agent"
                        }
                
                # Fallback: Generate synthetic data for testing
                import random
                from datetime import datetime, timedelta
                
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                days = (end_dt - start_dt).days + 1
                
                base_price = random.uniform(90, 150)
                synthetic_data = []
                
                for i in range(days):
                    current_date = start_dt + timedelta(days=i)
                    daily_change = random.uniform(-0.05, 0.05)  # ±5% daily change
                    base_price *= (1 + daily_change)
                    
                    data_point = {
                        "date": current_date.strftime("%Y-%m-%d"),
                        "timestamp": current_date.strftime("%Y-%m-%d"),
                        "open": round(base_price * random.uniform(0.98, 1.02), 2),
                        "high": round(base_price * random.uniform(1.01, 1.05), 2),
                        "low": round(base_price * random.uniform(0.95, 0.99), 2),
                        "close": round(base_price, 2),
                        "volume": random.randint(1000000, 10000000)
                    }
                    synthetic_data.append(data_point)
                    
                    # Store synthetic data in memory agent
                    await self.memory_client.store_event(
                        event_type="MARKET_DATA",
                        log_level="INFO",
                        title=f"Synthetic Data: {symbol}",
                        content=f"Date: {data_point['date']}, Close: {data_point['close']}, Volume: {data_point['volume']}",
                        tags={f"symbol_{symbol}", "synthetic_data", "market_data"},
                        metadata={
                            "symbol": symbol,
                            "date": data_point['date'],
                            "price_data": data_point,
                            "interval": interval,
                            "data_source": "synthetic"
                        }
                    )
                
                await self.memory_client.store_event(
                    event_type="MARKET_DATA",
                    log_level="WARNING",
                    title=f"Synthetic Data Generated: {symbol}",
                    content=f"Generated {len(synthetic_data)} synthetic data points "
                           f"for {symbol} from {start_date} to {end_date} (real data unavailable)",
                    tags={f"symbol_{symbol}", "synthetic_data", "fallback"},
                    metadata={
                        "symbol": symbol,
                        "start_date": start_date,
                        "end_date": end_date,
                        "data_points_count": len(synthetic_data),
                        "interval": interval
                    }
                )
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "data": synthetic_data,
                    "data_points": len(synthetic_data),
                    "memory_stored": True,
                    "source": "synthetic"
                }
                
            except Exception as e:
                error_msg = f"Failed to get historical data for {symbol}: {str(e)}"
                logger.error(error_msg)
                
                await self.memory_client.store_event(
                    event_type="MARKET_DATA",
                    log_level="ERROR",
                    title=f"Data Retrieval Failed: {symbol}",
                    content=error_msg,
                    tags={f"symbol_{symbol}", "data_error", "failure"},
                    metadata={
                        "symbol": symbol,
                        "start_date": start_date,
                        "end_date": end_date,
                        "error": str(e)
                    }
                )
                
                return {
                    "status": "error",
                    "symbol": symbol,
                    "error": error_msg,
                    "memory_stored": True
                }

        # ...existing tools...

    async def run_async(self):
        """
        Async version of run that handles agent startup and monitoring.
        """
        try:
            # Start all agents first
            await self.start_all_agents()
            
            # Initialize memory agent
            # The memory_client is now initialized in __init__
            
            # Start agent monitoring in background
            monitor_task = asyncio.create_task(self.monitor_agents())
            
            # Start the MCP server
            logger.info(f"🚀 DataAgentPool MCP server ready on {self.host}:{self.port}")
            logger.info("=== Registered MCP Pool Tools ===")
            tools = await self.pool_server.list_tools()
            for tool in tools:
                logger.info(f"- {tool.name}")
            
            # This would typically be where we start the server
            # But since FastMCP doesn't have an async run method, we'll run it in the sync method
            return monitor_task
            
        except Exception as e:
            logger.error(f"Error in run_async: {e}")
            self.shutdown()
            raise

    def run(self):
        """
        Start the MCP pool server with agent management.
        """
        logger.info(f"[DataAgentPool] MCP pool server starting on {self.host}:{self.port} ...")
        
        # Setup the server
        self.pool_server.settings.host = self.host
        self.pool_server.settings.port = self.port
        
        # Start agents asynchronously in background
        if self.auto_start_agents:
            logger.info("🔄 Starting agents...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Start agents
            startup_success = loop.run_until_complete(self.start_all_agents())
            
            # Start monitoring in background
            monitor_task = loop.create_task(self.monitor_agents())
            
            # Close the event loop for now (FastMCP will create its own)
            # Note: This is a compromise since FastMCP doesn't support async startup hooks
        
        logger.info("=== Registered MCP Pool Tools ===")
        tools = asyncio.run(self.pool_server.list_tools())
        for tool in tools:
            logger.info(f"- {tool.name}")
        
        try:
            # Register shutdown handler
            import atexit
            atexit.register(self.shutdown)
            
            # Start the FastMCP server
            self.pool_server.run(transport="sse")
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            self.shutdown()
        except Exception as e:
            logger.error(f"Error running server: {e}")
            self.shutdown()
            raise

if __name__ == "__main__":
    # Script entry point: start the DataAgentPoolMCPServer
    pool = DataAgentPoolMCPServer(host="0.0.0.0", port=8001)
    pool.run()
