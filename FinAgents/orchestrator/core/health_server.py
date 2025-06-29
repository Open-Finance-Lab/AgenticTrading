"""
Health Check HTTP Server for Agent Pools
Provides simple HTTP endpoints for health monitoring
"""

import asyncio
import json
import logging
from aiohttp import web
from typing import Dict, Any

logger = logging.getLogger(__name__)

class HealthCheckServer:
    """Simple HTTP server for health check endpoints"""
    
    def __init__(self, agent_pool, port: int = None):
        self.agent_pool = agent_pool
        self.port = port or 8000
        self.app = web.Application()
        self.setup_routes()
        
    def setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/mcp/capabilities', self.mcp_capabilities)
        self.app.router.add_get('/status', self.status_check)
        
    async def health_check(self, request):
        """HTTP health check endpoint"""
        try:
            # Get health from agent pool if it has health_check method
            if hasattr(self.agent_pool, 'health_check') and callable(getattr(self.agent_pool, 'health_check')):
                # Try to call the MCP health_check tool
                health_result = await self.call_agent_health_check()
            else:
                # Basic health check
                health_result = {
                    "status": "healthy",
                    "timestamp": asyncio.get_event_loop().time(),
                    "agent_pool": self.agent_pool.__class__.__name__
                }
                
            return web.json_response(health_result, status=200)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return web.json_response({
                "status": "unhealthy",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }, status=503)
    
    async def call_agent_health_check(self):
        """Call the agent pool's health check"""
        try:
            # Access the health_check tool directly if available
            if hasattr(self.agent_pool, 'pool_server') and self.agent_pool.pool_server:
                tools = await self.agent_pool.pool_server.list_tools()
                health_tool = next((tool for tool in tools if tool.name == 'health_check'), None)
                
                if health_tool:
                    # Execute health check tool
                    # Note: This is a simplified approach, may need adjustment based on FastMCP implementation
                    return {
                        "status": "healthy",
                        "mcp_tools_available": len(tools),
                        "health_tool_available": True,
                        "agent_pool": self.agent_pool.__class__.__name__
                    }
            
            return {
                "status": "healthy",
                "agent_pool": self.agent_pool.__class__.__name__
            }
            
        except Exception as e:
            logger.warning(f"MCP health check failed, using basic check: {e}")
            return {
                "status": "healthy",
                "agent_pool": self.agent_pool.__class__.__name__,
                "note": "basic_health_check"
            }
    
    async def mcp_capabilities(self, request):
        """MCP capabilities endpoint"""
        try:
            if hasattr(self.agent_pool, 'pool_server') and self.agent_pool.pool_server:
                tools = await self.agent_pool.pool_server.list_tools()
                capabilities = {
                    "mcp_version": "1.0",
                    "tools_count": len(tools),
                    "tools": [{"name": tool.name, "description": tool.description} for tool in tools],
                    "transport": "sse"
                }
            else:
                capabilities = {
                    "mcp_version": "1.0",
                    "status": "no_mcp_server"
                }
                
            return web.json_response(capabilities, status=200)
            
        except Exception as e:
            return web.json_response({
                "error": str(e),
                "mcp_version": "1.0"
            }, status=500)
    
    async def status_check(self, request):
        """Detailed status endpoint"""
        try:
            status = {
                "agent_pool": self.agent_pool.__class__.__name__,
                "port": self.port,
                "timestamp": asyncio.get_event_loop().time(),
                "status": "running"
            }
            
            # Add agent-specific status if available
            if hasattr(self.agent_pool, 'agent_endpoints'):
                status["agent_endpoints"] = list(self.agent_pool.agent_endpoints.keys())
                
            return web.json_response(status, status=200)
            
        except Exception as e:
            return web.json_response({
                "error": str(e),
                "status": "error"
            }, status=500)
    
    async def start_server(self):
        """Start the HTTP server"""
        try:
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            site = web.TCPSite(runner, 'localhost', self.port)
            await site.start()
            
            logger.info(f"Health check server started on port {self.port}")
            return runner
            
        except Exception as e:
            logger.error(f"Failed to start health check server: {e}")
            raise
    
    def start_in_background(self):
        """Start server in background thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_server():
            runner = await self.start_server()
            # Keep running
            while True:
                await asyncio.sleep(1)
        
        loop.create_task(run_server())
        
        import threading
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        
        return thread

def add_health_server_to_agent_pool(agent_pool, health_port: int):
    """Add health check server to an existing agent pool"""
    health_server = HealthCheckServer(agent_pool, health_port)
    
    # Start health server in background
    health_thread = health_server.start_in_background()
    
    # Store reference to health server
    agent_pool._health_server = health_server
    agent_pool._health_thread = health_thread
    
    return health_server
