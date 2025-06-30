#!/usr/bin/env python3

"""
Simplified Alpha Agent Pool for debugging
"""

import os
import sys
import logging
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlphaAgentPool")

class SimpleAlphaAgentPool:
    def __init__(self, host="0.0.0.0", port=8081):
        logger.info(f"Initializing SimpleAlphaAgentPool on {host}:{port}")
        
        self.host = host
        self.port = port
        self.pool_server = FastMCP("SimpleAlphaAgentPool")
        
        logger.info("Registering tools...")
        self._register_tools()
        logger.info("Tools registered successfully")

    def _register_tools(self):
        @self.pool_server.tool(name="ping", description="Simple ping test")
        def ping() -> str:
            return "pong"
        
        @self.pool_server.tool(name="status", description="Get status")
        def status() -> dict:
            return {
                "status": "running",
                "host": self.host,
                "port": self.port
            }

if __name__ == "__main__":
    logger.info("🚀 Starting Simple Alpha Agent Pool...")
    
    try:
        alpha_pool = SimpleAlphaAgentPool(host="0.0.0.0", port=8081)
        
        # Configure server
        alpha_pool.pool_server.settings.host = "0.0.0.0"
        alpha_pool.pool_server.settings.port = 8081
        
        logger.info("Starting FastMCP server...")
        alpha_pool.pool_server.run()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Simple Alpha Agent Pool shutting down...")
    except Exception as e:
        logger.error(f"❌ Simple Alpha Agent Pool error: {e}")
        import traceback
        traceback.print_exc()
