#!/usr/bin/env python3
"""
Enhanced MCP Server with additional endpoints for tools, health checks, and better integration.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the enhanced demo system
from demo_decoupled_system import EnhancedAlphaPoolDemo

class EnhancedMCPServer:
    """Enhanced MCP Server with additional endpoints and better integration."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8081):
        self.host = host
        self.port = port
        
        # Initialize the demo system
        self.demo = EnhancedAlphaPoolDemo(test_mode=True)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Alpha Agent Pool MCP Server",
            description="Enhanced MCP Server with additional endpoints",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Create MCP server
        self.mcp_server = FastMCP("AlphaAgentPoolMCPServer")
        
        # Setup routes
        self._setup_routes()
        
        # Setup MCP tools
        self._setup_mcp_tools()
    
    def _setup_routes(self):
        """Setup additional HTTP routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with server information."""
            return {
                "server": "Alpha Agent Pool MCP Server",
                "version": "1.0.0",
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "endpoints": {
                    "mcp": "/sse",
                    "tools": "/tools",
                    "health": "/health",
                    "status": "/status",
                    "info": "/info"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            try:
                # Check if demo system is accessible
                agent_count = len(self.demo.agent_registry)
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "agent_count": agent_count,
                    "demo_system": "accessible"
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
        
        @self.app.get("/status")
        async def status():
            """Detailed status endpoint."""
            try:
                return {
                    "status": "running",
                    "timestamp": datetime.now().isoformat(),
                    "server_info": {
                        "host": self.host,
                        "port": self.port,
                        "transport": "sse"
                    },
                    "demo_system": {
                        "session_id": self.demo.session_id,
                        "test_mode": self.demo.test_mode,
                        "agent_count": len(self.demo.agent_registry)
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/info")
        async def info():
            """Server information endpoint."""
            return {
                "server": "Alpha Agent Pool MCP Server",
                "version": "1.0.0",
                "description": "Enhanced MCP Server for Alpha Agent Pool",
                "features": [
                    "MCP Protocol Support",
                    "SSE Transport",
                    "Alpha Signal Generation",
                    "Factor Discovery",
                    "Strategy Development",
                    "Backtesting",
                    "Memory Management"
                ],
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/tools")
        async def list_tools():
            """List available MCP tools."""
            try:
                # Get tools from MCP server
                tools = []
                
                # Add core tools
                core_tools = [
                    "generate_alpha_signals",
                    "discover_alpha_factors",
                    "develop_strategy_configuration",
                    "run_comprehensive_backtest",
                    "submit_strategy_to_memory",
                    "run_integrated_backtest",
                    "validate_strategy_performance"
                ]
                
                for tool_name in core_tools:
                    tools.append({
                        "name": tool_name,
                        "type": "core",
                        "description": f"Core {tool_name.replace('_', ' ')} functionality"
                    })
                
                # Add agent management tools
                agent_tools = [
                    "start_agent",
                    "list_agents",
                    "get_agent_status",
                    "momentum_health"
                ]
                
                for tool_name in agent_tools:
                    tools.append({
                        "name": tool_name,
                        "type": "agent_management",
                        "description": f"Agent management: {tool_name.replace('_', ' ')}"
                    })
                
                # Add memory tools
                memory_tools = [
                    "get_memory",
                    "set_memory",
                    "delete_memory",
                    "list_memory_keys"
                ]
                
                for tool_name in memory_tools:
                    tools.append({
                        "name": tool_name,
                        "type": "memory",
                        "description": f"Memory operations: {tool_name.replace('_', ' ')}"
                    })
                
                return {
                    "tools": tools,
                    "total_count": len(tools),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/sse")
        async def sse_endpoint():
            """SSE endpoint for MCP communication."""
            async def event_stream():
                # Send initial connection info
                yield f"event: endpoint\ndata: /sse/messages/?session_id={self.demo.session_id}\n\n"
                
                # Keep connection alive
                while True:
                    await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                    yield f"event: heartbeat\ndata: {datetime.now().isoformat()}\n\n"
            
            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*"
                }
            )
    
    def _setup_mcp_tools(self):
        """Setup MCP tools."""
        
        @self.mcp_server.tool()
        async def generate_alpha_signals(symbol: str = None, symbols: list = None, 
                                       date: str = None, lookback_period: int = 20, 
                                       price: float = None, memory: dict = None) -> dict:
            """Generate alpha signals using momentum agent."""
            try:
                await self.demo._initialize_system()
                result = await self.demo.generate_alpha_signals(
                    symbol=symbol, symbols=symbols, date=date,
                    lookback_period=lookback_period, price=price, memory=memory
                )
                await self.demo.shutdown()
                return result
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        @self.mcp_server.tool()
        async def discover_alpha_factors(factor_categories: list = None, 
                                       significance_threshold: float = 0.05) -> dict:
            """Discover alpha factors."""
            try:
                await self.demo._initialize_system()
                result = await self.demo.discover_alpha_factors(
                    factor_categories=factor_categories,
                    significance_threshold=significance_threshold
                )
                await self.demo.shutdown()
                return result
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        @self.mcp_server.tool()
        async def develop_strategy_configuration(risk_level: str = "moderate", 
                                               target_volatility: float = 0.15) -> dict:
            """Develop strategy configuration."""
            try:
                await self.demo._initialize_system()
                result = await self.demo.develop_strategy_configuration(
                    risk_level=risk_level,
                    target_volatility=target_volatility
                )
                await self.demo.shutdown()
                return result
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        @self.mcp_server.tool()
        async def run_comprehensive_backtest(strategy_id: str, 
                                           start_date: str = "2018-01-01", 
                                           end_date: str = "2023-12-31") -> dict:
            """Run comprehensive backtest."""
            try:
                await self.demo._initialize_system()
                result = await self.demo.run_comprehensive_backtest(
                    strategy_id=strategy_id,
                    start_date=start_date,
                    end_date=end_date
                )
                await self.demo.shutdown()
                return result
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        @self.mcp_server.tool()
        async def submit_strategy_to_memory(strategy_id: str, backtest_id: str = None) -> dict:
            """Submit strategy to memory."""
            try:
                await self.demo._initialize_system()
                result = await self.demo.submit_strategy_to_memory(
                    strategy_id=strategy_id,
                    backtest_id=backtest_id
                )
                await self.demo.shutdown()
                return result
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        @self.mcp_server.tool()
        async def run_integrated_backtest(strategy_id: str, symbols: list, 
                                        start_date: str = "2020-01-01", 
                                        end_date: str = "2023-12-31", 
                                        risk_level: str = "moderate") -> dict:
            """Run integrated backtest pipeline."""
            try:
                await self.demo._initialize_system()
                result = await self.demo.run_integrated_backtest(
                    strategy_id=strategy_id,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    risk_level=risk_level
                )
                await self.demo.shutdown()
                return result
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        @self.mcp_server.tool()
        async def validate_strategy_performance(strategy_id: str, backtest_id: str = None) -> dict:
            """Validate strategy performance."""
            try:
                await self.demo._initialize_system()
                result = await self.demo.validate_strategy_performance(
                    strategy_id=strategy_id,
                    backtest_id=backtest_id
                )
                await self.demo.shutdown()
                return result
            except Exception as e:
                return {"status": "error", "message": str(e)}
    
    async def start(self):
        """Start the enhanced MCP server."""
        try:
            # Initialize demo system
            await self.demo._initialize_system()
            
            # Start momentum agent
            self.demo.start_agent('momentum_agent')
            
            print(f"🚀 Starting Enhanced MCP Server on {self.host}:{self.port}")
            print("📡 Available endpoints:")
            print("   - / (root)")
            print("   - /health (health check)")
            print("   - /status (detailed status)")
            print("   - /info (server info)")
            print("   - /tools (available tools)")
            print("   - /sse (MCP SSE endpoint)")
            
            # Start the server
            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            raise

async def main():
    """Main entry point."""
    server = EnhancedMCPServer()
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())
