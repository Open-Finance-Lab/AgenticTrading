#!/usr/bin/env python3
"""
FinAgent A2A Server

Dedicated A2A # Unified components
try:
    from unified_database_manager import create_database_manager
    from unified_interface_manager import create_interface_manager, UnifiedInterfaceManager
    UNIFIED_COMPONENTS_AVAILABLE = True
except ImportError:
    UNIFIED_COMPONENTS_AVAILABLE = False
    UnifiedInterfaceManager = object  # Fallback for type hints
    print("⚠️ Unified components not available")o-Agent) communication server implementation for FinAgent Memory operations.
This server focuses exclusively on agent-to-agent protocol compliance and inter-agent communication,
using the unified database and interface managers for actual operations.

Features:
- Agent-to-Agent protocol implementation
- Signal transmission and strategy sharing
- Real-time agent communication
- Unified architecture integration
- Performance monitoring and analytics

Author: FinAgent Team
License: Open Source
"""

# ═══════════════════════════════════════════════════════════════════════════════════
# IMPORTS AND DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# FastAPI for A2A HTTP server
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create fallback classes
    class BaseModel:
        def __init__(self, **kwargs):
            pass
        def dict(self):
            return {}
        def json(self):
            return "{}"
    class Field:
        def __init__(self, *args, **kwargs):
            pass
    print("⚠️ FastAPI not available. Install with: pip install fastapi uvicorn")

# WebSocket support for real-time communication
try:
    from fastapi import WebSocket, WebSocketDisconnect
    import websockets
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Unified components
try:
    from unified_database_manager import create_database_manager
    from unified_interface_manager import create_interface_manager, UnifiedInterfaceManager
    UNIFIED_COMPONENTS_AVAILABLE = True
except ImportError:
    UNIFIED_COMPONENTS_AVAILABLE = False
    print("⚠️ Unified components not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════
# DATA MODELS FOR A2A COMMUNICATION
# ═══════════════════════════════════════════════════════════════════════════════════

class AgentMessageType(Enum):
    """Types of A2A messages"""
    SIGNAL = "signal"
    STRATEGY = "strategy"
    PERFORMANCE_UPDATE = "performance_update"
    MEMORY_SHARE = "memory_share"
    HEALTH_CHECK = "health_check"
    COLLABORATION_REQUEST = "collaboration_request"

class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    ALERT = "alert"

class A2ASignal(BaseModel):
    """A2A Signal message model"""
    signal_id: str = Field(..., description="Unique signal identifier")
    source_agent: str = Field(..., description="Agent sending the signal")
    target_agents: List[str] = Field(default=[], description="Target agents (empty for broadcast)")
    signal_type: SignalType = Field(..., description="Type of signal")
    symbol: str = Field(..., description="Trading symbol")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence (0-1)")
    price: Optional[float] = Field(None, description="Associated price")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default={}, description="Additional signal data")

class A2AStrategy(BaseModel):
    """A2A Strategy sharing model"""
    strategy_id: str = Field(..., description="Unique strategy identifier")
    source_agent: str = Field(..., description="Agent sharing the strategy")
    strategy_name: str = Field(..., description="Strategy name")
    strategy_data: Dict[str, Any] = Field(..., description="Strategy parameters and logic")
    performance_metrics: Dict[str, float] = Field(default={}, description="Strategy performance")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class A2AMemoryShare(BaseModel):
    """A2A Memory sharing model"""
    memory_id: str = Field(..., description="Memory identifier")
    source_agent: str = Field(..., description="Agent sharing the memory")
    target_agents: List[str] = Field(default=[], description="Target agents")
    memory_content: Dict[str, Any] = Field(..., description="Memory content")
    relevance_score: float = Field(default=1.0, description="Memory relevance score")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class A2AMessage(BaseModel):
    """Generic A2A message wrapper"""
    message_id: str = Field(..., description="Unique message identifier")
    message_type: AgentMessageType = Field(..., description="Type of message")
    source_agent: str = Field(..., description="Source agent")
    target_agents: List[str] = Field(default=[], description="Target agents")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    priority: int = Field(default=1, description="Message priority (1-5)")
    expires_at: Optional[datetime] = Field(None, description="Message expiration")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# ═══════════════════════════════════════════════════════════════════════════════════
# A2A SERVER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════

A2A_SERVER_NAME = "FinAgent-A2A-Server"
A2A_SERVER_VERSION = "2.0.0"
A2A_SERVER_PORT = 8002

DATABASE_CONFIG = {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "FinOrchestration", 
    "database": "neo4j"
}

# Global components
interface_manager = None
connected_agents: Dict[str, Any] = {}
message_history: List[Any] = []

# ═══════════════════════════════════════════════════════════════════════════════════
# A2A SERVER IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════════

if FASTAPI_AVAILABLE and UNIFIED_COMPONENTS_AVAILABLE:
    
    # Create FastAPI app
    app = FastAPI(
        title=A2A_SERVER_NAME,
        version=A2A_SERVER_VERSION,
        description="FinAgent Agent-to-Agent Communication Server"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SERVER LIFECYCLE EVENTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize A2A server components."""
        global interface_manager
        
        try:
            logger.info("🚀 Starting FinAgent A2A Server...")
            
            # Initialize interface manager
            interface_manager = create_interface_manager(DATABASE_CONFIG)
            
            if await interface_manager.initialize():
                logger.info("✅ A2A Server initialized successfully")
            else:
                logger.error("❌ Failed to initialize A2A server")
                
        except Exception as e:
            logger.error(f"❌ A2A server startup failed: {e}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup A2A server components."""
        try:
            if interface_manager:
                await interface_manager.shutdown()
            
            # Close all WebSocket connections
            for agent_id, websocket in connected_agents.items():
                try:
                    await websocket.close()
                except:
                    pass
            
            logger.info("✅ A2A Server shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ A2A server shutdown error: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # HEALTH AND STATUS ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @app.get("/health")
    async def health_check():
        """A2A server health check endpoint."""
        try:
            health_info = {
                "status": "healthy",
                "server": A2A_SERVER_NAME,
                "version": A2A_SERVER_VERSION,
                "timestamp": datetime.utcnow().isoformat(),
                "connected_agents": len(connected_agents),
                "message_history_size": len(message_history),
                "components": {
                    "interface_manager": interface_manager is not None,
                    "websocket_support": WEBSOCKET_AVAILABLE
                }
            }
            
            # Get system health from interface manager
            if interface_manager:
                system_health = await interface_manager.execute_tool("health_check", {})
                health_info["system_health"] = system_health
            
            return health_info
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"status": "unhealthy", "error": str(e)}
            )
    
    @app.get("/status")
    async def get_server_status():
        """Get detailed A2A server status."""
        try:
            status_info = {
                "server_info": {
                    "name": A2A_SERVER_NAME,
                    "version": A2A_SERVER_VERSION,
                    "uptime": datetime.utcnow().isoformat()
                },
                "connections": {
                    "active_agents": list(connected_agents.keys()),
                    "total_connections": len(connected_agents)
                },
                "message_stats": {
                    "total_messages": len(message_history),
                    "recent_messages": len([m for m in message_history if (datetime.utcnow() - m.timestamp).seconds < 300])
                }
            }
            
            return status_info
            
        except Exception as e:
            logger.error(f"❌ Status check failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SIGNAL TRANSMISSION ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @app.post("/api/v1/signals/send")
    async def send_signal(signal: A2ASignal, background_tasks: BackgroundTasks):
        """Send a trading signal to target agents."""
        try:
            logger.info(f"📡 Sending signal {signal.signal_id} from {signal.source_agent}")
            
            # Store signal in memory system
            if interface_manager:
                memory_result = await interface_manager.execute_tool("store_graph_memory", {
                    "query": f"Signal: {signal.signal_type.value} {signal.symbol}",
                    "keywords": ["signal", signal.signal_type.value, signal.symbol],
                    "summary": f"{signal.source_agent} sent {signal.signal_type.value} signal for {signal.symbol} with confidence {signal.confidence}",
                    "agent_id": signal.source_agent,
                    "event_type": "SIGNAL_TRANSMISSION"
                })
            
            # Create A2A message
            a2a_message = A2AMessage(
                message_id=f"signal_{signal.signal_id}",
                message_type=AgentMessageType.SIGNAL,
                source_agent=signal.source_agent,
                target_agents=signal.target_agents,
                payload=signal.dict(),
                priority=3 if signal.signal_type in [SignalType.BUY, SignalType.SELL] else 2
            )
            
            # Add to message history
            message_history.append(a2a_message)
            
            # Broadcast to connected agents
            background_tasks.add_task(broadcast_message, a2a_message)
            
            return {
                "status": "success",
                "message": f"Signal {signal.signal_id} sent successfully",
                "signal_id": signal.signal_id,
                "targets": signal.target_agents if signal.target_agents else "broadcast"
            }
            
        except Exception as e:
            logger.error(f"❌ Signal transmission failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/signals/history")
    async def get_signal_history(agent_id: Optional[str] = None, limit: int = 50):
        """Get signal transmission history."""
        try:
            # Filter messages for signals
            signal_messages = [
                msg for msg in message_history 
                if msg.message_type == AgentMessageType.SIGNAL
                and (not agent_id or msg.source_agent == agent_id or agent_id in msg.target_agents)
            ]
            
            # Sort by timestamp and limit
            signal_messages.sort(key=lambda x: x.timestamp, reverse=True)
            signal_messages = signal_messages[:limit]
            
            return {
                "status": "success",
                "signals": [msg.dict() for msg in signal_messages],
                "total_count": len(signal_messages)
            }
            
        except Exception as e:
            logger.error(f"❌ Signal history retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # STRATEGY SHARING ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @app.post("/api/v1/strategies/share")
    async def share_strategy(strategy: A2AStrategy, background_tasks: BackgroundTasks):
        """Share a trading strategy with other agents."""
        try:
            logger.info(f"📈 Sharing strategy {strategy.strategy_id} from {strategy.source_agent}")
            
            # Store strategy in memory system
            if interface_manager:
                memory_result = await interface_manager.execute_tool("store_graph_memory", {
                    "query": f"Strategy: {strategy.strategy_name}",
                    "keywords": ["strategy", strategy.strategy_name, "shared"],
                    "summary": f"{strategy.source_agent} shared strategy {strategy.strategy_name}",
                    "agent_id": strategy.source_agent,
                    "event_type": "STRATEGY_SHARING"
                })
            
            # Create A2A message
            a2a_message = A2AMessage(
                message_id=f"strategy_{strategy.strategy_id}",
                message_type=AgentMessageType.STRATEGY,
                source_agent=strategy.source_agent,
                target_agents=[],  # Broadcast to all
                payload=strategy.dict(),
                priority=2
            )
            
            # Add to message history
            message_history.append(a2a_message)
            
            # Broadcast to connected agents
            background_tasks.add_task(broadcast_message, a2a_message)
            
            return {
                "status": "success",
                "message": f"Strategy {strategy.strategy_id} shared successfully",
                "strategy_id": strategy.strategy_id
            }
            
        except Exception as e:
            logger.error(f"❌ Strategy sharing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/strategies/list")
    async def list_shared_strategies(agent_id: Optional[str] = None):
        """List shared strategies."""
        try:
            # Filter messages for strategies
            strategy_messages = [
                msg for msg in message_history 
                if msg.message_type == AgentMessageType.STRATEGY
                and (not agent_id or msg.source_agent == agent_id)
            ]
            
            # Sort by timestamp
            strategy_messages.sort(key=lambda x: x.timestamp, reverse=True)
            
            return {
                "status": "success",
                "strategies": [msg.dict() for msg in strategy_messages],
                "total_count": len(strategy_messages)
            }
            
        except Exception as e:
            logger.error(f"❌ Strategy listing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # MEMORY SHARING ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @app.post("/api/v1/memory/share")
    async def share_memory(memory_share: A2AMemoryShare, background_tasks: BackgroundTasks):
        """Share memory content with target agents."""
        try:
            logger.info(f"🧠 Sharing memory {memory_share.memory_id} from {memory_share.source_agent}")
            
            # Create A2A message
            a2a_message = A2AMessage(
                message_id=f"memory_{memory_share.memory_id}",
                message_type=AgentMessageType.MEMORY_SHARE,
                source_agent=memory_share.source_agent,
                target_agents=memory_share.target_agents,
                payload=memory_share.dict(),
                priority=1
            )
            
            # Add to message history
            message_history.append(a2a_message)
            
            # Broadcast to target agents
            background_tasks.add_task(broadcast_message, a2a_message)
            
            return {
                "status": "success",
                "message": f"Memory {memory_share.memory_id} shared successfully",
                "memory_id": memory_share.memory_id,
                "targets": memory_share.target_agents
            }
            
        except Exception as e:
            logger.error(f"❌ Memory sharing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # WEBSOCKET COMMUNICATION
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    if WEBSOCKET_AVAILABLE:
        @app.websocket("/ws/{agent_id}")
        async def websocket_endpoint(websocket: WebSocket, agent_id: str):
            """WebSocket endpoint for real-time agent communication."""
            try:
                await websocket.accept()
                connected_agents[agent_id] = websocket
                logger.info(f"🔌 Agent {agent_id} connected via WebSocket")
                
                # Send welcome message
                welcome_message = {
                    "type": "connection_established",
                    "agent_id": agent_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": f"Welcome to FinAgent A2A Communication, {agent_id}!"
                }
                await websocket.send_text(json.dumps(welcome_message))
                
                # Listen for messages
                while True:
                    try:
                        data = await websocket.receive_text()
                        message_data = json.loads(data)
                        
                        # Process incoming message
                        await process_websocket_message(agent_id, message_data)
                        
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.error(f"❌ WebSocket message error for {agent_id}: {e}")
                        break
                        
            except Exception as e:
                logger.error(f"❌ WebSocket connection error for {agent_id}: {e}")
            finally:
                # Clean up connection
                if agent_id in connected_agents:
                    del connected_agents[agent_id]
                logger.info(f"🔌 Agent {agent_id} disconnected")
        
        async def process_websocket_message(agent_id: str, message_data: Dict[str, Any]):
            """Process incoming WebSocket message from agent."""
            try:
                message_type = message_data.get("type", "unknown")
                
                if message_type == "ping":
                    # Respond to ping
                    response = {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    await connected_agents[agent_id].send_text(json.dumps(response))
                
                elif message_type == "signal":
                    # Process signal message
                    signal_data = message_data.get("payload", {})
                    signal_data["source_agent"] = agent_id
                    
                    # Create signal and broadcast
                    signal = A2ASignal(**signal_data)
                    await send_signal(signal, BackgroundTasks())
                
                # Add more message type handlers as needed
                
            except Exception as e:
                logger.error(f"❌ WebSocket message processing failed: {e}")
        
        async def broadcast_message(message: A2AMessage):
            """Broadcast A2A message to connected agents."""
            try:
                # Determine target agents
                target_agents = message.target_agents if message.target_agents else list(connected_agents.keys())
                
                # Remove source agent from targets
                if message.source_agent in target_agents:
                    target_agents.remove(message.source_agent)
                
                # Send to each target agent
                for agent_id in target_agents:
                    if agent_id in connected_agents:
                        try:
                            websocket = connected_agents[agent_id]
                            await websocket.send_text(message.json())
                            logger.info(f"📤 Sent message to {agent_id}")
                        except Exception as e:
                            logger.error(f"❌ Failed to send message to {agent_id}: {e}")
                
            except Exception as e:
                logger.error(f"❌ Message broadcast failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # ANALYTICS AND MONITORING
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @app.get("/api/v1/analytics/summary")
    async def get_analytics_summary():
        """Get A2A communication analytics summary."""
        try:
            # Message type distribution
            message_types = {}
            for msg in message_history:
                msg_type = msg.message_type.value
                message_types[msg_type] = message_types.get(msg_type, 0) + 1
            
            # Agent activity
            agent_activity = {}
            for msg in message_history:
                agent_id = msg.source_agent
                agent_activity[agent_id] = agent_activity.get(agent_id, 0) + 1
            
            # Recent activity (last hour)
            recent_cutoff = datetime.utcnow()
            recent_cutoff = recent_cutoff.replace(hour=recent_cutoff.hour-1)
            recent_messages = len([msg for msg in message_history if msg.timestamp > recent_cutoff])
            
            analytics = {
                "summary": {
                    "total_messages": len(message_history),
                    "active_connections": len(connected_agents),
                    "recent_messages_1h": recent_messages
                },
                "message_distribution": message_types,
                "agent_activity": agent_activity,
                "connection_status": {
                    "connected_agents": list(connected_agents.keys()),
                    "websocket_support": WEBSOCKET_AVAILABLE
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Analytics summary failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

else:
    # Fallback when dependencies are missing
    app = None
    
    def print_dependency_error():
        print("❌ A2A Server dependencies missing")
        if not FASTAPI_AVAILABLE:
            print("   Install FastAPI with: pip install fastapi uvicorn")
        if not UNIFIED_COMPONENTS_AVAILABLE:
            print("   Unified components not available")

# ═══════════════════════════════════════════════════════════════════════════════════
# SERVER INFORMATION AND UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════════

def print_a2a_server_info():
    """Print A2A server information."""
    if app:
        print("\n" + "="*80)
        print("🚀 FINAGENT A2A SERVER - AGENT-TO-AGENT COMMUNICATION")
        print("="*80)
        print(f"📋 Server Info:")
        print(f"   🏷️  Name: {A2A_SERVER_NAME}")
        print(f"   📦 Version: {A2A_SERVER_VERSION}")
        print(f"   🔧 Protocol: Agent-to-Agent (A2A)")
        print(f"   🗄️  Architecture: Unified Components")
        print(f"   🌐 WebSocket Support: {'✅ Available' if WEBSOCKET_AVAILABLE else '❌ Unavailable'}")
        print("\n📡 Available A2A Endpoints:")
        print("   • POST /api/v1/signals/send - Send trading signals")
        print("   • GET  /api/v1/signals/history - Get signal history")
        print("   • POST /api/v1/strategies/share - Share trading strategies")
        print("   • GET  /api/v1/strategies/list - List shared strategies")
        print("   • POST /api/v1/memory/share - Share memory content")
        print("   • GET  /api/v1/analytics/summary - Get communication analytics")
        print("   • GET  /health - Health check")
        print("   • GET  /status - Server status")
        if WEBSOCKET_AVAILABLE:
            print("   • WS   /ws/{agent_id} - Real-time WebSocket communication")
        print("\n🔧 Server Configuration:")
        print(f"   📍 Database URI: {DATABASE_CONFIG['uri']}")
        print(f"   👤 Database User: {DATABASE_CONFIG['username']}")
        print(f"   🗄️  Database Name: {DATABASE_CONFIG['database']}")
        print(f"   🚪 Server Port: {A2A_SERVER_PORT}")
        print("="*80)
    else:
        print_dependency_error()

# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_a2a_server_info()
    if app:
        print(f"\n🚀 Starting FinAgent A2A Server on port {A2A_SERVER_PORT}...")
        print(f"   Use: uvicorn a2a_server:app --host 0.0.0.0 --port {A2A_SERVER_PORT}")
        print("="*80)
        
        # Run the server
        uvicorn.run(app, host="0.0.0.0", port=A2A_SERVER_PORT)
    else:
        print("\n❌ Cannot start A2A server - dependencies missing")
        print("="*80)
