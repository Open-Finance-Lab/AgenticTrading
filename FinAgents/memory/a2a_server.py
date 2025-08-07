"""
FinAgent A2A Memory Server

A2A (Agent-to-Agent) Protocol compliant server for FinAgent Memory operations.
Built using the official A2A Python SDK.

Based on A2A Protocol specification and samples from:
https://github.com/a2aproject/a2a-samples
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, AsyncGenerator

import click
import uvicorn

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    TextPart,
    DataPart,
    UnsupportedOperationError,
    InvalidParamsError,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact
from a2a.utils.errors import ServerError

# Memory system imports - using optional imports for graceful degradation
try:
    from unified_database_manager import create_database_manager
    from unified_interface_manager import create_interface_manager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("⚠️ Memory components not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════
# A2A SERVER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════

SERVER_NAME = "FinAgent Memory A2A Server"
SERVER_VERSION = "1.0.0"
SERVER_PORT = 8002
SERVER_HOST = "0.0.0.0"

# Memory database configuration
DATABASE_CONFIG = {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "finagent123",
    "database": "neo4j"
}# Global components
interface_manager = None
message_history: List[Dict[str, Any]] = []

# ═══════════════════════════════════════════════════════════════════════════════════
# MEMORY AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════════

class MemoryAgent:
    """FinAgent Memory Agent for handling memory operations."""
    
    def __init__(self):
        """Initialize the Memory Agent."""
        self.interface_manager = None
        self.database_manager = None
        self.message_history: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize the memory interface manager and database manager."""
        if MEMORY_AVAILABLE:
            try:
                # Initialize interface manager
                self.interface_manager = create_interface_manager(DATABASE_CONFIG)
                if await self.interface_manager.initialize():
                    logger.info("✅ Memory interface manager initialized")
                    
                    # Initialize database manager for direct database operations
                    self.database_manager = create_database_manager(DATABASE_CONFIG)
                    if await self.database_manager.connect():
                        logger.info("✅ Database manager connected")
                        return True
                    else:
                        logger.warning("⚠️ Database manager connection failed")
                        return False
                else:
                    logger.warning("⚠️ Memory interface manager initialization failed")
                    return False
            except Exception as e:
                logger.warning(f"⚠️ Memory system initialization failed: {e}")
                return False
        return False
    
    async def stream(self, query: str, context_id: str = None, task_id: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a query and stream results."""
        try:
            logger.info(f"🧠 Processing query: {query[:100]}...")
            
            # Add to message history
            message_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "content": query,
                "source": "a2a_client",
                "type": "query",
                "context_id": context_id,
                "task_id": task_id
            }
            self.message_history.append(message_entry)
            
            # Keep only last 100 messages
            if len(self.message_history) > 100:
                self.message_history.pop(0)
            
            # Initial status update
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': f"Processing memory query: {query[:50]}...",
                'response_type': 'text'
            }
            
            # Process with memory system if available
            if MEMORY_AVAILABLE and self.interface_manager:
                try:
                    # Store the query in memory
                    memory_result = await self.interface_manager.execute_tool("store_graph_memory", {
                        "query": query,
                        "keywords": ["a2a", "query", "external"],
                        "summary": f"A2A query: {query[:100]}",
                        "agent_id": "a2a_client",
                        "event_type": "A2A_QUERY"
                    })
                    
                    # Search for relevant memories
                    search_result = await self.interface_manager.execute_tool("search_memories", {
                        "query": query,
                        "limit": 5
                    })
                    
                    memories = search_result.get('memories', [])
                    
                    # Stream progress update
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': f"Found {len(memories)} relevant memories, generating response...",
                        'response_type': 'text'
                    }
                    
                    # Format response
                    response_parts = [
                        f"Memory query processed successfully!",
                        f"Query: {query}",
                        f"Stored in memory: ✅",
                        f"Relevant memories found: {len(memories)}",
                        "",
                        "Memory search results:"
                    ]
                    
                    for i, memory in enumerate(memories[:3], 1):  # Show top 3
                        summary = memory.get('summary', 'No summary available')[:100]
                        response_parts.append(f"{i}. {summary}...")
                    
                    if len(memories) > 3:
                        response_parts.append(f"... and {len(memories) - 3} more memories")
                    
                    response_text = "\n".join(response_parts)
                    
                except Exception as memory_error:
                    logger.warning(f"Memory processing failed: {memory_error}")
                    response_text = f"Query received but memory processing failed: {str(memory_error)}"
            else:
                response_text = f"Query received: {query}\nMemory system not available, but query has been logged."
            
            # Final completion
            yield {
                'is_task_complete': True,
                'require_user_input': False,
                'content': response_text,
                'response_type': 'text'
            }
            
        except Exception as e:
            logger.error(f"❌ Memory agent error: {e}")
            yield {
                'is_task_complete': True,
                'require_user_input': False,
                'content': f"Error processing query: {str(e)}",
                'response_type': 'text'
            }
    
    async def shutdown(self):
        """Cleanup resources."""
        try:
            if self.interface_manager and MEMORY_AVAILABLE:
                await self.interface_manager.shutdown()
                logger.info("✅ Memory interface manager shutdown")
            
            if self.database_manager and MEMORY_AVAILABLE:
                await self.database_manager.disconnect()
                logger.info("✅ Database manager disconnected")
                
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════
# A2A AGENT EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════════

class MemoryAgentExecutor(AgentExecutor):
    """A2A Agent Executor for FinAgent Memory operations."""
    
    def __init__(self):
        """Initialize the Memory Agent Executor."""
        logger.info("🧠 MemoryAgentExecutor initialized")
        self.agent = MemoryAgent()
        self.memory_store = {}  # Simple fallback storage
        
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute memory operations based on user input."""
        try:
            # Initialize agent if not already done
            if not self.agent.interface_manager and MEMORY_AVAILABLE:
                await self.agent.initialize()
            
            # Get user input and task
            query = context.get_user_input()
            task = context.current_task
            
            if not context.message:
                raise Exception('No message provided')
            
            if not task:
                task = new_task(context.message)
                await event_queue.enqueue_event(task)
            
            # Create task updater
            updater = TaskUpdater(event_queue, task.id, task.context_id)
            
            logger.info(f"Processing memory query: {query}")
            
            # Process the memory query
            response_text = await self._process_memory_query(query)
            
            # Add final artifact (remove description parameter as it's not supported)
            await updater.add_artifact(
                [TextPart(text=response_text)],
                name='memory-result'
            )
            await updater.complete()
            
            logger.info("✅ Memory operation completed")
                    
        except Exception as e:
            logger.error(f"❌ Memory executor error: {e}")
            if 'task' in locals() and task:
                updater = TaskUpdater(event_queue, task.id, task.context_id)
                await updater.add_artifact(
                    [TextPart(text=f"Memory Error: {str(e)}")],
                    name='error-result'
                )
                await updater.complete()
    
    async def _process_memory_query(self, query: str) -> str:
        """Process memory query and return response."""
        try:
            # Try to use the full memory system first
            if MEMORY_AVAILABLE and self.agent.interface_manager:
                try:
                    # Parse query for structured operations
                    import json
                    try:
                        data = json.loads(query)
                        action = data.get("action", "").lower()
                        
                        if action == "store":
                            result = await self.agent.interface_manager.execute_tool("store_graph_memory", {
                                "query": data.get("value", ""),
                                "keywords": [data.get("key", "")],
                                "summary": f"Stored: {data.get('key', '')}",
                                "agent_id": "a2a_client",
                                "event_type": "MEMORY_STORE"
                            })
                            return f"✅ Stored memory: {data.get('key', '')}"
                            
                        elif action == "retrieve" or action == "search":
                            result = await self.agent.interface_manager.execute_tool("search_memories", {
                                "query": data.get("key", "") or data.get("query", ""),
                                "limit": 5
                            })
                            if result.get("success") and result.get("data"):
                                memories = result["data"]
                                if memories:
                                    response = f"🔍 Found {len(memories)} memories:\n"
                                    for mem in memories[:3]:  # Show top 3
                                        response += f"• {mem.get('summary', mem.get('content', ''))}\n"
                                    return response
                                else:
                                    return "📝 No memories found"
                            else:
                                return "❌ Search failed"
                                
                        elif action == "list":
                            result = await self.agent.interface_manager.execute_tool("search_memories", {
                                "query": "",
                                "limit": 10
                            })
                            if result.get("success") and result.get("data"):
                                memories = result["data"]
                                if memories:
                                    response = f"📝 Found {len(memories)} memories:\n"
                                    for mem in memories:
                                        response += f"• {mem.get('summary', mem.get('content', ''))}\n"
                                    return response
                                else:
                                    return "📝 Memory is empty"
                            else:
                                return "❌ List failed"
                        
                        elif action == "stats":
                            # Use database manager for statistics
                            if self.agent.database_manager:
                                try:
                                    stats = await self.agent.database_manager.get_memory_statistics()
                                    return f"📊 Database Statistics:\n" + \
                                           f"• Total memories: {stats.get('total_memories', 0)}\n" + \
                                           f"• Total agents: {stats.get('total_agents', 0)}\n" + \
                                           f"• Total events: {stats.get('total_events', 0)}\n" + \
                                           f"• Database health: {'✅ Good' if stats.get('healthy', False) else '❌ Issues'}"
                                except Exception as e:
                                    return f"❌ Statistics retrieval failed: {str(e)}"
                            else:
                                return "❌ Database manager not available for statistics"
                        
                        elif action == "health":
                            # Database health check
                            if self.agent.database_manager:
                                try:
                                    health = await self.agent.database_manager.health_check()
                                    return f"🏥 Database Health Check:\n" + \
                                           f"• Connection: {'✅ Active' if health.get('connected', False) else '❌ Failed'}\n" + \
                                           f"• Response time: {health.get('response_time', 'N/A')}ms\n" + \
                                           f"• Neo4j version: {health.get('version', 'Unknown')}"
                                except Exception as e:
                                    return f"❌ Health check failed: {str(e)}"
                            else:
                                return "❌ Database manager not available for health check"
                        
                        else:
                            return f"❓ Unknown action '{action}'. Available: store, retrieve, search, list, stats, health"
                    
                    except json.JSONDecodeError:
                        # Handle as plain text search
                        result = await self.agent.interface_manager.execute_tool("search_memories", {
                            "query": query,
                            "limit": 5
                        })
                        if result.get("success") and result.get("data"):
                            memories = result["data"]
                            if memories:
                                response = f"🔍 Found {len(memories)} relevant memories:\n"
                                for mem in memories[:3]:
                                    response += f"• {mem.get('summary', mem.get('content', ''))}\n"
                                return response
                            else:
                                # Store as new memory
                                store_result = await self.agent.interface_manager.execute_tool("store_graph_memory", {
                                    "query": query,
                                    "keywords": ["a2a", "note"],
                                    "summary": f"Note: {query[:100]}",
                                    "agent_id": "a2a_client",
                                    "event_type": "A2A_NOTE"
                                })
                                return f"📝 Stored as new memory: {query[:50]}..."
                        else:
                            return f"❌ Memory system error: {result.get('error', 'Unknown error')}"
                            
                except Exception as e:
                    logger.warning(f"Memory system error, falling back to simple storage: {e}")
                    # Fall back to simple storage
                    return await self._simple_memory_fallback(query)
            
            else:
                # Use simple fallback storage
                return await self._simple_memory_fallback(query)
                
        except Exception as e:
            return f"❌ Error processing memory query: {str(e)}"
    
    async def _simple_memory_fallback(self, query: str) -> str:
        """Simple memory fallback when full system is not available."""
        try:
            import json
            from datetime import datetime
            
            try:
                data = json.loads(query)
                action = data.get("action", "").lower()
                key = data.get("key", "")
                value = data.get("value", "")
                search_query = data.get("query", "")
                
                if action == "store" and key and value:
                    self.memory_store[key] = {
                        "value": value,
                        "timestamp": datetime.now().isoformat(),
                        "type": "stored"
                    }
                    return f"✅ Stored key '{key}' with value '{value}'"
                
                elif action == "retrieve" and key:
                    if key in self.memory_store:
                        item = self.memory_store[key]
                        return f"📋 Retrieved '{key}': {item['value']} (stored: {item['timestamp']})"
                    else:
                        return f"❌ Key '{key}' not found in memory"
                
                elif action == "search" and search_query:
                    matches = []
                    for k, v in self.memory_store.items():
                        if search_query.lower() in k.lower() or search_query.lower() in str(v['value']).lower():
                            matches.append(f"  • {k}: {v['value']}")
                    
                    if matches:
                        return f"🔍 Found {len(matches)} matches for '{search_query}':\n" + "\n".join(matches)
                    else:
                        return f"🔍 No matches found for '{search_query}'"
                
                elif action == "list":
                    if self.memory_store:
                        items = []
                        for k, v in self.memory_store.items():
                            items.append(f"  • {k}: {v['value']} ({v['timestamp']})")
                        return f"📝 Memory contains {len(self.memory_store)} items:\n" + "\n".join(items)
                    else:
                        return "📝 Memory is empty"
                
                else:
                    return f"❓ Unknown action '{action}'. Available: store, retrieve, search, list, stats, health"
                    
            except json.JSONDecodeError:
                # Handle as plain text query
                if query.strip():
                    # Search for the query in memory
                    matches = []
                    for k, v in self.memory_store.items():
                        if query.lower() in k.lower() or query.lower() in str(v['value']).lower():
                            matches.append(f"  • {k}: {v['value']}")
                    
                    if matches:
                        return f"🔍 Memory search for '{query}' found {len(matches)} matches:\n" + "\n".join(matches)
                    else:
                        # Store as simple key-value
                        timestamp = datetime.now().isoformat()
                        key = f"note_{timestamp}"
                        self.memory_store[key] = {
                            "value": query,
                            "timestamp": timestamp,
                            "type": "note"
                        }
                        return f"📝 Stored note as '{key}': {query}"
                else:
                    return "❓ Empty query received"
        except Exception as e:
            return f"❌ Fallback memory error: {str(e)}"
    
    async def cancel(
        self, 
        context: RequestContext, 
        event_queue: EventQueue
    ) -> None:
        """Cancel the current memory operation."""
        logger.info("🛑 Memory operation cancelled")
        raise ServerError(error=UnsupportedOperationError())

# ═══════════════════════════════════════════════════════════════════════════════════
# A2A SERVER SETUP
# ═══════════════════════════════════════════════════════════════════════════════════

def get_agent_card(host: str, port: int) -> AgentCard:
    """Create the agent card for FinAgent Memory Server."""
    return AgentCard(
        name=SERVER_NAME,
        description="FinAgent Memory A2A Server for agent-to-agent communication and memory operations. Provides memory storage, retrieval, and search capabilities for trading agents.",
        url=f"http://{host}:{port}/",
        version=SERVER_VERSION,
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(
            input_modes=['text'],
            output_modes=['text'],
            streaming=True
        ),
        skills=[
            AgentSkill(
                id='memory_operations',
                name='Memory Operations',
                description='Store and retrieve trading memories, strategies, and agent communications using graph-based memory system',
                tags=['memory', 'storage', 'retrieval', 'trading'],
                examples=[
                    'Store this trading strategy in memory',
                    'Search for memories about Apple stock',
                    'What do you remember about previous market analysis?'
                ]
            ),
            AgentSkill(
                id='trading_signals',
                name='Trading Signal Storage',
                description='Process and store trading signals from other agents',
                tags=['trading', 'signals', 'storage'],
                examples=[
                    'Store this buy signal for AAPL',
                    'Remember this market sentiment analysis'
                ]
            ),
            AgentSkill(
                id='agent_communication',
                name='Agent Communication Memory',
                description='Facilitate memory-backed communication between trading agents',
                tags=['communication', 'agents', 'memory'],
                examples=[
                    'What did the alpha agent say about this stock?',
                    'Store this communication from portfolio agent'
                ]
            )
        ],
        examples=[
            'Store this trading analysis in memory',
            'What do you remember about Tesla stock performance?',
            'Search for previous risk assessments'
        ]
    )

# ═══════════════════════════════════════════════════════════════════════════════════
# SERVER LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════════

async def shutdown_server():
    """Cleanup server resources."""
    try:
        if interface_manager and MEMORY_AVAILABLE:
            await interface_manager.shutdown()
            logger.info("✅ Memory interface manager shutdown")
        
        logger.info("✅ A2A Server shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Server shutdown error: {e}")

def print_server_info():
    """Print server information."""
    print("\n" + "="*80)
    print("🚀 FINAGENT A2A SERVER - AGENT-TO-AGENT PROTOCOL")
    print("="*80)
    print(f"📋 Server Configuration:")
    print(f"   🏷️  Name: {SERVER_NAME}")
    print(f"   📦 Version: {SERVER_VERSION}")
    print(f"   🌐 Host: {SERVER_HOST}")
    print(f"   🚪 Port: {SERVER_PORT}")
    print(f"   🔧 Protocol: A2A (Agent-to-Agent)")
    print(f"   🧠 Memory: {'✅ Available' if MEMORY_AVAILABLE else '❌ Not Available'}")
    print("\n📡 Supported Operations:")
    print("   • Memory storage and retrieval")
    print("   • Trading signal processing")
    print("   • Agent communication memory")
    print("   • Knowledge graph operations")
    print("\n🔧 A2A Protocol Endpoints:")
    print(f"   • Agent Card: http://{SERVER_HOST}:{SERVER_PORT}/.well-known/agent-card")
    print(f"   • Health Check: http://{SERVER_HOST}:{SERVER_PORT}/health")
    print(f"   • Message Send: http://{SERVER_HOST}:{SERVER_PORT}/message/send")
    print(f"   • Message Stream: http://{SERVER_HOST}:{SERVER_PORT}/message/stream")
    print("\n🗄️ Database Configuration:")
    print(f"   📍 URI: {DATABASE_CONFIG['uri']}")
    print(f"   👤 User: {DATABASE_CONFIG['username']}")
    print(f"   🗄️  Database: {DATABASE_CONFIG['database']}")
    print("="*80)

# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════════

@click.command()
@click.option('--host', default=SERVER_HOST, help='Host to run the server on')
@click.option('--port', default=SERVER_PORT, help='Port to run the server on')
def main(host: str, port: int):
    """Main entry point for the A2A server."""
    
    print_server_info()
    
    try:
        # Initialize memory components if available
        global interface_manager
        if MEMORY_AVAILABLE:
            try:
                # We'll initialize this in the executor when first used
                logger.info("✅ Memory components available for initialization")
            except Exception as e:
                logger.warning(f"⚠️ Memory system setup warning: {e}")
        
        # Create request handler with memory agent executor
        request_handler = DefaultRequestHandler(
            agent_executor=MemoryAgentExecutor(),
            task_store=InMemoryTaskStore(),
        )
        
        # Create A2A server application
        server = A2AStarletteApplication(
            agent_card=get_agent_card(host, port),
            http_handler=request_handler
        )
        
        logger.info(f"✅ A2A Server configured successfully")
        logger.info(f"   📋 Name: {SERVER_NAME}")
        logger.info(f"   🔢 Version: {SERVER_VERSION}")
        logger.info(f"   🌐 Host: {host}")
        logger.info(f"   🚪 Port: {port}")
        logger.info(f"   🧠 Memory: {'✅ Available' if MEMORY_AVAILABLE else '❌ Unavailable'}")
        
        print(f"\n🚀 Starting FinAgent A2A Server...")
        print(f"   Use Ctrl+C to stop the server")
        print("="*80)
        
        # Run the server
        uvicorn.run(server.build(), host=host, port=port)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Server shutdown requested...")
        asyncio.run(shutdown_server())
        print("👋 FinAgent A2A Server stopped")
        
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        asyncio.run(shutdown_server())

if __name__ == "__main__":
    main()
