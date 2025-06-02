from datetime import datetime, UTC
from typing import Dict, Any, Optional
import logging
from mcp.server.fastmcp import FastMCP, Context
from .registry import alpha_registry, AlphaAgent
from .schema.agent_config import AlphaAgentConfig
from .schema.config_schema import config_manager
from .agents.technical_agent import TechnicalAlphaAgent
from .agents.event_agent import EventAlphaAgent
from .agents.ml_agent import MLAlphaAgent

logger = logging.getLogger(__name__)

# Create FastMCP instance
mcp = FastMCP("Alpha Agent Pool", stateless_http=True)

# Register all agent classes to the registry
alpha_registry.register("technical_agent", TechnicalAlphaAgent)
alpha_registry.register("event_agent", EventAlphaAgent)
alpha_registry.register("ml_agent", MLAlphaAgent)

# Initialize agents on startup
def initialize_agents():
    """Initialize all configured agents on server startup"""
    try:
        agent_configs = config_manager.get_agent_configs()
        for agent_id, config in agent_configs.items():
            try:
                agent = alpha_registry.create_agent(config)
                logger.info(f"Successfully initialized agent: {agent_id}")
            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_id}: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to load agent configurations: {str(e)}")
        raise

# Initialize agents when server starts
initialize_agents()

# -----------------------------------------------------------------------------
# Tool: agent.execute
# Executes a registered alpha agent's function and returns its result
# -----------------------------------------------------------------------------
@mcp.tool(
    name="agent.execute",
    description="Execute a function on a registered alpha agent"
)
async def agent_execute(
    agent_id: str,
    function: str,
    input: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute a function on a registered alpha agent"""
    agent = alpha_registry.get_agent_instance(agent_id)
    if not agent:
        raise ValueError(f"Unknown agent_id: {agent_id}")
        
    try:
        if function == "generate_alpha":
            result = await agent.generate_alpha(input)
            is_valid = await agent.validate_signal(result)
            return {
                "result": result,
                "meta": {
                    "status": "success",
                    "is_valid": is_valid,
                    "agent_type": agent.config.agent_type.value,
                    "data_sources": agent.config.data_sources,
                    "timestamp": datetime.now(UTC).isoformat()
                }
            }
        else:
            raise ValueError(f"Unsupported function: {function}")
            
    except Exception as e:
        logger.error(f"Execution failed for {agent_id}.{function}: {str(e)}")
        raise RuntimeError(f"Execution failed: {str(e)}")

# -----------------------------------------------------------------------------
# Resource: agent.capabilities
# Returns the capabilities of a registered agent
# -----------------------------------------------------------------------------
@mcp.resource("agent://{agent_id}/capabilities")
async def get_agent_capabilities(agent_id: str) -> Dict[str, Any]:
    """Get capabilities of a registered agent"""
    agent = alpha_registry.get_agent_instance(agent_id)
    if not agent:
        raise ValueError(f"Unknown agent_id: {agent_id}")
        
    return {
        "agent_id": agent_id,
        "agent_type": agent.config.agent_type.value,
        "description": agent.config.description,
        "data_sources": agent.config.data_sources,
        "parameters": agent.config.parameters,
        "signal_rules": agent.config.signal_rules,
        "risk_parameters": agent.config.risk_parameters
    }

# -----------------------------------------------------------------------------
# Resource: agent.list
# Returns list of all registered agents
# -----------------------------------------------------------------------------
@mcp.resource("agent://list")
async def list_agents() -> Dict[str, Any]:
    """List all registered alpha agents"""
    agents = alpha_registry.list_agents()
    return {
        "agents": [
            {
                "agent_id": agent_id,
                "agent_type": agent_type.value
            }
            for agent_id, agent_type in agents.items()
        ],
        "total": len(agents)
    }

# -----------------------------------------------------------------------------
# Tool: agent.register
# Register a new alpha agent
# -----------------------------------------------------------------------------
@mcp.tool(
    name="agent.register",
    description="Register a new alpha agent"
)
async def register_agent(config: Dict[str, Any]) -> Dict[str, Any]:
    """Register a new alpha agent"""
    try:
        agent_config = AlphaAgentConfig(**config)
        agent = alpha_registry.create_agent(agent_config)
        return {
            "result": {
                "agent_id": agent.config.agent_id,
                "status": "registered"
            },
            "meta": {
                "timestamp": datetime.now(UTC).isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Agent registration failed: {str(e)}")
        raise RuntimeError(f"Registration failed: {str(e)}")

# -----------------------------------------------------------------------------
# Tool: agent.heartbeat
# Periodic health check for agents
# -----------------------------------------------------------------------------
@mcp.tool(
    name="agent.heartbeat",
    description="Send heartbeat for agent health check"
)
async def agent_heartbeat(agent_id: str) -> Dict[str, Any]:
    """Send heartbeat for agent health check"""
    agent = alpha_registry.get_agent_instance(agent_id)
    if not agent:
        raise ValueError(f"Unknown agent_id: {agent_id}")
        
    return {
        "result": {
            "agent_id": agent_id,
            "status": "active"
        },
        "meta": {
            "timestamp": datetime.now(UTC).isoformat()
        }
    }
    
app = mcp.streamable_http_app()
# Server entry point
if __name__ == "__main__":
    logger.info("Starting Alpha Agent Pool MCP Server...")
    agent_configs = config_manager.get_agent_configs()
    logger.info(f"Initialized {len(agent_configs)} agents")
    mcp.run() 