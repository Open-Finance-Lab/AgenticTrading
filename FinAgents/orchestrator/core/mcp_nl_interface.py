"""
MCP Natural Language Interface
Provides natural language interaction capabilities for all FinAgent components via MCP protocol
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import JSONRPCMessage, JSONRPCRequest, JSONRPCResponse

from .llm_integration import NaturalLanguageProcessor, ConversationManager, LLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPNaturalLanguageInterface:
    """
    MCP-based Natural Language Interface for FinAgent system
    Handles communication between natural language requests and agent pools
    """
    
    def __init__(self, orchestrator_config: Dict[str, Any]):
        self.config = orchestrator_config
        self.agent_pool_endpoints = self._extract_agent_endpoints()
        
        # Initialize LLM integration
        llm_config = LLMConfig(
            provider=self.config.get("llm", {}).get("provider", "openai"),
            model=self.config.get("llm", {}).get("model", "gpt-4"),
            temperature=self.config.get("llm", {}).get("temperature", 0.7)
        )
        
        self.nlp = NaturalLanguageProcessor(llm_config)
        self.conversation_manager = ConversationManager(self.nlp)
        
        # Initialize MCP server
        self.mcp_server = FastMCP("FinAgent-NL-Interface")
        self._register_nl_tools()
        
        # Agent pool clients
        self.agent_clients: Dict[str, httpx.AsyncClient] = {}
        
    def _extract_agent_endpoints(self) -> Dict[str, str]:
        """Extract agent pool endpoints from configuration"""
        endpoints = {}
        agent_pools = self.config.get("agent_pools", {})
        
        for pool_name, pool_config in agent_pools.items():
            if pool_config.get("enabled", True):
                endpoints[pool_name] = pool_config.get("url", "")
                
        return endpoints
    
    def _register_nl_tools(self):
        """Register natural language tools for MCP"""
        
        @self.mcp_server.tool(
            name="process_natural_language",
            description="Process natural language requests and execute corresponding actions"
        )
        async def process_natural_language(
            user_input: str,
            user_id: str = "default",
            context: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Process natural language input and execute corresponding actions
            
            Args:
                user_input: Natural language request from user
                user_id: Unique user identifier
                context: Additional context information
                
            Returns:
                Execution result with natural language response
            """
            try:
                # Get current system context
                system_context = await self._get_system_context()
                if context:
                    system_context.update(context)
                
                # Process natural language request
                nl_response = await self.conversation_manager.handle_user_message(
                    user_input, user_id, system_context
                )
                
                if not nl_response["success"]:
                    return nl_response
                
                # Execute the parsed action
                execution_result = await self._execute_parsed_action(
                    nl_response["response"]
                )
                
                # Combine results
                return {
                    "success": True,
                    "natural_language_response": nl_response["response"]["explanation"],
                    "action_executed": nl_response["response"]["action"],
                    "execution_result": execution_result,
                    "suggestions": nl_response["response"].get("suggestions", []),
                    "session_id": nl_response["session_id"]
                }
                
            except Exception as e:
                logger.error(f"Error processing natural language request: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "natural_language_response": f"I encountered an error: {str(e)}"
                }
        
        @self.mcp_server.tool(
            name="chat_with_system",
            description="Have a conversational interaction with the FinAgent system"
        )
        async def chat_with_system(
            message: str,
            session_id: Optional[str] = None,
            user_id: str = "default"
        ) -> Dict[str, Any]:
            """
            Conversational interface for the FinAgent system
            
            Args:
                message: User message
                session_id: Optional session ID to continue conversation
                user_id: User identifier
                
            Returns:
                Conversational response with system actions
            """
            try:
                system_context = await self._get_system_context()
                
                response = await self.conversation_manager.handle_user_message(
                    message, user_id, system_context
                )
                
                return {
                    "success": True,
                    "message": response["response"]["explanation"],
                    "intent": response["response"]["intent"],
                    "confidence": response["response"]["confidence"],
                    "suggestions": response["response"].get("suggestions", []),
                    "session_id": response["session_id"]
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "I'm having trouble understanding your request."
                }
        
        @self.mcp_server.tool(
            name="execute_strategy_from_description",
            description="Execute a trading strategy based on natural language description"
        )
        async def execute_strategy_from_description(
            strategy_description: str,
            portfolio_size: Optional[float] = None,
            risk_level: str = "medium"
        ) -> Dict[str, Any]:
            """
            Execute trading strategy from natural language description
            
            Args:
                strategy_description: Natural language description of strategy
                portfolio_size: Portfolio size for execution
                risk_level: Risk tolerance (low, medium, high)
                
            Returns:
                Strategy execution results
            """
            try:
                # Use NLP to parse strategy description
                nl_response = await self.nlp.process_natural_language_request(
                    f"Execute this strategy: {strategy_description}",
                    f"strategy_{datetime.now().timestamp()}"
                )
                
                if nl_response["success"]:
                    parsed_strategy = nl_response["response"]
                    
                    # Execute via orchestrator
                    execution_result = await self._call_orchestrator_api(
                        "execute_strategy",
                        {
                            "strategy_type": parsed_strategy.get("action", "momentum"),
                            "parameters": parsed_strategy.get("parameters", {}),
                            "portfolio_size": portfolio_size,
                            "risk_level": risk_level
                        }
                    )
                    
                    return {
                        "success": True,
                        "strategy_executed": parsed_strategy.get("action"),
                        "parameters_used": parsed_strategy.get("parameters"),
                        "execution_result": execution_result,
                        "natural_language_summary": f"Successfully executed {strategy_description}"
                    }
                
                return nl_response
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "natural_language_summary": f"Failed to execute strategy: {str(e)}"
                }
        
        @self.mcp_server.tool(
            name="get_system_status_summary",
            description="Get a natural language summary of the entire system status"
        )
        async def get_system_status_summary() -> Dict[str, Any]:
            """Get natural language summary of system status"""
            try:
                system_context = await self._get_system_context()
                
                # Generate natural language summary
                summary_request = "Provide a comprehensive status summary of all system components"
                nl_response = await self.nlp.process_natural_language_request(
                    summary_request,
                    f"status_{datetime.now().timestamp()}",
                    system_context
                )
                
                return {
                    "success": True,
                    "summary": nl_response["response"]["explanation"],
                    "system_health": system_context.get("system_health", "unknown"),
                    "active_components": list(system_context.get("agent_pools", {}).keys()),
                    "recommendations": nl_response["response"].get("suggestions", [])
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "summary": "Unable to generate system status summary"
                }
    
    async def _get_system_context(self) -> Dict[str, Any]:
        """Get current system context from all agent pools"""
        context = {
            "timestamp": datetime.now().isoformat(),
            "agent_pools": {},
            "system_health": "unknown"
        }
        
        # Check each agent pool
        for pool_name, endpoint in self.agent_pool_endpoints.items():
            try:
                pool_status = await self._check_agent_pool_health(pool_name, endpoint)
                context["agent_pools"][pool_name] = pool_status
            except Exception as e:
                context["agent_pools"][pool_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Determine overall system health
        healthy_pools = sum(1 for pool in context["agent_pools"].values() 
                          if pool.get("status") == "healthy")
        total_pools = len(context["agent_pools"])
        
        if healthy_pools == total_pools:
            context["system_health"] = "healthy"
        elif healthy_pools > total_pools / 2:
            context["system_health"] = "degraded"
        else:
            context["system_health"] = "critical"
        
        return context
    
    async def _check_agent_pool_health(self, pool_name: str, endpoint: str) -> Dict[str, Any]:
        """Check health of a specific agent pool"""
        try:
            if pool_name not in self.agent_clients:
                self.agent_clients[pool_name] = httpx.AsyncClient(timeout=10.0)
            
            client = self.agent_clients[pool_name]
            
            # Try to ping the health endpoint
            health_url = f"{endpoint}/health"
            response = await client.get(health_url)
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "endpoint": endpoint,
                    "response_time": response.elapsed.total_seconds(),
                    "details": response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                }
            else:
                return {
                    "status": "unhealthy",
                    "endpoint": endpoint,
                    "status_code": response.status_code
                }
                
        except Exception as e:
            return {
                "status": "unreachable",
                "endpoint": endpoint,
                "error": str(e)
            }
    
    async def _execute_parsed_action(self, parsed_response: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the action parsed from natural language"""
        intent = parsed_response.get("intent")
        action = parsed_response.get("action")
        parameters = parsed_response.get("parameters", {})
        
        try:
            if intent == "execute_strategy":
                return await self._call_orchestrator_api("execute_strategy", parameters)
            
            elif intent == "run_backtest":
                return await self._call_orchestrator_api("run_backtest", parameters)
            
            elif intent == "train_model":
                return await self._call_orchestrator_api("train_rl_model", parameters)
            
            elif intent == "system_status":
                return await self._get_system_context()
            
            elif intent == "manage_agents":
                return await self._manage_agent_pools(action, parameters)
            
            else:
                return {
                    "success": True,
                    "action": "information_provided",
                    "result": f"Provided guidance for {intent}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": action
            }
    
    async def _call_orchestrator_api(self, endpoint: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call orchestrator API"""
        try:
            orchestrator_url = self.config.get("orchestrator", {}).get("url", "http://localhost:9000")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{orchestrator_url}/api/{endpoint}",
                    json=parameters
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "success": False,
                        "error": f"Orchestrator API error: {response.status_code}",
                        "details": response.text
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to call orchestrator: {str(e)}"
            }
    
    async def _manage_agent_pools(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Manage agent pool operations"""
        try:
            results = {}
            
            if action == "check_agent_health":
                for pool_name, endpoint in self.agent_pool_endpoints.items():
                    results[pool_name] = await self._check_agent_pool_health(pool_name, endpoint)
            
            elif action == "restart_agents":
                # Implementation for restarting agents
                results["message"] = "Agent restart functionality would be implemented here"
            
            return {
                "success": True,
                "action": action,
                "results": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": action
            }
    
    async def start_server(self, host: str = "localhost", port: int = 8020):
        """Start the MCP natural language interface server"""
        try:
            logger.info(f"Starting MCP Natural Language Interface on {host}:{port}")
            await self.mcp_server.start(host, port)
        except Exception as e:
            logger.error(f"Failed to start MCP NL Interface: {e}")
            raise
    
    async def stop_server(self):
        """Stop the MCP server and cleanup"""
        try:
            # Close all agent clients
            for client in self.agent_clients.values():
                await client.aclose()
            
            logger.info("MCP Natural Language Interface stopped")
        except Exception as e:
            logger.error(f"Error stopping MCP NL Interface: {e}")


# Standalone server for testing
async def main():
    """Test the MCP Natural Language Interface"""
    
    # Sample configuration
    config = {
        "agent_pools": {
            "data_agent_pool": {
                "url": "http://localhost:8001",
                "enabled": True
            },
            "alpha_agent_pool": {
                "url": "http://localhost:5050",
                "enabled": True
            },
            "risk_agent_pool": {
                "url": "http://localhost:7000",
                "enabled": True
            },
            "transaction_cost_agent_pool": {
                "url": "http://localhost:6000",
                "enabled": True
            }
        },
        "orchestrator": {
            "url": "http://localhost:9000"
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7
        }
    }
    
    # Initialize interface
    nl_interface = MCPNaturalLanguageInterface(config)
    
    # Test natural language processing
    test_messages = [
        "Execute a momentum strategy for AAPL and GOOGL",
        "What's the status of all agent pools?",
        "Run a backtest for the last 6 months",
        "Help me train a new reinforcement learning model"
    ]
    
    print("🤖 Testing MCP Natural Language Interface")
    print("=" * 50)
    
    for message in test_messages:
        print(f"\n👤 User: {message}")
        
        try:
            # Simulate MCP tool call
            system_context = await nl_interface._get_system_context()
            
            response = await nl_interface.conversation_manager.handle_user_message(
                message, "test_user", system_context
            )
            
            print(f"🤖 Intent: {response['response']['intent']}")
            print(f"🎯 Action: {response['response']['action']}")
            print(f"📝 Response: {response['response']['explanation']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ MCP Natural Language Interface test completed!")

if __name__ == "__main__":
    asyncio.run(main())
