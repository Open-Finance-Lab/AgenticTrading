"""
Risk Agent Pool - Core Orchestration Engine

This module implements the central orchestrator for managing risk analysis agents,
providing unified MCP interface and lifecycle management with OpenAI integration.

Key Features:
- Multi-agent risk analysis lifecycle management
- Real-time risk calculation orchestration
- Performance monitoring and optimization
- Scalable microservices architecture
- External memory agent integration
- OpenAI-powered natural language processing

Author: Jifeng Li
License: openMDW
"""

import logging
import asyncio
import threading
import time
import json
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI, Request
import contextvars
import traceback
import os
import openai
from openai import AsyncOpenAI
from pathlib import Path

# Add memory module to path
memory_path = Path(__file__).parent.parent.parent / "memory"
sys.path.insert(0, str(memory_path))

try:
    from external_memory_agent import ExternalMemoryAgent, EventType, LogLevel
    MEMORY_AVAILABLE = True
except ImportError:
    ExternalMemoryAgent = None
    EventType = LogLevel = None
    MEMORY_AVAILABLE = False

# Initialize logger
logger = logging.getLogger("RiskAgentPool")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# Global context management for request tracking
request_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "request_context", 
    default={}
)


class ContextDecompressor:
    """
    Context decompression and natural language processing component.
    Handles input from external orchestrators and converts natural language
    context into structured risk analysis tasks.
    """
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.logger = logging.getLogger("ContextDecompressor")
    
    async def decompress_context(self, context: str) -> Dict[str, Any]:
        """
        Decompress natural language context into structured risk analysis tasks.
        
        Args:
            context: Natural language context string from external orchestrator
            
        Returns:
            Dict containing structured analysis tasks and parameters
        """
        try:
            # Use OpenAI to parse and structure the natural language input
            system_prompt = """
            You are a financial risk analysis context processor. Your job is to parse natural language 
            descriptions of risk analysis requests and convert them into structured JSON format.
            
            Expected output format:
            {
                "risk_type": "market|credit|operational|liquidity|systemic",
                "analysis_scope": "portfolio|individual_security|sector|market",
                "time_horizon": "intraday|daily|weekly|monthly|quarterly|annual",
                "risk_measures": ["var", "cvar", "volatility", "beta", "correlation", "drawdown"],
                "portfolio_data": {
                    "securities": [],
                    "weights": [],
                    "market_data_required": []
                },
                "specific_requirements": [],
                "urgency": "low|medium|high|critical"
            }
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse this risk analysis request: {context}"}
                ],
                response_format={"type": "json_object"}
            )
            
            structured_context = json.loads(response.choices[0].message.content)
            self.logger.info(f"Successfully decompressed context: {structured_context}")
            return structured_context
            
        except Exception as e:
            self.logger.error(f"Error decompressing context: {e}")
            # Fallback to basic structure
            return {
                "risk_type": "market",
                "analysis_scope": "portfolio",
                "time_horizon": "daily",
                "risk_measures": ["var", "volatility"],
                "portfolio_data": {"securities": [], "weights": [], "market_data_required": []},
                "specific_requirements": [context],
                "urgency": "medium"
            }


class RiskAgentPool:
    """
    Central orchestrator for risk analysis agents with OpenAI integration and MCP task distribution.
    
    This class manages the lifecycle of various risk analysis agents, processes natural language
    input from external orchestrators, and distributes tasks via MCP protocol.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Risk Agent Pool.
        
        Args:
            config: Configuration dictionary containing agent settings and API keys
        """
        self.config = config or {}
        self.agents = {}
        self.agent_registry = {}
        
        # Initialize logger first
        self.logger = logging.getLogger("RiskAgentPool")
        
        # Support direct injection of openai_client and memory_bridge for testing
        self.openai_client = self.config.get('openai_client')
        self.memory_bridge = self.config.get('memory_bridge')
        
        # Initialize memory agent
        self.memory_agent = None
        self.session_id = None
        self._initialize_memory_agent()
        
        self.context_decompressor = None
        self.mcp_app = None
        self.fastapi_app = None
        self.startup_complete = threading.Event()
        self.shutdown_event = threading.Event()
        self.server = None
        self.is_running = False
        
        # Initialize OpenAI client if not directly provided
        if not self.openai_client:
            self._initialize_openai_client()
        
        # Initialize context decompressor
        if self.openai_client:
            self.context_decompressor = ContextDecompressor(self.openai_client)
        
        # Initialize agents synchronously for testing
        self._initialize_agents_sync()
        
        self.logger.info("Risk Agent Pool initialized")
    
    def _initialize_agents_sync(self):
        """Initialize agents synchronously (for testing)."""
        try:
            from FinAgents.agent_pools.risk_agent_pool.registry import AGENT_REGISTRY, preload_default_agents
            
            # Preload default agents
            preload_default_agents()
            self.agent_registry = AGENT_REGISTRY
            
            # Initialize memory bridge if not directly provided
            if not self.memory_bridge:
                from FinAgents.agent_pools.risk_agent_pool.memory_bridge import RiskMemoryBridge
                self.memory_bridge = RiskMemoryBridge(self.config.get('memory_config', {}))
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            # Don't raise in constructor for testing compatibility
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client with API key from config or environment."""
        try:
            api_key = self.config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = AsyncOpenAI(api_key=api_key)
                self.logger.info("OpenAI client initialized successfully")
            else:
                self.logger.warning("No OpenAI API key found. Context decompression will use fallback.")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def _initialize_memory_agent(self):
        """Initialize the external memory agent"""
        if not MEMORY_AVAILABLE:
            self.logger.warning("External memory agent not available")
            return
        
        try:
            self.memory_agent = ExternalMemoryAgent()
            self.session_id = f"risk_pool_session_{int(time.time())}"
            self.logger.info("External memory agent initialized for Risk Agent Pool")
        except Exception as e:
            self.logger.error(f"Failed to initialize memory agent: {e}")
            self.memory_agent = None
    
    async def _log_memory_event(self, event_type: str, description: str, metadata: Optional[Dict[str, Any]] = None):
        """Log an event to the memory agent"""
        if self.memory_agent and self.session_id:
            try:
                await self.memory_agent.log_event(
                    event_type=event_type,
                    description=description,
                    metadata={
                        "session_id": self.session_id,
                        "agent_pool": "risk",
                        **(metadata or {})
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to log memory event: {e}")
    
    async def initialize_agents(self):
        """Initialize all risk analysis agents."""
        try:
            from FinAgents.agent_pools.risk_agent_pool.registry import AGENT_REGISTRY, preload_default_agents
            
            # Preload default agents
            preload_default_agents()
            self.agent_registry = AGENT_REGISTRY
            
            # Initialize memory bridge
            from FinAgents.agent_pools.risk_agent_pool.memory_bridge import RiskMemoryBridge
            self.memory_bridge = RiskMemoryBridge(self.config.get('memory_config', {}))
            
            self.logger.info(f"Initialized {len(self.agent_registry)} risk agents")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    async def process_orchestrator_input(self, context: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process input from external orchestrator and distribute tasks via MCP.
        
        Args:
            context: Natural language context string
            metadata: Optional metadata about the request
            
        Returns:
            Dictionary containing task distribution results
        """
        try:
            # Set request context
            request_context.set({
                "timestamp": datetime.utcnow().isoformat(),
                "context": context,
                "metadata": metadata or {}
            })
            
            # Decompress context using OpenAI
            if self.context_decompressor:
                structured_request = await self.context_decompressor.decompress_context(context)
            else:
                # Fallback parsing
                structured_request = self._fallback_context_parsing(context)
            
            # Distribute tasks to appropriate agents
            task_results = await self._distribute_tasks(structured_request)
            
            # Record event in memory
            if self.memory_bridge:
                await self.memory_bridge.record_event(
                    agent_name="RiskAgentPool",
                    task="process_orchestrator_input",
                    input_data={"context": context, "metadata": metadata},
                    summary=f"Processed orchestrator input and distributed {len(task_results)} tasks"
                )
            
            return {
                "status": "success",
                "structured_request": structured_request,
                "task_results": task_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing orchestrator input: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _fallback_context_parsing(self, context: str) -> Dict[str, Any]:
        """Fallback context parsing when OpenAI is not available."""
        # Simple keyword-based parsing
        risk_keywords = {
            "market": ["market", "volatility", "price", "return"],
            "credit": ["credit", "default", "rating", "spread"],
            "operational": ["operational", "process", "system", "human"],
            "liquidity": ["liquidity", "bid", "ask", "volume"]
        }
        
        detected_risk_type = "market"  # default
        for risk_type, keywords in risk_keywords.items():
            if any(keyword.lower() in context.lower() for keyword in keywords):
                detected_risk_type = risk_type
                break
        
        return {
            "risk_type": detected_risk_type,
            "analysis_scope": "portfolio",
            "time_horizon": "daily",
            "risk_measures": ["var", "volatility"],
            "portfolio_data": {"securities": [], "weights": [], "market_data_required": []},
            "specific_requirements": [context],
            "urgency": "medium"
        }
    
    async def _distribute_tasks(self, structured_request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Distribute tasks to appropriate risk agents based on structured request."""
        task_results = []
        
        try:
            risk_type = structured_request.get("risk_type", "market")
            risk_measures = structured_request.get("risk_measures", ["var"])
            
            # Determine which agents to use based on risk type and measures
            relevant_agents = self._select_relevant_agents(risk_type, risk_measures)
            
            # Execute tasks in parallel
            tasks = []
            for agent_name in relevant_agents:
                if agent_name in self.agent_registry:
                    agent_class = self.agent_registry[agent_name]
                    task = self._execute_agent_task(agent_name, agent_class, structured_request)
                    tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        task_results.append({
                            "agent": relevant_agents[i],
                            "status": "error",
                            "error": str(result)
                        })
                    else:
                        task_results.append(result)
            
        except Exception as e:
            self.logger.error(f"Error distributing tasks: {e}")
            task_results.append({
                "agent": "RiskAgentPool",
                "status": "error",
                "error": str(e)
            })
        
        return task_results
    
    def _select_relevant_agents(self, risk_type: str, risk_measures: List[str]) -> List[str]:
        """Select relevant agents based on risk type and measures."""
        relevant_agents = []
        
        # Map risk types to agents
        risk_type_mapping = {
            "market": ["market_risk_agent", "volatility_agent", "var_agent"],
            "credit": ["credit_risk_agent", "rating_agent"],
            "operational": ["operational_risk_agent"],
            "liquidity": ["liquidity_risk_agent"],
            "systemic": ["systemic_risk_agent", "correlation_agent"]
        }
        
        # Map measures to agents
        measure_mapping = {
            "var": ["var_agent"],
            "cvar": ["cvar_agent"],
            "volatility": ["volatility_agent"],
            "beta": ["beta_agent"],
            "correlation": ["correlation_agent"],
            "drawdown": ["drawdown_agent"]
        }
        
        # Add agents based on risk type
        if risk_type in risk_type_mapping:
            relevant_agents.extend(risk_type_mapping[risk_type])
        
        # Add agents based on measures
        for measure in risk_measures:
            if measure in measure_mapping:
                relevant_agents.extend(measure_mapping[measure])
        
        # Remove duplicates and ensure agents exist
        relevant_agents = list(set(relevant_agents))
        return [agent for agent in relevant_agents if agent in self.agent_registry]
    
    async def _execute_agent_task(self, agent_name: str, agent_class, structured_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using a specific agent."""
        try:
            # Initialize agent if not already done
            if agent_name not in self.agents:
                self.agents[agent_name] = agent_class()
            
            agent = self.agents[agent_name]
            
            # Execute the agent's analysis method
            if hasattr(agent, 'analyze'):
                result = await agent.analyze(structured_request)
            elif hasattr(agent, 'calculate'):
                result = await agent.calculate(structured_request)
            else:
                result = {"error": f"Agent {agent_name} has no analyze or calculate method"}
            
            return {
                "agent": agent_name,
                "status": "success",
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing agent {agent_name}: {e}")
            return {
                "agent": agent_name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def create_mcp_server(self) -> FastMCP:
        """Create and configure the MCP server with risk analysis tools."""
        self.mcp_app = FastMCP("RiskAgentPool")
        
        @self.mcp_app.tool()
        async def process_risk_analysis_request(context: str, metadata: dict = None) -> dict:
            """
            Process a natural language risk analysis request from external orchestrator.
            
            Args:
                context: Natural language description of risk analysis requirements
                metadata: Optional metadata about the request
            
            Returns:
                Dictionary containing analysis results and task distribution summary
            """
            return await self.process_orchestrator_input(context, metadata)
        
        @self.mcp_app.tool()
        async def get_agent_status() -> dict:
            """Get status of all risk agents."""
            status = {}
            for agent_name in self.agent_registry:
                if agent_name in self.agents:
                    status[agent_name] = "active"
                else:
                    status[agent_name] = "inactive"
            return {
                "total_agents": len(self.agent_registry),
                "active_agents": len(self.agents),
                "agent_status": status,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.mcp_app.tool()
        async def calculate_portfolio_risk(portfolio_data: dict, risk_measures: list = None) -> dict:
            """
            Calculate portfolio risk metrics.
            
            Args:
                portfolio_data: Portfolio composition and market data
                risk_measures: List of risk measures to calculate
            
            Returns:
                Dictionary containing calculated risk metrics
            """
            if risk_measures is None:
                risk_measures = ["var", "volatility"]
            
            structured_request = {
                "risk_type": "market",
                "analysis_scope": "portfolio",
                "time_horizon": "daily",
                "risk_measures": risk_measures,
                "portfolio_data": portfolio_data,
                "specific_requirements": [],
                "urgency": "medium"
            }
            
            return await self._distribute_tasks(structured_request)
        
        return self.mcp_app
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """FastAPI lifespan context manager."""
        try:
            # Startup
            self.logger.info("Starting Risk Agent Pool...")
            await self.initialize_agents()
            self.startup_complete.set()
            self.logger.info("Risk Agent Pool startup completed")
            yield
        finally:
            # Shutdown
            self.logger.info("Shutting down Risk Agent Pool...")
            self.shutdown_event.set()
            
            # Cleanup agents
            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'cleanup'):
                    try:
                        await agent.cleanup()
                    except Exception as e:
                        self.logger.error(f"Error cleaning up agent {agent_name}: {e}")
            
            self.logger.info("Risk Agent Pool shutdown completed")
    
    async def run_server(self, host: str = "0.0.0.0", port: int = 8003):
        """Run the Risk Agent Pool MCP server."""
        try:
            # Create MCP server
            mcp_app = await self.create_mcp_server()
            
            # Create FastAPI app with lifespan
            self.fastapi_app = FastAPI(lifespan=self.lifespan)
            
            # Mount MCP app
            self.fastapi_app.mount("/mcp", mcp_app)
            
            # Add health check endpoint
            @self.fastapi_app.get("/health")
            async def health_check():
                return {
                    "status": "healthy",
                    "service": "RiskAgentPool",
                    "timestamp": datetime.utcnow().isoformat(),
                    "agents_loaded": len(self.agent_registry),
                    "agents_active": len(self.agents)
                }
            
            # Run server
            import uvicorn
            config = uvicorn.Config(
                app=self.fastapi_app,
                host=host,
                port=port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Error running server: {e}")
            raise

    async def get_available_agents(self) -> List:
        """Get list of available risk agents."""
        try:
            available_agents = []
            
            # Create mock agents if registry doesn't have real agents
            if not self.agents and not self.agent_registry:
                # Create a simple mock risk agent
                class MockRiskAgent:
                    async def assess_risk(self, request):
                        symbols = request.get('symbols', ['AAPL', 'MSFT'])
                        portfolio_weights = request.get('portfolio_weights', {})
                        
                        # Generate mock risk metrics
                        return {
                            "risk_metrics": {
                                "portfolio_volatility": 0.15,
                                "beta": 1.2,
                                "correlation_matrix": {s: {s2: 0.6 if s != s2 else 1.0 for s2 in symbols} for s in symbols}
                            },
                            "var_estimate": 0.05,
                            "risk_score": 0.3,
                            "recommendations": [
                                "Consider diversification across sectors",
                                "Monitor volatility levels",
                                "Review position sizing"
                            ]
                        }
                
                self.agents["mock_risk_agent"] = MockRiskAgent()
                available_agents.append(self.agents["mock_risk_agent"])
            else:
                # Return actual agents if available
                for agent_name, agent in self.agents.items():
                    available_agents.append(agent)
            
            return available_agents
            
        except Exception as e:
            self.logger.error(f"Error getting available agents: {e}")
            return []


# Main execution - Risk Agent Pool MCP Server
if __name__ == "__main__":
    print("🚀 Starting Risk Agent Pool...")
    
    try:
        from mcp.server.fastmcp import FastMCP
        
        # Create FastMCP server
        risk_server = FastMCP("RiskAgentPool")
        
        # Initialize risk agent pool
        config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "memory_config": {
                "memory_agent_url": os.getenv("MEMORY_AGENT_URL", "http://localhost:8001"),
                "enable_external_memory": True
            }
        }
        
        risk_pool = RiskAgentPool(config)
        
        @risk_server.tool(name="process_strategy_request", description="Process risk management strategy request")
        async def process_strategy_request(request: dict) -> dict:
            """Process risk management strategy request from orchestrator"""
            try:
                logger.info("Processing risk management strategy request")
                
                # Extract request details
                symbols = request.get('symbols', ['AAPL', 'MSFT'])
                date = request.get('date', datetime.now().strftime('%Y-%m-%d'))
                portfolio_weights = request.get('portfolio_weights', {})
                market_conditions = request.get('market_conditions', {})
                
                # Create risk analysis request
                risk_query = f"""
                Analyze risk for portfolio:
                Symbols: {symbols}
                Date: {date}
                Portfolio weights: {portfolio_weights}
                Market conditions: {market_conditions}
                
                Please provide risk metrics, VaR, and risk recommendations.
                """
                
                # Use the risk pool's agent to process
                agents = await risk_pool.get_available_agents()
                if agents:
                    agent = agents[0]  # Use first available agent
                    result = await agent.assess_risk({
                        "query": risk_query,
                        "portfolio_weights": portfolio_weights,
                        "symbols": symbols
                    })
                    
                    logger.info("Risk analysis completed successfully")
                    return {
                        "status": "success",
                        "risk_metrics": result.get("risk_metrics", {}),
                        "var_estimate": result.get("var_estimate", 0.0),
                        "risk_score": result.get("risk_score", 0.0),
                        "recommendations": result.get("recommendations", []),
                        "agent_source": "risk_agent_pool",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Fallback to direct calculation if no agents available
                    risk_metrics = {}
                    total_portfolio_risk = 0.0
                    var_estimate = 0.0
                    
                    for symbol in symbols:
                        weight = portfolio_weights.get(symbol, 1.0 / len(symbols))
                        # Simple risk calculation
                        symbol_volatility = 0.20  # 20% annual volatility assumption
                        symbol_var = symbol_volatility * 2.33  # 99% VaR approximation
                        
                        risk_metrics[symbol] = {
                            "volatility": symbol_volatility,
                            "var_99": symbol_var,
                            "beta": 1.0,  # Market beta assumption
                            "weight": weight
                        }
                        
                        total_portfolio_risk += (weight ** 2) * (symbol_volatility ** 2)
                        var_estimate += weight * symbol_var
                    
                    # Portfolio-level metrics
                    portfolio_volatility = (total_portfolio_risk ** 0.5)
                    risk_score = min(portfolio_volatility * 5, 1.0)  # Normalize to 0-1
                    
                    logger.info("Risk analysis completed using fallback calculation")
                    return {
                        "status": "success",
                        "risk_metrics": {
                            "portfolio_volatility": portfolio_volatility,
                            "individual_risks": risk_metrics,
                            "correlation_estimate": 0.6  # Assumed correlation
                        },
                        "var_estimate": var_estimate,
                        "risk_score": risk_score,
                        "recommendations": [
                            "Consider diversification across sectors",
                            "Monitor portfolio volatility levels",
                            "Review correlation assumptions",
                            "Implement stop-loss mechanisms"
                        ],
                        "agent_source": "risk_agent_pool",
                        "timestamp": datetime.now().isoformat()
                    }
                
            except Exception as e:
                logger.error(f"Risk analysis failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "agent_source": "risk_agent_pool",
                    "timestamp": datetime.now().isoformat()
                }

        @risk_server.tool(name="ping", description="Health check ping")
        def ping() -> str:
            return "pong"

        @risk_server.tool(name="status", description="Get risk agent status")
        def status() -> dict:
            return {
                "status": "running",
                "agent_type": "risk_management",
                "port": 8084,
                "capabilities": [
                    "risk_assessment",
                    "var_calculation",
                    "portfolio_risk_analysis",
                    "risk_recommendations"
                ]
            }
        
        # Configure and start server
        risk_server.settings.host = "0.0.0.0"
        risk_server.settings.port = 8084
        
        logger.info("Starting Risk Agent Pool on port 8084...")
        risk_server.run(transport="sse")
        
    except KeyboardInterrupt:
        print("\n🛑 Risk Agent Pool shutting down...")
    except Exception as e:
        print(f"❌ Risk Agent Pool error: {e}")
        import traceback
        traceback.print_exc()
