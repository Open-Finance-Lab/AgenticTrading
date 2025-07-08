"""
Transaction Cost Agent Pool - Core Orchestration Engine

This module implements the central orchestrator for managing transaction cost
analysis agents, providing unified MCP interface and lifecycle management.

Key Features:
- Multi-agent lifecycle management
- Real-time cost calculation orchestration
- Performance monitoring and optimization
- Scalable microservices architecture
- External memory agent integration

Author: FinAgent Development Team
License: OpenMDW
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
logger = logging.getLogger("TransactionCostAgentPool")
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


class TransactionCostAgentPool:
    """
    Central orchestrator for transaction cost analysis agents.
    
    This class manages the lifecycle, coordination, and unified access to all
    transaction cost agents within the FinAgent ecosystem. It provides:
    
    - Agent lifecycle management (initialization, start, stop, health monitoring)
    - Unified MCP protocol interface for external orchestration
    - Intelligent request routing and load balancing
    - Performance monitoring and optimization
    - Real-time cost calculation coordination
    - External memory agent integration
    
    The pool operates in a microservices architecture, enabling horizontal
    scaling and fault tolerance across multiple agent instances.
    """

    def __init__(self, pool_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Transaction Cost Agent Pool.
        
        Args:
            pool_id (str): Unique identifier for this pool instance
            config (Optional[Dict[str, Any]]): Configuration parameters
        """
        self.pool_id = pool_id
        self.config = config or {}
        
        # Agent management structures
        self.agents = {
            "pre_trade": {},      # Pre-trade cost estimation agents
            "post_trade": {},     # Post-trade analysis agents  
            "optimization": {},   # Cost optimization agents
            "risk_adjusted": {}   # Risk-adjusted cost agents
        }
        
        # Operational state tracking
        self.agent_threads = {}       # Running agent MCP server threads
        self.agent_adapters = {}      # MCP adapter instances
        self.agent_status = {}        # Agent lifecycle status
        self.performance_metrics = {} # Agent performance tracking
        
        # Initialize MCP server
        self.mcp = FastMCP(f"TransactionCostAgentPool-{pool_id}", stateless_http=True)
        
        # Register MCP endpoints
        self._register_mcp_endpoints()
        
        # Initialize memory agent
        self.memory_agent = None
        self.session_id = None
        self._initialize_memory_agent()

        logger.info(f"Initialized Transaction Cost Agent Pool with ID: {pool_id}")
        logger.info(f"Configuration parameters: {len(self.config)} loaded")

    def _initialize_memory_agent(self):
        """Initialize the external memory agent"""
        if not MEMORY_AVAILABLE:
            logger.warning("External memory agent not available")
            return
        
        try:
            self.memory_agent = ExternalMemoryAgent()
            self.session_id = f"transaction_cost_pool_session_{int(time.time())}"
            logger.info("External memory agent initialized for Transaction Cost Agent Pool")
        except Exception as e:
            logger.error(f"Failed to initialize memory agent: {e}")
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
                        "agent_pool": "transaction_cost",
                        **(metadata or {})
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log memory event: {e}")
    
    def _register_mcp_endpoints(self):
        """
        Register MCP protocol tools for external orchestration and management.
        
        This method exposes the core functionality of the agent pool through
        standardized MCP tools that can be invoked by external orchestrators.
        """

        @self.mcp.tool(
            name="estimate_transaction_cost",
            description="Estimate transaction costs for a given trade specification"
        )
        def estimate_transaction_cost(
            symbol: str,
            quantity: float,
            side: str,
            order_type: str = "market",
            venue: Optional[str] = None,
            market_conditions: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Perform pre-trade transaction cost estimation.
            
            Args:
                symbol (str): Trading symbol (e.g., 'AAPL', 'GOOGL')
                quantity (float): Trade quantity in shares
                side (str): Trade side ('buy' or 'sell')
                order_type (str): Order type ('market', 'limit', 'iceberg')
                venue (Optional[str]): Target venue for execution
                market_conditions (Optional[Dict]): Current market state
            
            Returns:
                Dict[str, Any]: Comprehensive cost estimation breakdown
            """
            try:
                # Set request context
                context = {
                    "operation": "estimate_transaction_cost",
                    "symbol": symbol,
                    "quantity": quantity,
                    "timestamp": datetime.utcnow().isoformat()
                }
                request_context.set(context)
                
                # Route to appropriate pre-trade agent
                if "cost_predictor" in self.agents["pre_trade"]:
                    agent = self.agents["pre_trade"]["cost_predictor"]
                    result = agent.estimate_costs(
                        symbol=symbol,
                        quantity=quantity,
                        side=side,
                        order_type=order_type,
                        venue=venue,
                        market_conditions=market_conditions or {}
                    )
                    
                    # Track performance metrics
                    self._update_performance_metrics("cost_estimation", result)
                    
                    return {
                        "success": True,
                        "cost_estimate": result,
                        "timestamp": datetime.utcnow().isoformat(),
                        "agent_id": "cost_predictor"
                    }
                else:
                    return {
                        "success": False,
                        "error": "Cost predictor agent not available",
                        "available_agents": list(self.agents["pre_trade"].keys())
                    }
                    
            except Exception as e:
                logger.error(f"Cost estimation failed: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }

        @self.mcp.tool(
            name="analyze_execution_quality",
            description="Analyze post-trade execution quality and cost attribution"
        )
        def analyze_execution_quality(
            trade_id: str,
            execution_data: Dict[str, Any],
            benchmark_type: str = "twap"
        ) -> Dict[str, Any]:
            """
            Perform comprehensive post-trade execution analysis.
            
            Args:
                trade_id (str): Unique trade identifier
                execution_data (Dict[str, Any]): Trade execution details
                benchmark_type (str): Benchmark for comparison ('twap', 'vwap', 'arrival')
            
            Returns:
                Dict[str, Any]: Detailed execution quality analysis
            """
            try:
                context = {
                    "operation": "analyze_execution_quality",
                    "trade_id": trade_id,
                    "benchmark_type": benchmark_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
                request_context.set(context)
                
                if "execution_analyzer" in self.agents["post_trade"]:
                    agent = self.agents["post_trade"]["execution_analyzer"]
                    analysis = agent.analyze_execution(
                        trade_id=trade_id,
                        execution_data=execution_data,
                        benchmark_type=benchmark_type
                    )
                    
                    # Update performance tracking
                    self._update_performance_metrics("execution_analysis", analysis)
                    
                    return {
                        "success": True,
                        "analysis": analysis,
                        "timestamp": datetime.utcnow().isoformat(),
                        "trade_id": trade_id
                    }
                else:
                    return {
                        "success": False,
                        "error": "Execution analyzer not available",
                        "available_agents": list(self.agents["post_trade"].keys())
                    }
                    
            except Exception as e:
                logger.error(f"Execution analysis failed: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "trade_id": trade_id
                }

        @self.mcp.tool(
            name="optimize_portfolio_execution",
            description="Optimize portfolio-level execution to minimize transaction costs"
        )
        def optimize_portfolio_execution(
            portfolio_trades: List[Dict[str, Any]],
            optimization_objective: str = "minimize_cost",
            constraints: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Optimize portfolio execution strategy for cost minimization.
            
            Args:
                portfolio_trades (List[Dict]): List of planned trades
                optimization_objective (str): Optimization target
                constraints (Optional[Dict]): Execution constraints
            
            Returns:
                Dict[str, Any]: Optimized execution strategy
            """
            try:
                context = {
                    "operation": "optimize_portfolio_execution",
                    "trade_count": len(portfolio_trades),
                    "objective": optimization_objective,
                    "timestamp": datetime.utcnow().isoformat()
                }
                request_context.set(context)
                
                if "portfolio_optimizer" in self.agents["optimization"]:
                    agent = self.agents["optimization"]["portfolio_optimizer"]
                    optimization = agent.optimize_execution(
                        portfolio_trades=portfolio_trades,
                        objective=optimization_objective,
                        constraints=constraints or {}
                    )
                    
                    # Track optimization performance
                    self._update_performance_metrics("portfolio_optimization", optimization)
                    
                    return {
                        "success": True,
                        "optimization": optimization,
                        "timestamp": datetime.utcnow().isoformat(),
                        "trade_count": len(portfolio_trades)
                    }
                else:
                    return {
                        "success": False,
                        "error": "Portfolio optimizer not available",
                        "available_agents": list(self.agents["optimization"].keys())
                    }
                    
            except Exception as e:
                logger.error(f"Portfolio optimization failed: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "trade_count": len(portfolio_trades)
                }

        @self.mcp.tool(
            name="calculate_risk_adjusted_costs",
            description="Calculate risk-adjusted transaction costs with VaR and volatility adjustments"
        )
        def calculate_risk_adjusted_costs(
            trades: List[Dict[str, Any]],
            risk_model: str = "parametric_var",
            confidence_level: float = 0.95
        ) -> Dict[str, Any]:
            """
            Calculate risk-adjusted transaction costs incorporating market risk.
            
            Args:
                trades (List[Dict]): Trade specifications
                risk_model (str): Risk model type ('parametric_var', 'historical_var')
                confidence_level (float): VaR confidence level
            
            Returns:
                Dict[str, Any]: Risk-adjusted cost analysis
            """
            try:
                context = {
                    "operation": "calculate_risk_adjusted_costs",
                    "trade_count": len(trades),
                    "risk_model": risk_model,
                    "confidence_level": confidence_level,
                    "timestamp": datetime.utcnow().isoformat()
                }
                request_context.set(context)
                
                if "var_adjusted_cost" in self.agents["risk_adjusted"]:
                    agent = self.agents["risk_adjusted"]["var_adjusted_cost"]
                    risk_analysis = agent.calculate_risk_adjusted_costs(
                        trades=trades,
                        risk_model=risk_model,
                        confidence_level=confidence_level
                    )
                    
                    # Update risk metrics
                    self._update_performance_metrics("risk_adjusted_analysis", risk_analysis)
                    
                    return {
                        "success": True,
                        "risk_analysis": risk_analysis,
                        "timestamp": datetime.utcnow().isoformat(),
                        "model_used": risk_model
                    }
                else:
                    return {
                        "success": False,
                        "error": "Risk-adjusted cost agent not available",
                        "available_agents": list(self.agents["risk_adjusted"].keys())
                    }
                    
            except Exception as e:
                logger.error(f"Risk-adjusted cost calculation failed: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "trade_count": len(trades)
                }

        @self.mcp.tool(
            name="get_agent_status",
            description="Retrieve comprehensive status information for all agents"
        )
        def get_agent_status() -> Dict[str, Any]:
            """
            Get comprehensive status information for all managed agents.
            
            Returns:
                Dict[str, Any]: Complete agent status and performance metrics
            """
            try:
                status_summary = {
                    "pool_id": self.pool_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_categories": {
                        category: {
                            "agent_count": len(agents),
                            "active_agents": [
                                agent_id for agent_id in agents.keys()
                                if self.agent_status.get(agent_id, {}).get("status") == "running"
                            ],
                            "agents": {
                                agent_id: {
                                    "status": self.agent_status.get(agent_id, {}).get("status", "unknown"),
                                    "last_activity": self.agent_status.get(agent_id, {}).get("last_activity"),
                                    "performance": self.performance_metrics.get(agent_id, {})
                                }
                                for agent_id in agents.keys()
                            }
                        }
                        for category, agents in self.agents.items()
                    },
                    "performance_summary": {
                        "total_requests_processed": sum(
                            metrics.get("request_count", 0) 
                            for metrics in self.performance_metrics.values()
                        ),
                        "average_response_time": self._calculate_average_response_time(),
                        "error_rate": self._calculate_error_rate()
                    }
                }
                
                return {
                    "success": True,
                    "status": status_summary
                }
                
            except Exception as e:
                logger.error(f"Status retrieval failed: {str(e)}")
                return {
                    "success": False,
                    "error": str(e)
                }

    def _update_performance_metrics(self, operation: str, result: Dict[str, Any]):
        """
        Update performance metrics for monitoring and optimization.
        
        Args:
            operation (str): Operation type performed
            result (Dict[str, Any]): Operation result for metrics extraction
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            
            if operation not in self.performance_metrics:
                self.performance_metrics[operation] = {
                    "request_count": 0,
                    "total_response_time": 0.0,
                    "error_count": 0,
                    "last_updated": timestamp
                }
            
            metrics = self.performance_metrics[operation]
            metrics["request_count"] += 1
            metrics["last_updated"] = timestamp
            
            # Track success/failure
            if result.get("success", True):
                response_time = result.get("response_time", 0.0)
                metrics["total_response_time"] += response_time
            else:
                metrics["error_count"] += 1
                
        except Exception as e:
            logger.warning(f"Performance metrics update failed: {str(e)}")

    def _calculate_average_response_time(self) -> float:
        """Calculate average response time across all operations."""
        try:
            total_time = sum(
                metrics.get("total_response_time", 0.0) 
                for metrics in self.performance_metrics.values()
            )
            total_requests = sum(
                metrics.get("request_count", 0) 
                for metrics in self.performance_metrics.values()
            )
            
            return total_time / total_requests if total_requests > 0 else 0.0
            
        except Exception:
            return 0.0

    def _calculate_error_rate(self) -> float:
        """Calculate overall error rate across all operations."""
        try:
            total_errors = sum(
                metrics.get("error_count", 0) 
                for metrics in self.performance_metrics.values()
            )
            total_requests = sum(
                metrics.get("request_count", 0) 
                for metrics in self.performance_metrics.values()
            )
            
            return (total_errors / total_requests * 100) if total_requests > 0 else 0.0
            
        except Exception:
            return 0.0

    def start_mcp_server(self, host: str = "0.0.0.0", port: int = 5060):
        """
        Start the MCP server for external orchestrator communication.
        
        Args:
            host (str): Server host address
            port (int): Server port number
        """
        try:
            logger.info(f"Starting Transaction Cost Agent Pool MCP server on {host}:{port}")
            self.mcp.settings.host = host
            self.mcp.settings.port = port
            self.mcp.run(transport="sse")
            
        except Exception as e:
            logger.error(f"MCP server startup failed: {str(e)}")
            raise

    def initialize_agent(self, agent_id: str, agent_config: Dict[str, Any]) -> bool:
        """
        Initialize a specific transaction cost agent.
        
        Args:
            agent_id (str): Unique agent identifier
            agent_config (Dict[str, Any]): Agent configuration parameters
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info(f"Initializing agent: {agent_id}")
            
            # Agent initialization logic will be implemented with specific agent classes
            # This is a placeholder for the agent initialization framework
            
            self.agent_status[agent_id] = {
                "status": "initialized",
                "initialized_at": datetime.utcnow().isoformat(),
                "config": agent_config
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Agent initialization failed for {agent_id}: {str(e)}")
            return False

    def shutdown(self):
        """
        Gracefully shutdown the agent pool and all managed agents.
        """
        try:
            logger.info(f"Shutting down Transaction Cost Agent Pool: {self.pool_id}")
            
            # Shutdown all agent threads
            for agent_id, thread in self.agent_threads.items():
                if thread.is_alive():
                    logger.info(f"Stopping agent thread: {agent_id}")
                    # Implement graceful shutdown logic
            
            # Update status
            for agent_id in self.agent_status:
                self.agent_status[agent_id]["status"] = "shutdown"
                
            logger.info("Transaction Cost Agent Pool shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")


# Factory function for easy instantiation
def create_transaction_cost_agent_pool(
    pool_id: str, 
    config_path: Optional[str] = None
) -> TransactionCostAgentPool:
    """
    Factory function to create and configure a Transaction Cost Agent Pool.
    
    Args:
        pool_id (str): Unique pool identifier
        config_path (Optional[str]): Path to configuration file
    
    Returns:
        TransactionCostAgentPool: Configured agent pool instance
    """
    config = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Config loading failed, using defaults: {str(e)}")
    
    return TransactionCostAgentPool(pool_id, config)


# Main execution - Transaction Cost Agent Pool MCP Server
if __name__ == "__main__":
    print("🚀 Starting Transaction Cost Agent Pool...")
    
    try:
        from mcp.server.fastmcp import FastMCP
        
        # Create FastMCP server
        tc_server = FastMCP("TransactionCostAgentPool")
        
        # Create transaction cost agent pool
        tc_pool = create_transaction_cost_agent_pool("transaction_cost_pool")
        
        @tc_server.tool(name="process_strategy_request", description="Process transaction cost analysis strategy request")
        async def process_strategy_request(request: dict) -> dict:
            """Process transaction cost analysis strategy request from orchestrator"""
            try:
                logger.info("Processing transaction cost analysis strategy request")
                
                # Extract request details
                symbols = request.get('symbols', ['AAPL', 'MSFT'])
                date = request.get('date', datetime.now().strftime('%Y-%m-%d'))
                trades = request.get('trades', [])
                portfolio_weights = request.get('portfolio_weights', {})
                
                # Create transaction cost analysis request
                cost_query = f"""
                Analyze transaction costs for:
                Symbols: {symbols}
                Date: {date}
                Trades: {trades}
                Portfolio weights: {portfolio_weights}
                
                Please provide cost estimates, impact analysis, and optimization recommendations.
                """
                
                # Use the transaction cost pool's capabilities
                cost_results = {}
                for symbol in symbols:
                    # Simulate cost analysis (simplified for now)
                    trade_volume = portfolio_weights.get(symbol, 0) * 1000000  # Assume $1M base
                    spread_cost = trade_volume * 0.0001  # 1 bp spread
                    market_impact = trade_volume * 0.0005  # 5 bp market impact
                    commission = max(1.0, trade_volume * 0.00001)  # Commission
                    
                    cost_results[symbol] = {
                        "spread_cost": spread_cost,
                        "market_impact": market_impact,
                        "commission": commission,
                        "total_cost": spread_cost + market_impact + commission,
                        "cost_bps": ((spread_cost + market_impact + commission) / trade_volume) * 10000 if trade_volume > 0 else 0
                    }
                
                total_cost = sum(result["total_cost"] for result in cost_results.values())
                
                logger.info("Transaction cost analysis completed successfully")
                return {
                    "status": "success",
                    "cost_breakdown": cost_results,
                    "total_cost": total_cost,
                    "cost_optimization": {
                        "recommendations": [
                            "Consider using TWAP orders for large trades",
                            "Monitor market impact during execution",
                            "Use dark pools for large block trades"
                        ]
                    },
                    "agent_source": "transaction_cost_agent_pool",
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Transaction cost analysis failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "agent_source": "transaction_cost_agent_pool",
                    "timestamp": datetime.now().isoformat()
                }

        @tc_server.tool(name="ping", description="Health check ping")
        def ping() -> str:
            return "pong"

        @tc_server.tool(name="status", description="Get transaction cost agent status")
        def status() -> dict:
            return {
                "status": "running",
                "agent_type": "transaction_cost",
                "port": 8085,
                "capabilities": [
                    "cost_analysis",
                    "market_impact_estimation",
                    "execution_optimization",
                    "spread_analysis"
                ]
            }
        
        # Configure and start server
        tc_server.settings.host = "0.0.0.0"
        tc_server.settings.port = 8085
        
        logger.info("Starting Transaction Cost Agent Pool on port 8085...")
        tc_server.run(transport="sse")
        
    except KeyboardInterrupt:
        print("\n🛑 Transaction Cost Agent Pool shutting down...")
    except Exception as e:
        print(f"❌ Transaction Cost Agent Pool error: {e}")
        import traceback
        traceback.print_exc()
