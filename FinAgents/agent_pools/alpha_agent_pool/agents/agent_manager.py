"""
Alpha Agent Management Module

This module provides centralized management for all alpha agent functionalities
within the Alpha Agent Pool. It coordinates between theory-driven, empirical,
and autonomous agents to deliver comprehensive alpha strategy research.

Architecture:
- Theory-driven agents: Momentum, mean reversion, and academic factor models
- Empirical agents: Data mining and ML pattern recognition
- Autonomous agents: Self-directed strategy discovery and optimization

All agents are coordinated through the enhanced memory bridge system for
institutional-grade alpha factor research and strategy development.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

# Import agent implementations
from .theory_driven.momentum_agent import MomentumAgent
from .theory_driven.mean_reversion_agent import MeanReversionAgent
from .empirical.data_mining_agent import DataMiningAgent
from .empirical.ml_pattern_agent import MLPatternAgent
from .autonomous.autonomous_agent import AutonomousAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaAgentManager:
    """
    Centralized manager for all alpha agent operations.
    
    This class coordinates multiple specialized agents to provide comprehensive
    alpha strategy research capabilities. It manages agent lifecycle, coordinates
    cross-agent communication, and ensures optimal resource utilization.
    
    Key Features:
    - Multi-agent coordination and orchestration
    - Centralized memory bridge integration
    - Performance monitoring and optimization
    - Dynamic agent allocation based on market conditions
    - Academic-standard research methodology coordination
    """
    
    def __init__(self, memory_bridge=None, enhanced_a2a_bridge=None):
        """
        Initialize the Alpha Agent Manager.
        
        Args:
            memory_bridge: Primary memory bridge for agent coordination
            enhanced_a2a_bridge: Enhanced A2A memory bridge for cross-agent learning
        """
        self.memory_bridge = memory_bridge
        self.enhanced_a2a_bridge = enhanced_a2a_bridge
        
        # Initialize agent registry
        self.agents: Dict[str, Any] = {}
        self.agent_status: Dict[str, str] = {}
        
        # Performance tracking
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        
        logger.info("🎯 Alpha Agent Manager initialized")
    
    async def initialize_agents(self) -> Dict[str, str]:
        """
        Initialize all available alpha agents.
        
        Returns:
            Dictionary containing initialization status for each agent
        """
        initialization_results = {}
        
        try:
            # Initialize theory-driven agents
            logger.info("🔬 Initializing theory-driven agents...")
            
            # Momentum Agent
            try:
                self.agents['momentum'] = MomentumAgent(
                    memory_bridge=self.memory_bridge,
                    enhanced_bridge=self.enhanced_a2a_bridge
                )
                await self.agents['momentum'].initialize()
                self.agent_status['momentum'] = 'active'
                initialization_results['momentum'] = 'success'
                logger.info("✅ Momentum agent initialized")
            except Exception as e:
                logger.error(f"❌ Momentum agent initialization failed: {e}")
                initialization_results['momentum'] = f'failed: {e}'
            
            # Mean Reversion Agent
            try:
                self.agents['mean_reversion'] = MeanReversionAgent(
                    memory_bridge=self.memory_bridge,
                    enhanced_bridge=self.enhanced_a2a_bridge
                )
                await self.agents['mean_reversion'].initialize()
                self.agent_status['mean_reversion'] = 'active'
                initialization_results['mean_reversion'] = 'success'
                logger.info("✅ Mean reversion agent initialized")
            except Exception as e:
                logger.error(f"❌ Mean reversion agent initialization failed: {e}")
                initialization_results['mean_reversion'] = f'failed: {e}'
            
            # Initialize empirical agents
            logger.info("📊 Initializing empirical agents...")
            
            # Data Mining Agent
            try:
                self.agents['data_mining'] = DataMiningAgent(
                    memory_bridge=self.memory_bridge,
                    enhanced_bridge=self.enhanced_a2a_bridge
                )
                await self.agents['data_mining'].initialize()
                self.agent_status['data_mining'] = 'active'
                initialization_results['data_mining'] = 'success'
                logger.info("✅ Data mining agent initialized")
            except Exception as e:
                logger.error(f"❌ Data mining agent initialization failed: {e}")
                initialization_results['data_mining'] = f'failed: {e}'
            
            # ML Pattern Agent
            try:
                self.agents['ml_pattern'] = MLPatternAgent(
                    memory_bridge=self.memory_bridge,
                    enhanced_bridge=self.enhanced_a2a_bridge
                )
                await self.agents['ml_pattern'].initialize()
                self.agent_status['ml_pattern'] = 'active'
                initialization_results['ml_pattern'] = 'success'
                logger.info("✅ ML pattern agent initialized")
            except Exception as e:
                logger.error(f"❌ ML pattern agent initialization failed: {e}")
                initialization_results['ml_pattern'] = f'failed: {e}'
            
            # Initialize autonomous agents
            logger.info("🤖 Initializing autonomous agents...")
            
            # Autonomous Agent
            try:
                self.agents['autonomous'] = AutonomousAgent(
                    memory_bridge=self.memory_bridge,
                    enhanced_bridge=self.enhanced_a2a_bridge
                )
                await self.agents['autonomous'].initialize()
                self.agent_status['autonomous'] = 'active'
                initialization_results['autonomous'] = 'success'
                logger.info("✅ Autonomous agent initialized")
            except Exception as e:
                logger.error(f"❌ Autonomous agent initialization failed: {e}")
                initialization_results['autonomous'] = f'failed: {e}'
            
            successful_agents = sum(1 for result in initialization_results.values() if result == 'success')
            logger.info(f"🎯 Agent initialization completed: {successful_agents}/{len(initialization_results)} agents active")
            
            return initialization_results
            
        except Exception as e:
            logger.error(f"❌ Critical error during agent initialization: {e}")
            return {"error": str(e)}
    
    async def execute_alpha_research_workflow(self, 
                                            research_parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute comprehensive alpha research workflow using all available agents.
        
        Args:
            research_parameters: Parameters for alpha research configuration
            
        Returns:
            Comprehensive alpha research results from all agents
        """
        workflow_id = f"alpha_research_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"🚀 Starting alpha research workflow: {workflow_id}")
        
        research_params = research_parameters or {
            "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
            "lookback_period": 30,
            "significance_threshold": 0.05,
            "risk_level": "moderate"
        }
        
        workflow_results = {
            "workflow_id": workflow_id,
            "start_timestamp": datetime.utcnow().isoformat(),
            "research_parameters": research_params,
            "agent_results": {},
            "consolidated_insights": {},
            "performance_summary": {}
        }
        
        try:
            # Execute theory-driven research
            logger.info("🔬 Executing theory-driven alpha research...")
            
            # Momentum-based alpha discovery
            if 'momentum' in self.agents and self.agent_status.get('momentum') == 'active':
                try:
                    momentum_results = await self.agents['momentum'].discover_momentum_factors(
                        symbols=research_params["symbols"],
                        lookback_period=research_params["lookback_period"]
                    )
                    workflow_results["agent_results"]["momentum"] = momentum_results
                    logger.info("✅ Momentum alpha research completed")
                except Exception as e:
                    logger.error(f"❌ Momentum research failed: {e}")
                    workflow_results["agent_results"]["momentum"] = {"error": str(e)}
            
            # Mean reversion alpha discovery
            if 'mean_reversion' in self.agents and self.agent_status.get('mean_reversion') == 'active':
                try:
                    mean_rev_results = await self.agents['mean_reversion'].discover_mean_reversion_factors(
                        symbols=research_params["symbols"],
                        lookback_period=research_params["lookback_period"]
                    )
                    workflow_results["agent_results"]["mean_reversion"] = mean_rev_results
                    logger.info("✅ Mean reversion alpha research completed")
                except Exception as e:
                    logger.error(f"❌ Mean reversion research failed: {e}")
                    workflow_results["agent_results"]["mean_reversion"] = {"error": str(e)}
            
            # Execute empirical research
            logger.info("📊 Executing empirical alpha research...")
            
            # Data mining alpha discovery
            if 'data_mining' in self.agents and self.agent_status.get('data_mining') == 'active':
                try:
                    data_mining_results = await self.agents['data_mining'].mine_alpha_patterns(
                        symbols=research_params["symbols"],
                        significance_threshold=research_params["significance_threshold"]
                    )
                    workflow_results["agent_results"]["data_mining"] = data_mining_results
                    logger.info("✅ Data mining alpha research completed")
                except Exception as e:
                    logger.error(f"❌ Data mining research failed: {e}")
                    workflow_results["agent_results"]["data_mining"] = {"error": str(e)}
            
            # ML pattern recognition
            if 'ml_pattern' in self.agents and self.agent_status.get('ml_pattern') == 'active':
                try:
                    ml_results = await self.agents['ml_pattern'].discover_ml_patterns(
                        symbols=research_params["symbols"],
                        pattern_complexity="moderate"
                    )
                    workflow_results["agent_results"]["ml_pattern"] = ml_results
                    logger.info("✅ ML pattern alpha research completed")
                except Exception as e:
                    logger.error(f"❌ ML pattern research failed: {e}")
                    workflow_results["agent_results"]["ml_pattern"] = {"error": str(e)}
            
            # Execute autonomous research
            logger.info("🤖 Executing autonomous alpha research...")
            
            if 'autonomous' in self.agents and self.agent_status.get('autonomous') == 'active':
                try:
                    autonomous_results = await self.agents['autonomous'].autonomous_alpha_discovery(
                        research_context={
                            "symbols": research_params["symbols"],
                            "peer_results": workflow_results["agent_results"]
                        }
                    )
                    workflow_results["agent_results"]["autonomous"] = autonomous_results
                    logger.info("✅ Autonomous alpha research completed")
                except Exception as e:
                    logger.error(f"❌ Autonomous research failed: {e}")
                    workflow_results["agent_results"]["autonomous"] = {"error": str(e)}
            
            # Consolidate insights from all agents
            workflow_results["consolidated_insights"] = await self._consolidate_alpha_insights(
                workflow_results["agent_results"]
            )
            
            # Generate performance summary
            workflow_results["performance_summary"] = await self._generate_workflow_performance_summary(
                workflow_results["agent_results"]
            )
            
            workflow_results["completion_timestamp"] = datetime.utcnow().isoformat()
            workflow_results["status"] = "completed"
            
            logger.info(f"🎯 Alpha research workflow completed: {workflow_id}")
            
            # Store workflow results in memory bridge
            if self.enhanced_a2a_bridge:
                try:
                    await self.enhanced_a2a_bridge.store_agent_performance(
                        agent_id="alpha_agent_manager",
                        performance_data={
                            "workflow_id": workflow_id,
                            "agents_executed": len([r for r in workflow_results["agent_results"].values() if "error" not in r]),
                            "total_factors_discovered": workflow_results["consolidated_insights"].get("total_factors", 0),
                            "workflow_duration": (
                                datetime.fromisoformat(workflow_results["completion_timestamp"]) - 
                                datetime.fromisoformat(workflow_results["start_timestamp"])
                            ).total_seconds()
                        }
                    )
                    logger.info("✅ Workflow results stored in memory bridge")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to store workflow results: {e}")
            
            return workflow_results
            
        except Exception as e:
            logger.error(f"❌ Critical error in alpha research workflow: {e}")
            workflow_results["status"] = "failed"
            workflow_results["error"] = str(e)
            workflow_results["completion_timestamp"] = datetime.utcnow().isoformat()
            return workflow_results
    
    async def _consolidate_alpha_insights(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate insights from all agent research results."""
        consolidated = {
            "total_factors": 0,
            "factor_categories": set(),
            "consensus_factors": [],
            "unique_factors": [],
            "performance_metrics": {},
            "research_quality": "high"
        }
        
        try:
            for agent_name, results in agent_results.items():
                if "error" in results:
                    continue
                
                # Extract factors discovered by each agent
                factors = results.get("factors_discovered", [])
                consolidated["total_factors"] += len(factors)
                
                # Track factor categories
                for factor in factors:
                    if isinstance(factor, dict) and "category" in factor:
                        consolidated["factor_categories"].add(factor["category"])
                
                # Extract performance metrics
                if "performance" in results:
                    consolidated["performance_metrics"][agent_name] = results["performance"]
            
            consolidated["factor_categories"] = list(consolidated["factor_categories"])
            
            # Determine research quality based on agent coverage and factor count
            active_agents = len([r for r in agent_results.values() if "error" not in r])
            if active_agents >= 4 and consolidated["total_factors"] >= 10:
                consolidated["research_quality"] = "institutional"
            elif active_agents >= 3 and consolidated["total_factors"] >= 5:
                consolidated["research_quality"] = "high"
            elif active_agents >= 2:
                consolidated["research_quality"] = "moderate"
            else:
                consolidated["research_quality"] = "limited"
            
        except Exception as e:
            logger.error(f"❌ Error consolidating insights: {e}")
            consolidated["error"] = str(e)
        
        return consolidated
    
    async def _generate_workflow_performance_summary(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary for the workflow execution."""
        summary = {
            "agents_executed": 0,
            "successful_agents": 0,
            "failed_agents": 0,
            "total_execution_time": "estimated",
            "success_rate": 0.0,
            "agent_performance": {}
        }
        
        try:
            for agent_name, results in agent_results.items():
                summary["agents_executed"] += 1
                
                if "error" in results:
                    summary["failed_agents"] += 1
                    summary["agent_performance"][agent_name] = "failed"
                else:
                    summary["successful_agents"] += 1
                    summary["agent_performance"][agent_name] = "success"
            
            if summary["agents_executed"] > 0:
                summary["success_rate"] = summary["successful_agents"] / summary["agents_executed"]
            
        except Exception as e:
            logger.error(f"❌ Error generating performance summary: {e}")
            summary["error"] = str(e)
        
        return summary
    
    async def get_agent_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report for all managed agents."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_agents": len(self.agents),
            "active_agents": sum(1 for status in self.agent_status.values() if status == 'active'),
            "agent_details": {},
            "memory_bridge_status": "connected" if self.memory_bridge else "disconnected",
            "enhanced_bridge_status": "connected" if self.enhanced_a2a_bridge else "disconnected"
        }
        
        for agent_name, agent in self.agents.items():
            try:
                agent_health = await agent.get_health_status() if hasattr(agent, 'get_health_status') else "unknown"
                report["agent_details"][agent_name] = {
                    "status": self.agent_status.get(agent_name, "unknown"),
                    "health": agent_health,
                    "type": agent.__class__.__name__
                }
            except Exception as e:
                report["agent_details"][agent_name] = {
                    "status": "error",
                    "error": str(e),
                    "type": agent.__class__.__name__ if agent else "unknown"
                }
        
        return report
    
    async def shutdown_agents(self) -> Dict[str, str]:
        """Shutdown all managed agents gracefully."""
        shutdown_results = {}
        
        logger.info("🛑 Initiating agent shutdown sequence...")
        
        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
                self.agent_status[agent_name] = 'stopped'
                shutdown_results[agent_name] = 'success'
                logger.info(f"✅ {agent_name} agent shutdown successfully")
            except Exception as e:
                shutdown_results[agent_name] = f'error: {e}'
                logger.error(f"❌ {agent_name} agent shutdown failed: {e}")
        
        logger.info("🛑 Agent shutdown sequence completed")
        return shutdown_results


# Convenience functions for direct agent management
async def initialize_alpha_agents(memory_bridge=None, enhanced_bridge=None) -> AlphaAgentManager:
    """Initialize and return a fully configured Alpha Agent Manager."""
    manager = AlphaAgentManager(memory_bridge=memory_bridge, enhanced_a2a_bridge=enhanced_bridge)
    await manager.initialize_agents()
    return manager


async def quick_alpha_research(symbols: List[str] = None) -> Dict[str, Any]:
    """Perform quick alpha research with default parameters."""
    symbols = symbols or ["AAPL", "GOOGL", "MSFT"]
    
    manager = await initialize_alpha_agents()
    results = await manager.execute_alpha_research_workflow({
        "symbols": symbols,
        "lookback_period": 20,
        "significance_threshold": 0.05,
        "risk_level": "moderate"
    })
    
    await manager.shutdown_agents()
    return results
