"""
Main Orchestrator Application - FinAgent System Entry Point

This module provides the main entry point for the FinAgent orchestration system,
integrating all agent pools, DAG planner, RL engine, and sandbox environment.

Key Features:
- Complete system initialization and startup
- Agent pool discovery and registration
- Orchestrated strategy execution
- Comprehensive monitoring and logging
- RL-enhanced backtesting
- Sandbox testing environment

Usage:
    python main_orchestrator.py [--config config.yaml] [--mode production|development|sandbox]

Author: FinAgent Team
Version: 1.0.0
"""

import asyncio
import logging
import argparse
import signal
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

from core.finagent_orchestrator import FinAgentOrchestrator, OrchestratorStatus
from core.dag_planner import DAGPlanner, TradingStrategy, BacktestConfiguration
from core.rl_policy_engine import RLPolicyEngine, RLConfiguration, RLAlgorithm
from core.sandbox_environment import SandboxEnvironment, SandboxMode, TestScenario

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'orchestrator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger("MainOrchestrator")


class OrchestratorApplication:
    """Main orchestrator application for the FinAgent system"""
    
    def __init__(self, config_path: Optional[str] = None, mode: str = "development"):
        """
        Initialize the orchestrator application
        
        Args:
            config_path: Path to configuration file
            mode: Operating mode (production, development, sandbox)
        """
        self.mode = mode
        self.config = self._load_config(config_path)
        self.orchestrator: Optional[FinAgentOrchestrator] = None
        self.sandbox: Optional[SandboxEnvironment] = None
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Orchestrator application initialized in {mode} mode")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
        else:
            # Default configuration
            config = {
                "orchestrator": {
                    "host": "0.0.0.0",
                    "port": 9000,
                    "max_concurrent_tasks": 100,
                    "task_timeout": 300,
                    "memory_agent_url": "http://localhost:8010",
                    "enable_rl": True,
                    "enable_sandbox": True
                },
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
                "rl_engine": {
                    "algorithm": "TD3",
                    "learning_rate": 0.0003,
                    "buffer_size": 1000000,
                    "batch_size": 256,
                    "tau": 0.005,
                    "gamma": 0.99,
                    "policy_noise": 0.2,
                    "noise_clip": 0.5,
                    "policy_freq": 2
                },
                "sandbox": {
                    "enable_backtesting": True,
                    "enable_stress_testing": True,
                    "data_start_date": "2023-01-01",
                    "data_end_date": "2024-12-31",
                    "initial_capital": 1000000,
                    "commission_rate": 0.001
                }
            }
            logger.info("Using default configuration")
        
        return config
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def initialize_system(self):
        """Initialize all system components"""
        logger.info("Initializing FinAgent orchestration system...")
        
        try:
            # Initialize orchestrator
            self.orchestrator = FinAgentOrchestrator(
                config=self.config["orchestrator"]
            )
            
            # Initialize sandbox if enabled
            if self.config["orchestrator"].get("enable_sandbox", False):
                self.sandbox = SandboxEnvironment(
                    orchestrator=self.orchestrator,
                    config=self.config["sandbox"]
                )
                logger.info("Sandbox environment initialized")
            
            # Start orchestrator
            await self.orchestrator.initialize()
            logger.info("Orchestrator initialized successfully")
            
            # Register agent pools
            await self._register_agent_pools()
            
            # Initialize RL engine if enabled
            if self.config["orchestrator"].get("enable_rl", False):
                await self._initialize_rl_engine()
            
            logger.info("System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def _register_agent_pools(self):
        """Register all configured agent pools"""
        logger.info("Registering agent pools...")
        
        for pool_name, pool_config in self.config["agent_pools"].items():
            if pool_config.get("enabled", False):
                try:
                    await self.orchestrator.register_agent_pool(
                        pool_name=pool_name,
                        endpoint_url=pool_config["url"]
                    )
                    logger.info(f"Registered {pool_name} at {pool_config['url']}")
                except Exception as e:
                    logger.warning(f"Failed to register {pool_name}: {e}")
    
    async def _initialize_rl_engine(self):
        """Initialize reinforcement learning engine"""
        logger.info("Initializing RL engine...")
        
        rl_config = RLConfiguration(
            algorithm=RLAlgorithm(self.config["rl_engine"]["algorithm"]),
            learning_rate=self.config["rl_engine"]["learning_rate"],
            buffer_size=self.config["rl_engine"]["buffer_size"],
            batch_size=self.config["rl_engine"]["batch_size"],
            gamma=self.config["rl_engine"]["gamma"]
        )
        
        await self.orchestrator.initialize_rl_engine(rl_config)
        logger.info("RL engine initialized successfully")
    
    async def run_development_demo(self):
        """Run development mode demonstration"""
        logger.info("Running development demonstration...")
        
        # Example 1: Simple strategy execution
        await self._demo_simple_strategy()
        
        # Example 2: Multi-agent coordination
        await self._demo_multi_agent_coordination()
        
        # Example 3: Sandbox testing
        if self.sandbox:
            await self._demo_sandbox_testing()
        
        # Example 4: RL training
        if self.orchestrator.rl_engine:
            await self._demo_rl_training()
    
    async def _demo_simple_strategy(self):
        """Demonstrate simple strategy execution"""
        logger.info("Demo: Simple momentum strategy execution")
        
        strategy = TradingStrategy(
            strategy_id="demo_momentum",
            name="Demo Momentum Strategy",
            description="Simple momentum strategy for demonstration",
            symbols=["AAPL", "GOOGL", "MSFT"],
            lookback_period=20,
            rebalance_frequency="daily",
            parameters={
                "momentum_window": 10,
                "signal_threshold": 0.02
            }
        )
        
        try:
            result = await self.orchestrator.execute_strategy(strategy)
            logger.info(f"Strategy execution result: {result['status']}")
            
            if result["status"] == "completed":
                logger.info(f"Generated {len(result.get('signals', []))} signals")
                
        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
    
    async def _demo_multi_agent_coordination(self):
        """Demonstrate multi-agent coordination"""
        logger.info("Demo: Multi-agent coordination workflow")
        
        # Create a complex workflow involving multiple agent pools
        workflow_config = {
            "workflow_id": "multi_agent_demo",
            "steps": [
                {
                    "agent_pool": "data_agent_pool",
                    "action": "fetch_market_data",
                    "parameters": {"symbols": ["AAPL", "GOOGL"], "period": "1d"}
                },
                {
                    "agent_pool": "alpha_agent_pool", 
                    "action": "generate_signals",
                    "parameters": {"strategy_type": "momentum"}
                },
                {
                    "agent_pool": "risk_agent_pool",
                    "action": "assess_portfolio_risk",
                    "parameters": {"risk_model": "var"}
                },
                {
                    "agent_pool": "transaction_cost_agent_pool",
                    "action": "estimate_execution_cost",
                    "parameters": {"cost_model": "implementation_shortfall"}
                }
            ]
        }
        
        try:
            result = await self.orchestrator.execute_workflow(workflow_config)
            logger.info(f"Multi-agent workflow result: {result['status']}")
            
        except Exception as e:
            logger.error(f"Multi-agent workflow failed: {e}")
    
    async def _demo_sandbox_testing(self):
        """Demonstrate sandbox testing capabilities"""
        logger.info("Demo: Sandbox testing environment")
        
        # Historical backtest scenario
        backtest_scenario = TestScenario(
            scenario_id="historical_backtest_demo",
            name="Historical Backtest Demo",
            mode=SandboxMode.HISTORICAL_BACKTEST,
            parameters={
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "initial_capital": 100000,
                "strategy_type": "momentum"
            }
        )
        
        try:
            result = await self.sandbox.run_test_scenario(backtest_scenario)
            logger.info(f"Backtest completed with performance: {result.get('performance_metrics', {})}")
            
        except Exception as e:
            logger.error(f"Sandbox testing failed: {e}")
    
    async def _demo_rl_training(self):
        """Demonstrate RL training capabilities"""
        logger.info("Demo: RL training demonstration")
        
        training_config = {
            "training_episodes": 10,
            "symbols": ["AAPL", "GOOGL"],
            "start_date": "2023-01-01",
            "end_date": "2023-06-30",
            "initial_capital": 100000
        }
        
        try:
            training_result = await self.orchestrator.train_rl_policy(training_config)
            logger.info(f"RL training completed: {training_result['status']}")
            
        except Exception as e:
            logger.error(f"RL training failed: {e}")
    
    async def run_production_mode(self):
        """Run in production mode"""
        logger.info("Starting production mode...")
        
        # Start orchestrator server
        await self.orchestrator.start_server()
        
        # Production monitoring loop
        self.running = True
        while self.running:
            try:
                # Monitor system health
                health_status = await self.orchestrator.get_health_status()
                
                if health_status["status"] != "healthy":
                    logger.warning(f"System health issue detected: {health_status}")
                
                # Check for pending tasks
                pending_tasks = await self.orchestrator.get_pending_tasks()
                if pending_tasks:
                    logger.info(f"Processing {len(pending_tasks)} pending tasks")
                
                # Sleep before next check
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Production monitoring error: {e}")
                await asyncio.sleep(30)  # Longer sleep on error
    
    async def run_sandbox_mode(self):
        """Run in sandbox mode"""
        logger.info("Starting sandbox mode...")
        
        if not self.sandbox:
            logger.error("Sandbox not initialized")
            return
        
        # Run comprehensive sandbox tests
        test_scenarios = [
            TestScenario(
                scenario_id="comprehensive_backtest",
                name="Comprehensive Historical Backtest",
                mode=SandboxMode.HISTORICAL_BACKTEST,
                parameters={
                    "start_date": "2022-01-01",
                    "end_date": "2024-12-31",
                    "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
                    "initial_capital": 1000000,
                    "strategies": ["momentum", "mean_reversion", "pairs_trading"]
                }
            ),
            TestScenario(
                scenario_id="stress_test",
                name="Market Stress Test",
                mode=SandboxMode.STRESS_TEST,
                parameters={
                    "stress_scenarios": ["market_crash", "volatility_spike", "liquidity_crisis"],
                    "severity_levels": [0.5, 0.75, 1.0]
                }
            )
        ]
        
        for scenario in test_scenarios:
            try:
                logger.info(f"Running scenario: {scenario.name}")
                result = await self.sandbox.run_test_scenario(scenario)
                logger.info(f"Scenario {scenario.scenario_id} completed: {result['status']}")
                
            except Exception as e:
                logger.error(f"Scenario {scenario.scenario_id} failed: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down orchestrator system...")
        
        try:
            if self.orchestrator:
                await self.orchestrator.shutdown()
            
            if self.sandbox:
                await self.sandbox.cleanup()
            
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def run(self):
        """Main run method"""
        try:
            await self.initialize_system()
            
            if self.mode == "production":
                await self.run_production_mode()
            elif self.mode == "development":
                await self.run_development_demo()
            elif self.mode == "sandbox":
                await self.run_sandbox_mode()
            else:
                logger.error(f"Unknown mode: {self.mode}")
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            await self.shutdown()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="FinAgent Orchestration System")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["production", "development", "sandbox"],
        default="development",
        help="Operating mode"
    )
    
    args = parser.parse_args()
    
    # Create and run application
    app = OrchestratorApplication(
        config_path=args.config,
        mode=args.mode
    )
    
    await app.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)
