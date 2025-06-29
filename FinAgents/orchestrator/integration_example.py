"""
Real Integration Example - FinAgent Orchestrator with Live Agent Pools

This script demonstrates how to integrate the orchestrator with actual running
agent pools, memory agent, and execute real trading strategies.

Prerequisites:
- All agent pools must be running (data, alpha, risk, transaction_cost)
- Memory agent must be running
- Appropriate API keys configured

Usage:
    python integration_example.py [--strategy momentum|mean_reversion|pairs_trading]

Author: FinAgent Team
Version: 1.0.0
"""

import asyncio
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from core.finagent_orchestrator import FinAgentOrchestrator, OrchestratorStatus
from core.dag_planner import TradingStrategy, TaskNode, TaskStatus, AgentPoolType
from core.sandbox_environment import SandboxEnvironment, TestScenario, SandboxMode
from main_orchestrator import OrchestratorApplication

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("IntegrationExample")


class RealIntegrationExample:
    """Real integration example with live agent pools"""
    
    def __init__(self):
        self.config = {
            "orchestrator": {
                "host": "0.0.0.0",
                "port": 9000,
                "max_concurrent_tasks": 50,
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
            }
        }
        
        self.orchestrator: Optional[FinAgentOrchestrator] = None
        self.sandbox: Optional[SandboxEnvironment] = None
    
    async def initialize_system(self):
        """Initialize the orchestrator and check agent pool connections"""
        logger.info("🚀 Initializing FinAgent Orchestration System...")
        
        try:
            # Initialize orchestrator
            self.orchestrator = FinAgentOrchestrator(config=self.config["orchestrator"])
            await self.orchestrator.initialize()
            
            # Check orchestrator status
            if self.orchestrator.status != OrchestratorStatus.READY:
                raise Exception("Orchestrator failed to initialize properly")
            
            logger.info("✅ Orchestrator initialized successfully")
            
            # Register and verify agent pools
            await self._verify_agent_pools()
            
            # Initialize sandbox
            self.sandbox = SandboxEnvironment(
                orchestrator=self.orchestrator,
                config={"enable_backtesting": True, "initial_capital": 100000}
            )
            
            logger.info("✅ System initialization completed")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            logger.error("Please ensure all agent pools are running:")
            logger.error("  • Data Agent Pool: http://localhost:8001")
            logger.error("  • Alpha Agent Pool: http://localhost:5050")
            logger.error("  • Risk Agent Pool: http://localhost:7000") 
            logger.error("  • Transaction Cost Agent Pool: http://localhost:6000")
            logger.error("  • Memory Agent: http://localhost:8010")
            raise
    
    async def _verify_agent_pools(self):
        """Verify that all agent pools are accessible"""
        logger.info("🔍 Verifying agent pool connections...")
        
        for pool_name, pool_config in self.config["agent_pools"].items():
            if not pool_config.get("enabled", False):
                continue
                
            try:
                # Attempt to register the agent pool
                await self.orchestrator.register_agent_pool(
                    pool_name=pool_name,
                    endpoint_url=pool_config["url"]
                )
                
                # Test basic connectivity
                health_result = await self.orchestrator.check_agent_pool_health(pool_name)
                
                if health_result.get("status") == "healthy":
                    logger.info(f"✅ {pool_name}: Connected and healthy")
                else:
                    logger.warning(f"⚠️  {pool_name}: Connected but health check failed")
                    
            except Exception as e:
                logger.error(f"❌ {pool_name}: Connection failed - {e}")
                raise Exception(f"Failed to connect to {pool_name}")
    
    async def execute_momentum_strategy(self):
        """Execute a momentum-based trading strategy"""
        logger.info("📈 Executing Momentum Trading Strategy")
        logger.info("=" * 50)
        
        strategy = TradingStrategy(
            strategy_id="real_momentum_001",
            name="Real Momentum Strategy",
            description="Momentum strategy using real agent pools",
            symbols=["AAPL", "GOOGL", "MSFT"],
            lookback_period=20,
            rebalance_frequency="daily",
            parameters={
                "momentum_threshold": 0.03,
                "position_size": 0.25,
                "stop_loss": 0.05,
                "take_profit": 0.15
            }
        )
        
        try:
            # Execute the strategy
            result = await self.orchestrator.execute_strategy(strategy)
            
            # Process and display results
            await self._display_strategy_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Strategy execution failed: {e}")
            raise
    
    async def execute_mean_reversion_strategy(self):
        """Execute a mean reversion trading strategy"""
        logger.info("🔄 Executing Mean Reversion Trading Strategy")
        logger.info("=" * 50)
        
        strategy = TradingStrategy(
            strategy_id="real_mean_reversion_001",
            name="Real Mean Reversion Strategy", 
            description="Mean reversion strategy using real agent pools",
            symbols=["TSLA", "NVDA", "AMD"],
            lookback_period=30,
            rebalance_frequency="daily",
            parameters={
                "reversion_threshold": 2.0,  # Standard deviations
                "position_size": 0.2,
                "holding_period": 5,  # Days
                "volatility_filter": 0.3
            }
        )
        
        try:
            result = await self.orchestrator.execute_strategy(strategy)
            await self._display_strategy_results(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Strategy execution failed: {e}")
            raise
    
    async def execute_pairs_trading_strategy(self):
        """Execute a pairs trading strategy"""
        logger.info("🔗 Executing Pairs Trading Strategy")
        logger.info("=" * 50)
        
        strategy = TradingStrategy(
            strategy_id="real_pairs_trading_001",
            name="Real Pairs Trading Strategy",
            description="Statistical arbitrage pairs trading using real agent pools",
            symbols=["KO", "PEP", "JPM", "BAC"],  # Coca-Cola/Pepsi, JPM/BAC pairs
            lookback_period=60,
            rebalance_frequency="daily",
            parameters={
                "correlation_threshold": 0.7,
                "z_score_entry": 2.0,
                "z_score_exit": 0.5,
                "position_size": 0.15,
                "max_holding_period": 10
            }
        )
        
        try:
            result = await self.orchestrator.execute_strategy(strategy)
            await self._display_strategy_results(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Strategy execution failed: {e}")
            raise
    
    async def _display_strategy_results(self, result: Dict[str, Any]):
        """Display formatted strategy execution results"""
        if result["status"] == "completed":
            logger.info("✅ Strategy Execution Completed Successfully")
            logger.info(f"   Strategy ID: {result.get('strategy_id', 'N/A')}")
            logger.info(f"   Execution Time: {result.get('execution_time', 0):.2f}s")
            
            # Display signals
            signals = result.get("signals", [])
            logger.info(f"   Signals Generated: {len(signals)}")
            
            for i, signal in enumerate(signals[:5]):  # Show first 5 signals
                symbol = signal.get("symbol", "N/A")
                signal_type = signal.get("signal", "N/A")
                confidence = signal.get("confidence", 0)
                logger.info(f"     {i+1}. {symbol}: {signal_type} (confidence: {confidence:.2f})")
            
            if len(signals) > 5:
                logger.info(f"     ... and {len(signals) - 5} more signals")
            
            # Display risk metrics
            risk_metrics = result.get("risk_metrics", {})
            if risk_metrics:
                logger.info("   Risk Assessment:")
                logger.info(f"     VaR (95%): {risk_metrics.get('var_95', 0):.2%}")
                logger.info(f"     Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}")
                logger.info(f"     Volatility: {risk_metrics.get('volatility', 0):.2%}")
                logger.info(f"     Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}")
            
            # Display transaction costs
            costs = result.get("transaction_costs", {})
            if costs:
                logger.info("   Transaction Cost Analysis:")
                logger.info(f"     Commission: {costs.get('commission', 0):.4f}")
                logger.info(f"     Market Impact: {costs.get('market_impact', 0):.4f}")
                logger.info(f"     Slippage: {costs.get('slippage', 0):.4f}")
                logger.info(f"     Total Cost: {costs.get('total_cost', 0):.4f}")
        
        else:
            logger.error("❌ Strategy Execution Failed")
            logger.error(f"   Error: {result.get('error', 'Unknown error')}")
            logger.error(f"   Execution Time: {result.get('execution_time', 0):.2f}s")
    
    async def run_comprehensive_backtest(self):
        """Run comprehensive backtesting across multiple strategies"""
        logger.info("📊 Running Comprehensive Backtesting")
        logger.info("=" * 50)
        
        if not self.sandbox:
            logger.error("Sandbox not initialized")
            return
        
        # Define backtest scenarios
        scenarios = [
            TestScenario(
                scenario_id="momentum_backtest_2023",
                name="Momentum Strategy 2023 Backtest",
                mode=SandboxMode.HISTORICAL_BACKTEST,
                parameters={
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31", 
                    "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
                    "initial_capital": 100000,
                    "strategy_type": "momentum",
                    "benchmark": "SPY"
                }
            ),
            TestScenario(
                scenario_id="mean_reversion_backtest_2023",
                name="Mean Reversion Strategy 2023 Backtest",
                mode=SandboxMode.HISTORICAL_BACKTEST,
                parameters={
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "symbols": ["TSLA", "NVDA", "AMD", "INTC"],
                    "initial_capital": 100000,
                    "strategy_type": "mean_reversion",
                    "benchmark": "QQQ"
                }
            )
        ]
        
        backtest_results = []
        
        for scenario in scenarios:
            try:
                logger.info(f"🔄 Running: {scenario.name}")
                result = await self.sandbox.run_test_scenario(scenario)
                backtest_results.append(result)
                
                # Display results
                perf = result.get("performance_metrics", {})
                logger.info(f"   ✅ {scenario.name} completed:")
                logger.info(f"     Total Return: {perf.get('total_return', 0):.2%}")
                logger.info(f"     Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
                logger.info(f"     Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
                logger.info(f"     Win Rate: {perf.get('win_rate', 0):.1%}")
                
            except Exception as e:
                logger.error(f"❌ Backtest failed for {scenario.name}: {e}")
        
        return backtest_results
    
    async def demonstrate_real_time_monitoring(self):
        """Demonstrate real-time monitoring capabilities"""
        logger.info("📡 Starting Real-time Monitoring Demo")
        logger.info("=" * 50)
        
        monitoring_duration = 30  # seconds
        check_interval = 5  # seconds
        
        logger.info(f"Monitoring system for {monitoring_duration} seconds...")
        
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < monitoring_duration:
            try:
                # Get system health status
                health_status = await self.orchestrator.get_health_status()
                
                # Get performance metrics
                performance = await self.orchestrator.get_performance_metrics()
                
                # Display monitoring information
                current_time = datetime.now().strftime("%H:%M:%S")
                logger.info(f"[{current_time}] System Status: {health_status.get('status', 'unknown')}")
                logger.info(f"[{current_time}] Active Tasks: {performance.get('active_tasks', 0)}")
                logger.info(f"[{current_time}] Total Executed: {performance.get('total_executed', 0)}")
                logger.info(f"[{current_time}] Success Rate: {performance.get('success_rate', 0):.1%}")
                
                # Check agent pool health
                for pool_name in self.config["agent_pools"]:
                    if self.config["agent_pools"][pool_name].get("enabled"):
                        pool_health = await self.orchestrator.check_agent_pool_health(pool_name)
                        status = pool_health.get("status", "unknown")
                        logger.info(f"[{current_time}] {pool_name}: {status}")
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(check_interval)
        
        logger.info("📡 Real-time monitoring demo completed")
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up resources...")
        
        try:
            if self.orchestrator:
                await self.orchestrator.shutdown()
            
            if self.sandbox:
                await self.sandbox.cleanup()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")


async def main():
    """Main entry point for real integration example"""
    parser = argparse.ArgumentParser(description="FinAgent Real Integration Example")
    parser.add_argument(
        "--strategy",
        choices=["momentum", "mean_reversion", "pairs_trading", "all"],
        default="all",
        help="Strategy type to execute"
    )
    parser.add_argument(
        "--include-backtest",
        action="store_true",
        help="Include comprehensive backtesting"
    )
    parser.add_argument(
        "--include-monitoring",
        action="store_true", 
        help="Include real-time monitoring demo"
    )
    
    args = parser.parse_args()
    
    logger.info("🎯 FinAgent Real Integration Example")
    logger.info("=" * 60)
    logger.info("This example requires live agent pools to be running!")
    logger.info("=" * 60)
    
    example = RealIntegrationExample()
    
    try:
        # Initialize system
        await example.initialize_system()
        
        # Execute strategies based on selection
        if args.strategy in ["momentum", "all"]:
            await example.execute_momentum_strategy()
            await asyncio.sleep(2)
        
        if args.strategy in ["mean_reversion", "all"]:
            await example.execute_mean_reversion_strategy()
            await asyncio.sleep(2)
        
        if args.strategy in ["pairs_trading", "all"]:
            await example.execute_pairs_trading_strategy()
            await asyncio.sleep(2)
        
        # Optional backtesting
        if args.include_backtest:
            await example.run_comprehensive_backtest()
        
        # Optional real-time monitoring
        if args.include_monitoring:
            await example.demonstrate_real_time_monitoring()
        
        logger.info("🎊 Integration example completed successfully!")
        logger.info("The FinAgent orchestration system is working properly with live agent pools.")
        
    except Exception as e:
        logger.error(f"❌ Integration example failed: {e}")
        logger.error("\nTroubleshooting steps:")
        logger.error("1. Ensure all agent pools are running")
        logger.error("2. Check network connectivity")
        logger.error("3. Verify configuration settings")
        logger.error("4. Check agent pool logs for errors")
        sys.exit(1)
        
    finally:
        await example.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Integration example terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
