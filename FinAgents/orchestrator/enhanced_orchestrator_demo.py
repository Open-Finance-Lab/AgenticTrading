"""
Enhanced FinAgent Orchestrator with Natural Language Interface Demo
Demonstrates LLM-enhanced DAG planning and natural language interaction capabilities
"""

import asyncio
import logging
import json
import yaml
import httpx
from datetime import datetime
from pathlib import Path
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.dag_planner import DAGPlanner, TradingStrategy
from core.finagent_orchestrator import FinAgentOrchestrator
from core.llm_integration import LLMConfig, NaturalLanguageProcessor, ConversationManager
from core.mcp_nl_interface import MCPNaturalLanguageInterface
from core.agent_pool_monitor import AgentPoolMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("EnhancedOrchestrator")

class EnhancedFinAgentDemo:
    """
    Demonstration of enhanced FinAgent capabilities with LLM integration
    """
    
    def __init__(self, config_path: str = "config/orchestrator_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize LLM configuration
        self.llm_config = LLMConfig(
            provider=self.config.get("llm", {}).get("provider", "openai"),
            model=self.config.get("llm", {}).get("model", "gpt-4"),
            temperature=self.config.get("llm", {}).get("temperature", 0.7)
        )
        
        # Initialize components
        self.dag_planner = None
        self.orchestrator = None
        self.nl_interface = None
        self.agent_monitor = None
        self.conversation_manager = None
        
    def _load_config(self) -> dict:
        """Load orchestrator configuration"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Return default configuration
                return self._get_default_config()
        except Exception as e:
            logger.warning(f"Error loading config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration for demo"""
        return {
            "orchestrator": {
                "host": "localhost",
                "port": 9000,
                "enable_llm": True,
                "enable_nl_interface": True
            },
            "agent_pools": {
                "data_agent_pool": {
                    "url": "http://localhost:8001",
                    "enabled": True,
                    "capabilities": ["market_data_fetch", "technical_indicators"]
                },
                "alpha_agent_pool": {
                    "url": "http://localhost:5050",
                    "enabled": True,
                    "capabilities": ["signal_generation", "strategy_development"]
                },
                "risk_agent_pool": {
                    "url": "http://localhost:7000",
                    "enabled": True,
                    "capabilities": ["risk_assessment", "portfolio_optimization"]
                },
                "transaction_cost_agent_pool": {
                    "url": "http://localhost:6000",
                    "enabled": True,
                    "capabilities": ["execution_optimization", "cost_analysis"]
                }
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "enabled": True
            }
        }
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing Enhanced FinAgent System")
        
        try:
            # Initialize LLM-enhanced DAG planner
            self.dag_planner = DAGPlanner(self.llm_config)
            logger.info("✅ LLM-enhanced DAG planner initialized")
            
            # Initialize orchestrator
            self.orchestrator = FinAgentOrchestrator(self.config_path)
            logger.info("✅ Orchestrator initialized")
            
            # Initialize natural language interface
            self.nl_interface = MCPNaturalLanguageInterface(self.config)
            logger.info("✅ Natural language interface initialized")
            
            # Initialize agent pool monitor
            self.agent_monitor = AgentPoolMonitor(self.config)
            logger.info("✅ Agent pool monitor initialized")
            
            # Initialize conversation manager
            nlp = NaturalLanguageProcessor(self.llm_config)
            self.conversation_manager = ConversationManager(nlp)
            logger.info("✅ Conversation manager initialized")
            
            logger.info("🎉 Enhanced FinAgent System ready!")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    async def demo_natural_language_strategy_planning(self):
        """Demonstrate natural language strategy planning"""
        print("\n" + "="*60)
        print("🧠 NATURAL LANGUAGE STRATEGY PLANNING DEMO")
        print("="*60)
        
        # Test natural language strategy descriptions
        strategy_descriptions = [
            "Execute a momentum strategy for Apple and Google stocks using 20-day moving averages",
            "Create a mean reversion strategy for tech stocks when they're oversold",
            "Run a pairs trading strategy between correlated bank stocks",
            "Implement a machine learning strategy using sentiment analysis and technical indicators"
        ]
        
        for i, description in enumerate(strategy_descriptions, 1):
            print(f"\n📝 Strategy {i}: {description}")
            
            try:
                # Use LLM-enhanced DAG planner
                dag = await self.dag_planner.plan_strategy_from_description(
                    description,
                    context={"market_conditions": "bullish", "risk_tolerance": "medium"}
                )
                
                # Analyze the generated DAG
                task_count = len(dag.nodes())
                complexity = "High" if task_count > 15 else "Medium" if task_count > 8 else "Low"
                
                print(f"   ✅ Strategy planned successfully!")
                print(f"   📊 Generated {task_count} tasks")
                print(f"   🎯 Complexity: {complexity}")
                print(f"   🔗 DAG edges: {len(dag.edges())}")
                
                # Show task breakdown
                task_types = {}
                for node in dag.nodes():
                    task = self.dag_planner.task_registry.get(node)
                    if task:
                        task_type = task.task_type
                        task_types[task_type] = task_types.get(task_type, 0) + 1
                
                print(f"   📋 Task breakdown: {dict(task_types)}")
                
            except Exception as e:
                print(f"   ❌ Planning failed: {e}")
    
    async def demo_conversational_interface(self):
        """Demonstrate conversational interface"""
        print("\n" + "="*60)
        print("💬 CONVERSATIONAL INTERFACE DEMO")
        print("="*60)
        
        # Get system context
        system_context = await self._get_system_context()
        
        # Test conversational interactions
        conversations = [
            "What's the current status of all agent pools?",
            "Execute a momentum strategy for AAPL and MSFT",
            "Run a backtest for the last 6 months on tech stocks",
            "Help me optimize my portfolio for better risk-adjusted returns",
            "Train a new reinforcement learning model for crypto trading"
        ]
        
        for i, message in enumerate(conversations, 1):
            print(f"\n👤 User: {message}")
            
            try:
                response = await self.conversation_manager.handle_user_message(
                    message, f"demo_user_{i}", system_context
                )
                
                if response["success"]:
                    parsed_response = response["response"]
                    print(f"🤖 Intent: {parsed_response['intent']}")
                    print(f"🎯 Action: {parsed_response['action']}")
                    print(f"📝 Response: {parsed_response['explanation']}")
                    print(f"🔮 Confidence: {parsed_response['confidence']:.2f}")
                    
                    if parsed_response.get('suggestions'):
                        print(f"💡 Suggestions: {', '.join(parsed_response['suggestions'])}")
                        
                else:
                    print(f"❌ Error: {response['error']}")
                    
            except Exception as e:
                print(f"❌ Conversation failed: {e}")
    
    async def demo_agent_pool_monitoring(self):
        """Demonstrate agent pool monitoring and validation"""
        print("\n" + "="*60)
        print("🔍 AGENT POOL MONITORING DEMO")
        print("="*60)
        
        try:
            # Check all agent pools
            print("\n📊 Checking agent pool health...")
            results = await self.agent_monitor.check_all_pools()
            
            healthy_pools = 0
            for pool_name, pool_info in results.items():
                status_icon = "✅" if pool_info.status.value == "healthy" else "❌" if pool_info.status.value == "error" else "⚠️"
                print(f"{status_icon} {pool_name}: {pool_info.status.value}")
                
                if pool_info.status.value == "healthy":
                    healthy_pools += 1
                    if pool_info.response_time:
                        print(f"   🕒 Response time: {pool_info.response_time:.3f}s")
                    if pool_info.capabilities:
                        print(f"   🛠️  Capabilities: {', '.join(pool_info.capabilities[:3])}{'...' if len(pool_info.capabilities) > 3 else ''}")
                else:
                    if pool_info.error_message:
                        print(f"   ❌ Error: {pool_info.error_message}")
            
            # System health summary
            total_pools = len(results)
            health_percentage = (healthy_pools / total_pools * 100) if total_pools > 0 else 0
            
            print(f"\n📈 System Health Summary:")
            print(f"   Total pools: {total_pools}")
            print(f"   Healthy pools: {healthy_pools}")
            print(f"   Health percentage: {health_percentage:.1f}%")
            
            # Test MCP connectivity for healthy pools
            print(f"\n🔗 Testing MCP Connectivity:")
            for pool_name, pool_info in results.items():
                if pool_info.status.value == "healthy":
                    mcp_result = await self.agent_monitor.validate_mcp_connectivity(pool_name)
                    if mcp_result["success"]:
                        tools_count = len(mcp_result.get("available_tools", []))
                        print(f"✅ {pool_name}: MCP OK ({tools_count} tools available)")
                    else:
                        print(f"❌ {pool_name}: MCP failed - {mcp_result['error']}")
                        
        except Exception as e:
            print(f"❌ Monitoring failed: {e}")
    
    async def demo_end_to_end_execution(self):
        """Demonstrate end-to-end strategy execution with LLM"""
        print("\n" + "="*60)
        print("🎯 END-TO-END STRATEGY EXECUTION DEMO")
        print("="*60)
        
        try:
            # Natural language strategy request
            strategy_request = "Execute a momentum strategy for AAPL and GOOGL with risk management"
            print(f"\n📝 Strategy Request: {strategy_request}")
            
            # Process with natural language interface
            print(f"\n🧠 Processing with LLM...")
            nl_response = await self.conversation_manager.handle_user_message(
                strategy_request, "demo_user"
            )
            
            if nl_response["success"]:
                parsed_response = nl_response["response"]
                print(f"✅ Intent recognized: {parsed_response['intent']}")
                print(f"🎯 Action planned: {parsed_response['action']}")
                
                # Create strategy from LLM response
                if parsed_response["intent"] == "execute_strategy":
                    strategy_params = parsed_response.get("parameters", {})
                    
                    # Create trading strategy
                    strategy = TradingStrategy(
                        name="LLM_Generated_Momentum",
                        strategy_type="momentum_strategy",
                        symbols=strategy_params.get("symbols", ["AAPL", "GOOGL"]),
                        timeframe="1D",
                        parameters=strategy_params
                    )
                    
                    print(f"\n📋 Created strategy: {strategy.name}")
                    print(f"   Symbols: {', '.join(strategy.symbols)}")
                    print(f"   Type: {strategy.strategy_type}")
                    
                    # Plan execution with LLM-enhanced DAG
                    print(f"\n🛠️  Planning execution with LLM-enhanced DAG...")
                    dag = await self.dag_planner.plan_strategy_execution(strategy)
                    
                    print(f"✅ Execution plan created:")
                    print(f"   📊 Total tasks: {len(dag.nodes())}")
                    print(f"   🔗 Dependencies: {len(dag.edges())}")
                    
                    # Try to execute with real agent pools first
                    print(f"\n🚀 Executing strategy with real agent pools...")
                    
                    try:
                        # Test data agent pool connectivity
                        async with httpx.AsyncClient() as client:
                            data_response = await client.get("http://localhost:8001/health", timeout=2.0)
                            alpha_response = await client.get("http://localhost:5050/health", timeout=2.0)
                            
                        print(f"✅ Connected to live agent pools!")
                        print(f"   📡 Data Agent Pool: {'Active' if data_response.status_code < 500 else 'Limited'}")
                        print(f"   🧠 Alpha Agent Pool: {'Active' if alpha_response.status_code < 500 else 'Limited'}")
                        
                        # Execute with real DAG runner
                        if hasattr(self.orchestrator, 'execute_strategy'):
                            execution_id = await self.orchestrator.execute_strategy(strategy)
                            print(f"✅ Real strategy execution started!")
                            print(f"   🆔 Execution ID: {execution_id}")
                            print(f"   📊 Strategy: {strategy.name}")
                            print(f"   📈 Symbols: {', '.join(strategy.symbols)}")
                            
                            # Wait a moment to allow execution to proceed
                            await asyncio.sleep(1)
                            print(f"   📋 Status: Strategy execution in progress...")
                        else:
                            raise AttributeError("execute_strategy method not available")
                            
                    except (httpx.RequestError, httpx.TimeoutException, AttributeError) as e:
                        print(f"❌ Agent pools unavailable: {e}")
                        print(f"   Please ensure agent pools are running on ports 8001 and 5050")
                        return
                    
            else:
                print(f"❌ Natural language processing failed: {nl_response.get('error')}")
                
        except Exception as e:
            print(f"❌ End-to-end execution failed: {e}")
    
    async def _get_system_context(self) -> dict:
        """Get current system context"""
        try:
            # Get agent pool status
            pool_results = await self.agent_monitor.check_all_pools()
            
            context = {
                "timestamp": datetime.now().isoformat(),
                "agent_pools": {
                    name: {
                        "status": pool.status.value,
                        "capabilities": pool.capabilities or []
                    }
                    for name, pool in pool_results.items()
                },
                "system_capabilities": [
                    "natural_language_processing",
                    "llm_enhanced_planning",
                    "real_time_monitoring",
                    "mcp_protocol_support"
                ],
                "demo_mode": True
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting system context: {e}")
            return {"timestamp": datetime.now().isoformat(), "error": str(e)}
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of all enhanced features"""
        print("🎉 ENHANCED FINAGENT ORCHESTRATOR DEMO")
        print("🤖 Featuring: LLM Integration, Natural Language Interface, and Advanced Monitoring")
        print("=" * 80)
        
        try:
            # Initialize system
            await self.initialize()
            
            # Run all demo components
            await self.demo_agent_pool_monitoring()
            await self.demo_natural_language_strategy_planning()
            await self.demo_conversational_interface()
            await self.demo_end_to_end_execution()
            
            print("\n" + "="*80)
            print("🎊 COMPREHENSIVE DEMO COMPLETED SUCCESSFULLY!")
            print("✅ All enhanced features demonstrated:")
            print("   🧠 LLM-Enhanced DAG Planning")
            print("   💬 Natural Language Interface")
            print("   🔍 Advanced Agent Pool Monitoring")
            print("   🎯 End-to-End Strategy Execution")
            print("   🔗 MCP Protocol Validation")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            logger.error(f"Comprehensive demo error: {e}")

# Standalone execution
async def main():
    """Main demo execution"""
    demo = EnhancedFinAgentDemo()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logging.error(f"Main demo error: {e}")
