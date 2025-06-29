#!/usr/bin/env python3
"""
FinAgent Natural Language Command Line Interface
Interactive command line tool for natural language interaction with FinAgent system
"""

import asyncio
import cmd
import sys
import os
import json
import yaml
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.llm_integration import LLMConfig, NaturalLanguageProcessor, ConversationManager
from core.mcp_nl_interface import MCPNaturalLanguageInterface
from core.agent_pool_monitor import AgentPoolMonitor

class FinAgentCLI(cmd.Cmd):
    """
    Interactive command line interface for FinAgent natural language interactions
    """
    
    intro = """
🤖 Welcome to FinAgent Natural Language Interface!
   
Type natural language commands to interact with the trading system:
  • "Execute a momentum strategy for AAPL and GOOGL"
  • "Run a backtest for the last year"
  • "What's the status of all agent pools?"
  • "Help me train a new RL model"

Type 'help' for available commands or 'quit' to exit.
"""
    
    prompt = "FinAgent> "
    
    def __init__(self):
        super().__init__()
        self.config = self._load_config()
        self.conversation_manager = None
        self.agent_monitor = None
        self.nl_interface = None
        self.session_id = f"cli_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.initialized = False
    
    def _load_config(self):
        """Load configuration"""
        try:
            config_path = Path("config/orchestrator_config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            print(f"⚠️  Warning: Could not load config ({e}), using defaults")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Default configuration"""
        return {
            "agent_pools": {
                "data_agent_pool": {"url": "http://localhost:8001", "enabled": True},
                "alpha_agent_pool": {"url": "http://localhost:5050", "enabled": True},
                "risk_agent_pool": {"url": "http://localhost:7000", "enabled": True},
                "transaction_cost_agent_pool": {"url": "http://localhost:6000", "enabled": True}
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7
            }
        }
    
    async def _initialize_async(self):
        """Initialize async components"""
        if self.initialized:
            return
        
        try:
            print("🚀 Initializing FinAgent systems...")
            
            # Initialize LLM components
            llm_config = LLMConfig(
                provider=self.config.get("llm", {}).get("provider", "openai"),
                model=self.config.get("llm", {}).get("model", "gpt-4"),
                temperature=self.config.get("llm", {}).get("temperature", 0.7)
            )
            
            nlp = NaturalLanguageProcessor(llm_config)
            self.conversation_manager = ConversationManager(nlp)
            
            # Initialize monitoring
            self.agent_monitor = AgentPoolMonitor(self.config)
            
            # Initialize natural language interface
            self.nl_interface = MCPNaturalLanguageInterface(self.config)
            
            print("✅ FinAgent systems initialized!")
            self.initialized = True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            print("   Some features may not be available.")
    
    def default(self, line):
        """Handle natural language input"""
        if not line.strip():
            return
        
        # Run async processing
        asyncio.run(self._process_natural_language(line))
    
    def do_EOF(self, line):
        """Handle EOF (Ctrl+D)"""
        print("\n👋 Goodbye!")
        return True
    
    def emptyline(self):
        """Handle empty line input"""
        pass  # Don't repeat last command like default cmd behavior
    
    async def _process_natural_language(self, user_input):
        """Process natural language input"""
        try:
            # Initialize if not done
            await self._initialize_async()
            
            if not self.conversation_manager:
                print("❌ Natural language processing not available")
                return
            
            print(f"🤖 Processing: {user_input}")
            
            # Get system context
            system_context = await self._get_system_context()
            
            # Process with conversation manager
            response = await self.conversation_manager.handle_user_message(
                user_input, "cli_user", system_context
            )
            
            if response["success"]:
                parsed_response = response["response"]
                
                print(f"\n📋 Analysis:")
                print(f"   Intent: {parsed_response['intent']}")
                print(f"   Action: {parsed_response['action']}")
                print(f"   Confidence: {parsed_response['confidence']:.2f}")
                
                print(f"\n🤖 Response:")
                print(f"   {parsed_response['explanation']}")
                
                if parsed_response.get('suggestions'):
                    print(f"\n💡 Suggestions:")
                    for suggestion in parsed_response['suggestions'][:3]:
                        print(f"   • {suggestion}")
                
                # Try to execute the action if possible
                await self._try_execute_action(parsed_response)
                
            else:
                print(f"❌ Error: {response['error']}")
                
        except Exception as e:
            print(f"❌ Processing failed: {e}")
    
    async def _try_execute_action(self, parsed_response):
        """Try to execute the parsed action"""
        intent = parsed_response.get("intent")
        action = parsed_response.get("action")
        
        try:
            if intent == "system_status":
                await self._show_system_status()
            elif intent == "execute_strategy":
                await self._simulate_strategy_execution(parsed_response)
            elif intent == "run_backtest":
                await self._simulate_backtest(parsed_response)
            elif intent == "help":
                self._show_help()
        except Exception as e:
            print(f"⚠️  Action execution failed: {e}")
    
    async def _show_system_status(self):
        """Show system status"""
        print(f"\n📊 System Status:")
        
        if self.agent_monitor:
            try:
                results = await self.agent_monitor.check_all_pools()
                
                for pool_name, pool_info in results.items():
                    status_icon = "✅" if pool_info.status.value == "healthy" else "❌"
                    print(f"   {status_icon} {pool_name}: {pool_info.status.value}")
                    if pool_info.response_time:
                        print(f"      Response time: {pool_info.response_time:.3f}s")
                
                healthy_count = sum(1 for pool in results.values() if pool.status.value == "healthy")
                total_count = len(results)
                print(f"\n   Health: {healthy_count}/{total_count} pools operational")
                
            except Exception as e:
                print(f"   ❌ Status check failed: {e}")
        else:
            print("   ⚠️  System monitoring not available")
    
    async def _simulate_strategy_execution(self, parsed_response):
        """Simulate strategy execution"""
        parameters = parsed_response.get("parameters", {})
        symbols = parameters.get("symbols", ["AAPL", "GOOGL"])
        
        print(f"\n🚀 Simulating Strategy Execution:")
        print(f"   Strategy: {parameters.get('strategy_type', 'momentum')}")
        print(f"   Symbols: {', '.join(symbols)}")
        print(f"   Parameters: {json.dumps(parameters, indent=6)[6:-1]}")
        
        # Simulate processing
        import random
        await asyncio.sleep(1)
        
        print(f"\n✅ Execution Results:")
        print(f"   Signals generated: {random.randint(2, 5)}")
        print(f"   Expected return: {random.uniform(5, 15):.1f}%")
        print(f"   Risk score: {random.uniform(0.1, 0.3):.2f}")
    
    async def _simulate_backtest(self, parsed_response):
        """Simulate backtest execution"""
        parameters = parsed_response.get("parameters", {})
        
        print(f"\n📈 Simulating Backtest:")
        print(f"   Period: {parameters.get('start_date', '2023-01-01')} to {parameters.get('end_date', '2023-12-31')}")
        print(f"   Strategy: {parameters.get('strategy', 'momentum')}")
        print(f"   Capital: ${parameters.get('initial_capital', 100000):,}")
        
        # Simulate processing
        import random
        await asyncio.sleep(2)
        
        print(f"\n✅ Backtest Results:")
        print(f"   Total return: {random.uniform(10, 30):.1f}%")
        print(f"   Sharpe ratio: {random.uniform(1.2, 2.5):.2f}")
        print(f"   Max drawdown: -{random.uniform(5, 15):.1f}%")
    
    def _show_help(self):
        """Show help information"""
        print(f"\n📖 FinAgent Natural Language Interface Help:")
        print(f"\nSupported Commands:")
        print(f"   Strategy Execution:")
        print(f"     • 'Execute a momentum strategy for AAPL and GOOGL'")
        print(f"     • 'Run a mean reversion strategy for tech stocks'")
        print(f"   ")
        print(f"   Backtesting:")
        print(f"     • 'Run a backtest for the last year'")
        print(f"     • 'Backtest my strategy on crypto data'")
        print(f"   ")
        print(f"   System Management:")
        print(f"     • 'What is the status of all agent pools?'")
        print(f"     • 'Check system health'")
        print(f"   ")
        print(f"   Model Training:")
        print(f"     • 'Train a new RL model for options trading'")
        print(f"     • 'Help me optimize my neural network'")
        print(f"\nBuilt-in Commands:")
        print(f"   status    - Show system status")
        print(f"   history   - Show conversation history")
        print(f"   clear     - Clear screen")
        print(f"   quit      - Exit interface")
    
    async def _get_system_context(self):
        """Get system context for processing"""
        context = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "interface": "cli"
        }
        
        if self.agent_monitor:
            try:
                pool_results = await self.agent_monitor.check_all_pools()
                context["agent_pools"] = {
                    name: {"status": pool.status.value, "capabilities": pool.capabilities or []}
                    for name, pool in pool_results.items()
                }
            except:
                context["agent_pools"] = {}
        
        return context
    
    # Built-in commands
    def do_status(self, arg):
        """Show system status"""
        asyncio.run(self._show_system_status())
    
    def do_history(self, arg):
        """Show conversation history"""
        if self.conversation_manager and self.session_id in self.conversation_manager.nlp.conversations:
            history = self.conversation_manager.get_conversation_history(self.session_id)
            print(f"\n📜 Conversation History ({len(history)} messages):")
            for i, msg in enumerate(history[-10:], 1):  # Show last 10 messages
                role_icon = "👤" if msg["role"] == "user" else "🤖"
                print(f"   {i}. {role_icon} {msg['content'][:80]}{'...' if len(msg['content']) > 80 else ''}")
        else:
            print("   No conversation history available")
    
    def do_clear(self, arg):
        """Clear screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def do_quit(self, arg):
        """Exit the interface"""
        print("👋 Goodbye! Thank you for using FinAgent!")
        return True
    
    def do_exit(self, arg):
        """Exit the interface"""
        return self.do_quit(arg)
    
    def do_help(self, arg):
        """Show help"""
        if arg:
            super().do_help(arg)
        else:
            self._show_help()

def main():
    """Main entry point"""
    try:
        cli = FinAgentCLI()
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
