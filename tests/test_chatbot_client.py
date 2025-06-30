#!/usr/bin/env python3
"""
FinAgent Chatbot Client Test

This script simulates a chatbot interaction with the FinAgent system,
demonstrating how users can execute backtests using natural language
through a conversational interface.
"""

import asyncio
import logging
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger("FinAgentChatbot")


class FinAgentChatbotClient:
    """Simulated chatbot client for FinAgent interaction"""
    
    def __init__(self):
        self.session_id = f"chatbot_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history = []
        
    async def run_chatbot_test(self):
        """Run chatbot interaction test"""
        print("🤖 FinAgent Chatbot Client Test")
        print("=" * 50)
        
        # Simulate conversation flow
        await self._simulate_conversation()
        
        # Show conversation summary
        await self._show_conversation_summary()
    
    async def _simulate_conversation(self):
        """Simulate a realistic conversation with FinAgent"""
        
        # Define conversation turns
        conversation_turns = [
            {
                "user": "Hello! I want to run a backtest for my investment strategy.",
                "expected_intent": "greeting_and_backtest_request"
            },
            {
                "user": "I'm interested in trading AAPL and MSFT using a momentum strategy. Can you help me set this up?",
                "expected_intent": "strategy_setup_request"
            },
            {
                "user": "I want to test this strategy over the last 3 years with $1 million starting capital. Please include proper risk management.",
                "expected_intent": "backtest_configuration"
            },
            {
                "user": "Make sure to include transaction costs and optimize the portfolio allocation. I want realistic results.",
                "expected_intent": "advanced_backtest_requirements"
            },
            {
                "user": "Run the backtest now and show me the performance metrics and risk analysis.",
                "expected_intent": "backtest_execution_request"
            },
            {
                "user": "Can you explain why the strategy performed well or poorly? What were the main drivers?",
                "expected_intent": "performance_attribution_request"
            },
            {
                "user": "Based on the results, what modifications would you suggest to improve the strategy?",
                "expected_intent": "strategy_improvement_request"
            }
        ]
        
        print("💬 Starting Conversation Simulation...\n")
        
        for i, turn in enumerate(conversation_turns, 1):
            # User message
            print(f"👤 User: {turn['user']}")
            
            # Simulate processing
            response = await self._process_user_message(turn['user'], turn['expected_intent'])
            
            # Assistant response
            print(f"🤖 FinAgent: {response['message']}")
            
            # Add to conversation history
            self.conversation_history.append({
                "turn": i,
                "user_message": turn['user'],
                "assistant_response": response,
                "timestamp": datetime.now().isoformat()
            })
            
            print()
            
            # Simulate thinking time
            await asyncio.sleep(1)
    
    async def _process_user_message(self, user_message: str, expected_intent: str) -> Dict[str, Any]:
        """Process user message and generate appropriate response"""
        
        # Simulate different response types based on intent
        responses = {
            "greeting_and_backtest_request": {
                "message": "Hello! I'd be happy to help you run a backtest. I can coordinate with our data, alpha generation, portfolio optimization, risk management, and transaction cost analysis agents to provide comprehensive results. What strategy would you like to test?",
                "intent": expected_intent,
                "confidence": 0.95,
                "next_steps": ["strategy_specification", "symbol_selection", "timeframe_setup"]
            },
            
            "strategy_setup_request": {
                "message": "Excellent choice! I'll set up a momentum strategy for AAPL and MSFT. Our alpha agent pool can implement momentum signals using various lookback periods and technical indicators. Would you like me to include mean reversion filters to reduce false signals?",
                "intent": expected_intent,
                "confidence": 0.92,
                "next_steps": ["parameter_configuration", "timeframe_selection", "risk_preferences"]
            },
            
            "backtest_configuration": {
                "message": "Perfect! I'll configure a 3-year backtest with $1M initial capital. I'm coordinating with our risk management agents to implement proper position sizing, drawdown limits, and volatility targeting. What's your risk tolerance level - conservative, moderate, or aggressive?",
                "intent": expected_intent,
                "confidence": 0.88,
                "next_steps": ["risk_parameter_setup", "rebalancing_frequency", "benchmark_selection"]
            },
            
            "advanced_backtest_requirements": {
                "message": "Absolutely! I'm engaging our transaction cost agent pool to model realistic execution costs including market impact and bid-ask spreads. Our portfolio construction agents will optimize allocations using modern portfolio theory. This will give you highly realistic backtest results.",
                "intent": expected_intent,
                "confidence": 0.91,
                "execution_status": "configuring_agents"
            },
            
            "backtest_execution_request": {
                "message": "Executing your backtest now! I'm coordinating across all agent pools:\n\n📊 Data agents: Fetching 3 years of AAPL/MSFT data\n🧠 Alpha agents: Generating momentum signals\n📈 Portfolio agents: Optimizing allocations\n💰 Cost agents: Calculating execution costs\n🛡️ Risk agents: Monitoring drawdowns\n\nResults coming up...",
                "intent": expected_intent,
                "confidence": 0.96,
                "execution_results": await self._generate_sample_results()
            },
            
            "performance_attribution_request": {
                "message": "Great question! Based on the backtest analysis:\n\n✅ **Key Performance Drivers:**\n• Momentum signals captured major trends effectively (+12% alpha)\n• Portfolio optimization reduced volatility while maintaining returns (+3% risk-adjusted)\n• Transaction cost management saved 85bps annually\n\n⚠️ **Areas for improvement:**\n• Strategy struggled during sideways markets (Q2 2023)\n• Drawdown exceeded target during market volatility spikes",
                "intent": expected_intent,
                "confidence": 0.89,
                "attribution_details": True
            },
            
            "strategy_improvement_request": {
                "message": "Based on the performance analysis, here are my recommendations:\n\n🔧 **Strategy Enhancements:**\n1. Add regime detection to reduce exposure during sideways markets\n2. Implement dynamic position sizing based on volatility\n3. Include sector rotation signals for broader diversification\n4. Add earnings announcement filters to avoid event risk\n\nWould you like me to run a new backtest with these improvements?",
                "intent": expected_intent,
                "confidence": 0.87,
                "recommendations": ["regime_detection", "dynamic_sizing", "sector_rotation", "event_filters"]
            }
        }
        
        # Return appropriate response
        return responses.get(expected_intent, {
            "message": "I understand your request. Let me coordinate with the appropriate agent pools to provide you with the best possible assistance.",
            "intent": "general_request",
            "confidence": 0.75
        })
    
    async def _generate_sample_results(self) -> Dict[str, Any]:
        """Generate sample backtest results for demonstration"""
        return {
            "performance_metrics": {
                "total_return": "28.5%",
                "annual_return": "8.7%",
                "volatility": "15.2%",
                "sharpe_ratio": "0.54",
                "max_drawdown": "-18.3%",
                "win_rate": "58%"
            },
            "agent_contributions": {
                "data_quality": "High (99.8% complete data)",
                "alpha_generation": "Strong momentum signals (12% alpha)",
                "portfolio_optimization": "Effective risk-return balance",
                "transaction_costs": "Well-managed (85bps savings)",
                "risk_management": "Successful drawdown control"
            },
            "execution_summary": {
                "total_trades": 156,
                "avg_holding_period": "23 days",
                "turnover": "280% annually",
                "cost_ratio": "0.65%"
            }
        }
    
    async def _show_conversation_summary(self):
        """Show conversation summary and analysis"""
        print("📋 Conversation Summary")
        print("=" * 50)
        
        print(f"Session ID: {self.session_id}")
        print(f"Total Turns: {len(self.conversation_history)}")
        print(f"Duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n🎯 Intents Processed:")
        for turn in self.conversation_history:
            intent = turn['assistant_response'].get('intent', 'unknown')
            confidence = turn['assistant_response'].get('confidence', 0)
            print(f"  {turn['turn']}. {intent} (confidence: {confidence:.2f})")
        
        print("\n✅ Demonstrated Capabilities:")
        capabilities = [
            "Natural language understanding",
            "Multi-agent coordination",
            "Backtest configuration",
            "Real-time execution monitoring",
            "Performance attribution analysis",
            "Strategy improvement recommendations"
        ]
        
        for capability in capabilities:
            print(f"  • {capability}")
        
        print("\n🚀 This chatbot demonstrates how users can:")
        print("  • Interact naturally with complex financial systems")
        print("  • Execute sophisticated backtests without technical knowledge")
        print("  • Get real-time feedback and explanations")
        print("  • Receive actionable insights and recommendations")
        print("  • Coordinate multiple specialized agent pools seamlessly")


async def run_chatbot_demo():
    """Run the chatbot demonstration"""
    client = FinAgentChatbotClient()
    await client.run_chatbot_test()


if __name__ == "__main__":
    print("🤖 Starting FinAgent Chatbot Client Demo...")
    asyncio.run(run_chatbot_demo())
    print("✅ Chatbot demo completed successfully!")
