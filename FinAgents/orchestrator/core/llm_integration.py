"""
LLM Integration Module for FinAgent Orchestration System
Provides natural language processing capabilities for dynamic planning and interaction
"""

import json
import asyncio
import logging
import openai
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    provider: str = "openai"  # openai, anthropic, local
    model: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30

@dataclass
class ConversationContext:
    """Context for natural language conversations"""
    session_id: str
    user_id: str
    conversation_history: List[Dict[str, str]]
    system_state: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class NaturalLanguageProcessor:
    """
    Natural Language Processor for FinAgent interactions
    Handles intent recognition, context management, and response generation
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.conversations: Dict[str, ConversationContext] = {}
        self._initialize_llm_client()
        
    def _initialize_llm_client(self):
        """Initialize the LLM client based on configuration"""
        if self.config.provider == "openai":
            openai.api_key = self.config.api_key or "demo-key"
            if self.config.base_url:
                openai.base_url = self.config.base_url
                
    async def process_natural_language_request(
        self, 
        user_input: str, 
        session_id: str, 
        system_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process natural language input and generate structured responses
        
        Args:
            user_input: Natural language input from user
            session_id: Unique session identifier
            system_context: Current system state and context
            
        Returns:
            Structured response with intent, parameters, and actions
        """
        try:
            # Get or create conversation context
            context = self._get_or_create_context(session_id, system_context)
            
            # Prepare system prompt for intent recognition
            system_prompt = self._create_system_prompt(context)
            
            # Add user input to conversation history
            context.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # Generate response using LLM
            response = await self._generate_llm_response(
                system_prompt, 
                context.conversation_history
            )
            
            # Parse and structure the response
            structured_response = self._parse_llm_response(response)
            
            # Add assistant response to history
            context.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat(),
                "structured": structured_response
            })
            
            # Update context
            context.updated_at = datetime.now()
            self.conversations[session_id] = context
            
            return {
                "success": True,
                "response": structured_response,
                "raw_response": response,
                "session_id": session_id,
                "context": {
                    "conversation_length": len(context.conversation_history),
                    "last_updated": context.updated_at.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing natural language request: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    def _get_or_create_context(self, session_id: str, system_context: Dict[str, Any]) -> ConversationContext:
        """Get existing or create new conversation context"""
        if session_id not in self.conversations:
            now = datetime.now()
            self.conversations[session_id] = ConversationContext(
                session_id=session_id,
                user_id="default",
                conversation_history=[],
                system_state=system_context or {},
                metadata={},
                created_at=now,
                updated_at=now
            )
        return self.conversations[session_id]
    
    def _create_system_prompt(self, context: ConversationContext) -> str:
        """Create system prompt for LLM based on current context"""
        return f"""
You are an AI assistant for the FinAgent Orchestration System, a sophisticated financial trading platform.

SYSTEM CAPABILITIES:
- Strategy execution and backtesting
- Multi-agent coordination (Data, Alpha, Risk, Transaction Cost agents)
- Reinforcement learning model training
- Portfolio risk management
- Real-time market data processing
- DAG-based task planning and execution

CURRENT SYSTEM STATE:
{json.dumps(context.system_state, indent=2)}

YOUR ROLE:
1. Understand user requests in natural language
2. Translate requests into structured actions
3. Provide clear, actionable responses
4. Maintain conversation context
5. Guide users through complex financial operations

RESPONSE FORMAT:
Always respond with a JSON structure containing:
{{
    "intent": "primary_action_category",
    "action": "specific_action_to_take",
    "parameters": {{
        "key": "value"
    }},
    "confidence": 0.0-1.0,
    "explanation": "human_readable_explanation",
    "suggestions": ["related_actions"]
}}

SUPPORTED INTENTS:
- "execute_strategy": Run trading strategies
- "run_backtest": Execute historical backtesting
- "train_model": Train RL models
- "manage_agents": Control agent pools
- "analyze_data": Data analysis and visualization
- "manage_portfolio": Portfolio operations
- "system_status": System monitoring and health
- "configure_system": System configuration
- "help": User guidance and documentation

Be conversational but precise. Always prioritize financial safety and risk management.
"""
    
    async def _generate_llm_response(self, system_prompt: str, conversation_history: List[Dict]) -> str:
        """Generate response using configured LLM"""
        try:
            # Simulate LLM response for demo purposes
            # In production, this would make actual API calls
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add recent conversation history (last 10 messages)
            recent_history = conversation_history[-10:]
            for msg in recent_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Enhanced intent recognition with better keyword matching
            last_user_message = conversation_history[-1]["content"].lower()
            
            # Execute strategy patterns
            if any(word in last_user_message for word in ["strategy", "execute", "run", "trade", "momentum", "reversal", "arbitrage"]):
                return json.dumps({
                    "intent": "execute_strategy",
                    "action": "run_momentum_strategy",
                    "parameters": {
                        "symbols": ["AAPL", "GOOGL", "MSFT"],
                        "lookback_period": 20,
                        "threshold": 0.02
                    },
                    "confidence": 0.85,
                    "explanation": "I'll execute a momentum strategy with the specified parameters.",
                    "suggestions": ["run_backtest", "analyze_risk", "monitor_performance"]
                })
            
            # Backtest patterns
            elif any(word in last_user_message for word in ["backtest", "historical", "test", "history", "past"]):
                return json.dumps({
                    "intent": "run_backtest",
                    "action": "historical_backtest",
                    "parameters": {
                        "strategy": "momentum",
                        "start_date": "2023-01-01",
                        "end_date": "2023-12-31",
                        "initial_capital": 100000
                    },
                    "confidence": 0.90,
                    "explanation": "I'll run a historical backtest for the momentum strategy.",
                    "suggestions": ["analyze_results", "compare_strategies", "optimize_parameters"]
                })
            
            # System status patterns
            elif any(word in last_user_message for word in ["agent", "status", "health", "pool", "system", "check"]):
                return json.dumps({
                    "intent": "system_status",
                    "action": "check_agent_health",
                    "parameters": {},
                    "confidence": 0.95,
                    "explanation": "I'll check the health and status of all agent pools.",
                    "suggestions": ["restart_agents", "view_logs", "performance_metrics"]
                })
            
            # Training patterns
            elif any(word in last_user_message for word in ["train", "model", "learning", "rl", "neural", "optimize"]):
                return json.dumps({
                    "intent": "train_model",
                    "action": "train_rl_model",
                    "parameters": {
                        "model_type": "reinforcement_learning",
                        "data_period": "6_months",
                        "target_assets": "crypto"
                    },
                    "confidence": 0.80,
                    "explanation": "I'll help you train a reinforcement learning model for crypto trading.",
                    "suggestions": ["configure_model", "select_data", "monitor_training"]
                })
                
            # Portfolio patterns
            elif any(word in last_user_message for word in ["portfolio", "optimize", "risk", "allocation", "rebalance"]):
                return json.dumps({
                    "intent": "manage_portfolio",
                    "action": "optimize_portfolio",
                    "parameters": {
                        "optimization_method": "risk_adjusted_returns",
                        "constraints": "standard"
                    },
                    "confidence": 0.85,
                    "explanation": "I'll help you optimize your portfolio for better risk-adjusted returns.",
                    "suggestions": ["analyze_risk", "rebalance", "performance_metrics"]
                })
                
            # Data analysis patterns  
            elif any(word in last_user_message for word in ["analyze", "data", "chart", "visualize", "report"]):
                return json.dumps({
                    "intent": "analyze_data",
                    "action": "generate_analysis",
                    "parameters": {
                        "analysis_type": "comprehensive",
                        "data_source": "market_data"
                    },
                    "confidence": 0.88,
                    "explanation": "I'll generate a comprehensive data analysis and visualization.",
                    "suggestions": ["export_data", "create_dashboard", "schedule_reports"]
                })
                
            # Help patterns (more specific)
            elif any(word in last_user_message for word in ["help", "how", "what", "guide", "tutorial", "?"]):
                return json.dumps({
                    "intent": "help",
                    "action": "provide_guidance",
                    "parameters": {
                        "topic": "general"
                    },
                    "confidence": 0.70,
                    "explanation": "I can help you with strategy execution, backtesting, agent management, and more. What would you like to do?",
                    "suggestions": ["execute_strategy", "run_backtest", "check_status", "train_model"]
                })
                
            else:
                # Better fallback - try to extract action from context
                return json.dumps({
                    "intent": "clarification_needed",
                    "action": "request_clarification", 
                    "parameters": {
                        "user_input": last_user_message
                    },
                    "confidence": 0.60,
                    "explanation": f"I'm not sure what you'd like to do with '{last_user_message}'. Could you provide more details?",
                    "suggestions": ["execute_strategy", "run_backtest", "check_status", "analyze_data", "train_model"]
                })
                
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return json.dumps({
                "intent": "error",
                "action": "handle_error",
                "parameters": {"error": str(e)},
                "confidence": 1.0,
                "explanation": f"I encountered an error: {str(e)}",
                "suggestions": ["retry", "check_logs", "contact_support"]
            })
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            parsed = json.loads(response)
            
            # Validate required fields
            required_fields = ["intent", "action", "confidence", "explanation"]
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = f"unknown_{field}"
            
            # Ensure confidence is between 0 and 1
            if "confidence" in parsed:
                parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
            
            return parsed
            
        except json.JSONDecodeError:
            # Fallback for non-JSON responses
            return {
                "intent": "unknown",
                "action": "parse_error",
                "parameters": {},
                "confidence": 0.0,
                "explanation": response,
                "suggestions": []
            }

class ConversationManager:
    """
    Manages conversations and context across multiple sessions
    """
    
    def __init__(self, nlp: NaturalLanguageProcessor):
        self.nlp = nlp
        self.active_sessions: Dict[str, str] = {}
        
    async def handle_user_message(
        self, 
        message: str, 
        user_id: str = "default",
        system_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Handle incoming user message"""
        
        # Generate or get session ID
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if user_id in self.active_sessions:
            session_id = self.active_sessions[user_id]
        else:
            self.active_sessions[user_id] = session_id
        
        # Process message
        response = await self.nlp.process_natural_language_request(
            message, session_id, system_context
        )
        
        return response
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        if session_id in self.nlp.conversations:
            return self.nlp.conversations[session_id].conversation_history
        return []
    
    def clear_session(self, session_id: str):
        """Clear a conversation session"""
        if session_id in self.nlp.conversations:
            del self.nlp.conversations[session_id]

# Example usage and demo
if __name__ == "__main__":
    async def demo():
        """Demonstrate natural language processing capabilities"""
        
        # Initialize LLM config
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7
        )
        
        # Initialize NLP and conversation manager
        nlp = NaturalLanguageProcessor(config)
        conversation_manager = ConversationManager(nlp)
        
        # Example system context
        system_context = {
            "agent_pools": {
                "data": {"status": "running", "port": 8001},
                "alpha": {"status": "running", "port": 5050},
                "risk": {"status": "running", "port": 7000},
                "transaction_cost": {"status": "running", "port": 6000}
            },
            "orchestrator": {"status": "running", "port": 9000},
            "active_strategies": ["momentum", "mean_reversion"],
            "portfolio": {
                "total_value": 150000,
                "positions": ["AAPL", "GOOGL", "MSFT"]
            }
        }
        
        # Demo conversations
        test_messages = [
            "Execute a momentum strategy for tech stocks",
            "Run a backtest for the last year",
            "What's the status of all agent pools?",
            "Help me train a new RL model"
        ]
        
        print("🤖 FinAgent Natural Language Demo")
        print("=" * 50)
        
        for message in test_messages:
            print(f"\n👤 User: {message}")
            
            response = await conversation_manager.handle_user_message(
                message, "demo_user", system_context
            )
            
            if response["success"]:
                parsed_response = response["response"]
                print(f"🤖 Intent: {parsed_response['intent']}")
                print(f"🎯 Action: {parsed_response['action']}")
                print(f"📝 Explanation: {parsed_response['explanation']}")
                print(f"💡 Suggestions: {', '.join(parsed_response.get('suggestions', []))}")
            else:
                print(f"❌ Error: {response['error']}")
        
        print("\n✅ Demo completed!")

    # Run demo
    asyncio.run(demo())
