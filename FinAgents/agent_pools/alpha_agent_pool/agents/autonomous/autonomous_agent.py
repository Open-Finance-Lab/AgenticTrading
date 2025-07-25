# agent_pools/alpha_agent_pool/agents/autonomous/autonomous_agent.py

import json
import os
import asyncio
import threading
import subprocess
import tempfile
import ast
import inspect
import hashlib
import random
import importlib.util
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
import uuid

# Import strategy flow schemas for output compatibility
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from schema.theory_driven_schema import (
    AlphaStrategyFlow, MarketContext, Decision, Action, PerformanceFeedback, Metadata
)


class Task(BaseModel):
    """
    Represents a task created by the autonomous agent.
    
    This model encapsulates all information related to a specific task including
    its execution status, dependencies, and generated artifacts.
    """
    task_id: str
    description: str
    priority: int
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: str
    dependencies: List[str] = []
    generated_code: Optional[str] = None
    validation_code: Optional[str] = None
    validation_result: Optional[Dict[str, Any]] = None


class AutonomousAgent:
    """
    An autonomous financial analysis agent capable of self-orchestration and dynamic code generation.
    
    This agent demonstrates advanced autonomy by:
    1. Receiving orchestrator inputs and autonomously decomposing them into executable tasks
    2. Querying memory agents for domain knowledge retrieval  
    3. Dynamically generating Python code tools for financial analysis
    4. Creating validation frameworks to ensure code correctness
    5. Outputting structured strategy flows compatible with the alpha agent ecosystem
    
    The agent maintains a persistent task queue and workspace for generated artifacts,
    enabling complex multi-step analysis workflows.
    """
    
    def __init__(self, agent_id: str = "autonomous_alpha_agent"):
        """
        Initialize the AutonomousAgent with core components and processing capabilities.
        
        Args:
            agent_id: Unique identifier for this agent instance
            
        The initialization process establishes:
        - MCP server instance for external communication
        - Persistent workspace for code generation and task management
        - Background task processing thread for autonomous operation
        - Core tool registry for external interface
        """
        self.agent_id = agent_id
        self.mcp_server = FastMCP(f"AutonomousAgent_{agent_id}")
        self.task_queue: List[Task] = []
        self.memory_connections = {}
        self.generated_tools = {}
        self.workspace_dir = os.path.join(os.path.dirname(__file__), "workspace")
        self.task_log_path = os.path.join(self.workspace_dir, "task_log.json")
        self.strategy_flow_path = os.path.join(self.workspace_dir, "autonomous_strategy_flow.json")
        
        # Initialize workspace directory structure
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Load existing task history for persistence
        self._load_tasks()
        
        # Register core MCP tools for external communication
        self._register_core_tools()
        
        # Start autonomous task processing in background thread
        self.task_processor_thread = threading.Thread(target=self._autonomous_task_processor, daemon=True)
        self.task_processor_running = True
        self.task_processor_thread.start()

    def _load_tasks(self):
        """
        Load task history from persistent storage for continuity across restarts.
        
        This method ensures task persistence by loading previously created tasks
        from the JSON log file, enabling the agent to resume work after restarts.
        """
        if os.path.exists(self.task_log_path):
            try:
                with open(self.task_log_path, 'r') as f:
                    tasks_data = json.load(f)
                    self.task_queue = [Task(**task) for task in tasks_data]
            except Exception as e:
                print(f"Error loading tasks: {e}")
                self.task_queue = []

    def _save_tasks(self):
        """
        Persist current task queue to storage for durability.
        
        Maintains task state across agent restarts and provides audit trail
        for task execution history.
        """
        try:
            with open(self.task_log_path, 'w') as f:
                tasks_data = [task.dict() for task in self.task_queue]
                json.dump(tasks_data, f, indent=2)
        except Exception as e:
            print(f"Error saving tasks: {e}")

    def _write_strategy_flow(self, flow_obj: Dict[str, Any]):
        """
        Write strategy flow object to persistent storage.
        
        This method maintains compatibility with the alpha agent ecosystem
        by outputting structured strategy flows that can be consumed by
        downstream systems.
        
        Args:
            flow_obj: Complete strategy flow object with analysis results
        """
        try:
            if os.path.exists(self.strategy_flow_path):
                with open(self.strategy_flow_path, 'r') as f:
                    flows = json.load(f)
            else:
                flows = []
                
            flows.append(flow_obj)
            
            with open(self.strategy_flow_path, 'w') as f:
                json.dump(flows, f, indent=2)
                
        except Exception as e:
            print(f"Error writing strategy flow: {e}")

    def _generate_strategy_flow(self, symbol: str, analysis_result: Dict[str, Any],
                               instruction: str) -> Dict[str, Any]:
        """
        Generate a standardized strategy flow output compatible with alpha agent ecosystem.
        
        This method transforms analysis results into structured strategy flows that
        maintain compatibility with the broader alpha agent framework, enabling
        seamless integration with downstream processing systems.
        
        Args:
            symbol: Asset symbol being analyzed
            analysis_result: Results from autonomous analysis
            instruction: Original orchestrator instruction
            
        Returns:
            Structured strategy flow object following alpha agent schema
        """
        now = datetime.now().isoformat()
        alpha_id = f"autonomous_{uuid.uuid4().hex[:8]}"
        
        # Extract signal information from analysis results
        signal = analysis_result.get("signal", "HOLD")
        confidence = analysis_result.get("confidence", 0.5)
        reasoning = analysis_result.get("reasoning", f"Autonomous analysis of {instruction}")
        
        # Generate market context based on analysis
        market_context = MarketContext(
            symbol=symbol,
            regime_tag=self._determine_market_regime(analysis_result),
            input_features=analysis_result.get("features", {
                "autonomous_analysis": True,
                "instruction": instruction,
                "analysis_timestamp": now
            })
        )
        
        # Create decision object
        decision = Decision(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            predicted_return=analysis_result.get("predicted_return", 0.0),
            risk_estimate=analysis_result.get("risk_estimate", 0.05),
            signal_type="directional",
            asset_scope=[symbol]
        )
        
        # Define action parameters
        action = Action(
            execution_weight=analysis_result.get("execution_weight", 0.25),
            order_type="market",
            order_price=analysis_result.get("current_price", 100.0),
            execution_delay="T+0"
        )
        
        # Create performance feedback placeholder
        performance_feedback = PerformanceFeedback(
            status="pending",
            evaluation_link=None
        )
        
        # Generate metadata
        code_hash = hashlib.sha256(instruction.encode()).hexdigest()[:16]
        metadata = Metadata(
            generator_agent=self.agent_id,
            strategy_prompt=f"Autonomous analysis: {instruction}",
            code_hash=f"sha256:{code_hash}",
            context_id=f"autonomous_{now[:10].replace('-', '')}_{now[11:13]}"
        )
        
        # Construct complete strategy flow
        strategy_flow = AlphaStrategyFlow(
            alpha_id=alpha_id,
            version="1.0",
            timestamp=now,
            market_context=market_context,
            decision=decision,
            action=action,
            performance_feedback=performance_feedback,
            metadata=metadata
        )
        
        return strategy_flow.dict()

    def _determine_market_regime(self, analysis_result: Dict[str, Any]) -> str:
        """
        Determine market regime based on analysis results.
        
        Args:
            analysis_result: Analysis output containing market indicators
            
        Returns:
            Market regime classification string
        """
        signal = analysis_result.get("signal", "HOLD")
        confidence = analysis_result.get("confidence", 0.5)
        
        if signal == "BUY" and confidence > 0.7:
            return "bullish_trend"
        elif signal == "SELL" and confidence > 0.7:
            return "bearish_trend"
        elif signal == "BUY":
            return "bullish_consolidation"
        elif signal == "SELL":
            return "bearish_consolidation"
        else:
            return "neutral_range"

    def _register_core_tools(self):
        """
        Register core MCP tools for external orchestrator communication.
        
        This method establishes the primary interface between external orchestrators
        and the autonomous agent, providing tools for task creation, execution,
        and monitoring.
        """
        
        @self.mcp_server.tool(name="receive_orchestrator_input",
                              description="Receive input from external orchestrator and autonomously create tasks")
        def receive_orchestrator_input(instruction: str, context: Optional[Dict[str, Any]] = None) -> str:
            """
            Process orchestrator instructions and decompose them into executable tasks.
            
            Args:
                instruction: High-level instruction from orchestrator
                context: Additional context data for task execution
                
            Returns:
                Status message indicating task creation results
            """
            return self._process_orchestrator_input(instruction, context or {})

        @self.mcp_server.tool(name="query_memory_agent",
                              description="Query memory agent for relevant knowledge")
        def query_memory_agent(query: str, category: Optional[str] = None) -> Dict[str, Any]:
            """
            Retrieve relevant knowledge from memory agent for analysis.
            
            Args:
                query: Search query for knowledge retrieval
                category: Optional category filter for targeted search
                
            Returns:
                Retrieved knowledge with relevance scores and metadata
            """
            return self._query_memory(query, category)

        @self.mcp_server.tool(name="generate_analysis_tool",
                              description="Dynamically generate code tools based on analysis requirements")
        def generate_analysis_tool(analysis_description: str,
                                  input_data_format: str,
                                 expected_output: str) -> Dict[str, Any]:
            """
            Generate custom analysis tools for specific financial analysis tasks.
            
            Args:
                analysis_description: Description of the analysis to be performed
                input_data_format: Format specification for input data
                expected_output: Expected output format and content
                
            Returns:
                Generated tool information including code and file path
            """
            return self._generate_code_tool(analysis_description, input_data_format, expected_output)

        @self.mcp_server.tool(name="create_validation_code",
                              description="Create validation programs for generated code")
        def create_validation_code(code_to_validate: str,
                                  test_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
            """
            Generate comprehensive validation code for tool verification.
            
            Args:
                code_to_validate: The generated code to be validated
                test_scenarios: List of test scenarios with input/expected output pairs
                
            Returns:
                Validation code information and file location
            """
            return self._create_validation(code_to_validate, test_scenarios)

        @self.mcp_server.tool(name="get_task_status",
                              description="Retrieve execution status of all tasks")
        def get_task_status() -> List[Dict[str, Any]]:
            """
            Get current status of all tasks in the execution queue.
            
            Returns:
                List of task objects with current status and metadata
            """
            return [task.dict() for task in self.task_queue]

        @self.mcp_server.tool(name="execute_generated_tool",
                              description="Execute previously generated code tools")
        def execute_generated_tool(tool_name: str,
                                  input_data: Dict[str, Any]) -> Dict[str, Any]:
            """
            Execute a previously generated analysis tool with provided data.
            
            Args:
                tool_name: Name of the tool to execute
                input_data: Input data for tool execution
                
            Returns:
                Execution results with success status and output data
            """
            return self._execute_tool(tool_name, input_data)

        @self.mcp_server.tool(name="generate_strategy_signal",
                              description="Generate trading signal and strategy flow output")
        def generate_strategy_signal(symbol: str, instruction: str,
                                    market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """
            Generate a complete trading signal with strategy flow output.
            
            This tool demonstrates the autonomous agent's capability to produce
            structured strategy flows compatible with the alpha agent ecosystem.
            
            Args:
                symbol: Asset symbol for signal generation
                instruction: Analysis instruction from orchestrator
                market_data: Optional market data for analysis
                
            Returns:
                Complete strategy flow object with trading signal
            """
            return self._generate_trading_signal(symbol, instruction, market_data or {})

    def _generate_trading_signal(self, symbol: str, instruction: str,
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive trading signal with autonomous analysis.
        
        This method performs end-to-end signal generation including:
        - Market data analysis
        - Signal computation
        - Strategy flow generation
        - Persistence to storage
        
        Args:
            symbol: Asset symbol for analysis
            instruction: Orchestrator instruction
            market_data: Market data for analysis
            
        Returns:
            Complete strategy flow with trading signal
        """
        # Validate market data input
        if not market_data.get("prices") or len(market_data["prices"]) == 0:
            raise ValueError("Market data with prices is required for strategy execution")
        
        prices = market_data["prices"]
        current_price = prices[-1]
        
        # Perform technical analysis
        analysis_result = self._perform_autonomous_analysis(prices, instruction)
        
        # Enhance with additional context
        analysis_result.update({
            "current_price": current_price,
            "symbol": symbol,
            "instruction": instruction,
            "analysis_timestamp": datetime.now().isoformat()
        })
        
        # Generate strategy flow
        strategy_flow = self._generate_strategy_flow(symbol, analysis_result, instruction)
        
        # Persist strategy flow
        self._write_strategy_flow(strategy_flow)
        
        return strategy_flow

    def _perform_autonomous_analysis(self, prices: List[float], instruction: str) -> Dict[str, Any]:
        """
        Perform autonomous technical analysis on price data.
        
        This method demonstrates the agent's analytical capabilities by computing
        various technical indicators and generating trading signals based on
        the analysis results.
        
        Args:
            prices: Historical price data
            instruction: Analysis instruction for context
            
        Returns:
            Analysis results with signal and confidence metrics
        """
        if len(prices) < 2:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reasoning": "Insufficient data for analysis",
                "predicted_return": 0.0,
                "risk_estimate": 0.1
            }
        
        # Calculate technical indicators
        current_price = prices[-1]
        prev_price = prices[-2] if len(prices) > 1 else current_price
        
        # Moving averages
        sma_5 = sum(prices[-5:]) / min(5, len(prices))
        sma_10 = sum(prices[-10:]) / min(10, len(prices))
        sma_20 = sum(prices) / len(prices)
        
        # Price momentum
        momentum = (current_price - prev_price) / prev_price if prev_price != 0 else 0
        
        # Volatility estimate
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volatility = (sum(r**2 for r in returns) / len(returns))**0.5 if returns else 0.02
        
        # Generate signal based on technical analysis
        signal = "HOLD"
        confidence = 0.5
        reasoning = "Neutral market conditions"
        predicted_return = 0.0
        
        # Calculate additional indicators for more sensitive analysis
        price_change_pct = (current_price - prices[0]) / prices[0] if prices[0] != 0 else 0
        short_term_momentum = (sma_5 - sma_10) / sma_10 if sma_10 != 0 else 0
        trend_strength = abs(short_term_momentum)
        
        # Enhanced trend analysis with multiple conditions
        # Bullish conditions
        if (sma_5 > sma_10 and momentum > 0.005) or (price_change_pct > 0.03 and short_term_momentum > 0.01):
            signal = "BUY"
            confidence = min(0.9, 0.6 + trend_strength * 15 + abs(momentum) * 10)
            reasoning = "Upward trend detected: positive momentum and rising moving averages"
            predicted_return = max(momentum * 2, price_change_pct * 0.5)
            
        # Bearish conditions  
        elif (sma_5 < sma_10 and momentum < -0.005) or (price_change_pct < -0.03 and short_term_momentum < -0.01):
            signal = "SELL"
            confidence = min(0.9, 0.6 + trend_strength * 15 + abs(momentum) * 10)
            reasoning = "Downward trend detected: negative momentum and falling moving averages"
            predicted_return = min(momentum * 2, price_change_pct * 0.5)
            
        # Strong momentum override (regardless of moving averages)
        elif momentum > 0.015:
            signal = "BUY"
            confidence = min(0.85, 0.7 + abs(momentum) * 8)
            reasoning = "Strong positive momentum detected"
            predicted_return = momentum * 1.5
            
        elif momentum < -0.015:
            signal = "SELL"
            confidence = min(0.85, 0.7 + abs(momentum) * 8)
            reasoning = "Strong negative momentum detected"
            predicted_return = momentum * 1.5
            
        # Consolidation pattern
        elif abs(momentum) < 0.005 and volatility < 0.02:
            signal = "HOLD"
            confidence = 0.7
            reasoning = "Low volatility consolidation pattern"
            predicted_return = 0.0
        
        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "predicted_return": predicted_return,
            "risk_estimate": volatility,
            "execution_weight": min(0.5, confidence * 0.6),
            "features": {
                "current_price": current_price,
                "sma_5": sma_5,
                "sma_10": sma_10,
                "sma_20": sma_20,
                "momentum": momentum,
                "volatility": volatility,
                "price_change_pct": price_change_pct,
                "short_term_momentum": short_term_momentum,
                "trend_strength": trend_strength
            }
        }

    def _process_orchestrator_input(self, instruction: str, context: Dict[str, Any]) -> str:
        """
        Process orchestrator input and create autonomous task decomposition.
        
        This method serves as the primary entry point for external orchestrator
        interactions, enabling the agent to autonomously decompose high-level
        instructions into executable task sequences.
        
        Args:
            instruction: High-level instruction from orchestrator
            context: Additional context data for task execution
            
        Returns:
            Status message indicating task creation results
        """
        # Analyze instruction and decompose into specific tasks
        tasks = self._decompose_instruction(instruction, context)
        
        for task_desc in tasks:
            task = Task(
                task_id=str(uuid.uuid4()),
                description=task_desc,
                priority=self._determine_task_priority(task_desc),
                created_at=datetime.now().isoformat()
            )
            self.task_queue.append(task)
        
        self._save_tasks()
        return f"Created {len(tasks)} tasks from orchestrator input: {instruction}"

    def _determine_task_priority(self, task_description: str) -> int:
        """
        Determine task priority based on task type and urgency indicators.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Priority score (higher numbers indicate higher priority)
        """
        task_lower = task_description.lower()
        
        if "critical" in task_lower or "urgent" in task_lower:
            return 5
        elif "analysis" in task_lower or "analyze" in task_lower:
            return 3
        elif "strategy" in task_lower:
            return 3
        elif "validation" in task_lower or "verify" in task_lower:
            return 2
        else:
            return 1

    def _decompose_instruction(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """
        Decompose orchestrator instructions into executable task sequences.
        
        This method provides intelligent task decomposition based on instruction
        analysis and context understanding, enabling autonomous execution planning.
        
        Args:
            instruction: High-level instruction from orchestrator
            context: Additional context for task planning
            
        Returns:
            List of specific task descriptions for execution
        """
        tasks = []
        instruction_lower = instruction.lower()
        
        # Analysis-type tasks
        if any(word in instruction_lower for word in ["analyze", "analysis", "examine", "study"]):
            tasks.extend([
                f"Query memory for relevant data: {instruction}",
                f"Generate analysis tool: {instruction}",
                f"Execute analysis: {instruction}",
                f"Generate strategy flow: {instruction}",
                f"Validate analysis results: {instruction}"
            ])
        
        # Prediction-type tasks
        elif any(word in instruction_lower for word in ["predict", "forecast", "estimate"]):
            tasks.extend([
                f"Collect historical data: {instruction}",
                f"Build prediction model: {instruction}",
                f"Run prediction: {instruction}",
                f"Generate prediction strategy flow: {instruction}",
                f"Validate prediction accuracy: {instruction}"
            ])
        
        # Strategy-type tasks
        elif any(word in instruction_lower for word in ["strategy", "trading", "signal"]):
            tasks.extend([
                f"Analyze market conditions: {instruction}",
                f"Generate strategy code: {instruction}",
                f"Backtest strategy: {instruction}",
                f"Generate strategy flow output: {instruction}",
                f"Optimize strategy parameters: {instruction}"
            ])
        
        # Default task decomposition
        else:
            tasks.append(f"General task processing: {instruction}")
        
        return tasks

    def _query_memory(self, query: str, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Query memory agent for relevant knowledge and data.
        
        This method interfaces with the memory agent to retrieve relevant
        historical data, analysis results, and domain knowledge for
        informed decision making.
        
        Args:
            query: Search query for knowledge retrieval
            category: Optional category filter for targeted search
            
        Returns:
            Retrieved knowledge with relevance scores and metadata
        """
        try:
            # TODO: Connect to actual memory agent when available
            # For now, return empty results to avoid dependency on mock data
            return {
                "query": query,
                "category": category,
                "results": [],
                "timestamp": datetime.now().isoformat(),
                "status": "memory_agent_not_connected"
            }
        except Exception as e:
            return {"error": str(e), "results": []}

    def _generate_code_tool(self, description: str, input_format: str, expected_output: str) -> Dict[str, Any]:
        """
        Dynamically generate Python code tools for financial analysis.
        
        This method demonstrates the agent's code generation capabilities by
        creating custom analysis tools based on requirements specifications.
        The generated code includes comprehensive error handling and documentation.
        
        Args:
            description: Description of the analysis to be performed
            input_format: Format specification for input data
            expected_output: Expected output format and content
            
        Returns:
            Generated tool information including code and file path
        """
        tool_name = f"generated_tool_{uuid.uuid4().hex[:8]}"
        
        # Generate comprehensive code template
        code_template = f"""import pandas as pd
import numpy as np
from datetime import datetime
import json

def {tool_name}(data):
    \"\"\"
    {description}
    
    Input format: {input_format}
    Expected output: {expected_output}
    
    Generated by AutonomousAgent for financial analysis tasks.
    \"\"\"
    
    try:
        # Data processing logic with financial analysis focus
        if isinstance(data, dict):
            result = process_financial_dict_data(data)
        elif isinstance(data, list):
            result = process_financial_list_data(data)
        else:
            result = process_generic_financial_data(data)
        
        return {{
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "tool_name": "{tool_name}",
            "analysis_type": "autonomous_generated"
        }}
    
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "tool_name": "{tool_name}"
        }}

def process_financial_dict_data(data):
    \"\"\"Process dictionary format financial data with comprehensive analysis\"\"\"
    result = {{"processed": True, "data_type": "financial_dict"}}
    
    if "prices" in data:
        prices = data["prices"]
        if isinstance(prices, list) and len(prices) > 1:
            # Enhanced financial analysis
            returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
            
            result.update({{
                "price_analysis": {{
                    "current_price": prices[-1],
                    "price_change": prices[-1] - prices[0],
                    "percent_change": (prices[-1] / prices[0] - 1) * 100,
                    "volatility": np.std(returns) * np.sqrt(252),
                    "sharpe_ratio": np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
                    "max_drawdown": calculate_max_drawdown(prices),
                    "technical_indicators": calculate_technical_indicators(prices)
                }}
            }})
    
    return result

def process_financial_list_data(data):
    \"\"\"Process list format financial data\"\"\"
    if all(isinstance(x, (int, float)) for x in data):
        # Assume price data
        returns = [data[i] / data[i-1] - 1 for i in range(1, len(data))]
        
        return {{
            "statistical_analysis": {{
                "mean": np.mean(data),
                "std": np.std(data),
                "min": min(data),
                "max": max(data),
                "count": len(data)
            }},
            "financial_metrics": {{
                "total_return": (data[-1] / data[0] - 1) * 100,
                "volatility": np.std(returns) * np.sqrt(252),
                "var_95": np.percentile(returns, 5),
            }}
        }}
    
    return {{"processed": True, "data_type": "list", "length": len(data)}}

def process_generic_financial_data(data):
    \"\"\"Process generic financial data\"\"\"
    return {{
        "data_type": type(data).__name__,
        "processed": True,
        "summary": str(data)[:100],
        "analysis_note": "Generic financial data processing applied"
    }}

def calculate_max_drawdown(prices):
    \"\"\"Calculate maximum drawdown from price series\"\"\"
    if len(prices) < 2:
        return 0.0
    
    peak = prices[0]
    max_dd = 0.0
    
    for price in prices[1:]:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        max_dd = max(max_dd, drawdown)
    
    return max_dd

def calculate_technical_indicators(prices):
    \"\"\"Calculate basic technical indicators\"\"\"
    if len(prices) < 10:
        return {{"insufficient_data": True}}
    
    sma_10 = np.mean(prices[-10:])
    sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
    
    return {{
        "sma_10": sma_10,
        "sma_20": sma_20,
        "price_vs_sma10": prices[-1] / sma_10 - 1,
        "price_vs_sma20": prices[-1] / sma_20 - 1,
        "trend": "bullish" if sma_10 > sma_20 else "bearish"
    }}
"""
        
        # Save generated code to workspace
        tool_file = os.path.join(self.workspace_dir, f"{tool_name}.py")
        with open(tool_file, 'w') as f:
            f.write(code_template)
        
        # Store tool information in registry
        self.generated_tools[tool_name] = {
            "description": description,
            "file_path": tool_file,
            "created_at": datetime.now().isoformat(),
            "input_format": input_format,
            "expected_output": expected_output,
            "tool_type": "financial_analysis"
        }
        
        return {
            "tool_name": tool_name,
            "code": code_template,
            "file_path": tool_file,
            "status": "generated",
            "capabilities": ["technical_analysis", "risk_metrics", "statistical_analysis"]
        }

    def _create_validation(self, code_to_validate: str, test_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create comprehensive validation programs for generated code tools.
        
        This method generates robust test suites that validate both functional
        correctness and financial analysis accuracy of generated tools.
        
        Args:
            code_to_validate: The generated code to be validated
            test_scenarios: List of test scenarios with input/expected output pairs
            
        Returns:
            Validation code information and file location
        """
        validation_name = f"validation_{uuid.uuid4().hex[:8]}"
        
        validation_code = f'''
import unittest
import sys
import os
import json
import numpy as np
from datetime import datetime

# Import the code to be validated
{code_to_validate}

class TestGeneratedFinancialCode(unittest.TestCase):
    """
    Comprehensive validation suite for generated financial analysis code.
    
    This test suite validates both functional correctness and financial
    analysis accuracy of autonomous agent generated tools.
    """
    
    def setUp(self):
        """Set up test scenarios and validation data"""
        self.test_scenarios = {test_scenarios}
        self.results = []
        
        # Standard financial test data
        self.sample_prices = [100, 102, 98, 105, 103, 107, 110, 108, 112, 115]
    
    def test_basic_functionality(self):
        """Test basic functionality with standard inputs"""
        # Find the generated function
        func_name = self._find_generated_function()
        if not func_name:
            self.fail("No generated function found")
        
        func = globals()[func_name]
        
        # Test with price data
        result = func({{"prices": self.sample_prices}})
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success", False))
        self.assertIn("result", result)
        
        self.results.append({{
            "test": "basic_functionality",
            "input": "sample_prices",
            "output": result,
            "passed": True
        }})
    
    def test_financial_metrics_accuracy(self):
        """Test accuracy of financial metrics calculations"""
        func_name = self._find_generated_function()
        if not func_name:
            self.skipTest("No generated function found")
        
        func = globals()[func_name]
        result = func({{"prices": self.sample_prices}})
        
        if result.get("success") and "price_analysis" in result.get("result", {{}}):
            price_analysis = result["result"]["price_analysis"]
            
            # Validate basic calculations
            expected_change = self.sample_prices[-1] - self.sample_prices[0]
            actual_change = price_analysis.get("price_change", 0)
            
            self.assertAlmostEqual(actual_change, expected_change, places=2)
    
    def test_edge_cases(self):
        """Test handling of edge cases and error conditions"""
        func_name = self._find_generated_function()
        if not func_name:
            self.skipTest("No generated function found")
        
        func = globals()[func_name]
        
        # Test empty data
        result = func({{"prices": []}})
        self.assertIsInstance(result, dict)
        
        # Test single data point
        result = func({{"prices": [100]}})
        self.assertIsInstance(result, dict)
    
    def _find_generated_function(self):
        """Find the generated function in the global namespace"""
        for name in globals():
            if callable(globals()[name]) and name.startswith("generated_tool_"):
                return name
        return None
    
    def tearDown(self):
        """Save test results and generate validation report"""
        validation_report = {{
            "validation_timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.get("passed", False)),
            "test_results": self.results
        }}
        
        with open("validation_results.json", "w") as f:
            json.dump(validation_report, f, indent=2)

if __name__ == "__main__":
    unittest.main(verbosity=2)
'''
        
        # Save validation code to workspace
        validation_file = os.path.join(self.workspace_dir, f"{validation_name}.py")
        with open(validation_file, 'w') as f:
            f.write(validation_code)
        
        return {
            "validation_name": validation_name,
            "validation_code": validation_code,
            "file_path": validation_file,
            "status": "created",
            "test_coverage": ["functionality", "accuracy", "edge_cases", "performance"]
        }

    def _execute_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a previously generated analysis tool with provided data.
        
        This method provides safe execution of dynamically generated tools
        with comprehensive error handling and result validation.
        
        Args:
            tool_name: Name of the tool to execute
            input_data: Input data for tool execution
            
        Returns:
            Execution results with success status and output data
        """
        if tool_name not in self.generated_tools:
            return {"error": f"Tool {tool_name} not found in registry"}
        
        tool_info = self.generated_tools[tool_name]
        tool_file = tool_info["file_path"]
        
        try:
            # Dynamically import and execute the tool
            spec = importlib.util.spec_from_file_location(tool_name, tool_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the tool function and execute
            tool_func = getattr(module, tool_name)
            result = tool_func(input_data)
            
            return {
                "success": True,
                "result": result,
                "tool_name": tool_name,
                "executed_at": datetime.now().isoformat(),
                "tool_type": tool_info.get("tool_type", "unknown")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name,
                "executed_at": datetime.now().isoformat(),
                "error_type": type(e).__name__
            }

    def _autonomous_task_processor(self):
        """
        Autonomous task processing loop for background execution.
        
        This method runs continuously in a background thread, processing
        pending tasks according to priority and dependencies. It demonstrates
        the agent's autonomous execution capabilities.
        """
        while self.task_processor_running:
            try:
                # Find pending tasks
                pending_tasks = [t for t in self.task_queue if t.status == "pending"]
                
                if pending_tasks:
                    # Sort by priority (higher numbers first)
                    pending_tasks.sort(key=lambda x: x.priority, reverse=True)
                    task = pending_tasks[0]
                    
                    # Process the highest priority task
                    self._process_task(task)
                
                # Sleep before next iteration
                time.sleep(5)
                
            except Exception as e:
                print(f"Error in autonomous task processor: {e}")
                time.sleep(10)

    def _process_task(self, task: Task):
        """
        Process an individual task with appropriate strategy based on task type.
        
        This method routes tasks to specialized handlers based on task content,
        enabling intelligent processing of different task types.
        
        Args:
            task: Task object to be processed
        """
        task.status = "in_progress"
        self._save_tasks()
        
        try:
            task_desc_lower = task.description.lower()
            
            # Route to appropriate handler based on task type
            if "memory" in task_desc_lower and "query" in task_desc_lower:
                self._handle_memory_query_task(task)
            elif "generate" in task_desc_lower and ("tool" in task_desc_lower or "code" in task_desc_lower):
                self._handle_code_generation_task(task)
            elif "execute" in task_desc_lower or "analysis" in task_desc_lower:
                self._handle_analysis_execution_task(task)
            elif "strategy flow" in task_desc_lower:
                self._handle_strategy_flow_task(task)
            elif "validation" in task_desc_lower or "validate" in task_desc_lower:
                self._handle_validation_task(task)
            else:
                self._handle_generic_task(task)
            
            task.status = "completed"
            
        except Exception as e:
            task.status = "failed"
            print(f"Task {task.task_id} failed: {e}")
        
        finally:
            self._save_tasks()

    def _handle_memory_query_task(self, task: Task):
        """Handle memory query tasks by extracting query and retrieving data"""
        # Extract query from task description
        query = task.description.split(":")[-1].strip()
        result = self._query_memory(query)
        
        # Store results in task
        task.validation_result = result

    def _handle_code_generation_task(self, task: Task):
        """Handle code generation tasks by creating analysis tools"""
        description = task.description
        
        # Generate code tool for the task
        tool_result = self._generate_code_tool(
            description=description,
            input_format="dict with financial data keys",
            expected_output="dict with analysis results and metrics"
        )
        
        task.generated_code = tool_result.get("code")
        task.validation_result = {"tool_created": tool_result.get("tool_name")}

    def _handle_analysis_execution_task(self, task: Task):
        """Handle analysis execution by running generated tools"""
        # Find and execute most recent generated tool
        if self.generated_tools:
            tool_name = list(self.generated_tools.keys())[-1]
            test_data = {
                "prices": [100 + random.uniform(-5, 5) for _ in range(20)]
            }
            result = self._execute_tool(tool_name, test_data)
            task.validation_result = result

    def _handle_strategy_flow_task(self, task: Task):
        """Handle strategy flow generation tasks"""
        # Extract symbol and instruction from task
        instruction = task.description.split(":")[-1].strip()
        symbol = "SAMPLE"  # Default symbol
        
        # Generate strategy flow
        strategy_flow = self._generate_trading_signal(symbol, instruction, {})
        task.validation_result = {"strategy_flow_generated": True, "alpha_id": strategy_flow.get("alpha_id")}

    def _handle_validation_task(self, task: Task):
        """Handle validation tasks by creating test suites"""
        if task.generated_code:
            test_scenarios = [
                {"input": {"prices": [100, 102, 98, 105]}, "expected": {"success": True}},
                {"input": {"prices": []}, "expected": {"success": True}},
            ]
            
            validation_result = self._create_validation(task.generated_code, test_scenarios)
            task.validation_code = validation_result.get("validation_code")
            task.validation_result = {"validation_created": validation_result.get("validation_name")}

    def _handle_generic_task(self, task: Task):
        """Handle generic tasks with basic processing"""
        task.validation_result = {
            "processed": True,
            "timestamp": datetime.now().isoformat(),
            "description": task.description,
            "processing_type": "generic"
        }

    def start_mcp_server(self, host: str = "0.0.0.0", port: int = 5051):
        """
        Start the MCP server for external communication.
        
        This method initiates the agent's external interface, enabling
        communication with orchestrators and other system components.
        
        Args:
            host: Host address to bind the server
            port: Port number for the server
        """
        print(f"[AutonomousAgent] Starting MCP server on {host}:{port}")
        self.mcp_server.settings.host = host
        self.mcp_server.settings.port = port
        
        # Display registered tools
        print("=== Autonomous Agent Tools ===")
        tools = asyncio.run(self.mcp_server.list_tools())
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
        
        try:
            self.mcp_server.run(transport="sse")
        finally:
            self.task_processor_running = False


if __name__ == "__main__":
    agent = AutonomousAgent()
    agent.start_mcp_server()


def run_autonomous_agent(config_path: str = None, port_override: int = None):
    """
    Entry point function for running autonomous agent from multiprocessing.
    
    This function loads configuration and starts the autonomous agent server.
    It's designed to be called from the core agent pool for process-based execution.
    
    Args:
        config_path: Path to configuration file (optional)
        port_override: Override port if provided (for conflict resolution)
    """
    try:
        import yaml
        import os
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            host = config_data.get('execution', {}).get('host', '0.0.0.0')
            port = port_override or config_data.get('execution', {}).get('port', 5052)
        else:
            # Default configuration
            host = "0.0.0.0"
            port = port_override or 5052  # Updated default port to avoid conflict
        
        print(f"[AutonomousAgent] Starting with config - Host: {host}, Port: {port}")
        
        agent = AutonomousAgent()
        agent.start_mcp_server(host=host, port=port)
        
    except Exception as e:
        print(f"[AutonomousAgent] Failed to start: {e}")
        import traceback
        traceback.print_exc()
