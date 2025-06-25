# Autonomous Alpha Agent System

## Overview

This system implements an intelligent agent capable of autonomous task management, dynamic code tool generation, and comprehensive validation. The agent can:

1. **Receive external orchestrator inputs** and autonomously decompose them into specific tasks
2. **Retrieve knowledge from memory agents** for informed analysis and decision-making
3. **Dynamically generate code tools** to meet analysis requirements
4. **Create validation code** to ensure the correctness of generated tools
5. **Autonomous task scheduling** and execution management
6. **Generate stable strategy flows** compatible with the alpha agent ecosystem

## System Architecture

```
Orchestrator
     ↓ (instruction input)
AlphaAgentPool (core.py)
     ↓ (startup management)
AutonomousAgent
     ├── Task Decomposition Module
     ├── Memory Query Module  
     ├── Code Generation Module
     ├── Validation Module
     ├── Strategy Flow Generation Module
     └── Task Execution Engine
```

## Core Components

### 1. AutonomousAgent (`autonomous_agent.py`)

**Main Features:**
- Receive orchestrator instructions and autonomously decompose tasks
- Manage task queues and execution status
- Dynamically generate analysis tool code
- Create validation programs to ensure code correctness
- Interact with memory agents to obtain knowledge
- Generate AlphaStrategyFlow objects for ecosystem compatibility

**Core Methods:**
- `_process_orchestrator_input()`: Process external input
- `_decompose_instruction()`: Intelligent task decomposition
- `_generate_code_tool()`: Dynamic code generation
- `_create_validation()`: Validation code creation
- `_generate_trading_signal()`: Strategy flow generation
- `_autonomous_task_processor()`: Autonomous task processing loop

### 2. Task Model

```python
class Task(BaseModel):
    task_id: str              # Unique task identifier
    description: str          # Task description
    priority: int             # Priority level
    status: str               # Status: pending/in_progress/completed/failed
    created_at: str           # Creation timestamp
    dependencies: List[str]   # Task dependencies
    generated_code: Optional[str]     # Generated code
    validation_code: Optional[str]    # Validation code
    validation_result: Optional[Dict] # Validation results
```

### 3. Integration with AlphaAgentPool

Added to `core.py`:
- `start_agent("autonomous_agent")`: Start autonomous agent
- `send_orchestrator_input()`: Send orchestrator instructions

## Usage Examples

### 1. Basic Usage

```python
from FinAgents.agent_pools.alpha_agent_pool.agents.autonomous.autonomous_agent import AutonomousAgent

# Create autonomous agent
agent = AutonomousAgent(agent_id="my_autonomous_agent")

# Send orchestrator instruction
instruction = "Analyze AAPL stock trends and generate trading recommendations"
context = {"symbol": "AAPL", "timeframe": "1d"}
result = agent._process_orchestrator_input(instruction, context)

# View generated tasks
for task in agent.task_queue:
    print(f"Task: {task.description}")
    print(f"Status: {task.status}")
```

### 2. Strategy Flow Generation

```python
# Generate trading signal with strategy flow
strategy_flow = agent._generate_trading_signal(
    symbol="AAPL",
    instruction="Momentum-based trading analysis",
    market_data={"prices": [150, 152, 148, 155, 157, 160]}
)

print(f"Signal: {strategy_flow['decision']['signal']}")
print(f"Confidence: {strategy_flow['decision']['confidence']}")
print(f"Alpha ID: {strategy_flow['alpha_id']}")
```

### 3. Through AlphaAgentPool

```python
# Start AlphaAgentPool
pool = AlphaAgentPoolMCPServer()

# Start autonomous agent through MCP tools
pool_server.call_tool("start_agent", agent_id="autonomous_agent")

# Send orchestrator input
pool_server.call_tool("send_orchestrator_input", 
                     instruction="Create quantitative trading strategy",
                     context={"type": "momentum"})
```

### 4. MCP Server Mode

```python
# Run directly as MCP server
agent = AutonomousAgent()
agent.start_mcp_server(host="0.0.0.0", port=5051)
```

## MCP Tool Interface

The autonomous agent provides the following MCP tools:

| Tool Name | Description | Parameters |
|-----------|-------------|------------|
| `receive_orchestrator_input` | Receive orchestrator instructions | instruction, context |
| `query_memory_agent` | Query memory agent | query, category |
| `generate_analysis_tool` | Generate analysis tool code | description, input_format, expected_output |
| `create_validation_code` | Create validation code | code_to_validate, test_scenarios |
| `get_task_status` | Get task status | - |
| `execute_generated_tool` | Execute generated tools | tool_name, input_data |
| `generate_strategy_signal` | Generate trading signals | symbol, instruction, market_data |

## Task Types and Processing Strategies

### Analysis Tasks
```
Input: "Analyze AAPL stock momentum indicators"
Decomposed into:
1. Query memory for relevant data
2. Generate analysis tool  
3. Execute analysis
4. Generate strategy flow
5. Validate analysis results
```

### Prediction Tasks
```
Input: "Predict future stock price trends"
Decomposed into:
1. Collect historical data
2. Build prediction model
3. Run prediction
4. Generate prediction strategy flow
5. Validate prediction accuracy
```

### Strategy Tasks
```
Input: "Create trading strategy"
Decomposed into:
1. Analyze market conditions
2. Generate strategy code
3. Backtest strategy
4. Generate strategy flow output
5. Optimize strategy parameters
```

## Code Generation Capabilities

The autonomous agent can dynamically generate Python code tools based on requirement descriptions:

### Generated Code Features
- Complete error handling
- Support for multiple data format inputs
- Return structured results
- Comprehensive documentation
- Financial analysis focus

### Example Generated Tool
```python
def generated_tool_abc123(data):
    """
    Calculate stock technical indicators
    
    Input format: dict with prices array
    Expected output: dict with technical indicators
    """
    import pandas as pd
    import numpy as np
    
    # Data processing and analysis logic
    # Enhanced with financial metrics
    # ...
    
    return {
        "success": True,
        "result": analysis_result,
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "autonomous_generated"
    }
```

## Strategy Flow Output

The agent generates `AlphaStrategyFlow` objects compatible with the ecosystem:

```json
{
  "alpha_id": "autonomous_xxxxxxxx",
  "version": "1.0",
  "timestamp": "2025-06-24T20:51:11.059331",
  "market_context": {
    "symbol": "AAPL",
    "regime_tag": "bullish_trend",
    "input_features": {
      "current_price": 162.0,
      "sma_5": 158.0,
      "momentum": 0.038,
      "volatility": 0.027
    }
  },
  "decision": {
    "signal": "BUY",
    "confidence": 0.85,
    "reasoning": "Strong upward trend with positive momentum",
    "predicted_return": 0.076,
    "risk_estimate": 0.027
  },
  "action": {
    "execution_weight": 0.51,
    "order_type": "market",
    "order_price": 162.0
  }
}
```

## Validation Mechanism

Automatically creates validation code for each generated tool:

```python
class TestGeneratedCode(unittest.TestCase):
    def test_scenarios(self):
        # Test various input scenarios
        # Validate output format and correctness
        # Record test results
        
    def test_financial_metrics(self):
        # Test financial calculation accuracy
        # Validate signal generation logic
        # Check error handling
```

## Configuration

`config/autonomous.yaml`:
```yaml
agent_id: "autonomous_alpha_agent"
execution:
  port: 5051
  host: "0.0.0.0"
  
autonomous_config:
  task_processing_interval: 5
  max_concurrent_tasks: 3
  code_generation:
    workspace_dir: "./workspace"
    max_tool_cache: 50
    enable_validation: true
  strategy_flow:
    output_format: "AlphaStrategyFlow"
    persistence_enabled: true
```

## Testing and Usage

### Run Tests
```bash
# Comprehensive functionality test
python test_autonomous_agent.py

# Enhanced example test
python examples/autonomous_agent_example.py
```

### Start Complete System
```bash
# Start AlphaAgentPool (includes autonomous agent)
python FinAgents/agent_pools/alpha_agent_pool/core.py

# Test in another terminal
python examples/autonomous_agent_example.py
```

## Extension Capabilities

The system is designed to be extensible:

1. **Task Decomposition Strategies** can integrate LLMs for more intelligent decomposition
2. **Code Generation** can connect to code generation models
3. **Validation Mechanism** can add more complex test case generation
4. **Memory Connection** can connect to real knowledge base systems
5. **Execution Engine** can support distributed task execution
6. **Strategy Flow Generation** can integrate with portfolio optimization

## Workflow Example

```
1. Orchestrator → "Analyze AAPL stock momentum and generate trading signal"
2. AutonomousAgent → Decompose into 5 subtasks
3. Task1: Query Memory for AAPL historical data
4. Task2: Generate momentum analysis tool code
5. Task3: Execute analysis tool to get results
6. Task4: Generate AlphaStrategyFlow with trading signal
7. Task5: Create validation code to ensure correctness
8. All tasks complete, return comprehensive strategy flow
```

## Integration with Alpha Agent Ecosystem

The autonomous agent seamlessly integrates with:

1. **Momentum Agent**: Identical strategy flow output format
2. **Memory Agent**: Knowledge retrieval for informed decisions
3. **Orchestrator**: High-level instruction processing
4. **Alpha Agent Pool**: Compatible agent registration and communication
5. **Strategy Flow Schema**: Full AlphaStrategyFlow compatibility

This autonomous agent system provides a powerful framework that can autonomously complete complex analysis tasks based on external instructions while ensuring the quality and correctness of generated code and maintaining full compatibility with the alpha agent ecosystem.
