# FinAgent Orchestration System - User Guide

## Overview

The FinAgent Orchestration System is a comprehensive, protocol-driven multi-agent framework designed for algorithmic trading and financial analysis. It provides a unified platform for coordinating data ingestion, signal generation, risk management, and execution across multiple specialized agent pools.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Features](#advanced-features)

## Quick Start

### Prerequisites

- Python 3.8+
- Required dependencies (see `requirements.txt`)
- Access to market data APIs (optional for testing)

### Basic Setup

1. **Clone and Install**
   ```bash
   git clone <repository-url>
   cd FinAgent-Orchestration
   pip install -r requirements.txt
   ```

2. **Start Agent Pools** (in separate terminals)
   ```bash
   # Terminal 1 - Data Agent Pool
   cd FinAgents/agent_pools/data_agent_pool
   python core.py
   
   # Terminal 2 - Alpha Agent Pool  
   cd FinAgents/agent_pools/alpha_agent_pool
   python core.py
   
   # Terminal 3 - Risk Agent Pool
   cd FinAgents/agent_pools/risk_agent_pool
   python core.py
   
   # Terminal 4 - Transaction Cost Agent Pool
   cd FinAgents/agent_pools/transaction_cost_agent_pool
   python core.py
   ```

3. **Start Memory Agent**
   ```bash
   cd FinAgents/memory
   python external_memory_agent.py
   ```

4. **Start Orchestrator**
   ```bash
   cd FinAgents/orchestrator
   python main_orchestrator.py --mode development
   ```

### Your First Strategy

```python
from core.finagent_orchestrator import FinAgentOrchestrator
from core.dag_planner import TradingStrategy

# Initialize orchestrator
orchestrator = FinAgentOrchestrator()
await orchestrator.initialize()

# Define a simple momentum strategy
strategy = TradingStrategy(
    strategy_id="my_first_strategy",
    name="Simple Momentum Strategy",
    description="Buy high momentum stocks",
    symbols=["AAPL", "GOOGL", "MSFT"],
    lookback_period=20,
    rebalance_frequency="daily",
    parameters={
        "momentum_threshold": 0.02,
        "position_size": 0.33
    }
)

# Execute the strategy
result = await orchestrator.execute_strategy(strategy)
print(f"Strategy execution result: {result}")
```

## Architecture Overview

The FinAgent system follows a modular, protocol-driven architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                         │
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │   DAG Planner   │────│      Execution Engine      │ │
│  │                 │    │                             │ │
│  └─────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                  Agent Pool Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Data Agent  │  │ Alpha Agent │  │ Execution Agent │ │
│  │ Pool        │  │ Pool        │  │ Pool            │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│               Memory & RL Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Memory      │  │ RL Policy   │  │ Backtesting     │ │
│  │ Agent       │  │ Engine      │  │ Engine          │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Key Components

- **Orchestrator**: Central coordination engine
- **DAG Planner**: Task decomposition and workflow management
- **Agent Pools**: Specialized agents for different functions
- **Memory Agent**: Unified logging and data persistence
- **RL Engine**: Reinforcement learning for strategy optimization
- **Sandbox Environment**: Testing and backtesting framework

## Installation

### Standard Installation

```bash
# Clone repository
git clone <repository-url>
cd FinAgent-Orchestration

# Create virtual environment
python -m venv finagent_env
source finagent_env/bin/activate  # On Windows: finagent_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Docker Installation

```bash
# Build Docker image
docker build -t finagent-orchestrator .

# Run with Docker Compose
docker-compose up -d
```

### Dependencies

Core dependencies include:
- `asyncio` - Asynchronous programming
- `fastapi` - API framework
- `mcp` - Multi-agent Control Protocol
- `torch` - Deep learning for RL
- `numpy`, `pandas` - Data processing
- `networkx` - Graph algorithms for DAG
- `pydantic` - Data validation

## Configuration

### Main Configuration File

The system uses YAML configuration files. See `config/orchestrator_config.yaml` for the complete configuration template.

Key configuration sections:

```yaml
# Orchestrator settings
orchestrator:
  host: "0.0.0.0"
  port: 9000
  max_concurrent_tasks: 100
  enable_rl: true
  enable_sandbox: true

# Agent pool endpoints
agent_pools:
  data_agent_pool:
    url: "http://localhost:8001"
    enabled: true
  alpha_agent_pool:
    url: "http://localhost:5050"
    enabled: true

# RL engine configuration
rl_engine:
  algorithm: "TD3"
  learning_rate: 0.0003
  buffer_size: 1000000

# Sandbox settings
sandbox:
  initial_capital: 1000000
  commission_rate: 0.001
  data_start_date: "2020-01-01"
```

### Environment Variables

```bash
# API Keys
export POLYGON_API_KEY="your_polygon_key"
export ALPHA_VANTAGE_API_KEY="your_av_key"
export OPENAI_API_KEY="your_openai_key"

# System settings
export FINAGENT_ENV="production"  # or "development"
export FINAGENT_LOG_LEVEL="INFO"
```

## Usage Examples

### 1. Basic Strategy Execution

```python
import asyncio
from core.finagent_orchestrator import FinAgentOrchestrator
from core.dag_planner import TradingStrategy

async def basic_strategy_example():
    # Initialize orchestrator
    orchestrator = FinAgentOrchestrator()
    await orchestrator.initialize()
    
    # Define strategy
    strategy = TradingStrategy(
        strategy_id="momentum_001",
        name="Momentum Strategy",
        symbols=["AAPL", "GOOGL"],
        lookback_period=20,
        parameters={"threshold": 0.02}
    )
    
    # Execute strategy
    result = await orchestrator.execute_strategy(strategy)
    print(f"Generated {len(result['signals'])} signals")

# Run the example
asyncio.run(basic_strategy_example())
```

### 2. Multi-Agent Workflow

```python
async def multi_agent_workflow():
    orchestrator = FinAgentOrchestrator()
    await orchestrator.initialize()
    
    # Define complex workflow
    workflow = {
        "workflow_id": "multi_agent_example",
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
                "action": "assess_risk",
                "parameters": {"model": "var"}
            }
        ]
    }
    
    # Execute workflow
    result = await orchestrator.execute_workflow(workflow)
    return result
```

### 3. Backtesting with Sandbox

```python
from core.sandbox_environment import SandboxEnvironment, TestScenario, SandboxMode

async def backtesting_example():
    orchestrator = FinAgentOrchestrator()
    await orchestrator.initialize()
    
    sandbox = SandboxEnvironment(orchestrator)
    
    # Define backtest scenario
    scenario = TestScenario(
        scenario_id="momentum_backtest",
        name="Momentum Strategy Backtest",
        mode=SandboxMode.HISTORICAL_BACKTEST,
        parameters={
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "initial_capital": 100000,
            "strategy_type": "momentum"
        }
    )
    
    # Run backtest
    result = await sandbox.run_test_scenario(scenario)
    
    # Access performance metrics
    performance = result["performance_metrics"]
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
```

### 4. Reinforcement Learning Training

```python
from core.rl_policy_engine import RLConfiguration, RLAlgorithm

async def rl_training_example():
    orchestrator = FinAgentOrchestrator()
    await orchestrator.initialize()
    
    # Configure RL
    rl_config = RLConfiguration(
        algorithm=RLAlgorithm.TD3,
        learning_rate=0.0003,
        buffer_size=1000000,
        batch_size=256
    )
    
    await orchestrator.initialize_rl_engine(rl_config)
    
    # Start training
    training_config = {
        "training_episodes": 1000,
        "symbols": ["AAPL", "GOOGL"],
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    }
    
    result = await orchestrator.train_rl_policy(training_config)
    print(f"Training completed: {result['final_performance']}")
```

### 5. Real-time Strategy Monitoring

```python
async def monitoring_example():
    orchestrator = FinAgentOrchestrator()
    await orchestrator.initialize()
    
    # Start monitoring loop
    while True:
        # Check system health
        health = await orchestrator.get_health_status()
        print(f"System status: {health['status']}")
        
        # Check active strategies
        active_strategies = await orchestrator.get_active_strategies()
        print(f"Active strategies: {len(active_strategies)}")
        
        # Check performance metrics
        metrics = await orchestrator.get_performance_metrics()
        print(f"Current metrics: {metrics}")
        
        await asyncio.sleep(30)  # Check every 30 seconds
```

## API Reference

### Core Classes

#### FinAgentOrchestrator

Main orchestration class for coordinating all system components.

```python
class FinAgentOrchestrator:
    async def initialize(self) -> None
    async def execute_strategy(self, strategy: TradingStrategy) -> Dict[str, Any]
    async def execute_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]
    async def train_rl_policy(self, config: Dict[str, Any]) -> Dict[str, Any]
    async def get_health_status(self) -> Dict[str, Any]
    async def shutdown(self) -> None
```

#### TradingStrategy

Strategy configuration class.

```python
@dataclass
class TradingStrategy:
    strategy_id: str
    name: str
    description: str
    symbols: List[str]
    lookback_period: int
    rebalance_frequency: str
    parameters: Dict[str, Any] = field(default_factory=dict)
```

#### SandboxEnvironment

Testing and backtesting environment.

```python
class SandboxEnvironment:
    async def run_test_scenario(self, scenario: TestScenario) -> Dict[str, Any]
    async def run_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]
    async def run_stress_test(self, scenario: str) -> Dict[str, Any]
```

### REST API Endpoints

When running in server mode, the orchestrator exposes a REST API:

#### Strategy Execution
- `POST /strategies` - Execute a trading strategy
- `GET /strategies/{strategy_id}` - Get strategy status
- `DELETE /strategies/{strategy_id}` - Stop strategy execution

#### System Management
- `GET /health` - System health check
- `GET /status` - Detailed system status
- `POST /shutdown` - Graceful shutdown

#### Backtesting
- `POST /backtest` - Run backtest scenario
- `GET /backtest/{test_id}` - Get backtest results

## Testing

### Running Tests

```bash
# Run all tests
python test_orchestrator_comprehensive.py --test-type all

# Run specific test types
python test_orchestrator_comprehensive.py --test-type unit
python test_orchestrator_comprehensive.py --test-type integration
python test_orchestrator_comprehensive.py --test-type system
python test_orchestrator_comprehensive.py --test-type performance
python test_orchestrator_comprehensive.py --test-type demo
```

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Cross-component interaction testing
3. **System Tests**: End-to-end workflow testing
4. **Performance Tests**: Load and stress testing
5. **Demo Scenarios**: Real-world use case demonstrations

### Example Test Output

```
[2024-01-15 10:30:00] INFO - OrchestratorTests: Starting FinAgent Orchestrator Tests
[2024-01-15 10:30:01] INFO - OrchestratorTests: ✅ test_dag_planner_initialization: PASSED
[2024-01-15 10:30:02] INFO - OrchestratorTests: ✅ test_trading_strategy_creation: PASSED
[2024-01-15 10:30:03] INFO - OrchestratorTests: ✅ test_orchestrator_initialization: PASSED
```

## Troubleshooting

### Common Issues

#### 1. Agent Pool Connection Failures

**Problem**: Orchestrator cannot connect to agent pools
```
ERROR - FinAgentOrchestrator: Failed to connect to data_agent_pool at http://localhost:8001
```

**Solution**:
- Ensure all agent pools are running
- Check port availability
- Verify configuration URLs
- Check firewall settings

#### 2. Memory Agent Unavailable

**Problem**: Memory agent connection timeout
```
WARNING - FinAgentOrchestrator: Memory agent unavailable, running without persistence
```

**Solution**:
- Start memory agent: `python FinAgents/memory/external_memory_agent.py`
- Check memory agent configuration
- Verify storage directory permissions

#### 3. RL Training Convergence Issues

**Problem**: RL agent not converging during training
```
WARNING - RLPolicyEngine: Training not converging after 1000 episodes
```

**Solution**:
- Adjust learning rate
- Increase buffer size
- Check reward function design
- Verify environment state representation

#### 4. High Memory Usage

**Problem**: System consuming excessive memory
```
WARNING - FinAgentOrchestrator: High memory usage detected: 8.5GB
```

**Solution**:
- Reduce batch sizes
- Implement data streaming
- Clear unused caches
- Optimize buffer sizes

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Or via configuration
orchestrator_config["log_level"] = "DEBUG"
```

### Performance Monitoring

Monitor system performance:

```python
# Get system metrics
metrics = await orchestrator.get_performance_metrics()
print(f"CPU Usage: {metrics['cpu_usage']:.1%}")
print(f"Memory Usage: {metrics['memory_usage']:.1%}")
print(f"Active Tasks: {metrics['active_tasks']}")
```

## Advanced Features

### 1. Custom Agent Pool Integration

Create custom agent pools by implementing the MCP protocol:

```python
from mcp.server.fastmcp import FastMCP

class CustomAgentPool:
    def __init__(self):
        self.mcp = FastMCP("CustomAgentPool")
        self._register_tools()
    
    def _register_tools(self):
        @self.mcp.tool()
        def custom_analysis(data: dict) -> dict:
            # Custom analysis logic
            return {"result": "analysis_complete"}
    
    def run(self):
        self.mcp.run(host="localhost", port=8005)
```

### 2. Custom Strategy Templates

Define reusable strategy templates:

```python
strategy_template = {
    "name": "Custom Momentum Strategy",
    "description": "Custom momentum strategy with risk management",
    "required_data": ["price_data", "volume_data", "volatility_data"],
    "parameters": {
        "lookback_period": {"type": "int", "default": 20, "min": 5, "max": 100},
        "signal_threshold": {"type": "float", "default": 0.02, "min": 0.01, "max": 0.1}
    },
    "workflow": [
        {"step": "fetch_data", "agent": "data_agent_pool"},
        {"step": "calculate_momentum", "agent": "alpha_agent_pool"},
        {"step": "assess_risk", "agent": "risk_agent_pool"},
        {"step": "optimize_execution", "agent": "transaction_cost_agent_pool"}
    ]
}
```

### 3. Custom Reward Functions

Implement custom reward functions for RL:

```python
from core.rl_policy_engine import RewardFunction

class CustomRewardFunction(RewardFunction):
    def calculate_reward(self, state, action, next_state, portfolio_value, benchmark_value):
        # Custom reward calculation
        excess_return = (portfolio_value - benchmark_value) / benchmark_value
        volatility_penalty = np.std(portfolio_value) * 0.1
        return excess_return - volatility_penalty
```

### 4. Real-time Data Integration

Integrate real-time data feeds:

```python
async def setup_realtime_feed():
    orchestrator = FinAgentOrchestrator()
    await orchestrator.initialize()
    
    # Setup real-time data callback
    async def data_callback(data):
        await orchestrator.process_realtime_data(data)
    
    # Register callback with data provider
    data_provider.register_callback(data_callback)
```

### 5. Multi-Environment Deployment

Deploy across multiple environments:

```yaml
# Production environment
production:
  orchestrator:
    replicas: 3
    resources:
      cpu: "2"
      memory: "4Gi"
  
# Development environment  
development:
  orchestrator:
    replicas: 1
    resources:
      cpu: "1"
      memory: "2Gi"
```

## Best Practices

### 1. Strategy Development

- Start with simple strategies before complex ones
- Use sandbox environment for thorough testing
- Implement proper risk management
- Monitor performance continuously

### 2. System Configuration

- Use environment-specific configurations
- Implement proper logging and monitoring
- Set appropriate timeouts and retries
- Use resource limits in production

### 3. Performance Optimization

- Optimize agent pool communications
- Use batch operations where possible
- Implement caching for frequently accessed data
- Monitor and tune system resources

### 4. Error Handling

- Implement graceful degradation
- Use circuit breakers for external services
- Log errors with sufficient context
- Implement automatic recovery mechanisms

## Support and Contributing

### Getting Help

- Check the troubleshooting section
- Review test examples
- Check system logs
- Open an issue on GitHub

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### License

This project is licensed under the OpenMDW License. See LICENSE file for details.

---

For more information, visit the [FinAgent Documentation](docs/) or contact the development team.
