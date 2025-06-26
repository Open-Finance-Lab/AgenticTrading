# Risk Agent Pool

**Author:** Jifeng Li  
**License:** openMDW

## Overview

The Risk Agent Pool is a comprehensive financial risk management system that provides modular, extensible risk analysis capabilities. It integrates with external memory agents for data persistence, leverages OpenAI's language models for natural language context processing, and uses the Model Context Protocol (MCP) for task distribution and orchestration.

## Architecture

### Core Components

1. **RiskAgentPool** (`core.py`) - Main orchestrator with OpenAI integration and MCP server
2. **Agent Registry** (`registry.py`) - Dynamic agent discovery and lifecycle management
3. **Memory Bridge** (`memory_bridge.py`) - External memory integration and caching
4. **Specialized Agents** (`agents/`) - Risk analysis implementations

### Design Principles

- **Modular**: Each risk type has its own specialized agent
- **Extensible**: Easy to add new risk agents and analysis methods
- **Asynchronous**: Non-blocking operations for better performance
- **Memory-enabled**: Persistent storage and retrieval of risk data
- **LLM-powered**: Natural language to structured task conversion
- **MCP-compliant**: Standardized protocol for task distribution

## Features

### Risk Analysis Types

- **Market Risk**: Price risk, volatility analysis, VaR calculations
- **Credit Risk**: PD, LGD, EAD, credit VaR, portfolio analysis
- **Liquidity Risk**: Market and funding liquidity assessment
- **Operational Risk**: Fraud detection, KRI monitoring, OpVaR
- **Stress Testing**: Scenario analysis, sensitivity testing, Monte Carlo
- **Model Risk**: Model validation, performance monitoring, governance

### Advanced Capabilities

- **Historical and hypothetical scenario analysis**
- **Monte Carlo simulations with correlation modeling**
- **Reverse stress testing**
- **Real-time performance monitoring**
- **Comprehensive model lifecycle management**
- **Fraud detection and prevention**
- **Regulatory compliance reporting**

## Quick Start

### Basic Usage

```python
from FinAgents.agent_pools.risk import RiskAgentPool

# Initialize the risk agent pool
pool = RiskAgentPool(
    openai_api_key="your_api_key",
    external_memory_config={
        "host": "localhost",
        "port": 8000
    }
)

# Start the MCP server
await pool.start()

# Process natural language risk request
context = "Calculate VaR for my equity portfolio with 95% confidence level"
result = await pool.process_orchestrator_input(context)

print(result)
```

### Direct Agent Usage

```python
from FinAgents.agent_pools.risk.agents import MarketRiskAnalyzer

# Initialize market risk analyzer
analyzer = MarketRiskAnalyzer()

# Calculate portfolio VaR
portfolio_data = {
    "positions": [
        {"asset": "AAPL", "quantity": 100, "price": 150.0},
        {"asset": "GOOGL", "quantity": 50, "price": 2800.0}
    ],
    "returns_data": {...}  # Historical returns
}

var_result = await analyzer.calculate_var(
    portfolio_data, 
    confidence_level=0.95,
    method="historical"
)
```

## API Reference

### RiskAgentPool

The main orchestrator class for the risk agent pool.

#### Methods

- `process_orchestrator_input(context: str)` - Process natural language input
- `execute_structured_task(task: Dict)` - Execute structured risk task
- `register_mcp_tools()` - Register MCP endpoints
- `start()` - Start the MCP server
- `stop()` - Stop the MCP server

### Agent Registry

Dynamic agent discovery and management.

#### Methods

- `register_agent(name: str, agent_class: Type)` - Register new agent
- `get_agent_class(name: str)` - Get agent class by name
- `list_agents()` - List all registered agents

### Memory Bridge

Integration with external memory systems.

#### Methods

- `record_event(event_data: Dict)` - Record risk event
- `store_analysis_result(analysis: RiskAnalysisRecord)` - Store analysis
- `retrieve_analysis_history(filters: Dict)` - Retrieve historical data
- `store_model_parameters(params: RiskModelParameters)` - Store model data

## Risk Agents

### Market Risk Agent

Comprehensive market risk analysis including:
- Volatility calculations (historical, GARCH, implied)
- Value at Risk (VaR) and Expected Shortfall (ES)
- Beta analysis and factor decomposition
- Maximum drawdown and tail risk metrics

**Example Usage:**
```python
request = {
    "portfolio_data": portfolio_data,
    "risk_measures": ["var", "volatility", "beta"],
    "time_horizon": "daily"
}
result = await market_risk_agent.analyze(request)
```

### Credit Risk Agent

Credit risk assessment and modeling:
- Probability of Default (PD) estimation
- Loss Given Default (LGD) modeling
- Exposure at Default (EAD) calculation
- Credit VaR and portfolio analysis

**Example Usage:**
```python
request = {
    "borrower_data": {
        "credit_score": 720,
        "debt_to_income": 0.35,
        "loan_amount": 250000
    },
    "analysis_type": "pd_estimation"
}
result = await credit_risk_agent.analyze(request)
```

### Operational Risk Agent

Operational risk management and monitoring:
- Fraud detection and prevention
- Key Risk Indicator (KRI) monitoring
- Operational VaR calculation
- Event recording and analysis

**Example Usage:**
```python
request = {
    "analysis_type": "fraud_assessment",
    "transaction_data": {
        "amount": 50000,
        "location": "foreign_country",
        "timestamp": datetime.now()
    }
}
result = await operational_risk_agent.analyze(request)
```

### Stress Testing Agent

Comprehensive stress testing framework:
- Historical scenario replay
- Hypothetical scenario analysis
- Monte Carlo simulations
- Reverse stress testing

**Example Usage:**
```python
request = {
    "test_type": "scenario",
    "scenario_id": "2008_financial_crisis",
    "portfolio": portfolio_positions
}
result = await stress_testing_agent.analyze(request)
```

### Model Risk Agent

Model lifecycle management and governance:
- Model registration and inventory
- Validation and performance monitoring
- Change tracking and approval
- Governance reporting

**Example Usage:**
```python
request = {
    "action": "validate_model",
    "model_id": "PRICING_MODEL_001",
    "validator": "Risk Team",
    "validation_config": {
        "accuracy_tests": {"min_accuracy": 0.85},
        "stability_tests": {},
        "bias_tests": {}
    }
}
result = await model_risk_agent.analyze(request)
```

## Configuration

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# External Memory Configuration
EXTERNAL_MEMORY_HOST=localhost
EXTERNAL_MEMORY_PORT=8000

# MCP Server Configuration
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=3000

# Logging Configuration
LOG_LEVEL=INFO
```

### Configuration File

```yaml
risk_agent_pool:
  openai:
    api_key: ${OPENAI_API_KEY}
    model: "gpt-4"
    temperature: 0.1
    max_tokens: 4000
  
  external_memory:
    host: ${EXTERNAL_MEMORY_HOST}
    port: ${EXTERNAL_MEMORY_PORT}
    cache_enabled: true
    cache_ttl: 3600
  
  mcp_server:
    host: ${MCP_SERVER_HOST}
    port: ${MCP_SERVER_PORT}
    
  risk_thresholds:
    var_confidence_levels: [0.95, 0.99, 0.999]
    stress_test_scenarios: ["2008_crisis", "covid19", "rate_shock"]
    kri_monitoring_frequency: "daily"
```

## Integration Examples

### With External Orchestrator

```python
# External orchestrator sends natural language context
orchestrator_request = {
    "context": "Assess credit risk for new loan application with FICO 680 and DTI 0.42",
    "priority": "high",
    "requestor": "lending_system"
}

# Risk agent pool processes and responds
response = await risk_pool.process_orchestrator_input(
    orchestrator_request["context"]
)
```

### With External Memory Agent

```python
# Store risk analysis results
analysis_record = RiskAnalysisRecord(
    analysis_id="RISK_20240101_001",
    risk_type="credit",
    portfolio_id="LOAN_PORTFOLIO_A",
    analysis_results=credit_analysis_results,
    timestamp=datetime.now()
)

await memory_bridge.store_analysis_result(analysis_record)
```

### With MCP Client

```python
# MCP client calls risk analysis tools
mcp_client.call_tool(
    "calculate_portfolio_var",
    {
        "portfolio_data": portfolio_data,
        "confidence_level": 0.99,
        "method": "monte_carlo"
    }
)
```

## Testing

### Unit Tests

```bash
# Run all risk agent tests
python -m pytest tests/test_risk_agent_pool.py -v

# Run specific agent tests
python -m pytest tests/test_market_risk_agent.py -v
python -m pytest tests/test_credit_risk_agent.py -v
```

### Integration Tests

```bash
# Test full integration with external memory
python -m pytest tests/test_risk_integration.py -v

# Test MCP server functionality
python -m pytest tests/test_mcp_integration.py -v
```

### Performance Tests

```bash
# Load testing for risk calculations
python tests/performance/test_risk_performance.py

# Stress testing for concurrent requests
python tests/performance/test_concurrent_analysis.py
```

## Monitoring and Observability

### Logging

The risk agent pool provides structured logging with:
- Request/response tracing
- Performance metrics
- Error tracking
- Audit trails

### Metrics

Key performance indicators:
- Analysis response times
- Memory usage patterns
- Error rates by agent type
- Cache hit ratios

### Health Checks

```python
# Check agent pool health
health_status = await risk_pool.health_check()

# Check individual agent status
agent_status = await risk_pool.check_agent_health("market_risk_agent")
```

## Extension and Customization

### Adding New Risk Agents

1. Implement `BaseRiskAgent` interface
2. Register in the agent registry
3. Add MCP tool endpoints
4. Update documentation

```python
class CustomRiskAgent(BaseRiskAgent):
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation
        pass

# Register the new agent
register_agent("custom_risk_agent", CustomRiskAgent)
```

### Custom Risk Models

Extend existing agents with custom models:

```python
class EnhancedVaRCalculator(VaRCalculator):
    async def calculate_var_with_custom_model(self, data, model_params):
        # Custom VaR implementation
        pass
```

## Security Considerations

- **API Key Management**: Secure storage of OpenAI API keys
- **Access Control**: Role-based access to risk analysis functions
- **Data Privacy**: Anonymization of sensitive portfolio data
- **Audit Logging**: Complete audit trail of risk calculations

## Performance Optimization

- **Caching**: Intelligent caching of frequently accessed data
- **Parallel Processing**: Concurrent risk calculations
- **Memory Management**: Efficient handling of large datasets
- **Database Optimization**: Optimized queries for historical data

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**: Check API key and rate limits
2. **Memory Agent Connection**: Verify external memory service
3. **MCP Server Issues**: Check port availability and configuration
4. **Performance Issues**: Monitor memory usage and caching

### Debug Mode

```python
# Enable debug logging
risk_pool = RiskAgentPool(debug=True)

# Detailed error reporting
try:
    result = await risk_pool.process_orchestrator_input(context)
except Exception as e:
    logger.error(f"Risk analysis failed: {e}", exc_info=True)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code quality checks
black FinAgents/agent_pools/risk/
flake8 FinAgents/agent_pools/risk/
mypy FinAgents/agent_pools/risk/
```

## License

This project is licensed under the openMDW license.

## Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation wiki
