# Risk Signal Agent Demo

A simple risk agent demo that uses OpenAI Agent SDK to generate risk signals from real market data using Qlib.

## Overview

This demo implements a Risk Signal Agent that:
- Loads real market data from Qlib
- Calculates comprehensive risk metrics (volatility, VaR, CVaR, max drawdown, beta, correlation, liquidity)
- Generates risk signals using LLM reasoning
- Provides structured risk assessment and recommendations

## Features

### Risk Metrics Calculation
- **Volatility**: Standard deviation of returns (annualized)
- **VaR (Value at Risk)**: Maximum expected loss at a given confidence level
- **CVaR (Conditional VaR)**: Expected loss beyond VaR
- **Max Drawdown**: Largest peak-to-trough decline
- **Beta**: Sensitivity to market movements
- **Correlation Risk**: Measure of asset co-movement
- **Liquidity Risk**: Based on volume and price impact

### Risk Signal Generation
- Automatic risk level assessment (LOW, MODERATE, HIGH)
- Risk score calculation
- Signal-based recommendations
- LLM-enhanced risk interpretation

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set OpenAI API key (optional, for LLM features):
```bash
export OPENAI_API_KEY='your-api-key'
```

## Usage

### Basic Example

```python
from risk_signal_agent import RiskSignalAgent
import pandas as pd

# Initialize agent
agent = RiskSignalAgent(name="RiskAgent", model="gpt-4o-mini")

# Load your data (must have: date, symbol, close, volume columns)
data = pd.DataFrame(...)  # Your market data

# Generate risk signals
result = agent.generate_risk_signals_from_data(
    data=data,
    risk_metrics=["volatility", "var", "cvar", "max_drawdown"]
)

# Check results
if result['status'] == 'success':
    print(f"Risk Level: {result['overall_risk_level']}")
    print(f"Risk Score: {result['risk_score']}")
    print(f"Risk Signals: {result['risk_signals']}")
```

### Run Examples

1. **Basic example usage**:
```bash
python example_usage.py
```

2. **Test with real Qlib data**:
```bash
python test_with_real_data.py
```

### Interactive Agent Usage

```python
# Use the agent for interactive risk analysis
agent = RiskSignalAgent()

# Ask questions about risk
response = agent.run(
    "Calculate VaR and CVaR for my portfolio and explain the difference"
)
print(response)
```

## Data Format

The agent expects data in the following format:

```python
# DataFrame with columns:
# - date: datetime
# - symbol: str (optional, for multi-asset analysis)
# - close: float (required)
# - volume: float (optional, for liquidity risk)
# - open, high, low: float (optional)
```

Example:
```python
data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=252),
    'symbol': 'AAPL',
    'close': [100, 101, 99, ...],
    'volume': [1000000, 1200000, ...]
})
```

## Output Format

The agent returns a structured dictionary:

```python
{
    "status": "success",
    "overall_risk_level": "MODERATE",  # LOW, MODERATE, or HIGH
    "risk_score": 0.45,  # 0.0 to 1.0
    "risk_metrics": {
        "volatility": {"current": 0.25, "mean": 0.23, ...},
        "var": {"historical_var": -0.03, ...},
        "cvar": {"cvar_value": -0.05, ...},
        "max_drawdown": {"value": -0.15, "percentage": -15.0},
        ...
    },
    "risk_signals": {
        "volatility": "MODERATE",
        "var": "LOW",
        "max_drawdown": "MODERATE",
        ...
    },
    "n_observations": 252,
    "date_range": {"start": "2023-01-01", "end": "2023-12-31"}
}
```

## Risk Signal Interpretation

### Risk Levels
- **LOW**: Risk levels are acceptable for normal trading operations
- **MODERATE**: Standard risk management practices recommended
- **HIGH**: Implement comprehensive risk management measures

### Risk Signals
- **Volatility**: LOW, MODERATE, HIGH
- **VaR**: LOW, MODERATE, SEVERE
- **Max Drawdown**: LOW, MODERATE, SEVERE
- **Beta**: LOW_MARKET_SENSITIVITY, MODERATE, HIGH_MARKET_SENSITIVITY
- **Correlation**: ACCEPTABLE, HIGH_DIVERSIFICATION_RISK
- **Liquidity**: Based on volume ratios

## Integration with Qlib

The agent uses Qlib data interfaces for:
- Data loading from Qlib data directory
- Data processing and normalization
- Factor calculation (if needed)

Data path: `/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/stock_backup`

## LLM Enhancement

When `OPENAI_API_KEY` is set, the agent can:
- Provide natural language explanations of risk metrics
- Generate risk management recommendations
- Answer questions about risk analysis
- Interpret complex risk scenarios

## Examples

See `example_usage.py` for:
- Basic risk analysis
- Beta calculation with market returns
- Comprehensive risk analysis
- Interactive agent usage

See `test_with_real_data.py` for:
- Real data testing
- Structured output formatting
- LLM-enhanced interpretation

## Requirements

- Python 3.8+
- pandas
- numpy
- openai (for LLM features)
- Qlib data (for real data testing)

## Notes

- All risk calculations use standard financial formulas
- Risk signals are generated based on configurable thresholds
- The agent can work with single assets or portfolios
- LLM features require OpenAI API key

## License

Part of the FinAgent-Orchestration project.

