# Alpha Signal Agent

A simple agent built with OpenAI Agent SDK for generating alpha trading signals. This agent integrates Qlib factor construction, technical indicators, and machine learning inference to produce trading signals based on algorithmic trading research.

## Features

- **Qlib Factor Construction**: Build factors using Qlib's standardized factor calculation framework
- **Technical Indicators**: Calculate RSI, MACD, Bollinger Bands, Moving Averages, Momentum, and Volume indicators
- **ML Inference**: Train and use linear regression, LightGBM, or Random Forest models for alpha prediction
- **Signal Generation**: Generate long/short/neutral trading signals from model predictions
- **OpenAI Agent SDK Integration**: Interactive agent that can answer questions and execute trading workflows

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set OpenAI API key (optional, for interactive agent features):
```bash
export OPENAI_API_KEY='your-api-key'
```

## Quick Start

### Basic Usage

```python
from alpha_signal_agent import AlphaSignalAgent
import pandas as pd

# Initialize agent
agent = AlphaSignalAgent(name="MyAlphaAgent")

# Load your market data (DataFrame with MultiIndex: Date, instrument)
data = pd.DataFrame(...)  # Your OHLCV data

# Define factors
factors = [
    {
        'factor_name': 'momentum_20d',
        'factor_type': 'alpha',
        'calculation_method': 'expression',
        'lookback_period': 20
    }
]

# Generate signals
result = agent.generate_signals_from_data(
    data=data,
    factors=factors,
    indicators=['RSI', 'MACD', 'Bollinger'],
    model_type='linear',  # or 'lightgbm' or 'random_forest'
    signal_threshold=0.01
)

# Access results
if result['status'] == 'success':
    signals = result['signals']
    performance = result['model_performance']
    print(f"Test Correlation: {performance['test_correlation']:.4f}")
```

### Interactive Agent Usage

```python
from alpha_signal_agent import AlphaSignalAgent

# Initialize agent
agent = AlphaSignalAgent()

# Ask questions or request actions
response = agent.run(
    "Generate alpha signals using momentum factors and RSI indicators"
)
print(response)
```

## Examples

Run the example script to see various usage patterns:

```bash
python example_usage.py
```

The examples demonstrate:
1. Basic signal generation pipeline
2. Qlib factor construction
3. Comparing different ML models
4. Interactive agent queries

## Architecture

### Components

1. **Factor Construction Tools**:
   - `construct_qlib_factor`: Create factor specifications
   - `calculate_qlib_factor`: Calculate factor values

2. **Technical Indicator Tools**:
   - `calculate_technical_indicators`: Compute RSI, MACD, Bollinger, MA, Momentum, Volume

3. **ML Inference Tools**:
   - `train_linear_regression_model`: Train linear regression model
   - `train_ml_model`: Train LightGBM or Random Forest model
   - `generate_alpha_signals`: Convert predictions to trading signals

4. **AlphaSignalAgent**:
   - Main agent class that orchestrates the workflow
   - Integrates all tools via OpenAI Agent SDK
   - Provides high-level API for signal generation

## Data Format

Input data should be a pandas DataFrame with:
- **MultiIndex**: (Date, instrument)
- **Columns**: `$open`, `$high`, `$low`, `$close`, `$volume` (or `open`, `high`, `low`, `close`, `volume`)

Example:
```python
data = pd.DataFrame({
    '$open': [...],
    '$high': [...],
    '$low': [...],
    '$close': [...],
    '$volume': [...]
})
data.index = pd.MultiIndex.from_tuples([(date, symbol) for ...])
```

## Factor Types

Supported factor calculation methods:

1. **Expression-based**: Use Qlib expressions like `close / Ref(close, 20) - 1`
2. **Function-based**: Use predefined functions (momentum, volatility, etc.)

## Technical Indicators

Supported indicators:
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger**: Bollinger Bands position
- **MA**: Moving Average ratios (5, 10, 20, 50 day)
- **Momentum**: Price momentum (5, 10, 20 day)
- **Volume**: Volume-based indicators

## Model Types

1. **Linear Regression**: Fast, interpretable, good baseline
2. **LightGBM**: Gradient boosting, handles non-linear relationships
3. **Random Forest**: Ensemble method, robust to overfitting

## Signal Generation

Signals are generated based on model predictions:
- **Long signals**: Prediction > threshold
- **Short signals**: Prediction < -threshold
- **Neutral signals**: Prediction between thresholds

## Research Basis

This agent implements concepts from algorithmic trading research:
- Factor-based alpha generation (Gu et al., 2020)
- Technical indicator analysis (Brock et al., 1992)
- Machine learning for return prediction (Jiang et al., 2020)
- Qlib framework for factor research (Yang et al., 2020)

## Notes

- All code comments are in English
- The agent uses existing Qlib infrastructure from the parent alpha_agent_pool
- For production use, ensure proper data validation and risk management
- Model performance should be validated on out-of-sample data

## License

Part of the FinAgent-Orchestration project.

