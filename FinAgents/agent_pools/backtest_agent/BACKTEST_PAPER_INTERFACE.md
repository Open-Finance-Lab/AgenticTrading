# Backtest Paper Interface Documentation

## Overview

This document describes the simple backtest function that follows the interface design from the Algorithmic Trading Review paper.

## Function: `run_simple_backtest_paper_interface`

### Interface Design (Based on Paper)

The function implements the five-stage AT system as described in the paper:

1. **Data Access/Cleaning**: Historical market data with cleaning rules
2. **Pre-trade Analysis**: 
   - Alpha Model (predictions)
   - Risk Model (risk constraints)
   - Transaction Cost Model (cost parameters)
3. **Portfolio Construction**: Combines Alpha, Risk, and Cost models
4. **Trade Execution**: Simulated execution with transaction costs
5. **Post-trade Analysis**: Performance metrics and visualizations

### Parameters

```python
run_simple_backtest_paper_interface(
    predictions=None,              # pd.Series with MultiIndex (datetime, instrument)
    start_time="2023-01-01",      # Backtest start date
    end_time="2023-12-31",        # Backtest end date
    look_back_period=20,          # Time window for pre-trade metrics (days)
    investment_horizon=5,          # Position holding period (days)
    topk=50,                      # Number of top stocks to select
    risk_thresholds=None,         # Risk constraints dict
    transaction_costs=None,        # Transaction cost parameters dict
    data_cleaning_rules=None,     # Data cleaning rules dict
    plot_results=True,            # Generate visualizations
    output_dir=None               # Directory to save plots
)
```

### Default Parameters

**Risk Thresholds:**
```python
{
    'max_position_size': 0.1,    # Maximum position size per stock
    'max_drawdown': 0.15,        # Maximum drawdown limit
    'var_limit': 0.02            # VaR limit (2%)
}
```

**Transaction Costs:**
```python
{
    'open_cost': 0.0005,         # Opening cost (0.05%)
    'close_cost': 0.0015,        # Closing cost (0.15%)
    'slippage': 0.0005           # Slippage (0.05%)
}
```

**Data Cleaning Rules:**
```python
{
    'remove_extreme_returns': True,  # Remove extreme returns
    'threshold': 3.0                 # Threshold in standard deviations
}
```

### Returns

The function returns a dictionary with:

```python
{
    'status': 'success',
    'backtest_period': {
        'start_time': '2023-01-01',
        'end_time': '2023-12-31',
        'total_days': 365
    },
    'strategy_parameters': {
        'look_back_period': 20,
        'investment_horizon': 5,
        'topk': 50
    },
    'performance_metrics': {
        'total_return': 0.15,
        'cumulative_return': 0.15,
        'sharpe_ratio': 1.2,
        'volatility': 0.18,
        'max_drawdown': -0.08,
        'calmar_ratio': 1.875
    },
    'risk_metrics': {
        'max_position_size_used': 0.1,
        'total_trades': 500,
        'transaction_costs': 0.01
    },
    'returns_series': pd.Series(...),  # Daily returns
    'positions': [...],                # Position history
    'visualizations': {
        'plots_generated': True,
        'plot_count': 4,
        'output_dir': 'output/'
    }
}
```

## Visualization Module (Decoupled)

The visualization functionality is implemented in a separate module `backtest_visualizer.py` for better decoupling.

### Available Plots

1. **P&L Curve**: Cumulative profit and loss over time
2. **Drawdown Analysis**: Drawdown curve showing portfolio drawdowns
3. **Metrics Table**: Tabular display of performance metrics
4. **Returns Distribution**: Histogram of daily returns
5. **Monthly Returns Heatmap**: Monthly returns visualization

### Usage Example

```python
from backtest_agent import BacktestAgent
from backtest_visualizer import BacktestVisualizer

# Initialize agent
agent = BacktestAgent()

# Run backtest
results = agent.run_simple_backtest_paper_interface(
    predictions=None,  # Will generate sample predictions
    start_time="2023-01-01",
    end_time="2023-12-31",
    look_back_period=20,
    investment_horizon=5,
    topk=50,
    plot_results=True,
    output_dir="backtest_output/"
)

# Access results
print(f"Total Return: {results['performance_metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.3f}")

# Use visualizer independently
visualizer = BacktestVisualizer()
visualizer.plot_pnl_curve(
    results['returns_series'],
    save_path="pnl_curve.png"
)
```

## Key Features

1. **Paper-Based Interface**: Follows the five-stage AT system design from the paper
2. **Decoupled Visualization**: Visualization module can be used independently
3. **Flexible Parameters**: All key parameters are configurable
4. **Comprehensive Metrics**: Includes all standard performance metrics
5. **Automatic Plotting**: Generates multiple visualization plots automatically

## Notes

- The function can work with or without qlib
- If predictions are not provided, sample predictions will be generated
- Visualization requires matplotlib
- All plots can be saved to files or displayed interactively

