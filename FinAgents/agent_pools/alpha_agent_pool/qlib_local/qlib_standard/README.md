# Qlib Standard Framework

A complete, production-ready implementation of quantitative trading research workflows using Qlib standard APIs and interfaces.

##  Overview

This framework provides a comprehensive solution for quantitative finance research and trading strategy development, fully compliant with Qlib standards. It includes data loading, factor calculation, model training, strategy execution, and backtesting capabilities.

## 📦 Key Components

### 1. Data Loading (`data_loader.py`)
- **QlibCSVDataLoader**: Load data from CSV files with proper Qlib formatting
- **QlibSyntheticDataLoader**: Generate synthetic market data for testing
- Full multi-index DataFrame support (datetime, instrument)
- Automatic data validation and formatting

### 2. Factor Calculation (`factor_calculator.py`) 
- **QlibFactorCalculator**: 30+ technical indicators using Qlib Expression engine
- RSI, Bollinger Bands, Moving Averages, Momentum indicators
- Cross-sectional ranking and normalization
- Custom factor expression support

### 3. Model Training (`model_trainer.py`)
- **QlibModelTrainer**: ML model training with Qlib Model interface
- Support for LightGBM, Linear, Ridge, Lasso, RandomForest
- Feature importance analysis
- Model persistence and loading

### 4. Strategy Execution (`strategy_executor.py`)
- **QlibStrategyExecutor**: Risk-managed trading strategies
- **QlibTopKStrategy**: Top-K selection strategy
- Long-only, short-only, and long-short strategies
- Position sizing and risk management

### 5. Data Processing (`data_handler.py`)
- **QlibDataHandler**: Comprehensive data preprocessing
- Normalization, missing value handling
- Train/validation/test splits
- Data quality validation

### 6. Complete Framework (`framework.py`)
- **QlibStandardFramework**: End-to-end pipeline integration
- Automatic component initialization
- Configuration management
- Results saving and analysis

##  Quick Start

### Basic Usage

```python
from qlib_standard import QlibStandardFramework

# Use default configuration
framework = QlibStandardFramework()
results = framework.run_complete_pipeline()
```

### Custom Configuration

```python
config = {
    'data': {
        'source_type': 'synthetic',
        'instruments': ['AAPL', 'MSFT', 'GOOGL'],
        'start_time': '2022-01-01',
        'end_time': '2023-12-31'
    },
    'model': {
        'type': 'lightgbm',
        'config': {
            'num_leaves': 31,
            'learning_rate': 0.1
        }
    },
    'strategy': {
        'type': 'long_short',
        'top_k': 5,
        'bottom_k': 5
    }
}

framework = QlibStandardFramework(config=config)
results = framework.run_complete_pipeline()
```

##  Demo Examples

### Synthetic Data Demo
```python
from qlib_standard.demo import run_synthetic_data_demo
results = run_synthetic_data_demo()
```

### CSV Data Demo
```python
from qlib_standard.demo import run_csv_data_demo
results = run_csv_data_demo('/path/to/your/data.csv')
```

### Model Comparison
```python
from qlib_standard.demo import compare_models_demo
comparison = compare_models_demo()
```

##  Configuration Options

### Data Configuration
- `source_type`: 'csv' or 'synthetic'
- `data_path`: Path to CSV data (for CSV source)
- `instruments`: List of instrument symbols
- `start_time`, `end_time`: Date range
- `freq`: Data frequency ('1D', '1H', etc.)

### Factor Configuration
- `enabled`: Enable/disable factor calculation
- `factor_list`: List of specific factors to calculate
- `custom_factors`: Custom factor expressions

### Model Configuration
- `type`: Model type ('lightgbm', 'linear', 'ridge', etc.)
- `config`: Model-specific parameters

### Strategy Configuration
- `type`: Strategy type ('long_short', 'topk', etc.)
- `top_k`, `bottom_k`: Number of positions
- `max_position_weight`: Maximum position size

##  Results and Analysis

The framework provides comprehensive results including:

- **Data Summary**: Raw data statistics, feature counts
- **Model Summary**: Training results, feature importance
- **Backtest Summary**: Performance metrics, risk analysis
- **Complete Data**: All intermediate and final results

### Key Performance Metrics
- Annual Return
- Sharpe Ratio
- Maximum Drawdown
- Volatility
- Risk-adjusted returns

## 🏗️ Architecture

The framework follows Qlib's standard architecture:

```
QlibStandardFramework
├── QlibCSVDataLoader/QlibSyntheticDataLoader
├── QlibFactorCalculator
├── QlibDataHandler
├── QlibModelTrainer
└── QlibStrategyExecutor
```

Each component inherits from the appropriate Qlib base class:
- `DataLoader` → `qlib.data.dataset.loader.DataLoader`
- `Model` → `qlib.model.base.Model`
- `Strategy` → `qlib.contrib.strategy.WeightStrategyBase`
- `DataHandler` → `qlib.data.dataset.handler.DataHandlerLP`

##  Features

###  Qlib API Compliance
- Full inheritance from Qlib base classes
- Standard method signatures and interfaces
- Compatible with Qlib ecosystem

###  Comprehensive Factor Library
- 30+ technical indicators
- Custom expression support
- Cross-sectional operations
- Proper handling of missing data

###  Multiple ML Algorithms
- LightGBM with hyperparameter tuning
- Linear regression variants
- Ensemble methods
- Feature importance analysis

###  Risk Management
- Position sizing controls
- Maximum weight constraints
- Portfolio rebalancing
- Drawdown monitoring

###  Production Ready
- Robust error handling
- Comprehensive logging
- Configuration validation
- Result persistence

## Requirements

- Python 3.8+
- qlib >= 0.9.0
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- lightgbm >= 3.3.0

##  File Structure

```
qlib_standard/
├── __init__.py          # Package initialization
├── data_loader.py       # Data loading components
├── factor_calculator.py # Factor calculation
├── model_trainer.py     # Model training
├── strategy_executor.py # Strategy execution
├── data_handler.py      # Data preprocessing
├── framework.py         # Complete framework
├── demo.py             # Demo examples
└── README.md           # This file
```

## 🚨 Error Handling

The framework includes comprehensive error handling for:
- Data loading failures
- Factor calculation errors
- Model training issues
- Strategy execution problems
- Configuration validation

All errors are logged with descriptive messages and debugging information.

##  Workflow

1. **Initialize**: Setup framework with configuration
2. **Load Data**: Import raw market data
3. **Calculate Factors**: Generate technical indicators
4. **Process Data**: Clean and normalize features
5. **Train Model**: Fit ML model to training data
6. **Generate Predictions**: Create trading signals
7. **Execute Strategy**: Run backtesting simulation
8. **Analyze Results**: Performance metrics and analysis

## - Getting Started

To get started quickly:

1. Import the framework:
   ```python
   from qlib_standard import QlibStandardFramework
   ```

2. Run with defaults:
   ```python
   framework = QlibStandardFramework()
   results = framework.run_complete_pipeline()
   ```

3. Check the results in the generated output directory!

## 📞 Support

This framework is designed to be self-contained and well-documented. All components include comprehensive docstrings and error messages to guide usage.

For advanced usage, refer to the individual component documentation and the Qlib official documentation for underlying concepts.
