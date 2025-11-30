#  Comprehensive Quantitative Backtesting Framework

A professional quantitative investment backtesting framework that supports comprehensive evaluation of Alpha factors and machine learning models. This framework integrates the complete workflow from data processing, factor calculation, model training, backtesting validation to performance evaluation.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Qlib](https://img.shields.io/badge/Framework-Qlib-green.svg)](https://github.com/microsoft/qlib)

## ✨ Key Features

- ** Alpha Factor Evaluation**: Support for technical indicators, momentum, mean reversion, and various Alpha factors
- ** Machine Learning Models**: Integrated mainstream ML models including LightGBM, Random Forest
- ** Information Coefficient (IC) Analysis**: Complete IC calculation and statistical analysis
- ** Unified Backtesting Interface**: Standardized backtesting pipeline for both factors and models
- ** Benchmark Comparison**: Detailed comparative analysis with market benchmarks
- ** Visualization Reports**: Rich charts and analytical reports
- ** Modular Design**: Easy to extend and customize
- ** Acceptance Criteria**: Automated validation with configurable acceptance thresholds

##  Project Structure

```
qlib/
├── comprehensive_demo.py         # Main demonstration entry point
├── enhanced_visualization_demo.py # Enhanced visualization and comparison pipeline
├── config.py                    # Configuration management
├── interfaces.py                # Standard interface definitions
├── factor_pipeline.py           # Factor evaluation pipeline
├── model_pipeline.py            # Model evaluation pipeline
├── data_interfaces.py           # Data interface abstractions
├── utils.py                     # Utility functions
├── output_processor.py          # Result processing
├── real_data_support.py         # Real data support
├── setup_data.py                # Data setup utilities
├── setup_data_download.py       # Data download utilities
└── qlib_data/                   # Data storage
    ├── stock_backup/            # Stock data backup
    ├── etf_backup/              # ETF data backup
    └── bitcoin_etfs/            # Bitcoin ETF data
```
- **🖼️ Enhanced Visualization & Comparison Pipeline**: `enhanced_visualization_demo.py` provides advanced visual analytics, including performance attribution, risk-return scatter plots, rolling metrics, and strategy-vs-benchmark comparison. It generates interactive HTML reports and charts for deeper insight into factor/model performance and portfolio risk.

##  Quick Start

### 4. Run Enhanced Visualization Demo

```bash
python enhanced_visualization_demo.py
```

This will generate a suite of interactive visual reports (HTML, Excel, CSV) in the `enhanced_visualizations/` folder, including:
- Performance charts
- Drawdown and underwater plots
- Monthly heatmaps
- Rolling Sharpe and beta analysis
- Risk-return scatter
- Factor exposure and attribution
- Position concentration and signal analysis
- Benchmark comparison tables

These visualizations help users intuitively compare strategies, analyze risk, and validate factor/model robustness against benchmarks.

### 1. Environment Setup

```bash
pip install qlib pandas numpy scikit-learn lightgbm plotly
```

### 2. Data Preparation

```bash
python setup_data_download.py
```

### 3. Run Comprehensive Demo

```bash
python comprehensive_demo.py
```

##  Data Preparation
## 🖼️ Enhanced Visualization Demo

### File: `enhanced_visualization_demo.py`

This module is designed for advanced visual analytics and strategy comparison. It automates the generation of rich, interactive reports for both alpha factors and machine learning models, enabling:
- Deep-dive performance attribution
- Rolling risk and return analysis
- Interactive charts (HTML)
- Excel and CSV summary exports
- Benchmark and strategy comparison

**Typical Workflow:**
1. Load backtest results and benchmark data
2. Generate visual reports (performance, drawdown, heatmap, attribution, etc.)
3. Export results to `enhanced_visualizations/` for review and sharing

**Usage Example:**
```bash
python enhanced_visualization_demo.py
```
Output files:
- `enhanced_visualizations/performance_chart.html`
- `enhanced_visualizations/rolling_sharpe_chart.html`
- `enhanced_visualizations/summary_report.html`
- ... (see folder for full list)

**Recommended for:**
- Strategy comparison and presentation
- In-depth risk analysis
- Portfolio review meetings
- Research documentation

### Supported Data Sources

The framework supports multiple data formats and sources:

- **Stock Data**: Historical OHLCV data for individual stocks
- **ETF Data**: Exchange-traded fund data
- **Crypto ETF Data**: Bitcoin and cryptocurrency ETF data

### Data Format Requirements

All data should be in CSV format with the following columns:
```
Date,Open,High,Low,Close,Volume,Dividends,Stock Splits
```

### Data Setup Process

1. **Download Historical Data**
   ```bash
   python setup_data_download.py
   ```
   This will download data for predefined instruments:
   - Stocks: AAPL, MSFT, GOOGL, AMZN, etc.
   - ETFs: SPY, QQQ, VTI, IWM, etc.
   - Bitcoin ETFs: GBTC, IBIT, FBTC

2. **Data Validation**
   ```python
   from config import QlibConfig
   
   config = QlibConfig(
       provider_uri="./qlib_data",
       instruments=["AAPL", "MSFT", "GOOGL"],
       basic_fields=["$open", "$high", "$low", "$close", "$volume"],
       freq="day"
   )
   ```

3. **Data Quality Checks**
   - Missing value detection and handling
   - Outlier detection
   - Data consistency validation
   - Date range verification

##  Alpha Factor Evaluation

### Built-in Alpha Factors

The framework includes six pre-built alpha factors:

1. **Momentum Factor (momentum_20d)**
   - **Description**: 20-day price momentum
   - **Calculation**: 20-period percentage change
   - **Signal**: Positive momentum = Buy signal

2. **Mean Reversion Factor (mean_reversion_10d)**
   - **Description**: Price deviation from 10-day moving average
   - **Calculation**: `(close - ma(close, 10)) / ma(close, 10)`
   - **Signal**: Negative deviation = Buy signal (mean reversion)

3. **RSI Divergence Factor (rsi_divergence)**
   - **Description**: RSI-based overbought/oversold signals
   - **Calculation**: 14-period RSI with divergence detection
   - **Signal**: RSI < 30 = Buy, RSI > 70 = Sell

4. **Volume Surge Factor (volume_surge)**
   - **Description**: Volume anomaly detection
   - **Calculation**: `volume / ma(volume, 20) - 1`
   - **Signal**: High volume surge = Potential breakout

5. **Volatility Factor (volatility_factor)**
   - **Description**: Inverse volatility weighting
   - **Calculation**: `1 / rolling_std(returns, 20)`
   - **Signal**: Lower volatility = Higher allocation

6. **MA Crossover Factor (ma_crossover)**
   - **Description**: Moving average ratio
   - **Calculation**: `close / ma(close, 20) - 1`
   - **Signal**: Price above MA = Buy signal

### Custom Factor Development

#### Method 1: Expression-based Factors
```python
custom_factor = FactorInput(
    factor_name="custom_momentum",
    factor_type="alpha",
    calculation_method="expression",
    expression="(close - delay(close, 5)) / delay(close, 5)",
    lookback_period=5
)
```

#### Method 2: Function-based Factors
```python
def custom_calculation(data, params):
    period = params.get('period', 20)
    return data['$close'].pct_change(period).fillna(0)

custom_factor = FactorInput(
    factor_name="custom_function",
    factor_type="alpha",
    calculation_method="function",
    function_name="custom_calculation",
    function_params={"period": 20}
)
```

### Factor Evaluation Metrics

#### Information Coefficient (IC) Analysis
- **IC Mean**: Average correlation between factor values and forward returns
- **IC Standard Deviation**: Stability of factor predictions
- **IC Information Ratio (IR)**: IC Mean / IC Std (risk-adjusted IC)
- **Rank IC**: Spearman correlation for non-linear relationships

#### Performance Metrics
- **Annual Return**: Annualized strategy returns
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Annual return / Maximum drawdown
- **Volatility**: Standard deviation of returns

### Acceptance Criteria

Factors are automatically evaluated against configurable thresholds:

```python
acceptance_criteria = StandardAcceptanceCriteria(
    min_annual_return=0.03,      # 3% minimum annual return
    min_sharpe_ratio=0.5,        # 0.5 minimum Sharpe ratio
    max_drawdown_threshold=0.3,  # 30% maximum drawdown
    min_ic_mean=0.01            # 1% minimum IC
)
```

##  Machine Learning Model Evaluation

### Supported Models

The framework supports two main categories of ML models:

#### 1. LightGBM Model (alpha_lgb_model)
```python
lgb_model = ModelInput(
    model_name="alpha_lgb_model",
    model_type="lightgbm",
    model_params={
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,           # Reduced for stability
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_estimators': 50,         # Optimized to avoid overfitting
        'max_depth': 3,             # Reduced complexity
        'min_child_samples': 20,
        'lambda_l1': 0.1,           # L1 regularization
        'lambda_l2': 0.1,           # L2 regularization
        'random_state': 42,
        'verbosity': -1
    }
)
```

#### 2. Random Forest Model (alpha_rf_model)
```python
rf_model = ModelInput(
    model_name="alpha_rf_model", 
    model_type="sklearn",
    model_class="RandomForestRegressor",
    model_params={
        'n_estimators': 50,         # Balanced complexity
        'max_depth': 6,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
)
```

### Feature Engineering

The framework automatically generates technical features:

1. **Price Features**
   - Returns (1d, 5d, 20d)
   - Moving averages (5, 10, 20, 60 periods)
   - Moving average ratios

2. **Volatility Features**
   - Rolling volatility (20 periods)
   - High-low spread

3. **Momentum Features**
   - Momentum indicators (1, 5, 10, 20 periods)

4. **Volume Features**
   - Volume moving average (20 periods)
   - Volume ratio

5. **Technical Indicators**
   - RSI (14 periods)

### Model Training Process

1. **Data Splitting**
   - Training: 70% of historical data
   - Testing: 30% of recent data (out-of-sample)

2. **Feature Preparation**
   ```python
   def _prepare_features_targets(self, data):
       feature_cols = [
           'returns_1d', 'returns_5d', 'returns_20d',
           'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_60',
           'volatility_20', 'momentum_1', 'momentum_5', 'momentum_10', 'momentum_20',
           'volume_ratio', 'hl_spread', 'rsi'
       ]
       # Forward returns as target
       targets = data.groupby('instrument')['$close'].pct_change().shift(-1)
       return data[feature_cols], targets
   ```

3. **Cross-Validation**
   - Time-series aware splitting
   - Walk-forward validation
   - Performance stability checks

### Model Evaluation Metrics

#### Performance Metrics
- **Annual Return**: Annualized model predictions return
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest portfolio decline
- **Direction Accuracy**: Percentage of correct direction predictions

#### Acceptance Criteria
```python
model_acceptance = StandardAcceptanceCriteria(
    min_annual_return=0.05,      # 5% minimum annual return
    min_sharpe_ratio=1.0,        # 1.0 minimum Sharpe ratio
    max_drawdown_threshold=0.25, # 25% maximum drawdown
)
```

##  Benchmark Comparison

### Benchmark Strategy

The framework compares strategies against a simple buy-and-hold benchmark:

1. **Buy-and-Hold Strategy**
   - Equal-weighted portfolio of all instruments
   - Rebalanced monthly
   - No transaction costs (baseline comparison)

2. **Market Benchmark**
   - SPY ETF performance (if available in dataset)
   - Risk-free rate (Treasury bills) for Sharpe calculation

### Performance Attribution

#### Return Decomposition
```python
def calculate_attribution(strategy_returns, benchmark_returns):
    """
    Calculate performance attribution metrics
    """
    excess_returns = strategy_returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(252)
    information_ratio = excess_returns.mean() * 252 / tracking_error
    
    return {
        'excess_return': excess_returns.mean() * 252,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio
    }
```

#### Risk-Adjusted Comparison
- **Alpha**: Excess return above benchmark
- **Beta**: Sensitivity to market movements  
- **Tracking Error**: Standard deviation of excess returns
- **Information Ratio**: Risk-adjusted excess return

### Benchmark Results Interpretation

#### Successful Strategy Criteria
1. **Absolute Performance**
   - Annual return > 5%
   - Sharpe ratio > 1.0
   - Maximum drawdown < 25%

2. **Relative Performance**
   - Positive alpha vs benchmark
   - Information ratio > 0.5
   - Reasonable tracking error (< 10%)

#### Example Results Analysis
```
 PERFORMANCE COMPARISON TABLE:
Type     Name                 Return   Sharpe   Status    
------------------------------------------------------------
Factor   mean_reversion_10d   11.43%   1.15      PASS    
Factor   rsi_divergence       13.61%   1.56      PASS    
Model    alpha_lgb_model      43.65%   1.33      PASS    
Model    alpha_rf_model       65.85%   2.01      PASS    
```

**Interpretation:**
- **rsi_divergence**: Best alpha factor with 13.61% return and 1.56 Sharpe
- **alpha_rf_model**: Top performer with 65.85% return and 2.01 Sharpe
- Both show strong risk-adjusted performance above acceptance thresholds

##  Strategy Deployment Guidelines

### Production Recommendations

1. **Risk Management**
   - Implement position sizing rules
   - Set stop-loss levels at 5-10%
   - Monitor real-time factor decay

2. **Portfolio Construction**
   - Combine multiple accepted factors/models
   - Diversify across different factor types
   - Regular rebalancing (monthly/quarterly)

3. **Performance Monitoring**
   - Track live IC vs backtest IC
   - Monitor strategy correlation changes
   - Alert on significant performance degradation

### Live Trading Considerations

1. **Transaction Costs**
   - Add realistic commission rates (0.1-0.5%)
   - Consider bid-ask spreads
   - Factor in market impact for large orders

2. **Data Quality**
   - Ensure real-time data feeds
   - Handle missing/delayed data gracefully
   - Validate data consistency

3. **Model Maintenance**
   - Retrain models quarterly
   - Update feature engineering as needed
   - Monitor for concept drift

## 💻 Usage Examples

### Basic Factor Evaluation

```python
from comprehensive_demo import *

# Configure instruments and date range
config = QlibConfig(
    provider_uri="./qlib_data",
    instruments=["AAPL", "MSFT", "GOOGL"],
    basic_fields=["$open", "$high", "$low", "$close", "$volume"],
    freq="day"
)

# Define custom factor
custom_factor = FactorInput(
    factor_name="custom_momentum_5d",
    factor_type="alpha", 
    calculation_method="expression",
    expression="(close - delay(close, 5)) / delay(close, 5)",
    lookback_period=5
)

# Create factor adapter
factor_adapter = FactorInputAdapter(custom_factor)

# Setup backtester
acceptance_criteria = StandardAcceptanceCriteria(
    min_annual_return=0.03,
    min_sharpe_ratio=0.5,
    max_drawdown_threshold=0.3,
    min_ic_mean=0.01
)

backtester = CSVBacktester(config, acceptance_criteria)
factor_evaluator = FactorEvaluator(acceptance_criteria)

# Run evaluation
data = backtester.prepare_data("2022-08-01", "2023-12-31")
result = factor_evaluator.evaluate_factor(
    backtester=backtester,
    factor=factor_adapter,
    start_date="2022-08-01", 
    end_date="2023-12-31"
)

# Print results
print(f"Factor: {result['factor_name']}")
print(f"Accepted: {result['is_accepted']}")
print(f"Annual Return: {result['metrics'].annual_return:.2%}")
print(f"Sharpe Ratio: {result['metrics'].sharpe_ratio:.2f}")
print(f"IC Mean: {result['metrics'].ic_mean:.4f}")
```

### Batch Evaluation Demo

```bash
# Run comprehensive demo for multiple strategies
python comprehensive_demo.py
```

Expected output:
```
 Starting Comprehensive Alpha Factor and Model Evaluation...
 Instruments: ['AAPL', 'MSFT', 'GOOGL']
 Alpha Factors to evaluate: 6
 ML Models to evaluate: 2

 ACCEPTED ALPHA FACTORS:
  • mean_reversion_10d - Annual Return: 11.43%, Sharpe: 1.15
  • rsi_divergence - Annual Return: 13.61%, Sharpe: 1.56

 ACCEPTED ML MODELS:
  • alpha_lgb_model - Annual Return: 43.65%, Sharpe: 1.33
  • alpha_rf_model - Annual Return: 65.85%, Sharpe: 2.01

- Comprehensive Alpha Factor and Model Evaluation Complete!
```

##  Configuration Options

### Data Configuration

```python
# Minimal configuration
config = QlibConfig(
    provider_uri="./qlib_data",
    instruments=["AAPL"],
    basic_fields=["$close"],
    freq="day"
)

# Full configuration
config = QlibConfig(
    provider_uri="/path/to/qlib_data",
    instruments=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    basic_fields=["$open", "$high", "$low", "$close", "$volume"],
    freq="day",
    start_time="2020-01-01",
    end_time="2023-12-31"
)
```

### Acceptance Criteria Tuning

```python
# Conservative criteria (high performance bar)
conservative_criteria = StandardAcceptanceCriteria(
    min_annual_return=0.10,      # 10% minimum return
    min_sharpe_ratio=1.5,        # 1.5 minimum Sharpe 
    max_drawdown_threshold=0.15, # 15% max drawdown
    min_ic_mean=0.03            # 3% minimum IC
)

# Relaxed criteria (lower performance bar)
relaxed_criteria = StandardAcceptanceCriteria(
    min_annual_return=0.02,      # 2% minimum return
    min_sharpe_ratio=0.3,        # 0.3 minimum Sharpe
    max_drawdown_threshold=0.4,  # 40% max drawdown  
    min_ic_mean=0.005           # 0.5% minimum IC
)
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Data Loading Errors
```
Error: File not found: /path/to/AAPL_daily.csv
```
**Solution:**
- Run `python setup_data_download.py` to download data
- Check that `qlib_data/stock_backup/` contains CSV files
- Verify file naming convention: `{INSTRUMENT}_daily.csv`

#### 2. MultiIndex Errors
```
Error: "The name instrument occurs multiple times, use a level number"
```
**Solution:**
- Already fixed in current implementation
- Use `level=1` instead of `level='instrument'` for groupby operations

#### 3. IC Calculation Issues
```
Error: IC Mean is None or NaN
```
**Solution:**
- Ensure sufficient data points (>20 observations)
- Check for missing factor values or target returns
- Verify data alignment between factor and returns

#### 4. Model Training Warnings
```
Warning: No further splits with positive gain, best_split=0 
```
**Solution:**
- Already optimized in current LightGBM parameters
- Reduce `n_estimators` and `max_depth`
- Add regularization (`lambda_l1`, `lambda_l2`)

##  Advanced Topics

### Custom Factor Development

```python
def bollinger_bands_factor(data, params):
    """
    Bollinger Bands factor implementation
    """
    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2)
    
    price = data['$close']
    ma = price.rolling(period).mean()
    std = price.rolling(period).std()
    
    upper_band = ma + (std * std_dev)
    lower_band = ma - (std * std_dev)
    
    # Signal: -1 when above upper band, +1 when below lower band
    signal = pd.Series(0, index=price.index)
    signal[price > upper_band] = -1  # Sell signal
    signal[price < lower_band] = 1   # Buy signal
    
    return signal
```

### Model Ensemble Methods

```python
class EnsembleModel:
    """
    Combine multiple models for improved performance
    """
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
    
    def predict(self, features):
        """Weighted average of predictions"""
        predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(features)
            predictions.append(pred * self.weights[i])
        
        return sum(predictions)
```

##  Performance Monitoring

### Factor Decay Detection

```python
def monitor_factor_decay(factor_ic_history, lookback_window=60):
    """
    Monitor factor performance decay over time
    """
    if len(factor_ic_history) < lookback_window:
        return None
    
    recent_ic = factor_ic_history[-lookback_window:].mean()
    historical_ic = factor_ic_history[:-lookback_window].mean()
    
    decay_ratio = recent_ic / historical_ic if historical_ic != 0 else 0
    
    if decay_ratio < 0.5:
        return "CRITICAL: Factor performance decayed >50%"
    elif decay_ratio < 0.7:
        return "WARNING: Factor performance decayed >30%"
    else:
        return "OK: Factor performance stable"
```

---


## 📝 License

This project is licensed under the OpenMDW License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For questions and support, please open an issue in the GitHub repository or contact the development team.

---

** Happy Backtesting! **

**Disclaimer**: This framework is for educational and research purposes only. It does not constitute investment advice. Please consult with professional financial advisors for actual investment decisions.
