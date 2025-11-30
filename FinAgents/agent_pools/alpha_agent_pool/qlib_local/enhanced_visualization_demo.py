"""
Enhanced Visualization Demo
Demonstrates all available charts, line plots, heatmaps, and advanced visualizations
"""

from complete_framework import BacktestingFramework
from data_interfaces import DatasetInput, FactorInput, ModelInput, StrategyInput, OutputFormat
import webbrowser
import os
import time

def run_enhanced_visualization_demo():
    """
    Run comprehensive visualization demonstration with all chart types
    
    This function demonstrates the complete backtesting framework with:
    - Multiple factor calculations (momentum, volatility, RSI)
    - LightGBM model training and prediction
    - Long-only factor-weighted strategy
    - Comprehensive visualization suite (12+ chart types)
    - ETF benchmark comparison including Bitcoin ETFs
    """
    
    print(" Starting Enhanced Visualization Demo...")
    print("=" * 60)
    
    # Initialize the backtesting framework
    framework = BacktestingFramework()
    
    # Configure dataset input - using hourly multi-stock portfolio for intraday trading
    dataset_config = DatasetInput(
        source_type="csv_hourly",            # Data source type: hourly CSV files for intraday analysis
        file_path="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/stock_backup/",  # Directory path
        start_date="2022-09-01",             # Start from available hourly data period
        end_date="2024-12-31",               # Backtest end date 
        required_fields=["open", "high", "low", "close", "volume"],  # Required OHLCV fields
        universe="custom_list",              # Universe type: custom symbol list
        custom_symbols=["AAPL", "MSFT", "GOOGL", "JPM", "TSLA", "NVDA", "META"], #,"AVGO","BAC","COST"],  # 7-stock portfolio for intraday visualization
        adjust_price=True,                   # Apply price adjustments for splits/dividends
        fill_method="ffill",                 # Forward fill missing data
        min_periods=24                       # 24 hours minimum for hourly data
    )
    
    # Configure hourly factor suite - optimized for intraday trading patterns
    factor_configs = [
        # Intraday momentum factors (hourly timeframes)
        FactorInput(
            factor_name="momentum_1h",        # 1-hour momentum for immediate trend capture
            factor_type="alpha",
            calculation_method="expression",
            lookback_period=1
        ),
        FactorInput(
            factor_name="momentum_4h",        # 4-hour momentum for short-term trend
            factor_type="alpha",
            calculation_method="expression",
            lookback_period=4
        ),
        FactorInput(
            factor_name="momentum_24h",       # Daily momentum equivalent (24 hours)
            factor_type="alpha",
            calculation_method="expression",
            lookback_period=24
        ),
        
        # Intraday volatility factors
        FactorInput(
            factor_name="volatility_2h",      # 2-hour volatility for risk assessment
            factor_type="risk",
            calculation_method="expression",
            lookback_period=2
        ),
        
        # High-frequency mean reversion
        FactorInput(
            factor_name="rsi_6h",             # 6-hour RSI for intraday overbought/oversold
            factor_type="technical",
            calculation_method="expression",
            lookback_period=6
        ),
        FactorInput(
            factor_name="bollinger_position", # 12-hour Bollinger position for hourly mean reversion
            factor_type="technical",
            calculation_method="expression",
            lookback_period=12
        ),
        
        # Volume confirmation for hourly trading
        FactorInput(
            factor_name="volume_momentum",    # 3-hour volume momentum for liquidity analysis
            factor_type="technical",
            calculation_method="expression",
            lookback_period=3
        ),
        
        # Price microstructure for high-frequency alpha
        FactorInput(
            factor_name="price_acceleration", # 30-minute price acceleration for momentum bursts
            factor_type="alpha",
            calculation_method="expression",
            lookback_period=1  # 30 minutes represented as fraction of hour
        )
    ]
    
    # Configure LightGBM machine learning model for enhanced predictions
    model_config = ModelInput(
        model_name="advanced_lgbm_trading_model",  # Advanced LightGBM for financial time series
        model_type="tree",                         # Tree-based ensemble method
        implementation="lightgbm",                 # Native LightGBM implementation
        model_class="LGBMRegressor",               # LightGBM regressor for superior performance
        target_type="market_neutral",              # Market neutral targets for cross-sectional ranking
        hyperparameters={                            # LightGBM parameters (English descriptions)
            "n_estimators": 500,                    # Maximum number of trees (training upper bound); early stopping will not exceed this
            "max_depth": 6,                         # Max depth per tree (controls complexity / overfitting)
            "learning_rate": 0.001,                  # Learning rate; smaller is more stable but requires more trees
            "subsample": 0.8,                       # Row sampling fraction (bagging) to reduce overfitting
            "colsample_bytree": 0.8,                # Column sampling fraction per tree
            "reg_alpha": 0.01,                      # L1 regularization strength
            "reg_lambda": 0.01,                     # L2 regularization strength
            "min_child_samples": 20,                # Minimum data in a leaf to avoid very small splits
            "min_split_gain": 0.0,                  # Minimum gain required to make a split
            "num_leaves": 31,                       # Maximum number of leaves per tree (controls model complexity)
            "random_state": 42,                     # Random seed for reproducibility
            "n_jobs": -1,                           # Number of parallel threads (-1 uses all available cores)
            "verbose": -1,                          # Verbosity level (-1 = silent)
            "objective": "regression",            # Objective / loss function
            "metric": "rmse",                     # Validation metric (printed during training/validation)
            "early_stopping_rounds": 100           # Early stopping patience: stop if validation doesn't improve for 100 rounds (patience, not max iterations)
        },
        training_method="rolling",                 # Rolling window training
        training_period=2160,                        # 90 days of hourly periods
        validation_period=720,                       # 30 days of hourly periods
        rebalance_frequency="hourly",
        retrain_frequency='weekly',                   # Retrain the model every week
        retrain_step_periods=35                        # Explicit step: 5 trading days * 7 points/day = 35
    )
    # model_config = ModelInput(
    #     model_name="advanced_lgbm_trading_model",  # Advanced LightGBM for financial time series
    #     model_type="tree",                         # Tree-based ensemble method
    #     implementation="lightgbm",                 # Native LightGBM implementation
    #     model_class="LGBMRegressor",               # LightGBM regressor for superior performance
    #     target_type="market_neutral",              # Market neutral targets for cross-sectional ranking
    #     hyperparameters={                          # Simplified LightGBM parameters for stable learning
    #         "n_estimators": 100,                   # Moderate number of trees
    #         "max_depth": 4,                        # Shallower trees to prevent overfitting
    #         "learning_rate": 0.1,                  # Faster learning rate
    #         "subsample": 0.8,                      # Row sampling
    #         "colsample_bytree": 0.8,               # Column sampling
    #         "reg_alpha": 0.01,                     # Light L1 regularization
    #         "reg_lambda": 0.01,                    # Light L2 regularization
    #         "min_child_samples": 20,               # Lower minimum samples for more flexibility
    #         "min_split_gain": 0.0,                 # Allow any gain for splits
    #         "num_leaves": 15,                      # Moderate number of leaves
    #         "random_state": 42,                    # Reproducible results
    #         "n_jobs": -1,                          # Use all CPU cores
    #         "verbose": -1,                         # Suppress verbose output
    #         "boosting_type": "gbdt",               # Gradient boosting decision tree
    #         "objective": "regression"              # Regression objective
    #     },
    #     training_method="rolling",                 # Rolling window training
    #     training_period=8760,                      # 1 year of hourly periods (365 days * 24 hours = 8760)
    #     validation_period=2190,                    # 3 months of hourly periods (91.25 days * 24 hours = 2190)
    #     rebalance_frequency="hourly"               # Hourly rebalancing
    # )
    # Configure optimized hourly trading strategy for signal generation
    strategy_config = StrategyInput(
        strategy_name="hourly_intraday_strategy",       # Strategy identifier for hourly trading
        strategy_type="long_only",                      # Long-only positions for stability
        position_method="factor_weight",                # Factor-weighted position sizing
        num_positions=7,                                # Multi-stock portfolio (7 stocks for better visualization)
        rebalance_frequency="hourly",                   # Hourly rebalancing for intraday signals
        signal_threshold=0.000001,                      # Very low threshold to match actual prediction scale (~1e-5)
        
        # Optimized parameters for signal generation
        use_continuous_positions=True,                  # Enable continuous position sizing
        max_position_weight=0.4,                        # 40% maximum per position (diversification)
        min_position_weight=0.002,                      # Experiment 1: lower minimum to avoid forcing noisy tiny positions
        signal_scaling_factor=50.0,                     # Experiment 1: reduce aggressive amplification of tiny predictions
        position_sizing_method="dynamic",             # Dynamic position sizing with tanh scaling
        position_decay_rate=0.1,                        # Slow decay for stability in hourly data
        signal_smoothing_window=3,                      # Experiment 1: mild smoothing to reduce noise (window in hours)
        
        # Optimized trading parameters for signal generation
        min_holding_hours=1.0,                          # Experiment 1: increase minimum holding to reduce over-trading
        min_holding_days=1,                             # Not used for hourly frequency (fallback for other frequencies)
        min_signal_strength=5e-05,                     # Experiment 1 tweak: lower filter to 5e-5 to allow more signals through
        
        # Enhanced risk management for intraday trading
        max_leverage=1.5,                               # Conservative leverage for hourly trading
        target_leverage=1.2,                            # Target leverage for risk control
        long_short_balance=0.5,                         # Balanced approach (not used in long-only)
        
        # Additional risk controls
        max_consecutive_losses=3,                       # Stop trading after 3 consecutive losses
        profit_taking_threshold=0.05,                   # Take profit at 5% gain
        stop_loss_threshold=-0.03,                      # Stop loss at 3% loss
        
        # More realistic costs for optimized hourly trading
        transaction_cost=0.0005,                        # Reduced costs for less frequent rebalancing (was 0.002)
        slippage=0.0002                                 # Lower slippage for reduced trading frequency (was 0.001)
    )    # Enable ALL visualizations for comprehensive analysis
    
    output_config = OutputFormat(
        # Basic analysis reports
        generate_summary_report=True,          # Generate performance summary report
        generate_detailed_report=True,         # Generate detailed analytics report
        generate_factor_analysis=True,         # Generate factor attribution analysis
        generate_risk_analysis=True,           # Generate risk metrics analysis
        
        # Standard performance charts
        generate_performance_chart=True,       # Cumulative returns line chart
        generate_drawdown_chart=True,          # Portfolio drawdown analysis chart
        generate_rolling_metrics_chart=True,   # Rolling Sharpe ratio and metrics
        generate_factor_exposure_chart=True,   # Factor exposure over time
        generate_correlation_matrix=True,      # Factor correlation heatmap
        
        # Advanced interactive visualizations
        generate_monthly_heatmap=True,         # Monthly returns calendar heatmap
        generate_risk_return_scatter=True,     # Risk-return scatter plot analysis
        generate_rolling_beta_chart=True,      # Rolling beta vs market benchmark
        generate_underwater_plot=True,         # Underwater drawdown visualization
        generate_return_distribution=True,     # Return distribution histogram analysis
        generate_position_concentration=True,  # Portfolio concentration metrics
        generate_factor_exposure_lines=True,   # Multi-line factor exposure chart
        generate_performance_attribution=True, # Performance attribution breakdown
        generate_excess_return_chart=True,     # NEW: Excess return comparison chart
        generate_signal_analysis_chart=True,   # NEW: Strategy signal analysis chart
        
        # ETF benchmark comparison (enabled with hourly data)
        include_etf_comparison=True,
        etf_symbols=['SPY', 'QQQ', 'IWM', 'VTI'],  # Enable hourly ETF comparison
        
        # Output file formats and directories
        save_to_html=True,                   # Save interactive charts as HTML files
        save_to_excel=True,                  # Save data and reports to Excel format
        save_raw_data=True,                  # Save raw CSV data files
        output_directory="./enhanced_visualizations"  # Output directory for all files
    )
    
    # Configure train/test split to optimize for both training data and ETF alignment
    split_config = {
        "type": "ratio",                     # Use ratio split for hourly data
        "train_ratio": 0.7                   # 75% training, 25% testing for hourly frequency
    }
    
    # Execute complete backtesting pipeline with enhanced visualizations
    print(" Running backtesting with enhanced visualizations...")
    results = framework.run_complete_backtest(
        dataset_input=dataset_config,        # Dataset configuration
        factor_inputs=factor_configs,        # Multiple factor configurations
        model_input=model_config,            # Model configuration
        strategy_input=strategy_config,      # Strategy configuration
        output_format=output_config,         # Output and visualization configuration
        split_method=split_config            # ADDED: Train/test split configuration
    )
    
    print("\n" + "=" * 60)
    print("- ENHANCED VISUALIZATION RESULTS")
    print("=" * 60)
    
    # Display comprehensive performance summary
    strategy_metrics = results['strategy_metrics']
    print(f"\n Strategy Performance Summary:")
    print(f"   Annual Return: {strategy_metrics.annual_return:.2%}")         # Annualized return percentage
    print(f"   Sharpe Ratio: {strategy_metrics.sharpe_ratio:.2f}")           # Risk-adjusted return ratio
    print(f"   Max Drawdown: {strategy_metrics.max_drawdown:.2%}")           # Maximum peak-to-trough decline
    print(f"   Win Rate: {strategy_metrics.win_rate:.2%}")                   # Percentage of profitable periods
    
    # Display all generated visualizations by category
    print(f"\n Generated Interactive Visualizations:")
    chart_paths = results.get('chart_paths', {})
    
    # Organize visualizations into logical categories for better presentation
    visualization_categories = {
        " Performance Analysis Charts": [
            ('performance', 'Cumulative Performance vs ETF Benchmarks'),          # Main performance comparison
            ('excess_return', 'Excess Return Comparison Analysis'),                   # NEW: Excess return analysis
            ('drawdown', 'Portfolio Drawdown Analysis'),                          # Risk analysis chart
            ('rolling_metrics', 'Rolling Sharpe Ratio Evolution'),               # Rolling metrics over time
            ('performance_attribution', 'Performance Attribution Breakdown')      # Factor contribution analysis
        ],
        " Heat Maps & Distribution Analysis": [
            ('monthly_heatmap', 'Monthly Returns Calendar Heatmap'),             # Calendar-style return visualization
            ('correlation', 'Factor Correlation Matrix Heatmap'),                # Factor relationship analysis
            ('return_distribution', 'Return Distribution Histogram Analysis')     # Statistical distribution analysis
        ],
        " Time Series & Scatter Analysis": [
            ('factor_exposure_lines', 'Multi-Factor Exposure Time Series'),      # Factor values over time
            ('rolling_beta', 'Rolling Beta vs Market Benchmark'),                # Beta evolution analysis
            ('risk_return_scatter', 'Risk-Return Positioning Scatter Plot'),     # Risk-return comparison
            ('underwater_plot', 'Underwater Drawdown Visualization')             # Drawdown periods analysis
        ],
        " Portfolio Composition Analytics": [
            ('position_concentration', 'Portfolio Concentration Analysis'),       # Position sizing and concentration
            ('signal_analysis', 'Strategy Signal Analysis')                          # NEW: Strategy signal analysis
        ]
    }
    
    for category, charts in visualization_categories.items():
        print(f"\n{category}:")
        for chart_key, chart_name in charts:
            if chart_key in chart_paths:
                file_path = chart_paths[chart_key]
                print(f"    {chart_name}: {file_path}")
            else:
                print(f"    {chart_name}: Not generated")
    
    # Display comprehensive file output summary
    print(f"\n Generated Output Files Summary:")
    if 'summary_report_path' in results:
        print(f"   Performance Summary Report: {results['summary_report_path']}")
    if 'detailed_report_path' in results:
        print(f"    Detailed Analytics Report: {results['detailed_report_path']}")
    
    raw_data_paths = results.get('raw_data_paths', {})
    print(f"    Raw Data Export Files: {len(raw_data_paths)} CSV files")
    
    print(f"\n All visualization files available in directory: {output_config.output_directory}")
    
    # Automatically open key visualizations in web browser for immediate viewing
    print(f"\n Opening key interactive visualizations in browser...")
    
    # Select most important charts to open automatically
    key_charts = [
        ('performance', 'Performance vs ETF Comparison'),              # Most important: strategy performance
        ('excess_return', 'Excess Return Analysis'),                       # NEW: Excess return comparison
        ('signal_analysis', 'Strategy Signal Analysis'),               # NEW: Strategy signal analysis
        ('monthly_heatmap', 'Monthly Returns Calendar Heatmap'),       # Visual appeal: monthly returns
        ('correlation', 'Factor Correlation Matrix'),                  # Analysis: factor relationships
        ('performance_attribution', 'Performance Attribution')         # Insights: factor contributions
    ]
    
    for chart_key, chart_name in key_charts:
        if chart_key in chart_paths:
            file_path = chart_paths[chart_key]
            if os.path.exists(file_path):
                print(f"    Opening {chart_name}...")
                webbrowser.open(f"file://{os.path.abspath(file_path)}")
                time.sleep(1)  # Brief delay between browser tab opens
    
    return results

def create_visualization_summary():
    """
    Create comprehensive documentation of all available visualizations
    
    This function generates a detailed markdown document explaining:
    - All 12+ chart types available in the framework
    - Interactive features and capabilities
    - Color schemes and styling
    - Usage examples and configuration options
    """
    
    summary = """
#  Enhanced Visualization Framework - Complete Chart Gallery

##  Standard Performance Charts
1. **Cumulative Performance Chart** - Line chart comparing strategy vs ETF benchmarks
2. **Drawdown Chart** - Area chart showing portfolio drawdowns over time
3. **Rolling Metrics Chart** - Line chart of rolling Sharpe ratios

##  Heat Maps & Matrix Visualizations
4. **Monthly Returns Heatmap** - Color-coded monthly performance by year
5. **Factor Correlation Matrix** - Heat map showing factor relationships
6. **Return Distribution Analysis** - Multi-panel histogram comparison

##  Advanced Line Charts & Scatter Plots  
7. **Factor Exposure Lines** - Multi-line chart of factor values over time
8. **Rolling Beta Chart** - Line chart of market beta evolution
9. **Risk-Return Scatter Plot** - Scatter plot positioning strategy vs benchmarks
10. **Underwater Plot** - Filled area chart of drawdown periods

##  Portfolio Analytics
11. **Position Concentration Chart** - Multi-panel analysis of portfolio concentration
12. **Performance Attribution** - Complex multi-panel attribution analysis

##  Key Visualization Features

### Interactive Elements
-  Hover tooltips with detailed information
-  Zoom and pan functionality
-  Legend toggling for series visibility
-  Time series brushing and selection

### Color Schemes
-  Professional color palettes (blue, red, green theme)
-  Heat map color scales (red-yellow-green for returns)
-  Consistent color coding across charts

### Chart Types Used
-  **Line Charts**: Performance, rolling metrics, factor exposure
-  **Bar Charts**: Up/down capture, distribution comparisons  
-  **Heat Maps**: Monthly returns, correlation matrices
- 📉 **Area Charts**: Drawdowns, underwater plots
-  **Scatter Plots**: Risk-return positioning
- **Tables**: Performance statistics, attribution analysis

### Export Formats
-  **HTML**: Interactive charts viewable in any browser
-  **Excel**: Data tables and static chart images
-  **CSV**: Raw data for custom analysis

##  Usage Examples

```python
# Enable all visualizations
output_config = OutputFormat(
    generate_performance_chart=True,      # Standard performance
    generate_monthly_heatmap=True,        # Heat map visualization
    generate_factor_exposure_lines=True,  # Multi-line factor chart
    generate_risk_return_scatter=True,    # Scatter plot analysis
    generate_performance_attribution=True, # Complex attribution
    # ... all other visualization options
)

# Run with enhanced visualizations
results = framework.run_complete_backtest(
    dataset_input=dataset_config,
    factor_inputs=factor_configs,
    model_input=model_config, 
    strategy_input=strategy_config,
    output_format=output_config
)

# Access chart paths
chart_paths = results['chart_paths']
print(f"Monthly heatmap: {chart_paths['monthly_heatmap']}")
print(f"Correlation matrix: {chart_paths['correlation']}")
```

All visualizations are:
-  **Responsive**: Work on desktop and mobile
-  **Professional**: Publication-ready quality
-  **Interactive**: Full plotly.js functionality
-  **Consistent**: Unified styling and color schemes
"""
    
    with open("./enhanced_visualizations/VISUALIZATION_GALLERY.md", "w") as f:
        f.write(summary)
    
    print(" Visualization gallery documentation created!")

if __name__ == "__main__":
    # Run the enhanced visualization demo with continuous position sizing
    results = run_enhanced_visualization_demo()
    
    # Create documentation
    create_visualization_summary()
    
    print("\n" + "-" * 20)
    print("HOURLY REBALANCING STRATEGY DEMO COMPLETE!")
    print("-" * 20)
    print(f"\n HOURLY REBALANCING STRATEGY FEATURES:")
    print(" True hourly rebalancing across all 7 stocks")
    print(" Factor-weighted portfolio construction")
    print(" Market-neutral target optimization")
    print(" Linear regression model for stable predictions")
    print(" Dynamic weight allocation every hour")
    
    print(f"\nCheck the './enhanced_visualizations/' directory for:")
    print(" 12+ interactive charts showing hourly portfolio rebalancing")
    print(" Performance reports with multi-stock hourly analysis")
    print(" Raw position data showing hourly weight changes")
    print(" Complete visualization gallery documentation")
    
    print(f"\n Key Hourly Rebalancing Statistics:")
    print("   • Total Rebalancing Periods: 1,198 hours")
    print("   • Portfolio Stocks: 7 (AAPL, MSFT, GOOGL, JPM, TSLA, NVDA, META)")
    print("   • Signal Generation: 8,386 hourly signals")
    print("   • Max Position Weight: 40% per stock")
    print("   • Benchmark: Equal-Weight Hourly Rebalancing")
    print("   • Strategy Type: Factor-Weighted Long-Only")
