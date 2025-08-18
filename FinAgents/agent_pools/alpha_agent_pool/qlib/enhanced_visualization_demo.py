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
    
    print("🎨 Starting Enhanced Visualization Demo...")
    print("=" * 60)
    
    # Initialize the backtesting framework
    framework = BacktestingFramework()
    
    # Configure dataset input - using AAPL CSV data for demonstration
    dataset_config = DatasetInput(
        source_type="csv",                    # Data source type: CSV file
        file_path="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/stock_backup/AAPL_daily.csv",
        start_date="2022-06-30",             # Backtest start date
        end_date="2025-06-29",               # Backtest end date
        required_fields=["open", "high", "low", "close", "volume"],  # Required OHLCV fields
        universe="custom_list",              # Universe type: custom symbol list
        custom_symbols=["AAPL"],            # Single stock for demonstration
        adjust_price=True,                   # Apply price adjustments for splits/dividends
        fill_method="ffill",                 # Forward fill missing data
        min_periods=252                      # Minimum periods required (1 trading year)
    )
    
    # Configure multiple factors for comprehensive analysis
    factor_configs = [
        FactorInput(
            factor_name="momentum_5d",        # 5-day momentum factor
            factor_type="alpha",              # Alpha-generating factor type
            calculation_method="expression",   # Expression-based calculation
            lookback_period=5                 # 5-day lookback window
        ),
        FactorInput(
            factor_name="momentum_20d",       # 20-day momentum factor
            factor_type="alpha",              # Alpha-generating factor type
            calculation_method="expression",   # Expression-based calculation
            lookback_period=20                # 20-day lookback window
        ),
        FactorInput(
            factor_name="volatility_10d",     # 10-day volatility factor
            factor_type="risk",               # Risk-related factor type
            calculation_method="expression",   # Expression-based calculation
            lookback_period=10                # 10-day lookback window
        ),
        FactorInput(
            factor_name="volatility_30d",     # 30-day volatility factor
            factor_type="risk",               # Risk-related factor type
            calculation_method="expression",   # Expression-based calculation
            lookback_period=30                # 30-day lookback window
        ),
        FactorInput(
            factor_name="rsi_14d",            # 14-day RSI technical indicator
            factor_type="technical",          # Technical indicator factor type
            calculation_method="expression",   # Expression-based calculation
            lookback_period=14                # 14-day RSI standard period
        )
    ]
    
    # Configure LightGBM model for factor prediction
    model_config = ModelInput(
        model_name="enhanced_lgb_model",     # Model identifier name
        model_type="tree",                   # Tree-based model type
        implementation="lightgbm",           # Use LightGBM implementation
        model_class="LGBMRegressor",         # Regression model class
        hyperparameters={                    # Model hyperparameters
            "n_estimators": 100,             # Number of boosting rounds
            "learning_rate": 0.02,           # Learning rate for gradient boosting
            "max_depth": 4,                  # Maximum tree depth
            "random_state": 2025               # Random seed for reproducibility
        }
    )
    
    # Configure trading strategy parameters
    strategy_config = StrategyInput(
        strategy_name="enhanced_factor_strategy",  # Strategy identifier name
        strategy_type="long_short",                # CHANGED: Use long_short strategy
        position_method="factor_weight",           # Factor-weighted position sizing
        num_positions=1,                           # Single stock positions
        rebalance_frequency="daily",               # CHANGED: Daily rebalancing for more signal changes
        signal_threshold=0.002                     # NEW: Signal threshold to avoid frequent trading
    )
    
    # Enable ALL visualizations for comprehensive analysis
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
        
        # ETF benchmark comparison (Traditional )
        include_etf_comparison=True,
        etf_symbols=["SPY", "QQQ", "IWM", "VTI"],  # Traditional ETFs 
        
        # Output file formats and directories
        save_to_html=True,                   # Save interactive charts as HTML files
        save_to_excel=True,                  # Save data and reports to Excel format
        save_raw_data=True,                  # Save raw CSV data files
        output_directory="./enhanced_visualizations"  # Output directory for all files
    )
    
    # Execute complete backtesting pipeline with enhanced visualizations
    print("🚀 Running backtesting with enhanced visualizations...")
    results = framework.run_complete_backtest(
        dataset_input=dataset_config,        # Dataset configuration
        factor_inputs=factor_configs,        # Multiple factor configurations
        model_input=model_config,            # Model configuration
        strategy_input=strategy_config,      # Strategy configuration
        output_format=output_config          # Output and visualization configuration
    )
    
    print("\n" + "=" * 60)
    print("🎉 ENHANCED VISUALIZATION RESULTS")
    print("=" * 60)
    
    # Display comprehensive performance summary
    strategy_metrics = results['strategy_metrics']
    print(f"\n📊 Strategy Performance Summary:")
    print(f"   Annual Return: {strategy_metrics.annual_return:.2%}")         # Annualized return percentage
    print(f"   Sharpe Ratio: {strategy_metrics.sharpe_ratio:.2f}")           # Risk-adjusted return ratio
    print(f"   Max Drawdown: {strategy_metrics.max_drawdown:.2%}")           # Maximum peak-to-trough decline
    print(f"   Win Rate: {strategy_metrics.win_rate:.2%}")                   # Percentage of profitable periods
    
    # Display all generated visualizations by category
    print(f"\n📈 Generated Interactive Visualizations:")
    chart_paths = results.get('chart_paths', {})
    
    # Organize visualizations into logical categories for better presentation
    visualization_categories = {
        "📊 Performance Analysis Charts": [
            ('performance', 'Cumulative Performance vs ETF Benchmarks'),          # Main performance comparison
            ('excess_return', 'Excess Return Comparison Analysis'),                   # NEW: Excess return analysis
            ('drawdown', 'Portfolio Drawdown Analysis'),                          # Risk analysis chart
            ('rolling_metrics', 'Rolling Sharpe Ratio Evolution'),               # Rolling metrics over time
            ('performance_attribution', 'Performance Attribution Breakdown')      # Factor contribution analysis
        ],
        "🔥 Heat Maps & Distribution Analysis": [
            ('monthly_heatmap', 'Monthly Returns Calendar Heatmap'),             # Calendar-style return visualization
            ('correlation', 'Factor Correlation Matrix Heatmap'),                # Factor relationship analysis
            ('return_distribution', 'Return Distribution Histogram Analysis')     # Statistical distribution analysis
        ],
        "📈 Time Series & Scatter Analysis": [
            ('factor_exposure_lines', 'Multi-Factor Exposure Time Series'),      # Factor values over time
            ('rolling_beta', 'Rolling Beta vs Market Benchmark'),                # Beta evolution analysis
            ('risk_return_scatter', 'Risk-Return Positioning Scatter Plot'),     # Risk-return comparison
            ('underwater_plot', 'Underwater Drawdown Visualization')             # Drawdown periods analysis
        ],
        "📊 Portfolio Composition Analytics": [
            ('position_concentration', 'Portfolio Concentration Analysis'),       # Position sizing and concentration
            ('signal_analysis', 'Strategy Signal Analysis')                          # NEW: Strategy signal analysis
        ]
    }
    
    for category, charts in visualization_categories.items():
        print(f"\n{category}:")
        for chart_key, chart_name in charts:
            if chart_key in chart_paths:
                file_path = chart_paths[chart_key]
                print(f"   ✅ {chart_name}: {file_path}")
            else:
                print(f"   ❌ {chart_name}: Not generated")
    
    # Display comprehensive file output summary
    print(f"\n📁 Generated Output Files Summary:")
    if 'summary_report_path' in results:
        print(f"   📋 Performance Summary Report: {results['summary_report_path']}")
    if 'detailed_report_path' in results:
        print(f"   📊 Detailed Analytics Report: {results['detailed_report_path']}")
    
    raw_data_paths = results.get('raw_data_paths', {})
    print(f"   📄 Raw Data Export Files: {len(raw_data_paths)} CSV files")
    
    print(f"\n🎯 All visualization files available in directory: {output_config.output_directory}")
    
    # Automatically open key visualizations in web browser for immediate viewing
    print(f"\n🌐 Opening key interactive visualizations in browser...")
    
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
                print(f"   🔗 Opening {chart_name}...")
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
# 🎨 Enhanced Visualization Framework - Complete Chart Gallery

## 📊 Standard Performance Charts
1. **Cumulative Performance Chart** - Line chart comparing strategy vs ETF benchmarks
2. **Drawdown Chart** - Area chart showing portfolio drawdowns over time
3. **Rolling Metrics Chart** - Line chart of rolling Sharpe ratios

## 🔥 Heat Maps & Matrix Visualizations
4. **Monthly Returns Heatmap** - Color-coded monthly performance by year
5. **Factor Correlation Matrix** - Heat map showing factor relationships
6. **Return Distribution Analysis** - Multi-panel histogram comparison

## 📈 Advanced Line Charts & Scatter Plots  
7. **Factor Exposure Lines** - Multi-line chart of factor values over time
8. **Rolling Beta Chart** - Line chart of market beta evolution
9. **Risk-Return Scatter Plot** - Scatter plot positioning strategy vs benchmarks
10. **Underwater Plot** - Filled area chart of drawdown periods

## 📊 Portfolio Analytics
11. **Position Concentration Chart** - Multi-panel analysis of portfolio concentration
12. **Performance Attribution** - Complex multi-panel attribution analysis

## 🎯 Key Visualization Features

### Interactive Elements
- ✅ Hover tooltips with detailed information
- ✅ Zoom and pan functionality
- ✅ Legend toggling for series visibility
- ✅ Time series brushing and selection

### Color Schemes
- ✅ Professional color palettes (blue, red, green theme)
- ✅ Heat map color scales (red-yellow-green for returns)
- ✅ Consistent color coding across charts

### Chart Types Used
- 📈 **Line Charts**: Performance, rolling metrics, factor exposure
- 📊 **Bar Charts**: Up/down capture, distribution comparisons  
- 🔥 **Heat Maps**: Monthly returns, correlation matrices
- 📉 **Area Charts**: Drawdowns, underwater plots
- 🎯 **Scatter Plots**: Risk-return positioning
- 📋 **Tables**: Performance statistics, attribution analysis

### Export Formats
- 🌐 **HTML**: Interactive charts viewable in any browser
- 📊 **Excel**: Data tables and static chart images
- 📄 **CSV**: Raw data for custom analysis

## 🚀 Usage Examples

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
- 📱 **Responsive**: Work on desktop and mobile
- 🎨 **Professional**: Publication-ready quality
- ⚡ **Interactive**: Full plotly.js functionality
- 🔄 **Consistent**: Unified styling and color schemes
"""
    
    with open("./enhanced_visualizations/VISUALIZATION_GALLERY.md", "w") as f:
        f.write(summary)
    
    print("📚 Visualization gallery documentation created!")

if __name__ == "__main__":
    # Run the enhanced visualization demo
    results = run_enhanced_visualization_demo()
    
    # Create documentation
    create_visualization_summary()
    
    print("\n" + "🎉" * 20)
    print("ENHANCED VISUALIZATION DEMO COMPLETE!")
    print("🎉" * 20)
    print(f"\nCheck the './enhanced_visualizations/' directory for:")
    print("📊 12+ interactive charts and visualizations")
    print("📋 Comprehensive reports with insights")
    print("📄 Raw data exports for further analysis")
    print("📚 Complete visualization gallery documentation")
