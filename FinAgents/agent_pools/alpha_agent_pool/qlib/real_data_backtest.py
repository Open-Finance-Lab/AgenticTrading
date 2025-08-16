"""
Real Data Backtesting with Actual Benchmark ETFs
Uses real data from qlib_data directory for proper benchmark comparison
"""

# Import real data support first to patch the framework
import real_data_support

from complete_framework import BacktestingFramework
from data_interfaces import DatasetInput, FactorInput, ModelInput, StrategyInput, OutputFormat
import pandas as pd
import os
import webbrowser
import time

def load_benchmark_data(data_dir, symbol, data_type="daily"):
    """
    Load real benchmark data from qlib_data directory
    
    Args:
        data_dir: Path to qlib_data directory
        symbol: ETF symbol (IBIT, FBTC, GBTC, SPY, QQQ, etc.)
        data_type: "daily" or "hourly" or "1min"
    
    Returns:
        pandas.DataFrame: Loaded benchmark data
    """
    
    if data_type == "1min":
        # For Bitcoin ETFs minute data
        file_path = os.path.join(data_dir, "bitcoin_etfs", f"{symbol}_1min_7d.csv")
    else:
        # For backup daily/hourly data
        file_path = os.path.join(data_dir, "etf_backup", f"{symbol}_{data_type}.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {symbol} {data_type} data: {len(df)} records from {file_path}")
        return df, file_path
    else:
        print(f"❌ File not found: {file_path}")
        return None, None

def run_btc_real_benchmark_backtest():
    """
    Run BTC backtesting with real Bitcoin ETF benchmarks (IBIT, FBTC, GBTC)
    """
    
    print("₿ Starting BTC Backtesting with Real Bitcoin ETF Benchmarks...")
    print("=" * 70)
    
    # Initialize framework
    framework = BacktestingFramework()
    
    # Path to qlib_data directory with real benchmark data
    qlib_data_dir = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data"
    
    # Load and verify Bitcoin ETF benchmark data
    print("\n📊 Loading Real Bitcoin ETF Benchmark Data...")
    bitcoin_etfs = ["IBIT", "FBTC", "GBTC"]
    
    for symbol in bitcoin_etfs:
        df, file_path = load_benchmark_data(qlib_data_dir, symbol, "1min")
        if df is not None:
            print(f"   📈 {symbol}: {len(df)} minute-level records")
            # Show date range
            if 'Date' in df.columns:
                df['Date'] = pd.to_Date(df['Date'])
                print(f"      📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Configure BTC dataset (using actual BTC minute data)
    dataset_config = DatasetInput(
        source_type="csv",
        file_path="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/data/btcusd_1-min_data.csv",
        start_date="2025-08-05",    # Last 7 days within BTC data range
        end_date="2025-08-12",      # Before 8.12.2025 as requested
        required_fields=["timestamp", "open", "high", "low", "close", "volume"],
        universe="custom_list",
        custom_symbols=["BTCUSD"],
        adjust_price=False,         # Crypto data doesn't need adjustment
        fill_method="ffill",
        min_periods=10              # Reduced minimum periods for short timeframe
    )
    
    # Configure factors for BTC analysis
    factor_configs = [
        FactorInput(
            factor_name="btc_momentum_30min",     # 30-minute momentum for BTC
            factor_type="alpha",
            calculation_method="expression",
            lookback_period=30                    # 30 minute lookback
        ),
        FactorInput(
            factor_name="btc_volatility_60min",   # 1-hour volatility
            factor_type="risk",
            calculation_method="expression",
            lookback_period=60                    # 60 minute lookback
        ),
        FactorInput(
            factor_name="btc_rsi_120min",         # 2-hour RSI
            factor_type="technical",
            calculation_method="expression",
            lookback_period=120                   # 120 minute lookback
        )
    ]
    
    # Configure model for BTC prediction
    model_config = ModelInput(
        model_name="btc_lgb_model",
        model_type="tree",
        implementation="lightgbm",
        model_class="LGBMRegressor",
        hyperparameters={
            "n_estimators": 50,                   # Reduced for small dataset
            "learning_rate": 0.1,                # Higher learning rate
            "max_depth": 3,                      # Shallower trees
            "random_state": 42
        }
    )
    
    # Configure strategy for BTC
    strategy_config = StrategyInput(
        strategy_name="btc_momentum_strategy",
        strategy_type="long_only",
        position_method="factor_weight",
        num_positions=1,                          # Single asset (BTC)
        rebalance_frequency="hourly"              # More frequent rebalancing for minute data
    )
    
    # Configure output with real Bitcoin ETF benchmarks
    output_config = OutputFormat(
        # Core reports and analysis
        generate_summary_report=True,
        generate_detailed_report=True,
        generate_factor_analysis=True,
        generate_risk_analysis=True,
        
        # Standard performance charts
        generate_performance_chart=True,
        generate_drawdown_chart=True,
        generate_rolling_metrics_chart=True,
        generate_factor_exposure_chart=True,
        generate_correlation_matrix=True,
        
        # Advanced visualizations
        generate_monthly_heatmap=True,
        generate_risk_return_scatter=True,
        generate_rolling_beta_chart=True,
        generate_underwater_plot=True,
        generate_return_distribution=True,
        generate_position_concentration=True,
        generate_factor_exposure_lines=True,
        generate_performance_attribution=True,
        
        # REAL Bitcoin ETF benchmark comparison
        include_etf_comparison=True,
        etf_symbols=["IBIT", "FBTC", "GBTC"],    # Real Bitcoin ETFs as benchmarks
        etf_data_source="qlib_data",             # Use real data from qlib_data directory
        etf_data_directory=qlib_data_dir,        # Specify data directory
        
        # Output formats
        save_to_html=True,
        save_to_excel=True,
        save_raw_data=True,
        output_directory="./btc_real_benchmark_results"
    )
    
    # Execute BTC backtesting with real benchmarks
    print("\n🚀 Running BTC backtesting with real Bitcoin ETF benchmarks...")
    try:
        results = framework.run_complete_backtest(
            dataset_input=dataset_config,
            factor_inputs=factor_configs,
            model_input=model_config,
            strategy_input=strategy_config,
            output_format=output_config
        )
        
        print("\n" + "=" * 70)
        print("₿ BTC BACKTESTING WITH REAL BENCHMARKS COMPLETE!")
        print("=" * 70)
        
        # Display results
        strategy_metrics = results['strategy_metrics']
        print(f"\n📊 BTC Strategy Performance vs Real Bitcoin ETFs:")
        print(f"   Annual Return: {strategy_metrics.annual_return:.2%}")
        print(f"   Sharpe Ratio: {strategy_metrics.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {strategy_metrics.max_drawdown:.2%}")
        print(f"   Win Rate: {strategy_metrics.win_rate:.2%}")
        
        # Display benchmark comparison info
        print(f"\n📈 Real Benchmark Comparison:")
        print(f"   IBIT (iShares Bitcoin Trust): Real minute-level data")
        print(f"   FBTC (Fidelity Wise Origin Bitcoin): Real minute-level data")
        print(f"   GBTC (Grayscale Bitcoin Trust): Real minute-level data")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in BTC backtesting: {e}")
        return None

def run_stock_real_benchmark_backtest():
    """
    Run stock backtesting with real traditional ETF benchmarks
    """
    
    print("\n📊 Starting Stock Backtesting with Real Traditional ETF Benchmarks...")
    print("=" * 70)
    
    # Initialize framework
    framework = BacktestingFramework()
    
    # Path to qlib_data directory
    qlib_data_dir = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data"
    
    # Load and verify traditional ETF benchmark data
    print("\n📊 Loading Real Traditional ETF Benchmark Data...")
    traditional_etfs = ["SPY", "QQQ", "IWM", "VTI", "VXUS"]
    
    for symbol in traditional_etfs:
        df, file_path = load_benchmark_data(qlib_data_dir, symbol, "daily")
        if df is not None:
            print(f"   📈 {symbol}: {len(df)} daily records")
    
    # Configure stock dataset (using AAPL as example)
    dataset_config = DatasetInput(
        source_type="csv",
        file_path="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/data/cache/AAPL_2022-06-30_2025-06-29_1d.csv",
        start_date="2024-05-15",    # 3-month period for common overlap
        end_date="2024-08-15",
        required_fields=["open", "high", "low", "close", "volume"],
        universe="custom_list",
        custom_symbols=["AAPL"],
        adjust_price=True,
        fill_method="ffill",
        min_periods=60              # 3 months minimum
    )
    
    # Configure factors for stock analysis
    factor_configs = [
        FactorInput(
            factor_name="momentum_10d",
            factor_type="alpha",
            calculation_method="expression",
            lookback_period=10
        ),
        FactorInput(
            factor_name="volatility_20d",
            factor_type="risk",
            calculation_method="expression",
            lookback_period=20
        ),
        FactorInput(
            factor_name="rsi_14d",
            factor_type="technical",
            calculation_method="expression",
            lookback_period=14
        )
    ]
    
    # Configure model
    model_config = ModelInput(
        model_name="stock_lgb_model",
        model_type="tree",
        implementation="lightgbm",
        model_class="LGBMRegressor",
        hyperparameters={
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 4,
            "random_state": 42
        }
    )
    
    # Configure strategy
    strategy_config = StrategyInput(
        strategy_name="stock_factor_strategy",
        strategy_type="long_only",
        position_method="factor_weight",
        num_positions=1,
        rebalance_frequency="weekly"
    )
    
    # Configure output with real traditional ETF benchmarks
    output_config = OutputFormat(
        # Core reports
        generate_summary_report=True,
        generate_detailed_report=True,
        generate_factor_analysis=True,
        generate_risk_analysis=True,
        
        # All visualizations
        generate_performance_chart=True,
        generate_drawdown_chart=True,
        generate_rolling_metrics_chart=True,
        generate_factor_exposure_chart=True,
        generate_correlation_matrix=True,
        generate_monthly_heatmap=True,
        generate_risk_return_scatter=True,
        generate_rolling_beta_chart=True,
        generate_underwater_plot=True,
        generate_return_distribution=True,
        generate_position_concentration=True,
        generate_factor_exposure_lines=True,
        generate_performance_attribution=True,
        
        # REAL traditional ETF benchmark comparison
        include_etf_comparison=True,
        etf_symbols=["SPY", "QQQ", "IWM", "VTI", "VXUS"],  # Real traditional ETFs
        etf_data_source="qlib_data",
        etf_data_directory=qlib_data_dir,
        
        # Output formats
        save_to_html=True,
        save_to_excel=True,
        save_raw_data=True,
        output_directory="./stock_real_benchmark_results"
    )
    
    # Execute stock backtesting with real benchmarks
    print("\n🚀 Running stock backtesting with real traditional ETF benchmarks...")
    try:
        results = framework.run_complete_backtest(
            dataset_input=dataset_config,
            factor_inputs=factor_configs,
            model_input=model_config,
            strategy_input=strategy_config,
            output_format=output_config
        )
        
        print("\n" + "=" * 70)
        print("📊 STOCK BACKTESTING WITH REAL BENCHMARKS COMPLETE!")
        print("=" * 70)
        
        # Display results
        strategy_metrics = results['strategy_metrics']
        print(f"\n📊 Stock Strategy Performance vs Real Traditional ETFs:")
        print(f"   Annual Return: {strategy_metrics.annual_return:.2%}")
        print(f"   Sharpe Ratio: {strategy_metrics.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {strategy_metrics.max_drawdown:.2%}")
        print(f"   Win Rate: {strategy_metrics.win_rate:.2%}")
        
        # Display benchmark comparison info
        print(f"\n📈 Real Benchmark Comparison:")
        print(f"   SPY (S&P 500 ETF): Real daily data")
        print(f"   QQQ (Nasdaq 100 ETF): Real daily data")
        print(f"   IWM (Russell 2000 ETF): Real daily data")
        print(f"   VTI (Total Stock Market ETF): Real daily data")
        print(f"   VXUS (International ETF): Real daily data")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in stock backtesting: {e}")
        return None

def open_results_in_browser(results_dir):
    """Open key visualization results in browser"""
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Look for key HTML files
    html_files = []
    for file in os.listdir(results_dir):
        if file.endswith('.html'):
            html_files.append(os.path.join(results_dir, file))
    
    print(f"\n🌐 Opening {len(html_files)} visualization files in browser...")
    for html_file in html_files[:5]:  # Open first 5 files to avoid overwhelming
        if os.path.exists(html_file):
            print(f"   🔗 Opening {os.path.basename(html_file)}...")
            webbrowser.open(f"file://{os.path.abspath(html_file)}")
            time.sleep(1)

def main():
    """
    Main execution function for real benchmark backtesting
    """
    
    print("🚀 REAL BENCHMARK DATA BACKTESTING")
    print("=" * 50)
    print("Using actual ETF data from qlib_data directory")
    print("BTC Benchmarks: IBIT, FBTC, GBTC")
    print("Stock Benchmarks: SPY, QQQ, IWM, VTI, VXUS")
    print("=" * 50)
    
    # Run BTC backtesting with real Bitcoin ETF benchmarks
    print("\n🔥 Part 1: BTC Analysis")
    btc_results = run_btc_real_benchmark_backtest()
    
    # Run stock backtesting with real traditional ETF benchmarks  
    print("\n🔥 Part 2: Stock Analysis")
    stock_results = run_stock_real_benchmark_backtest()
    
    # Open results in browser
    if btc_results:
        print("\n🌐 Opening BTC results...")
        open_results_in_browser("./btc_real_benchmark_results")
    
    if stock_results:
        print("\n🌐 Opening Stock results...")
        open_results_in_browser("./stock_real_benchmark_results")
    
    print("\n" + "🎉" * 30)
    print("REAL BENCHMARK BACKTESTING COMPLETE!")
    print("🎉" * 30)
    print("\n📁 Results available in:")
    print("   ₿ BTC: ./btc_real_benchmark_results/")
    print("   📊 Stock: ./stock_real_benchmark_results/")
    print("\n✅ All comparisons use REAL ETF data from qlib_data directory!")

if __name__ == "__main__":
    main()
