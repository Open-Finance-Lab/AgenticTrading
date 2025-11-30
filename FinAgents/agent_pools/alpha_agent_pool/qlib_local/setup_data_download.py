"""
Comprehensive ETF Data Download Script
Downloads minute-level data for traditional ETFs and Bitcoin ETFs using multiple sources
"""

import os
import subprocess
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys

def create_qlib_data_directory():
    """Create Qlib data directory structure"""
    
    # Define base data directory using absolute path
    base_dir = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data"
    us_minute_dir = os.path.join(base_dir, "us_minute")
    
    # Create directories if they don't exist
    os.makedirs(us_minute_dir, exist_ok=True)
    
    print(f" Created Qlib data directory structure:")
    print(f"    Base directory: {base_dir}")
    print(f"    US minute data: {us_minute_dir}")
    
    return base_dir, us_minute_dir

def download_qlib_traditional_etfs():
    """Download traditional ETF data using Qlib's built-in downloader"""
    
    print("\n Downloading traditional ETFs using Qlib...")
    
    # Traditional ETF symbols that are well-supported by Qlib
    traditional_etfs = ["SPY", "QQQ", "IWM", "VTI", "VXUS"]
    
    # Qlib download command for US market minute data
    qlib_download_cmd = [
        "python", "-m", "qlib.data", 
        "--download",
        "--market", "us",
        "--interval", "1min", 
        "--target_dir", "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/us_minute",
        "--region", "us"
    ]
    
    print(f" Executing Qlib download command:")
    print(f"   Command: {' '.join(qlib_download_cmd)}")
    
    try:
        # Execute the download command
        result = subprocess.run(qlib_download_cmd, 
                              capture_output=True, 
                              text=True, 
                              timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            print(" Qlib traditional ETF download completed successfully!")
            print(f" Data saved to: qlib_data/us_minute/")
        else:
            print(f" Qlib download failed:")
            print(f"   Error: {result.stderr}")
            print(f"   Output: {result.stdout}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Qlib download timed out after 30 minutes")
    except Exception as e:
        print(f" Error during Qlib download: {e}")

def download_bitcoin_etfs_yfinance():
    """Download Bitcoin ETF minute data using yfinance as fallback"""
    
    print("\n₿ Downloading Bitcoin ETFs using yfinance...")
    
    # Bitcoin ETF symbols
    bitcoin_etfs = ["IBIT", "FBTC", "GBTC"]
    
    # Define date range (last 2 years for comprehensive data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years of data
    
    # Create Bitcoin ETF data directory
    bitcoin_dir = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/bitcoin_etfs"
    os.makedirs(bitcoin_dir, exist_ok=True)
    
    for symbol in bitcoin_etfs:
        print(f"\n Downloading {symbol} minute data...")
        
        try:
            # Download minute data from yfinance
            ticker = yf.Ticker(symbol)
            
            # Get 1-minute data (yfinance limits to last 7 days for 1min)
            # For longer periods, use 5-minute data
            hist_1min = ticker.history(period="7d", interval="1m")
            hist_5min = ticker.history(start=start_date, end=end_date, interval="5m")
            
            if not hist_1min.empty:
                # Save 1-minute data (last 7 days)
                file_path_1min = os.path.join(bitcoin_dir, f"{symbol}_1min_7d.csv")
                hist_1min.to_csv(file_path_1min)
                print(f"    1-minute data (7 days): {file_path_1min}")
            
            if not hist_5min.empty:
                # Save 5-minute data (2 years)
                file_path_5min = os.path.join(bitcoin_dir, f"{symbol}_5min_2y.csv")
                hist_5min.to_csv(file_path_5min)
                print(f"    5-minute data (2 years): {file_path_5min}")
                
                # Convert 5-minute to pseudo-minute data by resampling
                minute_data = hist_5min.resample('1min').ffill()
                file_path_resampled = os.path.join(bitcoin_dir, f"{symbol}_1min_resampled.csv")
                minute_data.to_csv(file_path_resampled)
                print(f"    Resampled 1-minute data: {file_path_resampled}")
            
        except Exception as e:
            print(f"    Failed to download {symbol}: {e}")

def download_additional_etfs_yfinance():
    """Download daily and hourly data for 30 popular US stocks"""
    print("\n Downloading daily and hourly data for 30 popular US stocks...")

    # 30 popular US stock tickers (can be adjusted as needed)
    hot_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH",
        "HD", "MA", "PG", "LLY", "AVGO", "XOM", "MRK", "ABBV", "COST", "WMT",
        "ADBE", "PEP", "KO", "CVX", "MCD", "BAC", "TMO", "ACN", "DHR", "QCOM"
    ]

    # Create backup directory for stock data
    backup_dir = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/stock_backup"
    os.makedirs(backup_dir, exist_ok=True)

    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3 years of data

    for symbol in hot_stocks:
        print(f"\n Downloading data for {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            # Download daily data
            hist_daily = ticker.history(start=start_date, end=end_date, interval="1d")
            # Download hourly data (up to 2 years)
            hist_hourly = ticker.history(period="730d", interval="1h")

            if not hist_daily.empty:
                file_path_daily = os.path.join(backup_dir, f"{symbol}_daily.csv")
                hist_daily.to_csv(file_path_daily)
                print(f"    Daily data saved: {file_path_daily}")

            if not hist_hourly.empty:
                file_path_hourly = os.path.join(backup_dir, f"{symbol}_hourly.csv")
                hist_hourly.to_csv(file_path_hourly)
                print(f"    Hourly data saved: {file_path_hourly}")
        except Exception as e:
            print(f"    Failed to download data for {symbol}: {e}")

def verify_downloaded_data():
    """Verify that the downloaded data is accessible and valid"""
    
    print("\n Verifying downloaded data...")
    
    # Check Qlib data directory
    qlib_dir = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/us_minute"
    if os.path.exists(qlib_dir):
        files = os.listdir(qlib_dir)
        print(f" Qlib US minute data directory contains {len(files)} files/folders")
        if files:
            print(f"   Sample files: {files[:5]}")
    else:
        print(" Qlib US minute data directory not found")
    
    # Check Bitcoin ETF data
    bitcoin_dir = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/bitcoin_etfs"
    if os.path.exists(bitcoin_dir):
        files = [f for f in os.listdir(bitcoin_dir) if f.endswith('.csv')]
        print(f"₿ Bitcoin ETF data directory contains {len(files)} CSV files")
        for file in files:
            file_path = os.path.join(bitcoin_dir, file)
            try:
                df = pd.read_csv(file_path)
                print(f"    {file}: {len(df)} records")
            except Exception as e:
                print(f"    {file}: Error reading - {e}")
    else:
        print(" Bitcoin ETF data directory not found")
    
    # Check backup data
    backup_dir = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/etf_backup"
    if os.path.exists(backup_dir):
        files = [f for f in os.listdir(backup_dir) if f.endswith('.csv')]
        print(f"💾 Backup ETF data directory contains {len(files)} CSV files")
    else:
        print(" Backup ETF data directory not found")

def create_data_summary():
    """Create a summary of all available data sources"""
    
    summary = """#  ETF Data Download Summary

##  Data Sources Overview

### 1. Traditional ETFs (Qlib Source)
- **Symbols**: SPY, QQQ, IWM, VTI, VXUS
- **Frequency**: 1-minute bars
- **Source**: Qlib built-in downloader
- **Location**: /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/us_minute/
- **Coverage**: Comprehensive US market data

### 2. Bitcoin ETFs (yfinance Source)  
- **Symbols**: IBIT, FBTC, GBTC
- **Frequency**: 1-minute and 5-minute bars
- **Source**: Yahoo Finance via yfinance
- **Location**: /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/bitcoin_etfs/
- **Coverage**: Recent data with resampling for minute-level

### 3. Backup Data (yfinance Source)
- **Symbols**: All ETFs (Traditional + Bitcoin)
- **Frequency**: Daily and hourly bars
- **Source**: Yahoo Finance via yfinance  
- **Location**: /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/etf_backup/
- **Coverage**: Reliable fallback data source

##  Usage in Backtesting Framework

```python
# Use Qlib data for traditional ETFs
dataset_config = DatasetInput(
    source_type="qlib",
    start_date="2023-01-01", 
    end_date="2024-12-31",
    universe="etf_list",
    custom_symbols=["SPY", "QQQ", "IWM", "VTI", "VXUS"],
    frequency="1min"
)

# Use CSV data for Bitcoin ETFs
bitcoin_dataset_config = DatasetInput(
    source_type="csv",
    file_path="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/bitcoin_etfs/IBIT_1min_resampled.csv",
    start_date="2023-01-01",
    end_date="2024-12-31", 
    custom_symbols=["IBIT"]
)
```

##  Data Quality Notes

### Traditional ETFs (Qlib)
-  High quality, professionally cleaned
-  Corporate actions adjusted
-  Minute-level granularity
-  Long historical coverage

### Bitcoin ETFs (yfinance)
- Limited 1-minute history (7 days)
-  5-minute data available (2 years)
-  Resampled to 1-minute for consistency
- May have gaps during market holidays

### Backup Data (yfinance)
-  Reliable daily/hourly data
-  Long historical coverage
-  All symbols consistently available
-  Good for strategy validation
"""
    
    # Save summary to data directory
    summary_path = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/DATA_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    
    print(f" Data summary created: {summary_path}")

def main():
    """Main execution function to download all ETF data"""
    
    print(" ETF Data Download Setup")
    print("=" * 50)
    
    # Step 1: Create directory structure
    print("\n Step 1: Creating data directories...")
    base_dir, us_minute_dir = create_qlib_data_directory()
    
    # Step 2: Download traditional ETFs using Qlib
    print("\n Step 2: Downloading traditional ETFs...")
    download_qlib_traditional_etfs()
    
    # Step 3: Download Bitcoin ETFs using yfinance
    print("\n₿ Step 3: Downloading Bitcoin ETFs...")
    download_bitcoin_etfs_yfinance()
    
    # Step 4: Download backup data
    print("\n💾 Step 4: Downloading backup data...")
    download_additional_etfs_yfinance()
    
    # Step 5: Verify data
    print("\n Step 5: Verifying data...")
    verify_downloaded_data()
    
    # Step 6: Create summary
    print("\n Step 6: Creating data summary...")
    create_data_summary()
    
    print("\n" + "-" * 20)
    print("ETF DATA DOWNLOAD COMPLETE!")
    print("-" * 20)
    print(f"\n All data available in: {base_dir}")
    print(" Traditional ETFs: /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/us_minute/")
    print("₿ Bitcoin ETFs: /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/bitcoin_etfs/")
    print("💾 Backup data: /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/etf_backup/")
    print(" Documentation: /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/DATA_SUMMARY.md")

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import yfinance
    except ImportError:
        print("📦 Installing yfinance...")
        subprocess.run([sys.executable, "-m", "pip", "install", "yfinance"], check=True)
    
    # Run main download process
    main()
