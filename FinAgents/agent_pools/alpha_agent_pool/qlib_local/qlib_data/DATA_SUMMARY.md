#  ETF Data Download Summary

##  Data Sources Overview

### 1. Traditional ETFs (Qlib Source)
- **Symbols**: SPY, QQQ, IWM, VTI, VXUS
- **Frequency**: 1-minute bars
- **Source**: Qlib built-in downloader
- **Location**: /FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/us_minute/
- **Coverage**: Comprehensive US market data

### 2. Bitcoin ETFs (yfinance Source)  
- **Symbols**: IBIT, FBTC, GBTC
- **Frequency**: 1-minute and 5-minute bars
- **Source**: Yahoo Finance via yfinance
- **Location**: /FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/bitcoin_etfs/
- **Coverage**: Recent data with resampling for minute-level

### 3. Backup Data (yfinance Source)
- **Symbols**: All ETFs (Traditional + Bitcoin)
- **Frequency**: Daily and hourly bars
- **Source**: Yahoo Finance via yfinance  
- **Location**: /FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/etf_backup/
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
    file_path="/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/bitcoin_etfs/IBIT_1min_resampled.csv",
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
