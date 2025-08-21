"""
Synthetic Dataset Provider for Backtesting Framework
Handles data loading from CSV files and synthetic data generation
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Any
from data_interfaces import DatasetInput, DatasetInterface


class SyntheticDatasetProvider(DatasetInterface):
    """Synthetic dataset provider for demonstration"""
    
    def load_data(self, dataset_input: DatasetInput) -> pd.DataFrame:
        """
        Load market data for selected symbols and time period.
        If source_type is 'csv', load from CSV file; otherwise, generate synthetic data.
        """
        import os

        start_date = pd.to_datetime(dataset_input.start_date)
        end_date = pd.to_datetime(dataset_input.end_date)

        if dataset_input.source_type in ["csv", "csv_hourly"] and hasattr(dataset_input, "file_path"):
            # Check if file_path is a directory or a single file
            if os.path.isdir(dataset_input.file_path):
                # Determine file suffix based on source type
                file_suffix = "_hourly.csv" if dataset_input.source_type == "csv_hourly" else "_daily.csv"
                data_type = "hourly" if dataset_input.source_type == "csv_hourly" else "daily"
                
                print(f" Debug: Loading multiple {data_type} CSV files from directory: {dataset_input.file_path}")
                # Load multiple files from directory
                all_data = []
                
                for symbol in dataset_input.custom_symbols:
                    file_path = os.path.join(dataset_input.file_path, f"{symbol}{file_suffix}")
                    if os.path.exists(file_path):
                        print(f" Debug: Loading {symbol} {data_type} data from {file_path}")
                        
                        # Handle different column names for hourly vs daily data
                        if dataset_input.source_type == "csv_hourly":
                            df_symbol = pd.read_csv(file_path, parse_dates=['Datetime'])
                            df_symbol.columns = [col.lower() for col in df_symbol.columns]
                            # Rename datetime column to date for consistency
                            df_symbol = df_symbol.rename(columns={'datetime': 'date'})
                        else:
                            df_symbol = pd.read_csv(file_path, parse_dates=['Date'])
                            df_symbol.columns = [col.lower() for col in df_symbol.columns]
                        
                        df_symbol['date'] = pd.to_datetime(df_symbol['date'], utc=True).dt.tz_convert(None)
                        df_symbol['symbol'] = symbol
                        all_data.append(df_symbol)
                    else:
                        print(f"Warning: {data_type.title()} file not found for {symbol}: {file_path}")
                
                if not all_data:
                    raise ValueError(f"No {data_type} data files found in directory: {dataset_input.file_path}")
                
                # Combine all data
                df = pd.concat(all_data, ignore_index=True)
                print(f" Debug: Combined {data_type} data shape: {df.shape}, symbols: {df['symbol'].unique()}")
                
            elif os.path.exists(dataset_input.file_path):
                print(f" Debug: Loading single CSV file: {dataset_input.file_path}")
                # Load single file (legacy behavior)
                df = pd.read_csv(dataset_input.file_path, parse_dates=['Date'])
                print(f" Debug: Original shape: {df.shape}, columns: {df.columns.tolist()}")
                # Normalize all column names to lowercase
                df.columns = [col.lower() for col in df.columns]
                # Ensure Date column is tz-naive (handle tz-aware and tz-naive)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert(None)
                else:
                    raise ValueError(f"CSV file missing 'date' column: {dataset_input.file_path}. Columns found: {df.columns.tolist()}")
                # Add symbol column from filename if missing
                if 'symbol' not in df.columns:
                    import re
                    # Try to extract symbol from filename (e.g., .../AAPL_daily.csv)
                    # Use basename to get just the filename, then extract symbol before underscore
                    filename = os.path.basename(dataset_input.file_path)
                    match = re.search(r"([A-Za-z0-9]+)_", filename)
                    if match:
                        symbol = match.group(1)
                    else:
                        symbol = os.path.splitext(filename)[0]
                    df['symbol'] = symbol
                    print(f" Debug: Added symbol '{symbol}' from filename '{filename}'")
            else:
                raise ValueError(f"File path does not exist: {dataset_input.file_path}")
            
            print(f" Debug: Unique symbols in data: {df['symbol'].unique()}")
            start_date_naive = pd.to_datetime(start_date)
            end_date_naive = pd.to_datetime(end_date)
            print(f" Debug: Date range filtering: {start_date_naive} to {end_date_naive}")
            df = df[(df['date'] >= start_date_naive) & (df['date'] <= end_date_naive)]
            print(f" Debug: After date filtering: {df.shape}")
            if dataset_input.custom_symbols:
                print(f" Debug: Custom symbols requested: {dataset_input.custom_symbols}")
                df = df[df['symbol'].isin(dataset_input.custom_symbols)]
                print(f" Debug: After symbol filtering: {df.shape}")
            # Only keep business days if needed
            if getattr(dataset_input, "business_days_only", True):
                df = df[df['date'].dt.dayofweek < 5]
            print(f" Debug: Final shape: {df.shape}")
            return df.sort_values(['date', 'symbol']).reset_index(drop=True)

        # Generate synthetic data as fallback
        # Ensure dates are tz-naive to avoid conversion issues
        dates = pd.date_range(start_date, end_date, freq='D', tz=None)
        if dataset_input.custom_symbols:
            symbols = dataset_input.custom_symbols
        else:
            symbols = [f"STOCK_{i:03d}" for i in range(100)]  # 100 synthetic stocks

        data_rows = []
        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)
            initial_price = np.random.uniform(20, 200)
            returns = np.random.normal(0.0005, 0.02, len(dates))
            for i in range(1, len(returns)):
                returns[i] += 0.1 * returns[i-1]
            prices = initial_price * (1 + returns).cumprod()
            for i, (date, price) in enumerate(zip(dates, prices)):
                daily_vol = abs(returns[i]) * 2
                high = price * (1 + daily_vol/2)
                low = price * (1 - daily_vol/2)
                open_price = prices[i-1] if i > 0 else price
                close_price = price
                volume = np.random.lognormal(10, 1) * (1 + abs(returns[i]) * 5)
                data_rows.append({
                    'date': date,  # Use lowercase 'date'
                    'symbol': symbol,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': int(volume),
                    'amount': close_price * volume
                })
        df = pd.DataFrame(data_rows)
        if dataset_input.source_type != "synthetic_all_days":
            df = df[df['date'].dt.dayofweek < 5]
        return df.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality"""
        
        return {
            'total_rows': len(data),
            'symbols': data['symbol'].nunique(),
            'date_range': f"{data['date'].min()} to {data['date'].max()}",
            'missing_values': data.isnull().sum().sum(),
            'negative_prices': (data['close'] <= 0).sum(),
            'valid_ratio': 1.0 - (data.isnull().sum().sum() / len(data))
        }
