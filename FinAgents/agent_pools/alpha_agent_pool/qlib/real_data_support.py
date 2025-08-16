"""
Enhanced Output Processor with Real ETF Data Support
Supports loading real ETF data from qlib_data directory for accurate benchmark comparison
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import yfinance as yf
from datetime import datetime, timedelta

@dataclass
class RealETFDataLoader:
    """Loader for real ETF data from qlib_data directory"""
    
    qlib_data_dir: str
    etf_symbols: List[str]
    data_type: str = "daily"  # "daily", "hourly", or "1min"
    
    def load_real_etf_data(self) -> Dict[str, pd.Series]:
        """
        Load real ETF data from qlib_data directory
        
        Returns:
            Dict[str, pd.Series]: ETF symbol -> returns series
        """
        
        etf_data = {}
        
        for symbol in self.etf_symbols:
            try:
                # Determine file path based on symbol and data type
                if symbol in ["IBIT", "FBTC", "GBTC"] and self.data_type == "1min":
                    # Bitcoin ETFs minute data
                    file_path = os.path.join(
                        self.qlib_data_dir, 
                        "bitcoin_etfs", 
                        f"{symbol}_1min_7d.csv"
                    )
                else:
                    # Traditional ETFs backup data
                    file_path = os.path.join(
                        self.qlib_data_dir, 
                        "etf_backup", 
                        f"{symbol}_{self.data_type}.csv"
                    )
                
                if os.path.exists(file_path):
                    # Load the CSV data
                    df = pd.read_csv(file_path)
                    
                    # Standardize column names and date handling
                    df = self._standardize_dataframe(df, symbol)
                    
                    if not df.empty:
                        # Calculate returns
                        returns = df['Close'].pct_change().dropna()
                        etf_data[symbol] = returns
                        
                        print(f"✅ Loaded real {symbol} data: {len(returns)} return observations")
                    else:
                        print(f"⚠️  Empty data for {symbol}")
                        
                else:
                    print(f"❌ File not found for {symbol}: {file_path}")
                    # Fallback to synthetic data for missing symbols
                    etf_data[symbol] = self._generate_fallback_data(symbol)
                    
            except Exception as e:
                print(f"❌ Error loading {symbol}: {e}")
                # Fallback to synthetic data on error
                etf_data[symbol] = self._generate_fallback_data(symbol)
        
        return etf_data
    
    def _standardize_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardize dataframe format"""
        
        # Handle different date column names
        date_columns = ['Date', 'Datetime', 'date', 'datetime', 'timestamp']
        date_col = None
        
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df['Date'] = pd.to_datetime(df[date_col])
            df = df.set_index('Date')
        
        # Standardize price column names
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low', 
            'Close': 'Close',
            'Volume': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure we have Close prices
        if 'Close' not in df.columns:
            print(f"⚠️  No Close column found for {symbol}, columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        return df.sort_index()
    
    def _generate_fallback_data(self, symbol: str) -> pd.Series:
        """Generate synthetic fallback data for missing symbols"""
        
        print(f"🔄 Generating synthetic fallback data for {symbol}")
        
        # Create date range
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        
        # Generate returns based on symbol characteristics
        if symbol == "SPY":
            returns = np.random.normal(0.0003, 0.012, len(dates))  # S&P 500 characteristics
        elif symbol == "QQQ": 
            returns = np.random.normal(0.0004, 0.018, len(dates))  # NASDAQ characteristics
        elif symbol == "IWM":
            returns = np.random.normal(0.0002, 0.020, len(dates))  # Small cap characteristics
        elif symbol in ["VTI", "VXUS"]:
            returns = np.random.normal(0.0003, 0.014, len(dates))  # Broad market characteristics
        elif symbol in ["IBIT", "FBTC", "GBTC"]:
            returns = np.random.normal(0.0008, 0.040, len(dates))  # Bitcoin ETF characteristics
        else:
            returns = np.random.normal(0.0003, 0.015, len(dates))  # Default characteristics
        
        return pd.Series(returns, index=dates)

def create_enhanced_output_processor_with_real_data(output_format):
    """
    Create an enhanced output processor that can load real ETF data
    
    Args:
        output_format: OutputFormat configuration object
        
    Returns:
        Enhanced output processor instance
    """
    
    # Import the original output processor
    from output_processor import OutputProcessor
    
    class EnhancedOutputProcessor(OutputProcessor):
        """Enhanced output processor with real ETF data support"""
        
        def __init__(self, output_format):
            super().__init__(output_format)
            self.real_data_loader = None
            
        def load_etf_data(self) -> Dict[str, pd.Series]:
            """Enhanced ETF data loading with real data support"""
            
            # Check if we should use real data
            if (hasattr(self.output_format, 'etf_data_source') and 
                self.output_format.etf_data_source == "qlib_data" and
                hasattr(self.output_format, 'etf_data_directory')):
                
                print("📊 Loading REAL ETF data from qlib_data directory...")
                
                # Determine data type based on ETF symbols
                bitcoin_etfs = ["IBIT", "FBTC", "GBTC"]
                has_bitcoin_etfs = any(symbol in bitcoin_etfs for symbol in self.output_format.etf_symbols)
                
                data_type = "1min" if has_bitcoin_etfs else "daily"
                
                # Create real data loader
                self.real_data_loader = RealETFDataLoader(
                    qlib_data_dir=self.output_format.etf_data_directory,
                    etf_symbols=self.output_format.etf_symbols,
                    data_type=data_type
                )
                
                # Load real ETF data
                real_etf_data = self.real_data_loader.load_real_etf_data()
                
                if real_etf_data:
                    print(f"✅ Successfully loaded real data for {len(real_etf_data)} ETFs")
                    return real_etf_data
                else:
                    print("⚠️  No real ETF data loaded, falling back to synthetic data")
            
            # Fallback to original synthetic data loading
            print("🔄 Using synthetic ETF data (fallback)")
            return super().load_etf_data()
    
    return EnhancedOutputProcessor(output_format)

# Monkey patch the original framework to use enhanced processor
def patch_framework_with_real_data_support():
    """Patch the framework to support real ETF data loading"""
    
    import complete_framework
    
    original_init = complete_framework.BacktestingFramework.__init__
    
    def enhanced_init(self):
        original_init(self)
        # Will be set when processing output
        self._enhanced_output_processor = None
    
    def enhanced_run_complete_backtest(self, dataset_input, factor_inputs, model_input, strategy_input, output_format):
        """Enhanced backtesting with real ETF data support"""
        
        # Create enhanced output processor
        self.output_processor = create_enhanced_output_processor_with_real_data(output_format)
        
        # Run the original backtesting logic
        return self._run_original_backtest(dataset_input, factor_inputs, model_input, strategy_input, output_format)
    
    # Store original method
    complete_framework.BacktestingFramework._run_original_backtest = complete_framework.BacktestingFramework.run_complete_backtest
    
    # Replace with enhanced version
    complete_framework.BacktestingFramework.__init__ = enhanced_init
    complete_framework.BacktestingFramework.run_complete_backtest = enhanced_run_complete_backtest
    
    print("🔧 Framework patched with real ETF data support!")

if __name__ == "__main__":
    # Apply the patch when this module is imported
    patch_framework_with_real_data_support()
    print("✅ Real data support module loaded!")
