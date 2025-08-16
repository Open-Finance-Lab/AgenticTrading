"""
Data setup script for Qlib backtesting pipeline
This script helps download and setup sample data for testing
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import qlib
from qlib.data import D

def create_sample_data():
    """Create sample data for testing when real data is not available"""
    print("Creating sample synthetic data for testing...")
    
    # Create data directory
    data_dir = Path.home() / ".qlib" / "qlib_data" / "cn_data_sample"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data for CSI300-like stocks
    start_date = "2008-01-01"
    end_date = "2023-12-31"
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    # Filter to trading days (rough approximation)
    dates = dates[dates.dayofweek < 5]  # Monday=0, Friday=4
    
    # Create sample stocks
    stocks = [f"SH{600000 + i:03d}" for i in range(50)]  # Sample 50 stocks
    
    print(f"Generating data for {len(stocks)} stocks from {start_date} to {end_date}")
    
    # Create price data
    np.random.seed(42)  # For reproducible results
    
    all_data = []
    
    for stock in stocks:
        print(f"Generating data for {stock}...")
        
        # Generate price series with realistic properties
        initial_price = np.random.uniform(10, 100)
        prices = [initial_price]
        
        for i in range(1, len(dates)):
            # Random walk with slight upward drift
            daily_return = np.random.normal(0.0002, 0.02)  # ~5% annual drift, 30% vol
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 0.1))  # Ensure positive prices
        
        # Create OHLCV data
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Generate OHLC from close price
            noise = np.random.normal(0, 0.005)  # Small intraday noise
            
            open_price = price * (1 + noise)
            high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(100000, 10000000)
            
            all_data.append({
                'date': date,
                'instrument': stock,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': price,
                'volume': volume,
                'factor': 1.0  # Adjustment factor
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date', 'instrument']).sort_index()
    
    # Save to CSV (simple format for testing)
    output_file = data_dir / "sample_data.csv"
    df.to_csv(output_file)
    
    print(f"Sample data saved to: {output_file}")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
    print(f"Stocks: {len(df.index.get_level_values('instrument').unique())}")
    
    return output_file

def try_download_real_data():
    """Try to download real market data using various methods"""
    print("Attempting to download real market data...")
    
    # Method 1: Try the CLI data download
    try:
        from qlib.cli.data import GetData
        
        data_dir = Path.home() / ".qlib" / "qlib_data" / "cn_data"
        
        # Create GetData instance and try download
        get_data = GetData()
        
        # Try different download approaches
        print("Trying CLI data download...")
        
        # This might work depending on your qlib version
        try:
            result = get_data(
                qlib_data_name="qlib_data",
                target_dir=str(data_dir),
                region="cn"
            )
            print(f"Data download successful: {result}")
            return True
        except Exception as e:
            print(f"CLI download failed: {e}")
        
    except Exception as e:
        print(f"CLI method not available: {e}")
    
    # Method 2: Try direct URL download (if available)
    try:
        print("Trying direct download...")
        
        # Note: Actual URLs would need to be provided by qlib maintainers
        # This is just a template for future implementation
        data_urls = {
            "cn_data": "https://github.com/microsoft/qlib-data/releases/download/v1.0.0/cn_data.tar.gz"
        }
        
        # This would require actual download URLs
        print("Direct download URLs not available in this version")
        
    except Exception as e:
        print(f"Direct download failed: {e}")
    
    return False

def setup_qlib_with_sample_data():
    """Setup qlib to use sample data"""
    try:
        sample_file = create_sample_data()
        
        # Try to initialize qlib with sample data
        data_dir = sample_file.parent
        
        print(f"Initializing qlib with sample data from: {data_dir}")
        
        # Initialize qlib (this might fail if provider format is wrong)
        try:
            qlib.init(provider_uri=str(data_dir), region="cn")
            print("✓ Qlib initialized successfully with sample data!")
            return True
        except Exception as e:
            print(f"Qlib initialization failed: {e}")
            print("Sample data created but qlib initialization failed.")
            print("You can still use the CSV data for manual testing.")
            return False
            
    except Exception as e:
        print(f"Sample data creation failed: {e}")
        return False

def test_data_access():
    """Test if data can be accessed through qlib"""
    try:
        # Try to get instruments
        instruments = D.instruments()
        print(f"✓ Found {len(instruments)} instruments")
        
        # Try to get basic market data
        fields = ["$open", "$high", "$low", "$close", "$volume"]
        data = D.features(
            instruments=instruments[:5],  # Test with first 5 instruments
            fields=fields,
            start_time="2020-01-01",
            end_time="2020-01-31"
        )
        print(f"✓ Successfully loaded market data: {data.shape}")
        print(f"Sample data:\n{data.head()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data access test failed: {e}")
        return False

def main():
    """Main data setup function"""
    print("="*60)
    print("QLIB DATA SETUP")
    print("="*60)
    
    print("Checking current qlib installation...")
    print(f"Qlib version: {qlib.__version__}")
    
    # Try to download real data first
    if try_download_real_data():
        print("✓ Real data download successful!")
        
        # Test data access
        if test_data_access():
            print("✅ Data setup complete - ready for backtesting!")
            return True
    
    print("\nReal data download failed. Creating sample data for testing...")
    
    # Create sample data as fallback
    if setup_qlib_with_sample_data():
        if test_data_access():
            print("✅ Sample data setup complete - ready for testing!")
            return True
    
    print("\n⚠️  Data setup failed. You can still test the pipeline logic with synthetic data.")
    print("Check the test_basic.py script for pipeline testing without external data.")
    
    return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n" + "="*60)
        print("ALTERNATIVE TESTING OPTIONS")
        print("="*60)
        print("1. Run 'python test_basic.py' for basic pipeline testing")
        print("2. Use synthetic data in your factors and models")
        print("3. Manually download qlib data from official sources")
        print("4. Use alternative data sources (Yahoo Finance, etc.)")
        print("\nThe pipeline is ready to use once data is available!")
