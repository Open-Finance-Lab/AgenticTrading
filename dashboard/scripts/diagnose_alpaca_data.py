#!/usr/bin/env python3
"""
Diagnose Alpaca API data quality.

Why are we getting missing data for DJIA stocks?
Let's inspect what Alpaca is actually returning.
"""

import json
from datetime import datetime, timedelta
import sys

# Direct-execution bootstrap: make the repo root importable so canonical
# `dashboard.backend.*` imports resolve (no-op when run as part of the package).
from _bootstrap import ensure_repo_root

ensure_repo_root()
from dashboard.backend.paths import CREDENTIALS_DIR

# Load credentials
creds_path = CREDENTIALS_DIR / "alpaca.json"
with open(creds_path) as f:
    creds = json.load(f)

api_key = creds.get("api_key")
secret_key = creds.get("secret_key")

print("=" * 70)
print("🔍 ALPACA DATA QUALITY DIAGNOSIS")
print("=" * 70)

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    print("❌ alpaca-py not installed")
    sys.exit(1)

client = StockHistoricalDataClient(api_key, secret_key)

# Test with a few DJIA stocks
test_symbols = ["AAPL", "MSFT", "JPM", "V", "JNJ"]
start_date = "2026-04-20"  # Last few days
end_date = "2026-04-23"

print(f"\nTest Parameters:")
print(f"  Symbols: {test_symbols}")
print(f"  Date range: {start_date} to {end_date}")
print(f"  Timeframe: Hourly (TimeFrame.Hour)")

# Fetch data
request = StockBarsRequest(
    symbol_or_symbols=test_symbols,
    timeframe=TimeFrame.Hour,
    start=start_date,
    end=end_date,
)

print(f"\n📊 Fetching from Alpaca...")
try:
    bars_data = client.get_stock_bars(request)
    
    # Inspect response structure
    print(f"\n✅ Response received!")
    print(f"  Type: {type(bars_data)}")
    print(f"  Has df attribute: {hasattr(bars_data, 'df')}")
    
    if hasattr(bars_data, 'df'):
        df = bars_data.df
        print(f"\n📈 DataFrame info:")
        print(f"  Shape: {df.shape}")
        print(f"  Index levels: {df.index.nlevels}")
        print(f"  Index names: {df.index.names}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check coverage per symbol
        print(f"\n📊 Data coverage per symbol:")
        for symbol in test_symbols:
            if symbol in df.index.get_level_values(0):
                symbol_df = df.loc[symbol]
                print(f"\n  {symbol}:")
                print(f"    Rows: {len(symbol_df)}")
                print(f"    Date range: {symbol_df.index.min()} to {symbol_df.index.max()}")
                print(f"    First 3 rows:")
                for idx, row in symbol_df.head(3).iterrows():
                    print(f"      {idx}: close=${row['close']:.2f}, volume={row['volume']:.0f}")
            else:
                print(f"\n  {symbol}: ❌ NO DATA")
        
        # Calculate expected hours
        from datetime import datetime
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Count business hours (market open 9:30 AM - 4:00 PM ET)
        current = start_dt
        business_hours = 0
        while current <= end_dt:
            # Skip weekends
            if current.weekday() < 5:  # Mon-Fri
                business_hours += 1
            current += timedelta(hours=1)
        
        print(f"\n⏰ Expected vs Actual:")
        print(f"  Expected business hours: {business_hours}")
        for symbol in test_symbols:
            if symbol in df.index.get_level_values(0):
                actual = len(df.loc[symbol])
                pct = 100 * actual / business_hours if business_hours > 0 else 0
                print(f"  {symbol}: {actual} ({pct:.1f}% of expected)")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Now test with daily data for comparison
print(f"\n\n{'='*70}")
print("🔍 COMPARISON: Daily data")
print(f"{'='*70}")

request_daily = StockBarsRequest(
    symbol_or_symbols=test_symbols,
    timeframe=TimeFrame.Day,
    start=start_date,
    end=end_date,
)

print(f"\n📊 Fetching daily data from Alpaca...")
try:
    bars_daily = client.get_stock_bars(request_daily)
    
    if hasattr(bars_daily, 'df'):
        df_daily = bars_daily.df
        print(f"\n✅ Daily data received!")
        print(f"  Shape: {df_daily.shape}")
        
        print(f"\n📊 Daily data coverage per symbol:")
        for symbol in test_symbols:
            if symbol in df_daily.index.get_level_values(0):
                actual = len(df_daily.loc[symbol])
                print(f"  {symbol}: {actual} days")
            else:
                print(f"  {symbol}: ❌ NO DATA")

except Exception as e:
    print(f"\n❌ Error: {e}")

# Test with a longer range
print(f"\n\n{'='*70}")
print("🔍 LONGER RANGE TEST: 6 weeks of hourly data")
print(f"{'='*70}")

start_long = "2026-03-01"
end_long = "2026-04-23"

request_long = StockBarsRequest(
    symbol_or_symbols=["AAPL"],  # Just AAPL for simplicity
    timeframe=TimeFrame.Hour,
    start=start_long,
    end=end_long,
)

print(f"\nFetching AAPL hourly data ({start_long} to {end_long})...")
try:
    bars_long = client.get_stock_bars(request_long)
    
    if hasattr(bars_long, 'df'):
        df_long = bars_long.df
        aapl_df = df_long.loc["AAPL"] if "AAPL" in df_long.index.get_level_values(0) else None
        
        if aapl_df is not None:
            print(f"\n✅ AAPL hourly data:")
            print(f"  Total rows: {len(aapl_df)}")
            print(f"  Date range: {aapl_df.index.min()} to {aapl_df.index.max()}")
            
            # Calculate gaps
            import pandas as pd
            time_diffs = aapl_df.index.to_series().diff()
            typical_gap = pd.Timedelta(hours=1)
            gaps = time_diffs[time_diffs > typical_gap]
            
            print(f"\n⏰ Data gaps (missing hours):")
            print(f"  Total gaps: {len(gaps)}")
            if len(gaps) > 0:
                print(f"  Gap sizes:")
                for gap_size, count in gaps.value_counts().head(5).items():
                    print(f"    {gap_size}: {count} occurrences")
        else:
            print(f"❌ AAPL not in response")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}")
print("📌 SUMMARY: Likely causes of missing data")
print(f"{'='*70}")
print("""
Possible reasons for gaps in DJIA hourly data:

1. **Market hours only**: Alpaca only returns data during 9:30 AM - 4:00 PM ET
   → Expected ~6.5 hours per day, ~32.5 hours per week

2. **Halts/suspensions**: Individual stocks may halt trading briefly
   → Would explain gaps in some symbols but not others

3. **API rate limits**: Fetching 30 symbols at once might be rate-limited
   → Some symbols might not get returned

4. **Data availability**: Alpaca may not have complete hourly data for all symbols
   → Check their documentation on historical data coverage

5. **Timezone issues**: Our timestamps might be in wrong timezone
   → Would create false "gaps" where data exists but doesn't match

6. **Premium data requirement**: Full hourly bars might require paid subscription
   → Free tier might have reduced coverage

RECOMMENDATION:
- Check Alpaca API docs on historical data availability
- Try requesting data with different parameters
- Consider using daily bars instead (more reliable)
- Filter to only high-liquidity symbols
""")
