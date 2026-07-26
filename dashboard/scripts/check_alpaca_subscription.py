#!/usr/bin/env python3
"""
Check your Alpaca account data subscription tier.
"""

import json
import sys

# Direct-execution bootstrap: make the repo root importable so canonical
# `dashboard.backend.*` imports resolve (no-op when run as part of the package).
from _bootstrap import ensure_repo_root

ensure_repo_root()
from dashboard.backend.paths import CREDENTIALS_DIR

creds_path = CREDENTIALS_DIR / "alpaca.json"
with open(creds_path) as f:
    creds = json.load(f)

api_key = creds.get("api_key")
secret_key = creds.get("secret_key")

print("=" * 70)
print("📊 CHECKING ALPACA DATA SUBSCRIPTION")
print("=" * 70)

try:
    from alpaca.trading.client import TradingClient
except ImportError:
    print("❌ alpaca-py not installed")
    sys.exit(1)

# Connect to Alpaca
client = TradingClient(api_key, secret_key, paper=True)

try:
    # Get account info
    account = client.get_account()
    
    print(f"\n✅ Connected to Alpaca!")
    print(f"\nAccount Info:")
    print(f"  Account ID: {account.id}")
    print(f"  Account Type: Paper Trading" if creds.get("paper") else "Live Trading")
    print(f"  Account Status: {account.status}")
    print(f"  Equity: ${account.equity:,.2f}")
    print(f"  Buying Power: ${account.buying_power:,.2f}")
    
    # Check if there's subscription info
    if hasattr(account, 'data_subscriptions'):
        print(f"\nData Subscriptions:")
        subs = account.data_subscriptions
        for key, value in subs.items():
            print(f"  {key}: {value}")
    else:
        print(f"\n⚠️ Data subscription info not available via this endpoint")
    
    # Alternative: Check account configurations
    if hasattr(account, 'configurations'):
        print(f"\nAccount Configurations:")
        configs = account.configurations
        if configs:
            for key, value in configs.items():
                print(f"  {key}: {value}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"\nYour API credentials may not have permission to check subscriptions.")
    print(f"Go to: https://app.alpaca.markets → Account → Data Subscriptions")

print(f"\n{'='*70}")
print("📌 NEXT STEPS")
print(f"{'='*70}")
print("""
To upgrade market data:

1. Log in to Alpaca dashboard: https://app.alpaca.markets
2. Go to Account → Data Subscriptions
3. Select premium tier for your needs:
   - Basic: Limited hourly data (current)
   - Standard: Full historical bars ($0-20/month)
   - Premium: Real-time data ($99/month)

4. Complete billing setup

5. Restart your backtest - should get complete data!

Alternative for now:
- Switch to DAILY bars instead of HOURLY
- Daily data is 100% complete on free tier
- Still gives good backtest data
""")
