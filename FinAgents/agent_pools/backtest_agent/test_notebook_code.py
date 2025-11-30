#!/usr/bin/env python3
"""
Test script to verify notebook code works correctly
This matches the exact code from demo.ipynb
"""
from backtest_agent import BacktestAgent
import pandas as pd
import numpy as np

print("=" * 60)
print("Testing notebook code...")
print("=" * 60)

agent = BacktestAgent()

# Generate random noise signals as predictions
dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='D')
instruments = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JNJ', 'V', 
               'JPM', 'MA', 'PG', 'UNH', 'HD', 'DIS', 'BAC', 'XOM', 'CVX', 'ABBV']

np.random.seed(42)
prediction_data = []
for date in dates:
    for instrument in instruments:
        prediction_data.append({
            'datetime': date,
            'instrument': instrument,
            'score': np.random.randn()
        })

pred_df = pd.DataFrame(prediction_data)
predictions = pred_df.set_index(['datetime', 'instrument'])['score']

print(f"Generated {len(predictions)} random noise predictions")
print(f"Date range: {predictions.index.get_level_values('datetime').min()} to {predictions.index.get_level_values('datetime').max()}")
print(f"Instruments: {len(instruments)}")
print(f"Prediction statistics: mean={predictions.mean():.4f}, std={predictions.std():.4f}")

# Run backtest
results = agent.run_simple_backtest_paper_interface(
    predictions=predictions,
    start_time="2023-01-01",
    end_time="2023-12-31",
    look_back_period=20,
    investment_horizon=5,
    topk=50,
    plot_results=False,  # Disable for testing
    output_dir=None
)

print(f"\n=== Backtest Results ===")
if results.get('status') == 'success':
    metrics = results.get('performance_metrics', {})
    print(f"Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"Volatility: {metrics.get('volatility', 0):.2%}")
    
    # Validate
    total_return = metrics.get('total_return', 0)
    if abs(total_return) > 0.15:
        print(f"\n⚠️  WARNING: Return {total_return:.2%} seems too high for random noise!")
    else:
        print(f"\n✅ Results look reasonable for random noise signals")
else:
    print(f"❌ Error: {results.get('message')}")
    print(f"Error type: {results.get('error_type', 'Unknown')}")

print("=" * 60)

