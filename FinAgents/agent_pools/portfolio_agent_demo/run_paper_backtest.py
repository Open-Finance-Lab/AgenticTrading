"""
Run Paper Backtest Script

This script assembles the Alpha, Risk, and Portfolio agents to run a backtest
following the interface design from the "Algorithmic Trading Review" paper.

Workflow:
1. Data Loading (Qlib/Mock)
2. Pre-trade Analysis:
   - Alpha Signal Generation (AlphaSignalAgent)
   - Risk Signal Generation (RiskSignalAgent)
   - Transaction Cost Model (Fixed)
3. Portfolio Construction (PortfolioAgent)
4. Trade Execution (Simulated)
5. Post-trade Analysis (Performance Metrics)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add paths
current_dir = Path(__file__).parent
agent_pools_dir = current_dir.parent
sys.path.append(str(agent_pools_dir))
sys.path.append(str(agent_pools_dir / "alpha_agent_pool"))
sys.path.append(str(agent_pools_dir / "risk_agent_demo"))
sys.path.append(str(agent_pools_dir / "alpha_agent_demo"))
sys.path.append(str(current_dir))

# Import Agents
try:
    from alpha_signal_agent import AlphaSignalAgent
    from risk_signal_agent import RiskSignalAgent
    from portfolio_agent import PortfolioAgent
except ImportError as e:
    print(f"Error importing agents: {e}")
    print("Please ensure you are in the correct directory structure.")
    sys.exit(1)

def generate_mock_data(dates, symbols):
    """Generate mock price data for testing if Qlib is not fully configured."""
    data = []
    np.random.seed(42)
    
    prices = {s: 100.0 for s in symbols}
    
    for date in dates:
        for s in symbols:
            # Random walk
            ret = np.random.normal(0.0005, 0.02)
            prices[s] *= (1 + ret)
            volume = np.random.randint(1000, 100000)
            
            data.append({
                'date': date,
                'symbol': s,
                'open': prices[s],
                'high': prices[s] * 1.01,
                'low': prices[s] * 0.99,
                'close': prices[s],
                'volume': volume
            })
            
    return pd.DataFrame(data)

def run_backtest(start_date="2023-01-01", end_date="2023-03-31", symbols=None):
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
    print(f"Starting Backtest from {start_date} to {end_date}")
    print(f"Universe: {symbols}")
    
    # Initialize Agents
    alpha_agent = AlphaSignalAgent(name="AlphaAgent")
    risk_agent = RiskSignalAgent(name="RiskAgent")
    portfolio_agent = PortfolioAgent(name="PortfolioAgent")
    
    # Generate Dates for Data (including lookback)
    data_start_date = pd.to_datetime(start_date) - timedelta(days=60)
    data_dates = pd.date_range(start=data_start_date, end=end_date, freq='B')
    
    # Backtest Dates
    backtest_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Load/Generate Data
    print("Loading Data...")
    market_data = generate_mock_data(data_dates, symbols)
    market_data['date'] = pd.to_datetime(market_data['date'])
    market_data = market_data.sort_values(['symbol', 'date']) # Ensure sorted
    
    # Backtest Variables
    portfolio_value = 1000000.0 # Initial Capital
    cash = portfolio_value
    holdings = {s: 0 for s in symbols} # Quantity
    portfolio_history = []
    
    # Transaction Cost Model (Fixed)
    transaction_costs = {
        'fixed_cost': 5.0,
        'slippage': 0.001 # 10 bps
    }
    
    print("\nRunning Simulation...")
    
    for i, date in enumerate(backtest_dates):
        current_date_str = date.strftime("%Y-%m-%d")
        
        # Get Data for this date (and lookback)
        # We need history for signals
        history_start = date - timedelta(days=60)
        current_data = market_data[
            (market_data['date'] <= date) & 
            (market_data['date'] >= history_start)
        ]
        
        if len(current_data) < 20 * len(symbols): # Need some history
            continue
            
        # 1. Alpha Signals
        # We use a simple momentum factor config for the agent
        alpha_factors = [{
            "factor_name": "momentum_20d",
            "factor_type": "technical",
            "calculation_method": "expression",
            "expression": "close / Ref(close, 20) - 1",
            "lookback_period": 20
        }]
        
        daily_alpha = {}
        
        # Simulate Agent Response/Logic Robustly
        # Calculate momentum using shift
        for s in symbols:
            s_data = current_data[current_data['symbol'] == s].sort_values('date')
            if len(s_data) > 20:
                curr_price = s_data.iloc[-1]['close']
                # Get price 20 periods ago (approx 20 business days)
                prev_price = s_data.iloc[-21]['close']
                
                ret = curr_price / prev_price - 1
                daily_alpha[s] = float(ret)
            else:
                daily_alpha[s] = 0.0
        
        # print(f"Debug: {current_date_str} Alpha: {daily_alpha}")


        # 2. Risk Signals
        try:
            risk_result = risk_agent.generate_risk_signals_from_data(
                data=current_data
            )
            risk_signals = {
                "overall_risk_level": risk_result.get("overall_risk_level", "LOW"),
                "risk_score": risk_result.get("risk_score", 0.0)
            }
        except Exception as e:
            print(f"Risk Agent Error: {e}")
            risk_signals = {"overall_risk_level": "LOW", "risk_score": 0.0}

        # 3. Portfolio Construction
        try:
            portfolio_result = portfolio_agent.inference(
                alpha_signals=daily_alpha,
                risk_signals=risk_signals,
                transaction_costs=transaction_costs
            )
            target_weights = portfolio_result.get("target_weights", {})
        except Exception as e:
            print(f"Portfolio Agent Error: {e}")
            target_weights = {}

        # 4. Execution (Rebalancing)
        # Calculate Portfolio Value
        current_prices = current_data[current_data['date'] == date].set_index('symbol')['close']
        if current_prices.empty:
            continue
            
        total_value = cash
        for s, qty in holdings.items():
            if s in current_prices:
                total_value += qty * current_prices[s]
        
        # Execute Trades
        # Target Value per asset
        new_holdings = holdings.copy()
        transaction_cost_today = 0.0
        
        for s, weight in target_weights.items():
            if s in current_prices:
                target_val = total_value * weight
                current_val = holdings[s] * current_prices[s]
                diff_val = target_val - current_val
                
                if abs(diff_val) > 100: # Minimum trade size
                    price = current_prices[s]
                    qty_change = int(diff_val / price)
                    
                    if qty_change != 0:
                        cost = abs(qty_change * price) * transaction_costs['slippage'] + transaction_costs['fixed_cost']
                        transaction_cost_today += cost
                        
                        new_holdings[s] += qty_change
                        cash -= (qty_change * price + cost)

        holdings = new_holdings
        
        # Update Portfolio Value
        current_total_value = cash
        for s, qty in holdings.items():
            if s in current_prices:
                current_total_value += qty * current_prices[s]
                
        portfolio_history.append({
            'date': date,
            'value': current_total_value,
            'cash': cash,
            'risk_level': risk_signals['overall_risk_level'],
            'cost': transaction_cost_today
        })
        
        if i % 10 == 0:
            print(f"Date: {current_date_str}, Value: ${current_total_value:,.2f}, Risk: {risk_signals['overall_risk_level']}")

    # 5. Post-trade Analysis
    results_df = pd.DataFrame(portfolio_history).set_index('date')
    
    if results_df.empty:
        print("No results generated.")
        return

    results_df['returns'] = results_df['value'].pct_change()
    
    total_return = (results_df['value'].iloc[-1] / results_df['value'].iloc[0]) - 1
    sharpe_ratio = results_df['returns'].mean() / results_df['returns'].std() * np.sqrt(252)
    max_drawdown = (results_df['value'] / results_df['value'].cummax() - 1).min()
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown*100:.2f}%")
    print(f"Final Value: ${results_df['value'].iloc[-1]:,.2f}")
    print("="*50)
    
    # Save results
    output_path = Path("backtest_results.csv")
    results_df.to_csv(output_path)
    print(f"Detailed results saved to {output_path}")

    # Generate Plots and Tables
    plot_results(results_df)

def plot_results(results_df):
    """Generate and save P&L plot and performance table."""
    try:
        # 1. P&L Curve
        plt.figure(figsize=(12, 6))
        
        # Portfolio Value
        plt.subplot(2, 1, 1)
        plt.plot(results_df.index, results_df['value'], label='Portfolio Value', color='blue')
        plt.title('Portfolio Value (P&L) Over Time')
        plt.ylabel('Value ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Drawdown
        drawdown = (results_df['value'] / results_df['value'].cummax() - 1) * 100
        plt.subplot(2, 1, 2)
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
        plt.plot(drawdown.index, drawdown, color='red', linewidth=1)
        plt.title('Drawdown (%)')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('backtest_pnl.png')
        print("Saved P&L plot to backtest_pnl.png")
        
        # 2. Performance Table
        total_return = (results_df['value'].iloc[-1] / results_df['value'].iloc[0]) - 1
        daily_returns = results_df['value'].pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        volatility = daily_returns.std() * np.sqrt(252)
        max_drawdown = drawdown.min()
        
        metrics = {
            "Total Return": f"{total_return*100:.2f}%",
            "Annualized Return": f"{(1+total_return)**(252/len(results_df))-1:.2%}",
            "Annualized Volatility": f"{volatility:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2f}%",
            "Win Rate": f"{(daily_returns > 0).mean():.2%}",
        }
        
        # Create a simple table plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('tight')
        ax.axis('off')
        table_data = [[k, v] for k, v in metrics.items()]
        table = ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc='center', cellLoc='center')
        table.scale(1, 2)
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        plt.title("Performance Metrics Summary")
        plt.savefig('backtest_metrics.png')
        print("Saved metrics table to backtest_metrics.png")
        
    except Exception as e:
        print(f"Error plotting results: {e}")

if __name__ == "__main__":
    run_backtest()

