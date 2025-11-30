"""
Example Usage of Risk Signal Agent

This script demonstrates how to use the Risk Signal Agent to:
1. Load market data
2. Calculate risk metrics (volatility, VaR, CVaR, max drawdown, beta, etc.)
3. Generate risk signals
4. Get LLM-enhanced risk assessment
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "alpha_agent_pool"))

from risk_signal_agent import RiskSignalAgent

# Try to import Qlib utilities (optional)
try:
    sys.path.insert(0, str(parent_dir / "alpha_agent_pool" / "qlib"))
    from utils import QlibConfig, DataProcessor
except ImportError:
    # Use minimal stubs if Qlib not available
    from dataclasses import dataclass, field
    from typing import List
    
    @dataclass
    class QlibConfig:
        provider_uri: str = ""
        instruments: List[str] = field(default_factory=list)
    
    class DataProcessor:
        def __init__(self, config):
            self.config = config


def create_sample_data(n_days: int = 252, n_stocks: int = 5) -> pd.DataFrame:
    """
    Create sample market data for demonstration
    
    Args:
        n_days: Number of trading days
        n_stocks: Number of stocks
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    stocks = [f'STOCK_{i:02d}' for i in range(1, n_stocks + 1)]
    
    data_list = []
    for stock in stocks:
        # Generate random walk prices with different volatilities
        volatility = np.random.uniform(0.15, 0.35)  # Different risk levels
        returns = np.random.normal(0.001, volatility/np.sqrt(252), n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        open_price = prices * (1 + np.random.normal(0, 0.005, n_days))
        volume = np.random.lognormal(10, 1, n_days)
        
        df = pd.DataFrame({
            'date': dates,
            'symbol': stock,
            'open': open_price,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume
        })
        data_list.append(df)
    
    combined = pd.concat(data_list, ignore_index=True)
    
    return combined


def example_1_basic_risk_analysis():
    """
    Example 1: Basic risk analysis with sample data
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Risk Signal Generation")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample market data...")
    data = create_sample_data(n_days=252, n_stocks=5)
    print(f"   Data shape: {data.shape[0]} rows, {len(data['symbol'].unique())} symbols")
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
    
    # Initialize agent
    print("\n2. Initializing Risk Signal Agent...")
    agent = RiskSignalAgent(name="ExampleRiskAgent")
    
    # Generate risk signals
    print("\n3. Calculating risk metrics and generating signals...")
    result = agent.generate_risk_signals_from_data(
        data=data,
        risk_metrics=["volatility", "var", "cvar", "max_drawdown", "correlation", "liquidity"]
    )
    
    # Display results
    print("\n4. Results:")
    if result['status'] == 'success':
        print(f"   Overall Risk Level: {result['overall_risk_level']}")
        print(f"   Risk Score: {result['risk_score']:.4f}")
        print(f"   Observations: {result['n_observations']}")
        
        print(f"\n   Risk Metrics:")
        metrics = result['risk_metrics']
        if 'volatility' in metrics:
            vol = metrics['volatility']
            print(f"   - Volatility: {vol.get('current', 0):.4f} (annualized)")
        if 'var' in metrics:
            var = metrics['var']
            print(f"   - VaR (95%): {var.get('historical_var', 0):.4f}")
        if 'cvar' in metrics:
            cvar = metrics['cvar']
            print(f"   - CVaR (95%): {cvar.get('cvar_value', 0):.4f}")
        if 'max_drawdown' in metrics:
            mdd = metrics['max_drawdown']
            print(f"   - Max Drawdown: {mdd.get('percentage', 0):.2f}%")
        if 'correlation_risk' in metrics:
            corr = metrics['correlation_risk']
            print(f"   - Avg Correlation: {corr.get('average_correlation', 0):.4f}")
        if 'liquidity_risk' in metrics:
            liq = metrics['liquidity_risk']
            print(f"   - Liquidity Score: {liq.get('liquidity_score', 0):.4f}")
        
        print(f"\n   Risk Signals:")
        signals = result['risk_signals']
        for signal_type, signal_value in signals.items():
            print(f"   - {signal_type}: {signal_value}")
    else:
        print(f"   Error: {result.get('message', 'Unknown error')}")


def example_2_beta_calculation():
    """
    Example 2: Beta calculation with market returns
    """
    print("\n" + "=" * 60)
    print("Example 2: Beta Calculation with Market Returns")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data(n_days=252, n_stocks=3)
    
    # Create market returns (benchmark)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    market_returns = pd.Series(
        np.random.normal(0.0005, 0.15/np.sqrt(252), 252),
        index=dates
    )
    
    # Initialize agent
    agent = RiskSignalAgent()
    
    print("\n1. Calculating beta with market returns...")
    result = agent.generate_risk_signals_from_data(
        data=data,
        market_returns=market_returns,
        risk_metrics=["volatility", "beta", "var"]
    )
    
    if result['status'] == 'success':
        print(f"\n   Risk Metrics:")
        metrics = result['risk_metrics']
        if 'beta' in metrics:
            beta = metrics['beta']
            print(f"   - Current Beta: {beta.get('current', 0):.4f}")
            print(f"   - Mean Beta: {beta.get('mean', 0):.4f}")
        
        if 'volatility' in metrics:
            vol = metrics['volatility']
            print(f"   - Volatility: {vol.get('current', 0):.4f}")


def example_3_comprehensive_risk_analysis():
    """
    Example 3: Comprehensive risk analysis with all metrics
    """
    print("\n" + "=" * 60)
    print("Example 3: Comprehensive Risk Analysis")
    print("=" * 60)
    
    # Create sample data with different risk profiles
    data = create_sample_data(n_days=500, n_stocks=10)
    
    # Initialize agent
    agent = RiskSignalAgent()
    
    print("\n1. Running comprehensive risk analysis...")
    result = agent.generate_risk_signals_from_data(
        data=data,
        risk_metrics=["volatility", "var", "cvar", "max_drawdown", "correlation", "liquidity"]
    )
    
    if result['status'] == 'success':
        print(f"\n   Overall Risk Assessment:")
        print(f"   - Risk Level: {result['overall_risk_level']}")
        print(f"   - Risk Score: {result['risk_score']:.4f}")
        
        print(f"\n   Detailed Risk Metrics:")
        metrics = result['risk_metrics']
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict):
                print(f"\n   {metric_name.upper()}:")
                for key, value in metric_value.items():
                    if isinstance(value, (int, float)):
                        print(f"     - {key}: {value:.4f}")
                    else:
                        print(f"     - {key}: {value}")
        
        print(f"\n   Risk Signals:")
        signals = result['risk_signals']
        for signal_type, signal_value in signals.items():
            print(f"   - {signal_type}: {signal_value}")


def example_4_agent_interaction():
    """
    Example 4: Using the agent with OpenAI API for interactive risk analysis
    """
    print("\n" + "=" * 60)
    print("Example 4: Interactive Agent Usage (requires OpenAI API key)")
    print("=" * 60)
    
    import os
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set. Skipping interactive example.")
        print("   To use this example, set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key'")
        return
    
    # Initialize agent
    agent = RiskSignalAgent(name="InteractiveRiskAgent")
    
    # Example queries
    queries = [
        "Explain how to calculate Value at Risk (VaR) for a portfolio",
        "What is the difference between VaR and CVaR?",
        "How should I interpret beta values in risk analysis?",
        "What risk metrics are most important for portfolio management?"
    ]
    
    print("\nExample queries:")
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")
        print("   (Uncomment the line below to execute)")
        # response = agent.run(query)
        # print(f"   Response: {response[:200]}...")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Risk Signal Agent - Example Usage")
    print("=" * 60)
    
    # Run examples
    try:
        example_1_basic_risk_analysis()
        example_2_beta_calculation()
        example_3_comprehensive_risk_analysis()
        example_4_agent_interaction()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

