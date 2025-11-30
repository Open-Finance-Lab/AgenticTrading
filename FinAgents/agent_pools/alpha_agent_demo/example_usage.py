"""
Example Usage of Alpha Signal Agent

This script demonstrates how to use the Alpha Signal Agent to:
1. Load market data
2. Construct Qlib factors
3. Calculate technical indicators
4. Train ML models
5. Generate alpha signals
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "alpha_agent_pool"))

from alpha_signal_agent import AlphaSignalAgent

# Try to import Qlib utilities (optional)
try:
    sys.path.insert(0, str(parent_dir / "alpha_agent_pool" / "qlib"))
    from utils import QlibConfig, DataProcessor
    from data_interfaces import FactorInput
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


def create_sample_data(n_days: int = 252, n_stocks: int = 10) -> pd.DataFrame:
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
        # Generate random walk prices
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        open_price = prices * (1 + np.random.normal(0, 0.005, n_days))
        volume = np.random.lognormal(10, 1, n_days)
        
        df = pd.DataFrame({
            'Date': dates,
            'instrument': stock,
            '$open': open_price,
            '$high': high,
            '$low': low,
            '$close': prices,
            '$volume': volume
        })
        data_list.append(df)
    
    combined = pd.concat(data_list, ignore_index=True)
    combined = combined.set_index(['Date', 'instrument'])
    
    return combined


def example_1_basic_usage():
    """
    Example 1: Basic usage with sample data
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Alpha Signal Generation")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample market data...")
    data = create_sample_data(n_days=252, n_stocks=5)
    print(f"   Data shape: {data.shape}")
    print(f"   Date range: {data.index.get_level_values('Date').min()} to {data.index.get_level_values('Date').max()}")
    
    # Initialize agent
    print("\n2. Initializing Alpha Signal Agent...")
    agent = AlphaSignalAgent(name="ExampleAgent")
    
    # Define factors
    print("\n3. Defining factors...")
    factors = [
        {
            'factor_name': 'momentum_20d',
            'factor_type': 'alpha',
            'calculation_method': 'expression',
            'expression': 'close / Ref(close, 20) - 1',
            'lookback_period': 20
        },
        {
            'factor_name': 'volatility_20d',
            'factor_type': 'risk',
            'calculation_method': 'function',
            'function_name': 'volatility',
            'lookback_period': 20
        }
    ]
    
    # Define indicators
    indicators = ['RSI', 'MACD', 'Bollinger', 'Momentum']
    
    # Generate signals
    print("\n4. Generating alpha signals...")
    result = agent.generate_signals_from_data(
        data=data,
        factors=factors,
        indicators=indicators,
        model_type='linear',
        signal_threshold=0.01
    )
    
    # Display results
    print("\n5. Results:")
    if result['status'] == 'success':
        print(f"   Model Performance:")
        perf = result['model_performance']
        print(f"   - Test Correlation: {perf.get('test_correlation', 0):.4f}")
        print(f"   - Test MAE: {perf.get('test_mae', 0):.6f}")
        print(f"   - Features: {result['n_features']}")
        print(f"   - Samples: {result['n_samples']}")
        
        print(f"\n   Signal Summary:")
        sig_summary = result['signal_summary']
        print(f"   - Total Signals: {sig_summary['total_signals']}")
        print(f"   - Long Signals: {sig_summary['long_signals']}")
        print(f"   - Short Signals: {sig_summary['short_signals']}")
        print(f"   - Neutral Signals: {sig_summary['neutral_signals']}")
        
        print(f"\n   Top Feature Importance:")
        if 'feature_importance' in result:
            importance = result['feature_importance']
            sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for feat, imp in sorted_importance:
                print(f"   - {feat}: {imp:.6f}")
    else:
        print(f"   Error: {result.get('message', 'Unknown error')}")


def example_2_qlib_factors():
    """
    Example 2: Using Qlib factor construction
    """
    print("\n" + "=" * 60)
    print("Example 2: Qlib Factor Construction")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data(n_days=252, n_stocks=10)
    
    # Initialize agent
    agent = AlphaSignalAgent()
    
    # Use Qlib factor construction
    print("\n1. Constructing Qlib factors...")
    
    factor_configs = [
        {
            'factor_name': 'momentum_5d',
            'factor_type': 'alpha',
            'calculation_method': 'expression',
            'lookback_period': 5
        },
        {
            'factor_name': 'momentum_10d',
            'factor_type': 'alpha',
            'calculation_method': 'expression',
            'lookback_period': 10
        },
        {
            'factor_name': 'momentum_20d',
            'factor_type': 'alpha',
            'calculation_method': 'expression',
            'lookback_period': 20
        }
    ]
    
    # Generate signals with multiple momentum factors
    print("\n2. Generating signals with multiple momentum factors...")
    result = agent.generate_signals_from_data(
        data=data,
        factors=factor_configs,
        indicators=['RSI', 'MA'],
        model_type='lightgbm',
        signal_threshold=0.005
    )
    
    if result['status'] == 'success':
        print(f"\n   Model Performance:")
        print(f"   - Test Correlation: {result['model_performance']['test_correlation']:.4f}")
        print(f"   - Long Signals: {result['signal_summary']['long_signals']}")
        print(f"   - Short Signals: {result['signal_summary']['short_signals']}")


def example_3_ml_models():
    """
    Example 3: Comparing different ML models
    """
    print("\n" + "=" * 60)
    print("Example 3: Comparing ML Models")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data(n_days=500, n_stocks=10)
    
    # Define factors and indicators
    factors = [
        {
            'factor_name': 'momentum_20d',
            'factor_type': 'alpha',
            'calculation_method': 'expression',
            'lookback_period': 20
        }
    ]
    indicators = ['RSI', 'MACD', 'Bollinger', 'MA', 'Momentum']
    
    # Test different models
    models = ['linear', 'lightgbm', 'random_forest']
    results = {}
    
    agent = AlphaSignalAgent()
    
    for model_type in models:
        print(f"\nTesting {model_type} model...")
        result = agent.generate_signals_from_data(
            data=data,
            factors=factors,
            indicators=indicators,
            model_type=model_type,
            signal_threshold=0.01
        )
        
        if result['status'] == 'success':
            results[model_type] = result['model_performance']
            print(f"  Test Correlation: {result['model_performance']['test_correlation']:.4f}")
            print(f"  Test MAE: {result['model_performance']['test_mae']:.6f}")
    
    # Compare results
    print("\n" + "-" * 60)
    print("Model Comparison:")
    print("-" * 60)
    print(f"{'Model':<20} {'Test Correlation':<20} {'Test MAE':<15}")
    print("-" * 60)
    for model_type, perf in results.items():
        print(f"{model_type:<20} {perf['test_correlation']:<20.4f} {perf['test_mae']:<15.6f}")


def example_4_agent_interaction():
    """
    Example 4: Using the agent with OpenAI API for interactive queries
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
    agent = AlphaSignalAgent(name="InteractiveAgent")
    
    # Example queries
    queries = [
        "Explain how to construct a momentum factor using Qlib",
        "What technical indicators are most useful for alpha generation?",
        "How do I choose between linear regression and ML models for alpha signals?"
    ]
    
    print("\nExample queries:")
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")
        print("   (Uncomment the line below to execute)")
        # response = agent.run(query)
        # print(f"   Response: {response[:200]}...")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Alpha Signal Agent - Example Usage")
    print("=" * 60)
    
    # Run examples
    try:
        example_1_basic_usage()
        example_2_qlib_factors()
        example_3_ml_models()
        example_4_agent_interaction()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

