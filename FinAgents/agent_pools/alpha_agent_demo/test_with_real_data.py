"""
Test Alpha Signal Agent with Real Data

This script loads real stock data from qlib_data and generates structured alpha signals
using LLM-enhanced model predictions.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json
from datetime import datetime

# Add paths
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "alpha_agent_pool"))
sys.path.insert(0, str(parent_dir / "alpha_agent_pool" / "qlib"))

from alpha_signal_agent import AlphaSignalAgent
from utils import QlibConfig, DataProcessor
from data_interfaces import FactorInput
from standard_factor_calculator import StandardFactorCalculator


def load_stock_data(symbols: list, data_dir: Path) -> pd.DataFrame:
    """
    Load stock data from CSV files
    
    Args:
        symbols: List of stock symbols (e.g., ['AAPL', 'MSFT'])
        data_dir: Path to stock_backup directory
        
    Returns:
        DataFrame with columns: date, symbol, close, high, low, open, volume
    """
    all_data = []
    
    for symbol in symbols:
        csv_file = data_dir / f"{symbol}_daily.csv"
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found, skipping {symbol}")
            continue
        
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
        
        # Rename columns to match Qlib format (both formats supported)
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Date': 'date'
        })
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Select and reorder columns
        df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        all_data.append(df)
    
    if not all_data:
        raise ValueError("No data loaded")
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    
    # Sort by date
    combined = combined.sort_values('date')
    
    return combined


def get_all_signals_summary(result: dict) -> list:
    """
    Get summary of all signals across all dates
    
    Args:
        result: Result dictionary from generate_signals_from_data
        
    Returns:
        List of signal summaries by date
    """
    signals_dict = result.get('signals', {})
    signals_series = pd.Series(signals_dict)
    
    if len(signals_series) == 0:
        return []
    
    # Convert to MultiIndex if possible
    if isinstance(signals_series.index[0], str):
        try:
            signals_series.index = pd.MultiIndex.from_tuples(
                [eval(idx) if isinstance(eval(idx), tuple) else (idx, 'UNKNOWN') 
                 for idx in signals_series.index]
            )
        except:
            pass
    
    if not isinstance(signals_series.index, pd.MultiIndex):
        return []
    
    # Group by date
    date_level = signals_series.index.get_level_values(0)
    symbol_level = signals_series.index.get_level_values(1)
    
    summaries = []
    for date in sorted(date_level.unique())[-10:]:  # Last 10 dates
        date_mask = date_level == date
        date_signals = signals_series[date_mask]
        date_symbols = symbol_level[date_mask]
        
        long_count = (date_signals > 0).sum()
        short_count = (date_signals < 0).sum()
        neutral_count = (date_signals == 0).sum()
        
        # Get top signals
        top_long = []
        top_short = []
        date_symbols_list = list(date_symbols)
        for i, (idx, val) in enumerate(date_signals.items()):
            symbol = date_symbols_list[i] if i < len(date_symbols_list) else 'UNKNOWN'
            if val > 0:
                top_long.append((symbol, float(val)))
            elif val < 0:
                top_short.append((symbol, float(val)))
        
        top_long.sort(key=lambda x: x[1], reverse=True)
        top_short.sort(key=lambda x: x[1])
        
        summaries.append({
            "date": str(date),
            "long_count": int(long_count),
            "short_count": int(short_count),
            "neutral_count": int(neutral_count),
            "top_long": top_long[:3],
            "top_short": top_short[:3]
        })
    
    return summaries


def format_structured_signals(result: dict, data: pd.DataFrame) -> dict:
    """
    Format alpha signals into structured output with LLM-friendly format
    
    Args:
        result: Result dictionary from generate_signals_from_data
        data: Original market data
        
    Returns:
        Structured signal dictionary
    """
    if result['status'] != 'success':
        return {
            "status": "error",
            "message": result.get('message', 'Unknown error')
        }
    
    # Extract signals
    signals_dict = result.get('signals', {})
    signals_series = pd.Series(signals_dict)
    
    # Get latest signals (most recent date)
    if len(signals_series) > 0:
        # Convert index if it's string representation of MultiIndex
        if isinstance(signals_series.index[0], str):
            # Try to parse MultiIndex
            try:
                signals_series.index = pd.MultiIndex.from_tuples(
                    [eval(idx) if isinstance(eval(idx), tuple) else (idx, 'UNKNOWN') 
                     for idx in signals_series.index]
                )
            except:
                pass
        
        # Get latest date signals
        if isinstance(signals_series.index, pd.MultiIndex):
            # Try to get date and symbol from MultiIndex
            try:
                date_level = signals_series.index.get_level_values(0)
                symbol_level = signals_series.index.get_level_values(1)
                latest_date = date_level.max()
                latest_mask = date_level == latest_date
                latest_signals = signals_series[latest_mask]
                latest_symbols = symbol_level[latest_mask]
                
                # Format as structured output
                structured_signals = []
                for i, (idx, signal_value) in enumerate(latest_signals.items()):
                    if isinstance(idx, tuple):
                        date, symbol = idx
                    else:
                        date = latest_date
                        symbol = latest_symbols.iloc[i] if i < len(latest_symbols) else 'UNKNOWN'
                    
                    signal_type = "LONG" if signal_value > 0 else "SHORT" if signal_value < 0 else "NEUTRAL"
                    
                    structured_signals.append({
                        "instrument": str(symbol),
                        "date": str(date),
                        "signal": float(signal_value),
                        "signal_type": signal_type,
                        "confidence": abs(float(signal_value))
                    })
                
                # Sort by confidence
                structured_signals.sort(key=lambda x: x['confidence'], reverse=True)
            except Exception as e:
                # Fallback: create signals from all data
                structured_signals = []
                for idx, signal_value in signals_series.items():
                    if isinstance(idx, tuple) and len(idx) == 2:
                        date, symbol = idx
                    else:
                        date = "unknown"
                        symbol = str(idx)
                    
                    signal_type = "LONG" if signal_value > 0 else "SHORT" if signal_value < 0 else "NEUTRAL"
                    
                    structured_signals.append({
                        "instrument": str(symbol),
                        "date": str(date),
                        "signal": float(signal_value),
                        "signal_type": signal_type,
                        "confidence": abs(float(signal_value))
                    })
                
                # Sort by confidence and take top 20
                structured_signals.sort(key=lambda x: x['confidence'], reverse=True)
                structured_signals = structured_signals[:20]
        else:
            structured_signals = []
    else:
        structured_signals = []
    
    # Get model performance
    performance = result.get('model_performance', {})
    feature_importance = result.get('feature_importance', {})
    
    # Format top features
    top_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
        reverse=True
    )[:10]
    
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "model_performance": {
            "test_correlation": performance.get('test_correlation', 0),
            "test_mae": performance.get('test_mae', 0),
            "train_correlation": performance.get('train_correlation', 0),
            "n_features": performance.get('n_features', 0),
            "n_samples": performance.get('n_test_samples', 0)
        },
        "alpha_signals": structured_signals,
        "top_features": [
            {"feature": feat, "importance": float(imp)} 
            for feat, imp in top_features
        ],
        "signal_summary": result.get('signal_summary', {}),
        "recommendations": generate_recommendations(result, structured_signals)
    }


def generate_recommendations(result: dict, signals: list) -> list:
    """
    Generate trading recommendations based on signals and model performance
    
    Args:
        result: Result dictionary
        signals: List of structured signals
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Model performance recommendations
    perf = result.get('model_performance', {})
    test_corr = perf.get('test_correlation', 0)
    
    if test_corr > 0.1:
        recommendations.append("Model shows strong predictive power. Consider using signals for trading.")
    elif test_corr > 0.05:
        recommendations.append("Model shows moderate predictive power. Use with caution and proper risk management.")
    else:
        recommendations.append("Model shows weak predictive power. Consider feature engineering or different model.")
    
    # Signal-based recommendations
    if signals:
        long_count = sum(1 for s in signals if s['signal_type'] == 'LONG')
        short_count = sum(1 for s in signals if s['signal_type'] == 'SHORT')
        
        if long_count > short_count * 2:
            recommendations.append(f"Strong bullish bias: {long_count} long signals vs {short_count} short signals.")
        elif short_count > long_count * 2:
            recommendations.append(f"Strong bearish bias: {short_count} short signals vs {long_count} long signals.")
        else:
            recommendations.append(f"Balanced market view: {long_count} long vs {short_count} short signals.")
        
        # Top signal recommendation
        if signals:
            top_signal = signals[0]
            recommendations.append(
                f"Strongest signal: {top_signal['signal_type']} on {top_signal['instrument']} "
                f"(confidence: {top_signal['confidence']:.4f})"
            )
    
    return recommendations


def print_structured_output(structured: dict):
    """
    Print structured alpha signals in a readable format
    """
    print("\n" + "=" * 80)
    print("STRUCTURED ALPHA SIGNALS - LLM Enhanced Output")
    print("=" * 80)
    
    print(f"\n📊 Model Performance:")
    perf = structured['model_performance']
    print(f"   Test Correlation: {perf['test_correlation']:.4f}")
    print(f"   Test MAE: {perf['test_mae']:.6f}")
    print(f"   Train Correlation: {perf['train_correlation']:.4f}")
    print(f"   Features Used: {perf['n_features']}")
    print(f"   Test Samples: {perf['n_samples']}")
    
    print(f"\n🎯 Alpha Signals (Latest Date):")
    signals = structured['alpha_signals']
    if signals:
        print(f"   Total Signals: {len(signals)}")
        print(f"\n   {'Instrument':<12} {'Signal Type':<12} {'Signal Value':<15} {'Confidence':<12}")
        print("   " + "-" * 55)
        for sig in signals[:20]:  # Show top 20
            print(f"   {sig['instrument']:<12} {sig['signal_type']:<12} {sig['signal']:<15.6f} {sig['confidence']:<12.4f}")
    else:
        print("   No signals generated")
    
    print(f"\n🔍 Top Features (by importance):")
    features = structured['top_features']
    if features:
        print(f"   {'Feature':<30} {'Importance':<15}")
        print("   " + "-" * 50)
        for feat in features[:10]:
            print(f"   {feat['feature']:<30} {feat['importance']:<15.6f}")
    
    print(f"\n💡 Recommendations:")
    recommendations = structured['recommendations']
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Show signal history
    if 'signal_history' in structured and structured['signal_history']:
        print(f"\n📈 Signal History (Last 10 Days):")
        print(f"   {'Date':<20} {'Long':<8} {'Short':<8} {'Neutral':<8} {'Top Signals':<30}")
        print("   " + "-" * 80)
        for hist in structured['signal_history']:
            top_sigs = []
            if hist['top_long']:
                top_sigs.append(f"LONG:{hist['top_long'][0][0]}")
            if hist['top_short']:
                top_sigs.append(f"SHORT:{hist['top_short'][0][0]}")
            sig_str = ", ".join(top_sigs) if top_sigs else "None"
            print(f"   {hist['date'][:19]:<20} {hist['long_count']:<8} {hist['short_count']:<8} {hist['neutral_count']:<8} {sig_str:<30}")
    
    print("\n" + "=" * 80)


def main():
    """
    Main test function
    """
    print("\n" + "=" * 80)
    print("Alpha Signal Agent - Real Data Test")
    print("=" * 80)
    
    # Configuration
    data_dir = Path("/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/stock_backup")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']  # Select 5 stocks
    
    print(f"\n1. Loading data for symbols: {symbols}")
    try:
        data = load_stock_data(symbols, data_dir)
        print(f"   ✓ Loaded data: {data.shape[0]} rows, {len(data['symbol'].unique())} symbols")
        print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return
    
    # Filter to recent data (last 6 months for faster processing)
    recent_date = data['date'].max() - pd.Timedelta(days=180)
    data = data[data['date'] >= recent_date]
    print(f"   Using recent data: {data.shape[0]} rows")
    
    # Initialize agent
    print(f"\n2. Initializing Alpha Signal Agent...")
    agent = AlphaSignalAgent(name="RealDataAgent")
    
    # Define factors
    print(f"\n3. Defining factors and indicators...")
    factors = [
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
    
    indicators = ['RSI', 'MACD', 'Bollinger', 'MA', 'Momentum']
    
    # Generate signals
    print(f"\n4. Generating alpha signals...")
    print(f"   Factors: {[f['factor_name'] for f in factors]}")
    print(f"   Indicators: {indicators}")
    print(f"   Model: Linear Regression")
    
    try:
        result = agent.generate_signals_from_data(
            data=data,
            factors=factors,
            indicators=indicators,
            model_type='linear',
            signal_threshold=0.005  # Lower threshold to generate more signals
        )
        
        if result['status'] == 'success':
            print(f"   ✓ Signal generation successful")
        else:
            print(f"   ✗ Signal generation failed: {result.get('message', 'Unknown error')}")
            return
        
    except Exception as e:
        print(f"   ✗ Error generating signals: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Format structured output
    print(f"\n5. Formatting structured alpha signals...")
    structured = format_structured_signals(result, data)
    
    # Add signal history
    signal_history = get_all_signals_summary(result)
    structured['signal_history'] = signal_history
    
    # Print output
    print_structured_output(structured)
    
    # Save to JSON
    output_file = Path(__file__).parent / "alpha_signals_output.json"
    with open(output_file, 'w') as f:
        json.dump(structured, f, indent=2, default=str)
    print(f"\n💾 Structured signals saved to: {output_file}")
    
    # Also try LLM-enhanced interpretation if API key is available
    import os
    if os.getenv("OPENAI_API_KEY"):
        print(f"\n6. Generating LLM-enhanced interpretation...")
        try:
            llm_request = f"""
            Based on the following alpha signal analysis:
            - Model test correlation: {structured['model_performance']['test_correlation']:.4f}
            - Number of signals: {len(structured['alpha_signals'])}
            - Top features: {', '.join([f['feature'] for f in structured['top_features'][:5]])}
            
            Provide a brief interpretation of these signals and their trading implications.
            """
            
            llm_response = agent.run(llm_request)
            print(f"\n🤖 LLM Interpretation:")
            print(f"   {llm_response[:500]}...")
        except Exception as e:
            print(f"   ⚠️  LLM interpretation skipped: {e}")
    else:
        print(f"\n6. LLM interpretation skipped (OPENAI_API_KEY not set)")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

