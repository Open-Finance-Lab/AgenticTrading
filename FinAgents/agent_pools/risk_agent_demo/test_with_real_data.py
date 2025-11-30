"""
Test Risk Signal Agent with Real Data

This script loads real stock data from qlib_data and generates structured risk signals
using LLM-enhanced risk analysis.
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

from risk_signal_agent import RiskSignalAgent
from utils import QlibConfig, DataProcessor


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
        
        # Rename columns to match expected format
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


def format_structured_risk_signals(result: dict, data: pd.DataFrame) -> dict:
    """
    Format risk signals into structured output with LLM-friendly format
    
    Args:
        result: Result dictionary from generate_risk_signals_from_data
        data: Original market data
        
    Returns:
        Structured risk signal dictionary
    """
    if result['status'] != 'success':
        return {
            "status": "error",
            "message": result.get('message', 'Unknown error')
        }
    
    # Extract risk metrics
    risk_metrics = result.get('risk_metrics', {})
    risk_signals = result.get('risk_signals', {})
    
    # Format risk metrics for output
    formatted_metrics = {}
    for metric_name, metric_value in risk_metrics.items():
        if isinstance(metric_value, dict):
            formatted_metrics[metric_name] = {}
            for key, value in metric_value.items():
                if isinstance(value, (int, float)):
                    formatted_metrics[metric_name][key] = float(value)
                else:
                    formatted_metrics[metric_name][key] = value
        else:
            formatted_metrics[metric_name] = metric_value
    
    # Generate risk recommendations
    recommendations = generate_risk_recommendations(result)
    
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "overall_risk_level": result.get('overall_risk_level', 'UNKNOWN'),
        "risk_score": result.get('risk_score', 0.0),
        "risk_metrics": formatted_metrics,
        "risk_signals": risk_signals,
        "recommendations": recommendations,
        "data_summary": {
            "n_observations": result.get('n_observations', 0),
            "date_range": result.get('date_range', {}),
            "n_symbols": len(data['symbol'].unique()) if 'symbol' in data.columns else 1
        }
    }


def generate_risk_recommendations(result: dict) -> list:
    """
    Generate risk management recommendations based on risk signals
    
    Args:
        result: Result dictionary
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    risk_level = result.get('overall_risk_level', 'UNKNOWN')
    risk_score = result.get('risk_score', 0.0)
    risk_signals = result.get('risk_signals', {})
    risk_metrics = result.get('risk_metrics', {})
    
    # Overall risk level recommendations
    if risk_level == "HIGH":
        recommendations.append("⚠️ HIGH RISK DETECTED: Consider reducing position sizes and implementing strict stop-losses.")
        recommendations.append("Monitor positions closely and be prepared to exit if risk metrics deteriorate further.")
    elif risk_level == "MODERATE":
        recommendations.append("⚠️ MODERATE RISK: Maintain normal position sizing but monitor risk metrics regularly.")
        recommendations.append("Consider hedging strategies if risk levels increase.")
    else:
        recommendations.append("✓ LOW RISK: Current risk levels are acceptable for normal trading operations.")
    
    # Specific risk signal recommendations
    if "volatility" in risk_signals:
        vol_signal = risk_signals["volatility"]
        if vol_signal == "HIGH":
            recommendations.append("High volatility detected: Consider reducing leverage and increasing diversification.")
        elif vol_signal == "LOW":
            recommendations.append("Low volatility environment: May be suitable for higher leverage strategies.")
    
    if "var" in risk_signals:
        var_signal = risk_signals["var"]
        if var_signal == "SEVERE":
            recommendations.append("⚠️ SEVERE VaR: Potential for significant losses. Reduce exposure immediately.")
        elif var_signal == "MODERATE":
            recommendations.append("Moderate VaR: Monitor positions and set appropriate stop-losses.")
    
    if "max_drawdown" in risk_signals:
        mdd_signal = risk_signals["max_drawdown"]
        if mdd_signal == "SEVERE":
            recommendations.append("⚠️ SEVERE DRAWDOWN: Portfolio has experienced significant decline. Review strategy.")
    
    if "beta" in risk_signals:
        beta_signal = risk_signals["beta"]
        if beta_signal == "HIGH_MARKET_SENSITIVITY":
            recommendations.append("High beta detected: Portfolio is highly sensitive to market movements. Consider hedging.")
        elif beta_signal == "LOW_MARKET_SENSITIVITY":
            recommendations.append("Low beta: Portfolio has low correlation with market. Good for diversification.")
    
    if "correlation" in risk_signals:
        corr_signal = risk_signals["correlation"]
        if corr_signal == "HIGH_DIVERSIFICATION_RISK":
            recommendations.append("High correlation detected: Limited diversification benefit. Consider adding uncorrelated assets.")
    
    # Risk score based recommendations
    if risk_score > 0.6:
        recommendations.append(f"High risk score ({risk_score:.2f}): Implement comprehensive risk management measures.")
    elif risk_score > 0.3:
        recommendations.append(f"Moderate risk score ({risk_score:.2f}): Standard risk management practices recommended.")
    else:
        recommendations.append(f"Low risk score ({risk_score:.2f}): Current risk levels are manageable.")
    
    return recommendations


def print_structured_output(structured: dict):
    """
    Print structured risk signals in a readable format
    """
    print("\n" + "=" * 80)
    print("STRUCTURED RISK SIGNALS - LLM Enhanced Output")
    print("=" * 80)
    
    print(f"\n📊 Overall Risk Assessment:")
    print(f"   Risk Level: {structured['overall_risk_level']}")
    print(f"   Risk Score: {structured['risk_score']:.4f}")
    
    print(f"\n📈 Risk Metrics:")
    metrics = structured['risk_metrics']
    for metric_name, metric_value in metrics.items():
        print(f"\n   {metric_name.upper()}:")
        if isinstance(metric_value, dict):
            for key, value in metric_value.items():
                if isinstance(value, (int, float)):
                    print(f"     - {key}: {value:.4f}")
                else:
                    print(f"     - {key}: {value}")
        else:
            print(f"     - {metric_value}")
    
    print(f"\n🎯 Risk Signals:")
    signals = structured['risk_signals']
    for signal_type, signal_value in signals.items():
        print(f"   - {signal_type}: {signal_value}")
    
    print(f"\n💡 Recommendations:")
    recommendations = structured['recommendations']
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Display agent analysis if available
    if 'agent_analysis' in structured:
        print(f"\n🤖 Agent Analysis (LLM-generated):")
        agent_analysis = structured['agent_analysis']
        # Print first 1000 characters
        print(f"   {agent_analysis[:1000]}")
        if len(agent_analysis) > 1000:
            print(f"   ... (truncated, full analysis in JSON output)")
    
    print(f"\n📋 Data Summary:")
    data_summary = structured.get('data_summary', {})
    print(f"   - Observations: {data_summary.get('n_observations', 0)}")
    print(f"   - Symbols: {data_summary.get('n_symbols', 0)}")
    date_range = data_summary.get('date_range', {})
    if date_range:
        print(f"   - Date Range: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")
    
    print("\n" + "=" * 80)


def main():
    """
    Main test function
    """
    print("\n" + "=" * 80)
    print("Risk Signal Agent - Real Data Test")
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
    print(f"\n2. Initializing Risk Signal Agent...")
    agent = RiskSignalAgent(name="RealDataRiskAgent", model="gpt-5")
    
    # Prepare data summary for agent context
    data_summary = {
        "n_rows": len(data),
        "n_symbols": len(data['symbol'].unique()) if 'symbol' in data.columns else 1,
        "date_range": {
            "start": str(data['date'].min()),
            "end": str(data['date'].max())
        },
        "symbols": list(data['symbol'].unique()) if 'symbol' in data.columns else ["SINGLE_ASSET"]
    }
    
    # Calculate returns for agent to use
    processed_data = data.copy()
    if 'symbol' in processed_data.columns:
        processed_data = processed_data.sort_values(['symbol', 'date'])
        returns = processed_data.groupby('symbol')['close'].pct_change().dropna()
    else:
        returns = processed_data['close'].pct_change().dropna()
    
    # Prepare context with data
    context = {
        "data": data,
        "returns": returns,
        "data_summary": data_summary
    }
    
    # Use agent to calculate risk metrics and generate signals
    print(f"\n3. Using Agent to calculate risk metrics and generate signals...")
    print(f"   Metrics: volatility, VaR, CVaR, max drawdown, correlation, liquidity")
    
    try:
        # Create a request for the agent to analyze risk
        agent_request = f"""
        Please analyze the risk for the portfolio with the following data:
        - Number of symbols: {data_summary['n_symbols']}
        - Date range: {data_summary['date_range']['start']} to {data_summary['date_range']['end']}
        - Number of observations: {data_summary['n_rows']}
        
        Please calculate the following risk metrics:
        1. Volatility (annualized, 20-day window)
        2. VaR (Value at Risk, 95% confidence level)
        3. CVaR (Conditional VaR, 95% confidence level)
        4. Maximum Drawdown
        5. Correlation risk (if multiple symbols)
        6. Liquidity risk
        
        Then generate risk signals based on these metrics and provide an overall risk assessment.
        """
        
        # Use agent.run() to execute the analysis
        print(f"   Sending request to agent...")
        agent_response = agent.run(agent_request, context=context, max_turns=15)
        
        # Also generate risk signals using the direct method for structured output
        print(f"   Generating structured risk signals...")
        result = agent.generate_risk_signals_from_data(
            data=data,
            risk_metrics=["volatility", "var", "cvar", "max_drawdown", "correlation", "liquidity"]
        )
        
        if result['status'] == 'success':
            print(f"   ✓ Risk analysis successful")
            # Add agent response to result
            result['agent_response'] = agent_response
        else:
            print(f"   ✗ Risk analysis failed: {result.get('message', 'Unknown error')}")
            return
        
    except Exception as e:
        print(f"   ✗ Error generating risk signals: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Format structured output
    print(f"\n4. Formatting structured risk signals...")
    structured = format_structured_risk_signals(result, data)
    
    # Add agent response to structured output
    if 'agent_response' in result:
        structured['agent_analysis'] = result['agent_response']
    
    # Print output
    print_structured_output(structured)
    
    # Save to JSON
    output_file = Path(__file__).parent / "risk_signals_output.json"
    with open(output_file, 'w') as f:
        json.dump(structured, f, indent=2, default=str)
    print(f"\n💾 Structured risk signals saved to: {output_file}")
    
    # Display agent analysis if available
    if 'agent_analysis' in structured:
        print(f"\n5. Agent Analysis (LLM-enhanced):")
        print(f"   {structured['agent_analysis'][:800]}...")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

