"""
Qlib Standard Framework - Example Usage and Demo

This module demonstrates how to use the Qlib Standard Framework
for complete quantitative trading workflows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .framework import QlibStandardFramework


def run_synthetic_data_demo():
    """
    Run a complete demo using synthetic data.
    
    This demonstrates the full pipeline with automatically generated data.
    """
    print(" Running Qlib Standard Framework Demo with Synthetic Data")
    print("=" * 60)
    
    # Configuration for synthetic data demo
    config = {
        'data': {
            'source_type': 'synthetic',
            'instruments': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'],
            'start_time': '2022-01-01',
            'end_time': '2023-12-31',
            'freq': '1D'
        },
        
        'factors': {
            'enabled': True,
            'factor_list': [
                'rsi_14', 'sma_20', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower',
                'momentum_5', 'momentum_10', 'roc_10', 'stoch_k', 'stoch_d',
                'williams_r', 'cci_14', 'atr_14', 'volume_sma_20'
            ],
            'custom_factors': {
                'custom_momentum': '(Ref($close, 0) / Ref($close, 5) - 1)',
                'price_volume': '$close * $volume',
                'volatility_5d': 'Std($close, 5)'
            }
        },
        
        'preprocessing': {
            'normalization': 'zscore',
            'handle_missing': True,
            'train_test_split': 0.7,
            'validation_split': 0.2
        },
        
        'model': {
            'type': 'lightgbm',
            'config': {
                'objective': 'regression',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'num_boost_round': 100,
                'verbosity': -1
            }
        },
        
        'strategy': {
            'type': 'long_short',
            'top_k': 3,
            'bottom_k': 3,
            'max_position_weight': 0.15
        }
    }
    
    # Initialize and run framework
    framework = QlibStandardFramework(config=config)
    
    try:
        results = framework.run_complete_pipeline(
            save_results=True,
            output_dir="qlib_demo_results"
        )
        
        print("\n Pipeline Results Summary:")
        print("-" * 40)
        
        # Data summary
        data_summary = results['data_summary']
        if 'raw_data' in data_summary:
            raw_info = data_summary['raw_data']
            print(f" Raw Data: {raw_info['shape']} points, {raw_info['instruments']} instruments")
        
        if 'factor_data' in data_summary:
            factor_info = data_summary['factor_data']
            print(f"🧮 Factor Data: {factor_info['shape']}, {factor_info['feature_count']} features")
        
        # Model summary
        model_summary = results['model_summary']
        if model_summary:
            print(f" Model: {model_summary['model_type']}, Fitted: {model_summary['is_fitted']}")
            
            if 'top_features' in model_summary:
                print(" Top Features:")
                for feature, importance in list(model_summary['top_features'].items())[:5]:
                    print(f"   - {feature}: {importance:.4f}")
        
        # Backtest summary
        backtest_summary = results['backtest_summary']
        if backtest_summary:
            print("\n Backtest Performance:")
            print(f"   Annual Return: {backtest_summary.get('annual_return', 0):.2%}")
            print(f"   Sharpe Ratio: {backtest_summary.get('sharpe_ratio', 0):.3f}")
            print(f"   Max Drawdown: {backtest_summary.get('max_drawdown', 0):.2%}")
            print(f"   Volatility: {backtest_summary.get('volatility', 0):.2%}")
        
        print(f"\n Demo completed successfully!")
        print(f" Results saved to: qlib_demo_results/")
        
        return results
        
    except Exception as e:
        print(f" Demo failed: {str(e)}")
        raise


def run_csv_data_demo(data_path: str):
    """
    Run a complete demo using CSV data.
    
    Args:
        data_path: Path to CSV data file or directory
        
    This demonstrates the full pipeline with user-provided CSV data.
    """
    print(" Running Qlib Standard Framework Demo with CSV Data")
    print("=" * 60)
    
    # Configuration for CSV data demo
    config = {
        'data': {
            'source_type': 'csv',
            'data_path': data_path,
            'start_time': '2022-01-01',
            'end_time': '2023-12-31',
            'freq': '1H'
        },
        
        'factors': {
            'enabled': True,
            'factor_list': None,  # Use all default factors
            'custom_factors': {
                'vwap': '($high + $low + $close) / 3',
                'price_change': 'Ref($close, 0) / Ref($close, 1) - 1',
                'volume_ratio': '$volume / Mean($volume, 20)'
            }
        },
        
        'preprocessing': {
            'normalization': 'robust',
            'handle_missing': True,
            'train_test_split': 0.8,
            'validation_split': 0.15
        },
        
        'model': {
            'type': 'ridge',
            'config': {
                'alpha': 1.0,
                'fit_intercept': True
            }
        },
        
        'strategy': {
            'type': 'topk',
            'top_k': 10
        }
    }
    
    # Initialize and run framework
    framework = QlibStandardFramework(config=config)
    
    try:
        results = framework.run_complete_pipeline(
            save_results=True,
            output_dir=f"qlib_csv_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        print("\n Pipeline Results Summary:")
        print("-" * 40)
        
        # Print comprehensive results
        _print_results_summary(results)
        
        return results
        
    except Exception as e:
        print(f" Demo failed: {str(e)}")
        raise


def run_custom_config_demo(custom_config: dict):
    """
    Run demo with completely custom configuration.
    
    Args:
        custom_config: User-defined configuration dictionary
    """
    print(" Running Qlib Standard Framework Demo with Custom Config")
    print("=" * 60)
    
    framework = QlibStandardFramework(config=custom_config)
    
    try:
        results = framework.run_complete_pipeline(
            save_results=True,
            output_dir="qlib_custom_results"
        )
        
        _print_results_summary(results)
        
        return results
        
    except Exception as e:
        print(f" Demo failed: {str(e)}")
        raise


def _print_results_summary(results: dict):
    """Print comprehensive results summary."""
    
    # Pipeline status
    status = results['pipeline_status']
    print(" Pipeline Status:")
    for step, completed in status.items():
        icon = "" if completed else ""
        print(f"   {icon} {step.replace('_', ' ').title()}")
    
    # Data summary
    data_summary = results['data_summary']
    if data_summary:
        print("\n Data Summary:")
        if 'raw_data' in data_summary:
            raw_info = data_summary['raw_data']
            print(f"    Raw Data: {raw_info['shape']} points")
            print(f"   🏢 Instruments: {raw_info['instruments']}")
            print(f"    Date Range: {raw_info['date_range'][0]} to {raw_info['date_range'][1]}")
        
        if 'factor_data' in data_summary:
            factor_info = data_summary['factor_data']
            print(f"   🧮 Features: {factor_info['feature_count']}")
    
    # Model summary  
    model_summary = results['model_summary']
    if model_summary:
        print(f"\n Model Summary:")
        print(f"   Type: {model_summary.get('model_type', 'Unknown')}")
        print(f"   Fitted: {model_summary.get('is_fitted', False)}")
        
        if 'top_features' in model_summary:
            print("    Top Features:")
            for i, (feature, importance) in enumerate(list(model_summary['top_features'].items())[:5]):
                print(f"      {i+1}. {feature}: {importance:.4f}")
    
    # Backtest summary
    backtest_summary = results['backtest_summary']
    if backtest_summary:
        print(f"\n Backtest Performance:")
        metrics = [
            ('Annual Return', 'annual_return', ':.2%'),
            ('Sharpe Ratio', 'sharpe_ratio', ':.3f'),
            ('Max Drawdown', 'max_drawdown', ':.2%'),
            ('Volatility', 'volatility', ':.2%')
        ]
        
        for label, key, fmt in metrics:
            value = backtest_summary.get(key, 0)
            formatted_value = f"{value:{fmt}}"
            print(f"   {label}: {formatted_value}")


def compare_models_demo():
    """
    Demo comparing different model types.
    """
    print(" Running Model Comparison Demo")
    print("=" * 50)
    
    models_to_test = [
        ('lightgbm', {
            'objective': 'regression',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'num_boost_round': 100,
            'verbosity': -1
        }),
        ('linear', {}),
        ('ridge', {'alpha': 1.0}),
        ('random_forest', {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        })
    ]
    
    base_config = {
        'data': {
            'source_type': 'synthetic',
            'instruments': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'start_time': '2022-01-01',
            'end_time': '2023-12-31',
            'freq': '1D'
        },
        'factors': {
            'enabled': True,
            'factor_list': ['rsi_14', 'sma_20', 'ema_12', 'momentum_5', 'volume_sma_20']
        },
        'preprocessing': {
            'train_test_split': 0.8
        },
        'strategy': {
            'type': 'long_short',
            'top_k': 2,
            'bottom_k': 2
        }
    }
    
    comparison_results = {}
    
    for model_type, model_config in models_to_test:
        print(f"\n Testing {model_type.upper()} model...")
        
        config = base_config.copy()
        config['model'] = {
            'type': model_type,
            'config': model_config
        }
        
        try:
            framework = QlibStandardFramework(config=config)
            results = framework.run_complete_pipeline(save_results=False)
            
            # Extract key metrics
            backtest_summary = results['backtest_summary']
            comparison_results[model_type] = {
                'annual_return': backtest_summary.get('annual_return', 0),
                'sharpe_ratio': backtest_summary.get('sharpe_ratio', 0),
                'max_drawdown': backtest_summary.get('max_drawdown', 0)
            }
            
            print(f"    {model_type}: "
                  f"Return: {comparison_results[model_type]['annual_return']:.2%}, "
                  f"Sharpe: {comparison_results[model_type]['sharpe_ratio']:.3f}")
            
        except Exception as e:
            print(f"    {model_type} failed: {str(e)}")
            comparison_results[model_type] = None
    
    # Print comparison summary
    print(f"\n Model Comparison Summary:")
    print("-" * 50)
    
    for model_type, metrics in comparison_results.items():
        if metrics:
            print(f"{model_type.upper():12} | "
                  f"Return: {metrics['annual_return']:7.2%} | "
                  f"Sharpe: {metrics['sharpe_ratio']:6.3f} | "
                  f"Drawdown: {metrics['max_drawdown']:7.2%}")
        else:
            print(f"{model_type.upper():12} | Failed")
    
    return comparison_results


if __name__ == "__main__":
    # Run synthetic data demo
    try:
        print("Starting Synthetic Data Demo...\n")
        synthetic_results = run_synthetic_data_demo()
        
        print("\n" + "="*60)
        print("Starting Model Comparison Demo...\n")
        comparison_results = compare_models_demo()
        
        print(f"\n- All demos completed successfully!")
        
    except Exception as e:
        print(f" Demo execution failed: {str(e)}")
