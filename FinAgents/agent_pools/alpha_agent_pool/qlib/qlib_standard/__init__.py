"""
Qlib Standard Package - Complete Qlib-compliant Framework

This package provides a complete, production-ready implementation of quantitative
trading research workflows using Qlib standard APIs and interfaces.

Key Components:
- QlibCSVDataLoader/QlibSyntheticDataLoader: Standard data loading
- QlibFactorCalculator: Feature engineering using Qlib Expression engine
- QlibModelTrainer: ML model training with Qlib Model interface
- QlibStrategyExecutor: Trading strategy execution with Qlib Strategy framework
- QlibDataHandler: Data preprocessing using Qlib DataHandler framework
- QlibStandardFramework: Complete end-to-end pipeline integration

Usage:
    from qlib_standard import QlibStandardFramework
    
    config = {
        'data': {'source_type': 'synthetic', ...},
        'model': {'type': 'lightgbm', ...},
        'strategy': {'type': 'long_short', ...}
    }
    
    framework = QlibStandardFramework(config=config)
    results = framework.run_complete_pipeline()

Features:
- Full Qlib API compliance and inheritance
- Comprehensive factor library (30+ technical indicators)
- Multiple ML algorithms (LightGBM, Linear, Ridge, Lasso, RandomForest)
- Risk-managed trading strategies
- Complete backtesting and performance analysis
- Robust data preprocessing and validation
- Production-ready error handling and logging
"""

__version__ = "1.0.0"
__author__ = "GitHub Copilot"

# Core imports
from .data_loader import QlibCSVDataLoader, QlibSyntheticDataLoader
from .factor_calculator import QlibFactorCalculator  
from .model_trainer import QlibModelTrainer
from .strategy_executor import QlibStrategyExecutor
from .data_handler import QlibDataHandler
from .framework import QlibStandardFramework

# Demo and utilities
try:
    from .demo import (
        run_synthetic_data_demo,
        run_csv_data_demo, 
        run_custom_config_demo,
        compare_models_demo
    )
except ImportError:
    # Demo module might not be available in production
    pass

__all__ = [
    # Core components
    'QlibCSVDataLoader',
    'QlibSyntheticDataLoader', 
    'QlibFactorCalculator',
    'QlibModelTrainer',
    'QlibStrategyExecutor',
    'QlibTopKStrategy',
    'QlibDataHandler',
    'QlibStandardFramework',
    
    # Demo functions
    'run_synthetic_data_demo',
    'run_csv_data_demo',
    'run_custom_config_demo', 
    'compare_models_demo'
]


def get_default_config():
    """
    Get a default configuration for the Qlib Standard Framework.
    
    Returns:
        dict: Default configuration that can be modified as needed
    """
    return {
        'data': {
            'source_type': 'synthetic',
            'instruments': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'start_time': '2022-01-01',
            'end_time': '2023-12-31',
            'freq': '1D'
        },
        
        'factors': {
            'enabled': True,
            'factor_list': None,  # Use all default factors
            'custom_factors': {}
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
            'top_k': 10,
            'bottom_k': 10,
            'max_position_weight': 0.1
        },
        
        'backtest': {
            'benchmark': 'SH000300',
            'account': 100000000
        }
    }


def quick_demo():
    """
    Run a quick demonstration of the framework with default settings.
    
    Returns:
        dict: Complete pipeline results
    """
    print(" Running Quick Qlib Standard Framework Demo")
    print("=" * 50)
    
    framework = QlibStandardFramework(config=get_default_config())
    
    try:
        results = framework.run_complete_pipeline(
            save_results=True,
            output_dir="quick_demo_results"
        )
        
        print("\n Quick demo completed!")
        print(" Results saved to: quick_demo_results/")
        
        return results
        
    except Exception as e:
        print(f" Quick demo failed: {str(e)}")
        raise


# Package metadata
PACKAGE_INFO = {
    'name': 'qlib_standard',
    'version': __version__,
    'description': 'Complete Qlib-compliant quantitative trading framework',
    'features': [
        'Full Qlib API compliance',
        'Comprehensive factor library',
        'Multiple ML algorithms',
        'Risk-managed strategies',
        'Complete backtesting',
        'Production-ready'
    ],
    'dependencies': [
        'qlib>=0.9.0',
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'scikit-learn>=1.0.0',
        'lightgbm>=3.3.0'
    ]
}
