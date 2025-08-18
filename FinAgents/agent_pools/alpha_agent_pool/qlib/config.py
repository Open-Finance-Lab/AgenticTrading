"""
Configuration file for Qlib backtesting pipeline settings
"""

# Default configuration for Qlib data and backtesting
DEFAULT_CONFIG = {
    "provider_uri": "qlib/qlib_data/cn_data",
    "instruments": "csi300", 
    "freq": "day",
    
    # Data fields
    "basic_fields": [
        "$open", "$high", "$low", "$close", "$volume", "$factor"
    ],
    
    # Time periods
    "train_start_date": "2008-01-01",
    "train_end_date": "2014-12-31", 
    "valid_start_date": "2015-01-01",
    "valid_end_date": "2016-12-31",
    "test_start_date": "2017-01-01", 
    "test_end_date": "2020-12-31",
    
    # Trading parameters
    "benchmark": "SH000300",
    "rebalance_frequency": "daily",
    "max_position_size": 0.1,
    "transaction_cost": 0.003
}

# Acceptance criteria presets
ACCEPTANCE_CRITERIA_PRESETS = {
    "strict": {
        "min_sharpe_ratio": 1.5,
        "max_drawdown_threshold": 0.15,
        "min_ic_mean": 0.03,
        "min_annual_return": 0.08
    },
    "standard": {
        "min_sharpe_ratio": 1.0,
        "max_drawdown_threshold": 0.20,
        "min_ic_mean": 0.02,
        "min_annual_return": 0.05
    },
    "relaxed": {
        "min_sharpe_ratio": 0.8,
        "max_drawdown_threshold": 0.25,
        "min_ic_mean": 0.01,
        "min_annual_return": 0.03
    }
}

# Factor evaluation settings
FACTOR_SETTINGS = {
    "min_validation_samples": 100,
    "ic_calculation_window": 20,
    "rebalance_frequency": 20,  # days
    "default_strategy_type": "long_short"
}

# Model evaluation settings  
MODEL_SETTINGS = {
    "default_train_ratio": 0.6,
    "default_val_ratio": 0.2,
    "default_target_horizon": 1,  # days
    "default_strategy_type": "long_short",
    "min_training_samples": 1000
}
