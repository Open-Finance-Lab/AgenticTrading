"""
ML Model Configurations for Financial Trading
Multiple model options with different complexity levels
"""

from data_interfaces import ModelInput

# 1. LightGBM - Advanced Gradient Boosting (Recommended for production)
LIGHTGBM_ADVANCED = ModelInput(
    model_name="lightgbm_advanced",
    model_type="tree",
    implementation="lightgbm",
    model_class="LGBMRegressor",
    target_type="market_neutral",
    hyperparameters={
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 0.05,
        "min_child_samples": 30,
        "min_split_gain": 0.01,
        "num_leaves": 31,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "early_stopping_rounds": 50,
        "eval_metric": "rmse"
    },
    training_method="rolling",
    training_period=504,  # 2 years hourly
    validation_period=126,  # 6 months
    rebalance_frequency="hourly"
)

# 2. XGBoost - Alternative Gradient Boosting
XGBOOST_MODEL = ModelInput(
    model_name="xgboost_model",
    model_type="tree",
    implementation="sklearn",
    model_class="XGBRegressor",
    target_type="market_neutral",
    hyperparameters={
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_weight": 3,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 30
    },
    training_method="rolling",
    training_period=504,
    validation_period=126,
    rebalance_frequency="hourly"
)

# 3. Random Forest - Ensemble Method
RANDOM_FOREST_MODEL = ModelInput(
    model_name="random_forest_model",
    model_type="ensemble",
    implementation="sklearn",
    model_class="RandomForestRegressor",
    target_type="market_neutral",
    hyperparameters={
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
        "oob_score": True
    },
    training_method="rolling",
    training_period=504,
    validation_period=126,
    rebalance_frequency="hourly"
)

# 4. Neural Network - Deep Learning Approach
NEURAL_NETWORK_MODEL = ModelInput(
    model_name="neural_network_model",
    model_type="neural",
    implementation="sklearn",
    model_class="MLPRegressor",
    target_type="market_neutral",
    hyperparameters={
        "hidden_layer_sizes": (100, 50, 25),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "learning_rate": "adaptive",
        "max_iter": 500,
        "random_state": 42,
        "early_stopping": True,
        "validation_fraction": 0.1
    },
    training_method="rolling",
    training_period=504,
    validation_period=126,
    rebalance_frequency="hourly"
)

# 5. Linear Model - Simple and Interpretable
LINEAR_MODEL = ModelInput(
    model_name="linear_model",
    model_type="linear",
    implementation="sklearn",
    model_class="LinearRegression",
    target_type="market_neutral",
    hyperparameters={
        "fit_intercept": True,
        "normalize": False
    },
    training_method="rolling",
    training_period=504,
    validation_period=126,
    rebalance_frequency="hourly"
)

# 6. Ridge Regression - Regularized Linear
RIDGE_MODEL = ModelInput(
    model_name="ridge_model",
    model_type="linear",
    implementation="sklearn",
    model_class="Ridge",
    target_type="market_neutral",
    hyperparameters={
        "alpha": 1.0,
        "fit_intercept": True,
        "normalize": False,
        "random_state": 42
    },
    training_method="rolling",
    training_period=504,
    validation_period=126,
    rebalance_frequency="hourly"
)

# 7. Elastic Net - Combined L1/L2 Regularization
ELASTIC_NET_MODEL = ModelInput(
    model_name="elastic_net_model",
    model_type="linear",
    implementation="sklearn",
    model_class="ElasticNet",
    target_type="market_neutral",
    hyperparameters={
        "alpha": 0.01,
        "l1_ratio": 0.5,
        "fit_intercept": True,
        "normalize": False,
        "random_state": 42,
        "max_iter": 1000
    },
    training_method="rolling",
    training_period=504,
    validation_period=126,
    rebalance_frequency="hourly"
)

# 8. Support Vector Regression - Non-linear
SVR_MODEL = ModelInput(
    model_name="svr_model",
    model_type="svm",
    implementation="sklearn",
    model_class="SVR",
    target_type="market_neutral",
    hyperparameters={
        "kernel": "rbf",
        "C": 1.0,
        "epsilon": 0.1,
        "gamma": "scale"
    },
    training_method="rolling",
    training_period=504,
    validation_period=126,
    rebalance_frequency="hourly"
)

# Model selection dictionary
MODEL_CONFIGS = {
    "lightgbm": LIGHTGBM_ADVANCED,
    "xgboost": XGBOOST_MODEL,
    "random_forest": RANDOM_FOREST_MODEL,
    "neural_network": NEURAL_NETWORK_MODEL,
    "linear": LINEAR_MODEL,
    "ridge": RIDGE_MODEL,
    "elastic_net": ELASTIC_NET_MODEL,
    "svr": SVR_MODEL
}

def get_model_config(model_name: str) -> ModelInput:
    """Get model configuration by name"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]

def list_available_models() -> list:
    """List all available model configurations"""
    return list(MODEL_CONFIGS.keys())

def get_model_complexity(model_name: str) -> str:
    """Get model complexity level"""
    complexity_map = {
        "linear": "Low",
        "ridge": "Low", 
        "elastic_net": "Low",
        "svr": "Medium",
        "random_forest": "Medium",
        "xgboost": "High",
        "lightgbm": "High",
        "neural_network": "High"
    }
    return complexity_map.get(model_name, "Unknown")

if __name__ == "__main__":
    print(" Available ML Models for Financial Trading:")
    print("=" * 50)
    
    for name, config in MODEL_CONFIGS.items():
        complexity = get_model_complexity(name)
        print(f" {name.upper()}: {complexity} complexity")
        print(f"   Class: {config.model_class}")
        print(f"   Type: {config.model_type}")
        print(f"   Target: {config.target_type}")
        print()
    
    print("💡 Usage: from ml_model_configs import get_model_config")
    print("💡 Example: model_config = get_model_config('lightgbm')")
