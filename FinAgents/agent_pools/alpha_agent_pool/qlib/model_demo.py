"""
Model Validation Demo using ModelInput Interface

This demo shows how to use the model pipeline to validate ML models
with the FactorInput approach, using real CSV data.
"""

from model_pipeline import ModelEvaluator
from interfaces import StandardAcceptanceCriteria, BacktestInterface, EvaluationMetrics, ModelInterface
from data_interfaces import ModelInput
from utils import QlibConfig, DataProcessor, ResultProcessor
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
try:
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
except ImportError as e:
    print(f"Warning: Missing ML libraries: {e}")

class ModelInputAdapter:
    """
    Adapter to convert ModelInput specifications into executable model training/prediction
    """
    
    def __init__(self, model_input: ModelInput):
        """
        Initialize adapter with ModelInput specification
        
        Args:
            model_input: ModelInput object containing model specification
        """
        self.model_input = model_input
    
    @property
    def model_name(self) -> str:
        """Get model name"""
        return self.model_input.model_name
    
    @property
    def model_type(self) -> str:
        """Get model type"""
        return self.model_input.model_type
    
    @property
    def model_description(self) -> str:
        """Get model description"""
        return f"{self.model_input.model_type} model: {self.model_input.model_name}"
    
    def train(self, features: pd.DataFrame, targets: pd.Series) -> Any:
        """
        Train model based on ModelInput specification
        
        Args:
            features: Feature matrix
            targets: Target values
            
        Returns:
            Trained model object
        """
        if self.model_input.implementation == "lightgbm":
            return self._train_lightgbm(features, targets)
        elif self.model_input.implementation == "sklearn":
            return self._train_sklearn(features, targets)
        else:
            raise ValueError(f"Unsupported implementation: {self.model_input.implementation}")
    
    def _train_lightgbm(self, features: pd.DataFrame, targets: pd.Series) -> Any:
        """Train LightGBM model"""
        # Clean data
        aligned_data = pd.concat([features, targets], axis=1).dropna()
        if len(aligned_data) < 50:
            raise ValueError("Insufficient training data")
        
        X = aligned_data.iloc[:, :-1]
        y = aligned_data.iloc[:, -1]
        
        # Create LightGBM model
        model = lgb.LGBMRegressor(**self.model_input.hyperparameters)
        
        # Train model
        model.fit(X, y)
        
        return model
    
    def _train_sklearn(self, features: pd.DataFrame, targets: pd.Series) -> Any:
        """Train sklearn model"""
        # Clean data
        aligned_data = pd.concat([features, targets], axis=1).dropna()
        if len(aligned_data) < 50:
            raise ValueError("Insufficient training data")
        
        X = aligned_data.iloc[:, :-1]
        y = aligned_data.iloc[:, -1]
        
        # Create sklearn model
        if self.model_input.model_class == "RandomForestRegressor":
            model = RandomForestRegressor(**self.model_input.hyperparameters)
        elif self.model_input.model_class == "LinearRegression":
            model = LinearRegression(**self.model_input.hyperparameters)
        else:
            raise ValueError(f"Unsupported sklearn model: {self.model_input.model_class}")
        
        # Train model
        model.fit(X, y)
        
        return model
    
    def predict(self, model: Any, features: pd.DataFrame) -> pd.Series:
        """
        Generate predictions using trained model
        
        Args:
            model: Trained model object
            features: Feature matrix
            
        Returns:
            Predictions
        """
        # Clean features
        clean_features = features.dropna()
        if len(clean_features) == 0:
            return pd.Series(dtype=float)
        
        # Generate predictions
        predictions = model.predict(clean_features)
        
        return pd.Series(predictions, index=clean_features.index)
    
    def validate_model(self, model: Any, validation_data: Tuple[pd.DataFrame, pd.Series]) -> Dict[str, Any]:
        """
        Validate trained model
        
        Args:
            model: Trained model object
            validation_data: Tuple of (features, targets) for validation
            
        Returns:
            Validation metrics
        """
        features, targets = validation_data
        
        # Generate predictions
        predictions = self.predict(model, features)
        
        # Align predictions and targets
        aligned_data = pd.concat([predictions, targets], axis=1).dropna()
        if len(aligned_data) < 10:
            return {
                'mse': float('inf'),
                'mae': float('inf'),
                'accuracy': 0.0,
                'correlation': 0.0
            }
        
        pred_values = aligned_data.iloc[:, 0]
        true_values = aligned_data.iloc[:, 1]
        
        # Calculate metrics
        mse = mean_squared_error(true_values, pred_values)
        mae = mean_absolute_error(true_values, pred_values)
        correlation = pred_values.corr(true_values)
        
        # Binary accuracy (direction prediction)
        pred_direction = (pred_values > 0).astype(int)
        true_direction = (true_values > 0).astype(int)
        accuracy = (pred_direction == true_direction).mean()
        
        return {
            'mse': mse,
            'mae': mae,
            'accuracy': accuracy if not pd.isna(accuracy) else 0.0,
            'correlation': correlation if not pd.isna(correlation) else 0.0
        }

class CSVModelBacktester(BacktestInterface):
    """
    Custom model backtester that works with CSV data files
    """
    
    def __init__(self, 
                 config: QlibConfig,
                 acceptance_criteria: Optional[StandardAcceptanceCriteria] = None):
        """
        Initialize CSV Model Backtester
        
        Args:
            config: Qlib configuration object
            acceptance_criteria: Criteria for accepting models
        """
        self.config = config
        self.acceptance_criteria = acceptance_criteria
        self.data_processor = DataProcessor(config)
        self.result_processor = ResultProcessor()
        
        # Don't initialize Qlib since we're using CSV data directly
    
    def prepare_data(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """
        Prepare market data and features from CSV files
        """
        instruments = kwargs.get('instruments', self.config.instruments)
        
        all_data = []
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        for instrument in instruments:
            # Load daily data from CSV
            csv_file = Path(self.config.provider_uri) / "stock_backup" / f"{instrument}_daily.csv"
            
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
                df = df.set_index('Date')
                
                # Filter by date range
                mask = (df.index >= start_dt) & (df.index <= end_dt)
                df = df[mask]
                
                # Rename columns to match Qlib format
                df = df.rename(columns={
                    'Open': '$open',
                    'High': '$high', 
                    'Low': '$low',
                    'Close': '$close',
                    'Volume': '$volume'
                })
                
                # Add instrument identifier
                df['instrument'] = instrument
                df = df.reset_index()
                all_data.append(df)
        
        if not all_data:
            raise ValueError("No data found for the specified instruments")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.set_index(['Date', 'instrument'])
        
        # Print raw data snapshot for provenance
        print("\n[Data Snapshot] Loaded raw market data for model training:")
        print(combined_data.head(10))
        print(f"Data shape: {combined_data.shape}")
        print(f"Data columns: {list(combined_data.columns)}")
        
        # Process and create features
        processed_data = self.data_processor.add_returns(combined_data)
        feature_data = self.data_processor.create_technical_features(processed_data)
        
        return feature_data
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    model_adapter: ModelInputAdapter,
                    **kwargs) -> Dict[str, Any]:
        """
        Run model backtesting with training and prediction
        """
        print(f"\n[Model Training] Training {model_adapter.model_name}...")
        
        # Prepare features and targets
        features, targets = self._prepare_features_targets(data)
        
        # Split data for training and testing
        split_point = int(len(data) * 0.7)  # 70% for training
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]
        
        train_features, train_targets = self._prepare_features_targets(train_data)
        test_features, test_targets = self._prepare_features_targets(test_data)
        
        # Train model
        try:
            trained_model = model_adapter.train(train_features, train_targets)
            print(f"Model training completed. Training data shape: {train_features.shape}")
        except Exception as e:
            print(f"Model training failed: {e}")
            return {
                'model': None,
                'train_predictions': pd.Series(dtype=float),
                'test_predictions': pd.Series(dtype=float),
                'validation_metrics': {'mse': float('inf'), 'mae': float('inf'), 'accuracy': 0.0, 'correlation': 0.0}
            }
        
        # Generate predictions
        train_predictions = model_adapter.predict(trained_model, train_features)
        test_predictions = model_adapter.predict(trained_model, test_features)
        
        # Validate model
        validation_metrics = model_adapter.validate_model(trained_model, (test_features, test_targets))
        
        print(f"[Model Validation] Test set performance:")
        print(f"- MSE: {validation_metrics['mse']:.6f}")
        print(f"- MAE: {validation_metrics['mae']:.6f}")
        print(f"- Direction Accuracy: {validation_metrics['accuracy']:.2%}")
        print(f"- Correlation: {validation_metrics['correlation']:.4f}")
        
        return {
            'model': trained_model,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'validation_metrics': validation_metrics,
            'train_targets': train_targets,
            'test_targets': test_targets,
            'model_metrics': validation_metrics  # Store model metrics separately
        }
    
    def _prepare_features_targets(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and targets from market data
        """
        # Select feature columns (technical indicators)
        feature_cols = [col for col in data.columns if col.startswith(('ma_', 'momentum_', 'volatility_', 'volume_', 'hl_', 'rsi'))]
        
        # If no technical features, create basic ones
        if not feature_cols:
            # Basic momentum features
            features = pd.DataFrame(index=data.index)
            for period in [5, 10, 20]:
                features[f'momentum_{period}'] = data.groupby('instrument')['$close'].pct_change(periods=period)
                features[f'ma_ratio_{period}'] = data['$close'] / data.groupby('instrument')['$close'].rolling(period).mean() - 1
            
            # Volume features
            if '$volume' in data.columns:
                features['volume_ratio'] = data['$volume'] / data.groupby('instrument')['$volume'].rolling(20).mean()
        else:
            features = data[feature_cols]
        
        # Target: next period return
        targets = data.groupby('instrument')['$close'].pct_change(periods=1).shift(-1)
        
        return features, targets
    
    def evaluate_performance(self, results: Dict[str, Any]) -> EvaluationMetrics:
        """
        Evaluate model backtest performance
        """
        test_predictions = results.get('test_predictions', pd.Series(dtype=float))
        validation_metrics = results.get('validation_metrics', {})
        
        # Calculate strategy returns based on predictions
        strategy_returns = self._calculate_strategy_returns(results)
        
        # Calculate performance metrics
        performance_metrics = self.result_processor.calculate_metrics(strategy_returns)
        
        # Create EvaluationMetrics with model-specific data
        return EvaluationMetrics(
            annual_return=performance_metrics.annual_return,
            cumulative_return=performance_metrics.cumulative_return,
            sharpe_ratio=performance_metrics.sharpe_ratio,
            max_drawdown=performance_metrics.max_drawdown,
            volatility=performance_metrics.volatility,
            downside_risk=performance_metrics.downside_risk,
            calmar_ratio=performance_metrics.calmar_ratio,
            # Store model-specific metrics in accessible attributes
            accuracy=validation_metrics.get('accuracy', 0.0),
            precision=None,  # Could be added if needed
            recall=None,     # Could be added if needed
            # Note: mse and correlation will be stored separately
        )
    
    def _calculate_strategy_returns(self, results: Dict[str, Any]) -> pd.Series:
        """
        Calculate strategy returns based on model predictions
        """
        test_predictions = results.get('test_predictions', pd.Series(dtype=float))
        test_targets = results.get('test_targets', pd.Series(dtype=float))
        
        if test_predictions.empty or test_targets.empty:
            return pd.Series(dtype=float)
        
        # Align predictions and targets
        aligned_data = pd.concat([test_predictions, test_targets], axis=1).dropna()
        
        if len(aligned_data) == 0:
            return pd.Series(dtype=float)
        
        predictions = aligned_data.iloc[:, 0]
        actual_returns = aligned_data.iloc[:, 1]
        
        # Simple strategy: long when prediction > 0, short when prediction < 0
        strategy_returns = predictions.apply(np.sign) * actual_returns
        
        return strategy_returns

# Configuration
config = QlibConfig(
    provider_uri="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data",
    instruments=["AAPL", "MSFT"],
    basic_fields=["$open", "$high", "$low", "$close", "$volume"],
    freq="day"
)

# Acceptance criteria for models
acceptance_criteria = StandardAcceptanceCriteria(
    min_annual_return=0.05,    # 5% minimum annual return
    min_sharpe_ratio=0.8,      # 0.8 minimum Sharpe ratio
    max_drawdown_threshold=0.25  # 25% maximum drawdown
)

# Initialize backtester and evaluator
backtester = CSVModelBacktester(config, acceptance_criteria)
evaluator = ModelEvaluator(acceptance_criteria)

# Define multiple models using ModelInput approach
model_configs = [
    ModelInput(
        model_name="enhanced_lgb_model",     # Model identifier name
        model_type="tree",                   # Tree-based model type
        implementation="lightgbm",           # Use LightGBM implementation
        model_class="LGBMRegressor",         # Regression model class
        hyperparameters={                    # Model hyperparameters
            "n_estimators": 100,             # Number of boosting rounds
            "learning_rate": 0.02,           # Learning rate for gradient boosting
            "max_depth": 4,                  # Maximum tree depth
            "random_state": 2025             # Random seed for reproducibility
        }
    ),
    ModelInput(
        model_name="random_forest_model",
        model_type="tree",
        implementation="sklearn",
        model_class="RandomForestRegressor",
        hyperparameters={
            "n_estimators": 50,
            "max_depth": 5,
            "random_state": 2025
        }
    ),
    ModelInput(
        model_name="linear_regression_model",
        model_type="linear",
        implementation="sklearn",
        model_class="LinearRegression",
        hyperparameters={}
    )
]

# Convert ModelInput to executable models
models = [ModelInputAdapter(model_config) for model_config in model_configs]

print("Starting model evaluation with Qlib pipeline...")
print(f"Instruments: {config.instruments}")
print(f"Date range: 2022-08-01 to 2023-12-31")
print(f"Number of models to evaluate: {len(models)}")

# Run model evaluation for each model
all_results = []

for i, model in enumerate(models):
    print(f"\n{'='*20} Evaluating Model {i+1}/{len(models)} {'='*20}")
    print(f"Model: {model.model_name} ({model_configs[i].model_type})")
    
    try:
        result = evaluator.evaluate_model(
            backtester=backtester,
            model=model,
            start_date="2022-08-01",
            end_date="2023-12-31"
        )
        
        # Store result for comparison
        all_results.append(result)
        
        # Display individual model results
        print(f"Model Name: {result.get('model_name', 'Unknown')}")
        print(f"Model Accepted: {result.get('is_accepted', False)}")
        
        if 'metrics' in result:
            metrics = result['metrics']
            model_metrics = result.get('model_metrics', {})
            print(f"- Annual Return: {getattr(metrics, 'annual_return', 0):.2%}")
            print(f"- Sharpe Ratio: {getattr(metrics, 'sharpe_ratio', 0):.2f}")
            print(f"- Max Drawdown: {getattr(metrics, 'max_drawdown', 0):.2%}")
            print(f"- Direction Accuracy: {getattr(metrics, 'accuracy', 0):.2%}")
            print(f"- Prediction Correlation: {model_metrics.get('correlation', 0):.4f}")
            print(f"- MSE: {model_metrics.get('mse', 0):.6f}")
            print(f"- MAE: {model_metrics.get('mae', 0):.6f}")
    
    except Exception as e:
        print(f"Error evaluating model {model.model_name}: {e}")
        all_results.append(None)

# Summary of all model evaluations
print("\n" + "="*60)
print("COMPREHENSIVE MODEL EVALUATION SUMMARY")
print("="*60)

accepted_models = []
rejected_models = []

for i, result in enumerate(all_results):
    if result is not None:
        model_name = result.get('model_name', f'Model_{i+1}')
        is_accepted = result.get('is_accepted', False)
        
        if is_accepted:
            accepted_models.append((model_name, result))
        else:
            rejected_models.append((model_name, result))

print(f"\nResults Overview:")
print(f"- Total models evaluated: {len(model_configs)}")
print(f"- Accepted models: {len(accepted_models)}")
print(f"- Rejected models: {len(rejected_models)}")

if accepted_models:
    print(f"\n ACCEPTED MODELS:")
    for model_name, result in accepted_models:
        metrics = result.get('metrics')
        if metrics:
            print(f"  • {model_name}")
            print(f"    - Annual Return: {getattr(metrics, 'annual_return', 0):.2%}")
            print(f"    - Sharpe Ratio: {getattr(metrics, 'sharpe_ratio', 0):.2f}")
            print(f"    - Direction Accuracy: {getattr(metrics, 'accuracy', 0):.2%}")

if rejected_models:
    print(f"\n REJECTED MODELS:")
    for model_name, result in rejected_models:
        metrics = result.get('metrics')
        if metrics:
            print(f"  • {model_name}")
            print(f"    - Annual Return: {getattr(metrics, 'annual_return', 0):.2%}")
            print(f"    - Sharpe Ratio: {getattr(metrics, 'sharpe_ratio', 0):.2f}")
            print(f"    - Direction Accuracy: {getattr(metrics, 'accuracy', 0):.2%}")

# Model comparison analysis
if len(all_results) > 1:
    print(f"\n MODEL COMPARISON:")
    print(f"{'Model Name':<20} {'Return':<8} {'Sharpe':<8} {'Accuracy':<10} {'MSE':<10} {'Status':<10}")
    print("-" * 70)
    
    for i, result in enumerate(all_results):
        if result is not None:
            model_name = result.get('model_name', f'Model_{i+1}')[:19]
            metrics = result.get('metrics')
            model_metrics = result.get('model_metrics', {})
            if metrics:
                ret = getattr(metrics, 'annual_return', 0)
                sharpe = getattr(metrics, 'sharpe_ratio', 0)
                accuracy = getattr(metrics, 'accuracy', 0)
                mse = model_metrics.get('mse', 0)
                status = " PASS" if result.get('is_accepted', False) else " FAIL"
                print(f"{model_name:<20} {ret:<8.2%} {sharpe:<8.2f} {accuracy:<10.2%} {mse:<10.6f} {status:<10}")

print(f"\n RECOMMENDATION:")
if accepted_models:
    best_model = max(accepted_models, key=lambda x: getattr(x[1].get('metrics'), 'sharpe_ratio', 0))
    print(f"Best performing model: {best_model[0]}")
    print(f"Consider deploying this model for live trading.")
else:
    print("No models passed acceptance criteria.")
    print("Consider:")
    print("- Tuning hyperparameters")
    print("- Adding more features")
    print("- Using different model architectures")
    print("- Extending training period")
    print("- Relaxing acceptance criteria for initial testing")
