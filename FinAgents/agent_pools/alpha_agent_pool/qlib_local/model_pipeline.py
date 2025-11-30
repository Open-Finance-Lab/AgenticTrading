"""
Model Backtesting Pipeline

This module implements model-specific backtesting functionality.
It evaluates machine learning models for quantitative trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import qlib
from qlib.data import D
from qlib.constant import REG_CN

try:
    from .interfaces import BacktestInterface, ModelInterface, EvaluationMetrics, AcceptanceCriteria
    from .utils import QlibConfig, DataProcessor, ResultProcessor
except ImportError:
    from interfaces import BacktestInterface, ModelInterface, EvaluationMetrics, AcceptanceCriteria
    from utils import QlibConfig, DataProcessor, ResultProcessor


class ModelBacktester(BacktestInterface):
    """
    Backtester specifically designed for evaluating machine learning models
    """
    
    def __init__(self, 
                 config: QlibConfig,
                 acceptance_criteria: Optional[AcceptanceCriteria] = None):
        """
        Initialize Model Backtester
        
        Args:
            config: Qlib configuration object
            acceptance_criteria: Criteria for accepting models
        """
        self.config = config
        self.acceptance_criteria = acceptance_criteria
        self.data_processor = DataProcessor(config)
        self.result_processor = ResultProcessor()
        
        # Initialize Qlib
        qlib.init(provider_uri=config.provider_uri, region=REG_CN)
    
    def prepare_data(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """
        Prepare feature data for model backtesting
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional parameters (factors, instruments, etc.)
            
        Returns:
            Prepared feature matrix with targets
        """
        instruments = kwargs.get('instruments', self.config.instruments)
        
        # Get basic market data
        basic_fields = ['$open', '$high', '$low', '$close', '$volume', '$factor']
        market_data = D.features(
            instruments=instruments,
            fields=basic_fields,
            start_time=start_date,
            end_time=end_date,
            freq=self.config.freq
        )
        
        # Process market data
        market_data = self.data_processor.clean_data(market_data)
        
        # Add factor features if provided
        factor_features = kwargs.get('factor_features', None)
        if factor_features is not None:
            # Combine market data with factor features
            feature_data = pd.concat([market_data, factor_features], axis=1)
        else:
            # Use basic technical indicators as features
            feature_data = self.data_processor.create_technical_features(market_data)
        
        # Create target variable (forward returns)
        target_horizon = kwargs.get('target_horizon', 1)  # Days ahead
        targets = self.data_processor.create_targets(market_data, target_horizon)
        
        # Combine features and targets
        data = pd.concat([feature_data, targets], axis=1)
        data = data.dropna()
        
        return data
    
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    model: ModelInterface,
                    **kwargs) -> Dict[str, Any]:
        """
        Run backtesting for a specific model
        
        Args:
            data: Prepared data with features and targets
            model: Model to be backtested
            **kwargs: Additional backtesting parameters
            
        Returns:
            Dictionary containing backtest results
        """
        try:
            # Split data into train/validation/test
            train_ratio = kwargs.get('train_ratio', 0.6)
            val_ratio = kwargs.get('val_ratio', 0.2)
            
            train_data, val_data, test_data = self._split_data(data, train_ratio, val_ratio)
            
            # Extract features and targets
            feature_cols = [col for col in data.columns if col != 'targets']
            
            X_train = train_data[feature_cols]
            y_train = train_data['targets']
            X_val = val_data[feature_cols]
            y_val = val_data['targets']
            X_test = test_data[feature_cols]
            y_test = test_data['targets']
            
            # Train the model
            model.train(X_train, y_train, validation_data=(X_val, y_val), **kwargs)
            
            # Validate model performance
            val_metrics = model.validate_model(X_val, y_val)
            
            # Generate out-of-sample predictions
            predictions = model.predict(X_test)
            
            # Calculate trading strategy performance
            strategy_results = self._calculate_strategy_performance(
                predictions, y_test, test_data.index, **kwargs
            )
            
            # Compile results
            results = {
                'model_name': model.model_name,
                'model_type': model.model_type,
                'validation_metrics': val_metrics,
                'predictions': predictions,
                'actual_returns': y_test,
                'strategy_results': strategy_results,
                'test_period': (test_data.index.min(), test_data.index.max()),
                'success': True
            }
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_name': model.model_name
            }
    
    def evaluate_performance(self, results: Dict[str, Any]) -> EvaluationMetrics:
        """
        Evaluate model performance and create standardized metrics
        
        Args:
            results: Raw backtest results from run_backtest
            
        Returns:
            Standardized evaluation metrics
        """
        if not results.get('success', False):
            # Return empty metrics for failed backtests
            return EvaluationMetrics(
                annual_return=0.0,
                cumulative_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                downside_risk=0.0,
                calmar_ratio=0.0
            )
        
        # Extract strategy performance
        strategy_results = results.get('strategy_results', {})
        returns = strategy_results.get('returns', pd.Series())
        
        # Calculate standard metrics
        metrics = self.result_processor.calculate_metrics(returns)
        
        # Add model-specific metrics
        val_metrics = results.get('validation_metrics', {})
        metrics.accuracy = val_metrics.get('accuracy', None)
        metrics.precision = val_metrics.get('precision', None)
        metrics.recall = val_metrics.get('recall', None)
        
        return metrics
    
    def _split_data(self, data: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple:
        """
        Split data into training, validation, and test sets chronologically
        
        Args:
            data: Input data to split
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        data_sorted = data.sort_index()
        n = len(data_sorted)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data_sorted.iloc[:train_end]
        val_data = data_sorted.iloc[train_end:val_end]
        test_data = data_sorted.iloc[val_end:]
        
        return train_data, val_data, test_data
    
    def _calculate_strategy_performance(self, 
                                      predictions: pd.Series,
                                      actual_returns: pd.Series,
                                      timestamps: pd.Index,
                                      **kwargs) -> Dict[str, Any]:
        """
        Calculate trading strategy performance based on model predictions
        
        Args:
            predictions: Model predictions
            actual_returns: Actual forward returns
            timestamps: Time index for the predictions
            **kwargs: Strategy parameters
            
        Returns:
            Dictionary with strategy performance metrics
        """
        # Create strategy based on predictions
        strategy_type = kwargs.get('strategy_type', 'long_short')
        position_size = kwargs.get('position_size', 1.0)
        
        # Generate positions based on predictions
        if strategy_type == 'long_only':
            # Long positions for positive predictions
            positions = (predictions > 0).astype(float) * position_size
        elif strategy_type == 'long_short':
            # Normalize predictions to create position sizes
            pred_ranks = predictions.rank(pct=True)
            positions = (pred_ranks - 0.5) * 2 * position_size  # Scale to [-1, 1]
        else:
            # Direct use of normalized predictions
            positions = predictions / predictions.abs().max() * position_size
        
        # Calculate strategy returns
        strategy_returns = positions * actual_returns
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        return {
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'positions': positions,
            'hit_rate': (np.sign(predictions) == np.sign(actual_returns)).mean(),
            'correlation': predictions.corr(actual_returns)
        }


class ModelEvaluator:
    """
    Evaluator for comprehensive model analysis and acceptance testing
    """
    
    def __init__(self, acceptance_criteria: AcceptanceCriteria):
        """
        Initialize Model Evaluator
        
        Args:
            acceptance_criteria: Criteria for model acceptance
        """
        self.acceptance_criteria = acceptance_criteria
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate_model(self, 
                      backtester: ModelBacktester,
                      model: ModelInterface,
                      start_date: str,
                      end_date: str,
                      **kwargs) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            backtester: Model backtester instance
            model: Model to evaluate
            start_date: Evaluation start date
            end_date: Evaluation end date
            **kwargs: Additional evaluation parameters
            
        Returns:
            Complete evaluation results including acceptance decision
        """
        # Prepare data
        data = backtester.prepare_data(start_date, end_date, **kwargs)
        
        # Run backtest
        backtest_results = backtester.run_backtest(data, model, **kwargs)
        
        # Evaluate performance
        metrics = backtester.evaluate_performance(backtest_results)
        
        # Make acceptance decision
        is_accepted = self.acceptance_criteria.evaluate_model(metrics)
        
        # Compile evaluation results
        evaluation_result = {
            'model_name': model.model_name,
            'model_type': model.model_type,
            'evaluation_period': f"{start_date} to {end_date}",
            'metrics': metrics,
            'is_accepted': is_accepted,
            'backtest_results': backtest_results,
            'acceptance_criteria': self.acceptance_criteria.get_criteria_description(),
            'timestamp': pd.Timestamp.now()
        }
        
        # Store in history
        self.evaluation_history.append(evaluation_result)
        
        return evaluation_result
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """
        Get summary of all model evaluations
        
        Returns:
            DataFrame with evaluation summary
        """
        if not self.evaluation_history:
            return pd.DataFrame()
        
        summary_data = []
        for eval_result in self.evaluation_history:
            metrics = eval_result['metrics']
            summary_data.append({
                'model_name': eval_result['model_name'],
                'model_type': eval_result['model_type'],
                'is_accepted': eval_result['is_accepted'],
                'annual_return': metrics.annual_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'accuracy': metrics.accuracy,
                'evaluation_date': eval_result['timestamp']
            })
        
        return pd.DataFrame(summary_data)
    
    def get_accepted_models(self) -> List[str]:
        """
        Get list of accepted model names
        
        Returns:
            List of model names that passed acceptance criteria
        """
        return [
            eval_result['model_name'] 
            for eval_result in self.evaluation_history 
            if eval_result['is_accepted']
        ]
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance of all evaluated models
        
        Returns:
            DataFrame comparing model performance metrics
        """
        summary_df = self.get_evaluation_summary()
        if summary_df.empty:
            return pd.DataFrame()
        
        # Sort by sharpe ratio descending
        comparison_df = summary_df.sort_values('sharpe_ratio', ascending=False)
        
        return comparison_df[['model_name', 'model_type', 'is_accepted', 
                           'annual_return', 'sharpe_ratio', 'max_drawdown', 'accuracy']]
    
    def export_results(self, filepath: Union[str, Path]) -> None:
        """
        Export evaluation results to file
        
        Args:
            filepath: Path to save results
        """
        summary_df = self.get_evaluation_summary()
        summary_df.to_csv(filepath, index=False)
        print(f"Model evaluation results exported to {filepath}")


# Example model implementations
class ExampleLGBModel(ModelInterface):
    """Example implementation using LightGBM"""
    
    def __init__(self, **model_params):
        """Initialize with model parameters"""
        self.model_params = model_params
        self.model = None
        self.is_trained = False
    
    def train(self, features: pd.DataFrame, targets: pd.Series, **kwargs) -> None:
        """Train LightGBM model"""
        try:
            import lightgbm as lgb
            
            # Prepare validation data if provided
            validation_data = kwargs.get('validation_data', None)
            
            # Set default parameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'verbosity': -1
            }
            params.update(self.model_params)
            
            # Create datasets
            train_set = lgb.Dataset(features, label=targets)
            valid_sets = [train_set]
            
            if validation_data is not None:
                X_val, y_val = validation_data
                valid_set = lgb.Dataset(X_val, label=y_val)
                valid_sets.append(valid_set)
            
            # Train model
            self.model = lgb.train(
                params,
                train_set,
                valid_sets=valid_sets,
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            
            self.is_trained = True
            
        except ImportError:
            raise ImportError("LightGBM is required but not installed")
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Generate predictions using trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(features)
        return pd.Series(predictions, index=features.index)
    
    def validate_model(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Validate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before validation")
        
        predictions = self.predict(features)
        
        # Calculate validation metrics
        mse = ((predictions - targets) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = (predictions - targets).abs().mean()
        corr = predictions.corr(targets)
        
        # Calculate classification metrics (direction prediction)
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(targets)
        accuracy = (pred_direction == actual_direction).mean()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'correlation': corr,
            'accuracy': accuracy
        }
    
    @property
    def model_name(self) -> str:
        return "LightGBM_Regressor"
    
    @property
    def model_type(self) -> str:
        return "Tabular"


class ExampleLinearModel(ModelInterface):
    """Example implementation using Linear Regression"""
    
    def __init__(self, **model_params):
        """Initialize with model parameters"""
        self.model_params = model_params
        self.model = None
        self.is_trained = False
    
    def train(self, features: pd.DataFrame, targets: pd.Series, **kwargs) -> None:
        """Train linear regression model"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            
            # Initialize scaler and model
            self.scaler = StandardScaler()
            self.model = LinearRegression(**self.model_params)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train model
            self.model.fit(features_scaled, targets)
            self.is_trained = True
            
        except ImportError:
            raise ImportError("scikit-learn is required but not installed")
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Generate predictions using trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        return pd.Series(predictions, index=features.index)
    
    def validate_model(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Validate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before validation")
        
        predictions = self.predict(features)
        
        # Calculate validation metrics
        mse = ((predictions - targets) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = (predictions - targets).abs().mean()
        corr = predictions.corr(targets)
        
        # Calculate classification metrics (direction prediction)
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(targets)
        accuracy = (pred_direction == actual_direction).mean()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'correlation': corr,
            'accuracy': accuracy
        }
    
    @property
    def model_name(self) -> str:
        return "Linear_Regression"
    
    @property
    def model_type(self) -> str:
        return "Tabular"
