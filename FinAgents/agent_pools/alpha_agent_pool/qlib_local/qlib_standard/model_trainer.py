"""
Qlib Standard Model Trainer Implementation

This module implements model training using Qlib's Model interface
and supports various ML models with proper integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import warnings
import pickle
from abc import ABC, abstractmethod

# Qlib model imports
from qlib.model.base import Model
from qlib.data.dataset import Dataset, DatasetH

# ML model imports
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not available. Only basic models will be supported.")

try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("Scikit-learn not available. Model functionality will be limited.")


class QlibModelTrainer(Model):
    """
    Standard Qlib Model implementation supporting multiple ML algorithms.
    
    This class follows the Qlib Model interface and provides training,
    prediction, and evaluation capabilities for various ML models.
    """
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the Qlib model trainer.
        
        Args:
            model_type: Type of model ('lightgbm', 'linear', 'ridge', 'lasso', 'rf')
            model_config: Configuration parameters for the model
            **kwargs: Additional arguments
        """
        super().__init__()
        self.model_type = model_type.lower()
        self.model_config = model_config or {}
        self.model = None
        self.feature_columns = None
        self.label_columns = None
        self.fitted = False
        
        # Set default configurations for different model types
        self._set_default_configs()
        
        # Validate model type
        self._validate_model_type()
    
    def _set_default_configs(self) -> None:
        """Set default configurations for different model types."""
        default_configs = {
            'lightgbm': {
                'objective': 'regression',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'num_boost_round': 100,
                'early_stopping_rounds': 10,
                'valid_ratio': 0.2
            },
            'linear': {
                'fit_intercept': True
            },
            'ridge': {
                'alpha': 1.0,
                'fit_intercept': True
            },
            'lasso': {
                'alpha': 1.0,
                'fit_intercept': True,
                'max_iter': 1000
            },
            'rf': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        }
        
        # Merge with user config
        default_config = default_configs.get(self.model_type, {})
        default_config.update(self.model_config)
        self.model_config = default_config
    
    def _validate_model_type(self) -> None:
        """Validate that the requested model type is available."""
        if self.model_type == 'lightgbm' and not HAS_LIGHTGBM:
            raise ValueError("LightGBM is not available. Please install lightgbm package.")
        
        if self.model_type in ['linear', 'ridge', 'lasso', 'rf'] and not HAS_SKLEARN:
            raise ValueError("Scikit-learn is not available. Please install sklearn package.")
        
        supported_models = ['lightgbm', 'linear', 'ridge', 'lasso', 'rf']
        if self.model_type not in supported_models:
            raise ValueError(f"Unsupported model type: {self.model_type}. "
                           f"Supported types: {supported_models}")
    
    def fit(self, dataset: Dataset, **kwargs) -> None:
        """
        Train the model using Qlib Dataset interface.
        
        Args:
            dataset: Qlib Dataset object containing training data
            **kwargs: Additional training arguments
        """
        try:
            # Prepare training data
            train_data, valid_data = self._prepare_training_data(dataset)
            
            if train_data is None or train_data[0].empty:
                raise ValueError("No training data available")
            
            X_train, y_train = train_data
            X_valid, y_valid = valid_data if valid_data[0] is not None else (None, None)
            
            # Store feature and label information
            self.feature_columns = X_train.columns.tolist()
            self.label_columns = [y_train.name] if hasattr(y_train, 'name') else ['label']
            
            # Train model based on type
            self._fit_model(X_train, y_train, X_valid, y_valid)
            
            self.fitted = True
            
        except Exception as e:
            raise RuntimeError(f"Model training failed: {str(e)}")
    
    def _prepare_training_data(
        self, 
        dataset: Dataset
    ) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[Optional[pd.DataFrame], Optional[pd.Series]]]:
        """
        Prepare training and validation data from Qlib dataset.
        
        Args:
            dataset: Qlib Dataset object
            
        Returns:
            Tuple containing (train_data, valid_data) where each is (X, y)
        """
        # Get training data
        try:
            train_df = dataset.prepare("train", col_set=["feature", "label"])
        except Exception:
            # Fallback: try to get data without specific segments
            train_df = dataset.prepare(col_set=["feature", "label"])
        
        if train_df.empty:
            raise ValueError("No training data available from dataset")
        
        # Extract features and labels
        X_train = self._extract_features(train_df)
        y_train = self._extract_labels(train_df)
        
        # Get validation data if available
        X_valid, y_valid = None, None
        try:
            valid_df = dataset.prepare("valid", col_set=["feature", "label"])
            if not valid_df.empty:
                X_valid = self._extract_features(valid_df)
                y_valid = self._extract_labels(valid_df)
        except Exception:
            # No validation data available
            pass
        
        return (X_train, y_train), (X_valid, y_valid)
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract feature columns from multi-level DataFrame.
        
        Args:
            data: DataFrame with potential multi-level columns
            
        Returns:
            pd.DataFrame: Feature data
        """
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-level columns
            feature_cols = [col for col in data.columns if col[0] == 'feature']
            if feature_cols:
                features = data[feature_cols]
                # Flatten column names
                features.columns = [col[1] for col in features.columns]
                return features
            else:
                # No feature columns found, use all columns
                return data
        else:
            # Simple columns - assume all are features
            return data
    
    def _extract_labels(self, data: pd.DataFrame) -> pd.Series:
        """
        Extract label column from multi-level DataFrame.
        
        Args:
            data: DataFrame with potential multi-level columns
            
        Returns:
            pd.Series: Label data
        """
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-level columns
            label_cols = [col for col in data.columns if col[0] == 'label']
            if label_cols:
                # Use first label column
                label_col = label_cols[0]
                return data[label_col]
            else:
                raise ValueError("No label columns found in data")
        else:
            # Simple columns - look for common label names
            label_candidates = ['label', 'target', 'y']
            for candidate in label_candidates:
                if candidate in data.columns:
                    return data[candidate]
            
            # If no standard label found, use last column
            if len(data.columns) > 0:
                return data.iloc[:, -1]
            else:
                raise ValueError("No suitable label column found")
    
    def _fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None
    ) -> None:
        """
        Fit the actual ML model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_valid: Validation features (optional)
            y_valid: Validation labels (optional)
        """
        # Clean data
        X_train_clean, y_train_clean = self._clean_data(X_train, y_train)
        
        if X_valid is not None and y_valid is not None:
            X_valid_clean, y_valid_clean = self._clean_data(X_valid, y_valid)
        else:
            X_valid_clean, y_valid_clean = None, None
        
        if self.model_type == 'lightgbm':
            self._fit_lightgbm(X_train_clean, y_train_clean, X_valid_clean, y_valid_clean)
        elif self.model_type == 'linear':
            self.model = LinearRegression(**self.model_config)
            self.model.fit(X_train_clean, y_train_clean)
        elif self.model_type == 'ridge':
            self.model = Ridge(**self.model_config)
            self.model.fit(X_train_clean, y_train_clean)
        elif self.model_type == 'lasso':
            self.model = Lasso(**self.model_config)
            self.model.fit(X_train_clean, y_train_clean)
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(**self.model_config)
            self.model.fit(X_train_clean, y_train_clean)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _fit_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None
    ) -> None:
        """
        Fit LightGBM model with proper configuration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_valid: Validation features (optional)
            y_valid: Validation labels (optional)
        """
        # Prepare datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # Extract training parameters
        train_params = self.model_config.copy()
        num_boost_round = train_params.pop('num_boost_round', 100)
        early_stopping_rounds = train_params.pop('early_stopping_rounds', None)
        
        # Setup callbacks for newer LightGBM versions
        callbacks = []
        if early_stopping_rounds is not None and len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        
        # Train model with modern LightGBM API
        self.model = lgb.train(
            train_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks if callbacks else None
        )
    
    def _clean_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Clean training data by handling missing values and outliers.
        
        Args:
            X: Feature data
            y: Label data
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Cleaned feature and label data
        """
        # Align X and y indices
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        # Remove rows with missing labels
        label_mask = ~y_aligned.isna()
        X_clean = X_aligned[label_mask]
        y_clean = y_aligned[label_mask]
        
        # Handle missing features
        # For now, simple forward fill and then backward fill
        X_clean = X_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining rows with all NaN features
        feature_mask = ~X_clean.isna().all(axis=1)
        X_clean = X_clean[feature_mask]
        y_clean = y_clean[feature_mask]
        
        # Handle infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(0)  # Replace remaining NaN with 0
        
        # Clip extreme outliers (beyond 5 standard deviations)
        for col in X_clean.select_dtypes(include=[np.number]).columns:
            col_data = X_clean[col]
            if col_data.std() > 0:
                mean_val = col_data.mean()
                std_val = col_data.std()
                lower_bound = mean_val - 5 * std_val
                upper_bound = mean_val + 5 * std_val
                X_clean[col] = X_clean[col].clip(lower_bound, upper_bound)
        
        return X_clean, y_clean
    
    def predict(self, dataset: Dataset, segment: str = "test") -> pd.Series:
        """
        Generate predictions using the trained model.
        
        Args:
            dataset: Qlib Dataset object containing test data
            segment: Data segment to predict on (default: "test")
            
        Returns:
            pd.Series: Predictions with proper index
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Prepare test data
        try:
            test_df = dataset.prepare(segment, col_set=["feature"])
        except Exception:
            # Fallback: try to get all data
            test_df = dataset.prepare(col_set=["feature"])
        
        if test_df.empty:
            warnings.warn(f"No test data available for segment: {segment}")
            return pd.Series(dtype=float)
        
        # Extract features
        X_test = self._extract_features(test_df)
        
        # Ensure feature alignment
        if self.feature_columns:
            missing_features = set(self.feature_columns) - set(X_test.columns)
            extra_features = set(X_test.columns) - set(self.feature_columns)
            
            if missing_features:
                # Add missing features with zeros
                for feature in missing_features:
                    X_test[feature] = 0.0
                    
            if extra_features:
                # Drop extra features
                X_test = X_test.drop(columns=extra_features)
            
            # Reorder columns to match training
            X_test = X_test[self.feature_columns]
        
        # Clean test data
        X_test_clean = X_test.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X_test_clean = X_test_clean.replace([np.inf, -np.inf], 0)
        
        # Generate predictions
        if self.model_type == 'lightgbm':
            predictions = self.model.predict(X_test_clean, num_iteration=self.model.best_iteration)
        else:
            predictions = self.model.predict(X_test_clean)
        
        # Create predictions series with proper index
        pred_series = pd.Series(predictions, index=X_test.index, name='prediction')
        
        return pred_series
    
    def evaluate(
        self,
        dataset: Dataset,
        segment: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance on given dataset segment.
        
        Args:
            dataset: Qlib Dataset object
            segment: Data segment to evaluate on
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        # Get predictions
        predictions = self.predict(dataset, segment)
        
        # Get true labels
        try:
            test_df = dataset.prepare(segment, col_set=["label"])
            y_true = self._extract_labels(test_df)
        except Exception:
            raise RuntimeError(f"Cannot load labels for segment: {segment}")
        
        # Align predictions and labels
        common_index = predictions.index.intersection(y_true.index)
        pred_aligned = predictions.loc[common_index]
        true_aligned = y_true.loc[common_index]
        
        # Remove missing values
        mask = ~(pred_aligned.isna() | true_aligned.isna())
        pred_clean = pred_aligned[mask]
        true_clean = true_aligned[mask]
        
        if len(pred_clean) == 0:
            warnings.warn("No valid predictions for evaluation")
            return {}
        
        # Calculate metrics
        metrics = {}
        try:
            metrics['mse'] = mean_squared_error(true_clean, pred_clean)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(true_clean, pred_clean)
            
            # Correlation
            correlation = np.corrcoef(true_clean, pred_clean)[0, 1]
            metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
            
            # Information Coefficient (IC)
            metrics['ic'] = metrics['correlation']
            
            # Mean absolute correlation
            metrics['rank_ic'] = pd.Series(true_clean).corr(pd.Series(pred_clean), method='spearman')
            if np.isnan(metrics['rank_ic']):
                metrics['rank_ic'] = 0.0
                
        except Exception as e:
            warnings.warn(f"Error calculating metrics: {str(e)}")
        
        return metrics
    
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'model_config': self.model_config,
            'feature_columns': self.feature_columns,
            'label_columns': self.label_columns,
            'fitted': self.fitted
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.model_config = model_data['model_config']
        self.feature_columns = model_data['feature_columns']
        self.label_columns = model_data['label_columns']
        self.fitted = model_data['fitted']
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance if available.
        
        Returns:
            Optional[pd.Series]: Feature importance scores
        """
        if not self.fitted:
            return None
        
        importance = None
        
        if self.model_type == 'lightgbm':
            importance = self.model.feature_importance(importance_type='gain')
        elif self.model_type in ['ridge', 'lasso', 'linear']:
            importance = np.abs(self.model.coef_)
        elif self.model_type == 'rf':
            importance = self.model.feature_importances_
        
        if importance is not None and self.feature_columns:
            return pd.Series(importance, index=self.feature_columns)
        
        return None
