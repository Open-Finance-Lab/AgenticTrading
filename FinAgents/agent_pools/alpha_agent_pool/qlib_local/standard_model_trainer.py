"""
Standard Model Trainer for Backtesting Framework
Handles training of various ML models for factor prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from data_interfaces import ModelInput, ModelInterface

# ML dependencies
try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("sklearn not available, using simplified models")


class StandardModelTrainer(ModelInterface):
    """Standard model trainer with multiple algorithm support"""
    
    def train(self, features: pd.DataFrame, targets: pd.Series, model_input: ModelInput) -> Any:
        """Train model according to specification"""
        
        # Align features and targets
        aligned_features, aligned_targets = self._align_data(features, targets)
        
        if model_input.model_type == "linear":
            return self._train_linear_model(aligned_features, aligned_targets, model_input)
        elif model_input.model_type == "tree":
            return self._train_tree_model(aligned_features, aligned_targets, model_input)
        else:
            # Default to simple linear model
            return self._train_simple_model(aligned_features, aligned_targets)
    
    def _align_data(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features and targets by index with robust data cleaning"""
        
        # Find common index
        common_index = features.index.intersection(targets.index)
        
        aligned_features = features.loc[common_index]
        aligned_targets = targets.loc[common_index]
        
        # Clean features: handle NaN and infinity values
        aligned_features = aligned_features.fillna(0)  # Fill NaN with 0
        
        # Replace infinity values with large but finite numbers
        aligned_features = aligned_features.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # Clip extreme values to prevent numerical issues
        for col in aligned_features.columns:
            col_data = aligned_features[col]
            # Use robust quartile-based clipping
            q1, q3 = col_data.quantile([0.01, 0.99])
            if q3 > q1:  # Avoid issues with constant columns
                aligned_features[col] = col_data.clip(q1, q3)
        
        # Clean targets: handle NaN and infinity values  
        aligned_targets = aligned_targets.fillna(0)
        aligned_targets = aligned_targets.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # Clip target values to reasonable range
        target_q1, target_q3 = aligned_targets.quantile([0.01, 0.99])
        if target_q3 > target_q1:
            aligned_targets = aligned_targets.clip(target_q1, target_q3)
        
        return aligned_features, aligned_targets
    
    def _train_linear_model(self, features: pd.DataFrame, targets: pd.Series, model_input: ModelInput) -> Dict[str, Any]:
        """Train linear regression model using sklearn"""
        
        X = features.values
        y = targets.values
        
        # Check if we have any data
        if len(X) == 0 or len(y) == 0:
            print("Warning: No training data available, returning dummy model")
            return {
                'model_type': 'linear',
                'model': None,
                'coefficients': np.zeros(features.shape[1] if len(features.columns) > 0 else 1),
                'feature_names': features.columns.tolist() if len(features.columns) > 0 else ['dummy'],
                'training_samples': 0,
                'scaler': None,
                'is_dummy': True
            }
        
        if not SKLEARN_AVAILABLE:
            # Fallback to numpy implementation
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            try:
                beta = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ y)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
            
            return {
                'model_type': 'linear',
                'model': None,
                'coefficients': beta,
                'feature_names': features.columns.tolist(),
                'training_samples': len(X),
                'scaler': None
            }
        
        # Use sklearn for proper ML training
        # Feature scaling for better performance
        scaler = StandardScaler()
        
        # Additional data cleaning before scaling
        X_cleaned = X.copy()
        
        # Replace any remaining infinite values
        X_cleaned = np.where(np.isfinite(X_cleaned), X_cleaned, 0)
        
        # Check for remaining problematic values
        if not np.all(np.isfinite(X_cleaned)):
            print("Warning: Non-finite values detected, applying robust cleaning")
            X_cleaned = np.nan_to_num(X_cleaned, nan=0.0, posinf=1e6, neginf=-1e6)
        
        X_scaled = scaler.fit_transform(X_cleaned)
        
        # Split data for validation
        if len(X) > 100:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = X_scaled, X_scaled, y, y
        
        # Choose model based on parameters
        alpha = model_input.hyperparameters.get('regularization', 0.01)
        if alpha > 0:
            l1_ratio = model_input.hyperparameters.get('l1_ratio', 0)
            if l1_ratio > 0:
                # Lasso regression for feature selection
                model = Lasso(alpha=alpha, random_state=42)
            else:
                # Ridge regression for stability
                model = Ridge(alpha=alpha, random_state=42)
        else:
            # Standard linear regression
            model = LinearRegression()
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Validation metrics
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        return {
            'model_type': 'linear',
            'model': model,
            'scaler': scaler,
            'feature_names': features.columns.tolist(),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'coefficients': model.coef_ if hasattr(model, 'coef_') else None,
            'intercept': model.intercept_ if hasattr(model, 'intercept_') else None
        }
    
    def _train_tree_model(self, features: pd.DataFrame, targets: pd.Series, model_input: ModelInput) -> Dict[str, Any]:
        """Train tree-based model using sklearn or LightGBM"""
        
        X = features.values
        y = targets.values
        
        # Check implementation type
        implementation = getattr(model_input, 'implementation', 'sklearn')
        
        if not SKLEARN_AVAILABLE and implementation != 'lightgbm':
            # Fallback to simplified tree logic
            if len(X) > 0:
                feature_mean = np.mean(X[:, 0])
                high_mask = X[:, 0] > feature_mean
                high_prediction = np.mean(y[high_mask]) if np.any(high_mask) else 0
                low_prediction = np.mean(y[~high_mask]) if np.any(~high_mask) else 0
            else:
                feature_mean = 0
                high_prediction = 0
                low_prediction = 0
            
            return {
                'model_type': 'tree',
                'model': None,
                'split_feature': 0,
                'split_value': feature_mean,
                'high_prediction': high_prediction,
                'low_prediction': low_prediction,
                'training_samples': len(X)
            }
        
        # Split data for validation
        if len(X) > 100:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y
        
        # Choose model based on implementation
        if implementation == 'lightgbm':
            try:
                import lightgbm as lgb
                
                # LightGBM specific parameters
                lgb_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': model_input.hyperparameters.get('num_leaves', 63),
                    'learning_rate': model_input.hyperparameters.get('learning_rate', 0.01),
                    'feature_fraction': model_input.hyperparameters.get('feature_fraction', 0.9),
                    'bagging_fraction': model_input.hyperparameters.get('bagging_fraction', 0.9),
                    'bagging_freq': model_input.hyperparameters.get('bagging_freq', 5),
                    'reg_alpha': model_input.hyperparameters.get('reg_alpha', 0.0),
                    'reg_lambda': model_input.hyperparameters.get('reg_lambda', 0.0),
                    'verbose': -1,
                    'random_state': model_input.hyperparameters.get('random_state', 42)
                }
                
                # Override with user parameters
                for key, value in model_input.hyperparameters.items():
                    if key in ['n_estimators', 'max_depth']:
                        continue  # Handle separately
                    lgb_params[key] = value
                
                # Create LightGBM datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Train model
                n_estimators = model_input.hyperparameters.get('n_estimators', 1000)
                max_depth = model_input.hyperparameters.get('max_depth', -1)
                if max_depth > 0:
                    lgb_params['max_depth'] = max_depth

                early_stop = model_input.hyperparameters.get('early_stopping_rounds', 200)

                model = lgb.train(
                    lgb_params,
                    train_data,
                    num_boost_round=n_estimators,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(early_stop), lgb.log_evaluation(0)]
                )
                
                # Predictions for validation
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                # Feature importance
                feature_importance = dict(zip(features.columns, model.feature_importance()))
                
                implementation_used = 'lightgbm'
                
            except ImportError:
                print("LightGBM not available, falling back to sklearn GradientBoosting")
                implementation_used = 'sklearn_fallback'
                
                # Fallback to sklearn
                model_params = {
                    'random_state': model_input.hyperparameters.get('random_state', 42),
                    'n_estimators': model_input.hyperparameters.get('n_estimators', 100),
                    'learning_rate': model_input.hyperparameters.get('learning_rate', 0.1),
                    'max_depth': model_input.hyperparameters.get('max_depth', 3)
                }
                
                model = GradientBoostingRegressor(**model_params)
                model.fit(X_train, y_train)
                
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                feature_importance = dict(zip(features.columns, model.feature_importances_))
        
        else:
            # Use sklearn for other tree models
            model_params = {
                'random_state': 42,
                'max_depth': model_input.hyperparameters.get('max_depth', 6),
                'min_samples_split': model_input.hyperparameters.get('min_samples_split', 5),
                'min_samples_leaf': model_input.hyperparameters.get('min_samples_leaf', 2)
            }
            
            # Select specific tree model
            tree_type = model_input.hyperparameters.get('tree_type', 'random_forest')
            
            if tree_type == 'random_forest':
                model_params['n_estimators'] = model_input.hyperparameters.get('n_estimators', 100)
                model_params['max_features'] = model_input.hyperparameters.get('max_features', 'sqrt')
                model = RandomForestRegressor(**model_params)
            elif tree_type == 'gradient_boosting':
                model_params['n_estimators'] = model_input.hyperparameters.get('n_estimators', 100)
                model_params['learning_rate'] = model_input.hyperparameters.get('learning_rate', 0.1)
                model = GradientBoostingRegressor(**model_params)
            else:
                # Default to decision tree
                model = DecisionTreeRegressor(**model_params)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Predictions for validation
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(features.columns, model.feature_importances_))
            
            implementation_used = 'sklearn'
        
        # Calculate validation metrics
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        return {
            'model_type': 'tree',
            'implementation': implementation_used,
            'model': model,
            'feature_names': features.columns.tolist(),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'feature_importance': feature_importance
        }
    
    def _train_simple_model(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Train simple mean-based model"""
        
        return {
            'model_type': 'simple',
            'prediction': targets.mean(),
            'training_samples': len(targets)
        }
    
    def predict(self, features: pd.DataFrame, model: Any) -> pd.Series:
        """Generate predictions from trained model"""
        
        if model['model_type'] == 'linear':
            return self._predict_linear(features, model)
        elif model['model_type'] == 'tree':
            return self._predict_tree(features, model)
        else:
            return self._predict_simple(features, model)
    
    def _predict_linear(self, features: pd.DataFrame, model: Dict) -> pd.Series:
        """Generate linear model predictions"""
        
        X = features.values
        
        # Handle dummy model
        if model.get('is_dummy', False):
            predictions = np.zeros(len(features))
            return pd.Series(predictions, index=features.index)
        
        # Use sklearn model if available
        if SKLEARN_AVAILABLE and model.get('model') is not None:
            sklearn_model = model['model']
            scaler = model.get('scaler')
            
            # Apply scaling if used during training
            if scaler is not None:
                X_scaled = scaler.transform(X)
                predictions = sklearn_model.predict(X_scaled)
            else:
                predictions = sklearn_model.predict(X)
        else:
            # Fallback to numpy implementation
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            predictions = X_with_intercept @ model['coefficients']
        
        return pd.Series(predictions, index=features.index)
    
    def _predict_tree(self, features: pd.DataFrame, model: Dict) -> pd.Series:
        """Generate tree model predictions"""
        
        X = features.values
        
        # Use sklearn or LightGBM model if available
        if SKLEARN_AVAILABLE and model.get('model') is not None:
            trained_model = model['model']
            predictions = trained_model.predict(X)
        else:
            # Fallback to simplified tree logic
            split_feature = model['split_feature']
            split_value = model['split_value']
            
            predictions = np.where(
                X[:, split_feature] > split_value,
                model['high_prediction'],
                model['low_prediction']
            )
        
        return pd.Series(predictions, index=features.index)
    
    def _predict_simple(self, features: pd.DataFrame, model: Dict) -> pd.Series:
        """Generate simple model predictions"""
        
        predictions = np.full(len(features), model['prediction'])
        return pd.Series(predictions, index=features.index)
    
    def validate_model(self, model: Any, validation_data: Tuple) -> Dict[str, Any]:
        """Validate trained model"""
        
        features, targets = validation_data
        predictions = self.predict(features, model)
        
        # Align predictions and targets
        common_index = predictions.index.intersection(targets.index)
        aligned_predictions = predictions.loc[common_index]
        aligned_targets = targets.loc[common_index]
        
        # Calculate metrics
        mse = np.mean((aligned_predictions - aligned_targets) ** 2)
        mae = np.mean(np.abs(aligned_predictions - aligned_targets))
        
        # Classification-like accuracy (directional)
        correct_direction = np.sign(aligned_predictions) == np.sign(aligned_targets)
        accuracy = np.mean(correct_direction)
        
        return {
            'mse': mse,
            'mae': mae,
            'accuracy': accuracy,
            'model_type': model['model_type']
        }
