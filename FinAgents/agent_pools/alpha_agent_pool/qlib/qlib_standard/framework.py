"""
Qlib Standard Framework Integration

This module provides a complete integration of all Qlib standard components
for end-to-end quantitative trading research workflows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import warnings
import time
from datetime import datetime

# Import all standard components
from .data_loader import QlibCSVDataLoader, QlibSyntheticDataLoader
from .factor_calculator import QlibFactorCalculator
from .model_trainer import QlibModelTrainer
from .strategy_executor import QlibStrategyExecutor
from .data_handler import QlibDataHandler

# Qlib core imports
from qlib.data.dataset import DatasetH
from qlib.contrib.evaluate import backtest_daily, risk_analysis


class QlibStandardFramework:
    """
    Complete Qlib-based quantitative trading framework.
    
    This class integrates all standard components to provide a comprehensive
    workflow for data loading, factor calculation, model training, strategy
    execution, and backtesting.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the Qlib standard framework.
        
        Args:
            config: Complete framework configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or {}
        self._setup_default_config()
        
        # Initialize components
        self.data_loader = None
        self.factor_calculator = None
        self.data_handler = None
        self.model_trainer = None
        self.strategy_executor = None
        
        # Data and results storage
        self.raw_data = None
        self.factor_data = None
        self.feature_name_mapping = {}  # Map generic names to meaningful names
        self.meaningful_feature_names = []  # Store original meaningful factor names
        self.processed_data = None
        self.trained_model = None
        self.predictions = None
        self.backtest_results = None
        
        # State tracking
        self.is_initialized = False
        self.pipeline_status = {
            'data_loaded': False,
            'factors_calculated': False,
            'data_processed': False,
            'model_trained': False,
            'strategy_executed': False,
            'backtest_completed': False
        }
    
    def _setup_default_config(self) -> None:
        """Setup default configuration for all components."""
        default_config = {
            # Data configuration
            'data': {
                'source_type': 'csv',  # 'csv' or 'synthetic'
                'data_path': None,
                'instruments': None,
                'start_time': '2020-01-01',
                'end_time': '2023-12-31',
                'freq': '1H'
            },
            
            # Factor configuration
            'factors': {
                'enabled': True,
                'factor_list': None,  # None for all default factors
                'custom_factors': {}
            },
            
            # Data processing configuration
            'preprocessing': {
                'normalization': 'zscore',
                'handle_missing': True,
                'train_test_split': 0.7,
                'validation_split': 0.2,
                'fit_start_time': None,
                'fit_end_time': None
            },
            
            # Model configuration
            'model': {
                'type': 'lightgbm',
                'config': {
                    'objective': 'regression',
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'num_boost_round': 100
                }
            },
            
            # Strategy configuration
            'strategy': {
                'type': 'long_short',
                'top_k': 10,
                'bottom_k': 10,
                'max_position_weight': 0.1,
                'rebalance_freq': 'daily'
            },
            
            # Backtest configuration
            'backtest': {
                'start_time': None,  # Will use data start_time if None
                'end_time': None,    # Will use data end_time if None
                'benchmark': 'SH000300',
                'account': 100000000,
                'exchange_kwargs': {}
            }
        }
        
        # Deep merge with user config
        self.config = self._deep_merge_config(default_config, self.config)
    
    def _deep_merge_config(self, default: Dict, user: Dict) -> Dict:
        """Deep merge user config with defaults."""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def initialize(self) -> None:
        """Initialize all framework components."""
        try:
            print(" Initializing Qlib Standard Framework...")
            
            # Initialize data loader
            self._initialize_data_loader()
            
            # Initialize factor calculator
            self._initialize_factor_calculator()
            
            # Initialize model trainer
            self._initialize_model_trainer()
            
            # Initialize strategy executor
            self._initialize_strategy_executor()
            
            self.is_initialized = True
            print(" Framework initialization completed")
            
        except Exception as e:
            raise RuntimeError(f"Framework initialization failed: {str(e)}")
    
    def _initialize_data_loader(self) -> None:
        """Initialize the data loader component."""
        data_config = self.config['data']
        
        if data_config['source_type'] == 'csv':
            if not data_config['data_path']:
                raise ValueError("data_path is required for CSV data source")
            
            self.data_loader = QlibCSVDataLoader(
                data_path=data_config['data_path'],
                freq=data_config['freq']
            )
        
        elif data_config['source_type'] == 'synthetic':
            instruments = data_config.get('instruments', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
            
            self.data_loader = QlibSyntheticDataLoader(
                instruments=instruments,
                start_time=data_config['start_time'],
                end_time=data_config['end_time'],
                freq=data_config['freq']
            )
        
        else:
            raise ValueError(f"Unknown data source type: {data_config['source_type']}")
    
    def _initialize_factor_calculator(self) -> None:
        """Initialize the factor calculator component."""
        factor_config = self.config['factors']
        
        self.factor_calculator = QlibFactorCalculator(
            factor_config=factor_config
        )
        
        # Add custom factors if specified
        for factor_name, expression in factor_config.get('custom_factors', {}).items():
            self.factor_calculator.add_custom_factor(factor_name, expression)
    
    def _initialize_model_trainer(self) -> None:
        """Initialize the model trainer component."""
        model_config = self.config['model']
        
        self.model_trainer = QlibModelTrainer(
            model_type=model_config['type'],
            model_config=model_config['config']
        )
    
    def _initialize_strategy_executor(self) -> None:
        """Initialize the strategy executor component."""
        strategy_config = self.config['strategy']
        
        self.strategy_executor = QlibStrategyExecutor(
            strategy_config=strategy_config
        )
    
    def run_complete_pipeline(
        self,
        save_results: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete quantitative trading pipeline.
        
        Args:
            save_results: Whether to save results to disk
            output_dir: Directory to save results (if save_results=True)
            
        Returns:
            Dict[str, Any]: Complete pipeline results
        """
        if not self.is_initialized:
            self.initialize()
        
        print(" Running complete Qlib pipeline...")
        start_time = time.time()
        
        try:
            # Step 1: Load data
            print(" Step 1: Loading data...")
            self.load_data()
            
            # Step 2: Calculate factors
            print("🧮 Step 2: Calculating factors...")
            self.calculate_factors()
            
            # Step 3: Process data
            print("⚙️ Step 3: Processing data...")
            self.process_data()
            
            # Step 4: Train model
            print(" Step 4: Training model...")
            self.train_model()
            
            # Step 5: Generate predictions
            print("🔮 Step 5: Generating predictions...")
            self.generate_predictions()
            
            # Step 6: Execute strategy and backtest
            print(" Step 6: Executing strategy and backtesting...")
            self.execute_backtest()
            
            # Compile results
            results = self.get_pipeline_results()
            
            # Save results if requested
            if save_results:
                self.save_results(output_dir or "qlib_results")
            
            end_time = time.time()
            print(f" Pipeline completed in {end_time - start_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            print(f" Pipeline failed: {str(e)}")
            raise
    
    def load_data(self) -> pd.DataFrame:
        """Load raw data using the configured data loader."""
        if not self.data_loader:
            raise RuntimeError("Data loader not initialized")
        
        data_config = self.config['data']
        
        self.raw_data = self.data_loader.load(
            instruments=data_config.get('instruments'),
            start_time=data_config['start_time'],
            end_time=data_config['end_time']
        )
        
        if self.raw_data.empty:
            raise RuntimeError("No data loaded")
        
        self.pipeline_status['data_loaded'] = True
        print(f"    Loaded {len(self.raw_data)} data points for {self.raw_data.index.get_level_values('instrument').nunique()} instruments")
        
        return self.raw_data
    
    def calculate_factors(self) -> pd.DataFrame:
        """Calculate factors using the raw data."""
        if self.raw_data is None:
            raise RuntimeError("Raw data not loaded")
        
        if not self.factor_calculator:
            raise RuntimeError("Factor calculator not initialized")
        
        factor_config = self.config['factors']
        
        if factor_config['enabled']:
            factor_names = factor_config.get('factor_list')
            
            self.factor_data = self.factor_calculator.calculate_factors(
                data=self.raw_data,
                factor_names=factor_names
            )
            
            # Combine raw data with calculated factors
            if not self.factor_data.empty:
                # Align indices and combine
                self.factor_data = pd.concat([self.raw_data, self.factor_data], axis=1)
                
                # Create meaningful feature name mapping
                self._create_feature_name_mapping()
            else:
                self.factor_data = self.raw_data.copy()
        else:
            # Use raw data as factor data
            self.factor_data = self.raw_data.copy()
        
        self.pipeline_status['factors_calculated'] = True
        
        if isinstance(self.factor_data.columns, pd.MultiIndex):
            feature_count = sum(1 for col in self.factor_data.columns if col[0] == 'feature')
            print(f"   🧮 Calculated {feature_count} features")
        else:
            print(f"   🧮 Using {len(self.factor_data.columns)} features")
        
        return self.factor_data

    def _create_feature_name_mapping(self):
        """Create mapping from generic feature names to meaningful factor names."""
        self.feature_name_mapping = {}
        self.meaningful_feature_names = []
        
        if isinstance(self.factor_data.columns, pd.MultiIndex):
            feature_idx = 1
            for col in self.factor_data.columns:
                if len(col) > 1 and col[0] == 'feature':
                    meaningful_name = col[1]  # Get the actual factor name
                    generic_name = f'feature_{feature_idx}'
                    
                    self.feature_name_mapping[generic_name] = meaningful_name
                    self.meaningful_feature_names.append(meaningful_name)
                    feature_idx += 1
        
        print(f"   📝 Created feature mapping for {len(self.meaningful_feature_names)} factors")
        if len(self.meaningful_feature_names) > 0:
            print(f"   🏷️ Sample mappings: {dict(list(self.feature_name_mapping.items())[:3])}")

    def get_meaningful_feature_names(self, generic_names=None):
        """Convert generic feature names back to meaningful factor names."""
        if generic_names is None:
            return self.meaningful_feature_names
        
        meaningful_names = []
        for name in generic_names:
            if name in self.feature_name_mapping:
                meaningful_names.append(self.feature_name_mapping[name])
            else:
                # If no mapping found, use the original name
                meaningful_names.append(name)
        
        return meaningful_names
    
    def process_data(self) -> DatasetH:
        """Process data using the data handler."""
        if self.factor_data is None:
            raise RuntimeError("Factor data not available")
        
        # Setup data handler with processed data
        processor_config = self.config['preprocessing']
        
        # Create data handler directly from DataFrame using Qlib's from_df method
        from qlib.data.dataset.handler import DataHandlerLP
        self.data_handler = DataHandlerLP.from_df(self.factor_data)
        
        print(f"    Data handler created with {len(self.factor_data)} records")
        
        # Create dataset with train/test splits
        train_test_split = processor_config['train_test_split']
        
        # Calculate split dates
        dates = self.factor_data.index.get_level_values('datetime').unique().sort_values()
        split_idx = int(len(dates) * train_test_split)
        train_end_date = dates[split_idx-1]
        test_start_date = dates[split_idx]
        
        segments = {
            'train': (dates[0], train_end_date),
            'test': (test_start_date, dates[-1])
        }
        
        # Add validation split if specified
        validation_split = processor_config.get('validation_split', 0.0)
        if validation_split > 0:
            val_split_idx = int(split_idx * (1 - validation_split))
            val_start_date = dates[val_split_idx]
            
            segments['train'] = (dates[0], dates[val_split_idx-1])
            segments['valid'] = (val_start_date, train_end_date)
        
        # Create dataset
        self.processed_data = DatasetH(
            handler=self.data_handler,
            segments=segments
        )
        
        self.pipeline_status['data_processed'] = True
        print(f"   ⚙️ Data processed with segments: {list(segments.keys())}")
        
        return self.processed_data
    
    def train_model(self) -> QlibModelTrainer:
        """Train the model using processed data."""
        if self.processed_data is None:
            raise RuntimeError("Processed data not available")
        
        if not self.model_trainer:
            raise RuntimeError("Model trainer not initialized")
        
        # Train the model
        self.model_trainer.fit(self.processed_data)
        self.trained_model = self.model_trainer
        
        self.pipeline_status['model_trained'] = True
        print(f"    Model trained: {self.model_trainer.model_type}")
        
        # Print training summary
        try:
            # Get feature importance if available
            importance = self.model_trainer.get_feature_importance()
            if importance is not None:
                top_features = importance.nlargest(5)
                
                # Create a direct mapping from actual factor data
                meaningful_names = []
                for generic_name in list(top_features.index):
                    # Try to find meaningful name from factor data columns
                    meaningful_name = generic_name
                    
                    if isinstance(self.factor_data.columns, pd.MultiIndex):
                        # Look for meaningful names in the factor data
                        feature_idx = 1
                        for col in self.factor_data.columns:
                            if len(col) > 1 and col[0] == 'feature':
                                if f'feature_{feature_idx}' == generic_name:
                                    meaningful_name = col[1]  # Get the actual factor name
                                    break
                                feature_idx += 1
                    
                    meaningful_names.append(meaningful_name)
                
                print(f"    Top 5 features: {meaningful_names}")
        except Exception:
            pass
        
        return self.trained_model
    
    def generate_predictions(self) -> pd.Series:
        """Generate predictions using the trained model."""
        if self.trained_model is None:
            raise RuntimeError("Model not trained")
        
        if self.processed_data is None:
            raise RuntimeError("Processed data not available")
        
        # Generate predictions on test set
        self.predictions = self.trained_model.predict(self.processed_data, segment='test')
        
        print(f"   🔮 Generated {len(self.predictions)} predictions")
        
        return self.predictions
    
    def execute_backtest(self) -> Dict[str, Any]:
        """Execute strategy and perform backtesting."""
        if self.predictions is None:
            raise RuntimeError("Predictions not available")
        
        backtest_config = self.config['backtest']
        strategy_config = self.config['strategy']
        
        # Setup strategy with predictions
        if strategy_config['type'] == 'topk':
            # 使用策略执行器创建TopK策略
            strategy_executor = QlibStrategyExecutor(strategy_config)
            strategy_results = strategy_executor.execute_strategy(
                self.predictions,
                strategy_type="TopkDropout",
                topk=strategy_config.get('top_k', 50),
                n_drop=strategy_config.get('n_drop', 5)
            )
            strategy = strategy_results.get('strategy')
        else:
            # Update strategy executor with predictions
            strategy_results = self.strategy_executor.execute_strategy(self.predictions)
            strategy = strategy_results.get('strategy')
        
        # Determine backtest time range
        start_time = backtest_config.get('start_time')
        end_time = backtest_config.get('end_time')
        
        if not start_time:
            # Use test data start time
            test_dates = self.predictions.index.get_level_values('datetime').unique()
            start_time = test_dates.min()
        
        if not end_time:
            # Use test data end time
            test_dates = self.predictions.index.get_level_values('datetime').unique()
            end_time = test_dates.max()
        
        # Run backtest
        try:
            backtest_results = backtest_daily(
                start_time=start_time,
                end_time=end_time,
                strategy=strategy,
                account=backtest_config.get('account', 100000000),
                benchmark=backtest_config.get('benchmark', 'SH000300'),
                exchange_kwargs=backtest_config.get('exchange_kwargs', {})
            )
            
            # Calculate additional metrics
            if 'positions_normal' in backtest_results:
                positions = backtest_results['positions_normal']
                returns = positions['return'] if 'return' in positions.columns else None
                
                if returns is not None:
                    risk_metrics = risk_analysis(returns)
                    backtest_results['risk_metrics'] = risk_metrics
            
            self.backtest_results = backtest_results
            self.pipeline_status['backtest_completed'] = True
            
            # Print summary
            if 'risk_metrics' in backtest_results:
                metrics = backtest_results['risk_metrics']
                annual_return = metrics.get('annual_return', 0)
                sharpe_ratio = metrics.get('information_ratio', 0)
                max_drawdown = metrics.get('max_drawdown', 0)
                
                print(f"    Annual Return: {annual_return:.2%}")
                print(f"    Sharpe Ratio: {sharpe_ratio:.3f}")
                print(f"   📉 Max Drawdown: {max_drawdown:.2%}")
            
        except Exception as e:
            warnings.warn(f"Backtest failed: {str(e)}")
            self.backtest_results = {}
        
        return self.backtest_results
    
    def get_pipeline_results(self) -> Dict[str, Any]:
        """Get comprehensive pipeline results."""
        results = {
            'pipeline_status': self.pipeline_status.copy(),
            'config': self.config,
            'data_summary': self._get_data_summary(),
            'model_summary': self._get_model_summary(),
            'backtest_summary': self._get_backtest_summary(),
            'raw_data': self.raw_data,
            'factor_data': self.factor_data,
            'predictions': self.predictions,
            'backtest_results': self.backtest_results
        }
        
        return results
    
    def _get_data_summary(self) -> Dict[str, Any]:
        """Get data summary statistics."""
        summary = {}
        
        if self.raw_data is not None:
            summary['raw_data'] = {
                'shape': self.raw_data.shape,
                'instruments': self.raw_data.index.get_level_values('instrument').nunique(),
                'date_range': (
                    self.raw_data.index.get_level_values('datetime').min(),
                    self.raw_data.index.get_level_values('datetime').max()
                )
            }
        
        if self.factor_data is not None:
            feature_count = 0
            if isinstance(self.factor_data.columns, pd.MultiIndex):
                feature_count = sum(1 for col in self.factor_data.columns if col[0] == 'feature')
            else:
                feature_count = len(self.factor_data.columns)
            
            summary['factor_data'] = {
                'shape': self.factor_data.shape,
                'feature_count': feature_count
            }
        
        return summary
    
    def _get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        summary = {}
        
        if self.trained_model:
            summary['model_type'] = self.trained_model.model_type
            summary['is_fitted'] = self.trained_model.fitted
            
            # Add feature importance if available
            try:
                importance = self.trained_model.get_feature_importance()
                if importance is not None:
                    summary['top_features'] = importance.nlargest(10).to_dict()
            except Exception:
                pass
        
        return summary
    
    def _get_backtest_summary(self) -> Dict[str, Any]:
        """Get backtest summary statistics."""
        summary = {}
        
        if self.backtest_results and 'risk_metrics' in self.backtest_results:
            metrics = self.backtest_results['risk_metrics']
            summary = {
                'annual_return': metrics.get('annual_return', 0),
                'sharpe_ratio': metrics.get('information_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'volatility': metrics.get('annualized_vol', 0)
            }
        
        return summary
    
    def save_results(self, output_dir: str) -> None:
        """Save all results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving results to {output_path}")
        
        # Save configuration
        import json
        config_file = output_path / 'config.json'
        with open(config_file, 'w') as f:
            # Convert timestamps to strings for JSON serialization
            config_copy = self._serialize_config(self.config)
            json.dump(config_copy, f, indent=2, default=str)
        
        # Save data
        if self.raw_data is not None:
            self.raw_data.to_csv(output_path / 'raw_data.csv')
        
        if self.factor_data is not None:
            self.factor_data.to_csv(output_path / 'factor_data.csv')
        
        if self.predictions is not None:
            self.predictions.to_csv(output_path / 'predictions.csv')
        
        # Save model
        if self.trained_model:
            self.trained_model.save_model(output_path / 'trained_model.pkl')
        
        # Save backtest results
        if self.backtest_results:
            import pickle
            with open(output_path / 'backtest_results.pkl', 'wb') as f:
                pickle.dump(self.backtest_results, f)
        
        print(f" Results saved to {output_path}")
    
    def _serialize_config(self, config: Dict) -> Dict:
        """Serialize config for JSON saving."""
        serialized = {}
        for key, value in config.items():
            if isinstance(value, dict):
                serialized[key] = self._serialize_config(value)
            elif isinstance(value, (pd.Timestamp, datetime)):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized
