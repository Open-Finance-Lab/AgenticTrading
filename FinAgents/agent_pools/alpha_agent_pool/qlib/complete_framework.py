"""
Complete Backtesting Framework with Standard Input/Output Interfaces
Integrates all components with exact input/output format specifications
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
    print("⚠️ sklearn not available, using simplified models")

from data_interfaces import (
    DatasetInput, FactorInput, ModelInput, StrategyInput, OutputFormat,
    DatasetInterface, FactorInterface, ModelInterface, StrategyInterface,
    EXAMPLE_DATASET_INPUT, EXAMPLE_FACTOR_INPUT, EXAMPLE_MODEL_INPUT,
    EXAMPLE_STRATEGY_INPUT, EXAMPLE_OUTPUT_FORMAT
)
from output_processor import OutputProcessor, PerformanceMetrics

class BacktestingFramework:
    """
    Main backtesting framework that orchestrates the entire process
    with standardized input/output interfaces
    """
    
    def __init__(self):
        self.dataset_provider = None
        self.factors = []
        self.models = []
        self.strategy = None
        self.output_processor = None
        
    def run_complete_backtest(self,
                            dataset_input: DatasetInput,
                            factor_inputs: List[FactorInput],
                            model_input: Optional[ModelInput],
                            strategy_input: StrategyInput,
                            output_format: OutputFormat) -> Dict[str, Any]:
        """
        Run complete backtesting process with standardized inputs/outputs
        
        Args:
            dataset_input: Dataset configuration
            factor_inputs: List of factor configurations
            model_input: Model configuration (optional)
            strategy_input: Strategy configuration
            output_format: Output format specification
            
        Returns:
            Complete results dictionary with all outputs
        """
        
        print("🚀 Starting Complete Backtesting Process...")
        
        # Initialize output processor
        self.output_processor = OutputProcessor(output_format)
        
        # Step 1: Load Dataset
        print("📊 Loading dataset...")
        dataset_provider = SyntheticDatasetProvider()
        data = dataset_provider.load_data(dataset_input)
        data_quality = dataset_provider.validate_data(data)
        print(f"✅ Dataset loaded: {len(data)} rows, {data_quality['symbols']} symbols")

        # Step 1.5: Load ETF benchmark data if specified in output_format
        benchmark_data = {}
        etf_date_range = None
        
        if hasattr(output_format, "etf_symbols") and output_format.etf_symbols:
            # First pass: determine the common date range across all ETFs
            etf_data_temp = {}
            for symbol in output_format.etf_symbols:
                file_path = f"/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/etf_backup/{symbol}_daily.csv"
                if os.path.exists(file_path):
                    df_etf = pd.read_csv(file_path, parse_dates=['Date'])
                    df_etf.columns = [col.lower() for col in df_etf.columns]
                    df_etf['date'] = pd.to_datetime(df_etf['date'], utc=True).dt.tz_convert(None)
                    df_etf.set_index('date', inplace=True)
                    etf_data_temp[symbol] = df_etf
                else:
                    print(f"⚠️ ETF benchmark file not found for {symbol}: {file_path}")
            
            if etf_data_temp:
                # Find the common date range across all ETFs
                start_dates = [df.index.min() for df in etf_data_temp.values()]
                end_dates = [df.index.max() for df in etf_data_temp.values()]
                etf_start_date = max(start_dates)  # Latest start date
                etf_end_date = min(end_dates)      # Earliest end date
                
                print(f"📅 ETF data available from {etf_start_date.date()} to {etf_end_date.date()}")
                
                # Update dataset input to match ETF date range
                original_start = pd.to_datetime(dataset_input.start_date)
                original_end = pd.to_datetime(dataset_input.end_date)
                
                aligned_start = max(original_start, etf_start_date)
                aligned_end = min(original_end, etf_end_date)
                
                print(f"📅 Aligning strategy data to {aligned_start.date()} - {aligned_end.date()}")
                
                # Re-filter the main dataset with aligned dates
                data = data[(data['date'] >= aligned_start) & (data['date'] <= aligned_end)]
                print(f"✅ Strategy dataset trimmed to {len(data)} rows for date alignment")
                
                # Process ETF data with aligned date range
                for symbol, df_etf in etf_data_temp.items():
                    df_etf_filtered = df_etf[(df_etf.index >= aligned_start) & (df_etf.index <= aligned_end)]
                    close_prices = df_etf_filtered['close']
                    daily_returns = close_prices.pct_change().dropna()
                    benchmark_data[symbol] = daily_returns
                    print(f"✅ Loaded {symbol} benchmark: {len(daily_returns)} days of returns data")
                
                etf_date_range = (aligned_start, aligned_end)
        
        # Add buy-and-hold benchmark using the main dataset
        if len(data) > 0:
            # Create buy-and-hold benchmark from main dataset
            symbols_in_data = data['symbol'].unique()
            if len(symbols_in_data) > 0:
                main_symbol = symbols_in_data[0]  # Use the first symbol (e.g., AAPL)
                symbol_data = data[data['symbol'] == main_symbol].copy()
                symbol_data = symbol_data.set_index('date')['close']
                buy_hold_returns = symbol_data.pct_change().dropna()
                benchmark_data[f'{main_symbol}_BuyHold'] = buy_hold_returns
                print(f"✅ Added {main_symbol} buy-and-hold benchmark: {len(buy_hold_returns)} days")
        
        # Step 2: Calculate Factors
        print("🔢 Calculating factors...")
        factor_calculator = StandardFactorCalculator()
        factor_values = {}
        
        for factor_input in factor_inputs:
            factor_data = factor_calculator.calculate(data, factor_input)
            factor_validation = factor_calculator.validate_factor(factor_data)
            factor_values[factor_input.factor_name] = factor_data
            print(f"✅ Factor '{factor_input.factor_name}' calculated: {factor_validation['valid_ratio']:.2%} valid values")
        
        factor_df = pd.DataFrame(factor_values) if factor_values else None
        
        # Step 3: Train Model (if specified)
        model_predictions = None
        if model_input is not None:
            print("🤖 Training model...")
            model_trainer = StandardModelTrainer()
            
            if factor_df is not None:
                # Use factors as features
                features = factor_df.fillna(0)
                # Create synthetic targets (future returns)
                targets = self._create_synthetic_targets(data)
                
                model = model_trainer.train(features, targets, model_input)
                model_predictions = model_trainer.predict(features, model)
                model_validation = model_trainer.validate_model(model, (features, targets))
                print(f"✅ Model trained: {model_validation['accuracy']:.2%} accuracy")
        
        # Step 4: Generate Strategy Signals
        print("📈 Generating strategy signals...")
        strategy_executor = StandardStrategyExecutor()
        
        # Use model predictions or factor values for signals
        signal_data = model_predictions if model_predictions is not None else factor_df.iloc[:, 0] if factor_df is not None else None
        
        if signal_data is not None:
            signals = strategy_executor.generate_signals(signal_data, strategy_input)
            portfolio_weights = strategy_executor.construct_portfolio(signals, strategy_input)
            print(f"✅ Strategy signals generated: {len(signals)} signals")
        else:
            # Fallback to simple buy-and-hold
            portfolio_weights = self._create_buyhold_portfolio(data, strategy_input)
            signals = pd.DataFrame()
            print("⚠️ No signals data, using buy-and-hold strategy")
        
        # Step 5: Calculate Strategy Returns
        print("💰 Calculating strategy returns...")
        strategy_returns = self._calculate_strategy_returns(data, portfolio_weights, strategy_input)
        
        # Step 6: Generate Comprehensive Output
        print("📋 Generating reports and charts...")
        results = self.output_processor.process_backtest_results(
            strategy_returns=strategy_returns,
            strategy_positions=portfolio_weights,
            factor_values=factor_df,
            model_predictions=model_predictions,
            benchmark_data=benchmark_data if benchmark_data else None
        )
        
        # Add additional metadata
        results['input_summary'] = {
            'dataset': dataset_input,
            'factors': factor_inputs,
            'model': model_input,
            'strategy': strategy_input,
            'output_format': output_format
        }
        
        results['data_quality'] = data_quality
        
        print("✅ Backtesting completed successfully!")
        print(f"📁 Results saved to: {output_format.output_directory}")
        
        return results
    
    def _create_synthetic_targets(self, data: pd.DataFrame) -> pd.Series:
        """Create synthetic target returns for model training"""
        
        # Get close prices
        close_prices = data.pivot(index='date', columns='symbol', values='close')
        
        # Calculate future 5-day returns as targets
        future_returns = close_prices.pct_change(5).shift(-5)
        
        # Stack to get multi-index series
        targets = future_returns.stack()
        targets.index.names = ['date', 'symbol']
        
        return targets.fillna(0)
    
    def _create_buyhold_portfolio(self, data: pd.DataFrame, strategy_input: StrategyInput) -> pd.DataFrame:
        """Create simple buy-and-hold portfolio weights"""
        
        symbols = data['symbol'].unique()
        dates = sorted(data['date'].unique())
        
        # Equal weight portfolio
        weight = 1.0 / len(symbols)
        
        portfolio_data = []
        for date in dates:
            for symbol in symbols:
                portfolio_data.append({
                    'date': date,
                    'symbol': symbol,
                    'weight': weight
                })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        return portfolio_df.pivot(index='date', columns='symbol', values='weight').fillna(0)
    
    def _calculate_strategy_returns(self, data: pd.DataFrame, 
                                  portfolio_weights: pd.DataFrame,
                                  strategy_input: StrategyInput) -> pd.Series:
        """Calculate strategy returns from portfolio weights"""
        
        # Get daily returns
        close_prices = data.pivot(index='date', columns='symbol', values='close')
        daily_returns = close_prices.pct_change().fillna(0)
        
        # Align weights with returns
        aligned_weights = portfolio_weights.reindex(daily_returns.index).fillna(method='ffill').fillna(0)
        aligned_returns = daily_returns.reindex(aligned_weights.index).fillna(0)
        
        # Calculate portfolio returns
        strategy_returns = (aligned_weights.shift(1) * aligned_returns).sum(axis=1)
        
        # Apply transaction costs
        weight_changes = aligned_weights.diff().abs().sum(axis=1)
        transaction_costs = weight_changes * strategy_input.transaction_cost
        
        # Net returns after costs
        net_returns = strategy_returns - transaction_costs
        
        return net_returns.fillna(0)


class SyntheticDatasetProvider(DatasetInterface):
    """Synthetic dataset provider for demonstration"""
    
    def load_data(self, dataset_input: DatasetInput) -> pd.DataFrame:
        """
        Load market data for selected symbols and time period.
        If source_type is 'csv', load from CSV file; otherwise, generate synthetic data.
        """
        import os

        start_date = pd.to_datetime(dataset_input.start_date)
        end_date = pd.to_datetime(dataset_input.end_date)

        if dataset_input.source_type == "csv" and hasattr(dataset_input, "file_path") and os.path.exists(dataset_input.file_path):
            print(f"🔍 Debug: Loading CSV file: {dataset_input.file_path}")
            # Load real data from CSV
            df = pd.read_csv(dataset_input.file_path, parse_dates=['Date'])
            print(f"🔍 Debug: Original shape: {df.shape}, columns: {df.columns.tolist()}")
            # Normalize all column names to lowercase
            df.columns = [col.lower() for col in df.columns]
            # Ensure Date column is tz-naive (handle tz-aware and tz-naive)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert(None)
            else:
                raise ValueError(f"CSV file missing 'date' column: {dataset_input.file_path}. Columns found: {df.columns.tolist()}")
            # Add symbol column from filename if missing
            if 'symbol' not in df.columns:
                import re
                # Try to extract symbol from filename (e.g., .../AAPL_daily.csv)
                # Use basename to get just the filename, then extract symbol before underscore
                filename = os.path.basename(dataset_input.file_path)
                match = re.search(r"([A-Za-z0-9]+)_", filename)
                if match:
                    symbol = match.group(1)
                else:
                    symbol = os.path.splitext(filename)[0]
                df['symbol'] = symbol
                print(f"🔍 Debug: Added symbol '{symbol}' from filename '{filename}'")
            print(f"🔍 Debug: Unique symbols in data: {df['symbol'].unique()}")
            start_date_naive = pd.to_datetime(start_date)
            end_date_naive = pd.to_datetime(end_date)
            print(f"🔍 Debug: Date range filtering: {start_date_naive} to {end_date_naive}")
            df = df[(df['date'] >= start_date_naive) & (df['date'] <= end_date_naive)]
            print(f"🔍 Debug: After date filtering: {df.shape}")
            if dataset_input.custom_symbols:
                print(f"🔍 Debug: Custom symbols requested: {dataset_input.custom_symbols}")
                df = df[df['symbol'].isin(dataset_input.custom_symbols)]
                print(f"🔍 Debug: After symbol filtering: {df.shape}")
            # Only keep business days if needed
            if getattr(dataset_input, "business_days_only", True):
                df = df[df['date'].dt.dayofweek < 5]
            print(f"🔍 Debug: Final shape: {df.shape}")
            return df.sort_values(['date', 'symbol']).reset_index(drop=True)

        # Generate synthetic data as fallback
        # Ensure dates are tz-naive to avoid conversion issues
        dates = pd.date_range(start_date, end_date, freq='D', tz=None)
        if dataset_input.custom_symbols:
            symbols = dataset_input.custom_symbols
        else:
            symbols = [f"STOCK_{i:03d}" for i in range(100)]  # 100 synthetic stocks

        data_rows = []
        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)
            initial_price = np.random.uniform(20, 200)
            returns = np.random.normal(0.0005, 0.02, len(dates))
            for i in range(1, len(returns)):
                returns[i] += 0.1 * returns[i-1]
            prices = initial_price * (1 + returns).cumprod()
            for i, (date, price) in enumerate(zip(dates, prices)):
                daily_vol = abs(returns[i]) * 2
                high = price * (1 + daily_vol/2)
                low = price * (1 - daily_vol/2)
                open_price = prices[i-1] if i > 0 else price
                close_price = price
                volume = np.random.lognormal(10, 1) * (1 + abs(returns[i]) * 5)
                data_rows.append({
                    'date': date,  # Use lowercase 'date'
                    'symbol': symbol,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': int(volume),
                    'amount': close_price * volume
                })
        df = pd.DataFrame(data_rows)
        if dataset_input.source_type != "synthetic_all_days":
            df = df[df['date'].dt.dayofweek < 5]
        return df.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality"""
        
        return {
            'total_rows': len(data),
            'symbols': data['symbol'].nunique(),
            'date_range': f"{data['date'].min()} to {data['date'].max()}",
            'missing_values': data.isnull().sum().sum(),
            'negative_prices': (data['close'] <= 0).sum(),
            'valid_ratio': 1.0 - (data.isnull().sum().sum() / len(data))
        }


class StandardFactorCalculator(FactorInterface):
    """Standard factor calculator supporting multiple calculation methods"""
    
    def calculate(self, data: pd.DataFrame, factor_input: FactorInput) -> pd.Series:
        """Calculate factor based on input specification"""
        
        if factor_input.calculation_method == "expression":
            return self._calculate_expression_factor(data, factor_input)
        elif factor_input.calculation_method == "function":
            return self._calculate_function_factor(data, factor_input)
        else:
            # Default momentum factor
            return self._calculate_momentum_factor(data, factor_input.lookback_period)
    
    def _calculate_expression_factor(self, data: pd.DataFrame, factor_input: FactorInput) -> pd.Series:
        """Calculate factor from expression"""
        
        # Convert to pivot format for easier calculation
        close_prices = data.pivot(index='date', columns='symbol', values='close')
        
        if "momentum" in factor_input.factor_name.lower():
            # Momentum factor: current price / price N days ago - 1
            factor_values = close_prices / close_prices.shift(factor_input.lookback_period) - 1
        elif "volatility" in factor_input.factor_name.lower():
            # Volatility factor: rolling standard deviation of returns
            returns = close_prices.pct_change()
            factor_values = returns.rolling(factor_input.lookback_period).std()
        elif "rsi" in factor_input.factor_name.lower():
            # RSI factor
            factor_values = self._calculate_rsi(close_prices, factor_input.lookback_period)
        else:
            # Default to momentum
            factor_values = close_prices.pct_change(factor_input.lookback_period)
        
        # Convert back to multi-index series
        result = factor_values.stack()
        result.index.names = ['date', 'symbol']
        
        return result
    
    def _calculate_function_factor(self, data: pd.DataFrame, factor_input: FactorInput) -> pd.Series:
        """Calculate factor using function"""
        # Placeholder for custom function calculations
        return self._calculate_momentum_factor(data, factor_input.lookback_period)
    
    def _calculate_momentum_factor(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate simple momentum factor"""
        
        close_prices = data.pivot(index='date', columns='symbol', values='close')
        momentum = close_prices.pct_change(period)
        
        result = momentum.stack()
        result.index.names = ['date', 'symbol']
        
        return result
    
    def _calculate_rsi(self, prices: pd.DataFrame, period: int) -> pd.DataFrame:
        """Calculate RSI factor"""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return (rsi - 50) / 50  # Normalize to [-1, 1]
    
    def validate_factor(self, factor_values: pd.Series) -> Dict[str, Any]:
        """Validate factor values"""
        
        factor_clean = factor_values.dropna()
        
        return {
            'total_values': len(factor_values),
            'valid_values': len(factor_clean),
            'valid_ratio': len(factor_clean) / len(factor_values) if len(factor_values) > 0 else 0,
            'mean': factor_clean.mean() if len(factor_clean) > 0 else 0,
            'std': factor_clean.std() if len(factor_clean) > 0 else 0,
            'min': factor_clean.min() if len(factor_clean) > 0 else 0,
            'max': factor_clean.max() if len(factor_clean) > 0 else 0
        }


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
        """Align features and targets by index"""
        
        # Find common index
        common_index = features.index.intersection(targets.index)
        
        aligned_features = features.loc[common_index]
        aligned_targets = targets.loc[common_index]
        
        return aligned_features.fillna(0), aligned_targets.fillna(0)
    
    def _train_linear_model(self, features: pd.DataFrame, targets: pd.Series, model_input: ModelInput) -> Dict[str, Any]:
        """Train linear regression model using sklearn"""
        
        X = features.values
        y = targets.values
        
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
        X_scaled = scaler.fit_transform(X)
        
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
                    'num_leaves': 31,
                    'learning_rate': model_input.hyperparameters.get('learning_rate', 0.02),
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
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
                n_estimators = model_input.hyperparameters.get('n_estimators', 100)
                max_depth = model_input.hyperparameters.get('max_depth', -1)
                if max_depth > 0:
                    lgb_params['max_depth'] = max_depth
                
                model = lgb.train(
                    lgb_params,
                    train_data,
                    num_boost_round=n_estimators,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                )
                
                # Predictions for validation
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                # Feature importance
                feature_importance = dict(zip(features.columns, model.feature_importance()))
                
                implementation_used = 'lightgbm'
                
            except ImportError:
                print("⚠️ LightGBM not available, falling back to sklearn GradientBoosting")
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


class StandardStrategyExecutor(StrategyInterface):
    """Standard strategy executor"""
    
    def generate_signals(self, predictions: pd.Series, strategy_input: StrategyInput) -> pd.DataFrame:
        """Generate trading signals from predictions with improved logic"""
        
        signals_data = []
        
        # Define thresholds for signal generation
        signal_threshold = getattr(strategy_input, 'signal_threshold', 0.0)  # Default threshold
        
        # Group by date
        for date in predictions.index.get_level_values('date').unique():
            date_predictions = predictions.loc[date]
            
            if strategy_input.strategy_type == "long_only":
                # For single stock: use prediction sign and threshold
                if len(date_predictions) == 1:
                    symbol = date_predictions.index[0]
                    prediction = date_predictions.iloc[0]
                    
                    # Generate signal based on prediction value and threshold
                    if prediction > signal_threshold:
                        signal = 1   # Buy signal
                    elif prediction < -signal_threshold:
                        signal = 0   # Hold/no position (long_only can't short)
                    else:
                        signal = 0   # Neutral/hold
                    
                    if signal != 0:  # Only add non-zero signals
                        signals_data.append({
                            'date': date,
                            'symbol': symbol,
                            'signal': signal,
                            'prediction': prediction
                        })
                else:
                    # Multiple stocks: select top N with positive predictions
                    positive_predictions = date_predictions[date_predictions > signal_threshold]
                    top_symbols = positive_predictions.nlargest(strategy_input.num_positions)
                    
                    for symbol, prediction in top_symbols.items():
                        signals_data.append({
                            'date': date,
                            'symbol': symbol,
                            'signal': 1,  # Long signal
                            'prediction': prediction
                        })
            
            elif strategy_input.strategy_type == "long_short":
                # Enhanced long-short strategy with thresholds
                if len(date_predictions) == 1:
                    # Single stock: use prediction sign directly
                    symbol = date_predictions.index[0]
                    prediction = date_predictions.iloc[0]
                    
                    if prediction > signal_threshold:
                        signal = 1   # Long signal
                    elif prediction < -signal_threshold:
                        signal = -1  # Short signal
                    else:
                        signal = 0   # Neutral/hold
                    
                    if signal != 0:  # Only add non-zero signals
                        signals_data.append({
                            'date': date,
                            'symbol': symbol,
                            'signal': signal,
                            'prediction': prediction
                        })
                else:
                    # Multiple stocks: select top and bottom
                    n_long = int(strategy_input.num_positions * strategy_input.long_ratio / 2)
                    n_short = strategy_input.num_positions - n_long
                    
                    # Only consider predictions above threshold for long
                    positive_predictions = date_predictions[date_predictions > signal_threshold]
                    top_symbols = positive_predictions.nlargest(n_long)
                    
                    # Only consider predictions below negative threshold for short
                    negative_predictions = date_predictions[date_predictions < -signal_threshold]
                    bottom_symbols = negative_predictions.nsmallest(n_short)
                    
                    # Long signals
                    for symbol, prediction in top_symbols.items():
                        signals_data.append({
                            'date': date,
                            'symbol': symbol,
                            'signal': 1,
                            'prediction': prediction
                        })
                    
                    # Short signals
                    for symbol, prediction in bottom_symbols.items():
                        signals_data.append({
                            'date': date,
                            'symbol': symbol,
                            'signal': -1,
                            'prediction': prediction
                        })
            
            elif strategy_input.strategy_type == "market_neutral":
                # New strategy type: always maintain both long and short positions
                for symbol, prediction in date_predictions.items():
                    if prediction > signal_threshold:
                        signal = 1   # Long signal
                    elif prediction < -signal_threshold:
                        signal = -1  # Short signal
                    else:
                        signal = 0   # Neutral
                    
                    if signal != 0:
                        signals_data.append({
                            'date': date,
                            'symbol': symbol,
                            'signal': signal,
                            'prediction': prediction
                        })
        
        return pd.DataFrame(signals_data)
    
    def construct_portfolio(self, signals: pd.DataFrame, strategy_input: StrategyInput) -> pd.DataFrame:
        """Construct portfolio weights from signals"""
        
        if len(signals) == 0:
            return pd.DataFrame()
        
        portfolio_data = []
        
        for date in signals['date'].unique():
            date_signals = signals[signals['date'] == date]
            
            if strategy_input.position_method == "equal_weight":
                # Equal weight allocation
                weight_per_position = 1.0 / len(date_signals)
                
                for _, row in date_signals.iterrows():
                    portfolio_data.append({
                        'date': date,
                        'symbol': row['symbol'],
                        'weight': weight_per_position * row['signal']
                    })
            
            elif strategy_input.position_method == "factor_weight":
                # Weight by prediction strength
                total_abs_prediction = date_signals['prediction'].abs().sum()
                
                if total_abs_prediction > 0:
                    for _, row in date_signals.iterrows():
                        weight = abs(row['prediction']) / total_abs_prediction * row['signal']
                        portfolio_data.append({
                            'date': date,
                            'symbol': row['symbol'],
                            'weight': weight
                        })
        
        if portfolio_data:
            portfolio_df = pd.DataFrame(portfolio_data)
            return portfolio_df.pivot(index='date', columns='symbol', values='weight').fillna(0)
        else:
            return pd.DataFrame()


# Example usage and testing
if __name__ == "__main__":
    
    # Initialize framework
    framework = BacktestingFramework()
    
    # Define exact inputs
    dataset_config = EXAMPLE_DATASET_INPUT
    
    factor_configs = [
        FactorInput(
            factor_name="momentum_10d",
            factor_type="alpha",
            calculation_method="expression",
            expression="close/Ref(close,10)-1",
            lookback_period=10
        ),
        FactorInput(
            factor_name="volatility_20d",
            factor_type="risk",
            calculation_method="expression",
            expression="std(returns,20)",
            lookback_period=20
        )
    ]
    
    model_config = EXAMPLE_MODEL_INPUT
    strategy_config = EXAMPLE_STRATEGY_INPUT
    output_config = EXAMPLE_OUTPUT_FORMAT
    
    # Run complete backtest
    results = framework.run_complete_backtest(
        dataset_input=dataset_config,
        factor_inputs=factor_configs,
        model_input=model_config,
        strategy_input=strategy_config,
        output_format=output_config
    )
    
    print("\n" + "="*60)
    print("🎉 BACKTESTING FRAMEWORK RESULTS SUMMARY")
    print("="*60)
    
    # Print key results
    strategy_metrics = results['strategy_metrics']
    print(f"📊 Strategy Performance:")
    print(f"   Annual Return: {strategy_metrics.annual_return:.2%}")
    print(f"   Sharpe Ratio: {strategy_metrics.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {strategy_metrics.max_drawdown:.2%}")
    print(f"   Win Rate: {strategy_metrics.win_rate:.2%}")
    
    print(f"\n📁 Output Files Generated:")
    for file_type, file_path in results.get('chart_paths', {}).items():
        print(f"   {file_type}: {file_path}")
    
    if 'summary_report_path' in results:
        print(f"   Summary Report: {results['summary_report_path']}")
    
    print(f"\n✅ Complete results available in: {output_config.output_directory}")
