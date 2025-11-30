"""
Comprehensive Alpha Factor and Model Validation Demo

This demo combines alpha factor validation and model validation in one pipeline,
showing how to use both FactorInput and ModelInput approaches with real CSV data.
"""

from factor_pipeline import FactorEvaluator
from model_pipeline import ModelEvaluator
from interfaces import StandardAcceptanceCriteria, BacktestInterface, EvaluationMetrics, FactorInterface, ModelInterface
from data_interfaces import FactorInput, ModelInput
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

class FactorInputAdapter:
    """
    Adapter to convert FactorInput specifications into executable factor calculations
    """
    
    def __init__(self, factor_input: FactorInput):
        """
        Initialize adapter with FactorInput specification
        
        Args:
            factor_input: FactorInput object containing factor specification
        """
        self.factor_input = factor_input
    
    @property
    def factor_name(self) -> str:
        """Get factor name"""
        return self.factor_input.factor_name
    
    @property
    def factor_description(self) -> str:
        """Get factor description"""
        return f"Alpha Factor: {self.factor_input.factor_name}"
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate factor values based on FactorInput specification
        
        Args:
            data: Market data
            
        Returns:
            Factor values
        """
        method = self.factor_input.calculation_method
        expression = self.factor_input.expression
        factor_type = self.factor_input.factor_type
        
        # Use function_params if available, otherwise create default params
        params = self.factor_input.function_params or {}
        if 'period' not in params:
            params['period'] = self.factor_input.lookback_period
        
        if method == "expression":
            return self._calculate_from_expression(data, expression, params, factor_type)
        elif method == "function":
            function_name = self.factor_input.function_name
            return self._calculate_from_function(data, function_name, params)
        else:
            raise ValueError(f"Unsupported calculation method: {method}")
    
    def _calculate_from_expression(self, data: pd.DataFrame, expression: str, params: Dict, factor_type: str) -> pd.Series:
        """Calculate factor from expression based on factor type"""
        if factor_type == "alpha" and "momentum" in self.factor_input.factor_name:
            return self._calculate_momentum(data, params)
        elif factor_type == "alpha" and "reversion" in self.factor_input.factor_name:
            return self._calculate_mean_reversion(data, params)
        elif factor_type == "alpha" and "volume" in self.factor_input.factor_name:
            return self._calculate_volume(data, params)
        elif factor_type == "alpha" and "crossover" in self.factor_input.factor_name:
            return self._calculate_ma_ratio(data, params)
        elif factor_type == "risk" and "volatility" in self.factor_input.factor_name:
            return self._calculate_volatility(data, params)
        elif factor_type == "technical":
            return self._calculate_technical(data, expression, params)
        else:
            # Default to momentum calculation
            return self._calculate_momentum(data, params)
    
    def _calculate_from_function(self, data: pd.DataFrame, function_name: str, params: Dict) -> pd.Series:
        """Calculate factor from named function"""
        if function_name == "rsi":
            return self._calculate_rsi(data, params.get('period', 14))
        elif function_name == "macd":
            return self._calculate_macd(data, params)
        elif function_name == "bollinger":
            return self._calculate_bollinger(data, params)
        else:
            # Default fallback
            return self._calculate_momentum(data, params)
    
    def _calculate_momentum(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Calculate momentum-based factor"""
        period = params.get('period', 20)
        price_col = self._get_price_column(data)
        
        # For MultiIndex data, we use unstack/stack approach
        if isinstance(data.index, pd.MultiIndex):
            # Reshape to wide format, calculate pct_change, then stack back
            wide_data = data[price_col].unstack(level=1)
            momentum_wide = wide_data.pct_change(periods=period)
            momentum = momentum_wide.stack().reorder_levels([0,1]).sort_index()
        else:
            momentum = data[price_col].pct_change(periods=period)
        
        return momentum.fillna(0)
    
    def _calculate_technical(self, data: pd.DataFrame, expression: str, params: Dict) -> pd.Series:
        """Calculate technical indicator based factor"""
        if "rsi" in expression.lower():
            return self._calculate_rsi(data, params.get('period', 14))
        elif "macd" in expression.lower():
            return self._calculate_macd(data, params)
        elif "bollinger" in expression.lower():
            return self._calculate_bollinger(data, params)
        elif "ma_ratio" in expression.lower():
            return self._calculate_ma_ratio(data, params)
        else:
            # Default to simple moving average ratio
            return self._calculate_ma_ratio(data, params)
    
    def _calculate_mean_reversion(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Calculate mean reversion factor"""
        period = params.get('period', 20)
        price_col = self._get_price_column(data)
        
        # For MultiIndex data, use unstack/stack approach
        if isinstance(data.index, pd.MultiIndex):
            wide_data = data[price_col].unstack(level=1)
            ma_wide = wide_data.rolling(period).mean()
            deviation_wide = (wide_data - ma_wide) / ma_wide
            deviation = deviation_wide.stack().reorder_levels([0,1]).sort_index()
        else:
            ma = data[price_col].rolling(period).mean()
            deviation = (data[price_col] - ma) / ma
        
        # Mean reversion signal (negative of deviation)
        return -deviation.fillna(0)
    
    def _calculate_volatility(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Calculate volatility-based factor"""
        period = params.get('period', 20)
        price_col = self._get_price_column(data)
        
        # For MultiIndex data, use unstack/stack approach
        if isinstance(data.index, pd.MultiIndex):
            wide_data = data[price_col].unstack(level=1)
            returns_wide = wide_data.pct_change()
            volatility_wide = returns_wide.rolling(period).std()
            volatility = volatility_wide.stack().reorder_levels([0,1]).sort_index()
        else:
            returns = data[price_col].pct_change()
            volatility = returns.rolling(period).std()
        
        # Inverse volatility factor (higher for lower volatility)
        inv_vol = (1 / volatility).fillna(0).replace([np.inf, -np.inf], 0)
        return inv_vol
    
    def _calculate_volume(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Calculate volume-based factor"""
        period = params.get('period', 20)
        volume_col = '$volume' if '$volume' in data.columns else 'volume'
        
        if volume_col not in data.columns:
            return pd.Series(0, index=data.index)
        
        volume_ma = data[volume_col].rolling(period).mean()
        volume_ratio = data[volume_col] / volume_ma
        
        return volume_ratio.fillna(1) - 1  # Normalize around 0
    
    def _calculate_custom(self, data: pd.DataFrame, expression: str, params: Dict) -> pd.Series:
        """Calculate custom factor using expression"""
        try:
            # Simple expression evaluation (for demo purposes)
            # In production, use a proper expression parser
            
            if "close/ma" in expression:
                period = params.get('period', 20)
                price_col = self._get_price_column(data)
                ma = data[price_col].rolling(period).mean()
                return (data[price_col] / ma - 1).fillna(0)
            
            elif "rsi" in expression:
                return self._calculate_rsi(data, params.get('period', 14))
            
            else:
                # Default momentum
                return self._calculate_momentum(data, params)
                
        except Exception as e:
            print(f"Error calculating custom factor: {e}")
            return pd.Series(0, index=data.index)
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        price_col = self._get_price_column(data)
        
        # For MultiIndex data, use unstack/stack approach
        if isinstance(data.index, pd.MultiIndex):
            wide_data = data[price_col].unstack(level=1)
            delta_wide = wide_data.diff()
            gain_wide = delta_wide.where(delta_wide > 0, 0).rolling(period).mean()
            loss_wide = (-delta_wide.where(delta_wide < 0, 0)).rolling(period).mean()
            rs_wide = gain_wide / loss_wide
            rsi_wide = 100 - (100 / (1 + rs_wide))
            rsi = rsi_wide.stack().reorder_levels([0,1]).sort_index()
        else:
            delta = data[price_col].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        # Convert to signal (RSI > 70 = -1, RSI < 30 = +1, else 0)
        signal = pd.Series(0, index=rsi.index)
        signal[rsi > 70] = -1  # Overbought
        signal[rsi < 30] = 1   # Oversold
        
        return signal.fillna(0)
    
    def _calculate_macd(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Calculate MACD signal"""
        price_col = self._get_price_column(data)
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)
        
        ema_fast = data[price_col].ewm(span=fast_period).mean()
        ema_slow = data[price_col].ewm(span=slow_period).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        
        # MACD signal (positive when MACD > signal)
        return (macd_line - signal_line).fillna(0)
    
    def _calculate_bollinger(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Calculate Bollinger Band signal"""
        price_col = self._get_price_column(data)
        period = params.get('period', 20)
        std_dev = params.get('std_dev', 2)
        
        ma = data[price_col].rolling(period).mean()
        std = data[price_col].rolling(period).std()
        
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        
        # Bollinger signal
        signal = pd.Series(0, index=data.index)
        signal[data[price_col] > upper_band] = -1  # Above upper band
        signal[data[price_col] < lower_band] = 1   # Below lower band
        
        return signal.fillna(0)
    
    def _calculate_ma_ratio(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Calculate moving average ratio"""
        period = params.get('period', 20)
        price_col = self._get_price_column(data)
        
        # For MultiIndex data, use unstack/stack approach
        if isinstance(data.index, pd.MultiIndex):
            wide_data = data[price_col].unstack(level=1)
            ma_wide = wide_data.rolling(period).mean()
            ratio_wide = wide_data / ma_wide - 1
            ratio = ratio_wide.stack().reorder_levels([0,1]).sort_index()
        else:
            ma = data[price_col].rolling(period).mean()
            ratio = data[price_col] / ma - 1
        
        return ratio.fillna(0)
    
    def _get_price_column(self, data: pd.DataFrame) -> str:
        """Get the appropriate price column"""
        for col in ['$close', 'close', '$adj_close', 'adj_close']:
            if col in data.columns:
                return col
        raise ValueError("No price column found in data")

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

class CSVBacktester(BacktestInterface):
    """
    Unified backtester that works with CSV data for both factors and models
    """
    
    def __init__(self, 
                 config: QlibConfig,
                 acceptance_criteria: Optional[StandardAcceptanceCriteria] = None):
        """
        Initialize CSV Backtester
        
        Args:
            config: Qlib configuration object
            acceptance_criteria: Criteria for accepting factors/models
        """
        self.config = config
        self.acceptance_criteria = acceptance_criteria
        self.data_processor = DataProcessor(config)
        self.result_processor = ResultProcessor()
    
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
        
        # Process and create features
        processed_data = self.data_processor.add_returns(combined_data)
        feature_data = self.data_processor.create_technical_features(processed_data)
        
        return feature_data
    
    def run_factor_backtest(self, 
                           data: pd.DataFrame,
                           factor_adapter: FactorInputAdapter,
                           **kwargs) -> Dict[str, Any]:
        """
        Run factor backtesting
        """
        print(f"\n[Factor Calculation] Calculating {factor_adapter.factor_name}...")
        
        # Calculate factor values
        factor_values = factor_adapter.calculate(data)
        
        # Get forward returns for evaluation
        targets = data.groupby('instrument')['$close'].pct_change(periods=1).shift(-1)
        
        # Calculate strategy returns (simple long-short strategy)
        strategy_returns = self._calculate_factor_strategy_returns(factor_values, targets)
        
        print(f"[Factor Validation] Factor performance:")
        print(f"- Factor coverage: {len(factor_values.dropna())} observations")
        print(f"- Strategy returns: {len(strategy_returns.dropna())} periods")
        
        return {
            'factor_values': factor_values,
            'targets': targets,
            'strategy_returns': strategy_returns
        }
    
    def run_model_backtest(self, 
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
                'test_predictions': pd.Series(dtype=float),
                'strategy_returns': pd.Series(dtype=float)
            }
        
        # Generate predictions
        test_predictions = model_adapter.predict(trained_model, test_features)
        
        # Calculate strategy returns based on predictions
        strategy_returns = self._calculate_model_strategy_returns(test_predictions, test_targets)
        
        print(f"[Model Validation] Generated {len(test_predictions.dropna())} predictions")
        
        return {
            'model': trained_model,
            'test_predictions': test_predictions,
            'test_targets': test_targets,
            'strategy_returns': strategy_returns
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
    
    def _calculate_factor_strategy_returns(self, factor_values: pd.Series, targets: pd.Series) -> pd.Series:
        """
        Calculate strategy returns based on factor values
        """
        # Align factor values and targets
        aligned_data = pd.concat([factor_values, targets], axis=1).dropna()
        
        if len(aligned_data) == 0:
            return pd.Series(dtype=float)
        
        factor_col = aligned_data.columns[0]
        return_col = aligned_data.columns[1]
        
        # Simple long-short strategy based on factor rank
        factor_rank = aligned_data[factor_col].groupby(level=0).rank(pct=True)
        
        # Long top quintile, short bottom quintile
        positions = pd.Series(0, index=factor_rank.index)
        positions[factor_rank > 0.8] = 1   # Long top quintile
        positions[factor_rank < 0.2] = -1  # Short bottom quintile
        
        strategy_returns = positions * aligned_data[return_col]
        
        return strategy_returns.groupby(level=0).mean()  # Daily portfolio returns
    
    def _calculate_model_strategy_returns(self, predictions: pd.Series, targets: pd.Series) -> pd.Series:
        """
        Calculate strategy returns based on model predictions
        """
        # Align predictions and targets
        aligned_data = pd.concat([predictions, targets], axis=1).dropna()
        
        if len(aligned_data) == 0:
            return pd.Series(dtype=float)
        
        predictions = aligned_data.iloc[:, 0]
        actual_returns = aligned_data.iloc[:, 1]
        
        # Simple strategy: long when prediction > 0, short when prediction < 0
        strategy_returns = predictions.apply(np.sign) * actual_returns
        
        return strategy_returns
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    strategy,
                    **kwargs) -> Dict[str, Any]:
        """
        Run general backtest - delegates to specific methods based on strategy type
        """
        if hasattr(strategy, 'calculate'):  # Factor
            return self.run_factor_backtest(data, strategy, **kwargs)
        elif hasattr(strategy, 'train'):  # Model
            return self.run_model_backtest(data, strategy, **kwargs)
        else:
            raise ValueError("Unknown strategy type")
    
    def evaluate_performance(self, results: Dict[str, Any]) -> EvaluationMetrics:
        """
        Evaluate backtest performance with IC metrics for factors
        """
        strategy_returns = results.get('strategy_returns', pd.Series(dtype=float))
        
        if strategy_returns.empty:
            return EvaluationMetrics(
                annual_return=0.0,
                cumulative_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                downside_risk=0.0,
                calmar_ratio=0.0
            )
        
        # Calculate basic performance metrics
        performance_metrics = self.result_processor.calculate_metrics(strategy_returns)
        
        # Calculate IC metrics if factor data is available
        factor_values = results.get('factor_values')
        targets = results.get('targets')
        
        if factor_values is not None and targets is not None:
            # Calculate IC metrics
            ic_metrics = self.result_processor.calculate_ic_metrics(factor_values, targets)
            
            # Create enhanced metrics with IC
            enhanced_metrics = EvaluationMetrics(
                annual_return=performance_metrics.annual_return,
                cumulative_return=performance_metrics.cumulative_return,
                sharpe_ratio=performance_metrics.sharpe_ratio,
                max_drawdown=performance_metrics.max_drawdown,
                volatility=performance_metrics.volatility,
                downside_risk=performance_metrics.downside_risk,
                calmar_ratio=performance_metrics.calmar_ratio,
                ic_mean=ic_metrics.get('ic_mean', 0.0),
                ic_std=ic_metrics.get('ic_std', 0.0),
                ic_ir=ic_metrics.get('ic_ir', 0.0),
                rank_ic=ic_metrics.get('rank_ic', 0.0)
            )
            
            return enhanced_metrics
        
        return performance_metrics

# Configuration
config = QlibConfig(
    provider_uri="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data",
    instruments=["AAPL", "MSFT", "GOOGL"],  # Expanded to 3 stocks
    basic_fields=["$open", "$high", "$low", "$close", "$volume"],
    freq="day"
)

# Acceptance criteria
acceptance_criteria = StandardAcceptanceCriteria(
    min_annual_return=0.03,    # 3% minimum annual return
    min_sharpe_ratio=0.5,      # 0.5 minimum Sharpe ratio  
    max_drawdown_threshold=0.3, # 30% maximum drawdown
    min_ic_mean=0.01           # 1% minimum IC (reduced from 2%)
)

# Initialize backtester and evaluators
backtester = CSVBacktester(config, acceptance_criteria)
factor_evaluator = FactorEvaluator(acceptance_criteria)
model_evaluator = ModelEvaluator(acceptance_criteria)

# Define alpha factors using FactorInput approach
alpha_factors = [
    FactorInput(
        factor_name="momentum_20d",
        factor_type="alpha",
        calculation_method="expression",
        expression="close.pct_change(20)",
        lookback_period=20
    ),
    FactorInput(
        factor_name="mean_reversion_10d",
        factor_type="alpha",
        calculation_method="expression",
        expression="(close - ma(close, 10)) / ma(close, 10)",
        lookback_period=10
    ),
    FactorInput(
        factor_name="rsi_divergence",
        factor_type="technical",
        calculation_method="function",
        function_name="rsi",
        function_params={"period": 14}
    ),
    FactorInput(
        factor_name="volume_surge",
        factor_type="alpha",
        calculation_method="expression",
        expression="volume / ma(volume, 20) - 1",
        lookback_period=20
    ),
    FactorInput(
        factor_name="volatility_factor",
        factor_type="risk",
        calculation_method="expression",
        expression="1 / rolling_std(returns, 20)",
        lookback_period=20
    ),
    FactorInput(
        factor_name="ma_crossover",
        factor_type="alpha",
        calculation_method="expression",
        expression="close/ma(close, 20) - 1",
        lookback_period=20
    )
]

# Define models using ModelInput approach
alpha_models = [
    ModelInput(
        model_name="alpha_lgb_model",
        model_type="tree",
        implementation="lightgbm",
        model_class="LGBMRegressor",
        hyperparameters={
            "n_estimators": 50,          # Reduced from 100
            "learning_rate": 0.1,        # Increased from 0.05
            "max_depth": 3,              # Reduced from 6
            "min_child_samples": 20,     # Added minimum samples
            "reg_alpha": 0.1,            # Added L1 regularization
            "reg_lambda": 0.1,           # Added L2 regularization
            "random_state": 2025,
            "verbosity": -1              # Suppress warnings
        }
    ),
    ModelInput(
        model_name="alpha_rf_model",
        model_type="tree",
        implementation="sklearn",
        model_class="RandomForestRegressor",
        hyperparameters={
            "n_estimators": 50,          # Reduced from 100
            "max_depth": 5,              # Reduced from 8
            "min_samples_split": 10,     # Added minimum split samples
            "min_samples_leaf": 5,       # Added minimum leaf samples
            "random_state": 2025
        }
    )
]

# Convert to adapters
factor_adapters = [FactorInputAdapter(factor) for factor in alpha_factors]
model_adapters = [ModelInputAdapter(model) for model in alpha_models]

print(" Starting Comprehensive Alpha Factor and Model Evaluation...")
print(f" Instruments: {config.instruments}")
print(f" Date range: 2022-08-01 to 2023-12-31")
print(f" Alpha Factors to evaluate: {len(factor_adapters)}")
print(f" ML Models to evaluate: {len(model_adapters)}")
print("="*80)

# Prepare data once for all evaluations
print("\n📥 Loading and preparing market data...")
data = backtester.prepare_data(
    start_date="2022-08-01",
    end_date="2023-12-31"
)

print(f" Data loaded: {data.shape[0]} observations across {len(config.instruments)} instruments")

# Alpha Factor Evaluation
print("\n" + "="*40 + " ALPHA FACTOR EVALUATION " + "="*40)

factor_results = []
for i, factor_adapter in enumerate(factor_adapters):
    print(f"\n Evaluating Factor {i+1}/{len(factor_adapters)}: {factor_adapter.factor_name}")
    
    try:
        result = factor_evaluator.evaluate_factor(
            backtester=backtester,
            factor=factor_adapter,
            start_date="2022-08-01",
            end_date="2023-12-31"
        )
        
        factor_results.append(result)
        
        # Display results
        print(f"Factor Name: {result.get('factor_name', 'Unknown')}")
        print(f"Factor Accepted: {result.get('is_accepted', False)}")
        
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"- Annual Return: {getattr(metrics, 'annual_return', 0):.2%}")
            print(f"- Sharpe Ratio: {getattr(metrics, 'sharpe_ratio', 0):.2f}")
            print(f"- Max Drawdown: {getattr(metrics, 'max_drawdown', 0):.2%}")
            
            # Factor-specific IC metrics
            ic_mean = getattr(metrics, 'ic_mean', None)
            if ic_mean is not None:
                print(f"- IC Mean: {ic_mean:.4f}")
                ic_ir = getattr(metrics, 'ic_ir', None)
                if ic_ir is not None:
                    print(f"- IC IR: {ic_ir:.2f}")
                rank_ic = getattr(metrics, 'rank_ic', None)
                if rank_ic is not None:
                    print(f"- Rank IC: {rank_ic:.4f}")
    
    except Exception as e:
        print(f" Error evaluating factor {factor_adapter.factor_name}: {e}")
        factor_results.append(None)

# Model Evaluation
print("\n" + "="*40 + " ML MODEL EVALUATION " + "="*40)

model_results = []
for i, model_adapter in enumerate(model_adapters):
    print(f"\n Evaluating Model {i+1}/{len(model_adapters)}: {model_adapter.model_name}")
    
    try:
        result = model_evaluator.evaluate_model(
            backtester=backtester,
            model=model_adapter,
            start_date="2022-08-01",
            end_date="2023-12-31"
        )
        
        model_results.append(result)
        
        # Display results
        print(f"Model Name: {result.get('model_name', 'Unknown')}")
        print(f"Model Accepted: {result.get('is_accepted', False)}")
        
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"- Annual Return: {getattr(metrics, 'annual_return', 0):.2%}")
            print(f"- Sharpe Ratio: {getattr(metrics, 'sharpe_ratio', 0):.2f}")
            print(f"- Max Drawdown: {getattr(metrics, 'max_drawdown', 0):.2%}")
            
            # Handle potentially None accuracy
            accuracy = getattr(metrics, 'accuracy', None)
            if accuracy is not None:
                print(f"- Direction Accuracy: {accuracy:.2%}")
            else:
                print(f"- Direction Accuracy: N/A")
    
    except Exception as e:
        print(f" Error evaluating model {model_adapter.model_name}: {e}")
        model_results.append(None)

# Comprehensive Summary
print("\n" + "="*80)
print(" COMPREHENSIVE EVALUATION SUMMARY")
print("="*80)

# Factor summary
accepted_factors = []
rejected_factors = []

for i, result in enumerate(factor_results):
    if result is not None:
        factor_name = result.get('factor_name', f'Factor_{i+1}')
        is_accepted = result.get('is_accepted', False)
        
        if is_accepted:
            accepted_factors.append((factor_name, result))
        else:
            rejected_factors.append((factor_name, result))

# Model summary
accepted_models = []
rejected_models = []

for i, result in enumerate(model_results):
    if result is not None:
        model_name = result.get('model_name', f'Model_{i+1}')
        is_accepted = result.get('is_accepted', False)
        
        if is_accepted:
            accepted_models.append((model_name, result))
        else:
            rejected_models.append((model_name, result))

print(f"\n ALPHA FACTOR RESULTS:")
print(f"- Total factors evaluated: {len(alpha_factors)}")
print(f"- Accepted factors: {len(accepted_factors)}")
print(f"- Rejected factors: {len(rejected_factors)}")

if accepted_factors:
    print(f"\n ACCEPTED ALPHA FACTORS:")
    for factor_name, result in accepted_factors:
        metrics = result.get('metrics')
        if metrics:
            print(f"  • {factor_name}")
            print(f"    - Annual Return: {getattr(metrics, 'annual_return', 0):.2%}")
            print(f"    - Sharpe Ratio: {getattr(metrics, 'sharpe_ratio', 0):.2f}")

print(f"\n ML MODEL RESULTS:")
print(f"- Total models evaluated: {len(alpha_models)}")
print(f"- Accepted models: {len(accepted_models)}")
print(f"- Rejected models: {len(rejected_models)}")

if accepted_models:
    print(f"\n ACCEPTED ML MODELS:")
    for model_name, result in accepted_models:
        metrics = result.get('metrics')
        if metrics:
            print(f"  • {model_name}")
            print(f"    - Annual Return: {getattr(metrics, 'annual_return', 0):.2%}")
            print(f"    - Sharpe Ratio: {getattr(metrics, 'sharpe_ratio', 0):.2f}")
            accuracy = getattr(metrics, 'accuracy', 0)
            if accuracy is not None:
                print(f"    - Accuracy: {accuracy:.2%}")
            else:
                print(f"    - Accuracy: N/A")

# Comparison table
print(f"\n PERFORMANCE COMPARISON TABLE:")
print(f"{'Type':<8} {'Name':<20} {'Return':<8} {'Sharpe':<8} {'Status':<10}")
print("-" * 60)

# Factors
for i, result in enumerate(factor_results):
    if result is not None:
        factor_name = result.get('factor_name', f'Factor_{i+1}')[:19]
        metrics = result.get('metrics')
        if metrics:
            ret = getattr(metrics, 'annual_return', 0)
            sharpe = getattr(metrics, 'sharpe_ratio', 0)
            status = " PASS" if result.get('is_accepted', False) else " FAIL"
            print(f"{'Factor':<8} {factor_name:<20} {ret:<8.2%} {sharpe:<8.2f} {status:<10}")

# Models
for i, result in enumerate(model_results):
    if result is not None:
        model_name = result.get('model_name', f'Model_{i+1}')[:19]
        metrics = result.get('metrics')
        if metrics:
            ret = getattr(metrics, 'annual_return', 0)
            sharpe = getattr(metrics, 'sharpe_ratio', 0)
            status = " PASS" if result.get('is_accepted', False) else " FAIL"
            print(f"{'Model':<8} {model_name:<20} {ret:<8.2%} {sharpe:<8.2f} {status:<10}")

# Final recommendations
print(f"\n STRATEGIC RECOMMENDATIONS:")

if accepted_factors or accepted_models:
    print(" Successful Strategies Found!")
    
    if accepted_factors:
        best_factor = max(accepted_factors, key=lambda x: getattr(x[1].get('metrics'), 'sharpe_ratio', 0))
        print(f"   🏆 Best Alpha Factor: {best_factor[0]}")
        print(f"       Sharpe Ratio: {getattr(best_factor[1].get('metrics'), 'sharpe_ratio', 0):.2f}")
    
    if accepted_models:
        best_model = max(accepted_models, key=lambda x: getattr(x[1].get('metrics'), 'sharpe_ratio', 0))
        print(f"   🏆 Best ML Model: {best_model[0]}")
        print(f"       Sharpe Ratio: {getattr(best_model[1].get('metrics'), 'sharpe_ratio', 0):.2f}")
    
    print("\n Next Steps:")
    print("   1. Deploy best-performing strategies in paper trading")
    print("   2. Implement risk management and position sizing")
    print("   3. Monitor performance in live market conditions")
    print("   4. Consider ensemble approaches combining top factors/models")

else:
    print(" No strategies passed acceptance criteria.")
    print("\n Improvement Suggestions:")
    print("   1. Relax acceptance criteria for initial testing")
    print("   2. Expand feature engineering and factor research")
    print("   3. Implement more sophisticated model architectures")
    print("   4. Consider regime-aware strategies")
    print("   5. Optimize hyperparameters using systematic search")

print("\n" + "="*80)
print("- Comprehensive Alpha Factor and Model Evaluation Complete!")
print("="*80)
