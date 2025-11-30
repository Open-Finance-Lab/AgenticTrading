"""
Alpha Signal Agent using OpenAI Agent SDK

This agent integrates Qlib factor construction, technical indicators, and ML inference
to generate alpha trading signals based on algorithmic trading research.

Key Features:
- Qlib-based factor construction
- Technical indicator calculation
- Linear regression and ML model inference
- Alpha signal generation
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
alpha_agent_pool_path = parent_dir / "alpha_agent_pool"
sys.path.insert(0, str(alpha_agent_pool_path))

# Import OpenAI Agent SDK (using existing agents.py pattern)
try:
    from agents import Agent, function_tool
except ImportError:
    # Fallback if agents.py is not found
    print("Warning: agents.py not found. Creating minimal Agent class.")
    from openai import OpenAI
    import json
    import inspect
    
    def function_tool(func, name=None, description=None):
        func.is_tool = True
        func.name = name or func.__name__
        func.description = description or func.__doc__ or "No description available"
        return func
    
    class Agent:
        def __init__(self, name="Agent", instructions="", model="gpt-4o-mini", tools=None):
            self.name = name
            self.instructions = instructions
            self.model = model
            self.tools = tools or []
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        def run(self, user_request, context=None, max_turns=10):
            return f"Agent {self.name} executed with request: {user_request[:100]}"

# Import Qlib utilities
try:
    qlib_path = alpha_agent_pool_path / "qlib"
    sys.path.insert(0, str(qlib_path))
    from utils import QlibConfig, DataProcessor
    from data_interfaces import FactorInput, ModelInput
    from standard_factor_calculator import StandardFactorCalculator
except ImportError as e:
    print(f"Warning: Qlib modules not found: {e}. Some features may be limited.")
    # Create minimal stubs
    from dataclasses import dataclass, field
    from typing import List
    
    @dataclass
    class QlibConfig:
        provider_uri: str = ""
        instruments: List[str] = field(default_factory=list)
    
    class DataProcessor:
        def __init__(self, config):
            self.config = config
        def add_returns(self, data):
            return data
        def create_technical_features(self, data):
            return data
    
    @dataclass
    class FactorInput:
        factor_name: str = ""
        factor_type: str = ""
        calculation_method: str = ""
        expression: str = None
        function_name: str = None
        lookback_period: int = 20
    
    class StandardFactorCalculator:
        def calculate(self, data, factor_input):
            return pd.Series(dtype=float)

# Import ML libraries
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Warning: ML libraries not found. Install sklearn and lightgbm for full functionality.")


# ==============================
# Qlib Factor Construction Tools
# ==============================

@function_tool
def construct_qlib_factor(
    factor_name: str,
    factor_type: str,
    calculation_method: str,
    lookback_period: int = 20,
    expression: Optional[str] = None,
    function_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Construct a factor using Qlib's factor construction framework.
    
    Args:
        factor_name: Name of the factor (e.g., 'momentum_20d')
        factor_type: Type of factor ('alpha', 'technical', 'risk', etc.)
        calculation_method: 'expression' or 'function'
        lookback_period: Number of periods to look back
        expression: Qlib expression string (e.g., 'close / Ref(close, 20) - 1')
        function_name: Function name if using function method
        
    Returns:
        Dictionary with factor configuration
    """
    try:
        factor_input = FactorInput(
            factor_name=factor_name,
            factor_type=factor_type,
            calculation_method=calculation_method,
            expression=expression,
            function_name=function_name,
            lookback_period=lookback_period,
            update_frequency="daily"
        )
        
        return {
            "status": "success",
            "factor_name": factor_name,
            "factor_config": {
                "factor_type": factor_type,
                "calculation_method": calculation_method,
                "lookback_period": lookback_period,
                "expression": expression,
                "function_name": function_name
            },
            "message": f"Factor {factor_name} constructed successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to construct factor: {str(e)}"
        }


@function_tool
def calculate_qlib_factor(
    data: pd.DataFrame,
    factor_config: Dict[str, Any]
) -> pd.Series:
    """
    Calculate factor values using Qlib's standard factor calculator.
    
    Args:
        data: Market data DataFrame with MultiIndex (Date, instrument)
        factor_config: Factor configuration dictionary
        
    Returns:
        Series with calculated factor values
    """
    try:
        factor_input = FactorInput(
            factor_name=factor_config.get("factor_name", "custom_factor"),
            factor_type=factor_config.get("factor_type", "alpha"),
            calculation_method=factor_config.get("calculation_method", "expression"),
            expression=factor_config.get("expression"),
            function_name=factor_config.get("function_name"),
            lookback_period=factor_config.get("lookback_period", 20)
        )
        
        calculator = StandardFactorCalculator()
        factor_values = calculator.calculate(data, factor_input)
        
        return {
            "status": "success",
            "factor_values": factor_values.to_dict(),
            "summary": {
                "mean": float(factor_values.mean()),
                "std": float(factor_values.std()),
                "min": float(factor_values.min()),
                "max": float(factor_values.max()),
                "valid_count": int(factor_values.notna().sum())
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to calculate factor: {str(e)}"
        }


# ==============================
# Technical Indicator Tools
# ==============================

@function_tool
def calculate_technical_indicators(
    data: pd.DataFrame,
    indicators: List[str],
    periods: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Calculate technical indicators from market data.
    
    Supported indicators:
    - RSI: Relative Strength Index
    - MACD: Moving Average Convergence Divergence
    - Bollinger: Bollinger Bands position
    - MA: Moving Average ratios
    - Momentum: Price momentum
    - Volume: Volume-based indicators
    
    Args:
        data: Market data with OHLCV columns
        indicators: List of indicator names to calculate
        periods: Dictionary mapping indicator names to periods
        
    Returns:
        Dictionary with calculated indicators
    """
    try:
        if periods is None:
            periods = {}
        
        # Identify price columns
        close_col = None
        for col in ['$close', 'close', 'Close']:
            if col in data.columns:
                close_col = col
                break
        
        if close_col is None:
            return {"status": "error", "message": "No close price column found"}
        
        results = {}
        
        # RSI calculation
        if 'RSI' in indicators or 'rsi' in indicators:
            period = periods.get('RSI', 14)
            delta = data[close_col].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            results['RSI'] = rsi.to_dict()
        
        # MACD calculation
        if 'MACD' in indicators or 'macd' in indicators:
            fast_period = periods.get('MACD_fast', 12)
            slow_period = periods.get('MACD_slow', 26)
            signal_period = periods.get('MACD_signal', 9)
            
            ema_fast = data[close_col].ewm(span=fast_period).mean()
            ema_slow = data[close_col].ewm(span=slow_period).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=signal_period).mean()
            histogram = macd - signal
            
            results['MACD'] = macd.to_dict()
            results['MACD_signal'] = signal.to_dict()
            results['MACD_histogram'] = histogram.to_dict()
        
        # Bollinger Bands
        if 'Bollinger' in indicators or 'bollinger' in indicators:
            period = periods.get('Bollinger', 20)
            std_mult = periods.get('Bollinger_std', 2)
            
            ma = data[close_col].rolling(window=period).mean()
            std = data[close_col].rolling(window=period).std()
            upper = ma + (std * std_mult)
            lower = ma - (std * std_mult)
            
            # Position within bands (normalized -1 to 1)
            position = (data[close_col] - ma) / (std + 1e-10)
            results['Bollinger_position'] = position.to_dict()
            results['Bollinger_upper'] = upper.to_dict()
            results['Bollinger_lower'] = lower.to_dict()
        
        # Moving Averages
        if 'MA' in indicators or 'ma' in indicators:
            for ma_period in [5, 10, 20, 50]:
                ma = data[close_col].rolling(window=ma_period).mean()
                ma_ratio = (data[close_col] / ma - 1)
                results[f'MA_{ma_period}_ratio'] = ma_ratio.to_dict()
        
        # Momentum
        if 'Momentum' in indicators or 'momentum' in indicators:
            for mom_period in [5, 10, 20]:
                momentum = data[close_col].pct_change(periods=mom_period)
                results[f'Momentum_{mom_period}'] = momentum.to_dict()
        
        # Volume indicators
        if 'Volume' in indicators or 'volume' in indicators:
            volume_col = None
            for col in ['$volume', 'volume', 'Volume']:
                if col in data.columns:
                    volume_col = col
                    break
            
            if volume_col:
                volume_ma = data[volume_col].rolling(window=20).mean()
                volume_ratio = data[volume_col] / (volume_ma + 1e-10)
                results['Volume_ratio'] = volume_ratio.to_dict()
        
        return {
            "status": "success",
            "indicators": results,
            "message": f"Calculated {len(results)} technical indicators"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to calculate indicators: {str(e)}"
        }


# ==============================
# ML Inference Tools
# ==============================

@function_tool
def train_linear_regression_model(
    features: pd.DataFrame,
    targets: pd.Series,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Train a linear regression model for alpha signal prediction.
    
    Args:
        features: Feature matrix (DataFrame)
        targets: Target values (forward returns)
        test_size: Proportion of data for testing
        
    Returns:
        Dictionary with model information and performance metrics
    """
    try:
        # Align features and targets
        aligned_data = pd.concat([features, targets], axis=1).dropna()
        if len(aligned_data) < 50:
            return {
                "status": "error",
                "message": "Insufficient data for training (need at least 50 samples)"
            }
        
        X = aligned_data.iloc[:, :-1]
        y = aligned_data.iloc[:, -1]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_corr = np.corrcoef(y_train, train_pred)[0, 1]
        test_corr = np.corrcoef(y_test, test_pred)[0, 1]
        
        train_mae = np.mean(np.abs(y_train - train_pred))
        test_mae = np.mean(np.abs(y_test - test_pred))
        
        return {
            "status": "success",
            "model_type": "LinearRegression",
            "performance": {
                "train_correlation": float(train_corr) if not np.isnan(train_corr) else 0.0,
                "test_correlation": float(test_corr) if not np.isnan(test_corr) else 0.0,
                "train_mae": float(train_mae),
                "test_mae": float(test_mae),
                "n_features": len(X.columns),
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test)
            },
            "feature_importance": dict(zip(X.columns, model.coef_)),
            "message": "Linear regression model trained successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to train model: {str(e)}"
        }


@function_tool
def train_ml_model(
    features: pd.DataFrame,
    targets: pd.Series,
    model_type: str = "lightgbm",
    hyperparameters: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Train a machine learning model (LightGBM or RandomForest) for alpha prediction.
    
    Args:
        features: Feature matrix
        targets: Target values (forward returns)
        model_type: 'lightgbm' or 'random_forest'
        hyperparameters: Model hyperparameters
        test_size: Proportion of data for testing
        
    Returns:
        Dictionary with model information and performance metrics
    """
    try:
        # Align features and targets
        aligned_data = pd.concat([features, targets], axis=1).dropna()
        if len(aligned_data) < 50:
            return {
                "status": "error",
                "message": "Insufficient data for training"
            }
        
        X = aligned_data.iloc[:, :-1]
        y = aligned_data.iloc[:, -1]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Default hyperparameters
        if hyperparameters is None:
            if model_type == "lightgbm":
                hyperparameters = {
                    "n_estimators": 100,
                    "learning_rate": 0.05,
                    "max_depth": 5,
                    "random_state": 42
                }
            else:
                hyperparameters = {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "random_state": 42
                }
        
        # Train model
        if model_type == "lightgbm":
            model = lgb.LGBMRegressor(**hyperparameters)
        elif model_type == "random_forest":
            model = RandomForestRegressor(**hyperparameters)
        else:
            return {
                "status": "error",
                "message": f"Unsupported model type: {model_type}"
            }
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_corr = np.corrcoef(y_train, train_pred)[0, 1]
        test_corr = np.corrcoef(y_test, test_pred)[0, 1]
        
        train_mae = np.mean(np.abs(y_train - train_pred))
        test_mae = np.mean(np.abs(y_test - test_pred))
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
        else:
            feature_importance = {}
        
        return {
            "status": "success",
            "model_type": model_type,
            "performance": {
                "train_correlation": float(train_corr) if not np.isnan(train_corr) else 0.0,
                "test_correlation": float(test_corr) if not np.isnan(test_corr) else 0.0,
                "train_mae": float(train_mae),
                "test_mae": float(test_mae),
                "n_features": len(X.columns),
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test)
            },
            "feature_importance": feature_importance,
            "message": f"{model_type} model trained successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to train model: {str(e)}"
        }


@function_tool
def generate_alpha_signals(
    model_predictions: pd.Series,
    signal_threshold: float = 0.0,
    signal_type: str = "long_short"
) -> Dict[str, Any]:
    """
    Generate alpha trading signals from model predictions.
    
    Args:
        model_predictions: Model predictions (expected returns)
        signal_threshold: Minimum prediction value to generate signal
        signal_type: 'long_only', 'short_only', or 'long_short'
        
    Returns:
        Dictionary with trading signals
    """
    try:
        signals = pd.Series(index=model_predictions.index, dtype=float)
        
        if signal_type == "long_only":
            signals = (model_predictions > signal_threshold).astype(float)
        elif signal_type == "short_only":
            signals = (model_predictions < -signal_threshold).astype(float) * -1
        else:  # long_short
            signals = np.where(
                model_predictions > signal_threshold, 1.0,
                np.where(model_predictions < -signal_threshold, -1.0, 0.0)
            )
        
        signals = pd.Series(signals, index=model_predictions.index)
        
        long_signals = (signals > 0).sum()
        short_signals = (signals < 0).sum()
        neutral_signals = (signals == 0).sum()
        
        return {
            "status": "success",
            "signals": signals.to_dict(),
            "summary": {
                "total_signals": len(signals),
                "long_signals": int(long_signals),
                "short_signals": int(short_signals),
                "neutral_signals": int(neutral_signals),
                "signal_strength_mean": float(signals.abs().mean()),
                "signal_strength_std": float(signals.abs().std())
            },
            "message": f"Generated {len(signals)} alpha signals"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to generate signals: {str(e)}"
        }


# ==============================
# Alpha Signal Agent
# ==============================

class AlphaSignalAgent:
    """
    Alpha Signal Agent using OpenAI Agent SDK
    
    This agent combines Qlib factor construction, technical indicators,
    and ML inference to generate alpha trading signals.
    """
    
    def __init__(
        self,
        name: str = "AlphaSignalAgent",
        model: str = "gpt-4o-mini",
        qlib_config: Optional[QlibConfig] = None
    ):
        """
        Initialize the Alpha Signal Agent
        
        Args:
            name: Agent name
            model: OpenAI model to use
            qlib_config: Qlib configuration (optional)
        """
        self.name = name
        self.model = model
        self.qlib_config = qlib_config or QlibConfig()
        self.data_processor = DataProcessor(self.qlib_config)
        
        # Register all tools
        self.tools = [
            construct_qlib_factor,
            calculate_qlib_factor,
            calculate_technical_indicators,
            train_linear_regression_model,
            train_ml_model,
            generate_alpha_signals
        ]
        
        # Initialize OpenAI Agent
        instructions = """
        You are an Alpha Signal Agent specialized in algorithmic trading.
        Your role is to:
        1. Construct factors using Qlib's factor construction framework
        2. Calculate technical indicators from market data
        3. Train linear regression or ML models to predict returns
        4. Generate alpha trading signals based on model predictions
        
        When working with market data:
        - Always validate data quality before processing
        - Use appropriate lookback periods for factors
        - Combine multiple factors and indicators for robust signals
        - Evaluate model performance before generating signals
        - Consider risk management in signal generation
        
        Provide clear explanations of your reasoning and methodology.
        """
        
        self.agent = Agent(
            name=name,
            instructions=instructions,
            model=model,
            tools=self.tools
        )
    
    def run(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute the agent with a user request
        
        Args:
            user_request: User's request or query
            context: Optional context dictionary (e.g., market data)
            
        Returns:
            Agent response
        """
        return self.agent.run(user_request, context=context, max_turns=10)
    
    def generate_signals_from_data(
        self,
        data: pd.DataFrame,
        factors: List[Dict[str, Any]],
        indicators: List[str],
        model_type: str = "linear",
        signal_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Complete pipeline: factors + indicators -> model -> signals
        
        Args:
            data: Market data DataFrame
            factors: List of factor configurations
            indicators: List of technical indicators to calculate
            model_type: 'linear' or 'lightgbm' or 'random_forest'
            signal_threshold: Threshold for signal generation
            
        Returns:
            Dictionary with signals and metadata
        """
        try:
            # Step 1: Normalize data format
            # Convert to standard format with date and symbol columns if needed
            if isinstance(data.index, pd.MultiIndex):
                # Convert MultiIndex to columns
                data = data.reset_index()
                if 'Date' in data.columns:
                    data = data.rename(columns={'Date': 'date'})
                if 'instrument' in data.columns:
                    data = data.rename(columns={'instrument': 'symbol'})
            elif 'date' not in data.columns and 'Date' in data.columns:
                data = data.rename(columns={'Date': 'date'})
            if 'symbol' not in data.columns and 'instrument' in data.columns:
                data = data.rename(columns={'instrument': 'symbol'})
            
            # Ensure we have date and symbol columns
            if 'date' not in data.columns:
                raise ValueError("Data must have 'date' column")
            if 'symbol' not in data.columns:
                raise ValueError("Data must have 'symbol' column")
            
            # Rename price columns if needed (support both formats)
            column_mapping = {}
            if '$close' in data.columns:
                column_mapping['$close'] = 'close'
            if '$open' in data.columns:
                column_mapping['$open'] = 'open'
            if '$high' in data.columns:
                column_mapping['$high'] = 'high'
            if '$low' in data.columns:
                column_mapping['$low'] = 'low'
            if '$volume' in data.columns:
                column_mapping['$volume'] = 'volume'
            if column_mapping:
                data = data.rename(columns=column_mapping)
            
            # Process data
            processed_data = self.data_processor.add_returns(data)
            processed_data = self.data_processor.create_technical_features(processed_data)
            
            # Step 2: Calculate factors
            all_features = []
            factor_names = []
            
            calculator = StandardFactorCalculator()
            for factor_config in factors:
                factor_input = FactorInput(**factor_config)
                factor_values = calculator.calculate(processed_data, factor_input)
                all_features.append(factor_values)
                factor_names.append(factor_config['factor_name'])
            
            # Step 3: Calculate technical indicators
            indicator_results = calculate_technical_indicators(
                processed_data,
                indicators
            )
            
            # Step 4: Prepare features and targets
            # Create MultiIndex from date and symbol
            processed_data = processed_data.set_index(['date', 'symbol'])
            
            feature_df = pd.DataFrame(index=processed_data.index)
            
            # Add factors (factors are already MultiIndex Series with date, symbol)
            for i, factor_series in enumerate(all_features):
                if isinstance(factor_series, pd.Series):
                    feature_df[factor_names[i]] = factor_series
            
            # Add technical indicators
            if indicator_results['status'] == 'success':
                for ind_name, ind_values in indicator_results['indicators'].items():
                    if isinstance(ind_values, dict):
                        ind_series = pd.Series(ind_values)
                        # Try to align with feature_df index
                        if len(ind_series) > 0:
                            # If indicator is per-row, align by position
                            if len(ind_series) == len(processed_data):
                                feature_df[ind_name] = ind_series.values
                            else:
                                # Try to match by index
                                common_idx = feature_df.index.intersection(ind_series.index)
                                if len(common_idx) > 0:
                                    feature_df.loc[common_idx, ind_name] = ind_series.loc[common_idx]
            
            # Create target (forward returns)
            close_col = 'close'
            if close_col not in processed_data.columns:
                close_col = '$close' if '$close' in processed_data.columns else 'Close'
            
            # Calculate forward returns grouped by symbol
            targets = processed_data.groupby('symbol')[close_col].pct_change(periods=1).shift(-1)
            
            # Align features and targets
            aligned_data = pd.concat([feature_df, targets], axis=1).dropna()
            if len(aligned_data) < 50:
                return {
                    "status": "error",
                    "message": "Insufficient data after alignment"
                }
            
            X = aligned_data.iloc[:, :-1]
            y = aligned_data.iloc[:, -1]
            
            # Step 5: Train model
            if model_type == "linear":
                model_result = train_linear_regression_model(X, y)
            else:
                model_result = train_ml_model(X, y, model_type=model_type)
            
            if model_result['status'] != 'success':
                return model_result
            
            # Step 6: Generate predictions and signals
            if model_type == "linear":
                from sklearn.linear_model import LinearRegression
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model = LinearRegression()
                model.fit(X_scaled, y)
                predictions = pd.Series(
                    model.predict(X_scaled),
                    index=X.index
                )
            else:
                if model_type == "lightgbm":
                    model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                predictions = pd.Series(model.predict(X), index=X.index)
            
            # Generate signals
            signal_result = generate_alpha_signals(
                predictions,
                signal_threshold=signal_threshold
            )
            
            return {
                "status": "success",
                "model_performance": model_result['performance'],
                "signals": signal_result['signals'],
                "signal_summary": signal_result['summary'],
                "feature_importance": model_result.get('feature_importance', {}),
                "n_features": len(X.columns),
                "n_samples": len(X)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Pipeline failed: {str(e)}"
            }


# ==============================
# Main execution
# ==============================

if __name__ == "__main__":
    # Example usage
    print("Alpha Signal Agent - Example Usage")
    print("=" * 50)
    
    # Initialize agent
    agent = AlphaSignalAgent(
        name="AlphaSignalAgent",
        model="gpt-4o-mini"
    )
    
    # Example request
    request = """
    I want to generate alpha signals using:
    1. Momentum factor (20-day lookback)
    2. RSI and MACD technical indicators
    3. Linear regression model
    
    Please explain the process and generate signals.
    """
    
    print("\nAgent Request:")
    print(request)
    print("\n" + "=" * 50)
    print("Agent Response:")
    print("=" * 50)
    
    # Note: This requires OpenAI API key
    # response = agent.run(request)
    # print(response)
    
    print("\nNote: To use the agent, set OPENAI_API_KEY environment variable")
    print("Example: export OPENAI_API_KEY='your-api-key'")

