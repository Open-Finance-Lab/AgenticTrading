"""
Data Input Interfaces for Qlib Backtesting Framework
Defines exact input formats for datasets, factors, models, and strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class DatasetInput:
    """
    Exact input format for dataset specification
    """
    # Data source configuration
    source_type: str  # "qlib", "csv", "parquet", "api", "synthetic"
    
    # Time range
    start_date: str  # "YYYY-MM-DD"
    end_date: str    # "YYYY-MM-DD"
    
    # For qlib data
    market: Optional[str] = None  # "csi300", "csi500", "nasdaq100", etc.
    
    # For file-based data
    file_path: Optional[str] = None
    
    # For API data
    api_config: Optional[Dict[str, Any]] = None
    
    # Data fields required
    required_fields: Optional[List[str]] = None  # ["open", "high", "low", "close", "volume", "amount"]
    
    # Universe selection
    universe: Optional[str] = None  # "all", "top_100", "custom_list"
    custom_symbols: Optional[List[str]] = None
    
    # Data preprocessing
    adjust_price: bool = True  # Adjust for splits/dividends
    fill_method: str = "ffill"  # "ffill", "bfill", "interpolate", "drop"
    min_periods: int = 252  # Minimum trading days required
    
    def __post_init__(self):
        if self.required_fields is None:
            self.required_fields = ["open", "high", "low", "close", "volume"]


@dataclass
class FactorInput:
    """
    Exact input format for factor specification
    """
    # Factor identification
    factor_name: str
    factor_type: str  # "alpha", "risk", "technical", "fundamental", "macro"
    
    # Factor calculation parameters
    calculation_method: str  # "expression", "function", "class"
    
    # For expression-based factors
    expression: Optional[str] = None  # "rank(close/Ref(close, 20) - 1)"
    
    # For function-based factors
    function_name: Optional[str] = None
    function_params: Optional[Dict[str, Any]] = None
    
    # For class-based factors
    factor_class: Optional[str] = None
    class_params: Optional[Dict[str, Any]] = None
    
    # Factor properties
    lookback_period: int = 20  # Days needed for calculation
    update_frequency: str = "daily"  # "daily", "weekly", "monthly"
    neutralization: Optional[List[str]] = None  # ["industry", "market_cap", "volatility"]
    
    # Factor validation
    expected_range: Optional[Tuple[float, float]] = None  # (min_value, max_value)
    expected_distribution: Optional[str] = None  # "normal", "uniform", "skewed"
    
    def __post_init__(self):
        if self.neutralization is None:
            self.neutralization = []


@dataclass
class ModelInput:
    """
    Exact input format for model specification
    """
    # Model identification
    model_name: str
    model_type: str  # "linear", "tree", "neural_network", "ensemble"
    
    # Model implementation
    implementation: str  # "sklearn", "lightgbm", "pytorch", "custom"
    model_class: str  # "LinearRegression", "LGBMRegressor", etc.
    
    # Model parameters
    hyperparameters: Dict[str, Any]
    
    # Training configuration
    training_method: str = "rolling"  # "rolling", "expanding", "fixed"
    training_period: int = 252  # Days for training
    validation_period: int = 63  # Days for validation
    rebalance_frequency: str = "monthly"  # "daily", "weekly", "monthly", "quarterly"
    
    # Feature engineering
    feature_engineering: Optional[Dict[str, Any]] = None
    
    # Model validation
    cross_validation: bool = True
    cv_folds: int = 5
    early_stopping: bool = True
    
    def __post_init__(self):
        if self.feature_engineering is None:
            self.feature_engineering = {
                "scaling": "standard",  # "standard", "minmax", "robust", "none"
                "missing_values": "median",  # "median", "mean", "forward_fill", "drop"
                "outlier_treatment": "winsorize"  # "winsorize", "clip", "remove", "none"
            }


@dataclass
class StrategyInput:
    """
    Exact input format for strategy specification
    """
    # Strategy identification
    strategy_name: str
    strategy_type: str  # "long_only", "long_short", "market_neutral"
    
    # Position sizing
    position_method: str = "equal_weight"  # "equal_weight", "factor_weight", "optimization"
    max_position_size: float = 0.05  # Maximum weight per position (5%)
    min_position_size: float = 0.001  # Minimum weight per position (0.1%)
    
    # Portfolio construction
    num_positions: int = 50  # Number of positions to hold
    long_ratio: float = 1.0  # For long-short strategies
    leverage: float = 1.0  # Total leverage
    
    # Rebalancing
    rebalance_frequency: str = "monthly"  # "daily", "weekly", "monthly", "quarterly"
    rebalance_threshold: float = 0.05  # Rebalance if weight deviation > 5%
    
    # Risk management
    stop_loss: Optional[float] = None  # Stop loss level (e.g., -0.1 for -10%)
    take_profit: Optional[float] = None  # Take profit level (e.g., 0.2 for 20%)
    max_drawdown_limit: Optional[float] = None  # Maximum allowed drawdown
    
    # Signal generation
    signal_threshold: float = 0.0  # Threshold for signal generation (default: 0.0)
    
    # Transaction costs
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    
    # Benchmark configuration
    benchmark_symbols: Optional[List[str]] = None  # ETFs for comparison
    
    def __post_init__(self):
        if self.benchmark_symbols is None:
            self.benchmark_symbols = ["SPY", "QQQ", "IWM"]  # Default ETF benchmarks


@dataclass
class OutputFormat:
    """
    Exact output format specification
    """
    # Report types to generate
    generate_summary_report: bool = True
    generate_detailed_report: bool = True
    generate_factor_analysis: bool = True
    generate_risk_analysis: bool = True
    
    # Chart types to generate
    generate_performance_chart: bool = True
    generate_drawdown_chart: bool = True
    generate_rolling_metrics_chart: bool = True
    generate_factor_exposure_chart: bool = True
    generate_correlation_matrix: bool = True
    
    # Advanced visualizations
    generate_monthly_heatmap: bool = True
    generate_risk_return_scatter: bool = True
    generate_rolling_beta_chart: bool = True
    generate_underwater_plot: bool = True
    generate_return_distribution: bool = True
    generate_position_concentration: bool = True
    generate_factor_exposure_lines: bool = True
    generate_performance_attribution: bool = True
    generate_excess_return_chart: bool = True      # NEW: Excess return comparison chart
    generate_signal_analysis_chart: bool = True    # NEW: Strategy signal analysis chart
    
    # Comparison analysis
    include_etf_comparison: bool = True
    etf_symbols: Optional[List[str]] = None  # ETFs to compare against
    
    # Real data support for ETF benchmarks
    etf_data_source: str = "synthetic"  # "synthetic" or "qlib_data" 
    etf_data_directory: Optional[str] = None  # Path to qlib_data directory when using real data
    
    # Output formats
    save_to_html: bool = True
    save_to_pdf: bool = False
    save_to_excel: bool = True
    save_raw_data: bool = True
    
    # Output directory
    output_directory: str = "./results"
    
    def __post_init__(self):
        if self.etf_symbols is None:
            self.etf_symbols = ["SPY", "QQQ", "IWM", "VTI", "VXUS"]


class DatasetInterface(ABC):
    """Interface for dataset providers"""
    
    @abstractmethod
    def load_data(self, dataset_input: DatasetInput) -> pd.DataFrame:
        """
        Load dataset according to specification
        
        Returns:
            pd.DataFrame: Multi-index DataFrame with (datetime, symbol) index
                         and columns for OHLCV data
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate loaded data quality"""
        pass


class FactorInterface(ABC):
    """Interface for factor implementations"""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, factor_input: FactorInput) -> pd.Series:
        """
        Calculate factor values
        
        Returns:
            pd.Series: Multi-index Series with (datetime, symbol) index
        """
        pass
    
    @abstractmethod
    def validate_factor(self, factor_values: pd.Series) -> Dict[str, Any]:
        """Validate calculated factor values"""
        pass


class ModelInterface(ABC):
    """Interface for model implementations"""
    
    @abstractmethod
    def train(self, features: pd.DataFrame, targets: pd.Series, 
              model_input: ModelInput) -> Any:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, features: pd.DataFrame, model: Any) -> pd.Series:
        """Generate predictions"""
        pass
    
    @abstractmethod
    def validate_model(self, model: Any, validation_data: Tuple) -> Dict[str, Any]:
        """Validate trained model"""
        pass


class StrategyInterface(ABC):
    """Interface for strategy implementations"""
    
    @abstractmethod
    def generate_signals(self, predictions: pd.Series, 
                        strategy_input: StrategyInput) -> pd.DataFrame:
        """
        Generate trading signals
        
        Returns:
            pd.DataFrame: Columns ['symbol', 'weight', 'signal'] for each date
        """
        pass
    
    @abstractmethod
    def construct_portfolio(self, signals: pd.DataFrame, 
                          strategy_input: StrategyInput) -> pd.DataFrame:
        """
        Construct portfolio from signals
        
        Returns:
            pd.DataFrame: Portfolio weights with datetime index and symbol columns
        """
        pass


# Example input configurations
EXAMPLE_DATASET_INPUT = DatasetInput(
    source_type="synthetic",
    start_date="2020-01-01",
    end_date="2023-12-31",
    required_fields=["open", "high", "low", "close", "volume"],
    universe="top_100",
    adjust_price=True,
    fill_method="ffill",
    min_periods=252
)

EXAMPLE_FACTOR_INPUT = FactorInput(
    factor_name="momentum_20d",
    factor_type="alpha",
    calculation_method="expression",
    expression="close / Ref(close, 20) - 1",
    lookback_period=20,
    update_frequency="daily",
    neutralization=["industry"],
    expected_range=(-0.5, 0.5)
)

EXAMPLE_MODEL_INPUT = ModelInput(
    model_name="lgb_alpha_model",
    model_type="tree",
    implementation="lightgbm",
    model_class="LGBMRegressor",
    hyperparameters={
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "random_state": 42
    },
    training_method="rolling",
    training_period=252,
    validation_period=63,
    rebalance_frequency="monthly"
)

EXAMPLE_STRATEGY_INPUT = StrategyInput(
    strategy_name="long_only_equity",
    strategy_type="long_only",
    position_method="factor_weight",
    max_position_size=0.05,
    num_positions=50,
    rebalance_frequency="monthly",
    transaction_cost=0.001,
    benchmark_symbols=["SPY", "QQQ", "IWM"]
)

EXAMPLE_OUTPUT_FORMAT = OutputFormat(
    generate_summary_report=True,
    generate_detailed_report=True,
    generate_performance_chart=True,
    include_etf_comparison=True,
    etf_symbols=["SPY", "QQQ", "IWM", "VTI"],
    save_to_html=True,
    save_to_excel=True,
    output_directory="./backtest_results"
)
