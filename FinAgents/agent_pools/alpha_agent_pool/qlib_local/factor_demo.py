from factor_pipeline import FactorEvaluator, ExampleMomentumFactor
from interfaces import StandardAcceptanceCriteria, BacktestInterface, EvaluationMetrics, FactorInterface
from data_interfaces import FactorInput
from utils import QlibConfig, DataProcessor, ResultProcessor
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import qlib
from qlib.data import D
from qlib.constant import REG_CN

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
        return f"{self.factor_input.factor_type} factor: {self.factor_input.factor_name}"
    
    def calculate_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate factor values based on FactorInput specification
        
        Args:
            data: Market data DataFrame with MultiIndex (Date, instrument)
            
        Returns:
            pd.Series: Calculated factor values
        """
        if self.factor_input.calculation_method == "expression":
            return self._calculate_expression_factor(data)
        elif self.factor_input.calculation_method == "function":
            return self._calculate_function_factor(data)
        else:
            raise ValueError(f"Unsupported calculation method: {self.factor_input.calculation_method}")
    
    def _calculate_expression_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate factor using expression-based method
        """
        expression = self.factor_input.expression
        lookback = self.factor_input.lookback_period
        
        if expression is None:
            # Default momentum calculation if no expression provided
            if self.factor_input.factor_name.startswith("momentum"):
                return data.groupby('instrument')['$close'].pct_change(periods=lookback)
            else:
                raise ValueError("No expression provided for expression-based factor")
        
        # Parse and evaluate simple expressions
        if "close / Ref(close," in expression:
            # Handle Ref(close, N) pattern - momentum calculation
            import re
            ref_match = re.search(r'Ref\(close,\s*(\d+)\)', expression)
            if ref_match:
                periods = int(ref_match.group(1))
                if "- 1" in expression:
                    # Momentum: close / Ref(close, N) - 1
                    return data.groupby('instrument')['$close'].pct_change(periods=periods)
                else:
                    # Ratio: close / Ref(close, N)
                    return data.groupby('instrument')['$close'].pct_change(periods=periods) + 1
        
        # Add more expression patterns as needed
        raise ValueError(f"Unsupported expression pattern: {expression}")
    
    def _calculate_function_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate factor using function-based method
        """
        function_name = self.factor_input.function_name
        params = self.factor_input.function_params or {}
        
        if function_name == "momentum":
            lookback = params.get('lookback_period', self.factor_input.lookback_period)
            return data.groupby('instrument')['$close'].pct_change(periods=lookback)
        elif function_name == "rsi":
            window = params.get('window', 14)
            return self._calculate_rsi(data, window)
        elif function_name == "ma_ratio":
            window = params.get('window', 20)
            return self._calculate_ma_ratio(data, window)
        else:
            raise ValueError(f"Unsupported function: {function_name}")
    
    def _calculate_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        def rsi_calc(group):
            prices = group['$close']
            if len(prices) < window + 1:
                return pd.Series(index=prices.index, dtype=float)
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window, min_periods=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window, min_periods=window).mean()
            rs = gain / loss
            rs = rs.replace([np.inf, -np.inf], np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        result = data.groupby('instrument').apply(rsi_calc)
        # Flatten the MultiIndex to match the original data structure
        if isinstance(result.index, pd.MultiIndex):
            result.index = result.index.droplevel(0)
        return result
    
    def _calculate_ma_ratio(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate moving average ratio"""
        def ma_ratio_calc(group):
            prices = group['$close']
            if len(prices) < window:
                return pd.Series(index=prices.index, dtype=float)
            ma = prices.rolling(window, min_periods=window).mean()
            ratio = prices / ma - 1
            return ratio
        
        result = data.groupby('instrument').apply(ma_ratio_calc)
        # Flatten the MultiIndex to match the original data structure
        if isinstance(result.index, pd.MultiIndex):
            result.index = result.index.droplevel(0)
        return result
    
    def validate_factor(self, factor_values: pd.Series) -> bool:
        """
        Validate factor values
        """
        if factor_values.empty:
            return False
        
        # Check for infinite or NaN values
        if factor_values.isna().all() or np.isinf(factor_values).any():
            return False
        
        # Check expected range if specified
        if hasattr(self.factor_input, 'expected_range') and self.factor_input.expected_range:
            min_val, max_val = self.factor_input.expected_range
            if factor_values.min() < min_val or factor_values.max() > max_val:
                print(f"Warning: Factor values outside expected range {self.factor_input.expected_range}")
        
        return True

class CSVFactorBacktester(BacktestInterface):
    """
    Custom backtester that works with CSV data files
    """
    
    def __init__(self, 
                 config: QlibConfig,
                 acceptance_criteria: Optional[StandardAcceptanceCriteria] = None):
        """
        Initialize CSV Factor Backtester
        
        Args:
            config: Qlib configuration object
            acceptance_criteria: Criteria for accepting factors
        """
        self.config = config
        self.acceptance_criteria = acceptance_criteria
        self.data_processor = DataProcessor(config)
        self.result_processor = ResultProcessor()
        
        # Don't initialize Qlib since we're using CSV data directly
    
    def prepare_data(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """
        Prepare market data from CSV files
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
        print("\n[Data Snapshot] Loaded raw market data:")
        print(combined_data.head(10))
        print(f"Data shape: {combined_data.shape}")
        print(f"Data columns: {list(combined_data.columns)}")
        print(f"Sample instruments: {combined_data.index.get_level_values('instrument').unique().tolist()[:5]}")
        print(f"Sample dates: {combined_data.index.get_level_values('Date').unique().tolist()[:5]}")
        # Process and clean data using existing processor
        processed_data = self.data_processor.add_returns(combined_data)
        
        return processed_data
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    factor: Any,
                    **kwargs) -> Dict[str, Any]:
        """
        Run factor backtesting
        """
        # Calculate factor values
        factor_values = factor.calculate_factor(data)
        
        # Calculate forward returns
        forward_returns = data.groupby('instrument')['$close'].pct_change(periods=1).shift(-1)
        
        # Align factor and returns
        common_index = factor_values.index.intersection(forward_returns.index)
        aligned_factor = factor_values.loc[common_index]
        aligned_returns = forward_returns.loc[common_index]
        
        # Calculate IC metrics
        ic_mean = aligned_factor.corr(aligned_returns)
        rank_ic = aligned_factor.corr(aligned_returns, method='spearman')
        
        # Simple quintile strategy for performance calculation
        strategy_data = pd.DataFrame({
            'factor': aligned_factor,
            'returns': aligned_returns
        }).dropna()
        
        # Calculate quintile-based strategy returns
        strategy_returns = pd.Series(dtype=float)
        
        if len(strategy_data) > 0:
            # Group by date and calculate quintiles
            dates = strategy_data.index.get_level_values('Date').unique()
            daily_returns = []
            
            for date in dates:
                day_data = strategy_data[strategy_data.index.get_level_values('Date') == date]
                if len(day_data) >= 2:
                    # Simple long-short: top 50% long, bottom 50% short
                    median_factor = day_data['factor'].median()
                    long_returns = day_data[day_data['factor'] >= median_factor]['returns'].mean()
                    short_returns = day_data[day_data['factor'] < median_factor]['returns'].mean()
                    
                    if not pd.isna(long_returns) and not pd.isna(short_returns):
                        daily_returns.append(long_returns - short_returns)
            
            if daily_returns:
                strategy_returns = pd.Series(daily_returns, index=dates[:len(daily_returns)])
        
        return {
            'factor_values': aligned_factor,
            'forward_returns': aligned_returns,
            'strategy_returns': strategy_returns,
            'ic_mean': ic_mean if not pd.isna(ic_mean) else 0.0,
            'rank_ic': rank_ic if not pd.isna(rank_ic) else 0.0
        }
    
    def evaluate_performance(self, results: Dict[str, Any]) -> EvaluationMetrics:
        """
        Evaluate backtest performance
        """
        strategy_returns = results.get('strategy_returns', pd.Series(dtype=float))
        ic_mean = results.get('ic_mean', 0.0)
        rank_ic = results.get('rank_ic', 0.0)
        
        # Calculate performance metrics
        performance_metrics = self.result_processor.calculate_metrics(strategy_returns)
        
        # Create EvaluationMetrics with factor-specific data
        return EvaluationMetrics(
            annual_return=performance_metrics.annual_return,
            cumulative_return=performance_metrics.cumulative_return,
            sharpe_ratio=performance_metrics.sharpe_ratio,
            max_drawdown=performance_metrics.max_drawdown,
            volatility=performance_metrics.volatility,
            downside_risk=performance_metrics.downside_risk,
            calmar_ratio=performance_metrics.calmar_ratio,
            ic_mean=ic_mean,
            ic_std=0.0,  # Simplified for this demo
            ic_ir=0.0,   # Simplified for this demo
            rank_ic=rank_ic
        )

# 1. Configure Qlib for local US market data
config = QlibConfig(
    provider_uri="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data",
    instruments=["AAPL", "MSFT"],  # Stock universe
    basic_fields=["$open", "$high", "$low", "$close", "$volume"],  # Qlib format fields
    freq="day"  # Data frequency
)

# 2. Define acceptance criteria
acceptance_criteria = StandardAcceptanceCriteria(
    min_ic_mean=0.02,  # IC threshold (lowered for demo)
    min_sharpe_ratio=0.5,  # Sharpe ratio threshold (lowered for demo)
    max_drawdown_threshold=0.3  # Max drawdown threshold
)

# 3. Initialize backtester and evaluator
backtester = CSVFactorBacktester(config, acceptance_criteria)
evaluator = FactorEvaluator(acceptance_criteria)

# 4. Define multiple factors using FactorInput approach
factor_configs = [
    FactorInput(
        factor_name="momentum_5d",        # 5-day momentum factor
        factor_type="alpha",              # Alpha-generating factor type
        calculation_method="expression",   # Expression-based calculation
        expression="close / Ref(close, 5) - 1",  # 5-day momentum expression
        lookback_period=5,                # 5-day lookback window
        update_frequency="daily",
        expected_range=(-0.3, 0.3)
    ),
    FactorInput(
        factor_name="momentum_10d",       # 20-day momentum factor
        factor_type="alpha",
        calculation_method="expression",
        expression="close / Ref(close, 10) - 1",  # 20-day momentum expression
        lookback_period=10,
        update_frequency="daily",
        expected_range=(-0.5, 0.5)
    )
]

# Convert FactorInput to executable factors
factors = [FactorInputAdapter(factor_config) for factor_config in factor_configs]

print("Starting factor evaluation with Qlib pipeline...")
print(f"Instruments: {config.instruments}")
print(f"Date range: 2022-08-01 to 2023-12-31")
print(f"Number of factors to evaluate: {len(factors)}")

# 5. Run factor evaluation for each factor
all_results = []

for i, factor in enumerate(factors):
    print(f"\n{'='*20} Evaluating Factor {i+1}/{len(factors)} {'='*20}")
    print(f"Factor: {factor.factor_name} ({factor_configs[i].factor_type})")
    
    try:
        result = evaluator.evaluate_factor(
            backtester=backtester,
            factor=factor,
            start_date="2022-08-01",
            end_date="2023-12-31"
        )
        # Print factor values snapshot for provenance
        print("[Factor Values Snapshot] First 10 values:")
        factor_values = None
        if 'backtest_results' in result:
            factor_values = result['backtest_results'].get('factor_values')
        if factor_values is not None:
            print(factor_values.head(10))
        # Store result for comparison
        all_results.append(result)
        
        # Display individual factor results
        print(f"Factor Name: {result.get('factor_name', 'Unknown')}")
        print(f"Factor Accepted: {result.get('is_accepted', False)}")
        
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"- IC Mean: {getattr(metrics, 'ic_mean', 0):.4f}")
            print(f"- Sharpe Ratio: {getattr(metrics, 'sharpe_ratio', 0):.2f}")
            print(f"- Annual Return: {getattr(metrics, 'annual_return', 0):.2%}")
            print(f"- Max Drawdown: {getattr(metrics, 'max_drawdown', 0):.2%}")
    
    except Exception as e:
        print(f"Error evaluating factor {factor.factor_name}: {e}")
        all_results.append(None)

# 6. Summary of all factor evaluations
print("\n" + "="*60)
print("COMPREHENSIVE FACTOR EVALUATION SUMMARY")
print("="*60)

accepted_factors = []
rejected_factors = []

for i, result in enumerate(all_results):
    if result is not None:
        factor_name = result.get('factor_name', f'Factor_{i+1}')
        is_accepted = result.get('is_accepted', False)
        
        if is_accepted:
            accepted_factors.append((factor_name, result))
        else:
            rejected_factors.append((factor_name, result))

print(f"\nResults Overview:")
print(f"- Total factors evaluated: {len(factor_configs)}")
print(f"- Accepted factors: {len(accepted_factors)}")
print(f"- Rejected factors: {len(rejected_factors)}")

if accepted_factors:
    print(f"\n ACCEPTED FACTORS:")
    for factor_name, result in accepted_factors:
        metrics = result.get('metrics')
        if metrics:
            print(f"  • {factor_name}")
            print(f"    - IC: {getattr(metrics, 'ic_mean', 0):.4f}")
            print(f"    - Sharpe: {getattr(metrics, 'sharpe_ratio', 0):.2f}")
            print(f"    - Annual Return: {getattr(metrics, 'annual_return', 0):.2%}")

if rejected_factors:
    print(f"\n REJECTED FACTORS:")
    for factor_name, result in rejected_factors:
        metrics = result.get('metrics')
        if metrics:
            print(f"  • {factor_name}")
            print(f"    - IC: {getattr(metrics, 'ic_mean', 0):.4f}")
            print(f"    - Sharpe: {getattr(metrics, 'sharpe_ratio', 0):.2f}")
            print(f"    - Annual Return: {getattr(metrics, 'annual_return', 0):.2%}")

# 7. Factor comparison analysis
if len(all_results) > 1:
    print(f"\n FACTOR COMPARISON:")
    print(f"{'Factor Name':<15} {'IC':<8} {'Sharpe':<8} {'Return':<8} {'Status':<10}")
    print("-" * 55)
    
    for i, result in enumerate(all_results):
        if result is not None:
            factor_name = result.get('factor_name', f'Factor_{i+1}')[:14]
            metrics = result.get('metrics')
            if metrics:
                ic = getattr(metrics, 'ic_mean', 0)
                sharpe = getattr(metrics, 'sharpe_ratio', 0)
                ret = getattr(metrics, 'annual_return', 0)
                status = " PASS" if result.get('is_accepted', False) else " FAIL"
                print(f"{factor_name:<15} {ic:<8.4f} {sharpe:<8.2f} {ret:<8.2%} {status:<10}")

print(f"\n RECOMMENDATION:")
if accepted_factors:
    best_factor = max(accepted_factors, key=lambda x: getattr(x[1].get('metrics'), 'ic_mean', 0))
    print(f"Best performing factor: {best_factor[0]}")
    print(f"Consider using this factor for trading strategy implementation.")
else:
    print("No factors passed acceptance criteria.")
    print("Consider:")
    print("- Adjusting factor parameters")
    print("- Trying different lookback periods")
    print("- Exploring different factor types")
    print("- Relaxing acceptance criteria for initial testing")