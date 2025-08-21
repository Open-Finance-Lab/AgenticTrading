"""
Output Processor for Qlib Backtesting Framework
Generates comprehensive reports and charts comparing strategy performance with ETF benchmarks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_interfaces import OutputFormat


@dataclass
class PerformanceMetrics:
    """Standardized performance metrics output"""
    
    # Return metrics
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    calmar_ratio: float
    
    # Risk metrics
    max_drawdown: float
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional Value at Risk
    
    # Trading metrics
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    
    # Advanced metrics
    sortino_ratio: float
    information_ratio: float
    beta: float
    alpha: float
    
    # Additional trading metrics
    total_trades: int
    turnover_rate: float
    hit_ratio: float  # Percentage of profitable trades
    
    # Risk-adjusted metrics
    treynor_ratio: float
    tracking_error: float
    downside_deviation: float
    
    # Factor-specific metrics
    factor_turnover: float
    max_position_weight: float
    avg_num_positions: float
    
    # Tail risk metrics
    skewness: float
    kurtosis: float
    max_consecutive_losses: int
    
    # Comparison metrics (vs benchmark)
    excess_return: float
    up_capture: float
    down_capture: float
    
    # Additional hourly-specific metrics for comprehensive analysis
    turnover_hourly: float = 0.0
    turnover_annual: float = 0.0
    cost_ratio: float = 0.0
    gross_return: float = 0.0
    net_return: float = 0.0
    r_squared: float = 0.0
    hit_ratio_hourly: float = 0.0
    hit_ratio_daily: float = 0.0
    concentration_hhi: float = 0.0  # Herfindahl-Hirschman Index for position concentration
    top3_weight_ratio: float = 0.0
    top5_weight_ratio: float = 0.0
    recovery_time_days: float = 0.0
    # Inferred temporal metadata for annualization
    periods_per_year: int = 252
    n_periods: int = 0
    avg_per_day: float = 0.0


class OutputProcessor:
    """
    Processes backtesting results and generates comprehensive reports and visualizations
    """
    
    def __init__(self, output_format: OutputFormat):
        self.output_format = output_format
        self.ensure_output_directory()
        
    def ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_format.output_directory, exist_ok=True)
        
    def process_backtest_results(self, 
                               strategy_returns: pd.Series,
                               strategy_positions: pd.DataFrame,
                               factor_values: Optional[pd.DataFrame] = None,
                               model_predictions: Optional[pd.Series] = None,
                               benchmark_data: Optional[Dict[str, pd.Series]] = None,
                               test_start_date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        Main method to process all backtest results
        
        Args:
            strategy_returns: Daily strategy returns
            strategy_positions: Portfolio positions over time
            factor_values: Factor values used in strategy
            model_predictions: Model predictions if applicable
            benchmark_data: ETF benchmark returns data
            
        Returns:
            Dict containing all results, metrics, and file paths
        """
        
        results = {}
        
        # Calculate performance metrics
        strategy_metrics = self.calculate_performance_metrics(strategy_returns)
        results['strategy_metrics'] = strategy_metrics
        
        # Robustly clean factor_values: parse date, preserve symbol, coerce to numeric, drop all-NaN cols
        if factor_values is not None:
            try:
                fv = factor_values.copy()
                # Normalize column names
                fv.columns = [c.strip() if isinstance(c, str) else c for c in fv.columns]

                # If there's a date column, ensure it's datetime and set as index
                if 'date' in fv.columns:
                    fv['date'] = pd.to_datetime(fv['date'], errors='coerce')
                    fv = fv.set_index('date')

                # Preserve symbol column if present
                symbol_col = None
                if 'symbol' in fv.columns:
                    symbol_col = fv['symbol'].copy()
                    fv = fv.drop(columns=['symbol'])

                # Replace empty strings with NaN, then coerce all remaining cols to numeric
                fv = fv.replace(['', ' '], np.nan)
                for col in fv.columns:
                    fv[col] = pd.to_numeric(fv[col], errors='coerce')

                # Drop columns that are entirely NaN
                fv = fv.loc[:, fv.notna().any(axis=0)]

                # If nothing numeric remains, drop factor_values to avoid chart errors
                if fv.shape[1] == 0:
                    factor_values = None
                else:
                    # restore symbol column if it existed
                    if symbol_col is not None:
                        fv['symbol'] = symbol_col
                    factor_values = fv
            except Exception:
                factor_values = None
        
        # Load and calculate benchmark metrics
        if benchmark_data is None:
            benchmark_data = self.load_etf_data()
        
        benchmark_metrics = {}
        for etf_symbol, etf_returns in benchmark_data.items():
            # Align dates with strategy returns
            aligned_returns = etf_returns.reindex(strategy_returns.index).fillna(0)
            benchmark_metrics[etf_symbol] = self.calculate_performance_metrics(aligned_returns)
        
        results['benchmark_metrics'] = benchmark_metrics
        
        # Generate comparison analysis
        comparison_results = self.generate_comparison_analysis(
            strategy_returns, benchmark_data, strategy_metrics, benchmark_metrics
        )
        results['comparison_analysis'] = comparison_results
        
        # Generate reports
        if self.output_format.generate_summary_report:
            summary_path = self.generate_summary_report(results)
            results['summary_report_path'] = summary_path
            
        if self.output_format.generate_detailed_report:
            detailed_path = self.generate_detailed_report(results, strategy_positions, 
                                                        factor_values, model_predictions)
            results['detailed_report_path'] = detailed_path
            
        # Generate charts
        chart_paths = self.generate_all_charts(strategy_returns, benchmark_data, 
                            strategy_positions, factor_values, test_start_date)
        results['chart_paths'] = chart_paths
        
        # Save raw data
        if self.output_format.save_raw_data:
            raw_data_paths = self.save_raw_data(strategy_returns, strategy_positions, 
                                              factor_values, model_predictions, benchmark_data)
            results['raw_data_paths'] = raw_data_paths
            
        return results
    
    def calculate_performance_metrics(self, returns: pd.Series) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        # If returns is a DataFrame, this shouldn't happen for returns data
        if isinstance(returns, pd.DataFrame):
            if 'close' in returns.columns:
                # This is raw OHLCV data, calculate returns from close prices
                close_prices = returns['close']
                returns = close_prices.pct_change().dropna()
            else:
                # Take the first column as Series
                returns = returns.iloc[:, 0]
        
        # Infer periods per year from timestamp index when possible (handles hourly/daily)
        periods_per_year = 252
        try:
            if isinstance(returns.index, pd.DatetimeIndex) and len(returns) > 0:
                counts = pd.Series(1, index=returns.index).groupby(returns.index.date).sum()
                avg_per_day = counts.mean()
                inferred = int(round(avg_per_day * 252))
                if inferred > 0:
                    periods_per_year = inferred
        except Exception:
            periods_per_year = 252

        # Basic return metrics
        clean_returns = returns.dropna()
        total_return = (1 + clean_returns).cumprod().iloc[-1] - 1
        # Geometric annualization based on cumulative return and observed number of periods
        n_periods = len(clean_returns)
        if n_periods > 0 and total_return > -1:
            annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        else:
            # Fallback to arithmetic mean annualization
            annual_return = (1 + returns.mean()) ** periods_per_year - 1
        volatility = clean_returns.std() * np.sqrt(periods_per_year)
        
        # Ensure metrics are scalars for comparison
        if hasattr(total_return, 'item'):
            total_return = total_return.item()
        if hasattr(annual_return, 'item'):
            annual_return = annual_return.item()
        if hasattr(volatility, 'item'):
            volatility = volatility.item()
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Ensure max_drawdown is a scalar
        if hasattr(max_drawdown, 'item'):
            max_drawdown = max_drawdown.item()
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Risk metrics
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Ensure risk metrics are scalars
        if hasattr(var_95, 'item'):
            var_95 = var_95.item()
        if hasattr(cvar_95, 'item'):
            cvar_95 = cvar_95.item()
        
        # Trading metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        average_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        average_loss = negative_returns.mean() if len(negative_returns) > 0 else 0

        # Compute win rate over actual trading periods (ignore zero returns which often indicate no position)
        nonzero_returns = returns[returns != 0]
        if len(nonzero_returns) > 0:
            win_rate = (nonzero_returns > 0).mean()
            total_trades = len(nonzero_returns)
        else:
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
            total_trades = len(returns)
        
        # Ensure trading metrics are scalars
        if hasattr(average_win, 'item'):
            average_win = average_win.item()
        if hasattr(average_loss, 'item'):
            average_loss = average_loss.item()
        profit_factor = abs(average_win / average_loss) if average_loss != 0 else 0
        
        # Advanced metrics
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(periods_per_year)
        # Ensure downside_volatility is a scalar
        if hasattr(downside_volatility, 'item'):
            downside_volatility = downside_volatility.item()
        sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0

        # Additional trading metrics
        hit_ratio = win_rate  # Same as win rate for period-based analysis

        # Risk-adjusted metrics
        downside_deviation = downside_volatility
        treynor_ratio = 0  # Will be calculated with beta comparison

        # Tail risk metrics
        skewness = returns.skew() if len(returns) > 3 else 0
        kurtosis = returns.kurtosis() if len(returns) > 3 else 0

        # Ensure tail risk metrics are scalars
        if hasattr(skewness, 'item'):
            skewness = skewness.item()
        if hasattr(kurtosis, 'item'):
            kurtosis = kurtosis.item()

        # Calculate consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            sortino_ratio=sortino_ratio,
            information_ratio=0,  # Will be calculated in comparison
            beta=0,  # Will be calculated in comparison
            alpha=0,  # Will be calculated in comparison

            # New additional metrics
            total_trades=total_trades,
            turnover_rate=0,  # Will be calculated with position data
            hit_ratio=hit_ratio,
            treynor_ratio=treynor_ratio,
            tracking_error=0,  # Will be calculated in comparison
            downside_deviation=downside_deviation,
            factor_turnover=0,  # Will be calculated with position data
            max_position_weight=0,  # Will be calculated with position data
            avg_num_positions=0,  # Will be calculated with position data
            skewness=skewness,
            kurtosis=kurtosis,
            max_consecutive_losses=max_consecutive_losses,
            
            # Comparison metrics (vs benchmark)
            excess_return=0,  # Will be calculated in comparison
            up_capture=0,  # Will be calculated in comparison
            down_capture=0,  # Will be calculated in comparison
            
            # Additional hourly-specific metrics
            turnover_hourly=0.0,
            turnover_annual=0.0,
            cost_ratio=0.0,
            gross_return=total_return,
            net_return=total_return,
            r_squared=0.0,
            hit_ratio_hourly=hit_ratio,
            hit_ratio_daily=hit_ratio,
            concentration_hhi=0.0,
            top3_weight_ratio=0.0,
            top5_weight_ratio=0.0,
            recovery_time_days=0.0,
            periods_per_year=periods_per_year,
            n_periods=n_periods,
            avg_per_day=float(avg_per_day) if 'avg_per_day' in locals() else 0.0
        )
    
    def enhance_metrics_with_positions(self, metrics: PerformanceMetrics,
                                      positions: pd.DataFrame) -> PerformanceMetrics:
        """Enhance performance metrics with position-specific calculations"""
        if positions is None or len(positions) == 0:
            return metrics

        # Calculate turnover rates
        position_changes = positions.diff().abs().sum(axis=1)
        turnover_hourly = position_changes.mean()

        # Annualize turnover using inferred periods_per_year if available
        ppy = getattr(metrics, 'periods_per_year', 252)
        hourly_factor = ppy / 252.0
        turnover_annual = turnover_hourly * hourly_factor * 24  # approximate annualization

        # Calculate max position weight
        max_position = positions.abs().max().max()

        # Calculate average number of positions
        non_zero_positions = (positions != 0).sum(axis=1)
        avg_positions = non_zero_positions.mean()

        # Factor turnover (same as turnover rate for now)
        factor_turnover = turnover_hourly

        # Calculate concentration metrics
        squared_weights = positions ** 2
        hhi = squared_weights.sum(axis=1).mean()

        # Top N position concentration
        abs_positions = positions.abs()
        sorted_positions = abs_positions.apply(lambda x: x.sort_values(ascending=False), axis=1)

        top3_weight_ratio = 0.0
        top5_weight_ratio = 0.0
        if len(positions.columns) >= 3:
            top3_weights = sorted_positions.iloc[:, :3].sum(axis=1)
            top3_weight_ratio = top3_weights.mean()
        if len(positions.columns) >= 5:
            top5_weights = sorted_positions.iloc[:, :5].sum(axis=1)
            top5_weight_ratio = top5_weights.mean()

        # Create updated metrics with position data (preserve existing metrics values)
        updated_metrics = PerformanceMetrics(
            total_return=metrics.total_return,
            annual_return=metrics.annual_return,
            volatility=metrics.volatility,
            sharpe_ratio=metrics.sharpe_ratio,
            calmar_ratio=metrics.calmar_ratio,
            max_drawdown=metrics.max_drawdown,
            var_95=metrics.var_95,
            cvar_95=metrics.cvar_95,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            average_win=metrics.average_win,
            average_loss=metrics.average_loss,
            sortino_ratio=metrics.sortino_ratio,
            information_ratio=metrics.information_ratio,
            beta=metrics.beta,
            alpha=metrics.alpha,
            total_trades=metrics.total_trades,
            turnover_rate=turnover_hourly,
            hit_ratio=metrics.hit_ratio,
            treynor_ratio=metrics.treynor_ratio,
            tracking_error=metrics.tracking_error,
            downside_deviation=metrics.downside_deviation,
            factor_turnover=factor_turnover,
            max_position_weight=max_position,
            avg_num_positions=avg_positions,
            skewness=metrics.skewness,
            kurtosis=metrics.kurtosis,
            max_consecutive_losses=metrics.max_consecutive_losses,
            excess_return=metrics.excess_return,
            up_capture=metrics.up_capture,
            down_capture=metrics.down_capture,

            # Updated hourly-specific metrics
            turnover_hourly=turnover_hourly,
            turnover_annual=turnover_annual,
            cost_ratio=metrics.cost_ratio,
            gross_return=metrics.gross_return,
            net_return=metrics.net_return,
            r_squared=metrics.r_squared,
            hit_ratio_hourly=metrics.hit_ratio_hourly,
            hit_ratio_daily=metrics.hit_ratio_daily,
            concentration_hhi=hhi,
            top3_weight_ratio=top3_weight_ratio,
            top5_weight_ratio=top5_weight_ratio,
            recovery_time_days=metrics.recovery_time_days,
            periods_per_year=getattr(metrics, 'periods_per_year', 252),
            n_periods=getattr(metrics, 'n_periods', 0)
        )

        return updated_metrics
    
    def load_etf_data(self) -> Dict[str, pd.Series]:
        """Load ETF benchmark data from qlib_data directory"""
        
        # Path to the ETF data directory
        data_dir = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/etf_backup"
        
        etf_data = {}
        etf_symbols = ['SPY', 'QQQ', 'IWM', 'VTI']
        
        for symbol in etf_symbols:
            try:
                # Load daily data
                file_path = os.path.join(data_dir, f"{symbol}_daily.csv")
                
                if os.path.exists(file_path):
                    # Read CSV file
                    df = pd.read_csv(file_path)
                    
                    # Convert Date column to datetime and set as index
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    
                    # Calculate daily returns from Close prices
                    close_prices = df['Close']
                    daily_returns = close_prices.pct_change().dropna()
                    
                    # Store returns data
                    etf_data[symbol] = daily_returns
                    
                    print(f"Loaded {symbol} data: {len(daily_returns)} days, "
                          f"from {daily_returns.index.min().date()} to {daily_returns.index.max().date()}")
                else:
                    print(f"Warning: {file_path} not found, using synthetic data for {symbol}")
                    # Fallback to synthetic data if file not found
                    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
                    if symbol == 'SPY':
                        returns = np.random.normal(0.0003, 0.012, len(dates))
                    elif symbol == 'QQQ':
                        returns = np.random.normal(0.0004, 0.018, len(dates))
                    elif symbol == 'IWM':
                        returns = np.random.normal(0.0002, 0.020, len(dates))
                    else:  # VTI
                        returns = np.random.normal(0.0003, 0.014, len(dates))
                    etf_data[symbol] = pd.Series(returns, index=dates)
                    
            except Exception as e:
                print(f"Error loading {symbol} data: {e}")
                # Fallback to synthetic data on error
                dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
                if symbol == 'SPY':
                    returns = np.random.normal(0.0003, 0.012, len(dates))
                elif symbol == 'QQQ':
                    returns = np.random.normal(0.0004, 0.018, len(dates))
                elif symbol == 'IWM':
                    returns = np.random.normal(0.0002, 0.020, len(dates))
                else:  # VTI
                    returns = np.random.normal(0.0003, 0.014, len(dates))
                etf_data[symbol] = pd.Series(returns, index=dates)
        
        return etf_data
    
    def load_hourly_etf_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.Series]:
        """Load ETF hourly benchmark data for strategy comparison"""
        
        # Path to the ETF data directory
        data_dir = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/etf_backup"
        
        etf_data = {}
        etf_symbols = ['SPY', 'QQQ', 'IWM', 'VTI']
        
        for symbol in etf_symbols:
            try:
                # Load hourly data
                file_path = os.path.join(data_dir, f"{symbol}_hourly.csv")
                
                if os.path.exists(file_path):
                    # Read CSV file
                    df = pd.read_csv(file_path)
                    
                    # Convert Datetime column and handle timezone
                    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True).dt.tz_convert(None)
                    df.set_index('Datetime', inplace=True)
                    
                    # Filter to realistic date range - no future data beyond current date
                    import datetime
                    current_date = datetime.datetime.now()
                    max_reasonable_date = pd.Timestamp('2024-12-31')  # Set reasonable upper bound
                    
                    # Apply date filtering
                    if start_date and end_date:
                        filter_end = min(pd.to_datetime(end_date), max_reasonable_date)
                        mask = (df.index >= start_date) & (df.index <= filter_end)
                        df = df[mask]
                    else:
                        # Default filtering to remove unrealistic future dates
                        mask = df.index <= max_reasonable_date
                        df = df[mask]
                    
                    # Calculate hourly returns from Close prices
                    close_prices = df['Close']
                    hourly_returns = close_prices.pct_change().dropna()
                    
                    # Store returns data
                    etf_data[symbol] = hourly_returns
                    
                    print(f" Loaded hourly ETF {symbol}: {len(hourly_returns)} periods, "
                          f"from {hourly_returns.index.min()} to {hourly_returns.index.max()}")
                else:
                    print(f"Hourly data file not found: {file_path}")
                    
            except Exception as e:
                print(f"Could not load hourly ETF data for {symbol}: {e}")
                
        return etf_data
    
    def generate_comparison_analysis(self, strategy_returns: pd.Series, 
                                   benchmark_data: Dict[str, pd.Series],
                                   strategy_metrics: PerformanceMetrics,
                                   benchmark_metrics: Dict[str, PerformanceMetrics]) -> Dict[str, Any]:
        """Generate detailed comparison analysis"""
        
        comparison = {}
        
        # Create comparison table
        metrics_comparison = pd.DataFrame()
        
        # Add strategy metrics
        strategy_dict = {
            'Total Return': f"{strategy_metrics.total_return:.2%}",
            'Annual Return': f"{strategy_metrics.annual_return:.2%}",
            'Volatility': f"{strategy_metrics.volatility:.2%}",
            'Sharpe Ratio': f"{strategy_metrics.sharpe_ratio:.2f}",
            'Max Drawdown': f"{strategy_metrics.max_drawdown:.2%}",
            'Calmar Ratio': f"{strategy_metrics.calmar_ratio:.2f}",
            'Win Rate': f"{strategy_metrics.win_rate:.2%}",
            'Sortino Ratio': f"{strategy_metrics.sortino_ratio:.2f}"
        }
        metrics_comparison['Strategy'] = strategy_dict
        
        # Add benchmark metrics
        for etf_symbol, metrics in benchmark_metrics.items():
            etf_dict = {
                'Total Return': f"{metrics.total_return:.2%}",
                'Annual Return': f"{metrics.annual_return:.2%}",
                'Volatility': f"{metrics.volatility:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Calmar Ratio': f"{metrics.calmar_ratio:.2f}",
                'Win Rate': f"{metrics.win_rate:.2%}",
                'Sortino Ratio': f"{metrics.sortino_ratio:.2f}"
            }
            metrics_comparison[etf_symbol] = etf_dict
        
        comparison['metrics_table'] = metrics_comparison
        
        # Calculate relative performance
        relative_performance = {}
        for etf_symbol in benchmark_data.keys():
            relative_performance[etf_symbol] = {
                'excess_return': strategy_metrics.annual_return - benchmark_metrics[etf_symbol].annual_return,
                'sharpe_difference': strategy_metrics.sharpe_ratio - benchmark_metrics[etf_symbol].sharpe_ratio,
                'volatility_difference': strategy_metrics.volatility - benchmark_metrics[etf_symbol].volatility
            }
        
        comparison['relative_performance'] = relative_performance
        
        return comparison
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML summary report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtesting Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ margin: 20px 0; }}
                .table {{ border-collapse: collapse; width: 100%; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Backtesting Summary Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metrics">
                <h2>Performance Comparison</h2>
                {self._format_comparison_table(results['comparison_analysis']['metrics_table'])}
            </div>
            
            <div class="metrics">
                <h2>Key Insights</h2>
                {self._generate_key_insights(results)}
            </div>
        </body>
        </html>
        """
        
        file_path = os.path.join(self.output_format.output_directory, "summary_report.html")
        with open(file_path, 'w') as f:
            f.write(html_content)
            
        return file_path
    
    def generate_detailed_report(self, results: Dict[str, Any], 
                               strategy_positions: pd.DataFrame,
                               factor_values: Optional[pd.DataFrame],
                               model_predictions: Optional[pd.Series]) -> str:
        """Generate detailed Excel report"""
        
        file_path = os.path.join(self.output_format.output_directory, "detailed_report.xlsx")
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Summary metrics
            results['comparison_analysis']['metrics_table'].to_excel(
                writer, sheet_name='Summary', index=True
            )
            
            # Strategy positions
            if strategy_positions is not None:
                strategy_positions.to_excel(writer, sheet_name='Positions')
            
            # Factor values
            if factor_values is not None:
                factor_values.to_excel(writer, sheet_name='Factors')
            
            # Model predictions
            if model_predictions is not None:
                model_predictions.to_frame('Predictions').to_excel(writer, sheet_name='Predictions')
        
        return file_path
    
    def generate_all_charts(self, strategy_returns: pd.Series,
                          benchmark_data: Dict[str, pd.Series],
                          strategy_positions: pd.DataFrame,
                          factor_values: Optional[pd.DataFrame],
                          test_start_date: Optional[pd.Timestamp] = None) -> Dict[str, str]:
        """Generate all visualization charts"""
        
        chart_paths = {}
        # Quick guard: if no chart flags are enabled, skip chart generation
        chart_flags = [
            self.output_format.generate_performance_chart,
            self.output_format.generate_drawdown_chart,
            self.output_format.generate_rolling_metrics_chart,
            getattr(self.output_format, 'generate_correlation_matrix', False),
            getattr(self.output_format, 'generate_excess_return_chart', False),
            getattr(self.output_format, 'generate_signal_analysis_chart', False),
            getattr(self.output_format, 'generate_monthly_heatmap', False),
            getattr(self.output_format, 'generate_return_distribution', False)
        ]
        if not any(chart_flags):
            return chart_paths
        
        if self.output_format.generate_performance_chart:
            chart_paths['performance'] = self._create_performance_chart(strategy_returns, benchmark_data, test_start_date)
            
        if self.output_format.generate_drawdown_chart:
            chart_paths['drawdown'] = self._create_drawdown_chart(strategy_returns, benchmark_data, test_start_date)
            
        if self.output_format.generate_rolling_metrics_chart:
            chart_paths['rolling_metrics'] = self._create_rolling_metrics_chart(strategy_returns, benchmark_data, test_start_date)
            
        if self.output_format.generate_correlation_matrix and factor_values is not None:
            chart_paths['correlation'] = self._create_correlation_matrix(factor_values)
            
        # NEW: Excess return comparison chart
        if self.output_format.generate_excess_return_chart:
            chart_paths['excess_return'] = self._create_excess_return_chart(strategy_returns, benchmark_data, test_start_date)
            
        # NEW: Strategy signal analysis chart
        if self.output_format.generate_signal_analysis_chart and strategy_positions is not None:
            chart_paths['signal_analysis'] = self._create_signal_analysis_chart(strategy_positions, strategy_returns)
            
        # Additional advanced visualizations
        chart_paths.update(self._create_advanced_visualizations(strategy_returns, benchmark_data, 
                                                               strategy_positions, factor_values, test_start_date))
        
        # New comprehensive charts for hourly analysis audit
        chart_paths['rolling_drawdown'] = self._create_rolling_drawdown_chart(strategy_returns, test_start_date)
        chart_paths['return_distribution'] = self._create_return_distribution_chart(strategy_returns, benchmark_data, test_start_date)
        chart_paths['up_down_capture'] = self._create_up_down_capture_chart(strategy_returns, benchmark_data, test_start_date)
            
        return chart_paths
    
    def _create_performance_chart(self, strategy_returns: pd.Series, 
                                benchmark_data: Dict[str, pd.Series],
                                test_start_date: Optional[pd.Timestamp] = None) -> str:
        """Create cumulative performance comparison chart from test period start"""
        
        fig = go.Figure()
        
        # Filter returns to show only test period if test_start_date is provided
        if test_start_date is not None:
            # Filter strategy returns to test period only
            test_strategy_returns = strategy_returns[strategy_returns.index >= test_start_date]
            chart_title = f'Cumulative Performance Comparison (Test Period from {test_start_date.strftime("%Y-%m-%d")})'
        else:
            # Use all data if no test start date provided
            test_strategy_returns = strategy_returns
            chart_title = 'Cumulative Performance Comparison (Full Period)'
        
        # Calculate cumulative returns starting from test period (ensuring start at 1.0)
        # Use fillna(0) to handle any missing values and ensure clean calculation
        clean_strategy_returns = test_strategy_returns.fillna(0)
        
        # Calculate cumulative returns and normalize to start from 1.0
        strategy_cumulative = (1 + clean_strategy_returns).cumprod()
        # Normalize to start from 1.0 by dividing by the first value
        if len(strategy_cumulative) > 0 and strategy_cumulative.iloc[0] != 0:
            strategy_cumulative = strategy_cumulative / strategy_cumulative.iloc[0]
        
        # Strategy performance
        fig.add_trace(go.Scatter(
            x=strategy_cumulative.index,
            y=strategy_cumulative.values,
            mode='lines',
            name='Ours',
            line=dict(color='blue', width=3),
            hovertemplate='<b>Ours</b><br>' +
                         'Date: %{x}<br>' +
                         'Cumulative Return: %{y:.3f}<extra></extra>'
        ))
        
        # Benchmark performances - also filter to test period
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896']
        for i, (etf_symbol, etf_returns) in enumerate(benchmark_data.items()):
            # Align with strategy returns index and ensure same date range
            aligned_returns = etf_returns.reindex(clean_strategy_returns.index).fillna(0)
            
            # Calculate cumulative returns for benchmark and normalize to start from 1.0
            etf_cumulative = (1 + aligned_returns).cumprod()
            # Normalize to start from 1.0 by dividing by the first value
            if len(etf_cumulative) > 0 and etf_cumulative.iloc[0] != 0:
                etf_cumulative = etf_cumulative / etf_cumulative.iloc[0]
            
            fig.add_trace(go.Scatter(
                x=etf_cumulative.index,
                y=etf_cumulative.values,
                mode='lines',
                name=etf_symbol,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{etf_symbol}</b><br>' +
                             'Date: %{x}<br>' +
                             'Cumulative Return: %{y:.3f}<extra></extra>'
            ))
        
        # Add horizontal line at 1.0 for reference
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                     annotation_text="Starting Value = 1.0")
        
        fig.update_layout(
            title=chart_title,
            xaxis_title='Date',
            yaxis_title='Cumulative Return (Starting Value = 1.0)',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        )
        
        file_path = os.path.join(self.output_format.output_directory, "performance_chart.html")
        fig.write_html(file_path)
        return file_path
    
    def _create_drawdown_chart(self, strategy_returns: pd.Series,
                             benchmark_data: Dict[str, pd.Series],
                             test_start_date: Optional[pd.Timestamp] = None) -> str:
        """Create drawdown comparison chart"""
        
        fig = go.Figure()
        
        # Strategy drawdown
        strategy_cumulative = (1 + strategy_returns).cumprod()
        strategy_rolling_max = strategy_cumulative.expanding().max()
        strategy_drawdown = (strategy_cumulative - strategy_rolling_max) / strategy_rolling_max
        
        fig.add_trace(go.Scatter(
            x=strategy_drawdown.index,
            y=strategy_drawdown.values,
            mode='lines',
            name='Strategy',
            fill='tonexty',
            line=dict(color='blue')
        ))
        
        # Add benchmark drawdowns
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896']
        for i, (etf_symbol, etf_returns) in enumerate(benchmark_data.items()):
            aligned_returns = etf_returns.reindex(strategy_returns.index).fillna(0)
            etf_cumulative = (1 + aligned_returns).cumprod()
            etf_rolling_max = etf_cumulative.expanding().max()
            etf_drawdown = (etf_cumulative - etf_rolling_max) / etf_rolling_max
            
            fig.add_trace(go.Scatter(
                x=etf_drawdown.index,
                y=etf_drawdown.values,
                mode='lines',
                name=f'{etf_symbol}',
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title='Drawdown Comparison',
            xaxis_title='Date',
            yaxis_title='Drawdown',
            yaxis_tickformat='.1%'
        )
        
        file_path = os.path.join(self.output_format.output_directory, "drawdown_chart.html")
        fig.write_html(file_path)
        return file_path
    
    def _create_rolling_metrics_chart(self, strategy_returns: pd.Series,
                                    benchmark_data: Dict[str, pd.Series],
                                    test_start_date: Optional[pd.Timestamp] = None) -> str:
        """Create rolling Sharpe ratio chart"""
        # Infer periods-per-year and choose rolling window dynamically so charts work for intraday/hourly data
        ppy = 252
        try:
            if isinstance(strategy_returns.index, pd.DatetimeIndex) and len(strategy_returns) > 0:
                counts = pd.Series(1, index=strategy_returns.index).groupby(strategy_returns.index.date).sum()
                avg_per_day = counts.mean()
                inferred = int(round(avg_per_day * 252))
                if inferred > 0:
                    ppy = inferred
        except Exception:
            ppy = 252

        # Use a rolling window equal to one "year" of periods (inferred) but cap to available data length
        window = int(min(max(1, ppy), max(1, len(strategy_returns))))

        fig = go.Figure()

        # Strategy rolling Sharpe (use inferred annualization)
        strategy_rolling_sharpe = (strategy_returns.rolling(window, min_periods=1).mean() /
                                   strategy_returns.rolling(window, min_periods=1).std()) * np.sqrt(ppy)
        
        fig.add_trace(go.Scatter(
            x=strategy_rolling_sharpe.index,
            y=strategy_rolling_sharpe.values,
            mode='lines',
            name='Strategy',
            line=dict(color='blue')
        ))
        
        # Benchmark rolling Sharpe
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896']
        for i, (etf_symbol, etf_returns) in enumerate(benchmark_data.items()):
            aligned_returns = etf_returns.reindex(strategy_returns.index).fillna(0)
            etf_rolling_sharpe = (aligned_returns.rolling(window, min_periods=1).mean() /
                                  aligned_returns.rolling(window, min_periods=1).std()) * np.sqrt(ppy)
            
            fig.add_trace(go.Scatter(
                x=etf_rolling_sharpe.index,
                y=etf_rolling_sharpe.values,
                mode='lines',
                name=etf_symbol,
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title='Rolling Sharpe Ratio (1 Year Window)',
            xaxis_title='Date',
            yaxis_title='Sharpe Ratio'
        )
        
        file_path = os.path.join(self.output_format.output_directory, "rolling_sharpe_chart.html")
        fig.write_html(file_path)
        return file_path
    
    def _create_correlation_matrix(self, factor_values: pd.DataFrame) -> str:
        """Create factor correlation matrix heatmap"""
        # Keep only numeric columns for correlation
        if factor_values is None:
            raise ValueError('factor_values is None')
        numeric_cols = factor_values.select_dtypes(include=[np.number]).columns.tolist()
        dropped = [c for c in factor_values.columns if c not in numeric_cols]
        if dropped:
            print('Dropping non-numeric factor columns for correlation:', dropped)
        fv_numeric = factor_values[numeric_cols]
        correlation_matrix = fv_numeric.corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Factor Correlation Matrix",
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )
        
        file_path = os.path.join(self.output_format.output_directory, "correlation_matrix.html")
        fig.write_html(file_path)
        return file_path
    
    def _create_excess_return_chart(self, strategy_returns: pd.Series, 
                                  benchmark_data: Dict[str, pd.Series],
                                  test_start_date: Optional[pd.Timestamp] = None) -> str:
        """Create excess return comparison chart from test period start"""
        
        # Calculate adaptive rolling window first to use in titles
        data_length = len(strategy_returns)
        rolling_window = min(30, max(5, data_length // 3))  # Adaptive window: 5-30 days
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Strategy vs Benchmarks: Cumulative Returns Comparison', 
                f'Rolling {rolling_window}-Day Returns Comparison',
                'Strategy Excess Returns vs Benchmarks'
            ),
            row_heights=[0.33, 0.33, 0.34],
            vertical_spacing=0.10,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Extended color palette for consistent coloring across charts
        colors = [
            '#ff7f0e',  # Orange  
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf',  # Cyan
            '#ff9896'   # Light Red
        ]
        
        # Filter returns to test period if test_start_date is provided
        if test_start_date is not None:
            test_strategy_returns = strategy_returns[strategy_returns.index >= test_start_date]
            chart_subtitle = f'Test Period Analysis from {test_start_date.strftime("%Y-%m-%d")}'
        else:
            test_strategy_returns = strategy_returns
            chart_subtitle = 'Full Period Analysis'
        
        # Calculate strategy cumulative performance for comparison (starting from test period)
        strategy_cumulative = (1 + test_strategy_returns).cumprod()
        
        # First subplot: Add strategy performance first
        fig.add_trace(go.Scatter(
            x=strategy_cumulative.index,
            y=(strategy_cumulative.values - 1) * 100,  # Convert to percentage
            mode='lines',
            name='Strategy',
            line=dict(color='blue', width=3),
            hovertemplate='<b>Strategy</b><br>' +
                         'Date: %{x}<br>' +
                         'Cumulative Return: %{y:.2f}%<extra></extra>'
        ), row=1, col=1)
        
        # Second subplot: Add strategy rolling returns (adaptive window)
        # Use the pre-calculated rolling window from above
        
        strategy_rolling = test_strategy_returns.rolling(rolling_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=strategy_rolling.index,
            y=strategy_rolling.values * 100,  # Convert to percentage
            mode='lines',
            name=f'Strategy {rolling_window}D',
            line=dict(color='blue', width=3),
            showlegend=True,
            hovertemplate=f'<b>Strategy {rolling_window}D</b><br>' +
                         'Date: %{x}<br>' +
                         'Rolling Return: %{y:.2f}%<extra></extra>'
        ), row=2, col=1)
        
        # Calculate and plot benchmark comparisons
        excess_stats = {}
        for i, (etf_symbol, etf_returns) in enumerate(benchmark_data.items()):
            # Align data to test period
            aligned_benchmark = etf_returns.reindex(test_strategy_returns.index).fillna(0)
            benchmark_cumulative = (1 + aligned_benchmark).cumprod()
            
            # Calculate excess returns for stats (using test period data)
            excess_returns = test_strategy_returns - aligned_benchmark
            cumulative_excess = excess_returns.cumsum()
            
            # Store stats for summary
            excess_stats[etf_symbol] = {
                'total_excess': cumulative_excess.iloc[-1] if len(cumulative_excess) > 0 else 0,
                'annualized_excess': excess_returns.mean() * 252,
                'win_rate': (excess_returns > 0).mean()
            }
            
            # First subplot: Benchmark cumulative returns
            fig.add_trace(go.Scatter(
                x=benchmark_cumulative.index,
                y=(benchmark_cumulative.values - 1) * 100,  # Convert to percentage
                mode='lines',
                name=etf_symbol,
                line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                hovertemplate=f'<b>{etf_symbol}</b><br>' +
                             'Date: %{x}<br>' +
                             'Cumulative Return: %{y:.2f}%<extra></extra>'
            ), row=1, col=1)
            
            # Second subplot: Benchmark rolling returns (adaptive window)
            benchmark_rolling = aligned_benchmark.rolling(rolling_window, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=benchmark_rolling.index,
                y=benchmark_rolling.values * 100,  # Convert to percentage
                mode='lines',
                name=f'{etf_symbol} {rolling_window}D',
                line=dict(color=colors[i % len(colors)], width=2, dash='dot'),
                showlegend=True,
                hovertemplate=f'<b>{etf_symbol} {rolling_window}D</b><br>' +
                             'Date: %{x}<br>' +
                             'Rolling Return: %{y:.2f}%<extra></extra>'
            ), row=2, col=1)
        
        # Third subplot: Excess returns analysis
        for i, (etf_symbol, etf_returns) in enumerate(benchmark_data.items()):
            # Align data to test period
            aligned_benchmark = etf_returns.reindex(test_strategy_returns.index).fillna(0)
            
            # Calculate excess returns (using test period data)
            excess_returns = test_strategy_returns - aligned_benchmark
            cumulative_excess = excess_returns.cumsum()
            
            fig.add_trace(go.Scatter(
                x=cumulative_excess.index,
                y=cumulative_excess.values * 100,  # Convert to percentage
                mode='lines',
                name=f'Excess vs {etf_symbol}',
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>Excess vs {etf_symbol}</b><br>' +
                             'Date: %{x}<br>' +
                             'Cumulative Excess: %{y:.2f}%<extra></extra>'
            ), row=3, col=1)
        
        # Add zero lines for reference
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        # Create summary text for the best performing comparison
        best_benchmark = max(excess_stats.items(), key=lambda x: x[1]['total_excess'])
        summary_text = f"Best vs {best_benchmark[0]}: {best_benchmark[1]['total_excess']:.1%} total excess, " + \
                      f"{best_benchmark[1]['annualized_excess']:.1%} annualized, " + \
                      f"{best_benchmark[1]['win_rate']:.1%} win rate"
        
        fig.update_layout(
            title={
                'text': f'Strategy vs Benchmarks Performance Analysis - {chart_subtitle}<br><sub>{summary_text}</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'y': 0.98
            },
            height=1200,
            hovermode='x unified',
            margin=dict(t=120, b=60, l=60, r=60),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.05,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1)
        fig.update_yaxes(title_text=f"{rolling_window}-Day Rolling Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Excess Return (%)", row=3, col=1)
        
        file_path = os.path.join(self.output_format.output_directory, "excess_return_chart.html")
        fig.write_html(file_path)
        return file_path
    
    def _create_signal_analysis_chart(self, strategy_positions: pd.DataFrame, 
                                    strategy_returns: pd.Series) -> str:
        """Create strategy signal analysis chart"""
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                'Portfolio Weight Distribution Over Time', 
                'Daily Turnover Analysis', 
                'Strategy Returns Distribution',
                'Cumulative Strategy Performance'
            ),
            row_heights=[0.25, 0.25, 0.25, 0.25],
            vertical_spacing=0.15,  # Increased spacing for bottom legend
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Update subplot title font sizes
        fig.update_annotations(font_size=16)
        
        # Prepare data
        if isinstance(strategy_positions, pd.DataFrame):
            # First subplot: Weight distribution time series
            for i, symbol in enumerate(strategy_positions.columns):
                if symbol == 'date':
                    continue
                    
                weights = strategy_positions[symbol]
                # Extended color palette for up to 10 stocks with distinct colors
                colors_cycle = [
                    '#1f77b4',  # Blue
                    '#ff7f0e',  # Orange  
                    '#2ca02c',  # Green
                    '#d62728',  # Red
                    '#9467bd',  # Purple
                    '#8c564b',  # Brown
                    '#e377c2',  # Pink
                    '#7f7f7f',  # Gray
                    '#bcbd22',  # Olive
                    '#17becf'   # Cyan
                ]
                
                fig.add_trace(go.Scatter(
                    x=strategy_positions.index,
                    y=weights,
                    mode='lines+markers',
                    name=f'{symbol} Position',
                    line=dict(color=colors_cycle[i % len(colors_cycle)], width=2),
                    marker=dict(size=4),
                    showlegend=True
                ), row=1, col=1)
            
            # Add zero line for reference
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
            
            # Second subplot: Daily turnover analysis
            weight_changes = strategy_positions.diff().abs().sum(axis=1)
            # Remove first NaN value
            weight_changes = weight_changes.dropna()
            
            if len(weight_changes) > 0:
                fig.add_trace(go.Scatter(
                    x=weight_changes.index,
                    y=weight_changes,
                    mode='lines+markers',
                    name='Daily Turnover',
                    line=dict(color='lightcoral', width=2),
                    marker=dict(size=4),
                    showlegend=True
                ), row=2, col=1)
                
                # Add mean turnover line
                mean_turnover = weight_changes.mean()
                fig.add_hline(y=mean_turnover, line_dash="dash", line_color="red", 
                             annotation_text=f"Mean: {mean_turnover:.2%}", row=2, col=1)
            
            # Third subplot: Strategy returns distribution
            if len(strategy_returns) > 1:
                # Create histogram of strategy returns
                fig.add_trace(go.Histogram(
                    x=strategy_returns,
                    nbinsx=50,
                    name='Return Distribution',
                    marker_color='skyblue',
                    opacity=0.7,
                    showlegend=True
                ), row=3, col=1)
                
                # Add vertical lines for mean and median
                mean_return = strategy_returns.mean()
                median_return = strategy_returns.median()
                
                fig.add_vline(x=mean_return, line_dash="dash", line_color="red",
                             annotation_text=f"Mean: {mean_return:.4f}", row=3, col=1)
                fig.add_vline(x=median_return, line_dash="dot", line_color="blue",
                             annotation_text=f"Median: {median_return:.4f}", row=3, col=1)
            
            # Fourth subplot: Cumulative performance comparison
            strategy_cumulative = (1 + strategy_returns).cumprod()
            
            fig.add_trace(go.Scatter(
                x=strategy_cumulative.index,
                y=strategy_cumulative.values,
                mode='lines',
                name='Strategy Performance',
                line=dict(color='blue', width=3),
                showlegend=True
            ), row=4, col=1)
            
            # Add buy-and-hold comparison if we have position data
            if len(strategy_positions.columns) > 0:
                # Create simple buy-and-hold benchmark using first stock's return
                symbol = [col for col in strategy_positions.columns if col != 'date'][0]
                
                # Calculate buy-and-hold returns (assuming constant position)
                buy_hold_returns = strategy_returns * 0 + strategy_returns.mean()  # Simplified
                buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
                
                fig.add_trace(go.Scatter(
                    x=buy_hold_cumulative.index,
                    y=buy_hold_cumulative.values,
                    mode='lines',
                    name=f'{symbol} Buy-and-Hold',
                    line=dict(color='gray', width=2, dash='dot'),
                    showlegend=True
                ), row=4, col=1)
        
        # Update layout with improved legend and larger fonts
        fig.update_layout(
            title={
                'text': 'Strategy Signal Analysis Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}  # Larger title font
            },
            height=1300,  # Increased height to accommodate bottom legend
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.05,  # Place legend at bottom of chart
                xanchor="center",
                x=0.5,
                font={'size': 14},  # Larger legend font
                bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
                bordercolor="lightgray",
                borderwidth=1
            ),
            font={'size': 12}  # Increase overall font size
        )
        
        # Update axes labels with larger fonts
        fig.update_xaxes(
            title_text="Date", 
            title_font={'size': 14},
            tickfont={'size': 12},
            row=1, col=1
        )
        fig.update_xaxes(
            title_text="Date", 
            title_font={'size': 14},
            tickfont={'size': 12},
            row=2, col=1
        )
        fig.update_xaxes(
            title_text="Signal Strength (Absolute Weight)", 
            title_font={'size': 14},
            tickfont={'size': 12},
            row=3, col=1
        )
        fig.update_xaxes(
            title_text="Date", 
            title_font={'size': 14},
            tickfont={'size': 12},
            row=4, col=1
        )
        
        fig.update_yaxes(
            title_text="Position Weight", 
            title_font={'size': 14},
            tickfont={'size': 12},
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Change Magnitude", 
            title_font={'size': 14},
            tickfont={'size': 12},
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Daily Return", 
            title_font={'size': 14},
            tickfont={'size': 12},
            row=3, col=1
        )
        fig.update_yaxes(
            title_text="Cumulative Return", 
            title_font={'size': 14},
            tickfont={'size': 12},
            row=4, col=1
        )
        
        file_path = os.path.join(self.output_format.output_directory, "signal_analysis_chart.html")
        fig.write_html(file_path)
        return file_path
    
    def save_raw_data(self, strategy_returns: pd.Series,
                     strategy_positions: pd.DataFrame,
                     factor_values: Optional[pd.DataFrame],
                     model_predictions: Optional[pd.Series],
                     benchmark_data: Dict[str, pd.Series]) -> Dict[str, str]:
        """Save raw data to files"""
        
        paths = {}
        
        # Strategy returns
        returns_path = os.path.join(self.output_format.output_directory, "strategy_returns.csv")
        strategy_returns.to_csv(returns_path)
        paths['strategy_returns'] = returns_path
        
        # Strategy positions
        if strategy_positions is not None:
            positions_path = os.path.join(self.output_format.output_directory, "strategy_positions.csv")
            strategy_positions.to_csv(positions_path)
            paths['strategy_positions'] = positions_path
        
        # Factor values
        if factor_values is not None:
            factors_path = os.path.join(self.output_format.output_directory, "factor_values.csv")
            factor_values.to_csv(factors_path)
            paths['factor_values'] = factors_path
        
        # Model predictions
        if model_predictions is not None:
            predictions_path = os.path.join(self.output_format.output_directory, "model_predictions.csv")
            model_predictions.to_csv(predictions_path)
            paths['model_predictions'] = predictions_path
        
        # Benchmark data
        benchmark_path = os.path.join(self.output_format.output_directory, "benchmark_returns.csv")
        benchmark_df = pd.DataFrame(benchmark_data)
        benchmark_df.to_csv(benchmark_path)
        paths['benchmark_returns'] = benchmark_path
        
        return paths
    
    def _format_comparison_table(self, metrics_table: pd.DataFrame) -> str:
        """Format metrics table as HTML"""
        return metrics_table.to_html(classes='table', escape=False)
    
    def _generate_key_insights(self, results: Dict[str, Any]) -> str:
        """Generate key insights from results"""
        
        strategy_metrics = results['strategy_metrics']
        benchmark_metrics = results['benchmark_metrics']
        
        insights = []
        
        # Compare with SPY (most common benchmark)
        if 'SPY' in benchmark_metrics:
            spy_metrics = benchmark_metrics['SPY']
            
            if strategy_metrics.sharpe_ratio > spy_metrics.sharpe_ratio:
                insights.append(f" Strategy outperformed SPY on risk-adjusted basis (Sharpe: {strategy_metrics.sharpe_ratio:.2f} vs {spy_metrics.sharpe_ratio:.2f})")
            else:
                insights.append(f" Strategy underperformed SPY on risk-adjusted basis (Sharpe: {strategy_metrics.sharpe_ratio:.2f} vs {spy_metrics.sharpe_ratio:.2f})")
                
            if strategy_metrics.annual_return > spy_metrics.annual_return:
                insights.append(f" Strategy delivered higher annual returns ({strategy_metrics.annual_return:.2%} vs {spy_metrics.annual_return:.2%})")
            else:
                insights.append(f" Strategy delivered lower annual returns ({strategy_metrics.annual_return:.2%} vs {spy_metrics.annual_return:.2%})")
        
        # Risk assessment
        if strategy_metrics.max_drawdown < -0.2:
            insights.append(f"High maximum drawdown: {strategy_metrics.max_drawdown:.2%}")
        else:
            insights.append(f" Reasonable maximum drawdown: {strategy_metrics.max_drawdown:.2%}")
            
        # Win rate assessment
        if strategy_metrics.win_rate > 0.55:
            insights.append(f" Good win rate: {strategy_metrics.win_rate:.2%}")
        elif strategy_metrics.win_rate < 0.45:
            insights.append(f"Low win rate: {strategy_metrics.win_rate:.2%}")
        else:
            insights.append(f"➖ Average win rate: {strategy_metrics.win_rate:.2%}")
        
        # Include inferred temporal metadata for auditability if present
        if hasattr(strategy_metrics, 'periods_per_year') and hasattr(strategy_metrics, 'n_periods'):
            ppy = getattr(strategy_metrics, 'periods_per_year')
            npds = getattr(strategy_metrics, 'n_periods')
            insights.append(f"Temporal inference: periods_per_year={ppy}, n_periods={npds}")
        
        return "<ul>" + "".join([f"<li>{insight}</li>" for insight in insights]) + "</ul>"
    
    def _create_advanced_visualizations(self, strategy_returns: pd.Series,
                                      benchmark_data: Dict[str, pd.Series],
                                      strategy_positions: pd.DataFrame,
                                      factor_values: Optional[pd.DataFrame],
                                      test_start_date: Optional[pd.Timestamp] = None) -> Dict[str, str]:
        """Create additional advanced visualizations"""
        
        chart_paths = {}
        
        # 1. Monthly Returns Heatmap
        chart_paths['monthly_heatmap'] = self._create_monthly_returns_heatmap(strategy_returns, benchmark_data, test_start_date)
        
        # 2. Risk-Return Scatter Plot
        chart_paths['risk_return_scatter'] = self._create_risk_return_scatter(strategy_returns, benchmark_data, test_start_date)
        
        # 3. Rolling Beta Chart
        chart_paths['rolling_beta'] = self._create_rolling_beta_chart(strategy_returns, benchmark_data, test_start_date)
        
        # 4. Underwater Plot (Drawdown)
        chart_paths['underwater_plot'] = self._create_underwater_plot(strategy_returns, benchmark_data, test_start_date)
        
        # 5. Return Distribution Histogram
        chart_paths['return_distribution'] = self._create_return_distribution(strategy_returns, benchmark_data, test_start_date)
        
        # 6. Position Concentration Chart
        if strategy_positions is not None:
            chart_paths['position_concentration'] = self._create_position_concentration_chart(strategy_positions)
        
        # 7. Factor Exposure Line Chart
        if factor_values is not None:
            chart_paths['factor_exposure_lines'] = self._create_factor_exposure_lines(factor_values)
        
        # 8. Performance Attribution Chart
        chart_paths['performance_attribution'] = self._create_performance_attribution(strategy_returns, benchmark_data, test_start_date)
        
        return chart_paths
    
    def _create_monthly_returns_heatmap(self, strategy_returns: pd.Series,
                                      benchmark_data: Dict[str, pd.Series],
                                      test_start_date: Optional[pd.Timestamp] = None) -> str:
        """Create monthly returns heatmap"""
        
        # Calculate monthly returns
        monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create year-month matrix
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_data = []
        
        for date, ret in monthly_returns.items():
            monthly_data.append({
                'Year': date.year,
                'Month': date.strftime('%b'),
                'Return': ret
            })
        
        df_monthly = pd.DataFrame(monthly_data)
        
        if len(df_monthly) > 0:
            # Pivot for heatmap
            heatmap_data = df_monthly.pivot(index='Year', columns='Month', values='Return')
            
            # Reorder months
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            heatmap_data = heatmap_data.reindex(columns=month_order)
            
            # Create heatmap
            fig = px.imshow(
                heatmap_data,
                title="Monthly Returns Heatmap (%)",
                aspect="auto",
                color_continuous_scale="RdYlGn",
                labels=dict(x="Month", y="Year", color="Return"),
                text_auto=".1%"
            )
            
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Year"
            )
        else:
            # Fallback empty chart
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for monthly heatmap", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            fig.update_layout(title="Monthly Returns Heatmap")
        
        file_path = os.path.join(self.output_format.output_directory, "monthly_heatmap.html")
        fig.write_html(file_path)
        return file_path
    
    def _create_risk_return_scatter(self, strategy_returns: pd.Series,
                                  benchmark_data: Dict[str, pd.Series],
                                  test_start_date: Optional[pd.Timestamp] = None) -> str:
        """Create risk-return scatter plot"""
        
        fig = go.Figure()
        
        # Calculate metrics for strategy
        strategy_annual_return = (1 + strategy_returns.mean()) ** 252 - 1
        strategy_volatility = strategy_returns.std() * np.sqrt(252)
        
        # Add strategy point
        fig.add_trace(go.Scatter(
            x=[strategy_volatility],
            y=[strategy_annual_return],
            mode='markers',
            name='Strategy',
            marker=dict(size=15, color='blue'),
            text=['Strategy'],
            textposition="top center"
        ))
        
        # Add benchmark points
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896']
        for i, (etf_symbol, etf_returns) in enumerate(benchmark_data.items()):
            aligned_returns = etf_returns.reindex(strategy_returns.index).fillna(0)
            etf_annual_return = (1 + aligned_returns.mean()) ** 252 - 1
            etf_volatility = aligned_returns.std() * np.sqrt(252)
            
            fig.add_trace(go.Scatter(
                x=[etf_volatility],
                y=[etf_annual_return],
                mode='markers',
                name=etf_symbol,
                marker=dict(size=12, color=colors[i % len(colors)]),
                text=[etf_symbol],
                textposition="top center"
            ))
        
        fig.update_layout(
            title='Risk-Return Profile',
            xaxis_title='Volatility (Annual)',
            yaxis_title='Return (Annual)',
            xaxis_tickformat='.1%',
            yaxis_tickformat='.1%'
        )
        
        file_path = os.path.join(self.output_format.output_directory, "risk_return_scatter.html")
        fig.write_html(file_path)
        return file_path
    
    def _create_rolling_beta_chart(self, strategy_returns: pd.Series,
                                 benchmark_data: Dict[str, pd.Series],
                                 test_start_date: Optional[pd.Timestamp] = None) -> str:
        """Create rolling beta chart against market benchmark"""
        
        fig = go.Figure()
        
        # Use SPY as market benchmark if available
        market_returns = None
        if 'SPY' in benchmark_data:
            market_returns = benchmark_data['SPY'].reindex(strategy_returns.index).fillna(0)
        else:
            # Use first benchmark
            market_returns = list(benchmark_data.values())[0].reindex(strategy_returns.index).fillna(0)
        
        # Calculate rolling beta (252-day window)
        window = 252
        rolling_beta = []
        dates = []
        
        for i in range(window, len(strategy_returns)):
            period_strategy = strategy_returns.iloc[i-window:i]
            period_market = market_returns.iloc[i-window:i]
            
            # Calculate beta using covariance
            covariance = np.cov(period_strategy, period_market)[0, 1]
            market_variance = np.var(period_market)
            
            if market_variance > 0:
                beta = covariance / market_variance
            else:
                beta = 0
                
            rolling_beta.append(beta)
            dates.append(strategy_returns.index[i])
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=rolling_beta,
            mode='lines',
            name='Rolling Beta (1Y)',
            line=dict(color='blue')
        ))
        
        # Add beta = 1 reference line
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                     annotation_text="Beta = 1.0")
        
        fig.update_layout(
            title='Rolling Beta vs Market (1-Year Window)',
            xaxis_title='Date',
            yaxis_title='Beta',
            hovermode='x unified'
        )
        
        file_path = os.path.join(self.output_format.output_directory, "rolling_beta.html")
        fig.write_html(file_path)
        return file_path
    
    def _create_underwater_plot(self, strategy_returns: pd.Series,
                              benchmark_data: Dict[str, pd.Series],
                              test_start_date: Optional[pd.Timestamp] = None) -> str:
        """Create underwater plot showing drawdown periods"""
        
        fig = go.Figure()
        
        # Strategy underwater plot
        strategy_cumulative = (1 + strategy_returns).cumprod()
        strategy_rolling_max = strategy_cumulative.expanding().max()
        strategy_drawdown = (strategy_cumulative - strategy_rolling_max) / strategy_rolling_max
        
        fig.add_trace(go.Scatter(
            x=strategy_drawdown.index,
            y=strategy_drawdown.values,
            mode='lines',
            name='Strategy Drawdown',
            fill='tonexty',
            line=dict(color='blue'),
            fillcolor='rgba(0, 100, 200, 0.3)'
        ))
        
        # Add benchmark underwater plots
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896']
        for i, (etf_symbol, etf_returns) in enumerate(list(benchmark_data.items())[:3]):
            aligned_returns = etf_returns.reindex(strategy_returns.index).fillna(0)
            etf_cumulative = (1 + aligned_returns).cumprod()
            etf_rolling_max = etf_cumulative.expanding().max()
            etf_drawdown = (etf_cumulative - etf_rolling_max) / etf_rolling_max
            
            fig.add_trace(go.Scatter(
                x=etf_drawdown.index,
                y=etf_drawdown.values,
                mode='lines',
                name=f'{etf_symbol} Drawdown',
                line=dict(color=colors[i % len(colors)], dash='dash')
            ))
        
        fig.update_layout(
            title='Underwater Plot - Drawdown Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown',
            yaxis_tickformat='.1%',
            hovermode='x unified'
        )
        
        file_path = os.path.join(self.output_format.output_directory, "underwater_plot.html")
        fig.write_html(file_path)
        return file_path
    
    def _create_return_distribution(self, strategy_returns: pd.Series,
                                  benchmark_data: Dict[str, pd.Series],
                                  test_start_date: Optional[pd.Timestamp] = None) -> str:
        """Create return distribution histogram"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Strategy Returns', 'SPY Returns', 'QQQ Returns', 'Distribution Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Strategy histogram
        fig.add_trace(
            go.Histogram(x=strategy_returns, name='Strategy', nbinsx=50, opacity=0.7),
            row=1, col=1
        )
        
        # Benchmark histograms
        benchmark_names = ['SPY', 'QQQ']
        positions = [(1, 2), (2, 1)]
        
        for i, etf_symbol in enumerate(benchmark_names):
            if etf_symbol in benchmark_data:
                aligned_returns = benchmark_data[etf_symbol].reindex(strategy_returns.index).fillna(0)
                fig.add_trace(
                    go.Histogram(x=aligned_returns, name=etf_symbol, nbinsx=50, opacity=0.7),
                    row=positions[i][0], col=positions[i][1]
                )
        
        # Comparison plot
        fig.add_trace(
            go.Histogram(x=strategy_returns, name='Strategy', nbinsx=30, opacity=0.5, 
                        histnorm='probability density'),
            row=2, col=2
        )
        
        if 'SPY' in benchmark_data:
            spy_returns = benchmark_data['SPY'].reindex(strategy_returns.index).fillna(0)
            fig.add_trace(
                go.Histogram(x=spy_returns, name='SPY', nbinsx=30, opacity=0.5,
                            histnorm='probability density'),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Return Distribution Analysis',
            showlegend=True,
            height=600
        )
        
        file_path = os.path.join(self.output_format.output_directory, "return_distribution.html")
        fig.write_html(file_path)
        return file_path
    
    def _create_position_concentration_chart(self, strategy_positions: pd.DataFrame) -> str:
        """Create position concentration analysis"""
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Number of Positions Over Time', 'Top 10 Position Weights'),
            vertical_spacing=0.15
        )
        
        # Number of positions over time
        position_counts = (strategy_positions != 0).sum(axis=1)
        
        fig.add_trace(
            go.Scatter(
                x=position_counts.index,
                y=position_counts.values,
                mode='lines',
                name='Active Positions',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Top position weights over time
        top_weights = strategy_positions.abs().apply(lambda x: x.nlargest(10).sum(), axis=1)
        
        fig.add_trace(
            go.Scatter(
                x=top_weights.index,
                y=top_weights.values,
                mode='lines',
                name='Top 10 Weight',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Portfolio Concentration Analysis',
            height=600
        )
        
        fig.update_yaxes(title_text="Number of Positions", row=1, col=1)
        fig.update_yaxes(title_text="Weight", tickformat='.1%', row=2, col=1)
        
        file_path = os.path.join(self.output_format.output_directory, "position_concentration.html")
        fig.write_html(file_path)
        return file_path
    
    def _create_factor_exposure_lines(self, factor_values: pd.DataFrame) -> str:
        """Create factor exposure line charts"""
        
        fig = go.Figure()
        
        # Check index structure and adjust groupby accordingly
        if isinstance(factor_values.index, pd.MultiIndex):
            # Use the first level for grouping (should be date)
            level_name = factor_values.index.names[0] if factor_values.index.names[0] else 0
            factor_means = factor_values.groupby(level=level_name).mean()
        else:
            # If no multi-index, just use the DataFrame as is (time series)
            factor_means = factor_values
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, factor_name in enumerate(factor_means.columns):
            fig.add_trace(go.Scatter(
                x=factor_means.index,
                y=factor_means[factor_name],
                mode='lines',
                name=factor_name,
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title='Factor Exposure Over Time (Cross-Sectional Means)',
            xaxis_title='Date',
            yaxis_title='Factor Value',
            hovermode='x unified'
        )
        
        file_path = os.path.join(self.output_format.output_directory, "factor_exposure_lines.html")
        fig.write_html(file_path)
        return file_path
    
    def _create_performance_attribution(self, strategy_returns: pd.Series,
                                      benchmark_data: Dict[str, pd.Series],
                                      test_start_date: Optional[pd.Timestamp] = None) -> str:
        """Create performance attribution analysis"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative Excess Returns vs SPY', 'Rolling Correlation vs Market',
                           'Up/Down Capture Analysis', 'Performance Statistics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        # Excess returns vs SPY
        if 'SPY' in benchmark_data:
            spy_returns = benchmark_data['SPY'].reindex(strategy_returns.index).fillna(0)
            excess_returns = strategy_returns - spy_returns
            cumulative_excess = (1 + excess_returns).cumprod() - 1
            
            fig.add_trace(
                go.Scatter(
                    x=cumulative_excess.index,
                    y=cumulative_excess.values,
                    mode='lines',
                    name='Excess Return',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # Rolling correlation
        if 'SPY' in benchmark_data:
            spy_returns = benchmark_data['SPY'].reindex(strategy_returns.index).fillna(0)
            rolling_corr = strategy_returns.rolling(63).corr(spy_returns)  # 3-month window
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode='lines',
                    name='Rolling Correlation',
                    line=dict(color='red')
                ),
                row=1, col=2
            )
        
        # Up/Down capture
        if 'SPY' in benchmark_data:
            spy_returns = benchmark_data['SPY'].reindex(strategy_returns.index).fillna(0)
            up_market = spy_returns > 0
            down_market = spy_returns < 0
            
            up_capture = strategy_returns[up_market].mean() / spy_returns[up_market].mean() if spy_returns[up_market].mean() != 0 else 0
            down_capture = strategy_returns[down_market].mean() / spy_returns[down_market].mean() if spy_returns[down_market].mean() != 0 else 0
            
            fig.add_trace(
                go.Bar(
                    x=['Up Capture', 'Down Capture'],
                    y=[up_capture, down_capture],
                    name='Capture Ratios',
                    marker_color=['green', 'red']
                ),
                row=2, col=1
            )
        
        # Performance statistics table
        stats_data = [
            ['Metric', 'Strategy', 'SPY' if 'SPY' in benchmark_data else 'Benchmark'],
            ['Annual Return', f"{((1 + strategy_returns.mean()) ** 252 - 1):.2%}", 
             f"{((1 + benchmark_data['SPY'].reindex(strategy_returns.index).fillna(0).mean()) ** 252 - 1):.2%}" if 'SPY' in benchmark_data else 'N/A'],
            ['Volatility', f"{(strategy_returns.std() * np.sqrt(252)):.2%}",
             f"{(benchmark_data['SPY'].reindex(strategy_returns.index).fillna(0).std() * np.sqrt(252)):.2%}" if 'SPY' in benchmark_data else 'N/A'],
            ['Sharpe Ratio', f"{((1 + strategy_returns.mean()) ** 252 - 1) / (strategy_returns.std() * np.sqrt(252)):.2f}",
             f"{((1 + benchmark_data['SPY'].reindex(strategy_returns.index).fillna(0).mean()) ** 252 - 1) / (benchmark_data['SPY'].reindex(strategy_returns.index).fillna(0).std() * np.sqrt(252)):.2f}" if 'SPY' in benchmark_data else 'N/A']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=stats_data[0], fill_color='lightgray'),
                cells=dict(values=list(zip(*stats_data[1:])), fill_color='white')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Performance Attribution Analysis',
            height=800,
            showlegend=False
        )
        
        file_path = os.path.join(self.output_format.output_directory, "performance_attribution.html")
        fig.write_html(file_path)
        return file_path
    
    def _create_rolling_drawdown_chart(self, strategy_returns: pd.Series, test_start_date: pd.Timestamp = None) -> str:
        """Create rolling drawdown analysis chart for hourly strategy audit"""
        
        # Calculate cumulative returns and drawdowns
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Calculate rolling max drawdowns (1 week and 1 month windows)
        rolling_1w_dd = drawdown.rolling(window=24*7).min()  # 1 week in hours
        rolling_1m_dd = drawdown.rolling(window=24*30).min()  # 1 month in hours
        
        # Calculate recovery time
        recovery_periods = []
        in_drawdown = False
        dd_start = None
        
        for i, dd in enumerate(drawdown):
            if dd < -0.001 and not in_drawdown:  # Start of drawdown
                in_drawdown = True
                dd_start = i
            elif dd >= -0.001 and in_drawdown:  # End of drawdown (recovery)
                recovery_time = i - dd_start
                recovery_periods.append(recovery_time)
                in_drawdown = False
        
        avg_recovery_hours = np.mean(recovery_periods) if recovery_periods else 0
        avg_recovery_days = avg_recovery_hours / 24
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Cumulative Returns vs Rolling Peak',
                'Drawdown Series with Recovery Analysis',
                'Rolling Maximum Drawdown (1W & 1M Windows)'
            ),
            vertical_spacing=0.12,
            row_heights=[0.35, 0.35, 0.3]
        )
        
        # First subplot: Cumulative returns and running peak
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                name='Cumulative Returns',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=running_max.index,
                y=running_max.values,
                name='Running Peak',
                line=dict(color='green', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # Second subplot: Drawdown series with recovery zones
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,  # Convert to percentage
                name='Drawdown %',
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # Add horizontal line at -1% for reference
        fig.add_hline(y=-1.0, line_dash="dot", line_color="gray", 
                     annotation_text="1% DD Level", row=2, col=1)
        
        # Third subplot: Rolling max drawdowns
        fig.add_trace(
            go.Scatter(
                x=rolling_1w_dd.index,
                y=rolling_1w_dd.values * 100,
                name='1-Week Max DD',
                line=dict(color='orange', width=2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_1m_dd.index,
                y=rolling_1m_dd.values * 100,
                name='1-Month Max DD',
                line=dict(color='red', width=2)
            ),
            row=3, col=1
        )
        
        # Add test start date line if provided
        if test_start_date:
            for row_num in [1, 2, 3]:
                # Simplified shape addition without subplot-specific yref
                if row_num == 1:  # Only add to first subplot to avoid complex yref issues
                    fig.add_shape(
                        type="line",
                        x0=test_start_date, x1=test_start_date,
                        y0=0, y1=1,
                        yref="paper",
                        line=dict(dash="dash", color="gray", width=2)
                    )
                # Add annotation separately
                if row_num == 1:  # Only add annotation to first subplot to avoid clutter
                    fig.add_annotation(
                        x=test_start_date,
                        y=0.9,
                        yref="paper",
                        text="Test Start",
                        showarrow=False,
                        font=dict(color="gray", size=10)
                    )
        
        # Update layout
        fig.update_layout(
            title=f'Rolling Drawdown Analysis - Hourly Strategy<br><sub>Avg Recovery Time: {avg_recovery_days:.1f} days</sub>',
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Cumulative Value", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Max Drawdown (%)", row=3, col=1)
        
        # Update x-axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        # Save chart
        import os
        chart_path = os.path.join(self.output_format.output_directory, "rolling_drawdown_analysis.html")
        fig.write_html(str(chart_path))
        print(f"💹 Rolling drawdown analysis chart saved: {chart_path}")
        
        return str(chart_path)
    
    def _create_return_distribution_chart(self, strategy_returns: pd.Series, 
                                        benchmark_data: Dict[str, pd.Series],
                                        test_start_date: pd.Timestamp = None) -> str:
        """Create return distribution analysis chart with skewness and kurtosis"""
        
        # Calculate distribution statistics
        strategy_stats = {
            'mean': strategy_returns.mean() * 100,  # Convert to percentage
            'std': strategy_returns.std() * 100,
            'skew': strategy_returns.skew(),
            'kurt': strategy_returns.kurtosis(),
            'var_95': strategy_returns.quantile(0.05) * 100,
            'var_99': strategy_returns.quantile(0.01) * 100
        }
        
        # Get benchmark comparison if available
        benchmark_stats = {}
        if 'SPY' in benchmark_data:
            spy_returns = benchmark_data['SPY'].reindex(strategy_returns.index).fillna(0)
            benchmark_stats = {
                'mean': spy_returns.mean() * 100,
                'std': spy_returns.std() * 100,
                'skew': spy_returns.skew(),
                'kurt': spy_returns.kurtosis(),
                'var_95': spy_returns.quantile(0.05) * 100,
                'var_99': spy_returns.quantile(0.01) * 100
            }
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Return Distribution Histogram',
                'Q-Q Plot vs Normal Distribution', 
                'Rolling Skewness (24H Window)',
                'Rolling Kurtosis (24H Window)'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Return distribution histogram
        fig.add_trace(
            go.Histogram(
                x=strategy_returns * 100,
                nbinsx=50,
                name='Strategy Returns',
                opacity=0.7,
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        if benchmark_stats:
            fig.add_trace(
                go.Histogram(
                    x=spy_returns * 100,
                    nbinsx=50,
                    name='SPY Returns',
                    opacity=0.5,
                    marker_color='red'
                ),
                row=1, col=1
            )
        
        # Add VaR lines
        fig.add_vline(x=strategy_stats['var_95'], line_dash="dash", line_color="red",
                     annotation_text=f"VaR 95%: {strategy_stats['var_95']:.2f}%", row=1, col=1)
        fig.add_vline(x=strategy_stats['mean'], line_dash="dot", line_color="green",
                     annotation_text=f"Mean: {strategy_stats['mean']:.3f}%", row=1, col=1)
        
        # 2. Q-Q Plot vs Normal
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(strategy_returns)))
        sample_quantiles = np.sort(strategy_returns)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(size=4, color='blue', opacity=0.6)
            ),
            row=1, col=2
        )
        
        # Add reference line for perfect normal distribution
        min_val, max_val = min(theoretical_quantiles), max(theoretical_quantiles)
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val * strategy_returns.std() + strategy_returns.mean(), 
                   max_val * strategy_returns.std() + strategy_returns.mean()],
                mode='lines',
                name='Normal Reference',
                line=dict(dash='dash', color='red')
            ),
            row=1, col=2
        )
        
        # 3. Rolling skewness
        rolling_skew = strategy_returns.rolling(window=24).apply(lambda x: x.skew())  # 24-hour window
        fig.add_trace(
            go.Scatter(
                x=rolling_skew.index,
                y=rolling_skew.values,
                mode='lines',
                name='Rolling Skewness',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=0, line_dash="dot", line_color="gray", 
                     annotation_text="Symmetric", row=2, col=1)
        
        # 4. Rolling kurtosis
        rolling_kurt = strategy_returns.rolling(window=24).apply(lambda x: x.kurtosis())  # 24-hour window
        fig.add_trace(
            go.Scatter(
                x=rolling_kurt.index,
                y=rolling_kurt.values,
                mode='lines',
                name='Rolling Kurtosis',
                line=dict(color='orange', width=2)
            ),
            row=2, col=2
        )
        
        fig.add_hline(y=3, line_dash="dot", line_color="gray", 
                     annotation_text="Normal Kurtosis", row=2, col=2)
        
        # Add test start date line for time series plots
        if test_start_date:
            for row_num, col_num in [(2, 1), (2, 2)]:
                y_axis_ref = f"y{(row_num-1)*2 + col_num}"
                fig.add_shape(
                    type="line",
                    x0=test_start_date, x1=test_start_date,
                    y0=0, y1=1,
                    yref=f"{y_axis_ref} domain",
                    line=dict(dash="dash", color="gray", width=2),
                    row=row_num, col=col_num
                )
        
        # Update layout
        title_text = f'Return Distribution Analysis - Hourly Strategy<br>'
        title_text += f'<sub>Mean: {strategy_stats["mean"]:.3f}%, Std: {strategy_stats["std"]:.3f}%, '
        title_text += f'Skew: {strategy_stats["skew"]:.2f}, Kurt: {strategy_stats["kurt"]:.2f}</sub>'
        
        fig.update_layout(
            title=title_text,
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Skewness", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Kurtosis", row=2, col=2)
        
        # Save chart
        chart_path = os.path.join(self.output_format.output_directory, "return_distribution_analysis.html")
        fig.write_html(str(chart_path))
        print(f" Return distribution analysis chart saved: {chart_path}")
        
        return str(chart_path)
    
    def _create_up_down_capture_chart(self, strategy_returns: pd.Series,
                                    benchmark_data: Dict[str, pd.Series],
                                    test_start_date: pd.Timestamp = None) -> str:
        """Create up/down capture analysis chart"""
        
        if 'SPY' not in benchmark_data:
            print("No SPY benchmark data available for up/down capture analysis")
            return ""
        
        spy_returns = benchmark_data['SPY'].reindex(strategy_returns.index).fillna(0)
        
        # Calculate rolling up/down capture ratios
        window = 24 * 7  # 1 week rolling window
        rolling_up_capture = []
        rolling_down_capture = []
        dates = []
        
        for i in range(window, len(strategy_returns)):
            end_idx = i
            start_idx = i - window
            
            strategy_window = strategy_returns.iloc[start_idx:end_idx]
            benchmark_window = spy_returns.iloc[start_idx:end_idx]
            
            # Up market periods
            up_market = benchmark_window > 0
            if up_market.sum() > 5:  # Need at least 5 observations
                up_capture = strategy_window[up_market].mean() / benchmark_window[up_market].mean()
                rolling_up_capture.append(up_capture)
            else:
                rolling_up_capture.append(np.nan)
            
            # Down market periods  
            down_market = benchmark_window < 0
            if down_market.sum() > 5:  # Need at least 5 observations
                down_capture = strategy_window[down_market].mean() / benchmark_window[down_market].mean()
                rolling_down_capture.append(down_capture)
            else:
                rolling_down_capture.append(np.nan)
            
            dates.append(strategy_returns.index[end_idx])
        
        # Create scatter plot of strategy vs benchmark returns
        up_market_mask = spy_returns > 0
        down_market_mask = spy_returns < 0
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Strategy vs Benchmark Returns (Scatter)',
                'Rolling Up/Down Capture Ratios',
                'Up Market Performance Distribution',
                'Down Market Performance Distribution'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Scatter plot of returns
        fig.add_trace(
            go.Scatter(
                x=spy_returns[up_market_mask] * 100,
                y=strategy_returns[up_market_mask] * 100,
                mode='markers',
                name='Up Market',
                marker=dict(color='green', opacity=0.6, size=4)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=spy_returns[down_market_mask] * 100,
                y=strategy_returns[down_market_mask] * 100,
                mode='markers',
                name='Down Market',
                marker=dict(color='red', opacity=0.6, size=4)
            ),
            row=1, col=1
        )
        
        # Add diagonal reference line
        min_ret, max_ret = min(spy_returns.min(), strategy_returns.min()) * 100, max(spy_returns.max(), strategy_returns.max()) * 100
        fig.add_trace(
            go.Scatter(
                x=[min_ret, max_ret],
                y=[min_ret, max_ret],
                mode='lines',
                name='1:1 Line',
                line=dict(dash='dash', color='gray')
            ),
            row=1, col=1
        )
        
        # 2. Rolling capture ratios
        dates_series = pd.Series(dates)
        up_capture_series = pd.Series(rolling_up_capture, index=dates)
        down_capture_series = pd.Series(rolling_down_capture, index=dates)
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_up_capture,
                mode='lines',
                name='Up Capture',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_down_capture,
                mode='lines',
                name='Down Capture',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray", 
                     annotation_text="100% Capture", row=1, col=2)
        
        # 3. Up market distribution
        up_strategy = strategy_returns[up_market_mask] * 100
        up_benchmark = spy_returns[up_market_mask] * 100
        
        fig.add_trace(
            go.Histogram(
                x=up_strategy,
                nbinsx=30,
                name='Strategy (Up)',
                opacity=0.7,
                marker_color='green'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=up_benchmark,
                nbinsx=30,
                name='SPY (Up)',
                opacity=0.5,
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
        
        # 4. Down market distribution
        down_strategy = strategy_returns[down_market_mask] * 100
        down_benchmark = spy_returns[down_market_mask] * 100
        
        fig.add_trace(
            go.Histogram(
                x=down_strategy,
                nbinsx=30,
                name='Strategy (Down)',
                opacity=0.7,
                marker_color='red'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=down_benchmark,
                nbinsx=30,
                name='SPY (Down)',
                opacity=0.5,
                marker_color='lightcoral'
            ),
            row=2, col=2
        )
        
        # Add test start date line for time series
        if test_start_date:
            fig.add_shape(
                type="line",
                x0=test_start_date, x1=test_start_date,
                y0=0, y1=1,
                yref="y2 domain",
                line=dict(dash="dash", color="gray", width=2),
                row=1, col=2
            )
        
        # Calculate overall capture ratios
        overall_up_capture = strategy_returns[up_market_mask].mean() / spy_returns[up_market_mask].mean() if spy_returns[up_market_mask].mean() != 0 else 0
        overall_down_capture = strategy_returns[down_market_mask].mean() / spy_returns[down_market_mask].mean() if spy_returns[down_market_mask].mean() != 0 else 0
        
        # Update layout
        title_text = f'Up/Down Capture Analysis - Hourly Strategy<br>'
        title_text += f'<sub>Overall Up Capture: {overall_up_capture:.2f}, Down Capture: {overall_down_capture:.2f}</sub>'
        
        fig.update_layout(
            title=title_text,
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Benchmark Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Strategy Return (%)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Capture Ratio", row=1, col=2)
        fig.update_xaxes(title_text="Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Return (%)", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        # Save chart
        chart_path = os.path.join(self.output_format.output_directory, "up_down_capture_analysis.html")
        fig.write_html(str(chart_path))
        print(f" Up/down capture analysis chart saved: {chart_path}")
        
        return str(chart_path)
