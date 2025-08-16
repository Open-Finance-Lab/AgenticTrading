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
    
    # Comparison metrics (vs benchmark)
    excess_return: float
    tracking_error: float
    up_capture: float
    down_capture: float


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
                               benchmark_data: Optional[Dict[str, pd.Series]] = None) -> Dict[str, Any]:
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
                                             strategy_positions, factor_values)
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
        
        # Basic return metrics
        total_return = (1 + returns).cumprod().iloc[-1] - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        
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
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        average_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        average_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        # Ensure trading metrics are scalars
        if hasattr(average_win, 'item'):
            average_win = average_win.item()
        if hasattr(average_loss, 'item'):
            average_loss = average_loss.item()
        profit_factor = abs(average_win / average_loss) if average_loss != 0 else 0
        
        # Advanced metrics
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        # Ensure downside_volatility is a scalar
        if hasattr(downside_volatility, 'item'):
            downside_volatility = downside_volatility.item()
        sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
        
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
            excess_return=0,  # Will be calculated in comparison
            tracking_error=0,  # Will be calculated in comparison
            up_capture=0,  # Will be calculated in comparison
            down_capture=0  # Will be calculated in comparison
        )
    
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
                          factor_values: Optional[pd.DataFrame]) -> Dict[str, str]:
        """Generate all visualization charts"""
        
        chart_paths = {}
        
        if self.output_format.generate_performance_chart:
            chart_paths['performance'] = self._create_performance_chart(strategy_returns, benchmark_data)
            
        if self.output_format.generate_drawdown_chart:
            chart_paths['drawdown'] = self._create_drawdown_chart(strategy_returns, benchmark_data)
            
        if self.output_format.generate_rolling_metrics_chart:
            chart_paths['rolling_metrics'] = self._create_rolling_metrics_chart(strategy_returns, benchmark_data)
            
        if self.output_format.generate_correlation_matrix and factor_values is not None:
            chart_paths['correlation'] = self._create_correlation_matrix(factor_values)
            
        # NEW: Excess return comparison chart
        if self.output_format.generate_excess_return_chart:
            chart_paths['excess_return'] = self._create_excess_return_chart(strategy_returns, benchmark_data)
            
        # NEW: Strategy signal analysis chart
        if self.output_format.generate_signal_analysis_chart and strategy_positions is not None:
            chart_paths['signal_analysis'] = self._create_signal_analysis_chart(strategy_positions, strategy_returns)
            
        # Additional advanced visualizations
        chart_paths.update(self._create_advanced_visualizations(strategy_returns, benchmark_data, 
                                                               strategy_positions, factor_values))
            
        return chart_paths
    
    def _create_performance_chart(self, strategy_returns: pd.Series, 
                                benchmark_data: Dict[str, pd.Series]) -> str:
        """Create cumulative performance comparison chart"""
        
        fig = go.Figure()
        
        # Strategy performance
        strategy_cumulative = (1 + strategy_returns).cumprod()
        fig.add_trace(go.Scatter(
            x=strategy_cumulative.index,
            y=strategy_cumulative.values,
            mode='lines',
            name='Strategy',
            line=dict(color='blue', width=2)
        ))
        
        # Benchmark performances
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (etf_symbol, etf_returns) in enumerate(benchmark_data.items()):
            aligned_returns = etf_returns.reindex(strategy_returns.index).fillna(0)
            etf_cumulative = (1 + aligned_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=etf_cumulative.index,
                y=etf_cumulative.values,
                mode='lines',
                name=etf_symbol,
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title='Cumulative Performance Comparison',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            hovermode='x unified'
        )
        
        file_path = os.path.join(self.output_format.output_directory, "performance_chart.html")
        fig.write_html(file_path)
        return file_path
    
    def _create_drawdown_chart(self, strategy_returns: pd.Series,
                             benchmark_data: Dict[str, pd.Series]) -> str:
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
        colors = ['red', 'green', 'orange', 'purple']
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
                                    benchmark_data: Dict[str, pd.Series]) -> str:
        """Create rolling Sharpe ratio chart"""
        
        window = 252  # 1 year rolling window
        
        fig = go.Figure()
        
        # Strategy rolling Sharpe
        strategy_rolling_sharpe = strategy_returns.rolling(window).mean() / strategy_returns.rolling(window).std() * np.sqrt(252)
        
        fig.add_trace(go.Scatter(
            x=strategy_rolling_sharpe.index,
            y=strategy_rolling_sharpe.values,
            mode='lines',
            name='Strategy',
            line=dict(color='blue')
        ))
        
        # Benchmark rolling Sharpe
        colors = ['red', 'green', 'orange', 'purple']
        for i, (etf_symbol, etf_returns) in enumerate(benchmark_data.items()):
            aligned_returns = etf_returns.reindex(strategy_returns.index).fillna(0)
            etf_rolling_sharpe = aligned_returns.rolling(window).mean() / aligned_returns.rolling(window).std() * np.sqrt(252)
            
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
        
        correlation_matrix = factor_values.corr()
        
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
                                  benchmark_data: Dict[str, pd.Series]) -> str:
        """Create excess return comparison chart"""
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Strategy vs Benchmarks: Cumulative Returns Comparison', 
                'Rolling 30-Day Returns Comparison',
                'Strategy Excess Returns vs Benchmarks'
            ),
            row_heights=[0.33, 0.33, 0.34],
            vertical_spacing=0.10,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        
        # Calculate strategy cumulative performance for comparison
        strategy_cumulative = (1 + strategy_returns).cumprod()
        
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
        
        # Second subplot: Add strategy rolling returns
        strategy_rolling = strategy_returns.rolling(30).mean()
        fig.add_trace(go.Scatter(
            x=strategy_rolling.index,
            y=strategy_rolling.values * 100,  # Convert to percentage
            mode='lines',
            name='Strategy 30D',
            line=dict(color='blue', width=3),
            showlegend=False,
            hovertemplate='<b>Strategy 30D</b><br>' +
                         'Date: %{x}<br>' +
                         'Rolling Return: %{y:.2f}%<extra></extra>'
        ), row=2, col=1)
        
        # Calculate and plot benchmark comparisons
        excess_stats = {}
        for i, (etf_symbol, etf_returns) in enumerate(benchmark_data.items()):
            # Align data
            aligned_benchmark = etf_returns.reindex(strategy_returns.index).fillna(0)
            benchmark_cumulative = (1 + aligned_benchmark).cumprod()
            
            # Calculate excess returns for stats
            excess_returns = strategy_returns - aligned_benchmark
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
            
            # Second subplot: Benchmark rolling returns
            benchmark_rolling = aligned_benchmark.rolling(30).mean()
            fig.add_trace(go.Scatter(
                x=benchmark_rolling.index,
                y=benchmark_rolling.values * 100,  # Convert to percentage
                mode='lines',
                name=f'{etf_symbol} 30D',
                line=dict(color=colors[i % len(colors)], width=2, dash='dot'),
                showlegend=False,
                hovertemplate=f'<b>{etf_symbol} 30D</b><br>' +
                             'Date: %{x}<br>' +
                             'Rolling Return: %{y:.2f}%<extra></extra>'
            ), row=2, col=1)
        
        # Third subplot: Excess returns analysis
        for i, (etf_symbol, etf_returns) in enumerate(benchmark_data.items()):
            # Align data
            aligned_benchmark = etf_returns.reindex(strategy_returns.index).fillna(0)
            
            # Calculate excess returns
            excess_returns = strategy_returns - aligned_benchmark
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
                'text': f'Strategy vs Benchmarks Performance Analysis<br><sub>{summary_text}</sub>',
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
        fig.update_yaxes(title_text="30-Day Rolling Return (%)", row=2, col=1)
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
                'Signal Change Frequency', 
                'Strategy Returns vs Signal Strength',
                'Cumulative Strategy Performance vs Buy-and-Hold'
            ),
            row_heights=[0.25, 0.25, 0.25, 0.25],
            vertical_spacing=0.12,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Prepare data
        if isinstance(strategy_positions, pd.DataFrame):
            # First subplot: Weight distribution time series
            for i, symbol in enumerate(strategy_positions.columns):
                if symbol == 'date':
                    continue
                    
                weights = strategy_positions[symbol]
                colors_cycle = ['blue', 'red', 'green', 'orange', 'purple']
                
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
            
            # Second subplot: Signal change frequency
            weight_changes = strategy_positions.diff().abs().sum(axis=1)
            
            fig.add_trace(go.Bar(
                x=weight_changes.index,
                y=weight_changes,
                name='Position Change Magnitude',
                marker_color='lightcoral',
                opacity=0.7,
                showlegend=False
            ), row=2, col=1)
            
            # Third subplot: Strategy returns vs signal strength
            if len(strategy_returns) > 1:
                # Get current period signals and next period returns
                total_weights = strategy_positions.abs().sum(axis=1)
                current_returns = strategy_returns
                
                # Align data
                common_index = total_weights.index.intersection(current_returns.index)
                if len(common_index) > 10:
                    aligned_weights = total_weights.loc[common_index]
                    aligned_returns = current_returns.loc[common_index]
                    
                    # Color points by return magnitude
                    fig.add_trace(go.Scatter(
                        x=aligned_weights,
                        y=aligned_returns,
                        mode='markers',
                        name='Signal Strength vs Returns',
                        marker=dict(
                            color=aligned_returns,
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(
                                title="Daily Return",
                                x=1.02,
                                len=0.3,
                                y=0.4
                            ),
                            size=8,
                            opacity=0.7
                        ),
                        showlegend=False
                    ), row=3, col=1)
            
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
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Strategy Signal Analysis Dashboard',
                'x': 0.5,
                'xanchor': 'center'
            },
            height=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Signal Strength (Absolute Weight)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
        fig.update_yaxes(title_text="Position Weight", row=1, col=1)
        fig.update_yaxes(title_text="Change Magnitude", row=2, col=1)
        fig.update_yaxes(title_text="Daily Return", row=3, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=4, col=1)
        
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
                insights.append(f"✅ Strategy outperformed SPY on risk-adjusted basis (Sharpe: {strategy_metrics.sharpe_ratio:.2f} vs {spy_metrics.sharpe_ratio:.2f})")
            else:
                insights.append(f"❌ Strategy underperformed SPY on risk-adjusted basis (Sharpe: {strategy_metrics.sharpe_ratio:.2f} vs {spy_metrics.sharpe_ratio:.2f})")
                
            if strategy_metrics.annual_return > spy_metrics.annual_return:
                insights.append(f"✅ Strategy delivered higher annual returns ({strategy_metrics.annual_return:.2%} vs {spy_metrics.annual_return:.2%})")
            else:
                insights.append(f"❌ Strategy delivered lower annual returns ({strategy_metrics.annual_return:.2%} vs {spy_metrics.annual_return:.2%})")
        
        # Risk assessment
        if strategy_metrics.max_drawdown < -0.2:
            insights.append(f"⚠️ High maximum drawdown: {strategy_metrics.max_drawdown:.2%}")
        else:
            insights.append(f"✅ Reasonable maximum drawdown: {strategy_metrics.max_drawdown:.2%}")
            
        # Win rate assessment
        if strategy_metrics.win_rate > 0.55:
            insights.append(f"✅ Good win rate: {strategy_metrics.win_rate:.2%}")
        elif strategy_metrics.win_rate < 0.45:
            insights.append(f"⚠️ Low win rate: {strategy_metrics.win_rate:.2%}")
        else:
            insights.append(f"➖ Average win rate: {strategy_metrics.win_rate:.2%}")
        
        return "<ul>" + "".join([f"<li>{insight}</li>" for insight in insights]) + "</ul>"
    
    def _create_advanced_visualizations(self, strategy_returns: pd.Series,
                                      benchmark_data: Dict[str, pd.Series],
                                      strategy_positions: pd.DataFrame,
                                      factor_values: Optional[pd.DataFrame]) -> Dict[str, str]:
        """Create additional advanced visualizations"""
        
        chart_paths = {}
        
        # 1. Monthly Returns Heatmap
        chart_paths['monthly_heatmap'] = self._create_monthly_returns_heatmap(strategy_returns, benchmark_data)
        
        # 2. Risk-Return Scatter Plot
        chart_paths['risk_return_scatter'] = self._create_risk_return_scatter(strategy_returns, benchmark_data)
        
        # 3. Rolling Beta Chart
        chart_paths['rolling_beta'] = self._create_rolling_beta_chart(strategy_returns, benchmark_data)
        
        # 4. Underwater Plot (Drawdown)
        chart_paths['underwater_plot'] = self._create_underwater_plot(strategy_returns, benchmark_data)
        
        # 5. Return Distribution Histogram
        chart_paths['return_distribution'] = self._create_return_distribution(strategy_returns, benchmark_data)
        
        # 6. Position Concentration Chart
        if strategy_positions is not None:
            chart_paths['position_concentration'] = self._create_position_concentration_chart(strategy_positions)
        
        # 7. Factor Exposure Line Chart
        if factor_values is not None:
            chart_paths['factor_exposure_lines'] = self._create_factor_exposure_lines(factor_values)
        
        # 8. Performance Attribution Chart
        chart_paths['performance_attribution'] = self._create_performance_attribution(strategy_returns, benchmark_data)
        
        return chart_paths
    
    def _create_monthly_returns_heatmap(self, strategy_returns: pd.Series,
                                      benchmark_data: Dict[str, pd.Series]) -> str:
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
                                  benchmark_data: Dict[str, pd.Series]) -> str:
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
        colors = ['red', 'green', 'orange', 'purple', 'brown']
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
                                 benchmark_data: Dict[str, pd.Series]) -> str:
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
                              benchmark_data: Dict[str, pd.Series]) -> str:
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
        colors = ['red', 'green', 'orange']
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
                                  benchmark_data: Dict[str, pd.Series]) -> str:
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
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
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
                                      benchmark_data: Dict[str, pd.Series]) -> str:
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
