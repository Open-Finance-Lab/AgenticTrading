"""
Backtest Visualization Module
Decoupled visualization functions for backtest results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

try:
    import qlib
    from qlib.contrib.evaluate import risk_analysis
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False


class BacktestVisualizer:
    """
    Decoupled visualization class for backtest results
    Can be used independently without BacktestAgent
    """
    
    def __init__(self, figsize=(12, 8)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    def plot_pnl_curve(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None, 
                      title: str = "P&L Curve", save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot P&L (Profit & Loss) curve from returns
        
        Args:
            returns: Portfolio returns series with datetime index
            benchmark_returns: Optional benchmark returns for comparison
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        cumulative_pnl = (cumulative_returns - 1) * 100  # Convert to percentage
        
        # Plot portfolio P&L
        ax.plot(cumulative_returns.index, cumulative_pnl, 
               label='Portfolio', linewidth=2, color='#2E86AB')
        
        # Plot benchmark if provided
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            benchmark_pnl = (benchmark_cumulative - 1) * 100
            ax.plot(benchmark_cumulative.index, benchmark_pnl, 
                   label='Benchmark', linewidth=2, color='#A23B72', linestyle='--')
        
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative P&L (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        if len(cumulative_returns) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(cumulative_returns)//10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ P&L curve saved to {save_path}")
        
        return fig
    
    def plot_drawdown(self, returns: pd.Series, title: str = "Drawdown Analysis", 
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot drawdown curve
        
        Args:
            returns: Portfolio returns series with datetime index
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max * 100
        
        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown, 0, 
                       color='#F18F01', alpha=0.6, label='Drawdown')
        ax.plot(drawdown.index, drawdown, color='#C73E1D', linewidth=1.5)
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        if len(drawdown) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(drawdown)//10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Drawdown chart saved to {save_path}")
        
        return fig
    
    def plot_metrics_table(self, metrics: Dict, title: str = "Performance Metrics", 
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a table visualization of performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
            title: Table title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data for table
        metric_names = []
        metric_values = []
        
        # Common metrics to display
        metric_mapping = {
            'total_return': 'Total Return (%)',
            'cumulative_return': 'Cumulative Return (%)',
            'sharpe_ratio': 'Sharpe Ratio',
            'volatility': 'Volatility (%)',
            'max_drawdown': 'Max Drawdown (%)',
            'calmar_ratio': 'Calmar Ratio',
            'win_rate': 'Win Rate (%)',
            'profit_factor': 'Profit Factor'
        }
        
        for key, display_name in metric_mapping.items():
            if key in metrics:
                value = metrics[key]
                # Format percentage values
                if 'Return' in display_name or 'Drawdown' in display_name or 'Volatility' in display_name or 'Win Rate' in display_name:
                    if isinstance(value, (int, float)):
                        metric_values.append(f"{value * 100:.2f}%")
                    else:
                        metric_values.append(str(value))
                else:
                    metric_values.append(f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
                metric_names.append(display_name)
        
        # Create table
        table_data = [[name, value] for name, value in zip(metric_names, metric_values)]
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(table_data) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F5F5F5')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Metrics table saved to {save_path}")
        
        return fig
    
    def plot_returns_distribution(self, returns: pd.Series, title: str = "Returns Distribution", 
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot returns distribution histogram
        
        Args:
            returns: Portfolio returns series
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert to percentage
        returns_pct = returns * 100
        
        # Plot histogram
        ax.hist(returns_pct, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
        
        # Add mean line
        mean_return = returns_pct.mean()
        ax.axvline(mean_return, color='#C73E1D', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_return:.2f}%')
        
        # Add zero line
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Daily Return (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Returns distribution saved to {save_path}")
        
        return fig
    
    def plot_monthly_returns(self, returns: pd.Series, title: str = "Monthly Returns Heatmap", 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot monthly returns as a heatmap
        
        Args:
            returns: Portfolio returns series with datetime index
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Resample to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create year-month matrix
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_df = pd.DataFrame({
            'return': monthly_returns.values,
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month
        }, index=monthly_returns.index)
        
        # Pivot for heatmap
        pivot_data = monthly_df.pivot_table(
            values='return',
            index='Year',
            columns='Month',
            aggfunc='first'
        )
        
        # Create heatmap
        im = ax.imshow(pivot_data.values * 100, cmap='RdYlGn', aspect='auto', 
                      vmin=-10, vmax=10)
        
        # Set ticks
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_yticklabels(pivot_data.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Monthly Return (%)', fontsize=10)
        
        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(12):
                value = pivot_data.iloc[i, j]
                if not pd.isna(value):
                    text = ax.text(j, i, f'{value*100:.1f}%',
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Monthly returns heatmap saved to {save_path}")
        
        return fig
    
    def generate_comprehensive_report(self, returns: pd.Series, metrics: Dict, 
                                     benchmark_returns: Optional[pd.Series] = None,
                                     output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate comprehensive visualization report
        
        Args:
            returns: Portfolio returns series
            metrics: Performance metrics dictionary
            benchmark_returns: Optional benchmark returns
            output_dir: Directory to save all plots
            
        Returns:
            Dictionary with paths to saved plots
        """
        saved_paths = {}
        
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate all plots
        plots = {
            'pnl_curve': self.plot_pnl_curve(returns, benchmark_returns, 
                                            save_path=f"{output_dir}/pnl_curve.png" if output_dir else None),
            'drawdown': self.plot_drawdown(returns, 
                                          save_path=f"{output_dir}/drawdown.png" if output_dir else None),
            'metrics_table': self.plot_metrics_table(metrics, 
                                                     save_path=f"{output_dir}/metrics_table.png" if output_dir else None),
            'returns_distribution': self.plot_returns_distribution(returns, 
                                                                  save_path=f"{output_dir}/returns_distribution.png" if output_dir else None),
            'monthly_returns': self.plot_monthly_returns(returns, 
                                                         save_path=f"{output_dir}/monthly_returns.png" if output_dir else None)
        }
        
        if output_dir:
            for name, path in saved_paths.items():
                saved_paths[name] = path
        
        # Close all figures to free memory
        plt.close('all')
        
        return plots


# Convenience functions for standalone use
def plot_pnl_curve(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None, 
                   save_path: Optional[str] = None) -> plt.Figure:
    """Convenience function to plot P&L curve"""
    visualizer = BacktestVisualizer()
    return visualizer.plot_pnl_curve(returns, benchmark_returns, save_path=save_path)


def plot_backtest_metrics(metrics: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """Convenience function to plot metrics table"""
    visualizer = BacktestVisualizer()
    return visualizer.plot_metrics_table(metrics, save_path=save_path)

