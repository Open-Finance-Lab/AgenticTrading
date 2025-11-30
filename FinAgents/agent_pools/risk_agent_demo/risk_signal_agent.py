"""
Risk Signal Agent using OpenAI Agent SDK

This agent integrates Qlib data access, risk metrics calculation, and LLM reasoning
to generate risk signals based on market data analysis.

Key Features:
- Qlib-based data loading
- Risk metrics calculation (Volatility, VaR, CVaR, Max Drawdown, Beta, etc.)
- Risk signal generation
- LLM-enhanced risk assessment
"""

import os
import sys
import json
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

# Import OpenAI Agents SDK
try:
    # First try standard SDK import
    from agents import Agent, Runner, function_tool
except ImportError:
    try:
        # Try importing from alpha_agent_pool/agents.py directly to avoid package shadowing
        import importlib.util
        agents_path = alpha_agent_pool_path / "agents.py"
        if agents_path.exists():
            spec = importlib.util.spec_from_file_location("agents", agents_path)
            agents_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agents_module)
            Agent = agents_module.Agent
            function_tool = agents_module.function_tool
            
            # Check if Runner exists in local module, otherwise define a compatibility wrapper
            if hasattr(agents_module, 'Runner'):
                Runner = agents_module.Runner
            else:
                class Runner:
                    @staticmethod
                    def run_sync(agent, input_text, context=None, max_turns=10):
                        # Compatibility with local Agent.run method
                        if hasattr(agent, 'run'):
                             # Check if run accepts max_turns
                            import inspect
                            sig = inspect.signature(agent.run)
                            if 'max_turns' in sig.parameters:
                                return agent.run(input_text, context=context, max_turns=max_turns)
                            return agent.run(input_text, context=context)
                        return "Agent execution failed: No run method found"

        else:
            raise ImportError("agents.py not found")
    except (ImportError, Exception) as e:
        # Fallback if agents.py is not found or fails to import
        print(f"Warning: Failed to import Agent from agents.py: {e}")
        print("Using minimal local Agent implementation.")
        
        # Minimal Agent class without OpenAI dependency if possible, or mock it
        def function_tool(func, name=None, description=None):
            func.is_tool = True
            func.name = name or func.__name__
            func.description = description or func.__doc__ or "No description available"
            return func
        
    class Agent:
        def __init__(self, name="Agent", instructions="", model="gpt-5", tools=None):
            self.name = name
            self.instructions = instructions
            self.model = model
            self.tools = tools or []
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                self.client = None
                print("Warning: openai module not found in fallback Agent.")
            
        def run(self, user_request, context=None, max_turns=10):
            if not self.client:
                return f"Agent {self.name} received request: {user_request[:50]}... (OpenAI client not available)"
            
            # Basic implementation of agent loop for fallback
            messages = [{"role": "system", "content": self.instructions}, {"role": "user", "content": user_request}]
            
            # Use a simplified context string if context is provided
            if context:
                context_str = f"\nContext Data: {json.dumps({k: str(v)[:200] for k, v in context.items()}, default=str)}"
                messages[0]["content"] += context_str

            try:
                # Simple single-turn for fallback (can be expanded to loop)
                # For function calling, we would need to construct tool schemas manually
                # Here we just ask the model directly for now to ensure we get a response
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error in fallback Agent run: {e}"

        class Runner:
            @staticmethod
            def run_sync(agent, input_text, context=None, max_turns=10):
                return agent.run(input_text, context=context, max_turns=max_turns)

# Import Qlib utilities
try:
    qlib_path = alpha_agent_pool_path / "qlib"
    sys.path.insert(0, str(qlib_path))
    from utils import QlibConfig, DataProcessor
    from data_interfaces import FactorInput
except ImportError as e:
    print(f"Warning: Qlib modules not found: {e}. Some features may be limited.")
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


# ==============================
# Risk Metrics Calculation Tools
# ==============================

@function_tool
def calculate_volatility(
    ctx=None,
    returns: pd.Series = None,
    window: int = 20,
    annualized: bool = True
) -> Dict[str, Any]:
    """
    Calculate volatility (standard deviation of returns).
    
    Args:
        ctx: Context dictionary (optional, may contain 'returns')
        returns: Series of returns (optional, can be extracted from ctx)
        window: Rolling window size for calculation
        annualized: Whether to annualize the volatility (multiply by sqrt(252))
        
    Returns:
        Dictionary with volatility metrics
    """
    try:
        # Extract returns from context if not provided
        if returns is None and ctx is not None:
            returns = ctx.get('returns')
        if returns is None:
            return {
                "status": "error",
                "message": "Returns data not provided. Please provide returns in context or as parameter."
            }
        rolling_vol = returns.rolling(window=window).std()
        
        if annualized:
            rolling_vol = rolling_vol * np.sqrt(252)
        
        current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0.0
        mean_vol = rolling_vol.mean()
        max_vol = rolling_vol.max()
        min_vol = rolling_vol.min()
        
        return {
            "status": "success",
            "volatility": {
                "current": float(current_vol) if not np.isnan(current_vol) else 0.0,
                "mean": float(mean_vol) if not np.isnan(mean_vol) else 0.0,
                "max": float(max_vol) if not np.isnan(max_vol) else 0.0,
                "min": float(min_vol) if not np.isnan(min_vol) else 0.0,
                "window": window
            },
            "volatility_series": rolling_vol.to_dict() if len(rolling_vol) > 0 else {}
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to calculate volatility: {str(e)}"
        }


@function_tool
def calculate_var(
    ctx=None,
    returns: pd.Series = None,
    confidence_level: float = 0.05,
    window: int = 252
) -> Dict[str, Any]:
    """
    Calculate Value at Risk (VaR) - the maximum loss expected at a given confidence level.
    
    Args:
        ctx: Context dictionary (optional, may contain 'returns')
        returns: Series of returns (optional, can be extracted from ctx)
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        window: Window size for calculation
        
    Returns:
        Dictionary with VaR metrics
    """
    try:
        # Extract returns from context if not provided
        if returns is None and ctx is not None:
            returns = ctx.get('returns')
        if returns is None:
            return {
                "status": "error",
                "message": "Returns data not provided. Please provide returns in context or as parameter."
            }
        if len(returns) < window:
            window = len(returns)
        
        recent_returns = returns.tail(window)
        var_value = np.percentile(recent_returns, confidence_level * 100)
        
        # Historical VaR
        historical_var = var_value
        
        # Parametric VaR (assuming normal distribution)
        mean_return = recent_returns.mean()
        std_return = recent_returns.std()
        parametric_var = mean_return + std_return * np.percentile(np.random.normal(0, 1, 10000), 0) if confidence_level == 0.05 else mean_return - 1.645 * std_return
        
        return {
            "status": "success",
            "var": {
                "historical_var": float(historical_var),
                "parametric_var": float(parametric_var) if not np.isnan(parametric_var) else float(historical_var),
                "confidence_level": confidence_level,
                "window": window
            },
            "message": f"VaR calculated at {confidence_level*100}% confidence level"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to calculate VaR: {str(e)}"
        }


@function_tool
def calculate_cvar(
    ctx=None,
    returns: pd.Series = None,
    confidence_level: float = 0.05,
    window: int = 252
) -> Dict[str, Any]:
    """
    Calculate Conditional Value at Risk (CVaR) - expected loss beyond VaR.
    
    Args:
        ctx: Context dictionary (optional, may contain 'returns')
        returns: Series of returns (optional, can be extracted from ctx)
        confidence_level: Confidence level (e.g., 0.05 for 95% CVaR)
        window: Window size for calculation
        
    Returns:
        Dictionary with CVaR metrics
    """
    try:
        # Extract returns from context if not provided
        if returns is None and ctx is not None:
            returns = ctx.get('returns')
        if returns is None:
            return {
                "status": "error",
                "message": "Returns data not provided. Please provide returns in context or as parameter."
            }
        if len(returns) < window:
            window = len(returns)
        
        recent_returns = returns.tail(window)
        var_value = np.percentile(recent_returns, confidence_level * 100)
        
        # CVaR is the mean of returns below VaR
        tail_returns = recent_returns[recent_returns <= var_value]
        cvar_value = tail_returns.mean() if len(tail_returns) > 0 else var_value
        
        return {
            "status": "success",
            "cvar": {
                "cvar_value": float(cvar_value) if not np.isnan(cvar_value) else 0.0,
                "var_value": float(var_value) if not np.isnan(var_value) else 0.0,
                "confidence_level": confidence_level,
                "window": window,
                "tail_observations": len(tail_returns)
            },
            "message": f"CVaR calculated at {confidence_level*100}% confidence level"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to calculate CVaR: {str(e)}"
        }


@function_tool
def calculate_max_drawdown(
    ctx=None,
    prices: pd.Series = None,
    window: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate Maximum Drawdown (MDD) - the largest peak-to-trough decline.
    
    Args:
        ctx: Context dictionary (optional, may contain 'data' with 'close' prices)
        prices: Series of prices (optional, can be extracted from ctx)
        window: Optional rolling window for calculation
        
    Returns:
        Dictionary with drawdown metrics
    """
    try:
        # Extract prices from context if not provided
        if prices is None and ctx is not None:
            data = ctx.get('data')
            if data is not None and 'close' in data.columns:
                if 'symbol' in data.columns:
                    # Use first symbol or aggregate
                    prices = data.groupby('date')['close'].first()
                else:
                    prices = data['close']
        if prices is None:
            return {
                "status": "error",
                "message": "Price data not provided. Please provide prices in context or as parameter."
            }
        if window is None:
            # Calculate overall max drawdown
            cumulative = (1 + prices.pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            max_dd_date = drawdown.idxmin() if hasattr(drawdown, 'idxmin') else None
        else:
            # Rolling max drawdown
            cumulative = (1 + prices.pct_change()).cumprod()
            rolling_max_dd = []
            for i in range(window, len(cumulative) + 1):
                window_data = cumulative.iloc[i-window:i]
                running_max = window_data.expanding().max()
                dd = (window_data - running_max) / running_max
                rolling_max_dd.append(dd.min())
            
            max_dd = min(rolling_max_dd) if rolling_max_dd else 0.0
            max_dd_date = None
        
        return {
            "status": "success",
            "max_drawdown": {
                "value": float(max_dd) if not np.isnan(max_dd) else 0.0,
                "percentage": float(max_dd * 100) if not np.isnan(max_dd) else 0.0,
                "date": str(max_dd_date) if max_dd_date else None,
                "window": window
            },
            "message": "Maximum drawdown calculated successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to calculate max drawdown: {str(e)}"
        }


@function_tool
def calculate_beta(
    ctx=None,
    asset_returns: pd.Series = None,
    market_returns: pd.Series = None,
    window: int = 60
) -> Dict[str, Any]:
    """
    Calculate Beta - measure of asset's sensitivity to market movements.
    
    Args:
        ctx: Context dictionary (optional, may contain 'returns' and 'market_returns')
        asset_returns: Series of asset returns (optional, can be extracted from ctx)
        market_returns: Series of market returns (benchmark, optional)
        window: Rolling window size for calculation
        
    Returns:
        Dictionary with beta metrics
    """
    try:
        # Extract returns from context if not provided
        if asset_returns is None and ctx is not None:
            asset_returns = ctx.get('returns')
        if asset_returns is None:
            return {
                "status": "error",
                "message": "Asset returns not provided. Please provide returns in context or as parameter."
            }
        # Align returns
        aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
        if len(aligned) < window:
            window = len(aligned)
        
        if len(aligned) < 10:
            return {
                "status": "error",
                "message": "Insufficient data for beta calculation"
            }
        
        # Calculate rolling beta
        betas = []
        for i in range(window, len(aligned) + 1):
            window_data = aligned.iloc[i-window:i]
            if len(window_data) < 10:
                continue
            
            cov = window_data.iloc[:, 0].cov(window_data.iloc[:, 1])
            market_var = window_data.iloc[:, 1].var()
            
            if market_var > 0:
                beta = cov / market_var
                betas.append(beta)
        
        current_beta = betas[-1] if betas else 0.0
        mean_beta = np.mean(betas) if betas else 0.0
        
        return {
            "status": "success",
            "beta": {
                "current": float(current_beta) if not np.isnan(current_beta) else 0.0,
                "mean": float(mean_beta) if not np.isnan(mean_beta) else 0.0,
                "window": window,
                "observations": len(betas)
            },
            "message": "Beta calculated successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to calculate beta: {str(e)}"
        }


@function_tool
def calculate_correlation_risk(
    ctx=None,
    returns_df: pd.DataFrame = None,
    window: int = 60
) -> Dict[str, Any]:
    """
    Calculate correlation risk - measure of how assets move together.
    
    Args:
        ctx: Context dictionary (optional, may contain 'data')
        returns_df: DataFrame with returns for multiple assets (optional, can be extracted from ctx)
        window: Rolling window size for calculation
        
    Returns:
        Dictionary with correlation metrics
    """
    try:
        # Extract returns from context if not provided
        if returns_df is None and ctx is not None:
            data = ctx.get('data')
            if data is not None and 'symbol' in data.columns:
                # Create returns DataFrame
                returns_df = data.pivot(index='date', columns='symbol', values='close').pct_change().dropna()
        if returns_df is None:
            return {
                "status": "error",
                "message": "Returns DataFrame not provided. Please provide data in context or as parameter."
            }
        if len(returns_df) < window:
            window = len(returns_df)
        
        # Calculate correlation matrix for recent window
        recent_returns = returns_df.tail(window)
        corr_matrix = recent_returns.corr()
        
        # Average correlation
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_correlation = corr_matrix.where(mask).stack().mean()
        
        # Maximum correlation
        max_correlation = corr_matrix.where(mask).stack().max()
        
        # Minimum correlation
        min_correlation = corr_matrix.where(mask).stack().min()
        
        return {
            "status": "success",
            "correlation_risk": {
                "average_correlation": float(avg_correlation) if not np.isnan(avg_correlation) else 0.0,
                "max_correlation": float(max_correlation) if not np.isnan(max_correlation) else 0.0,
                "min_correlation": float(min_correlation) if not np.isnan(min_correlation) else 0.0,
                "window": window,
                "n_assets": len(returns_df.columns)
            },
            "correlation_matrix": corr_matrix.to_dict(),
            "message": "Correlation risk calculated successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to calculate correlation risk: {str(e)}"
        }


@function_tool
def calculate_liquidity_risk(
    ctx=None,
    data: pd.DataFrame = None,
    volume_col: str = "volume",
    price_col: str = "close",
    window: int = 20
) -> Dict[str, Any]:
    """
    Calculate liquidity risk based on volume and price impact.
    
    Args:
        ctx: Context dictionary (optional, may contain 'data')
        data: DataFrame with volume and price data (optional, can be extracted from ctx)
        volume_col: Column name for volume
        price_col: Column name for price
        window: Rolling window size for calculation
        
    Returns:
        Dictionary with liquidity metrics
    """
    try:
        # Extract data from context if not provided
        if data is None and ctx is not None:
            data = ctx.get('data')
        if data is None:
            return {
                "status": "error",
                "message": "Data not provided. Please provide data in context or as parameter."
            }
        if volume_col not in data.columns or price_col not in data.columns:
            return {
                "status": "error",
                "message": f"Required columns not found: {volume_col}, {price_col}"
            }
        
        # Calculate average daily volume
        avg_volume = data[volume_col].rolling(window=window).mean()
        current_volume = data[volume_col].iloc[-1] if len(data) > 0 else 0.0
        
        # Volume ratio (current vs average)
        volume_ratio = current_volume / (avg_volume.iloc[-1] + 1e-10) if len(avg_volume) > 0 else 1.0
        
        # Price impact proxy (inverse of volume)
        price_impact = 1.0 / (avg_volume + 1e-10)
        
        # Liquidity score (higher is better)
        liquidity_score = volume_ratio
        
        return {
            "status": "success",
            "liquidity_risk": {
                "current_volume": float(current_volume),
                "average_volume": float(avg_volume.iloc[-1]) if len(avg_volume) > 0 else 0.0,
                "volume_ratio": float(volume_ratio) if not np.isnan(volume_ratio) else 1.0,
                "liquidity_score": float(liquidity_score) if not np.isnan(liquidity_score) else 1.0,
                "window": window
            },
            "message": "Liquidity risk calculated successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to calculate liquidity risk: {str(e)}"
        }


@function_tool
def generate_risk_signals(
    ctx=None,
    risk_metrics: Dict[str, Any] = None,
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Generate risk signals based on calculated risk metrics.
    
    Args:
        ctx: Context dictionary (optional, may contain 'risk_metrics')
        risk_metrics: Dictionary containing various risk metrics (optional, can be extracted from ctx)
        thresholds: Optional dictionary of risk thresholds
        
    Returns:
        Dictionary with risk signals
    """
    try:
        # Extract risk_metrics from context if not provided
        if risk_metrics is None and ctx is not None:
            risk_metrics = ctx.get('risk_metrics', {})
        if risk_metrics is None or len(risk_metrics) == 0:
            return {
                "status": "error",
                "message": "Risk metrics not provided. Please calculate risk metrics first."
            }
        if thresholds is None:
            thresholds = {
                "volatility_high": 0.30,  # 30% annualized volatility
                "volatility_low": 0.10,   # 10% annualized volatility
                "var_severe": -0.05,      # 5% daily loss
                "var_moderate": -0.02,    # 2% daily loss
                "max_dd_severe": -0.20,   # 20% drawdown
                "max_dd_moderate": -0.10,  # 10% drawdown
                "beta_high": 1.5,
                "beta_low": 0.5,
                "correlation_high": 0.7,
                "liquidity_low": 0.5
            }
        
        signals = {}
        risk_level = "LOW"
        risk_score = 0.0
        
        # Volatility signal
        if "volatility" in risk_metrics:
            vol = risk_metrics["volatility"].get("current", 0.0)
            if vol > thresholds["volatility_high"]:
                signals["volatility"] = "HIGH"
                risk_score += 0.3
            elif vol < thresholds["volatility_low"]:
                signals["volatility"] = "LOW"
            else:
                signals["volatility"] = "MODERATE"
                risk_score += 0.15
        
        # VaR signal
        if "var" in risk_metrics:
            var = risk_metrics["var"].get("historical_var", 0.0)
            if var < thresholds["var_severe"]:
                signals["var"] = "SEVERE"
                risk_score += 0.3
            elif var < thresholds["var_moderate"]:
                signals["var"] = "MODERATE"
                risk_score += 0.15
            else:
                signals["var"] = "LOW"
        
        # Max Drawdown signal
        if "max_drawdown" in risk_metrics:
            mdd = risk_metrics["max_drawdown"].get("value", 0.0)
            if mdd < thresholds["max_dd_severe"]:
                signals["max_drawdown"] = "SEVERE"
                risk_score += 0.2
            elif mdd < thresholds["max_dd_moderate"]:
                signals["max_drawdown"] = "MODERATE"
                risk_score += 0.1
            else:
                signals["max_drawdown"] = "LOW"
        
        # Beta signal
        if "beta" in risk_metrics:
            beta = risk_metrics["beta"].get("current", 1.0)
            if beta > thresholds["beta_high"]:
                signals["beta"] = "HIGH_MARKET_SENSITIVITY"
                risk_score += 0.1
            elif beta < thresholds["beta_low"]:
                signals["beta"] = "LOW_MARKET_SENSITIVITY"
            else:
                signals["beta"] = "MODERATE"
        
        # Correlation signal
        if "correlation_risk" in risk_metrics:
            avg_corr = risk_metrics["correlation_risk"].get("average_correlation", 0.0)
            if avg_corr > thresholds["correlation_high"]:
                signals["correlation"] = "HIGH_DIVERSIFICATION_RISK"
                risk_score += 0.1
            else:
                signals["correlation"] = "ACCEPTABLE"
        
        # Determine overall risk level
        if risk_score >= 0.6:
            risk_level = "HIGH"
        elif risk_score >= 0.3:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        return {
            "status": "success",
            "risk_signals": signals,
            "overall_risk_level": risk_level,
            "risk_score": float(risk_score),
            "message": f"Risk signals generated: {risk_level} risk level"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to generate risk signals: {str(e)}"
        }


# ==============================
# Risk Signal Agent
# ==============================

class RiskSignalAgent:
    """
    Risk Signal Agent using OpenAI Agent SDK
    
    This agent combines Qlib data access, risk metrics calculation,
    and LLM reasoning to generate risk trading signals.
    """
    
    def __init__(
        self,
        name: str = "RiskSignalAgent",
        model: str = "gpt-5",
        qlib_config: Optional[QlibConfig] = None
    ):
        """
        Initialize the Risk Signal Agent
        
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
            calculate_volatility,
            calculate_var,
            calculate_cvar,
            calculate_max_drawdown,
            calculate_beta,
            calculate_correlation_risk,
            calculate_liquidity_risk,
            generate_risk_signals
        ]
        
        # Initialize OpenAI Agent
        instructions = """
        You are a Risk Signal Agent specialized in financial risk analysis.
        Your role is to:
        1. Calculate various risk metrics (volatility, VaR, CVaR, max drawdown, beta, correlation, liquidity)
        2. Analyze risk patterns and trends
        3. Generate risk signals based on calculated metrics
        4. Provide risk assessment and recommendations
        
        When working with market data:
        - Always validate data quality before processing
        - Use appropriate time windows for risk calculations
        - Consider multiple risk dimensions (market risk, credit risk, liquidity risk)
        - Provide clear risk level assessments (LOW, MODERATE, HIGH)
        - Give actionable recommendations based on risk signals
        
        Provide clear explanations of your risk analysis and reasoning.
        """
        
        self.agent = Agent(
            name=name,
            instructions=instructions,
            model=model,
            tools=self.tools
        )
    
    def run(self, user_request: str, context: Optional[Dict[str, Any]] = None, max_turns: int = 10) -> str:
        """
        Execute the agent with a user request using Runner
        
        Args:
            user_request: User's request or query
            context: Optional context dictionary (e.g., market data)
            max_turns: Maximum number of turns for the agent conversation
            
        Returns:
            Agent response
        """
        # Use Runner.run_sync to execute the agent
        result = Runner.run_sync(self.agent, user_request, context=context, max_turns=max_turns)
        
        # Handle different return types (string or Result object)
        if hasattr(result, 'final_output'):
            return result.final_output
        return str(result)
    
    def generate_risk_signals_from_data(
        self,
        data: pd.DataFrame,
        market_returns: Optional[pd.Series] = None,
        risk_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Complete pipeline: data -> risk metrics -> risk signals
        
        Args:
            data: Market data DataFrame with date, symbol, and OHLCV columns
            market_returns: Optional market returns for beta calculation
            risk_metrics: List of risk metrics to calculate (default: all)
            
        Returns:
            Dictionary with risk signals and metadata
        """
        try:
            # Step 1: Normalize data format
            if isinstance(data.index, pd.MultiIndex):
                data = data.reset_index()
                if 'Date' in data.columns:
                    data = data.rename(columns={'Date': 'date'})
                if 'instrument' in data.columns:
                    data = data.rename(columns={'instrument': 'symbol'})
            elif 'date' not in data.columns and 'Date' in data.columns:
                data = data.rename(columns={'Date': 'date'})
            if 'symbol' not in data.columns and 'instrument' in data.columns:
                data = data.rename(columns={'instrument': 'symbol'})
            
            # Rename price columns if needed
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
            
            # Ensure we have required columns
            if 'close' not in data.columns:
                raise ValueError("Data must have 'close' column")
            if 'date' not in data.columns:
                raise ValueError("Data must have 'date' column")
            
            # Process data
            processed_data = self.data_processor.add_returns(data)
            
            # Step 2: Calculate returns
            if 'symbol' in processed_data.columns:
                # Group by symbol and calculate returns
                processed_data = processed_data.sort_values(['symbol', 'date'])
                returns = processed_data.groupby('symbol')['close'].pct_change()
                prices = processed_data.groupby('symbol')['close']
            else:
                returns = processed_data['close'].pct_change()
                prices = processed_data['close']
            
            # Remove NaN values
            returns = returns.dropna()
            
            if len(returns) < 20:
                return {
                    "status": "error",
                    "message": "Insufficient data for risk calculation (need at least 20 observations)"
                }
            
            # Step 3: Calculate risk metrics
            risk_metrics_result = {}
            
            if risk_metrics is None:
                risk_metrics = ["volatility", "var", "cvar", "max_drawdown", "beta", "correlation", "liquidity"]
            
            # Volatility
            if "volatility" in risk_metrics:
                vol_result = calculate_volatility(ctx=None, returns=returns, window=20, annualized=True)
                if vol_result["status"] == "success":
                    risk_metrics_result["volatility"] = vol_result["volatility"]
            
            # VaR
            if "var" in risk_metrics:
                var_result = calculate_var(ctx=None, returns=returns, confidence_level=0.05, window=min(252, len(returns)))
                if var_result["status"] == "success":
                    risk_metrics_result["var"] = var_result["var"]
            
            # CVaR
            if "cvar" in risk_metrics:
                cvar_result = calculate_cvar(ctx=None, returns=returns, confidence_level=0.05, window=min(252, len(returns)))
                if cvar_result["status"] == "success":
                    risk_metrics_result["cvar"] = cvar_result["cvar"]
            
            # Max Drawdown
            if "max_drawdown" in risk_metrics:
                if 'symbol' in processed_data.columns:
                    # Calculate for each symbol and take average
                    mdd_results = []
                    for symbol in processed_data['symbol'].unique():
                        symbol_data = processed_data[processed_data['symbol'] == symbol]
                        if len(symbol_data) > 0:
                            symbol_prices = symbol_data['close']
                            mdd_result = calculate_max_drawdown(ctx=None, prices=symbol_prices)
                            if mdd_result["status"] == "success":
                                mdd_results.append(mdd_result["max_drawdown"]["value"])
                    if mdd_results:
                        avg_mdd = np.mean(mdd_results)
                        risk_metrics_result["max_drawdown"] = {"value": float(avg_mdd), "percentage": float(avg_mdd * 100)}
                else:
                    mdd_result = calculate_max_drawdown(ctx=None, prices=prices)
                    if mdd_result["status"] == "success":
                        risk_metrics_result["max_drawdown"] = mdd_result["max_drawdown"]
            
            # Beta (if market returns provided)
            if "beta" in risk_metrics and market_returns is not None:
                beta_result = calculate_beta(ctx=None, asset_returns=returns, market_returns=market_returns, window=min(60, len(returns)))
                if beta_result["status"] == "success":
                    risk_metrics_result["beta"] = beta_result["beta"]
            
            # Correlation (if multiple symbols)
            if "correlation" in risk_metrics and 'symbol' in processed_data.columns:
                symbols = processed_data['symbol'].unique()
                if len(symbols) > 1:
                    # Create returns DataFrame
                    returns_df = processed_data.pivot(index='date', columns='symbol', values='close').pct_change().dropna()
                    if len(returns_df) > 0:
                        corr_result = calculate_correlation_risk(ctx=None, returns_df=returns_df, window=min(60, len(returns_df)))
                        if corr_result["status"] == "success":
                            risk_metrics_result["correlation_risk"] = corr_result["correlation_risk"]
            
            # Liquidity
            if "liquidity" in risk_metrics and 'volume' in processed_data.columns:
                if 'symbol' in processed_data.columns:
                    # Calculate for each symbol and take average
                    liq_results = []
                    for symbol in processed_data['symbol'].unique():
                        symbol_data = processed_data[processed_data['symbol'] == symbol]
                        if len(symbol_data) > 0:
                            liq_result = calculate_liquidity_risk(ctx=None, data=symbol_data, volume_col='volume', price_col='close')
                            if liq_result["status"] == "success":
                                liq_results.append(liq_result["liquidity_risk"]["liquidity_score"])
                    if liq_results:
                        avg_liq = np.mean(liq_results)
                        risk_metrics_result["liquidity_risk"] = {"liquidity_score": float(avg_liq)}
                else:
                    liq_result = calculate_liquidity_risk(ctx=None, data=processed_data, volume_col='volume', price_col='close')
                    if liq_result["status"] == "success":
                        risk_metrics_result["liquidity_risk"] = liq_result["liquidity_risk"]
            
            # Step 4: Generate risk signals
            signal_result = generate_risk_signals(ctx=None, risk_metrics=risk_metrics_result)
            
            return {
                "status": "success",
                "risk_metrics": risk_metrics_result,
                "risk_signals": signal_result.get("risk_signals", {}),
                "overall_risk_level": signal_result.get("overall_risk_level", "UNKNOWN"),
                "risk_score": signal_result.get("risk_score", 0.0),
                "n_observations": len(returns),
                "date_range": {
                    "start": str(processed_data['date'].min()),
                    "end": str(processed_data['date'].max())
                }
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
    print("Risk Signal Agent - Example Usage")
    print("=" * 50)
    
    # Initialize agent
    agent = RiskSignalAgent(
        name="RiskSignalAgent",
        model="gpt-4o-mini"
    )
    
    # Example request
    request = """
    I want to analyze risk for a portfolio:
    1. Calculate volatility, VaR, and max drawdown
    2. Assess overall risk level
    3. Provide risk management recommendations
    
    Please explain the process and generate risk signals.
    """
    
    print("\nAgent Request:")
    print(request)
    print("\n" + "=" * 50)
    print("Agent Response:")
    print("=" * 50)
    
    print("\nNote: To use the agent, set OPENAI_API_KEY environment variable")
    print("Example: export OPENAI_API_KEY='your-api-key'")

