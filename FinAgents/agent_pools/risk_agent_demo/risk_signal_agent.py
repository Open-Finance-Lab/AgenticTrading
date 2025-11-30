"""
Risk Signal Agent using OpenAI Agent SDK

This agent integrates Qlib data access, risk metrics calculation, and LLM reasoning
to generate risk signals. Supports ReAct workflow.
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
sys.path.append(str(alpha_agent_pool_path))

# Import OpenAI Agents SDK
import nest_asyncio
nest_asyncio.apply()
from agents import Agent, Runner, function_tool, RunContextWrapper

# Import Qlib utilities
try:
    qlib_path = alpha_agent_pool_path / "qlib_local"
    if not qlib_path.exists():
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
# Internal Logic
# ==============================

def _calculate_volatility(returns: pd.Series, window: int = 20) -> Dict[str, Any]:
    try:
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0.0
        return {"status": "success", "volatility": {"current": float(current_vol)}}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _calculate_var(returns: pd.Series, confidence: float = 0.05, window: int = 252) -> Dict[str, Any]:
    try:
        recent = returns.tail(window)
        var_val = np.percentile(recent, confidence * 100)
        return {"status": "success", "var": {"historical_var": float(var_val)}}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _generate_risk_signals(risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
    # Simplified logic for risk scoring
    score = 0.0
    signals = {}
    
    if 'volatility' in risk_metrics:
        vol = risk_metrics['volatility'].get('current', 0.0)
        if vol > 0.3: 
            signals['volatility'] = 'HIGH'
            score += 0.5
        else:
            signals['volatility'] = 'LOW'
    
    if 'var' in risk_metrics:
        var = risk_metrics['var'].get('historical_var', 0.0)
        if var < -0.03:
            signals['var'] = 'HIGH'
            score += 0.5
        else:
            signals['var'] = 'LOW'
            
    level = "HIGH" if score >= 0.5 else "LOW"
    return {"status": "success", "risk_signals": signals, "overall_risk_level": level, "risk_score": score}

def _run_risk_pipeline_impl(data: pd.DataFrame, market_returns: Any, metrics: List[str], processor: Any) -> Dict[str, Any]:
    # Full pipeline implementation
    try:
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()
        # Normalize columns
        col_map = {c: c.lower() for c in data.columns}
        data = data.rename(columns=col_map)
        if 'instrument' in data.columns: data = data.rename(columns={'instrument': 'symbol'})
        if 'datetime' in data.columns: data = data.rename(columns={'datetime': 'date'})
        
        if 'close' not in data.columns: return {"status": "error", "message": "No close price"}
        
        # Calculate returns (portfolio level or aggregate?)
        # Risk agent usually assesses portfolio components or market context
        # For simplicity, calculate average return of the universe provided
        if 'symbol' in data.columns:
            returns = data.groupby('date')['close'].mean().pct_change().dropna()
        else:
            returns = data['close'].pct_change().dropna()
            
        risk_res = {}
        if 'volatility' in metrics:
            res = _calculate_volatility(returns)
            if res['status'] == 'success': risk_res['volatility'] = res['volatility']
            
        if 'var' in metrics:
            res = _calculate_var(returns)
            if res['status'] == 'success': risk_res['var'] = res['var']
            
        sig_res = _generate_risk_signals(risk_res)
        return {
            "status": "success",
            "risk_metrics": risk_res,
            "risk_signals": sig_res.get('risk_signals'),
            "overall_risk_level": sig_res.get('overall_risk_level'),
            "risk_score": sig_res.get('risk_score')
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==============================
# Tools
# ==============================

@function_tool
def run_risk_pipeline(ctx: RunContextWrapper[Any]) -> str:
    """Execute standard risk pipeline."""
    print("DEBUG: 🛡️ run_risk_pipeline (Macro) INVOKED")
    try:
        data = ctx.context.get('data')
        if data is None: return "Error: No data."
        metrics = ctx.context.get('risk_metrics', ['volatility', 'var'])
        result = _run_risk_pipeline_impl(data, None, metrics, None)
        ctx.context['result'] = result
        return f"Risk analysis complete. Level: {result.get('overall_risk_level')}"
    except Exception as e:
        return f"Error: {e}"

@function_tool
def calculate_volatility_tool(ctx: RunContextWrapper[Any], window: int = 20) -> str:
    """Calculate volatility."""
    print("DEBUG: 🛡️ calculate_volatility_tool INVOKED")
    try:
        data = ctx.context.get('data')
        if data is None: return "Error: No data."
        # Calculate returns on the fly
        # Simplified: assume 'close' column
        if isinstance(data.index, pd.MultiIndex): data = data.reset_index()
        col_map = {c: c.lower() for c in data.columns}
        data = data.rename(columns=col_map)
        if 'close' in data.columns:
            returns = data.groupby('date')['close'].mean().pct_change().dropna() if 'symbol' in data.columns else data['close'].pct_change().dropna()
            res = _calculate_volatility(returns, window)
            if res['status'] == 'success':
                if 'risk_metrics' not in ctx.context: ctx.context['risk_metrics'] = {}
                ctx.context['risk_metrics']['volatility'] = res['volatility']
                return f"Volatility calculated: {res['volatility']}"
        return "Error: Could not calculate."
    except Exception as e:
        return f"Error: {e}"

@function_tool
def submit_risk_assessment_tool(ctx: RunContextWrapper[Any]) -> str:
    """Submit final risk assessment."""
    print("DEBUG: 🛡️ submit_risk_assessment_tool INVOKED")
    try:
        metrics = ctx.context.get('risk_metrics', {})
        res = _generate_risk_signals(metrics)
        ctx.context['result'] = res
        return "Risk assessment submitted."
    except Exception as e:
        return f"Error: {e}"

# ==============================
# Risk Agent
# ==============================

class RiskSignalAgent:
    def __init__(
        self,
        name: str = "RiskSignalAgent",
        model: str = "gpt-4o",
        qlib_config: Optional[QlibConfig] = None
    ):
        self.name = name
        self.model = model
        self.qlib_config = qlib_config or QlibConfig()
        self.data_processor = DataProcessor(self.qlib_config)
        
        self.tools = [
            run_risk_pipeline,
            calculate_volatility_tool,
            submit_risk_assessment_tool
        ]
        
        self.agent = Agent(
            name=name,
            instructions="""
            You are a Risk Agent.
            You have market data in context.
            
            Choose a path:
            1. FAST PATH: Use 'run_risk_pipeline'.
            2. CUSTOM PATH: Use 'calculate_volatility_tool' then 'submit_risk_assessment_tool'.
            """,
            model=model,
            tools=self.tools
        )
    
    def run(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> str:
        return self.agent.run(user_request, context=context, max_turns=10)
    
    def generate_risk_signals_from_data(
        self,
        data: pd.DataFrame,
        market_returns: Optional[pd.Series] = None,
        risk_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        context = {
            'data': data,
            'market_returns': market_returns,
            'risk_metrics': risk_metrics
        }
        
        print("DEBUG: 🛡️ Requesting Risk Agent LLM...")
        result = Runner.run_sync(self.agent, "Assess market risk.", context=context)
        
        if 'result' in context:
            return context['result']
        return {'status': 'error', 'message': 'No result'}

if __name__ == "__main__":
    print("Risk Agent Initialized")
