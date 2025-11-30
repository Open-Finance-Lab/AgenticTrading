"""
Portfolio Agent using OpenAI Agent SDK

This agent implements the Portfolio Construction model.
Supports ReAct workflow.
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


# ==============================
# Internal Logic
# ==============================

def _construct_portfolio_impl(
    alpha_signals: Any,
    risk_signals: Any,
    transaction_costs: Any,
    current_portfolio: Any = None,
    total_capital: float = 1000000.0
) -> Dict[str, Any]:
    try:
        # 1. Parse Inputs
        if isinstance(alpha_signals, dict):
            available_assets = list(alpha_signals.keys())
        else:
            available_assets = []
        
        # Risk Assessment
        risk_level = "LOW"
        if isinstance(risk_signals, dict):
            risk_level = risk_signals.get("overall_risk_level", "LOW")
        
        # Adjust position sizing based on risk
        max_allocation = 1.0
        if risk_level == "HIGH": max_allocation = 0.5
        elif risk_level == "MODERATE": max_allocation = 0.8
            
        # 2. Alpha Processing
        sorted_assets = []
        if isinstance(alpha_signals, dict):
            sorted_assets = sorted(alpha_signals.items(), key=lambda x: x[1], reverse=True)
        
        # Select top assets
        top_k = 5
        selected_assets = sorted_assets[:top_k]
        
        # 3. Weight Allocation
        target_weights = {}
        if selected_assets:
            weight_per_asset = max_allocation / len(selected_assets)
            for asset, score in selected_assets:
                if score > 0: # Only long positive alpha
                    target_weights[asset] = weight_per_asset
        
        return {
            "status": "success",
            "target_weights": target_weights,
            "risk_adjustment": {"risk_level": risk_level, "max_allocation": max_allocation},
            "selected_assets": [x[0] for x in selected_assets]
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==============================
# Tools
# ==============================

@function_tool
def run_portfolio_pipeline(ctx: RunContextWrapper[Any]) -> str:
    """Execute standard portfolio construction pipeline."""
    print("DEBUG: 💼 run_portfolio_pipeline (Macro) INVOKED")
    try:
        alpha_signals = ctx.context.get('alpha_signals')
        risk_signals = ctx.context.get('risk_signals')
        transaction_costs = ctx.context.get('transaction_costs')
        current_portfolio = ctx.context.get('current_portfolio')
        total_capital = ctx.context.get('total_capital', 1000000.0)
        
        result = _construct_portfolio_impl(
            alpha_signals, risk_signals, transaction_costs, current_portfolio, total_capital
        )
        
        ctx.context['result'] = result
        return f"Portfolio constructed. Assets: {len(result.get('target_weights', {}))}"
    except Exception as e:
        return f"Error: {e}"

@function_tool
def construct_portfolio_tool(ctx: RunContextWrapper[Any], max_allocation: float = 1.0) -> str:
    """
    Construct portfolio weights from signals in context.
    """
    print(f"DEBUG: 💼 construct_portfolio_tool INVOKED")
    try:
        alpha_signals = ctx.context.get('alpha_signals')
        if not alpha_signals: return "Error: No alpha signals."
        
        # Simple Equal Weight Logic for custom path
        sorted_assets = sorted(alpha_signals.items(), key=lambda x: x[1], reverse=True)
        top_k = 5
        selected = sorted_assets[:top_k]
        weights = {}
        if selected:
            w = max_allocation / len(selected)
            for asset, score in selected:
                if score > 0: weights[asset] = w
                
        ctx.context['target_weights'] = weights
        return f"Constructed weights for {len(weights)} assets."
    except Exception as e:
        return f"Error: {e}"

@function_tool
def submit_portfolio_tool(ctx: RunContextWrapper[Any]) -> str:
    """Submit final portfolio."""
    print("DEBUG: 💼 submit_portfolio_tool INVOKED")
    try:
        weights = ctx.context.get('target_weights')
        if weights is None: return "Error: No weights found."
        
        result = {
            "status": "success",
            "target_weights": weights,
            "message": "Custom portfolio submitted"
        }
        ctx.context['result'] = result
        return "Portfolio submitted."
    except Exception as e:
        return f"Error: {e}"

# ==============================
# Portfolio Agent
# ==============================

class PortfolioAgent:
    def __init__(
        self,
        name: str = "PortfolioAgent",
        model: str = "gpt-4o",
        mode: str = "backtest"
    ):
        self.name = name
        self.model = model
        self.mode = mode
        
        self.tools = [
            run_portfolio_pipeline,
            construct_portfolio_tool,
            submit_portfolio_tool
        ]
        
        self.agent = Agent(
            name=name,
            instructions=f"""
            You are a Portfolio Agent.
            Choose a path:
            1. FAST PATH: Use 'run_portfolio_pipeline'.
            2. CUSTOM PATH: Use 'construct_portfolio_tool' then 'submit_portfolio_tool'.
            Current Mode: {self.mode.upper()}
            """,
            model=model,
            tools=self.tools
        )
        
    def run(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> str:
        return Runner.run_sync(self.agent, user_request, context=context)

    def inference(
        self,
        alpha_signals: Dict[str, float],
        risk_signals: Dict[str, Any],
        transaction_costs: Dict[str, float],
        current_portfolio: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        context = {
            'alpha_signals': alpha_signals,
            'risk_signals': risk_signals,
            'transaction_costs': transaction_costs,
            'current_portfolio': current_portfolio,
            'total_capital': 1000000.0
        }
        
        print("DEBUG: 💼 Requesting Portfolio Agent LLM...")
        result = Runner.run_sync(self.agent, "Construct portfolio.", context=context)
        
        if 'result' in context:
            return context['result']
        return {'status': 'error', 'message': 'No result'}

if __name__ == "__main__":
    print("Portfolio Agent Initialized")
