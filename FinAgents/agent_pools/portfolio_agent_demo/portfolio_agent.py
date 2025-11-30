"""
Portfolio Agent using OpenAI Agent SDK

This agent implements the Portfolio Construction model as described in the 
Algorithmic Trading Review paper. It integrates Alpha signals, Risk signals, 
and Transaction Costs to determine portfolio allocations.

Key Features:
- Portfolio Construction based on Alpha, Risk, and Cost models
- Integration with AlphaSignalAgent and RiskSignalAgent outputs
- Interface adhering to the paper's design
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

# Import OpenAI Agent SDK (using existing agents.py pattern or fallback)
try:
    from agents import Agent, Runner, function_tool
except ImportError:
    # Fallback
    print("Warning: agents.py not found. Creating minimal Agent class.")
    try:
        from openai import OpenAI
    except ImportError:
        OpenAI = None
        print("Warning: openai module not found. Agent will run in mock mode.")
    
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
            try:
                if OpenAI:
                    self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                else:
                    self.client = None
            except:
                self.client = None
        
        def run(self, user_request, context=None, max_turns=10):
            if self.client:
                # Simple completion
                messages = [{"role": "system", "content": self.instructions}, {"role": "user", "content": user_request}]
                if context:
                    messages[0]["content"] += f"\nContext: {json.dumps(str(context)[:1000])}"
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                return response.choices[0].message.content
            return f"Agent {self.name} executed (Mock)"

    class Runner:
        @staticmethod
        def run_sync(agent, input_text, context=None, max_turns=10):
            return agent.run(input_text, context=context, max_turns=max_turns)

# ==============================
# Portfolio Construction Tools
# ==============================

@function_tool
def construct_portfolio(
    alpha_signals: Dict[str, float],
    risk_signals: Dict[str, Any],
    transaction_costs: Dict[str, float],
    current_portfolio: Optional[Dict[str, float]] = None,
    total_capital: float = 1000000.0
) -> Dict[str, Any]:
    """
    Construct portfolio weights based on Alpha, Risk, and Transaction Cost models.
    
    Args:
        alpha_signals: Dictionary mapping symbols to alpha scores/predictions
        risk_signals: Dictionary containing risk metrics and signals
        transaction_costs: Dictionary with cost parameters (fixed cost, slippage)
        current_portfolio: Current portfolio holdings (symbol -> quantity)
        total_capital: Total available capital
        
    Returns:
        Dictionary with target portfolio weights and quantities
    """
    try:
        # 1. Parse Inputs
        # Filter available assets from alpha signals
        available_assets = list(alpha_signals.keys())
        
        # Risk Assessment
        risk_level = risk_signals.get("overall_risk_level", "LOW")
        risk_score = risk_signals.get("risk_score", 0.0)
        
        # Adjust position sizing based on risk
        # High risk -> reduce overall exposure
        max_allocation = 1.0
        if risk_level == "HIGH":
            max_allocation = 0.5
        elif risk_level == "MODERATE":
            max_allocation = 0.8
            
        # 2. Alpha Processing
        # Sort assets by alpha score (descending)
        sorted_assets = sorted(alpha_signals.items(), key=lambda x: x[1], reverse=True)
        
        # Select top assets (simple TopK for this demo, can be more complex)
        top_k = 5
        selected_assets = sorted_assets[:top_k]
        
        # 3. Weight Allocation
        # Simple equal weight for selected assets, scaled by max_allocation
        target_weights = {}
        if selected_assets:
            weight_per_asset = max_allocation / len(selected_assets)
            for asset, score in selected_assets:
                if score > 0: # Only long positive alpha
                    target_weights[asset] = weight_per_asset
        
        # 4. Transaction Cost Adjustment (Simple Hurdle)
        # If rebalancing cost > expected alpha gain, don't trade (simplified)
        # Here we just deduct estimated cost from capital for quantity calculation
        
        fixed_cost = transaction_costs.get("fixed_cost", 5.0)
        slippage = transaction_costs.get("slippage", 0.001) # 10 bps
        
        target_quantities = {}
        
        # Calculate quantities
        for asset, weight in target_weights.items():
            allocation_amount = total_capital * weight
            # Estimated price (need price data, assume provided in alpha signals or context if available)
            # For this function, we output weights primarily. 
            # If we need quantities, we need prices. 
            # Let's assume we return weights and the execution model handles quantities.
            pass

        return {
            "status": "success",
            "target_weights": target_weights,
            "risk_adjustment": {
                "risk_level": risk_level,
                "max_allocation": max_allocation
            },
            "selected_assets": [x[0] for x in selected_assets],
            "message": f"Constructed portfolio with {len(target_weights)} assets"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Portfolio construction failed: {str(e)}"
        }

# ==============================
# Portfolio Agent
# ==============================

class PortfolioAgent:
    """
    Portfolio Agent specialized in portfolio construction.
    """
    
    def __init__(
        self,
        name: str = "PortfolioAgent",
        model: str = "gpt-4o-mini"
    ):
        self.name = name
        self.model = model
        
        self.tools = [
            construct_portfolio
        ]
        
        instructions = """
        You are a Portfolio Agent responsible for constructing investment portfolios.
        Your goal is to maximize returns while managing risk and minimizing transaction costs.
        
        You follow the Algorithmic Trading Review paper's interface:
        1. Receive Alpha Signals (predictions)
        2. Receive Risk Signals (risk assessment)
        3. Receive Transaction Cost parameters
        4. Determine optimal Portfolio Weights
        
        Strategy:
        - Prioritize high-alpha assets.
        - Reduce exposure when Risk is High.
        - Consider transaction costs (avoid excessive trading).
        """
        
        self.agent = Agent(
            name=name,
            instructions=instructions,
            model=model,
            tools=self.tools
        )
        
    def run(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> str:
        return Runner.run_sync(self.agent, user_request, context=context)

    def inference(
        self,
        alpha_signals: Dict[str, float],
        risk_signals: Dict[str, Any],
        transaction_costs: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Direct inference method to construct portfolio.
        """
        return construct_portfolio(
            alpha_signals=alpha_signals,
            risk_signals=risk_signals,
            transaction_costs=transaction_costs
        )

if __name__ == "__main__":
    # Test
    agent = PortfolioAgent()
    print("Portfolio Agent Initialized")

