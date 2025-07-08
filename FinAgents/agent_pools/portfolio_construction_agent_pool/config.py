"""
Portfolio Construction Agent Pool Configuration

This configuration file defines optimization models, constraints,
and integration settings for the Portfolio Construction Agent Pool.

Author: Jifeng Li
Created: 2025-06-30
License: openMDW
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from .memory_bridge import OptimizationType


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization models"""
    optimization_type: OptimizationType
    enabled: bool = True
    parameters: Dict[str, Any] = None
    constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.constraints is None:
            self.constraints = {}


# Default optimization configurations
DEFAULT_OPTIMIZATION_CONFIGS = {
    "mean_variance": OptimizationConfig(
        optimization_type=OptimizationType.MEAN_VARIANCE,
        enabled=True,
        parameters={
            "risk_aversion": 1.0,
            "min_expected_return": 0.05,
            "max_volatility": 0.25
        },
        constraints={
            "max_weight": 0.10,
            "min_weight": 0.01,
            "max_sector_weight": 0.30,
            "max_single_country": 0.40
        }
    ),
    
    "black_litterman": OptimizationConfig(
        optimization_type=OptimizationType.BLACK_LITTERMAN,
        enabled=True,
        parameters={
            "tau": 0.025,
            "risk_aversion": 3.0,
            "confidence_level": 0.95
        },
        constraints={
            "max_weight": 0.15,
            "min_weight": 0.005,
            "view_uncertainty": 0.10
        }
    ),
    
    "risk_parity": OptimizationConfig(
        optimization_type=OptimizationType.RISK_PARITY,
        enabled=True,
        parameters={
            "risk_budget": "equal",
            "rebalancing_frequency": "monthly"
        },
        constraints={
            "max_weight": 0.20,
            "min_weight": 0.01,
            "max_leverage": 1.0
        }
    ),
    
    "factor_based": OptimizationConfig(
        optimization_type=OptimizationType.FACTOR_BASED,
        enabled=True,
        parameters={
            "factor_model": "fama_french_5",
            "alpha_decay": 0.90,
            "factor_exposure_limits": True
        },
        constraints={
            "max_factor_exposure": 2.0,
            "max_weight": 0.08,
            "min_weight": 0.005
        }
    )
}

# Portfolio construction models configuration
PORTFOLIO_CONSTRUCTION_MODELS = {
    "rule_based_models": {
        "equal_weight": {
            "description": "Equal position weighting across all assets",
            "enabled": True,
            "constraints": {
                "max_weight": 1.0,  # No individual weight limits
                "min_weight": 0.0
            }
        },
        
        "equal_risk_rating": {
            "description": "Equal risk rating weighting based on risk scores",
            "enabled": True,
            "parameters": {
                "risk_adjustment": True,
                "volatility_scaling": True
            }
        },
        
        "alpha_driven_weighting": {
            "description": "Alpha-driven weighting based on expected returns",
            "enabled": True,
            "parameters": {
                "alpha_threshold": 0.05,
                "concentration_limit": 0.30
            }
        },
        
        "decision_tree_weighting": {
            "description": "Decision tree-based asset allocation",
            "enabled": True,
            "parameters": {
                "max_depth": 5,
                "min_samples_leaf": 10
            }
        }
    },
    
    "optimization_models": {
        "unconstrained_optimization": {
            "description": "Unconstrained portfolio optimization",
            "enabled": False,  # Typically disabled for practical applications
            "parameters": {
                "allow_short_selling": False,
                "leverage_limit": 1.0
            }
        },
        
        "constrained_optimization": {
            "description": "Constrained portfolio optimization with practical limits",
            "enabled": True,
            "constraints": {
                "max_weight": 0.10,
                "min_weight": 0.01,
                "sector_limits": {
                    "technology": 0.35,
                    "financials": 0.25,
                    "healthcare": 0.20,
                    "industrials": 0.15,
                    "consumer": 0.20,
                    "energy": 0.10,
                    "utilities": 0.10,
                    "materials": 0.10
                },
                "country_limits": {
                    "US": 0.60,
                    "developed_markets": 0.30,
                    "emerging_markets": 0.15
                }
            }
        },
        
        "black_litterman_optimization": {
            "description": "Black-Litterman optimization with views integration",
            "enabled": True,
            "parameters": {
                "tau": 0.025,
                "risk_aversion": 3.0,
                "view_confidence": 0.25
            },
            "constraints": {
                "max_weight": 0.15,
                "min_weight": 0.005
            }
        }
    }
}

# External integration settings
EXTERNAL_INTEGRATION_CONFIG = {
    "alpha_agent_pool": {
        "namespace": "alpha_agent_pool",
        "event_types": ["SIGNAL_GENERATED", "STRATEGY_UPDATED"],
        "data_freshness_hours": 24,
        "minimum_confidence": 0.6,
        "signal_aggregation": "weighted_average"
    },
    
    "risk_agent_pool": {
        "namespace": "risk_agent_pool", 
        "event_types": ["ANALYSIS_COMPLETED", "RISK_ALERT"],
        "data_freshness_hours": 12,
        "risk_measures": ["var", "volatility", "correlation", "beta"],
        "confidence_level": 0.95
    },
    
    "transaction_cost_agent_pool": {
        "namespace": "transaction_cost_agent_pool",
        "event_types": ["COST_ANALYSIS_COMPLETED", "MARKET_IMPACT_UPDATED"],
        "data_freshness_hours": 6,
        "cost_components": ["bid_ask_spread", "market_impact", "commission"],
        "cost_model": "linear"
    }
}

# Real-time monitoring configuration
MONITORING_CONFIG = {
    "portfolio_drift_threshold": 0.05,  # 5% drift triggers rebalancing consideration
    "performance_update_frequency": "daily",
    "risk_monitoring_frequency": "hourly",
    "alert_thresholds": {
        "max_drawdown": 0.10,
        "var_breach": 0.05,
        "tracking_error": 0.08,
        "concentration_risk": 0.25
    },
    "reporting": {
        "daily_performance": True,
        "weekly_attribution": True,
        "monthly_risk_report": True,
        "quarterly_rebalancing": True
    }
}

# Natural language processing configuration
NLP_CONFIG = {
    "supported_languages": ["english"],
    "optimization_keywords": {
        "maximize_return": ["maximize return", "highest return", "best performance"],
        "minimize_risk": ["minimize risk", "low risk", "conservative", "safe"],
        "maximize_sharpe": ["sharpe ratio", "risk-adjusted return", "efficient"],
        "equal_weight": ["equal weight", "equally weighted", "balanced"],
        "factor_based": ["factor", "style", "value", "growth", "momentum"]
    },
    "constraint_keywords": {
        "sector_limit": ["sector", "industry", "tech", "finance", "healthcare"],
        "concentration": ["concentration", "diversify", "spread"],
        "esg": ["esg", "sustainable", "green", "environmental"],
        "geography": ["region", "country", "us", "international", "emerging"]
    }
}

# Default portfolio templates
PORTFOLIO_TEMPLATES = {
    "conservative_balanced": {
        "name": "Conservative Balanced Portfolio",
        "description": "Low-risk balanced portfolio for capital preservation",
        "asset_allocation": {
            "equities": 0.40,
            "bonds": 0.50,
            "cash": 0.10
        },
        "risk_budget": 0.12,
        "rebalancing_frequency": "quarterly",
        "optimization_type": OptimizationType.MINIMUM_VARIANCE
    },
    
    "moderate_growth": {
        "name": "Moderate Growth Portfolio", 
        "description": "Balanced growth portfolio for long-term appreciation",
        "asset_allocation": {
            "equities": 0.70,
            "bonds": 0.25,
            "alternatives": 0.05
        },
        "risk_budget": 0.18,
        "rebalancing_frequency": "monthly",
        "optimization_type": OptimizationType.MEAN_VARIANCE
    },
    
    "aggressive_growth": {
        "name": "Aggressive Growth Portfolio",
        "description": "High-growth portfolio for maximum capital appreciation",
        "asset_allocation": {
            "equities": 0.90,
            "alternatives": 0.10
        },
        "risk_budget": 0.25,
        "rebalancing_frequency": "monthly",
        "optimization_type": OptimizationType.BLACK_LITTERMAN
    },
    
    "factor_momentum": {
        "name": "Factor Momentum Portfolio",
        "description": "Factor-based portfolio focusing on momentum signals",
        "factor_exposures": {
            "momentum": 1.5,
            "quality": 0.8,
            "low_volatility": -0.5
        },
        "risk_budget": 0.20,
        "rebalancing_frequency": "weekly",
        "optimization_type": OptimizationType.FACTOR_BASED
    }
}
