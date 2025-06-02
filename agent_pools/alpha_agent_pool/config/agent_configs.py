from typing import Dict, Any, List
from ..schema.agent_config import AlphaAgentConfig, AgentType, DataSource, SignalRule, RiskParameter

# Define common parameters for all agents
COMMON_PARAMETERS = {
    "min_confidence": 0.7,
    "max_position_size": 0.1,
    "max_drawdown": 0.05
}

# Define common data sources
COMMON_DATA_SOURCES = [
    DataSource(
        name="market_data",
        type="market",
        description="Real-time market data including price, volume, and order book"
    ),
    DataSource(
        name="fundamental_data",
        type="fundamental",
        description="Company financial statements and fundamental metrics"
    )
]

# Technical Analysis Agent Configuration
TECHNICAL_AGENT_CONFIG = AlphaAgentConfig(
    agent_id="technical_agent",
    agent_type=AgentType.TECHNICAL,
    description="Technical analysis based trading agent using various technical indicators",
    data_sources=COMMON_DATA_SOURCES + [
        DataSource(
            name="technical_indicators",
            type="technical",
            description="Various technical indicators including RSI, MACD, Moving Averages"
        )
    ],
    parameters={
        **COMMON_PARAMETERS,
        "indicator_timeframes": ["1h", "4h", "1d"],
        "signal_threshold": 0.8
    },
    signal_rules=[
        SignalRule(
            name="trend_following",
            description="Follow trend based on moving averages",
            parameters={"ma_period": 20, "ma_type": "EMA"}
        ),
        SignalRule(
            name="momentum",
            description="RSI based momentum signals",
            parameters={"rsi_period": 14, "overbought": 70, "oversold": 30}
        )
    ],
    risk_parameters=[
        RiskParameter(
            name="position_sizing",
            description="Dynamic position sizing based on volatility",
            parameters={"vol_lookback": 20, "max_risk_per_trade": 0.02}
        )
    ]
)

# Event-Driven Agent Configuration
EVENT_AGENT_CONFIG = AlphaAgentConfig(
    agent_id="event_agent",
    agent_type=AgentType.EVENT_DRIVEN,
    description="Event-driven trading agent that analyzes market events and news",
    data_sources=COMMON_DATA_SOURCES + [
        DataSource(
            name="news_data",
            type="news",
            description="Real-time news and event data from various sources"
        ),
        DataSource(
            name="sentiment_data",
            type="sentiment",
            description="Market sentiment indicators and social media analysis"
        )
    ],
    parameters={
        **COMMON_PARAMETERS,
        "event_impact_threshold": 0.6,
        "sentiment_weight": 0.3
    },
    signal_rules=[
        SignalRule(
            name="news_impact",
            description="Analyze news impact on price movement",
            parameters={"impact_threshold": 0.7, "time_window": "1h"}
        ),
        SignalRule(
            name="sentiment_analysis",
            description="Trading signals based on market sentiment",
            parameters={"sentiment_threshold": 0.6, "min_volume": 1000000}
        )
    ],
    risk_parameters=[
        RiskParameter(
            name="event_risk",
            description="Risk management for event-driven trades",
            parameters={"max_event_exposure": 0.05, "stop_loss": 0.03}
        )
    ]
)

# ML-based Agent Configuration
ML_AGENT_CONFIG = AlphaAgentConfig(
    agent_id="ml_agent",
    agent_type=AgentType.ML_BASED,
    description="Machine learning based trading agent using predictive models",
    data_sources=COMMON_DATA_SOURCES + [
        DataSource(
            name="ml_features",
            type="ml",
            description="Engineered features for machine learning models"
        ),
        DataSource(
            name="model_predictions",
            type="prediction",
            description="Real-time model predictions and confidence scores"
        )
    ],
    parameters={
        **COMMON_PARAMETERS,
        "model_confidence_threshold": 0.8,
        "prediction_horizon": "1d"
    },
    signal_rules=[
        SignalRule(
            name="model_prediction",
            description="Trading signals based on model predictions",
            parameters={"min_confidence": 0.8, "prediction_threshold": 0.6}
        ),
        SignalRule(
            name="ensemble_validation",
            description="Validate signals across multiple models",
            parameters={"min_models_agree": 2, "consensus_threshold": 0.7}
        )
    ],
    risk_parameters=[
        RiskParameter(
            name="ml_risk",
            description="Risk management for ML-based predictions",
            parameters={"max_model_exposure": 0.1, "prediction_stop_loss": 0.04}
        )
    ]
)

# Dictionary of all agent configurations
AGENT_CONFIGS: Dict[str, AlphaAgentConfig] = {
    "technical_agent": TECHNICAL_AGENT_CONFIG,
    "event_agent": EVENT_AGENT_CONFIG,
    "ml_agent": ML_AGENT_CONFIG
} 