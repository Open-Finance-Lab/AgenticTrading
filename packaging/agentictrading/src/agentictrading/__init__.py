"""Agentic Trading Lab - lightweight Python client.

Agentic Trading Lab is an open-source experimental playground for LLM-powered
trading agents: prototype agents, run backtests and paper-trading simulations,
inspect reasoning and decision logs, and benchmark against market baselines.

This package provides a small standard-library client for the Agentic Trading
Lab REST API so you can drive backtests and read results from Python. The
install is dependency-free on macOS and Linux; on Windows it also pulls the
``tzdata`` data wheel, because ``zoneinfo`` has no system IANA database there.

Links:
  - Live demo: https://agentic-trading-lab.vercel.app/
  - Docs:      https://finagent-orchestration.readthedocs.io/
  - Source:    https://github.com/Allan-Feng/AgenticTrading
"""

from __future__ import annotations

from .atl_client import ATLClient
from .client import AgenticTradingClient, ApiError
from .exceptions import (
    ATLAPIError,
    ATLAuthenticationError,
    ATLConflictError,
    ATLRunFailedError,
    ATLTimeoutError,
    ATLValidationError,
)
from .models import (
    AgentVersion,
    Decision,
    ExecutionResult,
    Observation,
    Order,
    Run,
    RunResult,
    RunStatus,
    Step,
)
from .runner import AgentRunner, TradingAgentProtocol

__all__ = [
    # Protocol v1 SDK
    "ATLClient",
    "AgentRunner",
    "TradingAgentProtocol",
    "AgentVersion",
    "Run",
    "RunStatus",
    "Step",
    "Observation",
    "Decision",
    "Order",
    "ExecutionResult",
    "RunResult",
    # Exceptions
    "ATLAPIError",
    "ATLAuthenticationError",
    "ATLValidationError",
    "ATLConflictError",
    "ATLTimeoutError",
    "ATLRunFailedError",
    # Legacy client (backtest workflow)
    "AgenticTradingClient",
    "ApiError",
    # Metadata
    "__version__",
    "info",
]

__version__ = "0.2.0"

LIVE_DEMO_URL = "https://agentic-trading-lab.vercel.app/"
DOCS_URL = "https://finagent-orchestration.readthedocs.io/"
SOURCE_URL = "https://github.com/Allan-Feng/AgenticTrading"


def info() -> dict:
    """Return basic package/project metadata as a dict."""
    return {
        "name": "agentictrading",
        "version": __version__,
        "summary": "Lightweight Python client for the Agentic Trading Lab REST API.",
        "live_demo": LIVE_DEMO_URL,
        "docs": DOCS_URL,
        "source": SOURCE_URL,
    }
