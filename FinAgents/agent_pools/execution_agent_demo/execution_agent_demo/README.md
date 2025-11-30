# Execution Agent Demo

This directory contains a demonstration of an Execution Agent that connects to Alpaca Paper Trading. It is designed to work within the FinAgent ecosystem, receiving portfolio target weights and executing the necessary trades.

## Features

- **Execution Agent**: 
    - Built with `openai-agents-python`.
    - Connects to Alpaca Trading API (Paper/Live).
    - Supports checking account status, positions, and executing batch orders.
    - Graceful fallback to "Mock Mode" if API keys are missing.
- **Live Trading Orchestration (`run_live_trading.py`)**:
    - Simulates a live trading loop.
    - Fetches market data (Alpaca or Mock).
    - Generates Alpha/Risk signals (Mock placeholders).
    - Calls `PortfolioAgent` (from `portfolio_agent_demo`) to construct a portfolio.
    - Instructs `ExecutionAgent` to rebalance the portfolio.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   (Ensure you also have dependencies for `portfolio_agent_demo` if using the full pipeline).

2. **Environment Variables**:
   Set your Alpaca and OpenAI keys. If skipped, the agent runs in Mock Mode.
   ```bash
   export OPENAI_API_KEY="sk-..."
   export ALPACA_API_KEY="PK..."
   export ALPACA_SECRET_KEY="sk..."
   export ALPACA_PAPER_TRADE="True"
   ```

3. **Run the Demo**:
   ```bash
   python run_live_trading.py
   ```

## File Structure

- `execution_agent.py`: Defines the `ExecutionAgent` class and Alpaca tools.
- `run_live_trading.py`: Main script running the data -> signal -> execution loop.
- `requirements.txt`: Python dependencies.

## Integration

The agent expects instructions in natural language but is optimized to handle "target portfolio weights" by checking current positions and calculating the necessary trades.

