# FinAgent Orchestration Examples

This directory contains example workflows for the FinAgent system, demonstrating how to train, optimize, and rigorously test AI trading agents using a "World Model" approach.

## Core Workflows

### 1. Training & Prompt Optimization
**File:** `1_backtest_training.ipynb`

This notebook runs the "In-Sample" training loop.
- **Goal**: Optimize agent system prompts (Instructions) using a Meta-Agent.
- **Mechanism**: 
  - Runs annual backtests.
  - If performance (Sharpe Ratio) is below threshold, the Meta-Agent analyzes logs and rewrites the sub-agent's instructions.
  - Saves the best prompts to `optimized_prompts.json`.

### 2. Out-of-Sample Inference ("World Model")
**File:** `2_out_of_sample_inference.ipynb`

This notebook performs rigorous, institutional-grade evaluation.
- **Goal**: Test how agents perform in a strictly unseen environment.
- **Methodology**: **Rolling Weekly Inference**.
  - The system advances time week-by-week.
  - At each step $t$, the agent can ONLY access data from $t-Lookback$ to $t$.
  - The agent trains a model on this historical window and predicts for week $t+1$.
  - **Zero Data Leakage**: Future data is strictly hidden from the agent.
- **Result**: A realistic equity curve that reflects actual trading conditions, including transaction costs and model decay.

### 3. Batch Backtesting Script
**File:** `run_backtest_fixed.py`

A pure Python script version of the backtesting logic, useful for debugging or running long batch jobs without a Jupyter kernel.

---

## Key Features

### 🌍 Strict "World Model" Simulation
Unlike traditional backtests that might inadvertently leak future data (e.g., scaling on the whole dataset), this framework implements a strict **Time-Travel** constraint. The `Orchestrator` incrementally reveals data to agents, simulating the passage of real time.

### 🧠 Dynamic Agent Routing (ReAct)
Agents are not hardcoded scripts. They use the `openai-agents-sdk` to dynamically:
- Choose which tools to use (e.g., `calculate_rsi` vs `calculate_macd`).
- Interpret market regimes (Bull/Bear).
- Route complex tasks to specialized sub-functions.

### 🛡️ Robust Risk Management
The system includes a specialized Risk Agent that:
- Calculates VaR (Value at Risk) and CVaR.
- Monitors volatility.
- Can reject or reduce positions proposed by the Alpha Agent if risk limits are breached.

### 📊 Real Market Data
- Uses **yfinance** to fetch real-time historical data for US equities.
- Supports multi-stock universes (e.g., Tech Giants: AAPL, MSFT, NVDA).

## Usage

1. **Install Dependencies**:
   Ensure you have `openai-agents-sdk`, `yfinance`, `pandas`, `numpy` installed.

2. **Run Training**:
   Open `1_backtest_training.ipynb` and run all cells to generate `optimized_prompts.json`.

3. **Run Inference**:
   Open `2_out_of_sample_inference.ipynb`.
   **Note**: Use `orchestrator.run_inference_rolling_week(...)` for the rigorous rolling-window mode.

4. **Analyze Results**:
   The notebooks will generate:
   - Cumulative Return Plots
   - Drawdown Analysis
   - Monthly Heatmaps
   - Trade Logs

