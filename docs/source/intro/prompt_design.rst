==================
Prompt Design
==================

This section describes the prompt engineering framework for the Agentic Trading system.
The design follows a modular **Alpha → Risk → Portfolio → Backtest** pipeline, 
where each agent receives structured inputs and produces standardized JSON outputs.

Pipeline Overview
==================

The agentic trading workflow consists of four specialized agents:

.. list-table:: Agent Pipeline
   :widths: 20 25 30 25
   :header-rows: 1

   * - Agent
     - Input
     - Core Task
     - Output
   * - **AlphaAgent**
     - Historical OHLCV + alpha factors
     - Mine and quantify expected excess return
     - ``expected_excess_return``
   * - **RiskAgent**
     - Alpha output + covariance matrix
     - Evaluate risk, volatility, constraints
     - ``risk_score``, adjusted weights
   * - **PortfolioAgent**
     - Alpha + Risk signals
     - Optimize portfolio weights
     - Investment weights, allocation plan
   * - **BacktestAgent**
     - Portfolio weights + execution rules
     - Simulate historical trading
     - Return, volatility, Sharpe ratio

Design Principles
==================

1. **Modularity**: Each agent has a single responsibility and can be independently tested.
2. **Standardization**: All agents communicate via JSON with predefined schemas.
3. **Reproducibility**: Prompts explicitly define lookback windows, constraints, and objectives.
4. **Cross-Asset Generality**: The same structure applies to both stock and cryptocurrency tasks.


1. Alpha Agent Prompt
======================

**Objective**: Estimate the expected excess return of each asset based on mined alpha factors.

Prompt Template
-----------------

.. code-block:: text

   You are an Alpha Agent analyzing financial time series and factor data.
   Your goal is to estimate the expected excess return of each asset
   based on mined alpha factors.

   Input:
   - Historical OHLCV data and derived alpha factors (e.g., momentum, volatility, value)
   - Lookback window: 60 bars
   - Target horizon: 20 bars ahead

   Task:
   1. Compute factor-weighted signals.
   2. Predict expected excess return for each asset.
   3. Output JSON with {"ticker","expected_excess_return","confidence"}.

   Output Example:
   {
     "AAPL": {"expected_excess_return": 0.015, "confidence": 0.82},
     "MSFT": {"expected_excess_return": 0.009, "confidence": 0.78},
     "TSLA": {"expected_excess_return": -0.006, "confidence": 0.73}
   }

Key Parameters
---------------

- **Lookback Window**: 60 trading days (≈3 months)
- **Prediction Horizon**: 20 trading days (≈1 month)
- **Alpha Factors**: Momentum, mean reversion, volatility, volume, fundamental ratios
- **Output Schema**: ``{ticker: {expected_excess_return: float, confidence: float}}``

Implementation Details
-----------------------

The Alpha Agent computes a weighted combination of alpha factors:

.. math::

   \alpha_i = \sum_{k=1}^{K} w_k \cdot f_{i,k}

where :math:`f_{i,k}` is the k-th factor for asset i, and :math:`w_k` are learned weights.

The confidence score reflects:

- Factor consistency across lookback window
- Signal-to-noise ratio
- Historical predictive performance


2. Risk Agent Prompt
======================

**Objective**: Evaluate and adjust alpha signals based on risk exposures and constraints.

Prompt Template
-----------------

.. code-block:: text

   You are a Risk Agent responsible for evaluating and adjusting alpha signals.
   You receive expected excess returns and estimate the corresponding risk exposures.

   Input:
   - AlphaAgent output (expected_excess_return)
   - Historical covariance matrix Σ
   - Factor exposure matrix B
   - Risk constraints: target_vol = 0.15, max_leverage = 2.0, beta_neutral = true

   Task:
   1. Estimate volatility and systematic risk per asset.
   2. Compute risk-adjusted score (alpha / σ).
   3. Return adjusted risk signal and suggested scaling factor.

   Output Example:
   {
     "AAPL": {"risk_score": 0.011, "volatility": 0.14, "risk_scaler": 0.91},
     "MSFT": {"risk_score": 0.007, "volatility": 0.12, "risk_scaler": 0.86},
     "TSLA": {"risk_score": -0.004, "volatility": 0.21, "risk_scaler": 0.73}
   }

Risk Metrics
-------------

**Risk Score**: Risk-adjusted expected return

.. math::

   \text{risk\_score}_i = \frac{\alpha_i}{\sigma_i}

where :math:`\sigma_i` is the estimated volatility of asset i.

**Risk Scaler**: Adjustment factor to meet portfolio-level constraints

.. math::

   \text{scaler}_i = \min\left(1, \frac{\sigma_{\text{target}}}{\sigma_i}\right)

Constraints
------------

- **Target Volatility**: 15% annualized
- **Max Leverage**: 2.0 (gross exposure ≤ 2× capital)
- **Beta Neutral**: Net market beta ≈ 0
- **Sector Limits**: No sector > 30% gross exposure


3. Portfolio Agent Prompt
===========================

**Objective**: Combine alpha and risk signals into actionable portfolio weights.

Prompt Template
-----------------

.. code-block:: text

   You are a Portfolio Agent combining alpha and risk signals into actionable portfolio weights.

   Input:
   - AlphaAgent expected_excess_return
   - RiskAgent risk_score and volatility
   - Constraints: gross_leverage ≤ 2, position_limit ≤ 0.25, sector_neutral = true

   Task:
   1. Optimize portfolio weights w to maximize Sharpe ratio:
      maximize (wᵀα) / sqrt(wᵀΣw)
   2. Apply constraints and normalization.
   3. Output final allocation plan and turnover suggestion.

   Output Example:
   {
     "weights": {"AAPL": 0.20, "MSFT": 0.15, "TSLA": -0.05},
     "expected_portfolio_return": 0.012,
     "expected_portfolio_vol": 0.14,
     "target_sharpe": 0.85
   }

Optimization Objective
-----------------------

The Portfolio Agent solves a mean-variance optimization problem:

.. math::

   \max_w \frac{w^T \alpha}{\sqrt{w^T \Sigma w}}

subject to:

.. math::

   \begin{aligned}
   &\sum_i |w_i| \leq 2 \quad &\text{(gross leverage)} \\
   &|w_i| \leq 0.25 \quad &\text{(position limit)} \\
   &\sum_i w_i \cdot \beta_i = 0 \quad &\text{(beta neutral)} \\
   &w^T \Sigma w \leq \sigma_{\text{target}}^2 \quad &\text{(volatility target)}
   \end{aligned}

Turnover Control
-----------------

To reduce transaction costs, the agent applies a turnover penalty:

.. math::

   \text{turnover\_cost} = \lambda \sum_i |w_i^{\text{new}} - w_i^{\text{old}}|

where :math:`\lambda` is the transaction cost rate (default: 10 bps).


4. Backtest Agent Prompt
==========================

**Objective**: Simulate portfolio performance and report risk-adjusted metrics.

Prompt Template
-----------------

.. code-block:: text

   You are a Backtest Agent simulating the performance of a trading portfolio.

   Input:
   - PortfolioAgent weights and rebalance rules
   - Historical market data (prices, costs, slippage)
   - Backtest period: 2024-01-01 → 2025-01-01

   Task:
   1. Apply daily rebalancing based on given weights.
   2. Compute cumulative return, volatility, Sharpe ratio, and drawdown.
   3. Generate textual performance summary and JSON metrics.

   Output Example:
   {
     "cumulative_return": 0.238,
     "annual_volatility": 0.148,
     "sharpe_ratio": 1.61,
     "max_drawdown": 0.056,
     "summary": "The strategy achieved a 23.8% annual return with a Sharpe of 1.61."
   }

Performance Metrics
--------------------

**Cumulative Return**:

.. math::

   R_{\text{cum}} = \prod_{t=1}^{T} (1 + r_t) - 1

**Sharpe Ratio**:

.. math::

   \text{Sharpe} = \frac{\mathbb{E}[r_t - r_f]}{\sigma(r_t)}

**Maximum Drawdown**:

.. math::

   \text{MDD} = \max_{t} \left(\frac{\max_{s \leq t} V_s - V_t}{\max_{s \leq t} V_s}\right)

where :math:`V_t` is the portfolio value at time t.

Execution Model
----------------

- **Rebalance Frequency**: Daily at market close
- **Transaction Costs**: 10 bps per trade + slippage
- **Slippage Model**: :math:`\text{slippage} = 0.05\% \times \sqrt{\text{trade\_size}/\text{ADV}}`
- **Cash Management**: Uninvested cash earns risk-free rate


5. BTC Agent Prompt (Cryptocurrency)
======================================

**Objective**: Apply the same Alpha–Risk–Portfolio–Backtest framework to minute-level BTC-USDT trading.

Prompt Template
-----------------

.. code-block:: text

   You are a Crypto Trading Agent for BTC-USDT using minute-level data.
   You follow the same Alpha → Risk → Portfolio → Backtest structure.

   Input:
   - Price, volume, EMA, RSI, MACD
   - Funding rate, open interest, volatility index
   - Position info: qty, leverage, unrealized_pnl

   Task:
   1. Predict short-horizon excess return from features.
   2. Adjust signal based on volatility and funding risk.
   3. Allocate position within leverage/risk limits.
   4. Evaluate via rolling-minute backtest.

   Output Example:
   {
     "alpha_signal": 0.012,
     "risk_adjustment": 0.86,
     "position_size": 0.42,
     "expected_pnl_next_hour": 0.009,
     "decision": "hold"
   }

Cryptocurrency-Specific Features
----------------------------------

1. **Funding Rate Risk**: Adjusts position size to avoid excessive funding payments
2. **Liquidation Risk**: Ensures leverage never exceeds safe threshold based on volatility
3. **24/7 Trading**: No market close; continuous position monitoring
4. **Higher Frequency**: 1-minute bars instead of daily bars

Risk Adjustments
-----------------

.. math::

   \text{position\_size} = \text{base\_size} \times \frac{\sigma_{\text{target}}}{\sigma_{\text{realized}}} \times (1 - \text{funding\_penalty})

where:

- :math:`\sigma_{\text{realized}}` is the rolling 1-hour realized volatility
- :math:`\text{funding\_penalty} = \max(0, \text{funding\_rate} \times 100)`


Prompt Flow Summary
====================

.. list-table:: Complete Prompt Structure
   :widths: 15 35 25 25
   :header-rows: 1

   * - Agent
     - Core Prompt Instruction
     - Key Output Fields
     - Constraints
   * - **Alpha**
     - "Estimate expected excess return based on alpha factors."
     - ``expected_excess_return``, ``confidence``
     - Lookback=60, Horizon=20
   * - **Risk**
     - "Evaluate volatility and scale exposure under constraints."
     - ``risk_score``, ``volatility``, ``risk_scaler``
     - ``target_vol=0.15``, ``beta_neutral``
   * - **Portfolio**
     - "Optimize weights to maximize risk-adjusted return."
     - ``weights``, ``expected_return``, ``target_sharpe``
     - ``gross_leverage≤2``, ``position≤0.25``
   * - **Backtest**
     - "Simulate portfolio and report performance metrics."
     - ``sharpe_ratio``, ``max_drawdown``, ``summary``
     - Transaction cost=10bps


Methodology Statement (For Paper)
===================================

.. note::

   **Prompt-Driven Agentic Trading Pipeline**

   Each trading prompt follows a unified Alpha–Risk–Portfolio–Backtest framework.
   The Alpha Agent computes the future expected excess return based on mined alpha factors.
   The Risk Agent evaluates volatility and systematic exposure, adjusting signal magnitude.
   The Portfolio Agent integrates alpha and risk signals into an optimized asset allocation 
   under leverage and exposure constraints.
   Finally, the Backtest Agent simulates trading outcomes, reporting Sharpe ratio, 
   drawdown, and turnover statistics.
   
   This modular prompt structure ensures **interpretability**, **reproducibility**, 
   and **cross-asset generality** across both stock and crypto tasks.


Reproducibility
=================

All prompts are version-controlled and logged with:

- **Model**: GPT-4 / Claude-3.5
- **Temperature**: 0.1 (deterministic)
- **Max Tokens**: 2048
- **Seed**: Fixed per experiment

Each agent prompt includes explicit:

- Input data schema
- Output JSON schema  
- Parameter specifications (windows, constraints, thresholds)

This enables exact replication of agent behavior across runs.



