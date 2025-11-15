# Orchestration Overview: End-to-End Agentic Trading Pipeline

![Orchestration Diagram](docs/source/intro/orchestration%202.png)

This diagram illustrates the end-to-end agentic trading pipeline that underlies our prompt design. The Data Agent sits at the left of the workflow and supplies a standardized market dataset that is fanned out to both the Alpha Agent and the Risk Agent.

The Alpha Agent consumes historical OHLCV data together with engineered factor inputs to generate expected excess returns. In parallel, the Risk Agent ingests the alpha signals along with covariance estimates to compute risk-adjusted scores and constraints-aware scaling.

These outputs are then merged and forwarded to the Portfolio Agent, which optimizes portfolio weights under leverage limits, position constraints, and neutrality requirements. The optimized allocation flows along two downstream paths: to the Execution Agent, which translates target weights into executable orders, and to the Backtest Agent, which applies the same portfolio rules to historical data to simulate performance and report metrics such as return, volatility, Sharpe ratio, and drawdown.
