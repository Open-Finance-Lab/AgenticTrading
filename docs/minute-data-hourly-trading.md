# 分钟源数据与小时级交易契约

Phase 0/1 固定四个时间概念，避免把数据频率误当成策略频率：

| 字段 | Phase 0/1 目标 | 含义 |
| --- | --- | --- |
| `source_timeframe` | `5m` | 数据供应商返回的原始 bar 周期 |
| `decision_timeframe` | `60m` | 策略观察的已完成 bar 周期 |
| `decision_frequency` | `1h` | agent/策略的调用频率 |
| `execution_timeframe` | `5m` | 后续成交模拟使用的 bar 周期 |
| `valuation_frequency` | `5m` | 后续组合盯市使用的周期 |

本阶段只完成契约和数据层能力，不改变现有 `HourlyBacktester` 的执行循环。
现有回测仍默认读取 `60m`，避免在 5 分钟聚合器接入前把每根 5 分钟 bar
误当作一次策略决策。Phase 2 完成 5m→60m 聚合后，再由回测引擎显式传入
profile 的 `source_timeframe`。

## 代码契约

`dashboard/backend/infrastructure/market_data/frequency.py` 提供：

- `normalize_bar_timeframe()`：将 `1min`、`5Min`、`hourly` 等别名归一化为
  `1m`、`5m`、`60m`。
- `TradingFrequency`：校验源周期、决策周期、决策频率、执行周期和估值周期。
- `TradingFrequency.minute_source_hourly_decisions()`：返回目标 5m 源数据、
  1h 决策契约。

`MarketProfile` 保留旧的 `timeframe` 字段作为决策 bar 周期，并新增：

```text
source_timeframe
decision_frequency
execution_timeframe
valuation_frequency
```

当前 Alpaca profile 已记录目标契约；iFinD 和 vn.py profile 继续使用现有
`60m` 数据能力。

## Alpaca 数据层

`AlpacaDataLoader` 保持默认 `60m`，保证旧调用方兼容；分钟数据调用方式：

```python
loader = AlpacaDataLoader(
    api_key="...",
    secret_key="...",
    source_timeframe="5m",
)
bars = loader.fetch_bars(["AAPL"], "2026-01-01", "2026-01-03")
```

也可以通过 provider factory 显式选择：

```python
provider = create_market_data_provider(
    "alpaca",
    source_timeframe="5m",
)
```

支持的 Alpaca 源周期为 `1m`、`5m` 和 `60m`。每次 fetch 的
`last_fetch["source_timeframe"]` 会记录实际请求周期。默认不传
`source_timeframe` 时仍请求 profile 的决策周期，防止旧小时回测行为改变。
