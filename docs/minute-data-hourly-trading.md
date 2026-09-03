# 分钟源数据与小时级交易契约

Phase 0/1 固定四个时间概念，避免把数据频率误当成策略频率：

| 字段 | Phase 0/1 目标 | 含义 |
| --- | --- | --- |
| `source_timeframe` | `5m` | 数据供应商返回的原始 bar 周期 |
| `decision_timeframe` | `60m` | 策略观察的已完成 bar 周期 |
| `decision_frequency` | `1h` | agent/策略的调用频率 |
| `execution_timeframe` | `5m` | 后续成交模拟使用的 bar 周期 |
| `valuation_frequency` | `5m` | 后续组合盯市使用的周期 |

Phase 2 已接入 `HourlyBacktester` 和 API 的 `ExternalBacktestSession`：Alpaca
回测会显式请求 profile 的 `5m`
源数据，按交易时段聚合成已完成的 `60m` 决策 bar。策略/LLM 仍每小时调用
一次；成交使用对应下一根源 bar 的开盘价，组合估值按每根 `5m` bar 更新。
共享数据集缓存也按源/决策频率隔离，避免同一日期窗口的旧小时数据和分钟数据
相互复用。非 Alpaca provider 和旧的小时级注入 loader 仍保持原有路径。

聚合器位于 `dashboard/backend/domain/backtesting/bar_aggregation.py`，使用
美股 09:30 和 A 股早午盘边界切桶，不跨越午休；聚合结果保留
`source_bar_count`、`expected_source_bars`、`is_complete`、`has_gap` 等质量字段。
最后一个没有下一根源 bar 的收盘桶不进入可执行决策。

## Phase 3：正确性硬化与审计

Phase 3 将分钟链路的数据质量和无未来数据约束固化为可测试契约：

- 小时 bar 使用左闭右开区间。例如 10:30 决策只包含 09:30–10:25 的
  5 分钟 bar；10:30 的源 bar 只可作为该决策的成交 bar。
- 完整性按预期的 5 分钟时间槽逐一校验，而非只比较行数。缺失、重复、
  非 5 分钟网格时间戳和无效 OHLCV 都会使该小时 bar 失效。
- 失效小时 bar 不进入 agent 决策集。多标的决策时点必须覆盖至少 80% 的
  标的，门槛向上取整；没有时点达标时不再回退到稀疏时间轴。
- 成交必须精确命中小时决策边界处的源 bar。该标的缺少这根 bar 时保持
  未成交，不会延迟到后续 5 分钟 bar 补成交。
- `market_data_quality` 随回测结果写入元数据，记录可用/丢弃小时 bar 数，
  以及缺失、重复、错位和无效源 bar 数；网页版决策审计同时记录
  `timestamp` 与 `execution_timestamp`。

质量统计以“标的 × 决策 bar”为计数单位。同一小时若 30 个标的均有数据，
会计为 30 个 decision bars。

## Phase 4：API 可观测性

Phase 4 保持唯一、固定的成交规则 `next_source_bar_open`，不增加成交策略
配置项。回测详情 API 和外部 Agent 的完成结果会返回：

- `frequency_contract`：5m 源数据、60m 决策 bar、1h 决策、5m 成交与估值；
- `market_data_quality`：聚合后的可用、丢弃及异常计数。

数据库仍保留逐标的质量明细；列表 API 仅返回有界的汇总字段，避免运行列表
随股票池大小无限膨胀。

## Phase 5：网页展示与产品契约

网页版回测在运行前和完成后均明确显示 Alpaca 5m 源数据与小时级决策。
高级详情展示固定成交时点、估值频率和数据质量摘要；历史旧回测缺少新元数据时
保持原有显示，不伪造分钟级来源。独立策略页面的旧 “Alpaca hourly bars”
文案同步更新。

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
