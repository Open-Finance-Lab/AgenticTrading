# Nemotron 回测诊断数据设计

## 1. 背景

Issue #148 需要解释 Nemotron 在排行榜中表现不佳的原因。当前系统只保存整次回测的汇总指标，例如收益率、交易次数和 LLM 调用次数，没有保存每个交易步骤的模型响应状态、解析结果、回退原因和实际成交结果。因此目前无法区分：

- 模型没有返回可执行文本；
- JSON 解析失败或触发了规则策略回退；
- 模型确实输出了交易动作，但动作质量较差或交易过于频繁；
- 回测曲线存储重复，导致排行榜展示失真。

本设计为第二个 Loop 增加结构化诊断信息，只观察和记录现有行为，不在这一阶段修改提示词、交易规则或模型选择。

## 2. 目标

1. 为每个 LLM 回测步骤保存一条结构化诊断记录。
2. 区分 API 调用、响应解析、模型驱动决策、规则回退和实际成交。
3. 记录 Nemotron 的 reasoning 配置，支持 `medium` 与 `none` 的可重复比较。
4. 保留足够信息定位过度交易、解析失败和回退发生的步骤，但不保存完整 prompt 或模型 reasoning 原文。
5. 在临时数据库中完成同一行情窗口的两组对照实验，形成可以反馈到 Issue #148 的证据。

## 3. 非目标

- 本阶段不修改 Nemotron 的提示词、仓位规则、买卖规则或风险参数。
- 本阶段不改变现有回测的交易结果和回退行为。
- 本阶段不建设前端诊断页面。
- 本阶段不保存模型完整回复、完整 reasoning、完整 prompt 或账户密钥。
- 本阶段不清理线上排行榜历史数据；曲线去重作为独立修复项处理。

## 4. 方案选择

### 方案 A：扩展 `backtest_decisions`

改动最少，但该表已经用于外部 Agent 提交的决策。将内部 LLM 调用和外部提交混在一起，会使数据来源和字段含义不清晰。

### 方案 B：新增 `llm_step_diagnostics`（采用）

单独记录每个内部 LLM 步骤，保持现有 `backtest_decisions` 的语义不变。数据库写入可以在回测结束时批量完成，避免每个小时打开一次数据库连接。

### 方案 C：只保存运行级汇总

实现简单，但无法回答“哪一个步骤没有文本”“哪一步触发回退”“模型在哪些时刻连续买卖”等关键问题，不足以诊断 Issue #148。

采用方案 B，因为它在诊断粒度和实现范围之间最平衡。

## 5. 数据模型

### 5.1 `llm_step_diagnostics`

每个 LLM 决策步骤最多写入一条记录，使用 `(run_id, step_index)` 建立查询索引。

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `id` | INTEGER | 自增主键 |
| `run_id` | TEXT | 关联 `agent_runs.run_id` |
| `step_index` | INTEGER | 回测中的小时步骤，沿用引擎循环的 0-based `i` |
| `timestamp` | TEXT | 该步骤的行情时间 |
| `model_id` | TEXT | 实际使用的模型 ID |
| `integration` | TEXT | `openrouter`、`anthropic` 等网关 |
| `reasoning_effort` | TEXT | 该步骤生效的 reasoning 配置 |
| `response_block_types` | TEXT | JSON 数组，例如 `["thinking", "text"]` |
| `text_present` | INTEGER | 是否提取到可解析文本，使用 0/1 |
| `parse_success` | INTEGER | 是否解析出决策对象，使用 0/1 |
| `fallback_reason` | TEXT | 空文本、解析失败、异常或空动作等原因；没有回退时为空 |
| `retry_count` | INTEGER | 因空文本或响应问题产生的重试次数 |
| `llm_call_count` | INTEGER | 该步骤实际消耗的 API 调用次数 |
| `actions_proposed` | TEXT | 结构化动作摘要 JSON，不包含 reasoning 原文 |
| `actions_accepted` | INTEGER | 通过本地校验并交给执行器的动作数量 |
| `trades_executed` | INTEGER | 该步骤实际新增成交数量 |
| `latency_ms` | INTEGER | 该步骤端到端耗时 |
| `error_type` | TEXT | 异常类型名称；正常时为空 |
| `created_at` | TIMESTAMP | 记录写入时间 |

`actions_proposed` 只保留 symbol、action、shares/position_size 和 confidence 等结构化字段，不保留模型的自然语言 reasoning。

### 5.2 `agent_runs.metadata`

在现有运行级 metadata 中增加诊断汇总：

```json
{
  "decision_steps": 161,
  "llm_decisions": 160,
  "fallback_steps": 0,
  "no_text_steps": 5,
  "parse_failures": 0,
  "total_retries": 20,
  "avg_latency_ms": 3200,
  "reasoning_effort": "medium",
  "model_id": "nvidia/nemotron-3-nano-30b-a3b",
  "integration": "openrouter"
}
```

原有字段含义不变；旧数据库的 `metadata` 为空时仍然可以正常读取。

## 6. 代码职责和数据流

### 6.1 PortfolioManager

`PortfolioManager` 增加内存列表 `llm_diagnostics`，每次调用 `make_trading_decision_with_llm` 时生成诊断对象：

1. 调用开始时记录计时和当前模型配置。
2. 每次 API 响应记录 block 类型和 token 调用次数。
3. `_extract_response_text` 成功时记录 `text_present`；空文本重试时累加 `retry_count`。
4. `_parse_llm_response` 返回对象时记录 `parse_success`。
5. 规则回退时记录稳定的 `fallback_reason`，但继续执行现有回退逻辑。
6. 记录模型动作摘要和通过本地过滤的动作数量。

方法的现有参数保持兼容；步骤编号和时间戳使用可选参数传入，旧调用方不传时仍能工作。

### 6.2 Backtest engine

回测引擎在调用策略时传入 `step_index` 和 `timestamp`。执行动作前记录已有成交数量，执行动作后用差值补充 `trades_executed`。回测结束时：

1. 汇总 `llm_diagnostics` 得到运行级计数。
2. 将汇总写入 `agent_runs.metadata`。
3. 将步骤记录批量写入 `llm_step_diagnostics`。

诊断写入失败不能改变已完成的回测结果；失败只应记录警告并保留可查询的运行汇总。

### 6.3 Database layer

`BacktestDatabase` 增加：

- 新表创建和旧数据库迁移逻辑；
- `insert_llm_diagnostics(run_id, diagnostics)` 批量写入；
- `get_llm_diagnostics(run_id)` 按步骤查询。

所有新增字段使用可空或有默认值的定义，确保已有数据库可以启动。新增表对 `(run_id, step_index)` 使用唯一约束，保证每个步骤最多一条记录；重复写入同一个运行时，应使用幂等写入策略，避免二次 finalize 产生重复行。

## 7. 错误处理

- API 返回只有 thinking 或 redacted thinking：记录 block 类型、`text_present=0` 和重试次数；沿用现有重试与 rescue 行为。
- 最终仍无文本：记录 `fallback_reason=no_text_after_retries`，沿用规则回退。
- JSON 解析失败：记录 `fallback_reason=parse_failed`；不保存原文。
- 动作字段非法或被本地约束过滤：记录解析成功，但 `actions_accepted` 小于提议数量；不把它误记为 API 回退。
- 执行器拒绝动作：记录解析和接受状态，使用成交差值记录实际结果。
- 诊断数据库写入失败：不回滚回测，不改变收益和交易结果。

## 8. 第二个 Loop 实验

### 固定条件

- 同一代码版本；
- 同一行情文件和日期窗口；
- 同一初始资金、资产集合和交易模式；
- 使用临时数据库；
- 记录数据源和模型配置，避免把不同数据窗口的结果直接比较。

### 对照组

| 实验 | 模型 | reasoning |
| --- | --- | --- |
| A | `nvidia/nemotron-3-nano-30b-a3b` | `medium` |
| B | `nvidia/nemotron-3-nano-30b-a3b` | `none` |

### 判断标准

- A 的空文本、重试和延迟明显高于 B：优先修 reasoning 配置或响应可靠性。
- 两组 fallback 都很少，但交易数和换手率都异常高且收益差：优先检查模型交易行为和提示词。
- fallback 占比较高：先修调用链，再评价模型能力。
- 收益指标正常但曲线点数重复：归因于数据库/排行榜展示问题，而不是模型表现。
- 诊断记录数量少于实际决策步骤：停止归因，先修诊断完整性。

## 9. 测试

新增测试覆盖：

1. 新数据库创建 `llm_step_diagnostics` 表。
2. 旧数据库迁移后可以写入和读取诊断记录。
3. 结构化动作摘要不包含完整 reasoning 或 prompt。
4. 空文本重试、最终 rescue、解析失败和异常回退分别产生正确的诊断字段。
5. 执行前后成交数差值正确。
6. 一条最小模拟回测产生“步骤数 = 诊断记录数”。
7. 原有 LLM、回测和数据库测试保持通过。

## 10. 完成标准

当以下条件全部满足时，第二个 Loop 完成：

- 两组对照实验都生成完整诊断记录；
- 可以按步骤统计 no-text、parse failure、fallback 和实际成交；
- 可以判断亏损来自调用链问题、模型交易行为或排行榜数据问题；
- 没有保存完整模型回复，也没有改变现有交易行为；
- 测试通过，且实验结果可以用数字和 run_id 反馈到 Issue #148。
