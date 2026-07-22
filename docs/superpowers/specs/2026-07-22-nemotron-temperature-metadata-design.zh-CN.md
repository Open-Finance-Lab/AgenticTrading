# Issue #148：Nemotron 温度固定与排行榜运行元数据设计

## 1. 背景

Nemotron 已通过 OpenRouter 接入 ATL 排行榜，但历史收益曲线明显弱于其他模型。Issue #148 要求检查从模型响应到最终资金曲线的完整链路，判断差曲线来自接入错误、解析与执行错误，还是模型本身的交易决策。

现有调查已经确认：

- 排行榜使用的模型是 `nvidia/nemotron-3-nano-30b-a3b`；
- PR #149 已为 Nemotron 设置 `reasoning_effort: "none"`，避免 reasoning 占用输出空间；
- 历史 `-0.22%` 曲线实际使用 16k 输出上限、关闭 reasoning 和 10 万美元初始资金，不是“medium reasoning 挤占 2k 输出预算”造成的；
- 一次 1,000 美元短验证得到 `+0.81%`，但它与 10 万美元历史排行榜运行条件不同，不能直接比较；
- 当前模型请求没有固定 `temperature`，因此相同代码和行情仍可能得到不同回答；
- 手动部署的排行榜 LLM 运行没有完整保存模型请求与回测配置，事后难以严格复现。

因此，当前不能严谨地断言 Nemotron 的差曲线一定是模型能力问题。下一步先消除一个仍未固定的模型参数，并给新运行留下完整的配置记录。

## 2. 目标

本次变更只解决“可复现性”问题：

1. 在 Nemotron 排行榜配置中显式设置 `temperature: 0`。
2. 建立通用的可选温度参数传递链，供其他模型以后按配置使用。
3. 未配置温度的模型继续使用 provider 默认行为。
4. 所有通过 `deploy_model_run()` 手动部署的新 LLM 排行榜运行都保存关键配置元数据。
5. 配置错误在真实模型调用前失败，避免产生费用后才发现实验无效。

本次变更不承诺提高 Nemotron 收益，也不修改提示词、响应格式、解析逻辑、交易规则、仓位管理、模拟成交、行情数据或数据库结构。

## 3. 方案选择

### 3.1 采用方案：模型级配置与通用参数传递

在 `leaderboard.json` 的 Nemotron 条目中增加 `temperature: 0`。`LLMAgentStrategy` 读取并校验它，然后经 `PortfolioManager` 传给 `request_trading_decision()`。请求层只在值存在时把 `temperature` 交给 SDK。

该方案让实验条件直接跟随模型配置，不依赖运行进程中的隐式状态；基础设施保持通用，不需要识别 Nemotron 的模型名称。温度参数在整个链路中不得读取或修改 `os.environ`。

### 3.2 未采用：环境变量

环境变量会影响整个进程。在并发运行多个模型时，一个模型的温度可能意外影响另一个模型，而且数据库无法仅从模型条目判断实际使用值。

### 3.3 未采用：按模型 ID 硬编码

在请求层判断 `nvidia/nemotron-3-nano-30b-a3b` 虽然改动小，但会把业务名单写进通用基础设施。模型升级或增加其他模型时还要继续增加条件分支，配置也无法完整说明行为。

## 4. 架构与参数流

```text
dashboard/config/leaderboard.json
  Nemotron: temperature=0
        ↓
LLMAgentStrategy
  读取并校验可选温度
        ↓
PortfolioManager.make_trading_decision_with_llm(..., temperature=0)
        ↓
request_trading_decision(..., temperature=0)
        ↓
client.messages.create(..., temperature=0)
        ↓
OpenRouter / Nemotron
```

### 4.1 排行榜配置

Nemotron 条目增加：

```json
{
  "id": "nemotron_3_nano_30b",
  "model_id": "nvidia/nemotron-3-nano-30b-a3b",
  "reasoning_effort": "none",
  "temperature": 0
}
```

其他模型不增加该字段，因此保持当前 provider 默认温度。

### 4.2 策略层

`LLMAgentStrategy` 保存可选的 `temperature`。配置存在时，它必须是 `0.0` 到 `2.0` 之间的数字；布尔值、字符串、负数和大于 2 的数字都视为错误。

校验发生在策略创建阶段。`deploy_model_run()` 在获取行情和发起真实 LLM 请求前创建策略，因此错误配置会尽早停止。

策略调用 `PortfolioManager.make_trading_decision_with_llm()` 时传递温度。该参数默认值为 `None`，所以现有调用方不需要修改。

### 4.3 交易决策层

`PortfolioManager.make_trading_decision_with_llm()` 接收可选温度，并在直接 LLM 请求路径的首次调用和无文本重试中始终传给 `request_trading_decision()`。现有 parsing、空动作、规则回退和执行行为不变。

本次不改变 sub-agent pipeline 的独立请求协议，因为排行榜 Nemotron 当前不经过该路径。

### 4.4 请求层

`request_trading_decision()` 增加可选 `temperature` 参数，并用请求参数字典调用 `client.messages.create()`：

- 值为 `0` 时，明确发送 `temperature=0`；
- 值为 `None` 时，完全省略 `temperature` 键，而不是发送 JSON `null`。

省略字段可以保证未配置模型继续使用各自 provider 的原有默认值。

## 5. 排行榜运行元数据

`deploy_model_run()` 为每个新保存的 LLM 排行榜运行构建配置快照，并通过数据库现有的 `agent_runs.metadata` JSON 字段与成绩一起写入。无需新增表或迁移字段。

元数据结构为：

```json
{
  "entry_id": "nemotron_3_nano_30b",
  "model_id": "nvidia/nemotron-3-nano-30b-a3b",
  "integration": "openrouter",
  "temperature": 0,
  "reasoning_effort": "none",
  "llm_max_output_tokens": 2000,
  "initial_capital": 10000,
  "start_date": "2026-04-15",
  "end_date": "2026-05-15"
}
```

这里的数值来自本次运行解析后的有效配置，不写死示例值。例如最新主线的排行榜初始资金是 10,000 美元，因此当前完整窗口的新运行会记录 `10000`；以后配置变化时记录新的实际值。

对没有显式温度的 LLM 条目，元数据记录 `"temperature": null`，表示该次运行使用 provider 默认温度，而不是错误地猜测一个数值。`reasoning_effort` 未配置时同样记录 `null`。

以下敏感或体积过大的内容不进入元数据：

- API Key、Endpoint Key、Secret；
- 完整提示词；
- 模型原始回答；
- reasoning 或 thinking 文本。

历史缓存运行不会被伪造或回填配置，因为无法可靠知道它们当时的完整有效参数。需要可复现记录时，应在新代码下使用 `--force` 重新运行。

## 6. 错误处理与兼容性

- 没有 `temperature`：接受配置，并保持旧请求行为。
- 合法温度：接受 `0` 到 `2` 之间的整数或浮点数。
- 非法温度：在策略创建时抛出说明字段和值的 `ValueError`，不获取完整行情、不调用模型。
- 模型请求被 provider 拒绝：沿用现有异常和回退逻辑；排行榜已有的 fallback guard 继续阻止把无效 LLM 运行误发为正常模型成绩。
- 响应为空或 JSON 无效：沿用现有重试、解析和动作回退行为，本次不改变 HOLD 或规则回退的语义。
- 元数据序列化或数据库写入失败：`insert_run()` 失败，成绩不发布；不允许出现“有曲线但没有运行条件”的新排行榜记录。
- 非 LLM 排行榜条目及普通回测：行为不变。

## 7. 测试设计

所有自动测试使用 fake client、fake strategy、临时数据库或 monkeypatch，不进行真实网络请求，不消耗 OpenRouter 额度。

1. 配置测试：只有 Nemotron 明确设置 `temperature: 0`。
2. 校验测试：`0`、合法小数和 `2` 通过；字符串、布尔值、负数和大于 2 的值失败。
3. 策略传递测试：`LLMAgentStrategy` 把模型级温度交给 `PortfolioManager`。
4. 决策层测试：首次请求和重试请求都传递同一温度。
5. 请求载荷测试：`temperature=0` 出现在 SDK 请求参数中。
6. 兼容性测试：温度缺失时，SDK 请求参数中不存在 `temperature` 键。
7. 元数据测试：手动部署的 LLM 运行保存模型、integration、温度、reasoning、输出上限、资金和日期。
8. 空值测试：未配置温度或 reasoning 的 LLM 运行将对应字段保存为 `null`。
9. 回归测试：现有排行榜 fallback guard、响应解析和数据库 metadata round-trip 测试继续通过。

## 8. 成功标准与下一轮实验

代码层成功标准：

- Nemotron 的 OpenRouter 请求可观察到 `temperature=0`；
- 其他未配置模型的请求载荷与修改前一致；
- 新手动部署的 LLM 排行榜记录可以从数据库读回完整配置快照；
- 非法配置不会产生真实模型调用；
- 相关自动测试全部通过。

该变更合并后，再运行一次完整 Nemotron 排行榜窗口，并固定模型 ID、`temperature=0`、`reasoning_effort=none`、输出上限、初始资金、日期和行情范围。然后检查有效响应率、解析结果、执行动作和最终曲线，才能继续判断差表现是否主要来自 Nemotron 的真实交易决策。

口语化地说，本次不是给选手换脑子，而是把赛场温度、试卷版本、考试时间和起始资金全部登记清楚。下一次成绩好或差，我们才知道是在相同条件下测出来的。

## 9. 预计修改范围

- `dashboard/config/leaderboard.json`
- `dashboard/backend/domain/leaderboard/strategies/llm_agent.py`
- `dashboard/backend/domain/backtesting/portfolio_manager.py`
- `dashboard/backend/infrastructure/llm/backtest_harness.py`
- `dashboard/backend/domain/leaderboard/service.py`
- 对应的 leaderboard、LLM harness、portfolio manager 和 deploy 测试文件
