# TradingAgents 接入 ATL 第一版设计规格

- **日期：** 2026-07-26
- **状态：** 设计与书面规格已批准，进入实施计划
- **项目：** AgenticTrading（ATL）
- **上游项目：** `TauricResearch/TradingAgents`
- **参考版本：** TradingAgents v0.3.1；ATL `origin/main` 3a7781a
- **迭代：** 三个热门项目接入计划的第 1 个项目

## 1. 背景与价值

TradingAgents 是一个本地运行的多 Agent 金融研究框架。它让基本面、新闻、情绪、
技术分析、研究辩论、交易和风险管理等 Agent 协作，最后针对“单只股票 + 一个分析
日期”给出五档评级：`Buy`、`Overweight`、`Hold`、`Underweight` 或 `Sell`。

TradingAgents 擅长生成一次研究结论，但它的完整多 Agent 分析不适合在 ATL 每个
小时的决策时限内现场运行。ATL 的价值是提供统一的市场环境、模拟成交、收益指标、
交易记录、曲线、Agent Card 和排行榜。第一版使用离线桥接把两者连接起来：

```text
TradingAgents 负责“研究和判断”
ATL 负责“模拟成交、算账和比较”
```

口语化理解：TradingAgents 用户带着自己的赛车和油费来到 ATL；ATL 提供赛道、计时、
成绩曲线和排名，不替用户购买模型调用额度。

## 2. 目标

让用户在自己的电脑上，对一只 ATL 支持的美股运行 TradingAgents，并用一条本地命令
完成以下链路：

```text
用户指定股票、分析日期和 TradingAgents 配置
  -> 本地预生成并保存 TradingAgents 决策
  -> 将五档评级转换为 ATL 订单
  -> 使用 ATL 的美股小时环境模拟成交
  -> 获得交易记录、指标、收益曲线、Agent Card 和排行榜记录
```

第一版成功的定义不是“能够 import TradingAgents”，而是一个新用户能按文档独立跑完
上述链路，并能解释每个动作、错误和结果的数据来源。

## 3. 第一版范围

### 3.1 包含

- 一次回测只选择一只 ATL 当前支持的美股。
- TradingAgents 在用户本地运行，使用用户自己的 LLM Key 和数据源。
- 用户显式指定若干分析日期，例如每周一次，不要求每天运行。
- 所有 TradingAgents 分析在 ATL 回测开始前完成并写入决策文件。
- ATL 使用现有 `us-equity-hourly-v1` 环境和 Alpaca 行情完成模拟成交与估值。
- 使用 ATL 当前 typed `ATLClient` 和 Agent-Environment Protocol v1，不使用旧的
  `AgenticTradingClient.run_backtest(strategy=...)` 接口。
- 决策文件可以反复回放；回放不再次调用 LLM。
- ATL API Key、LLM Key 和数据供应商 Key 只留在用户本机环境变量中。

### 3.2 不包含

- ATL 服务器托管或代付 TradingAgents 的 LLM 调用。
- 在 ATL 网页中直接配置或启动 TradingAgents。
- 强制把 ATL 市场快照注入 TradingAgents。
- 同时分析多只股票或直接覆盖 DJIA-30。
- 实盘、模拟盘、做空、融资、限价单或盘中重新调用 TradingAgents。
- 新增允许单股票 100% 满仓的 ATL 环境。
- 修改 TradingAgents 上游代码或重写它的多 Agent 图。
- 保证重新生成的 TradingAgents 决策完全一致或保证策略盈利。
- 为 TradingAgents 单独建设服务进程、任务队列或 Key 托管系统。
- 新增专用网页诊断面板；第一版通过本地摘要、决策文件和 ATL 决策日志查看诊断。

## 4. 已确认的设计决策

| 决策项 | 第一版选择 | 原因 |
|---|---|---|
| 运行位置 | 用户本地 | 用户自己保管 Key、控制模型并承担费用。 |
| 决策数据源 | TradingAgents 自带数据源 | 接入最快，不深改 TradingAgents。 |
| 成交和估值数据源 | ATL 现有 Alpaca 行情 | 所有参赛 Agent 使用 ATL 的统一成交与计分规则。 |
| 股票数量 | 一只 | 先验证完整链路，控制费用和查错范围。 |
| 分析频率 | 用户显式指定日期，例如每周一次 | 一次完整多 Agent 研究成本高，不适合每小时或默认每天调用。 |
| 调用方式 | 回测前离线生成，回测中只回放 | 避开 ATL 每步 60 秒决策时限，并支持确定性重放。 |
| 交易方向 | 只做多 | 与当前 ATL 环境一致。 |
| 执行时间 | 分析日之后的第一个 ATL 交易时段 | 防止使用当天完整数据后回到当天交易的未来数据泄漏。 |
| 目标仓位 | 当前环境允许的最大单股仓位，现为总资产 25% | 遵守 `max_position_weight=0.25`，不新增环境。 |
| SELL 含义 | 卖出该股票的全部现有持仓 | 没有持仓时不做空，动作降为 HOLD。 |
| 错误策略 | 单个分析日期重试一次，仍失败则记录错误 HOLD | 让部分失败的回测继续，同时不把失败伪装成主动 HOLD。 |
| 用户入口 | 本地示例命令 + 集成文档 | 第一版不建设网页托管入口。 |

## 5. 备选架构及取舍

### 5.1 回测时直接调用 TradingAgents

ATL 每到一个决策时点就现场调用 `TradingAgentsGraph.propagate()`。代码表面直接，但完整
多 Agent 分析可能超过 ATL 的决策时限，还会在小时循环中重复产生高额 LLM 费用。一次
延迟可能被 ATL 自动转为超时 HOLD，污染曲线，因此不采用。

### 5.2 将 TradingAgents 启动为本地 API 服务

ATL 通过 HTTP 请求本地 TradingAgents 服务。该方案适合未来多用户、并行和任务队列，
但第一版需要增加进程管理、鉴权、健康检查和部署文档，超出当前验证目标，因此不采用。

### 5.3 离线决策文件 + ATL 回放

先为用户指定的日期生成完整决策文件，再由轻量适配器驱动 ATL 小时回测。它把昂贵、
缓慢、非确定的研究阶段与快速、可重复的成交阶段分开，最容易定位“模型判断问题”与
“适配或执行问题”，因此采用。

## 6. 总体架构

```text
用户本地
┌──────────────────────────────────────────────────────────────┐
│ TradingAgentsDecisionGenerator                              │
│   TradingAgentsGraph.propagate(symbol, analysis_date)       │
│   使用用户自己的模型、Key、行情、新闻和配置                  │
└───────────────────────────┬──────────────────────────────────┘
                            │ 原始 final_trade_decision + 五档评级
                            v
┌──────────────────────────────────────────────────────────────┐
│ TradingAgentsDecisionArtifact                               │
│   校验评级、转换三档动作、记录版本/数据源/错误、写 JSON       │
└───────────────────────────┬──────────────────────────────────┘
                            │ 可重复回放的只读决策文件
                            v
┌──────────────────────────────────────────────────────────────┐
│ TradingAgentsATLRunner                                      │
│   ATLClient 创建 run、读取 Step 时间/价格/仓位/约束           │
│   在 T+1 第一个小时提交订单，其余小时提交显式 HOLD             │
└───────────────────────────┬──────────────────────────────────┘
                            │ Agent-Environment Protocol v1
                            v
┌──────────────────────────────────────────────────────────────┐
│ ATL us-equity-hourly-v1                                     │
│   Alpaca 行情 -> 模拟成交 -> 指标/曲线 -> Agent Card/排行榜   │
└──────────────────────────────────────────────────────────────┘
```

### 6.1 为什么不直接使用通用 `AgentRunner`

当前 `AgentRunner` 只把 `Observation` 传给 `agent.decide()`；离线回放还需要 `Step.timestamp`
来判断是否已经到达分析日之后的第一个交易时段，并需要 `Step.constraints` 获取允许股票和
最大仓位。因此第一版由一个小型 `TradingAgentsATLRunner` 使用现有 `ATLClient` 方法驱动
循环，不修改通用 `AgentRunner` 的公开协议。

## 7. TradingAgents 生成阶段

### 7.1 依赖边界

`tradingagents` 是可选的客户端依赖，不能加入 ATL 后端或轻量 SDK 的核心依赖。集成模块
只在“生成决策”功能被调用时延迟导入 TradingAgents。未安装时给出明确安装说明：

```text
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
pip install .
pip install -e /path/to/AgenticTrading/packaging/agentictrading
```

第一版针对 TradingAgents v0.3.1 测试。运行时记录实际包版本；发现不兼容的大版本时应
明确失败，不能猜测新的返回格式。

### 7.2 调用契约

每个用户指定的分析日期调用公开接口：

```python
state, processed_rating = graph.propagate(symbol, analysis_date)
raw_decision = state["final_trade_decision"]
```

适配器使用 TradingAgents 自带的 `parse_rating(raw_decision, default="")` 重新校验显式
评级。原因是 `propagate()` 的处理结果在无法解析时可能默认返回 `Hold`；第一版不能把
格式错误静默伪装成模型主动 HOLD。

### 7.3 五档到三档映射

| TradingAgents v0.3.1 | ATL 方向 | 含义 |
|---|---|---|
| `Buy` | `BUY` | 建立或补足多头目标仓位。 |
| `Overweight` | `BUY` | 建立或补足多头目标仓位。 |
| `Hold` | `HOLD` | 保持当前仓位，不提交订单。 |
| `Underweight` | `SELL` | 清空当前多头仓位；空仓时 HOLD。 |
| `Sell` | `SELL` | 清空当前多头仓位；空仓时 HOLD。 |

评级是方向等级，不是经过校准的概率。第一版不编造置信概率；提交给 ATL 的
`Decision.confidence` 保持空值，并在 `rationale` 中保留评级和分析日期。ATL 协议当前
会对空值使用执行默认置信度 0.75，该值必须标记为 ATL 默认值，不能解释为 TradingAgents
输出的胜率或概率。

### 7.4 调用次数与重试

- 用户通过可重复的 `--analysis-date YYYY-MM-DD` 参数提供一个或多个日期。
- 日期必须唯一并按时间排序；股票代码在一次运行中保持不变。
- 每个日期最多尝试两次，即第一次失败后重试一次。
- TradingAgents 内部仍可按其 `llm_max_retries` 配置处理单次 LLM 请求；适配器的重试是
  对整个日期分析的外层重试，次数和最终状态必须记录。
- 所有日期生成完成后才允许启动 ATL 回测。
- 如果所有日期都失败，则不创建 ATL run，避免生成一条看似正常的全空仓曲线。

## 8. 决策文件契约

决策文件为 UTF-8 JSON，默认写入用户目录而不是 Git 仓库：

```text
~/.agentictrading/tradingagents/decisions/<symbol>-<timestamp>.json
```

顶层结构：

```json
{
  "schema_version": "tradingagents-atl-v1",
  "manifest": {
    "symbol": "AAPL",
    "created_at": "2026-07-26T12:00:00Z",
    "tradingagents_version": "0.3.1",
    "atl_protocol_version": "1.0",
    "llm_provider": "openai",
    "deep_think_llm": "...",
    "quick_think_llm": "...",
    "selected_analysts": ["market", "social", "news", "fundamentals"],
    "data_vendors": {"core_stock_apis": "yfinance"},
    "safe_config_sha256": "..."
  },
  "decisions": [
    {
      "analysis_date": "2026-04-10",
      "rating": "Overweight",
      "atl_action": "BUY",
      "status": "valid",
      "attempts": 1,
      "raw_final_trade_decision": "...",
      "raw_sha256": "...",
      "error_type": null,
      "error_message": null
    }
  ]
}
```

只记录安全配置白名单，不序列化完整环境变量或任意用户配置。禁止写入名称包含
`KEY`、`TOKEN`、`SECRET`、`PASSWORD`、`CREDENTIAL` 的值。错误文本经过清理和长度限制，
防止 SDK 或供应商异常把凭据带入文件。

生成完成后计算整个决策文件的 SHA-256。ATL run 的 `config` 中记录集成名称、
TradingAgents 版本、股票、分析日期、决策文件哈希、两套数据来源以及有效/错误决策
数量，但不记录本地文件绝对路径和原始长文本。

## 9. ATL 回放与成交规则

### 9.1 启动前校验

回放器在创建 ATL run 前完成以下校验：

- 决策文件 schema、哈希、股票和日期合法。
- 至少有一个 `status=valid` 的决策。
- `ATL_API_KEY`、`ATL_BASE_URL` 和 `ATL_AGENT_VERSION_ID` 已配置。
- 所选股票属于 `us-equity-hourly-v1` 的允许股票列表。
- 每个分析日期都早于回测 `end_date`，且开始、结束日期顺序合法。

ATL 公共 SDK 在创建 run 前不提供未来全部小时 Step，因此启动前不能准确预知节假日后的
最后一个可执行时段。回放器在 run 完成后必须检查是否仍有未处理记录；若有，则打印
具体分析日期、返回非零退出状态并保留 `run_id`，不能把不完整回放报告为成功。

### 9.2 T+1 执行

每条决策记录只处理一次，包括有效评级和生成失败记录。它的处理点不是简单的“加一天”，
而是 ATL 返回的第一个满足以下条件的小时 Step：

```text
Step 的 America/New_York 交易日期 > analysis_date
```

这样周五记录会自然落到下周第一个实际交易日，节假日也由 ATL 的真实时间序列决定。
有效记录按照评级执行；`status=error` 的记录提交空订单，并在 `rationale` 中标记
`generation_error`。同一交易日的后续小时 Step 全部 HOLD。两个分析日期之间不重新调用
TradingAgents，只保留现有仓位并让 ATL 继续估值。

如果多个尚未执行的信号在同一个 Step 同时变为可执行，只执行分析日期最新的一个；
较旧记录标记为 `superseded`，防止在同一个价格上连续执行过期意见。

### 9.3 买入、卖出与 HOLD

回放器从 `Step.observation` 和 `Step.constraints` 读取当前价格、总权益、已有股数和
`max_position_weight`。

`BUY` 使用目标仓位而不是重复加仓：

```text
target_shares = floor(equity * max_position_weight / current_price)
buy_shares = max(0, target_shares - held_shares)
```

只有 `buy_shares > 0` 才提交市价买单；否则提交显式 HOLD，并注明
`already_at_target` 或 `price_too_high_for_target`。现有环境的最大仓位是 25%，因此第一版
不会触发 100% 满仓，也不会修改环境约束。

`SELL` 卖出当前持有的全部整数股数。没有持仓时提交显式 HOLD，并注明
`sell_without_position`。`HOLD` 使用空订单列表。第一版不做空。

每个非空订单必须使用 `quantity_type="shares"`、`order_type="market"`，并使用决策文件中
的评级和分析日期生成可审计的 `rationale`。

## 10. 数据来源与防未来信息

第一版明确存在两套数据：

| 数据 | 用途 | 负责人 |
|---|---|---|
| TradingAgents 自带行情、新闻、情绪和基本面 | 生成评级 | 用户本地 TradingAgents 配置 |
| ATL Alpaca 小时行情 | 决定模拟成交价格并计算净值曲线 | ATL 环境 |

两套数据可能存在复权、时间戳或供应商差异，所以结果页和本地摘要必须标注来源。第一版
不宣称这是所有 Agent 完全同输入的严格模型对比；它是“不同 Agent 按 ATL 统一成交和
计分规则参赛”的接入测试。

TradingAgents 在一个分析日期上可能使用该日期的完整数据。ATL 必须等到之后的第一个
交易时段才成交，不能在分析日期当天回填订单。若 TradingAgents 数据源自身返回了分析
日期之后的信息，属于上游数据契约问题；决策文件保留版本和数据源，便于复查。

## 11. 错误与诊断

### 11.1 启动前错误

以下错误立即停止且不创建 ATL run：TradingAgents 未安装、不兼容版本、缺少凭据、
配置无效、股票不受支持、决策文件损坏、全部分析日期失败。错误信息应告诉用户下一步
怎样修复。

### 11.2 单个分析日期失败

第一次失败后重试一次；仍失败则在决策文件中写入：

```text
status=error, atl_action=HOLD, attempts=2, error_type, sanitized error_message
```

只要仍有其他有效信号，ATL 回测可以继续。错误 HOLD 与 TradingAgents 主动 `Hold` 在
本地汇总和 ATL `rationale` 中使用不同标签，不能混为一类。

### 11.3 ATL 执行错误

- ATL 拒单必须打印并保存拒绝原因，不得报告为成功成交。
- 网络、鉴权或 API 错误直接抛出，并附上已创建的 `run_id` 供用户定位；不能在客户端
  静默转换为 HOLD。
- 如果 ATL 服务器因超过决策时限自动 HOLD，最终指标中的 `timeout_holds` 必须显示为
  非零。离线回放本身不应接近 60 秒时限，因此出现该值即视为需要调查的集成错误。

第一版本地完成摘要至少显示：有效信号数、主动 HOLD 数、错误 HOLD 数、被覆盖信号数、
未处理记录数、ATL 拒单数、超时 HOLD 数、成交数、run_id 和结果链接。存在未处理记录
时命令返回非零退出状态。每步 `rationale` 让错误原因也能在 ATL 决策日志中被检查。

## 12. 用户入口

示例命令采用明确参数，不隐式每天运行：

```bash
python dashboard/examples/tradingagents_atl_backtest.py \
  --symbol AAPL \
  --analysis-date 2026-04-03 \
  --analysis-date 2026-04-10 \
  --analysis-date 2026-04-17 \
  --start-date 2026-04-06 \
  --end-date 2026-04-24
```

用户凭据来自环境变量：

```text
ATL_API_KEY
ATL_BASE_URL
ATL_AGENT_VERSION_ID
TradingAgents 所选供应商要求的 LLM / 数据 Key
```

一条命令内部依次执行“生成决策文件”和“ATL 回放”。同时提供 `--decisions-file`，允许
跳过生成阶段，直接重放已有文件。这样用户可以先检查昂贵的模型输出，再重复验证 ATL
适配逻辑而不再次付费。

集成文档必须说明如何创建并复用 ATL AgentVersion。AgentVersion 使用
`architecture="multi_agent_debate"`，模型列表记录 TradingAgents 的 deep/quick 模型，
配置变化时应创建新版本，避免不同模型配置共用一个排行榜身份。

## 13. 代码边界与文件位置

```text
packaging/agentictrading/src/agentictrading/integrations/
  __init__.py
  tradingagents.py
    - 稳定公开导入门面
  _tradingagents_core.py
    - TradingAgentsDecisionGenerator
    - load/save/validate decision artifact
    - five-tier rating mapping
  _tradingagents_replay.py
    - T+1 信号选择、目标仓位和本地诊断
  _tradingagents_runner.py
    - ATLClient 生命周期和结果汇总

packaging/agentictrading/tests/
  test_tradingagents_integration.py

dashboard/examples/
  tradingagents_atl_backtest.py

docs/integrations/
  tradingagents.md

docs/source/lab/external_agents.rst
  - 增加 TradingAgents 集成文档入口
```

集成模块依赖 `ATLClient` 的公开类型和方法，不导入 ATL 后端内部模块。TradingAgents 的
导入被限制在生成器边界；加载已有决策文件和 ATL 回放在未安装 TradingAgents 时也必须
可用。

## 14. 测试策略

### 14.1 单元测试

- 五档评级正确映射到 BUY / HOLD / SELL。
- 缺少显式评级不会被上游默认值伪装成主动 Hold。
- 决策文件 round-trip、schema 版本、哈希和安全配置白名单。
- 敏感字段不会写入 artifact、日志或异常摘要。
- 单个日期第一次失败后重试，第二次失败记录错误 HOLD。
- 所有日期失败时不启动 ATL run。
- UTC Step 时间正确转换为 `America/New_York` 日期，周末和节假日由下一条实际 Step
  自然处理。
- 每条信号只执行一次，同日后续小时 HOLD；多个待执行信号只保留最新一个。
- 回放结束后仍有未处理记录时返回错误，并列出对应分析日期。
- BUY 补到 25% 目标而不是重复加 25%；SELL 全部清仓；空仓 SELL 不做空。
- 价格过高导致目标股数为 0 时产生带原因的 HOLD。

### 14.2 集成测试

使用假的 TradingAgentsGraph 和假的 ATLClient 跑完整流程：生成多日期决策文件、创建
单股票 run、按 Step 回放、产生买卖与 HOLD、返回结果摘要。测试不得访问网络，也不得
调用真实 LLM 或 Alpaca。

### 14.3 手动冒烟测试

用户提供真实 TradingAgents Key、ATL API Key 和 ATL 支持的短日期范围，至少完成：

```text
1 个真实 TradingAgents 有效评级
1 个 ATL run
至少 1 条可检查的 ATL 决策记录
完整收益曲线与指标
timeout_holds = 0
```

真实冒烟测试不进入 CI，避免泄露凭据和产生不可控费用。

## 15. 第一版验收标准

- 新用户只按集成文档即可在本地安装并运行。
- 一只支持的美股能完成“TradingAgents 决策 -> ATL 订单 -> 模拟成交 -> 收益曲线”。
- TradingAgents 和 ATL 的数据来源、版本、模型和决策文件哈希可追踪。
- T+1 规则通过测试，分析日信息不会在当天回填交易。
- 主动 HOLD、错误 HOLD、无仓可卖、已到目标仓位和 ATL 超时可以区分。
- LLM 与数据 Key 不进入 Git、决策文件、运行配置、错误日志或 ATL 请求。
- 同一决策文件重复回放时，适配器提交的动作序列相同；市场数据不变时 ATL 结果相同。
- 自动测试不需要安装 TradingAgents，不调用网络和付费模型。
- 现有非 TradingAgents 的 SDK、回测和 Dashboard 行为保持不变。

## 16. 后续迭代

第一版跑通并获得真实用户反馈后，才考虑：

1. 支持用户选择的 3 到 5 只股票。
2. 建立 ATL 原生单股票日频环境，减少小时回放适配。
3. 将 ATL 的标准市场快照注入 TradingAgents，实现更严格的同输入比较。
4. 增加网页配置、异步任务、用户预算上限和进度展示。
5. 在 ATL 结果页增加通用的集成来源与错误 HOLD 可视化组件。
6. 评估本地 API 服务或远程 Worker，但仍保持用户 Key 的明确所有权。

这些后续工作不阻塞第一版，也不能提前混入第一版实现。
