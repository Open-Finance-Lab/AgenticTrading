# iFinD A股 LLM 回测第一轮设计

## 1. 背景

ATL 已经可以通过 iFinD HTTP API 获取真实 A股 `60m` 历史行情，并使用规则 Agent 对
`A-Share Demo 6` 和 `CSI 300 Sample 20 (2026 H2)` 两个注册股票池执行模拟回测。
当前 Market Profile 将 iFinD 的 `llm_enabled` 固定为 `false`，前端也将 Model 控件锁定为
`Rule-based (data integration)`，因此 A股尚不能使用 ATL 已有的 LLM 决策链路。

本设计只覆盖下一次 Loop：让 `A-Share Demo 6` 可以主动选择 ATL 已有模型并完成一次
真实 LLM 回测，同时保留规则模式作为默认值。它不在本轮补齐全部 A股成交制度。

## 2. 已确认决策

1. 复用美股现有的 LLM 客户端、回测主循环、Portfolio Manager 和运行记录，不创建
   独立的 A股回测引擎。
2. `A-Share Demo 6` 支持 `rule_based` 和 `llm`，默认仍为 `rule_based`。
3. `CSI 300 Sample 20 (2026 H2)` 本轮继续只支持 `rule_based`。
4. A股 Model 下拉框复用美股已有的全部模型选项。
5. 用户显式选择 LLM 后，缺少凭证、调用失败、响应无法解析或返回非法股票代码都必须
   令本次回测失败，禁止静默降级为规则 Agent。
6. 真实验收只需选择一个已配置凭证的模型跑通，不要求逐个调用所有模型。

## 3. 目标

- 用户选择 iFinD 和 `A-Share Demo 6` 后，可以在规则模式与现有模型之间切换。
- LLM 模式在每个有效 `60m` 时间点真实调用所选模型。
- 模型只允许对固定6只 `.SH`/`.SZ` 股票生成合法决策。
- 运行记录能够证明决策来源，包含所选模型、调用次数、Token、估算费用和股票池来源。
- iFinD 规则回测、20只股票池、美股和 vn.py 原有行为保持兼容。

## 4. 非目标

本轮不实现：

- `CSI 300 Sample 20 (2026 H2)` 的 LLM 回测；
- `T+1`、100股一手、涨跌停、印花税和 A股专用手续费；
- 真实券商下单或任何真实资金交易；
- 真正沪深300指数基准；
- 为 A股复制一套独立的 Agent、投资组合或回测引擎；
- 保证每个界面列出的模型在本地都有可用凭证。

## 5. 方案选择

### 5.1 采用：统一回测链路加市场能力配置

Market Profile 从单一 `llm_enabled` 布尔判断演进为可以表达允许决策来源的能力配置。
概念行为如下：

```text
A-Share Demo 6:              rule_based, llm; default=rule_based
CSI 300 Sample 20 (2026 H2): rule_based;      default=rule_based
Alpaca DJIA 30:              保持现有行为
vn.py Simulation DJIA 30:    保持现有行为
```

前端负责展示能力，API 和 Engine 负责最终执行约束。即使调用者绕过页面直接请求 API，
不受支持的 `data_source + universe + decision_source` 组合也必须被拒绝。

### 5.2 未采用：只解除界面和 `llm_enabled` 限制

该方案改动最少，但无法明确表达规则/LLM 选择，也会继续保留缺凭证或模型失败时静默降级
的问题，无法证明回测结果确实由所选模型产生。

### 5.3 未采用：建立 A股专用 LLM 引擎

该方案会复制模型调用、回测循环、投资组合和存储逻辑，增加长期维护成本，并使 A股与
美股更难保持一致，不符合本轮范围。

## 6. 架构与组件边界

```text
Run Backtest modal
  -> POST /backtest/run
  -> Market Profile capability validation
  -> background runner
  -> backtest_hourly_agent.py
  -> HourlyBacktester
  -> existing LLM client + PortfolioManager
  -> database + Run Config
```

各组件职责：

- Market Profile：声明市场、股票池、时区、周期、默认决策来源和允许的决策来源。
- 前端：只展示已注册能力，并提交用户明确选择的决策来源与模型。
- Backtest API：合并请求参数，校验能力组合、模型格式与启动前配置。
- 后台运行器和 CLI：原样传播已验证的 `decision_source`、模型、股票池和周期。
- HourlyBacktester：根据显式决策来源初始化规则路径或 LLM 路径，并执行严格 LLM 模式。
- Portfolio Manager：继续负责组合状态、动作执行和权益曲线；严格模式下不得吞掉 LLM
  错误并回退规则决策。
- 数据库和 Run Config：记录并展示实际使用的决策来源，不根据界面选择进行推测。

## 7. 请求契约与兼容性

`BacktestRunRequest` 增加可选字段：

```text
decision_source: "rule_based" | "llm"
```

iFinD 6只股票的 LLM 请求包含：

```text
data_source: ifind_ashare
universe: a_share_demo_6
timeframe: 60m
decision_source: llm
model: <existing ATL model id>
```

兼容规则：

- iFinD 未传 `decision_source` 时使用 Profile 默认值 `rule_based`。
- iFinD 选择 `llm` 时必须提供或解析出有效模型 ID。
- iFinD 20只股票池选择 `llm` 时返回 `422`，不得创建后台任务。
- Alpaca 和 vn.py 未传该字段时沿用现有行为，避免本轮改变既有请求语义。
- `assets` 仍不能覆盖 iFinD 后端注册股票池。

后台运行器不再仅根据 Profile 的布尔值决定 `--use-llm`。它根据已验证的
`decision_source` 选择 `--use-llm` 或 `--no-llm`，并且只在 LLM 模式传递模型、策略
Prompt 和 Pipeline。

## 8. A股 LLM 上下文和决策约束

每个决策时间点继续使用现有组合状态、OHLCV 和技术指标。LLM Prompt 增加由 Profile
生成的最小市场上下文：

- 市场为中国 A股，数据时区为 `Asia/Shanghai`；
- 周期为 `60m`，当前任务是历史模拟回测，不会产生真实订单；
- 允许股票代码严格等于 `A_SHARE_DEMO_6_SYMBOLS`；
- 返回动作中的 symbol 必须保留 `.SH`/`.SZ` 后缀；
- 未产生交易机会时允许合法的 HOLD/空动作结果。

市场上下文应作为结构化、后端拥有的 Prompt 片段加入现有提示词，而不是信任浏览器传入
任意市场描述。自定义 Prompt 和 Pipeline 可以表达策略，但不能扩大股票允许列表。

`PortfolioManager` 已经接收 `allowed_symbols`；本轮在严格 LLM 路径中补足端到端约束：
模型响应必须可解析为既有决策结构，动作类型合法，所有 symbol 位于固定6只允许列表中。
任何违反约束的响应都抛出明确的 LLM 决策错误。

## 9. 严格错误处理

显式 A股 LLM 运行设置 `llm_required=true`。以下情况必须失败：

- 对应 Provider 的 API 凭证或 SDK 不可用；
- 模型 ID 为空、格式非法或网关无法使用该模型；
- LLM 请求超时、网关返回错误或调用异常；
- 响应无法解析为既有决策结构；
- 响应包含固定6只以外的股票代码或非法动作。

失败行为：

- 不调用规则 Agent 生成替代决策；
- 不把该运行保存为成功的 Agent 回测；
- 后台状态向页面返回可理解的失败摘要；
- 错误信息经过现有脱敏边界，不能包含 iFinD Token、模型 API Key、Authorization Header
  或完整上游敏感响应；
- 不改变本轮范围以外的美股历史降级行为。

能力组合错误应在创建后台任务前返回 `422`。缺少运行所需凭证属于服务不可用错误，应在
尽可能早的边界返回明确错误；运行期间的模型/解析错误由回测进程以非零状态退出并传播到
回测状态接口。

## 10. 前端行为

选择 `A-Share Demo 6`：

- Model 控件可用；
- 第一项为 `Rule-based` 并默认选中；
- 后续选项复用美股当前模型列表；
- 选择模型后提交 `decision_source=llm` 和所选模型 ID；
- 选择规则项后提交 `decision_source=rule_based`，不提交 LLM 模型作为实际运行配置。

选择 `CSI 300 Sample 20 (2026 H2)`：

- 自动恢复 `Rule-based`；
- Model 控件保持禁用；
- 不保留先前6只股票池中选择的 LLM 作为隐藏运行状态。

Run Config 显示 Market data、Universe、Symbols、Timeframe、Decision source 和实际模型。
运行完成后显示的模型与数据库事实一致；失败运行不得显示成规则策略成功结果。所有新增
界面文字继续使用英文。

## 11. 数据和运行记录

成功的 A股 LLM Agent 运行至少记录：

```text
data_source=ifind_ashare
market=CN
universe=a_share_demo_6
timeframe=60m
timezone=Asia/Shanghai
symbols=<exact six-symbol list>
decision_source=llm
llm_model=<selected model>
llm_calls>0
input_tokens>=0
output_tokens>=0
est_cost_usd>=0
```

规则运行继续记录 `decision_source=rule_based`、`llm_calls=0` 和规则模型标识。基准仍为
现有等权 buy-and-hold，iFinD 仍不生成 DJIA 基准。

## 12. 测试设计

### 12.1 Profile 和 API

- 6只 Profile 允许规则和 LLM，默认规则；
- 20只 Profile 只允许规则；
- API 接受6只 + LLM，拒绝20只 + LLM；
- 默认 iFinD 请求仍调度规则回测；
- 非法 decision source、非法模型或缺少必需配置在调度前失败；
- iFinD 的 `assets` 不能改变注册股票池。

### 12.2 Engine 和 LLM

- 使用假的 LLM 客户端跑完整6只 A股路径，并证明每个有效时间点确实调用模型；
- Prompt 包含 CN、上海时区、`60m` 和准确的6只允许代码；
- 合法 BUY/SELL/HOLD 结果可以进入现有模拟成交和权益计算；
- 非法 JSON、非法动作、越权代码、超时和上游错误均使严格运行失败；
- 失败后规则决策函数调用次数为0；
- 规则模式的 LLM 调用次数为0；
- 元数据准确区分规则和 LLM。

### 12.3 前端

- 6只股票池展示 Rule-based 与现有模型，默认 Rule-based；
- 切换到20只后恢复并锁定 Rule-based；
- 切回6只不会错误提交20只股票池或隐藏的模型状态；
- 请求 Payload 包含明确的 `decision_source`；
- Run Config 根据实际运行记录显示决策来源和模型。

### 12.4 集成与真实验证

- 离线集成测试使用假的 iFinD 响应和假的 LLM 客户端，覆盖 API、后台参数、CLI、Engine、
  Portfolio Manager、数据库和图表读取；
- 全量后端测试、`node --check`、`compileall` 和 `git diff --check` 通过；
- 真实 iFinD 返回准确6只股票，每只至少50根有效 `60m` Bar；
- 使用一个已配置凭证的真实模型完成一次运行；
- 真实运行记录满足 `llm_calls > 0`，模型、Token、费用和决策来源正确；
- 再运行一次规则模式，确认 `llm_calls=0`；
- 浏览器控制台没有新增错误，页面能区分两次运行。

## 13. 成功标准

本 Loop 只有在以下条件同时满足时完成：

1. `A-Share Demo 6` 的 Model 控件可选择规则或现有 LLM，默认规则。
2. 真实 iFinD + 一个真实 LLM 完成至少一次端到端回测。
3. 数据库证明该运行实际调用了所选模型，不存在规则降级。
4. 20只股票池的 LLM 请求在前端和后端均被阻止。
5. 规则回测、美股、vn.py 和现有完整测试没有回归。
6. 仓库中不包含任何 iFinD 或模型凭证、实验数据库、临时日志或真实上游响应。

## 14. 后续 Loop

本设计完成后，后续独立 Loop 再处理：

1. 20只 A股的 LLM 成本、Prompt 规模和性能验证；
2. A股 `T+1`、手数、涨跌停和费用模型；
3. 真正沪深300指数基准；
4. 更完整的 A股公司元数据和市场事件上下文。
