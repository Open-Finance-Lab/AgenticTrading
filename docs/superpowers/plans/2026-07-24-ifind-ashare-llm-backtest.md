# iFinD A股 LLM 回测第一轮实施计划

> 本计划按 Loop Engineering 执行。每个产品代码任务都遵循“失败测试 -> 最小实现 -> 聚焦验证 -> 本地提交”，并在最后使用真实 iFinD 和一个真实模型做端到端验收。

**目标：** 让 `A-Share Demo 6` 默认保持规则回测，同时允许用户主动选择 ATL 现有模型并完成真实 LLM 回测；`CSI 300 Sample 20 (2026 H2)` 本轮继续只允许规则模式。

**架构：** 复用现有 Backtest API、后台子进程、`HourlyBacktester`、LLM Provider、`PortfolioManager` 和数据库。Market Profile 声明默认及允许的决策来源；请求显式携带 `decision_source`；A股显式 LLM 路径启用市场感知 Prompt 和严格失败模式，不允许回退规则 Agent。

**技术栈：** Python 3.10+、FastAPI、Pydantic、Anthropic Messages 兼容 LLM Provider、pandas、pytest、原生 JavaScript、iFinD HTTP API、SQLite。

**设计规格：** `docs/superpowers/specs/2026-07-24-ifind-ashare-llm-backtest-design.zh-CN.md`

## 全局约束

- 在 `feat/ifind-ashare-market-data` 分支实施，基线包含本地提交 `b781934` 和设计提交 `3bf34a3`。
- 不推送 GitHub，除非用户后续明确要求。
- `dashboard/.env`、iFinD Token、模型 API Key、真实上游响应、实验数据库和日志不得提交。
- 运行中的 `dashboard/storage/data/backtest.db` 属于本地状态；测试使用临时数据库，不覆盖或提交它。
- `A-Share Demo 6` 默认 `rule_based`；只有用户显式选择模型时才进入严格 LLM 路径。
- `CSI 300 Sample 20 (2026 H2)` 的 LLM 请求必须在前端和后端均被阻止。
- 不改变 Alpaca 和 vn.py 未传 `decision_source` 时的历史行为。
- 不在本轮实现 `T+1`、100股一手、涨跌停、印花税、A股专用手续费、真实下单或沪深300指数基准。
- A股 LLM 失败不得静默改成规则结果；美股现有兼容性回退不在本轮修改范围。
- 自动测试不得访问真实 iFinD 或模型网络；真实验证单独执行并控制费用。
- 所有新增界面文字使用英文。
- 每个任务只提交列出的相关文件，便于维护者审阅和回滚。

---

## Task 1：让 Market Profile 表达默认和允许的决策来源

**文件：**

- 修改：`dashboard/backend/infrastructure/market_data/profiles.py`
- 修改：`dashboard/backend/tests/infrastructure/market_data/test_market_profiles.py`
- 修改：`dashboard/backend/tests/infrastructure/market_data/test_provider.py`

**接口目标：**

Market Profile 增加：

```python
default_decision_source: str
allowed_decision_sources: tuple[str, ...]
```

并提供一个统一解析函数：

```python
resolve_decision_source(profile, requested: str | None) -> str
```

旧调用读取 `profile.decision_source` 和 `profile.llm_enabled` 时保持兼容：前者等于默认决策来源，后者表示“默认路径是否为 LLM”，不能因为6只 A股允许主动选择 LLM 就让现有 iFinD 请求自动调用模型。

- [ ] **Step 1：写失败测试**

在 `test_market_profiles.py` 覆盖：

- `A-Share Demo 6` 默认 `rule_based`，允许 `rule_based` 和 `llm`；
- `CSI 300 Sample 20 (2026 H2)` 默认且只允许 `rule_based`；
- Alpaca 和 vn.py 默认及历史 `llm_enabled` 行为不变；
- `resolve_decision_source` 在未传值时返回 Profile 默认值；
- 6只 + `llm` 合法，20只 + `llm` 抛出包含 source/universe/decision source 的 `ValueError`；
- 未知决策来源被拒绝。

在 `test_provider.py` 保留默认 Profile 和旧属性兼容断言。

- [ ] **Step 2：运行测试并确认失败**

```bash
python -m pytest -q \
  dashboard/backend/tests/infrastructure/market_data/test_market_profiles.py \
  dashboard/backend/tests/infrastructure/market_data/test_provider.py
```

预期：FAIL，现有 Profile 只有单一 `decision_source` 和 `llm_enabled` 字段，无法表达“默认规则但允许主动选 LLM”。

- [ ] **Step 3：实现最小能力契约**

- 定义 `RULE_BASED_DECISION_SOURCE` 和 `LLM_DECISION_SOURCE` 常量；
- 为四个已注册 Profile 填写默认及允许来源；
- 让兼容属性从默认来源计算；
- 实现统一解析函数，不把决策来源判断复制到 API、CLI 和 Engine；
- 错误信息只包含非敏感的注册配置。

- [ ] **Step 4：运行聚焦测试**

```bash
python -m pytest -q \
  dashboard/backend/tests/infrastructure/market_data/test_market_profiles.py \
  dashboard/backend/tests/infrastructure/market_data/test_provider.py
```

预期：PASS。

- [ ] **Step 5：提交**

```bash
git add dashboard/backend/infrastructure/market_data/profiles.py \
  dashboard/backend/tests/infrastructure/market_data/test_market_profiles.py \
  dashboard/backend/tests/infrastructure/market_data/test_provider.py
git commit -m "feat(backtest): add profile decision capabilities"
```

---

## Task 2：增加市场感知 Prompt 和严格 LLM 决策模式

**文件：**

- 修改：`dashboard/backend/infrastructure/llm/backtest_harness.py`
- 修改：`dashboard/backend/domain/backtesting/portfolio_manager.py`
- 修改：`dashboard/backend/tests/llm/test_backtest_harness.py`
- 修改：`dashboard/backend/tests/backtesting/test_portfolio_manager_move.py`

**接口目标：**

保留现有调用默认行为，同时增加可选参数：

```python
request_trading_decision(..., market_context: dict | None = None)
PortfolioManager.make_trading_decision_with_llm(
    ...,
    market_context: dict | None = None,
    strict_llm: bool = False,
)
```

新增固定、脱敏的 `LLMDecisionError`。`strict_llm=False` 时保持当前美股回退逻辑；`strict_llm=True` 时配置、调用、解析或越权错误必须抛出该异常。

- [ ] **Step 1：写失败测试**

在 `test_backtest_harness.py` 覆盖：

- 未传市场上下文时，请求继续使用现有 DJIA System Prompt；
- 传入 CN 上下文时，System Prompt 明确为 Chinese A-share market，包含上海时区、`60m` 和历史模拟语义，不再声称分析 DJIA；
- Provider 请求参数、模型、Token 上限和温度的旧行为不变；
- System Prompt 不包含凭证或任意浏览器自由文本。

在 `test_portfolio_manager_move.py` 使用假的 LLM 客户端覆盖：

- 市场上下文同时进入固定 System Prompt 和 `market_snapshot`；
- 严格模式下缺少客户端、调用异常、无文本、解析失败、Pipeline 返回 `None`、未知动作、动作过多和越权股票均抛出 `LLMDecisionError`；
- 严格模式抛错后规则决策函数调用次数为0；
- 合法 `{"actions": []}` 被视为模型 HOLD，不回退且 `llm_decisions` 增加；
- 合法6只股票 BUY/SELL/HOLD 继续转换成既有动作结构；
- 非严格模式保留当前规则回退、截断或跳过行为。

- [ ] **Step 2：运行测试并确认失败**

```bash
python -m pytest -q \
  dashboard/backend/tests/llm/test_backtest_harness.py \
  dashboard/backend/tests/backtesting/test_portfolio_manager_move.py
```

预期：FAIL，当前 System Prompt 固定写着 DJIA，所有异常和空动作都会回退规则路径。

- [ ] **Step 3：实现市场感知 System Prompt**

- 保留现有 `SYSTEM_PROMPT` 作为无上下文默认值；
- 添加只从后端结构化市场字段生成 System Prompt 的 helper；
- CN Prompt 固定描述 A股、时区、周期、历史模拟和无真实订单；
- `request_trading_decision` 只选择 Prompt，不改变 Provider 调用协议；
- 将同一市场上下文写入 `market_snapshot["market"]`，保证普通 Prompt、自定义 Prompt 和 Pipeline 都能读取。

- [ ] **Step 4：实现严格决策边界**

- 定义 `LLMDecisionError`，错误消息使用固定分类，不包含原始响应全文；
- 每个现有回退点先检查 `strict_llm`：严格模式抛错，非严格模式保持旧行为；
- 严格模式对动作数量、动作类型和 symbol allow-list 做完整批次校验后再转换，避免处理一半才失败；
- 明确区分“解析失败”和“合法空动作 HOLD”；
- 只在模型响应成功解析并通过严格校验后增加 `llm_decisions`。

- [ ] **Step 5：运行聚焦和 LLM 回归测试**

```bash
python -m pytest -q \
  dashboard/backend/tests/llm \
  dashboard/backend/tests/backtesting/test_portfolio_manager_move.py \
  dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py
```

预期：PASS。

- [ ] **Step 6：提交**

```bash
git add dashboard/backend/infrastructure/llm/backtest_harness.py \
  dashboard/backend/domain/backtesting/portfolio_manager.py \
  dashboard/backend/tests/llm/test_backtest_harness.py \
  dashboard/backend/tests/backtesting/test_portfolio_manager_move.py
git commit -m "feat(backtest): add strict market-aware LLM decisions"
```

---

## Task 3：让 Engine 和 CLI 显式执行 A股 LLM 模式

**文件：**

- 修改：`dashboard/backend/domain/backtesting/engine.py`
- 修改：`dashboard/scripts/backtest_hourly_agent.py`
- 修改：`dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py`
- 修改：`dashboard/backend/tests/backtesting/test_engine_move.py`
- 修改：`dashboard/backend/tests/backtesting/test_canonical_consumers.py`

**接口目标：**

```python
HourlyBacktester(..., decision_source: str | None = None)
```

CLI 增加：

```text
--decision-source rule_based|llm
```

保留 `--use-llm`/`--no-llm` 作为旧调用兼容入口；显式 `--decision-source` 优先，并必须通过 Profile 能力校验。

- [ ] **Step 1：写失败测试**

在 `test_ifind_ashare_engine.py` 覆盖：

- 未传决策来源的6只和20只 iFinD Engine 仍为规则模式；
- 6只 + `decision_source="llm"` 创建真实 LLM 路径，传给 Portfolio Manager 的 `strict_llm=True`；
- 20只 + `llm` 在创建 Provider 或模型客户端前失败；
- A股市场上下文由 Profile 构造，包含准确6只 symbol、CN、上海时区和 `60m`；
- 显式 A股 LLM 缺少 SDK/客户端时抛出配置错误，不回退；
- 规则模式不初始化 LLM 客户端；
- 运行元数据使用实际 `decision_source`，LLM 模式记录所选模型和调用数据；
- 基准继续使用等权 buy-and-hold，且不调用 LLM。

在 `test_engine_move.py` 和 `test_canonical_consumers.py` 覆盖旧构造签名和 Alpaca 行为不变。

- [ ] **Step 2：运行测试并确认失败**

```bash
python -m pytest -q \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py \
  dashboard/backend/tests/backtesting/test_engine_move.py \
  dashboard/backend/tests/backtesting/test_canonical_consumers.py
```

预期：FAIL，Engine 当前只用 `profile.llm_enabled` 决定模式，且 A股会强制关闭 LLM。

- [ ] **Step 3：实现 Engine 决策来源解析**

- 通过 Task 1 的统一 helper 解析决策来源；
- 未显式传入时保持现有 `use_llm` 和 Profile 默认兼容语义；
- 仅显式 A股 LLM 设置严格模式；
- 严格 LLM 初始化失败直接抛错，非严格旧路径继续保留历史降级；
- `run_agent_backtest` 将 Profile 市场上下文和严格标志传给 Portfolio Manager；
- `_agent_run_metadata()` 写入实际决策来源，不能继续照抄 Profile 默认值；
- Agent 失败时不创建成功 Agent 运行记录，后续基准不伪装该 Agent 成功。

- [ ] **Step 4：实现 CLI 传播**

- 添加可选 `--decision-source`；
- 显式值与 `--use-llm`/`--no-llm` 冲突时给出清晰 parser error；
- 解析 Profile 后立即验证 source/universe/decision source 组合；
- 将最终值传给 `HourlyBacktester`；
- 日志输出实际 `Decision source`，不打印凭证。

- [ ] **Step 5：运行聚焦测试**

```bash
python -m pytest -q \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py \
  dashboard/backend/tests/backtesting/test_engine_move.py \
  dashboard/backend/tests/backtesting/test_canonical_consumers.py
```

预期：PASS。

- [ ] **Step 6：提交**

```bash
git add dashboard/backend/domain/backtesting/engine.py \
  dashboard/scripts/backtest_hourly_agent.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py \
  dashboard/backend/tests/backtesting/test_engine_move.py \
  dashboard/backend/tests/backtesting/test_canonical_consumers.py
git commit -m "feat(backtest): run explicit iFinD LLM decisions"
```

---

## Task 4：让 API 和后台运行器校验并传播 `decision_source`

**文件：**

- 修改：`dashboard/backend/api/routers/backtests.py`
- 修改：`dashboard/backend/infrastructure/llm/providers/__init__.py`
- 修改：`dashboard/backend/tests/test_backtests_router.py`
- 修改：`dashboard/backend/tests/test_market_data_features.py`
- 修改：`dashboard/backend/tests/llm/test_providers.py`

**请求变化：**

`BacktestRunRequest`、查询参数和后台函数增加：

```python
decision_source: Literal["rule_based", "llm"] | None
```

- [ ] **Step 1：写失败测试**

在 `test_market_data_features.py` 覆盖：

- iFinD 6只请求未传来源时按 `rule_based` 调度；
- 6只 + `llm` + 合法模型按 LLM 调度；
- 20只 + `llm` 返回 `422` 且后台 spy 调用次数为0；
- 规则模式不要求模型凭证；
- LLM 模式缺 SDK/有效 Provider 客户端时在调度前返回明确服务不可用错误；
- 错误文本不包含任何环境变量中的测试 secret；
- Query 和 JSON Body 合并后的最终 `decision_source` 遵循现有 body override 规则。

在 `test_backtests_router.py` 覆盖：

- 现有九个前端模型 ID 在6只 A股 LLM 请求中通过格式校验；
- LLM 模式模型为空或非法时返回 `422`；
- 背景 runner 只在 LLM 模式传 `--decision-source llm`、`--model`、Prompt 和 Pipeline；
- 规则模式传 `--decision-source rule_based`，不传实际模型配置；
- `run_backtest_background` 将 source、universe、timeframe 和 decision source 原样传递给 CLI；
- 子进程非零退出时状态为失败，错误经过现有脱敏函数。

在 `test_providers.py` 覆盖一个无网络的 LLM 可用性预检 helper：

- SDK/选中 Provider Key 不可用时抛出固定配置错误；
- CommonStack/Anthropic 自动解析继续沿用现有顺序；
- 预检不打印或返回 API Key。

- [ ] **Step 2：运行测试并确认失败**

```bash
python -m pytest -q \
  dashboard/backend/tests/test_market_data_features.py \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/llm/test_providers.py
```

预期：FAIL，API 当前没有显式决策来源，后台仅按 `profile.llm_enabled` 选择 CLI 标志。

- [ ] **Step 3：实现无网络 LLM 配置预检**

- 在 LLM Provider 注册表旁增加单一预检入口；
- 复用现有 integration 自动解析，不在 Router 中重复判断各种 Key；
- 只确认 SDK、Provider 配置和客户端可构造，不发网络请求；
- 错误只说明缺少哪类配置，不包含值。

- [ ] **Step 4：实现 API 能力校验和后台传播**

- Body 和 Query 都支持 `decision_source`；
- 合并请求后使用 Task 1 helper 校验；
- 只有 `decision_source=llm` 才要求模型并执行 LLM 预检；
- 将已解析的来源传入后台任务；
- 后台命令使用 `--decision-source`，而不是根据 Profile 布尔值自行推断；
- Prompt、Pipeline 和 model 只进入 LLM 子进程；
- 保持31天、模型格式、并发和速率限制不变。

- [ ] **Step 5：运行聚焦测试**

```bash
python -m pytest -q \
  dashboard/backend/tests/test_market_data_features.py \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/llm/test_providers.py
```

预期：PASS。

- [ ] **Step 6：提交**

```bash
git add dashboard/backend/api/routers/backtests.py \
  dashboard/backend/infrastructure/llm/providers/__init__.py \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/test_market_data_features.py \
  dashboard/backend/tests/llm/test_providers.py
git commit -m "feat(api): accept iFinD backtest decision source"
```

---

## Task 5：开放6只 A股 Model 控件并保持20只规则锁定

**文件：**

- 修改：`dashboard/frontend/app.html`
- 修改：`dashboard/frontend/app.js`
- 按需修改：`dashboard/frontend/styles.css`
- 修改：`dashboard/backend/tests/test_ifind_ashare_frontend.py`
- 修改：`dashboard/backend/tests/test_backtests_router.py`

- [ ] **Step 1：写失败测试**

在 `test_ifind_ashare_frontend.py` 覆盖：

- iFinD 6只股票池的 Model 控件可用，并包含 `Rule-based` 和现有九个模型；
- 进入 iFinD 或重新打开 modal 时默认 `Rule-based`；
- 选择20只后自动切回规则并禁用，隐藏状态不保留旧 LLM 运行选择；
- 从20只切回6只后重新允许选择模型；
- 6只规则请求发送 `decision_source=rule_based` 且不发送模型作为实际配置；
- 6只模型请求发送 `decision_source=llm`、选中模型、Agent Prompt/Pipeline；
- 20只请求始终发送规则来源；
- `launchConfig` 和 Run Config 根据实际来源显示 `Rule-based` 或模型名称；
- 退出 iFinD 后，美股模型选择状态恢复且 vn.py 继续锁定规则模式。

在 Router 测试中继续把 HTML 里的模型列表与后端接受列表绑定，避免页面新增模型后 API 自己拒绝。

- [ ] **Step 2：运行测试并确认失败**

```bash
python -m pytest -q \
  dashboard/backend/tests/test_ifind_ashare_frontend.py \
  dashboard/backend/tests/test_backtests_router.py
```

预期：FAIL，当前 `syncMarketDataSourceUI()` 对所有 iFinD 股票池禁用 Model，`runBacktest()` 也强制不提交模型或 Pipeline。

- [ ] **Step 3：实现股票池能力驱动的控件状态**

- 为前端注册的两个 iFinD universe 增加是否允许 LLM 的静态能力；
- 将 `Rule-based` 作为 iFinD 临时选项管理，进入 iFinD 默认选中；
- `renderIFindAshareUniverse()` 同步 Model 状态；
- 6只启用下拉框，20只强制规则并禁用；
- 离开 iFinD 时移除临时规则项并恢复进入前的美股模型；
- 不增加解释功能的额外页面卡片或营销式提示。

- [ ] **Step 4：实现显式请求 Payload**

- 从最终选项计算 `decisionSource`，不再使用 `isRuleBasedSource = isSimulation || isIFind`；
- A股 LLM 路径加载所选 Agent 的 Prompt/Pipeline；
- 规则路径不把隐藏模型写成实际运行模型；
- Payload、查询参数、launch config 和运行配置使用同一个最终值；
- 仍由后端固定 iFinD symbols，浏览器资产列表不成为授权来源。

- [ ] **Step 5：运行前端静态和语法测试**

```bash
python -m pytest -q \
  dashboard/backend/tests/test_ifind_ashare_frontend.py \
  dashboard/backend/tests/test_backtests_router.py
node --check dashboard/frontend/app.js
```

预期：PASS。

- [ ] **Step 6：提交**

```bash
git add dashboard/frontend/app.html \
  dashboard/frontend/app.js \
  dashboard/frontend/styles.css \
  dashboard/backend/tests/test_ifind_ashare_frontend.py \
  dashboard/backend/tests/test_backtests_router.py
git commit -m "feat(frontend): enable LLM models for A-share demo 6"
```

若 `styles.css` 没有实际变化，不得为凑文件而修改或暂存它。

---

## Task 6：离线端到端测试、完整回归和真实验收

**文件：**

- 修改：`dashboard/backend/tests/integration/test_ifind_ashare_backtest.py`
- 按失败覆盖需要修改：`dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py`
- 产品代码：仅在集成测试发现已批准设计缺口时做最小修复，并回到对应 Task 的聚焦测试。

- [ ] **Step 1：写离线端到端失败测试**

扩展现有 iFinD 集成测试，使用：

- 准确6只股票、每只至少52根合法 `60m` Bar 的假 iFinD 响应；
- 每个时间点返回合法 JSON 的假 Anthropic Messages 客户端；
- 临时 SQLite 数据库；
- API/后台 runner 使用同步测试替身，禁止真实网络和真实子进程。

覆盖完整链路：

```text
API decision_source=llm
  -> Profile capability
  -> background args
  -> CLI/Engine
  -> market-aware Prompt
  -> strict PortfolioManager
  -> Agent run + equal-weight baseline
  -> database metadata/trades/equity
  -> chart API
```

断言：

- 真实决策时间点均由假 LLM 驱动，`llm_calls > 0` 且规则 fallback 调用数为0；
- 模型只能看到并返回固定6只代码；
- Agent 和等权基准成功，DJIA 基准不存在；
- Agent 元数据为 `decision_source=llm`，模型、Token、费用、股票池和时区正确；
- 同一集成环境中的规则请求 `llm_calls=0`；
- 20只 + LLM 在进入该链路前失败。

- [ ] **Step 2：运行离线集成测试并确认失败，再做最小修复**

```bash
python -m pytest -q \
  dashboard/backend/tests/integration/test_ifind_ashare_backtest.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py
```

首次预期：若前五个任务仍有传播或存储缺口则 FAIL。只修复与已批准设计直接相关的缺口，然后重跑至 PASS；不得在此顺带增加 A股交易规则。

- [ ] **Step 3：提交集成测试和必要最小修复**

```bash
git add dashboard/backend/tests/integration/test_ifind_ashare_backtest.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py
git commit -m "test(backtest): cover iFinD A-share LLM workflow"
```

如果集成测试要求修改产品文件，提交前将对应文件明确加入命令并在提交说明中如实体现；不得使用宽泛 `git add .`。

- [ ] **Step 4：运行完整自动验证**

```bash
python -m pytest -q dashboard/backend/tests
node --check dashboard/frontend/app.js
python -m compileall -q dashboard/backend dashboard/scripts
git diff --check
```

预期：完整测试通过；已知第三方弃用 warning 可以记录，但不得出现本功能新增失败。

- [ ] **Step 5：运行真实 iFinD + 一个真实模型**

使用被 Git 忽略的本地 `.env`，不打印任何凭证。先做只读预检：

- `ENABLE_IFIND_ASHARE=true`；
- iFinD Token 可用；
- 至少一个现有 UI 模型对应的 Provider 客户端可构造；
- 使用实验用临时数据库，不修改或提交仓库数据库。

固定真实验收：

- source：`ifind_ashare`；
- universe：`a_share_demo_6`；
- timeframe：`60m`；
- decision source：`llm`；
- 时间窗口足以让每只股票返回至少50根 Bar；
- 模型：从已配置 Provider 中选择费用较低且现有美股路径已支持的一个模型；
- 不进行真实下单。

通过条件：

- 6/6 symbols，且每只至少50根有效 Bar；
- 回测完整结束，`llm_calls > 0`；
- 运行记录中的模型、Token、费用和 `decision_source=llm` 正确；
- 交易 symbol 全部位于固定6只 allow-list；
- 没有规则 fallback；
- Agent 和等权基准曲线可读取。

如真实模型失败，保留脱敏的错误类别和失败步骤，修复后从相关聚焦测试重新进入 Loop；不得把规则结果当作成功。

- [ ] **Step 6：运行真实规则对照**

使用相同 iFinD 股票池和时间窗口选择 `Rule-based`。通过条件：

- `llm_calls=0`；
- `decision_source=rule_based`；
- 规则 Agent 和等权基准成功；
- 证明开放 LLM 没有改变默认规则路径。

- [ ] **Step 7：浏览器 QA**

在新的空闲端口启动当前代码，使用浏览器验证：

- 桌面端和 `390x844` 移动端 modal 无横向溢出或文字重叠；
- 6只默认规则、可选模型；
- 20只自动规则并锁定；
- 6只真实 LLM 运行的 Run Config 和图表正确；
- 规则对照显示正确；
- 控制台无新增 error/warning。

结束时只保留当前代码的测试页面，旧端口不作为交付链接。

- [ ] **Step 8：最终仓库审计**

```bash
git status --short --branch
git diff --check
git log --oneline --decorate -10
```

同时检查：

- `.env`、数据库、日志和临时响应没有进入任何提交；
- 没有冲突标记；
- 所有产品和测试提交都位于当前功能分支；
- 不推送远端。

## 完成定义

只有以下条件全部满足，本 Loop 才完成：

1. 6只 A股默认规则，但可以选择现有任意 Model；
2. 20只 A股仍严格限制为规则模式；
3. 真实 iFinD + 一个真实模型完成端到端回测；
4. 数据库证明模型确实被调用，且不存在规则 fallback；
5. 规则对照、Alpaca、vn.py 和完整测试无回归；
6. 浏览器桌面与移动端通过 QA；
7. 仓库不包含凭证和实验运行数据；
8. 所有改动只在本地分支，等待用户决定是否推送或创建 PR。
