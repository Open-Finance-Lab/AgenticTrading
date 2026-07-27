# iFinD 全股票池 Agent/LLM 与资金校验实施计划

> 本计划按 Loop Engineering 执行：失败测试 -> 最小实现 -> 聚焦验证 -> 浏览器验收 -> 完整回归 -> 本地提交。

**目标：** 让 `A-Share Demo 6` 和 `CSI 300 Sample 20 (2026 H2)` 都默认使用规则决策，同时允许用户主动选择 ATL 现有的 9 个 Agent/LLM 模型；修正默认资金 `1000` 的 HTML 原生校验不一致。

**架构：** 复用现有 Market Profile、Backtest API、严格 A 股 LLM Engine、Portfolio Manager 和前端 Model 控件。后端 Profile 是能力授权边界，前端配置镜像相同能力；不新增 Agent 引擎或 Provider。

**设计规格：** `docs/superpowers/specs/2026-07-24-ifind-all-universes-llm-capital-validation-design.zh-CN.md`

## 全局约束

- 在 `feat/ifind-ashare-market-data` 分支实施。
- 不修改、暂存或提交用户本地的 `dashboard/storage/data/backtest.db`。
- 不提交 iFinD Token、模型 API Key、真实上游响应或日志。
- 两个 A 股股票池均默认 `rule_based`；只有显式选择模型才进入 `llm`。
- 显式 A 股 LLM 继续采用严格模式，失败时不得回退规则 Agent。
- 规则模式不传生效的模型/Pipeline，且 `llm_calls=0`。
- 不改变 iFinD OHLCV 数据链路、60 分钟粒度、股票白名单或等权买入持有基准。
- 不实现真实交易、券商连接或 A 股专用撮合制度。
- 自动测试不得访问真实 iFinD 或付费 LLM 网络。
- 所有新增界面文案使用英文。

---

## Task 1：后端开放 Sample 20 的 LLM 能力

**文件：**

- 修改：`dashboard/backend/infrastructure/market_data/profiles.py`
- 修改：`dashboard/backend/tests/infrastructure/market_data/test_market_profiles.py`
- 修改：`dashboard/backend/tests/test_market_data_features.py`
- 修改：`dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py`

1. 先把现有“Sample 20 只允许规则决策”的断言改成“默认规则，同时允许规则和 LLM”。
2. 增加 Sample 20 显式 `decision_source=llm` 能通过 Profile/API/Engine 配置的断言。
3. 运行聚焦测试，确认它们因后端 Profile 仍为 rule-only 而失败。
4. 只修改 Sample 20 的 `allowed_decision_sources`，加入 `LLM_DECISION_SOURCE`，默认值保持 `RULE_BASED_DECISION_SOURCE`。
5. 运行聚焦测试，确认未知股票池、未知决策来源、Demo 6 和美股旧行为没有回归。

聚焦命令：

```bash
python -m pytest -q \
  dashboard/backend/tests/infrastructure/market_data/test_market_profiles.py \
  dashboard/backend/tests/test_market_data_features.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py
```

---

## Task 2：前端开放两个股票池并修正 `1000` 原生有效性

**文件：**

- 修改：`dashboard/frontend/app.js`
- 修改：`dashboard/frontend/app.html`
- 修改：`dashboard/backend/tests/test_ifind_ashare_frontend.py`

1. 先修改/增加前端源码契约测试，要求 Demo 6 和 Sample 20 都声明
   `allowedDecisionSources: ['rule_based', 'llm']`。
2. 增加输入框契约测试，要求 `backtestInitialCapital` 使用 `min="1"`、
   `step="1"`、`max="10000"` 和 `value="1000"`。
3. 增加或复用 Node DOM 测试，确认选择两个股票池时 Model 控件都启用，切换股票池后都重置为 `Rule-based`。
4. 运行测试，确认 Sample 20 能力和 `step="100"` 两处断言失败。
5. 最小修改前端 Profile 和资金输入框；保留现有 JavaScript 的正数及 10,000 上限校验。
6. 运行 `node --check` 和聚焦前端测试。

聚焦命令：

```bash
python -m pytest -q dashboard/backend/tests/test_ifind_ashare_frontend.py
node --check dashboard/frontend/app.js
```

---

## Task 3：Sample 20 严格 LLM 全链路与回归验收

**文件：**

- 修改：`dashboard/backend/tests/integration/test_ifind_ashare_backtest.py`
- 修改：`dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py`（仅在需要补充 Engine 断言时）

1. 使用假的 iFinD Provider 和假的 LLM Provider，为 Sample 20 增加一次完整严格 LLM 回测。
2. 断言使用 20 股白名单、`decision_source=llm`、所选模型及 Pipeline，并且确实发生 LLM 调用。
3. 断言非法股票或 Provider 失败会明确终止，不产生规则回退结果。
4. 运行 A 股 Profile、API、Engine、集成和前端测试集合。
5. 运行完整 `dashboard/backend/tests`，执行 Python 编译、JavaScript 语法和 `git diff --check`。
6. 使用真实 iFinD 凭证分别跑 Demo 6 和 Sample 20 的规则模式 smoke test，确认真实数据链未回归且 `llm_calls=0`。
7. 在浏览器中确认两个股票池均能选择 9 个模型、切换后默认规则、`1000` 的 `validity.valid=true`，并检查控制台错误。
8. 审计最终 diff，确保不包含 `backtest.db`、凭证或无关文件。

## 提交策略

计划文档单独提交。实现完成后，将相关代码和测试作为一个聚焦提交，避免再次拆分已经非常小的 Profile/UI 对齐改动。全部验证通过后再推送功能分支并创建指向 `Open-Finance-Lab/AgenticTrading:main` 的 PR。

## 完成标准

- 两个 A 股股票池都可选择 `Rule-based` 和 9 个现有模型。
- 两个股票池切换后都默认 `Rule-based`。
- Sample 20 的严格 LLM 离线端到端测试通过，且不存在静默规则回退。
- 默认资金 `1000` 的浏览器原生有效性为真，实际回测行为保持不变。
- 聚焦测试、完整测试、真实 iFinD 规则 smoke test 和浏览器 QA 全部通过。
- 最终提交不包含 `dashboard/storage/data/backtest.db` 或任何凭证。
