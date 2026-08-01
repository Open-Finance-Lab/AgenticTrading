# iFinD A 股整手交易与资金适配实施计划

**目标：** 在 PR #272 的 T+1 基础上，为两个 iFinD A 股股票池增加 100 股整手校验、资金不足拒单、统一订单事件和可审计的 Trading Log，同时保持 `$1,000` 默认本金、可选更高本金、Alpaca 美股和 vn.py 的现有行为。

**设计规格：** `docs/superpowers/specs/2026-08-01-ifind-ashare-lot-size-design.md`

**实施方法：** 每个任务按“失败测试 -> 最小实现 -> 聚焦测试 -> 命名文件提交”执行。不得把本功能追加到 PR #272；必须等 #272 合并后，从最新 `main` 创建独立功能分支。

## 全局约束

- ATL 只做历史回测和模拟交易，不连接真实券商账户或真实资金。
- 两个目标 Profile 是 `A-Share Demo 6` 和 `CSI 300 Sample 20`。
- A 股买入、卖出都必须是 100 股整数倍；非整手订单整笔拒绝，禁止自动取整。
- A 股现金不足的买单全拒，禁止部分买入或自动增加本金。
- A 股卖单继续遵循 PR #272 的 T+1 和部分卖出语义。
- 默认报告本金保持 `$1,000`，选择范围保持 `$1` 至 `$10,000`；不要修改这些常量或前端上限。
- iFinD A 股继续按历史 USD/CNY 汇率使用人民币原生账本；不得把 `$1,000` 直接当作 `¥1,000`。
- `trades` 只保存真实成交；拒单不得进入交易数、收益、现金或权益曲线。
- 所有新增前端可见文字使用英文。
- 自动测试禁止访问真实 iFinD 或模型网络；真实凭证只用于最后的本地手动验收。
- 不打印、提交或截图暴露 iFinD、Alpaca、OpenRouter 或其他 LLM 凭证。
- 不修改或提交 `dashboard/storage/data/backtest.db`，不要使用 `git add -A`。
- 每个阶段先跑聚焦测试；最后再跑完整后端、vn.py、Packaging 和前端相关回归测试。

---

## Task 0：等待并同步 PR #272

**不修改产品代码。**

- [ ] 检查 [PR #272](https://github.com/Open-Finance-Lab/AgenticTrading/pull/272) 的状态：

```bash
gh pr view 272 --repo Open-Finance-Lab/AgenticTrading \
  --json state,mergedAt,mergeCommit,statusCheckRollup
```

- [ ] 只有当 `state=MERGED` 后才继续；未合并时停止在本任务。
- [ ] 从健康的新工作树同步最新 `origin/main`，不得操作旧的损坏工作树或覆盖其中的 `backtest.db`。
- [ ] 从最新 `origin/main` 创建 `feat/ifind-ashare-lot-size`。
- [ ] 将 `spec/ifind-ashare-lot-size` 分支上的设计和实施计划文档带入新功能分支。
- [ ] 确认新功能分支包含 PR #272 的 T+1 提交，并且工作树只有预期文档改动。

验证：

```bash
git log --oneline --decorate -8
git status --short --branch
git diff --stat origin/main -- dashboard/storage/data/backtest.db
```

最后一条命令必须没有输出。

---

## Task 1：让 Market Profile 声明整手规则并传播到执行上下文

**修改：**

- `dashboard/backend/infrastructure/market_data/profiles.py`
- `dashboard/backend/domain/backtesting/engine.py`
- `dashboard/backend/domain/backtesting/portfolio_manager.py`

**测试：**

- `dashboard/backend/tests/infrastructure/market_data/test_market_profiles.py`
- `dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py`
- `dashboard/backend/tests/backtesting/test_portfolio_manager_move.py`

- [ ] 先写失败测试：两个 iFinD Profile 的 `lot_size == 100`，Alpaca/vn.py 的 `lot_size == 1`。
- [ ] 测试 `HourlyBacktester` 创建 `PortfolioManager` 时传入 Profile 的 `lot_size`。
- [ ] 测试 `_llm_market_context()` 包含结构化 `lot_size`，让 Prompt 和托管 Agent 都能知道市场规则。
- [ ] 测试旧调用 `PortfolioManager()` 默认 `lot_size=1`，构造签名和行为保持兼容。
- [ ] 在 `MarketProfile` 尾部增加默认字段 `lot_size: int = 1`，只为两个 iFinD Profile 显式设置 100。
- [ ] `PortfolioManager` 保存 `lot_size`，Engine 从 Profile 注入；不要在执行器之外写 `if data_source == "ifind"`。
- [ ] 在 `_run_metadata()` 中暴露 `lot_size`，便于历史结果审计；旧运行读取时保持可选。

聚焦测试：

```bash
python -m pytest -q \
  dashboard/backend/tests/infrastructure/market_data/test_market_profiles.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py \
  dashboard/backend/tests/backtesting/test_portfolio_manager_move.py
```

预期：全部通过，非 iFinD Profile 没有新增拒单行为。

提交：

```bash
git add dashboard/backend/infrastructure/market_data/profiles.py \
  dashboard/backend/domain/backtesting/engine.py \
  dashboard/backend/domain/backtesting/portfolio_manager.py \
  dashboard/backend/tests/infrastructure/market_data/test_market_profiles.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py \
  dashboard/backend/tests/backtesting/test_portfolio_manager_move.py
git commit -m "feat(backtest): configure A-share lot sizes"
```

---

## Task 2：让 Rule-based 和 LLM 产生可审计的 A 股订单

**修改：**

- `dashboard/backend/domain/backtesting/reference_agent.py`
- `dashboard/backend/domain/backtesting/portfolio_manager.py`
- `dashboard/backend/infrastructure/llm/backtest_harness.py`
- `dashboard/backend/domain/backtesting/engine.py`

**测试：**

- `dashboard/backend/tests/backtesting/test_reference_agent.py`
- `dashboard/backend/tests/backtesting/test_portfolio_manager_move.py`
- `dashboard/backend/tests/llm/test_backtest_harness.py`
- `dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py`

- [ ] 先写失败测试：A 股 Rule-based 买入信号固定产生 100 股，不再按 2% 权益算出零散股数。
- [ ] 测试即使现金不足，Rule-based 仍把 100 股原始请求交给统一执行器；执行器负责记录拒单，Agent 层不得静默丢弃。
- [ ] 测试 `lot_size=1` 时 Rule-based 的 2% 仓位算法和现金预检查逐字节保持原行为。
- [ ] 在 A 股 System Prompt 中明确 BUY/SELL `position_size` 必须为 100 的正整数倍，但仍以执行器校验为最终权威。
- [ ] 测试 A 股 LLM 的 `position_size=100.5` 保持原始数量进入执行器，不能先经 `int()` 变成 100。
- [ ] 测试 A 股 LLM 的有效整手即使现金不足也进入执行器；非 A 股继续保持当前的 Agent 层现金预检查。
- [ ] 测试 LLM 缺少 `position_size` 时不自动把计算结果向下凑整手；原始/计算结果必须经过统一整手门禁。
- [ ] 保持 LLM 的 symbol allow-list、动作数量、信心阈值和安全上限不变。

聚焦测试：

```bash
python -m pytest -q \
  dashboard/backend/tests/backtesting/test_reference_agent.py \
  dashboard/backend/tests/backtesting/test_portfolio_manager_move.py \
  dashboard/backend/tests/llm/test_backtest_harness.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py
```

提交：

```bash
git add dashboard/backend/domain/backtesting/reference_agent.py \
  dashboard/backend/domain/backtesting/portfolio_manager.py \
  dashboard/backend/infrastructure/llm/backtest_harness.py \
  dashboard/backend/domain/backtesting/engine.py \
  dashboard/backend/tests/backtesting/test_reference_agent.py \
  dashboard/backend/tests/backtesting/test_portfolio_manager_move.py \
  dashboard/backend/tests/llm/test_backtest_harness.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py
git commit -m "feat(backtest): size A-share agent orders by lot"
```

---

## Task 3：在统一执行器实现整手、现金和订单事件

**修改：**

- `dashboard/backend/domain/trading/execution.py`
- `dashboard/backend/domain/backtesting/portfolio_manager.py`

**测试：**

- `dashboard/backend/tests/domain/trading/test_execution.py`
- `dashboard/backend/tests/domain/trading/test_portfolio_compatibility.py`

**接口目标：**

`execute_actions()` 增加向后兼容的可选参数：

```python
lot_size: int = 1
order_events: Optional[List[Dict]] = None
```

`PortfolioManager` 增加 `order_events = []`，并把自身 `lot_size` 和事件列表传给执行器。

- [ ] 先写失败测试，覆盖 100/200 股通过，50/150/250.5 股以 `invalid_lot_size` 整笔拒绝。
- [ ] 锁定原因优先级：非整手优先于现金不足、T+1 和持仓不足。
- [ ] 覆盖 A 股 100 股现金充足时 `FILLED`，现金不足时 `REJECTED / insufficient_cash_for_lot`。
- [ ] 覆盖 200 股只够买 100 股时整笔拒绝，现金、持仓和 `trades` 不变。
- [ ] 覆盖卖出 200 股、可卖 100 股时 `PARTIAL / t1_frozen`，只有 100 股进入 `trades`。
- [ ] 覆盖同一卖单同时包含冻结和持仓不足时：`rejected_orders` 保留分项，单个 `order_event` 主要原因为 `t1_frozen`。
- [ ] 覆盖次日解冻后的整手卖出为 `FILLED`。
- [ ] 每个被执行器处理的 BUY/SELL 订单只追加一条 `order_event`；HOLD 不追加。
- [ ] 事件字段包含时间、股票、方向、申请/成交/未成交数量、原生价格、原生成交额、状态、执行原因和独立的 `strategy_reason`。
- [ ] `trades` 仍然只保存实际成交；不得写入 0 股成交。
- [ ] `lot_size=1` 路径通过现有 golden/compatibility 测试，现金、持仓和成交记录不发生回归。

聚焦测试：

```bash
python -m pytest -q \
  dashboard/backend/tests/domain/trading/test_execution.py \
  dashboard/backend/tests/domain/trading/test_portfolio_compatibility.py \
  dashboard/backend/tests/backtesting/test_portfolio_manager_move.py
```

提交：

```bash
git add dashboard/backend/domain/trading/execution.py \
  dashboard/backend/domain/backtesting/portfolio_manager.py \
  dashboard/backend/tests/domain/trading/test_execution.py \
  dashboard/backend/tests/domain/trading/test_portfolio_compatibility.py \
  dashboard/backend/tests/backtesting/test_portfolio_manager_move.py
git commit -m "feat(backtest): enforce A-share lot execution"
```

---

## Task 4：转换、持久化并实时发布订单事件

**修改：**

- `dashboard/backend/domain/backtesting/currency.py`
- `dashboard/backend/domain/backtesting/engine.py`
- `dashboard/backend/api/routers/backtests.py`

**测试：**

- `dashboard/backend/tests/backtesting/test_currency_context.py`
- `dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py`
- `dashboard/backend/tests/test_backtests_router.py`

- [ ] 先写失败测试：跨币种订单事件把原生 CNY `price/executed_value` 转为报告 USD，同时保留 `native_price/native_value/fx_rate`。
- [ ] 完全拒单的报告和原生成交额都保持 0，不能把申请金额伪装成成交额。
- [ ] 增加专用 `CurrencyContext.reporting_order_event()`，不要用字符串替换或在前端重复汇率计算。
- [ ] Engine 增加 `_serialize_order_events()`，规范化时间和数值类型。
- [ ] live progress 同时返回 `trades`、`rejected_orders`、`order_events`。
- [ ] Agent run metadata 持久化 `lot_size` 和最终 `order_events`；历史运行缺失字段时按空列表兼容。
- [ ] `RunMetadata` 增加可选 `lot_size` 和默认空列表 `order_events`，并更新 `_run_metadata_response()` allow-list。
- [ ] 保证 `num_trades=len(manager.trades)`，不得改成订单事件数量。

聚焦测试：

```bash
python -m pytest -q \
  dashboard/backend/tests/backtesting/test_currency_context.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py \
  dashboard/backend/tests/test_backtests_router.py
```

提交：

```bash
git add dashboard/backend/domain/backtesting/currency.py \
  dashboard/backend/domain/backtesting/engine.py \
  dashboard/backend/api/routers/backtests.py \
  dashboard/backend/tests/backtesting/test_currency_context.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py \
  dashboard/backend/tests/test_backtests_router.py
git commit -m "feat(backtest): publish order execution events"
```

---

## Task 5：让最终运行 API 返回订单事件且保持兼容

**修改：**

- `dashboard/backend/api/routers/backtests.py`

**测试：**

- `dashboard/backend/tests/test_backtests_router.py`
- `dashboard/backend/tests/integration/test_ifind_ashare_backtest.py`

- [ ] 先写失败测试：`GET /runs/{run_id}/trades` 的 `trades` 字段仍只包含实际成交，同时响应增加 `order_events`。
- [ ] 从该运行已解码的 metadata 安全读取 `order_events`；缺失、`null` 或错误类型都回退为空列表。
- [ ] 保留现有 `count == len(trades)`，另加 `order_event_count == len(order_events)`，不改变旧客户端对 `count` 的理解。
- [ ] 验证 session ownership 仍在读取元数据和订单事件之前执行，不能造成跨用户泄露。
- [ ] 集成测试验证 0 笔成交时 `trades=[]`、`count=0`，但拒单 `order_events` 可读取。

聚焦测试：

```bash
python -m pytest -q \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/integration/test_ifind_ashare_backtest.py
```

提交：

```bash
git add dashboard/backend/api/routers/backtests.py \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/integration/test_ifind_ashare_backtest.py
git commit -m "feat(api): expose backtest order events"
```

---

## Task 6：升级现有 Trading Log 展示三种执行状态

**修改：**

- `dashboard/frontend/app.html`
- `dashboard/frontend/app.js`
- `dashboard/frontend/styles.css`

**测试：**

- `dashboard/backend/tests/test_ifind_ashare_frontend.py`
- 新建 `dashboard/backend/tests/test_trading_log_order_events_ui.py`

- [ ] 先按 `test_frontend_portfolio_panel.py` 的现有模式创建 Node 执行测试，提取并运行发货的 `app.js` 规范化/渲染函数，覆盖 `order_events` 优先、历史 `trades` 回退、状态和原因映射、筛选以及完全拒单金额。
- [ ] Trading Log 表头调整为 `Time / Action / Company / Asset / Quantity / Price / Total Value / Status / Reason` 的既定八列布局，其中 `Company / Asset` 保持为一个现有列。
- [ ] 把 `All Trades` 改为 `All Orders`，保留 `Buys Only` 和 `Sells Only`。
- [ ] `normalizeTradeRecord` 重构为兼容成交和订单事件的规范化函数；历史成交推导为 `FILLED`。
- [ ] live progress 优先渲染 `progress.order_events`，缺失时回退 `progress.trades`。
- [ ] 最终运行接口优先渲染 `data.order_events`，缺失时回退 `data.trades`。
- [ ] Quantity 显示 `executed / requested shares`；历史成交显示相同的成交/申请数量。
- [ ] 完全拒单的 Total Value 显示 `--`；部分成交只显示实际成交值。
- [ ] 固定映射英文原因：`Invalid lot size`、`Insufficient cash for one lot`、`T+1 frozen`、`Insufficient position`。
- [ ] 未知原因使用固定安全兜底文案，不能把任意 API 字符串直接插入 `innerHTML`。
- [ ] 状态使用紧凑的 `FILLED/PARTIAL/REJECTED` 标识，加入 tooltip/可访问文本并保证窄屏不重叠。
- [ ] 更新所有空状态的 `colspan`，避免表格错位。

聚焦测试：

```bash
python -m pytest -q \
  dashboard/backend/tests/test_ifind_ashare_frontend.py \
  dashboard/backend/tests/test_trading_log_order_events_ui.py
```

提交：

```bash
git add dashboard/frontend/app.html \
  dashboard/frontend/app.js \
  dashboard/frontend/styles.css \
  dashboard/backend/tests/test_ifind_ashare_frontend.py \
  dashboard/backend/tests/test_trading_log_order_events_ui.py
git commit -m "feat(frontend): show backtest order outcomes"
```

---

## Task 7：锁定本金、汇率、Rule-based/LLM 和非 A 股回归

**测试为主，必要时只修改前述领域文件。**

**测试：**

- `dashboard/backend/tests/integration/test_ifind_ashare_backtest.py`
- `dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py`
- `dashboard/backend/tests/backtesting/test_reference_agent.py`
- `dashboard/backend/tests/backtesting/test_vnpy_simulation_engine.py`
- `dashboard/backend/tests/infrastructure/market_data/test_vnpy_adapter.py`
- `dashboard/backend/tests/infrastructure/market_data/test_vnpy_simulation.py`
- `dashboard/backend/tests/test_vnpy_simulation_frontend.py`
- `packaging/agentictrading/tests/test_vnpy_cta_integration.py`

- [ ] 使用确定性 iFinD/FX 夹具验证 `$1,000` 按历史汇率转换后可以买入低价股票的一手。
- [ ] 验证 `$1,000` 买不起高价股票时回测成功、0 笔成交、水平权益曲线、拒单事件存在。
- [ ] 验证选择 `$3,000` 和 `$10,000` 时使用所选本金；继续拒绝大于 `$10,000` 的请求。
- [ ] Rule-based 和 LLM 两条 iFinD 路径都产生并执行整手订单。
- [ ] 两个 iFinD 股票池均启用相同整手规则。
- [ ] Alpaca、vn.py 和 Packaging 集成测试的成交结果保持原样。
- [ ] 检查测试输出和 Git diff 不包含任何真实凭证或数据库变化。

聚焦回归：

```bash
python -m pytest -q \
  dashboard/backend/tests/integration/test_ifind_ashare_backtest.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py \
  dashboard/backend/tests/backtesting/test_reference_agent.py \
  dashboard/backend/tests/backtesting/test_vnpy_simulation_engine.py \
  dashboard/backend/tests/infrastructure/market_data/test_vnpy_adapter.py \
  dashboard/backend/tests/infrastructure/market_data/test_vnpy_simulation.py \
  dashboard/backend/tests/test_vnpy_simulation_frontend.py \
  packaging/agentictrading/tests/test_vnpy_cta_integration.py
```

提交测试补强：

```bash
git add dashboard/backend/tests/integration/test_ifind_ashare_backtest.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py \
  dashboard/backend/tests/backtesting/test_reference_agent.py
git commit -m "test(backtest): cover A-share lot capital flows"
```

---

## Task 8：完整验证、真实 iFinD 页面验收和 PR

- [ ] 运行完整后端测试：

```bash
python -m pytest -q dashboard/backend/tests
```

- [ ] 运行 Packaging 测试：

```bash
python -m pytest -q packaging/agentictrading/tests
```

- [ ] 运行静态检查：

```bash
git diff --check
git status --short
git diff --stat origin/main -- dashboard/storage/data/backtest.db
```

- [ ] 使用本地已有环境变量启动一个空闲端口，不在命令输出中打印 Token。
- [ ] 使用真实 iFinD 历史数据分别运行：默认 `$1,000`、更高本金、至少一个 Rule-based 路径。
- [ ] 页面检查：初始/最终权益、曲线、CNY/USD 审计、Quantity、三种 Status、Reason 和 All Orders 筛选。
- [ ] 使用浏览器桌面和窄屏截图检查八列表格不重叠、不溢出；检查控制台没有新增错误。
- [ ] 若真实行情窗口恰好没有触发某种状态，使用自动化确定性夹具证明该状态，不伪造真实回测结果。
- [ ] 将功能分支推送到用户的 GitHub fork/远程，并创建以 `Open-Finance-Lab/AgenticTrading:main` 为 base 的独立 PR。
- [ ] PR 描述重点写明：iFinD A 股整手规则、历史汇率本金、订单审计、T+1 组合行为、0 笔成交合法、非 A 股兼容性和测试结果。
- [ ] 等待 CI 完成，确认 Backend tests、Packaging 和 CodeQL 全部通过；有失败则进入下一 Loop 修复，不带红灯交付。

## 完成定义

只有同时满足以下条件，才可以结束本 Loop：

1. PR #272 已合并，本功能基于最新 `main`；
2. 两个 iFinD A 股股票池只成交 100 股整数倍；
3. 非整手、资金不足和 T+1 都能在 Trading Log 审计；
4. 默认及更高本金按历史汇率正确运行；
5. 拒单不污染交易数、收益、现金和权益曲线；
6. Alpaca、vn.py、历史 API 和历史运行没有回归；
7. 自动测试、真实 iFinD 页面验收和 GitHub CI 全部通过；
8. 独立 PR 已创建且没有凭证或本地数据库改动。
