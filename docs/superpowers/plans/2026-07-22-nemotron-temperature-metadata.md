# Issue #148：Nemotron 可复现排行榜运行实施计划

> 按 loop engineering 执行：每个代码任务都遵循“先写失败测试 -> 最小实现 -> 聚焦验证 -> 独立提交”。

**目标：** 为 Nemotron 固定 `temperature=0`，让该参数从排行榜配置传到实际 OpenRouter 请求，并为所有新手动部署的 LLM 排行榜运行保存可复现配置。

**设计规格：** `docs/superpowers/specs/2026-07-22-nemotron-temperature-metadata-design.zh-CN.md`

**工作分支：** `fix/issue-148-nemotron-reproducibility`，基于 `origin/main@671c2d4`。

**技术栈：** Python 3.10+、Anthropic-shaped Messages API、OpenRouter、pytest、SQLite JSON metadata。

## 全局约束

- 不修改提示词、响应格式、解析逻辑、交易规则、仓位管理、模拟成交和行情来源。
- 不新增数据库表或字段，复用 `agent_runs.metadata`。
- 不通过环境变量传递温度，不按模型 ID 在基础设施中硬编码。
- 没有配置温度的模型不得多出 `temperature` 请求字段。
- 普通回测、规则策略和非 LLM 排行榜行为保持不变。
- 单元测试全部使用 fake/monkeypatch/临时数据库，不进行真实网络请求。
- 不保存 API Key、完整提示词、模型原始回答或 reasoning 文本。
- 收益率不作为代码修复通过标准；可复现配置和调用链正确性才是本轮标准。
- 每个任务独立提交，便于审阅和回滚。

---

## Task 1：请求层支持可选温度

**文件：**

- 修改：`dashboard/backend/infrastructure/llm/backtest_harness.py`
- 修改：`dashboard/backend/domain/backtesting/portfolio_manager.py`
- 修改：`dashboard/backend/tests/llm/test_backtest_harness.py`

### Step 1：写失败测试

在 `test_backtest_harness.py` 增加：

1. `request_trading_decision(..., temperature=0)` 调用 SDK 时包含 `temperature=0`。
2. 不传温度时，SDK 捕获的请求参数中完全没有 `temperature` 键。
3. `PortfolioManager.make_trading_decision_with_llm(..., temperature=0)` 的首次请求收到 `0`。
4. 无文本重试路径中的每次请求都收到相同的 `0`。
5. 旧调用不传温度时仍能得到原有决策结果。

测试使用已有 `_FakeClient`，必要时 monkeypatch `_request_trading_decision` 记录参数和返回响应序列。

### Step 2：确认测试先失败

```bash
pytest dashboard/backend/tests/llm/test_backtest_harness.py -q
```

预期：新增测试失败，因为现有函数签名不接受 `temperature`。

### Step 3：实现最小请求参数

`request_trading_decision()` 增加可选参数：

```python
def request_trading_decision(
    client,
    *,
    prompt: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
): ...
```

先构建原有请求字典，只有 `temperature is not None` 时才加入该键，再调用 `client.messages.create(**request_kwargs)`。必须保留 `0`，不能使用 `if temperature` 判断。

### Step 4：在决策层转发

`PortfolioManager.make_trading_decision_with_llm()` 增加默认值为 `None` 的温度参数，并在直接请求、无文本重试和最终 rescue 请求中传递。不得修改 `os.environ` 来传递温度。

### Step 5：运行聚焦测试

```bash
pytest dashboard/backend/tests/llm/test_backtest_harness.py -q
```

预期：PASS。

### Step 6：提交

```bash
git add dashboard/backend/infrastructure/llm/backtest_harness.py \
  dashboard/backend/domain/backtesting/portfolio_manager.py \
  dashboard/backend/tests/llm/test_backtest_harness.py
git commit -m "feat(llm): support optional trading temperature"
```

---

## Task 2：只为 Nemotron 配置并校验温度

**文件：**

- 修改：`dashboard/config/leaderboard.json`
- 修改：`dashboard/backend/domain/leaderboard/strategies/llm_agent.py`
- 修改：`dashboard/backend/tests/domain/leaderboard/test_strategies_move.py`

### Step 1：写失败测试

在 `test_strategies_move.py` 增加：

1. 读取 `leaderboard.json` 后，只有 `nemotron_3_nano_30b` 明确包含 `temperature: 0`。
2. `LLMAgentStrategy` 保存合法的 `0`、小数和 `2`。
3. 缺失温度时保存 `None`。
4. 字符串、布尔值、负数、大于 2、NaN 和 Infinity 抛出包含 `temperature` 的 `ValueError`。
5. 策略运行时将 `self.temperature` 传给 `PortfolioManager.make_trading_decision_with_llm()`。
6. 未配置温度的旧策略运行时传递 `None`。

策略运行测试使用最小 DataFrame 和 fake manager/client，不进行真实 LLM 或行情请求。

### Step 2：确认测试先失败

```bash
pytest dashboard/backend/tests/domain/leaderboard/test_strategies_move.py -q
```

预期：新增测试失败，因为策略尚未读取、校验或传递温度。

### Step 3：实现配置校验

在 `LLMAgentStrategy.__init__()` 中解析一次：

```text
字段缺失或 null -> None
int/float 且有限、0 <= value <= 1 -> float 或原数值
其他情况 -> ValueError
```

Python 的 `bool` 是 `int` 子类，必须在数字判断前单独拒绝。不得静默改成默认值，否则错误实验仍会被发布。

**（评审修订）上界取 1 而不是 2。** 这里所有 gateway（commonstack / openrouter / anthropic）最终都走
Anthropic Messages 形状的 `client.messages.create`，其 `temperature` 上限是 1.0。放行 OpenAI 的 0–2 区间
只会把配置笔误推迟成每步一次的 400，进而静默回退到 rule-based，最后由 H6 guard 拦下——一个很难查的失败模式。

**（评审修订）与 reasoning 的互斥检查。** Anthropic 不允许 `temperature` 与 extended thinking 同时出现，而
OpenRouter 包装层会在 effort 启用推理时注入 `thinking`。因此配置里显式写了非关闭、非透传（`auto`/`default`）
的 `reasoning_effort` 时，同时固定 `temperature` 直接报错。`reasoning_effort` 缺省时取决于环境变量，
配置期无法判定，按“未知”放行。

### Step 4：连接策略与决策层

调用 manager 时加入：

```python
temperature=self.temperature
```

这条链只传值，不做 provider 或模型名称判断。

### Step 5：配置 Nemotron

只在 `nemotron_3_nano_30b` 条目中增加：

```json
"temperature": 0
```

其他模型条目保持不变。

### Step 6：运行聚焦与回归测试

```bash
pytest dashboard/backend/tests/domain/leaderboard/test_strategies_move.py \
  dashboard/backend/tests/llm/test_backtest_harness.py -q
```

预期：PASS。

### Step 7：提交

```bash
git add dashboard/config/leaderboard.json \
  dashboard/backend/domain/leaderboard/strategies/llm_agent.py \
  dashboard/backend/tests/domain/leaderboard/test_strategies_move.py
git commit -m "fix(leaderboard): pin Nemotron temperature"
```

---

## Task 3：保存手动部署 LLM 运行的配置元数据

**文件：**

- 修改：`dashboard/backend/domain/leaderboard/service.py`
- 修改：`dashboard/backend/tests/domain/leaderboard/test_deploy_guard.py`
- 视测试边界需要修改：`dashboard/backend/tests/test_agent_runs_metadata.py`

### Step 1：写失败测试

扩展 deploy 测试，覆盖：

1. Nemotron 风格 LLM 条目保存以下 metadata：

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

2. 没有显式温度/reasoning 的 LLM 条目保存 `null`，不猜 provider 默认值。
3. `start_date`、`end_date` 和资金记录调用时解析后的有效值，包括 CLI 日期覆盖。
4. `model_id` 使用策略实际解析后的值，而不是只相信展示名称。
5. 规则策略仍按原行为发布，不被写入 LLM 配置快照。
6. 命中历史缓存时不伪造或回填 metadata，也不触发昂贵重跑。

使用临时 `BacktestDatabase` 读回真实 JSON 字段；monkeypatch 行情、策略和 token 上限。

### Step 2：确认测试先失败

```bash
pytest dashboard/backend/tests/domain/leaderboard/test_deploy_guard.py \
  dashboard/backend/tests/test_agent_runs_metadata.py -q
```

预期：新增 deploy metadata 断言失败，因为 `deploy_model_run()` 当前没有传 `metadata=`。

### Step 3：构建单一配置快照

在 leaderboard service 中增加一个小型纯函数，输入 entry 和本次解析后的有效值，输出固定字段字典。只对 `strategy == "llm_agent"` 的手动部署运行调用。

`llm_max_output_tokens` 必须取 `backtest_harness.DEFAULT_MAX_OUTPUT_TOKENS` 的有效值；测试应能 monkeypatch 该模块属性。不得读取或保存密钥和响应内容。

### Step 4：随成绩写入数据库

在现有 `db.insert_run()` 调用中传入 `metadata=run_metadata`。先保存 run，再保存 curve 的顺序保持不变；metadata 序列化或 run 写入失败时不能继续发布一条无配置曲线。

不得修改数据库 schema 或 `BacktestDatabase.insert_run()` 的接口，因为它已经支持 JSON metadata。

### Step 5：运行聚焦测试

```bash
pytest dashboard/backend/tests/domain/leaderboard/test_deploy_guard.py \
  dashboard/backend/tests/test_agent_runs_metadata.py -q
```

预期：PASS。

### Step 6：提交

```bash
git add dashboard/backend/domain/leaderboard/service.py \
  dashboard/backend/tests/domain/leaderboard/test_deploy_guard.py \
  dashboard/backend/tests/test_agent_runs_metadata.py
git commit -m "feat(leaderboard): record LLM run configuration"
```

如果 `test_agent_runs_metadata.py` 最终不需要修改，不应为了匹配计划而产生无意义改动，也不要把它加入提交。

---

## Task 4：回归验证与分支检查

**产品代码：** 不新增。

### Step 1：运行相关测试组

```bash
pytest dashboard/backend/tests/llm \
  dashboard/backend/tests/domain/leaderboard \
  dashboard/backend/tests/test_agent_runs_metadata.py -q
```

预期：PASS。

### Step 2：运行完整 backend 测试

```bash
pytest dashboard/backend/tests -q
```

若出现与本分支无关的失败，先在 `origin/main` 上确认是否为基线问题，再报告；不要顺带修改无关模块。

### Step 3：做静态和范围检查

```bash
git diff --check origin/main...HEAD
git diff --stat origin/main...HEAD
git status --short --branch
```

确认分支只包含：设计、计划、温度参数链、Nemotron 配置、运行元数据和对应测试。

### Step 4：整理实现结果

向用户报告：

- 哪些测试通过；
- 分支与提交；
- 是否存在基线失败；
- 为什么此时只能证明“实验条件可复现”，还不能证明“Nemotron 一定表现更好”。

---

## Task 5：下一轮真实 Nemotron 实验（代码验证后单独执行）

该任务会调用 OpenRouter 并产生费用。执行完整实验前再次向用户说明预计请求范围，并获得明确确认；不得把实验原始响应或密钥提交到仓库。

固定并记录：

- 模型：`nvidia/nemotron-3-nano-30b-a3b`；
- integration：`openrouter`；
- temperature：`0`；
- reasoning：`none`；
- 有效输出上限；
- 当前排行榜初始资金；
- 交易日期和行情范围；
- ATL 当前提交 SHA。

先跑短窗口检查响应、解析和动作，再决定是否运行完整窗口。完整实验关注：有效决策覆盖率、无效 JSON、规则回退、成交数量、收益率、Sharpe 和最大回撤。

只有相同配置下的新完整实验完成后，才能继续回答：Nemotron 的差曲线究竟主要来自随机参数、调用链异常，还是模型自身的交易决策。

---

## 完成定义

- 所有新增测试先红后绿。
- Nemotron 请求明确包含 `temperature=0`。
- 未配置模型的请求不包含温度字段。
- 新手动部署 LLM 运行可以读回完整配置元数据。
- 非法温度在真实模型调用前失败。
- 相关回归测试通过，工作区干净。
- 未经确认不执行付费完整窗口，也不推送分支或创建 PR。
