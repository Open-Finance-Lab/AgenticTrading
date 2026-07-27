# TradingAgents 接入 ATL 第一版实施计划

> 本计划按 loop engineering 执行：每个任务都是一个独立的“失败测试 -> 最小实现 ->
> 验证 -> 提交”闭环。按顺序完成，不在同一轮提前加入后续能力。

**目标：** 让 TradingAgents 用户在本地为一只美股和若干自选日期预生成决策，再通过
ATL 当前 Agent-Environment Protocol v1 完成 T+1 模拟成交、指标、曲线和排行榜记录。

**架构：** TradingAgents 在用户本地使用自己的 Key 和数据源生成五档评级，并写入安全、
可重放的 JSON artifact。ATL 集成适配器把评级转换为 BUY/HOLD/SELL，使用 typed
`ATLClient` 驱动现有 `us-equity-hourly-v1` 环境。模型生成阶段与 ATL 回放阶段完全分开。

**技术栈：** Python 3.10+、TradingAgents v0.3.1、ATL `agentictrading` 轻量 SDK、
标准库 `dataclasses/json/hashlib/zoneinfo/urllib`、pytest。

**设计规格：**
`docs/superpowers/specs/2026-07-26-tradingagents-atl-integration-design.zh-CN.md`

## 全局约束

- 分支固定为 `feat/tradingagents-integration`，基线是 `origin/main` 3a7781a。
- 第一版只支持一只 `us-equity-hourly-v1` 允许的美股，只做多。
- TradingAgents 只能延迟导入，不能成为 SDK 或 ATL 后端的核心依赖。
- ATL 后端不能调用用户的 TradingAgents、LLM 或数据供应商。
- 不修改通用 `AgentRunner` 的 `decide(observation)` 协议。
- 不使用旧的 `AgenticTradingClient.run_backtest(strategy=...)` 接口。
- 不把格式错误默认为主动 Hold；必须使用上游 `parse_rating(..., default="")` 校验。
- 生成阶段完成后才创建 ATL run，回放阶段不得调用 LLM。
- 不将完整环境变量或包含 Key、Token、Secret、Password、Credential 的值持久化。
- 第一版不增加网页启动按钮、专用服务、任务队列、多股票、日频环境或 100% 仓位环境。
- CI 和自动测试禁止真实网络、Alpaca、TradingAgents 数据源及付费 LLM 调用。
- 真实冒烟测试会产生外部费用，必须在实现和离线测试全部通过后单独获得用户确认。
- 每个任务只提交该任务相关文件，提交保持可回滚。

---

## Task 1：建立评级映射和安全决策 artifact

**文件：**

- 新建：`packaging/agentictrading/src/agentictrading/integrations/__init__.py`
- 新建：`packaging/agentictrading/src/agentictrading/integrations/tradingagents.py`
- 新建：`packaging/agentictrading/tests/test_tradingagents_integration.py`

**首轮接口：**

```python
ARTIFACT_SCHEMA_VERSION = "tradingagents-atl-v1"
RATING_TO_ACTION = {...}

@dataclass(frozen=True)
class TradingAgentsDecisionRecord: ...

@dataclass(frozen=True)
class TradingAgentsDecisionArtifact: ...

def map_rating(rating: str) -> str: ...
def build_safe_manifest(...): ...
def save_decision_artifact(artifact, path) -> str: ...  # 返回文件 SHA-256
def load_decision_artifact(path) -> TradingAgentsDecisionArtifact: ...
```

- [ ] **Step 1：写失败测试**

覆盖：

- 五档评级按已批准规则转换；未知值抛出明确异常。
- artifact 决策日期唯一、有序且股票一致。
- JSON round-trip 保留 Unicode 原始结论和空错误字段。
- 加载时拒绝未知 schema、损坏 JSON、非法状态、错误哈希和空决策列表。
- `raw_sha256` 从原始最终结论稳定计算。
- manifest 只保留安全白名单；嵌套的敏感键和值不会出现。
- 错误摘要去除常见 `key=value`、Bearer Token 和 URL 凭据，并限制长度。
- 只导入 artifact/replay 模块时，`tradingagents` 不会出现在 `sys.modules`。

- [ ] **Step 2：运行并确认失败**

```bash
python -m pytest packaging/agentictrading/tests/test_tradingagents_integration.py -v
```

预期：FAIL，集成包尚不存在。

- [ ] **Step 3：实现最小 artifact 边界**

仅实现 dataclass、显式序列化/反序列化、字段校验、SHA-256 和安全白名单。不得在本任务
导入 TradingAgents 或编写 ATL 网络循环。

默认文件名函数使用 `~/.agentictrading/tradingagents/decisions/`，但单元测试始终使用
pytest 临时目录。

- [ ] **Step 4：运行测试并提交**

```bash
python -m pytest packaging/agentictrading/tests/test_tradingagents_integration.py -v
git add packaging/agentictrading/src/agentictrading/integrations \
  packaging/agentictrading/tests/test_tradingagents_integration.py
git commit -m "feat(sdk): add TradingAgents decision artifact"
```

---

## Task 2：实现本地 TradingAgents 决策生成器

**文件：**

- 修改：`packaging/agentictrading/src/agentictrading/integrations/tradingagents.py`
- 修改：`packaging/agentictrading/tests/test_tradingagents_integration.py`

**新增接口：**

```python
class TradingAgentsDependencyError(RuntimeError): ...
class TradingAgentsVersionError(RuntimeError): ...

class TradingAgentsDecisionGenerator:
    def generate(
        self,
        *,
        symbol: str,
        analysis_dates: Sequence[str],
        config: Mapping[str, Any],
        selected_analysts: Sequence[str],
    ) -> TradingAgentsDecisionArtifact: ...
```

构造函数支持注入 `graph_factory`、`rating_parser` 和 `version_resolver`，让测试不安装
TradingAgents、不联网也能覆盖完整行为。

- [ ] **Step 1：扩展失败测试**

覆盖：

- 默认路径只有调用 `generate()` 时才导入 TradingAgents。
- 缺少依赖时错误包含官方 clone/install 方法。
- v0.3.1 可用；不兼容的大版本明确失败。
- 对每个日期调用 `graph.propagate(symbol, date)`。
- 从 `state["final_trade_decision"]` 取原始结论，并调用
  `parse_rating(raw, default="")`，不信任上游默认 Hold。
- 显式 `Hold` 是 `status=valid`；无评级是 `status=error`。
- 第一次异常后重试一次；第二次失败写入清理后的错误 HOLD，`attempts=2`。
- 部分失败仍生成 artifact；全部失败抛出错误且调用方不会启动 ATL run。
- manifest 记录实际版本、模型、分析员、数据源和安全配置哈希，但没有任何 Key。

- [ ] **Step 2：运行并确认失败**

```bash
python -m pytest packaging/agentictrading/tests/test_tradingagents_integration.py -v -k generator
```

- [ ] **Step 3：实现最小生成器**

默认工厂在函数内部导入：

```python
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.agents.utils.rating import parse_rating
```

合并配置时复制 `DEFAULT_CONFIG`，不原地修改上游全局字典。一个 generator 实例复用一个
graph，日期按排序顺序调用。每个日期最多两次完整尝试。

- [ ] **Step 4：运行完整 SDK 测试并提交**

```bash
python -m pytest packaging/agentictrading/tests/test_tradingagents_integration.py -v
python -m pytest packaging/agentictrading/tests -q
git add packaging/agentictrading/src/agentictrading/integrations/tradingagents.py \
  packaging/agentictrading/tests/test_tradingagents_integration.py
git commit -m "feat(sdk): generate TradingAgents decisions locally"
```

---

## Task 3：实现 T+1 回放、25% 目标仓位和诊断状态机

**文件：**

- 修改：`packaging/agentictrading/src/agentictrading/integrations/tradingagents.py`
- 修改：`packaging/agentictrading/tests/test_tradingagents_integration.py`

**新增接口：**

```python
@dataclass
class TradingAgentsReplayDiagnostics: ...

class TradingAgentsReplayPlanner:
    def decision_for_step(self, step) -> Decision: ...
    def finalize(self) -> TradingAgentsReplayDiagnostics: ...
```

Planner 是纯本地状态机，不负责创建 run 或发 HTTP。它接收 typed `Step`，输出 SDK
`Decision`，因此可以用手工构造的 Observation/Step 完整测试。

- [ ] **Step 1：扩展失败测试**

覆盖：

- UTC Step 时间转换成 `America/New_York` 日期后再比较 analysis_date。
- 分析日当天所有小时 HOLD，之后第一个实际 Step 才处理记录。
- 同一天后续小时 HOLD；每条记录只处理一次。
- 多条待处理记录同时可执行时选择最新记录，并把旧记录记为 superseded。
- `status=error` 在 T+1 提交带 `generation_error` 的空订单。
- Buy/Overweight 计算 `floor(equity * max_position_weight / price)`，只买目标股数与已持股数
  的正差额。
- 已到目标、价格过高、缺少价格产生带不同原因的 HOLD。
- Underweight/Sell 卖出全部现有整数股；空仓时 HOLD；永不做空。
- symbol 不在 `allowed_symbols`、缺少约束、错误仓位数据时明确失败。
- 订单固定为 shares + market；rationale 包含评级、analysis_date 和 artifact hash。
- `finalize()` 列出所有未处理日期，并返回非成功诊断。

- [ ] **Step 2：运行并确认失败**

```bash
python -m pytest packaging/agentictrading/tests/test_tradingagents_integration.py -v -k replay
```

- [ ] **Step 3：实现纯回放状态机**

不得调用 TradingAgents 或 ATL API。Planner 只消费 artifact、Step 时间、市场 feature、
portfolio 和 constraints。所有 HOLD 都使用空 orders；执行错误和主动 HOLD 使用不同
rationale 标签。

- [ ] **Step 4：运行测试并提交**

```bash
python -m pytest packaging/agentictrading/tests/test_tradingagents_integration.py -v
git add packaging/agentictrading/src/agentictrading/integrations/tradingagents.py \
  packaging/agentictrading/tests/test_tradingagents_integration.py
git commit -m "feat(sdk): replay TradingAgents signals at ATL T+1"
```

---

## Task 4：用 ATLClient 驱动完整回测闭环

**文件：**

- 修改：`packaging/agentictrading/src/agentictrading/integrations/tradingagents.py`
- 修改：`packaging/agentictrading/src/agentictrading/integrations/__init__.py`
- 修改：`packaging/agentictrading/tests/test_tradingagents_integration.py`

**新增接口：**

```python
class TradingAgentsReplayIncompleteError(RuntimeError):
    run_id: str
    analysis_dates: tuple[str, ...]

class TradingAgentsATLRunner:
    def run_backtest(
        self,
        *,
        artifact: TradingAgentsDecisionArtifact,
        artifact_sha256: str,
        agent_version_id: str,
        start_date: str,
        end_date: str,
    ) -> RunResult: ...
```

- [ ] **Step 1：使用 fake ATLClient 写失败集成测试**

fake client 提供 `create_run/get_next_step/submit_decision/get_run_result/get_run_metrics/wait`
并记录调用。覆盖：

- 创建 run 时固定 `environment_id="us-equity-hourly-v1"` 和单股票 symbols。
- run config 记录集成名、TradingAgents 版本、分析日期、artifact hash、两套数据来源和
  有效/错误数量，不记录本地路径或原始结论。
- loading/pending/executing 状态按 SDK 习惯轮询。
- awaiting_decision 每步调用 Planner，并以 typed Decision 提交。
- ATL 拒单、fills、主动 HOLD、错误 HOLD、superseded 和 timeout_holds 汇总正确。
- completed 后读取最终 RunResult；未处理记录抛出带 run_id 的专用异常。
- API 异常原样上抛并保留 SDK 的 run_id 行为，不转为客户端 HOLD。
- 所有日期失败的 artifact 在 `create_run` 前被拒绝。
- 非法日期范围在 `create_run` 前被拒绝。

- [ ] **Step 2：运行并确认失败**

```bash
python -m pytest packaging/agentictrading/tests/test_tradingagents_integration.py -v -k runner
```

- [ ] **Step 3：实现最小 ATL 循环**

沿用通用 `AgentRunner` 的状态处理和等待上限，但保留 Step 给 Planner；不要复制鉴权、
序列化或 HTTP 代码，全部复用 `ATLClient`。完成摘要以结构化 dataclass 返回，打印交给
CLI 层。

- [ ] **Step 4：运行 SDK 与协议契约回归测试并提交**

```bash
python -m pytest packaging/agentictrading/tests -q
python -m pytest dashboard/backend/tests/test_protocol_api.py -q
git add packaging/agentictrading/src/agentictrading/integrations \
  packaging/agentictrading/tests/test_tradingagents_integration.py
git commit -m "feat(sdk): run TradingAgents artifacts on ATL"
```

---

## Task 5：增加一条命令入口和中文接入文档

**文件：**

- 新建：`dashboard/examples/tradingagents_atl_backtest.py`
- 新建：`docs/integrations/tradingagents.zh-CN.md`
- 修改：`docs/source/lab/external_agents.rst`
- 修改：`packaging/agentictrading/tests/test_tradingagents_integration.py`

- [ ] **Step 1：写 CLI 参数与配置失败测试**

覆盖可测试的 `build_parser()` / `run_from_args()`：

- `--symbol`、可重复 `--analysis-date`、`--start-date`、`--end-date`。
- `--decisions-file` 跳过生成阶段，不导入 TradingAgents。
- `--output` 指定 artifact 路径；未指定时使用安全默认目录。
- `ATL_API_KEY`、`ATL_BASE_URL`、`ATL_AGENT_VERSION_ID` 缺失时给出明确提示。
- 可选的 provider/deep-model/quick-model/selected-analyst 参数只进入安全配置。
- `--help` 不需要任何 Key、TradingAgents 或网络。
- 结束摘要显示 run_id、结果 URL、有效/错误/主动 HOLD/拒单/超时/未处理/成交计数。

- [ ] **Step 2：运行并确认失败**

```bash
python -m pytest packaging/agentictrading/tests/test_tradingagents_integration.py -v -k cli
```

- [ ] **Step 3：实现一条命令的两阶段流程**

命令先生成或加载 artifact，再构造 ATLClient 和 TradingAgentsATLRunner。`--decisions-file`
必须只走回放路径，保证用户复测时不再次付费。

- [ ] **Step 4：编写面向用户的文档**

文档包含：

- 两项目各自职责和口语化类比。
- 独立虚拟环境安装步骤。
- TradingAgents 与 ATL 凭据的归属和安全边界。
- 创建并复用 AgentVersion 的方法。
- 一只 AAPL、三个显式分析日期的命令示例。
- 决策文件字段、两套数据来源、T+1、25% 仓位和错误 HOLD 解释。
- 如何只回放已有 artifact，避免重复 LLM 费用。
- 如何在 ATL 查找 run、曲线、交易、决策和排行榜结果。
- 明确说明该接入不证明策略盈利，也不是实盘交易。

`external_agents.rst` 只增加简短入口，不复制整篇文档。

- [ ] **Step 5：验证帮助信息、文档和测试并提交**

```bash
python dashboard/examples/tradingagents_atl_backtest.py --help
python -m pytest packaging/agentictrading/tests/test_tradingagents_integration.py -v
git diff --check
git add dashboard/examples/tradingagents_atl_backtest.py \
  docs/integrations/tradingagents.zh-CN.md docs/source/lab/external_agents.rst \
  packaging/agentictrading/tests/test_tradingagents_integration.py
git commit -m "docs: add TradingAgents ATL quickstart"
```

---

## Task 6：回归验证和付费冒烟测试门槛

**文件：** 原则上不新增功能文件；只修复本次集成直接造成的测试问题。

- [ ] **Step 1：运行完整相关测试**

```bash
python -m pytest packaging/agentictrading/tests -q
python -m pytest dashboard/backend/tests/test_protocol_api.py -q
python -m pytest dashboard/backend/tests/test_deadline_and_holds.py -q
python -m pytest dashboard/backend/tests/test_app_composition.py -q
git diff --check
```

- [ ] **Step 2：运行无 TradingAgents 的回放 smoke**

使用固定测试 artifact 和 fake/local ATL 测试环境，证明回放路径不导入 TradingAgents、
不读取 LLM Key、不联网生成决策。

- [ ] **Step 3：检查安全与变更范围**

```bash
git status --short
git diff origin/main...HEAD --stat
git log --oneline origin/main..HEAD
```

人工检查：没有 `.env`、Key、用户目录绝对路径、真实 artifact、数据库文件和模型原始日志
进入提交。

- [ ] **Step 4：请求真实冒烟测试授权**

在所有离线验证通过后，向用户说明预计 TradingAgents 调用次数、所用模型和可能费用。
只有用户明确同意，才使用其本地 Key 运行一个日期的真实 TradingAgents 分析，再执行短
ATL 回测。不同意或凭据不足时跳过，并在最终结果中明确写明未运行付费 smoke。

- [ ] **Step 5：验收真实 smoke（获得授权时）**

必须满足：

```text
至少 1 个有效五档评级
artifact 不含凭据
至少 1 条 ATL 决策记录
完整 RunResult 和收益曲线
timeout_holds = 0
相同 artifact 再回放时不调用 TradingAgents
```

- [ ] **Step 6：最终提交或报告**

如果 Task 6 没有代码修复，不制造空提交。最终报告列出分支、提交、测试结果、未运行项、
运行命令和用户查看结果的位置。此时再决定是否推送分支和创建 PR，不自动对 GitHub 产生
外部状态变更。
