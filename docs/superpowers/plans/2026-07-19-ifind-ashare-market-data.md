# iFinD A 股行情数据接入实施计划

> 本计划按 loop engineering 执行：每个任务都是一个独立的“失败测试 -> 最小实现 ->
> 验证 -> 提交”闭环。必须按顺序完成，不在同一轮混入后续任务。

**目标：** 在 ATL 现有 Backtest 页面增加 `iFinD A 股` 数据源，通过同花顺 iFinD 官方
HTTP API 获取 6 只代表性 A 股的历史 60 分钟 K 线，转换成 ATL 统一 OHLCV DataFrame，
再运行规则策略和同股票池等权买入持有基准。

**架构：** 浏览器只选择数据源和固定股票池；ATL 后端的 `IFindAshareProvider` 调用
iFinD HTTP Client，Adapter 校验并转换官方 `tables` 响应，回测引擎只消费统一 OHLCV。
Access Token 只存在于后端环境变量中。iFinD 模式不经过 vn.py、不调用 LLM、不生成
DJIA 基准，也不回退到任何其他数据源。

**技术栈：** Python 3.10+、requests、pandas、FastAPI/Pydantic、pytest、原生
HTML/CSS/JavaScript。

**设计规格：**

- `docs/superpowers/specs/2026-07-19-ifind-ashare-market-data-design.zh-CN.md`

## 全局约束

- 只在 `AgenticTrading-ifind-ashare` worktree 的
  `feat/ifind-ashare-market-data` 分支工作。
- 每个任务先写失败测试，再写最小实现，最后单独提交。
- 默认测试和 CI 不访问真实 iFinD 网络，使用官方响应结构的测试替身。
- Access Token 只从 `IFIND_ACCESS_TOKEN` 读取，不进入浏览器、日志、运行 metadata、
  测试 fixture 或 Git 提交。
- 数据源异常时明确失败，禁止切换到 Alpaca、vn.py、Yahoo、免费 A 股接口或伪造数据。
- `alpaca` 和 `vnpy_simulation` 的现有行为保持向后兼容。
- iFinD 模式固定为 60 分钟周期和 `a_share_demo_6` 股票池，不接受前端任意 `assets`。
- 日期区间统一为 `[start, end)`，继续遵守现有最长 31 天限制。
- 完整回测前，每只股票至少需要 50 根有效 K 线。
- iFinD 模式禁止 LLM，只运行现有规则策略；0 笔交易是合法结果。
- iFinD 模式只生成规则策略和同股票池等权买入持有两条曲线，不生成 DJIA。
- 新错误必须可操作且脱敏，不能包含 Token、请求头或完整上游响应。
- 每轮提交前运行相关测试和 `git diff --check`。

## 固定市场配置

```python
IFIND_ASHARE = "ifind_ashare"
A_SHARE_DEMO_6 = "a_share_demo_6"

A_SHARE_DEMO_6_SYMBOLS = (
    "600519.SH",  # 贵州茅台
    "601318.SH",  # 中国平安
    "600036.SH",  # 招商银行
    "000001.SZ",  # 平安银行
    "000858.SZ",  # 五粮液
    "300750.SZ",  # 宁德时代
)
```

市场配置至少描述：数据源、市场 `CN`、时区 `Asia/Shanghai`、周期 `60m`、固定股票池、
决策来源 `rule_based`、基准 `equal_weight_buyhold`、是否启用 LLM、是否生成指数基准。

---

## Task 1：注册 iFinD 数据源、市场配置和功能开关

**文件：**

- 新建：`dashboard/backend/infrastructure/market_data/profiles.py`
- 修改：`dashboard/backend/infrastructure/market_data/provider.py`
- 修改：`dashboard/backend/api/routers/config.py`
- 修改：`.env.example`
- 修改：`dashboard/backend/tests/infrastructure/market_data/test_provider.py`
- 修改：`dashboard/backend/tests/test_market_data_features.py`

**接口：**

```python
IFIND_ASHARE = "ifind_ashare"

class MarketDataCredentialsError(RuntimeError): ...

def ifind_ashare_enabled() -> bool: ...
def get_market_profile(data_source: str) -> MarketProfile: ...
```

- [ ] **Step 1：写失败测试**

覆盖：

- `SUPPORTED_DATA_SOURCES` 包含 `ifind_ashare`。
- `ENABLE_IFIND_ASHARE` 只有规范化后的 `true/1/yes/on` 启用。
- 功能关闭时选择 iFinD 抛出 `MarketDataSourceDisabled`。
- 功能开启但缺少 Token 时抛出 `MarketDataCredentialsError`，错误中不出现其他环境变量。
- `MarketProfile` 返回固定 6 股、`market=CN`、上海时区、60 分钟、规则策略、无 LLM、
  无 DJIA。
- `/config/features` 同时返回 vn.py 和 iFinD 功能状态，但不暴露 Token 是否存在。
- 正常导入 provider 不会发起网络请求，也不会读取或输出 Token。

- [ ] **Step 2：运行测试，确认失败**

```bash
pytest dashboard/backend/tests/infrastructure/market_data/test_provider.py \
  dashboard/backend/tests/test_market_data_features.py -v
```

- [ ] **Step 3：实现最小注册和配置**

在 `.env.example` 增加空值示例和安全注释：

```text
ENABLE_IFIND_ASHARE=false
IFIND_ACCESS_TOKEN=
IFIND_BASE_URL=https://quantapi.51ifind.com
```

工厂中的 iFinD 具体类使用函数内延迟导入。Task 4 之前用测试替身验证分支，不提前实现
HTTP 或数据转换逻辑。

- [ ] **Step 4：运行测试并提交**

```bash
pytest dashboard/backend/tests/infrastructure/market_data/test_provider.py \
  dashboard/backend/tests/test_market_data_features.py -v
git diff --check
git add .env.example dashboard/backend/infrastructure/market_data/profiles.py \
  dashboard/backend/infrastructure/market_data/provider.py \
  dashboard/backend/api/routers/config.py \
  dashboard/backend/tests/infrastructure/market_data/test_provider.py \
  dashboard/backend/tests/test_market_data_features.py
git commit -m "feat(market-data): register iFinD A-share profile"
```

---

## Task 2：实现安全、可测试的 iFinD HTTP Client

**文件：**

- 新建：`dashboard/backend/infrastructure/market_data/ifind_client.py`
- 新建：`dashboard/backend/tests/infrastructure/market_data/test_ifind_client.py`

**接口：**

```python
class IFindHttpClient:
    def fetch_hourly_bars(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> Mapping[str, object]: ...
```

- [ ] **Step 1：用注入式 HTTP session 写失败测试**

覆盖：

- 请求为 `POST /api/v1/high_frequency`。
- `codes` 是固定 6 股的逗号分隔字符串。
- `indicators` 严格为 `open,high,low,close,volume`，响应中的 `time` 由端点单独返回；
  `functionpara.Interval` 为 `60`。
- `start` 包含、`end` 不包含；请求使用 `end - 1 calendar day` 的 15:00 作为
  `effective-last-day`，Adapter 再按 `[start, end)` 过滤。
- 请求头含 `access_token`、`Content-Type`、`ifindlang`，但异常和日志不含 Token。
- 使用有限连接/读取超时。
- 连接错误、HTTP 429、HTTP 5xx 最多重试 2 次。
- 其他 HTTP 4xx、认证错误和 iFinD 业务错误不重试。
- 最终错误只包含端点、状态码、日期范围、股票数量和脱敏错误类型。
- 测试用假 session 断言没有真实网络请求。

- [ ] **Step 2：运行测试，确认失败**

```bash
pytest dashboard/backend/tests/infrastructure/market_data/test_ifind_client.py -v
```

- [ ] **Step 3：实现最小 Client**

要求：

- 构造函数允许注入 `session`、`base_url`、`token`、timeout 和重试等待函数。
- 生产默认域名为 `https://quantapi.51ifind.com`。
- 不使用 `Fill=Previous` 补造缺失行情。
- 重试间隔短且有上限，测试注入无等待函数。
- 返回解析后的 JSON mapping，业务结构转换留给 Task 3。

- [ ] **Step 4：运行测试并提交**

```bash
pytest dashboard/backend/tests/infrastructure/market_data/test_ifind_client.py -v
git diff --check
git add dashboard/backend/infrastructure/market_data/ifind_client.py \
  dashboard/backend/tests/infrastructure/market_data/test_ifind_client.py
git commit -m "feat(market-data): add iFinD HTTP client"
```

---

## Task 3：将官方 `tables` 响应转换成 ATL OHLCV

**文件：**

- 新建：`dashboard/backend/infrastructure/market_data/ifind_adapter.py`
- 新建：`dashboard/backend/tests/infrastructure/market_data/test_ifind_adapter.py`

**接口：**

```python
def response_to_frames(
    payload: Mapping[str, object],
    expected_symbols: Sequence[str],
    start: datetime,
    end: datetime,
    min_bars: int = 50,
) -> dict[str, pd.DataFrame]: ...
```

- [ ] **Step 1：用官方结构的模拟响应写失败测试**

覆盖：

- 只接受顶层 `errorcode == 0` 和 `tables` 数组。
- 6 个 symbol 必须一个不少、一个不多，symbol 不允许重复。
- 时间、open、high、low、close、volume 数组长度必须一致。
- 数字字符串可显式转换为数值，空值、NaN、Infinity 和非数字被拒绝。
- 时间转成 `Asia/Shanghai` 时区的 `DatetimeIndex`，命名为 `timestamp`，升序排列。
- 输出列顺序严格为 `open/high/low/close/volume`。
- 拒绝重复时间、越界时间、非交易时段数据和未升序原始时间。
- 拒绝非正价格、`high` 小于 open/close、`low` 大于 open/close、负成交量。
- 结果遵守 `[start, end)`。
- 每只股票少于 50 根有效 K 线时明确失败，并指出股票代码与实际数量。
- `errorcode != 0` 只保存脱敏业务错误摘要，不回显完整 payload。

- [ ] **Step 2：运行测试，确认失败**

```bash
pytest dashboard/backend/tests/infrastructure/market_data/test_ifind_adapter.py -v
```

- [ ] **Step 3：实现显式字段映射和校验**

不要用模糊字段猜测或字符串拼接解析。字段别名只能来自设计规格中确认的官方返回合同；
真实冒烟测试如发现合同变化，必须先补失败 fixture，再调整映射。

- [ ] **Step 4：运行测试并提交**

```bash
pytest dashboard/backend/tests/infrastructure/market_data/test_ifind_adapter.py -v
git diff --check
git add dashboard/backend/infrastructure/market_data/ifind_adapter.py \
  dashboard/backend/tests/infrastructure/market_data/test_ifind_adapter.py
git commit -m "feat(market-data): adapt iFinD tables to OHLCV"
```

---

## Task 4：组装 iFinD A 股 Provider

**文件：**

- 新建：`dashboard/backend/infrastructure/market_data/ifind_ashare.py`
- 新建：`dashboard/backend/tests/infrastructure/market_data/test_ifind_ashare.py`
- 修改：`dashboard/backend/infrastructure/market_data/provider.py`
- 修改：`dashboard/backend/tests/infrastructure/market_data/test_provider.py`

**接口：**

```python
class IFindAshareProvider:
    def fetch_bars(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]: ...
```

- [ ] **Step 1：写 Provider 失败测试**

覆盖：

- 只接受 `A_SHARE_DEMO_6_SYMBOLS`，顺序不同可规范化，集合不同必须拒绝。
- 一次批量请求 6 个 symbol，不循环发送 6 次请求。
- Client 返回值只经 Adapter 进入领域层。
- 每个结果都有固定列、上海时区、至少 50 根 K 线。
- Client、Adapter 任一失败时原样转成分类清晰的行情错误。
- 用 fail-on-call 替身证明不调用 Alpaca、vn.py、Yahoo 或其他 HTTP 客户端。
- 工厂只在功能开启且凭据存在时创建 Provider。

- [ ] **Step 2：运行测试，确认失败**

```bash
pytest dashboard/backend/tests/infrastructure/market_data/test_ifind_ashare.py \
  dashboard/backend/tests/infrastructure/market_data/test_provider.py -v
```

- [ ] **Step 3：实现薄编排层并接入工厂**

Provider 只负责固定股票池验证、调用 Client、调用 Adapter。HTTP 协议和 DataFrame
校验分别留在已有模块，避免把三种职责揉在一个类中。

- [ ] **Step 4：运行测试并提交**

```bash
pytest dashboard/backend/tests/infrastructure/market_data/test_ifind_ashare.py \
  dashboard/backend/tests/infrastructure/market_data/test_provider.py -v
git diff --check
git add dashboard/backend/infrastructure/market_data/ifind_ashare.py \
  dashboard/backend/infrastructure/market_data/provider.py \
  dashboard/backend/tests/infrastructure/market_data/test_ifind_ashare.py \
  dashboard/backend/tests/infrastructure/market_data/test_provider.py
git commit -m "feat(market-data): provide fixed iFinD A-share universe"
```

---

## Task 5：让回测引擎使用市场配置而不是 DJIA 硬编码

**文件：**

- 修改：`dashboard/backend/domain/backtesting/engine.py`
- 修改：`dashboard/backend/baseline_generator.py`（仅在现有接口无法接收固定股票池时）
- 修改：`dashboard/scripts/backtest_hourly_agent.py`
- 新建：`dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py`
- 修改：现有 engine/baseline 回归测试

- [ ] **Step 1：写引擎失败测试**

覆盖：

- iFinD 引擎从 `MarketProfile.symbols` 加载和循环 6 股，不引用 `DJIA_30/TOP_10`。
- 规则策略使用转换后的真实 OHLCV 指标输入，不调用 LLM。
- 买入持有基准等权使用同一 6 股、同一日期范围和同一价格数据。
- iFinD 模式不调用 `run_djia_baseline`，返回结果允许 DJIA run id 缺失。
- 规则策略 0 笔交易时运行仍成功，曲线和 metadata 正常保存。
- fail-on-call 替身证明不调用 Alpaca、vn.py 和 LLM。
- Alpaca/vn.py 模式仍使用原有美股股票池和 DJIA 基准。

- [ ] **Step 2：运行测试，确认现有 DJIA 硬编码导致失败**

```bash
pytest dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py \
  dashboard/backend/tests/backtesting/test_engine_move.py \
  dashboard/backend/tests/test_baseline_generator_offline.py -v
```

- [ ] **Step 3：最小参数化**

将 `load_data`、策略循环和基准股票列表统一改为 `self.profile.symbols`。仅按 profile 控制
LLM 和指数基准，不重写现有规则指标、成交模拟或基准收益算法。

- [ ] **Step 4：保存来源 metadata**

Agent 和买入持有运行至少保存：

```json
{
  "data_source": "ifind_ashare",
  "market": "CN",
  "universe": "a_share_demo_6",
  "timeframe": "60m",
  "timezone": "Asia/Shanghai",
  "decision_source": "rule_based",
  "benchmark": "equal_weight_buyhold"
}
```

历史记录缺少新增字段时继续兼容；任何 metadata 都不能含 Token 或完整上游响应。

- [ ] **Step 5：运行测试并提交**

```bash
pytest dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py \
  dashboard/backend/tests/backtesting/test_engine_move.py \
  dashboard/backend/tests/test_baseline_generator_offline.py -v
git diff --check
git add dashboard/backend/domain/backtesting/engine.py \
  dashboard/scripts/backtest_hourly_agent.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py
git add dashboard/backend/baseline_generator.py \
  dashboard/backend/tests/test_baseline_generator_offline.py
git commit -m "refactor(backtest): drive symbols and baselines by market profile"
```

只暂存实际修改的 baseline 文件；若无需修改，不加入该提交。

---

## Task 6：贯通 Backtest API、后台任务和 CLI

**文件：**

- 修改：`dashboard/backend/api/routers/backtests.py`
- 修改：`dashboard/scripts/backtest_hourly_agent.py`
- 修改：`dashboard/backend/tests/test_backtests_router.py`
- 修改：`dashboard/backend/tests/test_agent_runs_metadata.py`
- 修改：`dashboard/backend/tests/test_market_data_features.py`

- [ ] **Step 1：写 API/CLI 失败测试**

覆盖：

- `data_source=ifind_ashare` 和 `universe=a_share_demo_6` 被接受并传到后台任务与 CLI。
- 其他 universe、未知数据源和不支持的周期返回 HTTP 422。
- 功能关闭返回 HTTP 403。
- 功能开启但 Token 缺失时，在创建后台线程前返回 HTTP 503。
- 后台命令明确关闭 LLM，并且不传浏览器提交的自定义 `assets`。
- 运行响应和历史详情包含完整来源字段，旧运行仍可反序列化。
- 错误状态能清理 running 状态并保存脱敏错误摘要。

- [ ] **Step 2：运行测试，确认失败**

```bash
pytest dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/test_agent_runs_metadata.py \
  dashboard/backend/tests/test_market_data_features.py -v
```

- [ ] **Step 3：扩展请求模型和后台命令**

验证顺序为：数据源名称 -> 功能开关 -> 固定股票池 -> 凭据 -> 创建后台任务。这样配置
错误能同步返回，不会先创建一个注定失败的运行。

- [ ] **Step 4：运行测试并提交**

```bash
pytest dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/test_agent_runs_metadata.py \
  dashboard/backend/tests/test_market_data_features.py -v
git diff --check
git add dashboard/backend/api/routers/backtests.py \
  dashboard/scripts/backtest_hourly_agent.py \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/test_agent_runs_metadata.py \
  dashboard/backend/tests/test_market_data_features.py
git commit -m "feat(backtest): route iFinD A-share runs through API"
```

---

## Task 7：在现有 Backtest 页面增加受控的 A 股入口

**文件：**

- 修改：`dashboard/frontend/app.html`
- 修改：`dashboard/frontend/app.js`
- 修改：`dashboard/frontend/styles.css`（仅在现有样式不足时）
- 新建：`dashboard/backend/tests/test_ifind_ashare_frontend.py`

- [ ] **Step 1：写前端契约失败测试**

覆盖：

- `/config/features` 开启时动态出现 `iFinD A股`，关闭时不出现。
- 选择 iFinD 后固定显示 6 股 A 股股票池和 `60m`，禁止 DJIA、Mag 7、自定义股票池。
- 模型选择被禁用，提交请求使用 `data_source=ifind_ashare` 和
  `universe=a_share_demo_6`。
- 切回美股数据源后恢复用户先前的美股股票池和模型选择。
- 运行中、完成后和重新加载历史结果时显示 `iFinD A股 · 60m`。
- 缺少 DJIA run id 时不渲染 DJIA 图例、空曲线或误导文案。
- 403、503、认证、限流、结构错误和少于 50 bars 显示短而可操作的消息。
- HTML/JavaScript 中不存在 Access Token 输入框、存储键或日志输出。

- [ ] **Step 2：运行测试，确认失败**

```bash
pytest dashboard/backend/tests/test_ifind_ashare_frontend.py \
  dashboard/backend/tests/test_vnpy_simulation_frontend.py -v
```

- [ ] **Step 3：实现受控 UI 状态**

复用现有数据源选择器和股票池布局，不另建页面。A 股股票卡展示代码和中文简称；来源
标记使用现有 badge 体系。页面不解释功能原理，只给出运行必需的状态和错误。

- [ ] **Step 4：自动测试和视觉检查**

```bash
pytest dashboard/backend/tests/test_ifind_ashare_frontend.py \
  dashboard/backend/tests/test_vnpy_simulation_frontend.py -v
```

启动本地服务后，用 Playwright/浏览器检查桌面和移动视口：选择器、6 股列表、按钮、错误
状态、来源标记和无 DJIA 结果均无重叠或截断。真实 Token 仍不用于这一检查。

- [ ] **Step 5：提交**

```bash
git diff --check
git add dashboard/frontend/app.html dashboard/frontend/app.js \
  dashboard/backend/tests/test_ifind_ashare_frontend.py
git commit -m "feat(frontend): add controlled iFinD A-share backtest mode"
```

只有实际修改样式时，才在提交前额外运行
`git add dashboard/frontend/styles.css`。

---

## Task 8：完成离线端到端测试和全量回归

**文件：**

- 新建：`dashboard/backend/tests/integration/test_ifind_ashare_backtest.py`
- 新建：`dashboard/backend/tests/fixtures/ifind_hourly_bars.json`（若 fixture 大于适合内联的体积）
- 修改：受本次接入影响的现有回归测试

- [ ] **Step 1：写离线端到端失败测试**

使用确定性的官方 `tables` 形状 fixture，覆盖：

- 请求进入 iFinD Provider 并得到 6 个合规 DataFrame。
- 规则策略完成运行，0 笔或多笔交易都合法。
- 等权买入持有基准完成运行。
- 结果只有两类曲线，不存在 DJIA run id。
- API、后台任务、CLI、引擎和运行记录的来源字段一致。
- fail-on-call 替身确认全流程没有真实 HTTP、Alpaca、vn.py、Yahoo 或 LLM 调用。
- 日志、异常、序列化结果和 fixture 中都不包含测试 Token。

- [ ] **Step 2：运行端到端测试，修复最小跨层问题**

```bash
pytest dashboard/backend/tests/integration/test_ifind_ashare_backtest.py -v
```

- [ ] **Step 3：运行相关测试集合**

```bash
pytest dashboard/backend/tests/infrastructure/market_data \
  dashboard/backend/tests/backtesting \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/test_agent_runs_metadata.py \
  dashboard/backend/tests/test_market_data_features.py \
  dashboard/backend/tests/test_ifind_ashare_frontend.py -v
```

- [ ] **Step 4：运行完整后端测试**

```bash
pytest dashboard/backend/tests -v
git diff --check
git status --short
```

如果完整套件出现本分支基线之外的已知失败，先在干净 `origin/main` 复现并记录；不能通过
删除断言、跳过测试或改变无关行为来掩盖。

- [ ] **Step 5：提交集成测试或必要修复**

```bash
git add dashboard/backend/tests/integration/test_ifind_ashare_backtest.py
git commit -m "test(backtest): cover iFinD A-share flow offline"
```

若创建了 JSON fixture，则在提交前额外运行
`git add dashboard/backend/tests/fixtures/ifind_hourly_bars.json`。跨层测试发现的实现缺陷应
回到所属 Task，以“失败测试 + 对应源码”单独提交；Task 8 不使用模糊的批量暂存命令。

---

## Task 9：权限开通后的真实 iFinD 冒烟测试

**外部前置条件：** iFinD 审核通过、HTTP API Token 已取得、历史高频权限可用；如果
服务要求 IP 白名单，本机出口 IP 已绑定。

这一任务不进入默认 CI，也不把凭据写入 Git。用户只在本地 `.env` 配置：

```text
ENABLE_IFIND_ASHARE=true
IFIND_ACCESS_TOKEN=<本地真实值，不发送到聊天或提交到 Git>
```

- [ ] **Step 1：单股票、小日期范围探针**

直接调用 Client 检查 HTTP 状态、`errorcode`、`tables` 字段名和时间格式。只记录状态、
字段名、行数和脱敏错误类型。

- [ ] **Step 2：6 股、最长 31 天 Provider 冒烟**

确认每只至少 50 bars、上海时区、`[start, end)`、OHLCV 不变量以及无缺失股票。

- [ ] **Step 3：完整页面回测**

从 Backtest 页面选择 `iFinD A股`，确认规则策略和等权买入持有完成、来源标记正确、
DJIA 不出现、0 笔交易也不会被误报为失败。

- [ ] **Step 4：根据真实合同闭环**

若真实响应与 fixture 不一致，先把脱敏后的最小结构写成失败测试，再修改 Client 或
Adapter；禁止为了“先跑起来”放宽股票完整性、OHLCV 或 50 bars 校验。

真实冒烟通过后才满足最终停止条件；在权限未开通期间，Loop 0 的离线实现仍可独立完成。

---

## 最终验收命令

```bash
pytest dashboard/backend/tests -v
git diff --check
git status --short
```

最终人工验收：

1. iFinD 功能关闭时，旧 ATL 行为不变。
2. 功能开启但缺 Token 时，请求在创建后台任务前返回 503。
3. 真实权限可用时，6 股各有至少 50 根合法 60 分钟 K 线。
4. 回测只展示规则策略与同 6 股等权买入持有，不出现 DJIA。
5. 0 笔交易合法，错误不触发数据源回退。
6. 页面、日志、metadata、测试输出和 Git 历史均不含 Token。

## Loop 停止条件

- **Loop 0 停止：** 离线单元、API、前端和端到端测试全部通过，完整回归无新增失败。
- **Loop 1 停止：** 真实 iFinD 6 股回测成功，数据校验、来源标记、两类曲线和脱敏要求
  全部满足。
- **不以收益停止：** 盈利、跑赢基准或产生交易都不是本次数据接入的验收标准。
