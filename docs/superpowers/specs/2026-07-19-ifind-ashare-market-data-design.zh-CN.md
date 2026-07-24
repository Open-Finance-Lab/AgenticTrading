# 同花顺 iFinD A 股历史行情接入设计规格

- **日期：** 2026-07-19
- **状态：** 设计已逐段确认，等待书面规格复核
- **项目：** AgenticTrading（下文简称 ATL）
- **分支：** `feat/ifind-ashare-market-data`
- **迭代：** iFinD/A 股数据接入第 1 轮

## 1. 背景

ATL 当前主要使用 Alpaca 美股小时 K 线，并已通过 `MarketDataProvider` 支持
`alpaca` 和开发用 `vnpy_simulation` 两个行情源。回测引擎消费统一的 OHLCV
DataFrame，但股票池、DJIA 基准和部分执行循环仍然包含美股/DJIA 硬编码。

本项目需要增加 A 股数据能力。用户当前只有普通同花顺账号，已于 2026-07-19
提交同花顺数据接口试用申请，尚未获得 iFinD HTTP API 权限或 Access Token。
开发设备为 macOS，因此采用同花顺官方 HTTP API，不依赖 Windows/Linux SDK，
也不通过证券账户或交易客户端。

## 2. 目标

在现有 ATL Backtest 页面新增一个受功能开关保护的 `iFinD A股` 行情源。用户
选择后，系统使用同花顺 iFinD 官方 HTTP API 获取固定 6 只 A 股的真实历史
60 分钟 K 线，转换为 ATL 的标准 OHLCV DataFrame，并运行现有规则策略回测。

第一轮的用户可见结果包括：

- 规则策略权益曲线和指标；
- 同一组 6 只股票的等权买入持有基准；
- 模拟成交记录（可以为空）；
- 清晰、可持久化的数据来源标记；
- 明确的权限、上游接口和数据校验错误。

本轮验证的是“同花顺真实 A 股历史数据能够进入 ATL 并完成回测”，不验证策略
盈利能力，也不验证实时行情或真实交易。

## 3. 已确认需求

| 项目 | 决策 |
|---|---|
| 数据供应商 | 同花顺 iFinD 官方数据接口 |
| 访问方式 | macOS 本地后端调用官方 HTTP API |
| 行情类型 | 历史 60 分钟 K 线 |
| 初始股票池 | 固定 6 只代表性 A 股 |
| 交付入口 | 现有 ATL Backtest 页面 |
| 决策方式 | 现有规则策略，不调用 LLM |
| 比较基准 | 同一 6 股票等权买入持有 |
| 开发节奏 | 权限审核期间用官方形状的模拟 HTTP 响应开发；开通后做真实只读验收 |
| 数据错误行为 | 明确失败，禁止静默切换到其他行情源 |
| 交易范围 | 仅 ATL 内部模拟成交，不接券商或真实订单 |

固定股票池为：

```text
600519.SH  贵州茅台
601318.SH  中国平安
600036.SH  招商银行
000001.SZ  平安银行
000858.SZ  五粮液
300750.SZ  宁德时代
```

## 4. 外部前置条件

真实验收依赖以下外部状态：

1. 同花顺完成 iFinD 数据接口试用审核；
2. 试用权限包含 A 股历史高频/分钟序列；
3. 用户取得可供 HTTP API 使用的 Access Token；
4. 如果试用账号要求 IP 绑定，当前 Mac 出口 IP 已按同花顺要求登记；
5. 当前网络可以访问 `https://quantapi.51ifind.com`。

这些条件不阻塞离线实现和模拟响应测试，但会阻塞真实 API 冒烟测试。Access Token、
短信验证码、账号密码和完整上游响应不得发送到聊天、提交到 Git 或写入运行记录。

## 5. 不在本轮范围内

- iFinD 实时行情、实时订阅或逐笔数据；
- 同花顺普通 App 的非官方抓取、逆向接口或浏览器自动化取数；
- 同花顺交易、券商登录、开户、下单、撤单和持仓同步；
- iFinD Windows/Linux SDK、`iFinDPy` 或 `vnpy_ifind`；
- 把 iFinD 数据先转换为 vn.py `BarData`；
- 沪深 300 全成分、用户自定义 A 股股票池或跨市场组合；
- 沪深 300 指数基准、DJIA 基准或其他外部指数基准；
- LLM Agent、子 Agent Pipeline 或 A 股专用模型提示词；
- 长期历史缓存、实时数据服务或独立 iFinD Worker；
- 将回测收益视为投资建议或真实投资表现。

## 6. 备选方案

### 6.1 官方 iFinD HTTP API 直接接入 ATL（采用）

iFinD HTTP 响应由专用适配器直接转换为 ATL 标准 DataFrame，再交给现有回测消费端。
该路线适合 macOS，依赖最少，便于用模拟 HTTP 响应测试，也不会把 iFinD 原始格式
泄漏到领域层。

### 6.2 iFinD 经 vn.py 再进入 ATL（不采用）

该路线可以让外部行情都经过 vn.py，但第一轮会增加 `iFinDPy`/`vnpy_ifind`、
`BarData` 和 ATL DataFrame 两次转换。它不适合当前 Mac HTTP 环境，也不能为本轮
数据验收增加有效证明。

### 6.3 独立 iFinD Worker（不采用）

独立服务可以隔离凭据和上游连接，适合后续服务器部署，但第一轮会额外引入进程管理、
通信协议、健康检查和部署配置。当前只读、单机、低频历史数据不需要这层复杂度。

## 7. 总体架构

```text
ATL Backtest 页面
  |
  | POST /backtest/run
  | data_source=ifind_ashare
  | universe=a_share_demo_6
  v
Backtest API 校验与功能开关
  |
  v
MarketDataProvider 工厂
  |-- alpaca          -> AlpacaDataLoader（保持现状）
  |-- vnpy_simulation -> VnpySimulationProvider（保持现状）
  `-- ifind_ashare    -> IFindAshareProvider（新增）
                             |
                             v
                       IFindHttpClient
                             |
                             | POST /api/v1/high_frequency
                             v
                       iFinD 官方 HTTP API
                             |
                             v
                       IFindBarAdapter
                             |
                             | Dict[str, DataFrame]
                             v
    技术指标 -> 规则决策 -> 模拟成交 -> 权益/指标/交易记录
                             |
                             v
       Agent 曲线 + 6 股票等权买入持有基准
```

浏览器只提交非秘密配置。Access Token 仅由后端进程读取和发送给 iFinD；数据适配完成
后，回测领域层不理解 iFinD 请求头、`errorcode`、`tables` 或其他供应商字段。

## 8. 数据源、市场配置和股票池

### 8.1 数据源常量

行情提供器注册表增加：

```text
IFIND_ASHARE = "ifind_ashare"
```

`SUPPORTED_DATA_SOURCES` 增加该值。现有 `alpaca` 默认行为保持不变。

### 8.2 市场配置

新增一个后端唯一来源的 A 股演示市场配置，至少包含：

```text
key: a_share_demo_6
market: CN
timezone: Asia/Shanghai
timeframe: 60m
symbols: 固定 6 只股票
benchmark: equal_weight_buyhold
llm_enabled: false
```

`data_source=ifind_ashare` 必须绑定此配置。后端不能信任浏览器传入的任意 A 股代码；
即使请求同时带有其他 `assets`，第一轮也只使用后端固定的 6 股票配置。

### 8.3 消除回测引擎中的 DJIA 数据循环硬编码

`HourlyBacktester` 增加由市场配置解析出的 `self.symbols`，并在以下位置使用它：

- `fetch_bars` 请求股票列表；
- 每个时间步构建 `market_data`；
- 价格缓存和指标计算；
- 规则策略的当前市场信号；
- 等权买入持有基准。

此修改仅把数据循环的股票列表参数化。美股路径仍解析为原有 DJIA-30，保持向后兼容。
LLM 路径仍然只支持原有美股白名单；`ifind_ashare` 强制 `use_llm=false`，因此本轮不改
LLM 验证器或提示词。

## 9. iFinD HTTP Client

### 9.1 配置

后端从环境变量读取：

```text
ENABLE_IFIND_ASHARE=true
IFIND_ACCESS_TOKEN=<secret>
```

可选测试配置：

```text
IFIND_BASE_URL=https://quantapi.51ifind.com
```

生产默认值固定为官方域名。`IFIND_BASE_URL` 仅用于测试替身或受控开发环境，不允许
浏览器传入。`.env.example` 只提供空占位符，`.env` 继续由 Git 忽略。

### 9.2 请求

第一轮使用官方高频序列端点：

```text
POST https://quantapi.51ifind.com/api/v1/high_frequency
```

请求头：

```json
{
  "Content-Type": "application/json",
  "access_token": "<IFIND_ACCESS_TOKEN>",
  "ifindlang": "cn"
}
```

请求体使用一次批量请求获取 6 只股票：

```json
{
  "codes": "600519.SH,601318.SH,600036.SH,000001.SZ,000858.SZ,300750.SZ",
  "indicators": "open,high,low,close,volume",
  "starttime": "<start> 09:30:00",
  "endtime": "<effective-last-day> 15:00:00",
  "functionpara": {
    "Interval": "60",
    "CPS": "forward1",
    "Timeformat": "LocalTime",
    "Limitstart": "09:30:00",
    "Limitend": "15:00:00"
  }
}
```

不发送 `Fill=Previous`，避免把前值填充伪装成真实成交 K 线。ATL 的持仓估值价格缓存
仍可在已存在的合法 K 线之间使用最后价格，但输入行情本身不得伪造缺失 bar。

ATL provider 合同继续使用 `start` 包含、`end` 不包含的日期语义。Client 将
`end - 1 calendar day` 作为请求的 `effective-last-day`；周末和节假日由 iFinD 返回
实际交易日数据，适配器最终再次按 `[start, end)` 过滤。

### 9.3 超时和重试

- 连接超时 3 秒，读取超时 20 秒；
- 对连接错误、HTTP 429 和 HTTP 5xx 最多重试 2 次；
- 重试退避为 0.5 秒和 1.0 秒；
- HTTP 4xx（除 429）、权限错误和 iFinD 业务 `errorcode` 不重试；
- 最终失败后终止本次运行，不切换行情源。

日志只记录端点、股票数量、日期范围、状态码和脱敏错误类型，绝不记录请求头、Token
或完整上游响应。

## 10. iFinD 响应适配

### 10.1 接受的响应合同

第一轮只接受官方示例使用的 `tables` 结构：

```json
{
  "errorcode": 0,
  "errmsg": "",
  "tables": [
    {
      "thscode": "600519.SH",
      "time": ["2026-04-01 10:30:00"],
      "table": {
        "open": ["..."],
        "high": ["..."],
        "low": ["..."],
        "close": ["..."],
        "volume": ["..."]
      }
    }
  ]
}
```

不对未知顶层 `data`、嵌套字符串 JSON 或其他未见过的结构做猜测性兼容。真实冒烟测试
如果证明试用账号返回合同不同，Loop 1 记录脱敏的结构摘要并只调整适配器测试与映射。

### 10.2 逐股票校验

适配器必须执行：

1. `errorcode` 必须为 `0`；
2. 返回的 `thscode` 集合必须与请求的 6 股票集合完全一致；
3. 每只股票只允许一个 table；
4. `time/open/high/low/close/volume` 数组长度必须一致；
5. 时间戳按 iFinD LocalTime 解析并明确本地化为 `Asia/Shanghai`；
6. 最终索引必须严格递增且没有重复；
7. `open/high/low/close` 必须是有限正数；
8. `high >= max(open, close)`；
9. `low <= min(open, close)`；
10. `volume` 必须是有限非负数；
11. 时间戳必须落在 `[start, end)` 和 A 股交易时段内；
12. 每只股票在完整回测前至少有 50 根有效 K 线。

A 股 60 分钟 bar 预期以 bar 结束时间标记，并保留午间休市间隔。适配器不生成午休、
周末或节假日 bar，也不通过前值填充补齐缺失行情。

### 10.3 ATL 标准输出

输出继续满足 `MarketDataProvider` 合同：

```python
dict[str, pd.DataFrame]
```

每个 DataFrame：

- 索引为带 `Asia/Shanghai` 时区的 `DatetimeIndex`；
- 索引名为 `timestamp`；
- 列顺序为 `open, high, low, close, volume`；
- 所有数据列为数值类型；
- 股票键保留 iFinD 的 `.SH`/`.SZ` 后缀。

## 11. Backtest API、CLI 和运行记录

### 11.1 API

`BacktestRunRequest.data_source` 和查询参数 allow-list 增加 `ifind_ashare`。API 在创建
后台线程前完成：

- 数据源名称校验；
- `ENABLE_IFIND_ASHARE` 功能开关校验；
- Access Token 是否存在的可用性校验；
- 市场配置和固定股票池解析。

第一轮请求可以携带 `universe=a_share_demo_6`，但后端必须验证其与数据源绑定，不能
接受其他 universe。现有前端 `assets` 参数不作为 A 股股票池授权来源。

### 11.2 CLI 和回测引擎

后台运行器继续把 `--data-source ifind_ashare` 传给现有 CLI。选择该数据源时：

- 强制添加 `--no-llm`；
- 忽略模型和 Pipeline 配置；
- 使用 A 股固定市场配置；
- 不初始化 Alpaca、vn.py 或 LLM 客户端。

### 11.3 数据来源元数据

Agent 和买入持有基准运行的 metadata 至少保存：

```json
{
  "data_source": "ifind_ashare",
  "market": "CN",
  "universe": "a_share_demo_6",
  "timeframe": "60m",
  "timezone": "Asia/Shanghai",
  "decision_source": "rule_based"
}
```

不保存 Token、请求头或完整上游响应。历史运行缺少这些字段时保持当前兼容行为。

## 12. 基准与结果展示

### 12.1 等权买入持有

A 股模式只生成以下两条曲线：

1. 规则策略 Agent；
2. 相同 6 股票、相同 iFinD K 线、相同初始资金的等权买入持有。

等权基准在第一个全部股票均有有效价格的时间点平均分配资金，之后只估值、不再调仓。
允许保留因整股数量产生的现金余额。

### 12.2 禁止 DJIA 误导

`ifind_ashare` 模式不生成 DJIA 运行，不设置 `baseline_djia_run_id`，图表和文案不显示
DJIA。现有 Alpaca/vn.py 美股路径继续显示原基准。图表接口必须允许 DJIA 基准缺失，
并仍然返回 Agent 与买入持有两条有效序列。

### 12.3 交易可以为空

真实行情在选定日期内可能不会触发规则策略。`num_trades=0` 是合法结果；系统不能为了
演示效果强行制造交易。验收要求非空权益曲线和指标，不要求盈利或至少一笔交易。

## 13. 前端行为

### 13.1 功能状态

现有功能状态接口增加：

```json
{"ifind_ashare_enabled": true}
```

只有后端功能开关启用时，页面才动态添加：

```text
iFinD A股（60分钟）
```

页面不能通过读取本地文件或猜测环境变量决定是否显示。
功能状态不暴露 Token 是否存在；如果开关已启用但后端缺少 Token，创建运行时按
第 14 节返回 HTTP 503。这使功能开关语义稳定，也不会通过公开配置接口泄露秘密配置
状态。

### 13.2 受控选择

选择 iFinD 后：

- 自动选择并锁定 `A股代表6只` 股票池；
- 隐藏或禁用 DJIA、Magnificent 7 和自定义股票池；
- 模型显示为 `Rule-based (data integration)` 并禁用；
- 不提交模型或 Pipeline；
- 保持现有开始/结束日期控件和 31 天上限；
- 页面持续显示：
  `Historical iFinD A-share data for paper backtesting; not real-time and no real orders.`

切回 Alpaca 时恢复用户之前的美股股票池和模型选择，不永久覆盖页面状态。

### 13.3 结果来源

运行中、运行完成和重新加载历史结果后都显示 `iFinD A股 · 60m` 来源标记。错误状态
使用可操作的中文或英文短消息，不显示 Token、手机号、账号或完整上游响应。

## 14. 错误处理

| 情况 | 行为 |
|---|---|
| 未知 `data_source` | HTTP 422，不创建后台任务 |
| iFinD 功能开关关闭 | HTTP 403，不创建后台任务 |
| 缺少 Access Token | HTTP 503，提示配置 `IFIND_ACCESS_TOKEN` |
| Token 无效或无高频权限 | 运行失败，提示认证/权限问题，不回显凭据 |
| HTTP 429/5xx/网络超时 | 按有限重试策略执行；耗尽后运行失败 |
| iFinD `errorcode != 0` | 运行失败，保存脱敏业务错误摘要 |
| 响应结构未知 | 运行失败并记录结构键名摘要，禁止猜测解析 |
| 股票集合不完整 | 运行失败，列出缺失股票代码 |
| 少于 50 根有效 K 线 | 运行失败，提示扩大日期范围或检查权限 |
| OHLCV/时区/重复时间无效 | 运行失败，指出股票、时间和违规字段 |
| 后台子进程失败 | 清理运行状态并向页面返回错误 |

任何 iFinD 错误都禁止静默回退到 Alpaca、vn.py 模拟数据、免费 A 股接口或本地伪造
行情。数据来源比“尽量完成一条曲线”更重要。

## 15. 测试设计

### 15.1 单元测试

- 固定 6 股票常量和市场配置；
- iFinD 请求端点、请求头、批量 codes、Interval 和日期边界；
- Token 缺失且错误消息不包含秘密；
- 连接错误、429、5xx 的有限重试；
- 4xx 和业务错误不重试；
- 官方 `tables` 模拟响应逐字段转为 OHLCV；
- `.SH`/`.SZ` 代码保留；
- naive LocalTime 明确本地化为 `Asia/Shanghai`；
- 数组长度不一致、重复时间、错误价格和负成交量被拒绝；
- 股票缺失或每只少于 50 根数据被拒绝；
- 输出列、类型、排序和 `[start, end)` 日期边界。

### 15.2 Provider、API 和运行器测试

- `alpaca` 默认行为不变；
- 功能关闭时 `ifind_ashare` 返回 403；
- 缺 Token 时在创建线程前返回 503；
- 数据源、市场配置和固定股票池通过 API、后台线程、CLI 和引擎传递；
- iFinD 模式强制 `use_llm=false`；
- iFinD 模式不会构造 Alpaca、vn.py 或 LLM 客户端；
- metadata 保存完整来源字段且不包含秘密；
- 失败后运行状态可再次启动，不残留全局锁定。

### 15.3 基准与前端测试

- iFinD 模式生成 Agent 和 6 股票等权买入持有；
- iFinD 模式不生成或显示 DJIA；
- 图表接口可处理 `baseline_djia_run_id=None`；
- 功能开关控制 iFinD 选项；
- 选择 iFinD 后锁定 A 股股票池并禁用模型；
- 请求发送 `ifind_ashare` 和 `a_share_demo_6`；
- 切回 Alpaca 恢复原页面状态；
- 运行和历史结果持续显示 iFinD 来源。

### 15.4 离线端到端测试

使用本地 HTTP 替身返回官方形状的 6 股票响应，并设置“任何真实网络、Alpaca、vn.py
或 LLM 调用都会立即失败”的防线。运行完整回测，确认：

- 6 股票均进入指标计算；
- Agent 和等权买入持有曲线非空；
- 指标和运行记录成功生成；
- 交易记录允许为空；
- 所有结果来源均为 `ifind_ashare`。

真实 iFinD 测试不进入默认 CI，避免消耗试用额度和依赖外部权限。

## 16. Loop Engineering 执行模型

| Loop 要素 | 本项目定义 |
|---|---|
| 任务输入 | 6 股票、`[start,end)`、60 分钟周期、iFinD 模拟或真实响应 |
| 执行单元 | 一次 Provider 拉取和转换；随后一次完整规则回测 |
| 评估标准 | 数据集合完整、每只至少 50 bars、OHLCV 合法、曲线非空、来源正确 |
| 反馈修正 | 认证问题修配置；结构问题修适配器；时间问题修请求；消费问题修股票池硬编码 |
| 外部状态 | iFinD 审核、历史高频权限、Token、IP 绑定和网络 |
| 停止条件 | 真实 6 股票 60 分钟数据在 ATL 页面完成回测且所有自动测试通过 |

### 16.1 Loop 0：权限审核期间

1. 用官方形状的模拟 HTTP JSON 开发 Client、Adapter 和 Provider；
2. 完成 Provider 工厂、功能开关和 API/CLI 传递；
3. 参数化回测股票池并实现 A 股等权基准；
4. 完成页面选择、锁定、来源标记和错误状态；
5. 通过全部离线单元、API、前端和端到端测试。

### 16.2 Loop 1：试用权限开通后

1. 用户只在本地 `.env` 配置 Token；
2. 使用 1 只股票和短日期范围做只读认证/权限冒烟；
3. 使用 6 股票和不超过 31 天的范围检查真实数据结构与数量；
4. 如真实合同与模拟合同不同，仅修改适配器和对应测试；
5. 从 ATL 页面运行完整规则回测；
6. 检查两条曲线、指标、来源标记和无真实交易事实；
7. 记录脱敏验收结果。

## 17. 验收标准与停止条件

第一轮只有在以下条件全部满足时完成：

- iFinD 试用账号真实返回固定 6 股票的 60 分钟历史行情；
- 每只股票至少有 50 根有效 K 线；
- 所有 DataFrame 通过字段、数值、时区、排序和重复校验；
- ATL Backtest 页面能选择 iFinD 并完成规则回测；
- Agent 与 6 股票等权买入持有曲线和指标非空；
- 页面和数据库记录的数据来源均正确；
- 没有调用 LLM、券商或真实交易接口；
- iFinD 失败时明确失败且没有任何行情源回退；
- 相关自动测试全部通过，现有 Alpaca 和 vn.py 测试无回归。

策略盈利、Sharpe 高低、是否至少一笔交易和是否优于买入持有都不是数据接入的停止条件。

## 18. 后续迭代

完成本轮后，以下内容必须分别重新设计和审批：

1. 沪深 300 或用户自定义 A 股股票池；
2. 沪深 300 指数基准；
3. iFinD 实时行情和本地缓存；
4. LLM Agent 的 A 股白名单、提示词、交易单位和风险规则；
5. vn.py A 股 Gateway 或 `vnpy_ifind` 统一入口；
6. 券商模拟交易和真实交易。

## 19. 官方参考

- 同花顺数据接口主页：
  <https://quantapi.51ifind.com/gwstatic/static/ds_web/quantapi-web/>
- 同花顺数据接口基础概念与平台说明：
  <https://quantapi.51ifind.com/gwstatic/static/ds_web/quantapi-web/help-center/faq.html>
- 同花顺数据接口产品手册（高频序列与 HTTP 参数）：
  <https://quantapi.51ifind.com/gwstatic/static/ds_web/quantapi-web/help-center/manual.html>
- 同花顺数据接口应用示例：
  <https://quantapi.10jqka.com.cn/gwstatic/static/ds_web/quantapi-web/example.html>
- 同花顺数据接口使用流程：
  <https://quantapi.51ifind.com/gwstatic/static/ds_web/quantapi-web/help-center/deploy.html>
