# iFinD 沪深 300 行业均衡 20 股股票池设计

- **日期：** 2026-07-24
- **状态：** 设计已逐段确认，等待书面规格复核
- **项目：** AgenticTrading（下文简称 ATL）
- **分支：** `feat/ifind-ashare-market-data`
- **迭代：** iFinD/A 股数据接入第 2 轮

## 1. 背景

ATL 已在当前功能分支接入同花顺 iFinD HTTP API，可以获取固定 6 只 A 股的真实
60 分钟历史 OHLCV 行情，并运行规则 Agent 回测和同股票池等权买入持有基准。现有
实现把 `ifind_ashare` 数据源与唯一的 `a_share_demo_6` 股票池绑定，前后端均假设
一个数据源只有一个股票池。

本轮保留原有 6 股演示池，同时新增一个固定、带版本号、覆盖沪深 300 全部一级行业的
20 股测试池。目标是扩大 ATL 对 A 股跨行业数据和 Agent 回测链路的测试覆盖，不是把
沪深 300 全部 300 只股票接入，也不是提供真实证券交易。

当前功能分支相对 `origin/main` 同时有本地提交和上游新提交。功能实现前必须重新获取
并合并最新 `origin/main`，以最新 ATL 的 Run Backtest 弹窗、资产选择和资金配置流程
为基础实现本设计。

## 2. 目标与成功标准

本轮需要实现：

1. 保留 `a_share_demo_6`，继续作为 iFinD 默认和兼容股票池。
2. 新增 `csi300_sample_20_2026h2` 固定股票池。
3. 在最新 Run Backtest 弹窗中显式选择行情源和对应股票池。
4. 使用 iFinD 一次批量请求 20 只股票的真实 60 分钟行情。
5. 使用同一批标准化行情运行规则 Agent 和 20 股等权买入持有基准。
6. 在运行配置中持久化数据源、股票池、股票数量和周期。
7. 对缺失、不完整或非法数据明确失败，不静默缩小股票池或切换数据源。

完成标准是：最新主线已合入、原 6 股和新 20 股均可选择、离线自动测试通过、真实
iFinD 20 股批量取数和完整回测通过、浏览器英文界面通过检查，且 Git 中没有 Token、
本地数据库或其他秘密信息。

## 3. 范围边界

### 3.1 本轮包含

- iFinD 历史 60 分钟 A 股行情；
- 两个后端注册的固定 iFinD 股票池；
- 最新 Backtest 页面中的行情源和股票池选择；
- 现有规则 Agent；
- 同股票池等额分配、买入并持有基准；
- 离线、真实 API、回归和浏览器验收。

### 3.2 本轮不包含

- 沪深 300 全部 300 只股票；
- 根据最新成分股自动调整股票池；
- 用户自定义 A 股代码；
- 1 分钟、5 分钟、15 分钟、日线或实时行情；
- LLM Agent 或 A 股专用提示词；
- A 股 100 股整手、T+1、涨跌停、停牌撮合等专用交易规则；
- iFinD 研报、新闻、财务数据或选股接口；
- 同花顺、券商或任何真实账户的下单、撤单、持仓同步；
- 将回测结果解释为投资建议或真实投资表现。

## 4. 已确认的核心决策

| 项目 | 决策 |
|---|---|
| 原股票池 | 保留 `a_share_demo_6` |
| 新股票池 | 新增 `csi300_sample_20_2026h2` |
| 数据源 | 两个股票池均使用 `ifind_ashare` |
| 股票池维护 | 固定并带版本号，不自动随指数调样 |
| 选样方法 | 重新选取行业均衡 20 股，不在原 6 股上简单追加 |
| 周期 | 固定 `60m` |
| 日期范围 | 最长 31 个自然日 |
| 数据最低要求 | 每只股票至少 50 根合法 K 线 |
| Agent | 仅现有规则 Agent，禁止 LLM |
| 比较基准 | 同一 20 股等额分配、买入并持有 |
| 数据失败 | 整次运行失败，禁止静默降级或替代 |
| 界面语言 | 所有新增显示文本使用英文 |
| 远程操作 | 完成后只创建本地 commit，未经明确要求不 push |

`60m` 与 ATL 当前美股 Alpaca 回测周期一致。它不是 iFinD 的硬性限制，而是本轮统一
数据粒度的设计选择。A 股通常每天得到 4 根 60 分钟 K 线；31 天窗口能够覆盖至少 50
根数据并支持 `SMA50` 等现有指标，同时避免分钟级数据带来的额外数据量和复杂度。

## 5. 固定 20 股名单

股票池标识：

```text
csi300_sample_20_2026h2
```

用户可见英文名称：

```text
CSI 300 Sample 20 (2026 H2)
```

固定顺序和名单如下：

| 代码 | 中文简称 | 英文显示名 | 中证一级行业 |
|---|---|---|---|
| `600519.SH` | 贵州茅台 | Kweichow Moutai | Consumer Staples |
| `601318.SH` | 中国平安 | Ping An Insurance | Financials |
| `600036.SH` | 招商银行 | China Merchants Bank | Financials |
| `300750.SZ` | 宁德时代 | CATL | Industrials |
| `000333.SZ` | 美的集团 | Midea Group | Consumer Discretionary |
| `002594.SZ` | 比亚迪 | BYD | Consumer Discretionary |
| `600276.SH` | 恒瑞医药 | Hengrui Medicine | Health Care |
| `300760.SZ` | 迈瑞医疗 | Mindray | Health Care |
| `688981.SH` | 中芯国际 | SMIC | Information Technology |
| `002415.SZ` | 海康威视 | Hikvision | Information Technology |
| `601766.SH` | 中国中车 | CRRC | Industrials |
| `600309.SH` | 万华化学 | Wanhua Chemical | Materials |
| `601899.SH` | 紫金矿业 | Zijin Mining | Materials |
| `601857.SH` | 中国石油 | PetroChina | Energy |
| `600900.SH` | 长江电力 | China Yangtze Power | Utilities |
| `600050.SH` | 中国联通 | China Unicom | Communication Services |
| `000725.SZ` | 京东方 A | BOE Technology | Information Technology |
| `600030.SH` | 中信证券 | CITIC Securities | Financials |
| `600887.SH` | 伊利股份 | Yili | Consumer Staples |
| `600048.SH` | 保利发展 | Poly Developments | Real Estate |

行业数量分布为：信息技术 3、金融 3；工业、可选消费、主要消费、医药卫生和原材料
各 2；能源、公用事业、通信服务和房地产各 1。合计 20 只并覆盖当前沪深 300 的全部
11 个中证一级行业。

名单和行业已于 2026-07-24 通过中证指数官网的当前样本查询核验。官网沪深 300
详情和官方样本文件入口为：

```text
https://www.csindex.com.cn/#/indices/family/detail?indexCode=000300
https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000300cons.xls
```

该名单是 2026 年下半年版本化测试样本。未来指数调样不会自动修改它；需要更新时应
创建新的股票池标识和独立设计，保证历史回测可复现。

## 6. 方案比较

### 6.1 `data_source + universe` 组合注册表（采用）

市场配置改为由 `(data_source, universe)` 唯一定位。一个数据源可以注册多个股票池，
后端仍然拥有完整的代码和市场规则。该方案直接解决 iFinD 一源多池问题，也为未来
增加其他受控股票池保留清晰扩展点。

### 6.2 把 6 股直接替换成 20 股（不采用）

实现最少，但会破坏已经验证的演示路径和历史运行配置，也无法比较小股票池与扩大后
股票池的行为。

### 6.3 把每个股票池建成独立数据源（不采用）

例如创建 `ifind_demo6` 和 `ifind_csi300_sample20`。这会把数据供应商与股票池概念混在
一起，导致 client、凭证、功能开关和界面选项重复。

## 7. 配置架构

市场配置注册表使用二元键：

```text
(alpaca, djia_30)
(vnpy_simulation, djia_30)
(ifind_ashare, a_share_demo_6)
(ifind_ashare, csi300_sample_20_2026h2)
```

每个 `MarketProfile` 继续拥有：

- `data_source`
- `market`
- `universe`
- `timezone`
- `timeframe`
- `symbols`
- `benchmark`
- `decision_source`
- `llm_enabled`
- `index_baseline_enabled`

`get_market_profile(data_source, universe=None)` 负责解析配置。为兼容现有调用：

- `alpaca` 缺省为 `djia_30`；
- `vnpy_simulation` 缺省为 `djia_30`；
- `ifind_ashare` 缺省为 `a_share_demo_6`。

最新界面发送显式 `universe`。后端必须在创建 client 或启动子进程前验证组合；未知数据
源、未知股票池或不匹配组合返回受控校验错误，不能接受浏览器提交的任意 A 股代码。

`IFindAshareProvider` 不再只硬编码 6 股，而是验证调用股票集合必须与已解析配置完全
一致。顺序可被规范化为配置顺序；数量、代码集合或重复项不一致时必须拒绝。

## 8. 前端设计

以实现时最新 `origin/main` 的 Run Backtest 弹窗为基础，显示 `Market Data` 选择器：

- 选择 `Alpaca` 时，继续显示最新版已有的 DJIA、Magnificent 7 和 custom 资产流程；
- 选择 `iFinD A-Share` 时，显示 `A-Share Demo 6` 与
  `CSI 300 Sample 20 (2026 H2)`；
- 选择 iFinD 时隐藏或禁用自定义美股资产编辑；
- 选择 iFinD 时模型固定显示为 `Rule-based`，不能选择 LLM；
- iFinD 周期固定显示为 `60m`；
- 切换数据源或股票池时不保留不兼容的隐藏表单值。

提交新 20 股运行时，核心请求字段为：

```json
{
  "data_source": "ifind_ashare",
  "universe": "csi300_sample_20_2026h2",
  "timeframe": "60m",
  "use_llm": false
}
```

后端根据注册表决定 20 个代码。浏览器传入的 `assets` 不能覆盖 iFinD 注册股票池。

Run Config 至少显示并持久化：

```text
Market Data: iFinD A-Share
Universe: CSI 300 Sample 20 (2026 H2)
Symbols: 20
Timeframe: 60m
Decision Source: Rule-based
```

所有新增标签、选项、校验消息和错误显示使用英文。股票名称使用上表英文显示名并附代码，
避免外国用户只能看到中文简称。

## 9. 数据流与批量行为

```text
Run Backtest modal
  -> POST backtest request: ifind_ashare + csi300_sample_20_2026h2
  -> API validates registered pair and forces rule-based mode
  -> background runner passes explicit universe and 60m timeframe
  -> market profile resolves the exact ordered 20-symbol tuple
  -> IFindHttpClient sends one /api/v1/high_frequency batch request
  -> adapter validates and converts all tables to ATL OHLCV DataFrames
  -> the same normalized frames feed indicators, rule Agent and baseline
  -> run metadata, equity curve, metrics and simulated trades are persisted
```

iFinD 请求继续使用现有逗号分隔 `codes` 负载，一次请求全部 20 只。现有 429/5xx 和网络
错误重试策略保持不变，不增加逐股请求或无限重试。单批次可以保持同一时间窗口、减少
请求数量，并允许适配器对返回股票集合做完整性检查。

2026-07-24 已使用本地受忽略的 Token 对以下窗口做只读验证：

```text
start: 2026-06-23
end-exclusive: 2026-07-24
timeframe: 60m
symbols: 20
result: 20/20 symbols returned, 92 valid bars per symbol
```

该验证证明当前 iFinD 权限和端点支持一次批量获取本股票池，但实现仍必须保留所有错误
校验，不能把该次成功当作长期可用性的保证。

## 10. 完整性和错误处理

一次 20 股回测必须满足：

1. 请求股票集合与注册配置完全一致；
2. iFinD 返回恰好 20 个唯一 `thscode` 表；
3. 每只股票包含 `open/high/low/close/volume`；
4. 数组长度一致，数值有限，价格为正，成交量非负；
5. 时间戳唯一、递增、位于请求窗口和上海交易时段；
6. 每只股票至少有 50 根合法 K 线；
7. Agent 和基准所需的共同起始时间能够覆盖全部 20 只。

任一条件失败时整次运行失败。错误应指出缺失或不合格的股票及可执行原因，例如实际
K 线数量与最低数量；不得继续使用 19 只，不得切回 6 股、Alpaca 或模拟数据。

HTTP 状态、业务错误码和安全的请求上下文可以进入错误信息。Token、请求头、完整原始
响应和刷新凭证不得进入日志、API 响应、运行元数据或 Git。错误文案使用英文。

## 11. Agent 与比较基准

iFinD 两个股票池均继续强制 `use_llm=false`。本轮不修改 LLM 验证器、提示词或模型
选择逻辑。

20 股基准使用 Agent 已经取得并验证的同一批 OHLCV，不再次访问 iFinD。初始资金按
20 份等额目标分配，在共同首个时间点按现有 ATL 整数股逻辑买入，无法整除的资金保留
为现金，然后持有到结束。该基准回答“规则 Agent 是否优于简单持有同一测试股票池”，
不是按沪深 300 官方权重复制指数，也不生成 DJIA 基准。

若共同首个时间点不能覆盖全部 20 只，基准和整次运行失败，不能静默跳过股票。A 股
100 股整手、T+1、涨跌停和停牌撮合属于后续交易规则迭代，不在本轮改变现有模拟逻辑。

## 12. 上游同步策略

书面规格提交并经用户复核后，实施阶段按以下顺序开始：

1. 确认工作树干净并创建可恢复的本地备份引用；
2. `git fetch origin` 获取当时最新主线；
3. 使用普通 merge 把最新 `origin/main` 合入当前功能分支；
4. 不 rebase、不重写当前功能分支已有历史；
5. 解决冲突时优先保留上游最新 Run Backtest、资产选择、资金配置、安全修复和迁移；
6. 在最新结构中加入本设计的最小必要修改；
7. 合并和实现完成前不 push、不创建 PR。

如果上游在实施前已经提供等价的多股票池抽象，应复用其公开接口并保持本规格的行为
合同，不再平行创建重复抽象。

## 13. 测试与验收

### 13.1 单元测试

- 注册表可解析四个合法 `(data_source, universe)` 组合；
- 省略股票池时保持现有默认行为；
- 未知或错配组合在访问凭证和网络前被拒绝；
- 两个 iFinD 股票池都只能接受各自完整集合；
- 20 股顺序被规范化为配置顺序；
- 缺失、额外、重复股票或少于 50 根 K 线失败；
- 非法 OHLCV、时间戳和非上海交易时段继续失败；
- 20 股基准使用同一批数据并要求共同起始点。

### 13.2 API、CLI 和集成测试

- API 接受两个合法 iFinD 股票池并把选择传递到后台运行器；
- iFinD 强制规则模式、`60m` 和最长 31 天；
- 非法股票池、周期或日期范围返回受控错误；
- CLI 的显式 `--universe` 解析正确；
- 使用伪造官方响应离线完成 20 股 Agent 与等权基准；
- 运行配置保存来源、股票池、数量、周期和规则决策来源；
- 原 6 股 iFinD、Alpaca、vn.py 和现有美股资产选择测试继续通过。

### 13.3 真实 iFinD 验收

- 使用受 Git 忽略的本地环境变量请求真实 20 股批量行情；
- 验证 20/20 返回且每只至少 50 根合法 K 线；
- 使用临时数据库完成规则 Agent 与等权基准回测；
- 验证结果只标记为 `ifind_ashare`，且运行股票池为
  `csi300_sample_20_2026h2`；
- 检查标准输出、错误输出、数据库和 Git diff 中没有 Token。

### 13.4 浏览器验收

- 启动合并最新主线后的本地服务；
- Run Backtest 弹窗中的 `Market Data` 选择器可见；
- Alpaca 仍显示最新主线资产选项；
- iFinD 显示 `A-Share Demo 6` 和 `CSI 300 Sample 20 (2026 H2)`；
- iFinD 模型锁定为 Rule-based；
- 请求和 Run Config 正确记录 20 股配置；
- 所有新增文本为英文，长名称不溢出、不重叠；
- 桌面与移动宽度均完成截图检查。

## 14. 提交、安全和停止条件

设计文档单独提交。实现完成后创建范围明确的本地功能提交；用户未明确要求前不 push。

提交前必须确认：

- `dashboard/.env`、Token、数据库、日志和临时响应不在 Git 变更中；
- 没有冲突标记或意外生成文件；
- 专项测试、相关回归测试和真实 iFinD 验收通过；
- 浏览器界面和 Run Config 通过检查；
- 当前分支包含实施时最新的 `origin/main`。

若最新主线合并产生无法在本规格范围内安全解决的架构冲突，或者真实 iFinD 不再允许
20 股单批请求，则停止实现并重新评估设计，不把股票池拆小或降低完整性要求来伪造成功。
