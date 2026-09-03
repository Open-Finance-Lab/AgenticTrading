# 策略候选池与回测配置

`POST /backtest/run`、`dashboard/scripts/backtest_hourly_agent.py` 和
`HourlyBacktester` 支持先选类别，再选择代表性 30 只、按代码排序的前 30 只或全部候选。

## Run Backtest

每次打开 Run Backtest，可在股票范围下拉框中选择：

| 页面选项 | 本次回测的候选范围 |
| --- | --- |
| Dow Jones 30 | 原有道指名单 |
| Magnificent 7 | 原有七只科技股名单 |
| Ordinary stocks · 30 stocks, 11 sectors | `ordinary`：覆盖 11 个行业的 30 只普通股 |
| Funds & ETFs · 30 across asset classes | `fund`：覆盖美股宽基、行业、海外股票、债券及实物资产的 30 只基金／ETP |
| Stocks & funds · 15 stocks + 15 funds | `all`：15 只普通股加 15 只基金／ETP，共 30 只 |

展开名单可查看分组与全部 30 个代码。新选项统一提交 `pool_mode=representative30`；
其中 `all` 表示股票与基金混合类别，仍然只选 30 只。页面不提供全量模式或大小盘选项。
日期和模型沿用现有的本次回测配置，仍支持自定义代码。

代表性名单是按覆盖范围人工维护的固定集合，不是收益排名，也不是按代码取前 30 只。
普通股分组参考 [GICS 的 11 个行业](https://www.spglobal.com/spdji/en/landing/topic/gics/)，
基金覆盖参考 [State Street 行业 ETF](https://www.ssga.com/us/en/intermediary/capabilities/equities/sector-investing/select-sector-etfs)
及 [iShares 产品分类](https://www.ishares.com/us/products/etf-investments)。

## 后端类别与模式

| 参数 `stock_pool` | 范围 |
| --- | --- |
| `ordinary` | 普通股，沿用目录的 `ordinary_share` 分类，含普通股形式的 REIT |
| `fund` | 目录标记的基金、ETF／ETP，以及 FIGI 类型明确的封闭式等基金 |
| `large_cap` | 普通股中市值 ≥ 100 亿美元 |
| `small_mid_cap` | 普通股中市值 > 0 且 < 100 亿美元，包含微盘股 |
| `all` | 普通股与基金／ETP 的并集 |

所有类别均要求目录中 `status=active`、`tradable=true`，排除 OTC。
沿用普通股目录政策，存托凭证、优先股、权证、权利、债券和无法确定类型的记录
不在普通股／基金并集中。基金范围含 ETP，应按具体产品类型理解。

| 参数 `pool_mode` | 行为 |
| --- | --- |
| `top30` | 先筛选类别，再按代码升序取前 30 只；不足 30 只时全部保留。指定类别后的默认值。 |
| `all` | 使用该类别所有符合条件的代码，不受原有 30 只上限限制。 |
| `representative30` | 使用版本化的代表性 30 只名单，仅支持 `ordinary`、`fund`、`all`。前端新增选项使用此模式。 |

排序固定，便于复现实验；`top30` 是代码顺序，并非市值或收益排名。
候选池大小不等于持仓数量。现有 LLM 策略会对整个候选池计算信号，再向模型提供
前 12 个趋势信号和当前持仓；所有选中代码仍属于可交易范围。

不传新参数时，现有道指 30 只默认值和自定义 `assets` 行为保持兼容。
`stock_pool` 与 `assets` 互斥；单独传 `pool_mode` 返回 422。
这组池用于 `alpaca` 及开启后的 `vnpy_simulation` 美股回测；iFinD 固定 A 股池保持原有接口。

## API

读取可用参数：`GET /config/stock-pools`，不需要会话。
响应的 `representative_presets` 包含前端使用的名称、说明、分组、代码和名单版本。
预览与执行由同一后端配置生成。

回测请求需携带原有 `X-Session-Id` UUID：

```json
{
  "start_date": "2026-05-01",
  "end_date": "2026-05-02",
  "data_source": "alpaca",
  "decision_source": "rule_based",
  "stock_pool": "fund",
  "pool_mode": "representative30"
}
```

改成 `"pool_mode": "top30"` 即按代码取基金类别前 30 只；`"pool_mode": "all"` 选全量。
同样支持查询参数；JSON 中非空的对应字段覆盖查询参数。
LLM 模式继续使用现有模型、凭据及计费配置。

响应中 `assets` 是实际选中代码；`universe_selection` 记录类别、模式、符合条件数量、
选中数量、名单、排序、市值分界和目录 SHA-256。代表性模式还记录 `roster_version`、
`catalog_scope`、名称和分组；其 `eligible_count` 是内置参考快照中该类别的数量，
不是全市场数量。回测结果的 metadata 中也保存这份记录。
原有 `universe` 参数仍表示行情源的注册市场配置；实际策略候选范围以
`assets` / `symbols` 和 `universe_selection` 为准。

调度时冻结名单，工作进程通过临时 JSON 文件接收，避免 Windows 命令行长度限制，
也避免排队期间目录变化导致候选池改变。Alpaca 按每批最多 100 个代码取行情。
候选资格来自参考目录；某次历史区间能否交易还取决于该代码是否有价格数据。

## 命令行和 Python

```powershell
python -m dashboard.scripts.backtest_hourly_agent `
  --start 2026-05-01 --end 2026-05-02 --no-llm `
  --stock-pool ordinary --pool-mode all
```

```python
from dashboard.backend.domain.backtesting.engine import HourlyBacktester

backtester = HourlyBacktester(
    "2026-05-01", "2026-05-02", use_llm=False,
    stock_pool="large_cap", pool_mode="top30",
)
```

## 参考数据与市值配置

代表性模式使用随仓库发布的 `dashboard/config/representative_universes.json`，
包含三份名单以及 60 个去重标的的分类参考快照，不依赖工作站的完整 CSV 或市值文件。
调整名单时需同时更新版本、分组与证券参考数据；每份名单必须恰好包含 30 个唯一代码，
且全部属于所选类别。当前版本为 `representative-2026-09-03-v1`。

以下完整目录和市值配置用于后端 `top30` 与 `all` 模式。
后端策略配置位于 `dashboard/config/strategy_universes.json`。
相对路径以 `dashboard/config` 为基准。完整目录不随仓库发布，使用相关模式的环境需设置：

```powershell
$env:US_EQUITY_CATALOG = 'C:\data\us_equities.csv'
$env:US_EQUITY_MARKET_CAPS = 'C:\data\market_caps.csv'
```

市值可直接放在目录的 `market_cap_usd` 列，或用单独 CSV 按代码补齐：

```csv
symbol,market_cap_usd
EXAMPLE,12000000000
```

上面是字段格式示例，不是真实证券数据。市值必须是以美元计的有限正数，
应保留数据来源与日期。单独 CSV 通过 `market_caps_path` 配置或
`US_EQUITY_MARKET_CAPS` 指定；环境变量优先。`large_cap_min_usd` 可修改分界。

当前下载目录没有市值列，因此普通股、基金、全部池可直接使用；大盘／中小盘池
需要补齐所有当前可交易普通股的市值。缺少目录或市值时，API 返回 503 和可操作的
原因；不会退回道指、猜市值或悄悄漏掉缺少市值的普通股。空候选结果返回 422。

当前目录是当前时点的证券快照；它不代表历史逐日成分，也不保证历史行情齐全。
用于历史回测时应区分当前分类、市值快照与历史时点数据。
