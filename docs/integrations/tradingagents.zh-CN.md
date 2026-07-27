# TradingAgents 接入 ATL：本地决策、统一回测

> **Language / 语言:** This guide is written in Simplified Chinese. English
> readers: see the "TradingAgents integration" section of
> `docs/source/lab/external_agents.rst` for an overview, and
> `dashboard/examples/tradingagents_atl_backtest.py` for the runnable command —
> its `--help` output and all console messages are in English.

这项集成让 [TradingAgents](https://github.com/TauricResearch/TradingAgents) 用户把本地的
多 Agent 研究结论放到 Agentic Trading Lab（ATL）中模拟成交，得到交易记录、指标、
收益曲线、Agent Card 和排行榜结果。

一句话类比：TradingAgents 是用户自己的“研究委员会”，ATL 是统一的“交易赛道和
计分器”。ATL 不托管模型，也不替用户支付 LLM 费用。

## 第一版能做什么

```text
本地 TradingAgents
  -> 为一只股票和几个指定日期生成五档评级
  -> 保存为可检查的决策 artifact
  -> ATL 在 T+1 模拟买卖
  -> 生成收益曲线和排行榜记录
```

第一版只支持 ATL `us-equity-hourly-v1` 环境中的一只美股，只做多。TradingAgents 使用
自己的数据源做判断，ATL 使用 Alpaca 小时行情决定模拟成交价和计算收益。

## 1. 安装

建议创建独立 Python 3.12 虚拟环境，让 TradingAgents 的数据与 LLM 依赖不影响其他
项目：

```bash
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
python -m venv .venv
source .venv/bin/activate
pip install .
```

再安装 ATL 的轻量 SDK。下面的路径替换为本机 AgenticTrading 仓库路径：

```bash
pip install -e /path/to/AgenticTrading/packaging/agentictrading
```

当前集成针对 TradingAgents v0.3.1 和 0.3.x 兼容版本测试。它不是 ATL 后端的核心依赖；
已经生成 artifact 后，即使当前环境没有安装 TradingAgents，也可以继续重放。

## 2. 配置 TradingAgents

按所选模型和数据供应商设置本地环境变量。例如：

```bash
export OPENAI_API_KEY="..."
# 或 GOOGLE_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY 等
export ALPHA_VANTAGE_API_KEY="..."  # 仅在对应数据配置需要时设置
```

这些 Key 由 TradingAgents 在本地读取。集成不会把它们写入 artifact、ATL run config 或
日志，也不会发送给 ATL。

## 3. 准备 ATL AgentVersion

在 ATL Dashboard 的 **My Agents** 中创建或连接一个 Agent，保存页面只显示一次的
`ag_...` API Key，并记录 `agent_id`。然后为当前 TradingAgents 配置创建一个不可变的
AgentVersion：

```bash
export ATL_API_KEY="ag_xxxxxxxx"
export ATL_BASE_URL="https://agentictrading.onrender.com"
export ATL_AGENT_ID="agt_xxxxxxxx"

python - <<'PY'
import os
from agentictrading import ATLClient

client = ATLClient(
    base_url=os.environ["ATL_BASE_URL"],
    api_key=os.environ["ATL_API_KEY"],
)
version = client.create_agent_version(
    os.environ["ATL_AGENT_ID"],
    version="0.1.0",
    architecture="multi_agent_debate",
    model_backbones=["your-deep-model", "your-quick-model"],
    decision_frequency="user_specified",
    metadata={"config": {"integration": "tradingagents"}},
)
print(version.id)
PY
```

保存输出的版本 ID：

```bash
export ATL_AGENT_VERSION_ID="agv_xxxxxxxx"
```

模型、分析员或关键配置变化后应创建新的 AgentVersion，避免不同配置共用同一个排行榜
身份。

## 4. 运行一次完整测试

从 ATL 仓库根目录运行：

```bash
python dashboard/examples/tradingagents_atl_backtest.py \
  --symbol AAPL \
  --analysis-date 2026-04-03 \
  --analysis-date 2026-04-10 \
  --analysis-date 2026-04-17 \
  --start-date 2026-04-06 \
  --end-date 2026-04-24
```

这里不是每天调用 TradingAgents。只有三个显式 `--analysis-date` 会运行完整多 Agent
分析。建议第一轮先用一个日期，确认模型和数据 Key 可用，再增加日期。

可以覆盖模型配置：

```bash
python dashboard/examples/tradingagents_atl_backtest.py \
  --symbol AAPL \
  --analysis-date 2026-04-03 \
  --start-date 2026-04-06 \
  --end-date 2026-04-10 \
  --llm-provider openai \
  --deep-think-llm your-deep-model \
  --quick-think-llm your-quick-model
```

生成的 JSON 默认保存在：

```text
~/.agentictrading/tradingagents/decisions/
```

也可以使用 `--output /safe/path/aapl.json` 指定位置。

## 5. 不再调用 LLM，直接重放

先检查 artifact 中的评级、原始最终结论、错误和安全 manifest。确认后使用：

```bash
python dashboard/examples/tradingagents_atl_backtest.py \
  --symbol AAPL \
  --decisions-file ~/.agentictrading/tradingagents/decisions/aapl-xxx.json \
  --start-date 2026-04-06 \
  --end-date 2026-04-24
```

`--decisions-file` 路径不会构造 TradingAgents，也不会读取任何 LLM Key。它只校验 JSON、
计算 SHA-256 并把已有决策交给 ATL。口语化理解：第一次是专家委员会开会并写好答题卡，
以后只是拿同一张答题卡重复阅卷，不再付开会费用。

## 6. 评级怎样成交

| TradingAgents v0.3.1 | ATL 动作 |
|---|---|
| Buy / Overweight | 买入或补足目标仓位 |
| Hold | 保持现有仓位 |
| Underweight / Sell | 卖出全部现有持仓 |

当前 ATL 环境限制单只股票最多占总资产 25%。BUY 会计算达到 25% 目标所需的整数股数，
不会每次额外增加 25%；SELL 只卖掉已有股票，没有持仓时不会做空。

### 单股价格上限：约 250 美元

`us-equity-hourly-v1` 固定以 **1000 美元** 起始资金运行（SDK 会拒绝任何其他初始
资金），单个持仓上限 25%，因此单个持仓预算是 **250 美元**。只要一股价格高于
约 250 美元，`floor(250 / 价格)` 就等于 0，这条 BUY 在数学上无法成交，与当时行情
无关。

DJIA-30 中约有三分之一的成分股长期处于这个区间（例如 MSFT、GS、UNH、HD、MCD、
CAT、V），而 AAPL、KO 等价格较低的成分股可以正常买入。选股票时请先确认这一点：
否则一次“成功”的回测会以 0% 收益结束，而原因并不是模型判断。

这类 BUY 会单独记为 `price_too_high`，不会混进普通的 constraint Hold，rationale 里
写明当时的价格和预算，例如：

```text
price_too_high_for_target price=430.00 max_position_budget=250.00
```

如果一次回测里所有 BUY 都因此被丢弃，命令会在 stderr 打印 WARNING，明确说明这次
0% 收益反映的是资金上限而不是 TradingAgents 的结论。

TradingAgents 可能在一个分析日期使用当天完整数据，因此 ATL 只在该日期之后的第一个
实际小时 Step 执行。周五分析通常在下周第一个实际交易日执行，不会回到周五交易。

## 7. 怎样看错误

命令结束时会分别显示：

- 模型主动 Hold；
- 单个分析日期两次失败后的 error Hold；
- 两次分析之间的 passive Hold；
- 已到目标仓位、无仓可卖或缺少价格导致的 constraint Hold；
- 单股价格超过持仓预算导致的 `price_too_high` Hold（见第 6 节）；
- ATL 拒单、成交和服务器 timeout Hold；
- `autoheld_steps`：ATL 在收到本地决策之前就关闭 Step 的次数。

单个日期失败会重试一次。只要还有其他有效评级，回测可以继续；如果所有日期都失败，
命令不会创建一条看似正常的全空仓曲线。ATL 网络或鉴权错误也不会被静默改成 Hold。

连续 3 个分析日期全部失败时，命令会立即停止并报错，不再为后面的日期继续付费：
连续失败几乎都是凭证、模型或数据供应商的问题，而不是某个日期本身的问题。

`autoheld_steps` 对应的分析日期**不会**被消费：ATL 没有收到决策，就说明什么都
没有成交。这些记录会在下一个 Step 重试；如果回测在此之前结束，它们会出现在
“未执行的分析日期”里，命令以退出码 2 结束。因此本地计数器不会出现“报告了一笔
从未成交的订单”的情况，可以直接和 ATL 返回的 `timeout_holds` 对照。

## 8. 在 ATL 查看结果

命令会输出 `run_id` 和结果链接。也可以在 Dashboard 的 **My Agents** 中找到对应 Agent，
打开 **View All Runs** 查看：

- 资产净值和收益曲线；
- 模拟成交记录；
- 每小时决策与 rationale；
- 总收益、最大回撤等指标；
- 与基准和其他 Agent 的排行榜比较。

同一 artifact 的回放动作是确定的；但 TradingAgents 本身由 LLM 和动态数据驱动，重新
生成同一日期的 artifact 仍可能得到不同结论。两套数据供应商也可能存在复权或时间戳
差异，因此这项接入首先用于研究、调试和统一计分，不代表实盘能力或盈利保证。

## 9. 常见问题

**提示 TradingAgents 未安装**

确认当前命令使用的是安装过 TradingAgents v0.3.x 的虚拟环境。仅重放已有 artifact
不需要安装。

**提示股票不在 universe**

第一版只能使用 ATL 当前 `us-equity-hourly-v1` 支持的股票。先用 AAPL 验证链路。

**买入信号没有成交**

检查命令摘要和 rationale。最常见的原因是**股价高于约 250 美元的单股上限**（见第 6
节的完整说明，摘要里记作 `price_too_high`）；其他原因包括已经达到目标仓位、ATL
拒单，或回测日期没有覆盖 T+1 Step。

**提示缺少时区数据库（ZoneInfoNotFoundError）**

回放需要把 ATL 的 Step 时间戳换算成美东日期，这依赖系统的 IANA 时区数据库。
Windows 不自带，精简版容器镜像也常常删掉它。在 Windows 上安装 `agentictrading`
会自动带上 `tzdata` 依赖；其他环境下手动执行 `pip install tzdata` 即可。

**出现 timeout_holds**

离线 artifact 回放应该很快。非零 timeout Hold 通常表示服务拥塞或网络问题，应把这次
结果视为执行链路异常，而不是 TradingAgents 主动选择不交易。
