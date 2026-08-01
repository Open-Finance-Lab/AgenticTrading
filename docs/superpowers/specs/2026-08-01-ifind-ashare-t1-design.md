# iFinD A 股回测 T+1 执行语义设计

## 1. 背景

ATL 已经可以使用 iFinD 的 A 股 60 分钟历史行情运行模拟回测。当前回测执行器只保存每个股票的总持仓，买入后没有区分“当天买入、暂时不可卖出”的数量，因此当天买入的 A 股可能在同一交易日再次卖出。

本设计为两个 iFinD A 股股票池补上模拟回测中的 T+1 规则：

- `A-Share Demo 6`
- `CSI 300 Sample 20`

Rule-based 和 LLM 两种决策来源都使用同一套执行语义。Alpaca 美股和 vn.py 模拟数据保持现有行为。

## 2. 目标与非目标

### 目标

1. A 股当天买入的数量当天不可卖出。
2. 按回测数据中出现的下一个不同交易日期解冻，不调用固定周末或节假日表。
3. 卖出请求超过可卖数量时允许部分成交，并明确记录未成交原因。
4. 冻结数量不进入成交数、收益、现金变化或权益曲线。
5. 保持现有成交接口、数据库交易记录和非 A 股数据源的兼容性。

### 非目标

本轮不实现以下 A 股制度：

- 100 股整手限制；
- 涨跌停撮合；
- 停牌处理；
- 印花税、佣金或其他费用；
- 真实券商下单或实盘交易；
- Alpaca 或 vn.py 的 T+1 改造。

## 3. 选择的方案

采用方案 A：在现有总持仓之上增加可卖持仓和冻结批次账本。

选择理由：

- 保留 `positions[symbol]`，现有组合估值、前端和 API 不需要改成批次持仓模型；
- 用买入日期记录冻结批次，可以正确处理多个交易日分批买入和部分卖出；
- T+1 只由 iFinD A 股市场配置打开，其他市场默认关闭；
- 执行规则集中在领域执行器，Rule-based、LLM 和外部 Agent 回测共用。

## 4. 状态模型

PortfolioManager 保留现有状态，并增加以下状态：

```python
positions: Dict[str, int]
available_positions: Dict[str, int]
frozen_lots: Dict[str, List[Dict[str, Any]]]
rejected_orders: List[Dict[str, Any]]
```

其中：

- `positions[symbol]` 是总持仓，继续供估值和现有 UI 使用；
- `available_positions[symbol]` 是当前可卖数量；
- `frozen_lots[symbol]` 是当天买入的批次，每批至少包含 `quantity` 和 `buy_date`；
- `rejected_orders` 只保存未完全成交的订单审计，不计入 `trades`。

非 T+1 市场可以继续使用原有总持仓执行路径，不创建冻结批次。

### 4.1 交易日期推进

执行器保存当前已处理的回测交易日期。遇到新的日期时，释放该股票所有 `buy_date < current_date` 的冻结批次：

```text
买入日 2026-04-01：批次保持冻结
下一个回测数据日期 2026-04-02：批次转入 available_positions
```

解冻只发生在回测数据实际出现的日期。数据没有出现的周末或节假日不会触发步骤，因此自然会跳到下一个有效交易日。

## 5. 执行流程

### 5.1 BUY

在现金足够且数量为正时：

1. 扣除买入成本；
2. 增加 `positions[symbol]`；
3. 追加一个 `frozen_lots[symbol]` 批次；
4. 当天不增加 `available_positions[symbol]`；
5. 向 `trades` 写入真实 BUY 成交。

同一根 K 线内后续的 SELL 不能使用刚刚买入的批次。

### 5.2 SELL

1. 读取当前 `available_positions[symbol]`；
2. 实际成交数量为 `min(requested_shares, available_shares)`；
3. 只从可卖数量和总持仓中扣除实际成交数量；
4. 实际成交数量大于 0 时，向 `trades` 写入 SELL 成交并增加现金；
5. 未成交部分不改变现金和收益，并写入 `rejected_orders`。

当总持仓为 100 股、其中 40 股可卖、60 股冻结，Agent 请求卖 100 股时：

```text
实际成交：40 股
未成交：60 股，reason = t1_frozen
```

如果请求量同时超过总持仓和冻结数量，未成交部分按原因拆分为 `t1_frozen` 和 `insufficient_position`，避免把普通持仓不足误报为 T+1。

## 6. 审计和 API 合同

### 6.1 成交记录

现有 `trades` 只表示真实成交：

- 数量始终大于 0；
- 继续写入现有 trades 表；
- 继续用于交易数、收益、权益曲线和现有交易日志。

### 6.2 未成交审计

新增 `rejected_orders` 结果字段，并在运行元数据中持久化完整列表。记录至少包含：

```json
{
  "timestamp": "2026-04-01T10:00:00",
  "symbol": "600519.SH",
  "action": "sell",
  "requested_shares": 100,
  "executed_shares": 40,
  "unfilled_shares": 60,
  "status": "partial",
  "reason": "t1_frozen"
}
```

完全不能卖时，`status` 为 `rejected`，`executed_shares` 为 0，但该对象不是 `trades` 中的 0 股成交。

`rejected_orders` 作为可选字段加入回测结果和 live progress：

- 旧客户端忽略该字段即可继续工作；
- 现有 `/runs/{run_id}/trades` 仍然只返回成交；
- 回测结果读取接口同时返回 `rejected_orders`，便于诊断 T+1 行为；
- 现有 Alpaca/vn.py 结果返回空列表。

所有面向用户的新增文案使用英文；不增加凭证输入或打印任何凭证。

## 7. 测试方案

### 7.1 执行器单元测试

覆盖：

1. 同日买入后卖出，卖出被 `t1_frozen` 拒绝；
2. 下一有效交易日卖出成功；
3. 周末和缺失节假日日期被跳过；
4. 可卖 40 股、请求 100 股时部分成交 40 股；
5. 全部数量冻结时没有成交记录和现金变化；
6. 多个交易日分批买入时，旧批次先解冻、新批次继续冻结；
7. 同一决策内先买后卖时，刚买数量不可卖。

### 7.2 回测集成测试

使用确定性模拟 OHLCV 数据验证：

- 两个 iFinD A 股股票池都打开 T+1；
- Rule-based 和 LLM/外部 Agent 两条路径都传递同一 T+1 配置；
- 回测结果包含 `rejected_orders`；
- 成交数、收益、权益曲线排除冻结部分。

### 7.3 回归测试

运行现有 backtesting、execution、iFinD、vn.py 和 Alpaca 测试，确认：

- 非 A 股数据源的成交结果不变；
- 旧 `trades` API 字段和语义不变；
- 旧回测结果没有 `rejected_orders` 时按空列表兼容。

## 8. 验收标准

功能完成必须同时满足：

1. iFinD A 股当天买入的股票当天无法卖出；
2. 下一个回测数据交易日可以卖出；
3. 超出可卖数量时能部分成交；
4. `t1_frozen` 原因可从回测结果读取；
5. 冻结部分不进入成交数、收益和现金；
6. 两个 A 股股票池、Rule-based 和 LLM 路径均通过测试；
7. Alpaca 和 vn.py 回归测试通过；
8. 代码、测试和日志不包含任何 API 凭证。

