# Issue #148：历史资金曲线唯一性迁移实施计划

> 按 loop engineering 执行：失败测试 -> 最小实现 -> 聚焦验证 -> 真实副本验证 -> 回归测试 -> 提交。

**目标：** 修复历史 SQLite 数据库缺少 `(run_id, timestamp)` 唯一约束的问题，让同一回测重新运行时覆盖旧曲线点，不再把新旧曲线追加到一起。

**设计规格：** `docs/superpowers/specs/2026-07-22-equity-timeseries-uniqueness-migration-design.zh-CN.md`

**工作分支：** `fix/issue-148-nemotron-reproducibility`，基于 `origin/main@671c2d4`。

**技术栈：** Python 3.10+、SQLite、pytest。

## 全局约束

- 只修改资金曲线数据库迁移和对应测试。
- 不修改 Nemotron 配置、提示词、解析、交易、指标和排行榜 API。
- 去重固定保留同组 `MAX(id)`，即最后写入的记录。
- 删除重复数据和创建唯一索引必须在同一事务中完成。
- 数据问题（残留重复、索引名被占用）导致迁移失败时必须阻止数据库继续初始化，不能只打印 warning。
- **（评审修订）** 数据库被其他进程锁住属于瞬时故障，不算数据问题：重试一次后降级为 warning 并推迟到下次启动。
  `database.py` 末尾是模块级单例 `db = BacktestDatabase()`，在这里硬失败等于让 `import` 失败、整个应用无法启动。
- 自动测试使用临时数据库，不连接网络，不产生 OpenRouter 费用。
- 真实验证只操作本地实验数据库的临时副本。
- 不提交任何 API Key。
- **（评审修订）** `dashboard/storage/data/backtest.db` 需要一并迁移后提交：只做运行时迁移会让每次本地
  `uvicorn` 都静默改写这个被 Git 跟踪的二进制文件。提交前用 `.dump` 逐行 diff 证明改动范围
  （实测：仅删除 483 行 `equity_timeseries` INSERT + 新增 1 个唯一索引）。
- 代码和测试作为一个独立提交，便于审阅和回滚。

---

## Task 1：用失败测试复现旧数据库问题

**文件：**

- 新增：`dashboard/backend/tests/test_equity_timeseries_migration.py`

### Step 1：构造旧版数据库

测试直接用 `sqlite3` 创建不带唯一约束的 `equity_timeseries` 表，并为相同的 `run_id` 和 `timestamp` 插入两条不同资金值。第二条记录的 `id` 更大，代表最后一次写入。

### Step 2：定义迁移后的预期

初始化 `BacktestDatabase` 后断言：

1. 相同键只剩一条记录；
2. 保留的是第二条记录的资金、现金和持仓价值；
3. 通过原始 SQLite 连接再次插入相同键时触发 `sqlite3.IntegrityError`，证明数据库本身已经具备唯一保护；
4. 通过 `insert_equity_point()` 写入相同键时，现有 `INSERT OR REPLACE` 能替换值，记录数仍为一；
5. 再次初始化数据库时结果不变，证明迁移幂等。

### Step 3：确认测试先失败

```bash
pytest dashboard/backend/tests/test_equity_timeseries_migration.py -q
```

预期：至少一个新增断言失败。当前旧数据库不会去重，也不会自动获得唯一约束。

---

## Task 2：实现最小唯一性迁移

**文件：**

- 修改：`dashboard/backend/database.py`
- 修改：`dashboard/backend/tests/test_equity_timeseries_migration.py`

### Step 1：增加严格迁移入口

在 `BacktestDatabase.__init__()` 的现有 schema 和普通迁移之后，调用独立的资金曲线唯一性迁移方法。该方法不放进现有会吞掉异常的宽松迁移分支。

### Step 2：检测等价唯一约束

读取 `PRAGMA index_list(equity_timeseries)`，只检查标记为 unique 的索引；再通过 `PRAGMA index_info(...)` 确认索引列按顺序恰好是：

```text
run_id, timestamp
```

新建数据库自带的 SQLite 自动索引也应被识别，因此不创建重复索引。

### Step 3：在单一事务中迁移旧表

没有等价唯一约束时：

```sql
DELETE FROM equity_timeseries
WHERE id NOT IN (
    SELECT MAX(id)
    FROM equity_timeseries
    GROUP BY run_id, timestamp
);

CREATE UNIQUE INDEX uq_equity_timeseries_run_timestamp
ON equity_timeseries(run_id, timestamp);
```

创建后再次验证唯一索引确实存在。若删除、建索引或验证失败，事务回滚并向调用方抛出异常。

### Step 4：运行聚焦测试

```bash
pytest dashboard/backend/tests/test_equity_timeseries_migration.py -q
```

预期：PASS。

### Step 5：运行邻近数据库测试

```bash
pytest dashboard/backend/tests/test_sqlite_wal.py \
  dashboard/backend/tests/test_agent_runs_metadata.py \
  dashboard/backend/tests/test_leaderboard_api.py \
  dashboard/backend/tests/test_external_backtest_api.py -q
```

预期：PASS。

---

## Task 3：验证真实 Nemotron 数据库副本

**产品代码：** 不新增。

### Step 1：复制实验数据库

把 `dashboard/storage/data/backtest.db` 复制到系统临时目录。记录原文件校验值和 Git 状态，后续所有迁移与查询只指向副本。

### Step 2：触发迁移并核对曲线

使用 `BacktestDatabase(temp_path)` 打开副本，然后通过 SQLite 查询：

```sql
SELECT COUNT(*), COUNT(DISTINCT timestamp)
FROM equity_timeseries
WHERE run_id = 'lb_nemotron_3_nano_30b_20260415_20260515';
```

预期结果：

```text
161 | 161
```

并抽查资金量级，确认保留的是最后写入的 1 万美元实验曲线，而不是旧的 10 万美元曲线。

### Step 3：确认原文件没有被验证过程修改

再次比较原文件校验值和 Git 状态。临时副本在验证结束后删除，不加入 Git。

---

## Task 4：回归测试、范围检查与提交

### Step 1：运行完整后端测试

```bash
pytest dashboard/backend/tests -q
```

如果仍出现已在未修改 `origin/main` 复现的 vn.py 随机模拟测试失败，则记录为基线问题，不顺带修改无关模块；其他新增失败必须先解决。

### Step 2：静态检查

```bash
git diff --check
git status --short --branch
git diff -- dashboard/backend/database.py \
  dashboard/backend/tests/test_equity_timeseries_migration.py
```

确认代码变化只有数据库迁移和测试，本地实验数据库仍未暂存。

### Step 3：提交代码与测试

```bash
git add dashboard/backend/database.py \
  dashboard/backend/tests/test_equity_timeseries_migration.py
git commit -m "fix(database): migrate equity curve uniqueness"
```

不得使用 `git add .`，避免把实验数据库带入提交。

### Step 4：最终检查

```bash
git status --short --branch
git log -3 --oneline
```

向用户报告测试结果、真实副本验证结果、提交 SHA，以及这一修复对 Issue #148 结论的影响。

---

## 完成定义

- 旧数据库能自动去重并建立唯一约束；
- 最新记录得到保留，重复运行只覆盖不追加；
- 迁移是幂等、原子且失败时明确中止；
- Nemotron 真实数据库副本从 644 点恢复为 161 个唯一时间点；
- 数据库相关测试和后端回归测试通过，或只剩已确认的主线基线失败；
- 提交中不包含实验数据库、密钥或无关改动。

口语化地说：先做一叠故意重复的假成绩单，证明系统确实会收重；再加整理和防重规则；最后拿真实成绩单的复印件验收，原件不动。
