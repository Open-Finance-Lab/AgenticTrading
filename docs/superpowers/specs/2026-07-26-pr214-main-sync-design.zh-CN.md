# PR #214 与最新 main 同步设计

## 背景

PR #214 在 `feat/ifind-ashare-integration` 分支上完成了同花顺 iFinD A 股行情、A 股 Agent 回测、历史汇率和双币种记账。PR 的 CI 已通过，但 `main` 在此后新增了 19 个分支独有提交，GitHub 当前将 PR 标记为冲突且不可合并。

Git 的无工作区合并预检确认，大部分文件可以自动合并，只有以下两个文件存在内容冲突：

- `dashboard/backend/baseline_generator.py`
- `dashboard/frontend/app.html`

## 目标

1. 使用普通 merge 将最新 `origin/main` 合入 PR 分支，保留现有提交历史。
2. 保留 PR #214 的完整 iFinD A 股功能和双币种回测合同。
3. 同时保留 `main` 最新的 CodeQL 清理、首页调整和 Agent UI 更新。
4. 重新运行测试并推送，使 GitHub 恢复可合并状态。

## 非目标

- 不增加新的 A 股交易规则或真实交易功能。
- 不改变 iFinD、Alpaca 或 vn.py 的现有行为。
- 不重写 PR 分支历史，不进行 rebase 或强制推送。
- 不提交本地数据库、测试截图、API Token 或其他凭证。

## 合并策略

执行 `git merge origin/main`，让 Git 自动合并无冲突文件。对两个冲突文件采用语义合并，不直接选择整份 ours 或 theirs。

### 后端基准生成器

`dashboard/backend/baseline_generator.py` 以 PR 分支的 iFinD 多币种基准能力为功能主体，同时接受 `main` 的 CodeQL 清理：

- 保留由已有 bars 生成基准的离线路径；
- 保留 iFinD 原生 CNY 账本和 USD 报告曲线所需接口；
- 保留 Agent 与 Buy-and-Hold 共享汇率序列的行为；
- 删除上游已经确认未使用的导入和死代码；
- 不恢复运行时自动安装未使用依赖的逻辑。

### 前端页面

`dashboard/frontend/app.html` 同时保留两侧的用户可见合同：

- 保留 iFinD 市场数据选择器、两个 A 股股票池、60 分钟标识和英文公司名；
- 保留 Rule-based/LLM 选择、CNY trading capital、历史汇率及双币种成交展示所需节点；
- 接受 `main` 最新首页结构、虚构结果文案清理和 Agent UI 调整；
- 使用 `main` 的最新静态资源版本号，避免浏览器继续加载旧缓存；
- 不重新加入被 `main` 明确删除的重复首页或虚构绩效信息。

## 数据与兼容性

本次同步不修改现有数据合同。iFinD 回测继续在 CNY 账本内决策和成交，并将权益按历史 USD/CNY 汇率报告为 USD；Alpaca 和 vn.py 继续使用 USD。旧数据库记录中的新增审计字段仍可为空。

本地 `dashboard/storage/data/backtest.db` 和 `artifacts/` 是测试产物，必须保持未暂存并排除在提交之外。

## 错误处理

- 若合并后出现超出预检范围的新冲突，停止提交并重新评估，不批量接受任一侧版本。
- 若测试失败，先判断是同步回归还是既有外部网络失败；只有与本次合并无关且有证据的既有失败才能单独记录。
- 若推送后 GitHub 仍显示冲突，重新获取远端 `main` 并检查是否在处理期间出现了新提交。
- 任何日志、测试输出和提交都不得包含 iFinD、Alpaca 或 LLM 凭证。

## 验证标准

1. 对两个冲突文件运行针对性的后端、前端合同和 iFinD 测试。
2. 运行完整后端测试与现有前端测试入口。
3. 检查差异中不存在冲突标记、凭证、数据库或截图。
4. 使用真实 iFinD 本地配置完成一次 A 股 Rule-based 回测，确认曲线和成交仍可生成。
5. 推送 merge 提交后，确认 PR #214 为 `OPEN`、`MERGEABLE`，并等待新的 GitHub CI 全部通过。

## 完成条件

只有以下条件同时满足，本次同步才算完成：

- 两个冲突按上述语义解决；
- iFinD A 股功能和 `main` 最新行为同时保留；
- 本地验证通过；
- 分支已推送；
- PR 不再显示合并冲突。
