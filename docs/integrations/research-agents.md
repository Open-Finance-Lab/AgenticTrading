# Research Agents — 研报 Agent 模块（N2/PR2）

> **Language / 语言:** 中文为主，接口字段为英文。对应设计：2026-09-15 admin layer
> redesign 的 D5/N2（本模块）与 09-17 nav consolidation (#488) / universal top nav
> (#490) 的既有实现。

## 这是什么

ATL 原有的外部 Agent 只有**交易决策型**一种：外部程序按小时循环接收行情快照、
提交买卖指令（见 `docs/source/lab/external_agents.rst`）。研报模块新增了
**报告输出型 Agent** 分类：外部 Deep Research 服务接收一份表单委托
（mandate），异步产出结构化研究报告，ATL 负责发现、安装、提交、展示与下载。

首批接入的两个 Agent（Lin-Feihan）：

| Agent | 输入 | 输出 |
|---|---|---|
| Shell Company Screening Agent | 客户委托表单（12 字段） | 壳公司筛选报告（md/docx/pdf）+ 证据 JSON |
| Due Diligence Agent | 目标公司 + 截止日 + 补充背景 | 公开信息尽调报告（md/pdf）+ 证据 JSON |

## 架构

```
Community "Research Agents" 货架
   │ Add to My Agents（仅记录 user→template 链接，不占资金、不进排行榜）
   ▼
My Agents "Research Agents" 货架 → Open workbench
   │ 动态表单（字段由 Agent 服务的 manifest 实时定义）
   ▼
ATL POST /api/v1/research/agents/{id}/runs ──HTTP──▶ Agent 服务 (FastAPI)
   │ ◀── 轮询状态 / 拉取结果 ──────────────────────┘
   ▼
报告入库 → 在线阅读 + 下载（md/docx/pdf/evidence.json）→ 完成邮件（可选）
```

原则与交易型接入一致：**ATL 不托管模型、不代付 LLM 费用**；真实研究发生在
Agent 作者自己的服务与 API key 上。

## 服务间契约（4 端点）

完整契约见 `docs/integrations/research-agent-contract.md`（v1.0-draft，与
Lin-Feihan 侧共享同一份），摘要：

| 端点 | 作用 |
|---|---|
| `GET /manifest?agent_id=…` | Agent 卡片 + 表单 schema（字段 id/类型/必填/默认值） |
| `POST /runs` | 提交 `{agent_id, settings}` → `{run_id}`，后台异步执行 |
| `GET /runs/{id}` | 状态：`queued → running → completed \| failed` |
| `GET /runs/{id}/result` | `report_markdown`（必返）+ `artifacts`（base64 docx/pdf，可缺省）+ `evidence` |

- 鉴权：请求头 `X-Service-Token`（`RESEARCH_SERVICE_TOKEN`，双方同值）。
- 服务必须持久化 run（部署重启后仍可查询）。
- markdown 与 evidence 必返；docx/pdf 生成失败可缺省（fail-visible，不是静默空报告）。

## ATL 侧实现

| 件 | 位置 |
|---|---|
| 模板目录 | `dashboard/config/marketplace.json` 的 `shelf: "research"` 条目（含服务地址 env 名与缺省 URL） |
| 货架/卡片 | Community `Research Agents` 柜台（无收益曲线，卡片为服务事实）+ My Agents 同名货架 |
| 存取 | `dashboard/backend/domain/agents/research_store.py`（adds / runs / artifacts 三张 sqlite 表，产物 base64 入库——Render 文件系统易失） |
| 路由 | `dashboard/backend/api/routers/research.py`（manifest 代理带短缓存、run 提交/轮询/完成入库、报告与下载、完成邮件钩子） |
| 工作台 | `researchWorkbenchView`（app.html）+ `app.js` 研报模块（动态表单、10 秒轮询、安全 Markdown 渲染、下载按钮组） |
| 邮件 | 完成 discovered-by-poll 时经 Brevo `send_email()` 发报告页链接（纯文本 v1；附件待 v1.5） |

**完成发现的语义（v1 限制）**：run 完成的副作用（入库、邮件）发生在
"某次状态轮询首次发现 completed" 时。前端每 10-15 秒轮询一次；用户关页后
完成事件推迟到下一次任何人拉取该 run 状态时补发。后台 sweeper 是后续增强。

## 权限与计费

- manifest/runs/报告/下载全部要求登录（`get_current_user`）；报告**仅提交人可见**。
- 研报 Agent 不参与资金分配、不进排行榜、不产生回测。
- 计费暂未接入：每次运行消耗 Agent 作者侧的 Deep Research 配额；周期订阅
  （v2，未实现）上线前需先定计费与限额。

## 尚未完成（按序）

1. **真服务接入**：把模板的 `service_base_url_default` 换成已部署的
   Lin-Feihan 服务 URL（或配 `RESEARCH_*_SERVICE_URL` 环境变量），替换开发用 stub。
2. **邮件本地验证**：本地无 Brevo 密钥，发信路径已实现、待 Render 环境实测。
3. **周期订阅（链路 2 v2）**：`research_subscriptions` 设计已备（见设计讨论），
   等订阅语义（完成通知 vs 周期重跑）与计费确认后实现。
4. 旧 `/app?view=admin` 控制台代码保留一个发布周期后删除（现已被重定向取代）。
