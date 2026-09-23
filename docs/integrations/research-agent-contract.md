# 研报 Agent 接入契约（v1.0-draft）

> 双方：**ATL 平台**（Agentic Trading Lab 后端，调用方）与 **研报 Agent 服务**（Lin-Feihan 的
> Shell Company Screening / Due Diligence Agent，被调用方）。
> 本文档是两边开发的唯一依据；定稿后改动需双方在群里确认并更新版本号。
>
> 状态：**draft，待双方确认** · 2026-09-18

---

## 0. 总览

- Agent 服务以 **HTTP + JSON** 暴露 4 个端点，部署后给 ATL 一个 **base URL**（如
  `https://xxx.onrender.com`）和一个 **服务间密钥**（任意随机串，双方各存一份）。
- ATL 每个请求带请求头 `X-Service-Token: <密钥>`；密钥不对返回 `401`。
- 除本契约 4 个端点外，ATL 不调用服务的任何其他接口。

调用时序：

```
ATL ── GET  /manifest            ──▶ 启动时/打开表单时拉取（可缓存）
ATL ── POST /runs   {表单值}      ──▶ 服务后台开始研究，立即返回 run_id
ATL ── GET  /runs/{id}           ──▶ 每 15 秒轮询一次状态
ATL ── GET  /runs/{id}/result    ──▶ status=completed 后取结果
```

---

## 1. `GET /manifest` — Agent 自我介绍 + 表单说明

**响应 200：**

```json
{
  "agent_id": "shell-company-screening",
  "name": "Shell Company Screening Agent",
  "description": "Deep Research agent for identifying and evaluating listed shell company candidates in M&A transactions.",
  "category": "M&A Research",
  "provider_note": "Runs on OpenAI / OpenRouter / Gemini / Perplexity Deep Research (configured server-side).",
  "estimated_runtime_seconds": 300,
  "max_runtime_seconds": 1800,
  "output_formats": ["markdown", "docx", "pdf", "evidence_json"],
  "settings_schema": {
    "fields": [
      {
        "id": "client_name",
        "label": "Client Name",
        "type": "text",
        "required": true,
        "default": null,
        "placeholder": "Name of the commissioning client",
        "description": "Name of the client or organization commissioning the screening task."
      },
      {
        "id": "target_market",
        "label": "Target Listed Market",
        "type": "text",
        "required": true,
        "default": null,
        "placeholder": "e.g. HKEX Main Board",
        "description": "Listed market or jurisdiction for candidate identification."
      },
      {
        "id": "research_cutoff_date",
        "label": "Research Cut-off Date",
        "type": "date",
        "required": false,
        "default": "today",
        "description": "Latest date up to which information is considered."
      },
      {
        "id": "control_requirement",
        "label": "Control Requirement",
        "type": "longtext",
        "required": false,
        "default": null,
        "description": "Client requirement on ownership, effective control, or control path."
      }
    ]
  }
}
```

**`type` 允许的取值**（ATL 前端按此渲染控件）：

| type | 前端控件 |
|---|---|
| `text` | 单行输入框 |
| `longtext` | 多行文本框 |
| `date` | 日期选择器，值格式 `YYYY-MM-DD` |
| `number` | 数字输入框 |
| `select` | 下拉框，字段需带 `options: ["a", "b"]` |

**约定：**

- `fields[].id` 全表唯一；用户提交的表单值以 `id` 为键。
- `required: false` 且用户未填的字段，ATL **不下发该键**（服务端用 `default_settings.md` 的默认值语义处理，如 "Not specified"）。
- Due Diligence 的 manifest 同构（字段：`target_company` 必填 text、`research_cutoff_date` 必填 date、
  `additional_context` 可选 longtext），`agent_id` 用 `due-diligence-agent`。

---

## 2. `POST /runs` — 提交一次研究任务

**请求头**：`X-Service-Token` + `Content-Type: application/json`

**请求体：**

```json
{
  "settings": {
    "client_name": "华泰资本",
    "target_market": "港股主板",
    "research_cutoff_date": "2026-09-01",
    "control_requirement": "需取得 ≥60% 控制权",
    "transaction_objective": "注入新能源资产借壳上市"
  }
}
```

**响应 202：**

```json
{
  "run_id": "run_8f3k2",
  "status": "queued",
  "created_at": "2026-09-18T08:30:00Z"
}
```

**错误：**

| 状态码 | 场景 | 响应体 |
|---|---|---|
| 401 | 密钥错误/缺失 | `{"detail": "unauthorized"}` |
| 422 | 表单校验失败 | `{"detail": "validation failed", "field_errors": {"target_market": "required"}}` |

**约定：**

- 服务收到请求后**立即返回**，研究在后台异步执行（Deep Research 分钟级，不能占着 HTTP 请求等）。
- `run_id` 全局唯一即可（如 `run_` + 随机串）。
- 必填字段缺失/类型不对 → 422，并尽量给出 `field_errors`（ATL 会在对应表单项下方标红）。
- 服务重启后已完成的 run 仍可查询：**run 状态与结果需持久化**（SQLite 即可，内存会在部署重启时丢失）。

---

## 3. `GET /runs/{run_id}` — 查询状态

**响应 200：**

```json
{
  "run_id": "run_8f3k2",
  "status": "running",
  "created_at": "2026-09-18T08:30:00Z"
}
```

**`status` 只允许 4 个值：** `queued` → `running` → `completed` | `failed`（终态不可逆）。

**约定：**

- ATL 每 15 秒轮询一次；超过 manifest 的 `max_runtime_seconds` 仍未终态，ATL 判定超时并停止轮询。
- `failed` 时本端点可附带 `"error": "人类可读的失败原因"`（可选）。
- 未知 run_id → 404 `{"detail": "run not found"}`。

---

## 4. `GET /runs/{run_id}/result` — 取结果

仅 `status=completed` 时调用（其他状态调用返回 `409 {"detail": "run not completed"}`）。

**响应 200：**

```json
{
  "run_id": "run_8f3k2",
  "status": "completed",
  "completed_at": "2026-09-18T08:37:12Z",
  "report_markdown": "# Shell Company Screening Report\n\n## 1. Executive Summary\n...",
  "artifacts": {
    "docx": {
      "filename": "shell_screening_huatai_20260901.docx",
      "content_base64": "UEsDBBQAAAAIAA..."
    },
    "pdf": {
      "filename": "shell_screening_huatai_20260901.pdf",
      "content_base64": "JVBERi0xLjcK..."
    }
  },
  "evidence": {
    "provider": "openai",
    "citations": [ { "title": "HKEX announcement", "url": "https://..." } ],
    "sources": [ "https://..." ],
    "metadata": {}
  }
}
```

**约定：**

- `report_markdown` **必返**（ATL 在线渲染的主体）；docx/pdf 生成失败时对应键可缺省
  （她的 runtime 已有此降级逻辑），但 markdown + evidence 必须有。
- 二进制文件走 `content_base64`（base64 编码传输，体积膨胀约 33%，几百 KB 的文档可接受；
  避免 ATL 再去拉带鉴权的文件 URL）。单文件建议 ≤ 5 MB。
- `evidence` 直接内联为 JSON 对象（对应她 `output_handler.py` 的 `save_evidence_json` 结构）。
- 结果可重复获取（ATL 可能把同一报告发给多个使用者），服务端无需“取后即焚”。

---

## 5. 通用错误与超时

| 状态码 | 含义 |
|---|---|
| 401 | `X-Service-Token` 缺失或不匹配 |
| 404 | 路径或 run_id 不存在 |
| 409 | 结果端点被调用但 run 未完成 |
| 422 | 表单校验失败 |
| 5xx | 服务内部错误（ATL 会在轮询中容忍偶发 5xx，连续失败才判 failed） |

- 服务不必自己做速率限制（调用方只有 ATL）。
- ATL 侧总超时 = manifest 的 `max_runtime_seconds`（默认 1800 秒），超时后 run 标记 failed 并提示用户。

---

## 6. 待确认清单（双方各答一遍）

1. base URL 与部署平台（建议 Render，与 ATL 后端同栈）
2. 服务间密钥的生成与存放方式（建议：Joy 生成随机串，双方各配环境变量）
3. 单次研究实际耗时分布（决定 `estimated_runtime_seconds` 填多少）
4. docx/pdf 生成在她部署的 Linux 环境是否可用（她 README 提到 PDF 依赖 Windows Word；
   Linux 下 markdown + evidence 保底是否可接受——按本契约 §4 可缺省）
5. 两个 Agent 是否共用一个服务（一个 base URL、manifest 里区分 agent_id），还是各部署一份
   （建议：**共用一个服务**，`GET /manifest` 变为 `GET /manifest?agent_id=...` 或返回数组，减少部署份数——定稿时二选一写死）
