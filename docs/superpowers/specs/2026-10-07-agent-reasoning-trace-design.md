# Agent Reasoning Trace Design

**Date:** 2026-10-07  
**Status:** Approved design  
**Scope:** First production-oriented slice of reasoning trace for ATL

## Problem

ATL already persists the important business objects in an Agent-Environment run:
AgentVersion, Run, Step, Decision, execution results, and artifacts. Decisions
may include a rationale, but there is no single object that connects the complete
story of one run for an administrator.

The first trace slice must let an administrator answer:

> What happened during this Agent run, in what order, and what decision and
> execution result did each step produce?

The design must later support tool calls, data retrieval, child Agents, and
incremental Admin updates without replacing the initial data model.

## Goals

- Create one stable `trace_id` for one Agent execution.
- Append ordered events while a run is executing.
- Reuse existing Run, Step, Decision, ExecutionResult, and Artifact records by
  reference instead of copying their business payloads.
- Provide an Admin-readable event timeline.
- Keep the first implementation deterministic and testable without an LLM.
- Store bounded decision rationale summaries, not hidden model chain-of-thought.
- Keep the implementation in the existing Agent Postgres/Neon data boundary.

## Non-goals for the first slice

- Capturing every internal model token or hidden chain-of-thought.
- Replacing the existing Run or Decision APIs.
- Introducing OpenTelemetry, a collector, or an external observability product.
- WebSocket delivery before the event query contract is stable.
- Moving binary artifacts out of the existing artifact storage path.

## Architecture

The trace layer is an append-only observability layer around the existing run
model:

```text
AgentVersion
  -> Run
    -> Step
      -> Observation
      -> Decision
      -> ExecutionResult

Trace
  -> TraceEvent (references Run/Step/Decision/Artifact)
```

`agent_traces` is the run-level envelope. `agent_trace_events` is the ordered
timeline. A trace event may reference an existing business record through an
ID, while its JSON payload contains only the bounded fields needed for display
and audit.

The first implementation uses the existing Agent Postgres/Neon database and
connection-pool boundary. The trace tables are a separate logical data domain,
but remain physically co-located so Admin queries do not require cross-database
joins. A later split is allowed only after measured volume or retention needs
justify it.

### Admin navigation

The operator path is layered rather than a second trace surface:

```text
User Analytics -> User profile -> Agents & traces -> Run -> Reasoning trace
```

The profile's `Agents & traces` section reads the admin-only
`GET /api/admin/users/{user_id}/agent-activity` projection. It joins Agent,
Run, and Trace records by stable IDs in the application layer because those
records may live in different persistence boundaries. A run without a trace
remains visible and is labeled as unavailable. Runs with a running trace link
to the polling timeline, while completed and failed runs link to the same
timeline in read-only mode. The global `Agent traces` Admin entry remains the
cross-user operational view.

## Data model

### `agent_traces`

| Column | Meaning |
|---|---|
| `trace_id` | Stable trace identifier |
| `agent_id` | Logical Agent reference |
| `agent_version_id` | Immutable version used for the run |
| `run_id` | Existing Run reference |
| `user_id` | Owner/user reference, when available |
| `trace_kind` | Initial value `trading_run` |
| `status` | `running`, `completed`, or `failed` |
| `initial_input_json` | Bounded, redacted run input |
| `final_output_summary` | Bounded final result summary |
| `started_at` | Run start time |
| `ended_at` | Terminal time, nullable while running |
| `created_at` | Trace creation time |

Constraints:

- `trace_id` is the primary key.
- `run_id` is unique for the first slice.
- `status` follows the existing Run terminal state; a trace cannot claim
  `completed` when the referenced Run is failed.

### `agent_trace_events`

| Column | Meaning |
|---|---|
| `event_id` | Stable event identifier |
| `trace_id` | Parent trace |
| `sequence_no` | Strict per-trace order |
| `event_type` | Versioned event kind |
| `actor_type` | `system`, `agent`, `tool`, or `user` |
| `actor_id` | Optional actor/provider identifier |
| `step_id` | Existing Step reference, nullable |
| `decision_id` | Existing Decision reference, nullable |
| `artifact_id` | Existing Artifact reference, nullable |
| `payload_json` | Bounded display/audit payload |
| `occurred_at` | Time the event happened |
| `ingested_at` | Time ATL persisted it |
| `idempotency_key` | Optional caller/event retry key |
| `schema_version` | Event payload version, starting at `1` |

Constraints and indexes:

- Primary key: `event_id`.
- Unique key: `(trace_id, sequence_no)`.
- Unique nullable retry key: `(trace_id, idempotency_key)`.
- Query index: `(trace_id, sequence_no)`.
- Admin list index: `(agent_id, created_at)` on `agent_traces`.
- Payloads are bounded before persistence; raw provider secrets are never
  accepted into `payload_json`.

## Initial event contract

The first slice supports exactly these event types:

### `run_started`

```json
{
  "environment_id": "us-equity-hourly-v1",
  "config_summary": {"symbols": ["AAPL"], "mode": "safe_trading"}
}
```

### `decision_recorded`

```json
{
  "action": "BUY",
  "symbol": "AAPL",
  "quantity": 10,
  "confidence": 0.72,
  "reasoning_summary": "Positive signal under the configured rule set."
}
```

### `execution_result`

```json
{
  "accepted": true,
  "fills": [],
  "rejections": []
}
```

### `artifact_created`

```json
{
  "kind": "decision_log",
  "content_type": "application/json"
}
```

### `run_completed`

```json
{
  "result_summary": {"total_return": 0.03, "decision_count": 12}
}
```

Failures use the same event envelope with `event_type: "run_failed"` in the
next implementation slice if the existing failure paths require it; the trace
status is authoritative for the first MVP. The event registry must reserve the
name now so later failure instrumentation does not invent a second convention.

## Write and read flows

### Write flow

1. Run creation creates or associates one trace.
2. The run lifecycle appends `run_started`.
3. Each completed decision appends `decision_recorded` and then
   `execution_result`.
4. Artifact persistence appends `artifact_created`.
5. Run finalization updates the envelope and appends `run_completed`.

Every append obtains the next `sequence_no` under a per-trace transaction or
equivalent database constraint. A retried append with the same idempotency key
returns the existing event rather than creating a duplicate.

### Read flow

The first Admin API is read-only:

```text
GET /api/v1/admin/traces
GET /api/v1/admin/traces/{trace_id}
GET /api/v1/admin/traces/{trace_id}/events?after_sequence=0&limit=100
```

The event endpoint is incremental by design. The first Admin page may poll it;
SSE or WebSocket delivery is a later transport choice over the same contract.

## Admin experience

The first Admin detail view shows:

- Agent, AgentVersion, Run, status, start time, end time, and duration;
- an ordered timeline of trace events;
- expandable decision rationale and execution outcomes;
- linked existing artifacts;
- explicit running, failed, empty, and permission-denied states.

The page must distinguish a missing event from an event with an empty payload.
It must never render fabricated zero values for data that has not arrived.

## Privacy and security

- Trace reads require the existing Admin authorization boundary.
- User-facing Agent owners do not automatically receive Admin traces.
- Input and payload fields are redacted and size-limited before storage.
- API keys, cookies, authorization headers, and provider secrets are forbidden
  in event payloads.
- The system stores decision rationale summaries and provider evidence, not an
  implied promise to expose hidden chain-of-thought.

## Loop plan

Each loop follows: failing test -> smallest implementation -> focused
verification -> regression check -> short review note.

| Loop | Deliverable | Exit criterion |
|---|---|---|
| L0 | Current-path audit and frozen design | Existing write points and DB boundary documented |
| L1 | Tables, repository, and append/read primitives | Ordered events round-trip with duplicate protection |
| L2 | Rule-based/SDK run instrumentation | One run creates the five-event MVP trace |
| L3 | Admin trace list/detail/events API | Authorized Admin queries return stable JSON |
| L4 | Admin timeline page | A completed run is inspectable without database access |
| L5 | Tool/data retrieval events | Evidence path for a decision is inspectable |
| L6 | Incremental polling | Running traces update without full-page reload |
| L7 | Parent/child traces and production hardening | Multi-agent relationships, permissions, retention, and indexes are verified |

## Verification strategy

- Repository tests cover schema migration, ordering, idempotency, and bounded
  payloads.
- A deterministic rule-based run is the first end-to-end fixture; no external
  LLM credential is needed.
- API tests cover ownership/admin access, pagination, empty states, and terminal
  status consistency.
- Browser verification is reserved for L4 and L6, where rendered timeline and
  live incremental updates are part of the contract.

## Open decisions deferred by design

- Whether large artifacts move to object storage.
- Exact retention duration and archival schedule.
- SSE versus WebSocket after polling proves the event contract.
- Additional event types for LLM provider requests and child Agents.
