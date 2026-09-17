# Admin Footprint and User Analytics Design

**Date:** 2026-08-26  
**Status:** Approved design  
**Delivery:** Three sequential pull requests

## Status

This document is retained only for Collection Scope, Event Contract, Privacy
and Security, and Retention. Every other section was superseded by
`docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md` and deleted
from this file on 2026-09-15. The deleted sections remain in git history:
`git show c3bbf2ed:docs/superpowers/specs/2026-08-26-admin-user-analytics-design.md`.

## Collection Scope

### Server-authoritative events

The backend emits events only after the source operation reaches its authoritative outcome.

Event groups include:

- **Account:** account signup and authenticated session start
- **Credential:** credential saved, verified, defaulted, reverified, or revoked
- **Agent:** agent created, updated, or deleted
- **Run:** requested, queued, started, completed, failed, or cancelled
- **Resource:** model usage recorded, Credits reserved, settled, or refunded, and safe error classification

The frontend cannot claim server-authoritative outcomes such as `backtest_completed`, `credential_verified`, or `credits_settled`.

### Frontend experience events

The authenticated frontend may submit only an allowlisted set of events, initially:

- `page_viewed`
- `page_hidden`
- `session_heartbeat`

Allowed page identifiers are stable product views such as `home`, `agents`, `agent_editor`, `credits`, and `account`. The endpoint rejects unknown event names, unknown page identifiers, unknown property keys, oversized payloads, and unauthenticated requests.

The client never submits email, display name, role, API key metadata, prompt text, strategy text, error bodies, or arbitrary form values. Server authentication supplies `user_id`.

### Sessions and page duration

The browser creates a random analytics session identifier that is unrelated to the authentication token. A session ends after 30 minutes without accepted page activity. Page duration is estimated from accepted visibility and heartbeat events; background polling and authentication refreshes do not extend meaningful activity.

## Event Contract

`analytics_events` stores a narrow, versioned envelope:

```text
event_id
schema_version
event_name
event_group
user_id
session_id
occurred_at
received_at
event_source
source_event_id
source_record_type
source_record_id
correlation_id
page_view
provider_id
model_id
billing_mode
outcome
error_category
country_code
device_category
browser_family
network_hash
properties_json
```

`source_event_id` is unique when present and provides idempotency for server events and backfill. `correlation_id` links related lifecycle events for a run or another multi-step operation. `properties_json` accepts only event-specific allowlisted keys and has a strict serialized-size limit.

Analytics stores safe error categories, not raw exceptions or upstream bodies. Approved categories include stable values such as `credential_invalid`, `credential_missing`, `provider_timeout`, `provider_unavailable`, `credits_unavailable`, `model_not_allowed`, and `internal_error`.

## Privacy and Security

### Prohibited data

The following values must never be persisted, returned, or logged by Analytics:

- Full API keys or authentication tokens
- Passwords or verification codes
- Prompt, instruction, strategy, portfolio, or form-input text
- Raw upstream provider response bodies
- Full raw IP addresses
- Raw User-Agent headers
- Encrypted credential ciphertext

Safe credential references may include provider ID, credential lifecycle outcome, and last four characters only when the existing public credential contract already permits them.

### Network pseudonymization

Analytics uses a dedicated deployment secret:

```text
ANALYTICS_PSEUDONYMIZATION_KEY
```

The request IP may exist transiently in process memory while an HMAC network identifier is calculated. The HMAC input includes the UTC calendar month so the identifier cannot link the same network indefinitely across retention periods. The raw value is immediately discarded and is never written to Analytics storage. Raw User-Agent headers are reduced to an allowlisted browser family and device category before storage.

If the key is absent or invalid, Analytics omits `network_hash`. It must never downgrade to plaintext storage. Country or region is optional and may only come from a trusted deployment-platform header; otherwise it is `Unknown`. The first version does not call an external IP geolocation provider.

### Authorization

All `/api/admin/analytics/*` endpoints use the centralized Admin dependency. Non-admin users receive `403`. Opening a user profile creates an Admin analytics access record.

The frontend ingestion endpoint requires an authenticated user, CSRF protection where required by the existing API pattern, rate limiting, and a strict event/property allowlist.

## Retention

A scheduled retention task deletes raw `analytics_events` older than 180 days in bounded batches. Daily non-identifying rollups remain available for long-term trends. Current user snapshots are recalculated rather than treated as immutable history. Admin profile-access records are retained for 365 days because they are a security audit trail, then deleted in bounded batches.

Retention failures are observable but do not block the application. Repeated failures must surface an operator-facing warning because unbounded event retention would violate the approved privacy contract.
