# iFinD Access Token Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let the ATL backend use one Render-managed iFinD refresh token to obtain short-lived access tokens for all A-share backtests, while preserving the existing static access-token fallback and keeping all credentials server-side.

**Architecture:** Prefer IFIND_REFRESH_TOKEN over IFIND_ACCESS_TOKEN. Exchange the refresh token lazily on the first iFinD request, cache the access token in module-level process memory for six days (a client is built per backtester, so an instance-scoped cache would exchange once per backtest), and serialize refreshes with a process-local lock. On one 401/403 response, replace the token that was rejected -- compare-and-swap under a single lock acquisition, so a rotation seen by N threads costs one exchange -- refresh once, and retry. Do not persist tokens, expose them to the frontend, log them, or start a background timer.

**Tech Stack:** Python backend, requests-compatible sessions, pytest, existing ATL market-data and backtest routers, Render environment variables.

**Specification:** docs/superpowers/specs/2026-08-19-ifind-token-refresh-design.md

## Global Constraints

- Use the official endpoint POST /api/v1/get_access_token on the configured iFinD base URL.
- Send the refresh credential only in the refresh_token request header.
- Treat the access token as valid for six days locally; iFinD documents a seven-day lifetime.
- Refresh credentials take precedence over a static access token when both are configured.
- Keep IFIND_ACCESS_TOKEN as a compatibility fallback.
- Never print either credential or include either credential in command arguments, response metadata, frontend bundles, database rows, or progress logs.
- Preserve existing Alpaca, vn.py, and synthetic-data behavior.
- Use deterministic fake sessions and clocks in tests; do not call iFinD from the test suite.
- Keep all new source, tests, docs, commit messages, and UI strings in English.
- ATL remains a backtest/paper-simulation platform and must not place live orders.

## Task 1: Add refresh-token configuration and redaction

**Files:**
- Modify .env.example:81-87.
- Modify dashboard/backend/infrastructure/market_data/provider.py:80-92.
- Modify dashboard/backend/api/routers/backtests.py:1165-1188.
- Extend dashboard/backend/tests/infrastructure/market_data/test_ifind_ashare.py.
- Extend dashboard/backend/tests/test_market_data_features.py.
- Extend dashboard/backend/tests/test_backtests_router.py.

**Interfaces:**
- IFIND_REFRESH_TOKEN is the shared server-side configuration.
- IFIND_ACCESS_TOKEN remains a static fallback.
- Availability errors name both accepted variables without revealing values.
- Background-process redaction covers both token names and both credential values.

**Steps:**

1. Add failing tests for refresh-only availability, missing-both availability, and redaction of a refresh-token canary.
2. Run the focused tests and confirm they fail for the missing configuration behavior.
3. Add this English environment block:
   ~~~dotenv
   ENABLE_IFIND_ASHARE=false
   IFIND_REFRESH_TOKEN=
   IFIND_ACCESS_TOKEN=
   IFIND_BASE_URL=https://quantapi.51ifind.com
   ~~~
4. Update the provider error to:
   "iFinD credentials are not configured; set IFIND_REFRESH_TOKEN or IFIND_ACCESS_TOKEN".
5. Update subprocess-output redaction to replace both configured values and redact refresh_token and access_token fields.
6. Run the focused tests.
7. Commit:
   feat(ifind): accept refresh token configuration

## Task 2: Implement lazy access-token exchange and retry

**Files:**
- Modify dashboard/backend/infrastructure/market_data/ifind_client.py:18-117,207-333.
- Extend dashboard/backend/tests/infrastructure/market_data/test_ifind_client.py.

**Interfaces:**
- Constructor accepts optional refresh_token and monotonic clock dependencies in addition to existing arguments.
- Explicit constructor values take precedence over environment variables.
- IFindHttpClient continues to support static-token callers without making an exchange request.
- Add an internal IFindTokenRefreshError with sanitized messages.

**Steps:**

1. Add failing tests for:
   - exchanging the refresh token before the first data request;
   - static-token fallback without an exchange;
   - one 401/403 refresh-and-retry sequence;
   - six-day cache expiry using a fake clock;
   - concurrent first requests causing one exchange;
   - malformed or error responses with sanitized exceptions.
2. Run the focused client tests and confirm the new tests fail.
3. Add:
   - ACCESS_TOKEN_ENDPOINT = "/api/v1/get_access_token";
   - ACCESS_TOKEN_MAX_AGE_SECONDS = 6 * 24 * 60 * 60.
4. Resolve refresh and static credentials from explicit arguments first, then environment variables, with refresh precedence.
5. Implement a lock-protected _get_access_token() cache using the injected monotonic clock.
6. Exchange with a POST request containing Content-Type: application/json and the refresh_token header. Validate HTTP status, JSON shape, a mapping data, and a non-empty string access_token. Treat errorcode the way the data path does -- a failure only when the field is present and non-zero -- so a response that omits it but carries a valid token is accepted rather than reported as a credential problem.
7. Build data-request headers from the current provider token on every attempt.
8. Preserve existing connection, 429, and 5xx retry behavior. For a 401/403, replace the rejected token, perform one refresh, and retry once; do not loop indefinitely. The retry must not spend a transport retry slot -- an expired token discovered on the last attempt still has to be re-sent, and falling out of the retry loop raises an error the engine does not catch.
9. Sanitize all refresh failures and keep credentials out of exception text.
10. Run the focused client tests.
11. Commit:
    feat(ifind): refresh access tokens on demand

## Task 3: Verify shared Render configuration and user-facing boundaries

**Files:**
- Extend dashboard/backend/tests/integration/test_ifind_ashare_backtest.py.
- Extend dashboard/backend/tests/infrastructure/market_data/test_ifind_ashare.py.
- Extend dashboard/backend/tests/test_market_data_features.py.
- Extend dashboard/backend/tests/test_backtests_router.py.
- Inspect dashboard/backend/tests/test_ifind_ashare_frontend.py; modify only if a regression is found.

**Acceptance checks:**

1. The backtest subprocess inherits the server-side refresh credential through the existing copied environment, but the credential is absent from the command, progress payload, run metadata, and logs.
2. Constructing the iFinD provider does not perform a network request; the first data request does.
3. Refresh-only configuration passes feature availability and missing-both configuration fails with the documented message.
4. The frontend contains neither IFIND_REFRESH_TOKEN nor refresh_token.
5. Existing static-token integration tests continue to pass.
6. Run focused integration, router, provider, and frontend tests.
7. Commit:
   test(ifind): verify shared refresh credentials

## Task 4: Full verification and deployment handoff

**Files:** No production code changes expected unless a preceding test exposes a defect.

**Steps:**

1. Run backend tests:
   python -m pytest dashboard/backend/tests/ --timeout=180 -p no:cacheprovider
2. Run:
   node --check dashboard/frontend/app.js
3. Run:
   git diff --check
4. Scan tracked files for credential-like values and confirm the working tree contains no database, log, or temporary artifacts.
5. Confirm the branch is based on the fetched origin/main and list the commits.
6. Prepare the Render handoff outside Git:
   ~~~text
   ENABLE_IFIND_ASHARE=true
   IFIND_REFRESH_TOKEN=<admin enters the real secret directly in Render>
   ~~~
   Leave IFIND_ACCESS_TOKEN empty unless a temporary compatibility fallback is required.
7. Redeploy the service and verify an A-share backtest. Logs may say that an iFinD access token was refreshed, but must never print the token itself.
8. Report the branch, commits, test results, and the one-time Render configuration step. Do not request or record the secret in chat.

**Execution order:** Complete Tasks 1-4 in order. Keep each commit independently reviewable.
