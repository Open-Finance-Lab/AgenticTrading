# iFinD Access Token Refresh Design

**Status:** Draft for review  
**Date:** 2026-08-19  
**Scope:** Server-side iFinD A-share market-data access for ATL backtests and paper simulation

## Goal

Configure the official iFinD account once on the Render backend and let every ATL user use A-share backtests without entering iFinD credentials. ATL must automatically exchange the long-lived `refresh_token` for a seven-day `access_token` whenever a data request needs one.

This feature only authenticates market-data requests. It does not place live orders or expose a brokerage account.

## Current State

- The latest `origin/main` baseline is `e8473fe`.
- `IFindHttpClient` currently reads only `IFIND_ACCESS_TOKEN` from the backend environment.
- The iFinD provider already converts official responses into ATL OHLCV data and is feature-gated by `ENABLE_IFIND_ASHARE`.
- Existing local configurations may still contain `IFIND_ACCESS_TOKEN` and must continue to work during migration.

## User Experience

### Administrator

The administrator sets these Render environment variables once and redeploys:

```text
ENABLE_IFIND_ASHARE=true
IFIND_REFRESH_TOKEN=
```

The administrator fills the empty value in Render's secret configuration with the official iFinD refresh token. The refresh token remains a backend secret. It is never committed, returned by an API, sent to browser code, or written to run metadata.

### ATL User

The user selects the iFinD A-share market in ATL and starts a backtest. No iFinD login, token, or additional setup is required in the browser.

### Failure requiring administrator action

If the official account expires or its permissions change, the refresh token may become invalid. ATL shows a sanitized configuration error telling the administrator to replace `IFIND_REFRESH_TOKEN` in Render. The error does not include either token value.

## Architecture

### Token sources and precedence

1. `IFIND_REFRESH_TOKEN` is the preferred production credential.
2. When a refresh token is present, ATL exchanges it for an access token and ignores a simultaneously configured static access token.
3. `IFIND_ACCESS_TOKEN` remains a legacy fallback for local development and existing deployments that have not migrated yet.
4. If neither value is configured, the existing missing-credentials error is returned.

### Lazy refresh with an in-memory cache

Add a small token provider beside the existing iFinD HTTP client:

1. On the first iFinD data request in a process, call `POST /api/v1/get_access_token` with `Content-Type: application/json` and the `refresh_token` request header.
2. Parse `data.access_token` from the JSON response and keep it only in process memory.
3. Treat the cached token as usable for six days from the exchange time. The one-day buffer avoids depending on the exact server-side expiration boundary.
4. After six days, the next iFinD request exchanges the refresh token again before requesting data.
5. Protect the cache with a process-local lock so concurrent backtests perform one exchange rather than a request per worker thread.
6. Do not run a background timer. If the service is idle, no token refresh is needed; the next request refreshes it automatically.

### Request retry

If a data request returns HTTP 401 or 403 while a refresh token is configured, replace the access token that was rejected (compare-and-swap under one lock acquisition, so concurrent callers hit by the same rotation share a single exchange instead of clobbering each other's tokens), exchange the refresh token once, and retry the original request once without consuming a transport retry slot. Unknown iFinD business error codes are not retried automatically; they remain sanitized provider errors instead of risking an infinite loop or duplicate data request.

### Existing provider boundary

`IFindAshareProvider` and the OHLCV adapter continue to consume the HTTP client. They do not know whether the client uses a refreshed or legacy token. Alpaca, vn.py simulation, and the backtest execution engine keep their existing behavior.

## Configuration

Update `.env.example` with the following server-only settings:

```text
# iFinD A-share historical market data (default OFF).
ENABLE_IFIND_ASHARE=false
IFIND_REFRESH_TOKEN=
# Legacy local fallback; ignored when IFIND_REFRESH_TOKEN is set.
IFIND_ACCESS_TOKEN=
IFIND_BASE_URL=https://quantapi.51ifind.com
```

`IFIND_BASE_URL` remains available for controlled tests. Production must use the official iFinD endpoint.

## Error Handling and Observability

- Refresh HTTP failures, invalid JSON, missing `data.access_token`, and nonzero iFinD business responses become dedicated sanitized client errors.
- Logs may record the endpoint, status class, retry count, and whether refresh mode or legacy mode was used.
- Logs must never record a refresh token, access token, authorization header, full response body, or request payload containing credentials.
- A refresh failure must not silently fall back to an expired access token when refresh mode is configured.
- A successful refresh may emit a token-free debug/info event such as `iFinD access token refreshed`.

## Security and Deployment Boundaries

- The refresh token is configured only in Render's backend environment, not in Vercel/browser variables.
- No token is persisted in SQLite, PostgreSQL, run metadata, screenshots, API responses, or Git history.
- The refresh endpoint is called only from the backend.
- The feature remains read-only market-data access for backtest and paper-simulation workflows. It is not a live trading integration.
- A refresh token update or iFinD account permission change requires a new Render deployment/restart so all processes discard old in-memory state.

## Testing Strategy

Add deterministic tests using a fake HTTP session; no real iFinD credential is used in CI:

1. Exchange success sends the refresh token in the request header, extracts `data.access_token`, and never exposes the token in an exception or log.
2. A cached token is reused before six days and a new exchange occurs after the six-day boundary.
3. Concurrent first requests perform one exchange and all callers receive the same cached token.
4. HTTP 401/403 invalidates the cache, refreshes once, and retries the original data request once; a second authentication failure is returned.
5. A configured legacy `IFIND_ACCESS_TOKEN` works without calling the refresh endpoint.
6. Missing credentials, malformed exchange responses, and nonzero business errors produce stable sanitized error messages.
7. Existing iFinD adapter/provider/backtest tests continue to pass, including the guarantee that `refresh_token` never appears in frontend source or run metadata.

## Acceptance Criteria

- An administrator configures one Render refresh token and all users can run iFinD A-share backtests without user-side credentials.
- A process automatically obtains a new seven-day access token after the six-day cache window or an HTTP 401/403 response.
- The access token is never written to persistent storage, browser code, logs, API responses, or Git.
- Existing static-token local setups still work.
- No Alpaca/vn.py behavior or live-trading capability changes.
- Focused tests and the repository backend test suite pass.

## Out of Scope

- Per-user iFinD credentials or a browser credential-management screen.
- A Render API integration that rewrites environment variables.
- Automatic refresh-token rotation; if iFinD invalidates the long-lived token, an administrator must replace it in Render.
- Real-market order submission or broker account access.
