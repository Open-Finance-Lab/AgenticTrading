# BYOK Review Fixes Design

## Goal

Close the two independent review findings on PR #396 and make its architecture guard reliable in GitHub Actions without expanding the PR into model execution.

## Scope

This change is limited to credential verification transport selection, user credential creation consistency, and the failing architecture test. It does not change provider approval policy, credential API responses, frontend behavior, model execution, billing, or the existing encrypted storage format.

## Verified-Origin Proxy Policy

`BROKER_CREDENTIAL_VERIFICATION_PROXY` remains an explicit operator-only escape hatch for local environments that cannot resolve the four native providers directly. A request may use that proxy only when both of these conditions hold:

1. The adapter is one of `openrouter`, `openai`, `anthropic`, or `gemini`.
2. The request origin exactly matches that adapter's seeded official HTTPS origin, including the effective port.

Path differences do not affect the origin comparison. Userinfo, query data, fragments, non-HTTPS schemes, alternate ports, and all custom origins are ineligible for proxying. Ineligible requests use the existing DNS validation and IP-pinned transport, even if a proxy is configured. OpenAI-compatible providers therefore always remain on the pinned path.

The policy lives next to the adapter transport decision rather than in the provider repository. This keeps the repository responsible for approved configuration syntax and keeps network routing enforcement at the outbound security boundary.

## Atomic Credential Creation

Credential creation follows this order:

1. Resolve the enabled BYOK provider.
2. Fail closed if `BROKER_TOKEN_ENCRYPTION_KEY` is absent or invalid.
3. Validate the submitted secret through the provider adapter.
4. Normalize the adapter result to a fixed, non-sensitive verification message.
5. Call `create_user_credential` once with the final status, verification timestamp, and requested default flag.

The repository remains the transaction owner. SQLite and Postgres already create the row and switch the provider default within one transaction when final values are supplied, so the service must not create a placeholder row and mutate it afterward.

The network request must not run inside a database transaction. A slow provider therefore cannot hold a SQLite write lock or a Postgres transaction open. A validation result of `invalid` or `verification_unavailable` is still a valid final record; only an infrastructure or database failure prevents creation.

Reverification remains a mutation of an existing record and is outside this creation-atomicity finding.

## Encryption Preflight

The preflight reuses the existing fail-closed Fernet configuration path and never logs or returns the configured key. Its purpose is ordering: a missing or invalid encryption key must reject the request before the submitted provider key leaves the process.

The repository still encrypts the secret during the single create transaction. The preflight does not persist ciphertext and does not introduce a second encryption format.

## CI Architecture Guard

`test_key_vault_pr_has_no_model_execution_runtime` must not execute `git diff origin/main...HEAD`. GitHub Actions uses a shallow checkout where `origin/main` is not guaranteed to exist, so the current test fails after the full suite has otherwise passed.

The replacement combines local `git ls-files` inventory with a direct filesystem scan of the forbidden runtime paths. The Git inventory covers tracked files plus untracked, non-ignored files without depending on a remote ref. The filesystem scan also rejects ignored importable artifacts, including sourceless `.pyc` files and stale `__pycache__` entries that can keep deleted package paths importable. PR diff review remains responsible for identifying unrelated changes outside those forbidden paths.

## Tests

- Native adapters use the explicit proxy only for their exact seeded official HTTPS origins.
- Native adapters with custom origins and all OpenAI-compatible providers use the pinned transport.
- Missing or invalid encryption configuration prevents both adapter validation and persistence.
- Creation calls the repository once with the final verification state and default intent.
- SQLite and Postgres preserve one verified default per user and provider through their existing create transaction.
- The architecture guard passes without `origin/main` and still fails for forbidden tracked, untracked, ignored, source, bytecode, or extension-module runtime files.
- Existing credential API, adapter, repository parity, and Postgres tests remain green.

## Non-Goals

- Routing real model execution through BYOK credentials.
- Supporting arbitrary user-configured provider origins.
- Persisting raw provider errors or response bodies.
- Adding retries, background verification, or new proxy configuration.
- Changing the API Keys frontend.
