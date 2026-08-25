# Admin Account Management Pagination

## Goal

Allow an Admin to inspect every account returned by the Grant Credits user
endpoint without changing the Credits ledger or assignment behavior.

## Design

- Keep the existing backend contract: `total`, `limit`, and `offset`.
- Keep a page size of 100 accounts to minimize requests for the current admin
  workflow.
- Add a footer below the account table with a range label plus `Previous` and
  `Next` buttons.
- Disable `Previous` on the first page and `Next` on the last page.
- Reset to the first page whenever the account search is submitted.
- Refreshes and Grant mutations preserve the current page.
- Use the server-provided `total` so the range is explicit (`Showing 1–100 of
  240`), including the empty state (`Showing 0 of 0`).
- Bump the Admin Credits script cache version so the browser receives the new
  behavior after deployment.

## Non-goals

- No changes to Grant/Purchased balance calculations.
- No changes to BYOK or Platform Credits execution.
- No changes to legacy entitlement storage or metering compatibility code.

## Acceptance criteria

1. A user list of more than 100 accounts can be traversed with Previous/Next.
2. Search starts at page one and reports the filtered total.
3. Refreshing or assigning/reclaiming Credits keeps the current page visible.
4. The buttons expose disabled state at both boundaries and remain usable on
   narrow screens.
