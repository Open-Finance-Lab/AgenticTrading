# Admin Users and Grant Pool Integration Design

Date: 2026-08-24

Status: Approved design, pending implementation planning

## Goal

Make the Admin `Users` page the single working surface for account allocation
and Grant Pool management. Remove the standalone `Grant Pool` tab without
changing Credits accounting, authorization, or backend API contracts.

## User-visible structure

The Admin page will contain three tabs:

1. `Users` — site statistics, Grant Pool summary and adjustment, and user
   Grant/Purchased balances with allocation actions.
2. `Providers` — approved provider registry and platform credentials.
3. `Activity` — immutable Grant activity.

`Users` remains the default tab. The standalone `Grant Pool` tab and its
tabpanel are removed. The existing Grant Pool controls are moved into the
Users panel in this order:

1. site statistics;
2. Grant Pool ring showing Total Pool, Available, and Allocated;
3. `Adjust Grant Pool` signed amount and reason form; and
4. user search, Role, Grant/Purchased/Total balances, and Assign/Reclaim
   actions.

The existing Role control remains in the Users table. The legacy user-quota
table remains hidden and its backend behavior is preserved for compatibility.

## State and compatibility

The existing `admin-credits.js` module continues to own Grant Pool loading,
rendering, mutation, user allocation, and activity refresh. Moving the markup
does not introduce a second data source or duplicate mutation handlers.

The existing Grant Pool element IDs remain stable so the current client logic
continues to update the moved controls. The `Grant Pool` refresh button remains
available in the moved section. Existing Grant Pool endpoints, idempotency
payloads, fixed `admin-console` source, and audit reasons are unchanged.

If a URL contains `adminTab=grant-pool`, the tab controller treats it as the
Users tab. The URL is normalized to `adminTab=users` when the Admin surface is
entered. Keyboard navigation follows the remaining DOM order:

```text
Users -> Providers -> Activity -> Users
```

## Accessibility

- The Users tab's `aria-controls` points only to the Users panel.
- The removed Grant Pool tab has no remaining focusable control.
- The moved Grant Pool summary keeps its existing accessible label and live
  status messaging.
- Hidden legacy markup remains non-interactive through the existing `hidden`
  attribute.
- The three-tab keyboard cycle remains supported by left/right arrow keys.

## Data flow and error handling

No backend changes are included. The Users entry path continues to load:

- `/api/admin/credits/grant-pool` for pool summary;
- `/api/admin/credits/users` for account balances and Role;
- existing Grant Pool fund/reduce routes for signed adjustments;
- existing Grant assign/reclaim routes for user mutations; and
- `/api/admin/credits/activity` for the Activity tab.

Existing access-loss handling, sanitized errors, idempotency behavior, and
refresh status messages remain unchanged. A pool failure continues to surface
through the existing Admin Credits status area rather than blocking the user
table markup from rendering.

## Scope boundaries

Included:

- Admin tab removal and three-tab state handling;
- moving the existing Grant Pool UI into Users;
- preserving the existing ring visuals and adjustment interactions;
- updating static frontend contracts and cache versions if required.

Excluded:

- backend routes, database schema, ledger calculations, or Credits accounting;
- Purchased Credits behavior;
- BYOK provider behavior;
- legacy entitlement semantics; and
- browser automation or visual QA by the agent. Browser verification remains
  a manual user step.

## Verification plan

The user will perform the browser and interaction checks. The implementation
handoff will include focused commands for the user to run:

- the Admin frontend static contract tests;
- Node syntax checks for the changed Admin modules; and
- `git diff --check`.

Acceptance criteria:

- Admin shows exactly `Users`, `Providers`, and `Activity` tabs;
- Users displays the Grant Pool ring and adjustment form above the user table;
- `adminTab=grant-pool` opens Users;
- Grant Pool and user mutations still use the existing API contracts; and
- no legacy quota table is visible.
