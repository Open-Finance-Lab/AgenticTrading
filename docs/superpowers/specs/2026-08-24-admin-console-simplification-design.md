# Admin Console Simplification Design

Date: 2026-08-24

Status: Approved for implementation

## Goal

Make the administrator console scannable by grouping the existing controls into
four tabs and removing controls that duplicate the selected context.

## User-visible structure

The Admin page contains four tabs:

1. `Grant Pool` — pool balance and one signed adjustment form.
2. `Users` — site statistics and user Grant/Purchased balances and allocation actions.
3. `Providers` — approved provider registry and platform credentials.
4. `Activity` — immutable Grant activity.

`Users` is the default tab. The previous standalone Overview area is removed;
its statistics live at the top of Users. Provider controls remain admin-only
and are not mixed with the user API Keys tab under Credits & Billing.

## Grant Pool interaction

The two pool mutation forms become one `Adjust Grant Pool` form:

- positive values call the existing `fund` API route;
- negative values call the existing `reduce` API route;
- zero, malformed values, and values with more than six decimal places are rejected;
- the input uses `type="number"` and `step="0.000001"`;
- the source is not editable and the client always sends `admin-console`;
- the existing reason field remains required.

The backend routes, integer micro-Credit ledger, authorization, idempotency, and
accounting semantics are unchanged.

## State and accessibility

Tab state is held in the URL query string when possible and defaults to Users.
Each tab uses `role="tab"`, `aria-selected`, and an associated `tabpanel`.
Hidden panels are not interactive. Existing refresh and authorization handling
continues to operate against the active admin surface.

## Verification

- Static tests assert that the old Overview, monthly metrics, source inputs, and
  separate Fund/Reduce forms are absent.
- Static tests assert the signed-input dispatch and fixed `admin-console` source.
- JavaScript syntax checks run with Node.
- Existing Admin API tests remain unchanged because API contracts do not change.
- Browser verification checks default Users, all four tabs, signed pool input,
  and no secret-bearing fields.
