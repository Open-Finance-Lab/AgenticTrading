# Admin Provider Controls and Tab Order Design

Date: 2026-08-24

Status: Approved for implementation

## Goal

Make the Admin provider actions compact and put the most frequently used
`Users` surface before `Grant Pool` in the Admin navigation.

## Provider action controls

The `Save provider` and `Save and verify` buttons in the Admin Providers tab
will use a scoped compact control style. Their check icons will have explicit
15px dimensions, preventing the browser's default 300x150px inline SVG size
from expanding the buttons. On desktop, both buttons will size to their
content and align to the start of their form row. On narrow screens, the
platform credential action remains full width for reliable touch targeting;
the provider registry action remains compact unless the existing layout needs
to stack it.

No API payload, credential handling, validation, or authorization behavior
changes.

## Admin tab order

The tab DOM order will become:

1. `Users`
2. `Grant Pool`
3. `Providers`
4. `Activity`

`Users` remains the default tab. The URL query parameter continues to select
an explicit tab, and the existing left/right keyboard navigation follows the
DOM order automatically. Associated `aria-controls`, `aria-labelledby`, and
tabpanel IDs remain unchanged.

## Verification

- Static frontend tests assert the new tab order and default Users behavior.
- Static frontend tests assert scoped Provider button sizing and explicit SVG
  dimensions.
- Node syntax checks run for the affected Admin modules.
- `git diff --check` verifies whitespace and patch integrity.
- Browser login and visual interaction checks remain manual for the user.
