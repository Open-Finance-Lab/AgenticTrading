# Account Page Layout Redesign

## Scope

Redesign only the signed-in Account page content rendered by `#accountView` in
`dashboard/frontend/app.html`. The global header, brand bar, market ticker,
primary navigation, other page views, shared page shell, and backend behavior
are explicitly out of scope.

## Goals

- Use the available desktop width instead of leaving most of the viewport empty.
- Give the account identity a clear visual home before the editable settings.
- Keep all existing account capabilities and JavaScript contracts intact.
- Preserve the site's dark trading-lab visual language while improving hierarchy,
  scanning, focus states, and responsive behavior.

## Layout

The Account page keeps its existing page title and description, followed by a
responsive two-column workspace:

- **Identity column:** a compact profile summary containing the existing avatar
  preview, display name, email, and account role/status information. The logout
  action remains in this column as the destructive account action.
- **Settings column:** a two-column grid of the existing settings sections:
  display name, profile photo, change email address, and change password.
  Existing forms, field IDs, error containers, success containers, and submit
  buttons remain available so current JavaScript behavior does not change.

The workspace uses a scoped max width of approximately 1180px. On narrow
viewports the columns collapse to one column in a predictable order: identity,
display name, profile photo, email, password, logout.

## Visual Language

- Keep the existing dark background and typography foundation used by the app.
- Add Account-scoped surface, border, spacing, and heading rules rather than
  changing global selectors.
- Use a restrained cyan accent for primary actions and focus indicators,
  neutral slate borders for secondary controls, and red only for logout/error
  states.
- Remove the repeated summary rows currently shown above the editable display
  name form; the identity column becomes the single source of summary content.
- Use consistent panel padding, field heights, button alignment, and section
  headings so each setting reads as a deliberate unit.

## Behavior and Accessibility

- Do not add new account workflows, API calls, or navigation states.
- Keep all existing form submission, email-code stages, password policy hints,
  avatar upload/remove behavior, status announcements, and error handling.
- Preserve existing element IDs and event listeners in `app.js`.
- Keep visible labels associated with inputs and retain `role="status"`,
  `role="alert"`, and keyboard-submit behavior already present.
- Ensure focus-visible styles remain obvious against the dark surfaces and that
  buttons and inputs maintain comfortable touch targets on mobile.

## Implementation Boundaries

- Edit the Account markup only when needed to group the existing elements into
  the new identity/settings layout.
- Add or adjust CSS only with `.account-view`-scoped selectors. Do not alter
  shared `.page-view`, `.page-header`, `.auth-form`, or global button rules in
  ways that affect other routes.
- Do not change `dashboard/frontend/app.js` unless a purely presentational DOM
  grouping requires a selector update; existing IDs must remain stable.
- Do not change backend files, API contracts, or unrelated frontend assets.

## Verification

- Run the Account-related frontend contract tests and the existing fast-boot
  tests.
- Run `git diff --check` and verify the diff contains only Account markup/CSS
  plus any focused tests.
- Start the local frontend and inspect Account at desktop and mobile widths.
- Confirm Home, Community, Competition, Credits, and Admin views retain their
  existing layout by checking that no global selector was changed.

## Acceptance Criteria

1. Account desktop view presents the identity summary and settings in a balanced
   two-column workspace with no large unused right-side void.
2. All existing account actions still work without changing their API or event
   contracts.
3. Account mobile view is a readable single column with no horizontal overflow.
4. Other routes and the global navigation/header/ticker are unchanged.
5. Existing Account accessibility semantics and keyboard paths remain intact.
