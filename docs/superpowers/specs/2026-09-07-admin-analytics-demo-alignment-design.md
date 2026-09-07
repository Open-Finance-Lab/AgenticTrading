# Admin Analytics Demo Alignment Design

**Date:** 2026-09-07
**Status:** Approved visual direction; written-spec review pending
**Target branch:** `feature/admin-analytics-movement-profile`
**Visual reference:** `analytics-layout-demo.html` in the local worktree

## Summary

The production Admin Analytics surface must use the information hierarchy,
content density, and visual language approved in the local layout demo. The
current production markup combines older lifecycle, acquisition, and
operational concepts and therefore does not represent the approved product.

This change replaces the Admin Analytics presentation while preserving the
existing authenticated API contracts, URL-backed navigation, safe data
handling, partial-error behavior, and accessible keyboard interactions.

This document supersedes the Analytics page layout and headline metric choices
in the 2026-09-03 user-value design and the disclosure-first layout in the
2026-09-06 acquisition addendum. Their backend, attribution, privacy, audit,
and data-retention contracts remain authoritative.

## Goals

1. Make the real `/app?view=admin&adminTab=analytics` page visually and
   structurally match the approved demo.
2. Put the four core user-analysis metrics first and make their time scope
   explicit.
3. Show acquisition, return behavior, resource and purchase signals, and the
   action queue in one deliberate top-to-bottom story.
4. Keep the existing full User Analytics Profile as a dedicated, reversible
   drill-down rather than mixing user detail into the overview.
5. Preserve real data, accessibility, partial failures, and safe rendering.

## Non-goals

- Replacing the existing Admin APIs with demo fixtures or hard-coded values.
- Embedding the static demo in an iframe.
- Redesigning the global product navigation outside the Admin workspace.
- Changing account management, provider management, Credits, billing, or
  audit behavior.
- Displaying token counts, raw prompts, strategy content, provider response
  bodies, credentials, or other sensitive evidence.
- Adding benchmark, decision frequency, currency, transaction-cost rules,
  market rules, pipeline steps, output-token limits, or automatically selected
  provider fields to the Analytics overview or run detail.

## Source Of Truth

The local `analytics-layout-demo.html` is the visual baseline. The shipped page
must reproduce its layout proportions, section order, border-led depth,
typographic hierarchy, restrained color use, table density, and responsive
rail behavior. Demo numbers and names are illustrative only; production always
renders authenticated API data.

The global ATL header remains part of the application shell. The alignment
requirement starts at the Admin workspace immediately beneath that header.

## Information Architecture

### Admin workspace rail

The Admin workspace is a two-column layout with a fixed left rail and a fluid
content region. The rail order is:

1. Analytics
2. Users
3. Providers
4. Activity

The rail contains the ATL Admin identity at the top and a short operator-state
summary at the bottom. The horizontal Admin tab strip and redundant Admin page
heading are removed. `adminTab` remains the URL source of truth.

On narrow screens the rail remains on the left as an icon rail. Labels become
accessible names and tooltips; the navigation does not move above the content.

### Analytics header and filters

The content header contains the `Analytics` title, one concise description,
the range selector, last-updated time, and an icon-plus-label refresh action.
The only date ranges are `1D`, `1W`, `1M`, and `1Y`; `1W` is the default.

The filter row contains:

- Source
- Operating cohort
- Lifecycle: New, Active, At risk, Dormant
- Paid status
- Include internal accounts

Filters update the URL and reload affected panels without a separate Apply
button. A context line states the active source/cohort/user scope and clarifies
that progress metrics are lifetime while activity metrics use the selected
range.

### Section 01: Pulse

The first section is `What is happening now?` and contains exactly four
always-visible metrics:

| Metric | Definition | Scope |
| --- | --- | --- |
| Active users | Distinct users with an accepted `backtest_requested`, regardless of outcome | Selected range |
| Task completed | Users with at least one persisted, viewable successful backtest result | Lifetime |
| Repeat users | Users with successful backtests on at least two different UTC dates | Lifetime |
| ATL Credits settled | Settled platform-credit usage debits; excludes reservations, grants, refunds, and BYOK spend | Selected range |

These values come from the acquisition analytics contract so their definitions
remain identical to the group table. Each metric has a compact definition and
scope below its value. Loading or unavailable values retain their fixed space
and never shift the layout.

### Section 02: Acquisition

`Where users come from` is always visible. The acquisition table contains only
the approved columns, in this order:

1. Group
2. Users
3. Active
4. Task completed
5. Repeat
6. Runs / user
7. ATL Credits
8. Paid intent
9. Paid

The table defaults to source groups. Selecting a source filter or operating
cohort changes the slice, not the meaning of a metric. Clicking a group or a
count opens the Users workspace with matching URL filters. The chevron/collapse
control remains available, but the table is expanded by default.

### Sections 03 and 04: Return and value exchange

Two equal-width compact panels follow the acquisition table.

`Do they come back?` shows daily active users across the selected range and a
returning-user comparison when available. It links to the fuller retention
view without putting the cohort table on the overview.

`Resource and purchase signal` shows:

- Runs per active user
- ATL Credits settled per user
- Checkout starters
- Credits page viewed -> Checkout started -> Purchase settled

The funnel is an observed event sequence, not a prediction score. Admin Grants
never count as purchase intent or payment. When an upstream signal is not
available from an authoritative contract, that row displays `Unavailable`
rather than inventing or estimating a number.

### Section 05: Action queue

`Who needs attention?` is the final overview section. It shows a compact table
with:

- User
- Source
- Status
- Reason
- Last active
- Runs
- A visible profile chevron/action

The queue is ordered by operational urgency first, then inactivity. Lifecycle
and operational state stay separate in data; a blocking state may be shown as
the primary status because it is immediately actionable. The visible lifecycle
labels are the simplified New, Active, At risk, and Dormant set.

Selecting the user name, row action, or keyboard equivalent opens the full User
Analytics Profile. `Open all users` opens the Users workspace with the current
filters.

## User Analytics Profile

User detail remains a dedicated Analytics view with a breadcrumb and `Back to
analytics overview` action that restores the prior filters and scroll context.
It visually follows the demo profile treatment but is not a modal-only or
fixture-only drawer.

The first content is the four-step progress indicator:

```text
Account created -> Agent created -> Backtest attempted -> First successful result
```

The profile header shows identity, Source, Operating cohort, current lifecycle,
and operational state. Source and cohort remain editable only through the
existing audited admin action; no edit-reason field is required.

The profile keeps these sections:

1. Overview
2. Timeline
3. Runs
4. Usage
5. Sessions

Run rows show the approved core facts: request/start/end time, selected model,
billing mode, market-data source, market/universe or symbols, date range,
terminal status, result summary, and display-safe error reason. Token totals
and automatically selected provider are not rendered.

`Open account management` continues to the existing Users workspace and exact
account.

## Data And Failure Behavior

The UI continues to use the existing Admin Analytics endpoints and safe mock
fixtures in frontend contract tests. Independent requests use settled-result
handling so one failed endpoint does not blank successful sections.

Each panel has stable loading, ready, empty, unavailable, stale, and partial
states. A failed panel shows a concise message and a local Retry action. A
successful refresh updates the visible timestamp. Authentication or admin
access loss returns to the existing access-loss flow.

No demo fixture, synthetic user, or illustrative number ships in the runtime
path.

## Accessibility And Interaction

- The Admin rail remains a vertical tablist with roving `tabindex`, Up/Down,
  Home/End, synchronized `aria-selected`, and stable `aria-controls`.
- Date ranges use pressed-state buttons with keyboard focus styles.
- Charts have text/table equivalents and do not rely on color alone.
- Tables use captions and scoped headers; drill-down controls have explicit
  accessible names.
- Collapsible sections synchronize `aria-expanded` and `aria-controls`.
- Loading uses `aria-busy`; status and error updates use appropriate live
  regions without repeatedly announcing polling.
- Profile navigation moves focus to its heading and Back restores focus to the
  invoking control when possible.
- All interactive targets are at least 40 CSS pixels, with 44 pixels preferred.
- Reduced-motion preferences disable nonessential movement.

## Responsive Behavior

- Wide screens use the demo's fixed rail and constrained, fluid analytics
  content width.
- Medium screens keep the rail while allowing tables to scroll horizontally.
- At 760 CSS pixels and below the rail becomes an icon rail and the page header
  stacks.
- Metric cells form two columns on small tablets and one column on phones.
- The two compact analysis panels stack before content becomes unreadable.
- Text, controls, tables, badges, and profile content must not overlap at 1440,
  1024, 768, and 390 CSS-pixel viewport widths.

## Acceptance Criteria

1. The real Admin Analytics overview matches the approved demo's section order,
   fields, density, rail placement, and visual hierarchy.
2. The old Activated/Core/At risk/Paid headline and always-visible lifecycle
   distribution/movement composition no longer occupies the overview.
3. Pulse uses Active users, Task completed, Repeat users, and ATL Credits
   settled with the exact scopes defined above.
4. Acquisition is expanded by default and uses the exact nine approved
   columns.
5. Return, resource/purchase, and action-queue panels occupy the same hierarchy
   as the demo and render real data or an explicit unavailable state.
6. User drill-down opens the dedicated profile and Back returns to the prior
   filtered overview.
7. Existing partial-error, access-loss, safe-rendering, ARIA, keyboard, and URL
   contracts remain covered by tests.
8. Browser screenshots at desktop and mobile widths show no overlap and a
   materially faithful match to the demo.

