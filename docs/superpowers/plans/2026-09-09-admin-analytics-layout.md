# Admin Analytics Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make production Analytics match the approved demo layout and expose the agreed global and per-user analytics fields.

**Architecture:** Preserve existing analytics data contracts and routing. Refine the existing admin view into explicit sections, and keep the profile as a separate URL-backed view with shared filter context.

**Tech Stack:** Existing dashboard frontend, HTML/CSS/JavaScript, FastAPI analytics contracts, pytest, frontend contract tests.

**Spec:** `docs/superpowers/specs/2026-09-09-admin-analytics-layout-design.md`

## Global Constraints

- Date ranges are `1D`, `1W`, `1M`, and `1Y`; default is `1W`.
- Lifecycle states are New, Active, At risk, and Dormant only.
- Do not add token-volume cards, provider-selection UI, device/region/browser fields, benchmark/configuration fields, or agent-removal history.
- Preserve keyboard navigation, accessible names, visible focus, partial-section retry, and URL-backed state.

### Task 1: Map existing analytics surface and contracts

**Files:**
- Inspect: `dashboard/frontend/**`, `dashboard/backend/**`, `tests/**`
- Test: existing analytics contract tests

- [ ] Identify the current production analytics entry point, section renderers, URL state keys, profile route, and API response types.
- [ ] Record the exact existing selectors and fixtures that must remain compatible.
- [ ] Run the focused analytics test command and save the baseline result.

### Task 2: Align Analytics home structure

**Files:**
- Modify: existing Analytics page component/template and analytics styles
- Test: focused analytics UI/contract tests

- [ ] Render header, filters, Pulse, Acquisition, Return, Resource & purchase, and Action queue in that order.
- [ ] Ensure Acquisition and Action queue columns exactly match the spec, including Source and Open profile.
- [ ] Remove excluded fields from visible production markup while preserving safe fixture compatibility.
- [ ] Implement the four date ranges with `1W` default and preserve URL state.
- [ ] Add section-level loading, empty, error, and retry states without hiding unaffected sections.
- [ ] Run focused tests and commit: `feat: align analytics home layout`.

### Task 3: Align independent User Analytics Profile

**Files:**
- Modify: existing profile view/template, profile styles, and navigation helpers
- Test: profile navigation and rendering tests

- [ ] Put breadcrumb and Back first, preserving the prior analytics URL context.
- [ ] Render the progress-first tracker: Account created, Agent created, Backtest attempted, First successful result.
- [ ] Render the agreed summary fields and account-management link.
- [ ] Keep tabs Overview, Timeline, Runs, Usage, Sessions.
- [ ] In Runs, show initiation, selected model, market/index scope, backtest selections, events, and result status.
- [ ] Add keyboard-accessible row/detail navigation and URL-backed profile state.
- [ ] Run focused tests and commit: `feat: align analytics user profile`.

### Task 4: Accessibility and regression verification

**Files:**
- Modify: analytics templates/components only where needed
- Test: analytics accessibility, contract, and navigation suites

- [ ] Verify accessible names, focus order, Enter/Space activation, table semantics, and visible focus styles.
- [ ] Verify filters, date range, profile drill-down, and Back preserve state.
- [ ] Verify partial errors expose retry controls only in the failing section.
- [ ] Run focused analytics tests, JavaScript syntax checks, and the relevant backend suite.
- [ ] Commit: `test: verify analytics layout and navigation`.

### Task 5: Visual acceptance against demo

**Files:**
- Inspect: `analytics-layout-demo.html`

- [ ] Start the local app and compare 8766 Analytics against the demo at 8765.
- [ ] Check rail placement, section order, spacing, table columns, profile transition, and Back behavior at desktop and narrow widths.
- [ ] Fix only deviations covered by the spec, rerun tests, and report remaining intentional differences.
