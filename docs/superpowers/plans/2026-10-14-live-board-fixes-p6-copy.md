# Live-Board Fixes P6: Live Leaderboard copy, legacy label, stop-price disclaimer and catching-up badge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the board "Live Leaderboard" from one server-side owner, make its header copy static, label the Competition board as legacy, say on both boards that stop-loss and take-profit prices are not executed, and flag catching-up rows (#586), without changing any prompt, fee, slippage, fill or ranking rule.

**Architecture:** Two constants in `domain/leaderboard/service.py`, `LIVE_BOARD_TITLE` and `LIVE_PHASE_LABEL`, become the only owners of the Live board's name and season label: the payload, the freeze config and the service config read them, and tests pin the static `app.html` copy and the JS fallbacks to them. Everything else is static HTML copy plus four pure functions in `js/live-leaderboard.js`, which badge catching-up rows, sort them in their own tier and re-key the banner off the rows instead of `live_status.models_cached`. Characterization tests pin that the legacy single-prompt path drops the stop fields the new copy disclaims.

**Tech Stack:** Python 3.13 / FastAPI backend (`domain/leaderboard`, two routers, the refresh CLI's help text), pytest from the repo root, static `app.html` copy, and the vanilla-JS dashboard (`js/live-leaderboard.js`, `js/leaderboard.js`) tested under `node -e` against a fake DOM.

**Spec:** `docs/superpowers/specs/2026-10-14-atl-direction.md` §4 (Credibility bar: stop-loss/take-profit executed only by the new core) and §6 (Live Leaderboard: name, subtitle, cadence, "Season 0 · preview", the Competition label, and the stop-loss/take-profit disclaimer in the legacy-engine fix list). Decisions, Global Constraints, merge order and Review Focus live in the index plan `docs/superpowers/plans/2026-10-14-live-board-fixes.md`; every task here inherits its Global Constraints.

**Merge order:** P1 → P2 → P3 → P4 → P5 → P6 → P7 PR A → P7 PR B (owner-gated). Deadlines: P3 before 2026-11-01; P4 by 2026-10-31 (else bump both live_from values); P1 merged and deployed by 2026-11-20. One executor per contiguous range; phases are not safe to run in parallel (several append to the same test files and bump the same cache-busters).

## Phase 6: Live Leaderboard naming, legacy label, stop-price disclaimer and catching-up badge

**Why.** The direction spec renames the board **Live Leaderboard**, with the subtitle "Simulated fills · forward-only · no real money" and the cadence "Updated nightly after the close". It labels the months "Season 0 · preview" and the Competition board "legacy engine · historical window · may overlap model training data" (`docs/superpowers/specs/2026-10-14-atl-direction.md` §6). The legacy engine drops stop-loss and take-profit prices (conformance D12), so the board must say so until the new core executes them (§4 and the §6 legacy-engine fix list). Issue #586 also still leaves a catching-up row unflagged, and the banner then calls its curve missing. This phase is copy and rendering only. It changes no prompt, fee, slippage, fill or ranking rule.

**Depends on.** It needs no runtime interface from another phase. One Phase 3 test fixture does need updating; Task 6.5 does it (see below). Execute it after Phases 1–5 have merged (P1 `2026-10-14-live-board-fixes-p1-early-close.md`, P2 `2026-10-14-live-board-fixes-p2-memory-robustness.md`, P3 `2026-10-14-live-board-fixes-p3-attempt-ledger.md`, P4 `2026-10-14-live-board-fixes-p4-roster.md`, P5 `2026-10-14-live-board-fixes-p5-daily-sharpe.md`; all in `docs/superpowers/plans/`), and rebase on `main` first. (Phase 7's PR A, in P7 `2026-10-14-live-board-fixes-p7-records.md`, comes after this one: it reads `LIVE_BOARD_TITLE`.) It edits files those phases also touch:
- `js/live-leaderboard.js`: Phase 3 adds the budget-halted banner and the "Last nightly run" row; Phase 5 adds the Sharpe cells.
- `js/leaderboard.js` and `app.html`: Phase 5 adds Sharpe.
- `us_market_calendar.py`: Phase 1 adds early closes.
- `refresh_live_leaderboard.py`: Phase 4 edits the roster help text.
- `test_live_leaderboard.py`.
- Phase 3's `test_frontend_live_budget.py`.

Every anchor below is a short verbatim snippet chosen to survive those edits. Three anchors sit on lines that Phases 3 and 5 edit around. Each is given as **Form A** (main@17c552c5) and **Form B** (main@17c552c5 plus Phases 1–5), and each form was checked to match exactly once on its own tree. Where the two forms differ, neither matches on the other's tree. Apply whichever form your tree holds.
- **The type-badge line** in `populateLiveTable` (Task 6.5 edit 4). Phase 3 adds its `${attemptBadge ? … : ''}` line directly under it.
- **The banner's final `else`** in `updateLiveHeader` (Task 6.5 edit 3). Phase 3 adds its budget-halt block straight after the `if (banner) { … }` block, where main has the zone caption. That block appends "Model runs are paused: the monthly model budget is used up. Baselines keep updating." to whatever the branches chose.
- **The `setText` run** at the top of `updateLiveHeader` (Task 6.2 edit 13). Neither Phase 3 nor Phase 5 edits these six lines, so Form B is byte-identical to Form A. It matches once on each tree.

Stop rule for every other anchor, unless its step names a fallback: if it matches zero times or more than once on your tree, stop and report it. Do not improvise an edit.

- **Line hints** are positions on main@17c552c5 **before any task of this phase**. Earlier tasks of this phase move later hints in the same file. For example, once Tasks 6.3–6.4 are in, the `live-leaderboard.js` script tag sits at ≈:2540 of `app.html`, not ≈:2523. Match on the quoted text, never on the number.
- **Cache-busters.** The literal `?v=` anchors show bare main (`leaderboard.js?v=37`, `live-leaderboard.js?v=5`). After Phases 1–5, expect `leaderboard.js?v=38` (Phase 5 bumped it) and `live-leaderboard.js?v=7` (Phases 3 and 5 bumped it once each). Always read the value first and write N → N+1. Applying the literal 37 → 38 to a file that already says 38 changes nothing, so the browser keeps serving the old JS.
- **Phase 3 interaction (fixed in Task 6.5).** Task 6.5 makes the banner read the rows instead of `live_status.models_cached`. Phase 3's `test_frontend_live_budget.py::_banner_after` builds a "healthy" board with `entries: []` and `models_cached: 1`. Without the fixture edit in Task 6.5 Step 1, five of its tests fail: `test_a_budget_halt_shows_a_banner_line_on_a_healthy_board` and the four `test_no_budget_line_without_a_real_true` cases.
- **Stacked-tree check.** This phase's anchors were checked against main@17c552c5 plus Phases 1, 2, 3 and 5 as their plans write them. Every anchor matched exactly once there, except the cache-busters handled above, the two blocks edited with `replace_all` (two matches each, as their steps say) and the Form A halves of the pairs above, which match only on main@17c552c5. Phase 4's plan edits none of these anchors. With the whole phase applied on that tree, Task 6.5's fixture edit included, the `test_frontend_*`, `test_live_*`, `test_leaderboard_*`, `test_landing_*`, `test_app_copy_register`, `domain/leaderboard` and `domain/backtesting` tests all passed (`1158 passed, 43 skipped`, with node on PATH).

**Branch / PR title.** `feat/live-board-copy` → `feat(live): Live Leaderboard copy and catching-up badge`

**Interfaces this phase produces.**
- `dashboard/backend/domain/leaderboard/service.py`:
  - `LIVE_BOARD_TITLE: str = "Live Leaderboard"`.
  - `LIVE_PHASE_LABEL: str = f"Season {PREVIEW_SEASON_NUMBER} · preview"`, which is `"Season 0 · preview"`.
  - These two constants are the only owners of the Live board's name and season label.
- `GET /api/v1/leaderboard?period=live` (`live.get_live_leaderboard`) now answers `board_title: "Live Leaderboard"` and `phase_label: "Season 0 · preview"`. Before this phase it sent `"Live Trading Leaderboard"` and `"Season 0"`.
  - `live.live_freeze_config(...)` and `service.resolve_leaderboard_config("live")` carry the same two values.
  - `updated_at` is unchanged in the payload, but no UI renders it any more.
- `dashboard/frontend/js/live-leaderboard.js` gains four pure functions, all global in the browser:
  - `liveEntryIsCatchingUp(entry) -> boolean`: `status === 'catching_up'` and the row is printed.
  - `liveCatchingUpLabel(entry) -> string`: `"Catching up · through <snapshot_end>"`, or `"Catching up"`.
  - `liveHasPrintedModel(entries) -> boolean`.
  - `liveCatchingUpBanner(entries) -> string`, which is `''` when no row is catching up.
- `liveSortedEntries()` (existing, `js/live-leaderboard.js`) now orders rows in three tiers in either sort direction: current rows, then catching-up rows, then pending rows. The chosen column orders rows only within a tier.
- New DOM element `#boardEngineNote`: static, in the Competition header only.
- New test file `dashboard/backend/tests/domain/backtesting/test_stop_prices_not_enforced.py`.
- New test helpers:
  - in `test_frontend_live_leaderboard.py`: `_live_view_markup()`, `_static_text(html, element_id)`, `_LIVE_FAKE_DOM`, `_render_catching_up_board(payload, select_id)`, `_catching_up_entry(...)`, `_catching_up_payload(*entries)`, `_SPY_ROW` and `_deepseek(status, *, value, snapshot_end)`;
  - in `test_frontend_live_trading_board.py`: `_STOP_RULE` and `_rules_items(view, formula)`;
  - in Phase 3's `test_frontend_live_budget.py`: `_PRINTED_MODEL` (Task 6.5 Step 1, Edit B).
  - Python silently keeps the later of two same-named module functions, so before appending any of these, grep the target file for the name. If an earlier phase already defined it, add the suffix `_p6` to ours throughout the block instead of redefining.
- Cache-busters: `js/leaderboard.js?v=` +1 (Task 6.2) and `js/live-leaderboard.js?v=` +1 (Task 6.5), each exactly once in this PR.

**Copy this phase ships.** Approving this plan approves exactly these strings. Each one appears verbatim again in its task.

| Where | Before | After |
|---|---|---|
| Competition subtab, Home blurb link, About card heading, Live panel `<h2>`, payload `board_title` | Live Trading Leaderboard | Live Leaderboard |
| Live panel subtitle `#liveBoardSubtitle` (JS used to rewrite it to "`<YYYY-MM>` · paper trading · hourly US cash session") | Paper trading · hourly US cash session | Simulated fills · forward-only · no real money |
| Live season stat `#livePhaseLabel`, payload `phase_label` | Season 0 | Season 0 · preview |
| Live stat label, and its value `#liveUpdatedLabel` | Last Updated, then the request time (e.g. "10/14/2026, 9:12:03 PM") | Updated, then "Nightly after the close" |
| Competition header, new line `#boardEngineNote` | (none) | legacy engine · historical window · may overlap model training data |
| Live Rules sidebar, new last item | (none) | Stop-loss and take-profit prices a model writes are not executed |
| Competition Rules sidebar, new last item | (none) | Stop-loss and take-profit prices a model writes are not executed |
| Live About card, new paragraph after "Each entry is one continuous book…" | (none) | A model may write a stop-loss or take-profit price into its decision. The simulator does not act on either: orders are placed only at decision times, and a position is never sold because the price reached the stop or target the model wrote. |
| Competition About card, end of its paragraph | …Official rank is by final portfolio value. | …Official rank is by final portfolio value. Stop-loss and take-profit prices a model writes are not executed. |
| Live About card, last sentence | An entry more than one trading session behind is shown but not ranked. | An entry more than one trading session behind is marked Catching up: shown, but not ranked. |
| Live table, Entry cell of a catching-up row: new badge (hover title) | (none) | Catching up (title: "Catching up · through 2026-08-20") |
| Live Entry Details, new row (catching-up rows only) | (none) | Status: Catching up · through 2026-08-20 |
| Live banner, one row catching up | "Baselines frozen through …. Model curves appear once the nightly refresh deploys them; they are never computed on page load." (wrongly, whenever every printed model was catching up) | DeepSeek V4 Pro is catching up (through 2026-08-20): it is shown but not ranked until it reaches the latest session. |
| Live banner, several rows catching up | (same wrong line) | 2 entries are catching up (DeepSeek V4 Pro through 2026-08-20, Qwen3.7 Plus through 2026-08-22): they are shown but not ranked until they reach the latest session. |
| Dead `isLiveBoard()` branch of `js/leaderboard.js` (unreachable; pinned by a test) | 'Live Trading Leaderboard' | 'Live Leaderboard' |
| Docstrings and help text: OpenAPI text for `GET /api/v1/leaderboard`, the `POST /api/v1/leaderboard/live/refresh` docstring, and `refresh_live_leaderboard.py`'s module docstring and `--help` description | Live Trading Leaderboard | Live Leaderboard |

Two strings stay as they are on purpose. The Home blurb keeps "Season 0 preview" and "calendar month" (pinned by `test_the_home_blurb_does_not_claim_the_advance_is_running` and `test_live_board_copy_agrees_across_surfaces`), and the Live About card keeps "simulated", "no real capital" and "Season 0" (pinned by `test_the_live_name_is_disclaimed_as_simulated` and `test_about_card_names_the_current_season`).

**Out of this phase** (each is listed again in the PR body or in open issues):
- The landing page (`Race.tsx` plus the committed bundle) and `docs/source/**/*.rst` still say "Live Trading Leaderboard". They are owner-coordinated (Decision 9 in the index plan), and `test_landing_copy_register.py:268,630-631` pins the landing's old name until that PR.
- Comments in `styles.css` (≈:4330, ≈:13851) and `app.js` (≈:11594) still say "Live Trading". Editing them would force bumping the exact `?v=159` pins: `styles.css?v=159` in three test files and `app.js?v=159` in four.
- `SAFE_TRADING_PROMPT` is unchanged. It makes no enforcement claim, and editing it mid-month would silently splice two prompts into one month's curve, because `_RESUME_CONFIG_KEYS` records no prompt hash.
- Spec §4 links the methodology page from the Live Leaderboard subtitle. The subtitle stays plain text here; the methodology-page plan adds the link.

### Task 6.1: One owner for the Live board's name and season label (service.py)

**Files:**
- Modify:
  - `dashboard/backend/domain/leaderboard/service.py`
  - `dashboard/backend/domain/leaderboard/live.py`
  - `dashboard/backend/api/routers/leaderboard.py`
  - `dashboard/backend/api/routers/backtests.py` (a comment only)
  - `dashboard/backend/infrastructure/market_data/us_market_calendar.py` (a docstring only)
  - `dashboard/scripts/refresh_live_leaderboard.py`
- Test:
  - `dashboard/backend/tests/test_leaderboard_api.py`
  - `dashboard/backend/tests/test_leaderboard_season.py`
  - `dashboard/backend/tests/test_live_leaderboard.py`

**Interfaces:**
- Consumes: `service.PREVIEW_SEASON_NUMBER` (existing, `0`) and the existing `lb_service` import in `live.py`. `live.py` already imports `service` as `lb_service`, and `service.py` must never import `live.py`: the module comment at `_PERIOD_BOARDS` warns that the return import is a cycle.
- Produces:
  - `service.LIVE_BOARD_TITLE` and `service.LIVE_PHASE_LABEL`.
  - The payload values `board_title == "Live Leaderboard"` and `phase_label == "Season 0 · preview"`.

- [ ] **Step 1: Write the failing tests**

  Edit the existing assertions:

1. In `dashboard/backend/tests/test_leaderboard_api.py` (≈:794 on main@17c552c5), find:

```python
    assert body["board_title"] == "Live Trading Leaderboard"
```

   Replace with:

```python
    assert body["board_title"] == "Live Leaderboard"
    assert body["phase_label"] == "Season 0 · preview"
```

2. In `dashboard/backend/tests/test_leaderboard_season.py` (≈:257 on main@17c552c5), find:

```python
    live = service.resolve_leaderboard_config("live")
    assert live["phase_label"] == f"Season {service.PREVIEW_SEASON_NUMBER}"
```

   Replace with:

```python
    live = service.resolve_leaderboard_config("live")
    assert service.LIVE_PHASE_LABEL == f"Season {service.PREVIEW_SEASON_NUMBER} · preview"
    assert live["phase_label"] == service.LIVE_PHASE_LABEL
    assert live["board_title"] == service.LIVE_BOARD_TITLE == "Live Leaderboard"
```

3. In `dashboard/backend/tests/test_live_leaderboard.py` (≈:1 on main@17c552c5), find:

```python
"""Tests for the calendar-month Live Trading Leaderboard.
```

   Replace with:

```python
"""Tests for the calendar-month Live Leaderboard.
```

  Then append, after two blank lines, to the end of `dashboard/backend/tests/test_live_leaderboard.py`, after checking that neither function name already exists there:

```python
# ── The board's name and season label have one owner (service.py) ───────────


def test_every_live_surface_reads_the_shared_title_and_phase_label(no_alpaca):
    """Three dicts carry the Live board's name and season label: the payload the
    public GET serves, the freeze config the refresh writes under, and the
    service-side live config. Each used to spell both out, so a rename had three
    places to miss and the season number had three owners."""
    as_of = datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    assert lb_service.LIVE_BOARD_TITLE == "Live Leaderboard"
    assert lb_service.LIVE_PHASE_LABEL == "Season 0 · preview"
    for source in (
        live.get_live_leaderboard(as_of=as_of),
        live.live_freeze_config(as_of),
        lb_service.resolve_leaderboard_config("live"),
    ):
        assert source["board_title"] == lb_service.LIVE_BOARD_TITLE
        assert source["phase_label"] == lb_service.LIVE_PHASE_LABEL


def test_no_backend_text_still_says_live_trading_leaderboard():
    """'Trading' was dropped from the name (direction spec §6): it read as real
    orders and collided with the Robinhood live-trading docs. Docstrings, the
    OpenAPI text and the refresh CLI's --help are read by people too."""
    from pathlib import Path

    backend = Path(live.__file__).resolve().parents[2]
    scripts = backend.parent / "scripts"
    offenders = sorted(
        str(path.relative_to(backend.parent))
        for path in [*backend.rglob("*.py"), *scripts.rglob("*.py")]
        if "tests" not in path.relative_to(backend.parent).parts
        and "Live Trading Leaderboard" in path.read_text(encoding="utf-8")
    )
    assert offenders == [], f"rename these to 'Live Leaderboard': {offenders}"
```

- [ ] **Step 2: Run them and confirm they fail**

```bash
pytest dashboard/backend/tests/test_leaderboard_api.py::test_live_leaderboard_api_serves_the_calendar_month_freeze dashboard/backend/tests/test_leaderboard_season.py::test_the_live_phase_label_is_derived_from_the_season_number dashboard/backend/tests/test_live_leaderboard.py::test_every_live_surface_reads_the_shared_title_and_phase_label dashboard/backend/tests/test_live_leaderboard.py::test_no_backend_text_still_says_live_trading_leaderboard -q -p no:cacheprovider
```

  Expected: `4 failed`.
  - `test_live_leaderboard_api_serves_the_calendar_month_freeze`: `AssertionError: assert 'Live Trading Leaderboard' == 'Live Leaderboard'`.
  - `test_the_live_phase_label_is_derived_from_the_season_number`: `AttributeError: module 'dashboard.backend.domain.leaderboard.service' has no attribute 'LIVE_PHASE_LABEL'`.
  - `test_every_live_surface_reads_the_shared_title_and_phase_label`: `AttributeError: … no attribute 'LIVE_BOARD_TITLE'`.
  - `test_no_backend_text_still_says_live_trading_leaderboard`: `AssertionError: rename these to 'Live Leaderboard': ['backend/api/routers/backtests.py', 'backend/api/routers/leaderboard.py', 'backend/domain/leaderboard/live.py', 'backend/domain/leaderboard/service.py', 'backend/infrastructure/market_data/us_market_calendar.py', 'scripts/refresh_live_leaderboard.py']`. The list can differ: a file an earlier phase already reworded will be missing, and that is fine. A file an earlier phase added with the old name will be extra; rename it in Step 3 as well.

- [ ] **Step 3: Implement**

1. In `dashboard/backend/domain/leaderboard/service.py` (≈:588 on main@17c552c5), find:

```python
PREVIEW_SEASON_NUMBER = 0
DEFAULT_SEASON_TRADING_DAYS = 10
```

   Replace with:

```python
PREVIEW_SEASON_NUMBER = 0
# The Live board's name and season label have exactly one owner: this pair.
# resolve_leaderboard_config("live"), live.live_freeze_config and the payload
# live.get_live_leaderboard serves all read them, and the static fallbacks in
# app.html / js/live-leaderboard.js are pinned to them by
# tests/test_frontend_live_leaderboard.py. "Trading" is dropped from the name
# because it read as real orders and collided with the Robinhood live-trading
# docs (docs/superpowers/specs/2026-10-14-atl-direction.md §6). The label is
# built from PREVIEW_SEASON_NUMBER so the number keeps one owner, and says
# "preview" until a season counts (Season 1 waits for the new engine).
LIVE_BOARD_TITLE = "Live Leaderboard"
LIVE_PHASE_LABEL = f"Season {PREVIEW_SEASON_NUMBER} · preview"
DEFAULT_SEASON_TRADING_DAYS = 10
```

2. In `dashboard/backend/domain/leaderboard/service.py` (≈:733 on main@17c552c5), find:

```python
    """The season block the Live Trading tab renders.
```

   Replace with:

```python
    """The season block the Live tab renders.
```

3. In `dashboard/backend/domain/leaderboard/service.py` (≈:826 on main@17c552c5), find:

```python
            "board_title": "Live Trading Leaderboard",
            # Derived, never spelled out: a literal "Season 0" here is a second
            # owner of the season number, and the first thing the advance engine
            # does is bump PREVIEW_SEASON_NUMBER.
            "phase_label": f"Season {PREVIEW_SEASON_NUMBER}",
```

   Replace with:

```python
            "board_title": LIVE_BOARD_TITLE,
            # Derived, never spelled out: a literal "Season 0" here is a second
            # owner of the season number, and the first thing the advance engine
            # does is bump PREVIEW_SEASON_NUMBER, which LIVE_PHASE_LABEL is built from.
            "phase_label": LIVE_PHASE_LABEL,
```

4. In `dashboard/backend/domain/leaderboard/live.py` (≈:1 on main@17c552c5), find:

```python
"""Calendar-month Live Trading Leaderboard.
```

   Replace with:

```python
"""Calendar-month Live Leaderboard.
```

5. In `dashboard/backend/domain/leaderboard/live.py` (≈:282 on main@17c552c5) (this block occurs exactly 2 times; replace every occurrence, Edit with `replace_all: true`), find:

```python
        "board_title": "Live Trading Leaderboard",
        "phase_label": "Season 0",
```

   Replace with:

```python
        "board_title": lb_service.LIVE_BOARD_TITLE,
        "phase_label": lb_service.LIVE_PHASE_LABEL,
```

6. In `dashboard/backend/api/routers/leaderboard.py` (≈:77 on main@17c552c5), find:

```python
    Pass ?period=live for the current calendar-month Live Trading Leaderboard
```

   Replace with:

```python
    Pass ?period=live for the current calendar-month Live Leaderboard
```

7. In `dashboard/backend/api/routers/leaderboard.py` (≈:184 on main@17c552c5), find:

```python
    """Cron hook: enqueue a Live Trading Leaderboard increment (non-blocking).
```

   Replace with:

```python
    """Cron hook: enqueue a Live Leaderboard increment (non-blocking).
```

8. In `dashboard/backend/api/routers/backtests.py` (≈:3601 on main@17c552c5), find:

```python
# Two weeks, matching the fortnight the Live Trading Leaderboard's seasons use
# so the product has one window vocabulary. It was 31, set when the fixed
```

   Replace with:

```python
# Two weeks, chosen to match the two-week seasons the Live Leaderboard had
# then (seasons are calendar months now). It was 31, set when the fixed
```

9. In `dashboard/backend/infrastructure/market_data/us_market_calendar.py` (≈:3 on main@17c552c5), find:

```python
Shared US market-data infrastructure: the Live Trading Leaderboard walks it
```

   Replace with:

```python
Shared US market-data infrastructure: the Live Leaderboard walks it
```

10. In `dashboard/scripts/refresh_live_leaderboard.py` (≈:2 on main@17c552c5), find:

```python
"""Persist the Live Trading Leaderboard freeze snapshot for the current month.
```

   Replace with:

```python
"""Persist the Live Leaderboard freeze snapshot for the current month.
```

11. In `dashboard/scripts/refresh_live_leaderboard.py` (≈:53 on main@17c552c5), find:

```python
        description="Refresh the Live Trading Leaderboard freeze snapshot"
```

   Replace with:

```python
        description="Refresh the Live Leaderboard freeze snapshot"
```

  For edit 9: if the early-closes phase rewrote that module docstring and the line no longer exists, run `grep -n "Live Trading" dashboard/backend/infrastructure/market_data/us_market_calendar.py` and rename whatever it prints to "Live Leaderboard". For edits 10–11, the roster phase edits the `--models` help line of the same parser; the `description=` line anchored here is a different line. Do **not** run the script to look at `--help`: importing `dashboard.backend` outside pytest mutates the committed seed `backtest.db`.

- [ ] **Step 4: Run them and confirm they pass**

```bash
pytest dashboard/backend/tests/test_leaderboard_api.py dashboard/backend/tests/test_leaderboard_season.py dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/domain/leaderboard/test_service_move.py -q -p no:cacheprovider
```

  Expected: all pass. During planning this was `118 passed` on main@17c552c5 with this task applied; earlier phases add tests to these files, so expect more. `test_service_move.py` pins the payload key set, which this task does not change.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/leaderboard/service.py dashboard/backend/domain/leaderboard/live.py dashboard/backend/api/routers/leaderboard.py dashboard/backend/api/routers/backtests.py dashboard/backend/infrastructure/market_data/us_market_calendar.py dashboard/scripts/refresh_live_leaderboard.py dashboard/backend/tests/test_leaderboard_api.py dashboard/backend/tests/test_leaderboard_season.py dashboard/backend/tests/test_live_leaderboard.py
git commit -m "fix(live): name the board Live Leaderboard from one owner in service.py"
```

### Task 6.2: Rename the tab, make the subtitle and Updated stat static, show "Season 0 · preview"

**Files:**
- Modify:
  - `dashboard/frontend/app.html`
  - `dashboard/frontend/js/live-leaderboard.js`
  - `dashboard/frontend/js/leaderboard.js`
- Test:
  - `dashboard/backend/tests/test_frontend_live_leaderboard.py`
  - `dashboard/backend/tests/test_frontend_live_trading_board.py`
  - `dashboard/backend/tests/test_landing_copy_register.py`
  - `dashboard/backend/tests/test_frontend_fast_boot.py`

**Interfaces:**
- Consumes `service.LIVE_BOARD_TITLE` and `service.LIVE_PHASE_LABEL` from Task 6.1. The new frontend tests compare the static HTML and the JS fallbacks against them, so the server constant stays the single owner.
- Produces:
  - Static copy: `#liveBoardSubtitle` = "Simulated fills · forward-only · no real money"; `#liveUpdatedLabel` = "Nightly after the close" under the label "Updated"; `#livePhaseLabel` = "Season 0 · preview".
  - `updateLiveHeader` no longer writes `#liveBoardSubtitle` or `#liveUpdatedLabel`.
  - `js/leaderboard.js?v=` bumped by one.

- [ ] **Step 1: Write the failing tests**

  Read the current cache-buster first:

```bash
grep -o 'js/leaderboard\.js?v=[0-9]*' dashboard/frontend/app.html dashboard/backend/tests/test_frontend_fast_boot.py
```

  Both lines print the same `v=N`. On bare main@17c552c5, N is 37. After Phases 1–5 have merged, expect 38, because Phase 5 bumped it. Write N → N+1 in edit 9 below **and** in Step 3 edit 11. The code blocks show bare main's 37 → 38, so substitute your N and N+1 in both. If the two lines print different values, stop. An earlier phase left them out of sync, so `test_cache_busters_bumped` is already red on `main`. Report that rather than guessing which value is right. `test_frontend_fast_boot.py::test_cache_busters_bumped` is the only exact pin, so do not add a second cache-buster guard anywhere.

1. In `dashboard/backend/tests/test_frontend_live_leaderboard.py` (≈:1 on main@17c552c5), find:

```python
"""Source guards for the Live Trading Leaderboard tab.
```

   Replace with:

```python
"""Source guards for the Live Leaderboard tab.
```

2. In `dashboard/backend/tests/test_frontend_live_leaderboard.py` (≈:27 on main@17c552c5), find:

```python
        r'data-competition-tab="live"[^>]*>\s*Live Trading Leaderboard',
```

   Replace with:

```python
        r'data-competition-tab="live"[^>]*>\s*Live Leaderboard<',
```

3. In `dashboard/backend/tests/test_frontend_live_trading_board.py` (≈:1 on main@17c552c5), find:

```python
"""Source guards for the Live Trading Leaderboard (frontend/js/leaderboard.js).
```

   Replace with:

```python
"""Source guards for the Live Leaderboard (frontend/js/leaderboard.js).
```

4. In `dashboard/backend/tests/test_frontend_live_trading_board.py` (≈:407 on main@17c552c5), find:

```python
    assert "'Live Trading Leaderboard'" in header, (
```

   Replace with:

```python
    assert "'Live Leaderboard'" in header, (
```

5. In `dashboard/backend/tests/test_frontend_live_trading_board.py` (≈:411 on main@17c552c5), find:

```python
    assert 'data-competition-tab="live">Live Trading Leaderboard<' in _APP_HTML
```

   Replace with:

```python
    assert 'data-competition-tab="live">Live Leaderboard<' in _APP_HTML
    assert "Live Trading Leaderboard" not in _SOURCE, (
        "'Trading' was dropped from the board name (direction spec §6)"
    )
```

6. In `dashboard/backend/tests/test_frontend_live_trading_board.py` (≈:415 on main@17c552c5), find:

```python
    """"Live Trading" is a claim. The About card is where it gets qualified.
```

   Replace with:

```python
    """"Live" is a claim about direction, not money. The About card is where
    it gets qualified.
```

7. In `dashboard/backend/tests/test_frontend_live_trading_board.py` (≈:422 on main@17c552c5) (this block occurs exactly 2 times; replace every occurrence, Edit with `replace_all: true`), find:

```python
    about = _APP_HTML_VISIBLE[_APP_HTML_VISIBLE.index("Live Trading Leaderboard</h3>") :]
```

   Replace with:

```python
    about = _APP_HTML_VISIBLE[_APP_HTML_VISIBLE.index("Live Leaderboard</h3>") :]
```

8. In `dashboard/backend/tests/test_landing_copy_register.py` (≈:807 on main@17c552c5), find:

```python
    about_at = visible.index("Live Trading Leaderboard</h3>")
```

   Replace with:

```python
    about_at = visible.index("Live Leaderboard</h3>")
```

9. In `dashboard/backend/tests/test_frontend_fast_boot.py` (≈:200 on main@17c552c5), find the following line, with N in place of 37:

```python
    assert "js/leaderboard.js?v=37" in APP_HTML
```

   Replace with (N+1; 38 on bare main, 39 after Phase 5):

```python
    assert "js/leaderboard.js?v=38" in APP_HTML
```

  Then append, after two blank lines, to the end of `dashboard/backend/tests/test_frontend_live_leaderboard.py` (grep first for `_live_view_markup` and `_static_text`):

```python
# ── Naming and header copy (direction spec §6) ──────────────────────────────


def _live_view_markup() -> str:
    """The #liveLeaderboardView markup, up to the Competition view."""
    return _APP_HTML.split('id="liveLeaderboardView"')[1].split('id="leaderboardView"')[0]


def _static_text(html: str, element_id: str) -> str:
    """The text inside the element carrying `id="<element_id>"` (no child tags)."""
    match = re.search(rf'id="{element_id}"[^>]*>([^<]*)<', html)
    assert match, f"#{element_id} is missing from app.html"
    return match.group(1)


def test_the_board_is_named_live_leaderboard_on_every_surface():
    """'Trading' read as real orders and collided with the Robinhood
    live-trading docs. The static title, the JS fallback and the payload all
    resolve to service.LIVE_BOARD_TITLE."""
    from dashboard.backend.domain.leaderboard import service

    assert _static_text(_APP_HTML, "liveBoardTitle") == service.LIVE_BOARD_TITLE
    assert f"payload.board_title || '{service.LIVE_BOARD_TITLE}'" in _LIVE_JS
    visible = _strip_js_comments(re.sub(r"<!--.*?-->", "", _APP_HTML, flags=re.DOTALL))
    assert "Live Trading Leaderboard" not in visible
    assert "Live Trading Leaderboard" not in _strip_js_comments(_LIVE_JS)


def test_the_subtitle_is_static_simulated_forward_only_copy():
    assert (
        _static_text(_APP_HTML, "liveBoardSubtitle")
        == "Simulated fills · forward-only · no real money"
    )
    source = _strip_js_comments(_LIVE_JS)
    assert "liveBoardSubtitle" not in source, (
        "the subtitle is static copy in app.html; rewriting it from JS put "
        "'paper trading' back on screen"
    )
    assert "paper trading" not in source.lower()


def test_the_updated_stat_states_the_cadence_not_the_request_clock():
    """payload.updated_at is the wall clock of the GET, so rendering it always
    read 'just now' on a board that last advanced days ago."""
    assert re.search(
        r'<span class="stat-label">Updated</span>\s*'
        r'<span class="stat-value" id="liveUpdatedLabel">Nightly after the close</span>',
        _live_view_markup(),
    )
    source = _strip_js_comments(_LIVE_JS)
    assert "liveUpdatedLabel" not in source
    assert "updated_at" not in source


def test_the_season_stat_reads_season_0_preview_from_one_owner():
    from dashboard.backend.domain.leaderboard import service

    assert service.LIVE_PHASE_LABEL == "Season 0 · preview"
    assert _static_text(_APP_HTML, "livePhaseLabel") == service.LIVE_PHASE_LABEL
    assert f"payload.phase_label || '{service.LIVE_PHASE_LABEL}'" in _LIVE_JS
```

- [ ] **Step 2: Run them and confirm they fail**

```bash
pytest dashboard/backend/tests/test_frontend_live_leaderboard.py dashboard/backend/tests/test_frontend_live_trading_board.py dashboard/backend/tests/test_landing_copy_register.py dashboard/backend/tests/test_frontend_fast_boot.py -q -p no:cacheprovider
```

  Expected: `10 failed`.
  - `test_live_subtab_is_visible_and_daily_stays_parked`: `assert None`.
  - `test_the_board_is_named_live_leaderboard_on_every_surface`: `assert 'Live Trading Leaderboard' == 'Live Leaderboard'`.
  - `test_the_subtitle_is_static_simulated_forward_only_copy`: `assert 'Paper tradin... cash session' == 'Simulated fi...no real money'`.
  - `test_the_updated_stat_states_the_cadence_not_the_request_clock`: `assert None`.
  - `test_the_season_stat_reads_season_0_preview_from_one_owner`: `assert 'Season 0' == 'Season 0 · preview'`.
  - `test_the_board_names_itself_on_screen`: `assert "'Live Leaderboard'" in "function updateLeaderboardHeader(…"`.
  - `test_the_live_name_is_disclaimed_as_simulated`, `test_about_card_names_the_current_season` and `test_live_board_copy_agrees_across_surfaces`: `ValueError: substring not found`.
  - `test_cache_busters_bumped`: `assert 'js/leaderboard.js?v=38' in '<!DOCTYPE html>…'`. You will see your N+1, which is 39 after Phase 5. If this test **passes** here, you wrote N instead of N+1 in edit 9.

- [ ] **Step 3: Implement**

1. In `dashboard/frontend/app.html` (≈:70 on main@17c552c5), find:

```html
            // The Daily Leaderboard became the Live Trading Leaderboard.
```

   Replace with:

```html
            // The Daily Leaderboard became the Live Leaderboard.
```

2. In `dashboard/frontend/app.html` (≈:588 on main@17c552c5), find:

```html
<button class="hm-inline-link" id="homeModuleLiveBtn" type="button">Live Trading Leaderboard</button>
```

   Replace with:

```html
<button class="hm-inline-link" id="homeModuleLiveBtn" type="button">Live Leaderboard</button>
```

3. In `dashboard/frontend/app.html` (≈:1571 on main@17c552c5), find:

```html
<button class="subtab-btn" data-competition-tab="live">Live Trading Leaderboard</button>
```

   Replace with:

```html
<button class="subtab-btn" data-competition-tab="live">Live Leaderboard</button>
```

4. In `dashboard/frontend/app.html` (≈:1591 on main@17c552c5), find:

```html
<h3 class="section-title">Live Trading Leaderboard</h3>
```

   Replace with:

```html
<h3 class="section-title">Live Leaderboard</h3>
```

5. In `dashboard/frontend/app.html` (≈:1641 on main@17c552c5), find:

```html
<!-- Live Trading Leaderboard (calendar month — isolated from contest/daily) -->
```

   Replace with:

```html
<!-- Live Leaderboard (calendar month — isolated from contest/daily) -->
```

6. In `dashboard/frontend/app.html` (≈:1647 on main@17c552c5), find:

```html
<h2 class="contest-title" id="liveBoardTitle">Live Trading Leaderboard</h2>
```

   Replace with:

```html
<h2 class="contest-title" id="liveBoardTitle">Live Leaderboard</h2>
```

7. In `dashboard/frontend/app.html` (≈:1650 on main@17c552c5), find:

```html
<p class="contest-subtitle" id="liveBoardSubtitle">Paper trading · hourly US cash session</p>
```

   Replace with:

```html
<p class="contest-subtitle" id="liveBoardSubtitle">Simulated fills · forward-only · no real money</p>
```

8. In `dashboard/frontend/app.html` (≈:1658 on main@17c552c5), find:

```html
<span class="stat-value" id="livePhaseLabel">Season 0</span>
```

   Replace with:

```html
<span class="stat-value" id="livePhaseLabel">Season 0 · preview</span>
```

9. In `dashboard/frontend/app.html` (≈:1665 on main@17c552c5), find:

```html
                    <span class="stat-label">Last Updated</span>
                    <span class="stat-value" id="liveUpdatedLabel">—</span>
```

   Replace with:

```html
                    <span class="stat-label">Updated</span>
                    <span class="stat-value" id="liveUpdatedLabel">Nightly after the close</span>
```

10. In `dashboard/frontend/app.html` (≈:1774 on main@17c552c5), find:

```html
                         alone it captions the Live Trading Leaderboard as a
```

   Replace with:

```html
                         alone it captions the Live Leaderboard as a
```

11. In `dashboard/frontend/app.html` (≈:2522 on main@17c552c5), find the following tag, with the N from Step 1 in place of 37:

```html
<script src="js/leaderboard.js?v=37" defer></script>
```

   Replace with (N+1, the same number you wrote in edit 9):

```html
<script src="js/leaderboard.js?v=38" defer></script>
```

12. In `dashboard/frontend/js/live-leaderboard.js` (≈:2 on main@17c552c5), find:

```js
// LIVE TRADING LEADERBOARD (calendar-month board — Phase 0)
```

   Replace with:

```js
// LIVE LEADERBOARD (calendar-month board — Season 0 preview)
```

13. In `dashboard/frontend/js/live-leaderboard.js` (≈:162 on main@17c552c5), at the top of `updateLiveHeader`, find the block below. It is Form A and Form B alike: no earlier phase edits these six lines, and the block matches once on main@17c552c5 and once on main@17c552c5 plus Phases 1–5. Find:

```js
  setText('liveBoardTitle', payload.board_title || 'Live Trading Leaderboard');
  setText('livePhaseLabel', payload.phase_label || 'Season 0');
  setText('liveWindowLabel', window.label || '—');
  setText('liveUpdatedLabel', payload.updated_at
    ? new Date(payload.updated_at).toLocaleString()
    : '—');
```

   Replace with:

```js
  setText('liveBoardTitle', payload.board_title || 'Live Leaderboard');
  setText('livePhaseLabel', payload.phase_label || 'Season 0 · preview');
  setText('liveWindowLabel', window.label || '—');
  // The Updated stat is static copy in app.html ("Nightly after the close").
  // payload.updated_at is the request clock, not the last refresh, so showing
  // it read "just now" on a board that last advanced days ago.
```

14. In `dashboard/frontend/js/live-leaderboard.js` (≈:176 on main@17c552c5), find:

```js
  const subtitle = document.getElementById('liveBoardSubtitle');
  if (subtitle) {
    const month = status.month || '';
    subtitle.textContent = month
      ? `${month} · paper trading · hourly US cash session`
      : 'Paper trading · hourly US cash session';
  }
```

   Replace with:

```js
  // #liveBoardSubtitle is static copy in app.html and is not rewritten here:
  // the Window stat already names the month.
```

15. In `dashboard/frontend/js/leaderboard.js` (≈:617 on main@17c552c5), find:

```js
/** True when the Live Trading tab is rendering something that is not a season.
```

   Replace with:

```js
/** True when the Live tab is rendering something that is not a season.
```

16. In `dashboard/frontend/js/leaderboard.js` (≈:699 on main@17c552c5), find:

```js
      ? 'Live Trading Leaderboard'
```

   Replace with:

```js
      ? 'Live Leaderboard'
```

  Edit 11 is the cache-buster. Use the N+1 you chose in Step 1, and confirm `grep -o 'js/leaderboard\.js?v=[0-9]*' dashboard/frontend/app.html dashboard/backend/tests/test_frontend_fast_boot.py` now prints N+1 on both lines. Do **not** bump `js/live-leaderboard.js` here: Task 6.5 bumps it, so this PR bumps it exactly once. Do not touch `styles.css` or `app.js`.

  Edit 13 has a single form (see the top of this phase). If its block does not match exactly once, stop and report it. Rule for this edit: nothing in `js/live-leaderboard.js` may read `updated_at`, `liveUpdatedLabel` or `liveBoardSubtitle`, and the new tests enforce it.

```bash
node --check dashboard/frontend/js/live-leaderboard.js
node --check dashboard/frontend/js/leaderboard.js
```

- [ ] **Step 4: Run them and confirm they pass**

```bash
pytest dashboard/backend/tests/test_frontend_live_leaderboard.py dashboard/backend/tests/test_frontend_live_trading_board.py dashboard/backend/tests/test_landing_copy_register.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_frontend_bundle_integrity.py dashboard/backend/tests/test_app_copy_register.py -q -p no:cacheprovider
```

  Expected: all pass. During planning this was `135 passed` on main@17c552c5 plus Tasks 6.1–6.2.

- [ ] **Step 5: Commit**

```bash
git add dashboard/frontend/app.html dashboard/frontend/js/live-leaderboard.js dashboard/frontend/js/leaderboard.js dashboard/backend/tests/test_frontend_live_leaderboard.py dashboard/backend/tests/test_frontend_live_trading_board.py dashboard/backend/tests/test_landing_copy_register.py dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "fix(live): rename the tab, static subtitle and Updated stat, Season 0 preview label"
```

### Task 6.3: Label the Competition board "legacy engine · historical window · may overlap model training data"

**Files:**
- Modify: `dashboard/frontend/app.html`
- Test: `dashboard/backend/tests/test_frontend_live_trading_board.py`

**Interfaces:**
- Consumes nothing new.
- Produces static `<p class="contest-subtitle" id="boardEngineNote">` inside `#leaderboardView`'s `.contest-identity`. No JS writes it, and only the existing `.contest-subtitle` class styles it, so the change needs no CSS edit and no cache-buster bump.

- [ ] **Step 1: Write the failing test**

  Append, after two blank lines, to the end of `dashboard/backend/tests/test_frontend_live_trading_board.py`:

```python
# ── The Competition board is labelled as legacy (direction spec §6) ─────────


def test_the_competition_header_names_the_legacy_engine_and_window():
    """The Competition board stays on the legacy engine and is retired, not
    re-run, once Season 1 has data; its fixed historical window may sit inside
    a model's training data. The header says both. Static copy: leaderboard.js
    rewrites #boardSubtitle on every render and must leave this line alone."""
    competition = _APP_HTML_VISIBLE.split('id="leaderboardView"')[1]
    identity = competition[: competition.index('class="leaderboard-header-bar compact"')]
    assert (
        '<p class="contest-subtitle" id="boardEngineNote">'
        "legacy engine · historical window · may overlap model training data</p>"
    ) in identity
    live = _APP_HTML_VISIBLE.split('id="liveLeaderboardView"')[1].split('id="leaderboardView"')[0]
    assert "legacy engine" not in live, "the Live board is not labelled legacy"
    assert "boardEngineNote" not in _SOURCE, "leaderboard.js must not rewrite the note"
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
pytest dashboard/backend/tests/test_frontend_live_trading_board.py::test_the_competition_header_names_the_legacy_engine_and_window -q -p no:cacheprovider
```

  Expected: `1 failed` with `assert '<p class="contest-subtitle" id="boardEngineNote">legacy engine · historical window · may overlap model training data</p>' in ' class="leaderboard-view competition-panel competition-leaderboard-panel" style="display: none;">…'`.

- [ ] **Step 3: Implement**

1. In `dashboard/frontend/app.html` (≈:1770 on main@17c552c5), find:

```html
                    <p class="contest-subtitle" id="boardSubtitle">Sep 1 – Oct 30, 2026</p>
```

   Replace with:

```html
                    <p class="contest-subtitle" id="boardSubtitle">Sep 1 – Oct 30, 2026</p>
                    <!-- Static, and Competition-only: leaderboard.js rewrites
                         #boardSubtitle on every render but never this line.
                         Direction spec 2026-10-14 §6: this board stays on the
                         legacy engine and is retired, not re-run, once Season 1
                         has data. -->
                    <p class="contest-subtitle" id="boardEngineNote">legacy engine · historical window · may overlap model training data</p>
```

  `updateLeaderboardHeader` rewrites `#boardSubtitle` on every render and hides `#contestOrganizerLine` on the live branch. That branch is unreachable today, because app.js only ever calls `loadLeaderboardData('contest')`, so the new line needs no hiding code. Do not add any.

- [ ] **Step 4: Run it and confirm it passes**

```bash
pytest dashboard/backend/tests/test_frontend_live_trading_board.py dashboard/backend/tests/test_frontend_live_leaderboard.py dashboard/backend/tests/test_app_copy_register.py -q -p no:cacheprovider
```

  Expected: all pass. Measured: `84 passed` on main@17c552c5 plus Tasks 6.1–6.3. Earlier phases add tests to these files, so expect more.

- [ ] **Step 5: Commit**

```bash
git add dashboard/frontend/app.html dashboard/backend/tests/test_frontend_live_trading_board.py
git commit -m "feat(leaderboard): label the Competition board as legacy engine"
```

### Task 6.4: Say stop-loss and take-profit are not executed, and pin that the legacy path ignores them

**Files:**
- Modify: `dashboard/frontend/app.html`. This is HTML copy only: no prompt, validator, live.py or leaderboard.json change.
- Create (test): `dashboard/backend/tests/domain/backtesting/test_stop_prices_not_enforced.py`
- Test: `dashboard/backend/tests/test_frontend_live_trading_board.py`, `dashboard/backend/tests/test_frontend_live_leaderboard.py`

**Interfaces:**
- Consumes:
  - `PortfolioManager.make_trading_decision_with_llm`;
  - `PortfolioManager.execute_actions`;
  - the module-level seams `portfolio_manager._request_trading_decision`, `_extract_response_text` and `_extract_token_usage`, the same three that `tests/domain/backtesting/test_escalate_ceiling.py` patches.
- Produces:
  - four disclaimer strings in `app.html`;
  - a test file that fails the day the legacy order path reads `stop_loss`/`take_profit`.

  These facts were verified on main@17c552c5:
  - The Live and Competition LLM entries all run `mode: "safe_trading"`, with no `strategy_prompt` and no pipeline.
  - The single-prompt loop copies only `symbol`, `action`, `confidence`, `reasoning` and `position_size` from each model action (`portfolio_manager.py` ≈:833-945).
  - `SAFE_TRADING_PROMPT` makes no enforcement claim; it only shows the two fields as `null` in its JSON shape (`validator.py` ≈:663-664).
  - The false "enforced separately in code" claim is in `infrastructure/llm/prompts.py`, the Trading Algo prompt. That prompt is out of scope here (see open issues).

- [ ] **Step 1: Write the failing tests**

  Create `dashboard/backend/tests/domain/backtesting/test_stop_prices_not_enforced.py`:

```python
"""The legacy engine does not execute stop-loss or take-profit prices (D12).

The safe-trading prompt's JSON shape carries ``stop_loss_price`` and
``take_profit_price`` (infrastructure/llm/validator.py), so a model can write
them, but the single-prompt decision loop that both boards' LLM entries run
(``PortfolioManager.make_trading_decision_with_llm``, reached from
``LLMAgentStrategy``) copies only symbol, action, size, confidence and reasoning
out of each reply. Both boards say so in their Rules and About cards
(dashboard/frontend/app.html). These tests are the other half of that copy: the
day this path starts reading the fields, they fail, and the copy changes in the
same PR. The new core executes stops for real (direction spec 2026-10-14 §4);
when it lands, invert these tests and delete the four disclaimers together.
"""

import json
from datetime import datetime
from pathlib import Path

from dashboard.backend.domain.backtesting import portfolio_manager as pm
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager

_BACKEND = Path(__file__).resolve().parents[3]
_STOP_FIELDS = {"stop_loss_price": 240.0, "take_profit_price": 300.0}


class _Resp:
    def __init__(self, text):
        self._text = text
        self.usage = None
        self.content = []


def _script(monkeypatch, replies):
    """Answer each decision request with the next reply, in order."""
    queue = [json.dumps(reply) for reply in replies]

    def fake_request(client, **_kwargs):
        return _Resp(queue.pop(0))

    monkeypatch.setattr(pm, "_request_trading_decision", fake_request)
    monkeypatch.setattr(pm, "_extract_response_text", lambda response: response._text)
    monkeypatch.setattr(pm, "_extract_token_usage", lambda response: (100, 50))
    return queue


def _state(manager, price, timestamp):
    held = manager.positions.get("AAPL", 0)
    positions = []
    if held:
        entry = manager.entry_prices.get("AAPL", price)
        positions.append({
            "symbol": "AAPL",
            "shares": held,
            "entry_price": entry,
            "current_price": price,
            "position_value": held * price,
            "pnl_pct": (price / entry - 1.0) * 100.0,
        })
    return {
        "timestamp": timestamp,
        "cash": manager.cash,
        "positions": positions,
        "positions_value": held * price,
        "total_equity": manager.cash + held * price,
        "market_signals": {
            "AAPL": {"price": price, "rsi": 50.0, "macd": 0.0,
                     "macd_signal": 0.0, "sma_20": price, "sma_50": price,
                     "bb_upper": price * 1.04, "bb_lower": price * 0.96,
                     "volume": 1000},
        },
    }


def _reply(action, **fields):
    return {"actions": [{
        "symbol": "AAPL",
        "action": action,
        "confidence": 0.8,
        "reasoning": "scripted",
        **_STOP_FIELDS,
        **fields,
    }]}


def _decide(manager, price, timestamp):
    return manager.make_trading_decision_with_llm(
        _state(manager, price, timestamp), llm_client=object(), model="m"
    )


def test_stop_and_take_profit_fields_never_reach_the_executed_action(monkeypatch):
    _script(monkeypatch, [_reply("buy", position_size=2)])
    manager = PortfolioManager(initial_capital=1000.0, allowed_symbols=["AAPL"])

    decision = _decide(manager, 250.0, datetime(2026, 4, 1, 14, 30))

    assert manager.llm_decisions == 1, "the model drove this step"
    assert len(decision["actions"]) == 1
    action = decision["actions"][0]
    assert set(action) == {"symbol", "action", "shares", "reason", "confidence"}
    assert (action["symbol"], action["action"], action["shares"]) == ("AAPL", "buy", 2)


def test_a_held_position_is_not_sold_when_the_price_crosses_its_stop_or_target(monkeypatch):
    remaining = _script(monkeypatch, [
        _reply("buy", position_size=2),
        _reply("hold"),
        _reply("hold"),
    ])
    manager = PortfolioManager(initial_capital=1000.0, allowed_symbols=["AAPL"])
    opened = datetime(2026, 4, 1, 14, 30)
    entry = _decide(manager, 250.0, opened)
    manager.execute_actions(entry["actions"], {"AAPL": {"close": 250.0}}, opened)
    assert manager.positions == {"AAPL": 2}

    # 200 is below the 240 stop; 320 is above the 300 target. Neither sells.
    for hour, price in ((15, 200.0), (16, 320.0)):
        timestamp = datetime(2026, 4, 1, hour, 30)
        decision = _decide(manager, price, timestamp)
        manager.execute_actions(decision["actions"], {"AAPL": {"close": price}}, timestamp)
        assert decision["actions"] == []
        assert manager.positions == {"AAPL": 2}

    assert remaining == []
    assert [trade["side"] for trade in manager.trades] == ["BUY"]
    assert manager.llm_decisions == 3


def test_the_legacy_order_path_has_no_stop_handling():
    """A source guard, because a stop could be wired in a branch the two cases
    above never reach. Lifting it is a decision, not a refactor: update the
    "not executed" copy in app.html (Live and Competition Rules + About cards)
    in the same PR."""
    for rel in (
        "domain/backtesting/portfolio_manager.py",
        "domain/trading/execution.py",
        "domain/leaderboard/strategies/llm_agent.py",
    ):
        text = (_BACKEND / rel).read_text(encoding="utf-8")
        for field in ("stop_loss", "take_profit"):
            assert field not in text, (
                f"{rel} now mentions {field}: if stops execute on the legacy "
                "engine, the 'not executed' disclaimers in app.html are false"
            )
```

  Append, after two blank lines, to the end of `dashboard/backend/tests/test_frontend_live_trading_board.py`:

```python
# ── Stop-loss / take-profit are not executed on the legacy engine (D12) ─────
#
# The other half of these is tests/domain/backtesting/test_stop_prices_not_enforced.py,
# which fails the day the legacy order path starts reading the two fields.

_STOP_RULE = "<li>Stop-loss and take-profit prices a model writes are not executed</li>"


def _rules_items(view: str, formula: str) -> str:
    rules = view[view.index(formula) :]
    return rules[: rules.index("</ul>")]


def test_both_boards_rules_say_stop_prices_are_not_executed():
    live = _APP_HTML_VISIBLE.split('id="liveLeaderboardView"')[1].split('id="leaderboardView"')[0]
    competition = _APP_HTML_VISIBLE.split('id="leaderboardView"')[1]
    assert _STOP_RULE in _rules_items(live, "Official rank = latest frozen portfolio value")
    assert _STOP_RULE in _rules_items(competition, "Official rank = final portfolio value")


def test_both_about_cards_say_stop_prices_are_not_executed():
    live = _APP_HTML_VISIBLE[_APP_HTML_VISIBLE.index("Live Leaderboard</h3>") :]
    live = live[: live.index("</div>")]
    assert (
        "A model may write a stop-loss or take-profit price into its decision. "
        "The simulator does not act on either: orders are placed only at decision "
        "times, and a position is never sold because the price reached the stop "
        "or target the model wrote."
    ) in live
    marker = '<h3 class="section-title">Competition Leaderboard</h3>'
    competition = _APP_HTML_VISIBLE[_APP_HTML_VISIBLE.index(marker) :]
    competition = competition[: competition.index("</div>")]
    assert (
        "Official rank is by final portfolio value. "
        "Stop-loss and take-profit prices a model writes are not executed."
    ) in competition
```

  Append, after two blank lines, to the end of `dashboard/backend/tests/test_frontend_live_leaderboard.py`:

```python
def test_neither_board_renders_a_stop_or_take_profit_price():
    """Both boards disclaim stops as not executed. A future trade-log or
    reasoning view is the first place a model's stop_loss_price could appear
    next to its decisions; it must label the value 'not executed' before this
    guard is lifted, or the page contradicts its own Rules card."""
    leaderboard_js = (_FRONTEND / "js" / "leaderboard.js").read_text(encoding="utf-8")
    for name, source in (
        ("js/live-leaderboard.js", _LIVE_JS),
        ("js/leaderboard.js", leaderboard_js),
    ):
        stripped = _strip_js_comments(source)
        assert "stop_loss" not in stripped, name
        assert "take_profit" not in stripped, name
```

- [ ] **Step 2: Run them and confirm the copy tests fail**

```bash
pytest dashboard/backend/tests/domain/backtesting/test_stop_prices_not_enforced.py dashboard/backend/tests/test_frontend_live_trading_board.py dashboard/backend/tests/test_frontend_live_leaderboard.py -q -p no:cacheprovider
```

  Expected: `2 failed`. Measured: `2 failed, 49 passed` on main@17c552c5 plus Tasks 6.1–6.3.
  - `test_both_boards_rules_say_stop_prices_are_not_executed`: `assert '<li>Stop-loss and take-profit prices a model writes are not executed</li>' in 'Official rank = latest frozen portfolio value</p>…<li>Entries with no frozen session are unranked</li>…'`.
  - `test_both_about_cards_say_stop_prices_are_not_executed`: `assert 'A model may write a stop-loss or take-profit price into its decision. …' in 'Live Leaderboard</h3>…'`, because the sentence is not in the card.

  The three tests in `test_stop_prices_not_enforced.py` and `test_neither_board_renders_a_stop_or_take_profit_price` **pass already**, and that is intended. They are characterization tests: they pin current behaviour as the tripwire for the copy, and they are not the change. They were mutation-checked during planning. Converting a HOLD into a SELL when the price is below `stop_loss_price` in `portfolio_manager.py` makes `test_a_held_position_is_not_sold_when_the_price_crosses_its_stop_or_target` and `test_the_legacy_order_path_has_no_stop_handling` fail.

- [ ] **Step 3: Implement**

1. In `dashboard/frontend/app.html` (≈:1745 on main@17c552c5), find:

```html
                                <li>Entries with no frozen session are unranked</li>
                            </ul>
```

   Replace with:

```html
                                <li>Entries with no frozen session are unranked</li>
                                <!-- D12: the legacy engine parses these two prices and drops
                                     them. Delete this line when the new core executes stops
                                     (spec 2026-10-14 §4); pinned by
                                     tests/domain/backtesting/test_stop_prices_not_enforced.py. -->
                                <li>Stop-loss and take-profit prices a model writes are not executed</li>
                            </ul>
```

2. In `dashboard/frontend/app.html` (≈:1886 on main@17c552c5), find:

```html
                                <li>Click column headers to re-sort</li>
                            </ul>
```

   Replace with:

```html
                                <li>Click column headers to re-sort</li>
                                <!-- D12: the legacy engine parses these two prices and drops
                                     them. Delete this line when the new core executes stops
                                     (spec 2026-10-14 §4); pinned by
                                     tests/domain/backtesting/test_stop_prices_not_enforced.py. -->
                                <li>Stop-loss and take-profit prices a model writes are not executed</li>
                            </ul>
```

3. In `dashboard/frontend/app.html` (≈:1595 on main@17c552c5), find this whole line. It is the Live About card's Season 0 paragraph, indented 20 spaces:

```html
                    <p class="about-text about-text-muted">We are in <strong>Season 0</strong>, the shakedown season, so read the standings as a preview rather than an official result. An entry more than one trading session behind is shown but not ranked.</p>
```

   Replace it with these two lines. The first is new, and the second is the original line, unchanged. Both carry the same 20-space indent:

```html
                    <p class="about-text">A model may write a stop-loss or take-profit price into its decision. The simulator does not act on either: orders are placed only at decision times, and a position is never sold because the price reached the stop or target the model wrote.</p>
                    <p class="about-text about-text-muted">We are in <strong>Season 0</strong>, the shakedown season, so read the standings as a preview rather than an official result. An entry more than one trading session behind is shown but not ranked.</p>
```

   The old string and the new string both start at the line's first space and end at its `</p>`. Add no trailing newline to either. With the Edit tool, paste them exactly as shown.

4. In `dashboard/frontend/app.html` (≈:1599 on main@17c552c5), find:

```html
Official rank is by final portfolio value.</p>
```

   Replace with:

```html
Official rank is by final portfolio value. Stop-loss and take-profit prices a model writes are not executed.</p>
```

  Edit 3 inserts the new paragraph **before** the Season 0 paragraph and leaves that paragraph untouched. Task 6.5 edit 7 rewrites its last sentence. No other phase edits that line. Edit 4 appends to the existing Competition sentence, and `test_app_copy_register.py` still finds "comparing leading AI models, baseline strategies, and market indices." inside it. All the copy is engine-neutral ("the simulator") and says nothing about fills, fees or slippage.

- [ ] **Step 4: Run them and confirm they pass**

```bash
pytest dashboard/backend/tests/domain/backtesting/test_stop_prices_not_enforced.py dashboard/backend/tests/domain/backtesting/test_escalate_ceiling.py dashboard/backend/tests/test_frontend_live_trading_board.py dashboard/backend/tests/test_frontend_live_leaderboard.py dashboard/backend/tests/test_app_copy_register.py dashboard/backend/tests/test_landing_copy_register.py -q -p no:cacheprovider
```

  Expected: all pass. Measured: `139 passed` on main@17c552c5 plus Tasks 6.1–6.4. Earlier phases add tests to these files, so expect more. Then check the indentation of the new line with `grep -c '^                    <p class="about-text">A model may write a stop-loss' dashboard/frontend/app.html`, which has 20 spaces after the `^`. It must print `1`.

- [ ] **Step 5: Commit**

```bash
git add dashboard/frontend/app.html dashboard/backend/tests/domain/backtesting/test_stop_prices_not_enforced.py dashboard/backend/tests/test_frontend_live_trading_board.py dashboard/backend/tests/test_frontend_live_leaderboard.py
git commit -m "fix(leaderboard): say stop-loss and take-profit prices are not executed"
```

### Task 6.5: Flag catching-up rows and stop calling their curves missing (#586)

**Files:**
- Modify: `dashboard/frontend/js/live-leaderboard.js`, `dashboard/frontend/app.html`
- Test:
  - `dashboard/backend/tests/test_frontend_live_leaderboard.py`
  - `dashboard/backend/tests/test_frontend_live_trading_board.py`
  - `dashboard/backend/tests/test_frontend_live_budget.py` (Phase 3's file; fixture only)

**Interfaces:**
- Consumes Phase 3's `test_frontend_live_budget.py::_banner_after(status)` and `_HEALTHY`. This task changes the banner's input, so it updates that fixture to match. Their tests are unchanged.
- Consumes the existing payload fields `entries[].status` (`frozen | catching_up | pending`), `entries[].snapshot_end`, `entries[].portfolio_value` and `live_status.snapshot_stale`/`frozen_through`/`has_prints`. They are set by `live._entry_from_strategy` and `get_live_leaderboard`, and their server contract is pinned by `test_live_leaderboard.py::test_an_entry_several_sessions_behind_is_shown_but_not_ranked`. The server is not changed: `live_status.models_cached`/`models_pending` keep counting ranked rows only, and `test_live_leaderboard.py:438-439` still pins them.
- Produces `liveEntryIsCatchingUp`, `liveCatchingUpLabel`, `liveHasPrintedModel` and `liveCatchingUpBanner`. `liveSortedEntries` now sorts catching-up rows in their own tier, between current and pending rows, in both directions. `js/live-leaderboard.js?v=` is bumped by one.

  The issue text is partly stale. Commit 1a7b550d already made a catching-up row print (the table has no rank column, and `liveEntryIsPending` keys on `status`/`portfolio_value`). What #586 still lacks:
  - no indication that the row is behind;
  - the banner keys on `live_status.models_cached`, which counts only ranked rows. When every printed model is catching up, the banner says "Model curves appear once the nightly refresh deploys them" while those curves are on screen;
  - sorting a column slots a catching-up row in among the current rows, although the server, the banner and the About card all say it is shown but not ranked.

- [ ] **Step 1: Write the failing tests**

  These run the **whole** of `js/live-leaderboard.js` under node against a fake DOM, through `_frontend_source.run_node`, with `escapeHtml` lifted from app.js by `fn_body`. Any helper an earlier phase added to the render path therefore runs too. If the harness throws a `TypeError` on a payload field another phase made required, add that field to `_catching_up_payload`/`_catching_up_entry` with the value the server sends for it, and do not stub the function out. Grep the file for each helper name first; the naming rule is under Interfaces above.

  Append, after two blank lines, to the end of `dashboard/backend/tests/test_frontend_live_leaderboard.py`:

```python
# ── #586: catching-up rows are printed, flagged, and not reported missing ───
#
# These evaluate the whole of js/live-leaderboard.js under node against a fake
# DOM instead of lifting single functions, so a helper another change adds to
# the render path is exercised too rather than raising a ReferenceError here.

_LIVE_FAKE_DOM = """
const els = {};
function el(id) {
  if (!els[id]) {
    els[id] = {
      id, textContent: '', innerHTML: '', hidden: false, dataset: {}, style: {},
      classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
      setAttribute() {}, querySelector() { return null; },
    };
  }
  return els[id];
}
globalThis.window = globalThis;
globalThis.document = {
  getElementById: el,
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener() {},
};
globalThis.API_BASE = '';
"""


def _render_catching_up_board(payload, select_id):
    """Render `payload` the way loadLiveLeaderboardData does; return the banner,
    the table body and the detail panel for the row `select_id`."""
    from dashboard.backend.tests._frontend_source import APP_JS, fn_body, run_node

    script = (
        _LIVE_FAKE_DOM
        + fn_body("function escapeHtml(", APP_JS)
        + "\n"
        + _LIVE_JS
        + f"\nconst payload = {json.dumps(payload)};\n"
        + "livePayload = payload;\n"
        + "updateLiveHeader(payload);\n"
        + "populateLiveTable();\n"
        + f"renderLiveDetail(payload.entries.find((e) => e.entry_id === {json.dumps(select_id)}));\n"
        + "process.stdout.write(JSON.stringify({\n"
        + "  banner: els.livePhaseBanner.textContent,\n"
        + "  bannerHidden: els.livePhaseBanner.hidden,\n"
        + "  table: els.liveLeaderboardTableBody.innerHTML,\n"
        + "  detail: els.liveSelectedDetail.innerHTML,\n"
        + "}));\n"
    )
    return run_node(script)


def _catching_up_entry(entry_id, name, status, *, value, snapshot_end, is_model=True):
    """One board row in the shape live._entry_from_strategy serves."""
    printed = value is not None
    return {
        "entry_id": entry_id,
        "model": name,
        "team_name": name,
        "team_badge": "Model" if is_model else "Index",
        "is_model": is_model,
        "status": status,
        "rank": None,
        "portfolio_value": value,
        "cumulative_return": 0.05 if printed else None,
        "sharpe_ratio": None,
        "max_drawdown": None,
        "num_trades": 3 if printed else None,
        "snapshot_end": snapshot_end,
        "equity_curve": (
            [{"timestamp": "2026-08-03T10:00:00-04:00", "equity": 100000}] if printed else []
        ),
    }


def _catching_up_payload(*entries):
    return {
        "period": "live",
        "board_title": "Live Leaderboard",
        "phase_label": "Season 0 · preview",
        "window": {"label": "2026-08-01 — 2026-08-31"},
        "entries": list(entries),
        "live_status": {
            "has_prints": True,
            "snapshot_stale": False,
            "frozen_through": "2026-08-27",
            "snapshot_end": "2026-08-27",
            "session_state": "closed",
            # The server counts only ranked (frozen) rows here, so a model
            # mid-replay reads 0. Held at 0 in every case: the banner must not
            # depend on it.
            "models_cached": 0,
            "catching_up": [e["entry_id"] for e in entries if e["status"] == "catching_up"],
            "trading_days_elapsed": 19,
            "trading_days_total": 21,
        },
    }


_SPY_ROW = _catching_up_entry(
    "spy_index", "SPY", "frozen", value=101000.0, snapshot_end="2026-08-27", is_model=False
)


def _deepseek(status, *, value=105000.0, snapshot_end="2026-08-20"):
    return _catching_up_entry(
        "deepseek_v4_pro", "DeepSeek V4 Pro", status, value=value, snapshot_end=snapshot_end
    )


@_needs_node
def test_a_catching_up_model_is_printed_and_flagged_with_its_own_date():
    out = _render_catching_up_board(
        _catching_up_payload(_SPY_ROW, _deepseek("catching_up")), "deepseek_v4_pro"
    )
    row = out["table"].split('data-live-entry="deepseek_v4_pro"')[1].split("</tr>")[0]
    assert (
        '<span class="team-badge" title="Catching up · through 2026-08-20">Catching up</span>'
    ) in row
    assert "5.00%" in row, "a catching-up row prints its figures"
    assert (
        '<span class="team-detail-label">Status</span>'
        '<span class="team-detail-value">Catching up · through 2026-08-20</span>'
    ) in out["detail"]
    assert "$105,000.00" in out["detail"]
    assert "Awaiting freeze" not in out["detail"]


@_needs_node
def test_the_banner_names_the_catching_up_model_instead_of_calling_it_missing():
    out = _render_catching_up_board(
        _catching_up_payload(_SPY_ROW, _deepseek("catching_up")), "deepseek_v4_pro"
    )
    assert out["bannerHidden"] is False
    assert "Model curves appear" not in out["banner"]
    assert out["banner"] == (
        "DeepSeek V4 Pro is catching up (through 2026-08-20): it is shown but "
        "not ranked until it reaches the latest session."
    )


@_needs_node
def test_the_banner_lists_every_catching_up_entry():
    qwen = _catching_up_entry(
        "qwen3_7_plus", "Qwen3.7 Plus", "catching_up", value=99000.0, snapshot_end="2026-08-22"
    )
    out = _render_catching_up_board(
        _catching_up_payload(_SPY_ROW, _deepseek("catching_up"), qwen), "qwen3_7_plus"
    )
    assert out["banner"] == (
        "2 entries are catching up (DeepSeek V4 Pro through 2026-08-20, "
        "Qwen3.7 Plus through 2026-08-22): they are shown but not ranked "
        "until they reach the latest session."
    )


@_needs_node
def test_a_model_with_no_curve_still_gets_the_models_pending_banner():
    out = _render_catching_up_board(
        _catching_up_payload(_SPY_ROW, _deepseek("pending", value=None, snapshot_end=None)),
        "deepseek_v4_pro",
    )
    assert out["banner"] == (
        "Baselines frozen through 2026-08-27. Model curves appear once the nightly "
        "refresh deploys them; they are never computed on page load."
    )
    assert "Catching up" not in out["table"]
    assert "Awaiting freeze" in out["detail"]
    assert "Status</span>" not in out["detail"]


@_needs_node
def test_a_catching_up_row_with_no_value_reads_as_pending_not_catching_up():
    """A row the server marks catching_up but sends no value for cannot print:
    it is pending, and a 'Catching up' badge beside 'Awaiting freeze' would say
    both at once."""
    out = _render_catching_up_board(
        _catching_up_payload(_SPY_ROW, _deepseek("catching_up", value=None)),
        "deepseek_v4_pro",
    )
    assert "Catching up" not in out["table"]
    assert "Status</span>" not in out["detail"]
    assert "Awaiting freeze" in out["detail"]
    assert out["banner"].startswith("Baselines frozen through 2026-08-27.")


@_needs_node
def test_a_stale_snapshot_banner_outranks_the_catching_up_line_and_the_badge_stays():
    payload = _catching_up_payload(_SPY_ROW, _deepseek("catching_up"))
    payload["live_status"]["snapshot_stale"] = True
    payload["live_status"]["frozen_through"] = "2026-08-28"
    out = _render_catching_up_board(payload, "deepseek_v4_pro")
    assert out["banner"].startswith("Showing the snapshot through 2026-08-27.")
    assert ">Catching up</span>" in out["table"]


@_needs_node
def test_a_current_board_shows_no_banner_and_no_badge():
    out = _render_catching_up_board(
        _catching_up_payload(_SPY_ROW, _deepseek("frozen", snapshot_end="2026-08-27")),
        "deepseek_v4_pro",
    )
    assert out["bannerHidden"] is True
    assert out["banner"] == ""
    assert "Catching up" not in out["table"]
    assert "Catching up" not in out["detail"]


@_needs_node
def test_catching_up_rows_sort_after_current_rows_and_before_pending_ones():
    """A catching-up row is shown but not ranked, and its figures are from an
    earlier session than the rows beside it. Sorting a column must not slot it
    in among the current rows, in either direction: DeepSeek's Sharpe would
    sort first descending and Qwen's first ascending."""
    from dashboard.backend.tests._frontend_source import run_node

    rows = [
        _catching_up_entry(
            "claude_haiku_4_5", "Claude Haiku 4.5", "pending", value=None, snapshot_end=None
        ),
        {**_deepseek("catching_up"), "sharpe_ratio": 9.0},
        {**_SPY_ROW, "sharpe_ratio": 1.0},
        {
            **_catching_up_entry(
                "qwen3_7_plus", "Qwen3.7 Plus", "catching_up",
                value=99000.0, snapshot_end="2026-08-22",
            ),
            "sharpe_ratio": -5.0,
        },
        {
            **_catching_up_entry(
                "equal_weight_djia", "Equal Weight DJIA", "frozen",
                value=100800.0, snapshot_end="2026-08-27", is_model=False,
            ),
            "sharpe_ratio": 0.5,
        },
    ]
    script = (
        _LIVE_FAKE_DOM
        + _LIVE_JS
        + f"\nlivePayload = {{ entries: {json.dumps(rows)} }};\n"
        + "liveSortKey = 'sharpe';\n"
        + "liveSortDir = 'desc'; const desc = liveSortedEntries().map((e) => e.entry_id);\n"
        + "liveSortDir = 'asc'; const asc = liveSortedEntries().map((e) => e.entry_id);\n"
        + "process.stdout.write(JSON.stringify({ desc, asc }));\n"
    )
    assert run_node(script) == {
        "desc": [
            "spy_index", "equal_weight_djia",
            "deepseek_v4_pro", "qwen3_7_plus",
            "claude_haiku_4_5",
        ],
        "asc": [
            "equal_weight_djia", "spy_index",
            "qwen3_7_plus", "deepseek_v4_pro",
            "claude_haiku_4_5",
        ],
    }


def test_the_banner_no_longer_reads_models_cached():
    assert "models_cached" not in _strip_js_comments(_LIVE_JS)
```

  Append, after two blank lines, to the end of `dashboard/backend/tests/test_frontend_live_trading_board.py`:

```python
def test_the_about_card_explains_the_catching_up_label():
    about = _APP_HTML_VISIBLE[_APP_HTML_VISIBLE.index("Live Leaderboard</h3>") :]
    about = about[: about.index("</div>")]
    assert (
        "An entry more than one trading session behind is marked Catching up: "
        "shown, but not ranked."
    ) in about
```

  Then update Phase 3's healthy-board fixture. Step 3 makes the banner read `payload.entries` instead of `live_status.models_cached`. Phase 3's `_banner_after` sends `entries: []`, so after Step 3 its "healthy" board would show "Baselines frozen through …" and five of its tests would fail: `test_a_budget_halt_shows_a_banner_line_on_a_healthy_board` and the four `test_no_budget_line_without_a_real_true` cases. The fix gives that board one printed, current model row, which is what a real payload with models cached carries. It passes both before and after Step 3; both states were checked on main@17c552c5 plus Phases 1, 2, 3 and 5. Skip both edits if `dashboard/backend/tests/test_frontend_live_budget.py` does not exist, or if `_banner_after` already sends a printed model row (`status` other than `pending`, and a `portfolio_value`) in `entries`. If it is missing, Phase 3 has not merged and must make this change when it lands.

  Edit A. In `dashboard/backend/tests/test_frontend_live_budget.py`, find:

```python
# A board with prints, current, and models cached: the existing banner is hidden.
_HEALTHY = {
```

   Replace with:

```python
# A board with prints, current, and a printed model row (_banner_after adds
# it): the existing banner is hidden.
_HEALTHY = {
```

  Edit B. In the same file, find:

```python
def _banner_after(status):
    payload = {"window": {"label": "2026-08-01 — 2026-08-31"}, "entries": [], "live_status": status}
```

   Replace with:

```python
# The banner reads the rows, not live_status.models_cached (#586): with no
# printed model row it says model curves have not appeared yet. A healthy board
# therefore needs one printed, current model row for the banner to stay hidden.
_PRINTED_MODEL = {
    "entry_id": "deepseek_v4_pro",
    "model": "DeepSeek V4 Pro",
    "is_model": True,
    "status": "frozen",
    "portfolio_value": 100500,
    "snapshot_end": "2026-08-27",
}


def _banner_after(status):
    payload = {
        "window": {"label": "2026-08-01 — 2026-08-31"},
        "entries": [_PRINTED_MODEL],
        "live_status": status,
    }
```

  Leave the rest of `_banner_after` as it is: its `return _run_live_js(...)` lines follow unchanged. If Edit B's anchor does not match, Phase 3 changed the line. Then make `_banner_after`'s payload carry `"entries": [_PRINTED_MODEL]`, keep everything else, and add `_PRINTED_MODEL` above it as shown.

- [ ] **Step 2: Run them and confirm they fail**

```bash
pytest dashboard/backend/tests/test_frontend_live_leaderboard.py dashboard/backend/tests/test_frontend_live_trading_board.py -q -p no:cacheprovider
```

  Expected: `8 failed`. Measured: `8 failed, 50 passed` on main@17c552c5 plus Tasks 6.1–6.4 and this step, and `8 failed, 58 passed` on main@17c552c5 plus Phases 1, 2, 3 and 5, Tasks 6.1–6.4 and this step.
  - The first three fail on their assertions:
    - `test_a_catching_up_model_is_printed_and_flagged_with_its_own_date`: `assert '<span class="team-badge" title="Catching up · through 2026-08-20">Catching up</span>' in '>\n        <td>…'`, because the badge is not in the row.
    - `test_the_banner_names_the_catching_up_model_instead_of_calling_it_missing`: `AssertionError: assert 'Model curves appear' not in 'Baselines f...n page load.'`.
    - `test_the_banner_lists_every_catching_up_entry`: `AssertionError: assert 'Baselines fr...on page load.' == '2 entries ar...test session.'`.
  - The next four show the bug directly:
    - `test_a_stale_snapshot_banner_outranks_the_catching_up_line_and_the_badge_stays`: `assert '>Catching up</span>' in '\n      <tr class="" data-live-entry="spy_index">…'`, because there is no badge in the table.
    - `test_a_current_board_shows_no_banner_and_no_badge`: `assert False is True`, because `models_cached: 0` raises the "Baselines frozen" banner over a board whose model is current.
    - `test_catching_up_rows_sort_after_current_rows_and_before_pending_ones`: `AssertionError: assert {'asc': ['qwe...e_haiku_4_5']} == {'asc': ['equ...e_haiku_4_5']}`. The rows still sort as one pool, so a catching-up row leads in each direction: `desc` is `['deepseek_v4_pro', 'spy_index', 'equal_weight_djia', 'qwen3_7_plus', 'claude_haiku_4_5']` and `asc` is `['qwen3_7_plus', 'equal_weight_djia', 'spy_index', 'deepseek_v4_pro', 'claude_haiku_4_5']`.
    - `test_the_banner_no_longer_reads_models_cached`: `AssertionError: assert 'models_cached' not in '…'`, because `models_cached` is still read.
  - `test_the_about_card_explains_the_catching_up_label`: `assert 'An entry more than one trading session behind is marked Catching up: shown, but not ranked.' in 'Live Leaderboard</h3>…'`.

  `test_a_model_with_no_curve_still_gets_the_models_pending_banner` and `test_a_catching_up_row_with_no_value_reads_as_pending_not_catching_up` **pass** before and after this task. They are the regression guards for the banner this task re-keys.

  If Phase 3 has merged, also run `pytest dashboard/backend/tests/test_frontend_live_budget.py -q -p no:cacheprovider`. Expected: all pass, with Edits A–B in and before Step 3. That is `10 passed` for the file as Phase 3's plan writes it.

- [ ] **Step 3: Implement**

  Banner order after edits 2–3, which must hold even if an earlier phase added a branch:
  1. no prints;
  2. stale snapshot;
  3. no printed model;
  4. catching up;
  5. hidden.

  The catching-up branch goes immediately before the final `else`. Phase 3's budget-halt line is **not** a branch. It is a separate block below this `if (banner) { … }` block (`const haltLine = liveBudgetHaltLine(status);`, just above `const zone = …`), and it appends its sentence ("Model runs are paused: the monthly model budget is used up. Baselines keep updating.") to whichever message the branches chose, the catching-up line included. Leave that block where it is. Do not move it into the chain.

1. In `dashboard/frontend/js/live-leaderboard.js` (≈:108 on main@17c552c5), find:

```js
// A row with no stored run has no value or return. Official rank still exists
// on the payload for the header/rules, but this table does not render it —
// catching-up rows are unranked and still have prints.
function liveEntryIsPending(entry) {
  return !entry || entry.status === 'pending' || entry.portfolio_value == null;
}
```

   Replace with:

```js
// A row with no stored run has no value or return. The table renders no rank,
// so a catching-up row (status 'catching_up': it has a curve, but its newest
// stored session is more than one behind the board's) is NOT pending: it
// shows its figures, flagged by liveEntryIsCatchingUp, because they are from
// an earlier day than the rows beside it (#586).
function liveEntryIsPending(entry) {
  return !entry || entry.status === 'pending' || entry.portfolio_value == null;
}

// Printed rows only: a catching_up row with no value is pending, and a
// 'Catching up' badge beside 'Awaiting freeze' would say both at once.
function liveEntryIsCatchingUp(entry) {
  return !!entry && entry.status === 'catching_up' && !liveEntryIsPending(entry);
}

// The date is the entry's own newest stored session, not the board's.
function liveCatchingUpLabel(entry) {
  const through = entry && entry.snapshot_end ? String(entry.snapshot_end) : '';
  return through ? `Catching up · through ${through}` : 'Catching up';
}

// Read off the rows, not live_status.models_cached: the server counts only
// ranked (frozen) rows there, so a model mid-replay with a curve on screen
// counted as missing and the banner said its curve had not appeared yet.
function liveHasPrintedModel(entries) {
  return (entries || []).some((e) => liveIsModel(e) && !liveEntryIsPending(e));
}

// The banner line for rows shown but not ranked; '' when there are none.
function liveCatchingUpBanner(entries) {
  const behind = (entries || []).filter(liveEntryIsCatchingUp);
  if (!behind.length) return '';
  if (behind.length === 1) {
    const e = behind[0];
    const through = e.snapshot_end ? ` (through ${e.snapshot_end})` : '';
    return `${liveSeriesLabel(e)} is catching up${through}: `
      + 'it is shown but not ranked until it reaches the latest session.';
  }
  const names = behind
    .map((e) => (e.snapshot_end ? `${liveSeriesLabel(e)} through ${e.snapshot_end}` : liveSeriesLabel(e)))
    .join(', ');
  return `${behind.length} entries are catching up (${names}): `
    + 'they are shown but not ranked until they reach the latest session.';
}
```

2. In `dashboard/frontend/js/live-leaderboard.js` (≈:208 on main@17c552c5), find:

```js
    } else if ((Number(status.models_cached) || 0) === 0) {
```

   Replace with:

```js
    } else if (!liveHasPrintedModel(payload.entries)) {
```

3. In `dashboard/frontend/js/live-leaderboard.js` (≈:214 on main@17c552c5), at the end of the banner chain in `updateLiveHeader`, apply the form your tree holds.

   **Form A** (main@17c552c5: the zone caption follows the banner block). Find:

```js
    } else {
      banner.hidden = true;
      banner.textContent = '';
    }
  }

  const zone = document.getElementById('liveZoneCaption');
```

   Replace with:

```js
    } else if (liveCatchingUpBanner(payload.entries)) {
      banner.hidden = false;
      banner.textContent = liveCatchingUpBanner(payload.entries);
    } else {
      banner.hidden = true;
      banner.textContent = '';
    }
  }

  const zone = document.getElementById('liveZoneCaption');
```

   **Form B** (main@17c552c5 plus Phases 1–5: Phase 3's halt block follows the banner block). Find:

```js
    } else {
      banner.hidden = true;
      banner.textContent = '';
    }
  }

  const haltLine = liveBudgetHaltLine(status);
```

   Replace with:

```js
    } else if (liveCatchingUpBanner(payload.entries)) {
      banner.hidden = false;
      banner.textContent = liveCatchingUpBanner(payload.entries);
    } else {
      banner.hidden = true;
      banner.textContent = '';
    }
  }

  const haltLine = liveBudgetHaltLine(status);
```

4. In `dashboard/frontend/js/live-leaderboard.js` (≈:318 on main@17c552c5), in the Entry cell of `populateLiveTable`, apply the form your tree holds.

   **Form A** (main@17c552c5: the type badge is the cell's last badge). Find:

```js
            <span class="team-badge">${escapeHtml(liveFormatBadge(entry.team_badge))}</span>
          </div>
```

   Replace with:

```js
            <span class="team-badge">${escapeHtml(liveFormatBadge(entry.team_badge))}</span>
            ${liveEntryIsCatchingUp(entry) ? `<span class="team-badge" title="${escapeHtml(liveCatchingUpLabel(entry))}">Catching up</span>` : ''}
          </div>
```

   **Form B** (main@17c552c5 plus Phases 1–5: Phase 3's attempt badge sits directly under the type badge). Find:

```js
            <span class="team-badge">${escapeHtml(liveFormatBadge(entry.team_badge))}</span>
            ${attemptBadge ? `<span class="team-badge">${escapeHtml(attemptBadge)}</span>` : ''}
```

   Replace with:

```js
            <span class="team-badge">${escapeHtml(liveFormatBadge(entry.team_badge))}</span>
            ${liveEntryIsCatchingUp(entry) ? `<span class="team-badge" title="${escapeHtml(liveCatchingUpLabel(entry))}">Catching up</span>` : ''}
            ${attemptBadge ? `<span class="team-badge">${escapeHtml(attemptBadge)}</span>` : ''}
```

   Either way the order is type badge, then Catching up, then (Form B) Paused or Last run failed.

5. In `dashboard/frontend/js/live-leaderboard.js` (≈:357 on main@17c552c5), find:

```js
    <div class="team-detail-row"><span class="team-detail-label">Through</span><span class="team-detail-value">${entry.snapshot_end || '—'}</span></div>
```

   Replace with:

```js
    ${liveEntryIsCatchingUp(entry) ? `<div class="team-detail-row"><span class="team-detail-label">Status</span><span class="team-detail-value">${escapeHtml(liveCatchingUpLabel(entry))}</span></div>` : ''}
    <div class="team-detail-row"><span class="team-detail-label">Through</span><span class="team-detail-value">${entry.snapshot_end || '—'}</span></div>
```

6. In `dashboard/frontend/js/live-leaderboard.js` (≈:278 on main@17c552c5, inside `liveSortedEntries`; no earlier phase edits these lines), find:

```js
  entries.sort((a, b) => {
    // Pending rows have nothing to sort on; keep them below every printed row
    // in either direction.
    const pa = liveEntryIsPending(a);
    const pb = liveEntryIsPending(b);
    if (pa !== pb) return pa ? 1 : -1;
    if (pa) return 0;
```

   Replace with:

```js
  // Three tiers, in this order in either direction: current rows; then
  // catching-up rows, which are shown but not ranked and whose figures are from
  // an earlier session, so no sort may slot one in among the current rows
  // (#586); then pending rows, which have nothing to sort on.
  const tier = (e) => (liveEntryIsPending(e) ? 2 : (liveEntryIsCatchingUp(e) ? 1 : 0));
  entries.sort((a, b) => {
    const ta = tier(a);
    const tb = tier(b);
    if (ta !== tb) return ta - tb;
    if (ta === 2) return 0;
```

   The rest of the comparator (`const av = liveSortValue(a, liveSortKey);` onwards) is unchanged, so within a tier rows still sort by the chosen column, with non-finite values last.

7. In `dashboard/frontend/app.html` (≈:1595 on main@17c552c5), find:

```html
An entry more than one trading session behind is shown but not ranked.</p>
```

   Replace with:

```html
An entry more than one trading session behind is marked Catching up: shown, but not ranked.</p>
```

8. Read the current value first: `grep -o 'js/live-leaderboard\.js?v=[0-9]*' dashboard/frontend/app.html`. Call it M. It is 5 on bare main@17c552c5, and expect 7 after Phases 3 and 5. In `dashboard/frontend/app.html` (≈:2523 on main@17c552c5, ≈:2540 once Tasks 6.3–6.4 are in), find the following tag, with M in place of 5:

```html
<script src="js/live-leaderboard.js?v=5" defer></script>
```

   Replace with (M+1; 6 on bare main, 8 after Phases 3 and 5):

```html
<script src="js/live-leaderboard.js?v=6" defer></script>
```

  Edit 8 is the cache-buster. No test pins it exactly (`test_live_script_is_cache_busted` only checks `>= 1`), so nothing catches a missed bump. Re-run the grep and confirm it prints M+1. The badge reuses the existing `.team-badge` class, so `styles.css` stays untouched.

```bash
node --check dashboard/frontend/js/live-leaderboard.js
```

- [ ] **Step 4: Run them and confirm they pass**

```bash
pytest dashboard/backend/tests/test_frontend_live_leaderboard.py dashboard/backend/tests/test_frontend_live_trading_board.py dashboard/backend/tests/test_landing_copy_register.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_app_copy_register.py dashboard/backend/tests/test_live_leaderboard.py -q -p no:cacheprovider
pytest dashboard/backend/tests/test_frontend_live_budget.py -q -p no:cacheprovider   # only if Phase 3 has merged
```

  Expected: all pass. Measured for the first command: `198 passed` on main@17c552c5 plus this phase, and `231 passed` on main@17c552c5 plus Phases 1, 2, 3 and 5 plus this phase. Phase 4 adds tests to `test_live_leaderboard.py`, so expect more on the full stack. The second command gives all passed, measured `10 passed` for Phase 3's plan version. The node-gated cases must report as passed, not skipped. If they show as `s`, `node` is not on `PATH` and this step has not been verified. If the second command shows `5 failed` (`{'hidden': False, 'text': 'Baselines frozen…'}` where a hidden or halt-only banner was expected), Edits A–B of Step 1 are missing.

- [ ] **Step 5: Commit**

```bash
git add dashboard/frontend/js/live-leaderboard.js dashboard/frontend/app.html dashboard/backend/tests/test_frontend_live_leaderboard.py dashboard/backend/tests/test_frontend_live_trading_board.py
git add dashboard/backend/tests/test_frontend_live_budget.py   # only if Step 1 edited it
git commit -m "fix(live): flag catching-up rows and stop calling their curves missing (#586)"
```

  Do not put a closing keyword ("Closes", "Fixes") in a commit message: GitHub reads commits too, and the PR body carries it.

### Task 6.6: Open the PR

**Files:** none. **Interfaces:** none.

- [ ] **Step 1: Run the whole suite and the JS parse checks**

```bash
node --check dashboard/frontend/js/live-leaderboard.js
node --check dashboard/frontend/js/leaderboard.js
pytest dashboard/backend/tests -q -p no:cacheprovider
```

  The run takes about 7.5 minutes (453 s measured on main@17c552c5 plus this phase). From an agent's Bash tool, set the timeout to 600000 ms or run it in the background. Run the command exactly as written:
  - Do not add `-x`. The `test_report_pdf.py` failures below would stop the run before the rest of the suite runs.
  - Do not add `--timeout`. `pytest-timeout` is not installed here, so pytest exits with `pytest: error: unrecognized arguments: --timeout=0`.

  Expected: green, apart from 7 `test_report_pdf.py` failures you get locally without `reportlab`. Those are environmental, and CI installs it. Measured on main@17c552c5 plus this phase, before Task 6.5 gained its sort test (which adds one pass): `7 failed, 6708 passed, 176 skipped, 25 xfailed`. All 7 failures were in `test_report_pdf.py`:
  - `test_produces_valid_pdf`
  - `test_empty_and_none_inputs`
  - `test_large_document_is_bounded`
  - `test_unicode_content`
  - `test_crossing_inline_markup_degrades_to_plain_text_not_a_crash`
  - `test_page_cap_truncates_instead_of_dropping_the_pdf`
  - `test_report_lists_available_artifacts_without_rendering_a_pdf`

  The same 7 fail on bare main@17c552c5 (`7 failed, 2 passed` for that file, measured). Adding `--ignore=dashboard/backend/tests/test_report_pdf.py` gave `6706 passed, 176 skipped, 25 xfailed` on the same tree, again before the sort test. With Phases 1–5 merged, the passed count is higher. Any failure outside `test_report_pdf.py` is a real regression. Confirm `git status` shows no change to `dashboard/storage/data/backtest.db`.

- [ ] **Step 2: Push and open the PR**

```bash
git push -u origin feat/live-board-copy
gh pr create --base main --head feat/live-board-copy \
  --title "feat(live): Live Leaderboard copy and catching-up badge" \
  --body-file - <<'EOF'
- Renames the Live board "Live Leaderboard" in-app and in the payload. `board_title` and `phase_label` ("Season 0 · preview") now come from `service.LIVE_BOARD_TITLE` and `service.LIVE_PHASE_LABEL`. This is the only API-visible change.
- Live header: the subtitle is now "Simulated fills · forward-only · no real money", and the Updated stat is now static ("Nightly after the close") instead of the request clock.
- The Competition header now reads "legacy engine · historical window · may overlap model training data".
- Both boards' Rules and About cards now say stop-loss and take-profit prices are not executed (D12). `test_stop_prices_not_enforced.py` pins that the legacy single-prompt path drops them. The prompt is unchanged.
- #586: catching-up rows get a "Catching up" badge and a "Status: Catching up · through <date>" detail row. The banner now reads the rows instead of `live_status.models_cached`, which counts ranked rows only, and a sorted table keeps catching-up rows below every current row. Phase 3's healthy-board test fixture now carries a printed model row to match.
- Not in this PR: the landing `Race.tsx`/bundle and `docs/source` still say "Live Trading Leaderboard" (owner-coordinated follow-up). Cache-busters: `js/leaderboard.js` and `js/live-leaderboard.js` are bumped by 1. Merging auto-deploys prod.

Do not merge between 22:15 and 23:30 UTC on a weekday. The merge auto-deploys prod, and the Render restart kills the 22:30 UTC live refresh thread mid-run, so that night's session is billed again on the next night.

Closes #586
EOF
```

  If the PR comes from a fork, CI and CodeQL wait at `action_required` until a maintainer approves the run. Merging to `main` auto-deploys prod (the CI deploy hook), and `main` has no required checks, so do not merge over red CI.

- [ ] **Step 3: Re-check the cache-busters against main immediately before merging**

  This PR bumps two tags: `js/leaderboard.js?v=` (Task 6.2) and `js/live-leaderboard.js?v=` (Task 6.5 edit 8).

```bash
git branch --show-current   # must print feat/live-board-copy (a sibling session can switch it)
git fetch origin main
for f in js/leaderboard.js js/live-leaderboard.js; do
  echo "$f main=$(git show origin/main:dashboard/frontend/app.html | grep -o "$f?v=[0-9]*") branch=$(grep -o "$f?v=[0-9]*" dashboard/frontend/app.html)"
done
```

  Expected: for each tag, main holds exactly the branch value minus one. With Phases 1–5 merged and nothing else in between, that is `js/leaderboard.js?v=38` on main and `?v=39` on the branch, and `js/live-leaderboard.js?v=7` on main and `?v=8` on the branch. If main has moved (another PR bumped the same tag), rebase onto it (`git rebase origin/main`) and bump again from main's value to that value plus one. For `leaderboard.js` (and `home-page.js`, if a rebase ever leaves this PR bumping that tag too), also update the exact pin in `dashboard/backend/tests/test_frontend_fast_boot.py` (≈:200-201, inside `test_cache_busters_bumped`) in the same commit. Then run `pytest dashboard/backend/tests/test_frontend_fast_boot.py -q -p no:cacheprovider`, push with `git push --force-with-lease`, and run this step again. Identical one-line bumps from two branches merge with no conflict, so this check is the only thing that stops the second PR's JS shipping under a version browsers already cached.

- [ ] **Step 4: Merge**

  Merge only once Step 3 holds and CI is green, and never inside the weekday window the PR body names (22:15–23:30 UTC). Once merged, the PR is write-only: file follow-ups as issues, not comments.

#### Phase 6 review-focus candidates

1. **Covered.** A row the server marks `catching_up` but sends with `portfolio_value: null` (its run row has no `final_equity`). Without care, that row would show a "Catching up" badge next to "Awaiting freeze". → Added in Task 6.5 as `test_a_catching_up_row_with_no_value_reads_as_pending_not_catching_up`, and `liveEntryIsCatchingUp` requires a printed row.
2. **Covered.** A stale snapshot (tonight's refresh failed) while a model is also catching up. Which banner wins, and does the badge survive? → Added in Task 6.5 as `test_a_stale_snapshot_banner_outranks_the_catching_up_line_and_the_badge_stays`. The static "Nightly after the close" then states the intended cadence, and the stale banner is what reports a missed night.
3. **No test.** Deploy skew. Vercel serves the new static `app.html`/JS while Render still runs the old backend, so the payload's `board_title` ("Live Trading Leaderboard") and `phase_label` ("Season 0") overwrite the new static copy until the backend deploys. It is not tested: the skew lasts minutes, it self-heals, and both deploys fire from the same merge.
4. **No test.** A phone-width viewport. With Phase 3 merged, the Entry cell can hold three `.team-badge` spans: the type, Catching up, and Paused or Last run failed. They sit inside `.team-name-badge`, which is `white-space: nowrap` (`styles.css` ≈:4243), so they may widen the column. No browser runs in this WSL environment, so the owner should look at it at 375px before merging. No CSS is added, which keeps `styles.css?v=159` and its exact pins in three test files unchanged.
5. **No test.** An API consumer outside this repo that matched `phase_label == "Season 0"` or `board_title == "Live Trading Leaderboard"`. No in-repo consumer reads either field (grep: only `live-leaderboard.js` and the dead `leaderboard.js` branch). The value change is deliberate and pinned by `test_leaderboard_api.py`, so the PR body names it as the only API-visible change.
6. **Covered (found on a stacked tree, not on main alone).** Phase 3's `test_frontend_live_budget.py` builds its "healthy" board with `entries: []` and relies on `models_cached: 1` to hide the banner. Once Task 6.5 reads the rows instead, five of its tests fail. A run of this phase on bare main cannot see this, because that file does not exist there. → Task 6.5 Step 1 Edits A–B give the fixture a printed, current model row. They were checked on main@17c552c5 plus Phases 1, 2, 3 and 5: `10 passed` both before and after Step 3, and `5 failed` after Step 3 without them. Reviewer: confirm the merged Phase 3 file still has the `entries: []` line, or that its owner already changed it.
7. **Covered.** The cache-buster no-op. Phase 5 already moves `leaderboard.js` from 37 to 38, so following the literal 37 → 38 anchors would leave the tag unchanged and the browser on the old JS. → Tasks 6.2 and 6.5 write N → N+1 from a grep. For `leaderboard.js`, `test_cache_busters_bumped` fails in Task 6.2 Step 2 if edit 9 wrote N. For `live-leaderboard.js` no test pins the value, so Task 6.5 edit 8 re-runs the grep. Task 6.6 Step 3 then re-checks both tags against `origin/main` just before the merge. The reviewer should check the diff shows +1 on both tags.
8. **Covered (raised by the whole-plan review).** Sorting the Live table interleaved catching-up rows with current ones, although the server, the banner and the About card call them shown but not ranked: a model several sessions behind could top the Sharpe column. → Task 6.5 edit 6 puts catching-up rows in their own tier between current and pending rows, in both directions, pinned by `test_catching_up_rows_sort_after_current_rows_and_before_pending_ones` (red before Step 3 and green after it, both on main@17c552c5 alone and on main@17c552c5 plus Phases 1, 2, 3 and 5).
