# Live-Board Fixes P7: Developer Records and Owner-Coordinated Surfaces — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repo `CLAUDE.md` and the dated live-board records describe the Live Leaderboard as it ships after P1–P6 (PR A), then carry the "Live Leaderboard" rename and the daily Sharpe with its 95% interval onto the landing page and the hosted docs once the owner approves the copy (PR B).

**Architecture:** PR A is documentation plus one text-only pytest module (`test_live_board_records.py`) that reads `CLAUDE.md`, both refresh workflows and the source tree back, so a renamed name, a revived stale claim, or a decision tape added to `live.py` turns it red. PR B edits the Vite/React landing source (`Race.tsx`, `BoardPreview.tsx`, `lib/leaderboard.ts`), rebuilds and hand-copies the bundle into `dashboard/frontend/`, and renames the board in four RST pages; the browser only displays the Sharpe numbers the server computes.

**Tech Stack:** Markdown and YAML comments; Python 3.11 pytest from the repo root (text and AST guards that never import `dashboard.backend`); TypeScript/React landing built with Vite, its helpers tested by transpiling with esbuild and running under `node`; reStructuredText checked with a throwaway docutils venv.

**Spec:** `docs/superpowers/specs/2026-10-14-atl-direction.md` §4 (Sharpe with its interval; stops), §6 (Live Leaderboard name, cadence, reasoning view), §9 (records to correct). Decisions, Global Constraints, merge order and Review Focus live in the index plan `docs/superpowers/plans/2026-10-14-live-board-fixes.md`; every task here inherits its Global Constraints.

**Merge order:** P1 → P2 → P3 → P4 → P5 → P6 → P7 PR A → P7 PR B (owner-gated). Deadlines: P3 before 2026-11-01; P4 by 2026-10-31 (else bump both live_from values); P1 merged and deployed by 2026-11-20. One executor per contiguous range; phases are not safe to run in parallel (several append to the same test files and bump the same cache-busters).

## Phase 7: Developer records and owner-coordinated surfaces (landing, docs)

**Why.** The direction spec (`docs/superpowers/specs/2026-10-14-atl-direction.md` §9, "Records to correct") found the repo `CLAUDE.md` still describing the live board as an unbuilt "Season 0 preview" three weeks after `live.py` (#534) began advancing a calendar-month board nightly, and silent on `LIVE_LEADERBOARD_DEPLOY_MODELS`, the GitHub variable that has billed DeepSeek every weekday since 2026-09-29. Phases 1–6 change the board further (early closes, trade memory, attempt ledger, roster, daily Sharpe, copy). This phase brings the developer record in line with the shipped board (PR A), then carries the §6 rename and the §4 Sharpe rule onto the two owner-coordinated public surfaces, the landing page and the hosted docs (PR B).

**Depends on.**
- **PR A** (Tasks 7.1, 7.2, 7.5 part A): Phases 1–6 merged to `main`: P1 `2026-10-14-live-board-fixes-p1-early-close.md`, P2 `2026-10-14-live-board-fixes-p2-memory-robustness.md`, P3 `2026-10-14-live-board-fixes-p3-attempt-ledger.md`, P4 `2026-10-14-live-board-fixes-p4-roster.md`, P5 `2026-10-14-live-board-fixes-p5-daily-sharpe.md` and P6 `2026-10-14-live-board-fixes-p6-copy.md`, all in `docs/superpowers/plans/`. Its test reads names those phases introduce (`nyse_early_closes`, `restored_trades`, `live_budget.py`, `live_from`, `live_overrides`, `daily_sharpe_interval`, `LIVE_BOARD_TITLE`, `_STOP_RULE`, …), so it cannot go green earlier, and the section it writes would describe code that is not on `main`. It does not need them deployed. Measured on main@17c552c5 (none of Phases 1–6 present): every edit applies cleanly and exactly the 18 Phase-owned cases stay red, which is why Task 7.1 Step 0 is a hard gate, not a formality.
- **PR B** (Tasks 7.3, 7.4, 7.5 part B): P5 `2026-10-14-live-board-fixes-p5-daily-sharpe.md` (Sharpe payload) and P6 `2026-10-14-live-board-fixes-p6-copy.md` (copy and labels, Decision 8 in the index plan) merged **and deployed to Render**, because Vercel deploys the landing the moment PR B merges. Phase 5 also rewrites the docstring that Task 7.3 Step 1d edits, so 7.3's anchors assume it is merged. **Execute PR B only after the owner approves the copy in Task 7.3 Step 0.** PR B does not depend on PR A.

**Out of scope.**
- **Public per-trade lists** (spec §6, the Format row's "public trades") and **the reasoning/trade-log view** (spec §6, "Reasoning and trade-log view"). Both belong to the separate reasoning/trade-log view plan, because nothing on the live path records a decision tape yet. PR A writes that fact into `CLAUDE.md` and pins it (Task 7.1).
- **The direction spec's Errata block.** The plan's own docs PR adds it before P1; this phase never edits `docs/superpowers/specs/2026-10-14-atl-direction.md`.

**Branch / PR title.**
- PR A: branch `docs/live-board-records`, title `docs: correct the live board records`.
- PR B: branch `docs/live-board-public-copy`, title `feat(landing): Live Leaderboard copy and Sharpe interval`. Opened as a **draft**.

**Interfaces this phase produces.**
- `CLAUDE.md` section heading `### Two leaderboards: Competition and Live` (replaces ``### Two leaderboards, and what `live` does *not* mean``). Other docs cite it as `CLAUDE.md`, "Two leaderboards", which still resolves.
- `CLAUDE.md` env bullet beginning ``- `LIVE_LEADERBOARD_DEPLOY_MODELS` `` (GitHub repository variable).
- A one-line banner under the H1 of six dated records (verbatim in Task 7.2 Step 3).
- `dashboard/backend/tests/test_live_board_records.py`: module constants `_RETIRED: tuple[str, ...]`, `_NAMED: tuple[tuple[str, tuple[str, ...]], ...]`, `_PATHS: tuple[tuple[str, str], ...]`, `_SUPERSEDED: tuple[str, ...]`; `_NO_TAPE_SENTENCE: str`, `_NO_TAPE_MODULES: tuple[str, ...]`; helpers `_env_bullet(name: str) -> str`, `_section(heading: str) -> str`, `_imported_modules(path: str) -> set[str]`.
- `.github/workflows/daily-leaderboard.yml` header comment that points at `live-leaderboard.yml`.
- `dashboard/landing/src/lib/leaderboard.ts`:
  - `export type SharpeBasis = { basis?: string; annualization_days?: number; confidence?: number; min_days?: number; rf_annual?: number; stderr?: string }`
  - `export type SharpeDisplay = { value: string; interval: string; detail: string }`
  - `export function formatSharpe(entry: Partial<LeaderboardEntry> | null | undefined, basis?: SharpeBasis | null): SharpeDisplay` (replaces `formatSharpe(value: unknown): string`; its only caller is `buildBoardData`).
  - `LeaderboardEntry` gains optional `sharpe_ci_low`, `sharpe_ci_high`, `sharpe_days`, `sharpe_status`; `BoardStanding` gains `sharpeInterval: string`, `sharpeDetail: string`; `buildBoardData` and `fetchLeaderboard` read the payload's top-level `sharpe_basis`.
- `dashboard/landing/src/components/home/BoardPreview.tsx`: `RANK_GRID` sm template `26px_minmax(0,1.4fr)_96px_78px_128px`.
- `dashboard/backend/tests/test_docs_live_board_names.py`.

**Interfaces this phase consumes** (each must exist on `main` before the task that uses it):
- Phase 1 (Decision 1 in the index plan): `nyse_early_closes` (the early-close rule table) in `dashboard/backend/infrastructure/market_data/us_market_calendar.py`, applied by `sessions.is_in_session` on its open-stamped branch only.
- Phase 2 (Decision 3 in the index plan): `restored_trades` and the snapshot key `recent_trades` in `dashboard/backend/domain/backtesting/portfolio_manager.py` (trade memory = the 24 hours up to the newest trade, at most `SNAPSHOT_TRADE_MEMORY_LIMIT` = 200).
- Phase 3 (Decision 5 in the index plan): the `CLAUDE.md` bullet beginning ``- `LEADERBOARD_MONTHLY_BUDGET_USD` ``, placed directly after the `LEADERBOARD_DAILY_AUTO_DEPLOY` bullet (P3 Task 3.6); `dashboard/backend/domain/leaderboard/live_budget.py` containing `LEADERBOARD_MONTHLY_BUDGET_USD` (`BUDGET_ENV`), `"leaderboard-live-spend"`, `"leaderboard_spend"`; `budget_halted` and `last_attempt` in `live.py` or `live_budget.py`.
- Phase 4 (Decision 6 in the index plan): `LIVE_MODEL_IDS = ("deepseek_v4_pro", "qwen3_7_plus", "claude_haiku_4_5")`; per-entry `live_from` and `live_overrides` read in `live.py`. Phase 4 corrected that Decision: Haiku's Live-only `model_id`/`temperature` ride a `live_overrides` block, and the shared keys stay as the Competition runs recorded them, because `service._entry_config_drift` would otherwise demote Haiku's Competition repeats. The section below records that, not the Decision's original wording.
- Phase 5 (Decision 7 in the index plan): `daily_sharpe_interval` and `daily_sharpe_basis` in `dashboard/backend/domain/backtesting/metrics.py`; per-entry payload fields `sharpe_ratio`, `sharpe_ci_low`, `sharpe_ci_high`, `sharpe_days`, `sharpe_status` on both boards; the board-level basis at `live_status.sharpe` (Live) and the top-level key **`sharpe_basis`** (Competition), which is the name Phase 5's interface list gives and Task 7.3 Step 0 confirms on the live payload; Phase 5 Task 5.4's rewrite of the `test_standings_carry_the_ending_value_and_sharpe_the_table_renders` docstring (it begins `Sharpe's em dash IS reachable through the API now`).
- Phase 6 (Decision 8 in the index plan): `LIVE_BOARD_TITLE`, `LIVE_PHASE_LABEL` in `service.py`, sent as `board_title`/`phase_label`; app.html About card heading `Live Leaderboard</h3>` (already re-pinned in `test_landing_copy_register.py::_app_html_live_surfaces` by that phase); `dashboard/backend/tests/domain/backtesting/test_stop_prices_not_enforced.py` and `_STOP_RULE` in `dashboard/backend/tests/test_frontend_live_trading_board.py`.

**Execution notes for every task.**
- Run commands from the repo root. Never run `python -c` (or any ad-hoc script) that imports `dashboard.backend`: it mutates the committed seed DB. `pytest` is safe.
- Another session may share this checkout. Run `git branch --show-current` immediately before every commit and stop if it is not the branch named in the task.
- PR A and PR B are two branches, each cut from `origin/main`; never stack them. Inside a pre-made worktree `git switch -c <branch> origin/main` still works, so use the branch names above (Task 7.1 Step 0, Task 7.3 Step 0) rather than committing on the worktree's own branch.
- A code block that sets shell variables (Task 7.1 Step 0, Task 7.3 Step 3k) must run as one shell invocation: each separate Bash call starts a fresh shell in the repo root.
- No browser runs in this WSL. Landing behaviour is tested by transpiling `leaderboard.ts` with `dashboard/landing/node_modules/.bin/esbuild` and running it under `node` (the `_run_ts` helper). Those tests skip in CI, which installs no landing `node_modules`, so they must be run locally.

---

### Task 7.1: Describe the shipped live board in CLAUDE.md (PR A)

**Files:**
- Create: `dashboard/backend/tests/test_live_board_records.py`
- Modify: `CLAUDE.md` (the `LEADERBOARD_DAILY_REFRESH_SECRET` bullet ≈:71, the `LEADERBOARD_DAILY_AUTO_DEPLOY` bullet ≈:89, the section ``### Two leaderboards, and what `live` does *not* mean`` ≈:213–291 on main@17c552c5)
- Modify: `.github/workflows/daily-leaderboard.yml` (header comment ≈:12–18)

**Interfaces:**
- Consumes: every name in the "Interfaces this phase consumes" list except the landing ones; Phase 3's `LEADERBOARD_MONTHLY_BUDGET_USD` bullet (referenced, never restated).
- Produces: the `### Two leaderboards: Competition and Live` section, the `LIVE_LEADERBOARD_DEPLOY_MODELS` bullet, `test_live_board_records.py` with `_RETIRED`, `_NAMED`, `_PATHS`, `_NO_TAPE_SENTENCE`, `_NO_TAPE_MODULES`, `_env_bullet`, `_section`, `_imported_modules`.

- [ ] **Step 0: Start the branch**

```bash
git fetch origin main
git switch -c docs/live-board-records origin/main
git branch --show-current   # must print docs/live-board-records
```

Confirm Phases 1–6 are on `origin/main`. This checks every Phase-owned name the Step 1 test reads, in the file the test reads it from (run it as one block):

```bash
need() {  # need <name> <file>...: prints "ok" when any listed file contains <name>
  local name=$1; shift
  for f in "$@"; do
    if [ -f "$f" ] && grep -qF -- "$name" "$f"; then echo "ok       $name"; return; fi
  done
  echo "MISSING  $name  (looked in: $*)"
}
B=dashboard/backend; L=$B/domain/leaderboard; M=$B/domain/backtesting/metrics.py
need nyse_early_closes              $B/infrastructure/market_data/us_market_calendar.py  # Phase 1
need restored_trades                $B/domain/backtesting/portfolio_manager.py           # Phase 2
need LEADERBOARD_MONTHLY_BUDGET_USD $L/live_budget.py                                   # Phase 3
need leaderboard-live-spend         $L/live_budget.py                                   # Phase 3
need leaderboard_spend              $L/live_budget.py                                   # Phase 3
need budget_halted                  $L/live.py $L/live_budget.py                        # Phase 3
need last_attempt                   $L/live.py $L/live_budget.py                        # Phase 3
need live_from                      $L/live.py $L/service.py                            # Phase 4
need live_overrides                 $L/live.py                                          # Phase 4
need daily_sharpe_interval          $M                                                  # Phase 5
need daily_sharpe_basis             $M                                                  # Phase 5
need sharpe_ci_low                  $L/live.py $L/service.py $M                         # Phase 5
need sharpe_status                  $L/live.py $L/service.py $M                         # Phase 5
need LIVE_BOARD_TITLE               $L/service.py                                       # Phase 6
need LIVE_PHASE_LABEL               $L/service.py                                       # Phase 6
need _STOP_RULE                     $B/tests/test_frontend_live_trading_board.py        # Phase 6
ls $B/tests/domain/backtesting/test_stop_prices_not_enforced.py                         # Phase 6
grep -c '^- `LEADERBOARD_MONTHLY_BUDGET_USD`' CLAUDE.md                                 # Phase 3: must print 1
grep -A4 '^LIVE_MODEL_IDS' $L/live.py                                                   # Phase 4: three ids, no nemotron
```

Expected: sixteen `ok` lines, the `test_stop_prices_not_enforced.py` path, `1`, and a `LIVE_MODEL_IDS` tuple naming `deepseek_v4_pro`, `qwen3_7_plus`, `claude_haiku_4_5` and not `nemotron_3_nano_30b`.

On main@17c552c5 every `need` prints `MISSING`, `ls` fails, the `grep -c` prints `0` and the tuple is `("deepseek_v4_pro", "nemotron_3_nano_30b")`. Any `MISSING`, a `0`, or Nemotron in the tuple means the phase in that line's comment has not merged: stop and wait for it. Exception: if `gh pr list --state merged --head <branch>` shows the phase merged (Phase 1 `fix/live-early-close`, 2 `fix/live-memory-robustness`, 3 `feat/live-attempt-ledger`, 4 `feat/live-roster-qwen-haiku`, 5 `feat/board-daily-sharpe`, 6 `feat/live-board-copy`) but one of its names still prints `MISSING`, it merged under another name. Find the real one with `grep -rn`, then use it in this check, in the section sentence (Step 3d) and in its `_NAMED` row (Step 1), all three in the same edit.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_live_board_records.py`:

```python
"""The developer record of the Live board, read back against the tree.

On 2026-10-14 the direction review found the repo CLAUDE.md still saying "the
season engine still does not exist" three weeks after domain/leaderboard/live.py
(#534) began advancing a calendar-month board nightly. Nothing failed, because
nothing read those sentences back. This module does, narrowly:

- the sentences that described the unbuilt board stay gone;
- the env bullets name both refresh workflows and the GitHub variable that
  bills, and quote the workflow's gate exactly;
- every name the "Two leaderboards" section cites still exists where the
  section says it does, so a rename turns this red instead of leaving the
  record pointing at nothing;
- the section's one negative claim, that live.py writes no decision tape,
  stays true: the day live.py imports the trace store or the decision tape,
  the sentence and its test have to change together.

Text and AST only: nothing here imports dashboard.backend, so it cannot touch
a DB. When a name test fails after a rename, fix the CLAUDE.md sentence and
the _NAMED row together; never delete the row to get green.
"""

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_CLAUDE_MD = (_REPO / "CLAUDE.md").read_text(encoding="utf-8")
_LIVE_WORKFLOW = (_REPO / ".github" / "workflows" / "live-leaderboard.yml").read_text(
    encoding="utf-8"
)
_DAILY_WORKFLOW = (_REPO / ".github" / "workflows" / "daily-leaderboard.yml").read_text(
    encoding="utf-8"
)
_SECTION_HEADING = "### Two leaderboards"


def _env_bullet(name: str) -> str:
    """The one top-level CLAUDE.md bullet that documents ``name``."""
    lines = [ln for ln in _CLAUDE_MD.splitlines() if ln.startswith(f"- `{name}`")]
    assert len(lines) == 1, (
        f"expected exactly one CLAUDE.md bullet for {name}, found {len(lines)}"
    )
    return lines[0]


def _section(heading: str) -> str:
    start = _CLAUDE_MD.find(f"\n{heading}")
    assert start != -1, f"CLAUDE.md has no section starting {heading!r}"
    end = _CLAUDE_MD.find("\n### ", start + 1)
    return _CLAUDE_MD[start : end if end != -1 else len(_CLAUDE_MD)]


_RETIRED = (
    "the season engine still does not exist",
    "Season 0 preview with no advance engine yet",
    "in **two-week seasons**",
    "The live board reuses the contest session and window byte-for-byte",
    "the season engine's nightly advance is this same call",
    "Live Trading Leaderboard",
)


@pytest.mark.parametrize("phrase", _RETIRED)
def test_retired_live_board_claims_are_gone(phrase):
    assert phrase not in _CLAUDE_MD, f"CLAUDE.md still says {phrase!r}"


def test_the_refresh_secret_bullet_names_both_workflows():
    bullet = _env_bullet("LEADERBOARD_DAILY_REFRESH_SECRET")
    for needle in (
        ".github/workflows/daily-leaderboard.yml",
        ".github/workflows/live-leaderboard.yml",
        "/api/v1/leaderboard/live/refresh",
        "LIVE_LEADERBOARD_DEPLOY_MODELS",
    ):
        assert needle in bullet, (
            f"the LEADERBOARD_DAILY_REFRESH_SECRET bullet must name {needle}"
        )


def test_the_auto_deploy_bullet_points_at_the_live_gate():
    bullet = _env_bullet("LEADERBOARD_DAILY_AUTO_DEPLOY")
    assert "LIVE_LEADERBOARD_DEPLOY_MODELS" in bullet
    assert "Season 0" not in bullet


def test_the_live_deploy_variable_has_its_own_bullet():
    bullet = _env_bullet("LIVE_LEADERBOARD_DEPLOY_MODELS")
    gate = "vars.LIVE_LEADERBOARD_DEPLOY_MODELS == 'true'"
    for needle in (
        "GitHub repository variable, not a Render env var",
        ".github/workflows/live-leaderboard.yml",
        gate,
        "2026-09-29",
        "LEADERBOARD_MONTHLY_BUDGET_USD",
    ):
        assert needle in bullet, f"the LIVE_LEADERBOARD_DEPLOY_MODELS bullet must say {needle}"
    assert gate in _LIVE_WORKFLOW, (
        "the bullet quotes the workflow's billing gate; the two must stay the same expression"
    )


def test_the_section_points_at_the_budget_bullet_instead_of_restating_it():
    # Phase 3 owns the LEADERBOARD_MONTHLY_BUDGET_USD bullet: exactly one.
    _env_bullet("LEADERBOARD_MONTHLY_BUDGET_USD")
    assert "LEADERBOARD_MONTHLY_BUDGET_USD" in _section(_SECTION_HEADING)


def test_the_daily_workflow_no_longer_claims_the_live_advance():
    assert "season engine's nightly advance is this same call" not in _DAILY_WORKFLOW
    assert "live-leaderboard.yml" in _DAILY_WORKFLOW, (
        "the paused daily job must point at the job that actually advances the live board"
    )


_BACKEND = "dashboard/backend"
_LIVE = f"{_BACKEND}/domain/leaderboard/live.py"
_SERVICE = f"{_BACKEND}/domain/leaderboard/service.py"
_ROSTER = f"{_BACKEND}/domain/leaderboard/roster_failures.py"
_BUDGET = f"{_BACKEND}/domain/leaderboard/live_budget.py"
_TRADE_STATS = f"{_BACKEND}/domain/leaderboard/live_trade_stats.py"
_BASELINES = f"{_BACKEND}/domain/leaderboard/baselines.py"
_PM = f"{_BACKEND}/domain/backtesting/portfolio_manager.py"
_METRICS = f"{_BACKEND}/domain/backtesting/metrics.py"
_CALENDAR = f"{_BACKEND}/infrastructure/market_data/us_market_calendar.py"
_BOARD_TESTS = f"{_BACKEND}/tests/test_frontend_live_trading_board.py"
_LB_JS = "dashboard/frontend/js/leaderboard.js"
_LIVE_YML = ".github/workflows/live-leaderboard.yml"

# (name the section cites, files of which at least one must contain it).
# The comment after a row names the phase of the 2026-10-14 live-board plan
# that introduces it (`# #668`: P2's merged half, which moved the roster's
# log lines into roster_failures.py); every other row was already on
# main@17c552c5.
_NAMED = (
    ("LIVE_SESSION_ID", (_LIVE,)),
    ('"leaderboard-live"', (_LIVE,)),
    ("register_period_board", (_LIVE,)),
    ("Live leaderboard module is not loaded", (_SERVICE,)),
    ("get_live_leaderboard", (_LIVE,)),
    ("latest_live_month_runs", (_LIVE,)),
    ("refresh_live_leaderboard", (_LIVE,)),
    ("enqueue_live_leaderboard_refresh", (_LIVE,)),
    ("/api/v1/leaderboard/live/refresh", (_LIVE_YML,)),
    ("prune_superseded_live_runs", (_LIVE,)),
    ("_LIVE_REFRESH_STATE_PATH", (_LIVE,)),
    ("leaderboard_live_refresh.json", (_LIVE,)),
    ("recent_trades", (_PM,)),
    ("restored_trades", (_PM,)),  # Phase 2
    ("fold_trade_stats", (_TRADE_STATS,)),
    ("_checkpoint_passes_h6", (_LIVE,)),
    ("_write_live_checkpoint", (_LIVE,)),
    ("live.config_changed", (_LIVE,)),
    ("_RESUME_CONFIG_KEYS", (_LIVE,)),
    ("live.session_gap", (_LIVE,)),
    ("model_deploy_failed", (_ROSTER,)),  # #668
    ("model_not_attempted", (_ROSTER,)),  # #668
    ("nyse_early_closes", (_CALENDAR,)),  # Phase 1
    ("LIVE_MODEL_IDS", (_LIVE,)),
    ("live_from", (_LIVE, _SERVICE)),  # Phase 4
    ("live_overrides", (_LIVE,)),  # Phase 4
    ("_entry_config_drift", (_SERVICE,)),
    ("LIVE_LEADERBOARD_DEPLOY_MODELS", (_LIVE_YML,)),
    ("LEADERBOARD_MONTHLY_BUDGET_USD", (_BUDGET,)),  # Phase 3
    ("leaderboard-live-spend", (_BUDGET,)),  # Phase 3
    ("leaderboard_spend", (_BUDGET,)),  # Phase 3
    ("clear_live_session_runs", (_LIVE,)),
    ("budget_halted", (_LIVE, _BUDGET)),  # Phase 3
    ("last_attempt", (_LIVE, _BUDGET)),  # Phase 3
    ("catching_up", (_LIVE,)),
    ("snapshot_stale", (_LIVE,)),
    ("LIVE_BOARD_TITLE", (_SERVICE,)),  # Phase 6
    ("LIVE_PHASE_LABEL", (_SERVICE,)),  # Phase 6
    ("daily_sharpe_interval", (_METRICS,)),  # Phase 5
    ("daily_sharpe_basis", (_METRICS,)),  # Phase 5
    ("sharpe_ci_low", (_LIVE, _SERVICE, _METRICS)),  # Phase 5
    ("sharpe_status", (_LIVE, _SERVICE, _METRICS)),  # Phase 5
    ("calc_metrics", (_BASELINES,)),
    ("_STOP_RULE", (_BOARD_TESTS,)),  # Phase 6
    ("build_season_payload", (_SERVICE,)),
    ("_preview_season_dates", (_SERVICE,)),
    ("_season_trading_days", (_SERVICE,)),
    ("isLivePreview", (_LB_JS,)),
    ("seasonHasAdvanced", (_LB_JS,)),
    ("displayedSeasonNumber", (_LB_JS,)),
)

# (path as the section writes it, the repo path it must resolve to).
_PATHS = (
    ("domain/leaderboard/live.py", _LIVE),
    ("dashboard/scripts/refresh_live_leaderboard.py", "dashboard/scripts/refresh_live_leaderboard.py"),
    ("infrastructure/market_data/us_market_calendar.py", _CALENDAR),
    ("domain/backtesting/metrics.py", _METRICS),
    ("js/live-leaderboard.js", "dashboard/frontend/js/live-leaderboard.js"),
    ("tests/test_live_leaderboard.py", f"{_BACKEND}/tests/test_live_leaderboard.py"),
    ("tests/test_frontend_live_leaderboard.py", f"{_BACKEND}/tests/test_frontend_live_leaderboard.py"),
    ("tests/test_live_trade_stats.py", f"{_BACKEND}/tests/test_live_trade_stats.py"),
    ("tests/test_leaderboard_season.py", f"{_BACKEND}/tests/test_leaderboard_season.py"),
    ("tests/test_frontend_live_trading_board.py", _BOARD_TESTS),
    (
        "tests/domain/backtesting/test_stop_prices_not_enforced.py",
        f"{_BACKEND}/tests/domain/backtesting/test_stop_prices_not_enforced.py",
    ),  # Phase 6
    ("tests/test_live_board_records.py", f"{_BACKEND}/tests/test_live_board_records.py"),
    (
        "docs/superpowers/specs/2026-10-14-atl-direction.md",
        "docs/superpowers/specs/2026-10-14-atl-direction.md",
    ),
)


@pytest.mark.parametrize("name,files", _NAMED, ids=[row[0] for row in _NAMED])
def test_every_name_the_live_section_cites_exists(name, files):
    assert name in _section(_SECTION_HEADING), (
        f"_NAMED lists {name!r} but the Two leaderboards section no longer cites it"
    )
    sources = [_REPO / f for f in files]
    assert any(p.is_file() and name in p.read_text(encoding="utf-8") for p in sources), (
        f"CLAUDE.md cites {name!r} but none of {list(files)} contains it -- "
        f"the code was renamed; correct the sentence and this row together"
    )


@pytest.mark.parametrize("cited,path", _PATHS, ids=[row[0] for row in _PATHS])
def test_every_path_the_live_section_cites_exists(cited, path):
    assert cited in _section(_SECTION_HEADING), (
        f"_PATHS lists {cited!r} but the Two leaderboards section no longer cites it"
    )
    assert (_REPO / path).is_file(), f"CLAUDE.md cites {cited} but {path} does not exist"


# The section's one negative claim. Spec §6 said the live reasoning "has been
# stored since #628"; #628 is an admin icon fix, and nothing on the live path
# writes a decision tape or a reasoning trace.
_NO_TAPE_SENTENCE = (
    "live.py writes no decision tape or reasoning trace; the reasoning/trade-log "
    "view plan must add one before that view can exist."
)
# The trace store, and the backtest decision tape that writes through it.
_NO_TAPE_MODULES = (
    "dashboard.backend.domain.traces",
    "dashboard.backend.domain.backtesting.decision_tape",
)


def _imported_modules(path: str) -> set[str]:
    """Every module ``path`` imports, function-level imports included, as an
    absolute dotted name. ``from X import y`` yields both X and X.y, because y
    may be a submodule (``from dashboard.backend.domain import traces``)."""
    package = path[: -len(".py")].replace("/", ".").rsplit(".", 1)[0]
    found: set[str] = set()
    tree = ast.parse((_REPO / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def test_live_py_writes_no_decision_tape_and_the_section_says_so():
    """The reasoning/trade-log view (spec §6) needs a stored decision tape, and
    the live board has none. The section says so in one sentence; the import
    scan is that sentence's tripwire. The day live.py imports the trace store or
    the decision tape, this fails, and the sentence and this test change in the
    same PR."""
    section = " ".join(_section(_SECTION_HEADING).split())
    assert _NO_TAPE_SENTENCE in section, (
        "the Two leaderboards section must say that live.py writes no decision tape"
    )
    tape = sorted(
        module
        for module in _imported_modules(_LIVE)
        if any(module == p or module.startswith(f"{p}.") for p in _NO_TAPE_MODULES)
    )
    assert not tape, (
        f"{_LIVE} now imports {tape}. If the live board writes a decision tape or "
        "reasoning trace, replace the CLAUDE.md sentence that says it does not, "
        "and this test, in the same PR"
    )
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
pytest dashboard/backend/tests/test_live_board_records.py -q -p no:cacheprovider
```

Expected: `64 failed, 11 passed` (75 cases). Measured as `63 failed, 11 passed` of 74 on main@17c552c5, and on main@17c552c5 with Phases 1–6 applied (Phase 3's `CLAUDE.md` bullet included), before #668 added the `model_not_attempted` row, which fails the same way. The count does not depend on Phases 1–6, because every name and path case fails first on the section text. The failures are:
- the six `test_retired_live_board_claims_are_gone[...]` cases, each `AssertionError: CLAUDE.md still says '<phrase>'`;
- `test_the_refresh_secret_bullet_names_both_workflows`: `the LEADERBOARD_DAILY_REFRESH_SECRET bullet must name .github/workflows/live-leaderboard.yml`;
- `test_the_auto_deploy_bullet_points_at_the_live_gate`: an `AssertionError` on `'LIVE_LEADERBOARD_DEPLOY_MODELS' in bullet`;
- `test_the_live_deploy_variable_has_its_own_bullet`: `expected exactly one CLAUDE.md bullet for LIVE_LEADERBOARD_DEPLOY_MODELS, found 0`;
- `test_the_section_points_at_the_budget_bullet_instead_of_restating_it`: `assert 'LEADERBOARD_MONTHLY_BUDGET_USD' in '\n### Two leaderboards, and what ...'`. If it fails instead with `expected exactly one CLAUDE.md bullet for LEADERBOARD_MONTHLY_BUDGET_USD, found 0`, Phase 3 has not merged and Step 0 should have stopped you: stop;
- `test_the_daily_workflow_no_longer_claims_the_live_advance`: `assert 'season engi...is same call' not in 'name: Daily...'` (pytest's truncation of the phrase and the workflow text);
- 43 `test_every_name_the_live_section_cites_exists[...]` and 9 `test_every_path_the_live_section_cites_exists[...]` cases, each `_NAMED lists '<name>' but the Two leaderboards section no longer cites it` (or `_PATHS lists ...`);
- `test_live_py_writes_no_decision_tape_and_the_section_says_so`: `AssertionError: the Two leaderboards section must say that live.py writes no decision tape`. Its import scan already passes: `live.py` imports neither `domain/traces` nor the decision tape, on main@17c552c5 or after Phases 1–6.

The 11 that already pass are the names the old section shares with the new one: `LIVE_LEADERBOARD_DEPLOY_MODELS`, `build_season_payload`, `_preview_season_dates`, `_season_trading_days`, `isLivePreview`, `seasonHasAdvanced`, `displayedSeasonNumber`, and the paths `domain/leaderboard/live.py`, `tests/test_leaderboard_season.py`, `tests/test_frontend_live_trading_board.py`, `docs/superpowers/specs/2026-10-14-atl-direction.md`.

- [ ] **Step 3: Implement**

Four edits to `CLAUDE.md` and one to the daily workflow. Each find block is verbatim from main@17c552c5 and occurs exactly once there and on main@17c552c5 with Phases 1–6 applied; no Phase 1–6 plan edits these passages (Phase 3 only inserts its own bullet after `LEADERBOARD_DAILY_AUTO_DEPLOY`, which 3c accounts for). If a find block does not match anyway, re-read the passage, apply the same intent to the current text, and keep the replacement wording below.

**3a.** In `CLAUDE.md`, inside the `LEADERBOARD_DAILY_REFRESH_SECRET` bullet, find:

```markdown
Driven by `.github/workflows/daily-leaderboard.yml` and by `dashboard/scripts/refresh_daily_leaderboard.py --remote`. **That workflow's `schedule:` is commented out as of PR #352** (2026-08-15) — the Daily Leaderboard tab it fed was replaced by the **Live Trading Leaderboard**, so on a schedule it would keep deploying all seven competition LLMs nightly, billable, for a board nobody can open. Only `workflow_dispatch` is live, and its `deploy_models` input now defaults to **false** for the same reason. The route, the secret and the 22:30 UTC Mon–Fri cron line all stay in the file because the season engine's nightly advance is this same call — re-enable by uncommenting the block, don't rewrite it. Set the secret in the **Render dashboard** *and* as a repository secret before enabling the workflow.
```

Replace with:

```markdown
**Two workflows send this one secret, and only one runs.** `.github/workflows/daily-leaderboard.yml` (and `dashboard/scripts/refresh_daily_leaderboard.py --remote`) calls the daily route; its `schedule:` has been commented out since PR #352 (2026-08-15), because the Daily Leaderboard tab it fed is retired and on a schedule it would deploy all seven competition LLMs nightly, billable, for a board nobody can open. Only its `workflow_dispatch` is live, and that input's `deploy_models` defaults to **false** for the same reason. `.github/workflows/live-leaderboard.yml` is the active one: cron `30 22 * * 1-5` (22:30 UTC Mon–Fri, after the 16:00 ET close) calls `POST /api/v1/leaderboard/live/refresh`, which checks this same secret through `verify_daily_refresh_secret` and has its own 20/hour limiter. That job bills only when the repository variable `LIVE_LEADERBOARD_DEPLOY_MODELS` is exactly `'true'` (bullet below); otherwise it refreshes baselines only. The Live Leaderboard's nightly advance is that live call, not the daily one, so re-enabling the daily schedule restores nothing and spends money. Set the secret in the **Render dashboard** *and* as a repository secret; the live job fails fast with "LEADERBOARD_DAILY_REFRESH_SECRET is not configured" when the repository copy is missing.
```

**3b.** In `CLAUDE.md`, inside the `LEADERBOARD_DAILY_AUTO_DEPLOY` bullet, find:

```markdown
Prod previously relied on the nightly refresh job instead — **but that schedule is paused as of PR #352** (see `LEADERBOARD_DAILY_REFRESH_SECRET` above), so prod currently runs *neither* this flag nor the cron. Nothing refreshes the daily board automatically today, by design: the Live Trading Leaderboard that replaced its tab is a Season 0 preview with no advance engine yet. Do not "restore" automation here to close the gap — arming this flag is the anonymous-billable-request path the strict opt-in exists to prevent.
```

Replace with:

```markdown
Prod previously relied on the nightly daily refresh job instead — **but that schedule is paused as of PR #352** (see `LEADERBOARD_DAILY_REFRESH_SECRET` above), so prod runs *neither* this flag nor the daily cron, and nothing refreshes the daily board automatically, by design: its tab is retired. The board that does advance nightly is the Live Leaderboard, through its own cron and its own billing gate (`LIVE_LEADERBOARD_DEPLOY_MODELS`, below); this flag has nothing to do with it. Do not "restore" automation here — arming this flag is the anonymous-billable-request path the strict opt-in exists to prevent.
```

**3c.** In `CLAUDE.md`, add the new bullet directly after the `LEADERBOARD_DAILY_AUTO_DEPLOY` bullet. Phase 3 put its `LEADERBOARD_MONTHLY_BUDGET_USD` bullet directly after that one, so the new bullet lands between the two, and the order becomes `LEADERBOARD_DAILY_AUTO_DEPLOY`, `LIVE_LEADERBOARD_DEPLOY_MODELS`, `LEADERBOARD_MONTHLY_BUDGET_USD`, `NEWS_FEED_FROZEN`. Find the last sentence of the `LEADERBOARD_DAILY_AUTO_DEPLOY` bullet:

```markdown
Note the in-progress guard and the per-window state file (`storage/data/leaderboard_daily_refresh.json`) are **per-process**: fine for the current single-instance Render deploy, but a multi-replica deploy would duplicate model deploys.
```

Replace with (the same sentence, a newline, then the new bullet):

```markdown
Note the in-progress guard and the per-window state file (`storage/data/leaderboard_daily_refresh.json`) are **per-process**: fine for the current single-instance Render deploy, but a multi-replica deploy would duplicate model deploys.
- `LIVE_LEADERBOARD_DEPLOY_MODELS` (**GitHub repository variable, not a Render env var**; `true` since 2026-09-29, check with `gh variable list --repo Open-Finance-Lab/AgenticTrading`): the billing gate of the Live Leaderboard's nightly cron. `.github/workflows/live-leaderboard.yml` sends `deploy_models=true` to `POST /api/v1/leaderboard/live/refresh` only when the variable is exactly the string `'true'` (`vars.LIVE_LEADERBOARD_DEPLOY_MODELS == 'true'`); any other value, unset included, refreshes baselines only. A `workflow_dispatch` run ignores it and uses its own `deploy_models` input, which defaults to false. With it on, every weekday each live LLM trades the newest session through `COMMONSTACK_API_KEY`, outside ATL Credits, bounded by `LEADERBOARD_MONTHLY_BUDGET_USD`. The server never reads this variable, so a deployment without the workflow (a fork, a self-host) never bills on a schedule. `tests/test_live_leaderboard.py::test_workflow_schedule_only_bills_behind_an_explicit_variable` pins the gate.
```

**3d.** In `CLAUDE.md`, replace the whole section from its heading through its last line. Find (79 lines, ≈:213–291):

```markdown
### Two leaderboards, and what `live` does *not* mean

> ⚠ **Partly stale — read `docs/superpowers/specs/2026-10-14-atl-direction.md` first.**
>
> **What changed since this section was written:**
> - Since 09-26, `domain/leaderboard/live.py` (#534) has advanced a calendar-month board nightly on the legacy engine.
> - Since 2026-09-29, `LIVE_LEADERBOARD_DEPLOY_MODELS=true`, so it bills DeepSeek through `COMMONSTACK_API_KEY` every weekday.
>
> **Decisions made on 2026-10-14:**
> - Seasons are calendar months, not two weeks.
> - Season 1 waits for the new engine.
> - The tab becomes "Live Leaderboard".
>
> The "Season 0 preview / nothing advances" description below predates all of this.

Since PR #352 the dashboard serves **two** boards: the **Competition Leaderboard** (one fixed
historical window, the acquisition hook) and the **Live Trading Leaderboard**, which is meant to
advance one session at a time in **two-week seasons**. The old Daily Leaderboard tab is retired.
Neither board takes user entries — `get_leaderboard` builds every row from the curated
`strategies` roster in `dashboard/config/leaderboard.json`, and `api/routers/leaderboard.py`
exposes no submission route. Don't write copy that implies otherwise.

**`live` is a real period; the season engine still does not exist.** As of PR #386
`VALID_PERIODS = ("contest", "daily", "live")` (`domain/leaderboard/service.py:50`) and
`GET /api/v1/leaderboard?period=live` answers with `period: "live"`, its own
`window.description`, and a `season` block (`build_season_payload`, `:621`). `_normalize_period`
(`:91-93`) still coerces genuinely unrecognised periods back to `contest` rather than 4xx-ing
them. **Nothing advances anything** — the block is hardcoded to the not-yet-advanced state — so
the tab renders a deliberate **Season 0 preview**: real Competition curves under season chrome,
plus a banner saying nothing here has advanced. `season` remains vocabulary the *engine* has yet
to earn.

**The live board reuses the contest session and window byte-for-byte, and that is a spend
control.** `resolve_leaderboard_config("live")` spreads the contest base so `_find_cached_run`
hits cache on all twelve entries; inventing a window would miss on every one and start
recomputing baselines — and, with `LEADERBOARD_DAILY_AUTO_DEPLOY` armed, billable LLM deploys —
from a public unauthenticated GET. The independent second lock is
`maybe_schedule_daily_leaderboard_refresh()` being gated on `period == "daily"`. Neither was
written as a backstop for the other.

⚠ **The preview banner is anchored on evidence of an advance, not on the period — keep it that
way.** `isLivePreview()` is `isLiveBoard() && !seasonHasAdvanced(payload)`, and
`seasonHasAdvanced()` (`js/leaderboard.js:312`) tests `season.last_advanced_date` /
`trading_days_elapsed > 0`: fields only a real nightly advance can write. It was originally
`payload.period !== 'live'`, which meant the season engine's most natural first commit — adding
`"live"` to `VALID_PERIODS`, needing no season payload at all — would have silently cleared every
banner while nothing had run. **That commit has now landed and changed nothing here, which was
the point.** Do not now "simplify" it to a period check on the grounds that the period is finally
real; it is real and still says nothing about an advance.

⚠ **Season 0 is falsy, and the number has exactly one owner.** Every read goes through
`displayedSeasonNumber()` + `Number.isFinite` (`:341`), because `season?.number ? … : '—'`
renders the shakedown season as no season at all; every *rendered* mention goes through
`displayedSeasonLabel()` (`:359`). `PREVIEW_SEASON_NUMBER` exists on both sides, but the JS copy
is only the fallback inside `displayedSeasonNumber` — the payload's `season.number` decides.
Interpolating the JS constant into copy gave the badge and the banner separate owners that
disagreed the moment the server's constant moved. `phase_label` is `f"Season {…}"` for the same
reason.

⚠ **The preview season's window expires by itself, on purpose.** Nothing advances Season 0, so
`season_zero_start` + `length_trading_days` describes a fortnight that becomes a *past* fortnight
with no code noticing. `_preview_season_dates` (`:566`) therefore returns `(None, None)` once the
window has elapsed — and also when `season_zero_start` is absent or unparseable — which lands on
copy the client already ships ("Dates set when the first season opens") and prints a one-time
operator log line. Do **not** restore the old `or config["start_date"]` fallback: it published
the *contest* window (2026-04-15 → 2026-04-28) as a season, so "nobody configured a season" and
"the season is the April contest window" came back byte-identical, `status: preview`, HTTP 200 —
the exact shape the fail-closed-is-not-fail-visible section below is about.

The season length is untrusted config on a public GET: `_season_trading_days` (`:494`) parses and
clamps it **once** to `1..MAX_SEASON_TRADING_DAYS`, and the payload reports the clamped value.
Clamping inside `season_window` while publishing the raw number let a negative length render a
**100%-full** progress bar (`elapsed / total`, both negative) directly under the banner denying
that anything advanced.

All of the above is pinned by tests: `tests/test_leaderboard_season.py` (the server contract),
`tests/test_leaderboard_api.py` (the route), and `tests/test_frontend_live_trading_board.py`
(source-shape guards — the only kind that can catch the falsy-season and two-owners bugs, since a
test passing period `"live"` or season `3` passes anyway).
```

Replace with:

```markdown
### Two leaderboards: Competition and Live

> **Read `docs/superpowers/specs/2026-10-14-atl-direction.md` first** (§4 credibility bar, §6 Live Leaderboard, §7 budget). It supersedes every earlier live-board record; the dated 08-09, 08-15 and 08-19 specs and plans carry a banner saying so. Decided there on 2026-10-14: seasons are calendar months; Season 1 waits for the new engine **and** the methodology page, and until then every month is "Season 0 · preview"; the Competition board is retired once Season 1 has data, not re-run. Everything below describes what runs today, on the legacy engine.

The dashboard serves two boards; the Daily Leaderboard tab was retired in #352. Neither takes user entries: every row comes from the curated `strategies` roster in `dashboard/config/leaderboard.json`, and `api/routers/leaderboard.py` exposes no submission route. Don't write copy that implies otherwise.

**Competition Leaderboard** (`?period=contest`, session `leaderboard-contest`): one fixed historical window (`start_date`/`end_date` in `leaderboard.json`), built by `service.get_leaderboard` and labelled "legacy engine · historical window · may overlap model training data". Repeat runs publish their median (see the H6 paragraph above).

**Live Leaderboard** (`?period=live`, `domain/leaderboard/live.py`, #534, advancing nightly since 2026-09-26) is a calendar-month board:

- **Dispatch.** `live.py` registers `get_live_leaderboard` through `lb_service.register_period_board("live", …)`, and the router calls `get_live_leaderboard()` directly for `?period=live`, so `?refresh=true` never recomputes it. With no builder registered, `service.get_leaderboard(period="live")` raises `RuntimeError("Live leaderboard module is not loaded")` rather than building the old contest-window preview, which answered 200 with the wrong board. Do not add a fallback.
- **GET is read-only.** `get_live_leaderboard` serves the newest stored freeze row per entry (`latest_live_month_runs`) and never fetches bars or calls a model; computing on a public GET would miss the run cache on the first request of every day. Every row is written by `refresh_live_leaderboard`, reached through `POST /api/v1/leaderboard/live/refresh` (the `live-leaderboard.yml` cron), `enqueue_live_leaderboard_refresh` and `dashboard/scripts/refresh_live_leaderboard.py`.
- **Storage.** Session `LIVE_SESSION_ID` (`"leaderboard-live"`), mode `leaderboard`, baselines and models alike: each freeze writes one `agent_runs` row per entry with `start_date` = month-open and `end_date` = the freeze, holding the whole month-to-date curve, and `prune_superseded_live_runs` keeps only the newest. Durable only where `AGENT_RUNS_DATABASE_URL` is set (it is in prod). `_LIVE_REFRESH_STATE_PATH` (`leaderboard_live_refresh.json` under `storage/data/`) sits on the ephemeral disk and only records which window the last refresh satisfied, so a redeploy costs one redundant baseline refresh and nothing else. Never put state that must survive there; that is why the spend ledger lives in `agent_runs`.
- **Increments and checkpoints.** A model increment restores cash, positions and its trade memory (the 24 hours up to its newest trade, at most 200) from the newest complete checkpoint, and trades only the unseen sessions. The trades ride the snapshot as `recent_trades` and reach the prompt through `PortfolioManager.restored_trades`; they are never copied into `self.trades`, because `fold_trade_stats` folds that list nightly and would count them twice. The snapshot also carries `trade_memory` (`TRADE_MEMORY_VERSION`) and `trade_memory_since`, the first bar from which the chain of snapshots has carried trade memory. A restore keeps that stamp, so the session where memory began stays identifiable after its own checkpoint is pruned. Every session close whose running coverage clears H6 writes its own checkpoint (`_checkpoint_passes_h6`, `_write_live_checkpoint`, since #582); a prefix that fails H6 is never checkpointed. Only an entry with no checkpoint this month, a `force` replay, or a config-change replay (`WARNING: live.config_changed`, keyed on `_RESUME_CONFIG_KEYS`) starts again from the 1st. A session with no bars fails the entry loudly (`ERROR: live.session_gap`) instead of being skipped, and `refresh_live_leaderboard` logs each entry's failure (`ERROR: live.model_deploy_failed`) and carries on with the rest. The exception is two entries in a row failing with the same error: that fault is shared, so the rest are logged `ERROR: live.model_not_attempted` instead of being billed for it. Either way the refresh still prunes and saves state.
- **Cadence.** Six model decisions per full NYSE session, on hourly bars stamped 10:00–15:00 ET; three on a 13:00 early close. Early closes are a rule table, `nyse_early_closes` in `infrastructure/market_data/us_market_calendar.py`, pinned by tests to the exchange's published 2025–2028 dates; `sessions.is_in_session` applies it on its open-stamped branch only, and the close-stamped path is left to the engine rewrite. A session is computed nightly once its closing bar has settled, and the board says "Nightly after the close" rather than printing a request clock.
- **Roster.** `LIVE_MODEL_IDS` (DeepSeek V4 Pro, Qwen3.7 Plus, Claude Haiku 4.5) plus every baseline. An entry with `live_from: "YYYY-MM-01"` joins from that month on; a malformed value warns once and is ignored. An entry's `live_overrides` block (`model_id`, `integration`, `temperature`, `reasoning_effort`) applies on the Live board only: that is how Haiku trades Live as `anthropic/claude-haiku-4-5` at temperature 0 while its shared keys stay as its Competition runs recorded them. Do not move those values into the shared keys: `_entry_config_drift` would mark Haiku's Competition repeats stale, and the board would publish its single July run in place of their median. Nemotron left the roster because CommonStack serves no NVIDIA model. A dropped model's rows stay where they are (GET renders roster entries only), so a re-add resumes rather than replays.
- **Spend.** Model calls go through `COMMONSTACK_API_KEY` and bypass ATL Credits. They happen only with `deploy_models=True`, which the cron sends only when `LIVE_LEADERBOARD_DEPLOY_MODELS` is `"true"`, and they stop at `LEADERBOARD_MONTHLY_BUDGET_USD` (both bullets above). Every attempt (successful, failed, H6-refused or replayed) is priced into the attempt ledger: `agent_runs` rows in session `leaderboard-live-spend`, mode `leaderboard_spend`, which `clear_live_session_runs` does not delete. The board, `_find_cached_run`, `GET /api/backtest/runs` and `/api/v2` never return them. The public `GET /runs?mode=leaderboard_spend` does list them, through the `RunMetadata` allow-list: run id, the entry's name and id (`agent_name`, `llm_model`), dates, `created_at`, `initial_equity` 0, the cumulative `llm_calls` and derived decision-provenance labels, with no `est_cost_usd`, no token counts and no `metadata` (so no `last_attempt`). `GET /api/backtest/{run_id}` returns one only to a caller that sends `X-Session-Id: leaderboard-live-spend`, as an empty curve with null return metrics, and `/runs/{run_id}` rejects that header because it is not a UUID. None of these routes exposes dollars or tokens from the ledger.
- **Payload.** `live_status`, never `season`: `frozen_through`, `snapshot_end`, `catching_up` (entries more than one session behind, shown but unranked), `snapshot_stale`, `roster` and `budget_halted`; each model entry also carries `last_attempt` (`on`, `outcome`, `kind`; no error text). The payload adds no new dollar figure; per-entry `est_cost_usd` predates this plan (`docs/superpowers/plans/2026-10-14-live-board-fixes.md`). The title and the season label have one owner each, `LIVE_BOARD_TITLE` and `LIVE_PHASE_LABEL` in `service.py`, sent as `board_title` and `phase_label`. The browser half is `js/live-leaderboard.js`.
- **Sharpe, on both boards.** Daily, computed at read time from the stored curve by `daily_sharpe_interval` (`domain/backtesting/metrics.py`): `sharpe_ratio` (null unless `sharpe_status` is `"ok"`, which needs at least 5 daily returns that moved), `sharpe_ci_low`/`sharpe_ci_high` (95%, iid standard error), `sharpe_days` and `sharpe_status`. The definition behind them, `daily_sharpe_basis()`, rides the payload once per board: `live_status.sharpe` on Live, top-level `sharpe_basis` on Competition. The browser displays them and never recomputes. The stored `agent_runs.sharpe_ratio` column is still the old hourly figure from `calc_metrics`, and neither board shows it.
- **Stops.** Models may write `stop_loss_price`/`take_profit_price`; nothing executes them, and both boards say so. Remove those disclaimers, and flip `tests/domain/backtesting/test_stop_prices_not_enforced.py` and the `_STOP_RULE` pins in `tests/test_frontend_live_trading_board.py`, in the PR where the new core executes stops (spec §4).
- **No decision tape.** live.py writes no decision tape or reasoning trace; the reasoning/trade-log view plan must add one before that view can exist.

**Legacy season-preview code: do not build on it.** `build_season_payload`, `_preview_season_dates` and `_season_trading_days` in `service.py`, the `"live"` branch of `resolve_leaderboard_config` (still the contest window), the `season` block in `leaderboard.json`, and the `isLivePreview()`/`seasonHasAdvanced()`/`displayedSeasonNumber()` branches of `js/leaderboard.js` describe the retired two-week preview. None is on a served path: `get_leaderboard` never reaches its `payload["season"]` line, and the Live tab renders through `js/live-leaderboard.js`. They survive only because `tests/test_leaderboard_season.py` and `tests/test_frontend_live_trading_board.py` pin them; remove each together with its pins rather than reviving it. Two lessons from that code still bind the live board: a "preview" or "has advanced" claim must rest on evidence that an advance ran, never on the period name; and a number rendered in copy has exactly one owner.

Pinned by `tests/test_live_leaderboard.py` (server), `tests/test_frontend_live_leaderboard.py` (live JS; node-gated), `tests/test_live_trade_stats.py`, and `tests/test_live_board_records.py`, which fails when a name this section cites stops existing or when `live.py` starts importing the trace store or the decision tape.
```

Keep the blank line that separates the section from `### LLM safety boundary`.

**3e.** In `.github/workflows/daily-leaderboard.yml`, find:

```yaml
# SCHEDULE PAUSED. The Daily Leaderboard tab was replaced by the Live Trading
# Leaderboard, so this job's output has no surface — left on schedule it would keep
# deploying every competition LLM nightly, billable, for a board nobody can
# open. The endpoint, the secret and workflow_dispatch all stay, because the
# season engine's nightly advance is this same call.
# To re-enable: uncomment the `schedule:` block below.
```

Replace with (the comment line that follows, the one citing #335, stays as it is):

```yaml
# SCHEDULE PAUSED. The Daily Leaderboard tab is retired, so this job's output
# has no surface — left on schedule it would keep deploying every competition
# LLM nightly, billable, for a board nobody can open. The Live Leaderboard's
# nightly advance is a different call: live-leaderboard.yml POSTs
# /api/v1/leaderboard/live/refresh with the same secret, so re-enabling this
# schedule feeds nothing on screen. The endpoint and workflow_dispatch stay for
# manual daily-board runs. To re-enable: uncomment the `schedule:` block below.
```

- [ ] **Step 4: Run it and confirm it passes**

```bash
pytest dashboard/backend/tests/test_live_board_records.py -q -p no:cacheprovider
```

Expected: `75 passed`. Measured as `74 passed` on main@17c552c5 with Phases 1–6 applied (Phase 3's `CLAUDE.md` bullet included), before #668 added a row; the env bullets came out in the order 3c gives.

If this step fails, read the failures against the 18 Phase-owned cases: the 16 `_NAMED` rows commented `# Phase N`, the `_PATHS` row for `tests/domain/backtesting/test_stop_prices_not_enforced.py`, and `test_the_section_points_at_the_budget_bullet_instead_of_restating_it` (`found 0`). With these edits applied to bare main@17c552c5 the result was exactly those, `18 failed, 56 passed` (measured before #668 added a row), the name rows failing with `none of [...] contains it` and the path row with `does not exist`. The two `# #668` rows pass on any main since 2026-10-30. Any of them here means that phase is not on this branch and Step 0 was skipped: stop, wait for the phase, then `git rebase origin/main`. A failure with `none of [...] contains it` for a name whose phase did merge is the rename case from Step 0: fix the sentence and the `_NAMED` row together. Never delete a row to get green. If `test_retired_live_board_claims_are_gone[Live Trading Leaderboard]` still fails, a Phase 1–6 PR wrote the old name into `CLAUDE.md`: change that occurrence to "Live Leaderboard".

Neighbouring suites (other readers of `CLAUDE.md` and of both workflow files):

```bash
pytest dashboard/backend/tests/integrations/test_docs_run_command.py dashboard/backend/tests/integrations/test_discord_wiring.py dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/test_frontend_live_trading_board.py -q -p no:cacheprovider
```

Expected: all pass (the discord module may skip if `discord` is not installed).

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # must print docs/live-board-records
git add CLAUDE.md .github/workflows/daily-leaderboard.yml dashboard/backend/tests/test_live_board_records.py
git commit -m "docs: describe the shipped live board in CLAUDE.md"
```

---

### Task 7.2: Mark the dated live-board records superseded (PR A)

**Files:**
- Modify: `docs/superpowers/specs/2026-08-09-participatory-competition-design.md`
- Modify: `docs/superpowers/plans/2026-08-09-participatory-competition-phase-0-1.md`
- Modify: `docs/superpowers/specs/2026-08-15-live-trading-leaderboard-ui.md`
- Modify: `docs/superpowers/specs/2026-08-19-nof1-leaderboard-frame-design.md`
- Modify: `docs/superpowers/plans/2026-08-19-nof1-leaderboard-frame.md`
- Modify: `docs/superpowers/notes/2026-09-10-user-facing-docs-backlog.md` (its entry 5 still says "the season engine does not exist yet")
- Test: `dashboard/backend/tests/test_live_board_records.py`

These are dated records: each gets one banner line under its title and is otherwise left as written. This task does not touch `docs/superpowers/specs/2026-10-14-atl-direction.md`: its Errata block lands in the plan's own docs PR, before P1. The 08-15 chart-first landing spec and plan mention "the season engine (issue #354)" only in out-of-scope lists and are not bannered; `docs/landing-narrative-copy.md` already carries its own "kept for the rationale" status note.

**Interfaces:**
- Consumes: `_REPO` and `pytest` from Task 7.1's module.
- Produces: `_SUPERSEDED`, `test_dated_live_board_records_carry_the_superseded_banner`, and the banner line on six files.

- [ ] **Step 1: Write the failing test**

Append to the end of `dashboard/backend/tests/test_live_board_records.py`, two blank lines after the last function:

```python
_SUPERSEDED = (
    "docs/superpowers/specs/2026-08-09-participatory-competition-design.md",
    "docs/superpowers/plans/2026-08-09-participatory-competition-phase-0-1.md",
    "docs/superpowers/specs/2026-08-15-live-trading-leaderboard-ui.md",
    "docs/superpowers/specs/2026-08-19-nof1-leaderboard-frame-design.md",
    "docs/superpowers/plans/2026-08-19-nof1-leaderboard-frame.md",
    "docs/superpowers/notes/2026-09-10-user-facing-docs-backlog.md",
)


@pytest.mark.parametrize("path", _SUPERSEDED)
def test_dated_live_board_records_carry_the_superseded_banner(path):
    """Dated records are history and are not rewritten; a banner is how a reader
    who lands on one learns the live board shipped. Five lines, so the banner
    has to sit directly under the title rather than anywhere in the file."""
    head = "\n".join((_REPO / path).read_text(encoding="utf-8").splitlines()[:5])
    assert "**Superseded for the live board**" in head, f"{path} has no superseded banner"
    assert "docs/superpowers/specs/2026-10-14-atl-direction.md" in head
    assert "#534" in head
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
pytest dashboard/backend/tests/test_live_board_records.py -q -p no:cacheprovider -k banner
```

Expected: `6 failed, 75 deselected`, each `AssertionError: <path> has no superseded banner`.

- [ ] **Step 3: Implement**

The banner, verbatim (one line):

```markdown
> **Superseded for the live board** by `docs/superpowers/specs/2026-10-14-atl-direction.md`: `domain/leaderboard/live.py` shipped in #534 and has advanced a calendar-month board nightly since 2026-09-26. Read this file's live-board passages as history.
```

In each of the six files, find its first line (the H1, which occurs once per file) and replace it with the H1, a blank line, and the banner. The H1 is already followed by a blank line, so the banner ends up framed by blank lines.

1. `docs/superpowers/specs/2026-08-09-participatory-competition-design.md`, find `# Participatory competition: two boards users can enter`
2. `docs/superpowers/plans/2026-08-09-participatory-competition-phase-0-1.md`, find `# Participatory Competition — Phase 0 (Gate) + Phase 1 (Evidence) Implementation Plan`
3. `docs/superpowers/specs/2026-08-15-live-trading-leaderboard-ui.md`, find `# Live Trading Leaderboard — UI and payload contract`
4. `docs/superpowers/specs/2026-08-19-nof1-leaderboard-frame-design.md`, find `# nof1-style leaderboard frame + live data on the landing hero`
5. `docs/superpowers/plans/2026-08-19-nof1-leaderboard-frame.md`, find `# nof1 Leaderboard Frame + Live Landing Hero — Implementation Plan`
6. `docs/superpowers/notes/2026-09-10-user-facing-docs-backlog.md`, find `# User-facing docs backlog — opened 2026-09-10`

Worked example for file 3. Find:

```markdown
# Live Trading Leaderboard — UI and payload contract
```

Replace with:

```markdown
# Live Trading Leaderboard — UI and payload contract

> **Superseded for the live board** by `docs/superpowers/specs/2026-10-14-atl-direction.md`: `domain/leaderboard/live.py` shipped in #534 and has advanced a calendar-month board nightly since 2026-09-26. Read this file's live-board passages as history.
```

- [ ] **Step 4: Run it and confirm it passes**

```bash
pytest dashboard/backend/tests/test_live_board_records.py -q -p no:cacheprovider
```

Expected: `81 passed` (measured as `80 passed` on main@17c552c5 with Phases 1–6 applied, before #668 added a row). The six banner cases pass whether or not Phases 1–6 have merged (on bare main@17c552c5, before #668: `18 failed, 62 passed`, the 18 being Task 7.1 Step 4's Phase-owned cases); any other failure is one of Task 7.1 Step 4's cases.

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # must print docs/live-board-records
git add docs/superpowers/specs/2026-08-09-participatory-competition-design.md docs/superpowers/plans/2026-08-09-participatory-competition-phase-0-1.md docs/superpowers/specs/2026-08-15-live-trading-leaderboard-ui.md docs/superpowers/specs/2026-08-19-nof1-leaderboard-frame-design.md docs/superpowers/plans/2026-08-19-nof1-leaderboard-frame.md docs/superpowers/notes/2026-09-10-user-facing-docs-backlog.md dashboard/backend/tests/test_live_board_records.py
git commit -m "docs: mark the dated live-board records superseded"
```

PR A is complete. Open it now (Task 7.5, part A); it does not wait for PR B.

---

### Task 7.3: Landing: rename the board and show Sharpe with its interval (PR B, owner-coordinated)

**Execute only after the owner approves the copy in Step 0.**

**Files:**
- Modify: `dashboard/landing/src/components/home/Race.tsx` (≈:27, ≈:150–162)
- Modify: `dashboard/landing/src/components/home/BoardPreview.tsx` (`RANK_GRID` ≈:88–112, Sharpe header ≈:647–652, Sharpe cell ≈:784–786)
- Modify: `dashboard/landing/src/lib/leaderboard.ts` (`LeaderboardEntry` ≈:19–25, `BoardStanding` ≈:62–63, `formatSharpe` ≈:157–170, `buildBoardData` ≈:249–252 and ≈:310–329, `fetchLeaderboard` ≈:507)
- Modify: `dashboard/frontend/index.html` (≈:274–275), `dashboard/frontend/assets/` (rebuilt bundle; superseded `index-*.js`/`index-*.css` deleted)
- Test: `dashboard/backend/tests/test_landing_copy_register.py`, `dashboard/backend/tests/test_landing_live_board.py`, `dashboard/backend/tests/test_landing_chart_first.py`

**Interfaces:**
- Consumes: Phase 5's per-entry `sharpe_ratio`, `sharpe_ci_low`, `sharpe_ci_high`, `sharpe_days`, `sharpe_status` and the Competition payload's top-level `sharpe_basis` (`{basis, annualization_days, confidence, min_days, rf_annual, stderr}`, written in `service.py` as `"sharpe_basis": daily_sharpe_basis(),`); Phase 5 Task 5.4's rewritten docstring in `test_landing_live_board.py` (the Step 1d anchor); Phase 6's `about_at = visible.index("Live Leaderboard</h3>")` in `test_landing_copy_register.py`; the conformance owner decision 5 tooltip wording.
- Produces: `SharpeBasis`, `SharpeDisplay`, `formatSharpe(entry, basis)`, `BoardStanding.sharpeInterval`, `BoardStanding.sharpeDetail`, `RANK_GRID` with a 128px Sharpe track, a rebuilt landing bundle.

- [ ] **Step 0: Gate on the owner, the deploy and the payload key**

1. Send the owner these exact strings and the layout choice, and wait for approval:
   - Race rule bullet: `Live Leaderboard: moves forward one trading session at a time, over the current calendar month.`
   - Race note: `The Live Leaderboard is in preview for Season 0: the current calendar month, updated nightly after the close. Read its standings as a preview, not an official result.`
   - Sharpe column header tooltip: `Return above what cash would have earned, divided by how much the daily returns swung, scaled to a yearly figure. The smaller figures are its 95% interval; hover a value for the trading days and cash rate behind it.`
   - Sharpe cell, one line: `2.61` followed by a smaller `-11.4 to 16.6`. Row tooltip: `95% interval -11.40 to 16.62. Based on 5 trading days; cash rate 0%/yr; scaled to a yearly figure.` Below 5 days: `—` with tooltip `Needs 5 trading days (have 3)`. Flat curve: `—` with tooltip `Returns never moved`.
   - Layout: the Sharpe track widens from 62px to 128px at `sm` and above, so value and interval share one line and every row keeps its 28px height. The name column pays for it and truncates sooner between 640px and 1023px. (Alternative if the owner rejects that: keep 62px and carry the interval in the row tooltip only. That breaks spec §4's "always with its 95% interval", so it needs the owner's explicit call.)
   - The RST strings of Task 7.4 Step 3.
   - The landing does not repeat the app's "no real money" subtitle: `test_landing_copy_register.py::_BROKERED_CLAIM_PATTERNS` bans `real (capital|money|…)` in landing components, comments included, and the hero's standing line already says it.
2. Confirm Render serves Phase 5's fields (public GET; the landing reads exactly this):

```bash
curl -s "https://agentictrading.onrender.com/api/v1/leaderboard?period=contest" | jq '{basis: .sharpe_basis, first: (.entries[0] | {sharpe_ratio, sharpe_ci_low, sharpe_ci_high, sharpe_days, sharpe_status})}'
```

   Expected: `basis` is an object with `rf_annual` and `min_days`, and `first.sharpe_status` is a string. If `basis` is `null` while `sharpe_status` is present, Phase 5 shipped the basis under another key. Find it with `grep -n "daily_sharpe_basis()" dashboard/backend/domain/leaderboard/service.py`, which prints the line `"<key>": daily_sharpe_basis(),`. (Grepping `service.py` for `daily_returns` finds nothing: that literal lives in `metrics.py`.) Substitute that key for `sharpe_basis` everywhere in this task: `leaderboard.ts` edits 3e (the `SharpeBasis` doc comment), 3f and 3g, and the two tests in 1e that name it (`test_the_board_carries_the_payload_basis_into_every_row`, `test_the_basis_key_the_landing_reads_is_the_one_the_server_writes`). Note it in the PR body. If `sharpe_status` is missing, Phase 5 is not deployed: stop.
3. Start the branch and confirm Phases 5 and 6 are in it:

```bash
git fetch origin main
git switch -c docs/live-board-public-copy origin/main
git branch --show-current   # must print docs/live-board-public-copy
grep -n '"sharpe_basis"' dashboard/backend/domain/leaderboard/service.py                    # Phase 5: one line
grep -c "Sharpe's em dash IS reachable through the API now" dashboard/backend/tests/test_landing_live_board.py   # Phase 5: 1
grep -c 'about_at = visible.index("Live Leaderboard</h3>")' dashboard/backend/tests/test_landing_copy_register.py  # Phase 6: 1
cd dashboard/landing && { [ -d node_modules ] || npm ci; } && cd ../..
```

   A missing line or a `0` means that phase is not on `origin/main`: stop. Install `node_modules` with `npm ci`, not a symlink to another checkout. `.gitignore`'s `node_modules/` matches directories only, so a symlink shows as untracked (`?? dashboard/landing/node_modules`). If you used one anyway, never `git add` it; Step 5 adds explicit paths for that reason.

- [ ] **Step 1: Write the failing tests**

**1a.** In `dashboard/backend/tests/test_landing_copy_register.py`, inside `test_race_names_the_two_boards_the_app_actually_serves`, find:

```python
    text = _shipped_text()
    assert "Live Trading Leaderboard" in text
    assert "Competition" in text
```

Replace with:

```python
    text = _shipped_text()
    assert "Live Leaderboard" in text
    assert "Competition" in text
    # Renamed 2026-10-14 (direction spec §6): "Trading" read as real orders and
    # collided with the Robinhood live-trading docs.
    assert "Live Trading Leaderboard" not in text
```

**1b.** In the same file, inside `test_race_discloses_that_the_live_board_is_a_preview`, find:

```python
    assert "not an official result" in text
    for stale in ("two-week seasons", "has not moved forward", "Season 1 is the first that counts"):
        assert stale not in text, f"retired Live board claim still ships: {stale!r}"
```

Replace with:

```python
    assert "not an official result" in text
    # Spec §6 states the cadence as "nightly after the close"; the app's Live tab
    # shows the same words in place of a request clock.
    assert "updated nightly after the close" in text
    for stale in (
        "two-week seasons",
        "has not moved forward",
        "Season 1 is the first that counts",
        "updated after each trading session closes",
    ):
        assert stale not in text, f"retired Live board claim still ships: {stale!r}"
```

**1c.** In the same file, find the whole `test_race_source_and_shipped_bundle_agree` function:

```python
def test_race_source_and_shipped_bundle_agree():
    """The register's thesis applied to this section specifically: every other
    assertion here reads the bundle, so a Race.tsx edit that was never rebuilt
    into ../frontend/ would leave them green against stale text. Anchoring one
    string on both sides makes the missing `npm run build` the failure."""
    source = _RACE_TSX.read_text(encoding="utf-8")
    assert "Live Trading Leaderboard" in source, "Race.tsx no longer names the board"
    assert "Live Trading Leaderboard" in _shipped_text(), (
        "Race.tsx names the board but the shipped bundle does not — "
        "rebuild per dashboard/landing/README.md"
    )
```

Replace with (the renamed pin, plus a second pin for the Sharpe column):

```python
def test_race_source_and_shipped_bundle_agree():
    """The register's thesis applied to this section specifically: every other
    assertion here reads the bundle, so a Race.tsx edit that was never rebuilt
    into ../frontend/ would leave them green against stale text. Anchoring one
    string on both sides makes the missing `npm run build` the failure."""
    source = _RACE_TSX.read_text(encoding="utf-8")
    assert "Live Leaderboard" in source, "Race.tsx no longer names the board"
    assert "Live Trading Leaderboard" not in source, "Race.tsx still uses the retired name"
    assert "Live Leaderboard" in _shipped_text(), (
        "Race.tsx names the board but the shipped bundle does not — "
        "rebuild per dashboard/landing/README.md"
    )


def test_board_preview_sharpe_source_and_shipped_bundle_agree():
    """The Sharpe column changed with the daily Sharpe (direction spec 2026-10-14
    §4): the header stopped calling the figure hourly, and each row carries its
    95% interval and the sentence behind it. Pinned on both sides for the reason
    the Race pin above gives: a BoardPreview.tsx or lib/leaderboard.ts edit that
    was never rebuilt leaves every bundle-reading guard green on stale text."""
    source = (_RACE_TSX.parent / "BoardPreview.tsx").read_text(encoding="utf-8")
    shipped = _shipped_text()
    for text, where in ((source, "BoardPreview.tsx"), (shipped, "the shipped bundle")):
        assert "divided by how much the daily returns swung" in text, where
        assert "annualized from hourly results" not in text, where
    assert "Returns never moved" in shipped, (
        "lib/leaderboard.ts's formatSharpe is not in the shipped bundle — "
        "rebuild per dashboard/landing/README.md"
    )
```

**1d.** In `dashboard/backend/tests/test_landing_live_board.py`, inside the docstring of `test_standings_carry_the_ending_value_and_sharpe_the_table_renders`, find the paragraph as Phase 5 Task 5.4 left it (main@17c552c5 still has the older "Sharpe's em dash is NOT reachable through the live API today" wording; if you see that, Phase 5 is not on this branch and Step 0 should have stopped you):

```python
    Sharpe's em dash IS reachable through the API now: the payload sends
    `sharpe_ratio: null` below five trading days or when the returns never
    moved (`sharpe_status` says which). It used to send
    `run.get("sharpe_ratio") or 0`, publishing absent as a real 0.00. The
    landing does not render the 95% interval (`sharpe_ci_low`/`sharpe_ci_high`)
    yet; that is the landing's own change."""
```

Replace with:

```python
    Sharpe's em dash is reachable now: the server sends `null` unless the
    ratio's `sharpe_status` is "ok", where it used to send
    `run.get("sharpe_ratio") or 0`. These two entries carry no status -- the
    shape of a payload from before the daily Sharpe -- so the number prints
    bare, with no interval and no claim about how it was made."""
```

**1e.** In the same test, find the end of its node script and its assertions:

```python
console.log(JSON.stringify({
  values: board.standings.map((s) => s.endingValue),
  sharpes: board.standings.map((s) => s.sharpe),
}));
"""
    )
    assert result["values"] == ["$105,000", "—"], (
        "a run with no recorded final equity must render as absent, never as $0"
    )
    assert result["sharpes"] == ["1.23", "—"]
```

Replace with (the same test extended, then five new tests):

```python
console.log(JSON.stringify({
  values: board.standings.map((s) => s.endingValue),
  sharpes: board.standings.map((s) => s.sharpe),
  intervals: board.standings.map((s) => s.sharpeInterval),
  details: board.standings.map((s) => s.sharpeDetail),
}));
"""
    )
    assert result["values"] == ["$105,000", "—"], (
        "a run with no recorded final equity must render as absent, never as $0"
    )
    assert result["sharpes"] == ["1.23", "—"]
    assert result["intervals"] == ["", ""]
    assert result["details"] == ["", ""]


# The Competition payload's board-level basis, as Phase 5 publishes it.
_SHARPE_BASIS = (
    "{basis: 'daily_returns', annualization_days: 252, confidence: 0.95,"
    " min_days: 5, rf_annual: 0, stderr: 'iid'}"
)


def _sharpe_cell(entry_js: str, basis_js: str = _SHARPE_BASIS):
    return _run_ts(
        f"""
const entry = Object.assign({{entry_id: 'a', team_name: 'A', team_badge: 'Model', model: 'A',
  is_model: true, cumulative_return: 0.01, portfolio_value: 101000, initial_equity: 100000,
  equity_curve: []}}, {entry_js});
console.log(JSON.stringify(module.exports.formatSharpe(entry, {basis_js})));
"""
    )


def test_a_sharpe_is_printed_with_its_interval_and_the_basis_behind_it():
    """Direction spec 2026-10-14 §4: shown from 5 trading days, always with its
    95% interval. The numbers are conformance case M03 at rf 0 -- the server's,
    passed through untouched; this module computes none of them (D20)."""
    cell = _sharpe_cell(
        "{sharpe_ratio: 2.611715, sharpe_ci_low: -11.39649, sharpe_ci_high: 16.619919,"
        " sharpe_days: 5, sharpe_status: 'ok'}"
    )
    assert cell == {
        "value": "2.61",
        "interval": "-11.4 to 16.6",
        "detail": (
            "95% interval -11.40 to 16.62. Based on 5 trading days; "
            "cash rate 0%/yr; scaled to a yearly figure."
        ),
    }


def test_the_cash_rate_comes_from_the_board_basis():
    """The rate is the server's to state: the bundle holds no copy of it."""
    cell = _sharpe_cell(
        "{sharpe_ratio: 2.568186, sharpe_ci_low: -11.436927, sharpe_ci_high: 16.573299,"
        " sharpe_days: 5, sharpe_status: 'ok'}",
        "{min_days: 5, rf_annual: 0.0252}",
    )
    assert cell["value"] == "2.57"
    assert "cash rate 2.52%/yr" in cell["detail"]


@pytest.mark.parametrize(
    "entry_js,basis_js,expected",
    [
        (
            "{sharpe_ratio: null, sharpe_days: 3, sharpe_status: 'insufficient_days'}",
            _SHARPE_BASIS,
            {"value": "—", "interval": "", "detail": "Needs 5 trading days (have 3)"},
        ),
        (
            "{sharpe_ratio: null, sharpe_days: 3, sharpe_status: 'insufficient_days'}",
            "null",
            {"value": "—", "interval": "", "detail": "Not enough trading days yet (have 3)"},
        ),
        (
            "{sharpe_ratio: 0.5, sharpe_days: 3, sharpe_status: 'insufficient_days'}",
            _SHARPE_BASIS,
            {"value": "—", "interval": "", "detail": "Needs 5 trading days (have 3)"},
        ),
        (
            "{sharpe_ratio: null, sharpe_days: 6, sharpe_status: 'no_variation'}",
            _SHARPE_BASIS,
            {"value": "—", "interval": "", "detail": "Returns never moved"},
        ),
        (
            "{sharpe_ratio: null, sharpe_days: 0, sharpe_status: 'pending'}",
            _SHARPE_BASIS,
            {"value": "—", "interval": "", "detail": ""},
        ),
        (
            "{sharpe_ratio: null, sharpe_status: 'ok'}",
            _SHARPE_BASIS,
            {"value": "—", "interval": "", "detail": ""},
        ),
    ],
    ids=[
        "below-minimum",
        "below-minimum-without-basis",
        "status-beats-a-stray-number",
        "flat-curve",
        "pending",
        "ok-but-null",
    ],
)
def test_a_sharpe_without_a_number_says_why_instead_of_printing_zero(
    entry_js, basis_js, expected
):
    assert _sharpe_cell(entry_js, basis_js) == expected


def test_the_board_carries_the_payload_basis_into_every_row():
    result = _run_ts(
        """
const mk = (o) => Object.assign({team_badge: 'Model', is_model: true, portfolio_value: 101000,
  initial_equity: 100000, equity_curve: []}, o);
const board = module.exports.buildBoardData({
  entries: [
    mk({entry_id: 'a', team_name: 'A', model: 'A', cumulative_return: 0.02, sharpe_ratio: 2.611715,
        sharpe_ci_low: -11.39649, sharpe_ci_high: 16.619919, sharpe_days: 5, sharpe_status: 'ok'}),
    mk({entry_id: 'b', team_name: 'B', model: 'B', cumulative_return: 0.01, sharpe_ratio: null,
        sharpe_days: 3, sharpe_status: 'insufficient_days'}),
  ],
  window: {label: 'w'},
  sharpe_basis: {min_days: 5, rf_annual: 0},
});
console.log(JSON.stringify(board.standings.map((s) => [s.sharpe, s.sharpeInterval, s.sharpeDetail])));
"""
    )
    assert result == [
        [
            "2.61",
            "-11.4 to 16.6",
            "95% interval -11.40 to 16.62. Based on 5 trading days; "
            "cash rate 0%/yr; scaled to a yearly figure.",
        ],
        ["—", "", "Needs 5 trading days (have 3)"],
    ]


def test_the_basis_key_the_landing_reads_is_the_one_the_server_writes():
    """The tooltip's cash rate and minimum-days figure come from the board-level
    basis the server publishes once per Competition payload. A key renamed on
    one side only drops both from every tooltip with nothing failing, so the
    server's spelling is pinned against this module's."""
    service = (_ROOT / "backend" / "domain" / "leaderboard" / "service.py").read_text(
        encoding="utf-8"
    )
    assert '"sharpe_basis"' in service
    assert "payload.sharpe_basis" in _LIB_TS


def test_the_landing_never_derives_a_sharpe_from_a_curve():
    """Conformance D20: the server computes the ratio once and every surface
    displays it. A browser recomputation is how two surfaces came to print two
    different Sharpes for one run (#626, blocked for this reason)."""
    code = strip_comments(_LIB_TS)
    for token in ("Math.sqrt", "stdev"):
        assert token not in code, f"leaderboard.ts computes a Sharpe locally ({token!r})"
```

**1f.** In `dashboard/backend/tests/test_landing_chart_first.py`, find the last assertion of the file (in the y-axis formatter test):

```python
    assert "tickFormatter={axisTick}" in yaxis.group(0), (
        "the y-axis must bind the percent formatter itself, not a second "
        "formatter that walks around the body pin above; found "
        f"{yaxis.group(0)!r}"
    )
```

Replace with (unchanged, then a new test appended):

```python
    assert "tickFormatter={axisTick}" in yaxis.group(0), (
        "the y-axis must bind the percent formatter itself, not a second "
        "formatter that walks around the body pin above; found "
        f"{yaxis.group(0)!r}"
    )


def test_the_sharpe_cell_carries_its_interval_on_the_same_line():
    """Direction spec 2026-10-14 §4: a Sharpe is shown only with its 95%
    interval. The cell prints both on ONE line, because a second line would
    make every row taller than the 22px rank badge that sets the 28px pitch the
    list height was derived against. So the Sharpe track widens instead, and the
    header tooltip stops calling the figure hourly."""
    board = _BOARD
    at = board.index("{item.sharpe}")
    opening = board[board.rindex("<span", 0, at) : at]
    assert "whitespace-nowrap" in opening, "the value and its interval must stay on one line"
    assert "title={item.sharpeDetail" in opening, "the per-row basis sentence is the cell's tooltip"
    assert "{item.sharpeInterval}" in board[at : at + 400], "the interval renders beside the value"
    grid = re.search(r'const RANK_GRID =\s*"([^"]*)"', board)
    assert grid, "RANK_GRID not found"
    sm_tracks = re.search(r"sm:grid-cols-\[([^\]]*)\]", grid.group(1)).group(1).split("_")
    assert sm_tracks[-1] == "128px", f"the Sharpe track must fit value and interval; got {sm_tracks[-1]}"
    head = _collapse(board)
    assert "annualized from hourly results" not in head
    assert "divided by how much the daily returns swung" in head
```

**1g.** Two comments in `dashboard/backend/tests/test_landing_copy_register.py` still give the old name as a fact and a reason. After 1a they would sit a few lines from an assertion that the name is gone. These are comment-only edits: no test count changes and no phase touches these lines. In the docstring of `test_race_sample_cards_have_no_live_pulse` (≈:222 on main@17c552c5), find:

```python
    prose may still say "live" — the Live Trading Leaderboard is a real board
    with a real name, and live *market prices* are a real product property.
```

Replace with:

```python
    prose may still say "live" — the Live Leaderboard is a real board with a
    real name, and live *market prices* are a real product property.
```

Then, above `_BROKERED_CLAIM_PATTERNS` (≈:309), find:

```python
# Claim shapes, not vocabulary. "live trading" is deliberately absent: it is now
# a board name ("Live Trading Leaderboard"), so banning the bare phrase would
# make naming the product a test failure — the mirror image of the scoping bug
# above, and the reason the sibling guard in test_landing_value_band.py was
# changed to match the claim instead.
```

Replace with:

```python
# Claim shapes, not vocabulary. "live trading" is deliberately absent. While the
# board was the "Live Trading Leaderboard" (renamed "Live Leaderboard" on
# 2026-10-14), banning the bare phrase would have made naming the product a test
# failure — the mirror image of the scoping bug above, and the reason the
# sibling guard in test_landing_value_band.py was changed to match the claim
# instead. The rename does not reopen it: ban the claim, not the vocabulary.
```

- [ ] **Step 2: Run them and confirm they fail**

```bash
pytest dashboard/backend/tests/test_landing_copy_register.py dashboard/backend/tests/test_landing_live_board.py dashboard/backend/tests/test_landing_chart_first.py -q -p no:cacheprovider
```

Expected: `16 failed, 106 passed`, none skipped (`node` and `dashboard/landing/node_modules` must exist or the `_run_ts` cases skip instead of failing; measured on main@17c552c5 with Phases 1–6 applied). The failures:
- `test_race_names_the_two_boards_the_app_actually_serves`: `assert 'Live Leaderboard' in ...`
- `test_race_discloses_that_the_live_board_is_a_preview`: `assert 'updated nightly after the close' in ...`
- `test_race_source_and_shipped_bundle_agree`: `AssertionError: Race.tsx no longer names the board`
- `test_board_preview_sharpe_source_and_shipped_bundle_agree`: `AssertionError: BoardPreview.tsx`
- `test_standings_carry_the_ending_value_and_sharpe_the_table_renders`: `assert [None, None] == ['', '']`
- `test_a_sharpe_is_printed_with_its_interval_and_the_basis_behind_it`, `test_the_cash_rate_comes_from_the_board_basis`, and the six `test_a_sharpe_without_a_number_says_why_instead_of_printing_zero[...]` cases: the old `formatSharpe(value)` returns the string `'—'` for an object, so each assertion compares `'—'` with a dict (`TypeError: string indices must be integers, not 'str'` in the cash-rate case)
- `test_the_board_carries_the_payload_basis_into_every_row`: `AssertionError: assert [['2.61', Non..., None, None]] == [['2.61', '-1...ys (have 3)']]` (the old rows carry no `sharpeInterval`/`sharpeDetail`)
- `test_the_basis_key_the_landing_reads_is_the_one_the_server_writes`: `assert 'payload.sharpe_basis' in ...`. If it fails on its first assertion instead (`assert '"sharpe_basis"' in '"""Leaderboard contest: …`), `service.py` does not write the key: Phase 5 is not on this branch (measured on main@17c552c5 without Phase 5), so stop.
- `test_the_sharpe_cell_carries_its_interval_on_the_same_line`: `AssertionError: the value and its interval must stay on one line`

`test_the_landing_never_derives_a_sharpe_from_a_curve` already passes; it is a guard against a future edit, not against today's code.

- [ ] **Step 3: Implement**

**3a.** In `dashboard/landing/src/components/home/Race.tsx`, in `BOARD_RULES`, find:

```tsx
    text: "Live Trading Leaderboard: moves forward one trading session at a time, over the current calendar month.",
```

Replace with:

```tsx
    text: "Live Leaderboard: moves forward one trading session at a time, over the current calendar month.",
```

**3b.** In the same file, find the comment and the preview sentence:

```tsx
            {/* "Live" names the direction the board runs, not brokered execution, and
                Season 0 is a shakedown: the nightly advance runs
                (domain/leaderboard/live.py, .github/workflows/live-leaderboard.yml),
                but its standings are not a record yet. Both are stated on the
                board's own About card; saying it here too keeps the landing from
                selling an official result. Same wording contract as app.html's
                Home blurb and About card -- pinned together by
                test_live_board_copy_agrees_across_surfaces. */}
            <p className="text-xs text-muted-foreground mb-8">
              The Live Trading Leaderboard is in preview for Season 0: the current calendar month,
              updated after each trading session closes. Read its standings as a preview, not an
              official result.
            </p>
```

Replace with:

```tsx
            {/* "Live" names the direction the board runs, not brokered execution --
                which is why the board dropped "Trading" from its name (direction
                spec 2026-10-14 §6: the word read as real orders and collided with
                the Robinhood live-trading docs). Season 0 is a shakedown: the
                nightly advance runs (domain/leaderboard/live.py,
                .github/workflows/live-leaderboard.yml), but its standings are not
                a record yet. Both are stated on the board's own About card; saying
                it here too keeps the landing from selling an official result. Same
                wording contract as app.html's Home blurb and About card -- pinned
                together by test_live_board_copy_agrees_across_surfaces. The app's
                subtitle also ends in a no-capital disclaimer this file does not
                repeat: test_no_landing_component_claims_brokered_or_real_capital_trading
                scans comments too, and the hero's standing safety line already
                says it. */}
            <p className="text-xs text-muted-foreground mb-8">
              The Live Leaderboard is in preview for Season 0: the current calendar month, updated
              nightly after the close. Read its standings as a preview, not an official result.
            </p>
```

**3c.** In `dashboard/landing/src/lib/leaderboard.ts`, in `type LeaderboardEntry`, find:

```ts
  /** Typed nullable even though `get_leaderboard` currently sends
   *  `run.get("sharpe_ratio") or 0` -- domain/leaderboard/service.py:2161 --
   *  which collapses an absent ratio onto a real 0.00. The nullable type is
   *  what lets `finiteNumber` keep rendering the em dash if that `or 0` is
   *  ever fixed at the source, rather than requiring a second change here.
   *  Until then a missing ratio does arrive as 0 and does print "0.00". */
  sharpe_ratio: number | null;
```

Replace with:

```ts
  /** Daily Sharpe, annualised, computed once by the server from the stored
   *  curve (domain/backtesting/metrics.py `daily_sharpe_interval`). `null`
   *  unless `sharpe_status` is "ok" -- absent, never 0. The server's old
   *  `run.get("sharpe_ratio") or 0`, which collapsed a missing ratio onto a
   *  real 0.00, is gone. */
  sharpe_ratio: number | null;
  /** The bounds of its 95% interval; `null` unless `sharpe_status` is "ok". */
  sharpe_ci_low?: number | null;
  sharpe_ci_high?: number | null;
  /** How many daily returns the ratio stands on (0 for an entry with no run). */
  sharpe_days?: number | null;
  /** Why there is or is not a number. Optional only so a payload from before
   *  the daily Sharpe still parses; every current entry carries it. */
  sharpe_status?: 'ok' | 'insufficient_days' | 'no_variation' | 'pending' | null;
```

**3d.** In the same file, in `type BoardStanding`, find:

```ts
  /** Preformatted to two decimals, `—` when non-finite. */
  sharpe: string;
```

Replace with:

```ts
  /** Preformatted to two decimals, `—` when the server sent no number. */
  sharpe: string;
  /** Its 95% interval to one decimal, "-4.9 to 8.6". Empty when there is no
   *  number, or when the server sent the number without its bounds. */
  sharpeInterval: string;
  /** The sentence behind both, for the cell's tooltip. Empty when there is
   *  nothing to say. */
  sharpeDetail: string;
```

**3e.** In the same file, find the whole `formatSharpe`:

```ts
/** Two decimals, em dash when absent.
 *
 *  The /app row renders `Number(entry.sharpe_ratio || 0).toFixed(2)`, which
 *  prints "0.00" for a missing ratio. That is not copied: `|| 0` is the same
 *  absent-reads-as-zero defect `finiteNumber` was written to remove, and a
 *  Sharpe of exactly 0.00 is a real claim about a real run. The server does
 *  currently send `or 0` itself (service.py:2161), so the em dash is not
 *  reachable today -- it is reachable the moment that is fixed, which is the
 *  point of putting the rule on this side of the wire too. */
export function formatSharpe(value: unknown): string {
  const n = finiteNumber(value);
  if (!Number.isFinite(n)) return '—';
  return n.toFixed(2);
}
```

Replace with:

```ts
/** How the board computed its Sharpe, sent once per board as `sharpe_basis`.
 *  The tooltip quotes `rf_annual` and `min_days` from here and nowhere else,
 *  so this bundle never becomes a second owner of either number. */
export type SharpeBasis = {
  basis?: string;
  annualization_days?: number;
  confidence?: number;
  min_days?: number;
  rf_annual?: number;
  stderr?: string;
};

export type SharpeDisplay = { value: string; interval: string; detail: string };

/** 0 -> "0", 0.0252 -> "2.52": an annual cash rate as a percentage. */
function formatRatePercent(rate: number): string {
  return String(Number((rate * 100).toFixed(2)));
}

/** One Sharpe cell: the number, its 95% interval, and the sentence behind both.
 *
 *  DISPLAY ONLY. Every number arrives from the server, which computes the
 *  ratio and its interval once from the stored curve (direction spec
 *  2026-10-14 §4; conformance D20). Nothing in this bundle derives a Sharpe,
 *  an interval or a standard error, and nothing should.
 *
 *  ABSENT IS NEVER ZERO. `finiteNumber`, not `Number`: `Number(null)` is 0, and
 *  a Sharpe of exactly 0.00 is a real claim about a real run. A status other
 *  than "ok" wins over any number sent beside it.
 *
 *  The tooltip wording is the owner's (conformance spec, owner decision 5):
 *  "Return above what cash would have earned, divided by how much the daily
 *  returns swung. Based on {N} trading days; cash rate {rf}%/yr; scaled to a
 *  yearly figure." The first sentence is the column header's; the rest is per
 *  row, because N is. */
export function formatSharpe(
  entry: Partial<LeaderboardEntry> | null | undefined,
  basis?: SharpeBasis | null,
): SharpeDisplay {
  const row = entry || {};
  const status = row.sharpe_status;
  const days = finiteNumber(row.sharpe_days);
  const have = Number.isFinite(days) ? ` (have ${days})` : '';
  if (status === 'insufficient_days') {
    const min = finiteNumber(basis?.min_days);
    return {
      value: '—',
      interval: '',
      detail: Number.isFinite(min)
        ? `Needs ${min} trading days${have}`
        : `Not enough trading days yet${have}`,
    };
  }
  if (status === 'no_variation') {
    return { value: '—', interval: '', detail: 'Returns never moved' };
  }
  const value = finiteNumber(row.sharpe_ratio);
  if ((status && status !== 'ok') || !Number.isFinite(value)) {
    return { value: '—', interval: '', detail: '' };
  }
  // A payload from before the daily Sharpe sends a bare number and no status:
  // show the number and claim nothing about how it was made.
  if (!status) return { value: value.toFixed(2), interval: '', detail: '' };
  const lo = finiteNumber(row.sharpe_ci_low);
  const hi = finiteNumber(row.sharpe_ci_high);
  const bounded = Number.isFinite(lo) && Number.isFinite(hi);
  const facts: string[] = [];
  if (Number.isFinite(days)) facts.push(`based on ${days} trading days`);
  const rf = finiteNumber(basis?.rf_annual);
  if (Number.isFinite(rf)) facts.push(`cash rate ${formatRatePercent(rf)}%/yr`);
  facts.push('scaled to a yearly figure');
  const joined = facts.join('; ');
  const sentence = `${joined.charAt(0).toUpperCase()}${joined.slice(1)}.`;
  return {
    value: value.toFixed(2),
    interval: bounded ? `${lo.toFixed(1)} to ${hi.toFixed(1)}` : '',
    detail: bounded ? `95% interval ${lo.toFixed(2)} to ${hi.toFixed(2)}. ${sentence}` : sentence,
  };
}
```

**3f.** In the same file, find the head of `buildBoardData`:

```ts
export function buildBoardData(payload: {
  entries?: LeaderboardEntry[];
  window?: { label?: string };
}): BoardData {
```

Replace with:

```ts
export function buildBoardData(payload: {
  entries?: LeaderboardEntry[];
  window?: { label?: string };
  sharpe_basis?: SharpeBasis | null;
}): BoardData {
```

Then, inside its `perEntry.forEach(({ entry, byTime }) => {` loop (the line occurs once in the file), find:

```ts
    const name = entry.model || entry.team_name;
```

Replace with:

```ts
    const name = entry.model || entry.team_name;
    const sharpe = formatSharpe(entry, payload.sharpe_basis);
```

And in the `standings.push({ … })` object a few lines below, find:

```ts
      sharpe: formatSharpe(entry.sharpe_ratio),
```

Replace with:

```ts
      sharpe: sharpe.value,
      sharpeInterval: sharpe.interval,
      sharpeDetail: sharpe.detail,
```

**3g.** In the same file, in `fetchLeaderboard`, find:

```ts
  let payload: { entries?: LeaderboardEntry[]; window?: { label?: string } };
```

Replace with:

```ts
  let payload: {
    entries?: LeaderboardEntry[];
    window?: { label?: string };
    sharpe_basis?: SharpeBasis | null;
  };
```

**3h.** In `dashboard/landing/src/components/home/BoardPreview.tsx`, find the end of the `RANK_GRID` doc comment and the constant:

```tsx
 *  BELOW `sm` THE TABLE IS THREE COLUMNS, and the tracks change with the cells.
 *  `hidden` is `display: none`, so a hidden cell stops occupying its grid
 *  column -- but the TRACK survives, and two dead rails at 96px and 62px take
 *  158px of a 285px card. The narrow template drops them, so rank, name and
 *  return get the whole width. Which two columns go is the answer to "what does
 *  a phone reader lose least by losing": the ranking and the return are the
 *  claim, the ending value restates the return against a known base, and Sharpe
 *  is the one number that needs a caption to mean anything. */
const RANK_GRID =
  "grid grid-cols-[26px_minmax(0,1fr)_78px] sm:grid-cols-[26px_minmax(0,1.4fr)_96px_78px_62px] gap-2 items-center";
```

Replace with:

```tsx
 *  BELOW `sm` THE TABLE IS THREE COLUMNS, and the tracks change with the cells.
 *  `hidden` is `display: none`, so a hidden cell stops occupying its grid
 *  column -- but the TRACK survives, and two dead rails at 96px and 128px would
 *  take 224px of a 285px card. The narrow template drops them, so rank, name and
 *  return get the whole width. Which two columns go is the answer to "what does
 *  a phone reader lose least by losing": the ranking and the return are the
 *  claim, the ending value restates the return against a known base, and Sharpe
 *  is the one number that needs a caption to mean anything.
 *
 *  THE SHARPE TRACK IS 128px, NOT /app's 62px. A Sharpe is never shown without
 *  its 95% interval (direction spec 2026-10-14 §4), so the cell carries both on
 *  ONE line: a 14px value and an 11px "-11.9 to 1.7". A second line would make
 *  every row taller than the 22px rank badge that sets the 28px pitch the list
 *  height below was derived against. The name column pays for the width, and
 *  truncates rather than wraps. */
const RANK_GRID =
  "grid grid-cols-[26px_minmax(0,1fr)_78px] sm:grid-cols-[26px_minmax(0,1.4fr)_96px_78px_128px] gap-2 items-center";
```

**3i.** In the same file, in the head row (`data-testid="board-rank-head"`), find:

```tsx
            <span
              className="hidden sm:block text-right"
              title="Risk-adjusted return, annualized from hourly results."
            >
              Sharpe
            </span>
```

Replace with:

```tsx
            <span
              className="hidden sm:block text-right"
              title="Return above what cash would have earned, divided by how much the daily returns swung, scaled to a yearly figure. The smaller figures are its 95% interval; hover a value for the trading days and cash rate behind it."
            >
              Sharpe
            </span>
```

**3j.** In the same file, in the standings row, find:

```tsx
                <span className="hidden sm:block text-right font-mono text-muted-foreground tabular-nums">
                  {item.sharpe}
                </span>
```

Replace with:

```tsx
                <span
                  className="hidden sm:block text-right font-mono text-sm text-muted-foreground tabular-nums whitespace-nowrap"
                  title={item.sharpeDetail || undefined}
                >
                  {item.sharpe}
                  {item.sharpeInterval ? (
                    <span className="ml-1.5 text-[11px] text-muted-foreground/70">
                      {item.sharpeInterval}
                    </span>
                  ) : null}
                </span>
```

**3k.** Typecheck, rebuild and hand-copy the bundle (`dashboard/landing/README.md`, "Refreshing the shipped `../frontend/` bundle"):

Run this as one block from the repo root (the variables carry between lines):

```bash
cd dashboard/landing
npm run typecheck          # must exit 0; CI runs the same command
npm run build              # vite.config.ts sets emptyOutDir, so dist/public holds only this build
ls dist/public/assets      # expect one index-*.js, one index-*.css, atltransparent-DMIv-Uax.png
NEW_JS=$(cd dist/public/assets && ls index-*.js)
NEW_CSS=$(cd dist/public/assets && ls index-*.css)
cd ../frontend
OLD_JS=$(grep -o 'index-[A-Za-z0-9_-]*\.js' index.html)
OLD_CSS=$(grep -o 'index-[A-Za-z0-9_-]*\.css' index.html)
echo "js:  $OLD_JS -> $NEW_JS"     # main@17c552c5: index-Dpn97z5M.js -> (new hash)
echo "css: $OLD_CSS -> $NEW_CSS"   # main@17c552c5: index-goeFawbv.css -> (new hash)
if [ "$OLD_JS" = "$NEW_JS" ] || [ "$OLD_CSS" = "$NEW_CSS" ]; then
  echo "STOP: a bundle name did not change, so the build did not pick up the edits"
else
  cp ../landing/dist/public/assets/* assets/
  git rm -q "assets/$OLD_JS" "assets/$OLD_CSS"
  sed -i "s#/assets/$OLD_JS\"#/assets/$NEW_JS\"#; s#/assets/$OLD_CSS\"#/assets/$NEW_CSS\"#" index.html
  grep -n 'assets/index-' index.html   # lines ≈274-275 now name $NEW_JS and $NEW_CSS
fi
cd ../..
```

Each `echo` must print one name per side; two names on a side means the build emitted an extra chunk, so stop and look. Measured: these exact edits, built on main@17c552c5 with Phases 1–6 applied, produce `index-B0tY7Up0.js` and `index-32KGyMLw.css`. Expect the same names only if your edits are byte-identical to these; any other pair is fine. The build changes the CSS hash too (new Tailwind classes such as `sm:grid-cols-[…_128px]` and `text-[11px]`). `atltransparent-DMIv-Uax.png` keeps its name; the copy overwrites it with identical bytes. If the build emits any other image under a new name, delete the superseded one, or `test_no_orphaned_assets` fails.

The `sed` rewrites only the two filenames on `index.html`'s `<script type="module" crossorigin src="/assets/index-….js">` and `<link rel="stylesheet" crossorigin href="/assets/index-….css">` lines (measured: the `index.html` diff is exactly those two lines). Leave the auth-gate `<script>`, `#landingAuthModal`, `<style id="landing-auth-patch">` and the end-of-body auth `<script>` untouched.

- [ ] **Step 4: Run them and confirm they pass**

```bash
pytest dashboard/backend/tests/test_landing_copy_register.py dashboard/backend/tests/test_landing_live_board.py dashboard/backend/tests/test_landing_chart_first.py -q -p no:cacheprovider
```

Expected: `122 passed`, none skipped (a skip means `node` or `dashboard/landing/node_modules` is missing; fix that and re-run). Measured on main@17c552c5 with Phases 1–6 applied. Without Phase 5, `test_the_basis_key_the_landing_reads_is_the_one_the_server_writes` fails on its first assertion (measured on main@17c552c5); Step 0 rules that out.

Neighbouring suites (bundle integrity, the other landing guards, and the /app tests that read the landing helpers):

```bash
pytest dashboard/backend/tests/test_frontend_bundle_integrity.py dashboard/backend/tests/test_landing_board_frame.py dashboard/backend/tests/test_landing_value_band.py dashboard/backend/tests/test_frontend_chart_first_home.py dashboard/backend/tests/test_leaderboard_curve_integrity.py dashboard/backend/tests/test_frontend_board_frame.py -q -p no:cacheprovider
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # must print docs/live-board-public-copy
git add dashboard/landing/src/components/home/Race.tsx dashboard/landing/src/components/home/BoardPreview.tsx dashboard/landing/src/lib/leaderboard.ts dashboard/frontend/index.html dashboard/frontend/assets dashboard/backend/tests/test_landing_copy_register.py dashboard/backend/tests/test_landing_live_board.py dashboard/backend/tests/test_landing_chart_first.py
git status --short dashboard/frontend/assets   # exactly: the two new index-* files added, the two old ones deleted
git status --short                             # nothing else staged; an untracked node_modules symlink stays unadded
git commit -m "feat(landing): name the Live Leaderboard and show Sharpe with its interval"
```

---

### Task 7.4: Hosted docs: rename the board and document ignored stop fields (PR B, owner-coordinated)

**Execute only after the owner approves the copy in Task 7.3 Step 0.**

**Files:**
- Create: `dashboard/backend/tests/test_docs_live_board_names.py`
- Modify: `docs/source/lab/key_features.rst` (≈:10), `docs/source/lab/operating_modes.rst` (≈:14), `docs/source/lab/agent_api.rst` (≈:50), `docs/source/lab/external_agents.rst` (≈:195)
- Leave alone: `docs/source/lab/live_trading.rst` (the Robinhood page; the rename is what removes the collision)

**Interfaces:**
- Consumes: the name "Live Leaderboard" (Decision 8 in the index plan); the legacy engine's handling of stop fields (`domain/runs/protocol.py:133-134` and `domain/backtesting/external_run_service.py:1177-1178` force them to `None`; nothing in `portfolio_manager.py`, `domain/trading/` or `execution/` reads them).
- Produces: `test_docs_live_board_names.py`; a throwaway `rst_diff.py` (not committed).

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_docs_live_board_names.py`:

```python
"""The hosted manual names the boards the way the app does.

The 2026-10-14 direction spec (§6) dropped "Trading" from the Live board's name:
it read as real orders and collided with docs/source/lab/live_trading.rst, which
is about a real Robinhood account. No CI job builds docs/source, so these text
guards are the only check that the rename reached the manual and stays there.
"""

from pathlib import Path

import pytest

_LAB = Path(__file__).resolve().parents[3] / "docs" / "source" / "lab"
_RETIRED_NAMES = ("Live Trading Leaderboard", "Live Trading board")


def _rst(name: str) -> str:
    return (_LAB / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(p.name for p in _LAB.glob("*.rst")))
def test_no_lab_page_uses_the_retired_board_name(name):
    text = " ".join(_rst(name).split())
    for retired in _RETIRED_NAMES:
        assert retired not in text, f"{name} still uses the retired board name"


def test_the_board_pages_use_the_current_name():
    assert "Live Leaderboard" in _rst("key_features.rst")
    assert "**Live Leaderboard**" in _rst("operating_modes.rst")
    assert "Competition and Live leaderboards" in " ".join(_rst("agent_api.rst").split())


def test_the_robinhood_page_keeps_its_own_title():
    """The collision is removed by renaming the board, not the brokerage page."""
    assert _rst("live_trading.rst").splitlines()[0] == "Live Trading with Robinhood"


def test_stop_and_take_profit_fields_are_documented_as_ignored():
    """Conformance D12 on the legacy engine: both fields are parsed and then
    dropped (domain/runs/protocol.py and external_run_service.py force them to
    None; nothing downstream reads them). An agent author reading "optional
    floats" alone would expect a stop to fire."""
    text = " ".join(_rst("external_agents.rst").split())
    assert (
        "``stop_loss_price`` / ``take_profit_price`` — optional floats. "
        "They are accepted and ignored"
    ) in text, "external_agents.rst must say the stop fields are accepted and ignored"
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
pytest dashboard/backend/tests/test_docs_live_board_names.py -q -p no:cacheprovider
```

Expected: `5 failed, 11 passed`:
- `test_no_lab_page_uses_the_retired_board_name[agent_api.rst]`, `[key_features.rst]`, `[operating_modes.rst]`: `AssertionError: <name> still uses the retired board name`;
- `test_the_board_pages_use_the_current_name`: `assert 'Live Leaderboard' in ...`;
- `test_stop_and_take_profit_fields_are_documented_as_ignored`: `AssertionError: external_agents.rst must say the stop fields are accepted and ignored`.

- [ ] **Step 3: Implement**

**3a.** In `docs/source/lab/key_features.rst`, find:

```rst
- **Two leaderboards** — A Competition board and a Live Trading board rank a curated roster of baselines and models.
```

Replace with:

```rst
- **Two leaderboards** — The Competition Leaderboard and the Live Leaderboard rank a curated roster of baselines and models.
```

(The rest of that line, from ` Neither accepts user or agent submissions.` on, is unchanged.)

**3b.** In `docs/source/lab/operating_modes.rst`, find:

```rst
and the **Live Trading Leaderboard** (a calendar-month board that advances after the US cash close).
```

Replace with:

```rst
and the **Live Leaderboard** (a calendar-month board that advances nightly after the US cash close, on simulated fills with no real money; it is not the **Live trading** mode above).
```

**3c.** In `docs/source/lab/agent_api.rst`, inside the `.. note::` block, find:

```rst
   separate from the dashboard's Competition and Live Trading boards, which
```

Replace with:

```rst
   separate from the dashboard's Competition and Live leaderboards, which
```

**3d.** In `docs/source/lab/external_agents.rst`, under "Field notes:", find:

```rst
- ``stop_loss_price`` / ``take_profit_price`` — optional floats.
```

Replace with (continuation lines indented two spaces, so they stay inside the bullet):

```rst
- ``stop_loss_price`` / ``take_profit_price`` — optional floats. They are
  accepted and ignored: no stop or take-profit order is placed, every order is
  a market order, and a position closes only when a later decision sells it.
```

**3e.** Validate the RST: no CI job builds `docs/source`, and neither sphinx nor docutils is installed. Build a throwaway venv outside the repo and run a differential parse (diagnostic multisets must match with line numbers ignored; every list keeps its kind and item count):

```bash
SP=$(mktemp -d)
python3 -m venv "$SP/rstvenv" && "$SP/rstvenv/bin/pip" install -q docutils
mkdir -p "$SP/base" "$SP/new"
for f in key_features operating_modes agent_api external_agents; do
  git show "origin/main:docs/source/lab/$f.rst" > "$SP/base/$f.rst"
  cp "docs/source/lab/$f.rst" "$SP/new/$f.rst"
done
cat > "$SP/rst_diff.py" <<'PYEOF'
"""Differential docutils parse for docs/source RST edits (no sphinx in CI).

usage: python rst_diff.py <base_dir> <new_dir> <file.rst> [<file.rst> ...]

Each file is parsed twice, as it was (<base_dir>/<file>) and as edited
(<new_dir>/<file>). The diagnostic multisets must match, line numbers ignored: bare
docutils always reports sphinx-only roles (:doc:, :ref:) and a missing Pygments,
so an absolute count means nothing, but a new message means the edit broke
markup. Every list must keep its kind and item count, so a reworded bullet
cannot silently become a sibling paragraph or swallow the next item.
"""
import io
import sys
from pathlib import Path

from docutils import nodes
from docutils.core import publish_doctree

_LISTS = (nodes.bullet_list, nodes.enumerated_list, nodes.definition_list, nodes.field_list)


def _parse(path: Path):
    tree = publish_doctree(
        path.read_text(encoding="utf-8"),
        source_path=str(path),
        settings_overrides={"report_level": 1, "halt_level": 5, "warning_stream": io.StringIO()},
    )
    # The message paragraph alone: astext() would prefix "path:line:", and
    # moving a line is exactly what an edit is allowed to do.
    messages = sorted(
        (m["level"], m[0].astext() if len(m) else m.astext())
        for m in tree.findall(nodes.system_message)
    )
    shapes = [(type(n).__name__, len(n.children)) for n in tree.findall(lambda n: isinstance(n, _LISTS))]
    return messages, shapes


def main() -> int:
    base, new, files = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    bad = 0
    for name in files:
        old_msgs, old_shapes = _parse(base / name)
        new_msgs, new_shapes = _parse(new / name)
        if old_msgs != new_msgs:
            bad += 1
            print(f"FAIL {name}: diagnostics changed")
            print("  before:", old_msgs)
            print("  after: ", new_msgs)
        if old_shapes != new_shapes:
            bad += 1
            print(f"FAIL {name}: list shapes changed {old_shapes} -> {new_shapes}")
        if old_msgs == new_msgs and old_shapes == new_shapes:
            print(f"ok   {name}: {len(new_msgs)} diagnostics unchanged, {len(new_shapes)} lists unchanged")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
PYEOF
"$SP/rstvenv/bin/python" -I "$SP/rst_diff.py" "$SP/base" "$SP/new" key_features.rst operating_modes.rst agent_api.rst external_agents.rst
```

Expected (measured on main@17c552c5 with Phases 1–6 applied, for these exact edits):

```text
ok   key_features.rst: 8 diagnostics unchanged, 1 lists unchanged
ok   operating_modes.rst: 4 diagnostics unchanged, 1 lists unchanged
ok   agent_api.rst: 0 diagnostics unchanged, 0 lists unchanged
ok   external_agents.rst: 14 diagnostics unchanged, 5 lists unchanged
```

A `FAIL external_agents.rst: diagnostics changed` whose `after:` list gains `Bullet list ends without a blank line; unexpected unindent.` means 3d's continuation lines lost their two-space indent (measured by removing them). A `FAIL … list shapes changed` means a bullet split or merged.

- [ ] **Step 4: Run it and confirm it passes**

```bash
pytest dashboard/backend/tests/test_docs_live_board_names.py dashboard/backend/tests/integrations/test_docs_run_command.py -q -p no:cacheprovider
```

Expected: `18 passed` (16 here, 2 in `test_docs_run_command.py`; measured on main@17c552c5 with Phases 1–6 applied).

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # must print docs/live-board-public-copy
git add docs/source/lab/key_features.rst docs/source/lab/operating_modes.rst docs/source/lab/agent_api.rst docs/source/lab/external_agents.rst dashboard/backend/tests/test_docs_live_board_names.py
git commit -m "docs: name the Live Leaderboard and note ignored stop fields"
```

---

### Task 7.5: Open the PRs

**Files:** none.
**Interfaces:** Consumes the commits of Tasks 7.1–7.4. Produces PR A (ready for review), PR B (draft) and, owner-only, an updated project record in the owner's central database.

**Part A (Step 2) runs as soon as Tasks 7.1–7.2 are committed. Step 3 runs only after PR A merges, and only for the owner. Part B (Step 4) runs only after Tasks 7.3–7.4.**

- [ ] **Step 1: Full suite (each branch, before its push)**

```bash
pytest dashboard/backend/tests -q -p no:cacheprovider
```

It takes about 8 minutes (462 s measured on main@17c552c5), so give the call a 10-minute timeout or run it in the background. `pytest-timeout` is not installed: do not add `--timeout`.

Expected: green apart from failures in `test_report_pdf.py`, which are environmental locally (no `reportlab`; 7 measured on main@17c552c5, and the count grows with that file). On bare main@17c552c5, without Phases 1–6, the full suite also shows the prerequisite failures this plan predicts: the Task 7.1 Phase-owned cases in `test_live_board_records.py`, and `test_landing_live_board.py::test_the_basis_key_the_landing_reads_is_the_one_the_server_writes`. On a branch whose Step 0 passed, neither appears. If one does, go back to Task 7.1 Step 0 or Task 7.3 Step 0; do not edit the test. Any other red test is a regression in this branch.

- [ ] **Step 2: Part A, push and open**

```bash
git switch docs/live-board-records && git branch --show-current
gh pr list --head docs/live-board-records --state all   # must be empty: never push to a merged branch
git push -u origin docs/live-board-records
gh pr create --base main --head docs/live-board-records --title "docs: correct the live board records" --body-file - <<'EOF'
- `CLAUDE.md` "Two leaderboards" now describes `live.py` as shipped: dispatch, read-only GET, per-session checkpoints, roster with `live_from` and the Live-only `live_overrides`, attempt ledger, early closes, daily Sharpe. The season-preview code is marked legacy, not to be revived.
- Env bullets: `LEADERBOARD_DAILY_REFRESH_SECRET` names both workflows that send it; `LEADERBOARD_DAILY_AUTO_DEPLOY` drops "no advance engine"; new `LIVE_LEADERBOARD_DEPLOY_MODELS` bullet (GitHub variable, `true` since 2026-09-29). The budget bullet stays Phase 3's.
- `daily-leaderboard.yml`: header comment no longer says the live advance rides the daily call.
- "Superseded for the live board" banner on six dated specs, plans and notes.
- `test_live_board_records.py` pins the retired claims as gone, every name the section cites as present (so a rename turns it red), and the section's "no decision tape" sentence against an import scan of `live.py`.
- Docs and one text-only test; no runtime change. Merging to main auto-deploys prod.

Do not merge on a weekday between 22:15 and ~02:30 UTC. The merge auto-deploys prod, and the Render restart kills the nightly live refresh thread mid-run (its cron says 22:30 UTC, but GitHub has started it as late as 01:52), so that night's session is billed again on the next night.
EOF
```

- [ ] **Step 3: After PR A merges, update the owner's project record (owner-only)**

**Owner-only.** This step writes to the owner's private central database, which lives outside this repo. An executor without access to it skips this step and says so in its hand-off ("Task 7.5 Step 3 skipped: no access to the owner's central database"). Run it only once PR A shows `MERGED` (`gh pr list --head docs/live-board-records --state merged`).

Update the Agentic Trading Lab entry in the central database's projects file, through that database's normal edit path. Never write to its `state/` or `identity/` folders. Record:
- the Live Leaderboard has advanced a calendar-month board nightly since 2026-09-26 (`domain/leaderboard/live.py`, #534), on the legacy engine;
- `LIVE_LEADERBOARD_DEPLOY_MODELS=true` (a GitHub repository variable) since 2026-09-29, so the weekday cron bills model calls through `COMMONSTACK_API_KEY`, outside ATL Credits;
- the roster: DeepSeek V4 Pro, with Qwen3.7 Plus and Claude Haiku 4.5 from 2026-11-01 (if P4 set a later `live_from` in `dashboard/config/leaderboard.json`, record that month instead); Nemotron is off the live roster because CommonStack serves no NVIDIA model;
- the $50 monthly cap on live model spend, `LEADERBOARD_MONTHLY_BUDGET_USD` (default 50, enforced in code since P3).

- [ ] **Step 4: Part B, push and open as a draft**

```bash
git switch docs/live-board-public-copy && git branch --show-current
gh pr list --head docs/live-board-public-copy --state all   # must be empty
git push -u origin docs/live-board-public-copy
gh pr create --draft --base main --head docs/live-board-public-copy --title "feat(landing): Live Leaderboard copy and Sharpe interval" --body-file - <<'EOF'
**DO NOT MERGE until the owner has checked `/` signed out and signed in on this PR's Vercel preview (checklist below).**

- Landing: Race renames the board "Live Leaderboard" and states the cadence as "updated nightly after the close" (spec §6). No "no real money" phrase: landing components ban it, and the hero already says it.
- Landing: the Sharpe column prints the server's daily Sharpe with its 95% interval on one line (track 62px to 128px at `sm`+); each row's tooltip names its trading days and cash rate; the header tooltip uses the owner's wording (spec §4, conformance owner decision 5). The browser computes nothing.
- Bundle rebuilt and hand-copied per `dashboard/landing/README.md`; the old `index-*.js`/`.css` are deleted. Two test comments in `test_landing_copy_register.py` stop giving the retired name as current.
- Docs: `key_features`, `operating_modes` and `agent_api` use the new name; `external_agents` says `stop_loss_price`/`take_profit_price` are accepted and ignored. Checked with a differential docutils parse, because no CI job builds the docs.
- Owner browser checklist: (1) signed out, every **Start Free** opens the modal in signup mode and the nav **Sign in** opens it in login mode (viewport wider than 1024px); (2) signed in, `/` renders and both CTAs read **Test a trading idea** and **Join our Discord community** and navigate; (3) sign out from `/app` returns to `/` with **Start Free** CTAs; (4) the Sharpe column at 640, 1024 and 1440px wide: value and interval on one line, no row taller than its rank badge, long names truncate with an ellipsis; (5) the only console error is the `/_vercel/insights/script.js` 404.
- Merging to main auto-deploys prod (Vercel for `/`, Render for the backend).

Do not merge on a weekday between 22:15 and ~02:30 UTC. The merge auto-deploys prod, and the Render restart kills the nightly live refresh thread mid-run (its cron says 22:30 UTC, but GitHub has started it as late as 01:52), so that night's session is billed again on the next night.
EOF
```

After the owner's check, the owner (not the executor) marks it ready for review.

#### Phase 7 review-focus candidates

1. **Phase 5 publishes the Competition basis under a key other than `sharpe_basis`.** The landing would then drop "cash rate" and the minimum-days figure from every tooltip with nothing failing. Phase 5's interface list names the key `sharpe_basis` (written in `service.py` as `"sharpe_basis": daily_sharpe_basis(),`). Covered: `test_the_basis_key_the_landing_reads_is_the_one_the_server_writes` (Task 7.3) pins the server's spelling against the landing's. Step 0 item 2 checks the live payload, and item 3 greps `service.py` for `"sharpe_basis"`. Item 2's fallback finds a renamed key by grepping for `daily_sharpe_basis()` (the `daily_returns` literal lives in `metrics.py`, so grepping `service.py` for it finds nothing).
2. **Render still serves the pre-Phase-5 payload when Vercel ships PR B.** Rows then carry a bare `sharpe_ratio` and no `sharpe_status`. Covered: the extended `test_standings_carry_the_ending_value_and_sharpe_the_table_renders` pins that such a row prints the bare number with no interval and no basis sentence, and Step 0 refuses to start before the fields are live.
3. **The 128px Sharpe track at 640–1023px.** Names truncate sooner, and an unusually wide interval (for example `-25.1 to -0.3` beside `-12.40`) can clip at the card edge rather than wrap. Not testable here: there is no layout engine in CI or in this WSL, and the derivation in `BoardPreview.tsx` is per measured viewport. It is item (4) of the owner's browser checklist in the PR body. The test pins only the invariant that matters for the list height: one line (`whitespace-nowrap`), so the 28px row pitch survives.
4. **PR A started before Phases 1–6 merged, or a phase merged under a different name.** Measured on bare main@17c552c5, before #668 added a row: every edit applies cleanly and the 18 Phase-owned cases stay red (`18 failed, 56 passed`), so starting early yields a section that describes code which is not there. Covered: Task 7.1 Step 0 now checks every Phase-owned name in the file the test reads it from, plus Phase 3's bullet and the Phase 4 roster tuple, and stops on any `MISSING`. Its rename clause, and Step 4's, say to fix the check, the sentence and the `_NAMED` row together.
5. **A later rename in `live.py` (say `_write_live_checkpoint`) turns `test_live_board_records.py` red in an unrelated PR.** Intended: it is the fail-visible property the 2026-10-14 review asked for. The assertion message tells that PR's author what to edit. Not a gap.
6. **Phase 5 rewrites the docstring Task 7.3 Step 1d edits.** P5 Task 5.4 replaces the "Sharpe's em dash is NOT reachable…" paragraph of `test_standings_carry_the_ending_value_and_sharpe_the_table_renders`. Step 1d anchors on Phase 5's wording, measured to match exactly once on main@17c552c5 with Phases 1–6 applied (it does not match bare main), and Step 0 greps for it. If Phase 5 changes that wording again before it merges, 1d's find block must follow it.
7. **The section records Phase 4's `live_overrides`, not the text of Decision 6 in the index plan.** That Decision said to set Haiku's shared `model_id`/`temperature`. Phase 4 found that `_entry_config_drift` would then demote Haiku's Competition repeats and publish the single July run, so it moved those values into a Live-only `live_overrides` block. The Roster bullet says so and warns against moving them back, and the `_NAMED` rows `live_overrides` (Phase 4) and `_entry_config_drift` (on main) keep both names honest. If Phase 4 is executed as that Decision originally read, Step 0's `need live_overrides` prints `MISSING`. That is a real disagreement to settle with the owner, not a rename.
