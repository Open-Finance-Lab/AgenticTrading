# Live-Board Fixes P4: Live Roster (Drop Nemotron, Add Qwen and Haiku) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Live board's nightly roster DeepSeek V4 Pro, Qwen3.7 Plus and Claude Haiku 4.5 (Nemotron dropped, because CommonStack does not serve it), with the two newcomers joining at the next month boundary and Haiku calling CommonStack's served id at temperature 0, without changing what the Competition board publishes.

**Architecture:** A per-entry `live_from` month gate and a `live_overrides` block are applied inside `live_board_strategies`, so every Live path (nightly deploy, resume check, deploy CLI, public GET) sees the same gated, overridden entries. `leaderboard.json`'s shared keys, which the Competition board compares its stored runs against, stay untouched. `LIVE_MODEL_IDS` becomes the ordered roster and deploy order, and config tests pin every live entry to CommonStack's 2026-10-14 model listing and listed prices.

**Tech Stack:** Python 3.13 / FastAPI backend (`domain/leaderboard/live.py`), JSON config (`dashboard/config/leaderboard.json`), one GitHub Actions workflow comment and one CLI help string, pytest from the repo root.

**Spec:** `docs/superpowers/specs/2026-10-14-atl-direction.md` §6 (Live Leaderboard: the "Until Season 1" row) and §7 (Budget and keys: per-model prices). Decisions, Global Constraints, merge order and Review Focus live in the index plan `docs/superpowers/plans/2026-10-14-live-board-fixes.md`; every task here inherits its Global Constraints.

**Merge order:** P1 → P2 → P3 → P4 → P5 → P6 → P7 PR A → P7 PR B (owner-gated). Deadlines: P3 before 2026-11-01; P4 by 2026-10-31 (else bump both live_from values); P1 merged and deployed by 2026-11-20. One executor per contiguous range; phases are not safe to run in parallel (several append to the same test files and bump the same cache-busters).

## Phase 4: Live roster — drop Nemotron, add Qwen and Haiku from the next month

**Why.** Spec §6 ("Until Season 1") and §7 (budget and keys). Nemotron 3 Nano has 404'd every night since 2026-10-06 because CommonStack serves no NVIDIA model (GET /v1/models with the prod key, 2026-10-14: 86 models, no Nemotron), so Decision 6 in the index plan drops it from the Live roster instead of "fixing its id". Qwen3.7 Plus and Claude Haiku 4.5 join at one run each, from the next calendar month, so neither backfills a month it did not see forward. Haiku's Live calls move to the id CommonStack actually serves (`anthropic/claude-haiku-4-5`) at temperature 0.

> **Deviation from Decision 6 (index plan).** Read before Task 4.2. Decision 6 in the index plan says to set Haiku's shared `model_id` and `temperature` in `leaderboard.json`. Two of its checks hold: the Competition Haiku row is never recomputed (`auto_compute: false`; `ensure_leaderboard_runs` skips it), and the results panel's Sampling label is read from the stored run (`api/routers/backtests.py::_leaderboard_llm_sampling`). A third config read does not hold. `service._entry_publication_with_pool` picks the Competition run to publish by comparing each stored run's recorded `model_id/integration/temperature/reasoning_effort` against the entry (`_entry_config_drift`). Haiku's three Competition repeats (deployed 2026-10-03, median +0.19%) recorded `claude-haiku-4-5-20251001` with no temperature. Its July primary recorded nothing. Changing the shared keys makes the repeats "stale" (rank 2) and the July row only "unrecorded" (rank 1). The public Competition board would then publish the July single run, drop the samples block and band, and print a WARNING on every process start. Measured on main@17c552c5 with those values put on the shared keys: the board published `lb_claude_haiku_4_5_20260415_20260515` in place of `…_s2`. This phase therefore applies Haiku's Live-only values through a new `live_overrides` block. The shared keys stay as the Competition runs recorded them, and a test pins that the Competition publication is unchanged.

**Depends on:** no code from another phase. It merges after P1 (`2026-10-14-live-board-fixes-p1-early-close.md`), P2 (`2026-10-14-live-board-fixes-p2-memory-robustness.md`) and P3 (`2026-10-14-live-board-fixes-p3-attempt-ledger.md`), all in `docs/superpowers/plans/`. All three edit `live.py` and `test_live_leaderboard.py`, so match every anchor by the quoted text, never by line number. The anchors below are quoted from main@17c552c5; each occurs exactly once both on main@17c552c5 and on main@17c552c5 with P1–P3 applied. P5 (`2026-10-14-live-board-fixes-p5-daily-sharpe.md`) merges after this phase; if it lands first anyway, see the note after Task 4.3 Step 1(a). Before Task 4.3, run `rg -ln nemotron_3_nano_30b dashboard/backend/tests`. On main@17c552c5 it lists `test_live_leaderboard.py`, `test_landing_live_board.py`, `domain/leaderboard/test_deploy_guard.py` and `domain/leaderboard/test_strategies_move.py`. Only the first uses Nemotron as a *Live* entry; the other three are Competition, landing or deploy-guard fixtures and stay untouched. If an earlier phase added a test file that uses Nemotron as a Live entry (for example a budget "skipped=" line), apply Task 4.3's rename to that file too. Any assertion there that lists the whole Live roster must then read `deepseek_v4_pro, qwen3_7_plus, claude_haiku_4_5` in that order. As planned, no earlier phase does this: P3's `test_live_budget.py` uses DeepSeek and a `deepseek_twin` copy of it.

**Running the command blocks.** Every line of a multi-line `bash` block is its own command. Run them one at a time, in order. Do not join them with `;`, `&&` or heredocs into one shell call: a worktree-isolation guard refuses exactly those chained calls as "too complex to verify". Make test-file appends with the Edit tool, not `cat >>`.

**Branch / PR title:** `feat/live-roster-qwen-haiku` / `feat(live): drop Nemotron, add Qwen and Haiku`

**Interfaces this phase produces** (all in `dashboard/backend/domain/leaderboard/live.py` unless noted):
- `LIVE_MODEL_IDS == ("deepseek_v4_pro", "qwen3_7_plus", "claude_haiku_4_5")`. Its order is also the nightly deploy order and the `live_status.roster` order.
- `LIVE_FROM_KEY = "live_from"` sets the optional per-LLM-entry config key `"live_from": "YYYY-MM-01"`. An entry is live in board month M iff its id is in `LIVE_MODEL_IDS` and (`live_from` is absent, null or malformed, or `live_from <= M`). A malformed value is one that is not a string matching `\d{4}-\d{2}-01`, or that is not a real date. It prints one `WARNING: live.config_invalid …` line per process and is treated as absent.
- `LIVE_OVERRIDES_KEY = "live_overrides"`, `LIVE_OVERRIDE_FIELDS = ("model_id", "integration", "temperature", "reasoning_effort")`. The optional config block `"live_overrides": {…}` is applied to the entry on the Live board only, and the block itself is dropped from the effective entry. If the block is not an object, it is warned once and ignored. An unknown key inside it is warned once and skipped.
- `live_board_strategies(config=None, *, month_start: Optional[str] = None) -> List[Dict]` returns baselines plus the LLM entries live in the board month, with overrides applied, in config order. The board month is taken from `month_start` when given. Otherwise it is `config["start_date"]` when `config["period"] == "live"` (a freeze config), and otherwise the current Eastern month.
- `live_llm_entries(config=None, *, month_start=None) -> List[Dict]` returns the month's live LLM entries in `LIVE_MODEL_IDS` order.
- Private helpers: `_warned_live_config: set`, `_LIVE_FROM_SHAPE`, `_warn_live_config(entry_id, key, value, expected) -> None`, `_live_from_month(entry) -> Optional[date]`, `_with_live_overrides(entry) -> Dict`, `_board_month(config, month_start) -> date`.
- Log line (stdout, flushed, once per `(entry, key, repr(value))` per process): `WARNING: live.config_invalid entry=<id> key=<live_from|live_overrides|live_overrides.<k>> value=<repr> expected=<…>; ignoring it`.
- Public GET `?period=live`: `live_status.roster` is now this month's live LLM ids in `LIVE_MODEL_IDS` order. It used to be `list(LIVE_MODEL_IDS)` whatever the month.
- `dashboard/config/leaderboard.json`: `qwen3_7_plus.live_from` and `claude_haiku_4_5.live_from` are `"2026-11-01"`, or the 1st of the calendar month after the merge month. `claude_haiku_4_5.live_overrides = {"model_id": "anthropic/claude-haiku-4-5", "temperature": 0}`. The shared Haiku keys are unchanged.
- Test-only constant `COMMONSTACK_SERVED` in `dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py` maps model id to the listed (input, output) USD per M tokens, dated 2026-10-14.

### Task 4.1: Optional per-entry `live_from` gate

**Files:**
- Modify: `dashboard/backend/domain/leaderboard/live.py`
- Modify: `dashboard/scripts/deploy_leaderboard_model.py`
- Create (test): `dashboard/backend/tests/test_live_roster.py`

**Interfaces:**
- Consumes: `lb_service.load_leaderboard_config()`, `live_month_dates()`, `live_freeze_config()`, `get_live_leaderboard()`, `refresh_live_leaderboard()`; the test helpers `SEED`, `_insert_live_run`, `_isolated_live_session` (autouse fixture) and `_pt` from `dashboard/backend/tests/test_live_leaderboard.py`; `_load_deploy_script` from `dashboard/backend/tests/test_leaderboard_samples.py`.
- Produces: `LIVE_FROM_KEY`, `_warned_live_config`, `_LIVE_FROM_SHAPE`, `_warn_live_config`, `_live_from_month`, `_board_month`, the new `live_board_strategies` / `live_llm_entries` signatures (keyword-only `month_start`), `LIVE_MODEL_IDS` ordering of `live_llm_entries`, month-gated `live_status.roster`, and the test-module helpers `_DROP`, `_EVERY_JOIN_OPEN`, `_config_with`, `_use_config`, `_live_ids`, `_AUG27`, `_SEP3`.

- [ ] **Step 1: Write the failing test.** Create `dashboard/backend/tests/test_live_roster.py` with exactly this content:

```python
"""Who is on the Live board in a given month.

An LLM entry trades on the Live board when its id is in ``LIVE_MODEL_IDS`` and
its optional ``live_from`` month has opened. A not-yet-live entry must be
neither deployed (that bills a month-to-date replay) nor rendered (that shows a
pending row for a model nobody is running).
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import dashboard.backend.domain.leaderboard.live as live
import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.tests.test_live_leaderboard import (  # noqa: F401
    SEED,
    _insert_live_run,
    _isolated_live_session,
    _pt,
)

_ET = ZoneInfo("America/New_York")
_AUG27 = datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
_SEP3 = datetime(2026, 9, 3, 17, 39, tzinfo=_ET)
# After every live_from in leaderboard.json, so no entry is gated out.
_EVERY_JOIN_OPEN = "2099-12-01"
_DROP = object()


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    """The once-per-process warning record would hide a second test's line."""
    monkeypatch.setattr(live, "_warned_live_config", set())


def _config_with(entry_id, **keys):
    """leaderboard.json with ``keys`` set on one entry (``_DROP`` removes one)."""
    cfg = lb_service.load_leaderboard_config()
    strategies = []
    for strategy in cfg["strategies"]:
        strategy = dict(strategy)
        if strategy["id"] == entry_id:
            for key, value in keys.items():
                if value is _DROP:
                    strategy.pop(key, None)
                else:
                    strategy[key] = value
        strategies.append(strategy)
    return {**cfg, "strategies": strategies}


def _use_config(monkeypatch, cfg):
    monkeypatch.setattr(lb_service, "load_leaderboard_config", lambda: cfg)


def _live_ids(as_of):
    return [e["id"] for e in live.live_llm_entries(live.live_freeze_config(as_of))]


def test_an_entry_joins_in_its_live_from_month(monkeypatch):
    _use_config(monkeypatch, _config_with("deepseek_v4_pro", live_from="2026-09-01"))
    assert "deepseek_v4_pro" not in _live_ids(_AUG27)
    assert "deepseek_v4_pro" in _live_ids(_SEP3)


def test_month_start_names_the_board_month_for_the_base_config(monkeypatch):
    cfg = _config_with("deepseek_v4_pro", live_from="2026-09-01")
    before = live.live_board_strategies(cfg, month_start="2026-08-01")
    after = live.live_board_strategies(cfg, month_start="2026-09-01")
    assert "deepseek_v4_pro" not in {s["id"] for s in before}
    assert "deepseek_v4_pro" in {s["id"] for s in after}
    # Baselines are never gated.
    assert "spy_index" in {s["id"] for s in before}


def test_the_get_gates_on_the_live_month_not_the_competition_window(monkeypatch):
    """The base config's start_date is the April Competition window. Gating on
    it would hide every entry whose live_from is after April."""
    _use_config(monkeypatch, _config_with("deepseek_v4_pro", live_from="2026-05-01"))
    payload = live.get_live_leaderboard(as_of=_AUG27)
    assert "deepseek_v4_pro" in {e["entry_id"] for e in payload["entries"]}
    assert "deepseek_v4_pro" in payload["live_status"]["roster"]


def test_a_not_yet_live_entry_is_neither_rendered_nor_deployed(monkeypatch, tmp_path):
    _use_config(monkeypatch, _config_with("deepseek_v4_pro", live_from="2026-09-01"))
    # A row left over from an earlier run must not put it back on the board.
    _insert_live_run(
        "lb_deepseek_v4_pro_20260801_20260826", "deepseek_v4_pro", "2026-08-26",
        [_pt("2026-08-26T19:00:00+00:00", SEED)],
    )
    payload = live.get_live_leaderboard(as_of=_AUG27)
    assert "deepseek_v4_pro" not in {e["entry_id"] for e in payload["entries"]}
    assert "deepseek_v4_pro" not in payload["live_status"]["roster"]

    deployed = []
    monkeypatch.setattr(live, "_LIVE_REFRESH_STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(
        live,
        "deploy_live_model_increment",
        lambda entry, freeze_cfg, **kwargs: deployed.append(entry["id"])
        or {"entry_id": entry["id"]},
    )
    live.refresh_live_leaderboard(deploy_models=True, as_of=_AUG27)
    assert "deepseek_v4_pro" not in deployed


@pytest.mark.parametrize(
    "value",
    ["2026-09", "2026-09-15", "2026-13-01", "Sept 2026", "", 20260901, ["2026-09-01"]],
)
def test_a_malformed_live_from_warns_once_and_leaves_the_entry_live(
    monkeypatch, capsys, value
):
    _use_config(monkeypatch, _config_with("deepseek_v4_pro", live_from=value))
    assert "deepseek_v4_pro" in _live_ids(_AUG27)
    assert "deepseek_v4_pro" in _live_ids(_AUG27)
    out = capsys.readouterr().out
    line = "WARNING: live.config_invalid entry=deepseek_v4_pro key=live_from"
    assert out.count(line) == 1
    assert "expected=YYYY-MM-01" in out


def test_a_null_live_from_is_absent_and_silent(monkeypatch, capsys):
    _use_config(monkeypatch, _config_with("deepseek_v4_pro", live_from=None))
    assert "deepseek_v4_pro" in _live_ids(_AUG27)
    assert "WARNING" not in capsys.readouterr().out


def test_live_llm_entries_follow_live_model_ids_order(monkeypatch):
    """Deploy order is roster order, not file order (the file lists Haiku first)."""
    roster = ("qwen3_7_plus", "deepseek_v4_pro", "claude_haiku_4_5")
    monkeypatch.setattr(live, "LIVE_MODEL_IDS", roster)
    entries = live.live_llm_entries(
        lb_service.load_leaderboard_config(), month_start=_EVERY_JOIN_OPEN
    )
    assert [e["id"] for e in entries] == list(roster)


def test_the_deploy_cli_says_why_it_refuses_a_not_yet_live_entry(monkeypatch, capsys):
    from argparse import Namespace

    from dashboard.backend.tests.test_leaderboard_samples import _load_deploy_script

    _use_config(monkeypatch, _config_with("deepseek_v4_pro", live_from="2026-09-01"))
    freeze_cfg = live.live_freeze_config(_AUG27)
    monkeypatch.setattr(live, "live_freeze_config", lambda *args, **kwargs: freeze_cfg)
    monkeypatch.setattr(
        live,
        "deploy_live_model_increment",
        lambda *args, **kwargs: pytest.fail("the CLI deployed a not-yet-live entry"),
    )
    script = _load_deploy_script(monkeypatch)
    args = Namespace(
        entry="deepseek_v4_pro", start=None, end=None, force=False, allow_fallback=False
    )
    assert script._deploy_live(args) is None
    assert "its live_from month has not opened" in capsys.readouterr().out
```

- [ ] **Step 2: Run it and confirm it fails.**

```bash
pytest dashboard/backend/tests/test_live_roster.py -q -p no:cacheprovider
```

Expected: `14 errors`. Every case errors at setup in the autouse `_fresh_warnings` with `AttributeError: <module 'dashboard.backend.domain.leaderboard.live' …> has no attribute '_warned_live_config'`.

- [ ] **Step 3: Implement.**

In `dashboard/backend/domain/leaderboard/live.py`, find (≈:22 on main@17c552c5):

```python
import os
import tempfile
```

Replace with:

```python
import os
import re
import tempfile
```

In `dashboard/backend/domain/leaderboard/live.py`, find (≈:74 on main@17c552c5; the roster tuple itself stays as it is until Task 4.3):

```python
LIVE_MODEL_IDS = (
    "deepseek_v4_pro",
    "nemotron_3_nano_30b",
)
```

Replace with:

```python
LIVE_MODEL_IDS = (
    "deepseek_v4_pro",
    "nemotron_3_nano_30b",
)
# Optional per-entry "live_from": "YYYY-MM-01", the first board month an LLM
# entry trades. An entry added to LIVE_MODEL_IDS without it replays the current
# month from the 1st on its first refresh (it has no checkpoint to resume
# from), backfilling sessions it never saw forward. With it, the entry stays
# off the board -- neither deployed nor rendered -- until that month opens.
# Anything but the 1st of a month is reported once and ignored, so the entry
# is live at once: this is read on the public GET, which must not raise on a
# config typo.
LIVE_FROM_KEY = "live_from"
# (entry id, key, repr(value)) already reported -- see _warn_live_config.
_warned_live_config: set = set()
_LIVE_FROM_SHAPE = re.compile(r"\d{4}-\d{2}-01")
```

In `dashboard/backend/domain/leaderboard/live.py`, find (≈:288 on main@17c552c5):

```python
def live_board_strategies(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Baselines plus the Season-0 Live LLM roster (not the full contest list)."""
    cfg = config or lb_service.load_leaderboard_config()
    out: List[Dict[str, Any]] = []
    for strategy in cfg.get("strategies", []):
        if strategy.get("strategy") == "llm_agent" and strategy.get("id") not in LIVE_MODEL_IDS:
            continue
        out.append(strategy)
    return out


def live_llm_entries(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return [
        s for s in live_board_strategies(config)
        if s.get("strategy") == "llm_agent"
    ]
```

Replace with:

```python
def _warn_live_config(entry_id: Any, key: str, value: Any, expected: str) -> None:
    """Report a malformed Live-only config value, once per process."""
    marker = (str(entry_id), key, repr(value))
    if marker in _warned_live_config:
        return
    _warned_live_config.add(marker)
    print(
        f"WARNING: live.config_invalid entry={entry_id} key={key} "
        f"value={value!r} expected={expected}; ignoring it",
        flush=True,
    )


def _live_from_month(entry: Dict[str, Any]) -> Optional[date]:
    """The entry's ``live_from`` month, or None when it is unset or malformed."""
    raw = entry.get(LIVE_FROM_KEY)
    if raw is None:
        return None
    if isinstance(raw, str) and _LIVE_FROM_SHAPE.fullmatch(raw):
        try:
            return date.fromisoformat(raw)
        except ValueError:
            pass
    _warn_live_config(entry.get("id"), LIVE_FROM_KEY, raw, "YYYY-MM-01")
    return None


def _board_month(config: Dict[str, Any], month_start: Optional[str]) -> date:
    """The first day of the month a roster is asked for.

    A freeze config (``live_freeze_config``) carries it as ``start_date``. The
    base leaderboard config's ``start_date`` is the Competition window's, so a
    caller holding that one passes ``month_start``. Neither: this month.
    """
    if month_start:
        return date.fromisoformat(month_start).replace(day=1)
    if config.get("period") == "live" and config.get("start_date"):
        return date.fromisoformat(str(config["start_date"])).replace(day=1)
    return date.fromisoformat(live_month_dates()[0])


def live_board_strategies(
    config: Optional[Dict[str, Any]] = None,
    *,
    month_start: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Baselines plus the LLM entries live in the board month, in config order.

    An LLM entry is live when its id is in ``LIVE_MODEL_IDS`` and its
    ``live_from`` month, if it sets one, has opened. ``month_start`` names the
    board month when ``config`` is not a freeze config (see ``_board_month``).
    """
    cfg = config or lb_service.load_leaderboard_config()
    board_month = _board_month(cfg, month_start)
    out: List[Dict[str, Any]] = []
    for strategy in cfg.get("strategies", []):
        if strategy.get("strategy") == "llm_agent":
            if strategy.get("id") not in LIVE_MODEL_IDS:
                continue
            live_from = _live_from_month(strategy)
            if live_from is not None and live_from > board_month:
                continue
        out.append(strategy)
    return out


def live_llm_entries(
    config: Optional[Dict[str, Any]] = None,
    *,
    month_start: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """The board month's live LLM entries in ``LIVE_MODEL_IDS`` order.

    That is also the order ``refresh_live_leaderboard`` deploys them in, so
    the roster's first entry is never queued behind a newcomer.
    """
    entries = [
        s for s in live_board_strategies(config, month_start=month_start)
        if s.get("strategy") == "llm_agent"
    ]
    return sorted(entries, key=lambda s: LIVE_MODEL_IDS.index(s["id"]))
```

In `dashboard/backend/domain/leaderboard/live.py`, find (≈:1407 on main@17c552c5, in get_live_leaderboard):

```python
    strategies = live_board_strategies(config)
```

Replace with:

```python
    strategies = live_board_strategies(config, month_start=start_date)
```

In `dashboard/backend/domain/leaderboard/live.py`, find (≈:1499 on main@17c552c5, in get_live_leaderboard's live_status):

```python
        "roster": list(LIVE_MODEL_IDS),
```

Replace with:

```python
        # This month's live LLM ids, in roster order (``strategies`` is gated).
        "roster": [
            model_id for model_id in LIVE_MODEL_IDS
            if any(s["id"] == model_id for s in strategies)
        ],
```

If an earlier phase already added `import re` to `live.py`, skip the first edit above.

In `dashboard/scripts/deploy_leaderboard_model.py`, find (≈:253 on main@17c552c5, in _deploy_live):

```python
        print(f"ERROR: '{args.entry}' is not on the Live roster.")
```

Replace with:

```python
        print(
            f"ERROR: '{args.entry}' is not on the Live roster this month "
            "(not in LIVE_MODEL_IDS, or its live_from month has not opened)."
        )
```

Every existing caller keeps working unchanged. `refresh_live_leaderboard`, `deploy_leaderboard_model.py` and `refresh_live_leaderboard.py` pass a freeze config, which carries the month. `get_live_leaderboard` now passes `month_start=start_date` explicitly, because the base config's `start_date` is the April Competition window.

- [ ] **Step 4: Run it and confirm it passes.**

```bash
pytest dashboard/backend/tests/test_live_roster.py -q -p no:cacheprovider
pytest dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/test_leaderboard_season.py dashboard/backend/tests/test_frontend_live_leaderboard.py dashboard/backend/tests/test_frontend_live_trading_board.py dashboard/backend/tests/test_live_trade_stats.py -q -p no:cacheprovider
```

Expected: `14 passed`. The second command runs the neighbouring suites and must stay green. On main@17c552c5 that is `121 passed`; earlier phases change the count, never the colour.

- [ ] **Step 5: Commit.**

```bash
git add dashboard/backend/domain/leaderboard/live.py dashboard/scripts/deploy_leaderboard_model.py dashboard/backend/tests/test_live_roster.py
git commit -m "feat(live): optional per-entry live_from gate on the Live roster"
```

### Task 4.2: Haiku's Live-only id and temperature via `live_overrides`

**Files:**
- Modify: `dashboard/backend/domain/leaderboard/live.py`
- Modify: `dashboard/config/leaderboard.json`
- Modify (test): `dashboard/backend/tests/test_live_roster.py`
- Modify (test): `dashboard/backend/tests/test_leaderboard_samples.py`
- Modify (test, comment only): `dashboard/backend/tests/domain/leaderboard/test_strategies_move.py`

**Interfaces:**
- Consumes: Task 4.1's `_warn_live_config`, `live_board_strategies`, `live_llm_entries`, `_config_with`, `_use_config`, `_DROP`, `_EVERY_JOIN_OPEN`, `_AUG27`; `test_live_leaderboard.py`'s `_ScriptedAgent` and `_one_bar_frame`; and `test_leaderboard_samples.py`'s `_config`, `_seed_sample`, `_START`, `_END`, `_DISPLAY_CAPITAL`, `_entry` and `board` (all already imported there).
- Produces: `LIVE_OVERRIDES_KEY`, `LIVE_OVERRIDE_FIELDS`, `_with_live_overrides`; `live_board_strategies` returns LLM entries with overrides applied; Haiku's `live_overrides` block in `leaderboard.json`.

- [ ] **Step 1: Write the failing test.** Extend the imports of `dashboard/backend/tests/test_live_roster.py`:

In `dashboard/backend/tests/test_live_roster.py`, find (added in Task 4.1):

```python
import dashboard.backend.domain.leaderboard.live as live
import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.tests.test_live_leaderboard import (  # noqa: F401
    SEED,
    _insert_live_run,
    _isolated_live_session,
    _pt,
)
```

Replace with:

```python
from dashboard.backend.database import db
import dashboard.backend.domain.leaderboard.live as live
import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.tests.test_live_leaderboard import (  # noqa: F401
    SEED,
    _ScriptedAgent,
    _insert_live_run,
    _isolated_live_session,
    _one_bar_frame,
    _pt,
)
```

Append to the end of `dashboard/backend/tests/test_live_roster.py`:

```python
# --------------------------------------------------------------------------
# live_overrides: what the Live board sends, without touching the shared entry
# --------------------------------------------------------------------------

_HAIKU = "claude_haiku_4_5"
_AUG26 = datetime(2026, 8, 26, 17, 39, tzinfo=_ET)


def _live_entry(entry_id, cfg=None):
    entries = live.live_llm_entries(
        cfg or lb_service.load_leaderboard_config(), month_start=_EVERY_JOIN_OPEN
    )
    return next(e for e in entries if e["id"] == entry_id)


def test_haiku_runs_live_on_its_served_id_at_temperature_zero(monkeypatch):
    monkeypatch.setattr(live, "LIVE_MODEL_IDS", ("deepseek_v4_pro", _HAIKU))
    haiku = _live_entry(_HAIKU)
    assert haiku["model_id"] == "anthropic/claude-haiku-4-5"
    assert haiku["temperature"] == 0
    assert haiku["integration"] == "commonstack"
    # Claude keeps the Messages client: no thinking control, provider default.
    assert "reasoning_effort" not in haiku
    assert live.LIVE_OVERRIDES_KEY not in haiku
    strategy = lb_service.get_strategy(haiku)
    assert strategy.model_id == "anthropic/claude-haiku-4-5"
    assert strategy.temperature == 0
    assert strategy.reasoning_effort is None


def test_the_override_is_recorded_and_the_next_night_resumes(monkeypatch, capsys):
    """The checkpoint records the overridden values, and the resume check
    compares against the same values, so night two trades one session."""
    _use_config(monkeypatch, _config_with(_HAIKU, live_from=_DROP))
    monkeypatch.setattr(live, "LIVE_MODEL_IDS", ("deepseek_v4_pro", _HAIKU))
    built, runs = [], []

    def build(entry):
        built.append(entry)
        agent = _ScriptedAgent(runs=runs)
        agent.model_id = entry["model_id"]
        return agent

    monkeypatch.setattr(lb_service, "get_strategy", build)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())

    night_one = live.live_freeze_config(_AUG26)
    row = live.deploy_live_model_increment(_live_entry(_HAIKU, night_one), night_one)
    recorded = db.get_run(row["run_id"])["metadata"]
    assert recorded["model_id"] == "anthropic/claude-haiku-4-5"
    assert recorded["temperature"] == 0
    assert built[-1]["model_id"] == "anthropic/claude-haiku-4-5"

    runs.clear()
    night_two = live.live_freeze_config(_AUG27)
    live.deploy_live_model_increment(_live_entry(_HAIKU, night_two), night_two)
    assert runs == [("2026-08-27", "2026-08-27", "08-26")]
    assert "live.config_changed" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("block", "key"),
    [
        ("anthropic/claude-haiku-4-5", "live_overrides"),
        ({"temprature": 0}, "live_overrides.temprature"),
        ({"live_from": "2026-11-01"}, "live_overrides.live_from"),
    ],
)
def test_a_malformed_override_warns_once_and_runs_the_shared_values(
    monkeypatch, capsys, block, key
):
    cfg = _config_with(_HAIKU, live_overrides=block, live_from=_DROP)
    monkeypatch.setattr(live, "LIVE_MODEL_IDS", ("deepseek_v4_pro", _HAIKU))
    for _ in range(2):
        haiku = _live_entry(_HAIKU, cfg)
    assert haiku["model_id"] == "claude-haiku-4-5-20251001"
    assert "temperature" not in haiku
    assert live.LIVE_OVERRIDES_KEY not in haiku
    out = capsys.readouterr().out
    assert out.count(f"WARNING: live.config_invalid entry={_HAIKU} key={key} ") == 1
```

Append to the end of `dashboard/backend/tests/test_leaderboard_samples.py`. Every name it uses (`_config`, `_seed_sample`, `_entry`, `_START`, `_END`, `_DISPLAY_CAPITAL`, `board`) is already defined or imported at the top of that file:

```python
# --------------------------------------------------------------------------
# the Live board's Haiku pin must not re-judge Haiku's Competition repeats
# --------------------------------------------------------------------------

_HAIKU = "claude_haiku_4_5"
# What Haiku's three Competition repeat runs recorded when they were deployed
# (2026-10-03). Stated, not read from leaderboard.json: the case is that the
# shared entry still agrees with them.
_HAIKU_REPEATS_RECORDED = {
    "entry_id": _HAIKU,
    "model_id": "claude-haiku-4-5-20251001",
    "integration": "commonstack",
    "temperature": None,
    "reasoning_effort": None,
}


def test_the_live_haiku_pin_leaves_its_competition_median_alone(board, capsys):
    """Haiku's Live-board id and temperature sit under ``live_overrides``. Set
    on the shared entry, they make its repeats a different config than the one
    configured, and the July single run replaces their median on the board."""
    july = board.seed_run(
        _HAIKU,
        initial_equity=_DISPLAY_CAPITAL,
        equities=[_DISPLAY_CAPITAL, _DISPLAY_CAPITAL * 1.03],
        total_return=0.03,
    )
    for n, total_return in ((1, -0.0252), (2, 0.0019), (3, 0.0108)):
        _seed_sample(
            board, n, total_return, entry=_HAIKU,
            metadata=_config(**_HAIKU_REPEATS_RECORDED),
        )

    entry = _entry(lb_service.get_leaderboard(), _HAIKU)

    assert entry["run_id"] == lb_service._sample_run_id(_HAIKU, _START, _END, 2)
    assert entry["run_id"] != july
    assert entry["samples"]["count"] == 3
    assert "repeat run(s) recorded under a different" not in capsys.readouterr().out
```

Record why Haiku is absent from the Competition sampling set:

In `dashboard/backend/tests/domain/leaderboard/test_strategies_move.py`, find (≈:143 on main@17c552c5; it occurs once, on the line just above `_PINNED_SAMPLING_ENTRIES = {`):

```python
# PINNED_NO_THINKING for both.
```

Replace with:

```python
# PINNED_NO_THINKING for both. Claude Haiku 4.5 pins temperature 0 on the Live
# board only, under its "live_overrides" block, so it is not counted here: its
# shared keys are what its Competition runs recorded (see
# test_leaderboard_samples.py::test_the_live_haiku_pin_leaves_its_competition_median_alone).
```

This is a comment-only edit. Leave the `_PINNED_SAMPLING_ENTRIES = {"nemotron_3_nano_30b", "deepseek_v4_pro", "qwen3_7_plus"}` line below it byte-for-byte as it is. An earlier execution of this step retyped that line and committed `_PINNED_SAMPLING_ENTRIES ={…}`, losing a space. `git diff dashboard/backend/tests/domain/leaderboard/test_strategies_move.py` must show one `-` line, which is the old comment line, and four `+` comment lines.

- [ ] **Step 2: Run it and confirm it fails.**

```bash
pytest dashboard/backend/tests/test_live_roster.py dashboard/backend/tests/test_leaderboard_samples.py -q -p no:cacheprovider
```

Expected: `5 failed`, the rest passing. The failures are:
- `test_haiku_runs_live_on_its_served_id_at_temperature_zero` and `test_the_override_is_recorded_and_the_next_night_resumes`, with `AssertionError: assert 'claude-haiku-4-5-20251001' == 'anthropic/claude-haiku-4-5'`;
- the three `test_a_malformed_override_warns_once_and_runs_the_shared_values[…]` cases, with `AttributeError: module 'dashboard.backend.domain.leaderboard.live' has no attribute 'LIVE_OVERRIDES_KEY'`.

`test_the_live_haiku_pin_leaves_its_competition_median_alone` **passes** already. It is the guard Step 4 proves can fail.

- [ ] **Step 3: Implement.**

In `dashboard/backend/domain/leaderboard/live.py`, find (added in Task 4.1):

```python
_LIVE_FROM_SHAPE = re.compile(r"\d{4}-\d{2}-01")
```

Replace with:

```python
_LIVE_FROM_SHAPE = re.compile(r"\d{4}-\d{2}-01")
# Keys an LLM entry may set for the Live board alone, under "live_overrides".
# leaderboard.json is shared with the Competition board, which picks the stored
# run it publishes by comparing each run's recorded model_id / integration /
# temperature / reasoning_effort with the entry's (service._entry_config_drift).
# Changing a shared key for the Live board would re-judge Competition runs made
# under the old value: Haiku's three repeat runs would stop publishing and its
# July single run would replace their median. An override changes what the
# Live board sends and records, and nothing else.
LIVE_OVERRIDES_KEY = "live_overrides"
LIVE_OVERRIDE_FIELDS = ("model_id", "integration", "temperature", "reasoning_effort")
```

In `dashboard/backend/domain/leaderboard/live.py`, find (added in Task 4.1):

```python
def _board_month(config: Dict[str, Any], month_start: Optional[str]) -> date:
```

Replace with:

```python
def _with_live_overrides(entry: Dict[str, Any]) -> Dict[str, Any]:
    """``entry`` as the Live board runs it: ``live_overrides`` applied, the block dropped.

    A block that is not an object, or a key outside ``LIVE_OVERRIDE_FIELDS``,
    is reported once and skipped; the entry then runs on its shared values.
    """
    if LIVE_OVERRIDES_KEY not in entry:
        return entry
    overrides = entry[LIVE_OVERRIDES_KEY]
    effective = {k: v for k, v in entry.items() if k != LIVE_OVERRIDES_KEY}
    if not isinstance(overrides, dict):
        _warn_live_config(entry.get("id"), LIVE_OVERRIDES_KEY, overrides, "an object")
        return effective
    for key, value in overrides.items():
        if key in LIVE_OVERRIDE_FIELDS:
            effective[key] = value
        else:
            _warn_live_config(
                entry.get("id"),
                f"{LIVE_OVERRIDES_KEY}.{key}",
                value,
                "|".join(LIVE_OVERRIDE_FIELDS),
            )
    return effective


def _board_month(config: Dict[str, Any], month_start: Optional[str]) -> date:
```

In `dashboard/backend/domain/leaderboard/live.py`, find (added in Task 4.1, live_board_strategies' docstring):

```python
    """Baselines plus the LLM entries live in the board month, in config order.

    An LLM entry is live when its id is in ``LIVE_MODEL_IDS`` and its
    ``live_from`` month, if it sets one, has opened. ``month_start`` names the
    board month when ``config`` is not a freeze config (see ``_board_month``).
    """
```

Replace with:

```python
    """Baselines plus the LLM entries live in the board month, in config order.

    An LLM entry is live when its id is in ``LIVE_MODEL_IDS`` and its
    ``live_from`` month, if it sets one, has opened. It is returned with its
    ``live_overrides`` applied, so every Live path (deploy, resume check, GET)
    sees the values the Live board actually sends. ``month_start`` names the
    board month when ``config`` is not a freeze config (see ``_board_month``).
    """
```

In `dashboard/backend/domain/leaderboard/live.py`, find (added in Task 4.1, the end of live_board_strategies):

```python
            live_from = _live_from_month(strategy)
            if live_from is not None and live_from > board_month:
                continue
        out.append(strategy)
    return out
```

Replace with:

```python
            live_from = _live_from_month(strategy)
            if live_from is not None and live_from > board_month:
                continue
            strategy = _with_live_overrides(strategy)
        out.append(strategy)
    return out
```

In `dashboard/config/leaderboard.json`, find (≈:65-66 on main@17c552c5, the claude_haiku_4_5 entry):

```json
      "model_id": "claude-haiku-4-5-20251001",
      "mode": "safe_trading",
```

Replace with:

```json
      "model_id": "claude-haiku-4-5-20251001",
      "live_overrides": {
        "model_id": "anthropic/claude-haiku-4-5",
        "temperature": 0
      },
      "mode": "safe_trading",
```

Do **not** touch Haiku's shared `model_id`, and do not add a shared `temperature`. The "Deviation from Decision 6 (index plan)" callout at the top of this phase explains why.

- [ ] **Step 4: Run it and confirm it passes.**

```bash
pytest dashboard/backend/tests/test_live_roster.py dashboard/backend/tests/test_leaderboard_samples.py dashboard/backend/tests/domain/leaderboard/test_strategies_move.py dashboard/backend/tests/infrastructure/llm/test_commonstack_thinking_off.py dashboard/backend/tests/test_marketplace_catalog_models.py dashboard/backend/tests/test_seed_database_integrity.py dashboard/backend/tests/test_live_leaderboard.py -q -p no:cacheprovider
```

Expected: all pass (`250 passed` on main@17c552c5 plus Tasks 4.1–4.2).

- [ ] **Step 5: Commit, then prove the Competition guard bites.** Run the lines one at a time.

```bash
git add dashboard/backend/domain/leaderboard/live.py dashboard/config/leaderboard.json dashboard/backend/tests/test_live_roster.py dashboard/backend/tests/test_leaderboard_samples.py dashboard/backend/tests/domain/leaderboard/test_strategies_move.py
git commit -m "feat(live): Haiku's Live-only served id and temperature 0 via live_overrides"
# Mutation: put the Live values on Haiku's SHARED keys (what Decision 6 in the index plan literally said), run the guard, restore.
sed -i '0,/      "model_id": "claude-haiku-4-5-20251001",/s//      "model_id": "anthropic\/claude-haiku-4-5",\n      "temperature": 0,/' dashboard/config/leaderboard.json
pytest "dashboard/backend/tests/test_leaderboard_samples.py::test_the_live_haiku_pin_leaves_its_competition_median_alone" -q -p no:cacheprovider
git checkout -- dashboard/config/leaderboard.json
git status --short   # must be clean
```

Expected from the `pytest` line: `1 failed` with `AssertionError: assert 'lb_claude_ha...0415_20260515' == 'lb_claude_ha...5_20260515_s2'`, meaning the July primary is published instead of the median repeat. Restoring leaves a clean tree.

### Task 4.3: Roster — DeepSeek, Qwen, Haiku; Qwen and Haiku from the next month

**Files:**
- Modify: `dashboard/backend/domain/leaderboard/live.py`
- Modify: `dashboard/config/leaderboard.json`
- Modify (test): `dashboard/backend/tests/test_live_leaderboard.py`
- Modify (test): `dashboard/backend/tests/test_live_roster.py`

**Interfaces:**
- Consumes: Task 4.1's `LIVE_FROM_KEY`, `live_llm_entries`, `_live_ids`, `_config_with`; Task 4.2's `_live_entry`; `test_live_leaderboard.py`'s `_ScriptedAgent`, `_insert_live_run`, `_one_bar_frame`, `_pt`, `SEED`; and `is_trading_day` from `dashboard/backend/infrastructure/market_data/us_market_calendar.py`.
- Produces: `LIVE_MODEL_IDS = ("deepseek_v4_pro", "qwen3_7_plus", "claude_haiku_4_5")`; `live_from` on `qwen3_7_plus` and `claude_haiku_4_5`; the autouse fixture `_roster_without_join_dates` in `test_live_leaderboard.py`; and the test helpers `_NEWCOMERS`, `_join_month`, `_trading_days_from`, `_evening` in `test_live_roster.py`.

**Pick the join month first.** Use the 1st of the first calendar month that starts after the PR's merge date. The default is `"2026-11-01"` (merge in October 2026). If the merge will land on or after 2026-11-01, use the 1st of the next month instead (for example `"2026-12-01"`) in **both** entries. No test hard-codes the month, because every case reads it from `leaderboard.json`.

- [ ] **Step 1: Write the failing test.**

(a) `test_live_leaderboard.py` uses Nemotron as "the Live model that is not DeepSeek" in about ten August-2026 cases. Rename it to Qwen in that one file. This is a whole-file rename, so it stays correct whatever earlier phases did to those cases:

```bash
rg -ln nemotron_3_nano_30b dashboard/backend/tests   # see "Depends on": only test_live_leaderboard.py is renamed
sed -i -e 's/nemotron_3_nano_30b/qwen3_7_plus/g' -e 's/Nemotron 3 Nano 30B/Qwen3.7 Plus/g' -e 's#nvidia/nemotron-3-nano-30b-a3b#qwen/qwen3.7-plus#g' -e 's/\bnemotron\b/qwen/g' dashboard/backend/tests/test_live_leaderboard.py
rg -c -i nemotron dashboard/backend/tests/test_live_leaderboard.py   # expect no output (zero matches)
git status --short   # expect exactly one line:  M dashboard/backend/tests/test_live_leaderboard.py
```

**If P5 merged before this phase.** The planned order puts P5 (`2026-10-14-live-board-fixes-p5-daily-sharpe.md`) after this phase. If it merged first, `test_live_leaderboard.py` already holds P5's Sharpe assertions in `test_returns_come_off_the_stored_run_and_keep_the_first_hour`, written against the main-only `nemotron[...]` form (P5 Task 5.2 Step 1, Form A). The sed renames them along with everything else. After the sed, read the diff on those lines:

```bash
git diff -U0 dashboard/backend/tests/test_live_leaderboard.py | rg sharpe
```

Without P5 it prints one pair: `-    assert nemotron["sharpe_ratio"] == pytest.approx(1.1)` and the same line with `qwen`. With P5 merged it prints five pairs: `nemotron["sharpe_ratio"]`, `["sharpe_ci_low"]`, `["sharpe_ci_high"]`, `["sharpe_days"]` and `["sharpe_status"]` as `-` lines, and the same five under `qwen[...]` as `+` lines, with every value unchanged (`2.906881`, `-4.487853`, `10.301616`, `18`, `"ok"`). Keep P5's assertions under the `qwen` name. Do not restore the `nemotron` form, change a value or drop a line: the figures come from the fixture curve, not from the entry. Measured on main@17c552c5 plus P5: exactly those five pairs, and with this phase applied on top, `test_live_leaderboard.py` and `test_live_roster.py` gave `87 passed`.

Run the sed once. Do not re-run it over a file that already holds these lines (after a reset, a rebase or a conflict) without checking first. `rg -n -i nemotron dashboard/backend/tests/test_live_leaderboard.py` must print nothing before Step 1(b), and exactly one line after it: `assert "nemotron_3_nano_30b" not in deployed`. That line is deliberate. A sed re-run after Step 1(b) turns it into `assert "qwen3_7_plus" not in deployed`, which then fails with `AssertionError: assert 'qwen3_7_plus' not in ['deepseek_v4_pro', 'qwen3_7_plus', 'claude_haiku_4_5']` (measured). If you re-run it anyway, restore that line and read `git diff` on the Sharpe lines again before committing.

(b) Then make these edits to the renamed file. The `find` blocks are the text *after* the rename:

In `dashboard/backend/tests/test_live_leaderboard.py`, find (≈:35-37 on main@17c552c5, just below _isolated_live_session):

```python
@pytest.fixture
def client():
    return TestClient(app)
```

Replace with:

```python
@pytest.fixture(autouse=True)
def _roster_without_join_dates(monkeypatch):
    """These cases are dated August 2026, before any entry's ``live_from``
    month, and are about the board, not about who joins when: read without
    that key, the whole roster is live. The join gate has its own cases in
    test_live_roster.py, against the real config."""
    real = lb_service.load_leaderboard_config

    def without_join_dates():
        cfg = real()
        return {
            **cfg,
            "strategies": [
                {k: v for k, v in s.items() if k != live.LIVE_FROM_KEY}
                for s in cfg["strategies"]
            ],
        }

    monkeypatch.setattr(lb_service, "load_leaderboard_config", without_join_dates)


@pytest.fixture
def client():
    return TestClient(app)
```

In `dashboard/backend/tests/test_live_leaderboard.py`, find (≈:439-447 on main@17c552c5, in test_live_month_run_shows_llm_snapshot):

```python
    assert payload["live_status"]["models_pending"] == 1
    pending = [e for e in payload["entries"] if e["status"] == "pending"]
    assert pending and all(e["rank"] is None for e in pending)
    assert payload["live_status"]["roster"] == [
        "deepseek_v4_pro",
        "qwen3_7_plus",
    ]
    model_ids = {e["entry_id"] for e in payload["entries"] if e.get("is_model")}
    assert model_ids == {"deepseek_v4_pro", "qwen3_7_plus"}
```

Replace with:

```python
    assert payload["live_status"]["models_pending"] == 2
    pending = [e for e in payload["entries"] if e["status"] == "pending"]
    assert pending and all(e["rank"] is None for e in pending)
    assert payload["live_status"]["roster"] == [
        "deepseek_v4_pro",
        "qwen3_7_plus",
        "claude_haiku_4_5",
    ]
    model_ids = {e["entry_id"] for e in payload["entries"] if e.get("is_model")}
    assert model_ids == {"deepseek_v4_pro", "qwen3_7_plus", "claude_haiku_4_5"}
```

In `dashboard/backend/tests/test_live_leaderboard.py`, find (≈:487-489 on main@17c552c5, in test_refresh_live_leaderboard_deploys_models):

```python
    assert "claude_haiku_4_5" not in deployed
    assert "gpt_5_5" not in deployed
    assert set(deployed) == {"deepseek_v4_pro", "qwen3_7_plus"}
```

Replace with:

```python
    assert "gpt_5_5" not in deployed
    assert "nemotron_3_nano_30b" not in deployed
    # Roster order, not file order: the file lists Haiku first.
    assert deployed == ["deepseek_v4_pro", "qwen3_7_plus", "claude_haiku_4_5"]
```

(c) Extend the imports of `dashboard/backend/tests/test_live_roster.py`:

In `dashboard/backend/tests/test_live_roster.py`, find (added in Task 4.1):

```python
from datetime import datetime
from zoneinfo import ZoneInfo
```

Replace with:

```python
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
```

In `dashboard/backend/tests/test_live_roster.py`, find (added in Task 4.1):

```python
import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.tests.test_live_leaderboard import (  # noqa: F401
```

Replace with:

```python
import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.infrastructure.market_data.us_market_calendar import (
    is_trading_day,
)
from dashboard.backend.tests.test_live_leaderboard import (  # noqa: F401
```

(d) Append to the end of `dashboard/backend/tests/test_live_roster.py`:

```python
# --------------------------------------------------------------------------
# the roster as configured
# --------------------------------------------------------------------------

_NEWCOMERS = ("qwen3_7_plus", "claude_haiku_4_5")


def _join_month():
    """The month Qwen and Haiku join, read from leaderboard.json."""
    months = {
        s.get("live_from")
        for s in lb_service.load_leaderboard_config()["strategies"]
        if s["id"] in _NEWCOMERS
    }
    assert len(months) == 1, f"the newcomers join together: {months}"
    return date.fromisoformat(months.pop())


def _trading_days_from(day, count):
    days = []
    while len(days) < count:
        if is_trading_day(day):
            days.append(day)
        day += timedelta(days=1)
    return days


def _evening(day):
    """After the close of ``day`` has settled, on any session length."""
    return datetime(day.year, day.month, day.day, 17, 39, tzinfo=_ET)


def test_the_roster_is_deepseek_qwen_haiku_in_that_order():
    assert live.LIVE_MODEL_IDS == ("deepseek_v4_pro", "qwen3_7_plus", "claude_haiku_4_5")


def test_before_the_join_month_only_deepseek_is_on_the_board():
    month_before = (_join_month() - timedelta(days=1)).replace(day=1)
    as_of = _evening(_trading_days_from(month_before, 1)[0])
    assert _live_ids(as_of) == ["deepseek_v4_pro"]
    payload = live.get_live_leaderboard(as_of=as_of)
    models = {e["entry_id"] for e in payload["entries"] if e.get("is_model")}
    assert models == {"deepseek_v4_pro"}
    assert payload["live_status"]["roster"] == ["deepseek_v4_pro"]


def test_from_the_join_month_the_whole_roster_is_on_the_board():
    as_of = _evening(_trading_days_from(_join_month(), 1)[0])
    assert _live_ids(as_of) == list(live.LIVE_MODEL_IDS)
    payload = live.get_live_leaderboard(as_of=as_of)
    assert payload["live_status"]["roster"] == list(live.LIVE_MODEL_IDS)
    assert payload["live_status"]["models_pending"] == 3


def test_newcomers_start_at_the_1st_and_deepseek_just_resumes(
    monkeypatch, tmp_path, capsys
):
    """Adding entries is not a config change for the ones already trading:
    DeepSeek appends one session, and only the newcomers trade from the 1st."""
    join = _join_month()
    first, second = _trading_days_from(join, 2)
    freeze_cfg = live.live_freeze_config(_evening(second))
    deepseek = next(
        e for e in live.live_llm_entries(freeze_cfg) if e["id"] == "deepseek_v4_pro"
    )
    _insert_live_run(
        f"lb_deepseek_v4_pro_{join:%Y%m%d}_{first:%Y%m%d}",
        "deepseek_v4_pro",
        first.isoformat(),
        [_pt(f"{first.isoformat()}T19:00:00+00:00", SEED)],
        start_date=join.isoformat(),
        metadata={
            "entry_id": "deepseek_v4_pro",
            "model_id": deepseek["model_id"],
            "integration": deepseek.get("integration"),
            "temperature": deepseek.get("temperature"),
            "reasoning_effort": deepseek.get("reasoning_effort"),
            "strategy_prompt": None,
            live.LIVE_SNAPSHOT_KEY: {
                "cash": SEED, "positions": {}, "day": first.strftime("%m-%d"),
            },
        },
    )
    runs = {}
    order = []

    def build(entry):
        order.append(entry["id"])
        agent = _ScriptedAgent(runs=runs.setdefault(entry["id"], []))
        agent.model_id = entry["model_id"]
        return agent

    monkeypatch.setattr(live, "_LIVE_REFRESH_STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "get_strategy", build)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())

    result = live.refresh_live_leaderboard(deploy_models=True, as_of=_evening(second))

    assert result["model_failures"] == []
    assert order == ["deepseek_v4_pro", "qwen3_7_plus", "claude_haiku_4_5"]
    assert runs["deepseek_v4_pro"] == [
        (second.isoformat(), second.isoformat(), first.strftime("%m-%d"))
    ]
    for newcomer in _NEWCOMERS:
        assert runs[newcomer] == [(join.isoformat(), second.isoformat(), None)]
    assert "live.config_changed" not in capsys.readouterr().out
```

- [ ] **Step 2: Run it and confirm it fails.**

```bash
pytest dashboard/backend/tests/test_live_roster.py dashboard/backend/tests/test_live_leaderboard.py -q -p no:cacheprovider
```

Expected: `12 failed`. The four new `test_live_roster.py` cases fail as follows:
- `test_the_roster_is_deepseek_qwen_haiku_in_that_order`: `AssertionError: assert ('deepseek_v4...n_3_nano_30b') == ('deepseek_v4...de_haiku_4_5')`;
- the other three: `TypeError: fromisoformat: argument must be str`, because there is no `live_from` yet.

The eight renamed `test_live_leaderboard.py` cases fail because `qwen3_7_plus` is not on the roster yet (`StopIteration`, `KeyError: 'qwen3_7_plus'` or set mismatches): `test_live_month_run_shows_llm_snapshot`, `test_refresh_live_leaderboard_deploys_models`, `test_deploy_live_increment_trades_only_the_new_session`, `test_a_model_with_no_run_is_never_ranked`, `test_returns_come_off_the_stored_run_and_keep_the_first_hour`, `test_live_entry_publishes_trade_stats_and_invested`, `test_dollar_axis_scales_by_the_recorded_seed_only`, `test_an_entry_several_sessions_behind_is_shown_but_not_ranked`.

- [ ] **Step 3: Implement.**

In `dashboard/backend/domain/leaderboard/live.py`, find (≈:66-77 on main@17c552c5):

```python
# Season-0 local roster: only these LLM curves are deployed and shown on Live.
# Contest/daily still use the full leaderboard.json list. GPT-5.5 is left out
# because it was most of the nightly token spend.
#
# A dropped model's rows are deliberately left in place, not pruned: GET only
# renders roster entries, so they are inert, and if the model comes back
# mid-month it resumes from its last snapshot and trades only the sessions it
# missed. Deleting them would make a re-add replay the month from the 1st.
LIVE_MODEL_IDS = (
    "deepseek_v4_pro",
    "nemotron_3_nano_30b",
)
```

Replace with:

```python
# Season-0 roster: only these LLM curves are deployed and shown on Live, in
# this order, which is also the nightly deploy order. Contest/daily still use
# the full leaderboard.json list. GPT-5.5, Gemini 3.1 Pro and Sonnet 4.6 are
# left out on cost (spec 2026-10-14 §7 prices each). Nemotron 3 Nano is out
# because CommonStack serves no NVIDIA model (GET /v1/models, 2026-10-14):
# every nightly call 404'd, so it never printed. Whether it returns for
# Season 1, and through which gateway, is the owner's call. Qwen3.7 Plus and
# Claude Haiku 4.5 join at a month boundary (``live_from`` in leaderboard.json).
#
# A dropped model's rows are deliberately left in place, not pruned: GET only
# renders roster entries, so they are inert, and if the model comes back
# mid-month it resumes from its last snapshot and trades only the sessions it
# missed. Deleting them would make a re-add replay the month from the 1st.
LIVE_MODEL_IDS = (
    "deepseek_v4_pro",
    "qwen3_7_plus",
    "claude_haiku_4_5",
)
```

In `dashboard/config/leaderboard.json`, find (≈:65 on main@17c552c5, claude_haiku_4_5):

```json
      "model_id": "claude-haiku-4-5-20251001",
```

Replace with:

```json
      "model_id": "claude-haiku-4-5-20251001",
      "live_from": "2026-11-01",
```

In `dashboard/config/leaderboard.json`, find (≈:133 on main@17c552c5, qwen3_7_plus):

```json
      "model_id": "qwen/qwen3.7-plus",
```

Replace with:

```json
      "model_id": "qwen/qwen3.7-plus",
      "live_from": "2026-11-01",
```

If you picked a join month other than November 2026, write it in both `live_from` values.

- [ ] **Step 4: Run it and confirm it passes.**

```bash
pytest dashboard/backend/tests/test_live_roster.py dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/test_leaderboard_samples.py dashboard/backend/tests/test_leaderboard_season.py dashboard/backend/tests/test_frontend_live_leaderboard.py dashboard/backend/tests/test_frontend_live_trading_board.py dashboard/backend/tests/test_live_trade_stats.py dashboard/backend/tests/domain/leaderboard/ dashboard/backend/tests/test_marketplace_catalog_models.py dashboard/backend/tests/infrastructure/llm/test_commonstack_thinking_off.py dashboard/backend/tests/test_seed_database_integrity.py -q -p no:cacheprovider
```

Expected: all pass (`377 passed` on main@17c552c5 plus Tasks 4.1–4.3). If a case added by an earlier phase fails, it is a roster list or roster count that now has three entries in August-dated fixtures. Update it to `deepseek_v4_pro, qwen3_7_plus, claude_haiku_4_5` in that order, and do not weaken the assertion.

- [ ] **Step 5: Commit.**

```bash
git add dashboard/backend/domain/leaderboard/live.py dashboard/config/leaderboard.json dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/test_live_roster.py
git commit -m "feat(live): roster DeepSeek, Qwen, Haiku; drop Nemotron; newcomers join next month"
```

### Task 4.4: Pin the Live roster to what CommonStack serves

**Files:**
- Create (test): `dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py`

**Interfaces:**
- Consumes: `live.live_llm_entries(config, month_start=…)`, `live.LIVE_MODEL_IDS`, `live._warned_live_config`, `pricing.listed_price_for_model(model) -> Optional[Tuple[float, float]]` (`dashboard/backend/infrastructure/llm/pricing.py`), `is_reasoning_off` (`dashboard/backend/infrastructure/llm/reasoning_controls.py`).
- Produces: `COMMONSTACK_SERVED` (test constant) and four config pins: every `LIVE_MODEL_IDS` id names a config entry; every live entry is `integration: commonstack` with a served model id; every thinking-off entry pins temperature 0; every live model is priced at its listed rate. A fifth pin checks that the committed `live_from`/`live_overrides` parse with no warning.

- [ ] **Step 1: Write the failing test.** Create `dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py` with exactly:

```python
"""The Live roster, checked against what its gateway serves.

Live-board calls go through CommonStack on ``COMMONSTACK_API_KEY``, outside the
Credits layer, so nothing at request time checks that a model id exists. An id
CommonStack does not serve fails every call, every night: Nemotron's OpenRouter
slug did exactly that until 2026-10-14. These cases pin the roster to the
gateway's own listing instead.
"""

import pytest

import dashboard.backend.domain.leaderboard.live as live
import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.infrastructure.llm import pricing
from dashboard.backend.infrastructure.llm.reasoning_controls import is_reasoning_off

# CommonStack GET /v1/models with the prod key on 2026-10-14: 86 models, none
# of them NVIDIA/Nemotron. Model id -> listed (input, output) USD per million
# tokens, copied for the ids this roster uses. Re-list before adding an id:
# the code's allowlists and pricing.py are not the listing.
COMMONSTACK_SERVED = {
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    "qwen/qwen3.7-plus": (0.40, 1.60),
    "anthropic/claude-haiku-4-5": (1.0, 5.0),
}
# After every live_from in leaderboard.json, so no entry is gated out.
_EVERY_JOIN_OPEN = "2099-12-01"


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(live, "_warned_live_config", set())


def _live_entries():
    """Every roster entry as the Live board runs it (overrides applied)."""
    return live.live_llm_entries(
        lb_service.load_leaderboard_config(), month_start=_EVERY_JOIN_OPEN
    )


def test_every_live_model_id_names_a_config_entry():
    """A typo in LIVE_MODEL_IDS silently drops that model from the board."""
    assert [e["id"] for e in _live_entries()] == list(live.LIVE_MODEL_IDS)


def test_every_live_entry_calls_a_model_commonstack_serves():
    for entry in _live_entries():
        assert entry["integration"] == "commonstack", entry["id"]
        assert entry["model_id"] in COMMONSTACK_SERVED, (entry["id"], entry["model_id"])


def test_live_entries_that_turn_thinking_off_pin_temperature_zero():
    for entry in _live_entries():
        if is_reasoning_off(entry.get("reasoning_effort")):
            assert entry.get("temperature") == 0, entry["id"]


def test_every_live_model_is_priced_at_its_listed_rate():
    """``est_cost_usd`` on every live row is priced from pricing.py. An id the
    table does not list is priced at ``_DEFAULT_PRICING``, which is a guess."""
    for entry in _live_entries():
        listed = COMMONSTACK_SERVED.get(entry["model_id"])
        assert listed is not None, (entry["id"], entry["model_id"])
        assert pricing.listed_price_for_model(entry["model_id"]) == pytest.approx(
            listed
        ), entry["id"]


def test_the_committed_live_config_raises_no_warning(capsys):
    """A malformed live_from or live_overrides is ignored with a WARNING at
    runtime, because the public GET must not fail on config. Committed, it is
    a bug: the entry would trade at once, or on its shared values."""
    _live_entries()
    assert "WARNING: live." not in capsys.readouterr().out
```

- [ ] **Step 2: Run it and confirm it fails.** These are guards over config that Task 4.3 already fixed, so they pass on the committed tree (`5 passed`). Prove each one bites against a deliberately broken tree. Run each mutation, then the file, then restore. Run the lines one at a time:

```bash
# (i) Nemotron back on the Live roster
sed -i 's/^    "claude_haiku_4_5",$/    "claude_haiku_4_5",\n    "nemotron_3_nano_30b",/' dashboard/backend/domain/leaderboard/live.py
pytest dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py -q -p no:cacheprovider
git checkout -- dashboard/backend/domain/leaderboard/live.py
# (ii) Haiku without its live_overrides
sed -i '/"live_overrides": {/,/},/d' dashboard/config/leaderboard.json
pytest dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py -q -p no:cacheprovider
git checkout -- dashboard/config/leaderboard.json
# (iii) a typo in LIVE_MODEL_IDS
sed -i 's/^    "qwen3_7_plus",$/    "qwen3_7_plux",/' dashboard/backend/domain/leaderboard/live.py
pytest dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py -q -p no:cacheprovider
git checkout -- dashboard/backend/domain/leaderboard/live.py
# (iv) a malformed live_from
sed -i '0,/"live_from": "\([0-9]\{4\}-[0-9]\{2\}\)-01",/s//"live_from": "\1",/' dashboard/config/leaderboard.json
pytest dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py -q -p no:cacheprovider
git checkout -- dashboard/config/leaderboard.json
git status --short   # only the new test file is untracked
```

Expected:
- (i): `2 failed`.
  - `test_every_live_entry_calls_a_model_commonstack_serves` fails on its `model_id in COMMONSTACK_SERVED` line with `AssertionError: ('nemotron_3_nano_30b', 'nvidia/nemotron-3-nano-30b-a3b')`.
  - `test_every_live_model_is_priced_at_its_listed_rate` fails one line before the `pricing.py` comparison, at `assert listed is not None`, because `COMMONSTACK_SERVED` has no row for the id. Its `E` lines are `AssertionError: ('nemotron_3_nano_30b', 'nvidia/nemotron-3-nano-30b-a3b')` followed by `assert None is not None`. The tuple appears only in the message line.
- (ii): `2 failed`, the same two tests on the same lines, with the tuple `('claude_haiku_4_5', 'claude-haiku-4-5-20251001')`. The priced test again shows `assert None is not None`.
- (iii): `1 failed`. `test_every_live_model_id_names_a_config_entry` fails with `At index 1 diff: 'claude_haiku_4_5' != 'qwen3_7_plux'`.
- (iv): `1 failed`. `test_the_committed_live_config_raises_no_warning` fails, showing `WARNING: live.config_invalid entry=claude_haiku_4_5 key=live_from value='2026-11' expected=YYYY-MM-01; ignoring it`.

- [ ] **Step 3: Implement.** No production change: Tasks 4.2–4.3 already made the config right. This task adds the guard.

- [ ] **Step 4: Run it and confirm it passes.**

```bash
pytest dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py dashboard/backend/tests/test_live_roster.py -q -p no:cacheprovider
```

Expected: all pass (`5 passed` for the new file).

- [ ] **Step 5: Commit.**

```bash
git add dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py
git commit -m "test(live): pin the Live roster to CommonStack's served models and listed prices"
```

### Task 4.5: Roster mentions name the constant, not the models

**Files:**
- Modify: `.github/workflows/live-leaderboard.yml` (comment only)
- Modify: `dashboard/scripts/refresh_live_leaderboard.py` (CLI help only)
- Modify (test): `dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py`

**Interfaces:**
- Consumes: Task 4.4's test module.
- Produces: `test_roster_mentions_name_the_constant_not_the_models`, a guard that the workflow comment and the CLI help reference `LIVE_MODEL_IDS` and never a model count or "Nemotron".

- [ ] **Step 1: Write the failing test.**

In `dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py`, find (added in Task 4.4):

```python
import pytest

import dashboard.backend.domain.leaderboard.live as live
```

Replace with:

```python
import re
from pathlib import Path

import pytest

import dashboard.backend.domain.leaderboard.live as live
```

Append to the end of `dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py`:

```python
_REPO = Path(__file__).resolve().parents[5]
_ROSTER_MENTIONS = (
    _REPO / ".github" / "workflows" / "live-leaderboard.yml",
    _REPO / "dashboard" / "scripts" / "refresh_live_leaderboard.py",
)


@pytest.mark.parametrize("path", _ROSTER_MENTIONS, ids=lambda p: p.name)
def test_roster_mentions_name_the_constant_not_the_models(path):
    """The workflow comment and the CLI help both listed the roster ("2 models
    (DeepSeek V4 Pro, Nemotron 3 Nano)") and went stale on its first change."""
    text = path.read_text(encoding="utf-8")
    assert "LIVE_MODEL_IDS" in text
    assert "Nemotron" not in text
    assert not re.search(r"\b\d+ models\b", text)
```

- [ ] **Step 2: Run it and confirm it fails.**

```bash
pytest dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py -q -p no:cacheprovider
```

Expected: `2 failed, 5 passed`. Both `test_roster_mentions_name_the_constant_not_the_models[live-leaderboard.yml]` and `[refresh_live_leaderboard.py]` fail on `assert 'LIVE_MODEL_IDS' in '…'`.

- [ ] **Step 3: Implement.**

In `.github/workflows/live-leaderboard.yml`, find (≈:9-12 on main@17c552c5):

```yaml
# when the repository variable LIVE_LEADERBOARD_DEPLOY_MODELS is exactly
# "true" — a recurring API cost for 2 models (DeepSeek V4 Pro, Nemotron 3
# Nano) every weekday. Unset, the schedule refreshes baselines only. A manual
# dispatch defaults to off too.
```

Replace with:

```yaml
# when the repository variable LIVE_LEADERBOARD_DEPLOY_MODELS is exactly
# "true" — a recurring API cost, every weekday, for each model on the Live
# roster: LIVE_MODEL_IDS in dashboard/backend/domain/leaderboard/live.py,
# each entry from its live_from month in dashboard/config/leaderboard.json.
# Unset, the schedule refreshes baselines only. A manual dispatch defaults to
# off too.
```

As planned, P3 (`2026-10-14-live-board-fixes-p3-attempt-ledger.md`, the budget hard stop) does not touch this file. If it, or any earlier phase, rewrote this comment anyway, keep its budget sentence. Only replace the "2 models (DeepSeek V4 Pro, Nemotron 3 Nano)" wording with the roster reference above.

In `dashboard/scripts/refresh_live_leaderboard.py`, find (≈:58 on main@17c552c5, the --models argument):

```python
        help="Append the Live LLM roster (DeepSeek / Nemotron) for new cash sessions (billable)",
```

Replace with:

```python
        help=(
            "Append this month's Live LLM roster (LIVE_MODEL_IDS, each entry "
            "from its live_from month) for new cash sessions (billable)"
        ),
```

- [ ] **Step 4: Run it and confirm it passes.**

```bash
pytest dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py "dashboard/backend/tests/test_live_leaderboard.py::test_workflow_schedule_only_bills_behind_an_explicit_variable" -q -p no:cacheprovider
python3 -m py_compile dashboard/scripts/refresh_live_leaderboard.py
```

Run the two lines as separate commands. Expected: `8 passed`, and `py_compile` prints nothing. It compiles without importing `dashboard.backend`, so the seed DB is untouched. Do **not** run the script itself or `--help`, because its imports touch the committed seed DB.

- [ ] **Step 5: Commit.**

```bash
git add .github/workflows/live-leaderboard.yml dashboard/scripts/refresh_live_leaderboard.py dashboard/backend/tests/domain/leaderboard/test_live_roster_config.py
git commit -m "docs(live): roster mentions name LIVE_MODEL_IDS, not the models"
```

### Task 4.6: Open the PR

**Files:** none changed.

**Interfaces:** consumes everything above; produces PR `feat(live): drop Nemotron, add Qwen and Haiku`.

- [ ] **Step 1: Run the full suite.**

```bash
pytest dashboard/backend/tests -q -p no:cacheprovider
git status --short   # must be clean; dashboard/storage/data/backtest.db must not appear
```

Run the command exactly as written. Do not add `-x`, which stops at the first `test_report_pdf.py` failure and hides everything after it. Do not add `--timeout=…` either: this environment has no pytest-timeout plugin, so pytest rejects the flag. The run takes about 7 minutes (425 s measured on main@17c552c5 plus this phase).

Expected: green except `test_report_pdf.py`. Its 7 failures are environmental locally, because `reportlab` is not installed, and this phase does not touch that file. Measured on main@17c552c5 plus this phase: `6717 passed, 176 skipped, 25 xfailed, 7 failed`. All 7 failures were in `test_report_pdf.py`: `test_produces_valid_pdf`, `test_empty_and_none_inputs`, `test_large_document_is_bounded`, `test_unicode_content`, `test_crossing_inline_markup_degrades_to_plain_text_not_a_crash`, `test_page_cap_truncates_instead_of_dropping_the_pdf` and `test_report_lists_available_artifacts_without_rendering_a_pdf`. Any failure outside `test_report_pdf.py` is a real regression. `git status --short` was clean afterwards.

- [ ] **Step 2: Push and open the PR.**

```bash
git push -u origin feat/live-roster-qwen-haiku
gh pr create --base main --head feat/live-roster-qwen-haiku \
  --title "feat(live): drop Nemotron, add Qwen and Haiku" \
  --body-file - <<'EOF'
Merge before 2026-11-01, or first move both `live_from` values in leaderboard.json to the 1st of the month after the merge.

- Drops Nemotron 3 Nano from `LIVE_MODEL_IDS`: CommonStack serves no NVIDIA model (GET /v1/models, 2026-10-14), so every nightly call 404'd and it never printed. Its leaderboard.json entry is untouched. **Nemotron for Season 1 (and on which gateway) is an open owner decision.**
- Adds Qwen3.7 Plus and Claude Haiku 4.5 via a new optional per-entry `live_from` (`YYYY-MM-01`). Neither is deployed or shown before 2026-11-01 (edit this date if the join month moved), so nothing is backfilled. Deploy order is now `LIVE_MODEL_IDS` order (DeepSeek first).
- Haiku's Live calls send `anthropic/claude-haiku-4-5` at temperature 0 through a new `live_overrides` block. The shared keys stay as its Competition runs recorded them. Changing them would make the Competition board publish Haiku's July single run instead of its median of 3; a test pins that this does not happen.
- **New recurring cost, from November: ≈ $2.1/month** (Qwen ≈ $0.63, Haiku ≈ $1.45 at 126 calls/month, spec §7 prices; `COMMONSTACK_API_KEY`, outside Credits). Nemotron removed, was $0 (every call 404'd). The Live board totals ≈ $2.4/month with DeepSeek.
- Tests: `pytest dashboard/backend/tests -q -p no:cacheprovider` (`test_report_pdf` failures are local-environmental: no reportlab).
- Merging to main auto-deploys prod.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
```

- [ ] **Step 3: Merge by 2026-10-31, outside the nightly window.**

This PR must merge by 2026-10-31. If it has not merged by then, bump both `live_from` values in `dashboard/config/leaderboard.json` (`qwen3_7_plus` and `claude_haiku_4_5`) by hand to the 1st of the month after the merge, in a commit on this branch before it merges (merging on 2026-11-05 means `"2026-12-01"`), and update the month in the PR body's join-date and cost bullets to match (`gh pr edit --body-file`). Otherwise Qwen and Haiku are live at once and replay the month to date. No test hard-codes the month, so only the two values change. The PR body's first line carries the same guard for whoever merges; keep it as the first line.

Do not merge on a weekday between 22:15 and ~02:30 UTC. The merge auto-deploys prod, and the Render restart kills the nightly live refresh thread mid-run (its cron says 22:30 UTC, but GitHub has started it as late as 01:52), so that night's session is billed again on the next night.

- [ ] **Step 4: After merge (no code).** On the first weekday evening of the join month, check the Render log for `live increment qwen3_7_plus` and `live increment claude_haiku_4_5`, and check that neither `ERROR: live.model_deploy_failed entry=claude_haiku_4_5` nor `ERROR: live.model_not_attempted entry=claude_haiku_4_5` appears. Since #668, two entries ahead of it failing with the same error skip Haiku without printing the first line. The first November nightly is the first real CommonStack `/v1/messages` call for Haiku at temperature 0. File anything found as an issue, not as a PR comment.

#### Phase 4 review-focus candidates

1. **Merge on or after the join month without bumping `live_from`.** The newcomers are then live at once and replay the month to date in one run: bounded, roughly $0.07 per elapsed session for Haiku. Covered: `test_newcomers_start_at_the_1st_and_deepseek_just_resumes` pins the replay-from-the-1st shape and that DeepSeek is not replayed. The operator guard is the PR body's first line and Task 4.6 Step 3.
2. **An operator hand-edits `live_from` or `live_overrides` on a deployed config with a typo.** Covered: the malformed-`live_from` cases (7 shapes) and malformed-override cases (3 shapes) assert one warning per process and the documented fallback. `test_the_committed_live_config_raises_no_warning` stops a typo from shipping in the repo.
3. **`deploy_leaderboard_model.py --period live --entry claude_haiku_4_5` run in October.** Covered: `test_the_deploy_cli_says_why_it_refuses_a_not_yet_live_entry` asserts it refuses without deploying and names `live_from`.
4. **The public Competition board's Haiku row after this change.** Covered: `test_the_live_haiku_pin_leaves_its_competition_median_alone`, mutation-proven in Task 4.2 Step 5. Measured on main@17c552c5: putting the Live values on the shared keys gave `1 failed`, with the July primary published in place of `…_s2`.
5. **CommonStack's `/v1/messages` actually honouring `temperature: 0` for `anthropic/claude-haiku-4-5`.** Not testable here: CI mocks the gateway, and a live call is billable. The served id is pinned by `COMMONSTACK_SERVED` (from the 2026-10-14 listing), and the temperature reaching the request by the existing `test_llm_agent_passes_temperature_to_portfolio_manager`. If the gateway rejects it, the failure is visible: H6 refuses the fallback curve and `ERROR: live.model_deploy_failed` is printed. Task 4.6 Step 4 is the post-merge check.
6. **Task 4.3 deletes `assert "claude_haiku_4_5" not in deployed` from `test_refresh_live_leaderboard_deploys_models`.** Reviewers should check that this is not a weakened assertion. The exact-list assertion that replaces it, `deployed == ["deepseek_v4_pro", "qwen3_7_plus", "claude_haiku_4_5"]`, now pins Haiku *in*, at roster position 3, and that is the intended new behaviour. The added `"nemotron_3_nano_30b" not in deployed` keeps the "dropped model is not deployed" check.
7. **The whole-file `sed` rename in Task 4.3 stays inside `test_live_leaderboard.py`.** Covered by the `git status --short` line in Step 1(a). Measured on main@17c552c5: `test_landing_live_board.py`, `domain/leaderboard/test_deploy_guard.py` and `domain/leaderboard/test_strategies_move.py` keep their Nemotron fixtures. The `test_strategies_move.py` diff is comment-only, so check it does not reformat the `_PINNED_SAMPLING_ENTRIES` line; an earlier execution of Task 4.2 did.
