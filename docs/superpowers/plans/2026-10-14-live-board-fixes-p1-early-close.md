# Live-Board Fixes P1: NYSE early closes on the live board — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End the live board's US session at 13:00 ET on NYSE early closes (first one: 2026-11-27), so that day gets four chart nodes, three model decisions and a 13:30 settle instead of seven nodes and three extra billed decisions on post-close bars, and pin the six-decisions-per-session cadence with tests.

**Architecture:** A rule-based early-close table (`nyse_early_closes`, `is_early_close`) joins the holiday rules in `us_market_calendar.py`, and `sessions.py`, still the only module allowed to spell a session bound, gains `session_close_time`. The open-stamped branch of the shared `is_in_session` filter (the board's model, its baselines, My Agents and protocol runs) reads that per-day close, and `live.py` replaces its fixed 16:00 close and seven-hour axis with per-day helpers (`_cash_close`, `_axis_hours`) used by the axis, clock, settle time, node mapping and reindex. The close-stamped path is untouched (D19 stays with the engine rewrite).

**Tech Stack:** Python 3.13 / FastAPI backend (pandas, pytz), pytest from the repo root, one static hint in `dashboard/frontend/app.html` (no JS change, so no cache-buster bump).

**Spec:** `docs/superpowers/specs/2026-10-14-atl-direction.md` §6 (Live Leaderboard: the Cadence row and "Fixes to the board while it runs on the legacy engine", the 2026-11-27 half day) and §7 (Budget and keys: 126 calls/month). Decisions, Global Constraints, merge order and Review Focus live in the index plan `docs/superpowers/plans/2026-10-14-live-board-fixes.md`; every task here inherits its Global Constraints.

**Merge order:** P1 → P2 → P3 → P4 → P5 → P6 → P7 PR A → P7 PR B (owner-gated). Deadlines: P3 before 2026-11-01; P4 by 2026-10-31 (else bump both live_from values); P1 merged and deployed by 2026-11-20. One executor per contiguous range; phases are not safe to run in parallel (several append to the same test files and bump the same cache-busters).

## Phase 1: NYSE early closes on the live board

**Why.** Spec §6 ("Fixes to the board while it runs on the legacy engine") makes the **2026-11-27** half-day session a hard deadline; Decisions 1 and 2 in the index plan scope it. On main the session close is a fixed 16:00 in four places, so on 2026-11-27 (13:00 ET close) the board would:
- draw seven axis nodes, three of them phantoms (`live.py` `_RTH_HOURS`);
- keep Alpaca's 13:00/14:00/15:00 late-session bars as decision bars, so the model makes three extra billed decisions filled off thin post-close prints (`sessions.is_in_session`, open-stamped branch);
- report "Cash session open" until 16:00 and settle at 16:30 (`live_clock`, `freeze_settle_time`).

No fee, slippage or prompt change, so the 10-08 "no retrofit" rule does not apply (spec §6). The cadence (spec §6, budget §7) is already right and only gets pinned (Decision 2 in the index plan).

**Depends on.** Nothing. This phase merges first: it must be **merged and deployed by 2026-11-20**. Later phases that touch `live.py`, `test_live_leaderboard.py`, `test_frontend_live_leaderboard.py` or `app.html` rebase onto it: P2 (`2026-10-14-live-board-fixes-p2-memory-robustness.md`), P3 (`2026-10-14-live-board-fixes-p3-attempt-ledger.md`), P4 (`2026-10-14-live-board-fixes-p4-roster.md`), P5 (`2026-10-14-live-board-fixes-p5-daily-sharpe.md`) and P6 (`2026-10-14-live-board-fixes-p6-copy.md`), all in `docs/superpowers/plans/`. Every anchor below is a verbatim block, not a line number.

**Branch / PR title.** `fix/live-early-close` / `fix(live): NYSE early closes on the live board`. Before Task 1.1:

```bash
git status --short                                   # must print nothing: switch -c carries uncommitted edits along
gh pr list --head fix/live-early-close --state all   # expect no rows; if one exists, stop and read it first
git fetch origin && git switch -c fix/live-early-close origin/main
```

Every `find` block below was checked to occur exactly once on main@17c552c5, and the whole phase was replayed on that commit task by task. If `origin/main` has moved past 17c552c5, the anchors still apply (they are verbatim blocks, not line numbers); the pass/fail counts quoted in "Expected" lines are main@17c552c5 figures, so treat a small drift in a neighbour-suite total as main having grown, not as a failure. The named failing tests and their messages are the real check.

> **Deviation from Decision 1 in the index plan — read before Task 1.1.** That decision puts `EARLY_CLOSE_TIME`, `REGULAR_CLOSE_TIME` and `session_close_time` in `us_market_calendar.py`. The code forbids that: `tests/test_market_sessions.py::test_no_module_restates_the_session_bounds` AST-scans every non-test backend module and fails on any `time(13, 0)` / `time(16, 0)` call outside `market_data/sessions.py` (verified: adding the two constants to the calendar fails it with `session bounds restated outside market_data/sessions.py: infrastructure/market_data/us_market_calendar.py:142, …:143`). So the **dates** live in the calendar (`nyse_early_closes`, `is_early_close`) and the **close times** live in `sessions.py` (`EARLY_CLOSE_TIME`, `REGULAR_CLOSE_TIME`, `session_close_time`), which is that module's stated job ("the one owner of … is it open"). Names and behaviour are as decided; only the module differs. Do not widen the guard.

**Interfaces this phase produces**
- `dashboard/backend/infrastructure/market_data/us_market_calendar.py`
  - `nyse_early_closes(year: int) -> FrozenSet[date]` (`@lru_cache(maxsize=16)`). Rule: the day after Thanksgiving; Dec 24 when Mon–Thu; Jul 3 when Jul 4 is Tue–Fri; any candidate that is not `is_trading_day` is dropped. Pinned to 2025: `{07-03, 11-28, 12-24}`, 2026: `{11-27, 12-24}`, 2027: `{11-26}`, 2028: `{07-03, 11-24}`.
  - `is_early_close(day: date) -> bool`.
  - `is_trading_day` is unchanged: an early close is still a trading day.
- `dashboard/backend/infrastructure/market_data/sessions.py`
  - `REGULAR_CLOSE_TIME: time` — `session_windows("US")[-1][1]`, i.e. `time(16, 0)`, derived rather than restated.
  - `EARLY_CLOSE_TIME: time` — `time(13, 0)`.
  - `session_close_time(day: date, market: object = "US") -> time` — `EARLY_CLOSE_TIME` when the market is on US sessions (US, or anything `session_windows` maps to US) and `is_early_close(day)`; otherwise the market's last window end (US 16:00, CN 15:00).
  - `is_in_session(..., open_stamped_minutes=N)` — the last window now ends at `session_close_time(local_date, market)`. The close-stamped branch, `time_in_session`, `session_windows` and `is_session_close` are byte-for-byte unchanged (D19 stays with the engine rewrite).
- `dashboard/backend/domain/leaderboard/live.py` (private helpers, used by tests)
  - `_cash_close(day: date) -> datetime` — the day's close in `America/New_York` (16:00 or 13:00).
  - `_axis_hours(day: date) -> Tuple[int, ...]` — `(10, …, 16)` or `(10, 11, 12, 13)`.
  - `_FIRST_NODE_HOUR = _US_CASH_OPEN.hour + 1` (10).
  - Removed: `_US_CASH_CLOSE`, `_RTH_HOURS` (a test asserts they stay gone).
  - `_coerce_as_of_eastern(date)` now returns that day's cash close (13:00 on an early close).
- Public payload `GET /api/v1/leaderboard?period=live`: `chart_axis` carries 4 nodes (`T10:00`…`T13:00`) on an early-close day; `live_status.axis_count`, `printed_through`, `now_index`, `next_tick`, `session_state`, `frozen_through` follow; `window.description` says `10:00–13:00 on NYSE early-close days`. No field added or removed.
- `dashboard/frontend/app.html` live chart hint: `… 10:00–16:00 ET (10:00–13:00 on early closes), NYSE trading days …`. No JS file changes, so no cache-buster bump.

Commit hygiene for every task: run `git branch --show-current` right before committing and confirm it prints `fix/live-early-close` (sibling sessions share this checkout). Append your harness's attribution trailer to commit messages if it requires one.

### Task 1.1: Rule-based NYSE early-close calendar and the per-day close time

**Files:**
- Modify: `dashboard/backend/infrastructure/market_data/us_market_calendar.py`
- Modify: `dashboard/backend/infrastructure/market_data/sessions.py`
- Create (test): `dashboard/backend/tests/infrastructure/market_data/test_us_market_calendar.py`

**Interfaces:**
- Consumes: `us_market_calendar._nth_weekday`, `nyse_holidays`, `is_trading_day` (existing); `sessions.session_windows`, `sessions.DEFAULT_MARKET` (existing).
- Produces: `nyse_early_closes(year) -> FrozenSet[date]`, `is_early_close(day) -> bool` (calendar); `REGULAR_CLOSE_TIME`, `EARLY_CLOSE_TIME`, `session_close_time(day, market="US") -> time` (sessions).

- [ ] **Step 1: Write the failing test.** Create `dashboard/backend/tests/infrastructure/market_data/test_us_market_calendar.py` (the package already has an `__init__.py`):

```python
"""NYSE early closes, by rule, pinned to the exchange's published calendars.

The Live board's 2026-11-27 session is why this exists: the calendar knew the
holidays but not the 13:00 early closes, so that day would have had seven
chart nodes and three model decisions on late-session bars. Sources: the NYSE
Group releases "2025, 2026 and 2027 Holiday and Early Closings Calendar"
(2024-11-08) and "2026, 2027 and 2028 ..." (2025-12-23), and nyse.com
"Holidays & Trading Hours".
"""

from datetime import date, time

import pytest

from dashboard.backend.infrastructure.market_data import sessions
from dashboard.backend.infrastructure.market_data.us_market_calendar import (
    is_early_close,
    is_trading_day,
    nyse_early_closes,
    nyse_holidays,
)

PUBLISHED_EARLY_CLOSES = {
    2025: ["2025-07-03", "2025-11-28", "2025-12-24"],
    2026: ["2026-11-27", "2026-12-24"],
    2027: ["2027-11-26"],
    2028: ["2028-07-03", "2028-11-24"],
}


@pytest.mark.parametrize("year", sorted(PUBLISHED_EARLY_CLOSES))
def test_early_closes_match_the_nyse_published_calendar(year):
    assert sorted(d.isoformat() for d in nyse_early_closes(year)) == (
        PUBLISHED_EARLY_CLOSES[year]
    )


def test_a_holiday_or_weekend_is_never_an_early_close():
    # Independence Day 2026 is observed on Friday July 3; Christmas 2027 on
    # Friday December 24; December 24, 2028 is a Sunday.
    assert date(2026, 7, 3) in nyse_holidays(2026)
    assert date(2026, 7, 3) not in nyse_early_closes(2026)
    assert date(2027, 12, 24) in nyse_holidays(2027)
    assert date(2027, 12, 24) not in nyse_early_closes(2027)
    assert date(2028, 12, 24) not in nyse_early_closes(2028)


@pytest.mark.parametrize("year", range(2022, 2031))
def test_an_early_close_is_always_a_trading_day(year):
    for day in nyse_early_closes(year):
        assert is_trading_day(day), day


def test_is_early_close():
    assert is_early_close(date(2026, 11, 27))
    assert is_early_close(date(2026, 12, 24))
    assert not is_early_close(date(2026, 11, 25))
    # Thanksgiving itself is a holiday, not an early close.
    assert not is_early_close(date(2026, 11, 26))


def test_the_close_times_have_one_owner():
    assert sessions.EARLY_CLOSE_TIME == time(13, 0)
    assert sessions.REGULAR_CLOSE_TIME == time(16, 0)
    assert sessions.REGULAR_CLOSE_TIME == sessions.session_windows("US")[-1][1]


@pytest.mark.parametrize(
    ("day", "market", "close"),
    [
        (date(2026, 11, 27), "US", time(13, 0)),
        (date(2026, 12, 24), "us", time(13, 0)),
        (date(2026, 11, 25), "US", time(16, 0)),
        (date(2026, 11, 30), None, time(16, 0)),
        # A-shares keep their own close; the NYSE calendar is not theirs.
        (date(2026, 11, 27), "CN", time(15, 0)),
    ],
)
def test_session_close_time(day, market, close):
    assert sessions.session_close_time(day, market) == close


def test_session_close_time_defaults_to_the_us_market():
    assert sessions.session_close_time(date(2026, 11, 27)) == time(13, 0)
```

- [ ] **Step 2: Run it and confirm it fails.**

```bash
pytest dashboard/backend/tests/infrastructure/market_data/test_us_market_calendar.py -q -p no:cacheprovider
```

Expected: a collection error, `ImportError: cannot import name 'is_early_close' from 'dashboard.backend.infrastructure.market_data.us_market_calendar'`, `1 error`.

- [ ] **Step 3: Implement.** Five edits; each `find` block occurs exactly once.

**Edit 1 — calendar docstring, first line.** In `dashboard/backend/infrastructure/market_data/us_market_calendar.py` (≈:1 on main@17c552c5), find:

```python
"""NYSE full-day holidays, computed by rule (no data file, no dependency).
```

Replace with:

```python
"""NYSE full-day holidays and early closes, computed by rule (no data file,
no dependency).
```

**Edit 2 — calendar docstring: the "not modelled" line.** In `dashboard/backend/infrastructure/market_data/us_market_calendar.py` (≈:14 on main@17c552c5), find:

```python
Covers the ten NYSE holidays in force since 2022 (Juneteenth added that year).
Early closes (13:00 ET) are not modelled; they are still trading days.
"""
```

Replace with:

```python
Covers the ten NYSE holidays in force since 2022 (Juneteenth added that year)
and the recurring 13:00 ET early closes (``nyse_early_closes``). An early close
is still a trading day. The hour it ends at is a session bound, so it lives
with the other bounds in ``sessions.py`` (``session_close_time``).
"""
```

**Edit 3 — the early-close rule, after `is_trading_day`.** In `dashboard/backend/infrastructure/market_data/us_market_calendar.py` (≈:83 on main@17c552c5), find:

```python
def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in nyse_holidays(day.year)
```

Replace with:

```python
def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in nyse_holidays(day.year)


@lru_cache(maxsize=16)
def nyse_early_closes(year: int) -> FrozenSet[date]:
    """NYSE's 13:00 ET early closes in ``year``, by rule.

    - the day after Thanksgiving;
    - December 24 when it falls Monday to Thursday (a Friday one is the
      observed Christmas holiday, a weekend one is no session);
    - July 3 when Independence Day falls Tuesday to Friday (a Saturday July 4
      makes July 3 the observed holiday; a Sunday or Monday one puts July 3 on
      a weekend).

    A candidate that is not a trading day is dropped, so this set can never
    contradict ``nyse_holidays``. The rules reproduce NYSE's published 2025-2028
    calendars; an unscheduled closure is not predictable by rule, and the Live
    board's session-gap check stays the backstop for one.
    """
    candidates = {_nth_weekday(year, 11, 3, 4) + timedelta(days=1)}
    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() < 4:
        candidates.add(christmas_eve)
    if 1 <= date(year, 7, 4).weekday() <= 4:
        candidates.add(date(year, 7, 3))
    return frozenset(day for day in candidates if is_trading_day(day))


def is_early_close(day: date) -> bool:
    """Whether ``day`` is an NYSE session that ends at 13:00 ET."""
    return day in nyse_early_closes(day.year)
```

**Edit 4 — sessions imports.** In `dashboard/backend/infrastructure/market_data/sessions.py` (≈:18 on main@17c552c5), find:

```python
from datetime import datetime, time, timedelta
from typing import Any, Mapping

import pytz

from dashboard.backend.infrastructure.market_data.profiles import (
    registered_market_timezones,
)
```

Replace with:

```python
from datetime import date, datetime, time, timedelta
from typing import Any, Mapping

import pytz

from dashboard.backend.infrastructure.market_data.profiles import (
    registered_market_timezones,
)
from dashboard.backend.infrastructure.market_data.us_market_calendar import (
    is_early_close,
)
```

**Edit 5 — the close times, after `session_windows`.** In `dashboard/backend/infrastructure/market_data/sessions.py` (≈:42 on main@17c552c5), find:

```python
    if canonical_market(market) == "CN":
        return ((time(9, 30), time(11, 30)), (time(13, 0), time(15, 0)))
    return ((time(9, 30), time(16, 0)),)
```

Replace with:

```python
    if canonical_market(market) == "CN":
        return ((time(9, 30), time(11, 30)), (time(13, 0), time(15, 0)))
    return ((time(9, 30), time(16, 0)),)


#: The US cash close on a full session. Read off ``session_windows`` rather
#: than restated, so the bound keeps one owner.
REGULAR_CLOSE_TIME: time = session_windows(DEFAULT_MARKET)[-1][1]
#: The US cash close on an NYSE early-close day
#: (``us_market_calendar.nyse_early_closes``).
EARLY_CLOSE_TIME: time = time(13, 0)


def session_close_time(day: date, market: object = DEFAULT_MARKET) -> time:
    """When ``market``'s last session ends on ``day``, in its own local time.

    A market on US sessions closes at :data:`EARLY_CLOSE_TIME` on the NYSE
    early-close days the calendar lists and at :data:`REGULAR_CLOSE_TIME`
    otherwise; any other market closes at its last window's end. Whether
    ``day`` trades at all is the calendar's question
    (``us_market_calendar.is_trading_day``), not this one's.

    The close-stamped helpers (:func:`time_in_session`, :func:`session_windows`,
    :func:`is_session_close`) do not consult it: bar aggregation and the
    engine still end every US day at 16:00. That gap is D19 in the US
    conformance design, and the engine rewrite owns it.
    """
    windows = session_windows(market)
    if windows == session_windows(DEFAULT_MARKET) and is_early_close(day):
        return EARLY_CLOSE_TIME
    return windows[-1][1]
```

`us_market_calendar.py` stays stdlib-only, so `sessions.py` importing it creates no cycle (`profiles.py` imports neither).

- [ ] **Step 4: Run it and confirm it passes.**

```bash
pytest dashboard/backend/tests/infrastructure/market_data/test_us_market_calendar.py -q -p no:cacheprovider
```

Expected: `22 passed`. Then the neighbours, including the AST guard that decided where the times live:

```bash
pytest dashboard/backend/tests/test_market_sessions.py dashboard/backend/tests/test_default_backtest_window.py dashboard/backend/tests/infrastructure/market_data "dashboard/backend/tests/test_live_leaderboard.py::test_nyse_holiday_rules" -q -p no:cacheprovider
```

Expected: all pass, `546 passed, 2 skipped` on main@17c552c5 + this task (`test_market_sessions.py` is 41 tests on main and stays 41 here).

- [ ] **Step 5: Commit.**

```bash
git branch --show-current   # must print fix/live-early-close
git add dashboard/backend/infrastructure/market_data/us_market_calendar.py \
        dashboard/backend/infrastructure/market_data/sessions.py \
        dashboard/backend/tests/infrastructure/market_data/test_us_market_calendar.py
git commit -m "feat(calendar): model NYSE early closes by rule"
```

### Task 1.2: End the open-stamped US session at 13:00 on an early close

Global on purpose (Decision 1 in the index plan): every open-stamped filter — the board's model, its Alpaca baselines (`baseline_generator._market_hours_only`), the dashboard engine (`engine._market_hours_only`, My Agents) and the protocol dataset store (`market_data_store._build_trading_timestamps`) — goes through `is_in_session`, so an agent and its baseline stay on one rule. My Agents runs that cover an early close (e.g. 2025-11-28) change on that day only: late-session bars stop being decisions or valuation marks, and in minute mode the post-13:00 decision buckets no longer fill. The close-stamped path stays wrong in the way conformance T05 measures (D19); the engine rewrite owns it.

**Files:**
- Modify: `dashboard/backend/infrastructure/market_data/sessions.py`
- Test: `dashboard/backend/tests/test_market_sessions.py`

**Interfaces:**
- Consumes: `sessions.session_close_time` (Task 1.1).
- Produces: `sessions.is_in_session(timestamp, *, market, timezone, open_stamped_minutes)` — same signature; with `open_stamped_minutes` set, the last window ends at `session_close_time(local.date(), market)`.

- [ ] **Step 1: Write the failing tests.** They reuse this file's existing imports (`pd`, `sessions`, `profiles`, `clock_time`).

The anchor is the last line of `test_open_stamped_hourly_day_is_six_bars`. In `dashboard/backend/tests/test_market_sessions.py` (≈:177 on main@17c552c5), find:

```python
    assert kept == ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00"]
```

Insert directly after it (keep the anchor as is):

```python


def _open_stamped_kept(stamps, minutes=60):
    return [
        ts.strftime("%H:%M") for ts in stamps
        if sessions.is_in_session(
            ts, market="US", timezone="US/Eastern", open_stamped_minutes=minutes
        )
    ]


def test_open_stamped_hourly_early_close_is_three_bars():
    """2026-11-27 closes at 13:00 ET. Alpaca serves the late session as bars
    stamped 13:00, 14:00 and 15:00, and a fixed 16:00 close kept them: three
    decisions priced off thin after-hours prints."""
    day = pd.date_range("2026-11-27 04:00", "2026-11-27 19:00", freq="h",
                        tz="US/Eastern")
    assert _open_stamped_kept(day) == ["10:00", "11:00", "12:00"]


def test_the_day_before_thanksgiving_is_a_full_session():
    day = pd.date_range("2026-11-25 04:00", "2026-11-25 19:00", freq="h",
                        tz="US/Eastern")
    assert _open_stamped_kept(day) == [
        "10:00", "11:00", "12:00", "13:00", "14:00", "15:00",
    ]


def test_open_stamped_five_minute_bars_stop_at_the_early_close():
    bars = pd.date_range("2026-12-24 12:45", "2026-12-24 13:15", freq="5min",
                         tz="US/Eastern")
    assert _open_stamped_kept(bars, minutes=5) == ["12:45", "12:50", "12:55"]


def test_the_early_close_is_read_on_the_market_local_date():
    """18:00Z on 2026-11-27 is 13:00 ET, after that day's close; a naive stamp
    is market-local, as everywhere else in this module."""
    def kept(ts):
        return sessions.is_in_session(
            ts, market="US", timezone="US/Eastern", open_stamped_minutes=60
        )

    assert kept(pd.Timestamp("2026-11-27 17:00", tz="UTC"))
    assert not kept(pd.Timestamp("2026-11-27 18:00", tz="UTC"))
    assert not kept(pd.Timestamp("2026-11-27 13:00"))


def test_an_early_close_leaves_the_close_stamped_path_alone():
    """Bar aggregation and the engine's close-stamped filter still end every US
    day at 16:00. That is D19, measured by conformance T05 and owned by the
    engine rewrite; only the open-stamped rule reads the day's close."""
    assert sessions.session_windows("US") == ((clock_time(9, 30), clock_time(16, 0)),)
    assert sessions.is_in_session(
        pd.Timestamp("2026-11-27 15:00", tz="US/Eastern"),
        market="US",
        timezone="US/Eastern",
    )
    assert sessions.is_session_close(
        pd.Timestamp("2026-11-27 16:00", tz="US/Eastern"),
        market="US",
        timezone="US/Eastern",
    )


def test_cn_open_stamped_sessions_ignore_the_nyse_calendar():
    day = pd.date_range("2026-11-27 09:00", "2026-11-27 15:00", freq="h",
                        tz="Asia/Shanghai")
    kept = [
        ts.strftime("%H:%M") for ts in day
        if sessions.is_in_session(
            ts, market="CN", timezone="Asia/Shanghai", open_stamped_minutes=60
        )
    ]
    assert kept == ["10:00", "13:00", "14:00"]


def test_every_us_filter_drops_the_late_session_on_an_early_close():
    """The board's model, its Alpaca baselines, a My Agents backtest and a
    protocol run all filter through ``is_in_session``. One rule for all of them
    is what keeps an agent and its baseline on the same bars on a half day."""
    from dashboard.backend import baseline_generator
    from dashboard.backend.domain.backtesting import market_data_store as mds
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester
    from dashboard.backend.domain.leaderboard.strategies import _common

    stamps = list(
        pd.date_range("2026-11-27 04:00", "2026-11-27 19:00", freq="h",
                      tz="US/Eastern").tz_convert("UTC")
    )
    frame = pd.DataFrame({"close": 1.0}, index=pd.DatetimeIndex(stamps))
    frame.attrs[sessions.FRAME_ATTR_OPEN_STAMPED_MINUTES] = 60
    expected = [
        ts for ts in stamps if ts.tz_convert("US/Eastern").hour in (10, 11, 12)
    ]
    engine = HourlyBacktester.__new__(HourlyBacktester)
    engine.profile = profiles.get_market_profile(profiles.ALPACA)

    assert baseline_generator._market_hours_only(
        stamps, "US/Eastern", {"X": frame}
    ) == expected
    assert mds._build_trading_timestamps(
        {"X": frame}, market="US", timezone="US/Eastern"
    ) == expected
    assert _common.market_timestamps({"X": frame}) == expected
    assert engine._market_hours_only(stamps, {"X": frame}) == expected


def test_minute_mode_stops_filling_at_an_early_close():
    """My Agents' minute mode values and fills on open-stamped 5m source bars,
    which now stop at 12:55 on 2026-11-27. The close-stamped aggregation still
    forms 13:30-16:00 buckets from late-session prints (D19), but none of them
    fills, so none is a step: nothing trades off a post-close print."""
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester

    source = pd.date_range("2026-11-27 09:30", "2026-11-27 15:55", freq="5min",
                           tz="US/Eastern").tz_convert("UTC")
    frame = pd.DataFrame({"close": 1.0}, index=source)
    frame.attrs[sessions.FRAME_ATTR_OPEN_STAMPED_MINUTES] = 5
    engine = HourlyBacktester.__new__(HourlyBacktester)
    engine.profile = profiles.get_market_profile(profiles.ALPACA)
    engine.intraday_mode = True
    engine.source_timeframe = "5m"
    engine.source_data = {"X": frame}
    decisions = [
        pd.Timestamp(f"2026-11-27 {clock}", tz="US/Eastern").tz_convert("UTC")
        for clock in ("10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00")
    ]
    steps, valuation, _fills = engine._plan_executions(decisions)
    assert [ts.tz_convert("US/Eastern").strftime("%H:%M") for ts in steps] == [
        "10:30", "11:30", "12:30",
    ]
    assert max(valuation).tz_convert("US/Eastern").strftime("%H:%M") == "12:55"
```

- [ ] **Step 2: Run them and confirm they fail.**

```bash
pytest dashboard/backend/tests/test_market_sessions.py -q -p no:cacheprovider
```

Expected: `5 failed, 44 passed`. The five:
- `test_open_stamped_hourly_early_close_is_three_bars` — `assert ['10:00', '11...:00', '15:00'] == ['10:00', '11:00', '12:00']`
- `test_open_stamped_five_minute_bars_stop_at_the_early_close` — the left list runs on to `'13:10'`
- `test_the_early_close_is_read_on_the_market_local_date` — `assert not True` on the 18:00Z stamp
- `test_every_us_filter_drops_the_late_session_on_an_early_close` — fails at the `baseline_generator._market_hours_only` assertion
- `test_minute_mode_stops_filling_at_an_early_close` — `Left contains 4 more items, first extra item: '13:30'`

`test_the_day_before_thanksgiving_is_a_full_session`, `test_an_early_close_leaves_the_close_stamped_path_alone` and `test_cn_open_stamped_sessions_ignore_the_nyse_calendar` already pass: they pin what must not move.

- [ ] **Step 3: Implement.**

**`is_in_session`, open-stamped branch only.** In `dashboard/backend/infrastructure/market_data/sessions.py` (≈:139 on main@17c552c5), find:

```python
    open, and that 09:00 bar is out too: its open, high, low and volume include
    half an hour of pre-market trading, so admitting it would price the day's
    first fill off a pre-market print.
    """
    local = market_local(timestamp, timezone)
    if open_stamped_minutes is None:
        return time_in_session(local.time(), market)
    close = local + timedelta(minutes=open_stamped_minutes)
    if close.date() != local.date():
        return False
    open_minute = local.time().replace(second=0, microsecond=0)
    close_minute = close.time().replace(second=0, microsecond=0)
    return any(
        start <= open_minute and close_minute <= end
        for start, end in session_windows(market)
    )
```

Replace with:

```python
    open, and that 09:00 bar is out too: its open, high, low and volume include
    half an hour of pre-market trading, so admitting it would price the day's
    first fill off a pre-market print.

    The day's last session ends at :func:`session_close_time`: 13:00 ET on an
    NYSE early close, where Alpaca still serves the late session as bars
    stamped 13:00, 14:00 and 15:00. Only this open-stamped rule reads the
    day's close; the close-stamped rule (:func:`time_in_session`) still ends
    every US day at 16:00 (D19, owned by the engine rewrite).
    """
    local = market_local(timestamp, timezone)
    if open_stamped_minutes is None:
        return time_in_session(local.time(), market)
    close = local + timedelta(minutes=open_stamped_minutes)
    if close.date() != local.date():
        return False
    open_minute = local.time().replace(second=0, microsecond=0)
    close_minute = close.time().replace(second=0, microsecond=0)
    windows = session_windows(market)
    windows = windows[:-1] + (
        (windows[-1][0], session_close_time(local.date(), market)),
    )
    return any(
        start <= open_minute and close_minute <= end
        for start, end in windows
    )
```

- [ ] **Step 4: Run it and confirm it passes, then run the conformance gate.**

```bash
pytest dashboard/backend/tests/test_market_sessions.py dashboard/backend/tests/infrastructure/market_data/test_us_market_calendar.py -q -p no:cacheprovider
```

Expected: `71 passed` (49 + 22).

Gate — the conformance suite's strict xfails must not move:

```bash
pytest dashboard/backend/tests/conformance dashboard/backend/tests/backtesting -q -p no:cacheprovider -rX
```

Expected: `449 passed, 9 skipped, 25 xfailed`, identical to main, and **no `XPASS` line** in the `-rX` summary. (Measured on main@17c552c5 with this change: exactly that. T05's synthesized bars end at 13:00, so the open-stamped change cannot reach it.) If a case does XPASS or a `known_failures.json` line moves, the suite's own convention applies (header of `tests/conformance/test_conformance_current_engine.py`): update `PREDICTIONS_D` / `PREDICTIONS_DL` in `tests/conformance/cases.py`, re-record with `CONFORMANCE_UPDATE_KNOWN_FAILURES=1 pytest dashboard/backend/tests/conformance/test_conformance_current_engine.py -q -p no:cacheprovider`, review the JSON diff, commit both in this task, and say so in the PR body.

Then the other callers of the shared filter:

```bash
pytest dashboard/backend/tests/test_baseline_generator_offline.py dashboard/backend/tests/test_market_data_store.py dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/infrastructure/market_data -q -p no:cacheprovider
```

Expected: all pass, `716 passed, 2 skipped` on main@17c552c5 + Tasks 1.1–1.2.

- [ ] **Step 5: Commit.**

```bash
git branch --show-current   # must print fix/live-early-close
git add dashboard/backend/infrastructure/market_data/sessions.py dashboard/backend/tests/test_market_sessions.py
git commit -m "fix(sessions): end the open-stamped US session at 13:00 on NYSE early closes"
```

Task 1.4 finds this commit by that exact subject, so keep it.

### Task 1.3: Per-day cash close on the live axis, clock, settle time and node mapping

Leaves `_session_gap`, `_trading_days_inclusive`, `next_session_day`, `_previous_trading_day` and `live_increment_bounds` unchanged: they work in whole trading days, and an early close is a trading day. Leaves `service.py`'s own `_US_CASH_CLOSE` (the retired Daily board) alone.

**Files:**
- Modify: `dashboard/backend/domain/leaderboard/live.py`
- Modify: `dashboard/frontend/app.html` (live chart hint only)
- Test: `dashboard/backend/tests/test_live_leaderboard.py`
- Test: `dashboard/backend/tests/test_frontend_live_leaderboard.py`

**Interfaces:**
- Consumes: `sessions.session_close_time` (Task 1.1), `sessions.is_in_session` early-close behaviour (Task 1.2, exercised end to end by the replay test), `us_market_calendar.is_early_close` (Task 1.1, in a test).
- Produces: `live._cash_close(day) -> datetime`, `live._axis_hours(day) -> Tuple[int, ...]`, `live._FIRST_NODE_HOUR`; `live._US_CASH_CLOSE` and `live._RTH_HOURS` removed; payload `chart_axis` with 4 nodes on an early close; `window.description` naming the early close.

- [ ] **Step 1: Write the failing tests.** Backend tests reuse this file's existing names: `live`, `lb_service`, `db`, `SEED`, `_ET`, `_pt`, `_insert_live_run`, `_deepseek`, `_month_rows`, the `no_alpaca` fixture, and `date`/`datetime`/`pytest` from its imports. They use only `deepseek_v4_pro`, which stays on the roster after P4 (`2026-10-14-live-board-fixes-p4-roster.md`) removes Nemotron.

The anchor is the end of `test_nyse_holiday_rules`. In `dashboard/backend/tests/test_live_leaderboard.py` (≈:244 on main@17c552c5), find:

```python
    # A Saturday New Year's Day is not observed on the Friday before.
    assert date(2021, 12, 31) not in nyse_holidays(2021)
    assert date(2022, 1, 1) not in nyse_holidays(2022)
```

Insert directly after it (keep the anchor as is):

```python


# ── NYSE early closes (13:00 ET) ────────────────────────────────────────────


@pytest.fixture
def sip_delay_15(monkeypatch):
    monkeypatch.delenv("ALPACA_ALLOW_RECENT_SIP", raising=False)
    monkeypatch.setenv("ALPACA_SIP_DELAY_MINUTES", "15")


def _node_clocks(axis, day):
    return [ts[11:16] for ts in axis if ts.startswith(day)]


def test_hourly_axis_ends_at_13_00_on_an_early_close():
    """2026-11-27 closes at 13:00: four nodes (10:00 carries the prior close,
    11:00-13:00 hold the 10:00-12:00 bars), not seven with three phantoms."""
    from dashboard.backend.infrastructure.market_data.us_market_calendar import (
        is_early_close,
    )

    axis = live.live_month_hourly_axis("2026-11-01", "2026-11-30")
    days = live._trading_days_inclusive(date(2026, 11, 1), date(2026, 11, 30))
    assert len(days) == 20
    assert [day for day in days if is_early_close(day)] == [date(2026, 11, 27)]
    assert len(axis) == 19 * 7 + 4
    assert _node_clocks(axis, "2026-11-27") == ["10:00", "11:00", "12:00", "13:00"]
    assert _node_clocks(axis, "2026-11-25") == [
        "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00",
    ]
    assert _node_clocks(axis, "2026-11-26") == []


def test_december_axis_has_one_early_close():
    axis = live.live_month_hourly_axis("2026-12-01", "2026-12-31")
    days = live._trading_days_inclusive(date(2026, 12, 1), date(2026, 12, 31))
    assert len(days) == 22
    assert len(axis) == 21 * 7 + 4
    assert _node_clocks(axis, "2026-12-24") == ["10:00", "11:00", "12:00", "13:00"]
    assert axis[-1].startswith("2026-12-31T16:00")


def test_an_early_close_settles_30_minutes_after_13_00(sip_delay_15):
    assert live.freeze_settle_time(date(2026, 11, 27)) == datetime(
        2026, 11, 27, 13, 30, tzinfo=_ET
    )
    assert live.freeze_settle_time(date(2026, 11, 25)) == datetime(
        2026, 11, 25, 16, 30, tzinfo=_ET
    )


@pytest.mark.parametrize(
    ("clock", "state", "frozen_through", "live_day"),
    [
        ((12, 59), "rth", "2026-11-25", "2026-11-27"),
        ((13, 5), "settling", "2026-11-25", "2026-11-27"),
        ((13, 30), "closed", "2026-11-27", None),
        ((15, 0), "closed", "2026-11-27", None),
    ],
)
def test_the_clock_closes_at_13_00_on_an_early_close(
    sip_delay_15, clock, state, frozen_through, live_day
):
    """Thanksgiving (11-26) is skipped: before the half day settles, the
    freeze is still Wednesday's."""
    status = live.live_clock(datetime(2026, 11, 27, *clock, tzinfo=_ET))
    assert (
        status["session_state"], status["frozen_through"], status["live_day"]
    ) == (state, frozen_through, live_day)


def test_a_bare_date_means_that_days_cash_close():
    assert live._coerce_as_of_eastern(date(2026, 11, 27)) == datetime(
        2026, 11, 27, 13, 0, tzinfo=_ET
    )
    assert live._coerce_as_of_eastern(date(2026, 11, 25)) == datetime(
        2026, 11, 25, 16, 0, tzinfo=_ET
    )


def test_axis_node_key_caps_at_the_early_close():
    def key(hour, minute=0, day=27):
        return live._axis_node_key(datetime(2026, 11, day, hour, minute, tzinfo=_ET))

    assert key(12) == "2026-11-27T13:00"  # Alpaca's last cash-session bar
    assert key(12, 30) == "2026-11-27T13:00"  # Yahoo's 12:30-13:00 half bar
    assert key(13) is None  # the late session has no node
    assert key(15) is None
    assert key(15, day=25) == "2026-11-25T16:00"


def test_frozen_curve_through_an_early_close_stops_at_13_00():
    """Open-stamped bars 10:00-15:00 ET (15:00Z-20:00Z in EST): the three
    late-session prints must not paint 14:00-16:00 nodes."""
    axis = live.live_month_hourly_axis("2026-11-01", "2026-11-30")
    points = [
        _pt(f"2026-11-27T{15 + i}:00:00+00:00", SEED + 100 * i) for i in range(6)
    ]
    curve = live.reindex_frozen_curve(points, axis, "2026-11-27", SEED)
    assert len(curve) == 18 * 7 + 4
    assert [
        (p["timestamp"][11:16], p["equity"])
        for p in curve if p["timestamp"].startswith("2026-11-27")
    ] == [
        ("10:00", SEED), ("11:00", SEED), ("12:00", SEED + 100), ("13:00", SEED + 200),
    ]


def test_the_window_description_names_the_early_close(no_alpaca):
    payload = live.get_live_leaderboard(as_of=datetime(2026, 11, 27, 17, 39, tzinfo=_ET))
    assert "10:00–13:00 on NYSE early-close days" in payload["window"]["description"]


def test_the_live_board_keeps_no_fixed_session_close():
    """A module-level 16:00 close and a fixed seven-hour axis are how the early
    close slipped through: every reader asks for the day's close instead."""
    assert not hasattr(live, "_US_CASH_CLOSE")
    assert not hasattr(live, "_RTH_HOURS")


def _alpaca_hourly_bars(*days):
    """AAPL bars shaped like Alpaca's: open-stamped hourly, 04:00-19:00 ET, so
    the late session after an early close is in the response."""
    import pandas as pd

    from dashboard.backend.infrastructure.market_data.sessions import (
        FRAME_ATTR_OPEN_STAMPED_MINUTES,
    )

    index = pd.DatetimeIndex([
        ts
        for day in days
        for ts in pd.date_range(f"{day} 04:00", f"{day} 19:00", freq="h",
                                tz="US/Eastern")
    ]).tz_convert("UTC")
    frame = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000},
        index=index,
    )
    frame.attrs[FRAME_ATTR_OPEN_STAMPED_MINUTES] = 60
    return {"AAPL": frame}


def test_a_replay_across_thanksgiving_and_the_early_close_is_not_a_gap(monkeypatch):
    """The real decision loop over 11-25 (full), 11-26 (holiday) and 11-27
    (13:00 close): six decisions then three, a checkpoint per session, no
    LiveSessionGapError, and four nodes on the half day."""
    from dashboard.backend.domain.leaderboard.strategies import llm_agent

    decided = []

    class Manager:
        def __init__(self, **kwargs):
            self.equity_history = []
            self.trades = []
            self.llm_calls = self.llm_decisions = 0
            self.input_tokens = self.output_tokens = 0

        def restore_state(self, snapshot):
            pass

        def snapshot_state(self):
            return {"cash": SEED, "positions": {}}

        def get_portfolio_state(self, *args):
            return {}

        def make_trading_decision_with_llm(self, state, client, **kwargs):
            decided.append(state["timestamp"])
            self.llm_calls += 1
            self.llm_decisions += 1
            return {"actions": []}

        def execute_actions(self, *args):
            pass

        def update_equity(self, market_data, price_cache, ts):
            equity = SEED + 1000 * (len(self.equity_history) + 1)
            self.equity_history.append(
                {"timestamp": ts, "equity": equity, "cash": equity, "positions_value": 0}
            )

        def get_equity_curve(self):
            return self.equity_history

    monkeypatch.setattr(llm_agent, "PortfolioManager", Manager)
    monkeypatch.setattr(llm_agent.LLMAgentStrategy, "_make_client", lambda self: object())
    monkeypatch.setattr(
        lb_service,
        "fetch_hourly_bars",
        lambda symbols, start, end: _alpaca_hourly_bars("2026-11-25", "2026-11-27"),
    )
    _insert_live_run(
        "lb_deepseek_v4_pro_20261101_20261124", "deepseek_v4_pro", "2026-11-24",
        [_pt("2026-11-24T20:00:00+00:00", SEED)],
        start_date="2026-11-01",
        metadata={live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}}},
    )
    as_of = datetime(2026, 11, 27, 17, 39, tzinfo=_ET)
    freeze_cfg = live.live_freeze_config(as_of)
    assert freeze_cfg["end_date"] == "2026-11-27"

    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)

    assert row["segment"] == {"start_date": "2026-11-25", "end_date": "2026-11-27"}
    by_day = {}
    for ts in decided:
        et = ts.tz_convert("US/Eastern")
        by_day.setdefault(et.date().isoformat(), []).append(et.strftime("%H:%M"))
    assert by_day == {
        "2026-11-25": ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00"],
        "2026-11-27": ["10:00", "11:00", "12:00"],
    }
    stored = db.get_run(row["run_id"])
    assert stored["end_date"] == "2026-11-27"
    assert stored["llm_calls"] == 9
    assert [r["end_date"] for r in _month_rows("deepseek_v4_pro")] == ["2026-11-27"]

    payload = live.get_live_leaderboard(as_of=as_of)
    assert _node_clocks(payload["chart_axis"], "2026-11-27") == [
        "10:00", "11:00", "12:00", "13:00",
    ]
    deepseek = next(e for e in payload["entries"] if e["entry_id"] == "deepseek_v4_pro")
    assert [
        (p["timestamp"][11:16], p["equity"])
        for p in deepseek["equity_curve"] if p["timestamp"].startswith("2026-11-27")
    ] == [
        ("10:00", SEED + 6000),
        ("11:00", SEED + 7000),
        ("12:00", SEED + 8000),
        ("13:00", SEED + 9000),
    ]
    assert deepseek["portfolio_value"] == SEED + 9000
```

Frontend guard. In `dashboard/backend/tests/test_frontend_live_leaderboard.py` (≈:64 on main@17c552c5), find:

```python
def test_live_chart_does_not_span_empty_future_with_fake_values():
    source = _strip_js_comments(_LIVE_JS)
    assert "has_prints" in source
    # Future nodes stay null; we must not invent contest-window timestamps.
    assert "2026-04-15" not in source
```

Insert directly after it (keep the anchor as is):

```python


def test_live_chart_hint_names_the_early_close():
    """The axis ends at 13:00 on an NYSE early close; the caption under the
    chart must not promise a 16:00 node that day."""
    view = _APP_HTML.split('id="liveLeaderboardView"')[1].split('id="leaderboardView"')[0]
    assert "10:00–13:00 on early closes" in view
```

- [ ] **Step 2: Run them and confirm they fail.**

```bash
pytest dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/test_frontend_live_leaderboard.py -q -p no:cacheprovider
```

Expected: `13 failed, 69 passed`. The failures and their messages:
- `test_hourly_axis_ends_at_13_00_on_an_early_close` — `assert 140 == ((19 * 7) + 4)` (`len(days) == 20` already passes: the count comes from the calendar).
- `test_december_axis_has_one_early_close` — `154 == 151` on the axis length.
- `test_an_early_close_settles_30_minutes_after_13_00` — the settle time is 16:30, not 13:30.
- `test_the_clock_closes_at_13_00_on_an_early_close[clock1…]`, `[clock2…]`, `[clock3…]` — the state is `rth` (the 12:59 case already passes).
- `test_a_bare_date_means_that_days_cash_close` — 16:00 returned for 2026-11-27.
- `test_axis_node_key_caps_at_the_early_close` — `assert '2026-11-27T14:00' is None`.
- `test_frozen_curve_through_an_early_close_stops_at_13_00` — the length is `18 * 7 + 7`.
- `test_the_window_description_names_the_early_close` — the string is absent.
- `test_the_live_board_keeps_no_fixed_session_close` — `live` still has `_US_CASH_CLOSE`.
- `test_a_replay_across_thanksgiving_and_the_early_close_is_not_a_gap` — gets past the deploy (Task 1.2 already makes the loop decide 6 then 3 times, with no gap) and fails on the axis: `Left contains 3 more items, first extra item: '14:00'`.
- `test_live_chart_hint_names_the_early_close` — the hint text is absent.

- [ ] **Step 3: Implement.** Ten edits in `live.py`, then one in `app.html`. Each `find` block occurs exactly once.

**Edit A — module docstring.** In `dashboard/backend/domain/leaderboard/live.py` (≈:5 on main@17c552c5), find:

```python
America/New_York calendar month, hourly nodes are the closes of the US cash
session's hourly bars (10:00–16:00 ET, NYSE trading days), and points after
the freeze close stay empty so the line cannot interpolate into the future.
```

Replace with:

```python
America/New_York calendar month, hourly nodes are the closes of the US cash
session's hourly bars (10:00–16:00 ET on NYSE trading days, 10:00–13:00 on
NYSE early closes), and points after the freeze close stay empty so the line
cannot interpolate into the future.
```

**Edit B — import.** In `dashboard/backend/domain/leaderboard/live.py` (≈:42 on main@17c552c5), find:

```python
from dashboard.backend.infrastructure.market_data.sessions import session_windows
```

Replace with:

```python
from dashboard.backend.infrastructure.market_data.sessions import (
    session_close_time,
    session_windows,
)
```

**Edit C — constants: drop the fixed close and the fixed seven-hour axis.** In `dashboard/backend/domain/leaderboard/live.py` (≈:48 on main@17c552c5), find:

```python
# The US cash session has one owner (market_data/sessions.py); restating it here
# is how the board's copies drifted apart before #529.
_US_CASH_OPEN, _US_CASH_CLOSE = session_windows("US")[0]
# Axis nodes are hourly bar *closes*. Since #529 the board's Alpaca bars are
# open-stamped (09:00 … 15:00, each closing an hour later) and Yahoo's index
# bars are open-stamped on the half hour (09:30 … 15:30), so a stored point is
# placed at the hour its bar closed in (see ``_axis_node_key``): seven nodes a
# day, the last one holding the 16:00 close.
_RTH_HOURS = (10, 11, 12, 13, 14, 15, 16)
_BAR_MINUTES = 60
```

Replace with:

```python
# The US cash session has one owner (market_data/sessions.py); restating it here
# is how the board's copies drifted apart before #529. The open is fixed; the
# close is per day (``_cash_close``), because an NYSE early close ends at 13:00.
_US_CASH_OPEN = session_windows("US")[0][0]
# Axis nodes are hourly bar *closes*. Since #529 the board's Alpaca bars are
# open-stamped (09:00 … 15:00, each closing an hour later) and Yahoo's index
# bars are open-stamped on the half hour (09:30 … 15:30), so a stored point is
# placed at the hour its bar closed in (see ``_axis_node_key``). The first node
# is the 09:00 bar's 10:00 close and the last is the day's cash close: seven
# nodes on a full session, four (10:00 … 13:00) on an early close.
_FIRST_NODE_HOUR = _US_CASH_OPEN.hour + 1
_BAR_MINUTES = 60
```

**Edit D — per-day close helpers, and a bare `as_of` date means that day's close.** In `dashboard/backend/domain/leaderboard/live.py` (≈:89 on main@17c552c5), find:

```python
def _coerce_as_of_eastern(as_of: Optional[Union[date, datetime]] = None) -> datetime:
    if as_of is None:
        return datetime.now(_US_EASTERN)
    if isinstance(as_of, datetime):
        if as_of.tzinfo is None:
            return as_of.replace(tzinfo=_US_EASTERN)
        return as_of.astimezone(_US_EASTERN)
    return datetime(
        as_of.year, as_of.month, as_of.day,
        _US_CASH_CLOSE.hour, _US_CASH_CLOSE.minute,
        tzinfo=_US_EASTERN,
    )
```

Replace with:

```python
def _cash_close(day: date) -> datetime:
    """``day``'s US cash close in New York: 16:00, or 13:00 on an NYSE early close."""
    return datetime.combine(day, session_close_time(day, "US"), tzinfo=_US_EASTERN)


def _axis_hours(day: date) -> Tuple[int, ...]:
    """Hours of ``day``'s axis nodes: 10 … 16, or 10 … 13 on an early close.

    NYSE closes on the hour, so the last node is the close itself.
    """
    return tuple(range(_FIRST_NODE_HOUR, _cash_close(day).hour + 1))


def _coerce_as_of_eastern(as_of: Optional[Union[date, datetime]] = None) -> datetime:
    """``as_of`` in New York; a bare date means that day's cash close."""
    if as_of is None:
        return datetime.now(_US_EASTERN)
    if isinstance(as_of, datetime):
        if as_of.tzinfo is None:
            return as_of.replace(tzinfo=_US_EASTERN)
        return as_of.astimezone(_US_EASTERN)
    return _cash_close(as_of)
```

**Edit E — the axis iterates the day's own hours.** In `dashboard/backend/domain/leaderboard/live.py` (≈:123 on main@17c552c5), find:

```python
def live_month_hourly_axis(start_date: str, end_date: str) -> List[str]:
    """Hourly bar-close ISO timestamps covering the month's NYSE trading days."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    out: List[str] = []
    for day in _trading_days_inclusive(start, end):
        for hour in _RTH_HOURS:
```

Replace with:

```python
def live_month_hourly_axis(start_date: str, end_date: str) -> List[str]:
    """Hourly bar-close ISO timestamps covering the month's NYSE trading days.

    Seven per full session (10:00 … 16:00), four per early close (10:00 … 13:00).
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    out: List[str] = []
    for day in _trading_days_inclusive(start, end):
        for hour in _axis_hours(day):
```

**Edit F — `freeze_settle_time`.** In `dashboard/backend/domain/leaderboard/live.py` (≈:180 on main@17c552c5), find:

```python
    delay = 0 if allow_recent_sip() else sip_delay_minutes()
    close = datetime.combine(day, _US_CASH_CLOSE, tzinfo=_US_EASTERN)
    return close + timedelta(minutes=delay + _FREEZE_SETTLE_MARGIN_MINUTES)
```

Replace with:

```python
    delay = 0 if allow_recent_sip() else sip_delay_minutes()
    close = _cash_close(day)
    return close + timedelta(minutes=delay + _FREEZE_SETTLE_MARGIN_MINUTES)
```

**Edit G — `live_clock`.** In `dashboard/backend/domain/leaderboard/live.py` (≈:207 on main@17c552c5), find:

```python
    elif now.time() < _US_CASH_CLOSE:
        session_state = "rth"
```

Replace with:

```python
    elif now.time() < session_close_time(today, "US"):
        session_state = "rth"
```

**Edit H — `_axis_node_key`.** In `dashboard/backend/domain/leaderboard/live.py` (≈:1216 on main@17c552c5), find:

```python
    Alpaca's 09:00 bar (09:30 open → 10:00) lands on 10:00 and its 15:00 bar on
    the 16:00 close; Yahoo's 09:30 bar closes 10:30 and lands on 10:00, and its
    15:30 half-bar is capped at the 16:00 close. A bar opening at or after the
    close is after-hours and has no node.
    """
    if bar_open.time() >= _US_CASH_CLOSE:
        return None
    close = bar_open + timedelta(minutes=_BAR_MINUTES)
    session_close = datetime.combine(close.date(), _US_CASH_CLOSE, tzinfo=_US_EASTERN)
    if close > session_close:
        close = session_close
```

Replace with:

```python
    Alpaca's 09:00 bar (09:30 open → 10:00) lands on 10:00 and its 15:00 bar on
    the 16:00 close; Yahoo's 09:30 bar closes 10:30 and lands on 10:00, and its
    15:30 half-bar is capped at the 16:00 close. On an NYSE early close the cap
    is 13:00, where Alpaca's 12:00 bar and Yahoo's 12:30 half-bar both land. A
    bar opening at or after the day's close is after-hours and has no node.
    """
    session_close = _cash_close(bar_open.date())
    if bar_open >= session_close:
        return None
    close = bar_open + timedelta(minutes=_BAR_MINUTES)
    if close > session_close:
        close = session_close
```

**Edit I — `reindex_frozen_curve`.** In `dashboard/backend/domain/leaderboard/live.py` (≈:1254 on main@17c552c5), find:

```python
    freeze_dt = datetime.combine(
        date.fromisoformat(freeze_end), _US_CASH_CLOSE, tzinfo=_US_EASTERN
    )
```

Replace with:

```python
    freeze_dt = _cash_close(date.fromisoformat(freeze_end))
```

**Edit J — the payload's `window.description` (in `get_live_leaderboard`).** In `dashboard/backend/domain/leaderboard/live.py` (≈:1525 on main@17c552c5), find:

```python
                "trading days; each node is the close of an hourly US cash-session "
                "bar (10:00–16:00 America/New_York). "
```

Replace with:

```python
                "trading days; each node is the close of an hourly US cash-session "
                "bar (10:00–16:00 America/New_York; 10:00–13:00 on NYSE early-close "
                "days). "
```

**Edit K — the live chart hint.** In `dashboard/frontend/app.html` (≈:1708 on main@17c552c5), find:

```html
<div class="chart-hint">Hourly bar closes · US cash session 10:00–16:00 ET, NYSE trading days · future days stay empty</div>
```

Replace with:

```html
<div class="chart-hint">Hourly bar closes · US cash session 10:00–16:00 ET (10:00–13:00 on early closes), NYSE trading days · future days stay empty</div>
```

Then confirm no fixed close is left behind:

```bash
grep -n "_US_CASH_CLOSE\|_RTH_HOURS" dashboard/backend/domain/leaderboard/live.py
```

Expected: no output.

- [ ] **Step 4: Run them and confirm they pass.**

```bash
pytest dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/test_frontend_live_leaderboard.py -q -p no:cacheprovider
```

Expected: `82 passed` (70 + 12; main has 57 + 11). Then the neighbouring live-board suites:

```bash
pytest dashboard/backend/tests/test_leaderboard_season.py dashboard/backend/tests/test_frontend_live_trading_board.py dashboard/backend/tests/test_live_trade_stats.py dashboard/backend/tests/test_leaderboard_api.py dashboard/backend/tests/test_market_sessions.py -q -p no:cacheprovider
```

Expected: all pass, `127 passed` on main@17c552c5 + Tasks 1.1–1.3. `app.html` changed but no JS file did, so no `?v=` bump and no `node --check` is needed.

- [ ] **Step 5: Commit.**

```bash
git branch --show-current   # must print fix/live-early-close
git add dashboard/backend/domain/leaderboard/live.py dashboard/frontend/app.html \
        dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/test_frontend_live_leaderboard.py
git commit -m "fix(live): per-day cash close on the live axis, clock and node mapping"
```

### Task 1.4: Pin the cadence: six decisions per full session, three per early close

Decision 2 in the index plan: no production change. The live decision loop already makes one model call per kept open-stamped bar, 10:00–15:00 on a full session (spec §6), and after Task 1.2 10:00–12:00 on an early close. Spec §7's 126 calls/month is that times a 21-session month. These tests stop it drifting, and tie the decision bars to the chart axis.

**Files:**
- Create (test): `dashboard/backend/tests/domain/leaderboard/test_llm_agent_cadence.py`

**Interfaces:**
- Consumes: `strategies._common.market_timestamps`, `llm_agent.LLMAgentStrategy._run_decision_loop` and `llm_agent.PortfolioManager` (monkeypatched), `live._trading_days_inclusive`, `live.live_month_hourly_axis`, `live._axis_node_key`, `live._parse_to_et`, `us_market_calendar.is_early_close`, `sessions.FRAME_ATTR_OPEN_STAMPED_MINUTES`.
- Produces: nothing new (test-only).

- [ ] **Step 1: Write the test.** Create `dashboard/backend/tests/domain/leaderboard/test_llm_agent_cadence.py`:

```python
"""The Live board's decision cadence, pinned (atl-direction spec §6, §7).

Six model decisions per full session, on the open-stamped hourly bars 10:00 to
15:00 ET, and three on an NYSE early close (10:00 to 12:00). The budget in spec
§7 (126 calls for a 21-session month) is this number times the calendar, so a
filter change that adds or drops one bar a day moves the bill by a sixth.
Nothing here is new behaviour: these stop the cadence drifting.
"""

from collections import Counter
from datetime import date

import pandas as pd
import pytest

from dashboard.backend.domain.leaderboard import live
from dashboard.backend.domain.leaderboard.strategies import _common, llm_agent
from dashboard.backend.infrastructure.market_data.sessions import (
    FRAME_ATTR_OPEN_STAMPED_MINUTES,
)
from dashboard.backend.infrastructure.market_data.us_market_calendar import (
    is_early_close,
)

FULL_SESSION = ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00"]
EARLY_CLOSE = ["10:00", "11:00", "12:00"]


def _alpaca_hourly(days):
    """Open-stamped hourly bars, 04:00-19:00 ET on each day: Alpaca serves
    extended hours, so the late session after an early close is present."""
    index = pd.DatetimeIndex([
        ts
        for day in days
        for ts in pd.date_range(f"{day} 04:00", f"{day} 19:00", freq="h",
                                tz="US/Eastern")
    ]).tz_convert("UTC")
    frame = pd.DataFrame({"close": 100.0}, index=index)
    frame.attrs[FRAME_ATTR_OPEN_STAMPED_MINUTES] = 60
    return {"AAPL": frame}


def _clocks_by_day(stamps):
    days = {}
    for ts in stamps:
        et = ts.tz_convert("US/Eastern")
        days.setdefault(et.date(), []).append(et.strftime("%H:%M"))
    return days


def test_six_decision_bars_on_a_full_session_and_three_on_an_early_close():
    stamps = _common.market_timestamps(_alpaca_hourly(["2026-11-25", "2026-11-27"]))
    assert _clocks_by_day(stamps) == {
        date(2026, 11, 25): FULL_SESSION,
        date(2026, 11, 27): EARLY_CLOSE,
    }


class _CountingManager:
    """Stands in for PortfolioManager: records every model call."""

    calls = []

    def __init__(self, **kwargs):
        self.equity_history = []
        self.trades = []
        self.llm_calls = self.llm_decisions = 0
        self.input_tokens = self.output_tokens = 0

    def restore_state(self, snapshot):
        pass

    def snapshot_state(self):
        return {"cash": 1.0, "positions": {}}

    def get_portfolio_state(self, *args):
        return {}

    def make_trading_decision_with_llm(self, state, client, **kwargs):
        _CountingManager.calls.append(state["timestamp"])
        self.llm_calls += 1
        self.llm_decisions += 1
        return {"actions": []}

    def execute_actions(self, *args):
        pass

    def update_equity(self, market_data, price_cache, ts):
        self.equity_history.append({"timestamp": ts, "equity": 1.0})

    def get_equity_curve(self):
        return self.equity_history


def test_the_decision_loop_asks_the_model_once_per_kept_bar(monkeypatch):
    _CountingManager.calls = []
    monkeypatch.setattr(llm_agent, "PortfolioManager", _CountingManager)
    stamps = _common.market_timestamps(_alpaca_hourly(["2026-11-25", "2026-11-27"]))
    strategy = llm_agent.LLMAgentStrategy({"strategy": "llm_agent", "model_id": "m"})
    closes = []
    strategy._run_decision_loop(
        client=object(),
        timestamps=stamps,
        symbols=["AAPL"],
        data={},
        price_cache={},
        initial_capital=1.0,
        model_id="m",
        on_session_close=lambda day, curve: closes.append((day, strategy.llm_calls)),
    )
    assert Counter(
        ts.tz_convert("US/Eastern").date() for ts in _CountingManager.calls
    ) == {date(2026, 11, 25): 6, date(2026, 11, 27): 3}
    # One checkpoint per session, the counters cumulative when it fires.
    assert closes == [(date(2026, 11, 25), 6), (date(2026, 11, 27), 9)]


@pytest.mark.parametrize(
    ("month_start", "month_end", "sessions", "early_closes", "decisions"),
    [
        ("2026-08-01", "2026-08-31", 21, 0, 126),
        ("2026-11-01", "2026-11-30", 20, 1, 117),
        ("2026-12-01", "2026-12-31", 22, 1, 129),
    ],
)
def test_a_months_decisions_follow_the_calendar(
    month_start, month_end, sessions, early_closes, decisions
):
    """Spec §7's 126 calls is 6 x 21 sessions. An early close takes three off;
    retries are billed on top of this floor."""
    days = live._trading_days_inclusive(
        date.fromisoformat(month_start), date.fromisoformat(month_end)
    )
    assert len(days) == sessions
    assert sum(is_early_close(day) for day in days) == early_closes
    stamps = _common.market_timestamps(_alpaca_hourly([d.isoformat() for d in days]))
    assert len(stamps) == decisions == 6 * sessions - 3 * early_closes


def test_every_axis_node_after_a_days_first_is_where_a_decision_bar_closes():
    """The 10:00 node carries the prior close (the 09:00 bar straddles the open
    and is never a decision); every later node is one decision bar's close.
    Pins the chart axis and the decision filter to one calendar."""
    days = live._trading_days_inclusive(date(2026, 11, 1), date(2026, 11, 30))
    axis = live.live_month_hourly_axis("2026-11-01", "2026-11-30")
    stamps = _common.market_timestamps(_alpaca_hourly([d.isoformat() for d in days]))
    decision_nodes = [live._axis_node_key(live._parse_to_et(ts.isoformat())) for ts in stamps]
    first_nodes = {f"{day.isoformat()}T10:00" for day in days}
    assert decision_nodes == [ts[:16] for ts in axis if ts[:16] not in first_nodes]
```

- [ ] **Step 2: Run it, then prove it bites.** This pins behaviour Tasks 1.2–1.3 already shipped, so it passes first time:

```bash
pytest dashboard/backend/tests/domain/leaderboard/test_llm_agent_cadence.py -q -p no:cacheprovider
```

Expected: `6 passed`. To show the pins fail without the fix, put back the pre-Task-1.2 `sessions.py` (it still has Task 1.1's `session_close_time`, so `live.py` imports cleanly), run, and restore:

```bash
SHA=$(git log -1 --format=%h --grep='^fix(sessions): end the open-stamped US session')
git show "$SHA^:dashboard/backend/infrastructure/market_data/sessions.py" > dashboard/backend/infrastructure/market_data/sessions.py
pytest dashboard/backend/tests/domain/leaderboard/test_llm_agent_cadence.py -q -p no:cacheprovider
git checkout HEAD -- dashboard/backend/infrastructure/market_data/sessions.py
git status --short   # only the new test file may be listed
```

Expected for the middle run: `5 failed, 1 passed`. Only the August row of `test_a_months_decisions_follow_the_calendar` passes (it has no early close); November counts 120, not 117, and December 132, not 129.

- [ ] **Step 3: Implement.** Nothing to implement: the cadence already matches spec §6 (Decision 2 in the index plan). Do not touch `llm_agent.py`.

- [ ] **Step 4: Run it and its neighbours.**

```bash
pytest dashboard/backend/tests/domain/leaderboard -q -p no:cacheprovider
pytest dashboard/backend/tests/test_live_leaderboard.py -q -p no:cacheprovider
```

Expected: all pass, `100 passed` for `tests/domain/leaderboard` (the cadence file's 6 included) and `70 passed` for `test_live_leaderboard.py`, on main@17c552c5 + Tasks 1.1–1.4.

- [ ] **Step 5: Commit.**

```bash
git branch --show-current   # must print fix/live-early-close
git add dashboard/backend/tests/domain/leaderboard/test_llm_agent_cadence.py
git commit -m "test(live): pin six decisions per full session and three per early close"
```

### Task 1.5: Open the PR (merge and deploy by 2026-11-20)

**Files:** none changed.

**Interfaces:**
- Consumes: Tasks 1.1–1.4 committed on `fix/live-early-close`.
- Produces: an open PR against `main`, not a draft (nothing should block this merge); after the merge, one issue that carries the two dated production checks.

The PR body's second paragraph is the merge-window rule. It is the only merge gate this PR carries, and whoever merges reads it there, so keep it directly under the deadline line.

- [ ] **Step 1: Run the full suite.** It takes 5–8 minutes (454 s measured on main@17c552c5 + this phase): in an agent harness give the command a 10-minute timeout or run it in the background. Do not add `-x` (it stops at the first environmental `test_report_pdf.py` failure) or `--timeout` (`pytest-timeout` is not installed, so pytest rejects the flag).

```bash
pytest dashboard/backend/tests -q -p no:cacheprovider 2>&1 | tee "$(git rev-parse --git-dir)/p1-full.log" | tail -3
grep '^FAILED' "$(git rev-parse --git-dir)/p1-full.log" | grep -v 'tests/test_report_pdf.py::'   # must print nothing
```

Expected: the second command prints nothing. Judge by file, not by count: every failure must be in `test_report_pdf.py`, 7 of whose 9 tests need `reportlab`, which is not installed locally (environmental; CI installs it, and there they pass). Any other `FAILED` line is a regression. The log sits in the git directory, so it survives a separate shell and never shows in `git status`. Measured on main@17c552c5 + Tasks 1.1–1.4 without `reportlab`: `7 failed, 6736 passed, 176 skipped, 25 xfailed`, all 7 in `test_report_pdf.py`.

- [ ] **Step 2: Confirm the branch and push.**

```bash
git branch --show-current                                  # fix/live-early-close
gh pr list --head fix/live-early-close --state all         # expect no rows
git log --oneline origin/main..HEAD                        # the four commits of Tasks 1.1-1.4
git push -u origin fix/live-early-close
```

- [ ] **Step 3: Open the PR.**

```bash
gh pr create --base main --head fix/live-early-close \
  --title "fix(live): NYSE early closes on the live board" \
  --body-file - <<'EOF'
Merge and deploy by 2026-11-20: 2026-11-27 (13:00 ET close) is the first NYSE early close the live board has to trade.

Do not merge between 22:15 and 23:30 UTC on a weekday. The merge auto-deploys prod, and the Render restart kills the 22:30 UTC live refresh thread mid-run, so that night's session is billed again on the next night.

- Calendar: `nyse_early_closes` / `is_early_close` in `us_market_calendar.py`, by rule (day after Thanksgiving; Dec 24 Mon–Thu; Jul 3 when Jul 4 is Tue–Fri), pinned to NYSE's published 2025–2028 lists. The close times (`EARLY_CLOSE_TIME`, `REGULAR_CLOSE_TIME`, `session_close_time`) live in `sessions.py`, the only module `test_no_module_restates_the_session_bounds` lets spell a session bound.
- `sessions.is_in_session`, open-stamped branch only: a US early-close day ends at 13:00, so Alpaca's 13:00–15:00 late-session bars are not decision bars (on main they would be 3 extra billed model calls filled off post-close prints). This also applies to My Agents and protocol runs on early-close days. The close-stamped path is untouched (D19 stays with the engine rewrite): conformance unchanged, 25 strict xfails, no XPASS.
- Live board: 4 axis nodes on an early close (10:00–13:00), settles at 13:30, and the clock, node mapping and reindex read the per-day close. The window description and chart hint say so. The Competition board is unaffected: its window has no early close.
- Cadence pinned by tests: 6 decisions per full session, 3 per early close (Aug 2026 126, Nov 117, Dec 129 calls before retries).
- No fee, slippage or prompt change.

Tests: `pytest dashboard/backend/tests -q -p no:cacheprovider` (`test_report_pdf.py` needs `reportlab` locally).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
```

Drop the last line if a person, not an agent, opens the PR.

- [ ] **Step 4: After the merge, verify the deploy.** Merging to `main` auto-deploys prod once CI's backend tests pass. The live axis covers the whole calendar month, future days included, so from 2026-11-01 the 11-27 nodes can be checked straight away:

```bash
curl -s 'https://agentictrading.onrender.com/api/v1/leaderboard?period=live' \
  | jq '[.chart_axis[] | select(startswith("2026-11-27")) | .[11:16]]'
```

Expected from 2026-11-01 on: `["10:00","11:00","12:00","13:00"]` (and the same for `2026-12-24` from 2026-12-01). A merge in October cannot be checked this way until November 1. In October, check only that `curl -s https://agentictrading.onrender.com/health` answers and that the October board still renders.

- [ ] **Step 5: File the dated checks as an issue (ask the owner first).** A merged PR is write-only, so the two post-deploy checks need an issue. Filing on the shared repo assigns work, so get the owner's OK, then:

```bash
gh issue create --title "Verify the live board on the 2026-11-27 and 2026-12-24 early closes" --body-file - <<'EOF'
Owner check after the nightly refresh (22:30 UTC cron, i.e. after 17:30 ET) on **2026-11-27**, and again on **2026-12-24** with that date substituted.

    curl -s 'https://agentictrading.onrender.com/api/v1/leaderboard?period=live' | jq '{
      frozen_through: .live_status.frozen_through,
      stale: .live_status.snapshot_stale,
      axis: [.chart_axis[] | select(startswith("2026-11-27")) | .[11:16]],
      models: [.entries[] | select(.is_model) | {entry_id, status, snapshot_end, portfolio_value,
        nodes: [.equity_curve[] | select(.timestamp | startswith("2026-11-27")) | [.timestamp[11:16], .equity]]}]
    }'

Pass when:
- `frozen_through` is the date and `stale` is false;
- `axis` is `["10:00","11:00","12:00","13:00"]`;
- every model with `snapshot_end` equal to the date has 4 nodes ending at `13:00`, and its `portfolio_value` equals that 13:00 node's equity. The last stored mark should be the 12:00 bar's; a mark on a post-close bar would usually make the two differ;
- the Render log for that refresh shows `live increment <entry>: trade 2026-11-27 → 2026-11-27` followed by `🤖 LLM agent baseline: … steps=3 llm=on`. A catch-up spanning more days shows `steps` = 6 per full session + 3. Either way, nothing was traded after 13:00.

Source: PR #<n> (fix/live-early-close).
EOF
```

Replace `<n>` with the PR number `gh pr create` printed in Step 3. Then link the issue from the PR with one comment: `gh pr comment <n> --body "Post-deploy checks for 2026-11-27 and 2026-12-24: #<issue>"`, where `<issue>` is the number `gh issue create` printed.

#### Phase 1 review-focus candidates

1. **My Agents minute-mode backtest across an early close** (5m source bars that include the late session). Covered: `test_minute_mode_stops_filling_at_an_early_close` was added to Task 1.2. It pins that only the 10:30, 11:30 and 12:30 decisions fill, and that valuation stops at 12:55.
2. **A manual refresh between 13:00 and 13:30 ET on 2026-11-27** (`workflow_dispatch` in the settle window). Covered by the `13:05 → settling, frozen 2026-11-25` case in `test_the_clock_closes_at_13_00_on_an_early_close`. It must not freeze a truncated 12:00 bar.
3. **A catch-up or forced replay whose segment spans Thanksgiving and the half day.** Covered: `test_a_replay_across_thanksgiving_and_the_early_close_is_not_a_gap` runs the real `LLMAgentStrategy` decision loop over 11-25 → 11-27. It checks 6 then 3 decisions, one checkpoint per session, no `LiveSessionGapError`, and 4 painted nodes. A full-month replay takes the same per-day path.
4. **Yahoo index baselines (`^DJI`, `^GSPC`) on the half day.** They are not session-filtered at all. Covered at the mapping layer: `test_axis_node_key_caps_at_the_early_close` maps Yahoo's 12:30 half bar to the 13:00 node and drops anything that opens at or after 13:00. It is not tested against a live Yahoo response, because the suite never calls the network.
5. **An unscheduled early close** (a closure NYSE announces at short notice). Not testable: no rule can know it. A full-day closure is still caught loudly by `_session_gap`. An unscheduled *early* close would need its date added to `nyse_early_closes` by hand.
6. **Cost on the shared filter's hot path.** `is_in_session`'s open-stamped branch now also calls `session_close_time` (a second `session_windows` plus an `lru_cache`d frozenset lookup) on every stamp. All four call sites (`baseline_generator._market_hours_only`, `engine._market_hours_only`, `market_data_store._build_trading_timestamps`, `_common.filter_market_hours` via `market_timestamps`) filter one list of timestamps, already unioned across symbols where there are several, not symbol × timestamp, so a 14-day 5m run is a few thousand calls. The full suite ran in 454 s on main@17c552c5 + this phase, inside the 5–8 minutes the same suite took on main with the other phases applied. Not measured on a prod-size run; worth a glance, not a blocker.
