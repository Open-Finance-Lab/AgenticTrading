# Live-Board Fixes P5: Daily Sharpe with its 95% interval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hourly `agent_runs.sharpe_ratio` that both leaderboards echo with a daily Sharpe and its 95% interval, computed once by the backend at read time from the stored curve and only displayed by the Live table, the Competition table and detail, and the home rank list.

**Architecture:** A pure `daily_sharpe_interval` in `domain/backtesting/metrics.py` (one return per US/Eastern trading day from the seed, ddof 1, ×√252, iid standard error) is called on the read path of `live.get_live_leaderboard` (off the reindexed display curve) and `service.get_leaderboard` (off the scaled hourly curve). Every entry gains `sharpe_ci_low`, `sharpe_ci_high`, `sharpe_days` and `sharpe_status`, and each payload a basis block. The JS formatters only map those fields to text, a parity test holds the two board formatters to one answer, and `calculate_sharpe`, `calc_metrics` and the stored column stay untouched.

**Tech Stack:** Python 3.13 / FastAPI backend (stdlib `statistics` and `zoneinfo`), pytest from the repo root, the vanilla-JS dashboard (`js/live-leaderboard.js`, `js/leaderboard.js`, `home-page.js`, `app.html` tooltips and cache-busters) tested under `node -e`.

**Spec:** `docs/superpowers/specs/2026-10-14-atl-direction.md` §4 (Credibility bar: the Sharpe rule) and §6 (Fixes to the board while it runs on the legacy engine: "The Sharpe rule from §4"). Decisions, Global Constraints, merge order and Review Focus live in the index plan `docs/superpowers/plans/2026-10-14-live-board-fixes.md`; every task here inherits its Global Constraints.

**Merge order:** P1 → P2 → P3 → P4 → P5 → P6 → P7 PR A → P7 PR B (owner-gated). Deadlines: P3 before 2026-11-01; P4 by 2026-10-31 (else bump both live_from values); P1 merged and deployed by 2026-11-20. One executor per contiguous range; phases are not safe to run in parallel (several append to the same test files and bump the same cache-busters).

## Phase 5: Daily Sharpe with its 95% interval on the Live and Competition boards

**Why.** Spec §4: Sharpe is daily, computed once by the backend (P19), never recomputed in the browser (D20), shown from 5 trading days and always with its 95% interval; §6 lists it among the legacy-engine board fixes (no fee or slippage change). Today both boards echo `agent_runs.sharpe_ratio`, which `calc_metrics` → `calculate_sharpe` writes from hourly marks with ddof 0, rf 0, ×√1638 and 0 for "undefined"; the Competition board adds `or 0` on top. Decision 7 in the index plan replaces that on the read path of both boards and leaves `calculate_sharpe`, `calc_metrics` and the stored column alone.

**Depends on.** **Phase 1 is a prerequisite** (P1, `2026-10-14-live-board-fixes-p1-early-close.md`): Task 5.2's `test_live_sharpe_counts_an_early_close_once` reads the November 2026 board through Phase 1's early-close axis, on which 2026-11-27 has four nodes ending at 13:00, and fails at its axis assertion without it. Before Task 5.1, confirm Phase 1 is on your base: `grep -c 'def _cash_close' dashboard/backend/domain/leaderboard/live.py` must print `1`. It prints `0` on main@17c552c5, which means Phase 1 has not merged: stop. Phases 2–4 (P2 `2026-10-14-live-board-fixes-p2-memory-robustness.md`, P3 `2026-10-14-live-board-fixes-p3-attempt-ledger.md`, P4 `2026-10-14-live-board-fixes-p4-roster.md`; all four files in `docs/superpowers/plans/`) are not functional dependencies: this phase only reads the curves they write. Merge it after all four: every anchor below is a function name plus a verbatim snippet, and line numbers are hints on main@17c552c5. The whole section was dry-run verbatim on main@17c552c5 alone (every find matched exactly once; every Step 2 failed as stated), and every find was checked again to match exactly once on main@17c552c5 plus Phase 1. Downstream, P6 (`2026-10-14-live-board-fixes-p6-copy.md`) and P7 PR B (`2026-10-14-live-board-fixes-p7-records.md`) must merge within the same week as this phase (Task 5.6 Step 4). Where Phases 1–4 move an anchor, the step gives both forms:
- **Phase 4 (roster)** renames Nemotron to Qwen across `test_live_leaderboard.py` with a whole-file `sed` (`nemotron_3_nano_30b` → `qwen3_7_plus`, the local variable `nemotron` → `qwen`) and adds the autouse fixture `_roster_without_join_dates`, which strips `live_from` so every roster entry is live in August 2026. Task 5.2 Step 1's first edit therefore has a main-only form (`nemotron[...]`) and a post-Phase-4 form (`qwen[...]`); apply whichever the file contains. The numbers do not depend on the entry.
- **Phase 3 (budget)** bumps `js/live-leaderboard.js?v=5` → `?v=6`, so Task 5.3 bumps `6` → `7` after it (the step gives both). Nothing in Phases 1–4 touches `leaderboard.js?v=37` or `home-page.js?v=55`.
- **Phase 3** also inserts `const attemptBadge = …` after `const dash = …` in `populateLiveTable`, `const lastRun = …` after `const hasPrints = …` in `renderLiveDetail`, a `last_attempt` key per entry and `budget_halted` at the end of `live_status`. None of Task 5.2/5.3's anchors includes those lines, so they still match; keep the inserted lines where they are. Each anchor here is one to three lines for that reason.

The Live tests use August 2026 (no holidays or early closes) and `deepseek_v4_pro`, which is on the Live roster before and after Decision 6 in the index plan. The one exception is `test_live_sharpe_counts_an_early_close_once`, which reads the November 2026 board through 2026-11-30 (Thanksgiving shut, 2026-11-27 closing at 13:00); it is why Phase 1 is a prerequisite. Tests appended to `test_live_leaderboard.py` inherit Phase 4's autouse fixture once it exists.

**Applying an "Append to the end of" block.** Append the fenced block's content, without the fence lines, to the end of the file. Each such block deliberately starts with two blank lines, so the new code sits two blank lines below the file's last line.

**Expected counts.** Every count in a Step 2 or Step 4 below was measured on main@17c552c5 plus this phase alone, except the ones marked "with Phase 1" (Task 5.2's, and the second command of Task 5.4 Step 4). Those include `test_live_leaderboard.py`, whose early-close test needs Phase 1, so they were measured on main@17c552c5 plus Phase 1 plus this phase. Where earlier phases added tests to the same files, the `failed` numbers stay the same and the `passed` numbers grow by however many tests they added.

**Branch / PR title.** `feat/board-daily-sharpe` / `feat(leaderboard): daily Sharpe with its 95% interval`

**Interfaces this phase produces.**
- `dashboard/backend/domain/backtesting/metrics.py` (new names only):
  - `SHARPE_TRADING_DAYS_PER_YEAR = 252`, `SHARPE_MIN_DAYS = 5`, `SHARPE_CONFIDENCE = 0.95`, `SHARPE_RF_ANNUAL = 0.0`.
  - `SHARPE_STATUS_OK = "ok"`, `SHARPE_STATUS_INSUFFICIENT_DAYS = "insufficient_days"`, `SHARPE_STATUS_NO_VARIATION = "no_variation"`, `SHARPE_STATUS_PENDING = "pending"`.
  - `sharpe_standard_error_daily(sharpe_daily: float, n: int) -> float`: `sqrt((1 + SR_d²/2)/n)`, not annualised. Raises `ValueError` for `n < 1`.
  - `daily_session_closes(points, *, tz=ZoneInfo("America/New_York")) -> List[Tuple[date, float]]`: the last finite equity per Eastern date, in date order. Naive stamps are read as UTC.
  - `daily_sharpe_basis(*, rf_annual=0.0, confidence=0.95, min_days=5) -> Dict` returns `{"basis": "daily_returns", "annualization_days": 252, "confidence": 0.95, "min_days": 5, "rf_annual": 0.0, "stderr": "iid"}`.
  - `daily_sharpe_interval(points, initial_equity, *, rf_annual=0.0, confidence=0.95, min_days=5) -> Dict` returns `{"sharpe", "ci_low", "ci_high", "days", "status", "rf_annual"}`. The three values are None unless `status == "ok"`, and never NaN. It raises `ValueError` for `min_days < 2`, a `confidence` outside (0, 1) or a non-finite `rf_annual`.
- Payload, every entry on both boards (Live: `get_live_leaderboard`; Competition and the parked daily board: `get_leaderboard`):
  - `sharpe_ratio`: `float | null`. It is now the daily figure, null unless the status is `ok`.
  - `sharpe_ci_low`, `sharpe_ci_high`: `float | null`.
  - `sharpe_days`: `int`.
  - `sharpe_status`: `"ok" | "insufficient_days" | "no_variation" | "pending"`.
- Board-level basis, equal to `daily_sharpe_basis()`: `live_status.sharpe` on Live, top-level `sharpe_basis` on Competition.
- JS:
  - `liveFormatSharpe(entry, basis) -> {main, sub, title}` in `js/live-leaderboard.js`.
  - `boardSharpeBasis() -> object | null`, `formatBoardSharpe(entry, basis) -> {main, sub, title}` and the `window.formatBoardSharpe` export in `js/leaderboard.js`.
  - `homeFormatSharpe(entry, basis) -> {text, title}` in `home-page.js`, where `renderEntries(entries, { sample, sharpeBasis })` gains `sharpeBasis`.
  - `liveFormatSharpe` and `formatBoardSharpe` return identical results for the same input, and a test pins that.
- Unchanged on purpose: `calculate_sharpe`, `calc_metrics`, the `agent_runs.sharpe_ratio` column and every run-level API that echoes it.

**Worked numbers every task reuses.** These come from conformance case M03: `r = [0, 0.02, −0.05, 0.05, 0.01]` from `E_0 = 100,000`, closes `[100000, 102000, 96900, 101745, 102762.45]`.
- At rf 0: mean 0.006 and stdev(ddof 1) 0.0364691651, so SR_d = 0.164522549 and SR = 0.164522549 × √252 = **2.611715**. SE = √((1 + SR_d²/2)/5) × √252 = 7.147175, and z = `NormalDist().inv_cdf(0.975)` = 1.959964, so the interval is **[−11.396490, 16.619919]**.
- At rf 0.0252 (rf_d = 0.0001): **2.568186**, the value M03 states, with the interval **[−11.436927, 16.573299]**.
- At n = 5 the 95% half-width is about ±14. The spec's "about ±7" is one standard error. No copy in this phase quotes a width.

### Task 5.1: Daily Sharpe and its interval in metrics.py

**Files:**
- Create: `dashboard/backend/tests/backtesting/test_daily_sharpe.py`
- Modify: `dashboard/backend/domain/backtesting/metrics.py`

**Interfaces:**
- Consumes: `dashboard.backend.tests.conformance.cases.M03` (its `expected.equity`, `initial_cash`, `rf_annual`, `expected.sharpe`). Nothing else beyond the stdlib (`math`, `statistics`, `datetime`, `zoneinfo`).
- Produces: every metrics name listed under **Interfaces this phase produces**.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/backtesting/test_daily_sharpe.py` with exactly:

```python
"""Daily Sharpe and its 95% interval -- the number both leaderboards publish.

Every expected value is hand-computed (conformance case M03, design doc
2026-10-07 §4) or derived from the formula's own terms, never read back from
the function under test. P19: one return per US trading day, ``E_0`` the seed,
ddof 1, ``rf_annual / 252`` simple; the interval is the iid standard error.
"""

import json
import math

import pytest

from dashboard.backend.domain.backtesting.metrics import (
    SHARPE_CONFIDENCE,
    SHARPE_MIN_DAYS,
    SHARPE_RF_ANNUAL,
    SHARPE_TRADING_DAYS_PER_YEAR,
    daily_session_closes,
    daily_sharpe_basis,
    daily_sharpe_interval,
    sharpe_standard_error_daily,
)
from dashboard.backend.tests.conformance.cases import M03

# M03: r = [0, 0.02, -0.05, 0.05, 0.01] from E_0 = 100,000.
M03_CLOSES = [100000.0, 102000.0, 96900.0, 101745.0, 102762.45]
SEED = 100000.0


def _daily(closes, *, first_day=3):
    """One 15:00 ET bar (19:00Z, open-stamped) per August 2026 session."""
    return [
        {"timestamp": f"2026-08-{first_day + i:02d}T19:00:00+00:00", "equity": equity}
        for i, equity in enumerate(closes)
    ]


def test_constants_are_the_p19_rule():
    assert SHARPE_TRADING_DAYS_PER_YEAR == 252
    assert SHARPE_MIN_DAYS == 5
    assert SHARPE_CONFIDENCE == 0.95
    assert SHARPE_RF_ANNUAL == 0.0


def test_m03_anchor_with_its_cash_rate():
    out = daily_sharpe_interval(_daily(M03_CLOSES), SEED, rf_annual=0.0252)
    assert out["status"] == "ok"
    assert out["days"] == 5
    assert out["rf_annual"] == 0.0252
    assert out["sharpe"] == pytest.approx(2.568186, abs=1e-6)
    assert out["ci_low"] == pytest.approx(-11.436927, abs=1e-6)
    assert out["ci_high"] == pytest.approx(16.573299, abs=1e-6)


def test_m03_at_the_boards_cash_rate_of_zero():
    out = daily_sharpe_interval(_daily(M03_CLOSES), SEED)
    assert out["status"] == "ok"
    assert out["rf_annual"] == 0.0
    assert out["sharpe"] == pytest.approx(2.611715, abs=1e-6)
    assert out["ci_low"] == pytest.approx(-11.396490, abs=1e-6)
    assert out["ci_high"] == pytest.approx(16.619919, abs=1e-6)


def test_matches_the_conformance_case_m03_as_stated():
    """The suite's own case object, so a transcription here cannot drift."""
    points = [
        {"timestamp": stamp.isoformat(), "equity": float(value)}
        for stamp, value in M03.expected.equity
    ]
    out = daily_sharpe_interval(
        points, float(M03.initial_cash), rf_annual=float(M03.rf_annual)
    )
    assert out["sharpe"] == pytest.approx(float(M03.expected.sharpe), abs=1e-6)


def test_standard_error_formula():
    assert sharpe_standard_error_daily(1.0, 5) == pytest.approx(0.5477225575, abs=1e-9)
    assert sharpe_standard_error_daily(0.0, 5) == pytest.approx(0.4472135955, abs=1e-9)
    assert sharpe_standard_error_daily(0.0, 20) == pytest.approx(0.2236067977, abs=1e-9)
    with pytest.raises(ValueError):
        sharpe_standard_error_daily(0.0, 0)


def test_five_days_is_the_boundary():
    four = daily_sharpe_interval(_daily(M03_CLOSES[:4]), SEED)
    assert four == {
        "sharpe": None,
        "ci_low": None,
        "ci_high": None,
        "days": 4,
        "status": "insufficient_days",
        "rf_annual": 0.0,
    }
    five = daily_sharpe_interval(_daily(M03_CLOSES), SEED)
    assert five["status"] == "ok"
    assert five["days"] == 5


def test_one_day_and_no_days():
    one = daily_sharpe_interval(_daily([101000.0]), SEED)
    assert one["status"] == "insufficient_days"
    assert one["days"] == 1
    assert one["sharpe"] is None
    empty = daily_sharpe_interval([], SEED)
    assert empty["status"] == "pending"
    assert empty["days"] == 0
    assert empty["sharpe"] is None and empty["ci_low"] is None and empty["ci_high"] is None


def test_a_flat_curve_is_no_variation_not_zero():
    """M03b's rule: undefined is null. ``calculate_sharpe`` answers 0 here."""
    out = daily_sharpe_interval(_daily([SEED] * 6), SEED)
    assert out["status"] == "no_variation"
    assert out["days"] == 6
    assert out["sharpe"] is None
    assert out["ci_low"] is None and out["ci_high"] is None


def test_float_noise_is_not_a_return():
    """One ulp above the seed on every day: r_1 = 2.2e-16 and nothing after.
    Without the noise floor this publishes a Sharpe of 6.48."""
    noisy = math.nextafter(SEED, math.inf)
    out = daily_sharpe_interval(_daily([noisy] * 6), SEED)
    assert out["status"] == "no_variation"
    assert out["sharpe"] is None


def test_intraday_points_collapse_to_the_last_point_of_each_et_day():
    points = _daily(M03_CLOSES)
    # Earlier marks on each day must not count; 01:00Z on Aug 4 is 21:00 ET on
    # Aug 3, so it is Aug 3's close and the 19:00Z Aug 3 mark is not.
    points += [
        {"timestamp": f"2026-08-{d:02d}T14:00:00+00:00", "equity": 1.0}
        for d in range(3, 8)
    ]
    points[0] = {"timestamp": "2026-08-03T19:00:00+00:00", "equity": 55555.0}
    points.append({"timestamp": "2026-08-04T01:00:00+00:00", "equity": M03_CLOSES[0]})
    closes = daily_session_closes(points)
    assert [day.isoformat() for day, _ in closes] == [
        "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07",
    ]
    assert [equity for _, equity in closes] == M03_CLOSES
    out = daily_sharpe_interval(points, SEED)
    assert out["sharpe"] == pytest.approx(2.611715, abs=1e-6)


def test_order_is_by_timestamp_not_list_position():
    out = daily_sharpe_interval(list(reversed(_daily(M03_CLOSES))), SEED)
    assert out["sharpe"] == pytest.approx(2.611715, abs=1e-6)


def test_naive_timestamps_are_read_as_utc():
    # 02:00 UTC on Aug 4 is 22:00 ET on Aug 3. Read as ET it would be Aug 4.
    closes = daily_session_closes([
        {"timestamp": "2026-08-03T14:00:00", "equity": 1.0},
        {"timestamp": "2026-08-04T02:00:00", "equity": 2.0},
    ])
    assert [(day.isoformat(), equity) for day, equity in closes] == [("2026-08-03", 2.0)]


def test_zero_or_nan_equity_never_yields_nan():
    # A stored NULL (issue #390) and a NaN are "no observation", never $0.
    gappy = _daily(M03_CLOSES)
    gappy.insert(2, {"timestamp": "2026-08-04T20:00:00+00:00", "equity": float("nan")})
    gappy.insert(4, {"timestamp": "2026-08-05T20:00:00+00:00", "equity": None})
    skipped = daily_sharpe_interval(gappy, SEED)
    assert skipped["sharpe"] == pytest.approx(2.611715, abs=1e-6)

    wiped = daily_sharpe_interval(_daily([100000.0, 0.0, 0.0, 0.0, 0.0, 0.0]), SEED)
    assert wiped["status"] == "no_variation"
    assert wiped["sharpe"] is None

    for seed in (0.0, -1.0, float("nan"), None, "junk"):
        out = daily_sharpe_interval(_daily(M03_CLOSES), seed)
        assert out["status"] == "no_variation", seed

    for out in (skipped, wiped):
        json.dumps(out, allow_nan=False)


def test_a_last_day_wipeout_is_a_real_return():
    """Only a return's denominator must be positive: ending at $0 is -100%."""
    out = daily_sharpe_interval(_daily(M03_CLOSES[:4] + [0.0]), SEED)
    assert out["status"] == "ok"
    assert out["sharpe"] < 0
    json.dumps(out, allow_nan=False)


def test_the_interval_brackets_the_estimate_and_is_symmetric():
    closes = [100400.0, 99800.0, 101300.0, 100900.0, 102100.0, 101700.0, 103000.0, 102600.0]
    out = daily_sharpe_interval(_daily(closes), SEED)
    assert out["status"] == "ok" and out["days"] == 8
    assert out["ci_low"] < out["sharpe"] < out["ci_high"]
    assert out["sharpe"] - out["ci_low"] == pytest.approx(out["ci_high"] - out["sharpe"])
    down = daily_sharpe_interval(_daily([SEED * SEED / c for c in closes]), SEED)
    assert down["sharpe"] < 0
    assert down["ci_low"] < down["sharpe"] < down["ci_high"]


def test_the_basis_block_states_every_number_the_ui_quotes():
    assert daily_sharpe_basis() == {
        "basis": "daily_returns",
        "annualization_days": 252,
        "confidence": 0.95,
        "min_days": 5,
        "rf_annual": 0.0,
        "stderr": "iid",
    }


def test_programmer_errors_raise_rather_than_publish():
    with pytest.raises(ValueError):
        daily_sharpe_interval(_daily(M03_CLOSES), SEED, min_days=1)
    with pytest.raises(ValueError):
        daily_sharpe_interval(_daily(M03_CLOSES), SEED, confidence=1.0)
    with pytest.raises(ValueError):
        daily_sharpe_interval(_daily(M03_CLOSES), SEED, rf_annual=math.inf)
```


- [ ] **Step 2: Run it and confirm it fails**

```bash
pytest dashboard/backend/tests/backtesting/test_daily_sharpe.py -q -p no:cacheprovider
```

Expected: a collection error and 0 tests run (`1 error`, `Interrupted: 1 error during collection`), with `ImportError: cannot import name 'SHARPE_CONFIDENCE' from 'dashboard.backend.domain.backtesting.metrics'`.

- [ ] **Step 3: Implement**

There are two edits. The second appends a block after `calculate_max_drawdown`, and `calculate_sharpe` / `calculate_max_drawdown` stay byte-identical. `_SHARPE_MIN_STDEV` goes beyond the text of Decision 7 in the index plan, for this reason: a flat curve that sits one ulp off `E_0` would otherwise publish 6.48 (`test_float_noise_is_not_a_return`). The committed contest board does carry a `100000.00000000003` seed.

In `dashboard/backend/domain/backtesting/metrics.py` (≈:12 on main@17c552c5, module level), find:

```python
from typing import Dict, List, Optional

import numpy as np
```

Replace with:

```python
import math
import statistics
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
```

In `dashboard/backend/domain/backtesting/metrics.py` (≈:88 on main@17c552c5, inside `calculate_max_drawdown`), find:

```python
            max_dd = dd

    return max_dd
```

Replace with:

```python
            max_dd = dd

    return max_dd


# ---------------------------------------------------------------------------
# Daily Sharpe with its 95% interval (P19 on the boards; spec 2026-10-14 §4)
# ---------------------------------------------------------------------------
#
# ``calculate_sharpe`` above annualises whatever steps a curve holds (hourly
# marks, ddof 0, rf 0, undefined -> 0). The leaderboards publish this one
# instead: one return per US trading day, ddof 1, a stated cash rate, an
# interval, and None -- never 0, never NaN -- when it is undefined. It is read
# at request time from a stored curve, so it must stay pure arithmetic: no I/O.

SHARPE_TRADING_DAYS_PER_YEAR = 252
SHARPE_MIN_DAYS = 5
SHARPE_CONFIDENCE = 0.95
# The legacy engine has no T-bill source, so the boards state 0%/yr and say so.
SHARPE_RF_ANNUAL = 0.0

SHARPE_STATUS_OK = "ok"
SHARPE_STATUS_INSUFFICIENT_DAYS = "insufficient_days"
SHARPE_STATUS_NO_VARIATION = "no_variation"
SHARPE_STATUS_PENDING = "pending"

_SHARPE_TZ = ZoneInfo("America/New_York")
# Below this a daily-return stdev is float noise, not variation. A flat curve
# rescaled onto the board's capital (the committed contest board carries a
# 100000.00000000003 seed) can sit one ulp off E_0 on day 1, and that alone
# would publish a Sharpe of about 6.5 for an account that never moved.
_SHARPE_MIN_STDEV = 1e-12


def sharpe_standard_error_daily(sharpe_daily: float, n: int) -> float:
    """Standard error of a per-day Sharpe estimated from ``n`` iid daily returns.

    Lo (2002), iid case: ``sqrt((1 + SR_d**2 / 2) / n)``. It is Mertens'
    formula with skew 0 and kurtosis 3; both moment estimates are noise at the
    5-20 returns a month holds, so the iid form is the stable, hand-checkable
    choice. Not annualised: multiply by ``sqrt(252)`` for the yearly figure.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    return math.sqrt((1.0 + sharpe_daily * sharpe_daily / 2.0) / n)


def _point_time(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        stamp = value
    elif value is None:
        return None
    else:
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    # Naive stamps are UTC: the convention of every stored curve in this repo
    # (live.py ``_parse_to_et``, service.py ``_curve_span``).
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def _finite_equity(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def daily_session_closes(
    points: Iterable[Dict[str, Any]], *, tz: ZoneInfo = _SHARPE_TZ
) -> List[Tuple[date, float]]:
    """The last finite equity of each ``tz`` calendar date, in date order.

    Points are ordered by timestamp, not by list position. A point with no
    parseable timestamp, or an equity that is None or not finite, is "no
    observation" and skipped; 0 and negatives are real observations and kept.
    """
    observed: List[Tuple[datetime, int, float]] = []
    for index, point in enumerate(points or []):
        stamp = _point_time(point.get("timestamp"))
        equity = _finite_equity(point.get("equity"))
        if stamp is None or equity is None:
            continue
        observed.append((stamp, index, equity))
    observed.sort(key=lambda item: (item[0], item[1]))
    closes: Dict[date, float] = {}
    for stamp, _index, equity in observed:
        closes[stamp.astimezone(tz).date()] = equity
    return sorted(closes.items())


def daily_sharpe_basis(
    *,
    rf_annual: float = SHARPE_RF_ANNUAL,
    confidence: float = SHARPE_CONFIDENCE,
    min_days: int = SHARPE_MIN_DAYS,
) -> Dict[str, Any]:
    """What a published ``sharpe_ratio`` means: one owner for the board copy."""
    return {
        "basis": "daily_returns",
        "annualization_days": SHARPE_TRADING_DAYS_PER_YEAR,
        "confidence": confidence,
        "min_days": min_days,
        "rf_annual": rf_annual,
        "stderr": "iid",
    }


def daily_sharpe_interval(
    points: Iterable[Dict[str, Any]],
    initial_equity: Any,
    *,
    rf_annual: float = SHARPE_RF_ANNUAL,
    confidence: float = SHARPE_CONFIDENCE,
    min_days: int = SHARPE_MIN_DAYS,
) -> Dict[str, Any]:
    """Annualised daily Sharpe and its interval, from an equity curve.

    ``E_0 = initial_equity``; ``E_d`` is the last finite point of US/Eastern
    date ``d``. ``r_d = E_d / E_{d-1} - 1``, ``x_d = r_d - rf_annual / 252``,
    ``SR = mean(x) / stdev(x, ddof=1) * sqrt(252)`` and the interval is
    ``SR -/+ z * sharpe_standard_error_daily(SR_d, n) * sqrt(252)``.

    Returns ``{"sharpe", "ci_low", "ci_high", "days", "status", "rf_annual"}``.
    ``days`` is the number of daily returns. The three values are floats only
    when ``status == "ok"`` and None otherwise -- never NaN, which Postgres
    round-trips and Starlette refuses to encode (``allow_nan=False``):

    * ``pending``: no daily close at all;
    * ``insufficient_days``: fewer than ``min_days`` returns;
    * ``no_variation``: a return is undefined (an ``E_{d-1}`` that is 0 or
      negative, or a seed that is not a positive number) or the returns never
      moved (stdev 0, or float noise below ``_SHARPE_MIN_STDEV``), or the
      arithmetic overflowed.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if min_days < 2:
        raise ValueError("min_days must be at least 2: a sample stdev needs two returns")
    if not math.isfinite(rf_annual):
        raise ValueError("rf_annual must be finite")

    closes = [equity for _day, equity in daily_session_closes(points)]
    days = len(closes)

    def _result(status: str, sharpe=None, low=None, high=None) -> Dict[str, Any]:
        return {
            "sharpe": sharpe,
            "ci_low": low,
            "ci_high": high,
            "days": days,
            "status": status,
            "rf_annual": rf_annual,
        }

    if days == 0:
        return _result(SHARPE_STATUS_PENDING)
    if days < min_days:
        return _result(SHARPE_STATUS_INSUFFICIENT_DAYS)
    seed = _finite_equity(initial_equity)
    previous_levels = [seed] + closes[:-1]
    if any(level is None or level <= 0 for level in previous_levels):
        return _result(SHARPE_STATUS_NO_VARIATION)

    rf_daily = rf_annual / SHARPE_TRADING_DAYS_PER_YEAR
    excess = [
        close / previous - 1.0 - rf_daily
        for previous, close in zip(previous_levels, closes)
    ]
    if not all(math.isfinite(value) for value in excess):
        return _result(SHARPE_STATUS_NO_VARIATION)
    try:
        mean = statistics.fmean(excess)
        stdev = statistics.stdev(excess)
    except (OverflowError, ValueError):
        return _result(SHARPE_STATUS_NO_VARIATION)
    if not math.isfinite(mean) or not math.isfinite(stdev) or stdev <= _SHARPE_MIN_STDEV:
        return _result(SHARPE_STATUS_NO_VARIATION)

    sharpe_daily = mean / stdev
    annualise = math.sqrt(SHARPE_TRADING_DAYS_PER_YEAR)
    sharpe = sharpe_daily * annualise
    half_width = (
        statistics.NormalDist().inv_cdf(0.5 + confidence / 2.0)
        * sharpe_standard_error_daily(sharpe_daily, days)
        * annualise
    )
    low, high = sharpe - half_width, sharpe + half_width
    if not all(math.isfinite(value) for value in (sharpe, low, high)):
        return _result(SHARPE_STATUS_NO_VARIATION)
    return _result(SHARPE_STATUS_OK, sharpe, low, high)
```


- [ ] **Step 4: Run it and confirm it passes**

```bash
pytest dashboard/backend/tests/backtesting/test_daily_sharpe.py -q -p no:cacheprovider
pytest dashboard/backend/tests/backtesting/test_metrics.py dashboard/backend/tests/conformance/test_case_data.py -q -p no:cacheprovider
```

Expected: `17 passed` for the first command and `81 passed` for the second. `test_metrics.py` must stay green: it pins `calculate_sharpe`, which this task does not touch.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/backtesting/metrics.py dashboard/backend/tests/backtesting/test_daily_sharpe.py
git commit -m "feat(metrics): daily Sharpe with its 95% interval (P19)"
```

### Task 5.2: Live payload publishes daily Sharpe, its interval and its basis

**Files:**
- Modify: `dashboard/backend/domain/leaderboard/live.py`
- Test: `dashboard/backend/tests/test_live_leaderboard.py`

**Interfaces:**
- Consumes: `daily_sharpe_interval`, `daily_sharpe_basis` (Task 5.1); `reindex_frozen_curve`'s output (the curve `_entry_from_strategy` already receives, on the display axis, opening at `display_capital` before the first print). The early-close test also consumes `daily_session_closes` (Task 5.1) and Phase 1's per-day axis (`live_month_hourly_axis` draws 10:00–13:00 on 2026-11-27).
- Produces: entry keys `sharpe_ratio` (now daily), `sharpe_ci_low`, `sharpe_ci_high`, `sharpe_days`, `sharpe_status`; `live_status["sharpe"]`.

The computation goes in `_entry_from_strategy`, which already holds `curve`, `display_capital` and `printed`. That curve is scale-free in ratios, so `E_0 = display_capital` is exact on Live. The curve is the one `reindex_frozen_curve` built, which means sessions before an entry's first print are filled at the seed and count as idle days with r = 0, as P19 requires. Do not use `run["sharpe_ratio"]`.

- [ ] **Step 1: Write the failing test**

First, update the existing test that echoed the stored 1.1. Do not delete it. Its run prints on Aug 3 and Aug 26 only, but the reindexed curve as-of fills every session in between. That makes 18 days, not 2, so the figure is defined.

Check which form the file has: `rg -n 'sharpe_ratio"\] == pytest.approx\(1.1\)' dashboard/backend/tests/test_live_leaderboard.py` prints exactly one line, naming either `nemotron` (main@17c552c5, Phase 4 not merged) or `qwen` (Phase 4 merged). Apply the matching one of the two edits below, not both.

**Form A — Phase 4 not merged.**

In `dashboard/backend/tests/test_live_leaderboard.py` (≈:817 on main@17c552c5, inside `test_returns_come_off_the_stored_run_and_keep_the_first_hour`), find:

```python
    assert nemotron["sharpe_ratio"] == pytest.approx(1.1)
```

Replace with:

```python
    # Sharpe is not the stored 1.1: it is the daily figure off the plotted
    # curve. Aug 3-26 is 18 sessions; the as-of fill holds 0.995 x seed from
    # Aug 3 to Aug 25, so r = [-0.005, 0 x 16, 1.02/0.995 - 1]. Measured from the
    # first mark instead of the seed, r_1 would be 0 and this would read 3.74.
    assert nemotron["sharpe_ratio"] == pytest.approx(2.906881, abs=1e-6)
    assert nemotron["sharpe_ci_low"] == pytest.approx(-4.487853, abs=1e-6)
    assert nemotron["sharpe_ci_high"] == pytest.approx(10.301616, abs=1e-6)
    assert nemotron["sharpe_days"] == 18
    assert nemotron["sharpe_status"] == "ok"
```

**Form B — Phase 4 merged (the file was renamed to Qwen).**

In `dashboard/backend/tests/test_live_leaderboard.py` (same test, same place; after Phase 4's rename), find:

```python
    assert qwen["sharpe_ratio"] == pytest.approx(1.1)
```

Replace with:

```python
    # Sharpe is not the stored 1.1: it is the daily figure off the plotted
    # curve. Aug 3-26 is 18 sessions; the as-of fill holds 0.995 x seed from
    # Aug 3 to Aug 25, so r = [-0.005, 0 x 16, 1.02/0.995 - 1]. Measured from the
    # first mark instead of the seed, r_1 would be 0 and this would read 3.74.
    assert qwen["sharpe_ratio"] == pytest.approx(2.906881, abs=1e-6)
    assert qwen["sharpe_ci_low"] == pytest.approx(-4.487853, abs=1e-6)
    assert qwen["sharpe_ci_high"] == pytest.approx(10.301616, abs=1e-6)
    assert qwen["sharpe_days"] == 18
    assert qwen["sharpe_status"] == "ok"
```

Append to the end of `dashboard/backend/tests/test_live_leaderboard.py`:

```python


# --- Daily Sharpe with its 95% interval (P19; spec 2026-10-14 §4) -------------

# Conformance case M03 as session closes: r = [0, 0.02, -0.05, 0.05, 0.01].
_M03_FACTORS = (1.0, 1.02, 0.969, 1.01745, 1.0276245)


def _insert_sharpe_run(entry_id, days, factors, *, seed=SEED, sharpe=9.99):
    """One 15:00 ET bar per August session (19:00Z, open-stamped -> the 16:00
    node). ``sharpe`` is the stored column, which the board must ignore."""
    end = f"2026-08-{days[-1]:02d}"
    _insert_live_run(
        f"lb_{entry_id}_20260801_{end.replace('-', '')}", entry_id, end,
        [_pt(f"2026-08-{d:02d}T19:00:00+00:00", seed * f) for d, f in zip(days, factors)],
        total_return=factors[-1] - 1, sharpe=sharpe,
        metadata={"initial_capital": seed},
    )


def _sharpe_entry(payload, entry_id):
    return next(e for e in payload["entries"] if e["entry_id"] == entry_id)


def test_live_sharpe_is_daily_with_its_interval(no_alpaca):
    _insert_sharpe_run("deepseek_v4_pro", range(3, 8), _M03_FACTORS)
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    assert deepseek["sharpe_ratio"] == pytest.approx(2.611715, abs=1e-6)
    assert deepseek["sharpe_ci_low"] == pytest.approx(-11.396490, abs=1e-6)
    assert deepseek["sharpe_ci_high"] == pytest.approx(16.619919, abs=1e-6)
    assert deepseek["sharpe_days"] == 5
    assert deepseek["sharpe_status"] == "ok"


def test_live_sharpe_is_hidden_below_five_days(no_alpaca):
    _insert_sharpe_run("deepseek_v4_pro", range(3, 7), _M03_FACTORS[:4])
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 6, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    assert deepseek["status"] == "frozen"
    assert deepseek["sharpe_ratio"] is None
    assert deepseek["sharpe_ci_low"] is None and deepseek["sharpe_ci_high"] is None
    assert deepseek["sharpe_days"] == 4
    assert deepseek["sharpe_status"] == "insufficient_days"


def test_live_sessions_before_the_first_print_count_as_idle_days(no_alpaca):
    """P19 counts every session in the window. Printing first on Aug 5 leaves
    Aug 3-4 at the seed: two zero returns, so five days, not three."""
    _insert_sharpe_run("deepseek_v4_pro", (5, 6, 7), (1.01, 1.0, 1.02))
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    assert deepseek["sharpe_days"] == 5
    assert deepseek["sharpe_status"] == "ok"
    assert deepseek["sharpe_ratio"] == pytest.approx(5.611652, abs=1e-6)


def test_live_sharpe_is_the_same_at_any_seed(no_alpaca):
    _insert_sharpe_run("deepseek_v4_pro", range(3, 8), _M03_FACTORS, seed=10_000.0)
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    assert deepseek["sharpe_ratio"] == pytest.approx(2.611715, abs=1e-6)


def test_a_pending_entry_has_no_sharpe(no_alpaca):
    _insert_sharpe_run("spy_index", range(3, 8), _M03_FACTORS)
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    assert deepseek["status"] == "pending"
    assert deepseek["sharpe_ratio"] is None
    assert deepseek["sharpe_ci_low"] is None and deepseek["sharpe_ci_high"] is None
    assert deepseek["sharpe_days"] == 0
    assert deepseek["sharpe_status"] == "pending"


def test_live_payload_states_its_sharpe_basis(no_alpaca):
    from dashboard.backend.domain.backtesting.metrics import daily_sharpe_basis

    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    assert payload["live_status"]["sharpe"] == {
        "basis": "daily_returns",
        "annualization_days": 252,
        "confidence": 0.95,
        "min_days": 5,
        "rf_annual": 0.0,
        "stderr": "iid",
    }
    assert payload["live_status"]["sharpe"] == daily_sharpe_basis()


def test_live_payload_survives_strict_json(no_alpaca):
    """A flat curve is no_variation, and nothing on the wire is NaN."""
    import json

    _insert_sharpe_run("deepseek_v4_pro", range(3, 8), (1.0,) * 5)
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    assert deepseek["sharpe_status"] == "no_variation"
    assert deepseek["sharpe_ratio"] is None
    json.dumps(payload, allow_nan=False)


# November 2026's NYSE sessions: Thanksgiving (the 26th) shut, the 27th an
# early close at 13:00.
_NOV_2026_SESSIONS = (2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 27, 30)


def test_live_sharpe_counts_an_early_close_once(no_alpaca):
    """November 2026 through the 30th: 20 sessions, Thanksgiving (11-26) shut,
    11-27 closing at 13:00. Each session returns the next of conformance case
    M03's five returns, four times over, so the figure is hand-checkable:
    mean 0.006, stdev 0.0334664 (ddof 1), SR = 0.006 / 0.0334664 x sqrt(252)
    = 2.846050, and E_0 = the seed gives 20 returns, not 19.

    One 15:00 ET bar per full session (20:00Z in EST, open-stamped -> the 16:00
    node). On 11-27 the last bar is the 12:00 one (17:00Z), which closes at
    13:00, so the 13:00 node is that day's close. An axis with a fixed 16:00
    close draws 14:00-16:00 nodes after it carrying the same equity, so the
    Sharpe figure alone cannot tell the two axes apart; the curve assertions can.
    """
    from dashboard.backend.domain.backtesting.metrics import daily_session_closes

    closes, level = [], SEED
    for r in (0.0, 0.02, -0.05, 0.05, 0.01) * 4:
        level *= 1 + r
        closes.append(level)
    close_by_day = dict(zip(_NOV_2026_SESSIONS, closes))
    points = []
    for day, equity in close_by_day.items():
        if day == 27:
            # The 11:00 bar lands on the 12:00 node, the 12:00 bar on 13:00.
            points.append(_pt("2026-11-27T16:00:00+00:00", equity * 0.99))
            points.append(_pt("2026-11-27T17:00:00+00:00", equity))
        else:
            points.append(_pt(f"2026-11-{day:02d}T20:00:00+00:00", equity))
    _insert_live_run(
        "lb_deepseek_v4_pro_20261101_20261130", "deepseek_v4_pro", "2026-11-30", points,
        total_return=closes[-1] / SEED - 1, sharpe=9.99,
        metadata={"initial_capital": SEED}, start_date="2026-11-01",
    )
    payload = live.get_live_leaderboard(as_of=datetime(2026, 11, 30, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")

    half_day = [
        (p["timestamp"][11:16], p["equity"])
        for p in deepseek["equity_curve"] if p["timestamp"].startswith("2026-11-27")
    ]
    assert half_day == [
        ("10:00", close_by_day[25]),
        ("11:00", close_by_day[25]),
        ("12:00", close_by_day[27] * 0.99),
        ("13:00", close_by_day[27]),
    ]
    daily = dict(daily_session_closes(deepseek["equity_curve"]))
    assert [day.day for day in daily] == list(_NOV_2026_SESSIONS)
    assert daily[date(2026, 11, 27)] == close_by_day[27]

    assert deepseek["sharpe_ratio"] == pytest.approx(2.846050, abs=1e-6)
    assert deepseek["sharpe_ci_low"] == pytest.approx(-4.166815, abs=1e-6)
    assert deepseek["sharpe_ci_high"] == pytest.approx(9.858915, abs=1e-6)
    assert deepseek["sharpe_days"] == 20
    assert deepseek["sharpe_status"] == "ok"
```

`test_live_sharpe_counts_an_early_close_once` is the board-level check of the 2026-11-27 early close: November 2026 through the 30th is 20 sessions, so with `E_0` the seed it has 20 daily returns, and 11-27 counts once with the 13:00 node's equity as its close. Its Sharpe figure cannot tell the two axes apart: an axis with a fixed 16:00 close carries the 13:00 equity to 14:00–16:00 nodes, and the day's last point is the same number. Measured on main@17c552c5 plus this phase without Phase 1, its curve assertion fails and every Sharpe value it asserts still holds. So what it pins is Phase 1's axis under the Sharpe reader, and its failure is how a missing Phase 1 shows here.


- [ ] **Step 2: Run it and confirm it fails**

```bash
pytest dashboard/backend/tests/test_live_leaderboard.py -q -p no:cacheprovider
```

Expected, with Phase 1: `9 failed, 69 passed` on main@17c552c5 plus Phase 1 (every other test passes; main@17c552c5 alone lacks Phase 1's 13 tests and reports `9 failed, 56 passed`, its early-close failure being the one described below this list):
- `test_returns_come_off_the_stored_run_and_keep_the_first_hour`: `assert 1.1 == 2.906881 ± 1.0e-06`.
- `test_live_sharpe_is_daily_with_its_interval` and `test_live_sharpe_is_the_same_at_any_seed`: `assert 9.99 == 2.611715 ± 1.0e-06`.
- `test_live_sharpe_is_hidden_below_five_days`: `assert 9.99 is None`.
- `test_live_sessions_before_the_first_print_count_as_idle_days`: `KeyError: 'sharpe_days'`.
- `test_a_pending_entry_has_no_sharpe`: `KeyError: 'sharpe_ci_low'`.
- `test_live_payload_states_its_sharpe_basis`: `KeyError: 'sharpe'`.
- `test_live_payload_survives_strict_json`: `KeyError: 'sharpe_status'`.
- `test_live_sharpe_counts_an_early_close_once`: `assert 9.99 == 2.84605 ± 1.0e-06`. Its curve assertions already pass on Phase 1's axis.

If `test_live_sharpe_counts_an_early_close_once` fails instead on its first assertion, with `Left contains 3 more items, first extra item: ('14:00', 110412.03790131217)`, Phase 1 is not on this branch (that is the main@17c552c5 result): stop and rebase onto a `main` that has it.

- [ ] **Step 3: Implement**

There are five edits, all in `live.py`. Leave `_write_live_checkpoint`'s `"sharpe_ratio": metrics["sharpe_ratio"]` alone, because the stored column keeps its old meaning.

In `dashboard/backend/domain/leaderboard/live.py` (≈:30 on main@17c552c5, module level), find:

```python
from dashboard.backend.database import db
```

Replace with:

```python
from dashboard.backend.database import db
from dashboard.backend.domain.backtesting.metrics import (
    daily_sharpe_basis,
    daily_sharpe_interval,
)
```

The next find ends on a whole line whose tail (`A row with no run is *pending*: it has`) the replacement keeps as the first line of a new paragraph; the rest of the docstring is unchanged.

In `dashboard/backend/domain/leaderboard/live.py` (≈:1315 on main@17c552c5, inside `_entry_from_strategy`), find:

```python
    """One board row. Returns and risk come off the stored run untouched.

    The stored run's metrics were computed against the capital it was seeded
    with, over exactly the curve this row plots, so only the dollar axis is
    scaled (the contest board's rule). A row with no run is *pending*: it has
```

Replace with:

```python
    """One board row. Return and drawdown come off the stored run untouched.

    The stored run's metrics were computed against the capital it was seeded
    with, over exactly the curve this row plots, so only the dollar axis is
    scaled (the contest board's rule). Sharpe is the exception: the stored
    column is hourly marks with ddof 0 and 0 for "undefined", so the row
    publishes P19's daily figure and its 95% interval instead, read off
    ``curve`` (``daily_sharpe_interval``). ``curve`` is on the display axis and
    opens at ``display_capital`` before the first print, so ``E_0`` is the seed
    and day 1 keeps the first hour's P&L.

    A row with no run is *pending*: it has
```

In `dashboard/backend/domain/leaderboard/live.py` (≈:1337 on main@17c552c5, inside `_entry_from_strategy`), find:

```python
        total_return = run.get("total_return")
        sharpe = run.get("sharpe_ratio")
        max_dd = run.get("max_drawdown")
    else:
        portfolio_value = None
        total_return = None
        sharpe = None
        max_dd = None
```

Replace with:

```python
        total_return = run.get("total_return")
        max_dd = run.get("max_drawdown")
    else:
        portfolio_value = None
        total_return = None
        max_dd = None
    sharpe = daily_sharpe_interval(curve if printed else [], display_capital)
```

In `dashboard/backend/domain/leaderboard/live.py` (≈:1358 on main@17c552c5, inside `_entry_from_strategy`), find:

```python
        "sharpe_ratio": sharpe,
```

Replace with:

```python
        "sharpe_ratio": sharpe["sharpe"],
        "sharpe_ci_low": sharpe["ci_low"],
        "sharpe_ci_high": sharpe["ci_high"],
        "sharpe_days": sharpe["days"],
        "sharpe_status": sharpe["status"],
```

In `dashboard/backend/domain/leaderboard/live.py` (≈:1482 on main@17c552c5, inside `get_live_leaderboard`), find:

```python
    live_status = {
        "phase": LIVE_PHASE,
```

Replace with:

```python
    live_status = {
        "phase": LIVE_PHASE,
        # What every entry's sharpe_* fields mean; the table quotes these.
        "sharpe": daily_sharpe_basis(),
```


- [ ] **Step 4: Run it and confirm it passes**

```bash
pytest dashboard/backend/tests/test_live_leaderboard.py -q -p no:cacheprovider
pytest dashboard/backend/tests/test_leaderboard_api.py dashboard/backend/tests/test_leaderboard_season.py dashboard/backend/tests/test_live_trade_stats.py -q -p no:cacheprovider
```

Expected, with Phase 1: `78 passed` (main's 57, Phase 1's 13 and the 8 appended here), or the file's count before this task plus 8 if Phases 2–4 added tests. Without Phase 1 the file reports `1 failed, 64 passed`, the failure being the early-close test's axis assertion. The neighbours report `49 passed` on main@17c552c5 with or without Phase 1, including `test_live_leaderboard_api_serves_the_calendar_month_freeze`, which reaches this path through the route.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/leaderboard/live.py dashboard/backend/tests/test_live_leaderboard.py
git commit -m "feat(live): publish daily Sharpe with its 95% interval"
```

### Task 5.3: Live table shows Sharpe with its interval; the browser never computes it

**Files:**
- Modify: `dashboard/frontend/js/live-leaderboard.js`
- Modify: `dashboard/frontend/app.html`
- Test: `dashboard/backend/tests/test_frontend_live_leaderboard.py`

**Interfaces:**
- Consumes: the entry keys and `live_status.sharpe` from Task 5.2; `escapeHtml` (app.js global); `fn_body`, `run_node` (`dashboard/backend/tests/_frontend_source.py`); `SHARPE_MIN_DAYS`, `daily_sharpe_basis` (Task 5.1).
- Produces: `liveFormatSharpe(entry, basis) -> {main, sub, title}`. It renders as follows:
  - status `ok`: `main` is the value to 2 dp and `sub` is `95% interval lo to hi`. `title` is the tooltip sentence of owner decision 5 in the conformance suite design (`docs/superpowers/specs/2026-10-07-us-conformance-suite-design.md`, "Owner decisions"), with N = `sharpe_days` and rf = `basis.rf_annual` × 100.
  - status `insufficient_days`: `—` with `Needs {min_days} trading days (have N)`.
  - status `no_variation`: `—` with `Returns never moved`.
  - anything else: `—`.
- Also produces: a null Sharpe sorts last in both directions.

Display rules:
- The sub-line reuses the existing `.leaderboard-sample-range` class, so `styles.css` is untouched.
- The header keeps the short label "Sharpe" in this 9-column table. The detail row uses the owner's label "Sharpe ratio".
- The header tooltip is the static form of that owner decision 5. The per-cell `title` carries the full sentence with that entry's N and the payload's cash rate.

- [ ] **Step 1: Write the failing test**

These tests load the whole shipped `live-leaderboard.js` under node: it has no top-level side effects beyond two `window.*` assignments. If an earlier phase made `populateLiveTable` or `renderLiveDetail` call a helper defined in `app.js`, node reports a `ReferenceError` naming it. Add `fn_body("function <thatName>(")` to `_render_live`'s script list next to `escapeHtml`.

Append to the end of `dashboard/backend/tests/test_frontend_live_leaderboard.py`:

```python


# --- Sharpe: displayed, never recomputed (D20), always with its interval -----

from dashboard.backend.domain.backtesting.metrics import (  # noqa: E402
    SHARPE_MIN_DAYS,
    daily_sharpe_basis,
)
from dashboard.backend.tests._frontend_source import fn_body, run_node  # noqa: E402

_SHARPE_BASIS = daily_sharpe_basis()
_SHARPE_OK = {
    "sharpe_status": "ok",
    "sharpe_ratio": 2.611715,
    "sharpe_ci_low": -11.39649,
    "sharpe_ci_high": 16.619919,
    "sharpe_days": 5,
}
_DECISION_5_TITLE = (
    "Return above what cash would have earned, divided by how much the daily "
    "returns swung. Based on 5 trading days; cash rate 0%/yr; scaled to a yearly figure."
)


def _format_sharpe(entry, basis=_SHARPE_BASIS):
    script = fn_body("function liveFormatSharpe(", _LIVE_JS) + (
        f"\nconsole.log(JSON.stringify(liveFormatSharpe({json.dumps(entry)}, {json.dumps(basis)})));"
    )
    return run_node(script)


def _sharpe_row(entry_id, **fields):
    return {
        "entry_id": entry_id, "model": entry_id, "team_badge": "Model", "is_model": True,
        "status": "frozen", "portfolio_value": 100000, "cumulative_return": 0.01,
        "max_drawdown": -0.01, "num_trades": 1, "equity_curve": [], **fields,
    }


def _render_live(entries, *, selected=None):
    """Load the whole shipped file under node, then paint the table and detail.

    Whole-file rather than lifted functions, so a helper another change adds
    to populateLiveTable cannot turn this into a ReferenceError. If node
    reports one for a helper defined in app.js, add its `fn_body(...)` here.
    """
    script = "\n".join([
        "const window = {};",
        fn_body("function escapeHtml("),
        _LIVE_JS,
        "const tbody = { innerHTML: '' }; const detail = { innerHTML: '' };",
        "const document = {",
        "  getElementById: (id) => (id === 'liveSelectedDetail' ? detail : tbody),",
        "  querySelectorAll: () => [],",
        "};",
        f"livePayload = {{ entries: {json.dumps(entries)}, live_status: {{ sharpe: {json.dumps(_SHARPE_BASIS)} }} }};",
        "liveSortKey = 'sharpe'; liveSortDir = 'desc';",
        "populateLiveTable();",
        f"renderLiveDetail(livePayload.entries.find((e) => e.entry_id === {json.dumps(selected)}) || null);",
        "console.log(JSON.stringify({ table: tbody.innerHTML, detail: detail.innerHTML,"
        " order: liveSortedEntries().map((e) => e.entry_id) }));",
    ])
    return run_node(script)


def test_live_js_never_recomputes_sharpe():
    source = _strip_js_comments(_LIVE_JS)
    assert "Math.sqrt" not in source
    assert "stdev" not in source.lower()
    assert "Number(entry.sharpe_ratio).toFixed" not in source


@_needs_node
def test_format_sharpe_shows_the_value_with_its_interval():
    out = _format_sharpe(_SHARPE_OK)
    assert out == {
        "main": "2.61",
        "sub": "95% interval -11.40 to 16.62",
        "title": _DECISION_5_TITLE,
    }


@_needs_node
def test_format_sharpe_tooltip_reads_the_payloads_cash_rate():
    out = _format_sharpe(_SHARPE_OK, {**_SHARPE_BASIS, "rf_annual": 0.0252})
    assert "cash rate 2.52%/yr" in out["title"]


@_needs_node
def test_format_sharpe_explains_every_dash():
    short = _format_sharpe({"sharpe_status": "insufficient_days", "sharpe_ratio": None, "sharpe_days": 3})
    assert short == {"main": "—", "sub": f"Needs {SHARPE_MIN_DAYS} trading days (have 3)", "title": ""}
    flat = _format_sharpe({"sharpe_status": "no_variation", "sharpe_ratio": None, "sharpe_days": 6})
    assert flat == {"main": "—", "sub": "Returns never moved", "title": ""}
    pending = _format_sharpe({"sharpe_status": "pending", "sharpe_ratio": None, "sharpe_days": 0})
    assert pending == {"main": "—", "sub": "", "title": ""}
    # An ok status with a missing bound is malformed: no number without its interval.
    torn = _format_sharpe({**_SHARPE_OK, "sharpe_ci_high": None})
    assert torn["main"] == "—"
    # A payload from before the basis existed still says what is missing.
    no_basis = _format_sharpe({"sharpe_status": "insufficient_days", "sharpe_days": 2}, None)
    assert no_basis["sub"] == "Needs more trading days (have 2)"


@_needs_node
def test_live_table_and_detail_render_the_interval_and_sort_null_last():
    out = _render_live(
        [
            _sharpe_row("up", **_SHARPE_OK),
            _sharpe_row("young", sharpe_status="insufficient_days", sharpe_ratio=None, sharpe_days=3),
            _sharpe_row("down", sharpe_status="ok", sharpe_ratio=-0.5, sharpe_ci_low=-14.2,
                        sharpe_ci_high=13.2, sharpe_days=5),
        ],
        selected="up",
    )
    assert out["order"] == ["up", "down", "young"]
    assert "95% interval -11.40 to 16.62" in out["table"]
    assert f'title="{_DECISION_5_TITLE}"' in out["table"]
    assert "Needs 5 trading days (have 3)" in out["table"]
    assert "Sharpe ratio" in out["detail"]
    assert "2.61 · 95% interval -11.40 to 16.62" in out["detail"]


@_needs_node
def test_null_sharpe_sorts_last_in_both_directions():
    rows = [
        _sharpe_row("none", sharpe_status="insufficient_days", sharpe_ratio=None, sharpe_days=3),
        _sharpe_row("neg", sharpe_status="ok", sharpe_ratio=-0.5, sharpe_ci_low=-14.0,
                    sharpe_ci_high=13.0, sharpe_days=5),
        _sharpe_row("pos", **_SHARPE_OK),
    ]
    script = "\n".join([
        "const window = {};",
        _LIVE_JS,
        f"livePayload = {{ entries: {json.dumps(rows)} }}; liveSortKey = 'sharpe';",
        "liveSortDir = 'desc'; const desc = liveSortedEntries().map((e) => e.entry_id);",
        "liveSortDir = 'asc'; const asc = liveSortedEntries().map((e) => e.entry_id);",
        "console.log(JSON.stringify({ desc, asc }));",
    ])
    out = run_node(script)
    assert out == {"desc": ["pos", "neg", "none"], "asc": ["neg", "pos", "none"]}


def test_live_sharpe_header_states_its_basis_in_plain_words():
    live_html = _APP_HTML.split('id="liveLeaderboardView"')[1].split('id="leaderboardView"')[0]
    match = re.search(r'<th[^>]*data-sort="sharpe"[^>]*title="([^"]*)"', live_html)
    assert match, "the Live Sharpe header lost its tooltip"
    title = match.group(1)
    assert "Return above what cash would have earned" in title
    assert "divided by how much the daily returns swung" in title
    assert "scaled to a yearly figure" in title
    assert f"Shown from {SHARPE_MIN_DAYS} trading days" in title
    assert "95% interval" in title
    for jargon in ("T-bill", "excess", "√252", "sqrt", "sample SD"):
        assert jargon not in title
    assert "noisy until a few weeks" not in _APP_HTML
```


- [ ] **Step 2: Run it and confirm it fails**

```bash
pytest dashboard/backend/tests/test_frontend_live_leaderboard.py -q -p no:cacheprovider
```

Expected: `7 failed, 11 passed` (node v24 is on PATH here; without node only the two source guards run and fail):
- The three `test_format_sharpe_*` tests fail with `ValueError: substring not found`, because `fn_body` cannot find `function liveFormatSharpe(`.
- `test_live_js_never_recomputes_sharpe` fails with `assert 'Number(entr...tio).toFixed' not in ...`.
- `test_live_table_and_detail_render_the_interval_and_sort_null_last` fails with `assert ['up', 'young', 'down'] == ['up', 'down', 'young']`.
- `test_null_sharpe_sorts_last_in_both_directions` fails because `Number(null)` sorts as 0: `{'desc': ['pos', 'none', 'neg']} != {'desc': ['pos', 'neg', 'none']}` (and `asc` likewise).
- `test_live_sharpe_header_states_its_basis_in_plain_words` fails with `assert 'Return above what cash would have earned' in 'Annualized; noisy until a few weeks of sessions'`.

- [ ] **Step 3: Implement**

The last edit is the cache buster. Read the current value with `grep -o 'js/live-leaderboard.js?v=[0-9]*' dashboard/frontend/app.html`. It prints `?v=6` once Phase 3 has merged (bump to 7, the form given below), or `?v=5` on main@17c552c5 alone (bump to 6, the alternative given below). If it prints any other `N`, use `N+1`. Nothing pins this one exactly; `test_live_script_is_cache_busted` only asserts `>= 1`.

In `dashboard/frontend/js/live-leaderboard.js` (≈:261 on main@17c552c5, at the signature of `liveSortValue`), find:

```js
function liveSortValue(entry, key) {
  switch (key) {
    case 'return': return Number(entry.cumulative_return);
    case 'sharpe': return Number(entry.sharpe_ratio);
```

Replace with:

```js
// Sharpe is computed once, on the server, from the stored curve (P19); this
// file only displays it (D20) -- never derive it from equity_curve here. The
// interval is always shown with the number: at five daily returns it is about
// +/-14, and showing that is the point. `basis` is `live_status.sharpe`.
// leaderboard.js's formatBoardSharpe must answer identically.
function liveFormatSharpe(entry, basis) {
  const read = (v) => (v === null || v === undefined || v === '' ? NaN : Number(v));
  const status = entry ? entry.sharpe_status : null;
  const value = read(entry && entry.sharpe_ratio);
  const lo = read(entry && entry.sharpe_ci_low);
  const hi = read(entry && entry.sharpe_ci_high);
  const days = read(entry && entry.sharpe_days);
  const have = Number.isFinite(days) ? days : 0;
  const need = read(basis && basis.min_days);
  const rf = read(basis && basis.rf_annual);
  if (status === 'ok' && Number.isFinite(value) && Number.isFinite(lo) && Number.isFinite(hi)) {
    const title = Number.isFinite(rf)
      ? 'Return above what cash would have earned, divided by how much the daily returns swung. '
        + `Based on ${have} trading days; cash rate ${Number((rf * 100).toFixed(2))}%/yr; scaled to a yearly figure.`
      : '';
    return { main: value.toFixed(2), sub: `95% interval ${lo.toFixed(2)} to ${hi.toFixed(2)}`, title };
  }
  if (status === 'insufficient_days') {
    const sub = Number.isFinite(need)
      ? `Needs ${need} trading days (have ${have})`
      : `Needs more trading days (have ${have})`;
    return { main: '—', sub, title: '' };
  }
  if (status === 'no_variation') return { main: '—', sub: 'Returns never moved', title: '' };
  return { main: '—', sub: '', title: '' };
}

function liveSortValue(entry, key) {
  switch (key) {
    case 'return': return Number(entry.cumulative_return);
    // null is "no figure", not 0.00: NaN sorts below every number.
    case 'sharpe': return entry.sharpe_ratio == null ? NaN : Number(entry.sharpe_ratio);
```

In `dashboard/frontend/js/live-leaderboard.js` (≈:299 on main@17c552c5, inside `populateLiveTable`), find:

```js
  const rows = liveSortedEntries();
  if (!rows.length) {
```

Replace with:

```js
  const rows = liveSortedEntries();
  const sharpeBasis = livePayload?.live_status?.sharpe || null;
  if (!rows.length) {
```

In `dashboard/frontend/js/live-leaderboard.js` (≈:312 on main@17c552c5, inside `populateLiveTable`), find:

```js
    const dash = (html) => (printed ? html : '—');
```

Replace with:

```js
    const dash = (html) => (printed ? html : '—');
    const sharpe = liveFormatSharpe(entry, sharpeBasis);
    const sharpeSub = sharpe.sub
      ? `<span class="leaderboard-sample-range">${escapeHtml(sharpe.sub)}</span>`
      : '';
```

In `dashboard/frontend/js/live-leaderboard.js` (≈:322 on main@17c552c5, inside `populateLiveTable`), find:

```js
        <td class="num-cell">${dash(entry.sharpe_ratio == null ? '—' : Number(entry.sharpe_ratio).toFixed(2))}</td>
```

Replace with:

```js
        <td class="num-cell"${sharpe.title ? ` title="${escapeHtml(sharpe.title)}"` : ''}>${dash(`${escapeHtml(sharpe.main)}${sharpeSub}`)}</td>
```

In `dashboard/frontend/js/live-leaderboard.js` (≈:348 on main@17c552c5, inside `renderLiveDetail`), find:

```js
  const ret = Number(entry.cumulative_return || 0);
  const retColor = ret >= 0 ? 'var(--success-color)' : 'var(--danger-color)';
  host.innerHTML = `
```

Replace with:

```js
  const ret = Number(entry.cumulative_return || 0);
  const retColor = ret >= 0 ? 'var(--success-color)' : 'var(--danger-color)';
  const sharpe = liveFormatSharpe(entry, livePayload?.live_status?.sharpe);
  const sharpeText = hasPrints ? [sharpe.main, sharpe.sub].filter(Boolean).join(' · ') : '—';
  host.innerHTML = `
```

In `dashboard/frontend/js/live-leaderboard.js` (≈:355 on main@17c552c5, inside `renderLiveDetail`), find:

```js
    <div class="team-detail-row"><span class="team-detail-label">Trades</span>
```

Replace with:

```js
    <div class="team-detail-row"><span class="team-detail-label">Sharpe ratio</span><span class="team-detail-value"${sharpe.title ? ` title="${escapeHtml(sharpe.title)}"` : ''}>${escapeHtml(sharpeText)}</span></div>
    <div class="team-detail-row"><span class="team-detail-label">Trades</span>
```

In `dashboard/frontend/app.html` (≈:1722 on main@17c552c5), find:

```html
<th class="num-col sortable" data-sort="sharpe" title="Annualized; noisy until a few weeks of sessions">Sharpe <span class="sort-arrow" aria-hidden="true"></span></th>
```

Replace with:

```html
<th class="num-col sortable" data-sort="sharpe" title="Sharpe ratio. Return above what cash would have earned, divided by how much the daily returns swung, scaled to a yearly figure. Shown from 5 trading days, always with its 95% interval.">Sharpe <span class="sort-arrow" aria-hidden="true"></span></th>
```

In `dashboard/frontend/app.html` (≈:2523 on main@17c552c5), with Phase 3 merged, find:

```html
<script src="js/live-leaderboard.js?v=6" defer></script>
```

Replace with:

```html
<script src="js/live-leaderboard.js?v=7" defer></script>
```

On main@17c552c5 alone the tag reads `<script src="js/live-leaderboard.js?v=5" defer></script>`; replace it with `<script src="js/live-leaderboard.js?v=6" defer></script>` instead.


- [ ] **Step 4: Run it and confirm it passes**

```bash
node --check dashboard/frontend/js/live-leaderboard.js
pytest dashboard/backend/tests/test_frontend_live_leaderboard.py -q -p no:cacheprovider
pytest dashboard/backend/tests/test_frontend_live_trading_board.py dashboard/backend/tests/test_frontend_xss_guards.py dashboard/backend/tests/test_frontend_fast_boot.py -q -p no:cacheprovider
```

Expected: `node --check` is silent, and the Live file reports `18 passed` (main's 11 plus 7). The neighbours report `56 passed` on main@17c552c5; `styles.css` and `app.js` are untouched, so their `?v=159` pins hold.

- [ ] **Step 5: Commit**

```bash
git add dashboard/frontend/js/live-leaderboard.js dashboard/frontend/app.html dashboard/backend/tests/test_frontend_live_leaderboard.py
git commit -m "feat(live): show Sharpe with its 95% interval, never recompute it"
```

### Task 5.4: Competition payload carries the same fields, read off the scaled hourly curve

**Files:**
- Modify: `dashboard/backend/domain/leaderboard/service.py`
- Test: `dashboard/backend/tests/test_leaderboard_api.py`
- Test: `dashboard/backend/tests/domain/leaderboard/test_service_move.py`
- Modify: `dashboard/backend/tests/test_landing_live_board.py` (docstring only)

**Interfaces:**
- Consumes: `daily_sharpe_interval`, `daily_sharpe_basis` (Task 5.1); `_scaled_hourly(equity_hourly, scale)` and `display_capital` inside `get_leaderboard`'s second pass.
- Produces: the five entry keys on every `get_leaderboard` entry, and the top-level `sharpe_basis`. `run.get("sharpe_ratio") or 0` is gone.

Feed `scaled_hourly`, never `equity_curve`. `chart_equity_curve` prepends a synthetic `{start_date}T00:00:00+00:00` tick, which falls on the previous Eastern day, so the board would gain a session that never traded.

E_0 is `display_capital`, and that is an approximation on this board. A Competition row's stored seed (`initial_equity`) is its curve's first mark, so the scaled curve opens at `display_capital` and day 1's return leaves out the first hour's P&L. Live keeps that hour. The code comment records this, and the Competition board is never re-run (spec §6).

On main, `test_service_move.py::test_get_leaderboard_schema_and_ranking` pins the exact top-level key set and the entry fields, so it has to learn the new keys. The `_seed_leaderboard_runs` fixture's DJIA run gains five daily closes on M03's equities. SPY keeps its two-point curve as the `insufficient_days` case.

One docstring in `test_landing_live_board.py` says the Competition payload sends `run.get("sharpe_ratio") or 0`. This task makes that false, so the docstring is corrected in the same commit. The landing's own code and copy are Phase 7's.

- [ ] **Step 1: Write the failing test**

In `dashboard/backend/tests/test_leaderboard_api.py` (≈:36 on main@17c552c5, inside `_seed_leaderboard_runs`), find:

```python
        initial_equity=100000,
        final_equity=105000,
        total_return=0.05,
        sharpe_ratio=1.2,
```

Replace with:

```python
        initial_equity=100000,
        final_equity=102762.45,
        total_return=0.0276245,
        sharpe_ratio=1.2,
```

In `dashboard/backend/tests/test_leaderboard_api.py` (≈:44 on main@17c552c5, inside `_seed_leaderboard_runs`), find:

```python
    db.insert_equity_points(
        "lb_djia_index_20260415_20260515",
        [
            {"timestamp": "2026-04-15T14:00:00", "equity": 100000, "cash": 0, "positions_value": 100000},
            {"timestamp": "2026-05-15T20:00:00", "equity": 105000, "cash": 0, "positions_value": 105000},
        ],
    )
```

Replace with:

```python
    # Five sessions closing on conformance case M03's equities (r = 0, 0.02,
    # -0.05, 0.05, 0.01), so the board's daily Sharpe is a hand-checked number.
    # Naive stamps are UTC: 20:00Z is the 16:00 ET close in April.
    db.insert_equity_points(
        "lb_djia_index_20260415_20260515",
        [
            {"timestamp": "2026-04-15T14:00:00", "equity": 100000, "cash": 0, "positions_value": 100000},
            {"timestamp": "2026-04-15T20:00:00", "equity": 100000, "cash": 0, "positions_value": 100000},
            {"timestamp": "2026-04-16T20:00:00", "equity": 102000, "cash": 0, "positions_value": 102000},
            {"timestamp": "2026-04-17T20:00:00", "equity": 96900, "cash": 0, "positions_value": 96900},
            {"timestamp": "2026-04-20T20:00:00", "equity": 101745, "cash": 0, "positions_value": 101745},
            {"timestamp": "2026-04-21T20:00:00", "equity": 102762.45, "cash": 0, "positions_value": 102762.45},
        ],
    )
```

Append to the end of `dashboard/backend/tests/test_leaderboard_api.py`:

```python


def _contest_board_payload(client, monkeypatch):
    _seed_leaderboard_runs(lb_service.db)
    monkeypatch.setattr(
        lb_service,
        "ensure_leaderboard_runs",
        lambda force_refresh=False, period="contest", config=None: {
            "session_id": "leaderboard-contest",
            "created": 0,
            "refreshed_at": "2026-06-18T00:00:00+00:00",
        },
    )
    resp = client.get("/api/v1/leaderboard")
    assert resp.status_code == 200
    return resp.json()


def test_competition_sharpe_is_daily_with_its_interval(client, monkeypatch):
    """Read off the stored curve, not the stored column (1.2 here)."""
    body = _contest_board_payload(client, monkeypatch)
    djia = next(e for e in body["entries"] if e["entry_id"] == "djia_index")
    assert djia["sharpe_ratio"] == pytest.approx(2.611715, abs=1e-6)
    assert djia["sharpe_ci_low"] == pytest.approx(-11.396490, abs=1e-6)
    assert djia["sharpe_ci_high"] == pytest.approx(16.619919, abs=1e-6)
    assert djia["sharpe_days"] == 5
    assert djia["sharpe_status"] == "ok"


def test_competition_sharpe_is_null_below_five_days_not_zero(client, monkeypatch):
    """SPY's fixture curve spans two sessions. `or 0` used to publish the
    stored 0.9 -- and an absent value as a real 0.00."""
    body = _contest_board_payload(client, monkeypatch)
    spy = next(e for e in body["entries"] if e["entry_id"] == "spy_index")
    assert spy["sharpe_ratio"] is None
    assert spy["sharpe_ci_low"] is None and spy["sharpe_ci_high"] is None
    assert spy["sharpe_days"] == 2
    assert spy["sharpe_status"] == "insufficient_days"


def test_competition_payload_states_its_sharpe_basis(client, monkeypatch):
    from dashboard.backend.domain.backtesting.metrics import daily_sharpe_basis

    body = _contest_board_payload(client, monkeypatch)
    assert body["sharpe_basis"] == {
        "basis": "daily_returns",
        "annualization_days": 252,
        "confidence": 0.95,
        "min_days": 5,
        "rf_annual": 0.0,
        "stderr": "iid",
    }
    assert body["sharpe_basis"] == daily_sharpe_basis()


def test_competition_sharpe_never_reads_the_stored_column():
    from pathlib import Path

    source = Path(lb_service.__file__).read_text(encoding="utf-8")
    assert 'run.get("sharpe_ratio") or 0' not in source
    assert "daily_sharpe_interval(scaled_hourly, display_capital)" in source
    assert "daily_sharpe_interval(equity_curve" not in source
```

In `dashboard/backend/tests/domain/leaderboard/test_service_move.py` (≈:196 on main@17c552c5, inside `test_get_leaderboard_schema_and_ranking`), find:

```python
        "window", "updated_at", "total_entries", "leader", "entries", "display_capital",
```

Replace with:

```python
        "window", "updated_at", "total_entries", "leader", "entries", "display_capital",
        # what every entry's sharpe_* fields mean (P19 daily Sharpe)
        "sharpe_basis",
```

In `dashboard/backend/tests/domain/leaderboard/test_service_move.py` (≈:218 on main@17c552c5, inside `test_get_leaderboard_schema_and_ranking`), find:

```python
        "output_tokens", "est_cost_usd", "equity_curve", "rank",
```

Replace with:

```python
        "output_tokens", "est_cost_usd", "equity_curve", "rank",
        "sharpe_ci_low", "sharpe_ci_high", "sharpe_days", "sharpe_status",
```

In `dashboard/backend/tests/test_landing_live_board.py` (≈:480 on main@17c552c5, inside `test_standings_carry_the_ending_value_and_sharpe_the_table_renders`), find:

```python
    Sharpe's em dash is NOT reachable through the live API today -- the payload
    builder sends `run.get("sharpe_ratio") or 0`, which collapses absent onto a
    real 0.00 before it leaves the server. It is pinned anyway: the client rule
    is the one that survives that being fixed, and an untested branch is how the
    fix would arrive with the display still reading "0.00"."""
```

Replace with:

```python
    Sharpe's em dash IS reachable through the API now: the payload sends
    `sharpe_ratio: null` below five trading days or when the returns never
    moved (`sharpe_status` says which). It used to send
    `run.get("sharpe_ratio") or 0`, publishing absent as a real 0.00. The
    landing does not render the 95% interval (`sharpe_ci_low`/`sharpe_ci_high`)
    yet; that is the landing's own change."""
```


- [ ] **Step 2: Run it and confirm it fails**

```bash
pytest dashboard/backend/tests/test_leaderboard_api.py dashboard/backend/tests/domain/leaderboard/test_service_move.py -q -p no:cacheprovider
```

Expected: `5 failed, 37 passed`:
- `test_competition_sharpe_is_daily_with_its_interval`: `assert 1.2 == 2.611715 ± 1.0e-06`.
- `test_competition_sharpe_is_null_below_five_days_not_zero`: `assert 0.9 is None`.
- `test_competition_payload_states_its_sharpe_basis`: `KeyError: 'sharpe_basis'`.
- `test_competition_sharpe_never_reads_the_stored_column`: the source still contains `run.get("sharpe_ratio") or 0`.
- `test_get_leaderboard_schema_and_ranking`: `Extra items in the right set: 'sharpe_basis'`.

`test_leaderboard_api_returns_baselines` stays green with the new fixture, because SPY now outranks DJIA and the test asserts only rank 1's existence.

- [ ] **Step 3: Implement**

In `dashboard/backend/domain/leaderboard/service.py` (≈:26 on main@17c552c5, module level), find:

```python
import dashboard.backend.domain.leaderboard.baselines as _baselines
```

Replace with:

```python
from dashboard.backend.domain.backtesting.metrics import (
    daily_sharpe_basis,
    daily_sharpe_interval,
)
import dashboard.backend.domain.leaderboard.baselines as _baselines
```

In `dashboard/backend/domain/leaderboard/service.py` (≈:2877 on main@17c552c5, inside `get_leaderboard`), find:

```python
        # just its dollar levels. Returns / Sharpe below come off the stored run
        # untouched, so the shim moves the dollar axis only. Re-running each
```

Replace with:

```python
        # just its dollar levels. Returns below come off the stored run
        # untouched and Sharpe is a ratio of the scaled curve's own returns, so
        # the shim moves the dollar axis only. Re-running each
```

In `dashboard/backend/domain/leaderboard/service.py` (≈:2896 on main@17c552c5, inside `get_leaderboard`), find:

```python
        scaled_hourly = _scaled_hourly(equity_hourly, scale)
        _report_curve_integrity(strategy["id"], run["run_id"], scaled_hourly)
```

Replace with:

```python
        scaled_hourly = _scaled_hourly(equity_hourly, scale)
        _report_curve_integrity(strategy["id"], run["run_id"], scaled_hourly)
        # P19 daily Sharpe with its 95% interval, computed here at read time
        # rather than read from `agent_runs.sharpe_ratio` (hourly marks, ddof 0,
        # 0 for "undefined"). Fed the scaled hourly curve, NOT `equity_curve`
        # below: `chart_equity_curve` prepends a synthetic
        # `{start_date}T00:00:00+00:00` tick, which is the previous US/Eastern
        # day and would add a session that never traded.
        #
        # ⚠ E_0 is an approximation on this board. The stored seed
        # (`initial_equity`) is the curve's first mark -- seed plus one hour of
        # P&L -- so the scaled curve opens AT display_capital and day 1's return
        # leaves out the first hour. Live keeps that hour (its rows record the
        # true seed in metadata). One hour of a multi-week window; the
        # Competition board is never re-run (spec 2026-10-14 §6).
        sharpe = daily_sharpe_interval(scaled_hourly, display_capital)
```

In `dashboard/backend/domain/leaderboard/service.py` (≈:2928 on main@17c552c5, inside `get_leaderboard`), find:

```python
                "sharpe_ratio": run.get("sharpe_ratio") or 0,
```

Replace with:

```python
                "sharpe_ratio": sharpe["sharpe"],
                "sharpe_ci_low": sharpe["ci_low"],
                "sharpe_ci_high": sharpe["ci_high"],
                "sharpe_days": sharpe["days"],
                "sharpe_status": sharpe["status"],
```

In `dashboard/backend/domain/leaderboard/service.py` (≈:3008 on main@17c552c5, inside `get_leaderboard`), find:

```python
        "display_capital": display_capital,
```

Replace with:

```python
        "display_capital": display_capital,
        # What every entry's sharpe_* fields mean; the table quotes these.
        "sharpe_basis": daily_sharpe_basis(),
```


- [ ] **Step 4: Run it and confirm it passes**

```bash
pytest dashboard/backend/tests/test_leaderboard_api.py dashboard/backend/tests/domain/leaderboard/test_service_move.py -q -p no:cacheprovider
pytest dashboard/backend/tests/test_leaderboard_samples.py dashboard/backend/tests/test_leaderboard_curve_integrity.py dashboard/backend/tests/test_leaderboard_season.py dashboard/backend/tests/domain/leaderboard dashboard/backend/tests/test_architecture_boundaries.py dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/test_landing_live_board.py -q -p no:cacheprovider
```

Expected: `42 passed` for the first command. With Phase 1, the second reports `398 passed, 34 skipped` on main@17c552c5 plus Phase 1 (no failures). Without Phase 1 it reports `1 failed, 378 passed, 34 skipped`, and the failure is Task 5.2's `test_live_sharpe_counts_an_early_close_once`.
- `test_architecture_boundaries.py` covers the new import: `domain/leaderboard` → `domain/backtesting.metrics` is an edge that already exists (`baselines.py`), and `metrics` imports only numpy and the stdlib.
- `tests/domain/leaderboard/test_deploy_guard.py` mocks `calc_metrics` and stays green, because `calc_metrics` is untouched.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/leaderboard/service.py dashboard/backend/tests/test_leaderboard_api.py dashboard/backend/tests/domain/leaderboard/test_service_move.py dashboard/backend/tests/test_landing_live_board.py
git commit -m "feat(leaderboard): daily Sharpe and its interval on the Competition board"
```

### Task 5.5: Competition table, detail and home rank list render Sharpe with its interval

**Files:**
- Modify: `dashboard/frontend/js/leaderboard.js`
- Modify: `dashboard/frontend/home-page.js`
- Modify: `dashboard/frontend/app.html`
- Create: `dashboard/backend/tests/test_frontend_board_sharpe.py`
- Test: `dashboard/backend/tests/test_frontend_xss_guards.py` (its driver lifts the row and detail renderers, which now call the formatter)
- Test: `dashboard/backend/tests/test_app_copy_register.py` (pins the old home tooltip)
- Test: `dashboard/backend/tests/test_frontend_fast_boot.py` (the single owner of the exact `leaderboard.js` and `home-page.js` cache-buster pins, :200–201)

**Interfaces:**
- Consumes:
  - the entry keys and `sharpe_basis` from Task 5.4;
  - `finiteNumber`, `escapeHtml`, `formatLeaderboardSamples`;
  - `liveFormatSharpe` (Task 5.3), for the parity test;
  - `strip_comments`, `fn_body`, `run_node`, `APP_HTML`, `FRONTEND` (`_frontend_source.py`).
- Produces:
  - `boardSharpeBasis()`, `formatBoardSharpe(entry, basis)` and `window.formatBoardSharpe` (leaderboard.js);
  - `homeFormatSharpe(entry, basis)` and `renderEntries(..., { sharpeBasis })` (home-page.js);
  - one header tooltip text, shared by the Live and Competition tables.

Display rules:
- The Competition cell renders like the Return cell beside it: `.metric-value-text` plus a `.leaderboard-sample-range` sub-line. The detail row reads `2.61 · 95% interval -11.40 to 16.62`.
- A null value is `—`, never `0.00`, and still sorts last. `finiteNumber` already handled the sort, so `test_competition_sort_puts_a_null_sharpe_last_both_ways` passes before and after as a characterization pin.
- On the home rank list the 48px Sharpe column holds the number only. The rows already run at `padding-block: 3px` so that seven fit without the pager clipping (see the comment on `.home-module-rank-list li` in `styles.css`), which leaves no room for a second line. So the interval, or the reason there is no number, goes in the cell's `title`. `homeFormatSharpe` delegates to `formatBoardSharpe`, the same way `homeFormatReturnPct` delegates to `boardSignedPercent`.
- The home rank list's sample rows (`HOME_MOCK_LEADERBOARD`, shown when the board is unreachable or has no model yet) carry an invented `sharpe_ratio` and no `sharpe_status`, and they are rendered with no basis. Their Sharpe cell therefore reads `—` with no tooltip. That is deliberate: a number with no interval is exactly what D20 retires, and inventing an interval for an invented number is worse. Leave `HOME_MOCK_LEADERBOARD` untouched; its `sharpe_ratio` values become unused, and the `({"sharpe_ratio": 1.5}, _BASIS)` row of `_CASES` pins that answer.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_frontend_board_sharpe.py` with exactly:

```python
"""Competition board + home rank list: Sharpe is displayed, never recomputed.

The server publishes P19's daily Sharpe with its 95% interval
(``sharpe_ratio``/``sharpe_ci_low``/``sharpe_ci_high``/``sharpe_days``/
``sharpe_status`` per entry, ``sharpe_basis`` on the payload). These run the
shipped JS under node -- the whole file where it has no top-level side effects,
so a helper added later cannot turn the harness into a ReferenceError.
"""

import json
import re
import shutil

import pytest

from dashboard.backend.domain.backtesting.metrics import SHARPE_MIN_DAYS, daily_sharpe_basis
from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    FRONTEND,
    fn_body,
    run_node,
    strip_comments,
)

_LEADERBOARD_JS = (FRONTEND / "js" / "leaderboard.js").read_text(encoding="utf-8")
_LIVE_JS = (FRONTEND / "js" / "live-leaderboard.js").read_text(encoding="utf-8")
_HOME_JS = (FRONTEND / "home-page.js").read_text(encoding="utf-8")

_needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

_BASIS = daily_sharpe_basis()
_OK = {
    "sharpe_status": "ok",
    "sharpe_ratio": 2.611715,
    "sharpe_ci_low": -11.39649,
    "sharpe_ci_high": 16.619919,
    "sharpe_days": 5,
}
_TITLE = (
    "Return above what cash would have earned, divided by how much the daily "
    "returns swung. Based on 5 trading days; cash rate 0%/yr; scaled to a yearly figure."
)
_HEADER_TITLE = (
    "Sharpe ratio. Return above what cash would have earned, divided by how much the "
    f"daily returns swung, scaled to a yearly figure. Shown from {SHARPE_MIN_DAYS} "
    "trading days, always with its 95% interval."
)
# Every shape the payload can send, plus the malformed ones a stale or broken
# payload could: the two boards must answer each the same way.
_CASES = [
    (_OK, _BASIS),
    ({**_OK, "sharpe_ratio": -0.5, "sharpe_ci_low": -14.5, "sharpe_ci_high": 13.5}, _BASIS),
    (_OK, {**_BASIS, "rf_annual": 0.0252}),
    (_OK, None),
    ({**_OK, "sharpe_ci_high": None}, _BASIS),
    ({"sharpe_status": "insufficient_days", "sharpe_ratio": None, "sharpe_days": 3}, _BASIS),
    ({"sharpe_status": "insufficient_days", "sharpe_days": 2}, None),
    ({"sharpe_status": "no_variation", "sharpe_ratio": None, "sharpe_days": 6}, _BASIS),
    ({"sharpe_status": "pending", "sharpe_ratio": None, "sharpe_days": 0}, _BASIS),
    ({"sharpe_ratio": 1.5}, _BASIS),
    (None, _BASIS),
]


def _board_script(*lines):
    """app.js's escapeHtml, the whole of leaderboard.js, then ``lines``."""
    return "\n".join(["const window = {};", fn_body("function escapeHtml("), _LEADERBOARD_JS, *lines])


def _board_row(entry_id, **fields):
    return {
        "entry_id": entry_id, "team_name": entry_id, "model": entry_id, "team_badge": "Model",
        "is_model": True, "rank": 1, "portfolio_value": 100000, "cumulative_return": 0.01,
        "max_drawdown": -0.01, "equity_curve": [], **fields,
    }


def test_board_js_never_recomputes_sharpe():
    for name, source in (("leaderboard.js", _LEADERBOARD_JS), ("home-page.js", _HOME_JS)):
        code = strip_comments(source)
        assert "Math.sqrt" not in code, name
        assert "stdev" not in code.lower(), name
        assert "Number(entry.sharpe_ratio || 0)" not in code, name


@_needs_node
def test_competition_formatter_shows_the_value_with_its_interval():
    out = run_node(_board_script(
        f"console.log(JSON.stringify(formatBoardSharpe({json.dumps(_OK)}, {json.dumps(_BASIS)})));"
    ))
    assert out == {"main": "2.61", "sub": "95% interval -11.40 to 16.62", "title": _TITLE}


@_needs_node
def test_both_boards_word_sharpe_identically():
    """Two copies of one mapping (leaderboard.js, live-leaderboard.js) is how
    they drift; this holds them to one answer for every payload shape."""
    script = "\n".join([
        "const window = {};",
        _LEADERBOARD_JS,
        fn_body("function liveFormatSharpe(", _LIVE_JS),
        f"const cases = {json.dumps(_CASES)};",
        "console.log(JSON.stringify(cases.map(([e, b]) => [formatBoardSharpe(e, b), liveFormatSharpe(e, b)])));",
    ])
    pairs = run_node(script)
    for board, live in pairs:
        assert board == live
    mains = [board["main"] for board, _live in pairs]
    assert mains == ["2.61", "-0.50", "2.61", "2.61", "—", "—", "—", "—", "—", "—", "—"]


@_needs_node
def test_competition_row_and_detail_render_the_interval_and_never_zero():
    ok = _board_row("up", **_OK)
    young = _board_row("young", sharpe_status="insufficient_days", sharpe_ratio=None, sharpe_days=2)
    out = run_node(_board_script(
        f"leaderboardPayload = {{ sharpe_basis: {json.dumps(_BASIS)}, entries: [] }};",
        "console.log(JSON.stringify({",
        f"  okRow: renderLeaderboardRowHtml({json.dumps(ok)}),",
        f"  youngRow: renderLeaderboardRowHtml({json.dumps(young)}),",
        f"  okDetail: renderLeaderboardDetailHtml({json.dumps(ok)}, 12),",
        f"  youngDetail: renderLeaderboardDetailHtml({json.dumps(young)}, 12),",
        "}));",
    ))
    assert "2.61" in out["okRow"] and "95% interval -11.40 to 16.62" in out["okRow"]
    assert f'title="{_TITLE}"' in out["okRow"]
    assert "Needs 5 trading days (have 2)" in out["youngRow"]
    assert '<span class="metric-value-text">—</span>' in out["youngRow"]
    assert '<span class="metric-value-text">0.00</span>' not in out["youngRow"]
    assert "Sharpe ratio" in out["okDetail"]
    assert "2.61 · 95% interval -11.40 to 16.62" in out["okDetail"]
    assert "— · Needs 5 trading days (have 2)" in out["youngDetail"]


@_needs_node
def test_competition_sort_puts_a_null_sharpe_last_both_ways():
    rows = [
        _board_row("none", sharpe_status="insufficient_days", sharpe_ratio=None, sharpe_days=2),
        _board_row("neg", sharpe_status="ok", sharpe_ratio=-0.5, sharpe_ci_low=-14.0, sharpe_ci_high=13.0, sharpe_days=5),
        _board_row("pos", **_OK),
    ]
    out = run_node(_board_script(
        f"leaderboardPayload = {{ entries: {json.dumps(rows)} }}; currentLeaderboardSort = 'sharpe';",
        "currentLeaderboardSortDir = 'desc'; const desc = getFilteredLeaderboardEntries().map((e) => e.entry_id);",
        "currentLeaderboardSortDir = 'asc'; const asc = getFilteredLeaderboardEntries().map((e) => e.entry_id);",
        "console.log(JSON.stringify({ desc, asc }));",
    ))
    assert out == {"desc": ["pos", "neg", "none"], "asc": ["neg", "pos", "none"]}


@_needs_node
def test_home_rank_row_borrows_the_boards_wording():
    lifted = fn_body("function homeFormatSharpe(", _HOME_JS)
    young = {"sharpe_status": "insufficient_days", "sharpe_ratio": None, "sharpe_days": 2}
    out = run_node(_board_script(
        lifted,
        "console.log(JSON.stringify({",
        f"  ok: homeFormatSharpe({json.dumps(_OK)}, {json.dumps(_BASIS)}),",
        f"  young: homeFormatSharpe({json.dumps(young)}, {json.dumps(_BASIS)}),",
        "}));",
    ))
    assert out["ok"] == {"text": "2.61", "title": f"95% interval -11.40 to 16.62 · {_TITLE}"}
    assert out["young"] == {"text": "—", "title": "Needs 5 trading days (have 2)"}
    alone = run_node("\n".join([
        "const window = {};", lifted,
        f"console.log(JSON.stringify(homeFormatSharpe({json.dumps(_OK)}, null)));",
    ]))
    assert alone == {"text": "—", "title": ""}


def test_home_rank_list_passes_the_basis_and_renders_the_cell():
    code = strip_comments(_HOME_JS)
    body = fn_body("function renderEntries(", code)
    assert "homeFormatSharpe(entry, sharpeBasis)" in body
    assert "sharpe.title" in body and "homeEscape(sharpe.text)" in body
    assert "renderEntries(models, { sharpeBasis: payload.sharpe_basis || null })" in code


def test_leaderboard_js_exports_the_formatter_for_home():
    assert "window.formatBoardSharpe = formatBoardSharpe;" in _LEADERBOARD_JS


def test_sharpe_headers_say_what_the_number_is():
    competition = APP_HTML.split('id="leaderboardTableBody"')[0]
    competition = competition[competition.rindex("<table") :]
    match = re.search(r'<th[^>]*data-sort="sharpe"[^>]*title="([^"]*)"', competition)
    assert match, "the Competition Sharpe header lost its tooltip"
    assert match.group(1) == _HEADER_TITLE
    live = APP_HTML.split('id="liveLeaderboardView"')[1].split('id="leaderboardView"')[0]
    live_title = re.search(r'<th[^>]*data-sort="sharpe"[^>]*title="([^"]*)"', live).group(1)
    assert live_title == _HEADER_TITLE, "one definition, worded once on both boards"
    assert "annualized from hourly results" not in APP_HTML
```

In `dashboard/backend/tests/test_frontend_xss_guards.py` (≈:199 on main@17c552c5, inside `_run_leaderboard_driver`), find:

```python
            _extract_function(src, "formatLeaderboardSamples"),
```

Replace with:

```python
            _extract_function(src, "formatLeaderboardSamples"),
            # The Sharpe cell and detail row (P19 daily Sharpe + interval).
            _extract_function(src, "boardSharpeBasis"),
            _extract_function(src, "formatBoardSharpe"),
```

In `dashboard/backend/tests/test_app_copy_register.py` (≈:288 on main@17c552c5, at the signature of `test_sharpe_tooltip_is_plain_language`), find:

```python
def test_sharpe_tooltip_is_plain_language():
    assert 'title="Risk-adjusted return, annualized from hourly results."' in _HTML
    assert "Annualized for hourly data (sqrt(252*6.5))" not in _HTML
```

Replace with:

```python
def test_sharpe_tooltip_is_plain_language():
    assert (
        'title="Sharpe ratio. Return above what cash would have earned, divided by how '
        'much the daily returns swung, scaled to a yearly figure. Hover a value for its '
        '95% interval."'
    ) in _HTML
    assert "annualized from hourly results" not in _HTML
    assert "Annualized for hourly data (sqrt(252*6.5))" not in _HTML
```

In `dashboard/backend/tests/test_frontend_fast_boot.py` (≈:200 on main@17c552c5, inside `test_cache_busters_bumped`), find:

```python
    assert "js/leaderboard.js?v=37" in APP_HTML
    assert "home-page.js?v=55" in APP_HTML
```

Replace with:

```python
    assert "js/leaderboard.js?v=38" in APP_HTML
    assert "home-page.js?v=56" in APP_HTML
```


- [ ] **Step 2: Run it and confirm it fails**

```bash
pytest dashboard/backend/tests/test_frontend_board_sharpe.py dashboard/backend/tests/test_frontend_xss_guards.py dashboard/backend/tests/test_app_copy_register.py dashboard/backend/tests/test_frontend_fast_boot.py -q -p no:cacheprovider
```

Expected: `13 failed, 62 passed`:
- 8 in `test_frontend_board_sharpe.py`. The node cases fail in one of three ways: an `AssertionError` raised by `run_node` (`_frontend_source.py`) whose text is node's stderr, `ReferenceError: formatBoardSharpe is not defined`; `ValueError: substring not found`, because `fn_body` cannot find `function homeFormatSharpe(`; or, for the row render, on the missing `95% interval` (the row still prints a bare `2.61`). The source cases fail on their `in` / `not in` assertions; the header case fails with `the Competition Sharpe header lost its tooltip`. The ninth test, `test_competition_sort_puts_a_null_sharpe_last_both_ways`, passes, which is expected.
- 3 in `test_frontend_xss_guards.py` (`test_leaderboard_row_escapes_every_string_field`, `..._onclick_round_trips_hostile_ids`, `test_leaderboard_detail_panel_escapes_string_fields`). They fail with `ValueError: substring not found`, because the driver cannot lift `boardSharpeBasis`.
- `test_sharpe_tooltip_is_plain_language` and `test_cache_busters_bumped`.

- [ ] **Step 3: Implement**

The last two `app.html` edits are cache busters. Check the current values with `grep -o 'js/leaderboard.js?v=[0-9]*\|home-page.js?v=[0-9]*' dashboard/frontend/app.html`. If an earlier phase moved either one off 37 / 55, bump from that `N` to `N+1`, and make the same change to the two pins in `test_frontend_fast_boot.py` from Step 1.

In `dashboard/frontend/js/leaderboard.js` (≈:1259 on main@17c552c5, inside `formatLeaderboardSamples`), find:

```js
  return long
    ? `${kind === 'median' ? 'Median' : 'Middle'} of ${count} runs${range}`
    : `${kind} of ${count}${range}`;
}
```

Replace with:

```js
  return long
    ? `${kind === 'median' ? 'Median' : 'Middle'} of ${count} runs${range}`
    : `${kind} of ${count}${range}`;
}

// The board's Sharpe definition (`sharpe_basis` on the payload), or null for a
// payload that predates it. `typeof` because the node harnesses lift single
// functions without this module's `let` state.
function boardSharpeBasis() {
  if (typeof leaderboardPayload === 'undefined' || !leaderboardPayload) return null;
  return leaderboardPayload.sharpe_basis || null;
}

// Sharpe is computed once, on the server, from the stored curve (P19); this
// file only displays it (D20) and never derives it from `equity_curve`. A
// number is never shown without its 95% interval. Returns `{ main, sub, title }`:
// the cell text, the muted second line (the interval, or why there is no
// number), and the plain-language tooltip. The Live board's `liveFormatSharpe`
// must answer identically -- test_frontend_board_sharpe.py pins the two.
function formatBoardSharpe(entry, basis) {
  const status = entry ? entry.sharpe_status : null;
  const value = finiteNumber(entry && entry.sharpe_ratio);
  const lo = finiteNumber(entry && entry.sharpe_ci_low);
  const hi = finiteNumber(entry && entry.sharpe_ci_high);
  const days = finiteNumber(entry && entry.sharpe_days);
  const have = Number.isFinite(days) ? days : 0;
  const need = finiteNumber(basis && basis.min_days);
  const rf = finiteNumber(basis && basis.rf_annual);
  if (status === 'ok' && Number.isFinite(value) && Number.isFinite(lo) && Number.isFinite(hi)) {
    const title = Number.isFinite(rf)
      ? 'Return above what cash would have earned, divided by how much the daily returns swung. '
        + `Based on ${have} trading days; cash rate ${Number((rf * 100).toFixed(2))}%/yr; scaled to a yearly figure.`
      : '';
    return { main: value.toFixed(2), sub: `95% interval ${lo.toFixed(2)} to ${hi.toFixed(2)}`, title };
  }
  if (status === 'insufficient_days') {
    const sub = Number.isFinite(need)
      ? `Needs ${need} trading days (have ${have})`
      : `Needs more trading days (have ${have})`;
    return { main: '—', sub, title: '' };
  }
  if (status === 'no_variation') return { main: '—', sub: 'Returns never moved', title: '' };
  return { main: '—', sub: '', title: '' };
}
```

In `dashboard/frontend/js/leaderboard.js` (≈:1278 on main@17c552c5, inside `renderLeaderboardRowHtml`), find:

```js
  const sampleLabel = formatLeaderboardSamples(entry);

  return `
      <tr onclick="selectLeaderboardTeam('${safeId}')">
```

Replace with:

```js
  const sampleLabel = formatLeaderboardSamples(entry);
  const sharpe = formatBoardSharpe(entry, boardSharpeBasis());

  return `
      <tr onclick="selectLeaderboardTeam('${safeId}')">
```

In `dashboard/frontend/js/leaderboard.js` (≈:1294 on main@17c552c5, inside `renderLeaderboardRowHtml`), find:

```js
        <td style="text-align: right; font-family: var(--font-mono);">${Number(entry.sharpe_ratio || 0).toFixed(2)}</td>
```

Replace with:

```js
        <td style="text-align: right; font-family: var(--font-mono);"${sharpe.title ? ` title="${escapeHtml(sharpe.title)}"` : ''}>
          <span class="metric-value-text">${escapeHtml(sharpe.main)}</span>
          ${sharpe.sub ? `<span class="leaderboard-sample-range">${escapeHtml(sharpe.sub)}</span>` : ''}
        </td>
```

In `dashboard/frontend/js/leaderboard.js` (≈:1325 on main@17c552c5, inside `renderLeaderboardDetailHtml`), find:

```js
  const runsLabel = formatLeaderboardSamples(entry, true);
```

Replace with:

```js
  const runsLabel = formatLeaderboardSamples(entry, true);
  const sharpe = formatBoardSharpe(entry, boardSharpeBasis());
```

In `dashboard/frontend/js/leaderboard.js` (≈:1348 on main@17c552c5, inside `renderLeaderboardDetailHtml`), find:

```js
        <span class="team-detail-label">Sharpe</span>
        <span class="team-detail-value">${Number(entry.sharpe_ratio || 0).toFixed(2)}</span>
```

Replace with:

```js
        <span class="team-detail-label">Sharpe ratio</span>
        <span class="team-detail-value"${sharpe.title ? ` title="${escapeHtml(sharpe.title)}"` : ''}>${escapeHtml([sharpe.main, sharpe.sub].filter(Boolean).join(' · '))}</span>
```

In `dashboard/frontend/js/leaderboard.js` (≈:2444 on main@17c552c5, module level), find:

```js
window.boardSignedPercent = boardSignedPercent;
```

Replace with:

```js
window.boardSignedPercent = boardSignedPercent;
window.formatBoardSharpe = formatBoardSharpe;
```

In `dashboard/frontend/home-page.js` (≈:966 on main@17c552c5, at the signature of `homeSparkPolyline`), find:

```js
function homeSparkPolyline(values, width = 52, height = 18) {
```

Replace with:

```js
/** A rank row's Sharpe cell, `{ text, title }`.
 *
 *  DELEGATES to leaderboard.js's `formatBoardSharpe`, so the rank row and the
 *  Competition table word one number the same way -- the reason
 *  `homeFormatReturnPct` delegates too. The server computes the number and its
 *  interval (P19); nothing here does (D20). The 48px column holds the number
 *  only, so the 95% interval -- or the reason there is no number -- is the
 *  cell's tooltip. Without leaderboard.js, a dash: there is no wording to
 *  borrow, and inventing one is the drift the delegation prevents. */
function homeFormatSharpe(entry, basis) {
    if (typeof window.formatBoardSharpe !== 'function') return { text: '—', title: '' };
    const fmt = window.formatBoardSharpe(entry, basis);
    return { text: fmt.main, title: [fmt.sub, fmt.title].filter(Boolean).join(' · ') };
}

function homeSparkPolyline(values, width = 52, height = 18) {
```

In `dashboard/frontend/home-page.js` (≈:2067 on main@17c552c5, at the signature of `renderEntries`), find:

```js
    function renderEntries(entries, { sample = null } = {}) {
```

Replace with:

```js
    function renderEntries(entries, { sample = null, sharpeBasis = null } = {}) {
```

In `dashboard/frontend/home-page.js` (≈:2087 on main@17c552c5, inside `renderEntries`), find:

```js
            const sharpe = Number(entry.sharpe_ratio || 0);
```

Replace with:

```js
            const sharpe = homeFormatSharpe(entry, sharpeBasis);
```

In `dashboard/frontend/home-page.js` (≈:2105 on main@17c552c5, inside `renderEntries`), find:

```js
                <span class="hm-rank-sharpe tabular-nums">${homeEscape(sharpe.toFixed(2))}</span>
```

Replace with:

```js
                <span class="hm-rank-sharpe tabular-nums"${sharpe.title ? ` title="${homeEscape(sharpe.title)}"` : ''}>${homeEscape(sharpe.text)}</span>
```

In `dashboard/frontend/home-page.js` (≈:2123 on main@17c552c5, inside `renderEntries`), find:

```js
        renderEntries(models);
```

Replace with:

```js
        renderEntries(models, { sharpeBasis: payload.sharpe_basis || null });
```

In `dashboard/frontend/app.html` (≈:570 on main@17c552c5), find:

```html
<span title="Risk-adjusted return, annualized from hourly results.">Sharpe</span>
```

Replace with:

```html
<span title="Sharpe ratio. Return above what cash would have earned, divided by how much the daily returns swung, scaled to a yearly figure. Hover a value for its 95% interval.">Sharpe</span>
```

In `dashboard/frontend/app.html` (≈:1865 on main@17c552c5), find:

```html
<th class="sharpe-col sortable" data-sort="sharpe">Sharpe <span class="sort-arrow" aria-hidden="true"></span></th>
```

Replace with:

```html
<th class="sharpe-col sortable" data-sort="sharpe" title="Sharpe ratio. Return above what cash would have earned, divided by how much the daily returns swung, scaled to a yearly figure. Shown from 5 trading days, always with its 95% interval.">Sharpe <span class="sort-arrow" aria-hidden="true"></span></th>
```

In `dashboard/frontend/app.html` (≈:2522 on main@17c552c5), find:

```html
<script src="js/leaderboard.js?v=37" defer></script>
```

Replace with:

```html
<script src="js/leaderboard.js?v=38" defer></script>
```

In `dashboard/frontend/app.html` (≈:2524 on main@17c552c5), find:

```html
<script src="home-page.js?v=55" defer></script>
```

Replace with:

```html
<script src="home-page.js?v=56" defer></script>
```


- [ ] **Step 4: Run it and confirm it passes**

```bash
node --check dashboard/frontend/js/leaderboard.js
node --check dashboard/frontend/home-page.js
pytest dashboard/backend/tests/test_frontend_board_sharpe.py dashboard/backend/tests/test_frontend_xss_guards.py dashboard/backend/tests/test_app_copy_register.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_frontend_live_leaderboard.py -q -p no:cacheprovider
pytest dashboard/backend/tests/test_frontend_chart_first_home.py dashboard/backend/tests/test_frontend_board_frame.py dashboard/backend/tests/test_frontend_leaderboard_hover.py dashboard/backend/tests/test_frontend_leaderboard_palette.py dashboard/backend/tests/test_leaderboard_band_frontend.py dashboard/backend/tests/test_leaderboard_samples.py dashboard/backend/tests/test_frontend_home_chart_height.py dashboard/backend/tests/test_frontend_home_get_started.py -q -p no:cacheprovider
```

Expected: both `node --check` runs are silent; the first pytest command reports `93 passed` and the second `192 passed` on main@17c552c5. `test_frontend_chart_first_home.py` extracts `renderEntries` by brace matching. Its first `)` is still the parameter list's closing paren, so the extraction holds.

- [ ] **Step 5: Commit**

```bash
git add dashboard/frontend/js/leaderboard.js dashboard/frontend/home-page.js dashboard/frontend/app.html dashboard/backend/tests/test_frontend_board_sharpe.py dashboard/backend/tests/test_frontend_xss_guards.py dashboard/backend/tests/test_app_copy_register.py dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "feat(leaderboard): Competition and home show Sharpe with its 95% interval"
```

### Task 5.6: Open the PR

**Files:** none changed.

**Interfaces:** Consumes Tasks 5.1–5.5. Produces the PR.

- [ ] **Step 1: Run the whole suite**

```bash
pytest dashboard/backend/tests -q -p no:cacheprovider
git status --short   # must print nothing: the suite must not have touched the committed seed DB
```

Expected: green apart from `test_report_pdf.py`, whose failures are environmental (no reportlab locally; CI is green). To get a clean run locally, add `--ignore=dashboard/backend/tests/test_report_pdf.py`. Do not add `--timeout`: `pytest-timeout` is not installed here and pytest rejects the flag (`unrecognized arguments: --timeout=0`). Measured on main@17c552c5 plus this phase alone, before Task 5.2 gained `test_live_sharpe_counts_an_early_close_once`, with that file ignored: `6728 passed, 176 skipped, 25 xfailed`, 0 failed, about 450 s. On this branch, with Phases 1–4 underneath it, the passed count is higher and the failed count is still 0. If the one failure is `test_live_sharpe_counts_an_early_close_once` with `Left contains 3 more items`, Phase 1 is missing from the base: stop. No `XPASS(strict)` in `tests/conformance`: this phase does not touch the engine path that suite scores, and its M03 case still reads the stored column through `adapters/current_engine.py`.

- [ ] **Step 2: Confirm the branch, push, open the PR as a draft**

The PR opens as a draft, with its merge gate in the body's first line, because Step 4's landing window must hold before anyone merges it.

```bash
git branch --show-current   # must print feat/board-daily-sharpe (a sibling session can switch it)
git push -u origin feat/board-daily-sharpe
gh pr create --draft --base main --head feat/board-daily-sharpe \
  --title "feat(leaderboard): daily Sharpe with its 95% interval" \
  --body "$(cat <<'EOF'
DO NOT MERGE until the owner has approved the landing copy (`docs/superpowers/plans/2026-10-14-live-board-fixes-p7-records.md`, Task 7.3 Step 0) and the copy PR (`feat(live): Live Leaderboard copy and catching-up badge`) and the landing PR (`feat(landing): Live Leaderboard copy and Sharpe interval`) can merge the same week. From this deploy until the landing PR merges, `/` shows the new daily Sharpe under its old "annualized from hourly results" tooltip, with no interval.

**Do not merge from 22:15 UTC on a weekday until that night's live refresh has finished** (the check is in `docs/superpowers/plans/2026-10-14-live-board-fixes.md`, Merging and records). GitHub starts the 22:30 UTC cron as late as 03:05 UTC, and the refresh keeps running on Render after the workflow run ends. The merge auto-deploys prod, and the restart kills a running refresh, so the sessions it had not checkpointed are billed again on the next night.

- Live and Competition boards now publish P19 daily Sharpe (one return per US/Eastern trading day, ddof 1, cash rate 0%/yr, ×√252) with its 95% interval (iid standard error), computed at read time from the stored curve. The browser only displays it (D20); guards keep `Math.sqrt`/`stdev` out of the board JS.
- **The published Sharpe numbers change meaning:** hourly-annualised → daily. Below 5 trading days the value is null and the UI says "Needs 5 trading days (have N)"; undefined is null, never 0 (`run.get("sharpe_ratio") or 0` is gone).
- New payload fields per entry: `sharpe_ci_low`, `sharpe_ci_high`, `sharpe_days`, `sharpe_status`; basis block at `live_status.sharpe` (Live) and `sharpe_basis` (Competition).
- Competition E_0 is approximate (its stored seed is the first mark, so day 1 omits the first hour); documented at the call site. `calc_metrics` and the stored `agent_runs.sharpe_ratio` are unchanged, so run-level APIs still show the hourly figure until the engine rewrite.
- Merging to main auto-deploys prod.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Note the merge consequence.** Merging to `main` auto-deploys prod: the Render deploy hook fires once backend tests pass. Vercel serves `dashboard/frontend`, so the new JS can briefly meet the old payload. When it does, every Sharpe cell shows `—`, because `sharpe_status` is absent; that case is pinned by `_CASES` in `test_frontend_board_sharpe.py`. It is honest and self-heals when Render finishes. Nothing else needs ordering between Render and Vercel; the landing page's ordering is Step 4's.

- [ ] **Step 4: Hold the landing window to one week, then mark the PR ready**

From this phase's deploy until P7 PR B merges, the landing page (`/`) shows the new daily Sharpe under its old "annualized from hourly results" tooltip, with no interval: it reads `sharpe_ratio` alone, so its numbers change meaning the moment Render serves this payload. Two conditions keep that window short:
1. Before this PR merges, the owner approves the landing copy of P7 Task 7.3 Step 0 (P7, `2026-10-14-live-board-fixes-p7-records.md`). Until then the PR stays a draft.
2. P6 (`2026-10-14-live-board-fixes-p6-copy.md`) and P7 PR B then merge within the same week as this PR. P7 PR B needs this phase and P6 merged and deployed first, so the order inside that week is P5, P6, P7 PR B. If either cannot land that week, do not merge this PR yet.

When both hold:

```bash
gh pr ready feat/board-daily-sharpe
```

- [ ] **Step 5: Re-check the cache-busters against main immediately before merging**

```bash
git branch --show-current   # must print feat/board-daily-sharpe
git fetch origin main
for f in js/live-leaderboard.js js/leaderboard.js home-page.js; do
  echo "$f main=$(git show origin/main:dashboard/frontend/app.html | grep -o "$f?v=[0-9]*") branch=$(grep -o "$f?v=[0-9]*" dashboard/frontend/app.html)"
done
```

Expected: for each of the three tags this PR bumped, main holds exactly the branch value minus one. With Phases 1–4 merged and nothing else in between, that is `js/live-leaderboard.js?v=6` on main and `?v=7` on the branch, `js/leaderboard.js?v=37` and `?v=38`, and `home-page.js?v=55` and `?v=56`. If main has moved (another PR bumped the same tag), rebase onto it (`git rebase origin/main`), bump again from main's value to that value plus one, and for `leaderboard.js` / `home-page.js` also update the exact pin in `dashboard/backend/tests/test_frontend_fast_boot.py` (≈:200-201, inside `test_cache_busters_bumped`) in the same commit. Then run `pytest dashboard/backend/tests/test_frontend_fast_boot.py -q -p no:cacheprovider`, push with `git push --force-with-lease`, and run this step again. Identical one-line bumps from two branches merge with no conflict, so this check is the only thing that stops the second PR's JS shipping under a version browsers already cached.

- [ ] **Step 6: Merge**

Merge only once Steps 4 and 5 hold, and never inside the nightly window named in the PR body's second paragraph. Then start P6.

#### Phase 5 review-focus candidates

1. **A NULL or NaN equity point stored mid-curve (issue #390), stamped after a session's real close.** It must be skipped as "no observation", not read as $0 or as the day's close. Added to `test_zero_or_nan_equity_never_yields_nan` (Task 5.1): the NULL and NaN points are stamped 20:00Z, after the 19:00Z closes, so a reader that took them would change the answer.
2. **A deploy where the frontend is ahead of the backend (Vercel vs Render).** The payload then has no `sharpe_status`. This is covered by the `({"sharpe_ratio": 1.5}, _BASIS)` case in `_CASES` (Task 5.5): both formatters answer `—`, never an hourly number without its interval.
3. **A Live row whose seed is unrecorded (no `metadata.initial_capital`) and whose capital differs from the board's.** `scale` stays 1.0, and `reindex_frozen_curve` pre-fills `display_capital` before the first print. Day 1 would then carry a fake jump, and the chart already shows the same jump. Not tested here:
   - every row the live writer produces records its seed (`_write_live_checkpoint` → `_llm_run_metadata`), or is a baseline seeded at the config capital, so this needs a hand-made row;
   - the defect belongs to the curve, which Sharpe only reads.
4. **The 2026-11-27 early close (Phase 1, a prerequisite).** On Phase 1's axis that session has four nodes, and the last one, 13:00, holds the 12:00 bar's equity. An axis with a fixed 16:00 close would carry that equity on to 14:00–16:00 nodes. Daily grouping takes the last point of the Eastern date, so the Sharpe figure is the same on either axis. Covered at board level by `test_live_sharpe_counts_an_early_close_once` (Task 5.2): November 2026 through the 30th, 11-27 counted once with the 13:00 node's equity as its close, `sharpe_days == 20` (20 sessions from a seeded `E_0`, so 20 returns), status `ok`, Sharpe 2.846050 with interval [−4.166815, 9.858915]. Measured on main@17c552c5 without Phase 1, it fails at its curve assertion (`Left contains 3 more items, first extra item: ('14:00', 110412.03790131217)`) while its Sharpe values still hold, so it pins Phase 1's axis under the Sharpe reader rather than a Sharpe difference. At helper level: `test_intraday_points_collapse_to_the_last_point_of_each_et_day` (Task 5.1).
5. **The parked daily board (`?period=daily`, a one-session window).** Every row there is `insufficient_days` with `sharpe_days == 1`, and the payload carries `sharpe_basis`. Not tested: the Daily tab is retired and hidden (PR #352), nothing renders that payload, and the code path is the Competition one that Task 5.4 tests.
6. **A Competition row that publishes the median of repeat runs (#604).** The row's `run` is the representative run the median selected, and Task 5.4 reads that run's curve, so the Sharpe and its interval are that one run's: the interval measures day-to-day noise inside one run, not the spread across the pooled runs. Before this phase the row echoed the same run's stored column, so which run is read has not changed; only the figure has. Not changed here: a pooled Sharpe is a statistics decision for the engine rewrite, and the row's "Median of N runs · lo to hi" label is about return only.
7. **The home rank list's sample rows now show `—` for Sharpe** (see Task 5.5's display rules). A reviewer may read a dashed column under "Sample standings" as a regression. It is the D20 rule applied to invented data, and `_CASES` pins it.
8. **Tooltip on the Live header vs the cell.** The `<th>` carries the static form of the conformance design's owner decision 5 (no N, no cash rate); each cell's `title` carries the full sentence with its own N and the payload's `rf_annual`. Both boards' headers are pinned equal (`test_sharpe_headers_say_what_the_number_is`); the home column header says "Hover a value for its 95% interval" instead, because its cells hold the interval in `title`.
