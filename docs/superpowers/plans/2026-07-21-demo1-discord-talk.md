# Demo 1 — "Talk": Discord idea → backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the already-built Discord → backtest flow so a stranger can join the lab's Discord, describe a trading idea, and get a real equity curve back — without burning unbounded LLM credit, silently succeeding on garbage input, or hitting a false "backend not running" on a cold start.

**Architecture:** No new routes and no new subsystems. Every change is a guard, a constant, or a doc on paths that already work. Two shared constants move into `domain/strategies/` so the API routers and the Discord bot stop drifting apart; the bot gains a warm-up ping and a deterministic (non-LLM) idea pre-check. Hosting is a **Render Starter worker** (decided 2026-07-21) and is a human console task — deliberately **out of this plan**.

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2, discord.py 2.x (optional dep), pytest, GitHub Actions.

## Global Constraints

- The backend is the `dashboard.backend` **package** — always import by full package path (`from dashboard.backend.x.y import z`). Never run modules by file path.
- `domain/` must **not** import `api/` or `app.py` — enforced by `dashboard/backend/tests/test_architecture_boundaries.py`. `api/` → `domain/` is allowed and is the direction this plan uses.
- **No new routes.** Three golden-set contract tests (`test_full_route_contract_unchanged`, `test_backtests_router_contract`, `test_agent_router_route_contract_unchanged`) turn CI red on *every* open PR if a route is added without updating them. Nothing in this plan adds one — if a task tempts you to, stop and re-scope.
- `discord.py` is an **optional** dep (`requirements-discord.txt`). Any test importing it must stay behind `pytest.importorskip("discord")` — an unguarded import aborts the whole pytest session at collection, not just one module.
- Run all commands from the **repo root**. Test command: `pytest dashboard/backend/tests/ -q`.
- `print()`, not `logger.info()`, is the operative logging convention — `logger.info` emits nothing under the deployed uvicorn config. Assert on `capsys`, never `caplog`.
- Hosting decision, already made: **Render Starter worker** (~$7/mo) via `render-discord-bot.yaml`. Docs written in Task 7 must describe this, not present it as open.

---

### Task 1: CI installs the Discord dependency

Today no GitHub Actions workflow mentions `discord` at all, so the bot's 12 behavioral tests in `dashboard/backend/tests/domain/chat/test_discord_bot.py` skip silently in CI (they sit behind `pytest.importorskip("discord")` at line 16). A regression in the bot ships green. This task must go **first** — every later task's bot-side test is invisible in CI until it lands.

**Files:**
- Modify: `.github/workflows/ci.yml:53-57`

**Interfaces:**
- Consumes: nothing.
- Produces: a CI environment where `import discord` succeeds, so all later bot tests actually execute.

- [ ] **Step 1: Confirm the tests currently skip without discord.py**

Run:
```bash
python -c "import discord" 2>&1 | head -1
pytest dashboard/backend/tests/domain/chat/test_discord_bot.py -q 2>&1 | tail -3
```

If `import discord` fails, expect pytest to report `12 skipped`. If discord.py *is* installed in your local venv, it reports `12 passed` — that is fine and expected locally; this task fixes **CI**, which installs only `requirements.txt`.

- [ ] **Step 2: Add the install line**

In `.github/workflows/ci.yml`, replace the `Install dependencies` step of the **`backend-tests`** job (lines 53-57):

```yaml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          # Installs discord.py so the bot's 12 behavioral tests in
          # tests/domain/chat/test_discord_bot.py actually run instead of
          # silently skipping past importorskip('discord'). This file also
          # re-includes requirements.txt, which pip resolves as already
          # satisfied. Demo 1 makes the bot a shipping surface, so a
          # regression in it must turn CI red.
          pip install -r requirements-discord.txt
          pip install pytest pytest-timeout matplotlib
```

Leave the `packaging-tests` job untouched — it tests the standalone PyPI SDK, which has no Discord dependency.

- [ ] **Step 3: Verify the workflow is still valid YAML**

Run:
```bash
python -c "import yaml,sys; d=yaml.safe_load(open('.github/workflows/ci.yml')); print(sorted(d['jobs'])); print(d['jobs']['backend-tests']['steps'][2]['run'].strip().splitlines()[-1])"
```
Expected: `['backend-tests', 'deploy-prod', 'packaging-tests']` then `pip install pytest pytest-timeout matplotlib`

- [ ] **Step 4: Verify the bot tests pass with discord.py present**

Run:
```bash
pip install -r requirements-discord.txt
pytest dashboard/backend/tests/domain/chat/test_discord_bot.py -q
```
Expected: `12 passed` (no skips).

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: install discord deps so bot tests run"
```

---

### Task 2: Tighten the backtest window cap to 14 days

`MAX_BACKTEST_DAYS = 31` today. A 31-day window is ~5x the 6-day default and can run toward the 30-minute subprocess limit while the bot stops polling at ~11 minutes (`discord_bot.py:518-520`, `max_polls = 130` × 5s = 650s) — so Discord reports a timeout on a run that is still alive server-side. 14 days keeps the demo path far from that edge.

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py:421`
- Test: `dashboard/backend/tests/test_backtests_router.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `MAX_BACKTEST_DAYS == 14` in `dashboard.backend.api.routers.backtests`. Task 4 references this module's rate-limiter constants but not this one.

- [ ] **Step 1: Write the failing boundary test**

Append to `dashboard/backend/tests/test_backtests_router.py`. The existing `test_backtest_run_rejects_excessive_date_range` uses a 6-year span, so it passes under any cap and pins nothing — this new test pins the actual boundary.

```python
def test_backtest_run_date_range_boundary_is_fourteen_days(monkeypatch):
    """14 days is allowed, 15 is rejected — the bot's ~11min poll ceiling
    (discord_bot.py max_polls) must not be reachable via a legal window."""
    assert bt.MAX_BACKTEST_DAYS == 14

    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    ok = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "2026-05-01", "end_date": "2026-05-15"},  # 14 days
        headers=_sess(),
    )
    assert ok.status_code == 200, ok.text

    monkeypatch.setattr(bt, "run_backtest_background", _Spy())
    too_long = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "2026-05-01", "end_date": "2026-05-16"},  # 15 days
        headers=_sess(),
    )
    assert too_long.status_code == 422
    assert "14 days" in too_long.json()["detail"]
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `pytest dashboard/backend/tests/test_backtests_router.py::test_backtest_run_date_range_boundary_is_fourteen_days -q`
Expected: FAIL on `assert bt.MAX_BACKTEST_DAYS == 14` (currently 31).

- [ ] **Step 3: Change the constant**

In `dashboard/backend/api/routers/backtests.py`, replace line 421:

```python
MAX_BACKTEST_DAYS = 31
```

with:

```python
# 14 days, not 31: the Discord bot stops polling for a result at ~11 minutes
# (discord_bot.py max_polls=130 x 5s) while the server lets a run continue to the
# 30-minute subprocess timeout. A window long enough to cross that gap looks like
# a failure in Discord while succeeding on the server. 14 keeps the demo path far
# from the edge; raise it only alongside the bot's poll ceiling.
MAX_BACKTEST_DAYS = 14
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest dashboard/backend/tests/test_backtests_router.py -q`
Expected: all pass, including the pre-existing `test_backtest_run_rejects_excessive_date_range`.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtests_router.py
git commit -m "fix: cap backtest window at 14 days"
```

---

### Task 3: Align the prompt-length caps behind one shared constant

A strategy can be **saved** at up to 5000 chars (`strategies.py:39`, `max_length=5000`) and then **rejected at run time** at 4000 (`backtests.py:420`, `MAX_STRATEGY_PROMPT_CHARS`). A user hits save-then-422 with no warning. The run-time cap is the cost-motivated one (the prompt is injected into *every* hourly LLM call), so 4000 wins — and the constant moves to `domain/` so the two can never drift again.

**Files:**
- Create: `dashboard/backend/domain/strategies/limits.py`
- Modify: `dashboard/backend/api/routers/backtests.py:420`
- Modify: `dashboard/backend/api/routers/strategies.py:21` (imports) and `:36-41` (the `prompt` field)
- Test: `dashboard/backend/tests/test_strategies_api.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `dashboard.backend.domain.strategies.limits.MAX_STRATEGY_PROMPT_CHARS` (`int`, value `4000`). Task 6's `idea_check` module lives in the same package and may import it.

- [ ] **Step 1: Write the failing consistency test**

Append to `dashboard/backend/tests/test_strategies_api.py`:

```python
def test_save_and_run_prompt_caps_are_identical():
    """A prompt that saves must also run. These two caps drifted (5000 vs 4000)
    and produced a save-then-422 trap; both now read one domain constant."""
    from dashboard.backend.api.routers import backtests as bt
    from dashboard.backend.domain.strategies.limits import MAX_STRATEGY_PROMPT_CHARS

    assert MAX_STRATEGY_PROMPT_CHARS == 4000
    assert bt.MAX_STRATEGY_PROMPT_CHARS == MAX_STRATEGY_PROMPT_CHARS

    field = CreateStrategyBody.model_fields["prompt"]
    max_lengths = [c for m in field.metadata for c in [getattr(m, "max_length", None)] if c]
    assert max_lengths == [MAX_STRATEGY_PROMPT_CHARS]


def test_create_strategy_accepts_prompt_at_the_cap():
    client = TestClient(app)
    at_cap = client.post("/api/strategies", json={"prompt": "b" * 4000})
    assert at_cap.status_code == 200, at_cap.text
    over_cap = client.post("/api/strategies", json={"prompt": "b" * 4001})
    assert over_cap.status_code == 422
```

Add `CreateStrategyBody` to the imports at the top of that test file:

```python
from dashboard.backend.api.routers.strategies import CreateStrategyBody
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `pytest dashboard/backend/tests/test_strategies_api.py -q -k "cap"`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.backend.domain.strategies.limits'`.

- [ ] **Step 3: Create the shared constant**

Create `dashboard/backend/domain/strategies/limits.py`:

```python
"""Size limits shared by every surface that accepts a strategy prompt.

These live in ``domain/`` because both the save path (api/routers/strategies.py)
and the run path (api/routers/backtests.py) must agree: a prompt that saves has
to be runnable. They drifted once -- save allowed 5000 chars, run rejected
anything over 4000 -- so a user could save a strategy and then get a 422 when
running it, with nothing explaining why.

The binding constraint is cost, not storage: the prompt is injected into every
hourly LLM call of a backtest, so its length multiplies across the whole run.
"""

from __future__ import annotations

MAX_STRATEGY_PROMPT_CHARS = 4000
```

- [ ] **Step 4: Point the run path at it**

In `dashboard/backend/api/routers/backtests.py`, replace line 420:

```python
MAX_STRATEGY_PROMPT_CHARS = 4000
```

with a re-export of the domain constant (keeping the module-level name so existing
references at `:462` and `:465` and any test monkeypatching keep working):

```python
# Re-exported from domain so the save path (api/routers/strategies.py) and this
# run path cannot drift apart again -- see domain/strategies/limits.py.
from dashboard.backend.domain.strategies.limits import MAX_STRATEGY_PROMPT_CHARS
```

Move that `from` statement up into the existing import block near line 49 (next to `from dashboard.backend.api.rate_limit import ...`) so imports stay grouped, and leave **only** a one-line comment where the constant used to be:

```python
# MAX_STRATEGY_PROMPT_CHARS is imported from domain/strategies/limits.py (see imports).
```

Do **not** re-declare `MAX_BACKTEST_DAYS` here — the next line already holds it (set to `14` by Task 2). Declaring it twice is a silent shadow, not an error, so nothing would catch it.

- [ ] **Step 5: Point the save path at it**

In `dashboard/backend/api/routers/strategies.py`, add to the import block after line 21:

```python
from dashboard.backend.domain.strategies.limits import MAX_STRATEGY_PROMPT_CHARS
```

and change the `prompt` field (lines 36-41) from `max_length=5000` to:

```python
class CreateStrategyBody(BaseModel):
    # max_length bounds per-row storage; the store also strips/validates non-empty.
    # The value is the shared domain constant, NOT a local literal: a prompt that
    # saves here must be runnable by /backtest/run, which enforces the same cap.
    prompt: str = Field(
        min_length=1,
        max_length=MAX_STRATEGY_PROMPT_CHARS,
        description="Free-form strategy prompt the agent will follow",
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run:
```bash
pytest dashboard/backend/tests/test_strategies_api.py dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/test_architecture_boundaries.py -q
```
Expected: all pass. `test_architecture_boundaries` is included deliberately — it proves the new `api/` → `domain/` import did not violate the layering ban (this direction is legal; the reverse is not).

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/domain/strategies/limits.py \
        dashboard/backend/api/routers/strategies.py \
        dashboard/backend/api/routers/backtests.py \
        dashboard/backend/tests/test_strategies_api.py
git commit -m "fix: align save and run prompt caps at 4000"
```

---

### Task 4: Add a daily per-client backtest quota

A 10-runs/**hour** limiter already exists (`backtests.py:437`) and is genuinely per-Discord-user, because the bot sends a deterministic per-user `X-Session-Id` (`discord_bot.py:144-146`) which `client_key()` keys on. But 10/hour is 240/day per user against lab-funded LLM credit. Add a second limiter beside it — same class, same pattern, different window.

The daily check runs **first**: with 3/day and 10/hour, the daily cap is always the binding constraint, and `allow()` records an attempt as a side effect, so checking the looser limiter first would burn hourly budget on a request the daily cap rejects anyway.

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py:432-437` (limiter definitions) and `:585-589` (the check)
- Test: `dashboard/backend/tests/test_backtests_router.py`

**Interfaces:**
- Consumes: `FixedWindowRateLimiter` and `client_key` from `dashboard.backend.api.rate_limit` (already imported at `backtests.py:49`).
- Produces: module-level `_backtest_daily_rate_limiter` in `dashboard.backend.api.routers.backtests`, monkeypatchable by tests exactly like the existing `_backtest_rate_limiter`.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_backtests_router.py`. Model it on the existing `test_backtest_run_rate_limited_per_client` — inject a fake clock so no test sleeps for a day:

```python
def test_backtest_run_daily_quota_per_client(monkeypatch):
    """A per-day cap sits beside the hourly one: 10/hr is still 240/day of
    lab-funded LLM credit per user."""
    now = [0.0]
    monkeypatch.setattr(bt, "run_backtest_background", _Spy())
    # Generous hourly limiter so this test isolates the DAILY bound.
    monkeypatch.setattr(
        bt, "_backtest_rate_limiter",
        FixedWindowRateLimiter(max_events=99, window_seconds=3600, clock=lambda: now[0]),
    )
    monkeypatch.setattr(
        bt, "_backtest_daily_rate_limiter",
        FixedWindowRateLimiter(max_events=3, window_seconds=86400, clock=lambda: now[0]),
    )
    client = TestClient(app)
    headers = _sess()  # same session -> same rate key across every call
    body = {"start_date": "2026-05-01", "end_date": "2026-05-02"}

    for i in range(3):
        assert client.post("/backtest/run", json=body, headers=headers).status_code == 200, i

    blocked = client.post("/backtest/run", json=body, headers=headers)
    assert blocked.status_code == 429
    assert "today" in blocked.json()["detail"].lower()

    # A day later the budget is back.
    now[0] += 86_401
    assert client.post("/backtest/run", json=body, headers=headers).status_code == 200


def test_daily_quota_default_is_three_per_day():
    assert bt._backtest_daily_rate_limiter.max_events == 3
    assert bt._backtest_daily_rate_limiter.window_seconds == 86_400
```

Note the hoisted `headers`: `_sess()` (`test_backtests_router.py:151-152`) mints a **fresh uuid on every call**, so calling it inline per request would give each request a different rate-limit key and the cap would never trigger. The existing `test_backtest_run_rate_limited_per_client` hoists it for exactly this reason — follow that pattern.

- [ ] **Step 2: Run it to make sure it fails**

Run: `pytest dashboard/backend/tests/test_backtests_router.py -q -k "daily"`
Expected: FAIL with `AttributeError: module 'dashboard.backend.api.routers.backtests' has no attribute '_backtest_daily_rate_limiter'`.

- [ ] **Step 3: Add the limiter**

In `dashboard/backend/api/routers/backtests.py`, directly after the existing limiter at line 437:

```python
_backtest_rate_limiter = FixedWindowRateLimiter(max_events=10, window_seconds=3600)

# Second bound, same best-effort caveats: 10/hour is still 240/day per client
# against lab-funded LLM credit. Sized for the Discord demo (~3 ideas/user/day is
# enough to try a strategy, tweak it, and try again); cheap to tune once real
# usage exists. Checked BEFORE the hourly limiter -- allow() records an attempt as
# a side effect, so testing the looser window first would spend hourly budget on a
# request the daily cap is about to reject anyway.
_backtest_daily_rate_limiter = FixedWindowRateLimiter(max_events=3, window_seconds=86_400)
```

- [ ] **Step 4: Wire the check**

Replace lines 585-589 of `dashboard/backend/api/routers/backtests.py`:

```python
    if not _backtest_rate_limiter.allow(client_key(request)):
        raise HTTPException(
            status_code=429,
            detail="Too many backtests started recently; please try again later.",
        )
```

with:

```python
    rate_key = client_key(request)
    if not _backtest_daily_rate_limiter.allow(rate_key):
        raise HTTPException(
            status_code=429,
            detail="Daily backtest limit reached; you have used all your runs for today.",
        )
    if not _backtest_rate_limiter.allow(rate_key):
        raise HTTPException(
            status_code=429,
            detail="Too many backtests started recently; please try again later.",
        )
```

The two messages must stay distinguishable — the bot surfaces the `detail` verbatim, and "try again later" vs "all your runs for today" is the difference between a user waiting 5 minutes and a user coming back tomorrow.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtests_router.py -q`
Expected: all pass, including the pre-existing `test_backtest_run_rate_limited_per_client`.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtests_router.py
git commit -m "feat: add daily backtest quota per client"
```

---

### Task 5: Survive the free-tier cold start

The Render web service sleeps when idle and can take longer to wake than the bot's 30-second HTTP timeout (`discord_bot.py:235,241`). The first `/backtest` after a quiet spell therefore reports a false "backend not running" — the worst possible first impression for a demo whose whole premise is "join and try it".

**Do not blanket-retry POSTs.** `/backtest/run` is not idempotent: if the request reached the server and only the *response* timed out, a retry starts a second run — which the global `backtest_status["running"]` flag then refuses with a confusing `success: false`. Instead: retry only idempotent GETs, and warm the backend with a GET before the backtest POST.

**Files:**
- Modify: `dashboard/backend/integrations/discord_bot.py:241-245` (`_http_get`) and inside `execute_backtest` (starts at `:439`)
- Test: `dashboard/backend/tests/domain/chat/test_discord_bot.py`

**Interfaces:**
- Consumes: `api_base()` (`discord_bot.py:65-67`), `_http_get` (`:241`).
- Produces: `dashboard.backend.integrations.discord_bot.warm_up_backend()` — an `async` function taking no arguments, returning `bool` (True if the backend answered). Never raises.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/domain/chat/test_discord_bot.py` (the file already begins with `discord = pytest.importorskip("discord")` at line 16, so these are correctly gated):

```python
def test_http_get_retries_once_on_timeout(monkeypatch):
    """A sleeping free-tier backend must not read as 'backend not running'."""
    import requests
    from dashboard.backend.integrations import discord_bot as db

    calls = []

    class _Resp:
        status_code = 200
        def json(self): return {"ok": True}
        def raise_for_status(self): return None

    def fake_get(url, headers=None, timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise requests.exceptions.ReadTimeout("cold start")
        return _Resp()

    monkeypatch.setattr(db.requests, "get", fake_get)
    assert db._http_get("/health") == {"ok": True}
    assert len(calls) == 2
    assert calls[1] > calls[0], "the retry must allow more time than the first try"


def test_http_get_gives_up_after_one_retry(monkeypatch):
    import requests
    from dashboard.backend.integrations import discord_bot as db

    calls = []

    def always_timeout(url, headers=None, timeout=None):
        calls.append(timeout)
        raise requests.exceptions.ReadTimeout("still asleep")

    monkeypatch.setattr(db.requests, "get", always_timeout)
    with pytest.raises(requests.exceptions.ReadTimeout):
        db._http_get("/health")
    assert len(calls) == 2, "exactly one retry, not an unbounded loop"


def test_warm_up_backend_swallows_failure(monkeypatch):
    import asyncio
    import requests
    from dashboard.backend.integrations import discord_bot as db

    def always_fail(url, headers=None, timeout=None):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(db.requests, "get", always_fail)
    # A dead backend must not raise out of the warm-up -- the real request that
    # follows produces the user-facing error message.
    assert asyncio.run(db.warm_up_backend()) is False
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `pytest dashboard/backend/tests/domain/chat/test_discord_bot.py -q -k "timeout or warm_up"`
Expected: FAIL — `_http_get` does not retry (first test sees `len(calls) == 1`), and `warm_up_backend` does not exist.

- [ ] **Step 3: Add the retry and the warm-up**

In `dashboard/backend/integrations/discord_bot.py`, replace `_http_get` (lines 241-245) with:

```python
# The free-tier backend sleeps when idle and can take longer to wake than a
# normal request timeout, so the first call after a quiet spell would otherwise
# surface as "backend not running" -- a false alarm on exactly the first command
# a new user tries. Retry ONCE with a longer allowance.
#
# GET only, deliberately: POST /backtest/run is not idempotent. If the request
# landed and only the response timed out, a retry starts a second run, which the
# global backtest_status flag then refuses with a confusing success:false. The
# POST path is covered by warm_up_backend() instead.
_COLD_START_TIMEOUT = 90


def _http_get(path: str, *, headers: Optional[dict] = None, timeout: int = 30) -> dict:
    try:
        resp = requests.get(f"{api_base()}{path}", headers=headers or {}, timeout=timeout)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        print(f"Backend GET {path} failed ({exc!r}); retrying once for cold start.", flush=True)
        resp = requests.get(
            f"{api_base()}{path}", headers=headers or {}, timeout=_COLD_START_TIMEOUT
        )
    resp.raise_for_status()
    return resp.json()


async def warm_up_backend() -> bool:
    """Wake a sleeping backend before a non-idempotent POST.

    Never raises: a False return just means the caller's real request will
    produce the user-facing error, rather than this probe inventing one.
    """
    try:
        await asyncio.to_thread(_http_get, "/health", timeout=_COLD_START_TIMEOUT)
        return True
    except Exception as exc:
        print("Backend warm-up failed:", repr(exc), flush=True)
        return False
```

**Before writing this**, read the current body of `_http_get` at `:241-245` and preserve its exact response handling — the code above assumes `resp.raise_for_status()` then `resp.json()`. If it differs, keep the original body and wrap only the `requests.get` call.

- [ ] **Step 4: Call the warm-up before the backtest POST**

In `execute_backtest` (starts at `discord_bot.py:439`), locate the `api_post("/backtest/run", ...)` call at line 494. Immediately before it, insert:

```python
    # Wake the free-tier backend before the non-idempotent run POST, so a cold
    # start costs a few extra seconds instead of a false "backend not running".
    await warm_up_backend()
```

Match the surrounding indentation exactly — read the ten lines above `:494` first to confirm nesting depth.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/domain/chat/test_discord_bot.py -q`
Expected: all pass (15 tests: the original 12 plus 3 new).

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/integrations/discord_bot.py dashboard/backend/tests/domain/chat/test_discord_bot.py
git commit -m "fix: survive free-tier cold start in discord bot"
```

---

### Task 6: Fail fast on ideas that cannot produce a strategy

Nothing checks idea content anywhere today, and a garbage prompt does not even fail loudly: the custom-prompt branch returns `{"actions": []}` (`domain/backtesting/portfolio_manager.py:364-366`) with no rule-based fallback and no `llm_decisions` increment, so the agent silently holds every hour and the run reports **success** with a flat curve. The user sees a flat line and concludes the platform is broken.

Two gates are needed, because `/backtest prompt:` bypasses `/strategy` synthesis entirely (`backtest_cmd`, `discord_bot.py:887-903`).

Keep the check **deterministic and cheap** — no LLM call. A false rejection is far worse than a false accept here, so the rules stay conservative: they catch empty/one-word/no-prose input, not bad trading logic.

**Files:**
- Create: `dashboard/backend/domain/strategies/idea_check.py`
- Create: `dashboard/backend/tests/domain/strategies/test_idea_check.py`
- Modify: `dashboard/backend/integrations/discord_bot.py` — `/strategy` handler (before `synthesize_strategy_prompt` at `:791`) and `execute_backtest` (`:439`, on the raw `prompt` argument)
- Test: `dashboard/backend/tests/domain/chat/test_discord_bot.py`

**Interfaces:**
- Consumes: nothing (may import `MAX_STRATEGY_PROMPT_CHARS` from Task 3's `limits.py`, same package).
- Produces: `dashboard.backend.domain.strategies.idea_check.check_trading_idea(text: str | None) -> str | None` — returns a user-facing error message, or `None` when the idea is acceptable.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/domain/strategies/test_idea_check.py`. Check whether `dashboard/backend/tests/domain/strategies/__init__.py` exists; if not, create an empty one (the sibling `tests/domain/chat/` has one).

```python
"""The idea pre-check is deliberately conservative: it catches input that cannot
possibly describe a strategy, not input that describes a bad one. A false
rejection turns a curious user away; a false accept costs one flat-curve run."""

import pytest

from dashboard.backend.domain.strategies.idea_check import check_trading_idea


@pytest.mark.parametrize("bad", [
    None,
    "",
    "   ",
    "hi",
    "test",
    "12345",
    "!!!!!!!!!!",
    "buy",
])
def test_rejects_input_that_cannot_be_a_strategy(bad):
    assert check_trading_idea(bad) is not None


@pytest.mark.parametrize("good", [
    "Buy the Magnificent 7 equally when the market is calm; cut exposure when volatility spikes.",
    "go long tech stocks on green days and hold cash otherwise",
    "Rotate into defensive names whenever the index drops more than two percent in a week.",
])
def test_accepts_a_plausible_idea(good):
    assert check_trading_idea(good) is None


def test_message_is_user_facing_and_actionable():
    msg = check_trading_idea("hi")
    assert msg is not None
    assert len(msg) > 20
    # Must tell the user what to do, not just that they were wrong.
    assert "describe" in msg.lower() or "example" in msg.lower()


def test_accepts_a_long_prompt_at_the_cap():
    from dashboard.backend.domain.strategies.limits import MAX_STRATEGY_PROMPT_CHARS
    long_idea = "buy low and sell high when momentum turns positive " * 60
    assert len(long_idea) > 1000
    assert check_trading_idea(long_idea[:MAX_STRATEGY_PROMPT_CHARS]) is None
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `pytest dashboard/backend/tests/domain/strategies/test_idea_check.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.backend.domain.strategies.idea_check'`.

- [ ] **Step 3: Write the checker**

Create `dashboard/backend/domain/strategies/idea_check.py`:

```python
"""A cheap, deterministic gate on user-submitted trading ideas.

Why this exists: a garbage prompt does not fail loudly. The custom-prompt branch
of the portfolio manager returns no actions (domain/backtesting/portfolio_manager.py),
so the agent holds every hour and the run reports SUCCESS with a flat equity
curve. The user sees a flat line and concludes the platform is broken.

Why it is not an LLM call: this runs before every /strategy and every raw
/backtest, on a public surface, and would double the cost of rejecting junk.

Why the rules are weak on purpose: this catches input that *cannot* describe a
strategy (empty, one word, punctuation, digits). It deliberately does NOT judge
whether the strategy is any good -- that is what running the backtest is for, and
a false rejection turns a curious user away at the door.
"""

from __future__ import annotations

import re

MIN_IDEA_CHARS = 15
MIN_IDEA_WORDS = 4

_WORD_RE = re.compile(r"[A-Za-z]{2,}")

_REJECTION = (
    "That does not look like a trading idea yet. Describe what to buy or sell "
    "and when, in a sentence or two. For example: \"Buy the Magnificent 7 "
    "equally when the market is calm; cut exposure when volatility spikes.\""
)


def check_trading_idea(text: str | None) -> str | None:
    """Return a user-facing rejection message, or None if the idea is usable."""
    if text is None:
        return _REJECTION
    cleaned = text.strip()
    if len(cleaned) < MIN_IDEA_CHARS:
        return _REJECTION
    if len(_WORD_RE.findall(cleaned)) < MIN_IDEA_WORDS:
        return _REJECTION
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/domain/strategies/test_idea_check.py -q`
Expected: all pass.

- [ ] **Step 5: Write the failing wiring tests**

Append to `dashboard/backend/tests/domain/chat/test_discord_bot.py`:

```python
def test_both_idea_entry_points_are_gated():
    """/strategy runs synthesis and /backtest prompt: bypasses it entirely, so
    the pre-check must appear on BOTH paths -- gating only synthesis leaves the
    raw path wide open."""
    import inspect
    from dashboard.backend.integrations import discord_bot as db

    strategy_src = inspect.getsource(db.strategy)
    backtest_src = inspect.getsource(db.execute_backtest)

    assert "check_trading_idea" in strategy_src, "/strategy is not gated"
    assert "check_trading_idea" in backtest_src, "raw /backtest prompt: is not gated"


def test_strategy_checks_idea_before_calling_the_model():
    """The gate must precede synthesis -- checking after the LLM call would
    spend the credit the gate exists to save."""
    import inspect
    from dashboard.backend.integrations import discord_bot as db

    src = inspect.getsource(db.strategy)
    assert src.index("check_trading_idea") < src.index("synthesize_strategy_prompt")
```

- [ ] **Step 6: Run them to make sure they fail**

Run: `pytest dashboard/backend/tests/domain/chat/test_discord_bot.py -q -k "idea"`
Expected: FAIL — `check_trading_idea` appears in neither handler.

- [ ] **Step 7: Wire the gate into `/strategy`**

Add to the import block near `discord_bot.py:17-21`:

```python
from dashboard.backend.domain.strategies.idea_check import check_trading_idea
```

In the `strategy` handler, insert immediately **before** the `try:` that wraps `synthesize_strategy_prompt` (currently at `:790-795`), after `agent_id` is resolved:

```python
    # Gate before synthesis: a rejected idea must not spend a model call.
    # Only check when the user typed one -- omitting `idea` deliberately means
    # "compile from my recent /ask chat", which has its own content.
    if idea is not None:
        problem = check_trading_idea(idea)
        if problem:
            await interaction.edit_original_response(content=problem)
            return
```

- [ ] **Step 8: Wire the gate into the raw backtest path**

In `execute_backtest` (`discord_bot.py:439`), add the check on the raw `prompt` argument near the top of the function, before any HTTP call. Read the function signature and its first ten lines first to confirm the parameter name is `prompt` and to find how it reports errors to the user (match that existing pattern — the surrounding code uses `interaction.edit_original_response`):

```python
    # The raw `/backtest prompt:` path bypasses /strategy synthesis entirely, so
    # it needs its own gate. `code` (a saved strategy) is already checked at save
    # time and is not re-validated here.
    if prompt is not None:
        problem = check_trading_idea(prompt)
        if problem:
            await interaction.edit_original_response(content=problem)
            return
```

- [ ] **Step 9: Run the full bot suite**

Run: `pytest dashboard/backend/tests/domain/chat/ dashboard/backend/tests/domain/strategies/ -q`
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add dashboard/backend/domain/strategies/idea_check.py \
        dashboard/backend/tests/domain/strategies/ \
        dashboard/backend/integrations/discord_bot.py \
        dashboard/backend/tests/domain/chat/test_discord_bot.py
git commit -m "feat: pre-check trading ideas on both discord entry points"
```

---

### Task 7: Docs truth pass

Four documented inaccuracies, each of which has already cost someone time:

1. `README.md` carries a Discord **badge** (`README.md:12-13`, "Join Community") but nowhere says you can *run a backtest* there. The demo's entire front door reads as a chat-room invite. Note the badge means a naive `"discord" in readme` assertion passes today and proves nothing — the test below checks for the command instead.
2. The bot-invite URL recipe (OAuth scopes + permission bits) exists **nowhere** in the repo — only the end-user `discord.gg` link does. Whoever deploys the worker next has to re-derive it from the code.
3. The Message Content privileged intent requirement is mentioned only in a runtime log line (`discord_bot.py:670-672`), not in any doc.
4. `domain/chat/service.py:131-135` states *"The Discord bot should not call Anthropic directly"* — literally true (no `AsyncAnthropic(...)` in `discord_bot.py`) but architecturally misleading: `chat_with_agent` executes **inside the bot's own process** via a direct Python import. This docstring is why the "bot needs its own LLM key" requirement was nearly missed.

**Files:**
- Modify: `README.md`
- Modify: `docs/discord-bot-instructions.md`
- Modify: `docs/architecture/discord-to-backtest.md`
- Modify: `dashboard/backend/domain/chat/service.py:131-143`
- Test: `dashboard/backend/tests/integrations/test_docs_run_command.py` (existing doc-assertion pattern — read it first and follow its style)

**Interfaces:**
- Consumes: everything built in Tasks 1-6 (the docs must describe the *new* 14-day cap, the daily quota, and the idea pre-check, not the old behavior).
- Produces: nothing importable.

- [ ] **Step 1: Write the failing doc test**

Read `dashboard/backend/tests/integrations/test_docs_run_command.py` first to match its existing assertion style, then append to it:

```python
def test_discord_docs_carry_the_deployment_recipe():
    """The invite URL and the privileged-intent requirement existed nowhere in
    the repo -- only in a runtime log line and in one engineer's head."""
    from dashboard.backend.paths import REPO_ROOT

    doc = (REPO_ROOT / "docs" / "discord-bot-instructions.md").read_text(encoding="utf-8")
    assert "applications.commands" in doc, "invite URL scope not documented"
    assert "117760" in doc, "permission bits not documented"
    assert "MESSAGE CONTENT INTENT" in doc.upper(), "privileged intent not documented"

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    # NOT `"discord" in readme` -- the Join-Community badge at README.md:12-13
    # already satisfies that and says nothing about running a backtest.
    assert "/backtest prompt:" in readme, "README does not show the Discord backtest command"
    assert "discord-bot-instructions" in readme, "README does not link the command reference"
```

Confirm `REPO_ROOT` is the correct exported name in `dashboard/backend/paths.py` before running — if it differs, use whatever that module actually exports.

- [ ] **Step 2: Run it to make sure it fails**

Run: `pytest dashboard/backend/tests/integrations/test_docs_run_command.py -q -k discord`
Expected: FAIL on the first assertion (`applications.commands` is absent).

- [ ] **Step 3: Add the deployment section to the Discord doc**

Append to `docs/discord-bot-instructions.md`:

```markdown
---

## Running the bot (operators)

The bot runs as a **Render Starter worker** (~$7/mo — Render workers have no free
tier), deployed from `render-discord-bot.yaml`: Dashboard → New → Blueprint →
connect the repo → set the Blueprint path to that file.

### Discord Developer Portal

1. **Bot → Reset Token** → this is `DISCORD_BOT_TOKEN` (shown once).
2. **Bot → Privileged Gateway Intents → enable MESSAGE CONTENT INTENT.** Leave
   *Server Members* and *Presence* off — the code requests neither.
   Without Message Content, free chat in server channels silently receives empty
   message bodies; DMs still work, which makes this failure confusing to diagnose.
3. Invite the bot with:

   ```
   https://discord.com/oauth2/authorize?client_id=<APPLICATION_ID>&permissions=117760&scope=bot%20applications.commands
   ```

   `117760` = View Channel + Send Messages + Embed Links + Attach Files +
   Read Message History. Attach Files carries the equity-curve PNG; Read Message
   History backs reply-to-bot detection. Nothing needs a moderation permission.

   The `applications.commands` scope is **not optional**: without it the token
   connects normally and `tree.sync()` fails, so the slash commands never appear.
4. Enable Developer Mode in a Discord client, right-click the server → **Copy
   Server ID** → this is `DISCORD_GUILD_ID`. Command sync is guild-scoped only;
   the bot never registers global commands.

### Worker environment

| Variable | Required | Failure if missing |
|---|---|---|
| `DISCORD_BOT_TOKEN` | yes | `RuntimeError` at startup, before connecting |
| `DISCORD_GUILD_ID` | yes | `RuntimeError` inside `setup_hook`, mid-startup |
| `ANTHROPIC_API_KEY` *or* `COMMONSTACK_API_KEY` | yes | `/ask` and `/strategy` fail; `/backtest` still works |
| `DISCORD_BOT_API_SECRET` | for `/agent` only | HTTP 401 → generic "fetch failed" |
| `COMMONSTACK_BASE_URL` | no | defaults to `https://api.commonstack.ai` |
| `DISCORD_CHANNEL_ID` | no | no channel allowlist |

`ATL_API_BASE` and `PUBLIC_APP_URL` are already set in `render-discord-bot.yaml`.

Two traps worth knowing before you debug:

- **A blank value fails exactly like a missing one** — `require_env` rejects the
  empty string, so a present-but-empty Render field produces an identical error.
- **`DISCORD_BOT_API_SECRET` must be byte-identical on the worker and the web
  service.** Unset-on-either-side and mismatched all collapse to the same HTTP
  401, folded into a generic error. Nothing logs which side is wrong.

The bot needs its **own** model key: `/ask` and `/strategy` call the model
**in-process** inside the worker (see `domain/chat/service.py`), so the backend's
key does not cover them.

### Current limits

- Backtest window: **14 days** maximum (`MAX_BACKTEST_DAYS`).
- Quotas: **10 runs/hour** and **3 runs/day** per Discord user.
- Prompts: **4000 characters**, enforced identically at save and at run.
- Ideas shorter than ~15 characters or 4 words are rejected before any model call.
```

- [ ] **Step 4: Add the Discord entry point to the README**

Find the README's quickstart / usage section and add (adapt the heading depth to match its surroundings):

```markdown
### Try it on Discord

The fastest way to run a backtest is to describe an idea in Discord — no account,
no install:

1. Join [discord.gg/9HnQ6XDG98](https://discord.gg/9HnQ6XDG98)
2. Run `/backtest prompt: Buy the Magnificent 7 equally when the market is calm; cut exposure when volatility spikes.`
3. The bot replies in-channel with metrics and an equity-curve image.

Full command reference: [docs/discord-bot-instructions.md](docs/discord-bot-instructions.md)
```

- [ ] **Step 5: Fix the misleading docstring**

In `dashboard/backend/domain/chat/service.py`, replace the docstring at lines 131-143. Read the current text first and preserve everything except the inaccurate boundary claim:

```python
    """
    Send a message to an Agentic Trading Lab agent.

    This function is the shared entry point for every chat surface. Note that it
    runs IN-PROCESS in whichever service calls it: the Discord bot imports it
    directly (integrations/discord_bot.py), so the bot's own host needs its own
    ANTHROPIC_API_KEY / COMMONSTACK_API_KEY -- the backend's key does not cover
    /ask or /strategy. This is not an HTTP boundary, despite the name.
    """
```

Keep the "Future implementation" list that follows it intact.

- [ ] **Step 6: Reconcile the architecture doc**

Read `docs/architecture/discord-to-backtest.md` in full. The known defect is the component table at line 41, which describes the entrypoint as:

> **Entrypoint** | `dashboard/backend/integrations/discord_bot.py` — HTTP client to the same API the website uses.

That is half true and is the same misconception as the `chat/service.py` docstring: the bot *is* a pure HTTP client for `/backtest`, `/prompt`, and `/agent`, but `/ask` and `/strategy` run the model **in-process**. Replace with:

```markdown
| **Entrypoint** | `dashboard/backend/integrations/discord_bot.py` — HTTP client to the same API the website uses for `/backtest`, `/prompt`, `/agent`; but `/ask` and `/strategy` import `domain/chat/service.py` and call the model **in-process**, so the bot's host needs its own model key. |
```

Then scan the rest of the file and correct anything else contradicted by Tasks 2-6 — a stated window cap (now 14 days), the absence of a daily quota (now 3/day), or the absence of idea validation (now gated on both entry points). If the file states hosting is undecided, update it to the Render Starter worker.

- [ ] **Step 7: Run the doc test and the full suite**

Run:
```bash
pytest dashboard/backend/tests/integrations/test_docs_run_command.py -q
pytest dashboard/backend/tests/ -q
```
Expected: doc test passes; full suite green with no new failures.

- [ ] **Step 8: Commit**

```bash
git add README.md docs/discord-bot-instructions.md docs/architecture/discord-to-backtest.md \
        dashboard/backend/domain/chat/service.py \
        dashboard/backend/tests/integrations/test_docs_run_command.py
git commit -m "docs: document discord bot deployment and current limits"
```

---

## After the worker is deployed (not code — do not dispatch a subagent for these)

The demo doc asks for one thing this plan cannot deliver, because it needs a
running worker against the live backend:

- **Time the default path end-to-end.** Run `/backtest prompt: …` on the default
  6-day window and record wall-clock time from command to posted PNG. The claim
  "minutes, not tens of minutes" must be *measured*, not assumed. If it lands
  above ~5 minutes, revisit `MAX_BACKTEST_DAYS` (Task 2) downward rather than
  shipping a demo whose headline experience is a long wait.
- **Confirm `DISCORD_BOT_API_SECRET` is actually set on the web service.** Whether
  the live API has it populated is unverified. Only `/agent` depends on it.

## Deliberately out of scope

- **FIFO run queue.** `backtest_status` (`backtests.py:187`) is an unsynchronized dict with a TOCTOU race (check at `:602`, set in-thread at `:235`). A rejected run returns HTTP 200 `{success: false}`, which the bot surfaces politely, and it self-clears via `finally` plus a 1800s subprocess timeout. Nothing gets stuck. This is the one medium-sized item and it is deferred per the demo doc's own note — but *"three concurrent users queue gracefully"* is a Demo 1 exit criterion, so it must land before the demo is called done. Track it separately.
- **Run-history persistence (#140).** Not a gate: the Discord embed and PNG are the durable copy of a Demo 1 result. It gates Demo 2's permalinks.
- **Bot hosting itself.** Two web consoles (Discord Developer Portal, Render), zero code. Human task.
- **Auth on the backtest routes.** The session header is self-minted and the plot PNG is fully public. The abuse bound is prompt length × window × quota, not identity. Accepted for a demo; revisit before a hardened public launch.
- **Memory headroom on the 512MB free instance.** Each backtest forks a second full Python process (numpy/pandas) beside uvicorn. Untested under demo load, and the first suspect if runs start dying. The *bot* worker is not implicated — importing it pulls neither numpy nor pandas.
