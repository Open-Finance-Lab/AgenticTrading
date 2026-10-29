# TradingAgents as a hosted ATL runtime

ATL runs TradingAgents itself, on the platform's own OpenRouter account, the
same shape as the AI Hedge Fund hosted runtime. A user clones the marketplace
template and runs a backtest; they never install anything and never supply a
key.

> This is **not** the integration described in
> `docs/integrations/tradingagents.zh-CN.md`. That one has the user run
> TradingAgents locally and replay the result through the SDK, and says ATL does
> not host the model or pay for it. Both exist; this document covers the hosted
> route only.

## The chain

```text
POST /backtest/run  (agent whose runtime_type is "tradingagents")
  -> HourlyBacktester            engine builds the runtime from the factory table
  -> TradingAgentsRuntime        daily guard, single symbol, data cutoff
  -> subprocess                  isolated interpreter, allow-listed environment
  -> bridge.py                   stdlib only; pins provider+model, fail-closed
  -> TradingAgentsDecisionGenerator   the SDK's generate(), reused unchanged
  -> TradingAgents 0.3.1         multi-agent debate on OpenRouter
  -> five-tier rating            Buy / Overweight / Hold / Underweight / Sell
  -> executable ATL action       sized here; TradingAgents returns no quantity
```

One invocation is one billed decision: one symbol, one analysis date.

## What the deployment must provide

| Requirement | Where it comes from | Required? |
| --- | --- | --- |
| a platform gateway credential | **one of** `COMMONSTACK_API_KEY` or `OPENROUTER_API_KEY`. The adapter picks the gateway (CommonStack first, matching `ATL_PLATFORM_PROVIDER_ORDER`), renames the secret to the variable upstream reads, and hands the child **only that one** | **yes** |
| `TRADINGAGENTS_PYTHON` | set by the Dockerfile (`/opt/tradingagents-venv/bin/python`) and by `render.yaml` | **yes** |
| The isolated venv | built from `requirements-tradingagents.txt` during the image build | **yes** |
| `packaging/` in the deployed tree | `COPY packaging ./packaging`; the bridge imports the SDK from it | **yes** |
| Egress to `openrouter.ai` | already needed by AI Hedge Fund | **yes** |
| Egress to Yahoo Finance | new: TradingAgents reads market, fundamentals and news through yfinance | **yes** |
| `FRED_API_KEY` | macro tool for the `news` analyst; absent, that vendor reports itself unavailable and the analyst continues without it | no |
| `TRADINGAGENTS_TIMEOUT_SECONDS` | per-decision budget, 1..900, default 600 | no |
| `TRADINGAGENTS_MAX_POSITION_WEIGHT` | share of equity one position may reach, default 0.2 | no |
| `TRADINGAGENTS_ROOT` | overrides where the child imports the SDK from; unset unless the SDK is vendored outside the repository layout | no |
| `TRADINGAGENTS_GATEWAY` | pins the gateway to `commonstack` or `openrouter` instead of autodetecting from which credential is set | no |
| `MAX_TRADINGAGENTS_TRADING_DAYS` | trading days one run may cover, 0..23, default **5**; `0` turns the runtime off. ⚠ Raising it above 5 makes `app.js` report "lost contact" for a healthy run — it polls for at most 4200s and cannot read a run's real budget (issue #474 item 5). 600s x 5 + 600 = 3600 is what keeps that honest | no |
| `TRADINGAGENTS_RESULTS_DIR` | where upstream writes its full state log. **Unset, the trajectory is lost by construction**: the child's `HOME` is the call's TemporaryDirectory, so the default `$HOME/.tradingagents/logs` is deleted the moment the call returns. Deployment-owned — a payload cannot redirect writes. Needs durable storage to survive a restart, which prod does not have (`disk: null`) | no |

No per-user credential exists for this runtime. Market data is keyless.

The isolated environment adds roughly 400 MB and ~114 packages to the image.
Both venvs are built in one layer, and each ends with an import check, so a
broken pin fails the build rather than shipping an image whose hosted runtime
only fails at request time.

## Verify the deployment before spending anything

```bash
# inside the deployed container / service shell
python -c "from dashboard.backend.infrastructure.tradingagents.adapter import runtime_unavailable_reason; print(runtime_unavailable_reason() or 'OK')"
```

`OK` means the interpreter, the bridge and the SDK sources are all reachable.
Any other string names the missing piece, and is the same text the API returns
as a 503 at request time.

## Minimal real smoke test

One analyst, one symbol, one trading day, one runtime call — the cheapest real
decision there is. Run from the repository root with `OPENROUTER_API_KEY` in the
environment. Never write the key into a file.

```bash
python - <<'PY'
from datetime import date, datetime, timezone

from dashboard.backend.domain.agents.runtime import AgentRuntimeContext
from dashboard.backend.infrastructure.tradingagents.adapter import (
    TradingAgentsRuntime,
    runtime_unavailable_reason,
)

print("availability:", runtime_unavailable_reason() or "OK")

runtime = TradingAgentsRuntime({"analysts": ["market"]})
context = AgentRuntimeContext(
    timestamp=datetime(2026, 3, 13, 15, tzinfo=timezone.utc),
    backtest_start_date="2026-03-01",
    symbols=["AAPL"],
    cash=10_000.0,
    total_equity=10_000.0,
    positions={},
    entry_prices={},
    current_prices={"AAPL": 50.0},
    latest_market_date_before_decision=date(2026, 3, 12),
)

print("actions:", runtime.decide(context)["actions"])
print("calls:", runtime.calls)
print("audit:", runtime.decision_audit_rows)
PY
```

Pass criteria:

- `availability: OK`
- `calls: 1` — one decision, not one per hourly bar
- `actions` holds one `buy`/`sell` order, or is empty on a `HOLD` rating
- the audit row's `data_cutoff_date` is strictly earlier than its
  `decision_date`
- the output contains no `sk-` string and no `raw_final_trade_decision`

> **The cutoff check above is not a no-look-ahead check, and must not be read
> as one.** It compares two date strings ATL itself wrote; it says nothing
> about what upstream's tools actually fetched. Three of the four analysts
> read data bounded by the wall clock rather than by the date they are handed
> — `social` unconditionally (StockTwits and Reddit are pre-fetched into the
> prompt with no date parameter at all), `news` through
> `get_prediction_markets`, and `fundamentals` through `get_fundamentals`,
> whose own signature annotates `curr_date` as "not used for yfinance". So
> this criterion passes while the run is contaminated, which is exactly the
> shape the repo's *fail-closed is not fail-visible* rule warns about.
>
> What actually holds the property is the refusal in
> `describe_tradingagents_lookahead` (`domain/agents/runtime.py`): the API
> answers **422** for any roster containing one of those three, and
> `TradingAgentsRuntime.decide` refuses again for the CLI path. The per-tool
> audit behind the classification is in `TRADINGAGENTS_LOOKAHEAD_EVIDENCE` in
> that module, read off the pinned revision. Re-enabling an analyst means
> changing that table, with evidence — not relaxing the guard.

Record how long the call took and what it cost. That number sizes every later
decision: the parent backtest timeout is `TRADINGAGENTS_TIMEOUT_SECONDS` times
the run's trading days, so a slow first call is the signal to shorten the date
range rather than raise the budget.

## Then, through the API

```bash
curl -X POST "$BASE/api/v1/agents/marketplace/tradingagents/clone" \
     -H "X-Session-Id: $SESSION" -H 'Content-Type: application/json' -d '{}'

curl -X POST "$BASE/backtest/run" -H "X-Session-Id: $SESSION" \
     -H 'Content-Type: application/json' \
     -d '{"start_date":"2026-03-12","end_date":"2026-03-13",
          "decision_source":"llm","assets":["AAPL"],"agent_id":"<id>"}'
```

The clone is required: `/backtest/run` only accepts built-in agents, and cloning
a marketplace template is the only path that produces one. Both calls must
carry a signed-in account's credentials (an `Authorization` bearer token / session cookie) in addition to `X-Session-Id`: unlike AI Hedge Fund, the TradingAgents lane refuses an unauthenticated run with **401** before it spends anything (`_require_signed_in_for_tradingagents`), so a bare `X-Session-Id` smoke test now 401s by design. A multi-symbol
`assets` list, or omitting it, is refused with 422 — the runtime is
single-symbol for now.

## Keeping the agent trajectory

Upstream writes one JSON per analysis date holding the whole trajectory --
every analyst report, the bull/bear debate with both histories, the research
manager's verdict, the trader's plan, all three risk debators and the portfolio
manager's final decision. Measured at **172KB** for a single-analyst run.

It lands at
`$TRADINGAGENTS_RESULTS_DIR/<TICKER>/TradingAgentsStrategy_logs/full_states_log_<date>.json`.

⚠ **Without `TRADINGAGENTS_RESULTS_DIR` the file is written and then deleted**,
because the child's `HOME` is this call's TemporaryDirectory. That is not a
container or `--rm` artifact; it happens on every run.

The trajectory holds full upstream text and is **server-side only**. ATL's own
audit row keeps a sha256 plus a <=500-character excerpt and nothing more, and
no API surfaces the file. Do not wire it into a response.

## Known limits

- **Single symbol per run.** Rejected at request time rather than mid-run.
- **No cross-day caching.** Each decision runs with `HOME` pointed at a fresh
  temporary directory, so TradingAgents' cache, results directory and memory log
  are recreated and discarded per call. Market data is re-fetched each day and
  the reflection layer never accumulates history. This is the price of the
  credential isolation; giving it a durable path is a separate decision.
- **Not metered against credits.** Hosted runtimes bypass the per-token
  reservation path, which is gated on `runtime_type == "pipeline"`. AI Hedge
  Fund is in the same position. `agent_runs.llm_calls` still counts every
  hosted call, so the leaderboard integrity guard sees an honest number.
- **Model pin is provisional.** `bridge.py` pins both the deep-think and
  quick-think models to ATL's registered OpenRouter default. A test asserts the
  two cannot drift apart. The right pair for TradingAgents specifically is still
  open.

## Decisions this runtime needs from the team

Each item states the current behaviour, why it is a problem, the options, and
whether it blocks merging. They are listed most urgent first. Nothing here was
decided unilaterally; where a safe fallback existed it is already in place and
is named.

### 1. Anonymous requests can still reach AI Hedge Fund's platform spend

**Current.** `POST /backtest/run` gates LLM runs on a signed-in account only
for `runtime_type == "pipeline"` (`run_backtest_endpoint`, the
`user_id is None -> 401` branch). Hosted runtimes skipped it. Cloning a
marketplace template with a self-chosen `X-Session-Id` makes the caller that
agent's owner, because `owner_browser_session` is set from that header, so an
unauthenticated caller could launch a run billed to the platform's own gateway
account.

**Done here.** TradingAgents now requires an account
(`_require_signed_in_for_tradingagents`), checked before the roster, window and
readiness checks so an anonymous caller cannot probe the deployment's
credential state either. Pinned by
`test_anonymous_session_cannot_launch_a_platform_paid_tradingagents_run`.

**Still open.** The AI Hedge Fund lane has the same pre-existing
unauthenticated-access gap and was left untouched here: it is shipped
behaviour, and requiring an account for it is a product decision. The
per-agent Financial Datasets credential it needs is a data-access barrier,
not an authentication one.

**Reproduction withheld from this public file.** This repository is public and the AI Hedge Fund lane is not yet fixed, so the step-by-step call sequence and the zero-cost test-based demonstration are not written here. They have been moved to a private maintainer note (see the hand-off to Felix). Maintainers can confirm the exposure from that note; no live request is needed.

**Blast radius.** Caller: anyone on the internet -- no account, no email, no
payment method. Per accepted run: one upstream subprocess per trading day, up
to `MAX_AI_HEDGE_FUND_TRADING_DAYS` (default 10), each doing its own LLM
calls on the platform's gateway. Concurrency bound: only the instance-wide
`MAX_ACTIVE_DASHBOARD_BACKTESTS` (5) actually holds -- the route's own
the per-client `FixedWindowRateLimiter(10, 3600)` is keyed on a
caller-supplied value (`client_key`, `api/rate_limit.py:208-219`) and so does
not bound a client that varies it. There is no Credits debit on this lane, so
nothing draws down and nothing alerts; the first signal is the gateway balance.

**Options.** (a) Extend the same gate to AI Hedge Fund, which makes both
hosted runtimes sign-in-only and is one line. (b) Leave it and accept that any
visitor can spend the operator's gateway balance. (c) Rate-limit anonymous
hosted runs instead of refusing them.

**Recommendation: (a).** It reuses the gate that already exists and matches
what the pipeline lane has required since the execution layer shipped.

**Blocks merge?** Not for this branch -- TradingAgents fails closed. It is a
live production exposure for AI Hedge Fund regardless of this branch.

### 2. Hosted runs are not metered, so spend is unbounded per account

**Current.** Both hosted runtimes bypass the reservation/settlement path
entirely (it is gated on the pipeline runtime). A signed-in account can
therefore start runs until `MAX_ACTIVE_DASHBOARD_BACKTESTS` (5) is reached,
repeatedly, with no Credits debit and no per-account budget. One TradingAgents
run is 11 sequential model calls per trading day, up to
`MAX_TRADINGAGENTS_TRADING_DAYS` (5) days.

**Why it matters.** The sign-in gate above ties spend to an identity but does
not bound it, and the bounds that look like they apply do not:

- The route's own limiter (`FixedWindowRateLimiter(10, 3600)`) is keyed on a
  caller-supplied value, so it does not bound a client that varies that value.
- `_max_concurrent_for_user` grants a signed-in account its
  `max_concurrent_backtests` entitlement, default 5 -- the same as
  `MAX_ACTIVE_DASHBOARD_BACKTESTS` -- so one account can hold every slot on
  the instance continuously.
- Signup is open, and one run is ~55 sequential model calls (11 per decision
  x 5 trading days).

So the control today is "any account", not "any account with balance".

**Demonstration withheld from this public file.** A concrete, zero-cost demonstration of the limiter bypass and the slot-exhaustion case is in the private maintainer note rather than here, for the same reason as section 1.

**Options.** (a) Route hosted runs through the existing
`LLMExecutionService` reservation path -- correct, but the bridge reports no
token usage back, so there is nothing to settle against; it needs a flat
per-decision estimate first. (b) A per-account daily cap on hosted runs,
reusing `user_entitlements`. (c) Leave unmetered and watch the gateway
balance.

**Recommendation: (b) now, (a) when the bridge can report usage.** (b) is a
bound rather than an accounting system, which is what the risk actually is.

**Blocks merge?** A judgement call. The branch is honest about it (this
section, and the "Not metered against credits" limit above), and the exposure
is no worse per-account than AI Hedge Fund's today.

### 3. The default analyst roster is one analyst

**Current.** The shipped template selects `["market"]`, and the API refuses the
other three in a historical backtest -- their data sources answer as of the
wall clock, so a backtest built on them describes the future. See the
no-look-ahead note above and `TRADINGAGENTS_LOOKAHEAD_EVIDENCE`.

**Why it matters.** It is a real reduction in what the product advertises: a
"research desk" whose debate is driven by one analyst's report.

**Options.** (a) Ship single-analyst and say so. (b) Contribute as-of support
upstream -- a `curr_date` on `get_fundamentals`, a dated prediction-markets
source -- and re-enable per analyst as each lands. (c) Allow the contaminated
analysts behind an explicit "exploratory, not a backtest" mode whose results
never reach a leaderboard.

**Recommendation: (a) now, (b) as the real fix.** (c) only with a hard
guarantee that such runs cannot be ranked or compared.

**Blocks merge?** No. The safe default is in place.

### 3b. The stored `model_name` on a hosted agent row is still wrong

**Current.** A cloned marketplace template copies the template's
`model_name`, and `dashboard/config/marketplace.json` gives the TradingAgents
template `nvidia/nemotron-3-nano-30b-a3b`. That row then reports a model the
agent does not run on through `GET /api/v1/agents`. Every *display* path is
now fixed -- the Configure panel reads the server
(`/api/v1/agents/hosted-runtimes`), the backtest agent dropdown names the
runtime, the results panel and progress message name the runtime, and
`agentDisplayName` already ignored it for hosted agents -- and the engine
overrides `agent_runs.llm_model` with what actually ran. The editor also
clears the field to `""` on the first save.

**Still open.** The stored value itself. Changing what `model_name` holds for
a hosted agent is a change to a field in a public API response, which is the
kind of protocol change this branch does not make unilaterally.

**Options.** (a) Store `""` in the template and let every consumer resolve the
model from the runtime, as the display paths now do. (b) Store a sentinel such
as `"hosted"`. (c) Leave it and rely on the display paths.

**Recommendation: (a)**, paired with a response-model note that `model_name`
is empty for hosted runtimes.

**Blocks merge?** No -- nothing user-visible reads it any more.

### 4. Position sizing cannot buy most of the Dow

**Current.** `INITIAL_CAPITAL` is $1,000, `MAX_BACKTEST_INITIAL_CAPITAL`
$3,000, and `TRADINGAGENTS_MAX_POSITION_WEIGHT` defaults to 0.2 -- a $200-$600
position budget, below one share of AAPL, MSFT, META, UNH, GS, CAT, AVGO or
NFLX. Every Buy sizes to zero and the curve is flat.

**Done here.** The cause is now recorded and aggregated
(`runtime_sizing_filters`), so a flat run is attributable instead of reading as
a model that held. The policy itself is untouched.

**Options.** (a) Allow at least one share when cash covers it. (b) Raise the
default target weight. (c) Refuse the configuration up front, before spending.

**Recommendation.** This is a trading decision, not a bug fix, so it is left
whole. (a) is the smallest change that makes the product work at its own
capital bounds.

**Blocks merge?** No, but the runtime is close to useless on large-cap names
until it is decided.

### 5. Timeout, retry and window budget

**The layers, with the real numbers.**

| layer | value | where |
|---|---|---|
| one upstream pass, 1-analyst roster | 290-415s measured | 11 sequential LLM calls on CommonStack/Haiku |
| attempts inside one decision | **2** | the SDK's `for attempt in (1, 2)`, inside a single subprocess call |
| two passes, worst case | **830s** | 2 x 415 |
| per-decision budget | `TRADINGAGENTS_TIMEOUT_SECONDS`, default **600**, range 1-900 | hard subprocess timeout |
| window bound | `MAX_TRADINGAGENTS_TRADING_DAYS`, default **5** | derived, see below |
| parent budget | `max(3600, step x days + 600)`, capped 14400 | `_backtest_subprocess_timeout` |
| worker cap | **14400s** | stops a window pinning a worker thread |
| page stops watching | **4200s** | `BACKTEST_POLL_MAX_SECONDS` in `app.js` |

**The conflict.** The default (600s x 5 days + 600) is exactly 3600s, inside
the 4200s the page watches -- but 600s cannot cover a decision that needs its
second attempt (580-830s), so such a decision is killed and the tokens already
spent are paid for and discarded. Covering the retry needs the 900s maximum,
and `830 x 5 + 600 = 4750 > 4200`, so **no step budget both covers the retry
and keeps a five-day window visible.**

**Fixed here.** `_tradingagents_trading_days_ceiling` was inverted against the
14400s worker cap rather than the 4200s poller bound, so the shipped default
was poller-safe only by arithmetic coincidence and an operator raising
`TRADINGAGENTS_TIMEOUT_SECONDS` to 900 silently got a 5-day window with a
5100s budget -- the page reporting failure for a run still in progress, after
the user had paid for decisions they could no longer see. Both dials looked in
range on their own. The ceiling now inverts the poller bound, which leaves the
shipped default unchanged at 5 days and makes 900s yield 4.

**Two self-consistent schemes.** Both are reachable today, pinned by tests.

| | step | days | parent budget | covers 2 attempts? | visible to the page? |
|---|---|---|---|---|---|
| **A: shipped** | 600s | 5 | 3600s | no | yes, 600s of margin |
| **B: retry-safe** | 900s | 4 | 4200s | yes (830s worst case) | yes, exactly at the bound |

The derived *ceiling* is `floor(3600/step)` -- 6 at 600s, 4 at 900s -- while
the shipped *default* is `min(5, ceiling)`. The two are different numbers and
an earlier draft of this table conflated them. An operator who raises
`MAX_TRADINGAGENTS_TRADING_DAYS` to the ceiling gets a budget of exactly
4200s, which the page still watches (it gives up strictly above) but with no
margin; the default keeps 600s.

**Recommendation: B**, set from the Render dashboard with no deploy. A killed
second attempt is the one failure that costs money and returns nothing, and
the window loses one day. A is the status quo and is safe, just wasteful on a
failing day.

**Not an option without a decision:** raising the poller bound in `app.js`,
which is the only way to get five days *and* the retry. That is a frontend
change to a shared file and makes every run's "lost contact" threshold later,
so it is listed rather than taken.

**Blocks merge?** No. Both schemes are self-consistent and the default is safe.

### 6. Dependency pinning for the isolated venv

**Current.** `requirements-tradingagents.txt` pins the upstream revision
exactly (`a33fd4c`) and nothing else. Measured on 2026-10-27 with
`uv pip compile --python-version 3.12 --python-platform linux`: the closure is
**110 packages**, of which **22 are upstream's own direct dependencies** --
`langchain-core`, `langchain-openai`, `langchain-anthropic`,
`langchain-google-genai`, `langgraph`, `langgraph-checkpoint-sqlite`,
`pandas`, `yfinance`, `stockstats`, `backtrader`, `parsel`, `redis`, `typer`,
`rich`, `questionary`, `requests`, `pytz`, `tqdm`, `python-dotenv`,
`setuptools`, `typing-extensions`, `langchain-experimental`.

Every one of those 22 is declared `>=` with **no upper bound**, which is
the mechanism: nothing in the build refuses a new major.

**Why it matters more than the usual drift.** Resolving today yields
`langchain-core 1.7.0`, `langgraph 1.2.18`, `openai 3.40.0`, `pandas 3.1.0`,
`numpy 2.5.4` -- all majors that have moved since this integration was built
and verified. The same pinned tarball therefore installs a materially
different stack on each rebuild, and a rebuild happens on **every push to
main**, because the image is built from `./Dockerfile`. If a resolution fails
or an incompatible major lands, it does not degrade this runtime -- it fails
the image build, which blocks the **whole application's** deploy.

**Not done here, deliberately.** A lockfile is the fix and the resolution
above is reproducible, but resolving is not the same as verifying: it proves
the graph is consistent, not that the pinned set imports and runs. Verifying
that needs one image build, and the Docker daemon is not available in this
environment. The user's instruction was explicit -- produce the plan, do not
introduce an unverified lockfile into the build path -- so none is committed.

**The plan, ready to run.**

1. Generate, on Linux with Python 3.12 (or in the image):

       uv pip compile requirements-tradingagents.txt            --python-version 3.12 --python-platform linux            --output-file requirements-tradingagents.lock

   `uv` rather than `pip freeze`: freeze records whatever happens to be
   installed on the machine that ran it, including host-specific and
   editable entries, and cannot target another platform.

2. Verify by building the image and running, inside it:

       /opt/tradingagents-venv/bin/python -c            "import tradingagents, langgraph, langchain_core;             from tradingagents.agents.utils.rating import parse_rating;             print(parse_rating('Rating: Buy'))"

   plus the repository's own
   `test_real_subprocess_reaches_the_sdk_and_reports_a_missing_upstream`,
   which exercises the bridge across the process boundary.

3. Only then point the Dockerfile at the lock file, and record the
   regeneration command beside it so the next upstream bump is a
   deliberate re-resolve rather than a silent one.

**Risks of the fix itself.** A lock pins the *whole* 110-package closure, so a
security patch in any of them needs a re-resolve; and a lock produced on a
different platform or Python version can be unsatisfiable in the image, which
is why step 2 is not optional. Both are ordinary and preferable to the current
state.

**Two facts that bound how bad this is, both verified in the tree.**

1. *The build fails closed.* Each venv's `uv pip install` is chained with
   `&&` to an import check -- for this one,
   `from tradingagents.graph.trading_graph import TradingAgentsGraph`. A
   resolution that produces an unimportable set therefore fails the
   `docker build`; it cannot ship an image whose hosted runtime 503s at
   request time. What the check does *not* cover is a set that imports and
   then behaves differently (a changed `langgraph` API contract reached only
   at run time), so it bounds the failure mode, it does not remove it.
2. *This is not a pattern this branch introduces.*
   `requirements-ai-hedge-fund.txt` is a single floating upstream tarball
   with an unpinned closure, it is already on `main`, and it is in production
   today. ATL's own `requirements.txt` is fully pinned (61 of 62 lines `==`);
   the isolated venvs are the exception, and there was already one. This
   branch adds a second resolver to the same build chain -- it roughly
   doubles the surface of an existing risk rather than creating a new class
   of it.

**Blocks merge?** No, but it is the likeliest cause of a future "it worked
yesterday" build failure, and that failure blocks the whole application's
deploy rather than just this runtime. Fixing both venvs together is the
change worth making; fixing only the new one leaves the deploy exposed to the
older, identical hazard.

### 7. Trajectory persistence in production

`TRADINGAGENTS_RESULTS_DIR` is unset in production and the service has no
persistent disk (`disk: null`), so the full upstream state log is written and
deleted on every run. Keeping it needs durable storage, which is a cost
decision.

### 8. Upstream's rating parser can reverse a decision's direction

**Current.** `tradingagents/agents/utils/rating.py::parse_rating` runs two
passes -- an explicit `Rating: X` label, then the first five-tier word
appearing anywhere in the text. They are not exclusive: a label whose value is
off the five-tier scale matches the regex but fails the set test, so pass one
falls through silently and pass two answers from the surrounding prose.

Measured against the pinned revision on 2026-10-27:

| transcript | `parse_rating` |
|---|---|
| `...the bear case recommends Sell.\nFinal Rating: Strong Buy` | `Sell` |
| `We should Buy here.\nRating: Strong Sell` | `Buy` |
| `Final Rating: Strong Buy` | `Buy` |
| `Rating: Accumulate` | `""` |

The direction reverses both ways, and row 3 shows why it is hard to catch:
with no competing rating word in the prose the same unsupported label parses
correctly by accident, so the fault is intermittent. "Strong Buy" is not on
upstream's scale and models emit it constantly.

**Nothing downstream can see it.** ATL cross-checks the rating against the
mapped action, but both come from that one parse, so they always agree. The
record validates, the digest matches, and it becomes an order in the wrong
direction.

**Done here.** The bridge refuses a decision whose raw text *states* a rating
off the scale (`unsupported_rating_label`), as a transient failure so the step
holds and the run continues. Narrow on purpose: a transcript with no rating
label at all also goes through pass two, but refusing every unlabelled
decision would refuse most legitimate output.

**Still open.** The real fix is upstream -- pass one should stop once it has
found a label, and an off-scale label should be reported rather than ignored.
That changes a shared module and trade direction, so it is not made here.

**Options.** (a) Upstream PR fixing the two-pass fall-through. (b) Pass ATL's
own parser to `TradingAgentsDecisionGenerator`, which already accepts a
`rating_parser` -- keeps the fix in ATL but forks the parsing behaviour from
upstream's own CLI. (c) Leave the guard as the whole fix and accept the
refused steps.

**Recommendation: (a), with (b) as the interim** if upstream is slow -- the
injection point already exists and needs no fork of the package.

**Blocks merge?** No, the guard fails safe. But it is the most serious defect
found in this integration, and the only one that can produce a *wrong trade*
rather than a missing or refused one.
