# Live Trading Leaderboard — UI and payload contract

**Status:** UI shipped in preview; season engine not built.
**Supersedes:** the Daily Leaderboard tab (`?period=daily`), retired by this change.
**Relates to:** PR #328 (participatory competition spec, draft, DO NOT MERGE).

## Why this exists

The plan was two user-facing boards: a fixed-window **Replay** (same data for everyone,
so ranks are comparable) and a **Forward Season** (bars nobody has seen, so ranks are
honest). A later simplification proposed collapsing them into one perpetually-advancing
board fed by a nightly backtest over the current day's real data.

The collapse does not work as stated, for one reason: **a single perpetually-advancing
board cannot hold both comparability and honesty.** Rank a March entrant against an
August entrant by raw return and the board is a signup-date lottery. Backfill the late
entrant over a standing window to fix that and you have handed them bars whose outcome is
already known — look-ahead worse than Replay's, because Replay at least discloses it.

Seasons are the resolution, and they are what survived: **everyone in a season starts on
the same Monday, on the same flat $10,000, and is ranked only over the days inside it.**

## Decisions this UI encodes

| # | Decision |
|---|----------|
| 1 | Two boards: **Competition** (fixed window, the acquisition hook) and the **Live Trading Leaderboard**. The Daily Leaderboard is retired; its window math, nightly cron and 202-background refresh are the season engine's foundation. |
| 2 | A season is **two weeks** — 10 US cash sessions, Monday to Friday. |
| 3 | **Every season resets.** Entries do not carry across seasons; joining is a per-season decision. This is also the cost control: perpetual entries would bill every signup ever, every night, forever. |
| 4 | A failed night **carries positions forward flat and is marked as a visible gap**. It is never re-run days later against a market that has already moved. |
| 5 | Gap copy differs per `failure_kind`. "The market was flat" and "our job died" must never render alike. |
| 6 | The board is the **Live Trading Leaderboard**. "Live" names the *direction* it runs, not brokered execution — the About card says "simulated trading on real market data, no broker, no real capital" because the name on its own over-claims. |
| 7 | The current season is **Season 0**, the shakedown season. Season 1 is the first that counts. |

Decision 7 has a trap attached. Season 0 is falsy, so `season?.number ? … : '—'` renders
the live season as *no season at all* — silently, and only for the season shipping right
now. Every read of the number goes through `displayedSeasonNumber()` and its explicit
`Number.isFinite` check, guarded by a source-shape test rather than a value test (a test
that passes `3` cannot see this bug).

Still open at the time of writing (the grilling session was cut short): whether the Replay
qualifier gate survives once Replay is unranked, and whether `instruction_sha256`
config-freeze applies to user-owned editable entries.

## The preview state, and why it is loud

The season engine does not exist. `_normalize_period` coerces any period it does not
recognise back to `contest` rather than 4xx-ing it, so `GET /api/v1/leaderboard?period=live`
returns **HTTP 200 carrying the Competition board**. Every other element on the tab —
chart, table, curve picker, rankings — renders identically either way, because those
shapes are shared between the two boards.

So the tab renders in **preview**: real curves from the Competition window, full season
chrome, and a banner that says in as many words that no season has been run and nothing
on the tab counts. Detection compares the period **requested** against the period
**returned** (`requestedBoardPeriod` vs `payload.period`); a check against the response
alone cannot see a coerced period, which is the whole failure mode.

This is the `CLAUDE.md` fail-closed-is-not-fail-visible rule applied before the fact
rather than after it. When the engine ships, the banner disappears on its own — no
frontend change required, because the payload starts answering `period: "live"`.

## Payload contract (proposed, not yet served)

`GET /api/v1/leaderboard?period=live` should return today's leaderboard payload plus a
`season` object. Everything below is optional from the frontend's point of view: each
field has a defined absent-state, so a partial rollout degrades rather than breaks.

```jsonc
{
  "period": "live",                // MUST be exactly this, or the UI renders preview
  "window":   { "start_date": "…", "end_date": "…", "label": "…" },
  "entries":  [ /* unchanged shape */ ],
  "season": {
    "number": 3,                   // 0 is valid and is the current season
    "status": "upcoming" | "running" | "closed",
    "start_date": "2026-08-17",    // Monday
    "end_date":   "2026-08-28",    // Friday, two weeks later
    "trading_days_total": 10,
    "trading_days_elapsed": 4,
    "last_advanced_date": "2026-08-20",
    "next_advance_at": "2026-08-21T22:30:00Z",
    "entries_open": true,
    "entry_closes_at": "2026-08-17T13:30:00Z",
    "entry_count": 42,
    "gaps": [
      {
        "date": "2026-08-19",
        "failure_kind": "market_data_unavailable",
        "detail": "Alpaca returned no bars for 12 of 30 symbols"
      }
    ]
  }
}
```

`failure_kind` is a closed set, mirroring the `leaderboard_attempts` table in PR #328:
`market_data_unavailable`, `model_error`, `job_not_run`, `budget_exhausted`. An unknown
kind renders as "the advance did not complete" rather than being dropped — a gap the UI
cannot explain is still a gap the reader must see.

## What is NOT in this change

* **No backend.** No `live` period, no `forward_positions` table, no nightly advance
  job. When those land, `forward_positions` belongs on `AGENT_RUNS_DATABASE_URL` — the
  Render free tier has no disk, so local SQLite resets to the seed DB on every deploy.
* **No user entries.** The board shows the same model + baseline roster as Competition —
  entries come from the curated `dashboard/config/leaderboard.json` roster, not from
  submissions. Entry flow, the qualifier gate and the practice range are downstream of
  the unresolved frontier questions above.

## What the landing page had to say instead

`Race.tsx` sold one "live leaderboard" you could enter: *"Race on the live leaderboard /
Paper trading on live markets. Watch your agent climb against the community"*, over
bullets promising live prices and rankings that "update as agents trade". Every clause
was false — there is no user entry path, the Competition board is a fixed historical
window, and brokered execution is a PR #328 non-goal with `paper_backend.py` a stub.

Rewritten to what the app serves: **"See where the bar is"**, naming both boards, plus a
preview note that the Live Trading Leaderboard is not ranking until Season 1. Requires a
`vite build` and the hand-patch recipe in `dashboard/landing/README.md` (the shipped
`frontend/index.html` keeps an inline auth layer the React bundle cannot carry).

The claim survived this long because its guard was scoped one file too narrowly:
`test_band_makes_no_paper_trading_claim` bans `paper[\s-]?trad` in `WhyCare.tsx` only,
so the claim lived on in the neighbouring section, pinned clean the whole time. The new
guards in `test_landing_copy_register.py` read the **shipped bundle**, so an unbuilt TSX
edit fails rather than passing against stale text.

## Data feasibility (settled, do not re-derive)

The Alpaca account is **free tier**. `infrastructure/market_data/quotes.py:257-341` tries
IEX first, then SIP capped at `now - 15min` to dodge the recent-SIP restriction. The
nightly cron fires 22:30 UTC = 18:30 ET, **2.5 hours after the 16:00 ET close**, so
same-day SIP hourly bars are available and free. The 15-minute rule only bites intraday —
which is what rules out a genuinely live board, independently of any design decision.
