# Chart-first landing — design

**Date:** 2026-08-15
**Surfaces:** `/` (marketing, `dashboard/landing/src`) and `/app` Home screen 0 (`dashboard/frontend/app.html`)
**Relates to:** `2026-08-15-live-trading-leaderboard-ui.md` (that spec defines the board's
payload and tab contract; this one defines how the board is *presented* on the two entry
surfaces). Follows PR #357, which put a board on both surfaces for the first time.

## Why this exists

PR #357 moved the leaderboard above the fold on both entry surfaces. It solved the
"four screens down" problem and created three new ones, all reported directly:

1. **The board is still too small.** On `/` it is a card in a 50/50 hero split, with a
   210–240px chart. On `/app` screen 0 it shares the row with a headline and two CTAs.
2. **It is the wrong artifact.** The ask was always a *chart* — interactive, immediately
   legible — not a ranking board. `/app` screen 0 has no chart at all; `/` has one, but
   subordinated to a five-row standings table beneath it.
3. **Its type is too small to read.** 11px chart axes, `text-xs` captions, `text-sm` rows.

And a fourth, on the copy: the `/app` screen 0 lede reads as neither marketing nor a call
to action.

## Reference

nof1.ai, measured at 1440×900. Its dominance comes from two things, and the ratio is only
one of them:

- **Full-bleed.** The chart column runs from x=0 to x≈1058 (**73%**) with no page gutter on
  its left edge; the right rail occupies x=1058→1440 (**26%**). ATL's hero sits inside
  `container mx-auto px-6`, which is why a 50% column still reads as a card.
- **Thin chrome, one-line captions.** Nav ~55px, ticker ~40px, then a single grey line
  stating exactly what the chart displays. That one line does the work ATL currently
  spends two paragraphs on.

## Shape (both surfaces)

Chart **left, 2/3**. Hero **right, 1/3**, keeping the container gutter.

"Full-bleed" means different things on the two surfaces and must not be copied across:

- On **`/`**, the chart column drops the `container mx-auto px-6` gutter on its **left edge
  only** and runs to the viewport edge, as nof1 does.
- On **`/app`**, it does not. Screen 0 lives inside `.home-pager-screen`, which is
  `height: 100%; overflow: hidden` inside a snap pager — the exact construct that clipped
  the board below 1200px in PR #357. There the chart goes flush to the *screen's* content
  edge and no further.

```
┌─────────────────────────────────────────────────────────────┐
│ nav + ticker                                                │
├──────────────────────────────────────────┬──────────────────┤
│ Where the AI models stand [Illustrative] │                  │
│ Each line is one model's account value.  │  Talk to Agents  │
│                                          │  Test Trading    │
│  $1200 ┤                        ╱‾‾‾     │  Ideas           │
│  $1100 ┤            ╱‾‾╲╱‾‾╱            │                  │
│  $1000 ┤═══════════╱═══════════════════  │  One line.       │
│   $900 ┤    ╲___                         │                  │
│        └──────────────────────────────── │  [ Start Free ]  │
│  ● DeepSeek +21.0%  ● Buy & Hold +5.5%   │                  │
│  ● DJIA +2.8%  ● Claude +1.4%  ● GPT −1.5│  small print     │
└──────────────────────────────────────────┴──────────────────┘
   full-bleed left, 2/3                    1/3, gutter kept
```

Below `lg:`, the columns stack: chart first, hero second. The chart keeps a hard minimum
height so a narrow viewport never collapses it to a strip.

## Components

### 1. Chart panel (left) — `BoardPreview.tsx` on `/`, `#homeModuleRanking` on `/app`

Same structure on both, different data source.

| Element | `/` | `/app` |
|---|---|---|
| Caption bar | One line + `Illustrative example` chip | One line; **no** chip — the data is real |
| Chart | `clamp(320px, 56vh, 520px)`, axis ticks **14px** | same clamp, but capped so the whole screen fits its pager without scrolling |
| Key beneath | Chip strip: `● Name +21.0%`, `text-base` | Compact keyed list, `text-base` |
| Detail line | One line stating what is measured | Existing `hm-rank-meta` line, unchanged |

**The standings table stops being the main event.** This is the "chart, not a ranking
board" change — but it is *demotion, not deletion*, and the two surfaces demote differently
because they carry different information.

On **`/`** it becomes a pure legend strip: five `● Name +21.0%` chips on one row. That
much is forced. `BoardPreview` ships **no** Recharts `<Legend>` on purpose — its own source
comment records that a five-item legend wraps to two rows at card width — so the standings
table is currently the *only* thing linking a curve colour to a model name. Delete it
outright and five unnamed lines are left. The chip strip preserves the swatch↔curve
identity link at a fraction of the height, and at the new width five chips fit on one row.
The full table survives in `Race.tsx`, which becomes the detail home.

On **`/app`** the list keeps its columns. `#homeModuleRankList` today carries rank, model,
ending value, return and Sharpe — real numbers a signed-in user came for, and there is no
`Race.tsx` on this surface to move them to. It restyles to a compact row that gains a
colour swatch (making it the chart's key as well) and loses vertical weight, but **ending
value and Sharpe stay**. Stripping them to match `/` would delete live data to satisfy a
marketing-page constraint.

### 2. Type scale

| | now | after |
|---|---|---|
| Chart axis ticks | 11px | **14px** |
| Chart height | 210–240px | **`clamp(320px, 56vh, 520px)`** |
| Panel title | `text-lg` | `text-xl` |
| Caption | `text-xs` | `text-sm` |
| Standings rows | `text-sm` | **`text-base`** chips |

### 3. `/app` screen 0 gets a real chart

`#homeModuleRanking` gains a Chart.js line chart above its list, built from
`entry.equity_curve`. No new endpoint and no new library: `domain/leaderboard/service.py:1251`
already puts `equity_curve` on every entry, `align_equity_curves` (`:1257`) already aligns
them across entries, `js/leaderboard.js:669` already builds curves from exactly this field,
and Chart.js is already loaded on `app.html`.

`#homeModuleRankList` stays — it is guard-pinned — and restyles into the same chip strip.

**Fallback honesty.** `home-page.js:1464` defines a sample-standings fallback with invented
returns, marked by a *"Sample standings —"* prefix, and distinguishes two reasons
(`unreachable` vs `empty`). A chart is more persuasive than a table, so five invented
curves are a larger claim than that prefix can retract.

> **Decision: on either fallback path, render no chart at all.** The note and the sample
> list ship exactly as they do today. The chart appears only when real curves arrive.

This follows the repo's fail-closed-is-not-fail-visible doctrine: the two reasons stay
distinguishable, and neither is dressed up as a result.

### 4. The `/app` screen 0 lede

Current (`app.html:462`):

> Your own agent — an AI trading assistant that follows your written instruction — is
> scored on the same numbers, in a test of its own.

The confusion has a traceable cause. The comment above it (`app.html:458`) shows the
sentence doing **two jobs at once**: glossing the word "agent", *and* pre-empting "is my
agent on this list?". It is a disclaimer wearing a value prop's clothes, which is why it
reads as neither marketing nor a call to action.

Split the jobs. The board's own meta line already reads **"AI models only · ranked by
return"** (`app.html:486`), so the no-entry fact is already stated where it belongs — on
the board making the claim. That frees the lede to be one plain thing:

> **See how the AI models did. Then test your own idea on the same days.**

Fact, then call to action. The "agent" gloss drops on this surface: the reader is signed
in and inside the app, where the word is glossed throughout. No guard pins this sentence
(verified: `test_app_copy_register.py:305-311` pins only `#homeModuleRanking`,
`#homeModuleRankList`, `#homeScrollHint`, and the absence of the `"Talk to Agents"` pitch).

### 5. Text reduction on `/`

Trim in place. Five sections stay five sections; the Navbar, the FooterCTA breadcrumb and
the section-order guards are untouched.

Roughly 45% less body copy is the *direction*, not an acceptance gate — no test asserts a
word count, and one that did would fail on any later copy edit. The per-section changes
below are the actual requirement.

| Section | Change |
|---|---|
| **Hero** | Two paragraphs → one line. The simulated-money sentence moves to small print under the CTA, **verbatim**. |
| **WhyCare** | Intro paragraph → one sentence; three ACT bodies → one line each. Headings unchanged. |
| **Talk** | Drop the three-step `<ol>` — it restates WhyCare's three acts one screen later. |
| **Test** | Trim the prose around its chart. |
| **Race** | Unchanged. This is where the detail lands. |

## Guard constraints (verified at source, not assumed)

These are the strings and shapes the existing suite pins. Every one of them survives this
design; they are listed so the implementation does not discover them by reddening CI.

**Must ship verbatim:**
- `"Every test here uses simulated money. Real money is involved only if you explicitly connect a brokerage account and turn on live trading."` — pinned twice, by
  `test_no_real_money_sentence_is_present_verbatim` **and** by the `_CLAIM_DISCLAIMERS`
  allowlist, whose staleness check (`test_the_disclaimer_allowlist_is_not_stale`) fails if
  the wording drifts. Moving it between components is fine; the allowlist is scanned across
  every `*.tsx` in `components/home/`.
- `"Illustrative example"` — must appear **≥2×** in the *minified bundle*
  (`test_illustrative_example_label_appears_at_least_twice`). esbuild interns a shared
  constant once, so the literal stays **duplicated at each site**. Do not DRY it.
- `"in preview for Season 0"` and `"Season 1 is the first that counts"` — bundle-wide.
- `"Live Trading Leaderboard"` — must be in **`Race.tsx` source** specifically
  (`test_race_source_and_shipped_bundle_agree`) as well as the bundle.
- `"Standings"` and `"Leaderboard"` in the bundle (`test_race_sample_cards_have_no_live_pulse`).
- WhyCare headings: `"Describe it in plain English"`, `"Prove it on real market data"`,
  `"See how it ranks"`, `"Pick the AI model"`, `"For developers: bring your own agent"`.
- Talk: `"Describe your idea"`, `"Discord"`, `id="talk"`, `<DiscordMock />`, and exactly
  one `"01 — Talk"`.

**Must not appear:**
- Brokered/real-capital claim *shapes* — `paper[\s\-]?trad`, `real (capital|money|cash|funds|dollars)`,
  `go live`, `trade live`, `turn on live trading`, `connect (a|an|your) brokerage` — scanned
  across **every** component, comments included. The bare noun "live trading" is allowed
  (it is a board name).
- `"0[1-9]"` as a quoted string anywhere in `WhyCare.tsx`, comments included.
- `STORY_AGENT_NAME` anywhere except `Test.tsx` — asserted as a **set**, so a new component
  naming it fails.
- `"yours"` in any landing component.
- `"Talk to Agents"` in `app.html`.

**Structural:**
- `SAMPLE_STANDINGS` must still be rendered by some component, and that corpus must contain
  `"DeepSeek V4 Pro"` and `dataKey=`. The chip strip satisfies this.
- Screen 0 must contain `#homeModuleRanking`, `#homeModuleRankList`, `#homeScrollHint`.
- `#landing-stats` must appear exactly once and keep a `scroll-mt-*` greater than
  `--landing-chrome-height` (120px).

## Build and deploy constraints

- **`/` requires a Vite rebuild.** `dashboard/frontend/index.html` is hand-patched build
  output: ~370 lines of auth-gate script, `#landingAuthModal`, `<style id="landing-auth-patch">`
  and the `[data-landing-auth]` delegated handler cannot be produced by `vite build`.
  Recipe in `dashboard/landing/README.md`: `npm install` → `npm run build` → copy
  `dist/public/assets/*` → delete superseded `index-*.{js,css}` → repoint the two refs,
  keeping the four auth markers. Verify by diffing vite's `index.html` against the shipped
  one: **every differing line must be `>`**; any `<` line means vite output was dropped.
- **`/app` requires cache-buster bumps** for whichever of `app.html`'s referenced assets
  change (`styles.css`, `home-page.js`). `test_frontend_fast_boot.py::test_cache_busters_bumped`
  is the single owner and matches exactly.
- Several landing copy guards read the **shipped bundle**, not the TSX. A source edit that
  is never rebuilt leaves them green against stale text.
- `/` deploys via Vercel (~1 min), `/app` via Render (~6 min after backend tests pass).
  Both hosts serve the landing page, so during that window `/` renders differently
  depending on which host is hit.

## Verification

Source-shape guard tests for the new invariants, plus a live browser pass — the clipping
bug PR #357 shipped below 1200px was invisible to DOM probes and only a screenshot caught
it. Read `getComputedStyle().display`, never the `hidden` attribute.

Viewports: 1000×700, 1280×800, 1440×900, 1920×1080, and one mobile width.

Checks:

- The chart column measures **≥60%** of content width at `lg:` and above. The target is
  2/3 (66.7%); the guard sits below it deliberately, so gutters and rounding cannot redden
  a correct layout while a reverted 50/50 split still fails.
- The chart's rendered height is inside the clamp on both surfaces.
- All five legend chips are on one row at 1440 on `/`.
- Nothing is clipped when the columns stack below `lg:` — measured, not inferred. On
  `/app` specifically, screen 0's full content must fit inside `.home-pager-screen` at
  every viewport, since that container is `overflow: hidden` and clips silently with no
  scrollbar.
- Both `/app` fallback paths (`unreachable`, `empty`) draw no chart and keep their
  distinct notes.

## Out of scope

- Making `/`'s chart real. It stays illustrative: `/` is served statically from Vercel, and
  a cross-origin fetch to Render on first paint is a cold-start gamble on the acquisition
  page.
- The season engine (issue #354) and the two open design questions (#355).
- Refreshing `README.md`'s `snapshot.png`, which this change makes stale for the second
  time this month. Filed as a follow-up rather than done inline.
