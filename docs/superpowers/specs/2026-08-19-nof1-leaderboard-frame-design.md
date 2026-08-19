# nof1-style leaderboard frame + live data on the landing hero

**Date:** 2026-08-19
**Status:** approved design, not yet planned
**Reference material:** `nof1_screenshot_1.jpg` (Alpha Arena by nof1) and
`wechat_20260817012805_78_188.jpg` (handwritten note), both under
`D:\FinGPT\Materials\Trade Materials`.

---

## 1. What exists today

Three leaderboard charts live in this repo, on three different stacks, drawing
from two different data sources.

| Surface | File | Stack | Data | Gutter today |
|---|---|---|---|---|
| Signed-out landing hero | `dashboard/landing/src/components/home/BoardPreview.tsx` | Vite/React + Recharts | hardcoded `SAMPLE_CURVES` / `SAMPLE_STANDINGS` | none (`margin.right: 10`) |
| Signed-in Home screen 0 | `dashboard/frontend/home-page.js` | vanilla Chart.js | live `GET /api/v1/leaderboard` | none |
| `/app` Leaderboard tab | `dashboard/frontend/js/leaderboard.js` | vanilla Chart.js | live `GET /api/v1/leaderboard` | 120px, `endpointLabelPlugin` |

The Leaderboard tab already implements most of the target pattern: a reserved
right gutter via `layout: { padding: { right: 120 } }`, per-curve endpoint labels
with vertical collision avoidance and dotted leader lines. It is the reference
implementation, not something to invent.

Screen 0 already reuses three exports from the Leaderboard tab
(`window.buildEquityCurvesFromEntries`, `window.getSeriesStyle`,
`window.formatShortDate`), so those two surfaces are already coupled by design.
The landing hero shares nothing with either.

### The reference material

**nof1's Alpha Arena** draws each model's curve ending short of the plot's right
edge, with the remaining width carrying a circular model avatar and a
colour-matched value pill per curve.

**The handwritten note** sketches a *Live Trading Leaderboard*: an axis spanning
a whole session (`Session 8/1 – 8/31`), curves labelled at their right endpoints
(`deepseek`, `Nemo`), each ending in a dot with a dotted continuation stub, a
`Now` marker at today (`Today 8/15`, mid-axis) and an `End` marker at the session
close. Annotated *"update each hour"* and *"refresh each month"* — the refresh
cadence is **two weeks** for this project, not a month. The lower half gives the
mechanism: ① now → ② check timer → ③ *"update by adding one new time stamp"* —
an hourly append, not a re-run.

---

## 2. Goals

1. The signed-out landing hero shows the **actual Competition leaderboard**, the
   same data the signed-in Home screen shows, instead of invented sample curves.
2. All three charts adopt one shared visual frame derived from nof1 and the note:
   curves ending short of the right edge, with the reserved space carrying each
   curve's **owner name and value**.
3. The x-axis reads as forward-moving, so a visitor understands the board
   advances.
4. Seasons are **two weeks** long, expressed somewhere real.

## 3. Non-goals

- **No future date ticks on the x-axis.** Considered and explicitly dropped. It
  would require extending the category-scale domain with null-padded future
  labels, which puts empty territory *inside* `chartArea` — the exact region
  `resolveHoverTarget` resolves hover against. The forward affordance is an
  arrowhead instead (§4.4).
- **No season advance engine.** The note's ①②③ timer mechanism — hourly append,
  two-week rollover, persistence of an advancing curve — is a separate
  workstream. This design ships the frame the engine will later fill.
- **No change to the `/app` Leaderboard tab's hover model.** `events: []`,
  `HOVER_HIT_RADIUS_PX`, `resolveHoverTarget` and the pointer-driven emphasis
  stay exactly as they are. Because the reserved space is `layout.padding` rather
  than domain, `chartArea.right` still marks the end of real data and every
  assumption that code makes remains true.

---

## 4. The shared chart frame

One visual contract, three implementations. The duplication is forced by the
stacks (React/Recharts vs two vanilla Chart.js surfaces) and is accepted — the
existing `test_the_two_surfaces_agree_on_the_numbers_that_must_agree` guard is
the model for keeping it from drifting silently.

```
│←──────────── plot area, curves (60%) ────────────→│←──── gutter (40%) ────→│
                                                     ⬤ DeepSeek V4 Pro +7.49%
                                                     ⬤ SPY             +5.95%
                                                     ⬤ Buy & Hold      +4.87%
─────────────────────────────────────────────────────────────────────────────→
 Apr 15     Apr 22     Apr 29     May 8     May 15
```

### 4.1 The 3:2 split

The plot area occupies **60%** of the chart's usable width; the reserved gutter
takes the remaining **40%**. Implemented as `layout.padding.right` on the two
Chart.js surfaces and `margin.right` on Recharts — computed as a fraction of
measured chart width, not a fixed pixel constant, so the ratio holds across
breakpoints.

This is deliberately more generous than the reference: nof1's own gutter is
roughly 18% of their plot width. The extra whitespace is the point — it is what
reads as room for the board to grow into. Revisit only after seeing it rendered.

**A pixel floor is required.** On a narrow viewport 40% of the chart width is
too little for `⬤ DeepSeek V4 Pro +7.49%`, and the labels clip. The gutter is
therefore `max(40% of chart width, GUTTER_MIN_PX)`.

> **As implemented (PR #382).** The 40% shipped as a *ceiling* on the measured
> floor rather than as the gutter's width: rendered, a fixed 40% reserved 640px
> on a 1600px canvas to hold ~200px of labels, leaving ~440px of the plot as an
> empty column. `boardFrameLayout` computes
> `max(floor, min(width * 0.4, floor + BOARD_GUTTER_SLACK))`.
>
> **Where the fraction actually binds**, since "the gutter takes 40%" now
> overstates it: only while `width * 0.4 < floor + slack`, i.e. below
> `2.5 * floor + 90` px of canvas — about 613px at today's ~209px measured
> floor. Every desktop width this dashboard renders at is above that, so on a
> 1440 or 1600px tab the slack binds and the fraction is inert; it does its job
> on the home panel and on mobile, which is where an unbounded rail would eat
> the plot. `BOARD_GUTTER_MAX_FRACTION` (0.5) is the one width-relative guard
> that fires at any size: past it the frame draws no labels at all.

`GUTTER_MIN_PX` is **derived at implementation time, not guessed**: render the
longest label the roster can produce — `shortName()` truncates at 18 characters,
so the worst case is 18 chars plus the dot, the gap and a `-12.34%` pill — in the
gutter's own font (`600 11px Inter`) and take the measured width plus the
existing `labelX` offset. Record the measured number and the string it came from
in a comment beside the constant, so a later font or formatter change has
something to re-measure against. `BoardPreview.tsx` already carries a cautionary
precedent: its `width={56}` y-axis reserve was measured, the tick font later went
11px → 14px, and four of five labels lost their leading `$`.

On the landing hero the floor also interacts with the card's measured height
reserves — see §9.

### 4.2 Endpoint labels in the gutter

Per visible curve, anchored at its final data point:

- a colour-matched filled dot, using the curve's own colour from
  `getSeriesStyle` / `LINE_COLORS`;
- the owner name, via the existing `shortName()` truncation;
- a **value pill** — a filled rounded rect in the curve's colour with contrasting
  text, carrying the curve's final value.

Vertical collision avoidance, the `GAP` stagger and the dotted leader lines for
displaced labels all already exist in `endpointLabelPlugin`. **Extend that
plugin; do not rewrite it.** Its `LEADER_MIN_DISPLACEMENT` logic and the
`gutterStart` / `labelX` split exist because labels drawn over the line endpoints
left visible stubs.

### 4.3 Endpoint dot and continuation stub

Each curve terminates in a filled dot in its own colour, followed by a short
dotted stub of the same colour extending a few pixels into the gutter — the
note's `•⋯` mark. It signals the curve continues without asserting any value.

### 4.4 The forward arrow

The x-axis line extends through the gutter and terminates in a right-pointing
arrowhead. Drawn in a plugin (Chart.js can draw anywhere on the canvas, not only
inside `chartArea`) / as an SVG overlay (Recharts).

No tick labels are *placed* in the gutter — but the strip below the plot is not
empty at the labels' x, and the implementation must not assume it is.

> **Correction (PR #382).** Chart.js centres the **last** x tick on
> `chartArea.right` and reserves its right-half overhang inside the same
> `layout.padding.right` this design uses for the gutter: roughly 18px at the
> Leaderboard tab's 12px ticks, ~21px at screen 0's 14px ones. That matters
> because endpoint labels legitimately hang *below* `chartArea.bottom` into the
> axis strip — the stagger is routinely taller than the plot — so a descending
> label and the final tick occupy overlapping pixels. Scales draw before
> `afterDatasetsDraw`, so the label always wins them rather than being
> occluded; `BOARD_TICK_CLEARANCE` indents descending labels past the overhang
> so the collision does not arise. The gutter is *reserved* space, not *empty
> canvas*, and the axis arrow's own baseline crosses it too (§4.4) — which is
> why that plugin draws under the labels rather than after them.

**Accepted imprecision, stated:** on the Competition board the arrow implies
advancement that a closed window does not do — that window ran
`2026-04-15 → 2026-05-15` and is finished. The mitigation is that the window
label sits directly above the chart on every surface that draws it. This is a
soft visual affordance rather than a data claim, and it is being accepted
knowingly rather than overlooked.

### 4.5 Units

**Every pill renders its own chart's y-axis unit.** No surface invents a unit its
axis does not show.

| Surface | Axis | Pill |
|---|---|---|
| Landing hero | `%` — **changed**, see §6 | `+7.49%` |
| Screen 0 | `%` | `+7.49%` |
| Leaderboard tab | follows `currentChartView` (`$` default, `%` toggle) | matches |

---

## 5. Landing hero: live data

`BoardPreview.tsx` drops `SAMPLE_CURVES` and `SAMPLE_STANDINGS` and fetches
`GET /api/v1/leaderboard`.

### 5.1 Transport

Follow the precedent already in the bundle: `MarketTicker.tsx`'s `apiBase()`,
which returns `window.location.origin` on localhost and
`https://agentictrading.onrender.com` otherwise, with an `AbortController`
timeout. Do not introduce a second base-URL convention —
`test_frontend_api_base.py` exists.

### 5.2 Entry selection

**Mirror screen 0's `homeChartEntries()` exactly**: every entry with
`is_model || team_badge === 'Model'`, plus the two reference baselines whose
`entry_id` is in `['buy_hold_djia', 'djia_index']`. That is 9 curves of the 12 the
API returns.

This *is* what "sync the signed-out page with the signed-in page" means
concretely. It is not a cosmetic choice: screen 0's own source explains that
drawing seven model curves with no baseline leaves the reader nothing to judge
them against, and the same is true on the hero.

### 5.3 States

Three, and they must be distinguishable:

| State | Render |
|---|---|
| loading | skeleton — axes drawn, shimmer in the plot area |
| loaded | real curves + real standings |
| failed | **a visible chart-shaped message** naming the failure |

The failed state is explicitly *not* a permanent shimmer and explicitly *not* a
fallback to sample curves. A silent fallback would make "the backend is down" and
"the backend is fine" render near-identically — the failure shape CLAUDE.md's
*fail-closed is not fail-visible* section is about, and the same one that
degraded the news panel in prod for hours.

Render's free tier cold-starts in 30–60s, so the loading state is a routine
occurrence on the first visit of the day, not an edge case. It must look
deliberate.

### 5.4 `Race.tsx`

`Race.tsx` imports `SAMPLE_STANDINGS` from `BoardPreview` and renders the full
standings table from it. It moves to the same fetched data. A hero with real
numbers above a table of invented ones, on the same page, is worse than either
alone.

The fetch therefore needs to be shared between the two components rather than
issued twice — a small context or a hook in `src/lib/`.

---

## 6. The units change is forced, not chosen

`test_the_two_surfaces_agree_on_the_numbers_that_must_agree` currently asserts an
asymmetry — `/app` is percent, `/` is dollars — and documents why:

> `/` plots fabricated curves that all share a base of 1000, so `$1210` is
> unambiguous and reads as `SAMPLE_STANDINGS`' +21.0%. Screen 0 plots LIVE
> entries, and every dollar level there is a ×0.1 rescale of a $100,000 backtest
> onto the config's $10,000 display base (leaderboard `service.py`), so a
> `$10,749` tick names an account that never existed while the percent is what
> actually ran.

The justification for the hero's dollar axis is precisely that its curves are
fabricated. This design removes that premise. **The hero's y-axis becomes
percent**, and the guard's asymmetry assertion becomes a symmetry assertion.

Two consequences in `BoardPreview.tsx`:

- `tickFormatter={(v) => `$${v}`}` becomes a percent formatter, one decimal
  (matching screen 0's axis; the tooltip and pill use two, matching the rank
  rows).
- `domain={[960, 1240]}` — a hardcoded dollar domain — is removed. Real returns
  span roughly −0.43% to +7.49%, so the domain must be derived.
- The `width={56}` y-axis reserve was measured against `$1030` at 14px. Percent
  labels are a different width and it must be **re-measured**, not assumed. The
  file already documents that four of five labels had their leading `$` sliced
  when this was last wrong.

---

## 7. Backend: the season block

Independent of the chart work — nothing above blocks on it, and it blocks
nothing above. It exists so "seasons are two weeks" is expressed in a real
contract rather than only in conversation.

`dashboard/backend/domain/leaderboard/service.py`:

- `VALID_PERIODS = ("contest", "daily", "live")`.
- `period=live` returns today's entries (the preview curves) with a `season`
  object attached.

`dashboard/config/leaderboard.json` gains a `season` block:

```json
"season": {
  "length_trading_days": 10,
  "season_zero_start": "2026-08-12"
}
```

Ten trading days is two calendar weeks of US cash sessions, Monday through
Friday. This is not a new number: `dashboard/frontend/js/leaderboard.js:282`
already declares `const SEASON_TRADING_DAYS = 10;` with exactly that comment. The
client contract for seasons is **already fully written and entirely
unimplemented server-side** — the render path already reads `season.number`,
`.status`, `.start_date`, `.end_date`, `.last_advanced_date`,
`.trading_days_elapsed`, `.entry_closes_at`, `.entry_count`, `.next_advance_at`
and `.gaps`. This change fills in fields the client already asks for.

### The one invariant that must hold

`season.last_advanced_date` stays `null` and `season.trading_days_elapsed` stays
`0`, because no season has advanced.

`seasonHasAdvanced()` is the anchor under every preview disclaimer on the Live
tab, and it tests those two fields specifically — *not* `payload.period !== 'live'`
— precisely so that adding `"live"` to `VALID_PERIODS` cannot clear the banner.
That was designed for this commit. Emitting a non-null `last_advanced_date`
before an engine exists would flip the badge to "Running" and promise a nightly
advance that nothing performs.

---

## 8. Guard tests affected

These red as a direct result of the work. They are part of it, not discoveries
for CI to make.

| Test | Why | Resolution |
|---|---|---|
| `test_landing_copy_register.py::test_no_landing_component_puts_a_user_agent_on_the_board` | Derives its corpus from files containing `SAMPLE_STANDINGS`, then asserts `"DeepSeek V4 Pro"` and `"dataKey="` are present. Both anchors disappear. | Re-anchor the corpus on whatever identifies a board-drawing component after the change. Keep it a **set** assertion, not a filename. |
| `test_landing_copy_register.py::test_illustrative_example_label_appears_at_least_twice` | The hero's "Illustrative example" chip must be **removed** — the data is no longer illustrative. Count drops. | Replace with a guard on whatever labels the real data's provenance (an "as of" / window label). |
| `test_landing_copy_register.py::test_race_sample_cards_have_no_live_pulse` | Scoped to Race's sample cards, which stop being samples. | Re-scope. |
| `test_landing_chart_first.py::test_the_two_surfaces_agree_on_the_numbers_that_must_agree` | Asserts `/` is dollars and `/app` is percent. §6 makes both percent. | Invert to a symmetry assertion; keep the rationale comment, rewritten. |
| `test_landing_chart_first.py::test_the_standings_table_becomes_a_chip_strip_that_can_show_every_chip` | Chip strip becomes data-driven; entry count goes 5 → 9. | Re-derive the wrap/measurement assertions. |
| `test_landing_chart_first.py::test_the_landing_chart_uses_its_own_measured_clamp` / `test_landing_chart_axis_ticks_are_14px` | Chart geometry changes. | Re-measure. |
| `test_frontend_leaderboard_hover.py` | Gutter width changes. | Verify the hover gate still clears in the wider gutter. |
| `test_frontend_live_trading_board.py` | `period=live` becomes a real period. | Extend to cover the season payload and the still-showing preview banner. |
| `test_landing_copy_register.py::test_race_source_and_shipped_bundle_agree` | Anchors a string in both `Race.tsx` and the shipped bundle so a missing rebuild fails. | **Do not weaken.** It is the only thing that catches a skipped `npm run build`. |

---

## 9. Risks and constraints

**The landing bundle is hand-patched.** `dashboard/frontend/index.html` is *not*
`vite build` output — the Vite template is 25 lines, the shipped file is 418.
The extra ~393 lines (auth-gate script, `#landingAuthModal`,
`<style id="landing-auth-patch">`, the delegated `[data-landing-auth]` handler)
must be re-applied by hand on every bundle refresh. Copying `dist/index.html`
over it silently kills every landing CTA with no console error. Tracked as issue
**#225**; guarded by `test_frontend_bundle_integrity.py`.

The bundle **is** byte-reproducible, so the definitive check on the landing PR is:
build from that branch's source and confirm the emitted asset hash equals the
committed filename.

```bash
cd dashboard/landing && npm ci && npm run build
sha256sum dist/public/assets/index-*.js
```

**Nothing in CI builds or type-checks the landing source.** Only CodeQL reads the
TS/JS. The correspondence between `frontend/assets/index-*.js` and `landing/src`
rests entirely on the author performing the rebuild.

**The card's height reserves are measured constants.** `BoardPreview.tsx` carries
two — `--board-chart-reserve` at 590px stacked and 390px at `lg+` — derived from
the measured height of the title, the chip, the caption and the chip strip. The
chip strip going from 5 entries to 9 changes its wrapped height, and removing the
"Illustrative example" chip changes it again. **Both reserves must be
re-derived.** The failure mode is a silently half-visible card below the fold,
not a broken build.

**The real board is visually flat.** Competition returns span −0.43% to +7.49%.
nof1's screenshot spans −34% to +34%. The dramatic fan-out that makes the
reference image compelling will not appear. Not a blocker; not a reason to
rescale the axis into implying more movement than occurred.

**Colour map pollution.** `getModelColor` mints a palette slot per unseen key and
`modelColorMap` is module-level state shared with the Leaderboard tab. Any new
code path that feeds it entries must pass real `entry_id` values from
`leaderboard.json`, never display labels — a previous mock did exactly this and
made the Leaderboard tab's curve colours depend on whether the home module had
failed earlier in the session.

**Nine curves in a hero card.** Screen 0's selection rule yields 9 series
(verified against the live payload: 7 models + `buy_hold_djia` + `djia_index`).
The hero card is roughly 850px wide at `lg+`; after the y-axis reserve and a 40%
gutter the curves have well under half that to separate in, and the board's real
spread is under 8 percentage points. Legibility at that density is **unverified**
and must be checked in a browser before the PR, not after. If it fails, the fix
is fewer curves — not a rescaled axis, which would imply more movement than
occurred.

---

## 10. Delivery

Three pieces, deliberately independent so none blocks the others.

| PR | Contents | Blocked by |
|---|---|---|
| **A** | Shared frame on both Chart.js surfaces: `js/leaderboard.js` (extend `endpointLabelPlugin`, widen gutter to 3:2, arrow, dot + stub) and `home-page.js` screen 0 (same frame, new). | — |
| **B** | `BoardPreview.tsx` + `Race.tsx` → live data, shared fetch, three states, percent axis, same frame. Bundle rebuild + hand-patch. Guard-test updates. | A, for the visual rule only |
| **C** | Backend `season` block + `live` period. | — |

## 11. Verification

- `pytest dashboard/backend/tests/ -v` green, including every guard in §8
  **updated rather than deleted**.
- Landing bundle rebuilt and hash-verified per §9.
- All three charts checked in a browser at 1920, 1440, 1280 and 390 wide —
  specifically: gutter labels not clipped, the hero card's bottom edge above the
  fold, and the Leaderboard tab's hover emphasis still clearing when the pointer
  enters the widened gutter.
- `period=live` returns a season payload **and the Live tab still shows the
  Season 0 preview banner** (§7).

## 12. Follow-ups, explicitly out of scope

- The season advance engine — the note's ①②③ hourly-append mechanism, two-week
  rollover, and persistence for an advancing live curve. This design ships the
  frame it will fill; it needs no further chart work when it lands.
- Issue **#225** — the hand-patched bundle. This work performs the hand-patch
  again rather than fixing the underlying problem.
- Model avatar icons. nof1 uses circular per-model logos in the gutter; this
  design uses colour dots, which is the existing visual language on all three
  surfaces.
