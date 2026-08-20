# Chart-first landing — design

**Date:** 2026-08-15
**Amended:** 2026-08-16 — the chart heights in the first draft were asserted, not
measured, and **both surfaces failed them**. Every height in this document is now a
browser measurement against the shipped build; see [Appendix: measured budgets](#appendix-measured-budgets).
Two design decisions the first draft left open (the `/app` panel cap, and whether
`/app`'s chart carries baselines) are settled below.
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

**Full-bleed is a ≥1300px effect only — measured.** Tailwind's `container` caps at 1280px
(`xl`) then 1536px (`2xl`), so the hero's left gutter is a function of viewport width:
72px at 1440, 25px at 1600, 185px at 1920 — but **0px at 1280 and below**, where the
container is already flush to the viewport. Escaping the gutter therefore changes nothing
under ~1300px, and the layout there is carried entirely by the 2/3 column split. Do not
describe full-bleed as the mechanism at laptop widths; it is not one.

Two mechanical consequences on `/`, neither of which is "remove a class":

- `Hero.tsx:139` puts `max-w-2xl` on the board column, capping it at 672px. Two-thirds of
  a 1280px container is 853px, so **that cap has to go** or the column silently stays at
  card width and every other change is cosmetic.
- Both columns are `flex-1` children of the single `container mx-auto px-6` div
  (`Hero.tsx:93`). Escaping it on one edge needs a negative inline-start margin
  (`margin-inline-start: calc((100% - 100vw) / 2)` on the chart column, or an equivalent
  `-ms-*` utility), not a class removal — the container is the same element that owns the
  hero's `lg:min-h-[calc(100dvh-var(--landing-chrome-height)-4rem)]` height contract.

**Order the columns with `order-*`, not by reordering the JSX.** The visual ask is chart
left / hero right at `lg:`, chart first when stacked — which reads as "move `<BoardPreview />`
above the copy in source". Doing that puts `BoardPreview`'s `<h2>` ahead of the page's only
`<h1>` in DOM order, so the document's heading outline opens on the board's title. Keep the
`<h1>` block first in source and set `order` at both breakpoints.

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

Same *reading order* on both — caption, chart, key, detail line — over different data
sources, different stacks and different height budgets. Not a shared component; see §3d.

| Element | `/` | `/app` |
|---|---|---|
| Caption bar | One line + `Illustrative example` chip | One line; **no** chip — the data is real |
| Chart | `clamp(300px, calc(100dvh - 390px), 520px)`, axis ticks **14px** | `clamp(140px, 26vh, 280px)`, axis ticks **14px** |
| Key beneath | Chip strip: `● Name +21.0%`, `text-base` | Compact keyed list, `text-base` |
| Detail line | One line stating what is measured | Existing `hm-rank-meta` line, unchanged |

**The two clamps are different numbers and must stay different.** The first draft gave both
surfaces `clamp(320px, 56vh, 520px)`; measured, that overflows on *both*. On `/` it pushes
the card 25–46px below the fold at 768px and 720px viewport heights. On `/app` it asks for
504px inside a panel with 265px left in it. The surfaces have different vertical envelopes,
so they get different formulas — this is the one place in the design where "same on both"
was actively wrong.

- **`/` — `clamp(300px, calc(100dvh - 390px), 520px)`.** The `390px` is not a taste
  constant: it is the card's own non-chart height (~227px: caption bar, chip strip, detail
  line, padding) plus the 120px `--landing-chrome-height` plus ~43px of fold margin. It
  yields 330px at 720px tall through 520px at 1080px, and clears the fold at every tested
  viewport with ≥27px to spare. **Re-derive it if the caption or chip strip changes height** —
  a two-line caption invalidates the constant, and the failure mode is a silently
  half-visible card, not a broken build.
- **`/app` — `clamp(140px, 26vh, 280px)`.** Yields 187–280px across 1280×720 → 1920×1080.
  Smaller than `/`'s because the panel it sits in is bounded by the pager, not the document.

> **⚠ Both formulas above are superseded — 2026-08-16. The source is the code, not this
> section.** Every `clamp(300px, calc(100dvh - 390px), 520px)` in this document is the
> original single-constant form; it was replaced twice during execution and review.
>
> - **`/` is now `clamp(260px, calc(100dvh - var(--board-chart-reserve)), 520px)`** with the
>   reserve **390px at `lg`+ and 590px below**. One constant could not serve both: the card's
>   non-chart height is 218–241px beside the copy but **443px** stacked at 390px wide, where
>   the title, the chip and the caption wrap *and* the chip strip runs to five rows. The
>   `~227px` figure quoted above was measured against a strip that was silently clipping four
>   of its five entries; unclipping it moved the strip from 24px to 152px, the stacked reserve
>   from 480 to 590, and then the **floor** from 300 to 260 — at 844px tall the card had 269px
>   left for a chart whose floor was 300, so the floor, not the reserve, was what put it 95px
>   past the fold on the second pass.
> - **`/app` keeps `clamp(140px, 26vh, 280px)` as a ceiling**, but `.hm-rank-chart` is now
>   `flex: 0 1 auto` with a 132px floor, so it renders shorter than the clamp wherever the
>   panel is tight. The clamp is no longer a prediction of the rendered height — see the
>   corrected table in §3a.
>
> The live values are pinned by `test_landing_chart_first.py` and
> `test_frontend_chart_first_home.py`, and measured by
> `dashboard/scripts/verify_chart_first_layout.py`.

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

**Decision: `/app`'s chart draws the passive baselines; `/app`'s *list* does not.**
`#homeModuleRankList` is fed by `homeModelEntries()` → `isHomeModelEntry()`
(`home-page.js:1423-1437`), which is `is_model || team_badge === 'Model'` — so **baselines
are filtered out of the panel's only data source today**. Build the chart from that source
unchanged and it draws seven model curves with nothing to judge them against, which fails
the one question the chart exists to answer: is +21.0% good? The whole rhetorical point of
`/`'s illustration, recorded in `BoardPreview.tsx:23-26`, is that exactly one model beat
the passive baselines.

So the chart takes a **second, wider selection** off the same payload — models plus the
`buy_hold` and `market_index` baseline entries — and renders the baselines as **dashed,
unranked reference curves**. The list keeps `isHomeModelEntry` and stays models-only.

This is why the split matters rather than being a detail: `app.html:486` ships the pinned
line **"AI models only · ranked by return"**. That sentence describes *the ranking*, and
the ranking stays models-only, so it remains literally true with baselines on the chart.
Ranking the baselines into the list — the other way to get them on screen — would make it
false and require a copy change on a line the design elsewhere relies on being stable
(§4 frees the lede precisely by leaning on it).

### 2. Type scale

| | now | after |
|---|---|---|
| Chart axis ticks | 11px (`/` only — `/app` has no chart) | **14px**, both |
| Chart height — `/` | 210–240px | **`clamp(300px, calc(100dvh - 390px), 520px)`** |
| Chart height — `/app` | — (no chart) | **`clamp(140px, 26vh, 280px)`** |
| Panel title | `text-lg` | `text-xl` |
| Caption | `text-xs` | `text-sm` |
| Standings rows | `text-sm` | **`text-base`** chips |

This table is the *only* thing keeping the two surfaces looking like one product, and
**nothing enforces it** — see "Two implementations of one look" below.

### 3. `/app` screen 0 gets a real chart

`#homeModuleRanking` gains a Chart.js line chart above its list, built from
`entry.equity_curve`. No new endpoint and no new library: `domain/leaderboard/service.py:1251`
already puts `equity_curve` on every entry, `align_equity_curves` (`:1257`) already aligns
them across entries, `js/leaderboard.js:669` already builds curves from exactly this field,
and Chart.js is already loaded on `app.html`.

`#homeModuleRankList` stays — it is guard-pinned — and restyles into a compact keyed row.

#### 3a. The panel has to grow first — there is no room in it today

This is the part the first draft got wrong, and it is not a tuning error. Measured at
1440×900 on the shipped build, `.home-landing-board .home-module` is capped at **520px**
by `styles.css:5312-5316` (`min(520px, calc(100dvh - var(--app-chrome-height) - 160px))`),
and its fixed chrome consumes **all but 265px** of that:

| child | height @1440×900 |
|---|---|
| `.home-module-head` | 44 |
| `.hm-rank-meta` | 49 *(wraps to two lines below 1920)* |
| `.hm-rank-table-head` | 26 |
| `.hm-rank-season` | **62** |
| `.hm-footer-btn` | **36** |
| panel padding | 36 |
| **fixed total** | **253** |

The last two are panel children the first draft never counted — the Season-0 preview note
and the "See both leaderboards" button — and together they are 98px, more than a third of
what is left. Seven standings rows at today's pitch (22px + 8px gap) need **202px**. So the
chart's actual budget inside the current cap is ~50px at 1440×900, and **negative at
1366×768**, where the list already has ~3px of slack with no chart at all.

Shrinking the list is not a way out either: `.home-module-rank-list` is
`flex: 1 1 auto; overflow-y: auto` (`styles.css:6215-6225`), so it does not overflow
visibly — it collapses toward zero and scrolls, producing exactly the state the adjacent
comment at `styles.css:5328-5331` already records as rejected: *"a visible header, a visible
footer and ZERO standings between them, which is worse than a scrollbar."*

> **Decision: lift the 520px cap and let the board panel fill the hero row.**
> ```css
> html[data-nav-page="home"] #homeView .home-landing-board .home-module {
>     height: 100%;
>     min-height: 0;
>     max-height: none;
> }
> ```
> Above 1200px the hero is a flex **row** — copy left, board right — so the 520px cap was
> never load-bearing for the copy column; it was a card-proportion choice made when the
> panel held only a table. With the board as the screen's subject, the panel takes the row.

**⚠ CORRECTED 2026-08-16 — the table below was measured through a broken probe.**
`height: 100%` on the panel is inert unless the board column is *stretched*:
`.home-landing-hero-inner` is `align-items: center`, so the board's cross size came from its
content, the percentage resolved against an indefinite height and fell back to `auto`. The
panel then sized itself to its content and **overran `.home-landing-hero`, which is
`overflow: hidden`** — cut, with no scrollbar. The pass that produced these numbers probed
`#homeScreenLanding` for overflow, but the hero is `height: 100%` of that screen and absorbs
its own overflow, so the screen reported a clean `0` at every viewport. "7/7 rows visible"
was true of the **list**, and the list was outside the **screen**.

The fix adds `align-self: stretch` to `.home-landing-board` and makes `.hm-rank-chart`
`flex: 0 1 auto` with a 132px floor, so the chart — the illustration — yields before the
standings do. Re-measured against a probe aimed at the hero:

| viewport | panel | chart | rows visible | hero clip |
|---|---|---|---|---|
| 1920×1080 | 866 | 280 | 7/7 | 0px |
| 1600×900 | 697 | 224 | 6/7 | 0px |
| 1440×900 | 697 | 212 | 6/7 | 0px |
| 1440×768 | 573 | 139 | 4/7 | 0px |
| 1366×768 | 573 | 139 | 4/7 | 0px |
| 1280×800 | 603 | 156 | 5/7 | 0px |
| 1280×720 | 528 | 132 | 3/7 | 0px |
| 1240×700 | 509 | 132 | 3/7 | 0px |
| 1201×760 | 565 | 132 | 4/7 | 0px |
| 1152×864 | — | 225 | 7/7 | 0px |
| 1024×900 | — | 234 | 7/7 | 0px |

**Seven rows do not fit at every height, and that is the honest outcome.** At 1280×720 the
panel gets 528px against 642px of content (253 chrome + 187 chart at full clamp + 202 rows);
something has to give, and `.home-module-rank-list` is `overflow-y: auto` precisely so it is
the thing that gives. The rows below its fold are scrolled, not lost. Nothing can show all
seven *and* a chart *and* the panel chrome in 528px — the original table only appeared to
because the overflow was being hidden.

Previously recorded (do not restore — every "screen clip: 0px" here is the probe's blind
spot, not a measurement): 1920×1080 722/280, 1600×900 676/234, 1440×900 700/234,
1440×768 666/200, 1366×768 666/200, 1280×800 674/208, 1280×720 653/187, all "7/7, 0px".

Below 1200px the existing `@media (max-width: 1200px)` block already lets
`#homeScreenLanding` scroll and gives the panel `height: auto; min-height: 400px`, so the
stacked case needs no new rule — but it does need the chart to honour its 140px floor
without pushing rows out, which is what the ≤1200px verification below measures.

#### 3b. Fallback honesty — three states, not two

`home-page.js:1464` (at `dashboard/frontend/home-page.js`, **not** `frontend/js/`, where
`leaderboard.js` lives) defines a sample-standings fallback with invented returns, marked by
a *"Sample standings —"* prefix, and distinguishes two reasons (`unreachable` vs `empty`).
A chart is more persuasive than a table, so five invented curves are a larger claim than
that prefix can retract.

> **Decision: render no chart unless real curves arrive.** On either sample path the note
> and the sample list ship exactly as they do today, with no chart.

**The gate is "curves present", not "sample reason is null".** There is a third state the
first draft's phrasing misses: `renderEntries(models)` runs with `sample: null` whenever
`models.length > 0`, **regardless of whether any entry carries an `equity_curve`**, and
`buildEquityCurvesFromEntries` silently drops curveless entries
(`if (!points.length) return;`, `js/leaderboard.js:872`). Real entries with no curves
therefore yield **zero series** — an empty chart with axes, under a real standings list,
carrying no sample note, because the data genuinely is real.

That is this repo's own fail-closed-is-not-fail-visible failure in miniature: *absent* and
*broken* render identically, and the honest-looking path is the one that lies. So the chart
element is created only when the built curve set is non-empty; otherwise the panel renders
exactly as it does today, list and all. Do not key the gate on `sample`, which cannot see
this case.

This follows the repo's fail-closed-is-not-fail-visible doctrine: the two reasons stay
distinguishable, and neither is dressed up as a result.

#### 3c. The chart cannot draw on first paint

Chart.js is a **deferred third-party script** (`app.html:19`, jsDelivr, `defer` +
SRI), and screen 0 is now the first thing `/app` paints. Between first paint and the
script landing there is a window — longer on a free-tier cold start — where the panel
knows its chart's height but has nothing to draw in it.

Reserve nothing. The chart element is inserted when both Chart.js and a non-empty curve
set exist (§3b), so before that the panel lays out exactly as it does today and the list
simply sits higher. A reserved-but-blank 234px box is the worse choice here: it looks like
a chart that failed rather than one that has not arrived, which is the same absent-vs-broken
confusion §3b exists to prevent. Layout shift on arrival is acceptable — it is one shift,
downward, of content the reader has not started reading.

#### 3d. Two implementations of one look — both surfaces

Worth stating plainly, because §2's type-scale table reads like a shared contract and is
not one. `/` is React + Recharts + Tailwind tokens; `/app` is vanilla JS + Chart.js +
`styles.css`. There is **no shared code and no shared token** between them, and after this
change there are two chart implementations with two different height formulas, two axis-tick
declarations, and two legend treatments.

That duplication is forced by the two stacks and is accepted. What is *not* acceptable is
calling it "the same structure" and leaving it unguarded: changing the axis tick to 14px
touches two Recharts props, a Chart.js options object, a cache-buster bump and a Vite
rebuild — and **nothing fails if you do three of the four**. The verification section
therefore pins the numbers that must agree across surfaces (axis tick size, chip type
scale) as source-shape assertions on both files, so the pair drifts loudly or not at all.
Heights are deliberately *excluded* from that pinning: they are different by design (§2).

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

**Superseded 2026-08-20 (PR #394).** Two things above are no longer true: the sentence
shipped here has been replaced, and it *is* pinned now. The current lede is:

> **Think you can beat them? Test your own idea on the same days.**

The fact half went because the headline ("See where each model ranks") and the board's
own meta line were already saying it — three elements describing one board. What did not
go, and what a future edit must not take, is the trailing clause. Neither leaderboard
accepts entries (`get_leaderboard` builds every row from the curated roster in
`dashboard/config/leaderboard.json`; `api/routers/leaderboard.py` has no submission
route), so a bare "Think you can beat them?" under a ranking headline and above a
"Create a free account" button promises a place on the board. Naming the mechanism is
what keeps the challenge true — it also aims at the CTA, which is the job the challenge
was reaching for. Pinned by
`test_the_screen_zero_lede_challenges_and_then_names_the_mechanism`, which asserts the
shape (challenge, then something after the question mark) as well as the string.

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
  `"DeepSeek V4 Pro"` and `dataKey=` (`test_landing_copy_register.py:362-365`). Note the
  corpus is *files containing `SAMPLE_STANDINGS`*, and the only `dataKey=` in that corpus is
  on `BoardPreview.tsx`'s `<Line>` elements — so the guard holds as long as the chart and
  the chip strip stay in the **same file**. Splitting the chip strip into its own component
  reddens it even though nothing was deleted.
- Screen 0 must contain `#homeModuleRanking`, `#homeModuleRankList`, `#homeScrollHint`.
- `#landing-stats` must appear exactly once and keep a `scroll-mt-*` greater than
  `--landing-chrome-height` (120px, `landing/src/index.css:114`).

**Re-verified 2026-08-16.** Every guard above was checked at source, and every one still
holds under the amended design. Two findings from that pass:

- **`Race.tsx` already renders the full standings table** (`Race.tsx:80-112`, headed
  "Competition Standings", consuming the same `SAMPLE_STANDINGS`). §1's "the full table
  survives in `Race.tsx`, which becomes the detail home" therefore describes the *current*
  state and needs **no implementation work** — do not budget for moving a table.
- **The `/app` lede really is unpinned.** A grep for its text across the whole test suite
  returns nothing, and `test_app_copy_register.py:305-311` pins only the three screen-0 ids
  plus the absence of the `"Talk to Agents"` pitch. §4 is safe as written.

## Build and deploy constraints

- **`/` requires a Vite rebuild.** `dashboard/frontend/index.html` is hand-patched build
  output: ~370 lines of auth-gate script, `#landingAuthModal`, `<style id="landing-auth-patch">`
  and the `[data-landing-auth]` delegated handler cannot be produced by `vite build`.
  Recipe in `dashboard/landing/README.md`: `npm install` → `npm run build` → copy
  `dist/public/assets/*` → delete superseded `index-*.{js,css}` → repoint the two refs,
  keeping the four auth markers. Verify by diffing vite's `index.html` against the shipped
  one: **every differing line must be `>`**; any `<` line means vite output was dropped.
- **`/app` requires cache-buster bumps** for whichever of `app.html`'s referenced assets
  change (`styles.css`, `home-page.js` — the latter at `dashboard/frontend/home-page.js`,
  a repo-root-of-frontend file, *not* under `frontend/js/` where `leaderboard.js` lives).
  `test_frontend_fast_boot.py::test_cache_busters_bumped` is the single owner and matches
  exactly.
- Several landing copy guards read the **shipped bundle**, not the TSX. A source edit that
  is never rebuilt leaves them green against stale text.
- `/` deploys via Vercel (~1 min), `/app` via Render (~6 min after backend tests pass).
  Both hosts serve the landing page, so during that window `/` renders differently
  depending on which host is hit.

## Verification

Source-shape guard tests for the new invariants, plus a live browser pass — the clipping
bug PR #357 shipped below 1200px was invisible to DOM probes and only a screenshot caught
it. Read `getComputedStyle().display`, never the `hidden` attribute.

**Viewports.** 1280×720, 1280×800, 1366×768, 1440×768, 1440×900, 1600×900, 1920×1080, and
one mobile width. **1366×768 and 1280×720 are not optional** — they are the two that
falsified the first draft's heights, and both are ordinary laptop sizes. A viewport list
that only samples 900px-tall screens cannot see the failure this design exists to fix.

Checks:

- **Column width.** The chart column measures **≥60% of the hero container's width** at
  `lg:` and above — target 2/3 (66.7%), guard below it so gutters and rounding cannot
  redden a correct layout while a reverted 50/50 split still fails. *State the denominator
  in the test:* the container, not the viewport. Because the column deliberately escapes
  the container's left edge, the same layout measures 66.7% of the container but only
  **63.0–65.9% of the viewport** depending on width (measured), and a guard that silently
  switched denominators would sit within 3pp of its own threshold.
- **Chart height.** The rendered height is inside that surface's clamp — `/` and `/app`
  have **different** clamps (§2); a single shared assertion is a bug.
- **The fold, on `/`.** The `BoardPreview` card's bottom edge is **above** the viewport
  bottom at every listed viewport. This is the check the first draft lacked, and the one
  that fails on `clamp(320px, 56vh, 520px)`.
- **The pager, on `/app`.** `#homeScreenLanding.scrollHeight <= clientHeight` (±1px) at
  every viewport ≥1200px, **and** every one of the 7 rows fully inside the list's visible
  box — `.home-pager-screen` is `overflow: hidden` and clips with no scrollbar, so a
  height assertion on the panel alone cannot see rows disappearing.
- All five legend chips are on one row at 1440 on `/`.
- Nothing is clipped when the columns stack below `lg:` — measured, not inferred.
- **Both `/app` sample paths** (`unreachable`, `empty`) draw no chart and keep their
  distinct notes, **and** the third state — real entries whose `equity_curve` set builds
  to zero series — also draws no chart and shows **no** sample note (§3b). A test that
  only exercises the two `sample` reasons cannot distinguish this case from a working one.
- **Baselines are on `/app`'s chart and not in its list** (§1): the chart's series set
  includes `buy_hold` and `market_index` as dashed curves, while `#homeModuleRankList`
  still contains only `is_model` entries.
- **Cross-surface source-shape pins** (§3d): axis tick size and chip type scale asserted on
  **both** `BoardPreview.tsx` and `home-page.js`, so the pair cannot drift silently.
  Heights are excluded — they differ by design.

**Mutation-test every source-shape guard** before trusting it (revert one fix, run that
single test, restore). PR #352's round wrote 15 and 2 passed against a broken implementation.

## Accepted knowingly

**`/`'s sample curves and sample standings describe different windows.** `SAMPLE_CURVES`
uses a relative-day axis (`"7d ago"` → `"Now"`) and its own comment says it illustrates the
Live Trading Leaderboard, "which advances one session at a time"; the `SAMPLE_STANDINGS`
returns beside it (+21.0%, +5.5%, …) are Competition-window figures over roughly a month,
and `Race.tsx` heads the same array "Competition Standings". The *numbers* agree — the
curves land exactly on the standings' returns — so nothing is contradictory, but the axis
label and the caption ("over the past week") name a different window than the data does.

This is pre-existing and out of scope to fix here. It is recorded because this design
**triples the chart's size and makes that axis the most prominent claim on the acquisition
page**, which raises the cost of leaving it wrong. If the caption is being rewritten anyway
(§2 moves it to `text-sm`), align it with the window the returns actually come from.

## Out of scope

- Making `/`'s chart real. It stays illustrative: `/` is served statically from Vercel, and
  a cross-origin fetch to Render on first paint is a cold-start gamble on the acquisition
  page.
- The season engine (issue #354) and the two open design questions (#355).
- Refreshing `README.md`'s `snapshot.png`, which this change makes stale for the second
  time this month. Filed as a follow-up rather than done inline.

## Appendix: measured budgets

Measured 2026-08-16 against the shipped build (`uvicorn dashboard.backend.app:app` on a
scratch copy of the seed DB, Chromium via Playwright, real leaderboard payload — 7 model
rows, sample note hidden). These are the numbers the design is derived from; re-measure
rather than re-reason if any of the panel's chrome changes.

**`/app` screen 0 — `.home-landing-board .home-module`, as shipped today**

| viewport | panel | head | meta | thead | list | season note | footer btn | left for chart |
|---|---|---|---|---|---|---|---|---|
| 1920×1080 | 520 | 44 | 24 | 26 | 290 | 62 | 36 | **~0** |
| 1440×900 | 520 | 44 | 49 | 26 | 266 | 62 | 36 | **~0** |
| 1366×768 | 459 | 44 | 49 | 26 | 205 | 62 | 36 | **negative** |
| 1280×800 | 491 | 44 | 49 | 26 | 237 | 62 | 36 | **negative** |
| 1000×700 | 414 | 44 | 24 | 26 | 202 | 45 | 36 | **negative** |

Note `.hm-rank-meta` is 24px at 1920 but **49px** at 1440 and below, where it wraps to two
lines. Budget against the wrapped height.

**`/` hero — candidate chart clamps, card bottom vs. fold**

Slack = viewport height − card bottom edge; negative means below the fold. Measured with
the 5-row standings table already replaced by a one-row chip strip.

| clamp | 1920×1080 | 1600×900 | 1440×900 | 1440×768 | 1366×768 | 1280×800 | 1280×720 |
|---|---|---|---|---|---|---|---|
| `clamp(320px, 56vh, 520px)` *(first draft)* | +132 | +50 | +33 | **−25** | **−25** | **−11** | **−46** |
| `clamp(300px, calc(100dvh - 390px), 520px)` ✅ | +132 | +47 | +27 | +27 | +27 | +27 | +27 |
| `clamp(280px, calc(100dvh - 400px), 520px)` | +132 | +52 | +37 | +37 | +37 | +37 | +37 |
| `clamp(260px, calc(100dvh - 420px), 480px)` | +152 | +74 | +53 | +53 | +53 | +53 | +53 |

The chosen formula is the first row that passes everywhere, i.e. the largest chart with
non-negative slack at every viewport. The three failing cells in row 1 are all ordinary
laptop heights.

**Container geometry on `/`** (informs the ≥60% guard's denominator)

| viewport | container width | left gutter | 2/3 column ends at | % of viewport |
|---|---|---|---|---|
| 1920×1080 | 1536 | 185 | x=1209 | 63.0% |
| 1600×900 | 1536 | 25 | x=1049 | 65.6% |
| 1440×900 | 1280 | 73 | x=926 | 64.3% |
| 1366×768 | 1280 | 36 | x=889 | 65.1% |
| 1280×800 | 1265 | **0** | x=843 | 65.9% |

The 0px gutter at 1280 is why full-bleed is a ≥1300px effect only.
