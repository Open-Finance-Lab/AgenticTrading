# Chart-First Landing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an equity-curve chart the subject of both entry surfaces — `/` (marketing) and `/app` screen 0 — at a size and type scale that can be read, without pushing anything below the fold or into a pager's clip region.

**Architecture:** Two surfaces, two stacks, no shared code (accepted — spec §3d). `/app` gains a real Chart.js chart fed by the leaderboard payload's existing `equity_curve` field, gated on curves actually arriving; `/` keeps its illustrative Recharts chart but triples its size and demotes the standings table to a legend strip. Both need their container to grow *before* the chart exists — on `/app` a 520px panel cap, on `/` a `max-w-2xl` column cap — which is why layout tasks precede chart tasks in each phase.

**Tech Stack:** Vanilla JS + Chart.js 4.4.0 + `styles.css` (`/app`); React + Recharts + Tailwind via Vite (`/`); pytest source-shape guards + node-under-pytest behavioural tests; Playwright for the measured layout pass.

**Spec:** `docs/superpowers/specs/2026-08-15-chart-first-landing-design.md` (amended 2026-08-16 — every height in it is a browser measurement; do not re-derive by arithmetic).

> **⚠ EXECUTED, THEN CORRECTED IN REVIEW (2026-08-16). The task bodies below are the
> historical record of what was written; several of their constants and code blocks no
> longer ship.** Read the code and the spec's §3a table for current values. What changed and
> why:
>
> | Task | Shipped as written | Corrected to |
> |---|---|---|
> | 1 (`/app` panel) | `height: 100%` on the panel | + `align-self: stretch` on `.home-landing-board` — without it the percentage resolved against an indefinite height, and the panel was **cut** by the hero's `overflow: hidden` at four viewports |
> | 3 (`/app` chart) | `flex: 0 0 auto` | `flex: 0 1 auto` + a 132px floor, so the chart yields before the standings |
> | 5 (`/` clamp) | `clamp(300px, … 390px …, 520px)` | `clamp(260px, … var(--board-chart-reserve) …, 520px)`, reserve 590/390 |
> | 8 (chip strip) | `flex-nowrap … overflow-hidden` | `flex-wrap` — the strip was clipping one to four of its five chips below ~1100px |
> | 12 (measurement) | probes `#homeScreenLanding` | probes `.home-landing-hero` — **the screen cannot see this clip**, which is why Task 12 passed over it |
>
> The lesson worth carrying forward is Task 12's: a measurement pass is only as good as the
> element it measures. This one aimed one node above the `overflow: hidden` ancestor and
> reported a clean sweep across a layout that was visibly cut at 1280×720.

---

## Why this is one plan and not two

The chart is not separable from "making room for the chart", on either surface:

- **The heights were measured as a pair.** `/`'s `clamp(300px, calc(100dvh - 390px), 520px)` was measured *with the 5-row standings table already collapsed to a chip strip*; `/app`'s `clamp(140px, 26vh, 280px)` was measured *with the 520px panel cap already lifted*. Ship either chart without its companion change and the card lands 100+px below the fold on `/`, or the pager clips standings rows on `/app` — silently, with no scrollbar and no error.
- **On `/app` there is no room at all today.** Measured, the panel has ~0px free at 1440×900 and a *negative* budget at 1366×768. §3a is the chart's precondition, not its sibling.
- **Two plans would become two PRs, and this repo cannot hold a stack.** `main` has no branch protection and the observed norm is that any collaborator merges any clean PR at any moment; stacked PRs here run no CI and no CodeQL. A split invites the chart half to land without the room half.

The plan is phased so the two *surfaces* stay separable (Phase A ships and deploys independently of Phase B), which is the seam that actually exists.

---

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the spec's "Guard constraints" section, which was re-verified at source on 2026-08-16.

**Must ship verbatim (landing, `/`):**
- `Every test here uses simulated money. Real money is involved only if you explicitly connect a brokerage account and turn on live trading.` — pinned twice (`test_no_real_money_sentence_is_present_verbatim` and the `_CLAIM_DISCLAIMERS` allowlist staleness check). May move between components; may not be reworded.
- `Illustrative example` — must appear **≥2× in the minified bundle**. esbuild interns a shared constant once, so **keep the literal duplicated at each site. Do not DRY it.**
- `in preview for Season 0` and `Season 1 is the first that counts` — bundle-wide.
- `Live Trading Leaderboard` — in **`Race.tsx` source** specifically, as well as the bundle.
- `Standings` and `Leaderboard` — in the bundle.
- WhyCare headings, unchanged: `Describe it in plain English`, `Prove it on real market data`, `See how it ranks`, `Pick the AI model`, `For developers: bring your own agent`.
- Talk: `Describe your idea`, `Discord`, `id="talk"`, `<DiscordMock />`, and exactly one `01 — Talk`.

**Must not appear (scanned across every landing component, comments included):**
- `paper[\s\-]?trad`, `real (capital|money|cash|funds|dollars)`, `go live`, `trade live`, `turn on live trading`, `connect (a|an|your) brokerage`. The bare noun `live trading` is allowed — it is a board name.
- `"0[1-9]"` as a quoted string anywhere in `WhyCare.tsx`.
- `STORY_AGENT_NAME` anywhere except `Test.tsx` (asserted as a **set** — a new component naming it fails).
- `yours` in any landing component.
- `Talk to Agents` in `app.html`.

**Structural:**
- `SAMPLE_STANDINGS` must still be rendered by some component, and the corpus of *files containing `SAMPLE_STANDINGS`* must contain `DeepSeek V4 Pro` and `dataKey=`. The only `dataKey=` in that corpus is on `BoardPreview.tsx`'s `<Line>` elements — **so the chart and the chip strip must stay in the same file.** Splitting the chip strip into its own component reddens this guard even though nothing was deleted.
- Screen 0 (`/app`) must contain `#homeModuleRanking`, `#homeModuleRankList`, `#homeScrollHint`.
- `#landing-stats` appears exactly once and keeps a `scroll-mt-*` greater than `--landing-chrome-height` (120px, `landing/src/index.css:114`).

**Cache busters (`/app`):** `test_frontend_fast_boot.py::test_cache_busters_bumped` is the **single owner** and matches **exactly**. Do not add a second cache-buster guard in a feature suite. Bump each asset **once per branch**, in the first task that touches it, and update the exact assertion in the same commit. Current floor, **re-read off `main` at `30c74e0`** (PR #344 advanced two of them after this plan was drafted): `app.js?v=115`, `styles.css?v=115`, `js/leaderboard.js?v=28`, `home-page.js?v=47`, `js/credits.js?v=2`. **Re-read this list before Task 1** — `command grep -n '?v=' dashboard/backend/tests/test_frontend_fast_boot.py`. The guard compares `app.html` against the test file, *not* against `main`, so a bump to a version `main` already serves is green **and** a no-op: browsers keep the cached asset and the whole branch ships invisibly. That failure is silent in CI and only visible as "my CSS change didn't appear in prod".

**Heights are per-surface and must never share an assertion.** `/` and `/app` have different clamps by design (spec §2). A single shared height assertion is a bug, not a simplification.

**Units: the `/app` chart plots percent return, never dollars.** *(Amendment 2026-08-16; **justification corrected 2026-08-16** after measurement — see "Amendments".)* Two reasons, both about what the labels **mean**. Neither is a scale-safety argument.

1. **The chart and its own key must speak one language.** Screen 0's rank list renders `+7.49%` via `homeFormatReturnPct` (`home-page.js:719`, called at `:1686`), and Task 4 makes that list **the chart's key**. A dollar y-axis beside a percent key labels the curve in units the key does not use, in the one panel where the two are meant to read as a single object.
2. **The dollar figures are synthetic; the percent is the only literally true number.** Every published curve was computed at **$100,000** — all 12 `lb_*` rows store `initial_equity = 100000` — while `dashboard/config/leaderboard.json` says `initial_capital: 10000`. `get_leaderboard` therefore multiplies every dollar level by `scale = display_capital / stored_initial` = 0.1 before serving (`service.py:1204-1230`). A `$10,749` axis tick describes an account that never existed. `cumulative_return` is read straight off the stored run (`service.py:1242`) and is untouched by the rescale, so `+7.49%` is exactly what the backtest produced.

**What this is NOT a defence against — do not re-derive the earlier reasoning.** The first draft of this constraint claimed a dollar axis would draw a 10× break under issue #365. **That is false, and it was measured false**, by serving a hand-built database in which one baseline sat at $10k while the models stayed at $100k: every curve still arrived at the client opening at $10,000. `get_leaderboard` rescales **each** entry by **its own** stored `initial_equity` and then reports `"initial_equity": display_capital` for **every** entry (`service.py:1196`, `:1204-1211`, `:1240`); `chart_equity_curve` opens every series at that same value (`baselines.py:115-145`). The served payload cannot carry a mixed-capital scale break, so on this data dollars and percent are an **affine transform of each other** — identical shapes, different tick labels. The choice is about labels and about matching the key, which is sufficient on its own.

**#365 is still real, and percent does not fix it either.** Its damage is to the *returns*, not the axis: a baseline recomputed at $10k trades in a much coarser share quantum (one DJIA share ≈ 2.5% of equity, several names unbuyable), so its curve genuinely differs from the $100k curves it is ranked against. No y-axis survives that. **Still do not force-refresh the board while building this** — the reason is corrupted comparability, not a broken chart.

Correspondingly, **`isHomeModelEntry` is not "the only thing hiding #365 here."** The rescale hides the scale half, and nothing hides the returns half. Task 2 removes the filter because the chart cannot answer "is +21% good?" without a reference line; that is reason enough.

**`/` may keep its dollar axis.** `SAMPLE_CURVES` is fabricated and every series shares a base of 1000, so `$1210` is unambiguous and reads as the `+21.0%` in `SAMPLE_STANDINGS`. The surfaces diverge here because one is illustrative with an honest base and the other relabels a rescaled one. Task 11 records the divergence so nobody "fixes" it by putting `/app` back into dollars.

**Build:** a source edit under `dashboard/landing/src` that is never rebuilt leaves the bundle-reading guards green against stale text. Phase B is not done until Task 10 runs.

---

## File Structure

**Modified — `/app` (Phase A):**
- `dashboard/frontend/styles.css` — panel cap (≈5310-5316), new `.hm-rank-chart` block, compact `.home-module-rank-list` rows.
- `dashboard/frontend/home-page.js` — new top-level pure functions (`homeChartEntries`, `homeChartSeries`), chart render, list restyle. **Top-level, not nested inside `loadHomeLeaderboardModule`** — nested closures cannot be extracted for node-under-pytest, and the gate logic in §3b is the part most worth testing.
- `dashboard/frontend/js/leaderboard.js` — explicit `window.` export of the curve builder; widen `MODEL_COLOR_PALETTE`.
- `dashboard/frontend/app.html` — lede copy, cache busters.

**Modified — `/` (Phase B):**
- `dashboard/landing/src/components/home/Hero.tsx` — column widths, ordering, full-bleed, copy trim.
- `dashboard/landing/src/components/home/BoardPreview.tsx` — chart clamp, 14px axes, chip strip. **One file — see the `dataKey=` constraint above.**
- `dashboard/landing/src/components/home/{WhyCare,Talk,Test}.tsx` — copy trim.
- `dashboard/frontend/index.html` + `dashboard/frontend/assets/*` — rebuilt bundle (Task 10).

**Created:**
- `dashboard/backend/tests/test_frontend_chart_first_home.py` — all `/app` guards for this change (CSS shape, gate behaviour under node, list/chart split).
- `dashboard/backend/tests/test_landing_chart_first.py` — all `/` guards (clamp string, 14px axes, chip strip, column split).
- `dashboard/scripts/verify_chart_first_layout.py` — the Playwright measurement pass (Task 12). Committed because the spec's acceptance criteria are measurements, and a measurement nobody can re-run is an assertion.

**Untouched deliberately:** `Race.tsx` (already renders the full standings table at `:80-112` — verified; budget no work), `Navbar`, `FooterCTA`, section order.

---

# Phase A — `/app` screen 0

Ships and deploys independently of Phase B. Render, ~6 min after backend tests pass on `main`.

---

### Task 1: The board panel fills the hero row

Spec §3a. Nothing else in Phase A fits until this lands: measured, the panel has ~0px free at 1440×900 and a negative budget at 1366×768.

**Files:**
- Modify: `dashboard/frontend/styles.css:5310-5316`
- Modify: `dashboard/frontend/app.html` (cache buster for `styles.css`)
- Modify: `dashboard/backend/tests/test_frontend_fast_boot.py:194`
- Test: `dashboard/backend/tests/test_frontend_chart_first_home.py` (create)

**Interfaces:**
- Produces: a `.home-landing-board .home-module` with no fixed height cap above 1200px, so Tasks 3 and 4 have vertical budget to spend.

- [x] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_frontend_chart_first_home.py`:

```python
"""Guards for the chart-first rebuild of /app screen 0 (2026-08-15 spec).

Screen 0 lives inside `.home-pager-screen`, which is `height:100%;
overflow:hidden` in a scroll-snap pager: it CLIPS rather than scrolls, with no
scrollbar and no error. Every constraint here exists because the failure mode is
silent -- rows vanish, the chart is a blank box, and nothing logs.

The behavioural cases run the real extracted functions under node, following
test_frontend_leaderboard_hover.py. The source-shape cases guard the seams that
node cannot see (CSS, DOM insertion points, cross-file globals).
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import STYLES, css_blocks

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_HOME_JS = (_FRONTEND / "home-page.js").read_text(encoding="utf-8")
_LEADERBOARD_JS = (_FRONTEND / "js" / "leaderboard.js").read_text(encoding="utf-8")

_PANEL_SELECTOR = (
    'html[data-nav-page="home"] #homeView .home-landing-board .home-module'
)


def _panel_block() -> str:
    """The unscoped (>1200px) rule for the board panel.

    `css_blocks` returns every block with this prelude; the ≤1200px media query
    re-declares the same selector, so taking [0] rather than the whole list is
    what makes "the cap is gone" mean the desktop cap and not the stacked one.
    """
    blocks = css_blocks(_PANEL_SELECTOR)
    assert blocks, "the board panel rule was renamed or deleted"
    return blocks[0]


def test_board_panel_is_not_capped_at_a_fixed_height():
    """Measured at 1440x900, the panel's own chrome (head, meta, table head,
    Season-0 note, footer button, padding) consumes 253px of a 520px cap, and
    seven standings rows need 202px -- leaving ~0px for a chart, and a negative
    budget at 1366x768. The cap was a card-proportion choice from when the panel
    held only a table; the board is the screen's subject now and takes the row.
    """
    block = _panel_block()
    assert "height: 100%" in block
    assert "min-height: 0" in block
    assert "max-height: none" in block
    assert "520px" not in block, (
        "the 520px cap leaves ~0px for the chart at 1440x900 and is negative at 1366x768"
    )
```

- [x] **Step 2: Run it and watch it fail**

```bash
pytest dashboard/backend/tests/test_frontend_chart_first_home.py::test_board_panel_is_not_capped_at_a_fixed_height -v
```

Expected: FAIL — `assert "height: 100%" in block`, because the block still reads `height: min(520px, ...)`.

- [x] **Step 3: Lift the cap**

In `dashboard/frontend/styles.css`, replace the block at 5310-5316:

```css
/* The pager clips rather than scrolls (.home-pager-screen is overflow:hidden),
   so the panel is sized against the viewport, not left to its content.
   It takes the whole hero row: measured at 1440x900 the panel's fixed chrome
   (head 44 + meta 49 + table head 26 + Season-0 note 62 + footer button 36 +
   padding 36 = 253) leaves 267px of a 520px cap, and seven rows need 202 -- so
   under the old cap a chart had ~0px, and a negative budget at 1366x768.
   Above 1200px the hero is a flex ROW, so the cap was never load-bearing for
   the copy column; it was a card proportion chosen when this panel held only a
   table. Verified: panel 653-722px and 7/7 rows visible with zero clip across
   1280x720 -> 1920x1080. Re-measure rather than re-reason if the chrome above
   changes -- every height in this design is a browser measurement. */
html[data-nav-page="home"] #homeView .home-landing-board .home-module {
    height: 100%;
    min-height: 0;
    max-height: none;
}
```

- [x] **Step 4: Run the test and watch it pass**

```bash
pytest dashboard/backend/tests/test_frontend_chart_first_home.py -v
```

Expected: PASS.

- [x] **Step 5: Bump the `styles.css` cache buster (once per branch)**

In `dashboard/frontend/app.html`, change `styles.css?v=115` to `styles.css?v=116`.
In `dashboard/backend/tests/test_frontend_fast_boot.py:194`, change the assertion to `assert "styles.css?v=116" in APP_HTML`.

⚠ **116 is correct only if `main` still serves 115.** Confirm with the grep in Global Constraints before editing; if another merge has advanced it, go one above whatever ships — never reuse a live version. Edit **only** the `styles.css` assertion line; `js/credits.js?v=2` sits below it and is not yours to touch.

Later Phase A tasks also edit `styles.css` — **do not bump again.** The invariant is that the shipped `?v=` is ahead of what `main` serves, and one bump per branch satisfies it. A second bump makes every open PR's exact-match assertion conflict for no gain.

- [x] **Step 6: Run the fast-boot suite**

```bash
pytest dashboard/backend/tests/test_frontend_fast_boot.py -v
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add dashboard/frontend/styles.css dashboard/frontend/app.html \
        dashboard/backend/tests/test_frontend_chart_first_home.py \
        dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "fix(home): board panel fills the hero row instead of a 520px cap"
```

---

### Task 2: The chart's series selection and its curves-present gate

Spec §3b. Pure functions, no DOM — this is the logic worth testing, and node-under-pytest can run it in CI.

**Files:**
- Modify: `dashboard/frontend/home-page.js` (add top-level functions above `loadHomeLeaderboardModule`)
- Modify: `dashboard/frontend/js/leaderboard.js:1599-1600` (explicit export)
- Modify: `dashboard/frontend/app.html` (cache busters for `home-page.js` and `js/leaderboard.js`)
- Modify: `dashboard/backend/tests/test_frontend_fast_boot.py:195-196`
- Test: `dashboard/backend/tests/test_frontend_chart_first_home.py` (extend)

**Interfaces:**
- Consumes: `window.buildEquityCurvesFromEntries(entries)` from `js/leaderboard.js`, returning `{ times, curves, trajectories, initials }`.
- Produces:
  - `HOME_CHART_BASELINE_IDS: string[]`
  - `homeChartEntries(entries): Entry[]`
  - `homeChartSeries(entries, build): {times: string[], series: Array<{label, values, color, dash, isBaseline}>}` — **`series.length === 0` means draw nothing.** Task 3 keys the canvas's existence on it. Returning `times` alongside is what stops the call site building the curve set twice.
  - **`values` are fractions, not dollars** (`0.0749` = +7.49%), each divided by *its own* entry's `initial_equity`. See "Units" in Global Constraints for why a shared dollar axis is wrong here. This is the one place the normalisation happens — Task 3 renders what it is given.

- [x] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_frontend_chart_first_home.py`:

```python
_requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _extract(source: str, name: str) -> str:
    """`function <name>(...) { ... }`, brace-matched, from `source`.

    Extracted rather than restated so a rename or deletion fails these tests
    instead of leaving them green against a copy that no longer ships.
    """
    marker = f"function {name}("
    start = source.index(marker)
    index = source.index("{", source.index(")", start))
    depth = 0
    while True:
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
        index += 1


def _const_block(source: str, name: str) -> str:
    """`const <name> = {...};` or `= [...];`, bracket-matched.

    A line-based reader returns the first line of a multi-line literal, which is
    syntactically incomplete: node then fails with a SyntaxError that reads like
    the code under test is broken. MODEL_COLOR_PALETTE and LEADERBOARD_STYLES
    are both multi-line.
    """
    start = source.index(f"const {name}")
    opener = min(
        i for i in (source.find("{", start), source.find("[", start)) if i != -1
    )
    closer = {"{": "}", "[": "]"}[source[opener]]
    depth, index = 0, opener
    while True:
        if source[index] == source[opener]:
            depth += 1
        elif source[index] == closer:
            depth -= 1
            if depth == 0:
                return source[start : index + 1] + ";"
        index += 1


def _harness() -> str:
    """Everything the extracted functions close over, in dependency order.

    Lifted from the shipped files rather than stubbed: a stub of
    `LEADERBOARD_STYLES` would quietly test the stub's dash patterns instead of
    the ones that ship, which is exactly the assertion these tests exist to make.
    """
    return "\n".join(
        [
            _const_block(_LEADERBOARD_JS, "LEADERBOARD_STYLES"),
            _const_block(_LEADERBOARD_JS, "MODEL_COLOR_PALETTE"),
            _const_block(_LEADERBOARD_JS, "TEAM_COLOR_PALETTE"),
            "const modelColorMap = {}; const teamColorMap = {};",
            _extract(_LEADERBOARD_JS, "isModelEntry"),
            _extract(_LEADERBOARD_JS, "getModelColor"),
            _extract(_LEADERBOARD_JS, "getTeamColor"),
            _extract(_LEADERBOARD_JS, "getSeriesStyle"),
            _extract(_LEADERBOARD_JS, "chartTimeKey"),
            # The real builder, so the gate is exercised against the actual
            # "silently drops curveless entries" behaviour rather than a stub
            # that would drop them the way the test author assumed.
            _extract(_LEADERBOARD_JS, "buildEquityCurvesFromEntries"),
            # The leaderboard tab's percent formula, so screen 0's copy of it
            # can be checked for equivalence rather than eyeballed. Pure and
            # closure-free (curveValues, viewType, initialValue).
            _extract(_LEADERBOARD_JS, "transformLeaderboardChartData"),
            _const_block(_HOME_JS, "HOME_CHART_BASELINE_IDS"),
            _extract(_HOME_JS, "homeChartEntries"),
            _extract(_HOME_JS, "homeChartSeries"),
        ]
    )


def _run_node(expr: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    script = f"{_harness()}\nconsole.log(JSON.stringify({expr}));"
    out = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def _entry(entry_id, model, *, is_model, curve, initial_equity=10000):
    """`initial_equity` is a parameter, not a constant, on purpose.

    A fixture where every row shares one capital base cannot fail on mixed
    capital, and mixed capital is a live condition on this board: the config
    says 10000, the published curves were computed at 100000, and
    `_find_cached_run` does not key on it (issue #365). The default keeps the
    existing cases unchanged; the mixed case below passes 100000 explicitly.
    """
    points = [
        {"timestamp": f"2026-04-{15 + i:02d}T14:00:00+00:00", "equity": v}
        for i, v in enumerate(curve)
    ]
    return {
        "entry_id": entry_id,
        "model": model,
        "team_name": model,
        "is_model": is_model,
        "team_badge": "Model" if is_model else "Baseline Strategy",
        "equity_curve": points,
        "initial_equity": initial_equity,
    }


@_requires_node
def test_chart_draws_the_baselines_the_rank_list_filters_out():
    """`isHomeModelEntry` is `is_model || team_badge === 'Model'`, so the rank
    list's source has no baselines in it at all. A chart built from that source
    draws seven model curves with nothing to judge them against, which fails the
    one question the chart exists to answer: is +21.0% good?
    """
    entries = [
        _entry("deepseek_v4_pro", "DeepSeek V4 Pro", is_model=True, curve=[10000, 12100]),
        _entry("buy_hold_djia", "Buy & Hold", is_model=False, curve=[10000, 10550]),
        _entry("djia_index", "DJIA", is_model=False, curve=[10000, 10280]),
        _entry("mean_variance_djia", "Mean-Variance", is_model=False, curve=[10000, 10100]),
    ]
    labels = _run_node(
        f"homeChartSeries({json.dumps(entries)}, buildEquityCurvesFromEntries)"
        ".series.map(s => s.label)"
    )
    assert "DeepSeek V4 Pro" in labels
    assert "Buy & Hold" in labels, "the chart must carry a strategy baseline"
    assert "DJIA" in labels, "the chart must carry an index baseline"
    assert "Mean-Variance" not in labels, (
        "two reference curves, not five -- the panel's chart is 187-280px tall"
    )


@_requires_node
def test_baselines_are_dashed_and_models_are_not():
    entries = [
        _entry("deepseek_v4_pro", "DeepSeek V4 Pro", is_model=True, curve=[10000, 12100]),
        _entry("buy_hold_djia", "Buy & Hold", is_model=False, curve=[10000, 10550]),
    ]
    series = _run_node(
        f"homeChartSeries({json.dumps(entries)}, buildEquityCurvesFromEntries).series"
    )
    by_label = {s["label"]: s for s in series}
    assert by_label["Buy & Hold"]["dash"], "baselines read as reference curves, not entrants"
    assert by_label["Buy & Hold"]["isBaseline"] is True
    assert not by_label["DeepSeek V4 Pro"]["dash"]
    assert by_label["DeepSeek V4 Pro"]["isBaseline"] is False


@_requires_node
def test_real_entries_with_no_curves_yield_no_series():
    """The third fallback state, and the reason the gate is NOT keyed on
    `sample`. `renderEntries` runs with `sample: null` whenever
    `models.length > 0`, regardless of whether any entry carries an
    `equity_curve`, and the builder silently drops curveless entries
    (`if (!points.length) return;`). Real entries + no curves therefore produce
    an empty chart with axes, under a real standings list, carrying no sample
    note -- absent and broken rendering identically.
    """
    entries = [
        {
            "entry_id": "deepseek_v4_pro",
            "model": "DeepSeek V4 Pro",
            "is_model": True,
            "team_badge": "Model",
            "equity_curve": [],
        }
    ]
    series = _run_node(
        f"homeChartSeries({json.dumps(entries)}, buildEquityCurvesFromEntries).series"
    )
    assert series == [], "no curves must mean no chart, not an empty chart"


@_requires_node
def test_a_missing_builder_yields_no_series_rather_than_throwing():
    """`buildEquityCurvesFromEntries` lives in another file and reaches this one
    as a global. If it is ever renamed, the panel must degrade to today's
    layout, not throw inside the leaderboard load and leave the list on
    "Loading the standings...". The source guard below is what makes the rename
    loud; this is the runtime floor under it.
    """
    entries = [
        _entry("deepseek_v4_pro", "DeepSeek V4 Pro", is_model=True, curve=[10000, 12100])
    ]
    assert _run_node(f"homeChartSeries({json.dumps(entries)}, undefined).series") == []


def test_the_curve_builder_is_an_explicit_cross_file_export():
    """home-page.js consumes this from js/leaderboard.js. Both are classic
    scripts sharing global scope, so an implicit top-level function would work
    -- and would break silently on rename, degrading to "no chart", which is
    indistinguishable from the honest no-curves state by design (see above).
    Pinning both sides of the seam is what turns that into a red test.
    """
    assert (
        "window.buildEquityCurvesFromEntries = buildEquityCurvesFromEntries;"
        in _LEADERBOARD_JS
    )
    assert "window.buildEquityCurvesFromEntries" in _HOME_JS


@_requires_node
def test_mixed_initial_equity_does_not_break_the_chart():
    """The board's rows do NOT share a capital base, so the chart cannot
    assume one.

    `dashboard/config/leaderboard.json` says `initial_capital: 10000` while
    every published curve was computed at $100,000, and `_find_cached_run`
    (`service.py:615`) does not key on `initial_equity` -- so one
    `?refresh=true` recomputes the five `auto_compute` baselines at $10k and
    leaves the seven model entries at $100k (issue #365, open). Plotted in
    dollars that renders as a 10x scale break: models near 100000, the
    reference baselines flat on the floor at 10000.

    This is the case the old fixture could not express, because it hardcoded
    one `initial_equity` for every row.
    """
    entries = [
        _entry(
            "deepseek_v4_pro", "DeepSeek V4 Pro", is_model=True,
            curve=[100000, 107490], initial_equity=100000,
        ),
        _entry(
            "buy_hold_djia", "Buy & Hold", is_model=False,
            curve=[10000, 10550], initial_equity=10000,
        ),
    ]
    series = _run_node(
        f"homeChartSeries({json.dumps(entries)}, buildEquityCurvesFromEntries).series"
    )
    by_label = {s["label"]: s["values"] for s in series}
    assert set(by_label) == {"DeepSeek V4 Pro", "Buy & Hold"}

    # Fractions, so the two are on one axis despite a 10x difference in base.
    # Raw dollars would put these finals 96,940 apart.
    assert by_label["DeepSeek V4 Pro"][-1] == pytest.approx(0.0749, abs=1e-4)
    assert by_label["Buy & Hold"][-1] == pytest.approx(0.0550, abs=1e-4)
    assert all(
        abs(v) < 1 for values in by_label.values() for v in values if v is not None
    ), "a value outside +/-100% means the series is still in dollars"


@_requires_node
def test_home_chart_matches_the_leaderboards_percent_formula():
    """Screen 0 and the Leaderboard tab must compute percent identically.

    They are two files with no shared module, so the formula is duplicated by
    force. Pinning it as a STRING in either file would pass while the other
    drifted; this runs both against the same input and compares outputs, which
    is the only version of this assertion that can fail for the right reason.
    """
    entries = [
        _entry(
            "deepseek_v4_pro", "DeepSeek V4 Pro", is_model=True,
            curve=[100000, 103000, 107490], initial_equity=100000,
        ),
        _entry(
            "buy_hold_djia", "Buy & Hold", is_model=False,
            curve=[10000, 9800, 10550], initial_equity=10000,
        ),
    ]
    pairs = _run_node(
        "(() => {"
        f"  const entries = {json.dumps(entries)};"
        "   const built = buildEquityCurvesFromEntries(entries);"
        "   return homeChartSeries(entries, buildEquityCurvesFromEntries).series.map("
        "     (s) => ({"
        "       label: s.label,"
        "       home: s.values,"
        "       leaderboard: transformLeaderboardChartData("
        "         built.curves[s.label], 'cumulative', built.initials[s.label]"
        "       ),"
        "     })"
        "   );"
        "})()"
    )
    assert pairs, "no series produced -- the fixture or the harness is wrong"
    for pair in pairs:
        assert pair["home"] == pair["leaderboard"], (
            f"{pair['label']}: screen 0 and the leaderboard tab disagree on percent"
        )
```

- [x] **Step 2: Run them and watch them fail**

```bash
pytest dashboard/backend/tests/test_frontend_chart_first_home.py -v -k "chart or curve or builder"
```

Expected: FAIL — `ValueError: substring not found` from `_extract`, since none of these functions exist yet.

- [x] **Step 3: Export the curve builder**

In `dashboard/frontend/js/leaderboard.js`, beside the existing exports at 1599-1600:

```js
window.loadLeaderboardData = loadLeaderboardData;
window.selectLeaderboardTeam = selectLeaderboardTeam;
// Consumed by home-page.js for screen 0's chart. Explicit rather than relying
// on the implicit global these classic scripts share: on rename the implicit
// form degrades to "no chart", which this design deliberately makes
// indistinguishable from the honest no-curves state -- so the break would be
// invisible. The export is pinned from both sides by
// test_frontend_chart_first_home.py.
window.buildEquityCurvesFromEntries = buildEquityCurvesFromEntries;
```

- [x] **Step 4: Add the pure functions**

In `dashboard/frontend/home-page.js`, immediately **above** `async function loadHomeLeaderboardModule()`. Top-level, closure-free, and side-effect-free — these are the two properties that let them run under node:

```js
/** Entry ids the screen-0 chart draws as passive reference curves.
 *
 *  Ids, not display labels. `LEADERBOARD_STYLES` in js/leaderboard.js keys on
 *  the label ("Buy & Hold", "DJIA"), but the label is copy and can be renamed
 *  in dashboard/config/leaderboard.json without anything failing; `id` is that
 *  file's primary key and reaches the client as `entry.entry_id`.
 *
 *  Two, not five. This chart is 187-280px tall and already carries seven model
 *  curves; the question it exists to answer -- is +21.0% good? -- needs one
 *  strategy baseline and one index, not the whole baseline roster. */
const HOME_CHART_BASELINE_IDS = ['buy_hold_djia', 'djia_index'];

/** Entries the CHART draws: every model, plus the two reference baselines.
 *
 *  A second, wider selection than `homeModelEntries()`, which the rank list
 *  keeps. That one filters on `is_model || team_badge === 'Model'`, so the
 *  panel's only data source today has no baselines in it -- build the chart
 *  from it unchanged and you draw seven curves with nothing to judge them
 *  against.
 *
 *  The LIST stays models-only on purpose: app.html ships the pinned line
 *  "AI models only - ranked by return", which describes the RANKING. Baselines
 *  on the chart leave it literally true; ranking them into the list would make
 *  it false and force a copy change on a line the rest of this design leans on
 *  being stable. */
function homeChartEntries(entries) {
    const all = entries || [];
    const models = all.filter((e) => e && (e.is_model || e.team_badge === 'Model'));
    const baselines = all.filter(
        (e) => e && !e.is_model && HOME_CHART_BASELINE_IDS.indexOf(e.entry_id) !== -1
    );
    return models.concat(baselines);
}

/** The chart's `{times, series}`, with an empty `series` when there is nothing
 *  honest to draw.
 *
 *  THE GATE IS "CURVES PRESENT", NOT "SAMPLE IS NULL". `renderEntries` runs
 *  with `sample: null` whenever `models.length > 0`, regardless of whether any
 *  entry carries an `equity_curve`, and `buildEquityCurvesFromEntries` silently
 *  drops curveless entries (`if (!points.length) return;`). Real entries with
 *  no curves therefore yield zero series: an empty chart with axes, under a
 *  real standings list, carrying no sample note -- because the data genuinely
 *  is real. Absent and broken would render identically, which is this repo's
 *  fail-closed-is-not-fail-visible failure in miniature.
 *
 *  `build` is injected rather than read off `window` so this function is a pure
 *  function of its arguments and can be exercised under node. */
function homeChartSeries(entries, build) {
    if (typeof build !== 'function') return { times: [], series: [] };
    const selected = homeChartEntries(entries);
    if (!selected.length) return { times: [], series: [] };
    const built = build(selected) || {};
    const times = built.times || [];
    const curves = built.curves || {};
    const styles = built.trajectories || {};
    const initials = built.initials || {};
    if (!times.length) return { times: [], series: [] };
    const series = Object.keys(curves)
        .map((label) => {
            const style = styles[label] || {};
            const raw = curves[label] || [];
            // Fractions, not dollars. Each entry carries its OWN
            // `initial_equity` and they do not currently agree across the
            // board (issue #365), so a shared dollar axis would draw a $10k
            // baseline against $100k models as a flat line on the floor.
            // Dividing by each series' own base makes the comparison the
            // chart exists to make. Same formula and same fallback order as
            // `transformLeaderboardChartData`'s 'cumulative' branch in
            // js/leaderboard.js -- pinned as an equivalence, not by eye, in
            // `test_home_chart_matches_the_leaderboards_percent_formula`.
            const base = Number(initials[label]) || raw.find((v) => v != null) || 10000;
            return {
                label,
                values: raw.map((v) => (v == null ? null : (v - base) / base)),
                color: style.color || '#94a3b8',
                dash: style.dash || [],
                isBaseline: (style.kind || 'model') !== 'model',
            };
        })
        .filter((s) => s.values.some((v) => v != null));
    return { times, series };
}
```

- [x] **Step 5: Run the tests and watch them pass**

```bash
pytest dashboard/backend/tests/test_frontend_chart_first_home.py -v
```

Expected: PASS (or SKIP on the node cases if `node` is absent — install node locally; CI has it).

- [x] **Step 6: Mutation-test the gate**

The spec requires this: PR #352's round wrote 15 source-shape guards and 2 passed against a broken implementation.

```bash
# Break the gate the way a "simplification" would, then confirm the test catches it.
# Temporarily change the `.filter(...)` tail of homeChartSeries to `;` (no filter):
pytest dashboard/backend/tests/test_frontend_chart_first_home.py::test_real_entries_with_no_curves_yield_no_series -v
```

Expected: FAIL. Restore the filter and re-run — PASS. Repeat for `test_chart_draws_the_baselines_the_rank_list_filters_out` by swapping `homeChartEntries` for `homeModelEntries`-style filtering.

- [x] **Step 7: Bump the two cache busters (once per branch)**

In `dashboard/frontend/app.html`: `home-page.js?v=47` → `home-page.js?v=48`, `js/leaderboard.js?v=28` → `js/leaderboard.js?v=29`.
In `dashboard/backend/tests/test_frontend_fast_boot.py:195-196`, update both exact assertions.

- [x] **Step 8: Commit**

```bash
git add dashboard/frontend/home-page.js dashboard/frontend/js/leaderboard.js \
        dashboard/frontend/app.html \
        dashboard/backend/tests/test_frontend_chart_first_home.py \
        dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "feat(home): chart series selection with a curves-present gate"
```

---

### Task 3: Render the chart on screen 0

Spec §3, §3c. The canvas is created only when Task 2's gate returns a non-empty set.

**Files:**
- Modify: `dashboard/frontend/home-page.js` (render fn + call site in `loadHomeLeaderboardModule`)
- Modify: `dashboard/frontend/styles.css` (new `.hm-rank-chart` block)
- Test: `dashboard/backend/tests/test_frontend_chart_first_home.py` (extend)

**Interfaces:**
- Consumes: `homeChartSeries(entries, build)` from Task 2; `window.Chart` (Chart.js 4.4.0, `app.html:20`, `defer` + SRI).
- Produces: `#homeModuleRankChartWrap` / `#homeModuleRankChart`, inserted before `.hm-rank-table-head`.

- [x] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_frontend_chart_first_home.py`:

```python
def test_the_chart_element_is_created_only_when_there_are_series():
    """Reserve nothing. Chart.js is a deferred third-party script and screen 0 is
    now the first thing /app paints, so there is a window -- longer on a
    free-tier cold start -- where the panel knows its chart's height and has
    nothing to draw in it. A reserved-but-blank 234px box looks like a chart that
    FAILED rather than one that has not arrived, which is the same absent-vs-
    broken confusion the gate exists to prevent. One downward layout shift, of
    content nobody has started reading, is the cheaper cost.
    """
    body = _extract(_HOME_JS, "renderHomeLeaderboardChart")
    assert "if (!series.length) return null;" in body, (
        "no series must mean no canvas -- not an empty canvas"
    )
    # The insertion is guarded, not unconditional at module scope.
    assert "document.createElement('canvas')" in body or "<canvas" in body
    assert "typeof window.Chart" in body, (
        "Chart.js is deferred; the render path must tolerate it not having landed"
    )


def test_chart_axis_ticks_are_14px():
    """Spec §2's type scale is the only thing keeping the two surfaces looking
    like one product, and nothing enforces it across stacks -- so it is pinned on
    each. The cross-surface pair check is in test_landing_chart_first.py.
    """
    body = _extract(_HOME_JS, "renderHomeLeaderboardChart")
    assert re.search(r"font:\s*\{\s*size:\s*14\s*\}", body), (
        "11px axis ticks were one of the three reported problems"
    )


def test_chart_height_is_the_app_clamp_and_not_the_landing_one():
    """The surfaces have different vertical envelopes and therefore different
    formulas. /app's panel is bounded by the pager; /'s card by the document.
    A shared assertion here would be a bug, not a simplification.
    """
    blocks = css_blocks(".hm-rank-chart")
    assert blocks, ".hm-rank-chart was renamed or deleted"
    assert "clamp(140px, 26vh, 280px)" in blocks[0]
    assert "100dvh" not in blocks[0], "that is /'s formula, measured against the fold"
```

- [x] **Step 2: Run it and watch it fail**

```bash
pytest dashboard/backend/tests/test_frontend_chart_first_home.py -v -k "chart_element or axis_ticks or chart_height"
```

Expected: FAIL — `renderHomeLeaderboardChart` does not exist; `.hm-rank-chart` has no CSS block.

- [x] **Step 3: Add the CSS**

In `dashboard/frontend/styles.css`, immediately after the `.hm-rank-table-head` rules (≈6214):

```css
/* Screen 0's equity chart. `clamp(140px, 26vh, 280px)` yields 187-280px across
   1280x720 -> 1920x1080 -- verified alongside 7/7 visible standings rows and
   zero pager clip. Deliberately smaller than /'s clamp: this panel is bounded
   by the pager, / 's card by the document. `flex: 0 0 auto` so the list, which
   is `flex: 1 1 auto; overflow-y: auto`, absorbs the remaining height rather
   than the chart collapsing. */
.hm-rank-chart {
    flex: 0 0 auto;
    position: relative;
    height: clamp(140px, 26vh, 280px);
    margin-bottom: 10px;
}
.hm-rank-chart canvas {
    width: 100% !important;
    height: 100% !important;
}
```

- [x] **Step 4: Add the render function**

In `dashboard/frontend/home-page.js`, below `homeChartSeries`:

```js
let homeRankChart = null;

/** Draw screen 0's equity chart, or nothing at all.
 *
 *  Returns null -- and creates no element -- when there are no series or when
 *  Chart.js has not landed yet. Both cases leave the panel laid out exactly as
 *  it is today, list and all, which is the point: a blank reserved box reads as
 *  a chart that failed. */
function renderHomeLeaderboardChart(series, times) {
    if (!series.length) return null;
    if (typeof window.Chart !== 'function') return null;
    const panel = document.getElementById('homeModuleRanking');
    const anchor = panel && panel.querySelector('.hm-rank-table-head');
    if (!panel || !anchor) return null;

    let wrap = document.getElementById('homeModuleRankChartWrap');
    if (!wrap) {
        wrap = document.createElement('div');
        wrap.id = 'homeModuleRankChartWrap';
        wrap.className = 'hm-rank-chart';
        const canvas = document.createElement('canvas');
        canvas.id = 'homeModuleRankChart';
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label', 'Account value for each AI model over the competition window');
        wrap.appendChild(canvas);
        panel.insertBefore(wrap, anchor);
    }
    if (homeRankChart) homeRankChart.destroy();

    const axis = { color: 'rgba(148, 163, 184, 0.85)', font: { size: 14 } };
    homeRankChart = new window.Chart(wrap.querySelector('canvas'), {
        type: 'line',
        data: {
            labels: times,
            datasets: series.map((s) => ({
                label: s.label,
                data: s.values,
                borderColor: s.color,
                borderWidth: s.isBaseline ? 1.5 : 2,
                borderDash: s.dash,
                pointRadius: 0,
                spanGaps: true,
                tension: 0,
            })),
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                // The rank list beneath is the key -- each row carries the
                // curve's colour swatch. A legend here would be the same five
                // names twice, in a panel that has no height to spare.
                legend: { display: false },
                tooltip: {
                    enabled: true,
                    callbacks: {
                        // Without this the tooltip prints the raw fraction
                        // (0.0749). The series are percent now; the readout
                        // has to be too, and in the key's own precision.
                        label: (c) =>
                            `${c.dataset.label}: ${c.parsed.y >= 0 ? '+' : ''}${(c.parsed.y * 100).toFixed(1)}%`,
                    },
                },
            },
            scales: {
                x: { ticks: { ...axis, maxTicksLimit: 6 }, grid: { display: false } },
                y: {
                    // Percent, not dollars -- see "Units" in Global
                    // Constraints. One decimal to match the rank list beside
                    // it (`homeFormatReturnBadge` uses toFixed(1)); at zero
                    // decimals a narrow domain renders duplicate tick labels.
                    ticks: { ...axis, callback: (v) => `${(v * 100).toFixed(1)}%` },
                    grid: { color: 'rgba(148, 163, 184, 0.12)' },
                },
            },
        },
    });
    return homeRankChart;
}
```

- [x] **Step 5: Wire the call site**

In `loadHomeLeaderboardModule`, replace the final `renderEntries(models);` (the real-data path only — **not** the two sample paths, which must stay chartless):

```js
        renderEntries(models);
        const build = window.buildEquityCurvesFromEntries;
        if (typeof build !== 'function') {
            // Distinguishable in the console from the honest no-curves case,
            // which is silent. Absent and broken must not look identical.
            console.warn('Home chart: buildEquityCurvesFromEntries is unavailable.');
        }
        const chart = homeChartSeries(payload.entries || [], build);
        renderHomeLeaderboardChart(chart.series, chart.times);
```

- [x] **Step 6: Run the suite and watch it pass**

```bash
pytest dashboard/backend/tests/test_frontend_chart_first_home.py -v
```

Expected: PASS.

- [x] **Step 7: Look at it**

```bash
cp dashboard/storage/data/backtest.db /tmp/claude-1000/scratch-backtest.db
DATABASE_PATH=/tmp/claude-1000/scratch-backtest.db \
ANTHROPIC_API_KEY= OPENAI_API_KEY= DEEPSEEK_API_KEY= COMMONSTACK_API_KEY= \
  uvicorn dashboard.backend.app:app --port 8077
```

Open `http://127.0.0.1:8077/app` at 1440×900. Expected: a chart above the standings, seven solid model curves plus two dashed reference curves, 14px axis labels, all seven rows still visible.

**Never point `DATABASE_PATH` at the committed `dashboard/storage/data/backtest.db`** — a bare backend import runs lazy `ALTER`s against it, and it is the prod seed database.

- [x] **Step 8: Commit**

```bash
git add dashboard/frontend/home-page.js dashboard/frontend/styles.css \
        dashboard/backend/tests/test_frontend_chart_first_home.py
git commit -m "feat(home): equity chart on screen 0, gated on real curves"
```

---

### Task 4: The rank list becomes the chart's key

Spec §1. Compact rows, a colour swatch per row, **ending value and Sharpe stay** — stripping them to match `/` would delete live data to satisfy a marketing-page constraint.

**Files:**
- Modify: `dashboard/frontend/js/leaderboard.js:75-77` (widen `MODEL_COLOR_PALETTE`)
- Modify: `dashboard/frontend/home-page.js` (`renderEntries`)
- Modify: `dashboard/frontend/styles.css` (compact row rules, swatch)
- Test: `dashboard/backend/tests/test_frontend_chart_first_home.py` (extend)

**Interfaces:**
- Consumes: `window.getSeriesStyle(label, entry).color` — the same call `homeChartSeries` uses, so a row's swatch and its curve cannot disagree.

- [x] **Step 1: Write the failing tests**

```python
def test_the_model_palette_has_a_distinct_colour_for_every_board_model():
    """`getModelColor` assigns `MODEL_COLOR_PALETTE[n % len]` in first-seen
    order. The board carries seven models and the palette had five, so models 6
    and 7 got models 1 and 2's colours -- two pairs of identically coloured
    curves. Harmless while the swatch was decoration; not harmless now that the
    swatch is the chart's only key.
    """
    block = _const_block(_LEADERBOARD_JS, "MODEL_COLOR_PALETTE")
    colours = re.findall(r"#[0-9A-Fa-f]{6}", block)
    assert len(colours) >= 7, "seven models are on the board"
    assert len(set(c.lower() for c in colours)) == len(colours), "duplicate colours"


def test_rank_rows_carry_the_swatch_from_the_same_source_as_the_curve():
    """A row whose swatch disagrees with its curve is worse than no swatch: it
    points the reader at the wrong line. Both sides therefore read
    `getSeriesStyle`, rather than the list picking its own colour.
    """
    body = _extract(_HOME_JS, "renderEntries")
    assert "getSeriesStyle" in body
    assert "hm-rank-swatch" in body


def test_rank_rows_keep_ending_value_and_sharpe():
    """/ demotes its table to a legend strip because it has Race.tsx to hold the
    detail. /app has no such page, and these are real numbers a signed-in user
    came for.
    """
    body = _extract(_HOME_JS, "renderEntries")
    assert "hm-rank-value" in body
    assert "hm-rank-sharpe" in body
```

- [x] **Step 2: Run and watch fail**

```bash
pytest dashboard/backend/tests/test_frontend_chart_first_home.py -v -k "palette or swatch or ending_value"
```

Expected: FAIL — palette has 5 colours; `renderEntries` has no `getSeriesStyle` and no `hm-rank-swatch`.

- [x] **Step 3: Widen the palette**

In `dashboard/frontend/js/leaderboard.js`, replace the `MODEL_COLOR_PALETTE` block:

```js
// Provided LLM models get their own warm, distinct palette (solid lines) so
// they read as a separate category from rule-based strategy baselines.
// Ten, not five: `getModelColor` assigns `PALETTE[n % len]` in first-seen
// order, and the board carries seven models -- at five, models 6 and 7 were
// handed models 1 and 2's colours. That was cosmetic while the colour was
// decoration and is not, now that screen 0's rank-row swatch is the chart's
// only key to which curve is which.
const MODEL_COLOR_PALETTE = [
  '#FBBF24', '#FB923C', '#F472B6', '#A78BFA', '#34D399',
  '#22D3EE', '#F87171', '#A3E635', '#E879F9', '#60A5FA',
];
```

Add the export beside the others at the file's end:

```js
window.getSeriesStyle = getSeriesStyle;
```

- [x] **Step 4: Add the swatch to `renderEntries`**

In `home-page.js`, inside `renderEntries`'s `map`, add above the `return`:

```js
            const style = (typeof window.getSeriesStyle === 'function')
                ? window.getSeriesStyle(label, entry)
                : { color: 'transparent' };
```

and as the first child of the `<li>`:

```js
                <span class="hm-rank-swatch" style="background:${homeEscape(style.color || 'transparent')}" aria-hidden="true"></span>
```

`homeEscape` on the colour is not decoration: it lands in an inline `style` attribute, and the value comes from a payload field.

- [x] **Step 5: Compact the rows**

In `dashboard/frontend/styles.css`, beside the existing `.home-module-rank-list` rules:

```css
/* The swatch is the identity link between a row and its curve; the list is the
   chart's legend, which is why the chart ships none of its own. */
.hm-rank-swatch {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex: 0 0 auto;
    margin-right: 6px;
}
.home-module-rank-list > li {
    padding-block: 3px;
    line-height: 1.35;
}
```

- [x] **Step 6: Run and watch pass**

```bash
pytest dashboard/backend/tests/test_frontend_chart_first_home.py dashboard/backend/tests/test_frontend_leaderboard_hover.py -v
```

Expected: PASS. The hover suite is included because it exercises `js/leaderboard.js`, which this task edits.

- [x] **Step 7: Commit**

```bash
git add dashboard/frontend/js/leaderboard.js dashboard/frontend/home-page.js \
        dashboard/frontend/styles.css dashboard/backend/tests/test_frontend_chart_first_home.py
git commit -m "feat(home): rank rows carry the curve swatch; widen the model palette"
```

---

### Task 5: The screen 0 lede

Spec §4. Copy-only. Verified unpinned: a grep for the sentence across the whole suite returns nothing, and `test_app_copy_register.py:305-311` pins only the three screen-0 ids plus the absence of the `Talk to Agents` pitch.

**Files:**
- Modify: `dashboard/frontend/app.html:458-462`
- Test: `dashboard/backend/tests/test_frontend_chart_first_home.py` (extend)

- [x] **Step 1: Write the failing test**

```python
def test_the_screen_zero_lede_is_a_fact_then_a_call_to_action():
    """The old sentence did two jobs at once -- glossing "agent" AND pre-empting
    "is my agent on this list?" -- which is why it read as neither marketing nor
    a CTA. The no-entry fact is already stated on the board itself
    ("AI models only - ranked by return"), so the lede is freed to be one plain
    thing. The gloss drops on this surface: the reader is signed in and inside
    the app, where the word is glossed throughout.
    """
    from dashboard.backend.tests._frontend_source import APP_HTML

    html = re.sub(r"<!--.*?-->", "", APP_HTML, flags=re.DOTALL)
    assert (
        "See how the AI models did. Then test your own idea on the same days."
        in html
    )
    assert "in a test of its own" not in html
    # The fact it used to carry must still be on screen, on the board making the
    # claim -- otherwise this is a deletion, not a split.
    assert "AI models only" in html
```

> **Superseded 2026-08-20 (PR #394).** This step executed as written; the lede and its
> guard have both moved since. Left verbatim because the block is a record of what ran,
> not a description of what ships — for the current copy, and why its second clause is
> not optional, see the "Superseded" note in the spec.

- [x] **Step 2: Run and watch fail**

```bash
pytest dashboard/backend/tests/test_frontend_chart_first_home.py -v -k lede
```

Expected: FAIL on the first assertion.

- [x] **Step 3: Replace the lede**

In `dashboard/frontend/app.html`, replace this exact block — the comment and the `<p>` immediately after the `<h1>` (it sat at 458-462 when this plan was written and at 463-467 after PR #344; **match the text, never the line numbers**, and do not let the range creep up into the `<h1>`, which no test pins):

```html
                            <!-- The one-per-surface gloss on "agent". Deliberately says
                                 "in a test of its own": no user agent is on any board and
                                 no entry path exists, so anything that reads as "yours is
                                 on this list" is a promise the product cannot keep. -->
                            <p class="home-landing-lede">Your own agent &mdash; an AI trading assistant that follows your written instruction &mdash; is scored on the same numbers, in a test of its own.</p>
```

with:

```html
                            <!-- Fact, then call to action. The "is my agent on
                                 this list?" pre-emption this sentence used to
                                 carry now lives on the board itself
                                 ("AI models only - ranked by return",
                                 #homeModuleRanking) -- on the element making the
                                 claim, rather than in a lede that had to be a
                                 disclaimer and a value prop at once. The gloss
                                 on "agent" drops here: the reader is signed in
                                 and inside the app, where the word is glossed
                                 throughout. -->
                            <p class="home-landing-lede">See how the AI models did. Then test your own idea on the same days.</p>
```

> **Superseded 2026-08-20 (PR #394)** — same as the step above; this is the markup that
> shipped at the time, not the markup that ships now.

- [x] **Step 4: Run the copy suites**

```bash
pytest dashboard/backend/tests/test_app_copy_register.py \
       dashboard/backend/tests/test_frontend_chart_first_home.py -v
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add dashboard/frontend/app.html dashboard/backend/tests/test_frontend_chart_first_home.py
git commit -m "ux(home): screen 0 lede states a fact, then asks for the click"
```

---

# Phase B — `/` landing

Vercel, ~1 min. Both hosts serve the landing page, so between the two deploys `/` renders differently depending on which is hit — expected, and short.

**Copy trim comes first.** At `lg:` the hero is a flex row with `items-center`: whichever column is taller sets the row height, and the card is centred inside it. Narrowing the copy column to 1/3 makes its text wrap *more*, so it can become the taller column and push the card's bottom edge down. The measured fold slack (+27px at every viewport) was taken with a 672px card beside a **full-width** copy column — it does not carry over to the 2/3 layout on its own. Trim first, then re-measure in Task 12.

---

### Task 6: Hero copy — two paragraphs to one line

Spec §5, Hero row. The simulated-money sentence moves to small print under the CTA, **verbatim** — it is pinned twice.

**Files:**
- Modify: `dashboard/landing/src/components/home/Hero.tsx:105-115`
- Test: `dashboard/backend/tests/test_landing_chart_first.py` (create)

- [x] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_landing_chart_first.py`:

```python
"""Guards for the chart-first rebuild of / (2026-08-15 spec).

These read the TSX SOURCE, not the shipped bundle. The bundle-reading guards in
test_landing_copy_register.py already catch "edited but never rebuilt"; what
they cannot catch is a layout constant, because minified Tailwind classes and
Recharts props survive the build as opaque strings that no copy guard inspects.

Heights are asserted per surface and never shared with /app: the two surfaces
have different vertical envelopes and therefore different formulas (spec §2).
"""

import re
from pathlib import Path

_HOME = (
    Path(__file__).resolve().parents[2] / "landing" / "src" / "components" / "home"
)
_HERO = (_HOME / "Hero.tsx").read_text(encoding="utf-8")
_BOARD = (_HOME / "BoardPreview.tsx").read_text(encoding="utf-8")

_NO_REAL_MONEY = (
    "Every test here uses simulated money. Real money is involved only if you "
    "explicitly connect a brokerage account and turn on live trading."
)


def _collapse(source: str) -> str:
    """JSX text with its line breaks and indentation collapsed, so a sentence
    split across lines by the formatter still matches as one string."""
    return re.sub(r"\s+", " ", source)


def test_the_hero_lede_is_one_line_and_still_glosses_agent():
    """/ is the acquisition page: the headline uses "agent" before anything else
    defines it, and the board beside it is the only other thing above the fold.
    The gloss has to land here or not at all -- unlike /app, where the reader is
    already inside the product. So this trims to one line; it does not drop.
    """
    hero = _collapse(_HERO)
    assert "An agent is an AI trading assistant that follows your written instruction" in hero
    assert "it trades the idea hour by hour, measured against buy-and-hold and the index" not in hero, (
        "the second clause is what makes this two lines at 1/3 column width"
    )


def test_the_simulated_money_sentence_survives_verbatim_as_small_print():
    """Pinned twice -- by test_no_real_money_sentence_is_present_verbatim and by
    the _CLAIM_DISCLAIMERS allowlist, whose staleness check fails if the wording
    drifts. Moving it between components is fine; rewording it is not.
    """
    assert _NO_REAL_MONEY in _collapse(_HERO)
```

- [x] **Step 2: Run and watch fail**

```bash
pytest dashboard/backend/tests/test_landing_chart_first.py -v
```

Expected: FAIL — the second clause is still present.

- [x] **Step 3: Trim the copy**

In `dashboard/landing/src/components/home/Hero.tsx`, replace the two `<p>` blocks (105-115):

```tsx
          {/* The one-per-surface gloss on "agent". The headline uses the word
              before anything else on the page defines it, and the board beside
              it is the only other thing above the fold — so the definition has
              to land here or not at all. One line, not two: at 1/3 column width
              the second clause wrapped to three lines and made the copy column
              taller than the card, which pushes the card's bottom edge below the
              fold (the hero row is `items-center`). */}
          <p className="max-w-xl mx-auto lg:mx-0 mb-5 text-base text-foreground/85 leading-relaxed">
            Write your trading idea in plain English. An agent is an AI trading assistant that
            follows your written instruction.
          </p>
```

and move the simulated-money paragraph below the CTA `motion.div`, **unchanged**:

```tsx
          <p className="max-w-xl mx-auto lg:mx-0 mt-5 text-sm text-foreground/75">
            Every test here uses simulated money. Real money is involved only if you explicitly
            connect a brokerage account and turn on live trading.
          </p>
```

- [x] **Step 4: Run the landing suites**

```bash
pytest dashboard/backend/tests/test_landing_chart_first.py \
       dashboard/backend/tests/test_landing_copy_register.py -v
```

Expected: `test_landing_chart_first.py` PASS. `test_landing_copy_register.py` also PASS — it reads the *bundle*, which is unchanged until Task 10. That green is not evidence; Task 10 is.

- [x] **Step 5: Commit**

```bash
git add dashboard/landing/src/components/home/Hero.tsx \
        dashboard/backend/tests/test_landing_chart_first.py
git commit -m "ux(landing): hero lede to one line, disclaimer to small print"
```

---

### Task 7: Hero becomes 2/3 chart, 1/3 copy, full-bleed left

Spec § Shape. Three mechanical facts, none of which is "remove a class":

1. `max-w-2xl` on the board column caps it at 672px. Two-thirds of a 1280px container is 853px, so **the cap has to go** or every other change is cosmetic.
2. Both columns are `flex-1` children of the single `container mx-auto px-6` div that also owns the hero's `lg:min-h-[...]` contract. Escaping it on one edge needs a negative inline-start margin, not a class removal.
3. **Order with `order-*`, not by reordering the JSX.** Moving `<BoardPreview />` above the copy puts its `<h2>` ahead of the page's only `<h1>`, so the document outline opens on the board's title.

**Full-bleed is a ≥1300px effect only** — measured: the container's left gutter is 185px at 1920, 73px at 1440, and **0px at 1280 and below**. Below ~1300px the 2/3 split carries the layout alone. Do not describe or test full-bleed as the mechanism at laptop widths.

**Files:**
- Modify: `dashboard/landing/src/components/home/Hero.tsx:93, 94, 139`
- Test: `dashboard/backend/tests/test_landing_chart_first.py` (extend)

- [x] **Step 1: Write the failing tests**

```python
def test_the_board_column_is_two_thirds_and_uncapped():
    hero = _HERO
    assert "max-w-2xl" not in hero, (
        "672px is card width; two-thirds of a 1280px container is 853px, so this "
        "cap silently reverts the layout to what PR #357 already shipped"
    )
    assert "lg:basis-2/3" in hero
    assert "lg:basis-1/3" in hero


def test_the_columns_are_ordered_with_utilities_not_by_source_order():
    """The visual ask is chart-left / hero-right at lg:, chart first when
    stacked -- which reads as "move <BoardPreview/> above the copy in source".
    Doing that puts BoardPreview's <h2> ahead of the page's only <h1>.
    """
    hero = _HERO
    assert hero.index("<h1") < hero.index("<BoardPreview"), (
        "the h1 block must stay first in source"
    )
    assert "order-first" in hero and "lg:order-first" in hero


def test_the_chart_column_escapes_the_container_on_its_left_edge_only():
    """Both columns live inside one `container mx-auto px-6` div that also owns
    the hero's min-height contract, so this is a negative inline-start margin at
    lg: and above -- not a class removal. It is a >=1300px effect: the container
    gutter is 0px at 1280 and below.
    """
    assert "lg:ms-[calc((100%-100vw)/2)]" in _HERO
```

- [x] **Step 2: Run and watch fail**

```bash
pytest dashboard/backend/tests/test_landing_chart_first.py -v -k "two_thirds or ordered or escapes"
```

Expected: FAIL on all three.

- [x] **Step 3: Restructure the hero row**

In `Hero.tsx`, the copy column (`:94`):

```tsx
        <div className="flex-1 lg:basis-1/3 lg:grow-0 order-last lg:order-last text-center lg:text-left">
```

and the board column (`:139`), losing `max-w-2xl`:

```tsx
        <motion.div
          className="w-full flex-1 lg:basis-2/3 lg:grow-0 shrink-0 order-first lg:order-first lg:ms-[calc((100%-100vw)/2)] lg:ps-6"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.7, delay: 0.3 }}
        >
```

`lg:ps-6` restores the 24px the container's `px-6` was providing on that edge, so the chart is flush to the viewport without its axis labels touching it.

- [x] **Step 4: Run and watch pass**

```bash
pytest dashboard/backend/tests/test_landing_chart_first.py -v
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add dashboard/landing/src/components/home/Hero.tsx \
        dashboard/backend/tests/test_landing_chart_first.py
git commit -m "feat(landing): hero splits 2/3 chart, 1/3 copy, full-bleed left"
```

---

### Task 8: BoardPreview — bigger chart, 14px axes, chip strip

Spec §1, §2. **Demotion, not deletion.** `BoardPreview` ships no Recharts `<Legend>` on purpose (its own comment records that a five-item legend wraps to two rows at card width), so the standings table is currently the *only* thing linking a curve colour to a model name. Delete it outright and five unnamed lines are left.

**The chip strip must stay in `BoardPreview.tsx`.** `test_landing_copy_register.py:362-365` scopes its corpus to files containing `SAMPLE_STANDINGS` and requires `dataKey=` in it — the only `dataKey=` there is on this file's `<Line>` elements. Splitting the strip into its own component reddens that guard though nothing was deleted.

**Files:**
- Modify: `dashboard/landing/src/components/home/BoardPreview.tsx:96-160`
- Test: `dashboard/backend/tests/test_landing_chart_first.py` (extend)

- [x] **Step 1: Write the failing tests**

```python
def test_the_landing_chart_uses_its_own_measured_clamp():
    """`clamp(320px, 56vh, 520px)` -- the first draft's number, shared with /app
    -- puts the card 25-46px BELOW the fold at 1440x768, 1366x768, 1280x800 and
    1280x720. All four are ordinary laptop heights. The replacement is the
    largest formula with non-negative fold slack at every tested viewport.

    The 390 is derived, not taste: the card's own non-chart height (~227px:
    caption bar, chip strip, detail line, padding) + 120px
    --landing-chrome-height + ~43px fold margin. RE-DERIVE IT if the caption or
    chip strip changes height -- the failure mode is a silently half-visible
    card, not a broken build.
    """
    assert "clamp(300px,calc(100dvh-390px),520px)" in _BOARD.replace(" ", "")
    assert "56vh" not in _BOARD, "the first draft's clamp fails at four viewports"
    assert "h-[210px]" not in _BOARD and "md:h-[240px]" not in _BOARD


def test_landing_chart_axis_ticks_are_14px():
    assert _BOARD.count("fontSize={14}") == 2, "both XAxis and YAxis"
    assert "fontSize={11}" not in _BOARD


def test_the_panel_title_is_text_xl():
    """Spec §2. The card is now two-thirds of the hero; a text-lg title reads as
    a widget label on it."""
    assert 'className="text-xl font-bold flex items-center gap-2 min-w-0"' in _BOARD


def test_the_standings_table_becomes_a_one_row_chip_strip():
    """Demotion, not deletion: the chart ships no <Legend> (a five-item legend
    wraps to two rows at this width), so the table is the ONLY thing linking a
    curve colour to a model name. The chips preserve that swatch-to-curve link
    at a fraction of the height. The full table already lives in Race.tsx.
    """
    board = _BOARD
    assert "grid-cols-12" not in board, "the 5-row table is what the chart needs the height of"
    assert "flex-nowrap" in board, "five chips, one row"
    assert "text-base" in board, "text-sm rows were one of the three reported problems"
    # The identity link and the guard corpus both depend on these staying here.
    assert "SAMPLE_STANDINGS" in board
    assert "dataKey=" in board
    assert "item.swatch" in board
```

- [x] **Step 2: Run and watch fail**

```bash
pytest dashboard/backend/tests/test_landing_chart_first.py -v -k "clamp or axis or chip_strip"
```

Expected: FAIL on all three.

- [x] **Step 3: Resize the chart, enlarge the axes, promote the title**

In `BoardPreview.tsx`, replace the chart wrapper (`:96`):

```tsx
      <div
        className="w-full px-3 pt-4"
        style={{ height: "clamp(300px, calc(100dvh - 390px), 520px)" }}
      >
```

An inline `style` rather than an arbitrary Tailwind value: the formula contains a comma and parentheses that Tailwind's arbitrary-value parser mangles, and the constant is load-bearing enough to want readable in source.

Change both `fontSize={11}` to `fontSize={14}`.

Promote the panel title (`:83`) per spec §2 — on a card that is now two-thirds of the hero, `text-lg` reads as a widget label:

```tsx
          <h2 className="text-xl font-bold flex items-center gap-2 min-w-0">
```

⚠ **Do not remove the `Illustrative example` chip at `:88`.** It ships at three render sites today (this file, `ChatSimulation.tsx:149`, `Race.tsx:77`) against a `≥2` guard in `test_landing_copy_register.py`. This task keeps BoardPreview's, so the count stays 3 — but the margin is a single site, so nothing else in this branch may drop one either.

- [x] **Step 4: Replace the table with the chip strip**

Replace the whole `<div className="px-5 pb-5 pt-3">` block (`:122-159`):

```tsx
      <div className="px-5 pb-5 pt-3">
        {/* The chart ships no Recharts <Legend> — at this card's width a
            five-item legend wraps to two rows and pushes the plot area down.
            These chips are the key, and the swatch is the identity link between
            a name and its curve. The full standings table, with ranks, lives in
            Race.tsx, which is the detail home.

            Kept in this file deliberately: test_landing_copy_register.py scopes
            its corpus to files containing SAMPLE_STANDINGS and requires
            `dataKey=` in it, and the only `dataKey=` is on the <Line> elements
            above. Splitting this strip into its own component reddens that
            guard though nothing was deleted. */}
        <div className="flex flex-nowrap items-center gap-x-4 gap-y-2 overflow-hidden text-base">
          {SAMPLE_STANDINGS.map((item) => (
            <span key={item.rank} className="flex items-center gap-2 whitespace-nowrap">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                style={{ backgroundColor: item.swatch }}
                aria-hidden="true"
              />
              <span className="font-medium text-foreground">{item.name}</span>
              <span
                className={`font-mono font-bold ${
                  item.ret.startsWith("-") ? "text-destructive" : "text-positive"
                }`}
              >
                {item.ret}
              </span>
            </span>
          ))}
        </div>
        <p className="mt-3 text-sm text-foreground/65">
          Account value over the competition window.
        </p>
      </div>
```

- [x] **Step 5: Shorten the caption to one line**

The caption at `:91-94` is two lines at the old card width and feeds the `227px` in the clamp's constant. Replace it:

```tsx
        <p className="text-sm text-foreground/65 leading-relaxed">
          Each line is one AI model&apos;s account value. Dashed lines are buy-and-hold and the index.
        </p>
```

Note `text-xs` → `text-sm` per spec §2. If this ever grows back to two lines, the `390` constant is invalid — see the test's docstring.

- [x] **Step 6: Run and watch pass**

```bash
pytest dashboard/backend/tests/test_landing_chart_first.py -v
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add dashboard/landing/src/components/home/BoardPreview.tsx \
        dashboard/backend/tests/test_landing_chart_first.py
git commit -m "feat(landing): chart takes the card, standings become a chip strip"
```

---

### Task 9: Trim WhyCare and Talk

Spec §5, remaining rows. **Two files, not three.** Two of the spec's rows need no work at all:
- `Race.tsx` already renders the full standings table at `:80-112`, which is exactly what §5 asks it to become.
- `Test.tsx` has **no prose to trim** — its chart section is the `02 — Test` eyebrow (`:143`) plus an `<h2>` (`:144`) and nothing else. §5's "trim the prose around its chart" is already satisfied. Do not invent a paragraph to shorten.

Roughly 45% less body copy is the *direction*, not an acceptance gate: no test asserts a word count, and one that did would fail on the next copy edit.

**Files:**
- Modify: `dashboard/landing/src/components/home/WhyCare.tsx:11-30,69-73` (intro → one sentence; ACT bodies → one line each; **headings and `EXTRAS` unchanged**)
- Modify: `dashboard/landing/src/components/home/Talk.tsx:2,18-31` (drop the three-step `<ol>` and the now-unused icon import)
- Test: `dashboard/backend/tests/test_landing_chart_first.py` (extend)

- [x] **Step 1: Write the failing test**

```python
def test_talk_drops_the_three_step_list_but_keeps_its_pinned_strings():
    """The <ol> restates WhyCare's three acts one screen later. Everything the
    existing suite pins about this section survives -- listed here so the trim
    does not discover them by reddening CI.
    """
    talk = (_HOME / "Talk.tsx").read_text(encoding="utf-8")
    assert "<ol" not in talk
    assert 'id="talk"' in talk
    assert "Describe your idea" in talk
    assert "Discord" in talk
    assert "<DiscordMock />" in talk
    assert talk.count("01 — Talk") == 1


def test_whycare_headings_are_untouched():
    whycare = _collapse((_HOME / "WhyCare.tsx").read_text(encoding="utf-8"))
    for heading in (
        "Describe it in plain English",
        "Prove it on real market data",
        "See how it ranks",
        "Pick the AI model",
        "For developers: bring your own agent",
    ):
        assert heading in whycare
    assert not re.search(r'"0[1-9]"', whycare), "quoted step numbers are banned here"
```

- [x] **Step 2: Run and watch fail**

```bash
pytest dashboard/backend/tests/test_landing_chart_first.py -v -k "talk or whycare"
```

Expected: FAIL on `assert "<ol" not in talk`.

- [x] **Step 3: Trim `Talk.tsx`**

Delete the `<ol>` at `:18-31` entirely — every `<li>` with it. Do **not** touch the surrounding `<section>`, the `<Button>`, `<ChatWindow />` or `<DiscordMock />`.

Then delete the whole lucide import at `:2`:

```tsx
import { MessageSquare, Bot, Hash } from "lucide-react";
```

Those three icons were used *only* by the `<li>` children. Nothing else in the file renders a lucide icon, so the line goes rather than shrinking — an unused import is a `noUnusedLocals` build failure in Task 10's `vite build`, not a lint nit you can leave.

- [x] **Step 4: Trim `WhyCare.tsx`**

The intro paragraph at `:69-73` becomes one sentence:

```tsx
          <p className="text-foreground/80 text-lg">
            Normally that means writing code, buying data, and waiting months to find out you were
            wrong.
          </p>
```

The three ACT bodies become one line each. The **third is already one line — leave its body and its comment exactly as they are**; only the first two change. The full array in its final state:

```tsx
const ACTS = [
  {
    icon: MessageSquare,
    title: "Describe it in plain English",
    body: "No code, no formulas — write it the way you would explain it to a person.",
  },
  {
    icon: LineChart,
    title: "Prove it on real market data",
    body: "Real prices and real market hours, measured against buy-and-hold and the index.",
  },
  {
    icon: Trophy,
    title: "See how it ranks",
    // Not "everyone else's agents": no user agent is on any board, and the
    // roster is curated (`dashboard/config/leaderboard.json`). The comparison
    // that actually exists is against the AI models and the passive baselines.
    body: "The same days and the same starting capital as every AI model on the board.",
  },
] as const;
```

`EXTRAS` (`:32-48`), all five `title`s, the `#landing-stats` anchor and both `scroll-mt-40`s are unchanged — the file's own comments explain why each exists, and the guard set pins the titles.

⚠ Do **not** write a quoted `"01"`–`"09"` anywhere in `WhyCare.tsx` while editing: `test_band_runs_no_second_step_sequence` greps for one, and the file's header comment says so for exactly this reason.

- [x] **Step 5: Run the full landing guard set**

```bash
pytest dashboard/backend/tests/test_landing_chart_first.py \
       dashboard/backend/tests/test_landing_copy_register.py \
       dashboard/backend/tests/test_landing_value_band.py -v
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add dashboard/landing/src/components/home/ dashboard/backend/tests/test_landing_chart_first.py
git commit -m "ux(landing): trim WhyCare and Talk body copy"
```

---

### Task 10: Rebuild the bundle and re-patch `index.html`

`dashboard/frontend/index.html` is hand-patched build output. Roughly 370 lines of auth-gate script, `#landingAuthModal`, `<style id="landing-auth-patch">` and the `[data-landing-auth]` delegated handler **cannot be produced by `vite build`**. Until this task runs, every bundle-reading guard is green against stale text — Tasks 6-9 proved nothing about what prod serves.

**Files:**
- Modify: `dashboard/frontend/index.html`
- Modify: `dashboard/frontend/assets/*`

- [x] **Step 1: Build**

```bash
cd dashboard/landing && npm install && npm run build
```

- [x] **Step 2: Copy the new assets in**

```bash
cp dashboard/landing/dist/public/assets/* dashboard/frontend/assets/
```

- [x] **Step 3: Delete the superseded bundles and repoint the two refs**

Remove the previous `index-*.js` and `index-*.css` from `dashboard/frontend/assets/`, then update the two `src=`/`href=` references in `dashboard/frontend/index.html` to the new hashed filenames. **Keep all four auth markers.**

- [x] **Step 4: Verify the patch did not drop vite output**

```bash
diff dashboard/landing/dist/public/index.html dashboard/frontend/index.html
```

Expected: **every differing line is `>`** (lines the hand-patch adds). Any `<` line means vite output was dropped — fix before committing.

- [x] **Step 5: Run the bundle-reading guards, which now read the new text**

```bash
pytest dashboard/backend/tests/test_landing_copy_register.py \
       dashboard/backend/tests/test_frontend_bundle_integrity.py \
       dashboard/backend/tests/test_landing_value_band.py -v
```

Expected: PASS. This is the first run of these that is evidence rather than a stale green.

- [x] **Step 6: Commit**

```bash
git add dashboard/frontend/index.html dashboard/frontend/assets/
git commit -m "build(landing): rebuild bundle for the chart-first hero"
```

---

# Phase C — cross-surface

---

### Task 11: Pin the numbers that must agree across surfaces

Spec §3d. §2's type-scale table reads like a shared contract and is not one: `/` is React + Recharts + Tailwind tokens, `/app` is vanilla JS + Chart.js + `styles.css`, and there is no shared code and no shared token between them. Changing the axis tick to 14px touches two Recharts props, a Chart.js options object, a cache-buster bump and a Vite rebuild — and **nothing fails if you do three of the four**.

**Heights are excluded from this pinning.** They differ by design.

**Files:**
- Test: `dashboard/backend/tests/test_landing_chart_first.py` (extend)

- [x] **Step 1: Write the test**

```python
def test_the_two_surfaces_agree_on_the_numbers_that_must_agree():
    """There is no shared code and no shared token between / and /app, so after
    this change there are two chart implementations with two axis-tick
    declarations and two legend treatments. That duplication is forced by the
    stacks and accepted; leaving it UNGUARDED is not. Pin the values that must
    match so the pair drifts loudly or not at all.

    Heights are deliberately absent: the surfaces have different vertical
    envelopes and therefore different clamps (spec §2). A shared height
    assertion here would be the bug it looks like a guard against. Units are
    the same kind of case and are asserted per-surface below, not shared.
    """
    home_js = (
        Path(__file__).resolve().parents[2] / "frontend" / "home-page.js"
    ).read_text(encoding="utf-8")

    # Axis ticks: 14px on both.
    assert "fontSize={14}" in _BOARD
    assert re.search(r"font:\s*\{\s*size:\s*14\s*\}", home_js)

    # The key's type scale: text-base on /, and /app's rows inherit the panel's
    # base size rather than the old 11px table register.
    assert "text-base" in _BOARD
    assert "hm-rank-swatch" in home_js

    # Neither surface draws a built-in legend: the standings/chips are the key.
    assert "<Legend" not in _BOARD
    assert re.search(r"legend:\s*\{\s*display:\s*false\s*\}", home_js)

    # UNITS: /app is percent. Asserted here rather than in the /app suite
    # because the pressure to break it comes from THIS comparison -- someone
    # noticing the two charts disagree and "aligning" screen 0 back to the
    # dollar axis / uses. That is a regression, not a tidy-up: / plots
    # fabricated curves that all share a base of 1000, so $1210 is unambiguous
    # and reads as SAMPLE_STANDINGS' +21.0%. Screen 0 plots live entries whose
    # `initial_equity` genuinely differs row to row (issue #365), so a shared
    # dollar axis there draws a $10k baseline against $100k models. The
    # divergence is load-bearing, exactly like the per-surface heights above.
    assert "(v * 100).toFixed(1)}%" in home_js
```

- [x] **Step 2: Run it**

```bash
pytest dashboard/backend/tests/test_landing_chart_first.py::test_the_two_surfaces_agree_on_the_numbers_that_must_agree -v
```

Expected: PASS if Tasks 3, 4 and 8 all landed. **If it fails, that is the point** — one of the four edits was missed.

- [x] **Step 3: Mutation-test it**

Revert `fontSize={14}` to `fontSize={11}` in `BoardPreview.tsx`, run the test (expect FAIL), restore. Then do the same to the Chart.js `size: 14`. A guard that passes against both breakages is decoration.

Then mutate the unit: swap screen 0's y-axis callback back to the dollar formatter (``callback: (v) => `$${Math.round(v).toLocaleString('en-US')}` ``) and run the test — expect FAIL. Restore. This is the mutation most likely to happen for a *plausible* reason, so it is the one that most needs to be red.

- [x] **Step 4: Commit**

```bash
git add dashboard/backend/tests/test_landing_chart_first.py
git commit -m "test(landing): pin the values the two chart surfaces must share"
```

---

### Task 12: The measured layout pass

The spec's acceptance criteria are measurements. **The clipping bug PR #357 shipped below 1200px was invisible to DOM probes and only a screenshot caught it** — read `getComputedStyle().display`, never the `hidden` attribute.

**1366×768 and 1280×720 are not optional.** They are the two viewports that falsified the first draft's heights, and both are ordinary laptop sizes. A viewport list that only samples 900px-tall screens cannot see the failure this design exists to fix.

**Files:**
- Create: `dashboard/scripts/verify_chart_first_layout.py`

- [x] **Step 1: Start the backend on a scratch DB with the LLM keys blanked**

```bash
cp dashboard/storage/data/backtest.db /tmp/claude-1000/scratch-backtest.db
DATABASE_PATH=/tmp/claude-1000/scratch-backtest.db \
ANTHROPIC_API_KEY= OPENAI_API_KEY= DEEPSEEK_API_KEY= COMMONSTACK_API_KEY= \
  uvicorn dashboard.backend.app:app --port 8077
```

python-dotenv loads with `override=False`, so an **empty** env var set in the process wins over `dashboard/.env` — which is what makes a paid LLM call impossible during the pass. Blank them, don't unset them. The scratch copy exists because a bare backend import runs lazy `ALTER`s against whatever `DATABASE_PATH` points at, and the committed seed DB is the prod database.

- [x] **Step 2: Write the script**

Create `dashboard/scripts/verify_chart_first_layout.py`:

```python
"""Measure the chart-first layout on both surfaces, at the viewports that matter.

Run against a local backend (see the plan's Task 12 Step 1 for the scratch-DB
invocation):

    ~/.venvs/htmlpdf/bin/python dashboard/scripts/verify_chart_first_layout.py

Exits non-zero on the first failed assertion, printing every measurement so a
near-miss is legible rather than a bare traceback.

WHY A SCRIPT AND NOT A PYTEST CASE: this needs a running server and a real
browser, and it is a pre-merge measurement pass, not a CI gate. The values it
confirms are pinned separately by Task 11's source guards, which do run in CI.
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8077"

# 1366x768 and 1280x720 are the two that falsified the first draft's heights.
# A list that only samples 900px-tall screens cannot see the bug this pass exists
# to catch. 390x844 is the stacked (below-lg) case.
VIEWPORTS = [
    (1280, 720),
    (1280, 800),
    (1366, 768),
    (1440, 768),
    (1440, 900),
    (1600, 900),
    (1920, 1080),
    (390, 844),
]

LG = 1024  # Tailwind's lg: breakpoint, where / stops stacking
PAGER_MIN = 1200  # below this /app stacks and the pager does not apply

failures: list[str] = []


def check(ok: bool, label: str, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: {detail}")
    if not ok:
        failures.append(f"{label}: {detail}")


def clamp(lo: float, preferred: float, hi: float) -> float:
    return max(lo, min(preferred, hi))


def measure_landing(page, width: int, height: int) -> None:
    page.goto(f"{BASE}/", wait_until="networkidle")
    m = page.evaluate(
        """() => {
        const card = document.querySelector('[data-testid="board-preview"]')
            || document.querySelector('#hero .rounded-xl, header .rounded-xl')
            || document.querySelector('main .rounded-xl');
        const chartBox = card && card.querySelector('.recharts-responsive-container');
        const column = card && card.closest('div[class*="basis-2/3"], div[class*="lg:basis-2/3"]');
        const container = column && column.closest('.container');
        const chips = card && card.querySelector('.flex-nowrap');
        const r = (el) => el ? el.getBoundingClientRect() : null;
        return {
            card: r(card),
            chart: r(chartBox),
            column: r(column),
            container: r(container),
            chips: chips ? {scrollWidth: chips.scrollWidth, clientWidth: chips.clientWidth} : null,
            innerHeight: window.innerHeight,
            // getComputedStyle, never the `hidden` attribute: PR #357's clipping
            // bug was invisible to attribute probes.
            cardDisplay: card ? getComputedStyle(card).display : null,
        };
    }"""
    )

    check(m["card"] is not None, "/ card found", str(bool(m["card"])))
    if not m["card"]:
        return

    check(
        m["cardDisplay"] not in (None, "none"),
        "/ card is displayed",
        f"display={m['cardDisplay']}",
    )

    # The fold. The check the first draft lacked -- and the one that failed at
    # four viewports.
    bottom = m["card"]["y"] + m["card"]["height"]
    check(
        bottom <= m["innerHeight"] + 0.5,
        "/ card sits above the fold",
        f"bottom={bottom:.1f} innerHeight={m['innerHeight']}",
    )

    # The chart's own clamp: clamp(300px, calc(100dvh - 390px), 520px).
    if m["chart"]:
        expected = clamp(300.0, height - 390.0, 520.0)
        actual = m["chart"]["height"]
        check(
            abs(actual - expected) <= 2.0,
            "/ chart height matches its clamp",
            f"actual={actual:.1f} expected={expected:.1f}",
        )

    # Column width. THE DENOMINATOR IS THE CONTAINER, NOT THE VIEWPORT: this
    # same layout is 66.7% of the container but only 63.0-65.9% of the viewport,
    # so a guard that quietly switched denominators would sit within 3pp of its
    # own threshold. Guarded at 60% -- below the 2/3 target so gutters and
    # rounding cannot redden a correct layout, above 50% so a reverted split
    # still fails.
    if width >= LG and m["column"] and m["container"]:
        ratio = m["column"]["width"] / m["container"]["width"]
        check(
            ratio >= 0.60,
            "/ chart column >= 60% OF THE CONTAINER",
            f"ratio={ratio:.3f} column={m['column']['width']:.0f}"
            f" container={m['container']['width']:.0f}",
        )

    # Five chips, one row. Checked at 1440 specifically -- the width the strip
    # was designed against.
    if width == 1440 and m["chips"]:
        check(
            m["chips"]["scrollWidth"] <= m["chips"]["clientWidth"] + 1,
            "/ chip strip fits on one row",
            f"scrollWidth={m['chips']['scrollWidth']} clientWidth={m['chips']['clientWidth']}",
        )


def measure_app(page, width: int, height: int) -> None:
    page.goto(f"{BASE}/app", wait_until="networkidle")
    page.wait_for_timeout(1500)  # the leaderboard module loads over fetch
    m = page.evaluate(
        """() => {
        const screen = document.querySelector('#homeScreenLanding');
        const wrap = document.querySelector('.hm-rank-chart');
        const canvas = document.querySelector('#homeModuleRankChart');
        const list = document.querySelector('#homeModuleRankList');
        const rows = list ? Array.from(list.children) : [];
        const listBox = list ? list.getBoundingClientRect() : null;
        const chart = (canvas && window.Chart && window.Chart.getChart)
            ? window.Chart.getChart(canvas) : null;
        return {
            screen: screen
                ? {scrollHeight: screen.scrollHeight, clientHeight: screen.clientHeight}
                : null,
            chartHeight: wrap ? wrap.getBoundingClientRect().height : null,
            chartDisplay: wrap ? getComputedStyle(wrap).display : null,
            rowCount: rows.length,
            rowsInside: listBox
                ? rows.filter(r => r.getBoundingClientRect().bottom
                    <= listBox.bottom + 0.5).length
                : 0,
            rowBadges: rows.map(r => (r.textContent || '').includes('Baseline')),
            datasets: chart
                ? chart.data.datasets.map(d => ({
                    label: d.label,
                    dash: (d.borderDash || []).length,
                  }))
                : null,
        };
    }"""
    )

    # The pager clips with overflow:hidden and NO scrollbar, so this is the only
    # way to see it. A height assertion on the panel alone cannot.
    if width >= PAGER_MIN and m["screen"]:
        overflow = m["screen"]["scrollHeight"] - m["screen"]["clientHeight"]
        check(
            overflow <= 1,
            "/app screen 0 does not clip",
            f"scrollHeight-clientHeight={overflow}",
        )

    if m["chartHeight"] is not None:
        expected = clamp(140.0, height * 0.26, 280.0)
        check(
            abs(m["chartHeight"] - expected) <= 2.0,
            "/app chart height matches its clamp",
            f"actual={m['chartHeight']:.1f} expected={expected:.1f}",
        )
        check(
            m["chartDisplay"] != "none",
            "/app chart is displayed",
            f"display={m['chartDisplay']}",
        )

    check(
        m["rowCount"] == 7,
        "/app renders all 7 models",
        f"rowCount={m['rowCount']}",
    )
    check(
        m["rowsInside"] == m["rowCount"],
        "/app every row is inside the list's visible box",
        f"{m['rowsInside']}/{m['rowCount']} visible",
    )
    # The list stays models-only, which is what keeps app.html's pinned
    # "AI models only - ranked by return" literally true.
    check(
        not any(m["rowBadges"]),
        "/app rank list carries no baseline rows",
        f"baseline rows={sum(m['rowBadges'])}",
    )

    if m["datasets"] is not None:
        labels = {d["label"]: d for d in m["datasets"]}
        for name in ("Buy & Hold", "DJIA"):
            present = name in labels
            check(present, f"/app chart carries the {name} baseline", str(present))
            if present:
                check(
                    labels[name]["dash"] > 0,
                    f"/app {name} is dashed",
                    f"borderDash length={labels[name]['dash']}",
                )


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for width, height in VIEWPORTS:
            page = browser.new_page(viewport={"width": width, "height": height})
            print(f"\n=== {width}x{height} ===")
            measure_landing(page, width, height)
            measure_app(page, width, height)
            page.close()
        browser.close()

    print(f"\n{'-' * 60}")
    if failures:
        print(f"{len(failures)} FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all measurements pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Two selector notes for whoever runs this: the `/` card selector falls back through three candidates because the shipped bundle's class names are minified — if all three miss, add a `data-testid="board-preview"` to `BoardPreview.tsx`'s outer card rather than guessing at a fourth. And the `/app` chart is read via `Chart.getChart(canvas)` (Chart.js 4's registry) rather than a `window.homeRankChart` global, so Task 3 does not have to leak its module-level handle just to be measurable.

- [x] **Step 3: Run it against Phase A only, before Phase B lands**

```bash
~/.venvs/htmlpdf/bin/python dashboard/scripts/verify_chart_first_layout.py
```

That venv already carries Playwright + Chromium — **do not `pip install playwright` into a fresh environment**, and do not install a second browser driver.

Expected: every `/app` row PASS; `/` rows still reflect the old layout.

- [x] **Step 4: Run it again after Task 10**

Expected: all PASS.

- [x] **Step 5: Force the three `/app` fallback states and confirm none draws a chart**

- `unreachable` — block `/api/v1/leaderboard` in devtools. Expect: sample list, `unreachable` note, **no chart**.
- `empty` — respond `{"entries": []}`. Expect: sample list, `empty` note, **no chart**.
- **The third state** — respond with real entries whose `equity_curve` is `[]`. Expect: **real list, no sample note, and no chart**. A pass that only exercises the two `sample` reasons cannot distinguish this case from a working one.

- [x] **Step 6: Confirm the seed DB is untouched**

```bash
git status --porcelain dashboard/storage/data/
ls -la dashboard/storage/data/backtest.db*
```

Expected: no output from `git status`; `backtest.db` mtime unchanged and the `-wal` sidecar 0 bytes. **Never `git add -A` in this repo** — a bare backend import runs lazy `ALTER`s against the committed prod seed database, and the ALTERs hide in the untracked `-wal` sidecar.

- [x] **Step 7: Commit**

```bash
git add dashboard/scripts/verify_chart_first_layout.py
git commit -m "test(home): committed Playwright pass for the chart-first layout"
```

---

## Deviations from the spec, found while planning

Three facts the spec asserts differently from what the code actually contains. Each is settled above; recorded here so a reader who checks the spec first is not confused.

1. **The baseline entry ids are `buy_hold_djia` and `djia_index`**, not `buy_hold` and `market_index`. The spec names the *strategy* keys; `dashboard/config/leaderboard.json` shows the strategy for the buy-and-hold entry is actually `equal_weight_buyhold`, and `market_index` covers **two** entries (`spy_index`, `djia_index`). Task 2 selects on `entry_id`, which is the config's primary key and reaches the client verbatim.

2. **Dashed baselines are free.** `LEADERBOARD_STYLES` already carries `dash` per baseline (`'Buy & Hold': [10, 6]`, `DJIA: [8, 4, 2, 4]`), and `getSeriesStyle` returns it. Reusing `buildEquityCurvesFromEntries` satisfies "dashed, unranked reference curves" with no new styling code.

3. **The model palette has five colours and the board has seven models.** `getModelColor` assigns `MODEL_COLOR_PALETTE[n % 5]` in first-seen order, so models 6 and 7 are handed models 1 and 2's colours. Cosmetic while the colour was decoration; not cosmetic once the rank row's swatch is the chart's only key. Task 4 widens the palette to ten — which also fixes the existing full leaderboard chart, where the collision ships today.

And one measurement gap:

4. **The `/` clamp was measured at the *current* 672px card width, beside a full-width copy column.** The sweep patched the chart height and the standings block, not the column split. At 1/3 width the copy column wraps more and can become the taller column, which — the hero row being `items-center` — pushes the card's bottom edge down. This is why Task 6 (copy trim) precedes Task 7, and why Task 12's fold check is a gate rather than a formality.

## Out of scope

- Making `/`'s chart real. It stays illustrative: `/` is served statically from Vercel, and a cross-origin fetch to Render on first paint is a cold-start gamble on the acquisition page.
- The season engine (issue #354) and the two open design questions (#355).
- Refreshing `README.md`'s `snapshot.png`, which this change makes stale for the second time this month. File as a follow-up issue at merge — a review finding left in a merged PR's comments is already dead when written.
- The `SAMPLE_CURVES` axis-vs-window mismatch (spec, "Accepted knowingly"). Pre-existing. If Task 8's caption rewrite makes it convenient, align the wording with the window the returns actually come from; do not chase it otherwise.
- **Fixing issue #365 itself.** The chart must survive mixed `initial_equity`; reconciling the config with the published curves is a backend change on the leaderboard service and does not belong in a frontend plan.

---

## Amendments

### 2026-08-16 — the screen 0 chart plots percent, not dollars

Found while briefing this plan against `main` at `04850df`. Two edits as drafted disagreed with each other and with the data:

- `homeChartSeries` returned `values: curves[label]` — **raw equity dollars** — and discarded `built.initials`, the per-entry `initial_equity` that `buildEquityCurvesFromEntries` already computes and returns.
- Task 3's y-axis then formatted those dollars: ``callback: (v) => `$${Math.round(v).toLocaleString('en-US')}` ``.

Two problems, one of them a correctness bug:

1. **The chart and its own key were in different units.** Screen 0's rank list renders `+7.5%` (`homeFormatReturnBadge`, `home-page.js:1206`), and Task 4's whole idea is that the list *is* the chart's key. A dollar axis labels the curves in units the key does not use.
2. **The rows do not share a capital base.** Config says `initial_capital: 10000`; all 12 published curves were computed at `$100,000`; `_find_cached_run` does not key on `initial_equity`; the 5 baselines are `auto_compute`-true and the 7 models are not. One `?refresh=true` therefore yields a payload mixing $10k baselines with $100k models (issue #365), which a dollar axis draws as a 10× scale break with the reference lines flat on the floor.

The second is not hypothetical-in-principle — it is reachable today by hand. It is *dormant* only because the daily cron is paused (PR #352) and `LEADERBOARD_DAILY_AUTO_DEPLOY` is off. **Do not force-refresh the board while building this**, which is otherwise a natural thing to do to see the new chart with fresh data.

Worth naming plainly: `isHomeModelEntry` filtering baselines out of the chart's source is currently the only thing hiding #365 on this surface. Task 2 removes that filter deliberately and correctly — the chart cannot answer "is +21% good?" without a reference line — which is precisely why the normalisation has to land in the same task.

**Changes made:**

| Where | Change |
|---|---|
| Global Constraints | New **Units** constraint: `/app` percent, `/` may keep dollars, with the reasoning |
| Task 2, `_entry` | `initial_equity` becomes a parameter (default `10000`) — a fixture with one shared base cannot fail on mixed capital |
| Task 2, harness | Extract `transformLeaderboardChartData` from `js/leaderboard.js` |
| Task 2, tests | `test_mixed_initial_equity_does_not_break_the_chart`, `test_home_chart_matches_the_leaderboards_percent_formula` |
| Task 2, `homeChartSeries` | Normalise per series by its own `initial_equity` |
| Task 2, Interfaces | `values` documented as fractions |
| Task 3, chart options | Percent y-axis at `toFixed(1)`; tooltip callback (the default prints the raw fraction) |
| Task 11 | Pin `/app`'s percent formatter; unit divergence documented as load-bearing; mutation step added |
| Out of scope | Fixing #365 itself |

**Not changed, deliberately:** `/`'s dollar axis. `SAMPLE_CURVES` is fabricated and every series shares a base of 1000, so `$1210` is unambiguous and equals `SAMPLE_STANDINGS`' `+21.0%`. Aligning it to percent is defensible but is churn in guarded copy for no correctness gain, and Task 8 already touches that component.

**Not re-verified:** the measured height budgets. Those were measured in-browser, which is the right method; re-deriving them by arithmetic is the mistake that falsified both earlier drafts.

### 2026-08-16 (later) — the units decision stands; its stated justification was wrong

Found while executing Task 12, by seeding a scratch database from the committed seed DB and reading what `/api/v1/leaderboard` actually serves. **The decision does not change** — Task 2's per-series normalisation, Task 3's percent axis and tooltip, and Task 11's formatter pin all stay exactly as amended above. Only the reasoning changes, and it was wrong in the dangerous direction: it asserted a correctness guarantee that does not exist, which reads as a licence to drop the guard the day someone simplifies it.

**Refuted: "one `?refresh=true` yields a payload a dollar axis draws as a 10× scale break."** Measured false. `get_leaderboard` rescales **each entry by its own** stored `initial_equity` (`scale = display_capital / stored_initial`, `service.py:1204-1211`), reports `"initial_equity": display_capital` for **every** entry (`:1240`), and `chart_equity_curve` opens every series at that same value (`baselines.py:115-145`). Serving a hand-built database with `buy_hold_djia` at $10k beside models at $100k produced curves that **all opened at $10,000**, with buy-and-hold's served points identical to its unmixed run. A dollar axis would have rendered fine. The live payload confirms it: all 12 entries report `initial_equity: 10000.0` and open at `10000.0`, while all 12 stored rows are at `100000.0`.

**What #365 actually costs**, since the plan should not carry a wrong version of a real issue: the harm is to the **returns**, not the axis. A baseline recomputed at $10k trades in a far coarser share quantum, so its curve differs from the $100k curves it is ranked against. No choice of y-axis repairs that. "Do not force-refresh the board while building this" still stands — for comparability, not for the chart.

**Also corrected: the formatter citation.** The rank list renders `homeFormatReturnPct` (`home-page.js:719`, called at `:1686`) → `+7.49%`, not `homeFormatReturnBadge` (`:1206`) → `+7.5%`. `homeFormatReturnBadge` takes a *run* and belongs to the My Agents activity list. The implementation already follows the correct one (tooltip at `toFixed(2)`, matching the neighbouring row; the axis keeps `toFixed(1)` for its own stated reason).

**A claim I made and then had refuted, recorded so it is not re-raised.** Reading the `or` in `run.get("initial_equity") or config.get("initial_capital", …)` (`service.py:1205`), I argued a stored `0.0` collapses `stored_initial` to the config capital, forces `scale = 1.0`, and leaves a series that opens at the synthetic `$10,000` and jumps to `$100,000`. The mechanism is real and I reproduced the rendering — but from an **unreachable** state. Both leaderboard writers set `initial_equity = float(equity_curve[0]["equity"])` (`baselines.py:91-96`) and store *that same curve* (`service.py:797/820`, `:1118/1142`), so the column is identical to the curve's first point by construction — visible in the seed data, where the Gemini row carries the float artifact `100000.00000000003` in both. `initial_equity` is also `NOT NULL`, so the NULL variant cannot exist either. It is a latent falsy-value smell worth tightening some day, **not** a live defect and not this plan's job. My fixture created a database state the write path cannot produce, which is the failure mode a hand-edited fixture invites.

**Consequent edits made on this branch:** the `homeChartSeries` comment (`home-page.js`) and `test_mixed_initial_equity_does_not_break_the_chart`'s docstring both stated the refuted #365 rationale; both now state the measured one and mark the per-series division as defence-in-depth on a pure function rather than a live fix. The test itself is unchanged and still earns its place.

### 2026-08-16 — executed; three plan defects found by measurement

All 12 tasks landed. Backend suite **3083 passed, 84 skipped**; the measurement pass is green at all 8 viewports on both surfaces plus all 3 fallback states. What the plan got wrong, all of it caught by measuring rather than by review:

1. **The `390` chart reserve is a desktop-only number.** The plan derived it from ~227px of non-chart card height and asserted non-negative fold slack "at every tested viewport", 390×844 included. Measured there, the card ran **77px past the fold**: stacked at phone width the title, the `Illustrative example` chip and the caption all wrap, and the non-chart height is **335px** against **132px** of section padding. Now two reserves on an arbitrary property (`480` below `lg`, `390` at `lg`+), because the formula's commas defeat Tailwind's arbitrary-*value* parser but an arbitrary *property* takes a breakpoint prefix. Both compile into the shipped CSS; verified.

2. **Two instrumentation bugs made a correct layout look broken.** The drafted script measured the Recharts container but compared it to the clamp, which sits on the wrapper — a permanent 16px deficit. Worse, it measured *during* the hero's framer-motion entrance: `getBoundingClientRect` reports the transformed box, so every reading came back at exactly **0.95×** (494 against a 520px clamp, 313.5 against 330). A uniform 5% error reads as a subtle CSS discrepancy, not an instrumentation fault. Note a "has the height stopped changing" wait does **not** catch it — during the 0.3s delay the scale is constant at 0.95, so two polls agree on a stationary wrong number. The wait keys on the transform reaching identity.

3. **A children-count wait is satisfied by static markup.** `app.html` ships `<li class="home-module-rank-empty">Loading the standings…</li>`, so waiting for the rank list to have children returned before any JS ran and produced a clean sweep of `/app` failures. The wait is now "the placeholder cleared".

**Deviations from the plan, all deliberate:**
- Task 12 Step 5's three fallback states are **scripted** (`--fallbacks`) rather than clicked through devtools, and their fixtures are the live payload mutated rather than hand-written. The third state — real entries with empty `equity_curve` — is the one a manual pass tends to skip, and it now runs every time.
- The `/`-source guards **strip comments before scanning**. Mutation-tested: a class deleted but still named in a comment satisfies an `in` guard otherwise, which is how PR #357's claim scans went green against the wrong file.
- Task 2 Step 5's cross-file export test was relocated into Task 3, where the consuming half exists; as drafted it could never pass in its own commit.

**Follow-ups for merge:** `README.md`'s `snapshot.png` is stale again (file an issue, do not regenerate here). `service.py:1205`'s `run.get("initial_equity") or …` is a latent falsy-value smell — presently unreachable, see the amendment above; not this branch's job.
