# nof1 Leaderboard Frame + Live Landing Hero — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give all three leaderboard charts one nof1-derived visual frame — curves stopping short of the right edge, a reserved gutter carrying each curve's owner name and value, an x-axis running on into a forward arrow — and put the signed-out landing hero on the same live Competition data the signed-in Home screen already draws.

**Architecture:** `dashboard/frontend/js/leaderboard.js` becomes the single source of the frame for both vanilla Chart.js surfaces: two plugin **factories** (`createEndpointLabelPlugin`, `createAxisArrowPlugin`) plus the pure geometry function they share (`boardFrameLayout`), exported on `window` exactly as `buildEquityCurvesFromEntries` and `getSeriesStyle` already are. `home-page.js` installs them. The landing is a separate stack (Vite/React + Recharts) and gets a second implementation of the same contract in `dashboard/landing/src/lib/boardFrame.ts` + a Recharts `Customized` overlay, with the shared numbers pinned across the two implementations by a guard test. Landing data moves from hardcoded constants to one shared fetch in `src/lib/useLeaderboard.tsx`, consumed by both `BoardPreview` and `Race`.

**Tech Stack:** Chart.js 4.4.0 (CDN UMD, classic scripts, no build step) · React 18 + Recharts 2.15.4 + Vite 6 + Tailwind v4 (`dashboard/landing`) · FastAPI + pytest (guards read frontend source and shell out to `node`).

**Source spec:** `docs/superpowers/specs/2026-08-19-nof1-leaderboard-frame-design.md` (approved 2026-08-19, commit `c30da53`).

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Three PRs, delivered in order A → B → C.** A and C are independent of everything; B depends on A for the visual rule only. Do not combine them.
- **Gutter split is 3:2.** The plot area takes **60%** of measured chart width, the reserved gutter **40%** (`BOARD_GUTTER_FRACTION = 0.4`). Deliberately more generous than nof1's ~18%; revisit only after seeing it rendered.
- **`layout.padding`, never the scale domain.** The gutter must stay *outside* `chartArea`. `resolveHoverTarget` (`js/leaderboard.js:1361`) rejects any `x > chartArea.right` outright; extending the category domain would move empty territory inside the plot and silently make the gutter hoverable.
- **No future date ticks.** Explicitly dropped by the user. The forward affordance is an arrowhead only. No tick labels appear in the gutter.
- **No "Now" marker.** The user's "move the Now point leftwards" is satisfied *by* the gutter — the last data point moves left because the plot narrows. Do not add a literal `Now` label: the Competition window closed `2026-05-15` and labelling its end "Now" is a false claim.
- **Every pill renders its own chart's y-axis unit.** Landing `%`, screen 0 `%`, Leaderboard tab follows `currentChartView` (`$` default, `%` toggle). No surface invents a unit its axis does not show.
- **Season length is 10 trading days** = two calendar weeks of US cash sessions, Mon–Fri. Already declared client-side at `dashboard/frontend/js/leaderboard.js:508` (`SEASON_TRADING_DAYS`).
- **`season.last_advanced_date` stays `null` and `season.trading_days_elapsed` stays `0`.** No season has advanced. `seasonHasAdvanced()` reads exactly those two fields, and it is the anchor under every preview disclaimer on the Live tab.
- **Never `git add -A` in this repo.** A bare backend import runs lazy `ALTER`s against the committed prod seed DB (`dashboard/storage/data/backtest.db`). Stage named paths only. Check `git status --short` before every commit and never stage `dashboard/storage/data/*`.
- **The shipped `dashboard/frontend/index.html` is hand-patched** (issue #225). It is *not* `vite build` output: the Vite template is 25 lines, the shipped file 418. Copying `dist/public/index.html` over it silently kills every landing CTA with no console error. Follow `dashboard/landing/README.md`.
- **Nothing in CI builds or type-checks the landing source.** The correspondence between `frontend/assets/index-*.js` and `landing/src` rests entirely on the author running `npm run build` and re-patching by hand.
- **Guard tests in the spec's §8 are updated, never deleted.** They red as a direct consequence of this work; that is part of the work.
- Run all pytest from the repo root: `pytest dashboard/backend/tests/ -v`.

### Deviations from the spec, deliberate and stated

Three places where implementation is stricter than §4.1/§6 asked for. Each removes a drift hazard the spec itself flagged rather than adding one.

1. **`GUTTER_MIN_PX` is measured at runtime, not frozen as a constant.** §4.1 says to measure the widest label once and record the number in a comment. `beforeLayout` has `chart.ctx`, so the plugin measures the *actual* label strings in the *actual* gutter font on every layout. A font or formatter change re-measures itself. This is the exact hazard §4.1 cites (`BoardPreview`'s `width={56}` measured at 11px, font later moved to 14px, four of five labels lost their leading `$`) — removed rather than re-documented.
2. **`BoardPreview`'s `width={56}` y-axis reserve is likewise measured, not re-guessed.** §6 says re-measure the percent labels. Measuring the two formatted domain endpoints with a canvas at render time achieves the same thing and cannot go stale.
3. **The landing fetches `/api/v1/leaderboard` root-relative, not through `MarketTicker.tsx`'s `apiBase()`.** §5.1 named `apiBase()` as the precedent. It is the wrong precedent: `dashboard/frontend/vercel.json` rewrites `/api/:path*` to Render, and `test_frontend_api_base.py::test_every_api_base_definition_uses_same_origin_off_localhost` requires an **empty** production base for exactly that reason, calling a hardcoded Render origin a "same-origin cookie auth regression". `MarketTicker` survives that guard only because it excludes minified `assets/`. A root-relative path is correct under Vercel *and* under local uvicorn, needs no localhost special case, and stays inside the `connect-src 'self'` CSP. (Under `npm run dev` at :5173 it hits the Vite server and fails — but so does `apiBase()`, which returns `window.location.origin` there. Neither pattern serves the dev server; note it and move on.) Bringing `MarketTicker` into line is a follow-up, not this work.

---

## File Structure

**PR A — the shared frame (two Chart.js surfaces)**

| File | Responsibility after this PR |
|---|---|
| `dashboard/frontend/js/leaderboard.js` | Owns the frame. New: geometry constants, `boardFrameLayout` (pure), `boardVisibleEndpoints`, `boardPillTextColor`, `boardRoundRect`, `createEndpointLabelPlugin`, `createAxisArrowPlugin`; both factories exported on `window`. `endpointLabelPlugin` the singleton is replaced by a factory call at the tab's own chart. |
| `dashboard/frontend/home-page.js` | Consumer. Installs both factories on screen 0's chart, degrading to no frame when the exports are absent. |
| `dashboard/backend/tests/test_frontend_board_frame.py` | **New.** Node-harness unit tests for `boardFrameLayout` + source-shape guards that both surfaces install the frame from the same factory. |
| `dashboard/backend/tests/test_frontend_leaderboard_hover.py` | Verified unchanged — the gutter widening must not make the gutter hoverable. |

**PR B — the landing hero on live data**

| File | Responsibility after this PR |
|---|---|
| `dashboard/landing/src/lib/boardFrame.ts` | **New.** The frame's numbers and pure geometry for the Recharts side: `BOARD_GUTTER_FRACTION`, gaps, pill/dot sizes, `stackLabels()`, `measureTextWidth()`, `pillTextColor()`. Mirrors `js/leaderboard.js`; pinned by a guard. |
| `dashboard/landing/src/lib/leaderboard.ts` | **New.** Transport + shaping: types, `fetchLeaderboard()`, `selectBoardEntries()`, `buildBoardSeries()`, `MODEL_COLOR_PALETTE`, `BASELINE_STYLES`, `BOARD_BASELINE_IDS`. No React. |
| `dashboard/landing/src/lib/useLeaderboard.tsx` | **New.** `LeaderboardProvider` + `useLeaderboard()`. One fetch shared by hero and Race. |
| `dashboard/landing/src/components/home/EndpointRail.tsx` | **New.** The Recharts `Customized` overlay: endpoint dots, stubs, staggered leaders, name + value pill, and the axis arrow. |
| `dashboard/landing/src/components/home/BoardPreview.tsx` | Modified. Loses `SAMPLE_CURVES`/`SAMPLE_STANDINGS`, gains three states, percent axis, measured y-width, gutter margin, the rail, a data-driven chip strip, re-derived height reserves. |
| `dashboard/landing/src/components/home/Race.tsx` | Modified. Standings table reads the same hook. |
| `dashboard/landing/src/pages/landing-page.tsx` | Modified. Wraps `<main>` in `LeaderboardProvider`. |
| `dashboard/frontend/index.html`, `dashboard/frontend/assets/*` | Rebuilt bundle + re-applied hand patch. |
| `dashboard/backend/tests/test_landing_copy_register.py`, `test_landing_chart_first.py` | Guards updated per spec §8. |

**PR C — the backend season block**

| File | Responsibility after this PR |
|---|---|
| `dashboard/config/leaderboard.json` | Gains a `season` block. |
| `dashboard/backend/domain/leaderboard/service.py` | `VALID_PERIODS` gains `"live"`; `resolve_leaderboard_config` gains a live branch; `get_leaderboard` attaches `season`. |
| `dashboard/backend/api/routers/leaderboard.py` | `period` query description names the third value. |
| `dashboard/backend/tests/test_leaderboard_season.py` | **New.** The payload contract and the not-advanced invariant. |
| `dashboard/backend/tests/test_frontend_live_trading_board.py` | Extended: the preview banner still shows once `live` is a real period. |

---

# PR A — the shared frame on both Chart.js surfaces

### Task 1: The frame's geometry, as one pure function

`boardFrameLayout` is the whole layout decision in one place: how wide the gutter is, whether labels are drawn at all, and how far apart they stack. Pure in `(chart.width, chart.height, label texts, fraction)`, so it is testable under `node` without a canvas or a DOM — which is the only way any of this gets tested in this repo.

It answers the two degradations up front. A card too **narrow** for the widest label (a 390px phone) and a chart too **short** to stack N labels (screen 0 at its 140px clamp floor) both resolve to *no labels, arrow only* — never to clipped text or an overlapping pile. On both surfaces there is already a complete key elsewhere (the tab's custom legend, screen 0's rank list), so the degradation loses nothing.

**Files:**
- Modify: `dashboard/frontend/js/leaderboard.js` — insert after `hexToRgba` (currently ends line 118)
- Test: `dashboard/backend/tests/test_frontend_board_frame.py` (create)

**Interfaces:**
- Consumes: `shortName(label)` and `hexToRgba(hex, alpha)`, both already module-level in `js/leaderboard.js`.
- Produces, for Tasks 2–4:
  - `boardFrameLayout(chart, labels, fraction) -> {gutter: number, drawLabels: boolean, gap: number}`
  - `boardVisibleEndpoints(chart, formatValue) -> Array<{i, lastIdx, anchorX, anchorY, color, name, value}>`
  - `boardSeriesColor(ds) -> string` (hex)
  - `boardPillTextColor(hex) -> string` (hex)
  - `boardDefaultValueText(ds, lastIdx) -> string`
  - `boardRoundRect(ctx, x, y, w, h, r) -> void` (paths only; caller fills)
  - constants `BOARD_GUTTER_FRACTION`, `BOARD_GUTTER_MAX_FRACTION`, `BOARD_GUTTER_FONT`, `BOARD_GUTTER_TEXT_INSET`, `BOARD_GUTTER_TRAILING_PAD`, `BOARD_LABEL_GAP_MAX`, `BOARD_LABEL_GAP_MIN`, `BOARD_LEADER_MIN_DISPLACEMENT`, `BOARD_PILL_PAD_X`, `BOARD_PILL_HEIGHT`, `BOARD_DOT_RADIUS`, `BOARD_STUB_LENGTH`, `BOARD_ARROW_PAD`, `BOARD_ARROW_HEAD_LENGTH`, `BOARD_ARROW_HEAD_HALF`, `BOARD_XAXIS_ALLOWANCE`, `BOARD_AXIS_COLOR`

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_frontend_board_frame.py`:

```python
"""Guards for the shared nof1-derived board frame (2026-08-19 spec §4).

One visual contract, two vanilla Chart.js implementations plus a Recharts one.
The duplication is forced by the stacks and accepted; leaving the *numbers*
unguarded is not, which is what test_the_two_surfaces_agree_on_the_numbers_that
_must_agree already establishes for this pair.

The geometry is exercised under node against the SHIPPING source, extracted by
name -- so a rename or deletion reddens these instead of leaving them passing
against a copy that no longer runs. Same harness shape as
test_frontend_leaderboard_hover.py.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_LEADERBOARD_JS = _FRONTEND / "js" / "leaderboard.js"
_HOME_JS = _FRONTEND / "home-page.js"
_SRC = _LEADERBOARD_JS.read_text(encoding="utf-8")
_HOME_SRC = _HOME_JS.read_text(encoding="utf-8")


def _extract_function(name: str) -> str:
    """The source of ``function <name>(...) { ... }``, brace-matched."""
    marker = f"function {name}("
    start = _SRC.index(marker)
    depth = 0
    index = _SRC.index("{", _SRC.index(")", start))
    while True:
        if _SRC[index] == "{":
            depth += 1
        elif _SRC[index] == "}":
            depth -= 1
            if depth == 0:
                return _SRC[start : index + 1]
        index += 1


def _board_constants() -> str:
    """Every ``const BOARD_* = <literal>;`` line, in source order.

    Extracted rather than restated: a number that only exists in this file is a
    number the guard cannot be wrong about.
    """
    lines = [
        ln
        for ln in _SRC.splitlines()
        if re.match(r"^const BOARD_[A-Z_]+ = .+;$", ln)
    ]
    assert lines, "no BOARD_* constants found -- the frame was renamed or deleted"
    return "\n".join(lines)


def _run_node(script: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    harness = "\n".join(
        [
            _board_constants(),
            _extract_function("boardFrameLayout"),
            _extract_function("boardLabelBlockWidth"),
            _extract_function("boardPillTextColor"),
            # Stub 2d context: every glyph is 6px wide. Enough to exercise the
            # width arithmetic without a canvas, and deliberately NOT a real
            # measurement -- the point of measuring at runtime is that no test
            # has to know the font metrics.
            """
function makeChart(width, height) {
  return {
    width,
    height,
    ctx: {
      save() {}, restore() {}, font: '',
      measureText(text) { return { width: String(text).length * 6 }; },
    },
  };
}
function makeLabels(n, name, value) {
  return Array.from({ length: n }, (_, i) => ({
    i, name: name || 'Model ' + i, value: value || '+1.00%',
  }));
}
""",
            script,
        ]
    )
    proc = subprocess.run(
        [node, "-e", harness], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_the_gutter_is_two_fifths_of_a_wide_chart():
    """Spec §4.1: plot 60%, gutter 40%, as a FRACTION of measured width so the
    ratio survives every breakpoint rather than holding at one design size."""
    result = _run_node(
        """
const frame = boardFrameLayout(makeChart(1200, 420), makeLabels(9), 0.4);
console.log(JSON.stringify({ gutter: frame.gutter, draw: frame.drawLabels }));
"""
    )
    assert result["gutter"] == pytest.approx(480.0)
    assert result["draw"] is True


def test_a_long_label_raises_the_gutter_above_the_fraction():
    """The 40% is a target, not a ceiling: at middling widths it clips, so the
    measured label block is a floor under it. Same width for both, so the only
    variable is the label."""
    result = _run_node(
        """
const short = boardFrameLayout(makeChart(400, 420), makeLabels(3, 'AI', '+1%'), 0.4);
const long = boardFrameLayout(
  makeChart(400, 420), makeLabels(3, 'DeepSeek V4 Pro', '-12.34%'), 0.4);
console.log(JSON.stringify({ short: short.gutter, long: long.gutter }));
"""
    )
    assert result["short"] == pytest.approx(160.0), "40% of 400 clears the short label"
    assert result["long"] > 160.0, "the measured block must be able to push past 40%"


def test_a_chart_too_narrow_for_its_labels_drops_them_rather_than_clipping():
    """A 390px phone card cannot carry `DeepSeek V4 Pro -12.34%`. Clipping is the
    failure this repo keeps re-learning (the chip strip cut four of five names
    with no scrollbar and nothing failing), so the frame gives the space back and
    draws the arrow alone. Both surfaces keep a complete key elsewhere."""
    result = _run_node(
        """
const frame = boardFrameLayout(
  makeChart(300, 420), makeLabels(5, 'DeepSeek V4 Pro', '-12.34%'), 0.4);
console.log(JSON.stringify({ gutter: frame.gutter, draw: frame.drawLabels }));
"""
    )
    assert result["draw"] is False
    assert result["gutter"] == pytest.approx(18.0), "arrow padding only"


def test_a_chart_too_short_to_stack_its_labels_drops_them_too():
    """Screen 0's panel clamps to `clamp(140px, 26vh, 280px)` and draws nine
    curves. Nine labels at the 13px minimum need 117px of plot; at the 140px
    floor there is not that much once the x-axis is taken out."""
    result = _run_node(
        """
console.log(JSON.stringify({
  tall: boardFrameLayout(makeChart(900, 280), makeLabels(9), 0.4).drawLabels,
  short: boardFrameLayout(makeChart(900, 140), makeLabels(9), 0.4).drawLabels,
}));
"""
    )
    assert result["tall"] is True
    assert result["short"] is False


def test_the_stagger_gap_tightens_before_it_gives_up():
    """20px is the comfortable gap and 13px the legibility floor. Between them
    the gap shrinks to fit rather than jumping straight to no labels."""
    result = _run_node(
        """
console.log(JSON.stringify({
  roomy: boardFrameLayout(makeChart(900, 600), makeLabels(4), 0.4).gap,
  tight: boardFrameLayout(makeChart(900, 200), makeLabels(9), 0.4).gap,
}));
"""
    )
    assert result["roomy"] == pytest.approx(20.0)
    assert 13.0 <= result["tight"] < 20.0


def test_no_labels_means_no_gap_to_stagger_by():
    """An empty board reserves nothing. Guards the divide-by-zero the gap
    formula would otherwise hit on `labels.length === 0`."""
    result = _run_node(
        """
const frame = boardFrameLayout(makeChart(900, 400), [], 0.4);
console.log(JSON.stringify({ gutter: frame.gutter, draw: frame.drawLabels, gap: frame.gap }));
"""
    )
    assert result["draw"] is False
    assert result["gap"] == 0


def test_pill_ink_follows_the_swatch_luminance():
    """Every palette entry today is a light tint on a dark page, so hardcoded
    dark ink would read -- until the first mid-dark colour lands, which is a
    one-line edit to dashboard/config/leaderboard.json away and would produce
    navy-on-navy with nothing failing."""
    result = _run_node(
        """
console.log(JSON.stringify({
  amber: boardPillTextColor('#FBBF24'),
  slate: boardPillTextColor('#94A3B8'),
  deep: boardPillTextColor('#1E3A8A'),
}));
"""
    )
    assert result["amber"] == "#0b1220"
    assert result["slate"] == "#0b1220"
    assert result["deep"] == "#f8fafc"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest dashboard/backend/tests/test_frontend_board_frame.py -v
```

Expected: every case FAILS. The first failure is in collection-time module setup — `_board_constants()`'s `assert lines` trips, because no `BOARD_*` constant exists yet.

- [ ] **Step 3: Add the constants and the pure geometry**

In `dashboard/frontend/js/leaderboard.js`, immediately after `hexToRgba` (which currently ends at line 118) and before `function getTeamColor`, insert:

```js
// ── The shared board frame (2026-08-19 spec §4) ──────────────────────────────
//
// One visual contract across three charts: curves stop short of the right edge,
// the reserved gutter carries each curve's owner and value, and the x-axis runs
// on through it to an arrowhead.
//
// The gutter is `layout.padding`, NOT extra scale domain, and that distinction
// is load-bearing rather than stylistic. Everything in this tab's hover gate
// reads `chartArea.right` as "the end of real data" -- `resolveHoverTarget`
// rejects `x > area.right` outright. Padding leaves that true. A domain padded
// with null future slots would put empty territory INSIDE the plot, and the
// gutter would silently become hoverable.
const BOARD_GUTTER_FRACTION = 0.4;
// The 40% is a target with a measured floor under it; this is the ceiling that
// stops the floor from eating the plot on a narrow card. Past it the frame
// gives the space back and draws no labels at all -- see boardFrameLayout.
const BOARD_GUTTER_MAX_FRACTION = 0.5;
const BOARD_GUTTER_FONT = '600 11px Inter, system-ui, sans-serif';
// Where the label block starts relative to chartArea.right, and the clear
// canvas left to the right of the widest one so the arrowhead has room.
const BOARD_GUTTER_TEXT_INSET = 12;
const BOARD_GUTTER_TRAILING_PAD = 16;
// Clear canvas kept between the plot's last tick label and the label block, so
// the gutter never reads as part of the axis.
const BOARD_TICK_CLEARANCE = 12;
// How far past the measured floor the gutter may grow. BOARD_GUTTER_FRACTION is
// the ceiling, the measured floor the minimum, and this is the slack between
// them -- a wide card should not donate plot width it has no labels to fill.
const BOARD_GUTTER_SLACK = 36;
// Gaps inside one label block: dot-to-name, then name-to-value pill.
const BOARD_DOT_GAP = 4;
const BOARD_NAME_GAP = 6;
const BOARD_PILL_PAD_X = 5;
const BOARD_PILL_HEIGHT = 15;
const BOARD_DOT_RADIUS = 3;
const BOARD_STUB_LENGTH = 7;
// Comfortable vertical spacing between stacked labels, and the floor below
// which 11px text stops being separable.
const BOARD_LABEL_GAP_MAX = 20;
// Ordering is load-bearing: BOARD_LABEL_GAP_MIN reads BOARD_PILL_HEIGHT, and a
// `const` read before its declaration is a TDZ ReferenceError at module load,
// not a lint warning. Keep the pill constants above the gap constants.
const BOARD_LABEL_GAP_MIN = BOARD_PILL_HEIGHT + 1;
// Only draw a leader line once collision-avoidance has displaced a label far
// enough that the connection is genuinely ambiguous; below this it is a stub.
const BOARD_LEADER_MIN_DISPLACEMENT = 7;
// Reserved right padding when the frame declines to draw labels: enough for the
// arrowhead and nothing else.
const BOARD_ARROW_PAD = 18;
const BOARD_ARROW_HEAD_LENGTH = 8;
const BOARD_ARROW_HEAD_HALF = 4;
// Conservative allowance for the x-axis, subtracted from canvas height to
// estimate plot height. Needed in `beforeLayout`, where chartArea is exactly
// what has not been computed yet -- and both the layout hook and the draw hook
// must reach the same drawLabels verdict or the gutter is reserved and empty.
const BOARD_XAXIS_ALLOWANCE = 34;
const BOARD_AXIS_COLOR = 'rgba(148, 163, 184, 0.45)';

/** The curve's own colour, from whichever field this surface carries it in.
 *
 *  This tab decorates datasets with `_style`; screen 0 sets a plain hex on
 *  `borderColor`. Reading both is what lets one factory serve both without
 *  either surface having to adopt the other's dataset shape. `borderColor` is
 *  second because `styleDatasets` rewrites it to an rgba with a hover alpha,
 *  and a faded swatch is not the series colour. */
function boardSeriesColor(ds) {
  return (ds && ds._style && ds._style.color) || (ds && ds.borderColor) || '#e5e7eb';
}

/** Dark or light pill ink, by the swatch's relative luminance. */
function boardPillTextColor(hex) {
  const h = String(hex || '').replace('#', '');
  const r = parseInt(h.slice(0, 2), 16) || 0;
  const g = parseInt(h.slice(2, 4), 16) || 0;
  const b = parseInt(h.slice(4, 6), 16) || 0;
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? '#0b1220' : '#f8fafc';
}

/** Width of the widest `dot name pill` block, measured in the gutter's own font.
 *
 *  MEASURED, not a recorded constant. The spec allowed either; this is the one
 *  that cannot go stale. `BoardPreview.tsx`'s `width={56}` is the cautionary
 *  case in this repo: measured correctly against `$1030` at 11px, then the tick
 *  font moved to 14px and four of five labels lost their leading `$` with
 *  nothing failing. A measurement taken at layout time re-takes itself. */
function boardLabelBlockWidth(chart, labels) {
  const ctx = chart.ctx;
  let widest = 0;
  ctx.save();
  ctx.font = BOARD_GUTTER_FONT;
  labels.forEach((lab) => {
    const block =
      BOARD_DOT_RADIUS * 2 +
      4 +
      ctx.measureText(lab.name).width +
      6 +
      ctx.measureText(lab.value).width +
      BOARD_PILL_PAD_X * 2;
    if (block > widest) widest = block;
  });
  ctx.restore();
  return BOARD_GUTTER_TEXT_INSET + widest + BOARD_GUTTER_TRAILING_PAD;
}

/** How much right padding to reserve, whether to draw labels, and how far apart.
 *
 *  Pure in (chart.width, chart.height, label texts, fraction) so it can be
 *  exercised under node without a canvas -- which, in this repo, is the only
 *  kind of chart test that exists.
 *
 *  TWO DEGRADATIONS, BOTH TO "ARROW ONLY". Too narrow for the widest label, or
 *  too short to stack N of them, and the frame gives the space back rather than
 *  clipping text or piling labels on each other. Clipping is the failure this
 *  codebase keeps re-learning: the chip strip on / silently cut four of five
 *  model names at 390px with no scrollbar, no ellipsis and nothing failing.
 *  Both surfaces keep a complete key elsewhere (this tab's custom legend,
 *  screen 0's rank list), so dropping the labels loses no information. */
function boardFrameLayout(chart, labels, fraction) {
  const none = { gutter: BOARD_ARROW_PAD, drawLabels: false, gap: 0 };
  if (!labels || !labels.length) return none;
  const usableHeight = chart.height - BOARD_XAXIS_ALLOWANCE;
  const gap = Math.min(BOARD_LABEL_GAP_MAX, usableHeight / labels.length);
  if (gap < BOARD_LABEL_GAP_MIN) return none;
  const floor = boardLabelBlockWidth(chart, labels);
  if (floor > chart.width * BOARD_GUTTER_MAX_FRACTION) return none;
  return { gutter: Math.max(chart.width * fraction, floor), drawLabels: true, gap };
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest dashboard/backend/tests/test_frontend_board_frame.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status --short   # confirm dashboard/storage/data/ is untouched
git add dashboard/frontend/js/leaderboard.js dashboard/backend/tests/test_frontend_board_frame.py
git commit -m "feat(leaderboard): board-frame geometry as one pure function"
```

---

### Task 2: The two plugin factories, wired to the Leaderboard tab

Factories, not singletons, because screen 0 draws the same frame over datasets carrying none of this tab's private fields and has no hover gate. Every surface-specific behaviour is an option whose default reproduces the tab.

This task also changes the tab's pill from always-percent to unit-following (spec §4.5). Today `endpointLabelPlugin` prints `+7.49%` even in the `$` view, which is the exact thing §4.5 forbids: a label in a unit the axis beside it does not show.

**Files:**
- Modify: `dashboard/frontend/js/leaderboard.js` — replace `endpointLabelPlugin` (currently lines 1283–1359); edit the chart config at `plugins:` (line 1458) and `layout:` (line 1461)
- Test: `dashboard/backend/tests/test_frontend_board_frame.py` (extend)

**Interfaces:**
- Consumes: everything Task 1 produced, plus `shortName`, `hexToRgba`, `formatLeaderboardNumber`, `currentChartView`, `hoveredDatasetIndex` (all already module-level).
- Produces, for Task 3:
  - `createEndpointLabelPlugin(options) -> ChartJsPlugin` where `options = {formatValue?, isFaded?, gutterFraction?}`; `formatValue(ds, lastIdx) -> string`, `isFaded(datasetIndex) -> boolean`, `gutterFraction` defaults to `BOARD_GUTTER_FRACTION`
  - `createAxisArrowPlugin() -> ChartJsPlugin`
  - Both plugins set/read `chart.$boardFrame`, the `boardFrameLayout` result for the current layout pass.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_frontend_board_frame.py`:

```python
def test_the_frame_is_built_by_factories_not_a_singleton():
    """Screen 0 draws the same frame over datasets that carry none of this tab's
    private fields (`_raw`, `_entry`, `_style`) and has no hover gate. A shared
    singleton would have to close over this module's `hoveredDatasetIndex` and
    `currentChartView`, so the only way to share it is to parameterise it."""
    assert "function createEndpointLabelPlugin(options)" in _SRC
    assert "function createAxisArrowPlugin(" in _SRC
    assert "const endpointLabelPlugin = {" not in _SRC, (
        "the singleton is replaced by a factory call, not kept beside it"
    )


def test_the_gutter_is_reserved_in_beforelayout_and_never_in_the_domain():
    """`layout.padding`, not scale domain. `resolveHoverTarget` rejects
    `x > chartArea.right` outright, so padding leaves the hover gate's whole
    premise true; a domain padded with future slots would move empty territory
    inside the plot and make the gutter hoverable.

    beforeLayout because the width is a FRACTION of the rendered width, which is
    not known at config time -- and beforeLayout is the last hook that can still
    move chartArea."""
    factory = _extract_function("createEndpointLabelPlugin")
    assert "beforeLayout(chart)" in factory
    assert "padding.right = frame.gutter" in factory
    assert "data.labels" not in factory, (
        "reserving space by appending empty category labels is the design that "
        "was dropped -- it puts the gutter INSIDE chartArea"
    )


def test_the_tab_lets_the_plugin_own_the_right_padding():
    """A literal `right: 120` left in the config is dead but not inert: it is
    what renders on the first frame before beforeLayout runs, and it is what a
    reader will believe."""
    layout = re.search(r"layout:\s*\{\s*padding:\s*\{[^}]*\}", _SRC)
    assert layout, "the tab's layout.padding block moved or was deleted"
    assert "right:" not in layout.group(0), (
        "the gutter is the frame's to compute; leaving a literal here renders it "
        "for one frame and misinforms every reader after that"
    )
    assert "top: 8" in layout.group(0), "the top padding is unrelated and stays"


def test_the_tab_pill_follows_the_axis_unit():
    """Spec §4.5: no surface invents a unit its axis does not show. This tab
    defaults to `$` (`currentChartView = 'absolute'`) and the endpoint label
    printed `+7.49%` in that view -- a percent beside a dollar axis."""
    call = _SRC[_SRC.index("createEndpointLabelPlugin({") :][:800]
    assert "currentChartView === 'absolute'" in call
    assert "formatLeaderboardNumber" in call, "the money branch reuses the axis formatter"
    assert "cumulative_return" in call, (
        "the percent branch keeps preferring the entry's stored return over the "
        "last plotted point"
    )


def test_the_hover_fade_stays_this_tabs_business():
    """`hoveredDatasetIndex` is this module's pointer-gate state. Screen 0 has no
    hover gate at all, so it must reach the factory as an injected predicate
    rather than a closed-over global."""
    factory = _extract_function("createEndpointLabelPlugin")
    assert "hoveredDatasetIndex" not in factory, (
        "the factory must not close over this tab's hover state"
    )
    assert "isFaded" in factory
    call = _SRC[_SRC.index("createEndpointLabelPlugin({") :][:800]
    assert "hoveredDatasetIndex" in call, "the tab injects it at the call site"


def test_each_curve_ends_in_a_dot_and_a_dotted_stub():
    """The handwritten note's `•⋯` mark: the curve carries on, and the stub
    asserts no value for where it goes."""
    factory = _extract_function("createEndpointLabelPlugin")
    assert "BOARD_DOT_RADIUS" in factory
    assert "BOARD_STUB_LENGTH" in factory
    assert factory.count("setLineDash([1, 3])") == 2, "the stub and the leader line"


def test_the_arrow_is_drawn_past_the_plot_and_not_as_an_axis_tick():
    """Chart.js can draw anywhere on the canvas, not only inside chartArea, so
    the forward affordance costs no scale configuration at all -- which is the
    whole reason the future-tick design was dropped."""
    arrow = _extract_function("createAxisArrowPlugin")
    assert "chart.width" in arrow, "the arrow tip is on the canvas edge, not chartArea"
    assert "BOARD_ARROW_HEAD_LENGTH" in arrow and "BOARD_ARROW_HEAD_HALF" in arrow
    assert "afterDraw(chart)" in arrow, (
        "chrome above the data: afterDatasetsDraw would let a curve running along "
        "the floor sit on top of the baseline"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_frontend_board_frame.py -v -k "factories or beforelayout or padding or pill or fade or stub or arrow"
```

Expected: all seven FAIL — `createEndpointLabelPlugin` does not exist.

- [ ] **Step 3: Replace the singleton with the two factories**

In `dashboard/frontend/js/leaderboard.js`, delete the whole `const endpointLabelPlugin = { … };` block (currently lines 1283–1359, the one whose comment begins "Right-endpoint labels: name + latest return") and put this in its place:

```js
/** The endpoint of each visible curve, and the two strings that label it.
 *
 *  Reads `ds.data` -- the PLOTTED values, already in whatever unit this chart's
 *  axis shows -- rather than any surface's private raw-curve field, which is
 *  what lets one factory serve two dataset shapes.
 *
 *  Called from `beforeLayout` as well as from the draw hook. In the layout pass
 *  `meta.data[i].x/y` are last frame's positions or undefined, which is fine:
 *  that pass only needs the label TEXT, to measure it. */
function boardVisibleEndpoints(chart, formatValue) {
  const out = [];
  chart.data.datasets.forEach((ds, i) => {
    const meta = chart.getDatasetMeta(i);
    if (meta.hidden || ds.hidden || !ds.data || !ds.data.length) return;
    let lastIdx = -1;
    for (let k = ds.data.length - 1; k >= 0; k -= 1) {
      if (ds.data[k] != null) { lastIdx = k; break; }
    }
    if (lastIdx < 0) return;
    const point = meta.data && meta.data[lastIdx];
    out.push({
      i,
      lastIdx,
      anchorX: point ? point.x : null,
      anchorY: point ? point.y : null,
      color: boardSeriesColor(ds),
      name: shortName(ds.label),
      value: formatValue(ds, lastIdx),
    });
  });
  return out;
}

/** Percent, two decimals, signed. The default for every surface whose axis is
 *  percent -- which is both of them except this tab in its money view. */
function boardDefaultValueText(ds, lastIdx) {
  const v = Number(ds.data[lastIdx]);
  if (!Number.isFinite(v)) return '';
  return `${v > 0 ? '+' : ''}${(v * 100).toFixed(2)}%`;
}

/** Path a rounded rect. Caller fills. `ctx.roundRect` is not in every browser
 *  this dashboard is opened in, and a missing method here would throw inside a
 *  draw hook -- which takes the whole chart down, not just the pill. */
function boardRoundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, h / 2, w / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

/** Right-endpoint labels: a colour-matched dot, the owner's name, and a value
 *  pill, laid out in the reserved gutter with vertical collision avoidance and
 *  dotted leaders to displaced labels.
 *
 *  A FACTORY. Screen 0 (home-page.js) draws this same frame over datasets that
 *  carry none of this tab's private fields and has no hover gate, so the two
 *  things that differ -- the value's unit and whether a series is faded -- are
 *  injected. The defaults reproduce this tab in its percent view.
 *
 *  The stagger, `BOARD_LEADER_MIN_DISPLACEMENT` and the `gutterStart`/`labelX`
 *  split are carried over unchanged from the plugin this replaces: labels drawn
 *  over the line endpoints left visible stubs, and a leader shorter than the
 *  displacement threshold is just a tick that connects nothing. */
function createEndpointLabelPlugin(options) {
  const opts = options || {};
  const formatValue =
    typeof opts.formatValue === 'function' ? opts.formatValue : boardDefaultValueText;
  const isFaded = typeof opts.isFaded === 'function' ? opts.isFaded : () => false;
  const fraction = Number.isFinite(opts.gutterFraction)
    ? opts.gutterFraction
    : BOARD_GUTTER_FRACTION;

  return {
    id: 'endpointLabels',
    // The gutter is a fraction of the RENDERED width, which is not known at
    // config time; beforeLayout is the last hook that can still move chartArea,
    // and `chart.width`/`chart.height` are already current there.
    beforeLayout(chart) {
      const labels = boardVisibleEndpoints(chart, formatValue);
      const frame = boardFrameLayout(chart, labels, fraction);
      chart.$boardFrame = frame;
      const layout = chart.options.layout || (chart.options.layout = {});
      const padding =
        layout.padding && typeof layout.padding === 'object'
          ? layout.padding
          : (layout.padding = {});
      padding.right = frame.gutter;
    },
    afterDatasetsDraw(chart) {
      const { ctx, chartArea } = chart;
      const frame = chart.$boardFrame;
      if (!frame || !frame.drawLabels || !chartArea) return;
      const labels = boardVisibleEndpoints(chart, formatValue).filter(
        (lab) => lab.anchorY != null,
      );
      if (!labels.length) return;

      // Stagger overlapping labels downward, keeping >= gap px spacing. Each
      // keeps its original endpoint y (anchorY) so we can tell later whether
      // collision-avoidance actually moved it.
      labels.forEach((lab) => { lab.y = lab.anchorY; });
      labels.sort((a, b) => a.y - b.y);
      for (let k = 1; k < labels.length; k += 1) {
        if (labels[k].y - labels[k - 1].y < frame.gap) {
          labels[k].y = labels[k - 1].y + frame.gap;
        }
      }
      // Shift the whole stack back inside the plot. Both clamps, not just the
      // bottom one: pushing an overflowing stack up can drive its head above
      // chartArea.top. It cannot exceed both bounds at once -- boardFrameLayout
      // only returns drawLabels when the stack fits in the plot height.
      const overflow = labels[labels.length - 1].y - chartArea.bottom;
      if (overflow > 0) labels.forEach((lab) => { lab.y -= overflow; });
      const underflow = chartArea.top - labels[0].y;
      if (underflow > 0) labels.forEach((lab) => { lab.y += underflow; });

      // Labels live entirely inside the reserved gutter so the plotted line
      // paths (which end at chartArea.right) never leave stubs under them.
      const gutterStart = chartArea.right + 6;
      const labelX = chartArea.right + BOARD_GUTTER_TEXT_INSET;
      ctx.save();
      ctx.font = BOARD_GUTTER_FONT;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      labels.forEach((lab) => {
        const faded = isFaded(lab.i);
        const alpha = faded ? 0.3 : 1;

        // The note's `•⋯`: a filled endpoint dot and a short dotted stub. It
        // says the curve continues; it asserts no value for where it goes.
        ctx.fillStyle = hexToRgba(lab.color, alpha);
        ctx.beginPath();
        ctx.arc(lab.anchorX, lab.anchorY, BOARD_DOT_RADIUS, 0, Math.PI * 2);
        ctx.fill();
        ctx.setLineDash([1, 3]);
        ctx.strokeStyle = hexToRgba(lab.color, faded ? 0.2 : 0.6);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(lab.anchorX + BOARD_DOT_RADIUS + 1, lab.anchorY);
        ctx.lineTo(lab.anchorX + BOARD_DOT_RADIUS + 1 + BOARD_STUB_LENGTH, lab.anchorY);
        ctx.stroke();
        ctx.setLineDash([]);

        if (Math.abs(lab.y - lab.anchorY) > BOARD_LEADER_MIN_DISPLACEMENT) {
          ctx.setLineDash([1, 3]);
          ctx.strokeStyle = hexToRgba(lab.color, faded ? 0.12 : 0.35);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(gutterStart, lab.anchorY);
          ctx.lineTo(labelX - 3, lab.y);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        let x = labelX;
        ctx.fillStyle = hexToRgba(lab.color, alpha);
        ctx.beginPath();
        ctx.arc(x + BOARD_DOT_RADIUS, lab.y, BOARD_DOT_RADIUS, 0, Math.PI * 2);
        ctx.fill();
        x += BOARD_DOT_RADIUS * 2 + 4;

        ctx.fillStyle = hexToRgba(lab.color, alpha);
        ctx.fillText(lab.name, x, lab.y);
        x += ctx.measureText(lab.name).width + 6;

        const pillWidth = ctx.measureText(lab.value).width + BOARD_PILL_PAD_X * 2;
        ctx.fillStyle = hexToRgba(lab.color, faded ? 0.25 : 1);
        boardRoundRect(ctx, x, lab.y - BOARD_PILL_HEIGHT / 2, pillWidth, BOARD_PILL_HEIGHT, 4);
        ctx.fill();
        const ink = boardPillTextColor(lab.color);
        ctx.fillStyle = faded ? hexToRgba(ink, 0.5) : ink;
        ctx.fillText(lab.value, x + BOARD_PILL_PAD_X, lab.y);
      });
      ctx.restore();
    },
  };
}

/** The x-axis line, running the full width and terminating in a right-pointing
 *  arrowhead out in the gutter.
 *
 *  The forward affordance, and the whole of it: the future-date-tick design was
 *  dropped precisely because Chart.js can draw anywhere on the canvas, so this
 *  costs no scale configuration and cannot touch the hover gate.
 *
 *  ACCEPTED IMPRECISION, STATED. On the Competition board the arrow implies an
 *  advancement a closed window does not perform -- that window ran 2026-04-15 →
 *  2026-05-15 and is finished. The mitigation is that the window label sits
 *  directly above the chart on every surface that draws this. It is a soft
 *  visual affordance rather than a data claim, and it was accepted knowingly. */
function createAxisArrowPlugin() {
  return {
    id: 'boardAxisArrow',
    // afterDraw, not afterDatasetsDraw: this is chrome, and a curve running
    // along the floor would otherwise sit on top of the baseline.
    afterDraw(chart) {
      const { ctx, chartArea } = chart;
      if (!chartArea) return;
      const y = Math.round(chartArea.bottom) + 0.5;
      const tipX = chart.width - 4;
      if (tipX <= chartArea.left + BOARD_ARROW_HEAD_LENGTH) return;
      ctx.save();
      ctx.strokeStyle = BOARD_AXIS_COLOR;
      ctx.fillStyle = BOARD_AXIS_COLOR;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(chartArea.left, y);
      ctx.lineTo(tipX - BOARD_ARROW_HEAD_LENGTH, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(tipX, y);
      ctx.lineTo(tipX - BOARD_ARROW_HEAD_LENGTH, y - BOARD_ARROW_HEAD_HALF);
      ctx.lineTo(tipX - BOARD_ARROW_HEAD_LENGTH, y + BOARD_ARROW_HEAD_HALF);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    },
  };
}
```

- [ ] **Step 4: Wire the tab's own chart to the factories**

In `renderEquityCurvesChart`, replace the `plugins:` array (currently line 1458) and the `layout:` line (currently line 1461):

```js
    plugins: [
      selectedGlowPlugin,
      hoverMarkerPlugin,
      createAxisArrowPlugin(),
      createEndpointLabelPlugin({
        // Spec §4.5: the pill renders THIS chart's axis unit. This tab defaults
        // to money, and the label printed `+7.49%` beside a `$` axis.
        formatValue(ds, idx) {
          if (currentChartView === 'absolute') {
            return `$${formatLeaderboardNumber(ds.data[idx])}`;
          }
          // The entry's stored return in preference to the last plotted point:
          // the point is the curve's own arithmetic, the field is what the run
          // recorded, and the rank column beside it renders the field.
          const ret =
            ds._entry && ds._entry.cumulative_return != null
              ? Number(ds._entry.cumulative_return)
              : Number(ds.data[idx]);
          if (!Number.isFinite(ret)) return '';
          return `${ret > 0 ? '+' : ''}${(ret * 100).toFixed(2)}%`;
        },
        isFaded: (i) => hoveredDatasetIndex != null && i !== hoveredDatasetIndex,
      }),
    ],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      // `right` is the frame's, computed per layout from the rendered width.
      // A literal here is not merely dead: it is what renders on the first
      // frame, and what every later reader believes.
      layout: { padding: { top: 8 } },
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_frontend_board_frame.py -v
pytest dashboard/backend/tests/test_frontend_leaderboard_hover.py -v
```

Expected: both PASS. The hover suite matters here specifically — its `test_endpoint_label_gutter_is_never_a_hit` probes `chartArea.right + 40` and `+ 110` against a synthetic 606px plot, and it must stay green because the gutter is still padding rather than domain.

- [ ] **Step 6: Commit**

```bash
git status --short
git add dashboard/frontend/js/leaderboard.js dashboard/backend/tests/test_frontend_board_frame.py
git commit -m "feat(leaderboard): endpoint-label and axis-arrow plugin factories"
```

---

### Task 3: Screen 0 draws the same frame

`home-page.js` already consumes four cross-file exports from `js/leaderboard.js` (`buildEquityCurvesFromEntries`, `getSeriesStyle`, `formatShortDate`, `formatChartTooltipLabel`), each explicit rather than implicit because the implicit form degrades to *no chart*, which this design deliberately makes indistinguishable from the honest no-curves state. The two frame factories join that list on the same terms.

**One thing that needs no change, and the evidence for it.** Screen 0's tooltip is `interaction: {mode: 'nearest', intersect: false}`, so the natural worry is that a 40% gutter becomes 40% of tooltip-firing surface. It does not. In Chart.js 4.4.0 `Interaction.modes.nearest` delegates to `getNearestItems`, whose first line is `return includeInvisible || chart.isPointInArea(position) ? … : []` — and `includeInvisible` defaults to false. A pointer outside `chartArea` resolves to zero items, so the gutter is inert for the tooltip exactly as it is for the tab's hover gate. Do not add a pointer gate to this panel.

**Files:**
- Modify: `dashboard/frontend/js/leaderboard.js` — append two `window.*` exports at the end of the file (after `window.formatChartTooltipLabel`, currently line 1623)
- Modify: `dashboard/frontend/home-page.js` — `renderHomeLeaderboardChart` (currently lines 1555–1694)
- Test: `dashboard/backend/tests/test_frontend_board_frame.py` (extend)

**Interfaces:**
- Consumes: `createEndpointLabelPlugin(options)`, `createAxisArrowPlugin()` from Task 2.
- Produces: `window.createEndpointLabelPlugin`, `window.createAxisArrowPlugin`; and in `home-page.js`, `homeBoardFramePlugins() -> Array<ChartJsPlugin>` (empty when the exports are absent).

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_frontend_board_frame.py`:

```python
def test_the_frame_factories_are_explicit_cross_file_exports():
    """Same contract as `buildEquityCurvesFromEntries`: explicit, not the
    implicit global these classic scripts share. On rename the implicit form
    degrades to a chart with no frame, and a frame that silently stops drawing
    looks exactly like a frame nobody asked for."""
    assert "window.createEndpointLabelPlugin = createEndpointLabelPlugin;" in _SRC
    assert "window.createAxisArrowPlugin = createAxisArrowPlugin;" in _SRC
    assert "window.createEndpointLabelPlugin" in _HOME_SRC
    assert "window.createAxisArrowPlugin" in _HOME_SRC


def test_screen_zero_installs_the_frame_on_its_chart():
    """Not merely defined next to it. The plugins array is what makes it draw."""
    assert "homeBoardFramePlugins()" in _HOME_SRC
    chart_call = _HOME_SRC[_HOME_SRC.index("new window.Chart(") :][:400]
    assert "plugins: homeBoardFramePlugins()" in chart_call


def test_screen_zero_says_so_when_the_frame_is_missing():
    """A missing export degrades to a frameless chart, which is a plausible
    design rather than a break -- so it needs a signal, exactly like the missing
    curve-builder case this module already warns about."""
    fn = _HOME_SRC[_HOME_SRC.index("function homeBoardFramePlugins()") :][:700]
    assert "console.warn" in fn
    assert "return []" in fn


def test_screen_zero_does_not_grow_a_pointer_gate():
    """`Interaction.modes.nearest` delegates to getNearestItems, which returns []
    unless `chart.isPointInArea(position)` -- so the widened gutter is already
    inert for this panel's tooltip. A hand-rolled gate here would be dead code
    imitating the Leaderboard tab, which needs one only because it sets
    `events: []`."""
    assert "pointermove" not in _HOME_SRC
    assert "resolveHoverTarget" not in _HOME_SRC


def test_screen_zero_keeps_its_percent_pill_by_taking_the_default():
    """The factory's default formatter is percent to two decimals, which is what
    the rank row beside each curve renders (`homeFormatReturnPct`). Passing a
    formatter here would be a second chance to render the same number two ways
    -- the reason this module borrows both axis formatters rather than writing
    its own."""
    fn = _HOME_SRC[_HOME_SRC.index("function homeBoardFramePlugins()") :][:700]
    assert "formatValue" not in fn
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_frontend_board_frame.py -v -k "cross_file or screen_zero"
```

Expected: all five FAIL — the exports and `homeBoardFramePlugins` do not exist.

- [ ] **Step 3: Export the factories**

At the very end of `dashboard/frontend/js/leaderboard.js`, after `window.formatChartTooltipLabel = formatChartTooltipLabel;`, add:

```js
// The shared board frame, consumed by home-page.js for screen 0's chart.
// Explicit for the same reason the four exports above are: the implicit global
// degrades on rename to a chart with no frame, and a frame that silently stops
// drawing is indistinguishable from a frame nobody asked for. Pinned from both
// sides by test_frontend_board_frame.py.
window.createEndpointLabelPlugin = createEndpointLabelPlugin;
window.createAxisArrowPlugin = createAxisArrowPlugin;
```

- [ ] **Step 4: Install the frame on screen 0**

In `dashboard/frontend/home-page.js`, insert immediately above `function renderHomeLeaderboardChart(series, times) {` (currently line 1555):

```js
/** The shared board frame's plugins, or none if js/leaderboard.js has not
 *  landed yet.
 *
 *  Screen 0 takes every default: the pill formatter is percent to two decimals,
 *  which is exactly what the rank row beside each curve renders
 *  (`homeFormatReturnPct`), and there is no hover gate here to fade against.
 *  Passing a formatter would be a second chance to render the same number two
 *  ways -- the same reason this module borrows both axis formatters rather than
 *  writing its own.
 *
 *  Warns on absence. A frameless chart is a plausible design rather than a
 *  visible break, so unlike the axis formatters (which degrade to an ugly label
 *  that is on screen and self-reporting) this one needs a signal. */
function homeBoardFramePlugins() {
    const labels = window.createEndpointLabelPlugin;
    const arrow = window.createAxisArrowPlugin;
    if (typeof labels !== 'function' || typeof arrow !== 'function') {
        console.warn('[home] board frame factories missing — drawing an unframed chart');
        return [];
    }
    return [arrow(), labels()];
}
```

Then, in `renderHomeLeaderboardChart`, add the `plugins` array to the Chart constructor. It currently reads:

```js
    homeRankChart = new window.Chart(wrap.querySelector('canvas'), {
        type: 'line',
        data: {
```

Change to:

```js
    homeRankChart = new window.Chart(wrap.querySelector('canvas'), {
        type: 'line',
        plugins: homeBoardFramePlugins(),
        data: {
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_frontend_board_frame.py dashboard/backend/tests/test_frontend_chart_first_home.py -v
```

Expected: PASS, including the whole existing screen-0 suite (the frame adds a plugin; it changes no series, no formatter and no height).

- [ ] **Step 6: Commit**

```bash
git status --short
git add dashboard/frontend/js/leaderboard.js dashboard/frontend/home-page.js dashboard/backend/tests/test_frontend_board_frame.py
git commit -m "feat(home): draw the shared board frame on screen 0"
```

---

### Task 4: Check it in a browser, and pin what the check settles

Everything above is source-shape and node-level. Two things can only be seen rendered: whether a 40% gutter reads well on screen 0's short, narrow panel, and whether nine staggered labels are legible at each breakpoint. Both are named risks in spec §9 and both have a defined remedy, so this is a gate with an outcome, not a look-and-see.

**Files:**
- Modify (only if the browser check fails): `dashboard/frontend/home-page.js`
- Test: `dashboard/backend/tests/test_frontend_board_frame.py` (extend)

**Interfaces:**
- Consumes: `homeBoardFramePlugins()` from Task 3.
- Produces: nothing new unless the remedy is taken, in which case `homeBoardFramePlugins()` passes `{gutterFraction: 0.3}` to `createEndpointLabelPlugin`.

- [ ] **Step 1: Start a backend against a scratch database**

```bash
cd /mnt/d/github/agent-trading-lab
DATABASE_PATH=/tmp/atl-frame-check.db ~/atl-venv/bin/python -m uvicorn dashboard.backend.app:app --port 8011
```

The scratch `DATABASE_PATH` is not optional: a bare backend import runs lazy `ALTER`s against the committed prod seed DB.

- [ ] **Step 2: Look at both charts at four widths**

Open `http://localhost:8011/app`, sign in, and check the **Leaderboard tab** and **Home screen 0** at 1920, 1440, 1280 and 390 CSS px wide. For each, confirm:

1. no gutter label is clipped at the canvas edge;
2. the arrowhead is fully on canvas and sits on the axis baseline;
3. every visible curve has an endpoint dot and a stub;
4. on the tab, toggling `$` ⇄ `%` changes the pill to match the axis;
5. on the tab, moving the pointer from a curve into the gutter *clears* the hover emphasis and does not re-light another curve;
6. at 390px and at screen 0's shortest height, the labels are cleanly **absent** rather than clipped or overlapping.

- [ ] **Step 3: Decide, and record the decision**

Two outcomes, no third:

- **Reads well** → change nothing. Add the note in Step 4 recording that 0.4 was checked on screen 0 and kept.
- **Screen 0's panel is too cramped** (a ~400px-wide panel gives a ~240px plot at 0.4, which is the specific risk §4.1 said to revisit after rendering) → the remedy is one option, on screen 0 only. In `homeBoardFramePlugins()`:

```js
    // 0.3, not the shared 0.4. Measured at <breakpoint>: this panel is <N>px
    // wide against the Leaderboard tab's <M>px, and the 3:2 split left the plot
    // too narrow to read <K> hourly points. The tab and the landing hero keep
    // 0.4 -- the split is a ratio for a full-width board, and this is a panel.
    return [arrow(), labels({ gutterFraction: 0.3 })];
```

Fill the bracketed measurements from what you saw. Leave the tab and the hero at `BOARD_GUTTER_FRACTION`.

- [ ] **Step 4: Write the guard that records the outcome**

Append to `dashboard/backend/tests/test_frontend_board_frame.py`:

```python
def test_the_shared_fraction_is_the_default_and_any_override_says_why():
    """Both surfaces take 0.4 unless a rendered check found otherwise. If screen
    0 overrides, the number must arrive as the factory's documented option --
    not as a second constant, and not by moving the shared default, which would
    silently narrow the plot on a full-width board to fix a panel."""
    assert "const BOARD_GUTTER_FRACTION = 0.4;" in _SRC
    fn = _HOME_SRC[_HOME_SRC.index("function homeBoardFramePlugins()") :][:900]
    if "gutterFraction" in fn:
        assert re.search(r"Measured at \S", fn), (
            "an override must carry the measurement that justified it"
        )
    assert "BOARD_GUTTER_FRACTION" not in _HOME_SRC, (
        "the fraction is the frame's; screen 0 either takes it or passes an option"
    )
```

- [ ] **Step 5: Run the whole frontend guard set**

```bash
pytest dashboard/backend/tests/test_frontend_board_frame.py \
       dashboard/backend/tests/test_frontend_leaderboard_hover.py \
       dashboard/backend/tests/test_frontend_chart_first_home.py \
       dashboard/backend/tests/test_frontend_live_trading_board.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit and open PR A**

```bash
git status --short
git add dashboard/backend/tests/test_frontend_board_frame.py dashboard/frontend/home-page.js
git commit -m "test(leaderboard): pin the browser-checked gutter fraction"
git push -u origin HEAD
gh pr create --title "feat(leaderboard): nof1-style board frame on both charts" --body "$(cat <<'BODY'
Shared visual frame on the Leaderboard tab and Home screen 0, per
docs/superpowers/specs/2026-08-19-nof1-leaderboard-frame-design.md §4.

- Reserved right gutter at a 3:2 plot:gutter split, computed per layout from
  measured width, with a measured label-block floor and a drop-to-arrow-only
  degradation at both narrow and short extremes.
- Endpoint dot + dotted stub, owner name, colour-matched value pill.
- x-axis runs through the gutter to a forward arrowhead.
- The tab's pill now follows `currentChartView` instead of always printing
  percent beside a dollar axis.

The gutter is `layout.padding`, never scale domain, so `chartArea.right` still
means "end of real data" and the hover gate is untouched.

Landing hero (PR B) and the backend season block (PR C) follow separately.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

---

# PR B — the landing hero on live Competition data

> **Blocked by PR A**, for the visual rule only. Land A first so the numbers this PR mirrors already exist in `js/leaderboard.js`.

### Task 5: The frame's geometry, mirrored for the Recharts side

The landing is a different stack, so the frame is implemented twice. That duplication is forced and accepted; leaving the *numbers* unguarded is not — which is the standing arrangement in this repo (`test_the_two_surfaces_agree_on_the_numbers_that_must_agree`). This task puts every shared number and the stagger algorithm in one TS module and pins it against `js/leaderboard.js`.

**On test coverage, honestly.** The mirror-constant guard is a source scan and runs everywhere. The behavioural test transpiles the TS with the `esbuild` already inside `dashboard/landing/node_modules` and runs it under `node` — so it **skips in CI**, which installs Python requirements only and never touches `npm`. It is a local-only tier, like the Postgres-only tests. Say so in the module docstring rather than letting a green CI imply the geometry was exercised.

**Files:**
- Create: `dashboard/landing/src/lib/boardFrame.ts`
- Test: `dashboard/backend/tests/test_landing_board_frame.py` (create)

**Interfaces:**
- Consumes: the constants in `dashboard/frontend/js/leaderboard.js` (mirrored, not imported — different bundles).
- Produces, for Tasks 8–9:
  - `frameLayout(input: {width, height, labels: LabelText[], fraction?}) -> {gutter: number, drawLabels: boolean, gap: number}` where `LabelText = {name: string; value: string}`
  - `stackLabels(anchors: Anchor[], opts: {gap, top, bottom}) -> Placed[]` where `Anchor = {key, anchorX, anchorY}` and `Placed = Anchor & {y: number; displaced: boolean}`
  - `measureTextWidth(text: string, font: string) -> number`
  - `labelBlockWidth(labels: LabelText[]) -> number`
  - `pillTextColor(hex: string) -> string`
  - constants `BOARD_GUTTER_FRACTION`, `BOARD_GUTTER_MAX_FRACTION`, `BOARD_GUTTER_FONT`, `BOARD_GUTTER_TEXT_INSET`, `BOARD_GUTTER_TRAILING_PAD`, `BOARD_LABEL_GAP_MAX`, `BOARD_LABEL_GAP_MIN`, `BOARD_LEADER_MIN_DISPLACEMENT`, `BOARD_PILL_PAD_X`, `BOARD_PILL_HEIGHT`, `BOARD_DOT_RADIUS`, `BOARD_STUB_LENGTH`, `BOARD_ARROW_PAD`, `BOARD_ARROW_HEAD_LENGTH`, `BOARD_ARROW_HEAD_HALF`, `BOARD_XAXIS_ALLOWANCE`, `BOARD_AXIS_COLOR`

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_landing_board_frame.py`:

```python
"""The board frame is implemented twice; these pin the two copies together.

`js/leaderboard.js` (vanilla Chart.js, two surfaces) and
`landing/src/lib/boardFrame.ts` (Recharts, one surface) draw the same contract
on different stacks. The duplication is forced by the stacks and accepted --
leaving the NUMBERS unguarded is not, which is the arrangement
test_the_two_surfaces_agree_on_the_numbers_that_must_agree already establishes
for the other pair.

TWO TIERS, AND THE SECOND SKIPS IN CI. The constant mirror is a source scan and
runs everywhere. The behavioural test transpiles the TS with the esbuild inside
dashboard/landing/node_modules and runs it under node, so it needs an `npm
install` CI does not do. A green CI therefore says the numbers agree, NOT that
the geometry was exercised -- run this suite locally before shipping a change to
either copy.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LEADERBOARD_JS = (_ROOT / "frontend" / "js" / "leaderboard.js").read_text(encoding="utf-8")
_BOARD_FRAME_TS_PATH = _ROOT / "landing" / "src" / "lib" / "boardFrame.ts"
_ESBUILD = _ROOT / "landing" / "node_modules" / ".bin" / "esbuild"

_JS_CONST = re.compile(r"^const (BOARD_[A-Z_]+) = (.+);$", re.M)
_TS_CONST = re.compile(r"^export const (BOARD_[A-Z_]+) = (.+);$", re.M)


def _constants(source: str, pattern: re.Pattern) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in pattern.finditer(source)}


def test_both_copies_declare_the_same_frame_constants():
    js = _constants(_LEADERBOARD_JS, _JS_CONST)
    ts = _constants(_BOARD_FRAME_TS_PATH.read_text(encoding="utf-8"), _TS_CONST)
    assert js, "no BOARD_* constants in js/leaderboard.js"
    assert set(js) == set(ts), (
        f"the two copies of the frame declare different constants; "
        f"only in js: {sorted(set(js) - set(ts))}, only in ts: {sorted(set(ts) - set(js))}"
    )
    for name in sorted(js):
        assert js[name] == ts[name], (
            f"{name} disagrees: js={js[name]} ts={ts[name]}"
        )


def _run_ts(script: str):
    """Transpile boardFrame.ts to CJS and run `script` against it under node."""
    node = shutil.which("node")
    if not node or not _ESBUILD.is_file():
        pytest.skip("node and dashboard/landing/node_modules are required")
    bundled = subprocess.run(
        [str(_ESBUILD), str(_BOARD_FRAME_TS_PATH), "--bundle", "--format=cjs",
         "--platform=node", "--log-level=error"],
        capture_output=True, text=True, timeout=60,
    )
    assert bundled.returncode == 0, bundled.stderr
    proc = subprocess.run(
        [node, "-e", bundled.stdout + "\n" + script],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_a_wide_card_reserves_the_measured_floor_plus_slack_not_two_fifths_of_the_width():
    result = _run_ts(
        """
const labels = Array.from({length: 9}, (_, i) => ({name: 'Model ' + i, value: '+1.00%'}));
const frame = module.exports.frameLayout({width: 900, height: 420, labels});
console.log(JSON.stringify({gutter: frame.gutter, draw: frame.drawLabels}));
"""
    )
    assert result["gutter"] == pytest.approx(180.0)
    assert result["draw"] is True


def test_a_narrow_card_drops_the_labels_rather_than_clipping_them():
    """390px is the stacked phone width the card is measured at."""
    result = _run_ts(
        """
const labels = Array.from({length: 9}, () => ({name: 'DeepSeek V4 Pro', value: '-12.34%'}));
const frame = module.exports.frameLayout({width: 300, height: 420, labels});
console.log(JSON.stringify({gutter: frame.gutter, draw: frame.drawLabels}));
"""
    )
    assert result["draw"] is False
    assert result["gutter"] == pytest.approx(18.0)


def test_the_stagger_separates_coincident_endpoints():
    """The real board spans -0.43% to +7.49%, so nine endpoints land within a
    few pixels of each other. Without the stagger they are one smear."""
    result = _run_ts(
        """
const anchors = Array.from({length: 5}, (_, i) => ({key: 'k' + i, anchorX: 500, anchorY: 200 + i}));
const placed = module.exports.stackLabels(anchors, {gap: 20, top: 0, bottom: 400});
console.log(JSON.stringify(placed.map((p) => ({y: p.y, displaced: p.displaced}))));
"""
    )
    ys = [row["y"] for row in result]
    assert ys == sorted(ys)
    assert all(b - a >= 20 for a, b in zip(ys, ys[1:])), "every pair clears the gap"
    assert result[0]["displaced"] is False, "the top label did not move"
    assert result[-1]["displaced"] is True


def test_an_overflowing_stack_is_pushed_back_inside_the_plot():
    """Both bounds, not just the bottom. Pushing an overflowing stack up can
    drive its head above the plot top, and a label drawn above the chart is not
    a smaller bug than one drawn below it."""
    result = _run_ts(
        """
const anchors = Array.from({length: 6}, (_, i) => ({key: 'k' + i, anchorX: 500, anchorY: 390 + i}));
const placed = module.exports.stackLabels(anchors, {gap: 20, top: 0, bottom: 400});
console.log(JSON.stringify(placed.map((p) => p.y)));
"""
    )
    assert min(result) >= 0
    assert max(result) <= 400
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_landing_board_frame.py -v
```

Expected: FAIL — `dashboard/landing/src/lib/boardFrame.ts` does not exist.

- [ ] **Step 3: Write the module**

Create `dashboard/landing/src/lib/boardFrame.ts`:

```ts
/** The nof1-derived board frame, for the Recharts side.
 *
 *  A MIRROR of the same constants and geometry in
 *  `dashboard/frontend/js/leaderboard.js`, which serves the two vanilla Chart.js
 *  surfaces. The duplication is forced -- different bundles, no shared module --
 *  and pinned by `dashboard/backend/tests/test_landing_board_frame.py`, which
 *  fails if a single number drifts. Change a value here and change it there in
 *  the same commit.
 */

export const BOARD_GUTTER_FRACTION = 0.4;
export const BOARD_GUTTER_MAX_FRACTION = 0.5;
export const BOARD_GUTTER_FONT = '600 11px Inter, system-ui, sans-serif';
export const BOARD_GUTTER_TEXT_INSET = 12;
export const BOARD_GUTTER_TRAILING_PAD = 16;
export const BOARD_TICK_CLEARANCE = 12;
export const BOARD_GUTTER_SLACK = 36;
export const BOARD_DOT_GAP = 4;
export const BOARD_NAME_GAP = 6;
export const BOARD_PILL_PAD_X = 5;
export const BOARD_PILL_HEIGHT = 15;
export const BOARD_DOT_RADIUS = 3;
export const BOARD_STUB_LENGTH = 7;
export const BOARD_LABEL_GAP_MAX = 20;
// An EXPRESSION, not a literal: a 15px pill needs at least 16px of pitch not to
// overlap its neighbour, so the minimum is derived from the pill height rather
// than restated beside it. Decoupling the two is how they drift apart.
// Ordering is load-bearing: BOARD_LABEL_GAP_MIN reads BOARD_PILL_HEIGHT, and a
// `const` read before its declaration is a TDZ ReferenceError at module load,
// not a lint warning. Keep the pill constants above the gap constants.
export const BOARD_LABEL_GAP_MIN = BOARD_PILL_HEIGHT + 1;
export const BOARD_LEADER_MIN_DISPLACEMENT = 7;
export const BOARD_ARROW_PAD = 18;
export const BOARD_ARROW_HEAD_LENGTH = 8;
export const BOARD_ARROW_HEAD_HALF = 4;
export const BOARD_XAXIS_ALLOWANCE = 34;
export const BOARD_AXIS_COLOR = 'rgba(148, 163, 184, 0.45)';

export type LabelText = { name: string; value: string };
export type Anchor = { key: string; anchorX: number; anchorY: number };
export type Placed = Anchor & { y: number; displaced: boolean };
export type Frame = { gutter: number; drawLabels: boolean; gap: number };

/** Text width in a given CSS font, measured on a module-level canvas.
 *
 *  MEASURED, not tabulated. `BoardPreview`'s `width={56}` y-axis reserve is the
 *  cautionary case: measured correctly at 11px, then the tick font moved to
 *  14px and four of five labels lost their leading `$` with nothing failing.
 *
 *  Returns a proportional estimate where there is no DOM (SSR, a node test), so
 *  callers never have to branch. */
let measureCtx: CanvasRenderingContext2D | null | undefined;
export function measureTextWidth(text: string, font: string): number {
  if (measureCtx === undefined) {
    measureCtx =
      typeof document === 'undefined'
        ? null
        : document.createElement('canvas').getContext('2d');
  }
  if (!measureCtx) return String(text).length * 6;
  measureCtx.font = font;
  return measureCtx.measureText(String(text)).width;
}

/** Width of the widest `dot name pill` block, plus the inset and trailing pad. */
export function labelBlockWidth(labels: LabelText[]): number {
  let widest = 0;
  for (const label of labels) {
    const block =
      BOARD_DOT_RADIUS * 2 +
      BOARD_DOT_GAP +
      measureTextWidth(label.name, BOARD_GUTTER_FONT) +
      BOARD_NAME_GAP +
      measureTextWidth(label.value, BOARD_GUTTER_FONT) +
      BOARD_PILL_PAD_X * 2;
    if (block > widest) widest = block;
  }
  return (
    BOARD_GUTTER_TEXT_INSET + widest + BOARD_TICK_CLEARANCE + BOARD_GUTTER_TRAILING_PAD
  );
}

/** How much right margin to reserve, whether to draw labels, and how far apart.
 *
 *  Same two degradations as the Chart.js copy, and for the same reason: too
 *  narrow for the widest label or too short to stack N of them, and the frame
 *  gives the space back and draws the arrow alone. Clipping is the failure this
 *  card has already shipped once -- `flex-nowrap` cut four of five chips at
 *  390px with no scrollbar, no ellipsis and nothing failing. */
export function frameLayout(input: {
  width: number;
  height: number;
  labels: LabelText[];
  fraction?: number;
}): Frame {
  const { width, height, labels } = input;
  const fraction = input.fraction ?? BOARD_GUTTER_FRACTION;
  const none: Frame = { gutter: BOARD_ARROW_PAD, drawLabels: false, gap: 0 };
  if (!labels.length || width <= 0 || height <= 0) return none;
  const gap = Math.min(BOARD_LABEL_GAP_MAX, (height - BOARD_XAXIS_ALLOWANCE) / labels.length);
  if (gap < BOARD_LABEL_GAP_MIN) return none;
  // The stack must also fit the canvas, not just the gap threshold.
  if ((labels.length - 1) * gap + BOARD_PILL_HEIGHT > height) return none;
  const floor = labelBlockWidth(labels);
  if (floor > width * BOARD_GUTTER_MAX_FRACTION) return none;
  // Cap the gutter at the measured floor plus a little slack rather than letting
  // it run to the full fraction: a wide card should not donate plot width it has
  // no labels to fill. `fraction` is the ceiling, `floor` the minimum.
  const room = Math.max(floor, Math.min(width * fraction, floor + BOARD_GUTTER_SLACK));
  return { gutter: room, drawLabels: true, gap };
}

/** Stagger coincident endpoints downward, then push the stack back inside.
 *
 *  Each label keeps its endpoint y as `anchorY` so `displaced` can say whether
 *  collision-avoidance actually moved it -- a leader line shorter than
 *  BOARD_LEADER_MIN_DISPLACEMENT connects nothing and just leaves a stub.
 *
 *  BOTH CLAMPS. Pushing an overflowing stack up can drive its head above the
 *  plot top, and a label drawn above the chart is not a smaller bug than one
 *  drawn below it. They cannot both bind at once: `frameLayout` only reports
 *  drawLabels when the stack fits in the plot height. */
export function stackLabels(
  anchors: Anchor[],
  opts: { gap: number; top: number; bottom: number },
): Placed[] {
  const placed: Placed[] = anchors
    .map((a) => ({ ...a, y: a.anchorY, displaced: false }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < placed.length; i += 1) {
    if (placed[i].y - placed[i - 1].y < opts.gap) {
      placed[i].y = placed[i - 1].y + opts.gap;
    }
  }
  if (placed.length) {
    const overflow = placed[placed.length - 1].y - opts.bottom;
    if (overflow > 0) placed.forEach((p) => { p.y -= overflow; });
    const underflow = opts.top - placed[0].y;
    if (underflow > 0) placed.forEach((p) => { p.y += underflow; });
  }
  placed.forEach((p) => {
    p.displaced = Math.abs(p.y - p.anchorY) > BOARD_LEADER_MIN_DISPLACEMENT;
  });
  return placed;
}

/** Dark or light pill ink, by the swatch's relative luminance. */
export function pillTextColor(hex: string): string {
  const h = String(hex || '').replace('#', '');
  const r = parseInt(h.slice(0, 2), 16) || 0;
  const g = parseInt(h.slice(2, 4), 16) || 0;
  const b = parseInt(h.slice(4, 6), 16) || 0;
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? '#0b1220' : '#f8fafc';
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_landing_board_frame.py -v
```

Expected: PASS, none skipped (`dashboard/landing/node_modules` is present locally). If any skip, run `cd dashboard/landing && npm ci` first — a skipped behavioural tier is the same as no tier.

- [ ] **Step 5: Commit**

```bash
git status --short
git add dashboard/landing/src/lib/boardFrame.ts dashboard/backend/tests/test_landing_board_frame.py
git commit -m "feat(landing): mirror the board-frame geometry for recharts"
```

---

### Task 6: Fetch and shape the live board

Transport and shaping, with no React in it, so the selection rule can be read against screen 0's without a component in the way.

**The selection rule is the whole point of "sync the two pages".** Screen 0 draws every model entry plus exactly two reference baselines, and its own source says why: seven model curves with no baseline leave the reader nothing to judge them against. The hero inherits that verbatim — 9 of the 12 entries the API returns.

**Colour is keyed on `entry_id`, never on a display label.** `LEADERBOARD_STYLES` on `/app` keys on the label ("Buy & Hold", "DJIA") for historical reasons, but the label is copy and can be renamed in `dashboard/config/leaderboard.json` with nothing failing; `id` is that file's primary key and reaches the client as `entry.entry_id`. Screen 0's `HOME_CHART_BASELINE_IDS` already made that correction; do not un-make it here.

**Files:**
- Create: `dashboard/landing/src/lib/leaderboard.ts`
- Test: `dashboard/backend/tests/test_landing_live_board.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, for Tasks 7–10:
  - `type LeaderboardEntry = {entry_id, team_name, team_badge, model, is_model, cumulative_return, portfolio_value, equity_curve: Array<{timestamp: string; equity: number}>}`
  - `type BoardSeries = {key: string; name: string; color: string; dash?: string; isBaseline: boolean; values: Array<number | null>}`
  - `type BoardStanding = {key: string; name: string; ret: string; color: string}`
  - `type BoardData = {times: string[]; series: BoardSeries[]; standings: BoardStanding[]; windowLabel: string}`
  - `fetchLeaderboard(signal: AbortSignal) -> Promise<BoardData>`
  - `selectBoardEntries(entries: LeaderboardEntry[]) -> LeaderboardEntry[]`
  - `buildBoardData(payload) -> BoardData`
  - `formatPercent(fraction: number, decimals: number) -> string`
  - `MODEL_COLOR_PALETTE: string[]`, `BASELINE_STYLES: Record<string, {color, dash}>`, `BOARD_BASELINE_IDS: string[]`

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_landing_live_board.py`:

```python
"""The landing hero draws the same live board the signed-in Home screen draws.

Source-shape guards. Nothing in CI builds or type-checks the landing, so these
read `landing/src` directly -- which is also the only layer that can compare the
landing's selection rule against screen 0's, since the two live in different
bundles and one of them ships minified.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LIB = _ROOT / "landing" / "src" / "lib"
_HOME_JS = (_ROOT / "frontend" / "home-page.js").read_text(encoding="utf-8")
_LEADERBOARD_JS = (_ROOT / "frontend" / "js" / "leaderboard.js").read_text(encoding="utf-8")
_LIB_TS = (_LIB / "leaderboard.ts").read_text(encoding="utf-8")


def _js_array(source: str, name: str) -> list[str]:
    match = re.search(rf"{name}\s*=\s*\[(.*?)\]", source, re.S)
    assert match, f"{name} not found"
    return re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))


def test_the_hero_and_screen_zero_pick_the_same_baselines():
    """Screen 0's own source states the reason: seven model curves with no
    baseline leave the reader nothing to judge them against. That is equally
    true on the acquisition page, and it is what "sync the two pages" means
    concretely rather than cosmetically."""
    assert _js_array(_HOME_JS, "HOME_CHART_BASELINE_IDS") == _js_array(
        _LIB_TS, "BOARD_BASELINE_IDS"
    )


def test_the_hero_uses_the_same_model_test_as_screen_zero():
    assert "is_model" in _LIB_TS and "team_badge" in _LIB_TS
    assert '"Model"' in _LIB_TS or "'Model'" in _LIB_TS


def test_baseline_colours_are_keyed_on_entry_id_not_on_a_display_label():
    """`LEADERBOARD_STYLES` on /app keys on the label for historical reasons, but
    the label is copy and can be renamed in dashboard/config/leaderboard.json
    with nothing failing. `id` is that file's primary key and reaches the client
    as `entry.entry_id`; screen 0's HOME_CHART_BASELINE_IDS already made this
    correction."""
    styles = re.search(r"BASELINE_STYLES[^=]*=\s*\{(.*?)\n\};", _LIB_TS, re.S)
    assert styles, "BASELINE_STYLES not found"
    body = styles.group(1)
    assert "buy_hold_djia" in body and "djia_index" in body
    assert "Buy & Hold" not in body and '"DJIA"' not in body


def test_the_model_palette_is_the_same_list_in_the_same_order():
    """The hero and /app must colour the same model the same way -- a visitor who
    signs up lands on a board whose curves they have already learned. The order
    matters as much as the members: /app assigns MODEL_COLOR_PALETTE[n] in
    first-seen order over the ranked payload, so the hero must index models in
    payload order too."""
    assert _js_array(_LEADERBOARD_JS, "MODEL_COLOR_PALETTE") == _js_array(
        _LIB_TS, "MODEL_COLOR_PALETTE"
    )


def test_the_fetch_is_root_relative_and_names_no_origin():
    """Vercel rewrites /api/:path* to Render (dashboard/frontend/vercel.json), and
    test_frontend_api_base.py requires an EMPTY production base for exactly that
    reason -- it calls a hardcoded Render origin a same-origin cookie auth
    regression. MarketTicker.tsx's apiBase() survives that guard only because it
    excludes minified assets/; do not copy it."""
    assert '"/api/v1/leaderboard' in _LIB_TS or "'/api/v1/leaderboard" in _LIB_TS
    assert "onrender.com" not in _LIB_TS
    assert "window.location.origin" not in _LIB_TS


def test_the_fetch_is_bounded_by_an_abort_signal():
    """Render's free tier cold-starts in 30-60s. A fetch with no ceiling leaves
    the card shimmering forever, which is the failure state this design most
    wants to be distinguishable."""
    assert "AbortSignal" in _LIB_TS or "signal" in _LIB_TS


def test_a_failed_request_throws_rather_than_returning_an_empty_board():
    """An empty board and a broken backend must not produce the same value. That
    is the fail-closed-is-not-fail-visible failure in miniature, and it is why
    the caller gets three states rather than two."""
    assert re.search(r"throw new Error", _LIB_TS), (
        "a non-ok response must raise, not resolve to an empty board"
    )
    assert "res.ok" in _LIB_TS or "response.ok" in _LIB_TS
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_landing_live_board.py -v
```

Expected: collection error — `dashboard/landing/src/lib/leaderboard.ts` does not exist.

- [ ] **Step 3: Write the module**

Create `dashboard/landing/src/lib/leaderboard.ts`:

```ts
/** The live Competition board, for the landing hero and the Race standings.
 *
 *  Same data the signed-in Home screen draws, selected by the same rule. See
 *  `dashboard/backend/tests/test_landing_live_board.py`, which pins the two
 *  selections against each other across the bundle boundary.
 */

export type EquityPoint = { timestamp: string; equity: number };

export type LeaderboardEntry = {
  entry_id: string;
  team_name: string;
  team_badge: string;
  model: string;
  is_model: boolean;
  cumulative_return: number;
  portfolio_value: number;
  initial_equity: number;
  equity_curve: EquityPoint[];
};

export type BoardSeries = {
  key: string;
  name: string;
  color: string;
  dash?: string;
  isBaseline: boolean;
  values: Array<number | null>;
};

export type BoardStanding = { key: string; name: string; ret: string; color: string };

export type BoardData = {
  times: string[];
  series: BoardSeries[];
  standings: BoardStanding[];
  windowLabel: string;
};

/** Entry ids the chart draws as passive reference curves.
 *
 *  Ids, not display labels: the label is copy and can be renamed in
 *  dashboard/config/leaderboard.json without anything failing, while `id` is
 *  that file's primary key and reaches the client as `entry.entry_id`.
 *
 *  Two, not five, and the same two screen 0 picks. The question the card exists
 *  to answer -- is +7.49% good? -- needs one strategy baseline and one index,
 *  not the whole baseline roster. */
export const BOARD_BASELINE_IDS = ['buy_hold_djia', 'djia_index'];

/** Mirrors `MODEL_COLOR_PALETTE` in dashboard/frontend/js/leaderboard.js, in
 *  order. A visitor who signs up lands on a board whose curves they have
 *  already learned here, so the same model must be the same colour on both. */
export const MODEL_COLOR_PALETTE = [
  '#FBBF24', '#FB923C', '#F472B6', '#A78BFA', '#34D399',
  '#22D3EE', '#F87171', '#A3E635', '#E879F9', '#60A5FA',
];

/** Mirrors the relevant rows of `LEADERBOARD_STYLES`, rekeyed onto entry ids. */
export const BASELINE_STYLES: Record<string, { color: string; dash: string }> = {
  buy_hold_djia: { color: '#38BDF8', dash: '10 6' },
  djia_index: { color: '#94A3B8', dash: '8 4 2 4' },
};

export function formatPercent(fraction: number, decimals: number): string {
  if (!Number.isFinite(fraction)) return '—';
  return `${fraction > 0 ? '+' : ''}${(fraction * 100).toFixed(decimals)}%`;
}

/** Every model, plus the two reference baselines -- 9 of the 12 entries the API
 *  returns. Order is preserved from the payload, which arrives ranked, because
 *  the model palette is assigned by position. */
export function selectBoardEntries(entries: LeaderboardEntry[]): LeaderboardEntry[] {
  const all = entries || [];
  const models = all.filter((e) => e && (e.is_model || e.team_badge === 'Model'));
  const baselines = all.filter(
    (e) => e && !e.is_model && BOARD_BASELINE_IDS.indexOf(e.entry_id) !== -1,
  );
  return models.concat(baselines);
}

/** `2026-04-15T14:00:00+00:00` → `2026-04-15T14:00`. Same normalisation
 *  js/leaderboard.js's `chartTimeKey` performs, so both surfaces bucket the
 *  same hourly stamps onto the same axis. */
function timeKey(ts: string): string {
  const s = String(ts || '');
  if (!s) return '';
  if (s.length >= 16 && s[10] === 'T') return s.slice(0, 16);
  if (s.length >= 10) return s.slice(0, 10);
  return s;
}

/** Fractions, not dollars, and not for scale safety -- because of what the
 *  labels MEAN. Every dollar level in this payload is a x0.1 rescale of a
 *  $100,000 backtest onto the config's $10,000 display base (leaderboard
 *  service.py), so a `$10,749` tick names an account that never existed, while
 *  the percent is exactly what ran. The old hero was allowed a dollar axis only
 *  because its curves were fabricated with a clean base of 1000; live data
 *  removes that premise. */
export function buildBoardData(payload: {
  entries?: LeaderboardEntry[];
  window?: { label?: string };
}): BoardData {
  const selected = selectBoardEntries(payload.entries || []);
  const timeSet = new Set<string>();
  const perEntry = selected.map((entry) => {
    const byTime: Record<string, number> = {};
    (entry.equity_curve || []).forEach((pt) => {
      const key = timeKey(pt.timestamp);
      if (!key) return;
      byTime[key] = Number(pt.equity) || 0;
      timeSet.add(key);
    });
    return { entry, byTime };
  });
  const times = Array.from(timeSet).sort();

  let modelIndex = 0;
  const series: BoardSeries[] = [];
  const standings: BoardStanding[] = [];
  perEntry.forEach(({ entry, byTime }) => {
    const isModel = !!(entry.is_model || entry.team_badge === 'Model');
    const style = isModel
      ? { color: MODEL_COLOR_PALETTE[modelIndex++ % MODEL_COLOR_PALETTE.length], dash: undefined }
      : BASELINE_STYLES[entry.entry_id] || { color: '#94A3B8', dash: '10 6' };
    const raw = times.map((t) => (t in byTime ? byTime[t] : null));
    const base = Number(entry.initial_equity) || raw.find((v) => v != null) || 10000;
    const values = raw.map((v) => (v == null ? null : (v - base) / base));
    if (!values.some((v) => v != null)) return;
    const name = entry.model || entry.team_name;
    series.push({
      key: entry.entry_id,
      name,
      color: style.color,
      dash: style.dash,
      isBaseline: !isModel,
      values,
    });
    standings.push({
      key: entry.entry_id,
      name,
      // Two decimals, matching /app's rank rows and this card's own tooltip.
      ret: formatPercent(Number(entry.cumulative_return), 2),
      color: style.color,
    });
  });

  standings.sort(
    (a, b) => parseFloat(b.ret) - parseFloat(a.ret),
  );
  return { times, series, standings, windowLabel: payload.window?.label || '' };
}

/** Root-relative, with no origin anywhere in it.
 *
 *  dashboard/frontend/vercel.json rewrites /api/:path* to Render, and
 *  test_frontend_api_base.py requires an EMPTY production base for exactly that
 *  reason -- it calls a hardcoded Render origin a same-origin cookie auth
 *  regression. MarketTicker.tsx's apiBase() survives that guard only because it
 *  excludes minified assets/. This path is correct under Vercel and under local
 *  uvicorn alike. (Under `npm run dev` at :5173 it hits the Vite server and
 *  fails -- but so does apiBase(), which returns the dev server's own origin
 *  there. Neither pattern serves the dev server.) */
export async function fetchLeaderboard(signal: AbortSignal): Promise<BoardData> {
  const res = await fetch('/api/v1/leaderboard?period=contest', { signal });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return buildBoardData(await res.json());
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_landing_live_board.py -v
cd dashboard/landing && npm run typecheck && cd ../..
```

Expected: pytest PASS, `tsc --noEmit` clean.

- [ ] **Step 5: Commit**

```bash
git status --short
git add dashboard/landing/src/lib/leaderboard.ts dashboard/backend/tests/test_landing_live_board.py
git commit -m "feat(landing): fetch and shape the live competition board"
```

---

### Task 7: One fetch, shared by the hero and the standings

The hero and `Race` are four screens apart but render the same board. Two fetches would double the load on a free-tier backend that cold-starts in 30–60s and, worse, could disagree — a hero showing real numbers above a table showing different ones is worse than either alone.

**Files:**
- Create: `dashboard/landing/src/lib/useLeaderboard.tsx`
- Modify: `dashboard/landing/src/pages/landing-page.tsx`
- Test: `dashboard/backend/tests/test_landing_live_board.py` (extend)

**Interfaces:**
- Consumes: `fetchLeaderboard`, `BoardData` from Task 6.
- Produces, for Tasks 9–10:
  - `type BoardState = {status: 'loading'; } | {status: 'ready'; data: BoardData} | {status: 'error'; message: string}`
  - `LeaderboardProvider({children}: {children: ReactNode}) -> JSX.Element`
  - `useLeaderboard() -> BoardState`

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_landing_live_board.py`:

```python
_HOOK_TS = None


def _hook() -> str:
    global _HOOK_TS
    if _HOOK_TS is None:
        _HOOK_TS = (_LIB / "useLeaderboard.tsx").read_text(encoding="utf-8")
    return _HOOK_TS


def test_the_board_is_fetched_once_for_the_whole_page():
    """The hero and the Race standings are four screens apart and render the same
    board. Two fetches double the load on a backend that cold-starts in 30-60s
    and, worse, can disagree -- real numbers in the hero above different ones in
    the table is worse than either alone."""
    assert "createContext" in _hook()
    assert "LeaderboardProvider" in _hook()
    page = (_ROOT / "landing" / "src" / "pages" / "landing-page.tsx").read_text(
        encoding="utf-8"
    )
    assert "<LeaderboardProvider>" in page
    hero_at = page.index("<Hero />")
    race_at = page.index("<Race />")
    provider_at = page.index("<LeaderboardProvider>")
    assert provider_at < hero_at < race_at, "both consumers must sit inside it"


def test_the_three_states_are_distinguishable_in_the_type():
    """Loading, ready and failed are three states, not two plus a fallback. A
    silent fallback to sample curves would make "the backend is down" and "the
    backend is fine" render near-identically -- the exact failure shape
    CLAUDE.md's fail-closed-is-not-fail-visible section is about."""
    src = _hook()
    for status in ('"loading"', '"ready"', '"error"'):
        assert status in src, f"{status} is not one of the states"
    assert "SAMPLE_" not in src, "no fallback to invented curves, ever"


def test_a_failed_fetch_carries_a_message_rather_than_a_bare_flag():
    """The failed card names the failure. "Something went wrong" with no cause is
    the dead end this landing's auth modal already had to be corrected for."""
    assert re.search(r"message:\s*", _hook())


def test_the_fetch_is_cancelled_on_unmount():
    assert "AbortController" in _hook()
    assert ".abort()" in _hook()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_landing_live_board.py -v -k "fetched_once or three_states or carries_a_message or cancelled"
```

Expected: FAIL — `useLeaderboard.tsx` does not exist.

- [ ] **Step 3: Write the provider**

Create `dashboard/landing/src/lib/useLeaderboard.tsx`:

```tsx
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { fetchLeaderboard, type BoardData } from "@/lib/leaderboard";

/** THREE states, and they must be distinguishable on screen.
 *
 *  Not two plus a fallback. A silent fallback to sample curves would make "the
 *  backend is down" and "the backend is fine" render near-identically -- the
 *  failure shape CLAUDE.md's fail-closed-is-not-fail-visible section is about,
 *  and the same one that degraded the news panel in prod for hours with a green
 *  test suite. Render's free tier cold-starts in 30-60s, so `loading` is a
 *  routine first-visit occurrence rather than an edge case. */
export type BoardState =
  | { status: "loading" }
  | { status: "ready"; data: BoardData }
  | { status: "error"; message: string };

const LeaderboardContext = createContext<BoardState>({ status: "loading" });

/** One fetch for the page. The hero and the Race standings are four screens
 *  apart and render the same board; fetching twice doubles the load on a
 *  cold-starting free-tier backend and lets the two disagree. */
export function LeaderboardProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<BoardState>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    // Generous, because a free-tier cold start is 30-60s and giving up at 10
    // would report a failure to every first visitor of the day. Same ceiling
    // MarketTicker already uses.
    const timeout = setTimeout(() => controller.abort(), 45_000);
    fetchLeaderboard(controller.signal)
      .then((data) => setState({ status: "ready", data }))
      .catch((err: unknown) => {
        if (controller.signal.aborted && !(err instanceof Error && err.name !== "AbortError")) {
          setState({ status: "error", message: "Timed out waiting for the board." });
          return;
        }
        setState({
          status: "error",
          message: err instanceof Error ? err.message : "Unknown error",
        });
      })
      .finally(() => clearTimeout(timeout));
    return () => {
      clearTimeout(timeout);
      controller.abort();
    };
  }, []);

  return (
    <LeaderboardContext.Provider value={state}>{children}</LeaderboardContext.Provider>
  );
}

export function useLeaderboard(): BoardState {
  return useContext(LeaderboardContext);
}
```

- [ ] **Step 4: Wrap the page**

In `dashboard/landing/src/pages/landing-page.tsx`, add the import and wrap `<main>` — both consumers must sit inside the provider:

```tsx
import { LeaderboardProvider } from "@/lib/useLeaderboard";
```

```tsx
      <LeaderboardProvider>
        <main>
          <Hero />
          <WhyCare />
          <Talk />
          <Test />
          <Race />
        </main>
      </LeaderboardProvider>
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_landing_live_board.py -v
cd dashboard/landing && npm run typecheck && cd ../..
```

Expected: PASS and clean.

- [ ] **Step 6: Commit**

```bash
git status --short
git add dashboard/landing/src/lib/useLeaderboard.tsx dashboard/landing/src/pages/landing-page.tsx dashboard/backend/tests/test_landing_live_board.py
git commit -m "feat(landing): one shared leaderboard fetch for hero and race"
```

---

### Task 8: The endpoint rail

The Recharts analogue of PR A's two plugins: endpoint dots, stubs, staggered leaders, name + value pill, and the axis arrow — one `Customized` overlay.

**What it depends on, and how it fails.** `Customized` is cloned with `{...chartProps, ...chartState}` (`recharts/es6/chart/generateCategoricalChart.js`, `renderCustomized`), so the rail receives `formattedGraphicalItems` and `offset` — internal shape, stable across 2.x but not contractual. When that shape is not what it expects the rail renders **nothing**, and the chip strip below the chart continues to key every curve exactly as it does today. That is a real degradation with a real fallback, not a silent one: the card stays complete and legible, it just loses the gutter labels.

Two hazards worth stating in the file: the clone spreads chart props **over** the element's own, so any extra prop passed to `<Customized>` must not collide with a chart prop or state key (`valueByKey` does not); and `formattedGraphicalItems` is in `<Line>` declaration order, not visual order, so nothing may assume it is sorted.

**Files:**
- Create: `dashboard/landing/src/components/home/EndpointRail.tsx`
- Test: `dashboard/backend/tests/test_landing_live_board.py` (extend)

**Interfaces:**
- Consumes: `frameLayout`, `stackLabels`, `measureTextWidth`, `pillTextColor` and the `BOARD_*` constants from Task 5.
- Produces, for Task 9: `EndpointRail(props)` — pass as `<Customized component={EndpointRail} valueByKey={…} drawLabels={…} />`, where `valueByKey: Record<string, string>` maps a series key to its pill text.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_landing_live_board.py`:

```python
_RAIL = None


def _rail() -> str:
    global _RAIL
    if _RAIL is None:
        _RAIL = (
            _ROOT / "landing" / "src" / "components" / "home" / "EndpointRail.tsx"
        ).read_text(encoding="utf-8")
    return _RAIL


def test_the_rail_degrades_to_nothing_when_recharts_internals_change():
    """`Customized` is cloned with the chart's props and state, which is internal
    shape rather than contract. When it is not what the rail expects, the rail
    renders nothing and the chip strip below keeps keying every curve -- a real
    fallback, not a silent one."""
    src = _rail()
    assert "Array.isArray(formattedGraphicalItems)" in src
    assert "return null" in src


def test_the_rail_draws_the_frame_and_not_a_second_geometry():
    """Every number comes from boardFrame.ts, which is pinned against
    js/leaderboard.js. A literal here would be a third copy nothing guards."""
    src = _rail()
    assert "from \"@/lib/boardFrame\"" in src or "from '@/lib/boardFrame'" in src
    assert "stackLabels(" in src
    assert "BOARD_DOT_RADIUS" in src and "BOARD_STUB_LENGTH" in src
    assert "BOARD_ARROW_HEAD_LENGTH" in src
    assert "BOARD_LABEL_GAP_MAX" in src, "even the fallback gap is the frame's"


def test_the_rail_never_sorts_by_declaration_order():
    """`formattedGraphicalItems` arrives in <Line> declaration order, not visual
    order. `stackLabels` sorts by y itself; anything that assumed the incoming
    order was meaningful would stagger the wrong labels."""
    src = _rail()
    assert "formattedGraphicalItems" in src
    assert ".sort(" not in src, "sorting is stackLabels' job and it does it by y"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_landing_live_board.py -v -k "rail"
```

Expected: FAIL — `EndpointRail.tsx` does not exist.

- [ ] **Step 3: Write the component**

Create `dashboard/landing/src/components/home/EndpointRail.tsx`:

```tsx
import {
  BOARD_ARROW_HEAD_HALF,
  BOARD_ARROW_HEAD_LENGTH,
  BOARD_AXIS_COLOR,
  BOARD_DOT_RADIUS,
  BOARD_GUTTER_FONT,
  BOARD_GUTTER_TEXT_INSET,
  BOARD_LABEL_GAP_MAX,
  BOARD_PILL_HEIGHT,
  BOARD_PILL_PAD_X,
  BOARD_STUB_LENGTH,
  measureTextWidth,
  pillTextColor,
  stackLabels,
  type Anchor,
} from "@/lib/boardFrame";

/** The reserved gutter's contents: an endpoint dot and dotted stub per curve,
 *  a staggered `dot name pill` label, a leader line to any label collision
 *  avoidance had to move, and the x-axis arrowhead.
 *
 *  The Recharts twin of `createEndpointLabelPlugin` + `createAxisArrowPlugin`
 *  in dashboard/frontend/js/leaderboard.js. Every number comes from
 *  `@/lib/boardFrame`, which is pinned against that file by
 *  dashboard/backend/tests/test_landing_board_frame.py -- a literal here would
 *  be a third copy nothing guards.
 *
 *  INTERNAL SHAPE, WITH A REAL FALLBACK. Recharts clones a `<Customized>` child
 *  with `{...chartProps, ...chartState}` (es6/chart/generateCategoricalChart.js,
 *  `renderCustomized`), which is how `formattedGraphicalItems` and `offset`
 *  arrive. That is stable across 2.x but is not a contract, so when it is not
 *  what this expects the rail renders NOTHING and the chip strip below the chart
 *  keeps keying every curve exactly as it does today. The card stays complete;
 *  it loses the gutter labels.
 *
 *  Two hazards the clone creates. Chart props are spread OVER the element's own,
 *  so an extra prop passed to `<Customized>` must not collide with a chart prop
 *  or state key -- `valueByKey` and `drawLabels` do not. And
 *  `formattedGraphicalItems` arrives in `<Line>` DECLARATION order, never visual
 *  order; `stackLabels` sorts by y itself, and nothing here may assume
 *  otherwise. */
type RailProps = {
  formattedGraphicalItems?: Array<{
    props?: { points?: Array<{ x: number; y: number }> };
    item?: { props?: { dataKey?: string; name?: string; stroke?: string } };
  }>;
  offset?: { top: number; left: number; width: number; height: number };
  width?: number;
  valueByKey?: Record<string, string>;
  drawLabels?: boolean;
  gap?: number;
};

type Row = Anchor & { name: string; value: string; color: string };

export function EndpointRail(props: RailProps) {
  const { formattedGraphicalItems, offset, width, valueByKey, drawLabels, gap } = props;
  if (!Array.isArray(formattedGraphicalItems) || !offset || !width) return null;

  const axisY = offset.top + offset.height;
  const tipX = width - 4;

  const arrow =
    tipX > offset.left + BOARD_ARROW_HEAD_LENGTH ? (
      <g key="arrow">
        <line
          x1={offset.left}
          y1={axisY}
          x2={tipX - BOARD_ARROW_HEAD_LENGTH}
          y2={axisY}
          stroke={BOARD_AXIS_COLOR}
          strokeWidth={1}
        />
        <polygon
          points={[
            `${tipX},${axisY}`,
            `${tipX - BOARD_ARROW_HEAD_LENGTH},${axisY - BOARD_ARROW_HEAD_HALF}`,
            `${tipX - BOARD_ARROW_HEAD_LENGTH},${axisY + BOARD_ARROW_HEAD_HALF}`,
          ].join(" ")}
          fill={BOARD_AXIS_COLOR}
        />
      </g>
    ) : null;

  const rows: Row[] = [];
  for (const entry of formattedGraphicalItems) {
    const points = entry?.props?.points;
    const item = entry?.item?.props;
    if (!Array.isArray(points) || !points.length || !item?.dataKey) continue;
    let last: { x: number; y: number } | undefined;
    for (let i = points.length - 1; i >= 0; i -= 1) {
      if (points[i] && Number.isFinite(points[i].y)) { last = points[i]; break; }
    }
    if (!last) continue;
    const key = String(item.dataKey);
    rows.push({
      key,
      anchorX: last.x,
      anchorY: last.y,
      name: item.name || key,
      value: valueByKey?.[key] ?? "",
      color: item.stroke || "#94a3b8",
    });
  }

  if (!drawLabels || !rows.length) return <>{arrow}</>;

  // `gap` comes from the card's single `frameLayout` call, never from a second
  // computation here: the gutter width that was RESERVED came out of that call,
  // and a gap derived independently is a second chance to disagree with it.
  const byKey = new Map(rows.map((row) => [row.key, row]));
  const placed = stackLabels(rows, {
    gap: gap ?? BOARD_LABEL_GAP_MAX,
    top: offset.top,
    bottom: offset.top + offset.height,
  });

  const gutterStart = offset.left + offset.width + 6;
  const labelX = offset.left + offset.width + BOARD_GUTTER_TEXT_INSET;

  return (
    <>
      {arrow}
      {placed.map((p) => {
        const row = byKey.get(p.key);
        if (!row) return null;
        const nameWidth = measureTextWidth(row.name, BOARD_GUTTER_FONT);
        const valueWidth = measureTextWidth(row.value, BOARD_GUTTER_FONT);
        const pillX = labelX + BOARD_DOT_RADIUS * 2 + 4 + nameWidth + 6;
        const pillWidth = valueWidth + BOARD_PILL_PAD_X * 2;
        return (
          <g key={row.key}>
            {/* The note's `•⋯`: the curve continues, and the stub asserts no
                value for where it goes. */}
            <circle cx={row.anchorX} cy={row.anchorY} r={BOARD_DOT_RADIUS} fill={row.color} />
            <line
              x1={row.anchorX + BOARD_DOT_RADIUS + 1}
              y1={row.anchorY}
              x2={row.anchorX + BOARD_DOT_RADIUS + 1 + BOARD_STUB_LENGTH}
              y2={row.anchorY}
              stroke={row.color}
              strokeWidth={1.5}
              strokeDasharray="1 3"
              opacity={0.6}
            />
            {p.displaced ? (
              <line
                x1={gutterStart}
                y1={row.anchorY}
                x2={labelX - 3}
                y2={p.y}
                stroke={row.color}
                strokeWidth={1}
                strokeDasharray="1 3"
                opacity={0.35}
              />
            ) : null}
            <circle cx={labelX + BOARD_DOT_RADIUS} cy={p.y} r={BOARD_DOT_RADIUS} fill={row.color} />
            <text
              x={labelX + BOARD_DOT_RADIUS * 2 + 4}
              y={p.y}
              fill={row.color}
              fontSize={11}
              fontWeight={600}
              dominantBaseline="middle"
            >
              {row.name}
            </text>
            <rect
              x={pillX}
              y={p.y - BOARD_PILL_HEIGHT / 2}
              width={pillWidth}
              height={BOARD_PILL_HEIGHT}
              rx={4}
              fill={row.color}
            />
            <text
              x={pillX + BOARD_PILL_PAD_X}
              y={p.y}
              fill={pillTextColor(row.color)}
              fontSize={11}
              fontWeight={600}
              dominantBaseline="middle"
            >
              {row.value}
            </text>
          </g>
        );
      })}
    </>
  );
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_landing_live_board.py -v -k "rail"
cd dashboard/landing && npm run typecheck && cd ../..
```

Expected: PASS and clean.

- [ ] **Step 5: Commit**

```bash
git status --short
git add dashboard/landing/src/components/home/EndpointRail.tsx dashboard/backend/tests/test_landing_live_board.py
git commit -m "feat(landing): endpoint rail and forward arrow for the hero chart"
```

---

### Task 9: The hero card on live data

The largest single change in the plan. `BoardPreview.tsx` loses both sample constants, gains three states, switches to a percent axis, reserves the gutter, mounts the rail, and drives its chip strip from the payload.

**The units change is forced, not chosen.** `test_the_two_surfaces_agree_on_the_numbers_that_must_agree` documents that `/` was *allowed* a dollar axis only because its curves were fabricated with a clean base of 1000, so `$1210` was unambiguous. Live entries' dollar levels are a ×0.1 rescale of a $100,000 backtest onto the config's $10,000 display base, so `$10,749` names an account that never existed. Switching the data removes the premise, which takes `domain={[960, 1240]}` and the `width={56}` reserve with it.

**"Illustrative example" comes off this card.** The data is no longer illustrative, and leaving the label on real numbers is its own false claim. It is replaced by the window the chart actually draws, taken from the payload.

**Files:**
- Modify: `dashboard/landing/src/components/home/BoardPreview.tsx` (full rewrite)
- Test: `dashboard/backend/tests/test_landing_chart_first.py` (update the four affected cases)

**Interfaces:**
- Consumes: `useLeaderboard()` (Task 7), `frameLayout`/`measureTextWidth` (Task 5), `formatPercent` (Task 6), `EndpointRail` (Task 8).
- Produces, for Task 10: nothing exported. `SAMPLE_CURVES`, `SAMPLE_STANDINGS` and `LINE_COLORS` are **deleted**; `Race.tsx` is the only importer of any of them, and **Task 9 rewrites that one import line itself** (ruled during execution) so that no intermediate commit on the branch is left failing `tsc`. Task 9 changes nothing else in `Race.tsx`; Task 10 still rewires its body to the hook. Passing `gap={frame.gap}` to `<Customized>` is what feeds `EndpointRail`'s `gap` prop from Task 8 — the rail never computes it.

- [ ] **Step 1: Update the failing guards**

In `dashboard/backend/tests/test_landing_chart_first.py`:

Replace `test_the_standings_table_becomes_a_chip_strip_that_can_show_every_chip` with:

```python
def test_the_standings_table_becomes_a_chip_strip_that_can_show_every_chip():
    """Demotion, not deletion: the chart ships no <Legend>, so the chips are the
    only thing linking a curve colour to a model name -- and they are now also
    the fallback when the endpoint rail declines to draw (a narrow card, a
    Recharts internal that moved). The full table lives in Race.tsx.

    THE STRIP MUST WRAP, and the pressure just went up: it went from five
    hardcoded entries to nine from the payload. `flex-nowrap` with
    `overflow-hidden` cut entries off the end wherever the strip was narrower
    than its content -- measured scrollWidth 910 against clientWidth 285 at 390
    (one chip survives, keying five drawn curves), 663 at 768, 895 at 1024, so
    the whole lg band and every phone, silently, because the only live-browser
    guard on it ran at 1440.
    """
    board = _BOARD
    assert "grid-cols-12" not in board, "the 5-row table is what the chart needs the height of"
    assert "flex-wrap" in board and "flex-nowrap" not in board, (
        "a legend that cannot show its entries is not a legend"
    )
    strip = board[board.index("standings.map") - 400 : board.index("standings.map")]
    assert "overflow-hidden" not in strip, (
        "clipping the strip is the same failure by another route -- no scrollbar, "
        "no ellipsis, and nothing fails"
    )
    assert "text-base" in board, "text-sm rows were one of the three reported problems"
    # The identity link. `swatch` is gone with the sample rows; the colour now
    # comes off the same BoardSeries the curve is drawn from, which is stronger:
    # a row and its curve cannot disagree because there is one value.
    assert "item.color" in board
    assert "dataKey=" in board


def test_the_hero_draws_the_board_the_signed_in_home_draws():
    """The whole point of the change. No component may reintroduce a curve that
    is not on the board, and the only way to be sure of that is for the data to
    come from the API rather than from a literal."""
    assert "useLeaderboard" in _BOARD
    assert "SAMPLE_CURVES" not in _BOARD and "SAMPLE_STANDINGS" not in _BOARD


def test_the_hero_reports_a_failed_load_instead_of_shimmering_forever():
    """Three states, and they must be distinguishable. A permanent skeleton and
    a silent fallback are the same defect: "the backend is down" and "the backend
    is fine" would render near-identically."""
    board = _collapse(_BOARD)
    assert 'status === "error"' in board or "status === 'error'" in board
    assert 'status === "loading"' in board or "status === 'loading'" in board
    assert "state.message" in board or "board.message" in board, (
        "the failed card must name the failure, not print a dead end"
    )
```

Replace `test_landing_chart_axis_ticks_are_14px` with:

```python
def test_landing_chart_axis_ticks_are_14px():
    assert _BOARD.count("fontSize={14}") == 2, "both XAxis and YAxis"
    assert "fontSize={11}" not in _BOARD, (
        "11px belongs to the gutter labels, which live in EndpointRail.tsx"
    )


def test_the_y_axis_reserve_is_measured_rather_than_guessed():
    """`width={56}` was measured against `$1030` at 11px; the tick font later
    moved to 14px and four of five labels lost their leading `$` with nothing
    failing. The axis is percent now, so the number would have to be re-measured
    anyway -- measuring it at render removes the whole class."""
    assert "width={56}" not in _BOARD
    assert "measureTextWidth" in _BOARD
    assert "domain={[960, 1240]}" not in _BOARD, "a hardcoded dollar domain"
```

Replace the units block at the end of `test_the_two_surfaces_agree_on_the_numbers_that_must_agree` (everything from the `# UNITS:` comment to the end of the function) with:

```python
    # UNITS: percent on BOTH, and this is the assertion that inverted.
    #
    # It used to pin an ASYMMETRY -- /app percent, / dollars -- and the
    # justification was precise: / plotted fabricated curves that all shared a
    # base of 1000, so `$1210` was unambiguous and read as SAMPLE_STANDINGS'
    # +21.0%. That premise is gone. / now plots the same LIVE entries screen 0
    # does, and every dollar level in that payload is a x0.1 rescale of a
    # $100,000 backtest onto the config's $10,000 display base (leaderboard
    # service.py), so a `$10,749` tick names an account that never existed while
    # the percent is what actually ran.
    #
    # NOT the reason, though an earlier draft of the chart-first plan said so:
    # issue #365 does NOT make a dollar axis draw a 10x break here.
    # get_leaderboard normalises every entry to one display base before serving
    # -- measured against a hand-built mixed-capital database -- so on this
    # payload dollars and percent are an affine transform. Do not re-derive the
    # scale argument and then "discover" it is false; the label argument above
    # is the one that holds.
    assert "(v * 100).toFixed(1)}%" in home_js
    assert "toFixed(1)" in _BOARD, "the landing axis is percent to one decimal too"
    assert not re.search(r"tickFormatter=\{\(v\) => `\$", _BOARD), (
        "a dollar tick on this card names an account that never existed"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_landing_chart_first.py -v
```

Expected: the five updated/new cases FAIL against the current `BoardPreview.tsx`.

- [ ] **Step 3: Rewrite the card**

Replace the entire contents of `dashboard/landing/src/components/home/BoardPreview.tsx` with:

```tsx
import { LineChart as LineChartIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Customized,
} from "recharts";
import { useLeaderboard } from "@/lib/useLeaderboard";
import { formatPercent, type BoardSeries } from "@/lib/leaderboard";
import { frameLayout, measureTextWidth } from "@/lib/boardFrame";
import { EndpointRail } from "./EndpointRail";

/** Matches `fontSize={14}` on both axes below. The y-axis reserve is measured
 *  in it rather than guessed: `width={56}` was measured correctly against
 *  `$1030` at 11px, the tick font later moved to 14px, and four of five labels
 *  lost their leading `$` with nothing failing. */
const AXIS_TICK_FONT = "14px Inter, system-ui, sans-serif";

/** One decimal on the axis, two in the tooltip and the pills.
 *
 *  Same split screen 0 makes, for the same reason: an axis tick is a scale
 *  marker with no neighbour to match, and over a domain under eight percentage
 *  points zero decimals renders duplicate labels while two renders noise. The
 *  tooltip and the chips sit beside each other and must agree, so both are two.
 */
function axisTick(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

/** Rows Recharts can plot: one object per timestamp, one column per curve. */
function toRows(times: string[], series: BoardSeries[]) {
  return times.map((t, i) => {
    const row: Record<string, string | number | null> = { t };
    series.forEach((s) => { row[s.key] = s.values[i]; });
    return row;
  });
}

/** The plotted range, padded.
 *
 *  Derived, because the hardcoded `[960, 1240]` it replaces was a dollar domain
 *  for fabricated curves. The real board spans about -0.43% to +7.49%, which is
 *  visually flat next to nof1's -34%..+34% -- and that is the honest picture.
 *  Do not widen the padding to manufacture a fan-out that did not happen. */
function percentDomain(series: BoardSeries[]): [number, number] {
  const values: number[] = [];
  series.forEach((s) => s.values.forEach((v) => { if (v != null) values.push(v); }));
  if (!values.length) return [-0.05, 0.05];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = Math.max((hi - lo) * 0.12, 0.005);
  return [lo - pad, hi + pad];
}

/**
 * The hero's right-hand card. Deliberately compact: it exists so the board is
 * on screen before any scroll, not to replace the full standings under
 * `#race`. Chart first, then the standings — a visitor should see the shape
 * before they read a single number.
 *
 * The curves are the LIVE Competition board, the same one the signed-in Home
 * screen draws and selected by the same rule: every model entry plus exactly two
 * reference baselines. Seven model curves with nothing to judge them against is
 * the failure that rule exists to prevent, and it is no less true here than on
 * screen 0.
 */
export function BoardPreview() {
  const board = useLeaderboard();
  const chartRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  // The gutter is a FRACTION of the rendered width, so the width has to be
  // observed. Recharts' own <ResponsiveContainer> knows it but does not hand it
  // to the parent, and `margin` is a prop on <LineChart>, which is the parent's
  // to set.
  useEffect(() => {
    const el = chartRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) setSize({ width: rect.width, height: rect.height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const data = board.status === "ready" ? board.data : null;
  const series = data?.series ?? [];
  const standings = data?.standings ?? [];

  const frame = useMemo(
    () =>
      frameLayout({
        width: size.width,
        height: size.height,
        labels: standings.map((s) => ({ name: s.name, value: s.ret })),
      }),
    [size.width, size.height, standings],
  );

  const domain = useMemo(() => percentDomain(series), [series]);
  const yAxisWidth = useMemo(() => {
    const widest = Math.max(
      measureTextWidth(axisTick(domain[0]), AXIS_TICK_FONT),
      measureTextWidth(axisTick(domain[1]), AXIS_TICK_FONT),
    );
    return Math.ceil(widest) + 12;
  }, [domain]);

  const rows = useMemo(() => toRows(data?.times ?? [], series), [data, series]);
  const valueByKey = useMemo(
    () => Object.fromEntries(standings.map((s) => [s.key, s.ret])),
    [standings],
  );

  return (
    <div className="bg-card border border-card-border rounded-xl shadow-2xl overflow-hidden flex flex-col">
      <div className="px-5 pt-5 pb-4 border-b border-border">
        <div className="flex items-start justify-between gap-3 mb-2">
          <h2 className="text-xl font-bold flex items-center gap-2 min-w-0">
            <LineChartIcon className="w-5 h-5 text-primary shrink-0" aria-hidden="true" />
            Where the AI models stand
          </h2>
          {/* Was "Illustrative example". The data is no longer illustrative, and
              that label on real numbers is its own false claim. What replaces it
              is the window the chart actually draws, off the payload -- so the
              chip is now a provenance statement rather than a disclaimer, and it
              is what keeps the forward arrow below from reading as a claim that
              this window is still running. */}
          <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded shrink-0">
            {data?.windowLabel ? `Competition window · ${data.windowLabel}` : "Competition window"}
          </span>
        </div>
        {/* One line at this card width, and that is load-bearing: the chart's
            clamp subtracts this bar's height. Two lines here invalidates the
            reserves below and the card goes half-visible without anything
            failing. */}
        <p className="text-sm text-foreground/65 leading-relaxed">
          Each line is one AI model&apos;s return. Dashed lines are buy-and-hold and the index.
        </p>
      </div>

      {/* The formula stays an inline style — its commas and parentheses get
          mangled by Tailwind's arbitrary-VALUE parser — while the one number
          that has to change per breakpoint rides an arbitrary PROPERTY, which
          does take a responsive prefix.

          TWO RESERVES, BOTH MEASURED, because the card's non-chart height is
          not one number: beside the copy at >=lg it is one thing, stacked at
          390px wide the title, the chip and the caption all wrap AND the chip
          strip runs to several rows. One constant cannot serve both, and the
          desktop one applied to a phone put the card 77px past the fold.

          RE-DERIVED for live data: the strip went from five entries to nine and
          the "Illustrative example" chip became a longer window label, so both
          numbers below were re-measured in a browser at 1920/1440/1280/390 --
          see the plan's Task 11. RE-DERIVE BOTH AGAIN if the caption, the title
          or the chip strip changes height. The failure mode is a silently
          half-visible card, not a broken build. */}
      <div
        ref={chartRef}
        className="w-full px-3 pt-4 [--board-chart-reserve:590px] lg:[--board-chart-reserve:390px]"
        style={{
          height: "clamp(260px, calc(100dvh - var(--board-chart-reserve)), 520px)",
        }}
      >
        {board.status === "loading" ? (
          // Deliberate, not a stall. Render's free tier cold-starts in 30-60s,
          // so this is what the first visitor of the day sees.
          <div className="h-full w-full rounded-lg bg-muted/40 animate-pulse" aria-hidden="true" />
        ) : board.status === "error" ? (
          // A chart-shaped message that NAMES the failure. Explicitly not a
          // permanent shimmer and explicitly not a fallback to sample curves:
          // either would make "the backend is down" and "the backend is fine"
          // render near-identically.
          <div className="h-full w-full rounded-lg border border-border bg-muted/20 flex flex-col items-center justify-center gap-2 px-6 text-center">
            <p className="text-sm text-foreground/80">The leaderboard didn&apos;t load.</p>
            <p className="text-xs font-mono text-muted-foreground">{board.message}</p>
            <p className="text-xs text-muted-foreground">
              The board itself is fine — reload to try again.
            </p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows} margin={{ top: 4, right: frame.gutter, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
              <XAxis
                dataKey="t"
                stroke="hsl(var(--muted-foreground))"
                fontSize={14}
                tickLine={false}
                axisLine={false}
                minTickGap={48}
                // `timeKey` slices to 16 chars, so the raw value here is
                // "2026-04-15T14:00". Without a formatter Recharts prints that
                // verbatim, while /app renders the byte-identical key as
                // "Apr 15" via formatShortDate (js/leaderboard.js:1096, used as
                // an axis tick callback at :2132). `formatAxisDate` mirrors it.
                // Omitting this was a defect in an earlier draft of this plan,
                // found in review of Task 9 and fixed before the bundle build.
                tickFormatter={formatAxisDate}
              />
              <YAxis
                stroke="hsl(var(--muted-foreground))"
                fontSize={14}
                tickLine={false}
                axisLine={false}
                domain={domain}
                width={yAxisWidth}
                tickFormatter={axisTick}
              />
              <Tooltip
                contentStyle={{ backgroundColor: "hsl(var(--card))", borderColor: "hsl(var(--border))", borderRadius: "8px" }}
                formatter={(value: number | string) =>
                  formatPercent(Number(value), 2)
                }
              />
              {series.map((s) => (
                <Line
                  key={s.key}
                  type="linear"
                  dataKey={s.key}
                  name={s.name}
                  stroke={s.color}
                  strokeWidth={s.isBaseline ? 1.5 : 2}
                  strokeDasharray={s.dash}
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
              ))}
              {/* Last, so it paints over the curves. `valueByKey`/`drawLabels`/
                  `gap` reach the rail because Recharts clones a <Customized>
                  child with the chart's own props and state spread OVER the
                  element's -- so an extra prop must not collide with a chart
                  prop or state key. These three do not. */}
              <Customized
                component={EndpointRail}
                valueByKey={valueByKey}
                drawLabels={frame.drawLabels}
                gap={frame.gap}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="px-5 pb-5 pt-3">
        {/* DEMOTION, NOT DELETION, and now doing two jobs. The chart ships no
            Recharts <Legend> — at this card's width a nine-item one wraps to
            three rows and pushes the plot area down — so this strip is the only
            thing linking a curve's colour to a model's name. It is ALSO the
            fallback whenever the endpoint rail declines to draw: a card too
            narrow or too short for the gutter labels, or a Recharts internal
            that moved under EndpointRail. Delete it and nine unnamed lines are
            left. The full standings, with ranks, live in Race.tsx.

            WRAPS, and must. `flex-nowrap` + `overflow-hidden` silently cut
            entries off the end whenever the strip was narrower than its
            content: measured scrollWidth 910 against clientWidth 285 at 390
            (four of five chips gone, leaving one model to key five drawn
            curves), 663 at 768, 895 at 1024 — so the whole lg band and every
            phone. No scrollbar, no ellipsis, nothing failing. The pressure is
            higher now, not lower: five entries became nine. */}
        <div
          data-testid="board-chip-strip"
          className="flex flex-wrap items-center gap-x-4 gap-y-2 text-base"
        >
          {standings.map((item) => (
            <span key={item.key} className="flex items-center gap-2 whitespace-nowrap">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                style={{ backgroundColor: item.color }}
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
        {/* Names the axis directly above it, and only that. The axis is percent
            now — see the plan's §6 — so a caption about "account value" would
            describe a chart that is not there. */}
        <p className="mt-3 text-sm text-foreground/65">
          Return over the competition window, hour by hour.
        </p>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_landing_chart_first.py dashboard/backend/tests/test_landing_live_board.py -v
cd dashboard/landing && npm run typecheck && cd ../..
```

Expected: PASS and clean. `test_landing_copy_register.py` is still red at this point — Task 10 owns it.

- [ ] **Step 5: Commit**

```bash
git status --short
git add dashboard/landing/src/components/home/BoardPreview.tsx dashboard/backend/tests/test_landing_chart_first.py
git commit -m "feat(landing): hero card draws the live competition board"
```

---

### Task 10: The Race standings, and the copy guards that anchored on samples

`Race.tsx` renders the full standings table from rows Task 9 left inline when it deleted `SAMPLE_STANDINGS` from `BoardPreview.tsx` (Task 9 rewrote the import so the branch stayed typecheck-clean; it did no other work here). The table moves to the same hook. A hero with real numbers above a table of invented ones, on the same page, is worse than either alone.

Three guards in `test_landing_copy_register.py` derive their corpus or their count from the sample constants and go red here. All three are **re-anchored, not relaxed** — each was catching something real, and what it was catching still needs catching.

**Files:**
- Modify: `dashboard/landing/src/components/home/Race.tsx`
- Modify: `dashboard/backend/tests/test_landing_copy_register.py`

**Interfaces:**
- Consumes: `useLeaderboard()` (Task 7).
- Produces: nothing.

- [ ] **Step 1: Update the failing guards**

In `dashboard/backend/tests/test_landing_copy_register.py`:

Replace `test_illustrative_example_label_appears_at_least_twice` with:

```python
def test_the_disclaimer_survives_where_the_data_is_still_invented():
    """The label was on three cards: the hero board, the Race standings, and the
    chat mock. Two of the three now draw the LIVE Competition board, and the
    label on real numbers is its own false claim -- so the count is 1, not 3,
    and asserting >= 2 would force the disclaimer back onto real data.

    The chat mock is still a mock, so it keeps it. Pinning that specifically,
    rather than dropping the guard, is what stops the next edit from removing the
    one place it is still true."""
    text = _shipped_text()
    assert text.count("Illustrative example") == 1, (
        "exactly one card is still illustrative — the chat simulation"
    )
    chat = (
        Path(__file__).resolve().parents[2]
        / "landing" / "src" / "components" / "home" / "ChatSimulation.tsx"
    ).read_text(encoding="utf-8")
    assert "Illustrative example" in chat


def test_the_board_cards_name_the_window_they_draw():
    """What replaced the disclaimer on the two board cards. They now draw real
    entries over a real window, and the window is the one detail that must not be
    left implicit -- the forward arrow under the chart otherwise reads as a claim
    that this window is still running, when it closed on 2026-05-15."""
    text = _shipped_text()
    assert text.count("Competition window") >= 2, (
        "both the hero card and the Race standings must state their provenance"
    )
```

Replace the closing block of `test_no_landing_component_puts_a_user_agent_on_the_board` — the four lines from `# Non-vacuity, scoped to whatever actually draws the board today.` to the end — with:

```python
    # Non-vacuity, scoped to whatever actually draws the board today. Membership
    # is derived rather than hardcoded to a filename: the chart moved to
    # BoardPreview.tsx when the board was promoted into the hero, and the sample
    # rows it was previously keyed on were deleted when the board went live. The
    # anchor is now the hook both board components read, which is the strongest
    # version of this check yet -- a component drawing a curve it did NOT get
    # from the API is exactly the thing being banned.
    board = {name: body for name, body in bodies.items() if "useLeaderboard" in body}
    assert board, "no component reads the live board"
    corpus = "".join(board.values())
    assert "dataKey=" in corpus, "one of them must actually draw curves"
    assert "SAMPLE_STANDINGS" not in corpus and "SAMPLE_CURVES" not in corpus, (
        "a board component that carries its own rows is drawing something the "
        "API did not send"
    )
```

Replace the docstring of `test_race_sample_cards_have_no_live_pulse` and add one assertion, leaving the existing ones intact:

```python
def test_race_sample_cards_have_no_live_pulse():
    """Race's Standings card carried "Illustrative example" yet a pulsing green
    "Live" badge sat beside it — animating exactly the claim the label
    disclaimed. The badge's ping dot was the landing's only use of Tailwind's
    ``animate-ping``, so its absence from the shipped text means the badge (not
    merely its caption) is gone.

    The card draws the live board now, which removes the contradiction but not
    the ban: a green pulse beside a FIXED historical window would be a fresh
    claim of its own, and a worse one for being on real numbers. The surrounding
    prose may still say "live" — the Live Trading Leaderboard is a real board
    with a real name, and live *market prices* are a real product property.

    The positive assertions pin that the cards themselves still ship AND that the
    bundle text was actually read: "Standings" and "Leaderboard" live only in the
    JS bundle, so a broken entry-bundle reference cannot turn the negative check
    vacuous (shown by fault injection during review)."""
    text = _shipped_text()
    assert "Standings" in text and "Leaderboard" in text
    assert "animate-ping" not in text
    assert "animate-pulse" not in text or "Competition window" in text, (
        "a pulse on this card must not outlive the window label that dates it"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_landing_copy_register.py -v
```

Expected: the new and updated cases FAIL — `Race.tsx` still imports the deleted constant and the bundle is still the old one.

- [ ] **Step 3: Move Race to the live board**

In `dashboard/landing/src/components/home/Race.tsx`:

Replace the `SAMPLE_STANDINGS` import with the hook:

```tsx
// The sample rows are gone: this table and the hero card render the SAME live
// Competition board, from one fetch. Real numbers in the hero above invented
// ones here, on the same page, is worse than either alone.
import { useLeaderboard } from "@/lib/useLeaderboard";
```

Inside `export function Race() {`, before the `return`:

```tsx
  const board = useLeaderboard();
  const standings = board.status === "ready" ? board.data.standings : [];
```

Replace the "Illustrative example" chip with the window label — a literal in this file rather than a constant shared with `BoardPreview.tsx`, for the reason that file already documents: `_shipped_text()` counts occurrences in the **minified bundle**, and esbuild collapses a shared constant to one string literal, so the DRY version renders on both cards while reading as one:

```tsx
              {/* Literal, not a shared constant — see the note in
                  BoardPreview.tsx: the guard counts occurrences in the minified
                  bundle. */}
              <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded shrink-0">
                {board.status === "ready" && board.data.windowLabel
                  ? `Competition window · ${board.data.windowLabel}`
                  : "Competition window"}
              </span>
```

Replace the rows block. The row's shape changed — rank is position in the standings, and the leader is the highlight (there is no user entry to highlight, and there never was one on any board):

```tsx
            <div className="space-y-2 mt-4">
              <div className="grid grid-cols-12 text-xs font-mono text-muted-foreground pb-2 px-2">
                <div className="col-span-2">Rank</div>
                <div className="col-span-7">AI model</div>
                <div className="col-span-3 text-right">Return</div>
              </div>
              {board.status === "loading" ? (
                <p className="px-2 py-6 text-sm text-muted-foreground">Loading the board…</p>
              ) : board.status === "error" ? (
                // Names the failure. Absent and broken must not render the same.
                <p className="px-2 py-6 text-sm text-muted-foreground">
                  The standings didn&apos;t load ({board.message}). Reload to try again.
                </p>
              ) : (
                standings.map((item, index) => (
                  <div
                    key={item.key}
                    className={`grid grid-cols-12 items-center p-3 border rounded-lg ${
                      index === 0
                        ? "bg-primary/10 border-primary/40"
                        : "bg-background border-border"
                    }`}
                  >
                    <div className="col-span-2 font-mono font-bold text-muted-foreground">
                      #{index + 1}
                    </div>
                    <div
                      className={`col-span-7 font-medium truncate pr-2 ${
                        index === 0 ? "text-primary" : "text-foreground"
                      }`}
                    >
                      {item.name}
                    </div>
                    <div
                      className={`col-span-3 text-right font-mono font-bold ${
                        index === 0
                          ? "text-primary"
                          : item.ret.startsWith("-")
                            ? "text-destructive"
                            : "text-positive"
                      }`}
                    >
                      {item.ret}
                    </div>
                  </div>
                ))
              )}
            </div>
```

Leave `BOARD_RULES` and the Season 0 paragraph exactly as they are. `"Live Trading Leaderboard"` in `BOARD_RULES[1].text` is the string `test_race_source_and_shipped_bundle_agree` anchors on both sides — it is the only thing that catches a skipped `npm run build`, and it must not move.

- [ ] **Step 4: Run the source-level tests**

```bash
pytest dashboard/backend/tests/test_landing_chart_first.py -v
cd dashboard/landing && npm run typecheck && cd ../..
```

Expected: PASS and clean. `test_landing_copy_register.py` stays red until the bundle is rebuilt — it reads the shipped bundle, which is the point of it.

- [ ] **Step 5: Commit**

```bash
git status --short
git add dashboard/landing/src/components/home/Race.tsx dashboard/backend/tests/test_landing_copy_register.py
git commit -m "feat(landing): race standings read the live board"
```

---

### Task 11: Re-derive the height reserves, rebuild the bundle, verify the hash

Two measured constants and one hand-patched file, in that order. The reserves have to be measured against the *built* card, and the bundle has to be built before the copy guards can read any of this — so this is one task, not three.

**The reserves are the silent failure in this plan.** `--board-chart-reserve` is 590px stacked and 390px at `lg+`, both derived from the measured height of the title, the chip, the caption and the chip strip. The strip goes from 5 entries to 9 and the chip's text changes length. The failure mode is a half-visible card below the fold with nothing failing.

**Files:**
- Modify: `dashboard/landing/src/components/home/BoardPreview.tsx` (the two reserve numbers only)
- Modify: `dashboard/frontend/index.html`, `dashboard/frontend/assets/*`
- Test: `dashboard/backend/tests/test_landing_chart_first.py` (the clamp assertion)

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

- [ ] **Step 1: Build and serve the real card**

```bash
cd /mnt/d/github/agent-trading-lab/dashboard/landing
npm ci
npm run build
cd /mnt/d/github/agent-trading-lab
DATABASE_PATH=/tmp/atl-hero-check.db ~/atl-venv/bin/python -m uvicorn dashboard.backend.app:app --port 8011
```

Serve `dashboard/landing/dist/public` directly (or copy the assets per Step 4 first and use `http://localhost:8011/`) — the reserves must be measured against built CSS, not the dev server.

- [ ] **Step 2: Measure the card's non-chart height at both bands**

At **1440×900** and at **390×844**, with the board loaded (not in the loading state — the chip strip's height is what is being measured):

```js
// Run in the browser console on the landing page.
const card = document.querySelector('[data-testid="board-chip-strip"]').closest('.bg-card');
const chart = card.querySelector('[style*="--board-chart-reserve"], [style*="clamp("]');
console.log({
  cardHeight: card.getBoundingClientRect().height,
  chartHeight: chart.getBoundingClientRect().height,
  nonChart: card.getBoundingClientRect().height - chart.getBoundingClientRect().height,
  stripHeight: document.querySelector('[data-testid="board-chip-strip"]').getBoundingClientRect().height,
  stripClipped:
    document.querySelector('[data-testid="board-chip-strip"]').scrollWidth >
    document.querySelector('[data-testid="board-chip-strip"]').clientWidth,
  cardBottom: card.getBoundingClientRect().bottom,
  viewport: window.innerHeight,
});
```

Record `nonChart` at each band. `stripClipped` must be `false` at both — if it is true the strip is cutting model names off the end again, which is the failure this card has already shipped once.

New reserves, using the same arithmetic the existing comment documents:

- `lg+` reserve = `nonChart@1440` + 120 (chrome) + fold margin
- stacked reserve = `nonChart@390` + 132 (section padding) + margin

Then confirm `cardBottom <= viewport` at 1920×1080, 1440×900, 1440×768, 1366×768, 1280×800, 1280×720 and 390×844.

- [ ] **Step 3: Write the measured numbers in**

Update the two arbitrary properties in `BoardPreview.tsx` and the arithmetic in the comment block above them with what you measured. Then update the guard so it pins the new values rather than the old:

```python
def test_the_landing_chart_uses_its_own_measured_clamp():
    """Both reserves are derived, not taste, and there are TWO because the card's
    non-chart height is not one number: one thing beside the copy at >=lg, and
    another stacked at 390px wide where the title, the window chip and the
    caption all wrap and the chip strip runs to several rows.

    RE-DERIVED when the board went live: the strip went from five hardcoded
    entries to nine from the payload, and the "Illustrative example" chip became
    a longer window label. Both numbers below were re-measured against the BUILT
    card at 1440x900 and 390x844 -- see the plan's Task 11.

    RE-DERIVE BOTH AGAIN if the caption, title or chip strip changes height. The
    failure mode is a silently half-visible card, not a broken build. That is not
    hypothetical: unclipping the chip strip once took it from 24px to 152px at
    390px wide, the stacked reserve from 480 to 590, and then the FLOOR from 300
    to 260 -- because at 844px tall the card had 269px left for a chart whose
    floor was 300.

    The var() indirection is load-bearing and not a tidy-up: the formula's
    commas defeat Tailwind's arbitrary-VALUE parser, so the breakpoint-dependent
    number rides an arbitrary PROPERTY instead, which does take a prefix.
    """
    board = _BOARD.replace(" ", "")
    assert "clamp(260px,calc(100dvh-var(--board-chart-reserve)),520px)" in board
    assert "[--board-chart-reserve:<STACKED>px]" in board, "the stacked-phone reserve"
    assert "lg:[--board-chart-reserve:<LG>px]" in board, "the side-by-side reserve"
    assert "56vh" not in _BOARD, "the first draft's clamp fails at four viewports"
    assert "h-[210px]" not in _BOARD and "md:h-[240px]" not in _BOARD
```

Substitute the two measured integers for `<STACKED>` and `<LG>` in both the guard and the component. If the measurements come back equal to the current 590/390, keep those numbers and say so in the comment — a re-derivation that lands on the old value is a result, not a no-op.

- [ ] **Step 4: Refresh the shipped bundle, keeping the hand-written auth layer**

```bash
cd /mnt/d/github/agent-trading-lab/dashboard/landing
npm run build
cp dist/public/assets/* ../frontend/assets/
```

Then, **by hand**, in `dashboard/frontend/index.html`: point the `<script>` and `<link>` at the new content-hashed filenames and delete the superseded `../frontend/assets/index-*.{js,css}`. **Keep** the auth-gate `<script>` in `<head>`, the `#landingAuthModal` markup, `<style id="landing-auth-patch">` and the end-of-body auth `<script>`.

Do **not** copy `dist/public/index.html` over `dashboard/frontend/index.html`. The Vite template is 25 lines; the shipped file is 418. The ~393-line difference is the auth layer, and overwriting it kills every landing CTA with no console error (issue #225).

- [ ] **Step 5: Verify the bundle is what this branch's source builds to**

The bundle is byte-reproducible, so the hash is a real check rather than a spot inspection:

```bash
cd /mnt/d/github/agent-trading-lab/dashboard/landing
npm run build
sha256sum dist/public/assets/index-*.js
sha256sum ../frontend/assets/index-*.js
ls ../frontend/assets/
```

The two hashes must match and the emitted asset's filename must be the one `dashboard/frontend/index.html` references. Confirm no superseded `index-*.js`/`index-*.css` is left behind.

- [ ] **Step 6: Run the whole suite**

```bash
cd /mnt/d/github/agent-trading-lab
pytest dashboard/backend/tests/ -v
```

Expected: green, including every guard in the spec's §8 — updated rather than deleted. `test_frontend_bundle_integrity.py` and `test_race_source_and_shipped_bundle_agree` are the two that specifically prove the rebuild happened.

- [ ] **Step 7: Look at the finished card**

At 1920, 1440, 1280 and 390 wide, on the landing page:

1. nine curves are distinguishable and no gutter label is clipped;
2. the card's bottom edge is above the fold at every viewport in Step 2's list;
3. the chip strip wraps rather than clipping;
4. the loading state looks deliberate (throttle the network, or stop the backend, to see it);
5. the failed state names the failure (stop the backend and reload).

**If nine curves are not legible at 1280 or below, the fix is fewer curves — not a rescaled axis.** The real board spans −0.43% to +7.49%; widening the domain to manufacture separation would imply more movement than occurred. The narrowing that keeps the design honest is to draw models only in the hero and leave all nine on screen 0, which costs the baselines the hero was given in order to make its numbers judgeable — so take it only if the browser says you must, and record which it was.

- [ ] **Step 8: Commit and open PR B**

```bash
git status --short   # dashboard/storage/data/ must be untouched
git add dashboard/landing/src/components/home/BoardPreview.tsx \
        dashboard/backend/tests/test_landing_chart_first.py \
        dashboard/frontend/index.html dashboard/frontend/assets
git commit -m "build(landing): rebuild the bundle for the live board hero"
git push -u origin HEAD
gh pr create --title "feat(landing): hero draws the live competition board" --body "$(cat <<'BODY'
The signed-out landing hero now draws the same live Competition board the
signed-in Home screen draws, with the shared nof1 frame from the previous PR.
Spec: docs/superpowers/specs/2026-08-19-nof1-leaderboard-frame-design.md §5-§6.

- One fetch shared by the hero and the Race standings; three distinguishable
  states (loading / ready / failed), no fallback to sample curves.
- Same selection rule as screen 0: every model plus `buy_hold_djia` and
  `djia_index` — 9 of 12 entries.
- Y-axis is percent. Forced, not chosen: the dollar axis was justified by the
  curves being fabricated with a base of 1000, and live dollar levels are a
  x0.1 rescale of a $100k backtest onto a $10k display base.
- "Illustrative example" comes off both board cards and is replaced by the
  window they draw. The chat mock keeps it.
- Bundle rebuilt and hash-verified; the hand-written auth layer re-applied by
  hand per dashboard/landing/README.md (issue #225).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

---

# PR C — the backend season block

> Independent of A and B. It exists so "seasons are two weeks" is expressed in a real contract rather than only in conversation.

### Task 12: `live` becomes a real period with a season payload

The client contract for seasons is **already fully written and entirely unimplemented server-side**. `js/leaderboard.js`'s render path reads `season.number`, `.status`, `.start_date`, `.end_date`, `.last_advanced_date`, `.trading_days_elapsed`, `.trading_days_total`, `.entries_open`, `.entry_closes_at`, `.entry_count`, `.next_advance_at` and `.gaps`. This fills in fields the client already asks for.

**The one invariant that must hold.** `season.last_advanced_date` stays `null` and `season.trading_days_elapsed` stays `0`, because no season has advanced. `seasonHasAdvanced()` tests exactly those two fields — deliberately, rather than `payload.period !== 'live'` — precisely so that adding `"live"` to `VALID_PERIODS` cannot clear the preview banner. That was designed for this commit. Emitting a non-null `last_advanced_date` here would flip the badge to "Running" and promise a nightly advance that nothing performs.

**Nothing here may spend money.** The live config reuses the contest `session_id` and the contest window, so `ensure_leaderboard_runs` finds every run cached and deploys nothing. A live branch that invented its own window would make `_find_cached_run` miss on all twelve entries and start recomputing baselines — and, with `LEADERBOARD_DAILY_AUTO_DEPLOY` armed, LLM deploys — from a *public, unauthenticated* GET.

**Files:**
- Modify: `dashboard/config/leaderboard.json`
- Modify: `dashboard/backend/domain/leaderboard/service.py`
- Test: `dashboard/backend/tests/test_leaderboard_season.py` (create)

**Interfaces:**
- Consumes: `load_leaderboard_config()`, `_normalize_period()`, `resolve_leaderboard_config()`, `get_leaderboard()`, all in `service.py`.
- Produces:
  - `VALID_PERIODS = ("contest", "daily", "live")`
  - `season_window(start_date: str, trading_days: int) -> tuple[str, str]`
  - `build_season_payload(config: Dict[str, Any]) -> Dict[str, Any]`
  - `get_leaderboard(period="live")` returns a payload with a `season` key.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_leaderboard_season.py`:

```python
"""The season contract: `live` is a real period, and Season 0 has not advanced.

The client half of this contract already ships in full and has never had a
server to talk to -- js/leaderboard.js reads eleven season fields. These pin the
shape it reads and, more importantly, the one thing that must NOT be true yet.
"""

import json

import pytest

from dashboard.backend.domain.leaderboard import service


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """`get_leaderboard` calls `ensure_leaderboard_runs`, which fetches bars.

    Stubbed here for every case in this module, following the pattern in
    tests/domain/leaderboard/test_service_move.py. Nothing in this file is about
    run production -- with the suite's temp DATABASE_PATH `_find_cached_run`
    misses on all twelve entries and `entries` comes back empty, which is
    exactly the shape these assertions want: the season block must be attached
    by the PERIOD, not by anything the roster happens to contain.
    """
    monkeypatch.setattr(
        service,
        "ensure_leaderboard_runs",
        lambda force_refresh=False, period="contest", config=None: {
            "session_id": (config or {}).get("session_id", "leaderboard-contest"),
            "created": 0,
            "refreshed_at": "2026-08-19T00:00:00+00:00",
        },
    )


def test_live_is_a_real_period():
    """`_normalize_period` coerces anything unrecognised back to 'contest', so
    before this change `?period=live` returned a perfectly successful HTTP 200
    carrying the Competition board."""
    assert "live" in service.VALID_PERIODS
    assert service._normalize_period("live") == "live"
    assert service._normalize_period("LIVE") == "live"
    assert service._normalize_period("season") == "contest", (
        "coercion stays the behaviour for genuinely unknown periods"
    )


def test_the_live_board_reuses_the_contest_runs_and_window():
    """Nothing in this change may spend money. A live branch with its own window
    would miss `_find_cached_run` on all twelve entries and start recomputing
    baselines -- and, with LEADERBOARD_DAILY_AUTO_DEPLOY armed, LLM deploys --
    from a public, unauthenticated GET."""
    base = service.load_leaderboard_config()
    live = service.resolve_leaderboard_config("live")
    assert live["session_id"] == base["session_id"]
    assert live["start_date"] == base["start_date"]
    assert live["end_date"] == base["end_date"]
    assert live["period"] == "live"


def test_a_season_is_ten_trading_days_which_is_two_calendar_weeks():
    """Ten US cash sessions, Monday through Friday. Not a new number:
    js/leaderboard.js already declares `const SEASON_TRADING_DAYS = 10;` with
    exactly that comment."""
    start, end = service.season_window("2026-08-12", 10)
    assert start == "2026-08-12"
    assert end == "2026-08-25", "Wed 12 Aug through Tue 25 Aug is ten sessions"


def test_the_season_payload_says_nothing_has_advanced():
    """THE invariant. `seasonHasAdvanced()` tests `last_advanced_date` and
    `trading_days_elapsed` -- deliberately, rather than the period string --
    precisely so that adding "live" to VALID_PERIODS cannot clear the preview
    banner. A non-null date here flips the badge to "Running" and promises a
    nightly advance that nothing performs."""
    season = service.build_season_payload(service.resolve_leaderboard_config("live"))
    assert season["last_advanced_date"] is None
    assert season["trading_days_elapsed"] == 0
    assert season["next_advance_at"] is None
    assert season["entries_open"] is False
    assert season["status"] != "running"


def test_season_zero_is_numbered_zero_and_survives_json():
    """Season 0 is the shakedown season by convention: numbered, so the board has
    a real identity to show, but explicitly the one whose results nobody should
    read as a standing. It is also FALSY, which is the whole hazard on the client
    side -- `displayedSeasonNumber()` exists for it."""
    season = service.build_season_payload(service.resolve_leaderboard_config("live"))
    assert season["number"] == 0
    assert json.loads(json.dumps(season))["number"] == 0


def test_the_season_payload_carries_every_field_the_client_reads():
    """The client contract was written before the server existed. A missing key
    is not a crash there -- the render path uses optional chaining throughout --
    it is a silently blank strip."""
    season = service.build_season_payload(service.resolve_leaderboard_config("live"))
    for field in (
        "number", "status", "start_date", "end_date", "last_advanced_date",
        "trading_days_elapsed", "trading_days_total", "entries_open",
        "entry_closes_at", "entry_count", "next_advance_at", "gaps",
    ):
        assert field in season, f"the client reads season.{field}"


def test_the_config_declares_the_season_rather_than_the_code():
    cfg = service.load_leaderboard_config()
    assert cfg["season"]["length_trading_days"] == 10
    assert cfg["season"]["season_zero_start"] == "2026-08-12"


def test_only_the_live_board_carries_a_season():
    """The Competition board is one fixed historical window and is not a season;
    attaching one would make the season strip render on a board that has none."""
    assert "season" not in service.get_leaderboard(period="contest")
    assert "season" in service.get_leaderboard(period="live")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_leaderboard_season.py -v
```

Expected: FAIL — `season_window` and `build_season_payload` do not exist and `"live"` is not in `VALID_PERIODS`.

- [ ] **Step 3: Declare the season in config**

In `dashboard/config/leaderboard.json`, add a `season` block as a sibling of `initial_capital` (before `"strategies"`):

```json
  "season": {
    "length_trading_days": 10,
    "season_zero_start": "2026-08-12"
  },
```

- [ ] **Step 4: Implement the period and the payload**

In `dashboard/backend/domain/leaderboard/service.py`:

Change the periods tuple (currently line 50):

```python
VALID_PERIODS = ("contest", "daily", "live")
```

Add these two functions immediately above `resolve_leaderboard_config`:

```python
# Season 0 is the shakedown season by convention: numbered, so the board has a
# real identity to show and so Season 1 means "the first one that counted", but
# explicitly the one whose results nobody should read as a standing. Mirrors
# PREVIEW_SEASON_NUMBER in dashboard/frontend/js/leaderboard.js.
PREVIEW_SEASON_NUMBER = 0
DEFAULT_SEASON_TRADING_DAYS = 10


def season_window(start_date: str, trading_days: int) -> Tuple[str, str]:
    """The (start, end) dates of a season ``trading_days`` sessions long.

    Ten sessions is two calendar weeks of US cash trading, Monday through
    Friday. Not a new number: ``js/leaderboard.js`` already declares
    ``const SEASON_TRADING_DAYS = 10;`` with exactly that comment.

    Weekdays only -- market holidays are NOT modelled. That is correct for
    Season 0, whose window (2026-08-12 → 2026-08-25) contains none, and it is
    knowingly insufficient for the advance engine, which will need a real
    calendar. It is stated here rather than left for that engine to discover:
    the failure would be a season that ends one session short with nothing
    reporting it.
    """
    start = date.fromisoformat(start_date)
    cursor = start
    counted = 0
    last = start
    while counted < max(int(trading_days), 1):
        if cursor.weekday() < 5:
            counted += 1
            last = cursor
        cursor += timedelta(days=1)
    return start.isoformat(), last.isoformat()


def build_season_payload(config: Dict[str, Any]) -> Dict[str, Any]:
    """The season block the Live Trading tab renders.

    NOTHING HERE MAY CLAIM AN ADVANCE. ``last_advanced_date`` stays None and
    ``trading_days_elapsed`` stays 0 because no season has advanced -- there is
    no advance engine. ``seasonHasAdvanced()`` on the client tests exactly those
    two fields, deliberately rather than the period string, precisely so that
    teaching the server the word "live" cannot clear the preview banner. A date
    here flips the badge to "Running" and prints "Next advance: nightly after
    the 16:00 ET close" under a board nothing updates.

    Every key the client reads is present. A missing one is not a crash there --
    the render path uses optional chaining throughout -- it is a silently blank
    season strip, which is worse.
    """
    season_cfg = config.get("season") or {}
    trading_days = int(season_cfg.get("length_trading_days") or DEFAULT_SEASON_TRADING_DAYS)
    start, end = season_window(
        season_cfg.get("season_zero_start") or config["start_date"], trading_days
    )
    return {
        "number": PREVIEW_SEASON_NUMBER,
        "status": "preview",
        "start_date": start,
        "end_date": end,
        "last_advanced_date": None,
        "trading_days_elapsed": 0,
        "trading_days_total": trading_days,
        "entries_open": False,
        "entry_closes_at": None,
        "entry_count": 0,
        "next_advance_at": None,
        "gaps": [],
    }
```

Add the `live` branch to `resolve_leaderboard_config`, immediately before the final `return`:

```python
    if period_key == "live":
        # The contest session and the contest window, deliberately. This board
        # is a Season 0 PREVIEW: real Competition curves under season chrome,
        # with a banner saying nothing here has advanced. Inventing a window
        # would make `_find_cached_run` miss on all twelve entries and start
        # recomputing baselines -- and with LEADERBOARD_DAILY_AUTO_DEPLOY armed,
        # LLM deploys -- from a public, unauthenticated GET.
        return {
            **base,
            "period": "live",
            "board_title": "Live Trading Leaderboard",
            "phase_label": "Season 0",
            "standings_label": "Ranking",
        }
```

In `get_leaderboard`, attach the block. After the `daily_status` assignment to `payload` at the end:

```python
    if config.get("period") == "live":
        payload["season"] = build_season_payload(config)
```

No new imports are needed: `date`, `timedelta` (line 18) and `Tuple` (line 19) are already imported in this module. If a later edit does need one, add it to those existing lines rather than writing a new import statement — `py/import-and-import-from` is a live CodeQL rule on this file (see the note above `INITIAL_CAPITAL`).

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_leaderboard_season.py -v
pytest dashboard/backend/tests/test_frontend_live_trading_board.py -v
```

Expected: PASS. The second suite matters most here: `test_preview_is_anchored_on_evidence_that_a_season_ran` is the guard that was written in advance of exactly this commit.

- [ ] **Step 6: Commit**

```bash
git status --short   # dashboard/storage/data/ must be untouched
git add dashboard/config/leaderboard.json dashboard/backend/domain/leaderboard/service.py dashboard/backend/tests/test_leaderboard_season.py
git commit -m "feat(leaderboard): live period with a two-week season block"
```

---

### Task 13: Say so at the route, and prove the banner survives

The route's own description still names two periods. And the claim that matters — that a real `live` period does **not** clear the Season 0 preview banner — is currently asserted only about the client's source shape. Now that the server can send a season, it can be asserted end to end.

**Files:**
- Modify: `dashboard/backend/api/routers/leaderboard.py`
- Test: `dashboard/backend/tests/test_frontend_live_trading_board.py` (extend)

**Interfaces:**
- Consumes: `get_leaderboard(period=...)` from Task 12.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_frontend_live_trading_board.py`:

```python
def test_the_live_route_answers_with_a_season_and_still_reads_as_preview(monkeypatch):
    """End to end, not source shape: the payload the route actually serves must
    fail `seasonHasAdvanced()`'s test.

    That function is `last_advanced_date || trading_days_elapsed > 0`, and it is
    the anchor under every disclaimer on this tab. Adding "live" to VALID_PERIODS
    was named in its docstring as the change most likely to break it silently --
    a one-line commit that needs no season payload at all -- so this is the case
    that closes that loop from the other side.
    """
    from dashboard.backend.domain.leaderboard import service

    # `ensure_leaderboard_runs` fetches bars; this module is about the payload
    # shape, not run production. Same stub tests/domain/leaderboard/
    # test_service_move.py uses.
    monkeypatch.setattr(
        service,
        "ensure_leaderboard_runs",
        lambda force_refresh=False, period="contest", config=None: {
            "session_id": "leaderboard-contest",
            "created": 0,
            "refreshed_at": "2026-08-19T00:00:00+00:00",
        },
    )
    payload = service.get_leaderboard(period="live")
    season = payload["season"]
    assert payload["period"] == "live"
    assert not season["last_advanced_date"]
    assert not (season["trading_days_elapsed"] > 0)


def test_the_route_documents_the_third_period():
    """The description is what /docs shows and what the next reader believes."""
    router = (
        Path(__file__).resolve().parents[1] / "api" / "routers" / "leaderboard.py"
    ).read_text(encoding="utf-8")
    assert "'live'" in router or '"live"' in router
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_frontend_live_trading_board.py -v -k "live_route or third_period"
```

Expected: the route-description case FAILS.

- [ ] **Step 3: Update the route description**

In `dashboard/backend/api/routers/leaderboard.py`, the `period` query parameter (currently line 33–35):

```python
    period: str = Query(
        "contest",
        description=(
            "Leaderboard period: 'contest' (fixed preseason window), "
            "'daily' (last completed weekday), or 'live' (Season 0 preview — "
            "the contest curves under season chrome; no season has advanced)."
        ),
    ),
```

Update the docstring below it in the same spirit, and leave the `?period=daily` polling note for `daily_status` intact.

- [ ] **Step 4: Run the full suite**

```bash
pytest dashboard/backend/tests/ -v
```

Expected: green.

- [ ] **Step 5: Check the tab in a browser**

```bash
DATABASE_PATH=/tmp/atl-season-check.db ~/atl-venv/bin/python -m uvicorn dashboard.backend.app:app --port 8011
```

On `/app` → Leaderboard → **Live Trading Leaderboard**, confirm:

1. the preview banner is still shown and still says the season has not been run;
2. the badge reads **Season 0**, not `Season —` (Season 0 is falsy — `displayedSeasonNumber()` exists for exactly this);
3. the season strip shows `Aug 12 – Aug 25` and `0 / 10`;
4. no "Next advance" line and no "last completed" date appears anywhere.

Any of those failing means the payload is claiming something the engine does not do — fix the payload, never the banner.

- [ ] **Step 6: Commit and open PR C**

```bash
git status --short
git add dashboard/backend/api/routers/leaderboard.py dashboard/backend/tests/test_frontend_live_trading_board.py
git commit -m "docs(leaderboard): name the live period at the route"
git push -u origin HEAD
gh pr create --title "feat(leaderboard): live period with a season block" --body "$(cat <<'BODY'
`?period=live` becomes a real period instead of being coerced back to the
Competition board, and carries a `season` block for the two-week season the
client already knows how to render.
Spec: docs/superpowers/specs/2026-08-19-nof1-leaderboard-frame-design.md §7.

- `season.length_trading_days: 10` and `season_zero_start: 2026-08-12` are
  declared in dashboard/config/leaderboard.json, not in code.
- Season 0 runs 2026-08-12 → 2026-08-25: ten US cash sessions, two calendar
  weeks. Market holidays are not modelled — August has none, and the advance
  engine will need a real calendar.
- `last_advanced_date` is null and `trading_days_elapsed` is 0, so the tab's
  Season 0 preview banner still shows. That is the invariant
  `seasonHasAdvanced()` was written to protect against this exact commit.
- The live config reuses the contest session and window, so nothing recomputes
  and nothing deploys.

No advance engine. This ships the contract it will fill.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

---

## Verification (all three PRs landed)

- [ ] `pytest dashboard/backend/tests/ -v` green, with every guard in the spec's §8 **updated rather than deleted**.
- [ ] The landing bundle rebuilt and its asset hash matching the committed filename (Task 11, Step 5).
- [ ] All three charts checked in a browser at 1920, 1440, 1280 and 390 wide: gutter labels not clipped, the hero card's bottom edge above the fold, and the Leaderboard tab's hover emphasis still clearing when the pointer enters the widened gutter.
- [ ] `?period=live` returns a season payload **and** the Live tab still shows the Season 0 preview banner.
- [ ] `cd dashboard/landing && npm run typecheck` clean.

## Follow-ups, explicitly out of scope

- **The season advance engine** — the handwritten note's ①②③ hourly-append mechanism, two-week rollover, and persistence for an advancing live curve. This plan ships the frame and the contract it will fill; it needs no further chart work when it lands. It will need a market-holiday calendar (see `season_window`).
- **Issue #225** — the hand-patched bundle. Task 11 performs the hand patch again rather than fixing the underlying problem.
- **`MarketTicker.tsx`'s `apiBase()`** — it hardcodes the Render origin, which `test_frontend_api_base.py` calls a same-origin cookie auth regression for every source it can read; it survives only because that guard excludes minified `assets/`. Bringing it to a root-relative path is a small separate change.
- **Model avatar icons.** nof1 uses circular per-model logos in the gutter; this design uses colour dots, the existing visual language on all three surfaces.
- **User-facing docs.** Nothing in `docs/` describes the landing chart's units or the leaderboard periods in a way this work invalidates — but the `?period` values are published through the RTD API reference, so if that page enumerates periods it needs the third one. Check before closing PR C.
