// ============================================================================
// LEADERBOARD (Competition tab — real baselines from API)
// ============================================================================

let leaderboardPayload = null;
let currentLeaderboardSort = 'rank'; // 'rank' | 'value' | 'return' | 'sharpe' | 'dd'
let currentLeaderboardSortDir = 'asc'; // rank: asc (1 best); metrics usually start desc
let selectedLeaderboardEntry = null;
let equityCurvesData = null;
let equityCurvesChartInstance = null;
let currentChartView = 'absolute'; // default: money ($). 'cumulative' = % return
let leaderboardListenersInitialized = false;
// The board the payload *on screen* was requested for, which is NOT always the
// board the API returned. `_normalize_period` server-side coerces any period it
// does not know back to 'contest', so asking for the live board the backend
// cannot serve yields a perfectly valid contest payload with no error anywhere.
// Keeping the request alongside the response is the only thing that can tell
// those two apart.
//
// Written only where `leaderboardPayload` is written, and never before the
// await: a global assigned at request time belongs to whichever request was
// issued last, not to the response currently being rendered. Set it early and a
// slow live response repaints itself under the Competition board's name.
let renderedBoardPeriod = 'contest';
// Monotonic request id. Board switches are user-paced and the API is not, so two
// loads can be in flight at once and resolve out of order; only the newest may
// touch the DOM.
let boardRequestSeq = 0;

// Chart visual-hierarchy state
let hiddenSeries = new Set();
let hiddenInitialized = false;
let hoveredDatasetIndex = null;
// Pixel position on the hovered curve directly under the pointer, or null. Kept
// alongside the index so the hover dot sits on the line rather than snapping to
// the nearest data point — on the Daily board those are ~97px apart.
let hoveredPoint = null;
let canvasPointerBound = false;
const selectedBenchmarkLabel = 'SPY';
/** Expanded groups in the Show-on-Chart picker: model | baseline | index */
let curvePickerExpanded = new Set();
let curvePickerOutsideBound = false;
// Max vertical distance from a curve before hover emphasis clears. Without it,
// Chart.js `nearest` + `intersect:false` keeps a curve selected anywhere inside
// the canvas — including empty plot space and the label gutter.
const HOVER_HIT_RADIUS_PX = 16;

// Stable per-series style presets. `kind` drives width/opacity hierarchy:
//   benchmark -> neutral gray, dotted / dash-dot, understated
//   strategy  -> colored, long-dashed, secondary
//   team      -> solid, prominent (colors assigned stably, never by rank)
const LEADERBOARD_STYLES = {
  SPY: { color: '#CBD5E1', kind: 'benchmark', dash: [2, 4] },
  DJIA: { color: '#94A3B8', kind: 'benchmark', dash: [8, 4, 2, 4] },
  'Buy & Hold': { color: '#38BDF8', kind: 'strategy', dash: [10, 6] },
  'Mean-Variance': { color: '#C084FC', kind: 'strategy', dash: [10, 6] },
  'Equal-Weight': { color: '#4ADE80', kind: 'strategy', dash: [10, 6] },
};

// Visual hierarchy: teams are boldest, provided models prominent (solid),
// strategy baselines secondary (dashed), market indices the most understated.
const KIND_WIDTH = { team: 2.25, model: 2.0, strategy: 1.6, benchmark: 1.1 };
const KIND_ALPHA = { team: 1.0, model: 0.95, strategy: 0.7, benchmark: 0.5 };
const EMPHASIS_WIDTH = 3;

// Stable bright palette for actual competition teams (assigned first-seen).
const TEAM_COLOR_PALETTE = [
  '#F97316', '#EAB308', '#EC4899', '#14B8A6', '#A855F7',
  '#EF4444', '#06B6D4', '#84CC16', '#F43F5E', '#8B5CF6',
];
const teamColorMap = {};

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
const modelColorMap = {};

function formatEntryBadge(badge) {
  const raw = String(badge || '').trim();
  if (!raw || raw === 'Baseline') return 'Baseline Strategy';
  if (raw === 'Index') return 'Market Index';
  if (raw === 'Strategy') return 'Baseline Strategy';
  return raw;
}

function isModelEntry(entry) {
  return !!(entry && (entry.is_model || entry.team_badge === 'Model'));
}

function getModelColor(stableId) {
  const key = String(stableId);
  if (!modelColorMap[key]) {
    const idx = Object.keys(modelColorMap).length % MODEL_COLOR_PALETTE.length;
    modelColorMap[key] = MODEL_COLOR_PALETTE[idx];
  }
  return modelColorMap[key];
}

function shortName(label) {
  // Model names are already canonical/short; only guard against long team names.
  return label && label.length > 18 ? `${label.slice(0, 17)}…` : (label || '');
}

/** `#rgb` / `#rrggbb` -> `{ r, g, b }`, black on anything unparseable.
 *
 *  ONE parse, TWO callers. `hexToRgba` (curve strokes, leader lines) and
 *  `boardPillTextColor` (pill ink) both need the channels, and carrying the
 *  same four lines twice meant the 3-digit form was wrong in both at once:
 *  `#fff` read as r=255 g=15 b=0, which is a light-ink verdict on a near-white
 *  pill in one and an orange-red stroke in the other. Every palette entry in
 *  this file is 6-digit, so neither ever fired -- but a fix applied to one copy
 *  would not have reached the other, which is the actual defect. */
function hexToRgb(hex) {
  let h = String(hex || '').replace('#', '');
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  return {
    r: parseInt(h.slice(0, 2), 16) || 0,
    g: parseInt(h.slice(2, 4), 16) || 0,
    b: parseInt(h.slice(4, 6), 16) || 0,
  };
}

function hexToRgba(hex, alpha) {
  const { r, g, b } = hexToRgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

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
// The 40% is a CEILING on the measured floor, not the width the rail always
// takes -- see boardFrameLayout.
//
// WHERE IT ACTUALLY BINDS, since "caps how much of the board the rail may
// take" overstates it: the gutter is `max(floor, min(width * 0.4, floor +
// SLACK))`, so the fraction only wins while `width * 0.4 < floor + SLACK` --
// i.e. below `2.5 * floor + 90` px of canvas, about 613px at today's ~209px
// measured floor. Every desktop width this dashboard renders at is above that,
// so on a 1440 or 1600px tab the gutter is floor + SLACK and the fraction is
// inert; it binds on the narrow home panel and on mobile, which is exactly
// where an unbounded rail would eat the plot. BOARD_GUTTER_MAX_FRACTION
// (below) is the one width-relative guard that fires at any size: past it the
// frame gives up entirely and gives the space back, drawing no labels at all.
const BOARD_GUTTER_MAX_FRACTION = 0.5;
const BOARD_GUTTER_FONT = '600 11px Inter, system-ui, sans-serif';
// Where the label block starts relative to chartArea.right, and the clear
// canvas left to the right of the widest one so the arrowhead has room.
const BOARD_GUTTER_TEXT_INSET = 12;
const BOARD_GUTTER_TRAILING_PAD = 16;
// Extra indent for a label that hangs into the x-axis strip -- see the band
// note on boardStackLabels for why the strip is not the empty canvas it looks
// like. Chart.js centres the LAST x tick on chartArea.right, so roughly half
// its width overhangs into our gutter: ~18px for this tab's 12px `May 15`,
// ~21px for screen 0's 14px ticks. Only the 6px leading dot reaches that far
// (the name starts at inset + 10), and labels draw after the scales so they
// win the pixels -- but a colour dot sitting on the tail of a tick label is
// still a collision, and it is the bottom-most labels, the ones the taller-
// than-the-plot stack routinely produces, that hit it.
//
// A LITERAL IS DEFENSIBLE HERE, unlike BOARD_XAXIS_ALLOWANCE below. This one
// is RESERVED unconditionally in boardLabelBlockWidth and spent only by the
// labels that descend, so getting it wrong costs a few px of residual overlap
// -- never clipped text, and never a wrong verdict. 12 clears both measured
// overhangs from the inset with margin.
const BOARD_TICK_CLEARANCE = 12;
// Breathing room ABOVE the measured floor, once the floor is smaller than the
// fraction would reserve. A 1600px Leaderboard tab at BOARD_GUTTER_FRACTION
// reserved 640px for a ~200px label block -- 440px of dead plot -- while the
// home panel's browser-checked floor (Task 4, 2026-08-19) came out within
// 0-36.2px of its own fraction-driven gutter at every desktop width measured
// (1280: 0px over the floor: it already bound; 1440: 9px; 1920: 36.2px). 36
// sits just under that observed ceiling, so every one of those three widths
// still lands within a fraction of a px of what a browser already confirmed
// looked right, while the tab's floor+slack (~240px) still gives back most of
// the 440px it used to waste. See boardFrameLayout.
const BOARD_GUTTER_SLACK = 36;
// The two gaps inside the "dot name pill" block: after the leading dot, and
// after the name. `boardLabelBlockWidth` (the measure) and the draw hook
// inside `createEndpointLabelPlugin` (~1400 lines apart) each total this block
// from scratch -- named constants are what keeps the two arithmetic copies
// from drifting the way two bare literals silently could.
const BOARD_DOT_GAP = 4;
const BOARD_NAME_GAP = 6;
const BOARD_PILL_PAD_X = 5;
const BOARD_PILL_HEIGHT = 15;
const BOARD_DOT_RADIUS = 3;
const BOARD_STUB_LENGTH = 7;
// Comfortable vertical spacing between stacked labels, and the floor below
// which the frame gives up and reserves the arrow alone.
//
// THE FLOOR IS GEOMETRIC, NOT TYPOGRAPHIC, and it is DERIVED for that reason.
// It was 13 against a 15px pill, so at any pitch in [13, 15) the guard passed
// -- it was reading "is 11px text still separable?" -- while consecutive
// FILLED colour pills overlapped by up to 2px. That is not a legibility
// judgement anyone can tune: two opaque rounded rects at a pitch below their
// own height intersect, and no font size changes it. Screen 0 reached it for
// real (9 series into `clamp(140px, 26vh, 280px)` is a 152-168px canvas at a
// 585-645px viewport), as does this tab once the curated roster hits 14 at the
// mobile height. Deriving it from BOARD_PILL_HEIGHT is what stops the two
// numbers being edited apart again; the +1 is a hairline so pills never touch.
const BOARD_LABEL_GAP_MAX = 20;
const BOARD_LABEL_GAP_MIN = BOARD_PILL_HEIGHT + 1;
// Only draw a leader line once collision-avoidance has displaced a label far
// enough that the connection is genuinely ambiguous; below this it is a stub.
const BOARD_LEADER_MIN_DISPLACEMENT = 7;
// Reserved right padding when the frame declines to draw labels: enough for the
// arrowhead and nothing else.
const BOARD_ARROW_PAD = 18;
const BOARD_ARROW_HEAD_LENGTH = 8;
const BOARD_ARROW_HEAD_HALF = 4;
// FIRST-FRAME allowance for the x-axis, subtracted from canvas height to
// estimate plot height. Needed in `beforeLayout`, where chartArea is exactly
// what has not been computed yet. See boardXAxisHeight: from the second update
// onwards the real number is read off the scale, and this is only the estimate
// that stands in before any layout has happened.
const BOARD_XAXIS_ALLOWANCE = 34;
const BOARD_AXIS_COLOR = 'rgba(148, 163, 184, 0.45)';

/** The x-axis strip's real height, or a conservative stand-in before layout.
 *
 *  `beforeLayout` runs at the one moment chartArea does not exist yet, which
 *  is why this began as a bare literal. But 34 was a guess and the tab's axis
 *  renders at 20.4px, so it threw away 14px of stacking room on every frame --
 *  and that is the height the gap divisor and the fits-the-canvas guard both
 *  spend, so the frame gave up on labels earlier than its own geometry
 *  required. Chart.js leaves the previous layout's scale on the chart, so from
 *  the second update onwards the true value is right here; the constant is the
 *  first frame's estimate and nothing more. Reading it also means a change to
 *  the tick font moves this number by itself, rather than leaving a literal
 *  quietly wrong the way `BoardPreview.tsx`'s `width={56}` was. */
function boardXAxisHeight(chart) {
  const h = chart && chart.scales && chart.scales.x && chart.scales.x.height;
  return Number.isFinite(h) ? h : BOARD_XAXIS_ALLOWANCE;
}

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
  const { r, g, b } = hexToRgb(hex);
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
      BOARD_DOT_GAP +
      ctx.measureText(lab.name).width +
      BOARD_NAME_GAP +
      ctx.measureText(lab.value).width +
      BOARD_PILL_PAD_X * 2;
    if (block > widest) widest = block;
  });
  ctx.restore();
  return (
    BOARD_GUTTER_TEXT_INSET + widest + BOARD_TICK_CLEARANCE + BOARD_GUTTER_TRAILING_PAD
  );
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
  const usableHeight = chart.height - boardXAxisHeight(chart);
  const gap = Math.min(BOARD_LABEL_GAP_MAX, usableHeight / labels.length);
  if (gap < BOARD_LABEL_GAP_MIN) return none;
  // ...and the stack that pitch produces must FIT THE CANVAS. The line above
  // sizes a gap; this is the extent of n pills at it -- (n-1) gaps plus one
  // pill's height -- against the band boardStackLabels is given. Checked here
  // so the layout hook refuses on the same geometry the draw hook would, on a
  // set that is a SUPERSET of what the draw hook will lay out (see
  // boardLayoutLabels) -- which makes this verdict conservative rather than
  // identical: the frame can reserve a gutter and draw fewer labels in it, but
  // it can never reserve too little and clip one.
  // With today's constants it cannot fire; it is here so that raising
  // BOARD_LABEL_GAP_MAX or shrinking BOARD_XAXIS_ALLOWANCE degrades to
  // arrow-only instead of silently reintroducing a clipped label.
  if ((labels.length - 1) * gap + BOARD_PILL_HEIGHT > chart.height) return none;
  const floor = boardLabelBlockWidth(chart, labels);
  if (floor > chart.width * BOARD_GUTTER_MAX_FRACTION) return none;
  // `fraction` is a CEILING on the floor, not the gutter's width. A wide board
  // (the 1600px Leaderboard tab) has `width * fraction` far above what the
  // labels measure, and reserving that much left ~440px of the plot rendering
  // as an empty column; a narrow one (the ~400px home panel) has the floor
  // already at or past the fraction, where it was binding correctly all
  // along. `Math.max(floor, ...)` is therefore LOAD-BEARING, not defensive:
  // `floor` is the hard lower bound clipped label text would otherwise
  // require, and it must survive the min() clamp below untouched. The
  // genuinely-impossible case -- floor itself past BOARD_GUTTER_MAX_FRACTION
  // -- is the `return none` above; this line never has to refuse on its own.
  const room = Math.max(floor, Math.min(chart.width * fraction, floor + BOARD_GUTTER_SLACK));
  return { gutter: room, drawLabels: true, gap };
}

function getTeamColor(stableId) {
  const key = String(stableId);
  if (!teamColorMap[key]) {
    const idx = Object.keys(teamColorMap).length % TEAM_COLOR_PALETTE.length;
    teamColorMap[key] = TEAM_COLOR_PALETTE[idx];
  }
  return teamColorMap[key];
}

function getSeriesStyle(label, entry) {
  if (isModelEntry(entry)) {
    return { color: getModelColor(entry?.entry_id || label), kind: 'model', dash: [] };
  }
  const preset = LEADERBOARD_STYLES[label];
  if (preset) return { ...preset };
  return { color: getTeamColor(entry?.entry_id || label), kind: 'team', dash: [] };
}

function getEntryKind(entry) {
  if (entry.entry_type && entry.entry_type !== 'baseline') return 'team';
  if (isModelEntry(entry)) return 'model';
  const label = entry.model || entry.team_name;
  const preset = LEADERBOARD_STYLES[label];
  return preset ? preset.kind : 'strategy';
}

/** Map series kind → chart-picker group id. */
function getFilterCategory(entry) {
  const kind = getEntryKind(entry);
  if (kind === 'model' || kind === 'team') return 'model';
  if (kind === 'benchmark') return 'index';
  return 'baseline'; // strategy baselines
}

const CURVE_PICKER_GROUPS = [
  { id: 'model', title: 'Models' },
  { id: 'baseline', title: 'Baseline Strategies' },
  { id: 'index', title: 'Market Indices' },
];

function entrySeriesLabel(entry) {
  return entry.model || entry.team_name || entry.entry_id || '';
}

function getCurvePickerGroups(entries) {
  const buckets = { model: [], baseline: [], index: [] };
  (entries || []).forEach((entry) => {
    const cat = getFilterCategory(entry);
    if (buckets[cat]) buckets[cat].push(entry);
  });
  // Models: sort by return desc for a familiar board order
  buckets.model.sort(
    (a, b) => (Number(b.cumulative_return) || 0) - (Number(a.cumulative_return) || 0)
  );
  return CURVE_PICKER_GROUPS.map((g) => ({
    ...g,
    entries: buckets[g.id] || [],
  })).filter((g) => g.entries.length > 0);
}

function seriesVisible(label) {
  return !hiddenSeries.has(label);
}

function countVisibleSeries(entries) {
  return (entries || []).filter((e) => seriesVisible(entrySeriesLabel(e))).length;
}

function setGroupVisibility(groupEntries, visible) {
  groupEntries.forEach((entry) => {
    const label = entrySeriesLabel(entry);
    if (!label) return;
    if (visible) hiddenSeries.delete(label);
    else hiddenSeries.add(label);
  });
}

function groupCheckState(groupEntries) {
  const total = groupEntries.length;
  if (!total) return 'none';
  const visible = groupEntries.filter((e) => seriesVisible(entrySeriesLabel(e))).length;
  if (visible === 0) return 'none';
  if (visible === total) return 'all';
  return 'partial';
}

function updateCurvePickerCount() {
  const el = document.getElementById('curvePickerCount');
  if (!el) return;
  const entries = leaderboardPayload?.entries || [];
  const n = countVisibleSeries(entries);
  const total = entries.length;
  el.textContent = total ? `${n} selected` : '0 selected';
}

function renderCurvePicker() {
  const body = document.getElementById('curvePickerBody');
  if (!body) return;
  const groups = getCurvePickerGroups(leaderboardPayload?.entries || []);

  body.innerHTML = groups.map((group) => {
    const state = groupCheckState(group.entries);
    const visibleN = group.entries.filter((e) => seriesVisible(entrySeriesLabel(e))).length;
    const expanded = curvePickerExpanded.has(group.id);
    const checkClass =
      state === 'all' ? 'is-checked' : state === 'partial' ? 'is-partial' : 'is-unchecked';
    // Always show selected/total so collapsed groups aren't mistaken for "all".
    const countLabel = `${visibleN}/${group.entries.length}`;

    const children = group.entries.map((entry, idx) => {
      const label = entrySeriesLabel(entry);
      const on = seriesVisible(label);
      const safeName = String(label)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/"/g, '&quot;');
      return `
        <label class="curve-picker-item">
          <input type="checkbox" data-group-id="${group.id}" data-entry-idx="${idx}" ${on ? 'checked' : ''} />
          <span class="curve-picker-item-name">${safeName}</span>
        </label>`;
    }).join('');

    return `
      <div class="curve-picker-group" data-group="${group.id}">
        <div class="curve-picker-group-row">
          <button type="button" class="curve-picker-group-check ${checkClass}"
            data-group-toggle="${group.id}" aria-label="Toggle ${group.title}"
            aria-checked="${state === 'all' ? 'true' : state === 'none' ? 'false' : 'mixed'}"></button>
          <button type="button" class="curve-picker-group-expand" data-group-expand="${group.id}"
            aria-expanded="${expanded ? 'true' : 'false'}">
            <span class="curve-picker-group-title">${group.title} (${countLabel})</span>
            <span class="curve-picker-group-chevron" aria-hidden="true">${expanded ? '▾' : '›'}</span>
          </button>
        </div>
        <div class="curve-picker-children${expanded ? '' : ' is-collapsed'}">${children}</div>
      </div>`;
  }).join('');

  updateCurvePickerCount();
}

function applyChartVisibilityChange() {
  updateCurvePickerCount();
  const open = document.getElementById('curvePickerTrigger')?.getAttribute('aria-expanded') === 'true';
  if (open) renderCurvePicker();
  else updateCurvePickerCount();
  renderEquityCurvesChart();
}

function setCurvePickerOpen(open) {
  const menu = document.getElementById('curvePickerMenu');
  const trigger = document.getElementById('curvePickerTrigger');
  const root = document.getElementById('curvePicker');
  if (!menu || !trigger) return;
  menu.hidden = !open;
  trigger.setAttribute('aria-expanded', open ? 'true' : 'false');
  root?.classList.toggle('is-open', open);
  if (open) renderCurvePicker();
}

// A season is two calendar weeks of US cash sessions, Monday through Friday.
const SEASON_TRADING_DAYS = 10;

// The season the board is in before the engine ships. Season 0 is the shakedown
// season by convention: numbered, so the board has a real identity to show and
// so Season 1 means "the first one that counted", but explicitly the one whose
// results nobody should read as a standing. The preview banner says the rest.
const PREVIEW_SEASON_NUMBER = 0;

/** The board on screen, from the request that produced the payload being shown. */
function isLiveBoard() {
  return renderedBoardPeriod === 'live';
}

/** True when a season has actually advanced at least one session.
 *
 * The anchor under every preview disclaimer on this tab, and deliberately not
 * `payload.period !== 'live'`. That tests whether the *backend recognises the
 * word* — so adding "live" to VALID_PERIODS, the natural first commit of the
 * season engine and one that needs no season payload at all, would clear the
 * banner, flip the badge to "Running" and promise a nightly advance while no
 * season had ever run. What the banner claims is that nothing here advanced, so
 * that is what it tests: a date the engine can only write after a real advance.
 *
 * That commit has since landed: `VALID_PERIODS` is now
 * `("contest", "daily", "live")` and the server sends a real `season` block. It
 * changed nothing here, which was the point — the block hardcodes the
 * not-yet-advanced state, so this still returns false and the banner still
 * shows. Do not now "simplify" this to a period check on the grounds that the
 * period is finally real; it is real and still says nothing about an advance.
 */
function seasonHasAdvanced(payload) {
  const season = (payload || {}).season;
  if (!season) return false;
  if (season.last_advanced_date) return true;
  return Number(season.trading_days_elapsed) > 0;
}

/** True when the Live Trading tab is rendering something that is not a season.
 *
 * The server now answers `?period=live` with `period: 'live'` and a season
 * block, so the old reason — an unknown period silently coerced to 'contest',
 * arriving as a perfectly successful HTTP 200 — no longer applies. The check
 * outlives it, because the *shape* it guards did not change: the season block
 * is hardcoded to the not-yet-advanced state and every other element on the tab
 * (chart, table, curve picker, rankings) renders identically whether or not a
 * season ran. Without this, "the season engine is not deployed" and "here are
 * the live standings" still render byte-identically — the exact failure shape
 * CLAUDE.md's fail-closed-is-not-fail-visible section is about.
 */
function isLivePreview(payload) {
  return isLiveBoard() && !seasonHasAdvanced(payload);
}

/** The season number to show, or null when there is genuinely none.
 *
 * Season 0 is a real season here, so the number can never be tested for
 * truthiness: `season?.number ? ... : '—'` renders the shakedown season as
 * "no season at all". Every read of the number goes through this.
 */
function displayedSeasonNumber(payload) {
  const n = Number((payload || {}).season?.number);
  if (Number.isFinite(n)) return n;
  return isLivePreview(payload) ? PREVIEW_SEASON_NUMBER : null;
}

/** "Season 0" / "Season 3" / "This season" — the number rendered as prose.
 *
 * Every sentence on this tab that names the season goes through here, so the
 * server's `season.number` is the single owner of it. Interpolating
 * PREVIEW_SEASON_NUMBER directly gave the number two: the badge read the
 * payload while the banner beneath it read this file's constant, so the first
 * act of the advance engine — bumping the server's PREVIEW_SEASON_NUMBER —
 * would have printed "Season 1" above "Season 0 has not been run".
 *
 * `null` (no season at all) becomes "This season" rather than "Season null",
 * and the sentences are written to read correctly either way.
 */
function displayedSeasonLabel(payload) {
  const n = displayedSeasonNumber(payload);
  return n === null ? 'This season' : `Season ${n}`;
}

/** A relative phrase, or null when the timestamp is absent or unparseable.
 *
 * Null rather than '': both call sites interpolate this into a label, so an
 * empty string renders as a dangling "Next advance " or "closes " — a caption
 * with its value silently deleted, which reads as a broken template rather than
 * as missing data. Callers drop the whole clause on null instead.
 *
 * Past timestamps say "now", not "due now": the phrase is composed into
 * "closes …" as well as "Next advance …", and an entry window cannot be
 * "closes due now".
 */
function formatRelativeFromNow(iso) {
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return null;
  const minutes = Math.round((then - Date.now()) / 60000);
  if (minutes <= 0) return 'now';
  if (minutes < 60) return `in ${minutes} min`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `in ${hours}h`;
  return `in ${Math.round(hours / 24)}d`;
}

/** Repaint the whole board identity for `payload`, which `board` was asked for.
 *
 * The single entry point of the render path, and the single place the
 * request/response pair is committed: every helper below — plus
 * populateLeaderboardTable, which the sort and filter handlers call long after
 * this returns — reads `renderedBoardPeriod`, so it must be written here from
 * the response's own request rather than by whoever fetched most recently.
 */
function updateLeaderboardHeader(payload, board = renderedBoardPeriod) {
  renderedBoardPeriod = board;
  payload = payload || {};
  const entries = payload.entries || [];
  // Best LLM / provided model by cumulative return (not overall board rank).
  const models = entries
    .filter((e) => isModelEntry(e))
    .slice()
    .sort((a, b) => (Number(b.cumulative_return) || 0) - (Number(a.cumulative_return) || 0));

  const totalEl = document.getElementById('totalTeams');
  const windowEl = document.getElementById('tradingWindow');
  const updatedEl = document.getElementById('lastUpdate');
  const leaderEl = document.getElementById('leaderTeam');
  const standingsEl = document.getElementById('leaderboardStandingsTitle');
  // By id, not `querySelector('.contest-subtitle')`: there are two, and the
  // positional selector silently retargets if either is ever reordered.
  const subtitleEl = document.getElementById('boardSubtitle');
  const titleEl = document.querySelector('#leaderboardView .contest-title');
  const badgeEl = document.querySelector('#leaderboardView .contest-live-badge');

  const season = payload.season || null;
  const liveBoard = isLiveBoard();

  if (titleEl) {
    // The board names itself; the season strip below names the season. Folding
    // the season number in here instead would leave the board unnamed on screen
    // the moment a season is running.
    titleEl.textContent = liveBoard
      ? 'Live Trading Leaderboard'
      : 'SecureFinAI Contest 2026';
  }
  if (badgeEl) {
    const state = liveBoard
      ? (isLivePreview(payload) ? 'preview' : (season?.status || 'upcoming'))
      : 'upcoming';
    badgeEl.textContent = {
      preview: 'Preview',
      running: 'Running',
      upcoming: 'Upcoming',
      closed: 'Closed',
    }[state] || 'Upcoming';
    badgeEl.className = `status-badge contest-live-badge ${state}`;
  }

  // Competition-only chrome, hidden rather than left standing. Both name the
  // SecureFinAI contest specifically — an organizing body, and a rules document
  // written around a fixed window and a registration deadline — so on a season
  // board they caption the wrong event, and the Rules button opens a modal that
  // describes rules this board does not run under.
  const organizerEl = document.getElementById('contestOrganizerLine');
  if (organizerEl) organizerEl.hidden = liveBoard;
  const rulesBtn = document.getElementById('competitionRulesBtn');
  if (rulesBtn) rulesBtn.hidden = liveBoard;

  // The stat is a label/value pair, so the label has to move with the value:
  // leaving it as "Phase" captions a season number with the contest's noun, and
  // making the value carry the word instead reads "Season: Season 0".
  const phaseLabelEl = document.getElementById('boardPhaseLabel');
  if (phaseLabelEl) phaseLabelEl.textContent = liveBoard ? 'Season' : 'Phase';
  if (totalEl) {
    const seasonNo = displayedSeasonNumber(payload);
    totalEl.textContent = liveBoard
      ? (seasonNo === null ? '—' : String(seasonNo))
      : (payload.phase_label || 'Preseason');
  }
  if (windowEl) windowEl.textContent = payload.window?.label || '—';
  if (updatedEl) {
    updatedEl.textContent = payload.updated_at
      ? new Date(payload.updated_at).toLocaleString()
      : '—';
  }
  if (leaderEl) {
    const top = models[0];
    leaderEl.textContent = top
      ? (top.model || top.team_name)
      : (payload.leader || '—');
  }
  if (standingsEl) {
    standingsEl.textContent = payload.standings_label || 'Ranking';
  }
  if (subtitleEl) {
    subtitleEl.textContent = liveBoard
      ? formatLiveBoardSubtitle(payload)
      : 'Sep 1 – Oct 30, 2026';
  }
  renderLivePreviewBanner(payload);
  renderSeasonStrip(payload);
  renderSeasonGaps(payload);
  updateCurvePickerCount();
}

function formatLiveBoardSubtitle(payload) {
  const season = payload.season || null;
  const dates = season?.start_date && season?.end_date
    ? `${formatShortDate(season.start_date)} – ${formatShortDate(season.end_date)} · `
    : '';
  if (isLivePreview(payload)) {
    // Stating the cadence here would promise a nightly advance that no deployed
    // job performs. The banner above carries the full explanation; this is the
    // one-line version that has to survive next to it.
    return `${dates}${displayedSeasonLabel(payload)} preview — no advance has run`;
  }
  // `window.start_date` is the board's display window, NOT a record that an
  // advance ran, and falling back to it printed the Competition window's first
  // day as "last completed" on a board that had never advanced once. The season
  // payload contract defines exactly one field that only a real advance can
  // write, so that is the one read here. (`daily_status.trading_date` was the
  // retired Daily board's shape; the server never sends it for this board.)
  const advanced = season?.last_advanced_date;
  const cadence = 'Advances nightly after the 16:00 ET cash session';
  return advanced
    ? `${dates}${cadence} · last completed ${advanced}`
    : `${dates}${cadence}`;
}

/** The load-bearing honesty control for this tab.
 *
 * Everything else on the board renders identically whether or not a season ran,
 * because the entry/curve/table shapes are shared with the Competition board.
 * This banner is the only element that distinguishes them, so it is written
 * first and deleted last.
 */
function renderLivePreviewBanner(payload) {
  const host = document.getElementById('seasonPreviewBanner');
  if (!host) return;
  host.textContent = '';
  if (!isLivePreview(payload)) {
    host.hidden = true;
    return;
  }
  host.hidden = false;
  const lead = document.createElement('strong');
  lead.textContent = 'Preview — the season engine is not deployed.';
  const body = document.createElement('span');
  const label = payload.window?.label || 'the Competition window';
  body.textContent = ` ${displayedSeasonLabel(payload)} has not been run. The curves below are the Competition board's fixed window (${label}), shown so this layout can be reviewed: nothing here is a live standing, and no ranking on this tab counts.`;
  host.append(lead, body);
}

function renderSeasonStrip(payload) {
  const host = document.getElementById('seasonStrip');
  if (!host) return;
  const season = payload.season || null;
  if (!isLiveBoard()) {
    host.hidden = true;
    return;
  }
  host.hidden = false;
  host.classList.toggle('is-placeholder', !season);

  // Both clamped before either is divided by the other. The server clamps
  // `trading_days_total` too, and that is the real fix — but this value reaches
  // `(elapsed / total) * 100` as a CSS width, so a zero or negative one draws a
  // full or backwards bar under the banner that says nothing has advanced.
  const rawTotal = Number(season?.trading_days_total);
  const total = Number.isFinite(rawTotal) && rawTotal >= 1
    ? Math.floor(rawTotal)
    : SEASON_TRADING_DAYS;
  const elapsed = Math.min(Math.max(Number(season?.trading_days_elapsed) || 0, 0), total);

  const badge = document.getElementById('seasonBadge');
  if (badge) {
    const seasonNo = displayedSeasonNumber(payload);
    badge.textContent = seasonNo === null ? 'Season —' : `Season ${seasonNo}`;
  }

  const dates = document.getElementById('seasonDates');
  if (dates) {
    dates.textContent = season?.start_date && season?.end_date
      ? `${formatShortDate(season.start_date)} – ${formatShortDate(season.end_date)}`
      : 'Dates set when the first season opens';
  }

  const entryText = document.getElementById('seasonEntryText');
  const entryState = document.getElementById('seasonEntryState');
  if (entryText && entryState) {
    let state = 'pending';
    let text = 'Entries not open yet';
    if (season?.entries_open) {
      state = 'open';
      // Composed from the formatted value, not from the raw field: an
      // unparseable `entry_closes_at` is present-but-useless, and testing the
      // field instead of the phrase renders a bare "· closes ".
      const closesIn = formatRelativeFromNow(season.entry_closes_at);
      const closes = closesIn ? ` · closes ${closesIn}` : '';
      const count = Number(season.entry_count);
      text = `Entries open${Number.isFinite(count) ? ` · ${count} entered` : ''}${closes}`;
    } else if (season?.status === 'running') {
      state = 'closed';
      text = 'Entries closed — season in progress';
    } else if (season?.status === 'closed') {
      state = 'closed';
      text = 'Season finished';
    }
    entryState.className = `season-entry-state is-${state}`;
    entryText.textContent = text;
  }

  const fill = document.getElementById('seasonProgressFill');
  const bar = document.getElementById('seasonProgressBar');
  const label = document.getElementById('seasonProgressLabel');
  if (fill) fill.style.width = `${total ? (elapsed / total) * 100 : 0}%`;
  if (bar) {
    bar.setAttribute('aria-valuemax', String(total));
    bar.setAttribute('aria-valuenow', String(elapsed));
  }
  if (label) {
    label.textContent = season
      ? `Day ${elapsed} of ${total}`
      : `${total} trading days per season`;
  }

  const advance = document.getElementById('seasonNextAdvance');
  if (advance) {
    // The formatted phrase is the only proof the timestamp was readable, so an
    // unparseable `next_advance_at` is treated as absent. Branching on the raw
    // field instead renders "Next advance null".
    const relative = formatRelativeFromNow(season?.next_advance_at);
    if (isLivePreview(payload)) {
      // Tested ahead of the timestamp rather than after it: a payload carrying
      // a schedule but no completed advance would otherwise announce a nightly
      // job that is not deployed — the one claim this tab exists to deny.
      advance.textContent = 'No advance scheduled';
    } else if (season?.status === 'closed') {
      advance.textContent = 'No further advances';
    } else if (relative) {
      advance.textContent = `Next advance ${relative} (${new Date(season.next_advance_at).toLocaleString()})`;
    } else {
      advance.textContent = 'Next advance: nightly after the 16:00 ET close';
    }
  }
}

// Copy is deliberately different per failure_kind. A gap that reads the same
// whatever caused it is worse than no gap marker at all: it teaches the reader
// that a flat day and a dead job look alike, which is the thing this list exists
// to prevent.
const SEASON_GAP_COPY = {
  market_data_unavailable: 'no usable market data for the session',
  model_error: 'the model returned no usable decision',
  job_not_run: 'the nightly advance never ran',
  budget_exhausted: 'the model budget for the month was exhausted',
};

function renderSeasonGaps(payload) {
  const host = document.getElementById('seasonGaps');
  if (!host) return;
  host.textContent = '';
  const gaps = (isLiveBoard() && payload.season?.gaps) || [];
  if (!gaps.length) {
    host.hidden = true;
    return;
  }
  host.hidden = false;
  gaps.forEach((gap) => {
    const item = document.createElement('li');
    item.className = 'season-gap';
    const date = document.createElement('span');
    date.className = 'season-gap-date';
    date.textContent = formatShortDate(gap.date) || gap.date || 'Unknown date';
    const why = document.createElement('span');
    why.className = 'season-gap-why';
    const reason = SEASON_GAP_COPY[gap.failure_kind] || 'the advance did not complete';
    why.textContent = `${reason} — positions carried forward unchanged, no trading counted`;
    item.append(date, why);
    if (gap.detail) {
      const detail = document.createElement('span');
      detail.className = 'season-gap-detail';
      detail.textContent = gap.detail;
      item.append(detail);
    }
    host.append(item);
  });
}

/** Resolve a caller's board name to one of the two the API actually serves.
 *
 * 'daily' is the retired Daily Leaderboard and 'season' was this tab's working
 * title before the board was named. Both are accepted so a stale deep link or a
 * cached app.js lands on the successor board rather than silently on
 * Competition, which looks like a working link to the wrong data.
 */
function normalizeBoardPeriod(period) {
  return (period === 'live' || period === 'season' || period === 'daily')
    ? 'live'
    : 'contest';
}

async function loadLeaderboardData(period = 'contest') {
  console.log('Loading leaderboard from API...', period);
  const boardPeriod = normalizeBoardPeriod(period);
  // Claimed before the await, re-checked after it. Subtab clicks are user-paced
  // and the API is not, so a slow first request can resolve after a fast second
  // one; without this the loser repaints the winner's board — and because the
  // two boards share every element except the chrome, it repaints it silently.
  const seq = ++boardRequestSeq;

  // Paint the board's identity before the request, not after it. Until the
  // fetch resolves the chrome on screen belongs to whichever board was there
  // last — for a shared ?view=live link that is the static Competition markup:
  // its name, its Sep–Oct dates, its "Upcoming" badge, its Rules button, and
  // the preview banner still hidden. On the free tier a cold start is 30–60s,
  // so that is not a flash, it is the whole first impression. The boot
  // stylesheet only covers up to app.js executing; this covers the rest.
  if (renderedBoardPeriod !== boardPeriod) {
    leaderboardPayload = null;
    updateLeaderboardHeader({}, boardPeriod);
    showLeaderboardTableLoading();
  }

  try {
    const url = `${API_BASE}/api/v1/leaderboard?period=${encodeURIComponent(boardPeriod)}&t=${Date.now()}`;
    const payload = await API.get(url);
    if (seq !== boardRequestSeq) return;
    leaderboardPayload = payload;
    equityCurvesData = buildEquityCurvesFromEntries(payload.entries || []);

    // Reset chart visibility when switching boards so season/contest don't share hide state.
    hiddenSeries = new Set();
    hiddenInitialized = true;

    updateLeaderboardHeader(payload, boardPeriod);
    populateLeaderboardTable();
    renderCurvePicker();

    if (!leaderboardListenersInitialized) {
      initLeaderboardListeners();
      leaderboardListenersInitialized = true;
    }

    await renderEquityCurvesChart();
  } catch (error) {
    if (seq !== boardRequestSeq) return;
    console.error('Error loading leaderboard:', error);
    // The board is torn down on the way out rather than left as it was. A
    // failed switch used to keep the previous board's identity on screen: the
    // preview banner stranded over Competition, or — the direction that
    // matters — the Competition title and curves still showing while the user
    // sits on the Live tab, with no preview banner anywhere. Free-tier cold
    // starts make a failed switch an ordinary path, not a corner case.
    leaderboardPayload = null;
    equityCurvesData = null;
    if (equityCurvesChartInstance) {
      equityCurvesChartInstance.destroy();
      equityCurvesChartInstance = null;
    }
    updateLeaderboardHeader({}, boardPeriod);
    renderCurvePicker();
    displayLeaderboardError(error.message);
  }
}

function initLeaderboardListeners() {
  const trigger = document.getElementById('curvePickerTrigger');
  const menu = document.getElementById('curvePickerMenu');
  const body = document.getElementById('curvePickerBody');
  const clearBtn = document.getElementById('curvePickerClear');

  // Keep the menu open while interacting: stop bubbles, and close only on
  // outside pointerdown (avoids the classic "rerender detaches target →
  // document click thinks it's outside" bug).
  menu?.addEventListener('click', (e) => e.stopPropagation());
  menu?.addEventListener('pointerdown', (e) => e.stopPropagation());

  trigger?.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = trigger.getAttribute('aria-expanded') === 'true';
    setCurvePickerOpen(!open);
  });

  clearBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    const entries = leaderboardPayload?.entries || [];
    entries.forEach((entry) => {
      const label = entrySeriesLabel(entry);
      if (label) hiddenSeries.add(label);
    });
    applyChartVisibilityChange();
  });

  body?.addEventListener('click', (e) => {
    e.stopPropagation();
    const expandBtn = e.target.closest('[data-group-expand]');
    if (expandBtn) {
      e.preventDefault();
      const id = expandBtn.dataset.groupExpand;
      if (curvePickerExpanded.has(id)) curvePickerExpanded.delete(id);
      else curvePickerExpanded.add(id);
      renderCurvePicker();
      return;
    }

    const groupToggle = e.target.closest('[data-group-toggle]');
    if (groupToggle) {
      e.preventDefault();
      const id = groupToggle.dataset.groupToggle;
      const group = getCurvePickerGroups(leaderboardPayload?.entries || [])
        .find((g) => g.id === id);
      if (!group) return;
      const state = groupCheckState(group.entries);
      // All on → turn off; partial or none → turn all on.
      setGroupVisibility(group.entries, state !== 'all');
      applyChartVisibilityChange();
    }
  });

  body?.addEventListener('change', (e) => {
    e.stopPropagation();
    const input = e.target;
    if (!(input instanceof HTMLInputElement) || input.dataset.entryIdx == null) return;
    const groupId = input.dataset.groupId;
    const idx = Number(input.dataset.entryIdx);
    const group = getCurvePickerGroups(leaderboardPayload?.entries || [])
      .find((g) => g.id === groupId);
    const entry = group?.entries?.[idx];
    if (!entry) return;
    const label = entrySeriesLabel(entry);
    if (input.checked) hiddenSeries.delete(label);
    else hiddenSeries.add(label);
    applyChartVisibilityChange();
  });

  if (!curvePickerOutsideBound) {
    document.addEventListener('pointerdown', (e) => {
      const root = document.getElementById('curvePicker');
      if (!root?.classList.contains('is-open')) return;
      if (e.target instanceof Node && root.contains(e.target)) return;
      setCurvePickerOpen(false);
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') setCurvePickerOpen(false);
    });
    curvePickerOutsideBound = true;
  }

  document.querySelectorAll('.leaderboard-table th.sortable').forEach((th) => {
    th.addEventListener('click', () => {
      const key = th.dataset.sort;
      if (!key) return;
      if (currentLeaderboardSort === key) {
        currentLeaderboardSortDir = currentLeaderboardSortDir === 'asc' ? 'desc' : 'asc';
      } else {
        currentLeaderboardSort = key;
        // Rank: 1 first. Value/Return/Sharpe: high first. Max DD: low first.
        currentLeaderboardSortDir = key === 'dd' || key === 'rank' ? 'asc' : 'desc';
      }
      updateLeaderboardSortHeaders();
      populateLeaderboardTable();
    });
  });
  updateLeaderboardSortHeaders();

  document.querySelectorAll('.view-toggle-btn').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      document.querySelectorAll('.view-toggle-btn').forEach((b) => b.classList.remove('active'));
      e.target.classList.add('active');
      currentChartView = e.target.dataset.view === 'absolute' ? 'absolute' : 'cumulative';
      await renderEquityCurvesChart();
    });
  });

  document.querySelectorAll('.chart-view-btn').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      document.querySelectorAll('.chart-view-btn').forEach((b) => b.classList.remove('active'));
      e.target.classList.add('active');
      currentChartView = e.target.dataset.view === 'absolute' ? 'absolute' : 'cumulative';
      await renderEquityCurvesChart();
    });
  });
}

function formatLeaderboardNumber(num) {
  return Number(num || 0).toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function formatShortDate(isoDay) {
  if (!isoDay) return '';
  const d = new Date(String(isoDay).includes('T') ? isoDay : `${isoDay}T00:00:00`);
  if (Number.isNaN(d.getTime())) return isoDay;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function formatChartTooltipLabel(ts) {
  if (!ts) return '';
  const raw = String(ts);
  const d = new Date(raw.includes('T') ? raw : `${raw}T00:00:00`);
  if (Number.isNaN(d.getTime())) return raw;
  // Hourly series: show date + hour so the open tick and intraday points differ.
  if (raw.includes('T')) {
    return d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  }
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

/** Normalize equity timestamps to hour precision for shared-axis alignment. */
function chartTimeKey(ts) {
  const s = String(ts || '');
  if (!s) return '';
  // 2026-04-15T14:00:00+00:00 → 2026-04-15T14:00
  if (s.length >= 16 && s[10] === 'T') return s.slice(0, 16);
  if (s.length >= 10) return s.slice(0, 10);
  return s;
}

function buildEquityCurvesFromEntries(entries) {
  // Align every series on a shared, sorted hourly axis rather than by
  // positional index or calendar-day bucket, so the open tick and each market
  // hour render at their true x-positions.
  const timeSet = new Set();
  const perEntry = [];

  entries.forEach((entry) => {
    const points = entry.equity_curve || [];
    if (!points.length) return;
    const byTime = {};
    points.forEach((pt) => {
      const key = chartTimeKey(pt.timestamp);
      if (!key) return;
      byTime[key] = Number(pt.equity) || 0;
      timeSet.add(key);
    });
    perEntry.push({ entry, seriesLabel: entry.model || entry.team_name, byTime });
  });

  const times = Array.from(timeSet).sort();
  const curves = {};
  const trajectories = {};
  const initials = {};

  perEntry.forEach(({ entry, seriesLabel, byTime }) => {
    const values = times.map((t) => (t in byTime ? byTime[t] : null));
    const firstReal = values.find((v) => v != null);
    curves[seriesLabel] = values;
    initials[seriesLabel] = Number(entry.initial_equity) || firstReal || 10000;
    trajectories[seriesLabel] = getSeriesStyle(seriesLabel, entry);
  });

  // Keep `days` alias so any older callers still unpack a familiar key.
  return { times, days: times, curves, trajectories, initials };
}

// Default chart visibility: top 5 teams + benchmarks. Strategy baselines are
// hidden by default (selectable via legend) — unless there are no teams yet,
// in which case we show them so the chart isn't just two gray lines.
function computeDefaultHidden(entries) {
  const hidden = new Set();
  const teams = entries.filter((e) => getEntryKind(e) === 'team');
  const hasTeams = teams.length > 0;
  const visibleTeamIds = new Set(teams.slice(0, 5).map((e) => e.entry_id));

  entries.forEach((entry) => {
    const label = entry.model || entry.team_name;
    const kind = getEntryKind(entry);
    if (kind === 'team' && !visibleTeamIds.has(entry.entry_id)) hidden.add(label);
    if (kind === 'strategy' && hasTeams) hidden.add(label);
  });
  return hidden;
}

function updateLeaderboardSortHeaders() {
  document.querySelectorAll('.leaderboard-table th.sortable').forEach((th) => {
    const active = th.dataset.sort === currentLeaderboardSort;
    th.classList.toggle('is-sorted', active);
    th.setAttribute('aria-sort', active
      ? (currentLeaderboardSortDir === 'asc' ? 'ascending' : 'descending')
      : 'none');
    const arrow = th.querySelector('.sort-arrow');
    if (arrow) {
      arrow.textContent = active ? (currentLeaderboardSortDir === 'asc' ? '↑' : '↓') : '';
    }
  });
}

function getFilteredLeaderboardEntries() {
  // Official rankings table always lists every entry; chart visibility is separate.
  // Left Rank stays the official portfolio-value rank; this only reorders rows.
  const entries = (leaderboardPayload?.entries || []).slice();
  const dir = currentLeaderboardSortDir === 'asc' ? 1 : -1;
  const num = (v) => Number(v) || 0;
  entries.sort((a, b) => {
    let cmp = 0;
    switch (currentLeaderboardSort) {
      case 'rank':
        cmp = num(a.rank) - num(b.rank);
        break;
      case 'value':
        cmp = num(a.portfolio_value) - num(b.portfolio_value);
        break;
      case 'return':
        cmp = num(a.cumulative_return) - num(b.cumulative_return);
        break;
      case 'sharpe':
        cmp = num(a.sharpe_ratio) - num(b.sharpe_ratio);
        break;
      case 'dd':
        cmp = Math.abs(num(a.max_drawdown)) - Math.abs(num(b.max_drawdown));
        break;
      default:
        cmp = num(a.rank) - num(b.rank);
    }
    return cmp * dir;
  });
  return entries;
}

// `entry.model` / `entry.team_name` are user-registered agent names, so every
// string field must go through `escapeHtml` (a global from app.js, loaded
// first). The onclick id additionally needs JS-string escaping — backslash
// before quote, or a trailing "\" would un-escape the closing quote — and the
// JS-escaped result is then HTML-escaped for the attribute context.
function renderLeaderboardRowHtml(entry) {
  const safeId = escapeHtml(
    String(entry.entry_id || entry.team_name).replace(/\\/g, '\\\\').replace(/'/g, "\\'")
  );
  const entryLabel = escapeHtml(entry.model || entry.team_name || '—');
  const ret = Number(entry.cumulative_return || 0);
  const retClass = ret >= 0 ? 'return-positive' : 'return-negative';
  const ddRaw = Number(entry.max_drawdown || 0);
  const dd = (Math.abs(ddRaw) * 100).toFixed(2);

  return `
      <tr onclick="selectLeaderboardTeam('${safeId}')">
        <td class="rank-cell">${escapeHtml(entry.rank)}</td>
        <td>
          <div class="team-name-badge">
            <span>${entryLabel}</span>
            <span class="team-badge">${escapeHtml(formatEntryBadge(entry.team_badge))}</span>
          </div>
        </td>
        <td style="text-align: right; font-family: var(--font-mono);">$${formatLeaderboardNumber(entry.portfolio_value)}</td>
        <td style="text-align: right;" class="${retClass}">
          <span class="metric-value-text">${(ret * 100).toFixed(2)}%</span>
        </td>
        <td style="text-align: right; font-family: var(--font-mono);">${Number(entry.sharpe_ratio || 0).toFixed(2)}</td>
        <td style="text-align: right; font-family: var(--font-mono);">${dd}%</td>
      </tr>
    `;
}

function populateLeaderboardTable() {
  const tbody = document.getElementById('leaderboardTableBody');
  if (!tbody) return;

  const filtered = getFilteredLeaderboardEntries();
  if (!filtered.length) {
    // Keyed on the board asked for, not the one returned. The server now
    // echoes `period: 'live'` so the two agree today, but this tab is defined
    // by what the user clicked: a coerced or fallback response explaining the
    // season tab as the contest board is how an empty season stops being
    // legible as one.
    const msg = isLiveBoard()
      ? 'No entries in this season yet. Baselines compute on first load; competition models advance via the nightly job.'
      : 'No leaderboard entries yet. Baselines compute on first load (requires market data).';
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--text-secondary);">${msg}</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(renderLeaderboardRowHtml).join('');
}

function renderLeaderboardDetailHtml(entry, totalEntries) {
  const ret = Number(entry.cumulative_return || 0);
  const retColor = ret >= 0 ? 'var(--success-color)' : 'var(--danger-color)';
  const entryLabel = escapeHtml(entry.model || entry.team_name || '—');
  return `
      <div class="team-detail-row">
        <span class="team-detail-label">Entry</span>
        <span class="team-detail-value">${entryLabel}</span>
      </div>
      <div class="team-detail-row">
        <span class="team-detail-label">Type</span>
        <span class="team-detail-value">${escapeHtml(formatEntryBadge(entry.team_badge || (entry.entry_type === 'baseline' ? 'Baseline Strategy' : 'Agent')))}</span>
      </div>
      <div class="team-detail-row">
        <span class="team-detail-label">Value</span>
        <span class="team-detail-value">$${formatLeaderboardNumber(entry.portfolio_value)}</span>
      </div>
      <div class="team-detail-row">
        <span class="team-detail-label">Return</span>
        <span class="team-detail-value" style="color: ${retColor};">${(ret * 100).toFixed(2)}%</span>
      </div>
      <div class="team-detail-row">
        <span class="team-detail-label">Sharpe</span>
        <span class="team-detail-value">${Number(entry.sharpe_ratio || 0).toFixed(2)}</span>
      </div>
      <div class="team-detail-row">
        <span class="team-detail-label">Max Drawdown</span>
        <span class="team-detail-value">${(Math.abs(Number(entry.max_drawdown || 0)) * 100).toFixed(2)}%</span>
      </div>
      <div class="team-detail-row">
        <span class="team-detail-label">Rank</span>
        <span class="team-detail-value">${escapeHtml(entry.rank)} / ${escapeHtml(totalEntries || '—')}</span>
      </div>
    `;
}

function selectLeaderboardTeam(entryId) {
  const entries = leaderboardPayload?.entries || [];
  selectedLeaderboardEntry =
    entries.find((e) => String(e.entry_id) === String(entryId)) ||
    entries.find((e) => e.team_name === entryId);
  if (!selectedLeaderboardEntry) return;

  const entry = selectedLeaderboardEntry;
  const detailPanel = document.getElementById('selectedTeamDetail');
  if (detailPanel) {
    detailPanel.innerHTML = renderLeaderboardDetailHtml(entry, leaderboardPayload?.total_entries);
  }

  // Selecting an entry also forces it visible and re-emphasizes it on the chart.
  const label = entry.model || entry.team_name;
  hiddenSeries.delete(label);
  renderEquityCurvesChart();
}

function getEmphasisLabel() {
  // Only an explicit row/detail selection stays emphasized when the pointer is
  // idle. Do not auto-emphasize the current leader — idle view shows the whole
  // figure with every visible curve at its kind weight.
  if (selectedLeaderboardEntry) {
    return selectedLeaderboardEntry.model || selectedLeaderboardEntry.team_name;
  }
  return null;
}

/**
 * The curve under the pointer at canvas pixel (x, y), or null.
 *
 * Distance is measured to the *rendered line* — `LineElement.interpolate` gives
 * the curve's y at the pointer's x — rather than to the nearest data point. A
 * point-distance test only works on a dense series: the Daily board plots 8
 * points per curve (~97px apart at typical widths), so a 16px radius around the
 * points rejects roughly two thirds of every segment while the cursor is
 * sitting directly on the line.
 */
function resolveHoverTarget(chart, x, y) {
  if (x == null || y == null) return null;
  const area = chart.chartArea;
  if (!area) return null;
  if (x < area.left || x > area.right || y < area.top || y > area.bottom) {
    return null;
  }

  let best = null;
  chart.data.datasets.forEach((ds, i) => {
    if (!chart.isDatasetVisible(i)) return;
    const line = chart.getDatasetMeta(i).dataset;
    if (!line || typeof line.interpolate !== 'function') return;
    // Undefined past the ends of the series; an array where several segments
    // cross this x (spanGaps can leave a curve in more than one piece).
    const found = line.interpolate({ x }, 'x');
    if (!found) return;
    (Array.isArray(found) ? found : [found]).forEach((pt) => {
      if (!pt || !Number.isFinite(pt.y)) return;
      const distance = Math.abs(pt.y - y);
      if (!best || distance < best.distance) {
        best = { datasetIndex: i, x: pt.x, y: pt.y, distance };
      }
    });
  });

  if (!best || best.distance > HOVER_HIT_RADIUS_PX) return null;
  // The tooltip reads real samples (`_raw[dataIndex]`), so pair the interpolated
  // position with the closest actual point on that curve.
  best.dataIndex = nearestDataIndex(chart.getDatasetMeta(best.datasetIndex), x);
  return best.dataIndex < 0 ? null : best;
}

/** Index of the closest non-skipped point of `meta` to canvas x. */
function nearestDataIndex(meta, x) {
  let index = -1;
  let smallest = Infinity;
  (meta.data || []).forEach((point, i) => {
    if (point.skip) return;
    const dx = Math.abs(point.x - x);
    if (dx < smallest) {
      smallest = dx;
      index = i;
    }
  });
  return index;
}

/** Commit a resolved hover target (or null) to the chart, redrawing on change. */
function applyHoverTarget(chart, target) {
  const index = target ? target.datasetIndex : null;
  const sameSpot =
    (hoveredPoint?.x ?? null) === (target?.x ?? null) &&
    (hoveredPoint?.y ?? null) === (target?.y ?? null);
  if (index === hoveredDatasetIndex && sameSpot) return;
  const curveChanged = index !== hoveredDatasetIndex;

  // State first: setActiveElements rebuilds the tooltip synchronously, and
  // `tooltip.filter` reads hoveredDatasetIndex while it does.
  hoveredDatasetIndex = index;
  hoveredPoint = target ? { x: target.x, y: target.y } : null;

  // The tooltip is driven from here rather than by Chart.js (see `events: []`),
  // so it can never disagree with the emphasis about which curve is hovered.
  if (chart.tooltip) {
    chart.tooltip.setActiveElements(
      target ? [{ datasetIndex: target.datasetIndex, index: target.dataIndex }] : [],
      target ? { x: target.x, y: target.y } : undefined,
    );
  }

  if (curveChanged) {
    // Every dataset's colour and width changed — needs a real update pass.
    styleDatasets(chart);
    chart.update('none');
    return;
  }
  // Only the marker slid along the curve already emphasized: repaint is enough.
  // `update('none')` here would re-run the whole pipeline on every mousemove.
  chart.render();
}

function clearChartHoverEmphasis() {
  if (hoveredDatasetIndex == null && hoveredPoint == null) return;
  // Clear the state even with no live chart, so a stale index can never dim the
  // wrong series after the next render rebuilds the dataset array.
  const chart = equityCurvesChartInstance;
  hoveredDatasetIndex = null;
  hoveredPoint = null;
  if (!chart) return;
  if (chart.tooltip) chart.tooltip.setActiveElements([], undefined);
  styleDatasets(chart);
  chart.update('none');
}

/**
 * Single source of truth for chart hover: every pointer position on the canvas.
 *
 * Chart.js' own `onHover` cannot do this job. It only fires while the pointer
 * is inside `chartArea` (plus `_minPadding`, a couple of px of dataset
 * overflow), so the endpoint-label gutter reserved by `layout.padding.right`
 * never reaches the proximity gate — emphasis would stick on whichever curve
 * was hovered last. (That gutter is no longer a literal: it is measured per
 * layout by `boardFrameLayout`, from 18px when the frame draws no labels up to
 * the measured block plus slack, ~245px on this tab. Do not re-introduce a
 * number here — the previous one said 120 and was wrong in both directions.) Worse, `chart.update()` replays
 * the last in-plot event, so clearing from `onHover` re-emphasized the curve
 * the clear had just dropped. Handling the DOM event directly sidesteps both.
 */
function handleCanvasPointerMove(event) {
  const chart = equityCurvesChartInstance;
  if (!chart || !chart.chartArea) return;
  const rect = chart.canvas.getBoundingClientRect();
  applyHoverTarget(
    chart,
    resolveHoverTarget(chart, event.clientX - rect.left, event.clientY - rect.top),
  );
}

function styleDatasets(chart) {
  const emphasisLabel = chart.$emphasisLabel;
  chart.data.datasets.forEach((ds, i) => {
    const st = ds._style || { kind: 'team' };
    const baseW = KIND_WIDTH[st.kind] || 2;
    let alpha;
    let width;
    if (hoveredDatasetIndex != null) {
      if (i === hoveredDatasetIndex) {
        alpha = 1.0;
        width = Math.max(baseW + 0.75, 2.5);
      } else {
        alpha = 0.25;
        width = baseW;
      }
    } else {
      const emph = ds.label === emphasisLabel;
      alpha = emph ? 1.0 : (KIND_ALPHA[st.kind] ?? 1.0);
      width = emph ? EMPHASIS_WIDTH : baseW;
    }
    ds.borderColor = hexToRgba(st.color, alpha);
    ds.borderWidth = width;
  });
}

// Subtle glow only on the explicitly selected curve (row click), not on hover.
const selectedGlowPlugin = {
  id: 'selectedGlow',
  beforeDatasetDraw(chart, args) {
    const ds = chart.data.datasets[args.index];
    if (ds && ds.label === chart.$emphasisLabel && hoveredDatasetIndex == null) {
      const { ctx } = chart;
      ctx.save();
      ctx.shadowColor = hexToRgba((ds._style && ds._style.color) || '#ffffff', 0.45);
      ctx.shadowBlur = 6;
    }
  },
  afterDatasetDraw(chart, args) {
    const ds = chart.data.datasets[args.index];
    if (ds && ds.label === chart.$emphasisLabel && hoveredDatasetIndex == null) {
      chart.ctx.restore();
    }
  },
};

// Dot marking the exact spot on the hovered curve. Chart.js' own point-hover
// styling is off (`hover: { mode: null }`): it tracks the nearest *data point*
// on the nearest curve regardless of our proximity gate, so it kept marking a
// curve out in empty plot space, and on a sparse board it snapped up to half a
// segment away from the cursor.
const hoverMarkerPlugin = {
  id: 'hoverMarker',
  afterDatasetsDraw(chart) {
    if (hoveredDatasetIndex == null || !hoveredPoint) return;
    const ds = chart.data.datasets[hoveredDatasetIndex];
    if (!ds) return;
    const { ctx } = chart;
    ctx.save();
    ctx.beginPath();
    ctx.arc(hoveredPoint.x, hoveredPoint.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = (ds._style && ds._style.color) || '#e5e7eb';
    ctx.strokeStyle = 'rgba(15, 23, 42, 0.9)';
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  },
};

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
    // `chart.isDatasetVisible(i)`, not a hand-rolled `meta.hidden || ds.hidden`:
    // `resolveHoverTarget` in this same file already asks Chart.js the question
    // this way, and two predicates for one chart is how the hover gate and the
    // label rail end up disagreeing about which curves are on screen. Chart.js
    // also resolves the pair properly -- meta.hidden wins only when it is an
    // actual boolean -- where `||` reads an explicit `meta.hidden === false`
    // as "fall through to ds.hidden".
    if (!chart.isDatasetVisible(i) || !ds.data || !ds.data.length) return;
    const meta = chart.getDatasetMeta(i);
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

/** The label set `beforeLayout` sizes the gutter from.
 *
 *  THE TWO HOOKS DO NOT SHARE A PREDICATE, and pretending otherwise was the
 *  bug. `beforeLayout` runs before any point has a position, so it cannot
 *  filter on `Number.isFinite(anchorY)` the way the draw hook does -- on the
 *  very first layout that would reject every series and reserve no gutter at
 *  all. It therefore counts every CANDIDATE, which keeps the reserve
 *  conservative in both directions (a wider measured floor, a larger gap) and
 *  means the draw set is always a subset: the frame can waste gutter, never
 *  clip a label.
 *
 *  What it CAN do is stop wasting it after the first frame. Once any anchor is
 *  positioned the unpositioned ones are genuinely absent rather than merely
 *  not-yet-laid-out, so from the second update onwards the two hooks see the
 *  same set and "gutter reserved and empty" stops being reachable at all. */
function boardLayoutLabels(chart, formatValue) {
  const all = boardVisibleEndpoints(chart, formatValue);
  const positioned = all.filter((lab) => Number.isFinite(lab.anchorY));
  return positioned.length ? positioned : all;
}

/** Re-lay-out once the gutter's webfont has actually landed.
 *
 *  `boardLabelBlockWidth` measures in `600 11px Inter`, and app.html loads
 *  Inter with `display=swap`. A chart laid out before the face arrives sizes
 *  its gutter against system-ui and then never re-measures: the next repaint
 *  is a bare `chart.render()` (the hover path), which does not re-run
 *  `beforeLayout`. The labels paint in Inter into a gutter measured for the
 *  fallback, and where Inter is the wider face the value pill runs off the
 *  canvas -- precisely the clipping a runtime measurement is supposed to be
 *  immune to, which is the claim `boardLabelBlockWidth`'s docstring makes.
 *
 *  One shot per chart. A no-op wherever `document.fonts` does not exist (node,
 *  older browsers), where nothing swapped under the measurement anyway. */
function boardWatchGutterFont(chart) {
  if (chart.$boardFontWatched) return;
  chart.$boardFontWatched = true;
  if (typeof document === 'undefined' || !document.fonts || !document.fonts.ready) return;
  document.fonts.ready
    .then(() => {
      // This tab destroys and rebuilds its chart on every render; Chart.js
      // nulls both of these on destroy.
      if (chart.canvas && chart.ctx) chart.update('none');
    })
    .catch(() => {});
}

/** Percent, two decimals, signed -- `+7.49%` / `-3.20%` / `''` for non-finite.
 *  Shared so the sign and precision cannot drift between this tab's money-view
 *  percent branch (Spec §4.5) and `boardDefaultValueText` below, the default
 *  every other pill on this frame takes. */
function boardSignedPercent(fraction) {
  const v = Number(fraction);
  if (!Number.isFinite(v)) return '';
  return `${v > 0 ? '+' : ''}${(v * 100).toFixed(2)}%`;
}

/** Percent, two decimals, signed. The default for every surface whose axis is
 *  percent -- which is both of them except this tab in its money view. */
function boardDefaultValueText(ds, lastIdx) {
  return boardSignedPercent(ds.data[lastIdx]);
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

/** Lay the endpoint labels out vertically: sorted by endpoint, never closer
 *  together than `gap`, and never outside the band `[top, bottom]`. Mutates
 *  `lab.y`; returns whether the stack fits, so a caller that cannot honour the
 *  band draws nothing rather than something clipped.
 *
 *  Pure in (labels, gap, top, bottom) for the same reason `boardFrameLayout`
 *  is: node without a canvas is the only kind of chart test this repo has, and
 *  the defect this replaces was invisible to every source-shape guard that
 *  already existed.
 *
 *  THE BAND IS THE CANVAS, NOT THE PLOT -- that distinction IS the bug this
 *  fixes. These labels live in the gutter, to the right of `chartArea.right`,
 *  so they may legitimately hang past `chartArea.bottom` into the x-axis strip.
 *  That strip is NEARLY empty at their x, not actually empty: Chart.js centres
 *  the last x tick on `chartArea.right` and reserves its right-half overhang
 *  inside our own `layout.padding.right`, which reaches about 18px into the
 *  gutter here and 21px on screen 0's larger ticks. Scales draw before
 *  `afterDatasetsDraw`, so a label always wins those pixels rather than being
 *  occluded by one; BOARD_TICK_CLEARANCE is what keeps it from landing on top
 *  of a tick in the first place. The version this replaces clamped
 *  them to `chartArea` instead, and since the forward pass enforces a MINIMUM
 *  gap the staggered stack is routinely taller than the plot: 255.3px against a
 *  237.4px plot on the Leaderboard tab at 1440, 202.5px against 168.8px on
 *  screen 0. Two whole-stack shifts then cancelled each other exactly -- lift
 *  the tail inside, put it straight back -- and a rendered check found the last
 *  label drawn 5px past the canvas bottom on the tab and 10.4px past it on
 *  screen 0, sliced through the middle.
 *
 *  THREE PASSES. Down to open the gaps, up to pull the tail inside `bottom`,
 *  down again to push the head inside `top`. The third is not redundant with
 *  the first: the second pass is what can drive the head out, and re-running
 *  the first is what puts it back without compressing anything. Walking a bound
 *  rather than shifting the whole stack is what keeps every gap at or above
 *  `gap` -- nothing is squeezed to buy the room. */
function boardStackLabels(labels, gap, top, bottom) {
  if (!labels || !labels.length) return false;
  labels.sort((a, b) => a.y - b.y);
  const last = labels.length - 1;
  for (let k = 1; k <= last; k += 1) {
    if (labels[k].y - labels[k - 1].y < gap) labels[k].y = labels[k - 1].y + gap;
  }
  if (labels[last].y > bottom) labels[last].y = bottom;
  for (let k = last - 1; k >= 0; k -= 1) {
    if (labels[k].y > labels[k + 1].y - gap) labels[k].y = labels[k + 1].y - gap;
  }
  if (labels[0].y < top) labels[0].y = top;
  for (let k = 1; k <= last; k += 1) {
    if (labels[k].y - labels[k - 1].y < gap) labels[k].y = labels[k - 1].y + gap;
  }
  // False only when the band cannot hold n pills at this pitch, which
  // boardFrameLayout now excludes before reserving the gutter. Reported rather
  // than asserted so the two hooks cannot drift into disagreeing again.
  return labels[last].y <= bottom;
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
      boardWatchGutterFont(chart);
      const labels = boardLayoutLabels(chart, formatValue);
      const frame = boardFrameLayout(chart, labels, fraction);
      // The measured texts ride along so the draw hook does not rebuild them.
      // `formatValue` per series plus 2N `measureText` is the whole cost of a
      // layout, and `applyHoverTarget` calls `chart.update('none')` on every
      // change of hovered curve -- so recomputing it in the draw hook ran the
      // entire measurement again on the mousemove path, for values that cannot
      // have changed without an update. Anchors are deliberately NOT carried:
      // they are last frame's here, and the draw hook re-reads them.
      frame.labels = labels;
      chart.$boardFrame = frame;
      const layout = chart.options.layout || (chart.options.layout = {});
      // `padding` may legally be a NUMBER -- Chart.js shorthand for uniform
      // padding. Replacing it with `{}` and setting only `.right` silently
      // dropped the other three sides. Neither caller in this repo passes one,
      // but this factory is exported on `window` precisely so other surfaces
      // can adopt the frame, and losing top/left/bottom padding renders their
      // curves flush to the canvas edges.
      const current = layout.padding;
      const padding =
        current && typeof current === 'object'
          ? current
          : (layout.padding = Number.isFinite(current)
            ? { top: current, right: current, bottom: current, left: current }
            : {});
      padding.right = frame.gutter;
    },
    afterDatasetsDraw(chart) {
      const { ctx, chartArea } = chart;
      const frame = chart.$boardFrame;
      if (!frame || !frame.drawLabels || !chartArea || !frame.labels) return;
      // Re-anchor the set `beforeLayout` measured, rather than rebuilding it:
      // the names and values are already current (nothing changes them without
      // an update, and an update re-runs beforeLayout), so only the positions
      // have to be re-read.
      //
      // Number.isFinite on BOTH axes, not `!= null`: a NaN anchorY (a malformed
      // point) would pass a null check, then poison every comparison inside
      // boardStackLabels, which returns false and drops the WHOLE stack rather
      // than the one bad series. anchorX is checked for a quieter reason -- the
      // canvas spec silently discards an arc/moveTo/lineTo containing NaN, so a
      // point with a finite y and a NaN x used to draw a name and a pill with
      // no endpoint dot and no stub, an inconsistent row nothing would report.
      const labels = [];
      frame.labels.forEach((lab) => {
        const meta = chart.getDatasetMeta(lab.i);
        const point = meta && meta.data && meta.data[lab.lastIdx];
        const anchorX = point ? point.x : null;
        const anchorY = point ? point.y : null;
        if (!Number.isFinite(anchorX) || !Number.isFinite(anchorY)) return;
        labels.push({ ...lab, anchorX, anchorY, y: anchorY });
      });
      if (!labels.length) return;

      // Stagger overlapping labels, keeping >= gap px spacing and staying on
      // canvas. Each keeps its original endpoint y (anchorY) so we can tell
      // later whether collision-avoidance actually moved it.
      //
      // The band is the CANVAS, not chartArea: a gutter label sits to the right
      // of the plot, so hanging below chartArea.bottom into the axis strip is
      // legitimate where clamping to it clipped the stack. The strip is not
      // quite empty there -- the last x tick overhangs into the gutter -- which
      // is what BOARD_TICK_CLEARANCE indents the descending labels past.
      // See boardStackLabels. A false return means the band cannot hold the
      // stack -- draw nothing rather than something cut off, which is the same
      // degradation boardFrameLayout takes at the narrow and short extremes.
      const half = BOARD_PILL_HEIGHT / 2;
      if (!boardStackLabels(labels, frame.gap, half, chart.height - half)) return;

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
        // Past the last x tick's overhang, but only for the labels that
        // actually hang into the axis strip -- see BOARD_TICK_CLEARANCE. The
        // clearance is reserved unconditionally by boardLabelBlockWidth, so
        // spending it here can never push a block out of the gutter.
        const lx = labelX + (lab.y + half > chartArea.bottom ? BOARD_TICK_CLEARANCE : 0);

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
          ctx.lineTo(lx - 3, lab.y);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        let x = lx;
        ctx.fillStyle = hexToRgba(lab.color, alpha);
        ctx.beginPath();
        ctx.arc(x + BOARD_DOT_RADIUS, lab.y, BOARD_DOT_RADIUS, 0, Math.PI * 2);
        ctx.fill();
        x += BOARD_DOT_RADIUS * 2 + BOARD_DOT_GAP;

        ctx.fillStyle = hexToRgba(lab.color, alpha);
        ctx.fillText(lab.name, x, lab.y);
        x += ctx.measureText(lab.name).width + BOARD_NAME_GAP;

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
    // afterDatasetsDraw, and registered BEFORE createEndpointLabelPlugin at
    // both call sites. Chart.js has already drawn the datasets by this hook,
    // so the original reason for afterDraw -- chrome above a curve running
    // along the floor -- still holds exactly. What afterDraw could NOT do is
    // let the labels sit above the chrome: this baseline runs the full canvas
    // width, straight through the reserved gutter, so it painted a line
    // through every label the stack pushes below chartArea.bottom. Measured on
    // a 1440x268 tab with 12 low-clustered curves: chartArea.bottom 247.6, a
    // label at y=241 whose pill spans 233.5-248.5, struck through at 247.6.
    // Plugins run a hook in array order, so the labels now paint last.
    afterDatasetsDraw(chart) {
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

function buildCustomLegend(chart) {
  const container = document.getElementById('equityCurvesLegend');
  if (!container) return;

  const order = { team: 0, model: 1, strategy: 2, benchmark: 3 };
  const items = chart.data.datasets
    .map((ds, i) => ({ ds, i }))
    .sort((a, b) => (order[a.ds._style.kind] ?? 9) - (order[b.ds._style.kind] ?? 9));

  container.innerHTML = items.map(({ ds }) => {
    const st = ds._style;
    const hidden = hiddenSeries.has(ds.label);
    const w = Math.min(KIND_WIDTH[st.kind] || 2, 2.4);
    const dash = (st.dash && st.dash.length) ? st.dash.join(',') : '';
    const stroke = hidden ? 'rgba(148,163,184,0.4)' : st.color;
    return `
      <button class="legend-item${hidden ? ' legend-hidden' : ''}" data-label="${ds.label.replace(/"/g, '&quot;')}">
        <svg class="legend-sample" width="26" height="10" viewBox="0 0 26 10">
          <line x1="1" y1="5" x2="25" y2="5" stroke="${stroke}" stroke-width="${w}"
            ${dash ? `stroke-dasharray="${dash}"` : ''} stroke-linecap="round" />
        </svg>
        <span class="legend-label">${shortName(ds.label)}</span>
      </button>`;
  }).join('');

  container.querySelectorAll('.legend-item').forEach((el) => {
    el.addEventListener('click', () => {
      const label = el.dataset.label;
      if (hiddenSeries.has(label)) hiddenSeries.delete(label);
      else hiddenSeries.add(label);
      updateCurvePickerCount();
      if (document.getElementById('curvePickerTrigger')?.getAttribute('aria-expanded') === 'true') {
        renderCurvePicker();
      }
      renderEquityCurvesChart();
    });
  });
}

async function renderEquityCurvesChart() {
  if (!equityCurvesData) return;

  const canvas = document.getElementById('equityCurvesChart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const { times, days, curves, initials } = equityCurvesData;
  const axisLabels = times || days;
  const orderedEntries = (leaderboardPayload?.entries || []);
  const emphasisLabel = getEmphasisLabel();

  // Hover state is a position into the dataset array about to be rebuilt, and
  // rank changes or a legend toggle reorder it. Drop it rather than carry an
  // index that now points at a different series.
  hoveredDatasetIndex = null;
  hoveredPoint = null;

  const datasets = [];
  orderedEntries.forEach((entry) => {
    const label = entry.model || entry.team_name;
    const raw = curves[label];
    if (!raw || !raw.length) return;
    const initial = initials[label] || raw[0] || 10000;
    const style = getSeriesStyle(label, entry);

    datasets.push({
      label,
      data: transformLeaderboardChartData(raw, currentChartView, initial),
      _raw: raw,
      _initial: initial,
      _entry: entry,
      _style: style,
      borderColor: style.color,
      backgroundColor: 'transparent',
      borderDash: style.dash || [],
      borderCapStyle: 'round',
      pointRadius: 0,
      // Marker comes from hoverMarkerPlugin, which honours the proximity gate.
      pointHoverRadius: 0,
      tension: 0.1,
      fill: false,
      // Series use different hour grids (e.g. SPY :30 vs LLM :00). On a shared
      // axis that leaves many nulls; span across them so each curve still draws.
      spanGaps: true,
      hidden: hiddenSeries.has(label),
    });
  });

  const baseInitial = (datasets[0] && datasets[0]._initial) || 10000;
  const isMoney = currentChartView === 'absolute';

  if (equityCurvesChartInstance) {
    equityCurvesChartInstance.destroy();
  }

  equityCurvesChartInstance = new Chart(ctx, {
    type: 'line',
    data: { labels: axisLabels, datasets },
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
          return boardSignedPercent(ret);
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
      // Chart.js does no interaction resolution of its own: hover, the marker
      // and the tooltip are all driven from handleCanvasPointerMove, which is
      // the only path that sees the endpoint-label gutter. Leaving the built-in
      // handling on would light up curves the proximity gate has rejected, and
      // its replay of the last in-plot event on every `update()` would undo the
      // clear that moving into the gutter had just performed.
      events: [],
      hover: { mode: null },
      plugins: {
        legend: { display: false },
        tooltip: {
          position: 'nearest',
          backgroundColor: 'rgba(15, 23, 42, 0.96)',
          borderColor: 'rgba(148, 163, 184, 0.25)',
          borderWidth: 1,
          titleColor: '#e5e7eb',
          bodyColor: '#cbd5e1',
          padding: 10,
          displayColors: false,
          // Belt-and-braces: the active element is set explicitly by
          // applyHoverTarget, so this only bites if Chart.js event handling is
          // ever switched back on above.
          filter(item) {
            return hoveredDatasetIndex != null && item.datasetIndex === hoveredDatasetIndex;
          },
          callbacks: {
            title(items) {
              if (!items.length) return '';
              return formatChartTooltipLabel(items[0].label);
            },
            label(context) {
              const ds = context.dataset;
              const idx = context.dataIndex;
              const equity = (ds._raw && ds._raw[idx]) || 0;
              const ret = (equity - ds._initial) / (ds._initial || 1);
              const entry = ds._entry || {};
              const lines = [
                ds.label,
                `Return: ${(ret * 100).toFixed(2)}%`,
                `Value: $${formatLeaderboardNumber(equity)}`,
                `Rank: ${entry.rank ?? '—'} / ${leaderboardPayload?.total_entries || '—'}`,
              ];

              const benchDs = context.chart.data.datasets.find((d) => d.label === selectedBenchmarkLabel);
              if (benchDs && benchDs.label !== ds.label && benchDs._raw && benchDs._raw[idx] != null) {
                const benchRet = (benchDs._raw[idx] - benchDs._initial) / (benchDs._initial || 1);
                const diff = (ret - benchRet) * 100;
                const sign = diff >= 0 ? '+' : '';
                lines.push(`vs ${shortName(selectedBenchmarkLabel)}: ${sign}${diff.toFixed(2)}%`);
              }
              return lines;
            },
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: '#6b7280',
            maxRotation: 0,
            minRotation: 0,
            autoSkip: true,
            maxTicksLimit: 7,
            callback(value) {
              return formatShortDate(this.getLabelForValue(value));
            },
          },
          grid: { color: 'rgba(148, 163, 184, 0.05)', drawTicks: false },
        },
        y: {
          ticks: {
            color: '#9ca3af',
            callback(value) {
              if (isMoney) return `$${formatLeaderboardNumber(value)}`;
              return `${(value * 100).toFixed(1)}%`;
            },
          },
          grid: {
            color(c) {
              const v = c.tick.value;
              const isRef = isMoney ? Math.abs(v - baseInitial) < 1 : Math.abs(v) < 1e-9;
              return isRef ? 'rgba(148, 163, 184, 0.45)' : 'rgba(148, 163, 184, 0.08)';
            },
            lineWidth(c) {
              const v = c.tick.value;
              const isRef = isMoney ? Math.abs(v - baseInitial) < 1 : Math.abs(v) < 1e-9;
              return isRef ? 1.4 : 1;
            },
          },
        },
      },
    },
  });

  equityCurvesChartInstance.$emphasisLabel = emphasisLabel;
  styleDatasets(equityCurvesChartInstance);
  equityCurvesChartInstance.update('none');

  if (!canvasPointerBound) {
    // Pointer events, not mouse events: `events: []` turned off Chart.js' own
    // touchstart/touchmove handling, and these cover mouse, touch and pen in
    // one pair. Bound once — the canvas outlives each destroy/recreate cycle.
    canvas.addEventListener('pointermove', handleCanvasPointerMove);
    canvas.addEventListener('pointerleave', clearChartHoverEmphasis);
    canvasPointerBound = true;
  }

  buildCustomLegend(equityCurvesChartInstance);
}

function transformLeaderboardChartData(curveValues, viewType, initialValue) {
  const base = initialValue || 10000;
  if (viewType === 'absolute') {
    return curveValues.slice();
  }
  return curveValues.map((v) => (v == null ? null : (v - base) / base));
}

/** Neutral table state for the gap between asking for a board and being given it.
 *
 * Deliberately not `populateLeaderboardTable()` over an empty payload: that
 * renders "No entries in this season yet", which during a 30-60s free-tier cold
 * start is a claim about the board nobody is in a position to make. Nor the rows
 * already on screen — those belong to the board the user just left, and under
 * the new board's title they read as its standings.
 */
function showLeaderboardTableLoading() {
  const tbody = document.getElementById('leaderboardTableBody');
  if (!tbody) return;
  tbody.innerHTML =
    '<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--text-secondary);">Loading…</td></tr>';
}

function displayLeaderboardError(message) {
  const tbody = document.getElementById('leaderboardTableBody');
  if (tbody) {
    // `escapeHtml` is a global from app.js, which app.html loads first — the same
    // contract js/leaderboard.js already relies on for API/API_BASE.
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 30px; color: var(--danger-color);">Error: ${escapeHtml(message)}</td></tr>`;
  }
}

window.loadLeaderboardData = loadLeaderboardData;
window.selectLeaderboardTeam = selectLeaderboardTeam;
// Consumed by home-page.js for screen 0's chart. Explicit rather than relying
// on the implicit global these classic scripts share: on rename the implicit
// form degrades to "no chart", which this design deliberately makes
// indistinguishable from the honest no-curves state -- so the break would be
// invisible. The export is pinned from both sides by
// test_frontend_chart_first_home.py.
window.buildEquityCurvesFromEntries = buildEquityCurvesFromEntries;
// Screen 0's rank rows read their swatch from here, so a row's colour and its
// curve's colour cannot disagree -- the swatch is the chart's only key.
window.getSeriesStyle = getSeriesStyle;
// Both axis formatters, for the same reason the two above are exported: screen
// 0 plots the SAME hourly `equity_curve` timestamps this tab does, and without
// these it printed them raw -- "2026-04-15T00:00", rotated 45 degrees and
// colliding across a chart 187px tall. A tick label is not copy either surface
// gets to invent; it is a rendering of the other's data.
window.formatShortDate = formatShortDate;
window.formatChartTooltipLabel = formatChartTooltipLabel;
// The shared board frame, consumed by home-page.js for screen 0's chart.
// Explicit for the same reason the four exports above are: the implicit global
// degrades on rename to a chart with no frame, and a frame that silently stops
// drawing is indistinguishable from a frame nobody asked for. Pinned from both
// sides by test_frontend_board_frame.py.
// Exported for home-page.js's rank rows: the pill beside a curve and the row
// beneath it render the same number, and two independent expressions producing
// it is how they drift.
window.boardSignedPercent = boardSignedPercent;
window.createEndpointLabelPlugin = createEndpointLabelPlugin;
window.createAxisArrowPlugin = createAxisArrowPlugin;
