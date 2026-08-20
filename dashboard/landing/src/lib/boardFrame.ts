/** The nof1-derived board frame, for the Recharts side.
 *
 *  A MIRROR of the same constants and geometry in
 *  `dashboard/frontend/js/leaderboard.js`, which serves the two vanilla Chart.js
 *  surfaces. The duplication is forced -- different bundles, no shared module --
 *  and pinned by `dashboard/backend/tests/test_landing_board_frame.py`, which
 *  fails if a single number drifts. Change a value here and change it there in
 *  the same commit.
 *
 *  THE CONSTANT GUARD CANNOT SEE ALGORITHM DRIFT. It diffs `BOARD_*` source
 *  text only, not function bodies -- so it stayed green through a full commit
 *  where `stackLabels` here used a different (and, per `js/leaderboard.js`'s
 *  own docstring, previously-buggy) overflow/underflow strategy than the
 *  shipped `boardStackLabels` it is supposed to mirror. Only a behavioural
 *  test with the right shape caught it (see
 *  `test_an_outlier_anchor_does_not_reintroduce_the_whole_stack_shift_bug`).
 *  Matching constants is necessary, not sufficient, for the two copies to
 *  actually agree -- when porting a function here, read the shipped body, not
 *  just its constants.
 *
 *  TWO TEST TIERS, AND THE SECOND SKIPS IN CI. The constant mirror in that test
 *  file is a source scan and runs everywhere. The behavioural tests transpile
 *  this file with the esbuild inside dashboard/landing/node_modules and run it
 *  under node, which needs an `npm install` CI does not do. A green CI therefore
 *  says the numbers agree, NOT that this geometry was exercised -- run the suite
 *  locally before shipping a change to either copy.
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

/** Width of the widest `dot name pill` block, plus the inset, the tick
 *  clearance and the trailing pad.
 *
 *  Mirrors `boardLabelBlockWidth` in `js/leaderboard.js` term-for-term,
 *  including `BOARD_TICK_CLEARANCE` -- the shipped comment there explains why a
 *  literal is defensible for that one constant: it is reserved unconditionally
 *  here and spent only by labels that actually descend into the axis strip, so
 *  getting it wrong costs a few px of residual overlap, never clipped text. */
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
  return BOARD_GUTTER_TEXT_INSET + widest + BOARD_TICK_CLEARANCE + BOARD_GUTTER_TRAILING_PAD;
}

/** How much right margin to reserve, whether to draw labels, and how far apart.
 *
 *  Mirrors `boardFrameLayout` in `js/leaderboard.js`. That function takes a
 *  live `chart` object and reads `chart.scales.x.height` when available; this
 *  module has no chart, so it always uses `BOARD_XAXIS_ALLOWANCE` -- the same
 *  first-frame estimate `boardXAxisHeight` falls back to before any Chart.js
 *  layout has run.
 *
 *  TWO DEGRADATIONS, BOTH TO "ARROW ONLY". Too narrow for the widest label, or
 *  too short to stack N of them, and the frame gives the space back rather than
 *  clipping text or piling labels on each other. Clipping is the failure this
 *  codebase keeps re-learning: the chip strip cut four of five model names at
 *  390px with no scrollbar, no ellipsis and nothing failing. Both surfaces keep
 *  a complete key elsewhere, so dropping the labels loses no information.
 *
 *  THE STACK MUST ALSO FIT THE CANVAS, not just clear the per-pair gap: an
 *  (n-1) gaps + one pill's height. Checked against the full `height`, not the
 *  usable plot height, because the stack lives in the gutter and may
 *  legitimately hang into the x-axis strip -- see `stackLabels` below. With
 *  today's constants (`BOARD_LABEL_GAP_MIN` derived as `BOARD_PILL_HEIGHT + 1`)
 *  this cannot fire; it stays wired in so that raising `BOARD_LABEL_GAP_MAX` or
 *  shrinking `BOARD_XAXIS_ALLOWANCE` degrades to arrow-only instead of silently
 *  reintroducing a clipped label. */
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
  if ((labels.length - 1) * gap + BOARD_PILL_HEIGHT > height) return none;
  const floor = labelBlockWidth(labels);
  if (floor > width * BOARD_GUTTER_MAX_FRACTION) return none;
  // `fraction` is a CEILING on the floor, not the gutter's width -- see the
  // comment above BOARD_GUTTER_FRACTION and BOARD_GUTTER_SLACK in
  // js/leaderboard.js. `Math.max(floor, ...)` is LOAD-BEARING, not defensive:
  // `floor` is the hard lower bound below which label text draws off-canvas,
  // and it must survive the inner Math.min() untouched. The genuinely-
  // impossible case -- floor itself past BOARD_GUTTER_MAX_FRACTION -- is the
  // `return none` above; this line never has to refuse on its own.
  const room = Math.max(floor, Math.min(width * fraction, floor + BOARD_GUTTER_SLACK));
  return { gutter: room, drawLabels: true, gap };
}

/** Stagger coincident endpoints downward, then walk the stack back inside
 *  `[top, bottom]` without re-opening the gaps the first pass just closed.
 *
 *  Each label keeps its endpoint y as `anchorY` so `displaced` can say whether
 *  collision-avoidance actually moved it -- a leader line shorter than
 *  BOARD_LEADER_MIN_DISPLACEMENT connects nothing and just leaves a stub.
 *
 *  THREE PASSES, NOT A SHIFT OF THE WHOLE STACK. This function used to correct
 *  an overflowing tail by shifting every label up by the overflow amount, then
 *  correct the underflow that shift could produce at the head by shifting
 *  every label back down by the underflow amount. For any stack whose SPREAD
 *  (not its count) exceeds the band, those two shifts cancel exactly -- the
 *  second undoes the first -- and the tail lands back outside `bottom` by the
 *  same amount it started. `js/leaderboard.js`'s `boardStackLabels` hit this
 *  for real: a rendered check found the last label 5-10px past the canvas
 *  bottom, sliced through the middle. The fix there, mirrored here, is to walk
 *  a BOUND instead of shifting a BLOCK: clamp only the last label to `bottom`,
 *  walk backward opening exactly the gaps that clamp compressed (never more),
 *  then clamp the first label to `top` and walk forward once more to restore
 *  any gap the backward pass over-closed while pulling the head up. That final
 *  forward pass is not redundant with the first: the backward pass is the one
 *  that can leave a label too close to the neighbour above it, and re-running
 *  the forward direction is what repairs that without re-compressing anything
 *  -- nothing here is ever squeezed below `gap` to buy room, only walked.
 *
 *  BOTH CLAMPS bind independently, and can bind on the same stack --
 *  `test_an_outlier_anchor_does_not_reintroduce_the_whole_stack_shift_bug` is
 *  exactly that case. `frameLayout`'s stack-fits-the-canvas guard keeps a
 *  caller from handing this a stack that cannot fit at ANY arrangement; it
 *  says nothing about whether the stack is well spread going in, which is
 *  what the walk above corrects for.
 *
 *  NO "DID IT FIT" RETURN. `boardStackLabels` also returns whether the last
 *  label ended at or before `bottom`, but its own comment says that value is
 *  "reported rather than asserted" -- a belt-and-suspenders cross-check
 *  against `boardFrameLayout`'s own stack-fits-the-canvas guard, not
 *  something a caller needs to branch on today. This mirror's `Placed[]`
 *  signature is pinned by the Task 8-9 contract, and the same fact is
 *  trivially recoverable without widening it: `placed[placed.length - 1].y <=
 *  opts.bottom`. */
export function stackLabels(
  anchors: Anchor[],
  opts: { gap: number; top: number; bottom: number },
): Placed[] {
  const placed: Placed[] = anchors
    .map((a) => ({ ...a, y: a.anchorY, displaced: false }))
    .sort((a, b) => a.y - b.y);
  const last = placed.length - 1;
  if (last < 0) return placed;
  for (let k = 1; k <= last; k += 1) {
    if (placed[k].y - placed[k - 1].y < opts.gap) placed[k].y = placed[k - 1].y + opts.gap;
  }
  if (placed[last].y > opts.bottom) placed[last].y = opts.bottom;
  for (let k = last - 1; k >= 0; k -= 1) {
    if (placed[k].y > placed[k + 1].y - opts.gap) placed[k].y = placed[k + 1].y - opts.gap;
  }
  if (placed[0].y < opts.top) placed[0].y = opts.top;
  for (let k = 1; k <= last; k += 1) {
    if (placed[k].y - placed[k - 1].y < opts.gap) placed[k].y = placed[k - 1].y + opts.gap;
  }
  placed.forEach((p) => {
    p.displaced = Math.abs(p.y - p.anchorY) > BOARD_LEADER_MIN_DISPLACEMENT;
  });
  return placed;
}

/** Dark or light pill ink, by the swatch's relative luminance.
 *
 *  MUST EXPAND A 3-DIGIT HEX BEFORE READING CHANNELS OUT OF IT. Shipped
 *  `hexToRgb`'s own docstring in js/leaderboard.js records this as a bug that
 *  existed in both of ITS callers at once -- unexpanded, `#fff` reads as
 *  r=255 g=15 b=0 (`slice(2,4)` on `'fff'` takes only the trailing `'f'`,
 *  `slice(4,6)` is empty), which is a light-ink verdict on a near-white pill.
 *  The docstring names the actual defect as "a fix applied to one copy would
 *  not have reached the other" -- and this mirror shipped as a THIRD copy
 *  with the same gap, for the same reason: one parse, repeated by hand. */
export function pillTextColor(hex: string): string {
  let h = String(hex || '').replace('#', '');
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  const r = parseInt(h.slice(0, 2), 16) || 0;
  const g = parseInt(h.slice(2, 4), 16) || 0;
  const b = parseInt(h.slice(4, 6), 16) || 0;
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? '#0b1220' : '#f8fafc';
}
