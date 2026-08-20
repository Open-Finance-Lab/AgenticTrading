import { useMemo } from "react";
import {
  BOARD_ARROW_HEAD_HALF,
  BOARD_ARROW_HEAD_LENGTH,
  BOARD_AXIS_COLOR,
  BOARD_DOT_GAP,
  BOARD_DOT_RADIUS,
  BOARD_GUTTER_FONT,
  BOARD_GUTTER_TEXT_INSET,
  BOARD_LABEL_GAP_MAX,
  BOARD_NAME_GAP,
  BOARD_PILL_HEIGHT,
  BOARD_PILL_PAD_X,
  BOARD_STUB_LENGTH,
  BOARD_TICK_CLEARANCE,
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
 *  ONLY STATIC SOURCE GUARDS COVER THIS FILE, AND THAT IS DELIBERATE. This is
 *  a React component; a behavioural tier would need `@testing-library/react`
 *  plus jsdom, neither of which is installed on this branch (tracked
 *  separately as issue #383). `test_landing_live_board.py`'s rail tests are
 *  therefore regex/substring checks against this source text, not a rendered
 *  DOM assertion -- they prove the right constants and guard clauses are
 *  present in the code, not that the geometry they produce is correct at
 *  runtime. The live-browser pass the controller runs is what actually closes
 *  that gap for this change.
 *
 *  INTERNAL SHAPE, WITH A REAL FALLBACK. Recharts clones a `<Customized>` child
 *  with `{...chartProps, ...chartState}` (es6/chart/generateCategoricalChart.js,
 *  `renderCustomized`), which is how `formattedGraphicalItems` and `offset`
 *  arrive. That is stable across 2.x but is not a contract, so when it is not
 *  what this expects the rail renders NOTHING and the chip strip below the chart
 *  keeps keying every curve exactly as it does today. The card stays complete;
 *  it loses the gutter labels. That is a real degradation with a real fallback,
 *  not a silent one -- it is not silent in the sense that matters (a reader
 *  tracing why labels are missing), because this file names the exact upstream
 *  shape change ("recharts changed shape") in the comment above the guard
 *  clause below, rather than the fallback reading as "there was nothing to
 *  draw" (no series, or `drawLabels` false because the gutter didn't fit).
 *  Those two causes are distinguished in code: the internals-changed case
 *  returns null without drawing anything; "nothing to draw" still renders the
 *  arrow. (The row and placement memos sit ABOVE that guard because hook order
 *  may not depend on a prop -- they run, and yield empty, in exactly the cases
 *  it rejects. Nothing between them and the guard reads their output.)
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
  height?: number;
  valueByKey?: Record<string, string>;
  drawLabels?: boolean;
  gap?: number;
};

type Row = Anchor & { name: string; value: string; color: string };

/** Half a pill: the inset that keeps the first and last labels on canvas.
 *
 *  The shipped hook computes this same `half` and spends it twice -- once as
 *  the stacking band's inset and once to decide which labels have descended
 *  into the x-axis strip. Both uses are mirrored below. */
const HALF_PILL = BOARD_PILL_HEIGHT / 2;

export function EndpointRail(props: RailProps) {
  const { formattedGraphicalItems, offset, width, height, valueByKey, drawLabels, gap } =
    props;

  // MEMOISED, AND RUN AHEAD OF THE GUARD BELOW. Recharts clones <Customized>
  // with `{...props, ...state}`, and `chartX`/`chartY`/`activeTooltipIndex` are
  // written into that state by a throttled 60fps mousemove handler -- so every
  // pointer frame over the card re-renders this component from scratch. Row
  // building, the three-pass walk and 2N canvas `measureText` calls do not
  // depend on the pointer, and the shipped Chart.js hook documents removing
  // exactly this waste ("ran the entire measurement again on the mousemove
  // path, for values that cannot change"). None of these deps move on hover:
  // `formattedGraphicalItems` and `offset` are set by the layout pass, never by
  // the tooltip handlers, and `valueByKey` is memoised by the card.
  //
  // Hook order may not depend on a prop, so both run unconditionally and return
  // an empty result in exactly the cases the guard below rejects.
  const rows = useMemo<Row[]>(() => {
    const built: Row[] = [];
    if (!Array.isArray(formattedGraphicalItems)) return built;
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
      built.push({
        key,
        anchorX: last.x,
        anchorY: last.y,
        name: item.name || key,
        value: valueByKey?.[key] ?? "",
        color: item.stroke || "#94a3b8",
      });
    }
    return built;
  }, [formattedGraphicalItems, valueByKey]);

  // THE BAND IS THE CANVAS INSET BY HALF A PILL, NOT THE PLOT. The shipped call
  // is `boardStackLabels(labels, frame.gap, half, chart.height - half)`, and its
  // docstring records the bug the distinction fixes: a gutter label legitimately
  // hangs BELOW the plot into the x-axis strip, so clamping the tail to
  // `offset.top + offset.height` re-clips the stack there, and dropping the
  // inset lets the head's 15px pill be centred on the band's edge and sliced by
  // the viewBox. `frameLayout`'s stack-fits-the-canvas guard is checked against
  // the full `height` for the same reason, so this is also the band that guard
  // was written to bound -- narrowing one without the other unhooks them.
  //
  // `gap` comes from the card's single `frameLayout` call, never from a second
  // computation here: the gutter width that was RESERVED came out of that call,
  // and a gap derived independently is a second chance to disagree with it.
  const placed = useMemo(
    () =>
      rows.length && height
        ? stackLabels(rows, {
            gap: gap ?? BOARD_LABEL_GAP_MAX,
            top: HALF_PILL,
            bottom: height - HALF_PILL,
          })
        : [],
    [rows, gap, height],
  );

  // Recharts internals changed shape (or this rendered before Recharts ever
  // called it): none of these are optional in a real render, so bail before
  // touching anything else rather than let a missing field throw mid-draw.
  if (!Array.isArray(formattedGraphicalItems) || !offset || !width || !height) return null;

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

  // "Nothing to draw" (no rows found, or the card's frameLayout call decided
  // the gutter doesn't fit and set drawLabels false) is NOT the internals-
  // changed case above: it still renders the arrow, and it is the fallback
  // the card's frame itself chose, not evidence Recharts broke anything.
  if (!drawLabels || !rows.length) return <>{arrow}</>;

  const byKey = new Map(rows.map((row) => [row.key, row]));

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
        // Past the last x tick's overhang, and only for the labels that
        // actually descend into the axis strip -- recharts centres the last x
        // tick on the plot's right edge with `textAnchor="middle"`, so it hangs
        // into the gutter directly under exactly those labels. The clearance is
        // reserved unconditionally by `labelBlockWidth`, so spending it here can
        // never push a block out of the gutter; the shipped hook computes the
        // same `lx`. Reserving it and never spending it left the reserve idle
        // and the overlap it was bought to prevent in place.
        const lx = labelX + (p.y + HALF_PILL > axisY ? BOARD_TICK_CLEARANCE : 0);
        const pillX =
          lx + BOARD_DOT_RADIUS * 2 + BOARD_DOT_GAP + nameWidth + BOARD_NAME_GAP;
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
                x2={lx - 3}
                y2={p.y}
                stroke={row.color}
                strokeWidth={1}
                strokeDasharray="1 3"
                opacity={0.35}
              />
            ) : null}
            <circle cx={lx + BOARD_DOT_RADIUS} cy={p.y} r={BOARD_DOT_RADIUS} fill={row.color} />
            <text
              x={lx + BOARD_DOT_RADIUS * 2 + BOARD_DOT_GAP}
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
