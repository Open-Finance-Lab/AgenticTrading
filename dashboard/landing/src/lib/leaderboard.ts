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

  // Colour is assigned once per SELECTED entry, in order -- never skipped by a
  // missing curve, because every selected entry reaches `standings` below
  // regardless of whether it has drawable values. That keeps this positional
  // MODEL_COLOR_PALETTE[n] indexing equivalent to /app's lazy per-entry_id
  // `getModelColor`, which mints a slot the first time an entry's style is
  // resolved: here every selected entry resolves exactly one style, in the
  // same order /app would resolve them in. Deriving the colour from a running
  // index that skipped curve-less entries shifted every later model's colour
  // by one slot -- the same failure mode home-page.js:1748 documents for a
  // stale key entering the SHARED modelColorMap: one model ends up wearing
  // another's colour, on whichever page built its map in a different order.
  let modelIndex = 0;
  const styleByEntryId = new Map<string, { color: string; dash?: string }>();
  perEntry.forEach(({ entry }) => {
    const isModel = !!(entry.is_model || entry.team_badge === 'Model');
    styleByEntryId.set(
      entry.entry_id,
      isModel
        ? { color: MODEL_COLOR_PALETTE[modelIndex++ % MODEL_COLOR_PALETTE.length], dash: undefined }
        : BASELINE_STYLES[entry.entry_id] || { color: '#94A3B8', dash: '10 6' },
    );
  });

  const series: BoardSeries[] = [];
  const standings: BoardStanding[] = [];
  perEntry.forEach(({ entry, byTime }) => {
    const isModel = !!(entry.is_model || entry.team_badge === 'Model');
    const style = styleByEntryId.get(entry.entry_id)!;
    const name = entry.model || entry.team_name;

    // Rank and return come from `cumulative_return`, independent of whether
    // this entry has a drawable curve -- mirrors /app's rank list
    // (`homeModelEntries`), which shows a model's rank regardless of chart
    // data. A curve-less entry must not vanish from the standings just
    // because it has nothing to plot.
    standings.push({
      key: entry.entry_id,
      name,
      // Two decimals, matching /app's rank rows and this card's own tooltip.
      ret: formatPercent(Number(entry.cumulative_return), 2),
      color: style.color,
    });

    const raw = times.map((t) => (t in byTime ? byTime[t] : null));
    const base = Number(entry.initial_equity) || raw.find((v) => v != null) || 10000;
    const values = raw.map((v) => (v == null ? null : (v - base) / base));
    if (!values.some((v) => v != null)) return;
    series.push({
      key: entry.entry_id,
      name,
      color: style.color,
      dash: style.dash,
      isBaseline: !isModel,
      values,
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
