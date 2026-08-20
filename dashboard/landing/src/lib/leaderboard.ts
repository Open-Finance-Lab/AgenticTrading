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

export type BoardStanding = {
  key: string;
  name: string;
  ret: string;
  /** The number `ret` was formatted FROM, carried so ranking never has to read
   *  it back out of the display string. See the sort in `buildBoardData`. */
  cumulativeReturn: number;
  color: string;
  /** Carried so a consumer can tell "no model results came back" from "the
   *  board is fine". `BoardSeries` answers the same question with
   *  `isBaseline`; the two collections are not interchangeable -- a curve-less
   *  model reaches `standings` and never reaches `series`. */
  isModel: boolean;
};

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

/** `Number()` WITH THE ABSENT CASES MAPPED TO NaN INSTEAD OF ZERO.
 *
 *  `Number(null)`, `Number('')` and `Number([])` are all `0`, so a plain
 *  `Number(x)` -- and even a `Number.isFinite(Number(x))` guard after it -- reads
 *  a MISSING value as a real zero and hands it downstream as fact. On this
 *  payload that is not a rounding difference: a missing equity became a $0
 *  account and rendered as -100%, dragging the whole percent axis to the floor,
 *  and a missing `cumulative_return` published as a confident "0.00%" beside
 *  models whose numbers were real.
 *
 *  A genuine `0` still passes through as `0` -- an account really at zero is
 *  -100%, and dropping that would be the mirror-image lie. This only separates
 *  "absent" from "zero", which is the distinction `Number` throws away. */
export function finiteNumber(value: unknown): number {
  if (value === null || value === undefined || value === '') return NaN;
  const n = Number(value);
  return Number.isFinite(n) ? n : NaN;
}

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

/** `2026-04-15T14:00` -> `Apr 15`. Mirrors `formatShortDate` in
 *  dashboard/frontend/js/leaderboard.js:1096 exactly, including the
 *  empty-string case, the T-vs-date-only branch, and the invalid-date
 *  passthrough (returns the input unchanged rather than "Invalid Date") --
 *  the dashboard renders the byte-identical `timeKey` output as a date, so
 *  the hero's x-axis must format it the same way rather than print the raw
 *  ISO key. */
export function formatAxisDate(isoDay: string): string {
  if (!isoDay) return '';
  const d = new Date(String(isoDay).includes('T') ? isoDay : `${isoDay}T00:00:00`);
  if (Number.isNaN(d.getTime())) return isoDay;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

/** `2026-04-15T14:00` -> `Apr 15, 2:00 PM`. Mirrors `formatChartTooltipLabel`
 *  in dashboard/frontend/js/leaderboard.js:1103 exactly, including the
 *  branch on the `T` and the invalid-date passthrough.
 *
 *  A SECOND formatter, not a reuse of `formatAxisDate`, and the split is the
 *  same one /app makes: the axis drops the hour because an hourly series would
 *  otherwise repeat `Apr 15` seven times in a row, while the tooltip names ONE
 *  point and has to say which hour it is. That is also why the axis fix did not
 *  reach here: recharts renders the raw category value as the tooltip header
 *  (`tooltipTicks[activeIndex].value`) and `XAxis.tickFormatter` never touches
 *  it, so until this was bound as `labelFormatter` the hero printed the machine
 *  key `2026-04-15T14:00` directly above an axis correctly reading `Apr 15`. */
export function formatTooltipDate(isoStamp: string): string {
  if (!isoStamp) return '';
  const raw = String(isoStamp);
  const d = new Date(raw.includes('T') ? raw : `${raw}T00:00:00`);
  if (Number.isNaN(d.getTime())) return raw;
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
      // SKIPPED, NOT ZEROED. `Number(pt.equity) || 0` read every unparseable
      // equity -- null, undefined, a string, NaN -- as a $0 account, and the
      // base-relative transform below turns $0 into (0 - 10000) / 10000 = -100%.
      // One such point does not misplace one marker: `percentDomain` then spans
      // roughly [-1.12, hi], and the real board's -0.43%..+7.49% spread
      // collapses into a sliver at the top of the axis with a spike to the
      // floor. A missing point is exactly what the null-fill path below already
      // handles, so hand it one.
      //
      // `finiteNumber`, NOT a `Number.isFinite(Number(...))` guard: `Number(null)`
      // is 0, so the obvious guard passes the single most likely malformed shape
      // straight through. A REAL zero still gets through and still reads -100%,
      // which is what an account at zero actually did.
      const equity = finiteNumber(pt.equity);
      if (!Number.isFinite(equity)) return;
      byTime[key] = equity;
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
      ret: formatPercent(finiteNumber(entry.cumulative_return), 2),
      cumulativeReturn: finiteNumber(entry.cumulative_return),
      color: style.color,
      isModel,
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

  // SORTED ON THE NUMBER, NOT ON THE FORMATTED STRING. `parseFloat(b.ret)`
  // re-read the display text and carried two defects at once. Precision: `ret`
  // is `formatPercent(..., 2)`, so any two entries within 0.005 percentage
  // points compared EQUAL and kept payload order while Race printed them under
  // distinct `#n` ranks. And totality: `formatPercent` returns the em-dash for
  // a non-finite return, `parseFloat('—')` is NaN, and a comparator that
  // returns NaN leaves V8's sort order implementation-defined for the WHOLE
  // array -- one bad entry could scramble every rank and every chip on the
  // page rather than just misplacing itself.
  //
  // Hence the explicit -1/0/1 form rather than `b.cumulativeReturn -
  // a.cumulativeReturn`: subtraction reintroduces NaN for a non-finite entry,
  // and sinking those to -Infinity does not help, because -Infinity minus
  // -Infinity is NaN again. Non-finite returns rank last, together, in a
  // defined order.
  standings.sort((a, b) => {
    const av = Number.isFinite(a.cumulativeReturn) ? a.cumulativeReturn : -Infinity;
    const bv = Number.isFinite(b.cumulativeReturn) ? b.cumulativeReturn : -Infinity;
    if (av === bv) return 0;
    return bv > av ? 1 : -1;
  });
  return { times, series, standings, windowLabel: payload.window?.label || '' };
}

/** How many models ran, how many reference curves they were ranked against,
 *  and how many finished ahead of ALL of them.
 *
 *  NUMBERS ONLY, DELIBERATELY. The sentence Race builds from these stays in
 *  Race.tsx, because `test_no_landing_component_claims_brokered_or_real_capital_trading`
 *  scans `components/home/*.tsx` and not `lib/`, so copy moved in here would
 *  leave that scan -- the guard's own docstring names relocation as the one
 *  thing this class of copy reliably does.
 *
 *  Race hardcoded "Seven leading AI models ... Only one finished ahead of both"
 *  beside a table that is now live off the same payload. An eighth `llm_agent`
 *  entry in dashboard/config/leaderboard.json -- the documented way the roster
 *  reached seven -- left that sentence saying "Seven" above eight rows, and a
 *  re-run that put a second model ahead of buy-and-hold falsified the second
 *  half with the counter-evidence rendered directly beside it.
 *
 *  "Ahead of ALL of them" rather than "of both": `BOARD_BASELINE_IDS` has two
 *  entries today and the caller's copy says "both", but this function is not
 *  the right place to assume that number, and it reports `baselines` so the
 *  caller can. A board that resolved only one baseline must not be described
 *  with "both". */
export type BoardHeadlineCounts = { models: number; baselines: number; ahead: number };

export function boardHeadlineCounts(standings: BoardStanding[]): BoardHeadlineCounts {
  const rows = standings || [];
  let models = 0;
  let baselines = 0;
  let bestBaseline = -Infinity;
  for (const row of rows) {
    if (row.isModel) {
      models += 1;
      continue;
    }
    baselines += 1;
    if (Number.isFinite(row.cumulativeReturn) && row.cumulativeReturn > bestBaseline) {
      bestBaseline = row.cumulativeReturn;
    }
  }
  let ahead = 0;
  if (Number.isFinite(bestBaseline)) {
    for (const row of rows) {
      if (row.isModel && Number.isFinite(row.cumulativeReturn) && row.cumulativeReturn > bestBaseline) {
        ahead += 1;
      }
    }
  }
  return { models, baselines, ahead };
}

/** What a 200 actually delivered -- the question `status: "ready"` cannot
 *  answer.
 *
 *  `get_leaderboard` skips any strategy with no cached run
 *  (domain/leaderboard/service.py, `if not run: continue`) and still answers
 *  200, so `entries: []` and "only the baselines resolved" are ordinary
 *  SUCCESSFUL responses. Without this the hero drew its entire frame over an
 *  empty payload -- a percent axis labelled -5.0%..5.0% off `percentDomain`'s
 *  hardcoded fallback, a scale no run produced, under the axis arrow, the
 *  title, the window chip and a caption naming the competition window -- and
 *  Race drew its Rank/AI model/Return header over zero rows. "The upstream
 *  returned nothing" and "everything is fine" rendered identically, which is
 *  the failure class CLAUDE.md's fail-closed-is-not-fail-visible section is
 *  about.
 *
 *  DO NOT answer this by substituting curves. An invented dataset under any
 *  name is the bug this whole module exists to remove; the fix is to say the
 *  board is empty. */
export type BoardCoverage = 'empty' | 'baselines-only' | 'full';

/** For the CHART, which draws `series`. */
export function chartCoverage(series: BoardSeries[]): BoardCoverage {
  if (!series.length) return 'empty';
  if (series.every((s) => s.isBaseline)) return 'baselines-only';
  return 'full';
}

/** For the TABLE, which lists `standings`.
 *
 *  A separate rule rather than a shared one, because the two collections
 *  genuinely disagree: a model with `equity_curve: []` stands in `standings`
 *  and never enters `series`, so a board that is `baselines-only` to the chart
 *  can be `full` to the table. Collapsing them would make Race announce a
 *  board with no models while listing models. */
export function standingsCoverage(standings: BoardStanding[]): BoardCoverage {
  if (!standings.length) return 'empty';
  if (!standings.some((s) => s.isModel)) return 'baselines-only';
  return 'full';
}

export const BOARD_ERROR_UNREACHABLE = 'The board service could not be reached.';
export const BOARD_ERROR_UNREADABLE =
  'The board service sent a response this page could not read.';

/** EVERY MESSAGE THAT ESCAPES THIS FUNCTION IS ALREADY VISITOR-FACING, which is
 *  what lets `classifyFetchFailure` keep showing `err.message` verbatim.
 *
 *  Two sources of raw transport text reached the card before this. The
 *  browser's own rejection is one: a dead backend rejects `fetch` with
 *  `TypeError: Failed to fetch`, which the hero rendered in `font-mono` and
 *  Race rendered mid-sentence. The response body is the other, and it is the
 *  worse of the two -- `res.json()` ran on ANY 2xx, so a Vercel or Render HTML
 *  error page delivered with a 2xx status made the public failure message
 *  `Unexpected token '<', "<!DOCTYPE "... is not valid JSON`.
 *
 *  AbortError is re-thrown rather than mapped: the provider's 45s ceiling
 *  aborts this same signal, and `classifyFetchFailure` needs that distinction
 *  to say "timed out" instead of "unreachable". Mapping it here would erase the
 *  one thing that branch reads.
 *
 *  Root-relative, with no origin anywhere in it. dashboard/frontend/vercel.json
 *  rewrites /api/:path* to Render, and test_frontend_api_base.py requires an
 *  EMPTY production base for exactly that reason -- it calls a hardcoded Render
 *  origin a same-origin cookie auth regression. MarketTicker.tsx's apiBase()
 *  survives that guard only because it excludes minified assets/. This path is
 *  correct under Vercel and under local uvicorn alike. (Under `npm run dev` at
 *  :5173 it hits the Vite server and fails -- but so does apiBase(), which
 *  returns the dev server's own origin there. Neither pattern serves the dev
 *  server.) */
export async function fetchLeaderboard(signal: AbortSignal): Promise<BoardData> {
  let res: Response;
  try {
    res = await fetch('/api/v1/leaderboard?period=contest', { signal });
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') throw err;
    throw new Error(BOARD_ERROR_UNREACHABLE);
  }
  if (!res.ok) throw new Error(`The board service returned HTTP ${res.status}.`);
  const contentType = res.headers?.get('content-type') || '';
  if (!contentType.includes('json')) throw new Error(BOARD_ERROR_UNREADABLE);
  let payload: { entries?: LeaderboardEntry[]; window?: { label?: string } };
  try {
    payload = await res.json();
  } catch {
    throw new Error(BOARD_ERROR_UNREADABLE);
  }
  return buildBoardData(payload || {});
}
