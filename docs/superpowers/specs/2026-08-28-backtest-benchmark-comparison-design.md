# Backtest Benchmark Comparison and Trading Log Layout Design

**Date:** 2026-08-28

**Status:** Approved
**Scope:** Backtest performance comparison and Trading Log presentation

## Goal

Make the Backtest page answer two questions without requiring the user to
estimate values from chart lines:

1. Did the selected Agent outperform the relevant benchmarks?
2. Which orders explain the Agent's result?

The approved desktop layout keeps Run config on the left, places the portfolio
value chart and benchmark metric table in the center, and moves Trading Log to
an independently scrolling right rail.

## Current problem

The chart already plots the Agent against DJIA and Nasdaq-100 for supported US
market runs, but Trading Performance Summary reports metrics for the Agent
only. A user can see that the curves differ but cannot scan exact Final Value,
Total Return, Max Drawdown, or Sharpe Ratio values across the series.

Trading Log currently uses an eight-column table below the chart. It consumes
vertical space and separates the decisions from the performance result they
produced. The table's minimum width also makes it unsuitable for the approved
right-rail position.

US backtests persist a paired Buy & Hold run, but the current chart-data route
includes that stored baseline only when market index baselines are disabled.
For a US run, the response therefore contains the Agent, DJIA, and Nasdaq-100,
but omits the available Buy & Hold series.

## Product decisions

1. Supported US runs compare Your Agent, DJIA, Nasdaq-100, and Buy & Hold.
2. The comparison table contains Final Value, Total Return, Max Drawdown, and
   Sharpe Ratio. Annualized Return and Annualized Volatility are not added.
3. The chart and comparison table form one performance surface. The table sits
   directly below the chart and uses the same series names and colors.
4. Trading Log moves to a right rail and changes from a wide table to a
   vertical activity feed without dropping existing order information.
5. Run config remains the left rail on wide desktop viewports.
6. Chart/comparison and Trading Log load independently so a failure in one does
   not erase the other.

Annualized Return is deliberately excluded because short backtests can make an
annualized projection look much more certain than the observed period warrants.
The selected four metrics describe the actual run and match the metrics already
shown for the Agent.

## Approved layout

### Wide desktop

At viewports at least 1280 CSS pixels wide, the Backtest content uses three
columns:

- a 220-250 pixel Run config rail;
- a flexible performance column with a practical minimum width of 620 pixels;
  and
- a 320-360 pixel Trading Log rail.

The performance card contains, in order:

1. the existing title, market-data badge, run selector, and range controls;
2. a DOM legend and the portfolio value chart; and
3. the Performance comparison table.

The former right-side Trading Performance Summary is removed. Its Agent-only
values are represented in the highlighted Your Agent column of the comparison
table.

Trading Log aligns with the top of Trading Performance and has an independent
vertical scrollbar. Its header and filter stay visible while the event feed
scrolls. The rail must not force the entire page to grow with a long run.

### Narrow desktop and tablet

Below 1280 CSS pixels, Trading Log moves below the performance card. Run config
follows the page's existing responsive behavior. The comparison table keeps all
available series and uses contained horizontal scrolling when its columns
cannot fit; it must not shrink values or headings until they overlap.

### Mobile

On small screens, Run config, Trading Performance, Performance comparison, and
Trading Log use one column. The chart keeps a stable minimum height. The
comparison table scrolls horizontally inside its own region, and the Trading
Log feed expands naturally instead of using a fixed-height nested scroller.

## Performance comparison

### Columns and order

The frontend normalizes available series into this canonical order regardless
of payload order:

1. Your Agent;
2. DJIA;
3. Nasdaq-100; and
4. Buy & Hold.

The Agent column receives a restrained blue background and remains the visual
anchor. Benchmark columns use the same swatches as their chart series. A value
that is best for its metric receives a textual `Best` marker as well as color,
so color is never the only signal.

Legend controls change chart-line visibility only. The comparison table keeps
all available series visible so hiding a visually noisy line cannot also hide
its exact benchmark result.

For Final Value, Total Return, and Sharpe Ratio, the largest finite value is
best. For Max Drawdown, the value closest to zero is best. Exact ties at the raw
numeric value may mark every tied series. Missing values never participate in
best-value selection.

### Metric formulas

Every column is calculated from the aligned `values` array in the same
chart-data response. This keeps the Agent and benchmark calculations on the
same timestamps and prevents a stored Agent metric from being compared with a
differently sampled benchmark metric.

For a cleaned finite value sequence `v`:

- `Final Value = v[last]`.
- `Total Return = (v[last] / v[first] - 1) * 100`.
- `Max Drawdown = min(v[i] / running_peak[i] - 1) * 100`.
- `Sharpe Ratio = mean(period_returns) / population_stddev(period_returns) *
  sqrt(252 * 6.5)`, with a zero risk-free rate, matching the existing hourly
  Backtest convention.

At least two finite portfolio values are required for Total Return and Max
Drawdown. Sharpe Ratio additionally requires at least two finite period returns
and non-zero population standard deviation. An undefined metric renders as the
unavailable marker rather than zero.

Currency values use the selected run's reporting currency and existing locale
formatting. Percentages use two decimal places. Sharpe Ratio uses two decimal
places. Calculations keep full precision; formatting occurs only at render
time.

### Relative Agent result

The Your Agent cell in the Total Return row includes one compact delta for each
available benchmark:

```text
Agent total return - benchmark total return
```

The delta is expressed in percentage points (`pp`), not percent. Positive
deltas use the existing success treatment, negative deltas use the existing
danger treatment, and exact zero uses neutral text. A missing benchmark produces
no fabricated delta.

## Chart-data contract

The existing authenticated endpoint remains:

```text
GET /api/backtest/{run_id}/chart-data
```

No new database table, migration, or endpoint is required. The response keeps
its current `agent_run_id`, `timestamps`, `x_labels`, `series`, and
`index_baselines_ok` fields.

For supported US runs, the route must pass the paired stored Buy & Hold curve
to `build_backtest_chart_data` even when market index baselines are enabled.
The expected successful series set is therefore:

```text
Agent + DJIA index + Nasdaq-100 + buy-and-hold
```

The frontend identifies series by stable run identifiers first:

- `agent_run_id` identifies Your Agent;
- `index:^DJI` identifies DJIA;
- `index:^NDX` identifies Nasdaq-100; and
- the selected run's `baseline_buyhold_run_id` identifies Buy & Hold.

Existing labels remain a backward-compatible fallback for older cached
payloads. Display labels are normalized to `Your Agent`, `DJIA`, `Nasdaq-100`,
and `Buy & Hold`; internal agent names and run ids are not exposed as headings.

For a market profile where US index baselines are not applicable, such as an
iFinD China A-share run, the page shows only the applicable Agent and Buy & Hold
columns. It must not imply that DJIA or Nasdaq-100 is a valid benchmark for that
market.

## Trading Log activity feed

The existing run-scoped order endpoint and normalized frontend record model
remain the data source. Only presentation changes.

Each event exposes the same information currently available in the table:

- action, symbol, and company or asset name;
- timestamp;
- requested and filled quantity;
- execution price and total value;
- order status;
- reporting-currency and native-currency details when present;
- transaction-cost details when present; and
- the complete decision reason.

The primary row contains the action badge, asset, timestamp, and status. A
secondary metadata row contains quantity, price, total value, and optional cost
details. The reason appears below the metadata and wraps within the rail. Long
content may wrap to additional lines but must not be silently discarded.

The existing All Orders, Buys Only, and Sells Only filtering behavior remains.
The header also reports the visible order count and settled-status summary when
those values are available from the normalized records.

Empty, loading, truncated, and error messages occupy the feed body and do not
change the rail width. The existing truncation notice remains explicit when the
backend limits the number of returned records.

## Loading and partial error behavior

Chart/comparison and Trading Log are independent render regions:

- While chart-data loads, the chart and comparison table show a stable loading
  state; Trading Log may render as soon as its request completes.
- If chart-data fails, Trading Log remains usable and the performance region
  shows a retryable error without stale values from another run.
- If Trading Log fails, the chart and comparison remain usable and only the
  right rail shows its retryable error.
- Switching the selected run clears or marks both regions as loading before
  new results arrive, and late responses for the previous run are ignored.

When `index_baselines_ok` is `false`, the Agent and available Buy & Hold data
remain visible. DJIA and Nasdaq-100 stay represented as unavailable columns so
the provider outage is explicit, and the existing benchmark notice remains
visible. An unavailable index metric never receives a `Best` marker or Agent
delta.

If a paired Buy & Hold run id or curve is absent for a historical run, the Buy
& Hold column renders unavailable with concise context. The frontend does not
substitute another run, another date range, or synthetic values.

## Accessibility and keyboard behavior

- The comparison uses a semantic table with column and row headers.
- The DOM legend uses native buttons with visible focus and `aria-pressed` to
  toggle series. Swatches are accompanied by text labels.
- The chart has an accessible name and references the comparison table as its
  exact-value alternative.
- The Trading Log is a semantic list of order events. Its scroll region is
  keyboard focusable on wide desktop and has an accessible name.
- The order filter has an associated accessible label and remains keyboard
  operable as a native select.
- Loading and error changes use the page's existing live-region conventions
  without repeatedly announcing chart hover updates.
- Focus order follows visual order: Run config, performance controls, legend,
  comparison, Trading Log filter, and Trading Log events.
- Positive, negative, best, action, and status states never rely on color
  alone.

## Security and privacy

The existing session-scoped route and run ownership checks remain unchanged.
The comparison exposes only portfolio values already returned by chart-data.
Trading Log continues to render the existing safe order payload and must use
the current escaping and DOM construction helpers for user- or model-provided
text.

Tests and fixtures use synthetic run ids, equity curves, trades, prices, and
reasons. Real API keys, local databases, `.superpowers/`, and `work/` are not
committed.

## Test coverage

Focused backend tests cover:

- US chart-data returning Agent, DJIA, Nasdaq-100, and the paired Buy & Hold
  series;
- non-US chart-data retaining only applicable Agent and Buy & Hold series;
- index-provider failure retaining Agent and Buy & Hold while setting
  `index_baselines_ok` to `false`;
- a missing or inaccessible paired Buy & Hold curve not substituting another
  run; and
- session ownership remaining enforced for the selected Agent run.

Focused frontend contract tests cover:

- canonical series identification and ordering independent of payload order;
- Final Value, Total Return, Max Drawdown, and Sharpe formulas for all series;
- flat, missing, non-finite, and insufficient series producing unavailable
  metrics instead of fabricated zeroes;
- percentage-point Agent deltas and best-value selection, including ties;
- partial index and Buy & Hold failure presentation;
- run-switch stale-response protection;
- DOM legend labels, `aria-pressed`, table headers, and focusable log region;
- vertical Trading Log event rendering without loss of quantity, price, value,
  status, currency, cost, reason, empty, error, or truncation information; and
- wide, narrow, and mobile layout contracts without overlapping text or page-
  level horizontal overflow.

The implementation is also visually checked at representative 1440, 1024, and
390 CSS-pixel viewports. The checks verify that the chart is nonblank, the
comparison columns remain readable, the right rail scrolls independently on
wide desktop, and the stacked layouts do not overlap.

## Non-goals

- Adding Annualized Return, Annualized Volatility, alpha, beta, or tracking
  error metrics.
- Changing backtest execution, trade generation, market data, transaction
  costs, or stored run metrics.
- Adding a database table, migration, or historical backfill.
- Comparing unrelated runs, models, date ranges, or initial capital amounts.
- Making DJIA or Nasdaq-100 appear for markets where those indexes are not an
  applicable baseline.
- Redesigning Run config, Backtest launch, Paper Trading, or Admin Analytics.
- Changing Trading Log filters or backend pagination limits.

## Acceptance criteria

1. A supported US Backtest displays Your Agent, DJIA, Nasdaq-100, and Buy & Hold
   on the chart and in one exact-value comparison table.
2. Each available series has Final Value, Total Return, Max Drawdown, and Sharpe
   Ratio calculated from the same aligned chart-data timestamps and values.
3. The Agent Total Return cell states its percentage-point difference from
   every available benchmark.
4. The best value in each metric is identified with text as well as color, and
   unavailable values are never treated as zero or best.
5. On a viewport at least 1280 CSS pixels wide, Run config is left, performance
   is center, Trading Log is right, and Trading Log scrolls independently.
6. On narrower viewports, Trading Log moves below performance and no text,
   controls, comparison values, or order details overlap.
7. Every existing Trading Log field, filter, empty state, error state, and
   truncation notice remains available in the vertical feed presentation.
8. Chart/comparison failure does not erase Trading Log, and Trading Log failure
   does not erase chart/comparison.
9. Index-provider or Buy & Hold partial failure is explicit and never replaced
   with a mismatched or fabricated baseline.
10. Keyboard users can operate range controls, legend toggles, order filtering,
    and the wide-screen log scroller with a visible focus indicator.
11. No response, fixture, commit, or PR exposes a real secret or local database.
