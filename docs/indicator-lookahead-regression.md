# Indicator look-ahead regression

`TechnicalIndicators.calculate_indicators` receives a complete backtest frame,
so anything computed from the frame as a whole can reach back into earlier
decision timestamps. Two such leaks existed.

**Future prices.** The insufficient-history and library-failure paths broadcast
the full frame's close mean, minimum, or maximum to every row. Closes
`[100, 110, 90]` produced an SMA fallback of `[100, 100, 100]`, although at the
second bar only 100 and 110 had been seen (105).

**Future bar count.** Whether a row got a fallback, a library value, or a
library warm-up NaN was decided by `len(df)`, i.e. by how many bars followed
it. pandas-ta adds to this: it returns `None` below a minimum frame length
(15 bars for RSI(14), 34 for MACD(12/26/9), 20 for BB(20) and SMA20, 50 for
SMA50) yet back-fills early rows of a longer frame (RSI from bar 2, the MACD
line from bar 26). A 49-bar US run therefore showed an SMA50 on every bar while
a 50-bar run with the same start showed NaN, which the LLM prompt renders as
`0.0`, for its first 49.

## The rule now

A row takes the library value only once its own prefix is at least the
library's minimum length, so the answer for row *i* is the same whether the
frame ends at bar *i* or a year later. Paired columns (MACD line and signal,
upper and lower band) switch together, so a real MACD line is never compared
against a placeholder signal. Every other row uses a causal fallback:

| Column | Fallback |
|---|---|
| `rsi_14` | 50 (neutral) |
| `macd`, `macd_signal` | 0 (neutral) |
| `sma20`, `sma50` | mean of the usable closes in the 20- or 50-bar window so far |
| `bb_upper`, `bb_lower` | that 20-bar mean ± 2 sample standard deviations |

The SMA and band fallbacks are the library's own formula with
`min_periods=1`: during warm-up they average every bar seen so far, and once a
window is full they equal the library value. So a row the library leaves empty
after warm-up (an unusable close inside its window) or a library error still
gets a 20- or 50-bar figure, not the mean of all history. The band is
zero-width only where its window holds one close. It replaced a running
max/min, which always contained the current close and so placed a rising price
at the upper band on every bar.

A library error keeps the groups computed before it and falls back for the
rest, with the same values as above. Non-numeric and non-finite closes count as
missing; SMA and band columns are NaN only on rows whose window holds no usable
close. pandas-ta runs on a positional index, since its MACD slices by label and
a repeated timestamp would otherwise raise and push every later group onto its
fallback.

## Verification

From the repository root with the project dependencies and pytest installed:

```sh
python -m pytest dashboard/backend/tests/backtesting/test_indicator_lookahead.py -q
```

- **Prefix invariance.** For a 60-bar series, every prefix of 1–60 bars must
  reproduce the full frame's rows (to 1e-12, since pandas-ta's rolling mean
  differs in the last bit with frame length).
- **Future perturbation.** At fixed lengths from 10 to 80 bars, replacing the
  second half with alternating extreme highs and lows must leave the first half
  bit-identical.
- Both run under normal library behavior, missing results, missing MACD/BB
  columns, and an exception raised early (`rsi`) or late (`bbands`). Each mode
  asserts the patched function was actually called and whether the error
  handler ran.
- Separate checks pin the fallback values, that an error falls back exactly like
  a missing library and keeps the groups computed before it, the band's
  behavior on a rising series, that fallbacks reproduce the library once a
  window is full, that ready rows match pandas-ta, that an unusable close
  mid-series keeps windowed fallbacks, that repeated timestamps leave the
  library path intact, and that unusable closes never raise.

The fixtures are synthetic and need no market-data or model API calls. The
comparison approach is inspired by
[Freqtrade's lookahead-analysis](https://www.freqtrade.io/en/stable/lookahead-analysis/);
these are original ATL unit tests, not a port of Freqtrade's command or engine.

## Scope

This covers the indicator calculator only. It does not establish whole-platform
freedom from look-ahead, measure portfolio returns, or prove restart
equivalence. The engine still loads bars from `start_date` with no earlier
warm-up history, so the first bars of every run use fallbacks; loading a
look-back window before the start date is a separate change.

Early bars change as a result. Rows that used to carry a library warm-up NaN
now carry the fallback, for every consumer:

- **LLM prompts** and **external agents** (v1/v2 step snapshots) used to see
  `0.0` there; a client treating `sma50 == 0` as "not warmed up" loses that
  signal.
- **The rule-based reference agent** skips rows whose RSI or SMA20 is NaN, so
  it now evaluates those rows too. Its RSI < 30 entry still cannot fire before
  bar 15, but its `price > sma50 * 1.02` exit can now fire before bar 50.
- **Leaderboard rows** cached before this change keep the old indicators until
  a forced refresh, so a partial refresh would rank old and new curves side by
  side.
