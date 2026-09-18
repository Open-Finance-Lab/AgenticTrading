# Indicator fallback lookahead regression

`TechnicalIndicators.calculate_indicators` receives a complete backtest frame.
Previously, insufficient-history and library-failure paths broadcast the full
frame's close mean, minimum, or maximum to earlier rows. A later price could
therefore change the information available at an earlier decision timestamp.

For example, closes `[100, 110, 90]` produced SMA fallback values
`[100, 100, 100]`. At the second bar only 100 and 110 were available, so the
fallback average should be 105. The corrected values are `[100, 105, 100]`.
Fallback upper/lower bounds similarly use running maxima/minima. The final-row
aggregate is unchanged for finite prices, but earlier fallback values change.

## Verification

From the repository root with the project dependencies and pytest installed:

```sh
python -m pytest dashboard/backend/tests/backtesting/test_indicator_lookahead.py -q
```

The tests hold timestamps and frame length fixed and replace only the second
half of the close prices with alternating extreme highs and lows. All seven
indicator columns in the unchanged first half must match exactly, including
their missing-value positions. Histories span 10, 19, 20, 25, 26, 33, 34, 49,
50, and 80 bars. Cases exercise normal library behavior, missing results,
missing MACD/Bollinger columns, and exceptions before or after partial results.
A separate check compares normal 80-bar outputs directly with pandas-ta.
The fixtures are synthetic and require no market-data or model API calls.

The comparison approach is inspired by
[Freqtrade's lookahead-analysis](https://www.freqtrade.io/en/stable/lookahead-analysis/),
which compares indicators and signals across verification backtests. These are
original ATL unit tests, not a port of Freqtrade's command or trading engine.

## Scope

This removes future-price dependence from the aggregate fallback paths. It
preserves the existing choice of fallback versus library calculation, neutral
RSI/MACD defaults, and library warm-up NaNs. Running aggregates remain fallback
approximations, not fully initialized SMAs or Bollinger Bands.

It does not establish whole-platform freedom from lookahead, measure portfolio
returns, or prove restart equivalence. In particular, extending a frame across
an indicator's minimum-length threshold can still switch an earlier row from
a fallback to a library warm-up NaN; defining a uniform readiness policy is a
separate change.
