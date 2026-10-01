"""The indicator warm-up pad (#540): history fetched before ``start_date`` so
the indicators are warm on a backtest's first decision bar.

One owner for both surfaces that build indicator-enriched bars -- the dashboard
engine (``engine.HourlyBacktester``) and the shared protocol / ``/api/v2``
dataset (``market_data_store._build_dataset``). Two copies of this logic would
let two runs over one window see different features.

The pad is indicator input only. ``split_at_start`` separates it before
anything trades, values, gates or records on the frames, and
``warm_indicators`` hands back window rows alone.

Two ways a pad can be worse than none, both handled here and both recorded:

- **It can be thin.** A suspension, a recent listing or a provider that
  answered only part of the range leaves fewer bars than the longest lookback,
  and the first bars fall back to cold figures. ``warmup_evidence`` records
  the bars each symbol actually got, so a cold start never reads as a warm one.
- **It can be on a different price scale.** Prices are unadjusted on both
  feeds, so a split or 除权除息 date inside the pad puts pre-action closes into
  the window's sma50 and bands with nothing in the window's own corporate-action
  check (which sees window dates only) to say so. ``trim_at_unadjusted_gaps``
  drops pad bars before such a gap, trading warmth for a correct scale.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Dict, Mapping, Optional, Tuple

import pandas as pd

from dashboard.backend.domain.backtesting.features import (
    LONGEST_LOOKBACK_BARS,
    TechnicalIndicators,
)
from dashboard.backend.infrastructure.market_data.ifind_market_rules import (
    CORPORATE_ACTION_GAP,
)
from dashboard.backend.infrastructure.market_data.provider import parse_ymd
from dashboard.backend.infrastructure.market_data.sessions import canonical_market

Frames = Dict[str, pd.DataFrame]

#: Pad bars a symbol needs for its first window bar to be warm in every
#: column: that bar plus this many before it make the longest lookback.
PAD_BARS_FOR_WARM_FIRST_BAR = LONGEST_LOOKBACK_BARS - 1

#: An overnight close-to-close move past this is read as a price-scale break
#: (a split or ex-rights date), not a trade. A-share reuses the market-rule
#: audit's bound, which clears every daily limit band. The US has no band, so
#: its bound sits above all but the rarest real overnight moves while catching
#: a 3-for-2 split (-33%). A false positive only costs one symbol's warmth.
_GAP_THRESHOLDS = {"CN": CORPORATE_ACTION_GAP}
_DEFAULT_GAP_THRESHOLD = Decimal("0.30")


def gap_threshold(market: object) -> Decimal:
    return _GAP_THRESHOLDS.get(canonical_market(market), _DEFAULT_GAP_THRESHOLD)


def _market_dates(index: pd.DatetimeIndex, timezone: str):
    """Each stamp's date on the market's clock (a naive stamp is market-local)."""
    if index.tz is not None:
        index = index.tz_convert(timezone)
    return index.date


def split_at_start(
    frames: Mapping[str, pd.DataFrame], start_date: str, timezone: str
) -> Tuple[Frames, Frames]:
    """``(window, warmup)``: each frame cut at ``start_date``'s midnight on the
    market's clock.

    A symbol with no window bar is left out of ``window``, as a provider omits
    a symbol with no bars in the range it was asked for. Not copied: a boolean
    ``.loc`` already returns a new frame (with ``attrs``), and a second copy
    held both halves twice while the padded original was still referenced.
    """
    start = pd.Timestamp(parse_ymd(start_date))
    window: Frames = {}
    warmup: Frames = {}
    for symbol, frame in frames.items():
        index = frame.index
        boundary = start if index.tz is None else start.tz_localize(timezone)
        in_window = index >= boundary
        if in_window.any():
            window[symbol] = frame.loc[in_window]
        if not in_window.all():
            warmup[symbol] = frame.loc[~in_window]
    return window, warmup


def _daily_closes(frame: pd.DataFrame, timezone: str) -> pd.Series:
    closes = pd.to_numeric(frame["close"], errors="coerce")
    closes = pd.Series(closes.to_numpy(), index=_market_dates(frame.index, timezone))
    return closes.dropna().groupby(level=0).last()


def trim_at_unadjusted_gaps(
    warmup: Mapping[str, pd.DataFrame],
    window: Mapping[str, pd.DataFrame],
    *,
    market: object,
    timezone: str,
) -> Tuple[Frames, Dict[str, str]]:
    """Drop each symbol's pad bars from before its last price-scale break.

    Checked across the pad and into the window's first day, since an action
    effective on ``start_date`` puts the whole pad on the old scale. Returns the
    trimmed pad and ``{symbol: first date kept}`` for every symbol trimmed (a
    break on ``start_date`` drops that symbol's pad entirely).
    """
    threshold = float(gap_threshold(market))
    trimmed: Frames = {}
    trims: Dict[str, str] = {}
    for symbol, pad in warmup.items():
        if pad.empty or "close" not in pad.columns:
            trimmed[symbol] = pad
            continue
        closes = _daily_closes(pad, timezone)
        first_window = window.get(symbol)
        if first_window is not None and not first_window.empty:
            window_closes = _daily_closes(first_window, timezone)
            if not window_closes.empty:
                closes = pd.concat([closes, window_closes.iloc[:1]])
        moves = closes.pct_change().abs()
        breaks = moves[moves > threshold]
        if breaks.empty:
            trimmed[symbol] = pad
            continue
        cut: date = breaks.index[-1]
        keep = _market_dates(pad.index, timezone) >= cut
        if keep.any():
            trimmed[symbol] = pad.loc[keep]
        trims[symbol] = cut.isoformat()
    return trimmed, trims


def warm_indicators(window: pd.DataFrame, warmup: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Indicators for ``window``'s rows, read over ``warmup`` + ``window``.

    Every indicator is causal (``features.py``), so a window row reads only
    bars at or before it; the pad just fills its lookback.
    """
    if warmup is None or warmup.empty:
        return TechnicalIndicators.calculate_indicators(window)
    padded = TechnicalIndicators.calculate_indicators(pd.concat([warmup, window]))
    result = padded.iloc[len(warmup):].copy()
    # `concat` keeps attrs only when every input agrees; the session filter
    # reads the open-stamp convention off them.
    result.attrs = dict(window.attrs)
    return result


def warmup_evidence(
    *,
    fetch_start: str,
    warmup: Mapping[str, pd.DataFrame],
    window_symbols,
    trims: Mapping[str, str],
) -> Dict[str, Any]:
    """What the pad actually delivered, for ``agent_runs.metadata``.

    ``fetch_start`` alone says what was asked for; a pad that came back empty
    or thin looked identical to a full one. ``short_symbols`` names each symbol
    whose first window bar is still cold, with the pad bars it got.
    """
    counts = {symbol: len(warmup.get(symbol, ())) for symbol in window_symbols}
    short = {
        symbol: bars
        for symbol, bars in sorted(counts.items())
        if bars < PAD_BARS_FOR_WARM_FIRST_BAR
    }
    return {
        "fetch_start": fetch_start,
        "bars_for_warm_start": PAD_BARS_FOR_WARM_FIRST_BAR,
        "min_pad_bars": min(counts.values()) if counts else 0,
        "short_symbols": short,
        "unadjusted_gap_trims": dict(sorted(trims.items())),
    }


def describe_evidence(evidence: Mapping[str, Any]) -> Optional[str]:
    """One stdout line when the pad fell short, else None."""
    short = evidence.get("short_symbols") or {}
    trims = evidence.get("unadjusted_gap_trims") or {}
    if not short and not trims:
        return None
    parts = []
    if short:
        sample = ", ".join(f"{s} {n}" for s, n in list(short.items())[:5])
        parts.append(
            f"{len(short)} symbol(s) have under "
            f"{evidence.get('bars_for_warm_start')} pad bars, so their first "
            f"bars use cold indicators ({sample}{' …' if len(short) > 5 else ''})"
        )
    if trims:
        sample = ", ".join(f"{s} from {d}" for s, d in list(trims.items())[:5])
        parts.append(
            f"pad trimmed at an unadjusted price break for {len(trims)} "
            f"symbol(s) ({sample}{' …' if len(trims) > 5 else ''})"
        )
    return "   ⚠️  Indicator warm-up: " + "; ".join(parts)
