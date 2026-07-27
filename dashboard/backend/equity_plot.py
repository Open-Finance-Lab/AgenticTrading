"""Gapless equity-curve PNG renderer.

Plot layout matches ``docs/examples/simple_trading_agent_backtest.py`` →
``plot_results()`` (title, baselines, gapless ET time axis). Colors follow the
Playground theme in ``chart_style.py``.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import pandas as pd
import pytz
from matplotlib.figure import Figure
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, NullFormatter

from dashboard.backend.chart_style import PLAYGROUND_THEME, series_color
from dashboard.backend.domain.leaderboard.strategies._yahoo import fetch_index_hourly

_ET = pytz.timezone("US/Eastern")
_DEFAULT_MARKET_TIMEZONE = "US/Eastern"
_HOUR_WIDTH = 1.0 / 24.0
DJIA_INDEX = "^DJI"
NASDAQ_100_INDEX = "^NDX"


def parse_equity_timestamp(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _to_et(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=pytz.UTC)
    return ts.astimezone(_ET)


def _to_market_timezone(
    ts: datetime, market_timezone: str = _DEFAULT_MARKET_TIMEZONE
) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=pytz.UTC)
    return ts.astimezone(pytz.timezone(market_timezone))


def is_market_hour(ts: datetime) -> bool:
    t = _to_et(ts)
    if t.weekday() >= 5:
        return False
    minutes = t.hour * 60 + t.minute
    return 9 * 60 + 30 <= minutes <= 16 * 60


def compute_index_baseline_values(
    index_symbol: str,
    timestamps: Sequence[datetime],
    start_date: str,
    end_date: str,
    initial_capital: float,
) -> Optional[List[float]]:
    """Scale a Yahoo index (^DJI, ^NDX, …) to ``initial_capital`` on the agent timeline."""
    points = fetch_index_hourly(index_symbol, start_date, end_date)
    if not points or not timestamps:
        return None

    idx = pd.DatetimeIndex([p[0] for p in points], tz="UTC")
    levels = pd.Series([p[1] for p in points], index=idx).sort_index()
    levels = levels[[is_market_hour(ts.to_pydatetime()) for ts in levels.index]]
    if levels.empty:
        return None

    ts_idx = pd.DatetimeIndex(list(timestamps))
    if ts_idx.tz is None:
        ts_idx = ts_idx.tz_localize("UTC")
    else:
        ts_idx = ts_idx.tz_convert("UTC")

    aligned = levels.reindex(ts_idx, method="nearest", tolerance=pd.Timedelta("30min"))
    if aligned.isna().any():
        aligned = aligned.ffill().bfill()
    if aligned.isna().any():
        return None

    base = float(aligned.iloc[0])
    if not base:
        return None
    return [float(initial_capital * (value / base)) for value in aligned]


def market_index_baselines_for_run(
    timestamps: Sequence[datetime],
    start_date: str,
    end_date: str,
    initial_capital: float,
) -> List[Tuple[str, str, List[float]]]:
    """DJIA + Nasdaq-100 index baselines (same pair as simple_trading_agent_backtest.py)."""
    baselines: List[Tuple[str, str, List[float]]] = []
    for label, symbol in (("DJIA index", DJIA_INDEX), ("Nasdaq-100", NASDAQ_100_INDEX)):
        values = compute_index_baseline_values(
            symbol, timestamps, start_date, end_date, initial_capital
        )
        if values:
            baselines.append((label, f"index:{symbol}", values))
    return baselines


def gapless_market_axis(
    timestamps: Sequence[datetime],
    market_timezone: str = _DEFAULT_MARKET_TIMEZONE,
) -> Tuple[List[float], List[datetime]]:
    """Map market datetimes to gapless matplotlib x coords (1 market hour = 1h wide)."""
    if not timestamps:
        return [], []
    ts_local = [_to_market_timezone(ts, market_timezone) for ts in timestamps]
    origin = mdates.date2num(ts_local[0])
    x = [origin + i * _HOUR_WIDTH for i in range(len(ts_local))]
    return x, ts_local


def equity_lookup(curve: Sequence[Dict[str, Any]]) -> Dict[datetime, float]:
    out: Dict[datetime, float] = {}
    for point in curve:
        try:
            out[parse_equity_timestamp(point["timestamp"])] = float(point["equity"])
        except Exception:
            continue
    return out


def align_equity(reference: Sequence[datetime], lookup: Dict[datetime, float]) -> List[float]:
    """Align a baseline curve to the agent run's market-hour timestamps."""
    values: List[float] = []
    last: Optional[float] = None
    for ts in reference:
        val = lookup.get(ts)
        if val is None:
            ts_naive = ts.replace(microsecond=0)
            for key, candidate in lookup.items():
                if key.replace(microsecond=0) == ts_naive:
                    val = candidate
                    break
        if val is not None:
            last = val
        if last is None:
            raise ValueError("baseline curve missing equity for agent timestamps")
        values.append(last)
    return values


def resolve_agent_chart_label(
    agent_name: Optional[str],
    llm_model: Optional[str] = None,
    card_name: Optional[str] = None,
) -> str:
    """Human-readable legend label for an agent equity series.

    Prefer the My Agents card name (``card_name``), then a non-generic run
    ``agent_name``. ``llm_model`` is ignored for display — model ids belong in
    metrics, not the chart legend.
    """
    _ = llm_model  # kept for call-site compatibility
    label = (card_name or "").strip()
    if label:
        return label
    name = (agent_name or "").strip()
    if name and name.lower() != "agent":
        return name
    return name or "Agent"


def gapless_chart_x_labels(
    timestamps: Sequence[datetime],
    market_timezone: str = _DEFAULT_MARKET_TIMEZONE,
) -> List[str]:
    """Chart.js x labels: YYYY-MM-DD on the first bar of each trading day."""
    if not timestamps:
        return []
    ts_local = [_to_market_timezone(ts, market_timezone) for ts in timestamps]
    labels = [""] * len(ts_local)
    i = 0
    while i < len(ts_local):
        labels[i] = ts_local[i].strftime("%Y-%m-%d")
        day = ts_local[i].date()
        i += 1
        while i < len(ts_local) and ts_local[i].date() == day:
            i += 1
    return labels


def build_backtest_chart_data(
    *,
    run_id: str,
    agent_name: str,
    llm_model: Optional[str],
    start_date: str,
    end_date: str,
    initial_capital: float,
    agent_curve: Sequence[Dict[str, Any]],
    card_name: Optional[str] = None,
    stored_baselines: Sequence[
        Tuple[str, str, Sequence[Dict[str, Any]]]
    ] = (),
    include_market_indexes: bool = True,
    market_timezone: str = _DEFAULT_MARKET_TIMEZONE,
) -> Dict[str, Any]:
    """JSON chart payload for the Playground backtest page (matches plot.png baselines)."""
    timestamps, agent_values = curve_timestamps_and_values(agent_curve)
    if not timestamps:
        raise ValueError("No equity data to plot for this run")

    label = resolve_agent_chart_label(agent_name, llm_model, card_name)
    series: List[Dict[str, Any]] = [
        {
            "run_id": run_id,
            "label": label,
            "values": list(agent_values),
            "color": series_color(run_id, label),
            "dashed": False,
        }
    ]

    baseline_values: List[Tuple[str, str, List[float]]] = []
    for bl_label, bl_run_id, bl_curve in stored_baselines:
        baseline_values.append(
            (bl_label, bl_run_id, align_equity(timestamps, equity_lookup(bl_curve)))
        )

    if include_market_indexes:
        baseline_values.extend(
            market_index_baselines_for_run(
                timestamps, start_date, end_date, initial_capital
            )
        )

    for bl_label, bl_run_id, bl_values in baseline_values:
        series.append(
            {
                "run_id": bl_run_id,
                "label": bl_label,
                "values": list(bl_values),
                "color": series_color(bl_run_id, bl_label),
                "dashed": True,
            }
        )

    return {
        "agent_run_id": run_id,
        "timestamps": [t.isoformat() for t in timestamps],
        "x_labels": gapless_chart_x_labels(timestamps, market_timezone),
        "series": series,
    }


def curve_timestamps_and_values(
    curve: Sequence[Dict[str, Any]],
) -> Tuple[List[datetime], List[float]]:
    timestamps: List[datetime] = []
    values: List[float] = []
    for point in curve:
        try:
            timestamps.append(parse_equity_timestamp(point["timestamp"]))
            values.append(float(point["equity"]))
        except Exception:
            continue
    return timestamps, values


def render_backtest_equity_png(
    *,
    agent_label: str,
    agent_run_id: str,
    timestamps: Sequence[datetime],
    agent_values: Sequence[float],
    baselines: Sequence[Tuple[str, str, Sequence[float]]],
    market_timezone: str = _DEFAULT_MARKET_TIMEZONE,
    title: str = "Trading Performance",
    xlabel: str = "Date",
    ylabel: str = "Portfolio value ($)",
) -> bytes:
    """Render agent vs baseline curves using the gapless market-hour x axis."""
    if not timestamps or not agent_values:
        raise ValueError("No equity data to plot")

    theme = PLAYGROUND_THEME
    x, ts_local = gapless_market_axis(timestamps, market_timezone)

    fig = Figure(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor(theme["figure_bg"])
    ax = fig.add_subplot(111)
    ax.set_facecolor(theme["axes_bg"])

    ax.plot(
        x,
        list(agent_values),
        label=agent_label,
        color=series_color(agent_run_id, agent_label),
        linewidth=theme["line_width"],
        linestyle="-",
    )
    for label, run_id, values in baselines:
        ax.plot(
            x,
            list(values),
            label=label,
            color=series_color(run_id, label),
            linewidth=theme["line_width"],
            linestyle="--",
        )

    ax.set_title(title, color=theme["title"], fontsize=12, pad=12)
    ax.set_ylabel(ylabel, color=theme["label"], fontsize=10)
    ax.set_xlabel(xlabel, color=theme["label"], fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="y", colors=theme["tick"], labelsize=9)
    ax.tick_params(axis="x", colors=theme["tick"])

    day_ticks, day_labels = [], []
    i = 0
    while i < len(ts_local):
        j = i
        while j < len(ts_local) and ts_local[j].date() == ts_local[i].date():
            j += 1
        # Anchor each date at the first market bar of that day (not the midpoint).
        day_ticks.append(x[i])
        day_labels.append(ts_local[i].strftime("%Y-%m-%d"))
        i = j

    ax.xaxis.set_major_locator(FixedLocator(day_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(day_labels))
    ax.xaxis.set_minor_locator(FixedLocator(x))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="major", length=6, pad=8, colors=theme["tick"])
    ax.tick_params(axis="x", which="minor", length=3, colors=theme["tick"])

    for label in ax.xaxis.get_majorticklabels():
        label.set_rotation(0)
        label.set_ha("left")
        label.set_color(theme["tick"])

    for spine in ax.spines.values():
        spine.set_color(theme["spine"])
    ax.grid(True, alpha=0.3, axis="y", color=theme["grid"])

    legend = ax.legend(
        loc="upper left",
        fontsize=9,
        facecolor=theme["legend_bg"],
        edgecolor=theme["legend_edge"],
    )
    for text in legend.get_texts():
        text.set_color(theme["legend_text"])

    fig.subplots_adjust(bottom=0.14)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()
