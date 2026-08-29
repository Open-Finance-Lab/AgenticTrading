"""Shared backtest chart styling aligned with the Playground Chart.js plot.

Source of truth for colors/labels: ``dashboard/frontend/app.js`` →
``initializeCharts()`` (``colorMap`` + ``formatTimestamps``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

# Keep in sync with app.js initializeCharts → colorMap
PLAYGROUND_SERIES_COLORS = {
    "my-algo": "#fbbf24",
    "external-agent": "#4FC3F7",
    "agent": "#4FC3F7",
    "djia": "#F5C04A",
    "djia-index": "#F5C04A",
    "nasdaq-100": "#9AA4B2",
    "buy-and-hold": "#34D399",
}

PLAYGROUND_THEME = {
    "figure_bg": "#0a0e27",   # --bg-primary
    "axes_bg": "#131a35",     # --bg-card (chart panel)
    "grid": "#1f2937",
    "spine": "#1f2937",
    "tick": "#e5e7eb",
    "title": "#e5e7eb",
    "label": "#e5e7eb",
    "legend_bg": "#131a35",
    "legend_edge": "#1f2937",
    "legend_text": "#e5e7eb",
    "note": "#f59e0b",       # amber: degraded-render caption (missing baselines)
    "line_width": 2.5,
}


def series_kind(run_id: str, agent_name: Optional[str]) -> str:
    """Map a run to the Playground chart series key."""
    rid = run_id or ""
    if rid.startswith("algo_"):
        return "my-algo"
    if rid.startswith("ext_"):
        return "external-agent"
    name = (agent_name or "").strip().lower()
    if name in {"djia", "djia index"}:
        return "djia-index"
    if name in {"nasdaq-100", "nasdaq 100"}:
        return "nasdaq-100"
    if name in {"buy-and-hold", "buy & hold"}:
        return "buy-and-hold"
    return "agent"


def series_color(run_id: str, agent_name: Optional[str]) -> str:
    return PLAYGROUND_SERIES_COLORS[series_kind(run_id, agent_name)]


def format_playground_timestamp(ts: datetime) -> str:
    """Match app.js formatTimestamps: ``May 2`` (month short + day)."""
    return f"{ts.strftime('%b')} {ts.day}"
