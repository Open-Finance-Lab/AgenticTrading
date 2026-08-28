"""Tests for Playground-aligned chart styling helpers."""

from dashboard.backend.chart_style import (
    PLAYGROUND_SERIES_COLORS,
    format_playground_timestamp,
    series_color,
    series_kind,
)
from dashboard.backend.equity_plot import (
    resolve_agent_chart_label,
)


def test_series_kind_and_colors_match_playground():
    assert series_kind("algo_123", "Agent") == "my-algo"
    assert series_kind("ext_abc", "My Bot") == "external-agent"
    assert series_kind("agent_123", "Agent") == "agent"
    assert series_kind("djia_index_1", "DJIA") == "djia-index"
    assert series_kind("idx1", "DJIA index") == "djia-index"
    assert series_kind("idx2", "Nasdaq-100") == "nasdaq-100"
    assert series_kind("buyhold_1", "buy-and-hold") == "buy-and-hold"

    assert series_color("agent_123", "Agent") == PLAYGROUND_SERIES_COLORS["agent"]
    assert series_color("idx1", "DJIA index") == "#F5C04A"
    assert series_color("idx2", "Nasdaq-100") == "#9AA4B2"
    assert series_color("buyhold_1", "buy-and-hold") == "#34D399"
    assert series_color("buyhold_1", "buy-and-hold") != series_color(
        "idx2", "Nasdaq-100"
    )


def test_format_playground_timestamp():
    from datetime import datetime

    assert format_playground_timestamp(datetime(2026, 5, 2, 14, 30)) == "May 2"


def test_resolve_agent_chart_label_prefers_card_name():
    assert resolve_agent_chart_label("Agent", "claude-haiku-4.5", "momentum alpha") == "momentum alpha"
    assert resolve_agent_chart_label("my-bot", "claude-haiku-4.5") == "my-bot"
    assert resolve_agent_chart_label("Agent", "claude-haiku-4.5") == "Agent"
    assert resolve_agent_chart_label(None, None) == "Agent"
