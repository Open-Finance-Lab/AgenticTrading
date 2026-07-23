"""Phase 3C3 — leaderboard service characterization.

Verifies the canonical ``dashboard.backend.domain.leaderboard`` package and
characterizes ranking, tie ordering, result schema, metric preservation, and the
contest-window helpers against isolated storage.
"""

import ast
from pathlib import Path

import pytest

from dashboard.backend.database import BacktestDatabase
from dashboard.backend.domain.leaderboard import baselines as canon_baselines
from dashboard.backend.domain.leaderboard import service as canon_service

_PUBLIC_SERVICE = [
    "LEADERBOARD_MODE",
    "load_leaderboard_config",
    "ensure_leaderboard_runs",
    "deploy_model_run",
    "get_leaderboard",
]
_PUBLIC_BASELINES = [
    "fetch_hourly_bars",
    "compute_equity_curve",
    "calc_metrics",
    "downsample_daily",
    "chart_equity_curve",
]


# ---------------------------------------------------------------------------
# Canonical imports + wiring
# ---------------------------------------------------------------------------

def test_canonical_modules_import():
    assert canon_service.get_leaderboard.__module__ == (
        "dashboard.backend.domain.leaderboard.service"
    )
    assert canon_baselines.calc_metrics.__module__ == (
        "dashboard.backend.domain.leaderboard.baselines"
    )
    for name in _PUBLIC_SERVICE:
        assert hasattr(canon_service, name), name
    for name in _PUBLIC_BASELINES:
        assert hasattr(canon_baselines, name), name


def test_service_wires_canonical_baselines():
    # The service composes the canonical baseline helpers (no duplication).
    assert canon_service.calc_metrics is canon_baselines.calc_metrics
    assert canon_service.fetch_hourly_bars is canon_baselines.fetch_hourly_bars
    assert canon_service.downsample_daily is canon_baselines.downsample_daily
    assert canon_service.chart_equity_curve is canon_baselines.chart_equity_curve


# ---------------------------------------------------------------------------
# Import boundary: no API / scripts / frontend imports
# ---------------------------------------------------------------------------

def _imported_modules(path: Path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_canonical_package_does_not_import_api_or_scripts():
    package_dir = Path(canon_service.__file__).parent
    for py in package_dir.rglob("*.py"):
        mods = _imported_modules(py)
        for m in mods:
            assert not m.startswith("dashboard.backend.api"), (py.name, m)
            assert not m.startswith("dashboard.scripts"), (py.name, m)
            assert m != "fastapi" and not m.startswith("fastapi."), (py.name, m)


def test_runtime_consumer_uses_canonical_import():
    # Phase 3C4: the leaderboard router moved to the canonical routers package,
    # which is the real runtime consumer of the leaderboard service.
    import dashboard.backend.api.routers.leaderboard as api_lb
    mods = _imported_modules(api_lb.__file__)
    assert "dashboard.backend.domain.leaderboard.service" in mods
    assert "dashboard.backend.services.leaderboard_service" not in mods


# ---------------------------------------------------------------------------
# Ranking + tie ordering (pure function)
# ---------------------------------------------------------------------------

def test_rank_entries_orders_by_portfolio_value():
    entries = [
        {"entry_id": "A", "portfolio_value": 110000, "cumulative_return": 0.10, "sharpe_ratio": 1.0},
        {"entry_id": "B", "portfolio_value": 105000, "cumulative_return": 0.05, "sharpe_ratio": 2.0},
        {"entry_id": "C", "portfolio_value": 101000, "cumulative_return": 0.01, "sharpe_ratio": 0.5},
    ]
    ranked = canon_service._rank_entries(entries)
    by_id = {e["entry_id"]: e for e in ranked}

    # Official rank is by final portfolio value (higher Wins), not composite score.
    assert by_id["A"]["rank"] == 1
    assert by_id["B"]["rank"] == 2
    assert by_id["C"]["rank"] == 3
    assert "final_score" not in by_id["A"]
    assert "rank_cr" not in by_id["A"]


def test_rank_entries_ties_break_on_return():
    entries = [
        {"entry_id": "A", "portfolio_value": 105000, "cumulative_return": 0.10},
        {"entry_id": "B", "portfolio_value": 105000, "cumulative_return": 0.05},
    ]
    ranked = canon_service._rank_entries(entries)
    assert ranked[0]["entry_id"] == "A"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["entry_id"] == "B"


def test_rank_entries_empty():
    assert canon_service._rank_entries([]) == []


# ---------------------------------------------------------------------------
# get_leaderboard end-to-end against isolated DB + controlled config
# ---------------------------------------------------------------------------

_CONFIG = {
    "session_id": "lb-test",
    "start_date": "2026-04-15",
    "end_date": "2026-05-15",
    "description": "Test contest",
    "initial_capital": 100000,
    "strategies": [
        {"id": "djia_index", "name": "DJIA", "label": "Baseline", "model": "DJIA", "strategy": "market_index"},
        {"id": "spy_index", "name": "SPY", "label": "Baseline", "model": "SPY", "strategy": "market_index"},
    ],
}


@pytest.fixture
def isolated_service(tmp_path, monkeypatch):
    test_db = BacktestDatabase(db_path=tmp_path / "lb.db")
    monkeypatch.setattr(canon_service, "db", test_db)
    monkeypatch.setattr(canon_service, "load_leaderboard_config", lambda: dict(_CONFIG))
    monkeypatch.setattr(
        canon_service,
        "ensure_leaderboard_runs",
        lambda force_refresh=False, period="contest", config=None: {
            "session_id": _CONFIG["session_id"],
            "start_date": _CONFIG["start_date"],
            "end_date": _CONFIG["end_date"],
            "created": 0,
            "refreshed_at": "2026-06-18T00:00:00+00:00",
        },
    )
    return test_db


def _seed(db, strategy_id, total_return, sharpe):
    run_id = f"lb_{strategy_id}_20260415_20260515"
    db.insert_run(
        run_id=run_id,
        session_id=_CONFIG["session_id"],
        agent_name=strategy_id.upper(),
        mode="leaderboard",
        start_date=_CONFIG["start_date"],
        end_date=_CONFIG["end_date"],
        initial_equity=100000,
        final_equity=100000 * (1 + total_return),
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=-0.02,
        num_trades=1,
        llm_model=strategy_id,
    )
    db.insert_equity_points(
        run_id,
        [
            {"timestamp": "2026-04-15T14:00:00", "equity": 100000, "cash": 0, "positions_value": 100000},
            {"timestamp": "2026-05-15T20:00:00", "equity": 100000 * (1 + total_return), "cash": 0, "positions_value": 100000 * (1 + total_return)},
        ],
    )


def test_get_leaderboard_schema_and_ranking(isolated_service):
    _seed(isolated_service, "djia_index", 0.05, 1.2)
    _seed(isolated_service, "spy_index", 0.03, 0.9)

    result = canon_service.get_leaderboard()
    assert set(result.keys()) == {
        "window", "updated_at", "total_entries", "leader", "entries", "display_capital",
        # period + board labelling fields (contest vs daily)
        "period", "board_title", "phase_label", "standings_label",
    }
    assert result["period"] == "contest"
    assert result["total_entries"] == 2
    assert result["window"]["label"] == "2026-04-15 → 2026-05-15"
    # Fixture config keeps initial_capital=100000; product default is tested below.
    assert result["display_capital"] == 100000

    entries = result["entries"]
    # djia_index has higher return and sharpe → rank 1.
    assert entries[0]["entry_id"] == "djia_index"
    assert entries[0]["rank"] == 1
    # No model entries in this fixture → leader falls back to overall rank-1 name.
    assert result["leader"] == "DJIA_INDEX"

    entry = entries[0]
    expected_fields = {
        "entry_id", "team_name", "team_badge", "model", "entry_type", "is_model",
        "initial_equity", "portfolio_value", "cumulative_return", "sharpe_ratio",
        "max_drawdown", "status", "run_id", "llm_calls", "input_tokens",
        "output_tokens", "est_cost_usd", "equity_curve", "rank",
    }
    assert expected_fields.issubset(set(entry.keys()))
    assert "win_loss_ratio" not in entry
    assert entry["entry_type"] == "baseline"
    assert entry["cumulative_return"] == 0.05
    assert isinstance(entry["equity_curve"], list)


def test_get_leaderboard_skips_uncached_strategies(isolated_service):
    # Only one of the two configured strategies has a cached run.
    _seed(isolated_service, "djia_index", 0.05, 1.2)
    result = canon_service.get_leaderboard()
    assert result["total_entries"] == 1
    assert result["entries"][0]["entry_id"] == "djia_index"


def test_chart_equity_curve_prepends_open_tick():
    hourly = [
        {"timestamp": "2026-04-15T14:00:00+00:00", "equity": 100500.0, "cash": 0, "positions_value": 100500},
        {"timestamp": "2026-04-15T15:00:00+00:00", "equity": 101000.0, "cash": 0, "positions_value": 101000},
        {"timestamp": "2026-04-16T14:00:00+00:00", "equity": 102000.0, "cash": 0, "positions_value": 102000},
    ]
    curve = canon_baselines.chart_equity_curve(
        hourly, initial_equity=100000.0, start_date="2026-04-15"
    )
    assert len(curve) == 4
    assert curve[0]["timestamp"] == "2026-04-15T00:00:00+00:00"
    assert curve[0]["equity"] == 100000.0
    assert curve[1]["equity"] == 100500.0
    assert curve[-1]["equity"] == 102000.0


def test_align_equity_curves_onto_alpaca_grid():
    # Open tick + Alpaca :00 hours (agent) vs Yahoo :30 hours (index).
    agent = [
        {"timestamp": "2026-04-15T00:00:00+00:00", "equity": 100000, "cash": 100000, "positions_value": 0},
        {"timestamp": "2026-04-15T14:00:00+00:00", "equity": 101000, "cash": 0, "positions_value": 101000},
        {"timestamp": "2026-04-15T15:00:00+00:00", "equity": 102000, "cash": 0, "positions_value": 102000},
    ]
    yahoo = [
        {"timestamp": "2026-04-15T00:00:00+00:00", "equity": 100000, "cash": 100000, "positions_value": 0},
        {"timestamp": "2026-04-15T13:30:00+00:00", "equity": 100500, "cash": 0, "positions_value": 100500},
        {"timestamp": "2026-04-15T14:30:00+00:00", "equity": 101500, "cash": 0, "positions_value": 101500},
    ]
    aligned = canon_baselines.align_equity_curves([agent, yahoo])
    assert [p["timestamp"] for p in aligned[0]] == [p["timestamp"] for p in aligned[1]]
    assert [p["timestamp"] for p in aligned[0]] == [p["timestamp"] for p in agent]
    # Yahoo 13:30 carries into the 14:00 slot; 14:30 into 15:00.
    assert aligned[1][1]["equity"] == 100500
    assert aligned[1][2]["equity"] == 101500


def test_get_leaderboard_equity_curve_is_hourly_with_open_tick(isolated_service):
    _seed(isolated_service, "djia_index", 0.05, 1.2)
    # Extra mid-day point so daily downsample would have collapsed to 2 days;
    # hourly chart must keep both Apr-15 hours plus the open tick.
    run_id = "lb_djia_index_20260415_20260515"
    isolated_service.insert_equity_points(
        run_id,
        [
            {"timestamp": "2026-04-15T15:00:00", "equity": 100200, "cash": 0, "positions_value": 100200},
        ],
        replace=False,  # extend the seeded curve; a rerun would replace it
    )
    result = canon_service.get_leaderboard()
    curve = result["entries"][0]["equity_curve"]
    assert curve[0]["timestamp"] == "2026-04-15T00:00:00+00:00"
    assert curve[0]["equity"] == 100000
    # Seeded points + mid-day insert remain (hourly, not daily-collapsed).
    assert len(curve) >= 3
    assert any(str(p["timestamp"]).startswith("2026-04-15T14") for p in curve)
    assert any(str(p["timestamp"]).startswith("2026-04-15T15") for p in curve)


def test_get_leaderboard_scales_to_display_capital_and_prefers_model_leader(
    isolated_service, monkeypatch
):
    """$100k seed runs are rescaled when config.initial_capital is $1k."""
    cfg = dict(_CONFIG)
    cfg["initial_capital"] = 1000
    cfg["strategies"] = [
        {"id": "djia_index", "name": "DJIA", "label": "Index", "model": "DJIA", "strategy": "market_index"},
        {
            "id": "gpt_demo",
            "name": "GPT Demo",
            "label": "Model",
            "model": "GPT Demo",
            "strategy": "llm_agent",
        },
    ]
    monkeypatch.setattr(canon_service, "load_leaderboard_config", lambda: cfg)

    _seed(isolated_service, "djia_index", 0.10, 1.0)
    _seed(isolated_service, "gpt_demo", 0.05, 0.8)

    result = canon_service.get_leaderboard()
    assert result["display_capital"] == 1000
    by_id = {e["entry_id"]: e for e in result["entries"]}
    assert by_id["djia_index"]["initial_equity"] == 1000
    assert abs(by_id["djia_index"]["portfolio_value"] - 1100) < 1e-6
    assert by_id["djia_index"]["cumulative_return"] == 0.10
    assert by_id["gpt_demo"]["is_model"] is True
    # Leader is best model even if an index outranks it overall.
    assert result["leader"] == "GPT Demo"
    assert by_id["gpt_demo"]["equity_curve"][0]["equity"] == 1000
