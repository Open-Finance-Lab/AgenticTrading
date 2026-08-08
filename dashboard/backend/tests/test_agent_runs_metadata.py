"""agent_runs.metadata column + effective LLM_MAX_OUTPUT_TOKENS recording.

LOW-sweep residual (PR #67): the per-request output-token ceiling is an env
knob (LLM_MAX_OUTPUT_TOKENS) that changes a run's spend and behavior, but no
row recorded which value was in effect. agent_runs gains a JSON metadata
column (config snapshot, additive) and the engine's LLM-driven agent run
records its effective cap there.
"""

import sqlite3

from dashboard.backend.database import BacktestDatabase


def _make_db(tmp_path, name="meta.db"):
    return BacktestDatabase(tmp_path / name)


def _insert_minimal(db, run_id, metadata=None):
    db.insert_run(
        run_id=run_id,
        session_id="meta-session",
        agent_name="meta-agent",
        mode="backtest",
        start_date="2026-01-01",
        end_date="2026-01-31",
        initial_equity=100000.0,
        metadata=metadata,
    )


def test_insert_run_roundtrips_metadata(tmp_path):
    db = _make_db(tmp_path)
    _insert_minimal(db, "run_meta_1", metadata={"llm_max_output_tokens": 600})
    run = db.get_run("run_meta_1")
    assert run["metadata"] == {"llm_max_output_tokens": 600}


def test_metadata_defaults_to_none(tmp_path):
    db = _make_db(tmp_path)
    _insert_minimal(db, "run_meta_2")
    assert db.get_run("run_meta_2")["metadata"] is None


def test_session_listings_parse_metadata_consistently(tmp_path):
    """SELECT * picks the new column up in the list readers too — they must
    return the same parsed shape as get_run, not raw JSON text."""
    db = _make_db(tmp_path)
    _insert_minimal(db, "run_meta_3", metadata={"llm_max_output_tokens": 1234})
    by_session = db.get_runs_by_session("meta-session")
    assert by_session[0]["metadata"] == {"llm_max_output_tokens": 1234}
    grouped = db.get_runs_by_sessions(["meta-session"])
    assert grouped["meta-session"][0]["metadata"] == {"llm_max_output_tokens": 1234}


def test_data_source_provenance_roundtrips_for_api_projection(tmp_path):
    db = _make_db(tmp_path)
    _insert_minimal(
        db,
        "run_vnpy_source",
        metadata={"data_source": "vnpy_simulation"},
    )

    assert db.get_run("run_vnpy_source")["metadata"]["data_source"] == (
        "vnpy_simulation"
    )


def test_ifind_profile_provenance_roundtrips_without_credentials(tmp_path):
    db = _make_db(tmp_path)
    metadata = {
        "data_source": "ifind_ashare",
        "market": "CN",
        "universe": "a_share_demo_6",
        "timeframe": "60m",
        "timezone": "Asia/Shanghai",
        "decision_source": "rule_based",
        "benchmark": "equal_weight_buyhold",
    }
    _insert_minimal(db, "run_ifind_source", metadata=metadata)

    stored = db.get_run("run_ifind_source")["metadata"]
    assert stored == metadata
    assert "token" not in str(stored).lower()


def test_migration_adds_metadata_column(tmp_path):
    """A DB created before the column must gain it on open (both
    _init_schema's CREATE IF NOT EXISTS and _migrate_schema must know it)."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE agent_runs (
            run_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            mode TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            initial_equity REAL NOT NULL,
            final_equity REAL,
            total_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            num_trades INTEGER DEFAULT 0,
            llm_calls INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            est_cost_usd REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

    db = BacktestDatabase(path)
    conn = sqlite3.connect(str(path))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(agent_runs)")}
    conn.close()
    assert "metadata" in cols
    _insert_minimal(db, "run_meta_migrated", metadata={"llm_max_output_tokens": 2000})
    assert db.get_run("run_meta_migrated")["metadata"] == {
        "llm_max_output_tokens": 2000
    }


def test_engine_llm_run_metadata_snapshot(monkeypatch):
    """The engine's agent run records the EFFECTIVE cap (whatever the env
    parse produced), while every run records data provenance."""
    import dashboard.backend.domain.backtesting.engine as engine_mod

    backtester = engine_mod.HourlyBacktester.__new__(engine_mod.HourlyBacktester)  # skip creds init
    # _agent_run_metadata() also reads the post-trade/pipeline attrs (added in the
    # post-trade-analysis work); __init__ sets them, but __new__ skips it, so set
    # the no-pipeline defaults here to keep this a pure llm_max_output_tokens test.
    backtester.prompt_adaptations = []
    backtester.initial_pipeline = None
    backtester.pipeline = None
    backtester.symbols = ["AAPL", "MSFT"]
    monkeypatch.setattr(engine_mod.llm_harness, "DEFAULT_MAX_OUTPUT_TOKENS", 777)
    backtester.data_source = "alpaca"

    backtester.use_llm = True
    assert backtester._agent_run_metadata() == {
        "data_source": "alpaca",
        "symbols": ["AAPL", "MSFT"],
        "native_currency": "USD",
        "reporting_currency": "USD",
        "lot_size": 1,
        "llm_max_output_tokens": 777,
    }
    backtester.use_llm = False
    assert backtester._agent_run_metadata() == {
        "data_source": "alpaca",
        "symbols": ["AAPL", "MSFT"],
        "native_currency": "USD",
        "reporting_currency": "USD",
        "lot_size": 1,
    }


def test_baseline_metadata_is_provenance_only(monkeypatch):
    """Baselines make no model calls and run no pipeline, so their rows must
    carry provenance and nothing else.

    Regression guard: the baselines call _run_metadata() *after*
    run_agent_backtest() has populated use_llm/prompt_adaptations/pipeline, so
    a shared metadata builder silently stamps the agent's LLM config and a copy
    of its pipeline onto Buy & Hold and DJIA rows."""
    import dashboard.backend.domain.backtesting.engine as engine_mod

    backtester = engine_mod.HourlyBacktester.__new__(engine_mod.HourlyBacktester)  # skip creds init
    backtester.data_source = "vnpy_simulation"
    backtester.symbols = ["AAPL"]
    monkeypatch.setattr(engine_mod.llm_harness, "DEFAULT_MAX_OUTPUT_TOKENS", 777)
    # State as it stands once the agent run has finished, which is when the
    # baselines actually build their metadata.
    backtester.use_llm = True
    backtester.prompt_adaptations = [{"day": "2026-04-02"}]
    backtester.initial_pipeline = [{"step": "a"}]
    backtester.pipeline = [{"step": "b"}]

    assert backtester._run_metadata() == {
        "data_source": "vnpy_simulation",
        "symbols": ["AAPL"],
        "native_currency": "USD",
        "reporting_currency": "USD",
        "lot_size": 1,
    }


def test_engine_agent_run_wires_the_metadata():
    """Wiring guard: the agent-run insert (the LLM one, not the baselines)
    passes the full config snapshot, while the baselines pass provenance only.

    Both needles matter: _run_metadata() alone would still be found in the
    source (the baselines call it), so this greps for the agent-specific one."""
    import re
    from pathlib import Path

    engine_src = (
        Path(__file__).resolve().parents[1]
        / "domain" / "backtesting" / "engine.py"
    ).read_text(encoding="utf-8")
    assert "metadata=self._agent_run_metadata()" in engine_src
    # The index baseline is the only bare call: it places no orders, so it
    # passes neither a cost ledger nor an allocation summary.
    assert engine_src.count("metadata=self._run_metadata()") == 1
    # Buy & Hold does trade, so its call carries both. Matched across newlines
    # because the argument list is long enough to wrap.
    assert re.search(
        r"metadata=self\._run_metadata\(\s*baseline_cost_totals,"
        r"[^)]*baseline_allocation=baseline_allocation,",
        engine_src,
    )
