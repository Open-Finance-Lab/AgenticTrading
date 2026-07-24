"""Offline end-to-end coverage for the fixed-universe iFinD A-share path."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import json
import subprocess
import sys
from types import SimpleNamespace
import uuid

from fastapi.testclient import TestClient
import pytest
import requests

from dashboard.backend.app import app
from dashboard.backend.api.routers import backtests as backtests_router
from dashboard.backend.database import BacktestDatabase
from dashboard.backend.domain.backtesting import engine as engine_module
from dashboard.backend.domain.backtesting import portfolio_manager as portfolio_module
from dashboard.backend import equity_plot
from dashboard.backend.infrastructure.market_data import provider as provider_module
from dashboard.backend.infrastructure.market_data.ifind_adapter import response_to_frames
from dashboard.backend.infrastructure.market_data.ifind_ashare import IFindAshareProvider
from dashboard.backend.infrastructure.market_data.profiles import (
    A_SHARE_DEMO_6,
    A_SHARE_DEMO_6_SYMBOLS,
    CSI300_SAMPLE_20_2026H2,
    CSI300_SAMPLE_20_2026H2_SYMBOLS,
    IFIND_ASHARE,
    get_market_profile,
)
from dashboard.scripts import backtest_hourly_agent


START = date(2026, 4, 1)
END = date(2026, 5, 1)


def _official_payload(
    symbols=A_SHARE_DEMO_6_SYMBOLS,
    count: int = 60,
) -> dict:
    timestamps = []
    current = START
    sessions = (time(10, 30), time(11, 30), time(14), time(15))
    while len(timestamps) < count:
        if current.weekday() < 5:
            for session in sessions:
                timestamps.append(
                    datetime.combine(current, session).strftime("%Y-%m-%d %H:%M:%S")
                )
                if len(timestamps) == count:
                    break
        current += timedelta(days=1)

    tables = []
    for offset, symbol in enumerate(symbols):
        opens = [10.0 + offset + row * 0.1 for row in range(count)]
        tables.append(
            {
                "thscode": symbol,
                "time": timestamps.copy(),
                "table": {
                    "open": opens,
                    "high": [value + 1.0 for value in opens],
                    "low": [value - 1.0 for value in opens],
                    "close": [value + 0.5 for value in opens],
                    "volume": [10_000 + row for row in range(count)],
                },
            }
        )
    return {"errorcode": 0, "errmsg": "", "tables": tables}


class _FakeIFindClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = []

    def fetch_hourly_bars(self, symbols, start, end):
        self.calls.append((tuple(symbols), start, end))
        return self.payload


class _CapturingThread:
    last = None

    def __init__(self, *, target, args, daemon):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False
        type(self).last = self

    def start(self):
        self.started = True


class _FakeLLMMessages:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"actions": []}')],
            usage=SimpleNamespace(input_tokens=12, output_tokens=2),
        )


class _FakeLLMClient:
    def __init__(self) -> None:
        self.messages = _FakeLLMMessages()


def _fail_external(name: str):
    def fail(*_args, **_kwargs):
        raise AssertionError(f"offline iFinD flow called forbidden dependency: {name}")

    return fail


@pytest.mark.parametrize(
    ("universe", "symbols"),
    [
        (A_SHARE_DEMO_6, A_SHARE_DEMO_6_SYMBOLS),
        (CSI300_SAMPLE_20_2026H2, CSI300_SAMPLE_20_2026H2_SYMBOLS),
    ],
)
def test_ifind_api_background_builds_one_controlled_cli_command(
    monkeypatch, capsys, universe, symbols
):
    secret = f"offline-{uuid.uuid4().hex}"
    session_id = str(uuid.uuid4())
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", secret)
    monkeypatch.setattr(backtests_router.threading, "Thread", _CapturingThread)
    backtests_router._backtest_rate_limiter.reset()
    backtests_router.backtest_status.update(
        {
            "running": False,
            "error": None,
            "runs_count": 0,
            "started_at": None,
            "progress_file": None,
            "live_run_id": None,
        }
    )

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": START.isoformat(),
            "end_date": END.isoformat(),
            "data_source": IFIND_ASHARE,
            "universe": universe,
            "timeframe": "60m",
        },
        headers={"X-Session-Id": session_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "success": True,
        "message": "Backtest started in background. Check /backtest/status for progress.",
        "status_url": "/backtest/status",
        "session_id": session_id,
        "data_source": IFIND_ASHARE,
        "live_run_id": body["live_run_id"],
        "run_id": body["live_run_id"],
        "market": "CN",
        "universe": universe,
        "timeframe": "60m",
        "timezone": "Asia/Shanghai",
        "decision_source": "llm",
        "benchmark": "equal_weight_buyhold",
        "assets": list(symbols),
    }

    thread = _CapturingThread.last
    assert thread is not None and thread.started
    assert thread.target is backtests_router.run_backtest_background
    assert thread.args[7:] == (
        IFIND_ASHARE,
        body["live_run_id"],
        universe,
        "60m",
        None,
        list(symbols),
        "llm",
    )

    captured = {}

    def fake_subprocess_run(command, **kwargs):
        captured["command"] = list(command)
        captured["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    thread.target(*thread.args)

    command = captured["command"]
    assert command[command.index("--data-source") + 1] == IFIND_ASHARE
    assert command[command.index("--universe") + 1] == universe
    assert command[command.index("--timeframe") + 1] == "60m"
    assert command[command.index("--decision-source") + 1] == "llm"
    assert "--use-llm" not in command
    assert "--no-llm" not in command
    assert secret not in " ".join(command)
    assert captured["environment"]["IFIND_ACCESS_TOKEN"] == secret
    assert secret not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("selected_universe", "symbols"),
    [
        (A_SHARE_DEMO_6, A_SHARE_DEMO_6_SYMBOLS),
        (CSI300_SAMPLE_20_2026H2, CSI300_SAMPLE_20_2026H2_SYMBOLS),
    ],
)
def test_ifind_llm_request_reaches_engine_database_and_chart_without_fallback(
    tmp_path,
    monkeypatch,
    capsys,
    selected_universe,
    symbols,
):
    secret = f"offline-{uuid.uuid4().hex}"
    session_id = str(uuid.uuid4())
    test_db = BacktestDatabase(tmp_path / "ifind_llm_e2e.db")
    fake_ifind = _FakeIFindClient(_official_payload(symbols))
    fake_llm = _FakeLLMClient()
    _CapturingThread.last = None

    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", secret)
    monkeypatch.setattr(backtests_router.threading, "Thread", _CapturingThread)
    monkeypatch.setattr(
        backtests_router,
        "ensure_llm_client_available",
        lambda: fake_llm,
    )
    backtests_router._backtest_rate_limiter.reset()
    backtests_router.backtest_status.update(
        {
            "running": False,
            "error": None,
            "runs_count": 0,
            "started_at": None,
            "progress_file": None,
            "live_run_id": None,
        }
    )

    def create_offline_provider(data_source, universe=None):
        assert data_source == IFIND_ASHARE
        assert universe == selected_universe
        return IFindAshareProvider(
            profile=get_market_profile(data_source, universe),
            client=fake_ifind,
            adapter=response_to_frames,
        )

    monkeypatch.setattr(engine_module, "create_market_data_provider", create_offline_provider)
    monkeypatch.setattr(engine_module, "HAS_ANTHROPIC", True)
    monkeypatch.setattr(engine_module, "make_llm_client", lambda: fake_llm)
    monkeypatch.setattr(portfolio_module, "HAS_ANTHROPIC", True)
    monkeypatch.setattr(engine_module, "db", test_db)
    monkeypatch.setattr(backtest_hourly_agent, "db", test_db)
    monkeypatch.setattr(backtests_router, "db", test_db)
    monkeypatch.setattr(
        backtests_router.agent_service.agents,
        "get_agent_by_session",
        lambda _session_id: None,
    )
    monkeypatch.setattr(requests.Session, "request", _fail_external("real HTTP"))
    monkeypatch.setattr(
        provider_module.AlpacaDataLoader,
        "fetch_bars",
        _fail_external("Alpaca"),
    )
    monkeypatch.setattr(equity_plot, "fetch_index_hourly", _fail_external("Yahoo"))
    monkeypatch.setattr(
        portfolio_module.PortfolioManager,
        "make_trading_decision",
        _fail_external("rule-based fallback"),
    )

    def run_cli_in_process(command, **_kwargs):
        assert command[command.index("--decision-source") + 1] == "llm"
        assert command[command.index("--model") + 1] == "gpt-5.2"
        monkeypatch.setattr(sys, "argv", [str(command[1]), *command[2:]])
        backtest_hourly_agent.main()
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run_cli_in_process)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": START.isoformat(),
            "end_date": END.isoformat(),
            "data_source": IFIND_ASHARE,
            "universe": selected_universe,
            "timeframe": "60m",
            "decision_source": "llm",
            "model": "gpt-5.2",
        },
        headers={"X-Session-Id": session_id},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision_source"] == "llm"
    thread = _CapturingThread.last
    assert thread is not None and thread.started
    assert thread.args[-1] == "llm"
    thread.target(*thread.args)

    assert backtests_router.backtest_status["error"] is None
    assert fake_ifind.calls == [(symbols, START, END)]
    assert len(fake_llm.messages.calls) == 60
    assert all(call["model"] == "gpt-5.2" for call in fake_llm.messages.calls)
    assert all("Chinese A-share" in call["system"] for call in fake_llm.messages.calls)
    assert all("DJIA" not in call["system"] for call in fake_llm.messages.calls)
    first_prompt = fake_llm.messages.calls[0]["messages"][0]["content"]
    assert all(symbol in first_prompt for symbol in symbols)

    runs = test_db.get_runs_by_session(session_id)
    assert len(runs) == 2
    agent_run = next(run for run in runs if run["run_id"] == body["run_id"])
    buyhold_run = next(run for run in runs if run["agent_name"] == "buy-and-hold")
    assert agent_run["metadata"]["decision_source"] == "llm"
    assert agent_run["metadata"]["symbols"] == list(symbols)
    assert agent_run["metadata"]["timezone"] == "Asia/Shanghai"
    assert agent_run["llm_model"] == "gpt-5.2"
    assert agent_run["llm_calls"] == 60
    assert agent_run["input_tokens"] == 60 * 12
    assert agent_run["output_tokens"] == 60 * 2
    assert agent_run["est_cost_usd"] > 0
    assert agent_run["baseline_djia_run_id"] is None
    assert agent_run["baseline_buyhold_run_id"] == buyhold_run["run_id"]
    assert test_db.get_equity_curve(agent_run["run_id"])
    assert test_db.get_equity_curve(buyhold_run["run_id"])
    assert test_db.get_trades(agent_run["run_id"]) == []

    chart_response = TestClient(app).get(
        f"/api/backtest/{agent_run['run_id']}/chart-data",
        headers={"X-Session-Id": session_id},
    )
    assert chart_response.status_code == 200
    chart = chart_response.json()
    assert [series["run_id"] for series in chart["series"]] == [
        agent_run["run_id"],
        buyhold_run["run_id"],
    ]
    assert all("DJIA" not in series["label"] for series in chart["series"])

    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err


@pytest.mark.parametrize(
    ("universe", "symbols", "agent_run_id"),
    [
        (A_SHARE_DEMO_6, A_SHARE_DEMO_6_SYMBOLS, "agent_ifind_offline_demo6"),
        (
            CSI300_SAMPLE_20_2026H2,
            CSI300_SAMPLE_20_2026H2_SYMBOLS,
            "agent_ifind_offline_csi300_sample20",
        ),
    ],
)
def test_ifind_offline_response_reaches_engine_database_and_chart(
    tmp_path, monkeypatch, capsys, universe, symbols, agent_run_id
):
    secret = f"offline-{uuid.uuid4().hex}"
    session_id = str(uuid.uuid4())
    test_db = BacktestDatabase(tmp_path / "ifind_e2e.db")
    fake_client = _FakeIFindClient(_official_payload(symbols))
    observed = {}

    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", secret)

    def recording_adapter(payload, **kwargs):
        frames = response_to_frames(payload, **kwargs)
        observed["frames"] = {
            symbol: frame.copy(deep=True) for symbol, frame in frames.items()
        }
        return frames

    def create_offline_provider(data_source, universe=None):
        assert data_source == IFIND_ASHARE
        return IFindAshareProvider(
            profile=get_market_profile(data_source, universe),
            client=fake_client,
            adapter=recording_adapter,
        )

    monkeypatch.setattr(engine_module, "create_market_data_provider", create_offline_provider)
    monkeypatch.setattr(engine_module, "db", test_db)
    monkeypatch.setattr(backtest_hourly_agent, "db", test_db)
    monkeypatch.setattr(backtests_router, "db", test_db)
    monkeypatch.setattr(
        backtests_router.agent_service.agents,
        "get_agent_by_session",
        lambda _session_id: None,
    )

    monkeypatch.setattr(requests.Session, "request", _fail_external("real HTTP"))
    monkeypatch.setattr(
        provider_module.AlpacaDataLoader,
        "fetch_bars",
        _fail_external("Alpaca"),
    )
    monkeypatch.setattr(equity_plot, "fetch_index_hourly", _fail_external("Yahoo"))
    monkeypatch.setattr(engine_module, "make_llm_client", _fail_external("LLM"))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backtest_hourly_agent.py",
            "--start",
            START.isoformat(),
            "--end",
            END.isoformat(),
            "--session-id",
            session_id,
            "--data-source",
            IFIND_ASHARE,
            "--universe",
            universe,
            "--timeframe",
            "60m",
            "--no-llm",
            "--run-id",
            agent_run_id,
        ],
    )
    backtest_hourly_agent.main()

    assert fake_client.calls == [(symbols, START, END)]
    frames = observed["frames"]
    assert tuple(frames) == symbols
    assert all(len(frame) == 60 for frame in frames.values())
    assert all(str(frame.index.tz) == "Asia/Shanghai" for frame in frames.values())
    assert all(
        list(frame.columns) == ["open", "high", "low", "close", "volume"]
        for frame in frames.values()
    )

    runs = test_db.get_runs_by_session(session_id)
    assert len(runs) == 2
    agent_run = next(run for run in runs if run["run_id"] == agent_run_id)
    buyhold_run = next(run for run in runs if run["agent_name"] == "buy-and-hold")

    expected_metadata = {
        "data_source": IFIND_ASHARE,
        "market": "CN",
        "universe": universe,
        "timeframe": "60m",
        "timezone": "Asia/Shanghai",
        "decision_source": "rule_based",
        "benchmark": "equal_weight_buyhold",
        "symbols": list(symbols),
    }
    assert agent_run["metadata"] == expected_metadata
    assert buyhold_run["metadata"] == expected_metadata
    assert agent_run["llm_calls"] == 0
    assert agent_run["baseline_djia_run_id"] is None
    assert agent_run["baseline_buyhold_run_id"] == buyhold_run["run_id"]
    assert test_db.get_equity_curve(agent_run_id)
    assert test_db.get_equity_curve(buyhold_run["run_id"])
    assert isinstance(test_db.get_trades(agent_run_id), list)

    chart_response = TestClient(app).get(
        f"/api/backtest/{agent_run_id}/chart-data",
        headers={"X-Session-Id": session_id},
    )
    assert chart_response.status_code == 200
    chart = chart_response.json()
    assert chart["agent_run_id"] == agent_run_id
    assert [series["run_id"] for series in chart["series"]] == [
        agent_run_id,
        buyhold_run["run_id"],
    ]
    assert [series["label"] for series in chart["series"]] == [
        "Agent",
        "buy-and-hold",
    ]
    assert all("DJIA" not in series["label"] for series in chart["series"])
    assert all("Nasdaq" not in series["label"] for series in chart["series"])

    captured = capsys.readouterr()
    serialized = json.dumps(
        {"runs": test_db.get_runs_by_session(session_id), "chart": chart},
        ensure_ascii=True,
    )
    assert secret not in captured.out
    assert secret not in captured.err
    assert secret not in serialized
