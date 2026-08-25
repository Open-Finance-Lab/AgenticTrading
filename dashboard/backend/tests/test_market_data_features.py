"""Feature gating and API propagation for selectable market-data sources."""

from __future__ import annotations

import subprocess
import sys
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
import dashboard.backend.api.routers.backtests as backtests
from dashboard.backend.infrastructure.market_data.provider import (
    IFIND_ASHARE,
    MarketDataDependencyError,
    VNPY_SIMULATION,
    validate_market_data_source,
)
from dashboard.backend.infrastructure.market_data.profiles import (
    A_SHARE_DEMO_6,
    A_SHARE_DEMO_6_SYMBOLS,
    CSI300_SAMPLE_20_2026H2,
    CSI300_SAMPLE_20_2026H2_SYMBOLS,
)
from dashboard.backend.domain.model_providers.execution_catalog import (
    ExecutionModelRoute,
)

REAL_RUN_BACKTEST_BACKGROUND = backtests.run_backtest_background


class Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


# The market-profile half of run_backtest_background's signature.
_PROFILE_KWARGS = (
    "data_source",
    "live_run_id",
    "universe",
    "timeframe",
    "initial_capital",
    "assets",
    "decision_source",
)


def session_headers():
    return {"X-Session-Id": str(uuid.uuid4())}


class _OpenRouterByokPreflight:
    def __init__(self, failure: Exception | None = None):
        self.failure = failure
        self.execution_calls = []
        self.credential_calls = []

    def preflight_execution_model(self, provider_id, catalog_model_id):
        self.execution_calls.append((provider_id, catalog_model_id))
        if self.failure is not None:
            raise self.failure
        assert provider_id == "openrouter"
        assert catalog_model_id == "openai/gpt-5.5"
        return ExecutionModelRoute(
            catalog_id=catalog_model_id,
            label="GPT-5.5",
            provider_model_id=catalog_model_id,
        )

    def preflight_user_default_credential(self, user_id, provider_id):
        self.credential_calls.append((user_id, provider_id))

    def preflight_platform_credential(self, provider_id):
        raise AssertionError(f"unexpected Platform Credits preflight: {provider_id}")


def _enable_authenticated_openrouter_byok(monkeypatch, *, failure=None):
    service = _OpenRouterByokPreflight(failure=failure)
    monkeypatch.setattr(backtests, "get_model_provider_service", lambda: service)
    monkeypatch.setattr(
        "dashboard.backend.api.dependencies._optional_user",
        lambda *_args, **_kwargs: {"id": 7},
    )
    return service


@pytest.fixture(autouse=True)
def reset_backtest_state(monkeypatch):
    backtests._backtest_rate_limiter.reset()
    backtests.backtest_status.update(
        {
            "running": False,
            "error": None,
            "runs_count": 0,
            "started_at": None,
            "progress_file": None,
            "live_run_id": None,
        }
    )
    monkeypatch.setattr(backtests, "run_backtest_background", lambda *a, **k: None)
    yield
    backtests._backtest_rate_limiter.reset()


def test_features_endpoint_reports_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_VNPY_SIMULATION", raising=False)
    monkeypatch.delenv("ENABLE_IFIND_ASHARE", raising=False)

    response = TestClient(app).get("/config/features")

    assert response.status_code == 200
    assert response.json() == {
        "vnpy_simulation_enabled": False,
        "ifind_ashare_enabled": False,
    }


def test_features_endpoint_reports_enabled(monkeypatch):
    monkeypatch.setenv("ENABLE_VNPY_SIMULATION", "true")
    monkeypatch.delenv("ENABLE_IFIND_ASHARE", raising=False)

    response = TestClient(app).get("/config/features")

    assert response.status_code == 200
    assert response.json() == {
        "vnpy_simulation_enabled": True,
        "ifind_ashare_enabled": False,
    }


def test_features_endpoint_reports_ifind_gate_without_token_status(monkeypatch):
    monkeypatch.delenv("ENABLE_VNPY_SIMULATION", raising=False)
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.delenv("IFIND_ACCESS_TOKEN", raising=False)

    response = TestClient(app).get("/config/features")

    assert response.status_code == 200
    assert response.json() == {
        "vnpy_simulation_enabled": False,
        "ifind_ashare_enabled": True,
    }
    assert "token" not in response.text.lower()


def test_unknown_source_returns_422_before_scheduling(monkeypatch):
    spy = Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-23",
            "data_source": "unknown",
        },
        headers=session_headers(),
    )

    assert response.status_code == 422
    assert spy.calls == []


def test_disabled_simulation_returns_403_before_scheduling(monkeypatch):
    monkeypatch.delenv("ENABLE_VNPY_SIMULATION", raising=False)
    spy = Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-23",
            "data_source": VNPY_SIMULATION,
        },
        headers=session_headers(),
    )

    assert response.status_code == 403
    assert spy.calls == []


def test_missing_vnpy_returns_503_before_scheduling(monkeypatch):
    monkeypatch.setenv("ENABLE_VNPY_SIMULATION", "true")
    spy = Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    def dependency_error(source):
        raise MarketDataDependencyError(
            "vn.py is not installed; run pip install -r requirements-vnpy.txt"
        )

    monkeypatch.setattr(backtests, "ensure_market_data_source_available", dependency_error)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-23",
            "data_source": VNPY_SIMULATION,
        },
        headers=session_headers(),
    )

    assert response.status_code == 503
    assert "requirements-vnpy.txt" in response.text
    assert spy.calls == []


def test_disabled_ifind_returns_403_before_scheduling(monkeypatch):
    monkeypatch.delenv("ENABLE_IFIND_ASHARE", raising=False)
    spy = Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-23",
            "data_source": IFIND_ASHARE,
            "universe": A_SHARE_DEMO_6,
            "timeframe": "60m",
            "decision_source": "rule_based",
        },
        headers=session_headers(),
    )

    assert response.status_code == 403
    assert spy.calls == []


def test_missing_ifind_token_returns_503_before_scheduling(monkeypatch):
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.delenv("IFIND_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("IFIND_ACCESS_TOKEN", raising=False)
    spy = Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-23",
            "data_source": IFIND_ASHARE,
            "universe": A_SHARE_DEMO_6,
            "timeframe": "60m",
            "decision_source": "rule_based",
        },
        headers=session_headers(),
    )

    assert response.status_code == 503
    assert "IFIND_REFRESH_TOKEN" in response.text
    assert "IFIND_ACCESS_TOKEN" in response.text
    assert spy.calls == []


def test_refresh_token_allows_ifind_backtest_scheduling(monkeypatch):
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_REFRESH_TOKEN", "refresh-token-canary")
    monkeypatch.delenv("IFIND_ACCESS_TOKEN", raising=False)
    spy = Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-23",
            "data_source": IFIND_ASHARE,
            "universe": A_SHARE_DEMO_6,
            "timeframe": "60m",
            "decision_source": "rule_based",
        },
        headers=session_headers(),
    )

    assert response.status_code == 200
    assert spy.calls


def test_ifind_rejects_wrong_universe_before_checking_credentials(monkeypatch):
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.delenv("IFIND_ACCESS_TOKEN", raising=False)
    spy = Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-23",
            "data_source": IFIND_ASHARE,
            "universe": "custom_a_share_pool",
            "timeframe": "60m",
        },
        headers=session_headers(),
    )

    assert response.status_code == 422
    assert A_SHARE_DEMO_6 in response.text
    assert spy.calls == []


def test_ifind_rejects_unsupported_timeframe_before_scheduling(monkeypatch):
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    spy = Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-23",
            "data_source": IFIND_ASHARE,
            "universe": A_SHARE_DEMO_6,
            "timeframe": "1d",
        },
        headers=session_headers(),
    )

    assert response.status_code == 422
    assert "60m" in response.text
    assert spy.calls == []


def test_enabled_simulation_is_passed_to_background_runner(monkeypatch):
    monkeypatch.setenv("ENABLE_VNPY_SIMULATION", "true")
    spy = Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)
    # Stand in for "vn.py is installed" without requiring the optional
    # dependency: validate_market_data_source is the real allow-list + feature
    # gate, minus the find_spec probe. The env var above therefore still has to
    # be set for this to reach the runner (test_disabled_... proves the 403),
    # and the probe itself is covered by test_missing_vnpy_returns_503.
    monkeypatch.setattr(
        backtests, "ensure_market_data_source_available", validate_market_data_source
    )

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-23",
            "data_source": VNPY_SIMULATION,
        },
        headers=session_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data_source"] == VNPY_SIMULATION
    assert len(spy.calls) == 1
    # The worker is scheduled by keyword, so assert by name: an index-based
    # assertion silently follows a mid-signature insertion onto the wrong value.
    args, kwargs = spy.calls[0]
    assert args == ()
    assert kwargs["data_source"] == VNPY_SIMULATION
    assert kwargs["live_run_id"] == body["live_run_id"]
    assert str(kwargs["live_run_id"]).startswith("agent_")


def test_enabled_ifind_profile_is_passed_to_background_runner(monkeypatch):
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    spy = Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    response = TestClient(app).post(
        "/backtest/run?assets=AAPL,MSFT",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-23",
            "data_source": IFIND_ASHARE,
            "universe": A_SHARE_DEMO_6,
            "timeframe": "60m",
            "decision_source": "rule_based",
        },
        headers=session_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "success": True,
        "message": "Backtest started in background. Check /backtest/status for progress.",
        "status_url": "/backtest/status",
        "session_id": body["session_id"],
        "data_source": IFIND_ASHARE,
        "live_run_id": body["live_run_id"],
        "run_id": body["live_run_id"],
        "market": "CN",
        "universe": A_SHARE_DEMO_6,
        "timeframe": "60m",
        "timezone": "Asia/Shanghai",
        "decision_source": "rule_based",
        "benchmark": "equal_weight_buyhold",
        "assets": list(A_SHARE_DEMO_6_SYMBOLS),
    }
    assert len(spy.calls) == 1
    args, kwargs = spy.calls[0]
    assert args == ()
    assert {key: kwargs[key] for key in _PROFILE_KWARGS} == {
        "data_source": IFIND_ASHARE,
        "live_run_id": body["live_run_id"],
        "universe": A_SHARE_DEMO_6,
        "timeframe": "60m",
        "initial_capital": None,
        "assets": list(A_SHARE_DEMO_6_SYMBOLS),
        "decision_source": "rule_based",
    }


@pytest.mark.parametrize("universe", [A_SHARE_DEMO_6, CSI300_SAMPLE_20_2026H2])
def test_ifind_explicit_llm_is_preflighted_and_scheduled(monkeypatch, universe):
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    service = _enable_authenticated_openrouter_byok(monkeypatch)
    spy = Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-23",
            "data_source": IFIND_ASHARE,
            "universe": universe,
            "timeframe": "60m",
            "decision_source": "llm",
            "billing_mode": "byok",
            "provider_id": "openrouter",
            "model": "openai/gpt-5.5",
            "strategy_prompt": "A-share momentum",
        },
        headers=session_headers(),
    )

    assert response.status_code == 200, response.text
    assert response.json()["decision_source"] == "llm"
    assert response.json()["billing_mode"] == "byok"
    assert response.json()["provider_id"] == "openrouter"
    assert service.execution_calls == [("openrouter", "openai/gpt-5.5")]
    assert service.credential_calls == [(7, "openrouter")]
    kwargs = spy.calls[0][1]
    assert kwargs["strategy_prompt"] == "A-share momentum"
    assert kwargs["model"] == "openai/gpt-5.5"
    assert kwargs["decision_source"] == "llm"
    assert kwargs["execution_handoff_payload"]


def test_ifind_body_decision_source_overrides_query(monkeypatch):
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    preflight_calls = []
    monkeypatch.setattr(
        backtests,
        "ensure_llm_client_available",
        lambda: preflight_calls.append(True),
        raising=False,
    )
    spy = Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    response = TestClient(app).post(
        "/backtest/run?decision_source=llm&model=gpt-5.2",
        json={
            "data_source": IFIND_ASHARE,
            "universe": A_SHARE_DEMO_6,
            "timeframe": "60m",
            "decision_source": "rule_based",
        },
        headers=session_headers(),
    )

    assert response.status_code == 200, response.text
    assert response.json()["decision_source"] == "rule_based"
    assert preflight_calls == []
    kwargs = spy.calls[0][1]
    assert (
        kwargs["strategy_prompt"],
        kwargs["model"],
        kwargs["pipeline"],
    ) == (None, None, None)
    assert kwargs["decision_source"] == "rule_based"


def test_ifind_explicit_llm_configuration_error_is_sanitized(monkeypatch):
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    secret = "provider-secret-that-must-not-leak"
    monkeypatch.setenv("COMMONSTACK_API_KEY", secret)
    _enable_authenticated_openrouter_byok(
        monkeypatch,
        failure=RuntimeError(secret),
    )
    spy = Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "data_source": IFIND_ASHARE,
            "universe": A_SHARE_DEMO_6,
            "timeframe": "60m",
            "decision_source": "llm",
            "billing_mode": "byok",
            "provider_id": "openrouter",
            "model": "openai/gpt-5.5",
        },
        headers=session_headers(),
    )

    assert response.status_code == 503
    assert "The selected model provider is unavailable." in response.text
    assert secret not in response.text
    assert spy.calls == []


def test_enabled_ifind_csi300_sample20_is_passed_to_background_runner(
    monkeypatch,
):
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    spy = Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-06-23",
            "end_date": "2026-07-23",
            "data_source": IFIND_ASHARE,
            "universe": CSI300_SAMPLE_20_2026H2,
            "timeframe": "60m",
            "assets": ["AAPL"],
            "decision_source": "rule_based",
        },
        headers=session_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["universe"] == CSI300_SAMPLE_20_2026H2
    assert body["assets"] == list(CSI300_SAMPLE_20_2026H2_SYMBOLS)
    assert body["decision_source"] == "rule_based"
    assert len(spy.calls) == 1
    kwargs = spy.calls[0][1]
    assert {key: kwargs[key] for key in _PROFILE_KWARGS} == {
        "data_source": IFIND_ASHARE,
        "live_run_id": body["live_run_id"],
        "universe": CSI300_SAMPLE_20_2026H2,
        "timeframe": "60m",
        "initial_capital": None,
        "assets": list(CSI300_SAMPLE_20_2026H2_SYMBOLS),
        "decision_source": "rule_based",
    }


@pytest.mark.parametrize(
    ("data_source", "expected_source"),
    [
        ("alpaca", "llm"),
        (VNPY_SIMULATION, "rule_based"),
        (IFIND_ASHARE, "llm"),
    ],
)
def test_background_command_uses_profile_decision_source(
    monkeypatch, data_source, expected_source
):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(backtests.db, "get_runs_by_mode", lambda mode: [])

    REAL_RUN_BACKTEST_BACKGROUND(
        "2026-04-01",
        "2026-04-23",
        "session-id",
        data_source=data_source,
    )

    command = captured["command"]
    assert command[command.index("--data-source") + 1] == data_source
    assert command[command.index("--decision-source") + 1] == expected_source
    assert "--use-llm" not in command
    assert "--no-llm" not in command
    assert "--assets" not in command
    if data_source == IFIND_ASHARE:
        assert command[command.index("--universe") + 1] == A_SHARE_DEMO_6
        assert command[command.index("--timeframe") + 1] == "60m"


def test_background_command_propagates_explicit_ifind_llm(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(backtests.db, "get_runs_by_mode", lambda mode: [])

    REAL_RUN_BACKTEST_BACKGROUND(
        "2026-04-01",
        "2026-04-23",
        "session-id",
        strategy_prompt="A-share momentum",
        model="gpt-5.2",
        data_source=IFIND_ASHARE,
        universe=A_SHARE_DEMO_6,
        timeframe="60m",
        decision_source="llm",
    )

    command = captured["command"]
    assert command[command.index("--decision-source") + 1] == "llm"
    assert command[command.index("--model") + 1] == "gpt-5.2"
    assert "--strategy-prompt-file" in command
    assert "--use-llm" not in command
    assert "--no-llm" not in command


def test_background_injects_only_resolved_financial_datasets_credential(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv(
        "FINANCIAL_DATASETS_API_KEY", "ambient-key-must-be-replaced"
    )
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(backtests.db, "get_runs_by_mode", lambda mode: [])

    REAL_RUN_BACKTEST_BACKGROUND(
        "2026-04-01",
        "2026-04-23",
        "session-id",
        runtime_type="ai_hedge_fund",
        runtime_config={"analysts": ["technical_analyst"]},
        financial_datasets_api_key="authorized-user-key",
    )

    assert captured["env"]["FINANCIAL_DATASETS_API_KEY"] == "authorized-user-key"
    assert "authorized-user-key" not in captured["command"]


def test_background_error_is_sanitized_and_clears_running_state(monkeypatch):
    secret = "test-ifind-token-that-must-not-leak"
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", secret)

    def fail_run(command, **kwargs):
        raise RuntimeError(f"upstream failed with access_token={secret}")

    monkeypatch.setattr(subprocess, "run", fail_run)

    REAL_RUN_BACKTEST_BACKGROUND(
        "2026-04-01",
        "2026-04-23",
        "session-id",
        data_source=IFIND_ASHARE,
        universe=A_SHARE_DEMO_6,
        timeframe="60m",
    )

    assert backtests.backtest_status["running"] is False
    assert backtests.backtest_status["error"]
    assert secret not in backtests.backtest_status["error"]
    assert "[REDACTED]" in backtests.backtest_status["error"]


def test_cli_rejects_universe_that_does_not_match_ifind_profile(monkeypatch, capsys):
    from dashboard.scripts import backtest_hourly_agent

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backtest_hourly_agent.py",
            "--data-source",
            IFIND_ASHARE,
            "--universe",
            "custom_a_share_pool",
            "--timeframe",
            "60m",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        backtest_hourly_agent.main()

    assert exc_info.value.code == 2
    assert A_SHARE_DEMO_6 in capsys.readouterr().err
