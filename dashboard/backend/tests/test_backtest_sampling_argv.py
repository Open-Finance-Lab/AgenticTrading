"""The parent resolves the sampling policy and the child receives it as argv.

The values are not secrets, so they ride beside --model rather than inside
the signed handoff envelope. Each flag is present only when its half of the
policy is set: an absent flag is how "not sent" reaches the request builder.
"""
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
import dashboard.backend.api.routers.backtests as backtests
from dashboard.backend.domain.model_providers.execution_catalog import (
    PINNED_NO_THINKING,
    PINNED_REASONING_LOW,
    PINNED_TEMPERATURE,
    ExecutionModelRoute,
)
from dashboard.backend.infrastructure.market_data.profiles import A_SHARE_DEMO_6
from dashboard.backend.infrastructure.market_data.provider import IFIND_ASHARE
from dashboard.backend.tests._fake_child import FakeChild

REAL_RUN_BACKTEST_BACKGROUND = backtests.run_backtest_background


class _Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _Preflight:
    def __init__(self, sampling):
        self.sampling = sampling

    def preflight_execution_model(self, provider_id, catalog_model_id):
        return ExecutionModelRoute(
            catalog_id=catalog_model_id,
            label=catalog_model_id,
            provider_model_id=catalog_model_id,
            sampling=self.sampling,
        )

    def preflight_user_default_credential(self, user_id, provider_id):
        return None


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    backtests._backtest_rate_limiter.reset()
    backtests.backtest_status.update({
        "running": False, "error": None, "runs_count": 0,
        "started_at": None, "progress_file": None, "live_run_id": None,
    })
    yield
    backtests._backtest_rate_limiter.reset()


def _capture_command(monkeypatch, **kwargs):
    captured = {}

    def fake_popen(command, **_kwargs):
        captured["command"] = command
        return FakeChild()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(backtests.db, "get_runs_by_mode", lambda mode: [])
    REAL_RUN_BACKTEST_BACKGROUND(
        "2026-04-01", "2026-04-08", "session-id",
        decision_source="llm", model="deepseek/deepseek-v4-pro",
        strategy_prompt="momentum", **kwargs,
    )
    return captured["command"]


def test_both_flags_ride_argv_when_both_are_set(monkeypatch):
    command = _capture_command(monkeypatch, llm_temperature=0.0, llm_reasoning_effort="none")
    assert command[command.index("--llm-temperature") + 1] == "0.0"
    assert command[command.index("--llm-reasoning-effort") + 1] == "none"


def test_an_unset_temperature_is_absent_from_argv(monkeypatch):
    """The `PINNED_REASONING_LOW` shape -- GPT-5.5, which rejects a
    temperature outright, so the absent flag is the whole point of the row.
    Also the one case that watches the `.lower()` normalisation."""
    command = _capture_command(monkeypatch, llm_reasoning_effort="LOW")
    assert "--llm-temperature" not in command
    assert command[command.index("--llm-reasoning-effort") + 1] == "low"


def test_an_unset_reasoning_effort_is_absent_from_argv(monkeypatch):
    """The `PINNED_TEMPERATURE` shape, and the mirror is not free.

    Three of the catalog's six rows carry it -- both Claude models and
    Gemini 3.1 Pro Preview -- so this is the argv a dashboard backtest emits
    most often, and it was the one policy class the argv layer never saw.
    It does not follow from the case above, because the two halves are built
    by different code in Step 4: the temperature branch guards on
    `is not None` and renders `repr(float(...))`, the effort branch guards on
    truthiness and renders `.strip().lower()`. Only a witness per half can
    catch one of them growing a guard that swallows a set value.
    """
    command = _capture_command(monkeypatch, llm_temperature=0.0)
    assert command[command.index("--llm-temperature") + 1] == "0.0"
    assert "--llm-reasoning-effort" not in command


def test_no_policy_means_no_flags(monkeypatch):
    command = _capture_command(monkeypatch)
    assert "--llm-temperature" not in command
    assert "--llm-reasoning-effort" not in command


@pytest.mark.parametrize(
    ("sampling", "expected"),
    [
        (PINNED_TEMPERATURE, {"llm_temperature": 0.0, "llm_reasoning_effort": None}),
        (PINNED_REASONING_LOW, {"llm_temperature": None, "llm_reasoning_effort": "low"}),
        (PINNED_NO_THINKING, {"llm_temperature": 0.0, "llm_reasoning_effort": "none"}),
    ],
)
def test_endpoint_hands_the_routes_policy_to_the_launcher(monkeypatch, sampling, expected):
    """All three policy shapes, on the route rather than in the catalog.

    The posted `model` stays `openai/gpt-5.5` across the sweep and the
    `_Preflight` stub answers with the parametrized policy regardless: what
    this asserts is that the endpoint forwards **whatever the route carried**,
    not that gpt-5.5 carries any particular thing. Which row carries which
    policy is Task 1's
    `test_each_model_pins_the_documented_policy`; asserting it twice would
    mean a table change reddens two files and someone edits the nearer one.
    """
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    monkeypatch.setattr(backtests, "get_model_provider_service", lambda: _Preflight(sampling))
    monkeypatch.setattr(
        "dashboard.backend.api.dependencies._optional_user",
        lambda *_args, **_kwargs: {"id": 7},
    )
    spy = _Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-15",
            "data_source": IFIND_ASHARE,
            "universe": A_SHARE_DEMO_6,
            "timeframe": "60m",
            "decision_source": "llm",
            "billing_mode": "byok",
            "provider_id": "openrouter",
            "model": "openai/gpt-5.5",
            "strategy_prompt": "A-share momentum",
        },
        headers={"X-Session-Id": str(uuid.uuid4())},
    )

    assert response.status_code == 200, response.text
    (_args, kwargs), = spy.calls
    assert {k: kwargs[k] for k in expected} == expected


def test_script_accepts_the_two_flags(tmp_path):
    import os

    result = subprocess.run(
        [sys.executable, "dashboard/scripts/backtest_hourly_agent.py", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "DATABASE_PATH": str(tmp_path / "backtest.db")},
    )
    assert result.returncode == 0, result.stderr
    assert "--llm-temperature" in result.stdout
    assert "--llm-reasoning-effort" in result.stdout


def test_reasoning_effort_without_a_handoff_is_refused(tmp_path):
    """The flag is worker-only, and that is enforced rather than documented.

    Without `--execution-handoff-stdin` the engine builds `make_llm_client()`
    -- the plain Anthropic SDK -- and `request_trading_decision` would hand
    `reasoning_effort` to `messages.create()`, which raises `TypeError` for an
    unknown kwarg. That would land on bar 1, after the bar fetch and the
    indicator pass have already been paid for, with a help string as the only
    thing that ever said not to.

    DEVNULL on stdin and a timeout on purpose: if the guard is ever dropped,
    this process goes on to attempt a real run, and the case should fail on
    the missing message rather than block on a read or a network call.
    """
    import os

    result = subprocess.run(
        [
            sys.executable,
            "dashboard/scripts/backtest_hourly_agent.py",
            "--use-llm",
            "--llm-reasoning-effort",
            "low",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "DATABASE_PATH": str(tmp_path / "backtest.db")},
    )

    assert result.returncode == 2, result.stdout
    assert "--llm-reasoning-effort requires --execution-handoff-stdin" in result.stderr


def test_temperature_alone_is_not_swept_up_by_that_refusal(tmp_path):
    """`--llm-temperature` is legal on every path and must stay legal.

    `request_trading_decision` has always accepted a temperature, so the flag
    means something even without a handoff. This run still stops at exit 2 --
    but at the *pre-existing* rule about who pays for the call ("explicit LLM
    execution requires a signed execution handoff", raised for
    `decision_source=llm` on the default pipeline runtime), not at the
    sampling rule. Asserting on which message comes back is what keeps the two
    distinguishable: they refuse the same command for different reasons, and
    only one of them is ours to relax.
    """
    import os

    result = subprocess.run(
        [
            sys.executable,
            "dashboard/scripts/backtest_hourly_agent.py",
            "--use-llm",
            "--llm-temperature",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "DATABASE_PATH": str(tmp_path / "backtest.db")},
    )

    assert result.returncode == 2, result.stdout
    # The refusal's own sentence, not the bare flag name: argparse's usage
    # line lists every option on any exit-2.
    assert "--llm-reasoning-effort requires" not in result.stderr
    assert "signed execution handoff" in result.stderr


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_reasoning_effort_means_unset(tmp_path, blank):
    """An empty `--llm-reasoning-effort` is "not set", on the wire and in metadata.

    Normalised once at parse time, so the refusal below and the value handed
    to the engine agree: a blank value neither trips the handoff rule nor
    reaches `agent_runs.metadata["llm_sampling"]` as an effort of `""`.
    """
    import os

    result = subprocess.run(
        [
            sys.executable,
            "dashboard/scripts/backtest_hourly_agent.py",
            "--use-llm",
            "--llm-reasoning-effort",
            blank,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "DATABASE_PATH": str(tmp_path / "backtest.db")},
    )

    assert result.returncode == 2, result.stdout
    assert "--llm-reasoning-effort requires" not in result.stderr
    assert "signed execution handoff" in result.stderr


def test_effort_is_trimmed_and_lowercased_before_the_engine_sees_it():
    """Source-shape pin: the engine gets the normalised value, not raw argv."""
    source = Path("dashboard/scripts/backtest_hourly_agent.py").read_text()
    assert 'args.llm_reasoning_effort = (args.llm_reasoning_effort or "").strip().lower() or None' in source
