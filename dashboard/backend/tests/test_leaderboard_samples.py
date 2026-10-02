"""Repeat runs of an LLM leaderboard entry publish a median and a range (#602).

One model run is one draw: three DeepSeek V4 Pro reruns with identical inputs
and pinned sampling diverged on the first bar and finished between -1.25% and
-0.37% (#539). A board that ranks one curve per model ranks the dice.

Repeats live under their own mode (``leaderboard_sample``), so the primary row
and every lookup keyed on ``leaderboard`` are untouched; deleting the samples
restores the board. These cases pin that isolation, the median choice, and the
pooling rule (only runs that recorded the same config are one experiment).
"""

import json
import shutil
import subprocess

import pytest

import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.tests._frontend_source import fn_body
from dashboard.backend.tests.test_leaderboard_curve_integrity import (  # noqa: F401
    _DISPLAY_CAPITAL,
    _END,
    _LEADERBOARD_JS,
    _SESSION,
    _START,
    _entry,
    board,
)

_ENTRY = "deepseek_v4_pro"


def _config(**overrides):
    md = {
        "entry_id": _ENTRY,
        "model_id": "deepseek/deepseek-v4-pro",
        "integration": "commonstack",
        "temperature": None,
        "reasoning_effort": None,
        "strategy_prompt": None,
        "llm_max_output_tokens": 2000,
        "initial_capital": _DISPLAY_CAPITAL,
        "market_data_feed": "sip",
    }
    md.update(overrides)
    return md


def _seed_sample(board, n, total_return, *, metadata="default", entry=_ENTRY, created_at=None):
    run_id = lb_service._sample_run_id(entry, _START, _END, n)
    final = _DISPLAY_CAPITAL * (1 + total_return)
    board.db.insert_run(
        run_id=run_id,
        session_id=_SESSION,
        agent_name="DeepSeek V4 Pro",
        mode=lb_service.LEADERBOARD_SAMPLE_MODE,
        start_date=_START,
        end_date=_END,
        initial_equity=_DISPLAY_CAPITAL,
        final_equity=final,
        total_return=total_return,
        sharpe_ratio=1.0,
        max_drawdown=-0.01,
        num_trades=3,
        llm_model=entry,
        llm_calls=10,
        metadata=_config() if metadata == "default" else metadata,
    )
    board._curves[run_id] = [
        {"timestamp": "2026-04-15T14:00:00+00:00", "equity": _DISPLAY_CAPITAL,
         "cash": _DISPLAY_CAPITAL, "positions_value": 0, "daily_return": 0},
        {"timestamp": "2026-04-16T14:00:00+00:00", "equity": final,
         "cash": final, "positions_value": 0, "daily_return": 0},
    ]
    return run_id


def _seed_primary(board, total_return=0.0749):
    return board.seed_run(
        _ENTRY,
        initial_equity=_DISPLAY_CAPITAL,
        equities=[_DISPLAY_CAPITAL, _DISPLAY_CAPITAL * (1 + total_return)],
        total_return=total_return,
    )


# --------------------------------------------------------------------------
# ids
# --------------------------------------------------------------------------


def test_sample_ids_extend_the_primary_id():
    primary = lb_service._run_id(_ENTRY, _START, _END)
    assert lb_service._sample_run_id(_ENTRY, _START, _END, 2) == primary + "_s2"


@pytest.mark.parametrize("bad", [0, -1, True, 1.0, "2", None])
def test_sample_index_must_be_a_positive_int(bad):
    with pytest.raises(ValueError):
        lb_service._sample_run_id(_ENTRY, _START, _END, bad)


# --------------------------------------------------------------------------
# publication
# --------------------------------------------------------------------------


def test_three_comparable_samples_publish_the_median_run(board):
    primary = _seed_primary(board, 0.0749)
    _seed_sample(board, 1, -0.0125)
    median = _seed_sample(board, 2, -0.0048)
    _seed_sample(board, 3, 0.0210)

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["run_id"] == median, "the median run, not the July primary"
    assert entry["run_id"] != primary
    assert entry["cumulative_return"] == pytest.approx(-0.0048)
    # The table and the chart come from the same row, so they agree.
    assert entry["portfolio_value"] == pytest.approx(_DISPLAY_CAPITAL * (1 - 0.0048))
    assert entry["equity_curve"][-1]["equity"] == pytest.approx(
        _DISPLAY_CAPITAL * (1 - 0.0048)
    )
    assert entry["samples"] == {
        "count": 3,
        "min_return": pytest.approx(-0.0125),
        "max_return": pytest.approx(0.0210),
        "returns": [pytest.approx(-0.0125), pytest.approx(-0.0048), pytest.approx(0.0210)],
    }


def test_an_even_count_publishes_the_lower_middle_run(board):
    """A real run, never an average of two curves nobody traded."""
    _seed_primary(board)
    low = _seed_sample(board, 1, -0.01)
    _seed_sample(board, 2, 0.02)

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)
    assert entry["run_id"] == low
    assert entry["samples"]["count"] == 2


def test_one_sample_beside_a_primary_changes_nothing(board):
    primary = _seed_primary(board)
    _seed_sample(board, 1, -0.02)

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)
    assert entry["run_id"] == primary
    assert entry["samples"] == {"count": 1}


def test_a_lone_sample_with_no_primary_still_publishes(board):
    only = _seed_sample(board, 1, 0.01)
    entry = _entry(lb_service.get_leaderboard(), _ENTRY)
    assert entry["run_id"] == only
    assert entry["samples"]["count"] == 1


def test_samples_of_different_configs_are_never_pooled(board):
    """The largest same-config group wins; the odd one out is a different experiment."""
    _seed_primary(board)
    _seed_sample(board, 1, -0.01)
    median = _seed_sample(board, 2, 0.00)
    _seed_sample(board, 3, 0.01)
    _seed_sample(board, 4, 0.50, metadata=_config(reasoning_effort="disabled"))

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)
    assert entry["samples"]["count"] == 3
    assert entry["samples"]["max_return"] == pytest.approx(0.01)
    assert entry["run_id"] == median


def test_samples_that_recorded_no_config_never_pool(board):
    primary = _seed_primary(board)
    _seed_sample(board, 1, -0.01, metadata=None)
    _seed_sample(board, 2, 0.02, metadata=None)

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)
    assert entry["run_id"] == primary
    assert entry["samples"] == {"count": 1}


def test_baselines_carry_no_samples_block(board):
    board.seed_run(
        "djia_index",
        initial_equity=_DISPLAY_CAPITAL,
        equities=[_DISPLAY_CAPITAL, _DISPLAY_CAPITAL * 1.01],
    )
    entry = _entry(lb_service.get_leaderboard(), "djia_index")
    assert "samples" not in entry


def test_the_median_decides_the_rank(board):
    """The single +7.49% primary would rank first; its median does not."""
    _seed_primary(board, 0.0749)
    for n, r in enumerate((-0.0125, -0.0048, 0.0210), start=1):
        _seed_sample(board, n, r)
    board.seed_run(
        "qwen3_7_plus",
        initial_equity=_DISPLAY_CAPITAL,
        equities=[_DISPLAY_CAPITAL, _DISPLAY_CAPITAL * 1.0249],
        total_return=0.0249,
    )

    payload = lb_service.get_leaderboard()
    assert _entry(payload, "qwen3_7_plus")["rank"] < _entry(payload, _ENTRY)["rank"]


# --------------------------------------------------------------------------
# isolation: the primary lookups never see a sample
# --------------------------------------------------------------------------


def test_cached_run_lookups_ignore_sample_rows(board):
    _seed_sample(board, 1, 0.05)
    assert lb_service._find_cached_run(_ENTRY, _START, _END, _SESSION) is None
    index, _ = lb_service._cached_run_index(_START, _END, _SESSION)
    assert _ENTRY not in index


# --------------------------------------------------------------------------
# deploy_model_run(sample=n)
# --------------------------------------------------------------------------


class _FakeLLMStrategy:
    """Just enough of LLMAgentStrategy for deploy_model_run's compute path."""

    integration = "commonstack"
    temperature = None
    reasoning_effort = None
    strategy_prompt = None
    model_id = "deepseek/deepseek-v4-pro"
    input_tokens = 1000
    output_tokens = 100
    llm_calls = 4
    llm_decisions = 4
    decision_steps = 4

    def __init__(self, calls):
        self._calls = calls

    def required_symbols(self):
        return ["AAPL"]

    def run(self, bars, start_date, end_date, initial_capital):
        self._calls.append(1)
        return [
            {"timestamp": "2026-04-15T14:00:00+00:00", "equity": initial_capital,
             "cash": initial_capital, "positions_value": 0},
            {"timestamp": "2026-04-16T14:00:00+00:00", "equity": initial_capital * 1.01,
             "cash": initial_capital * 1.01, "positions_value": 0},
        ]

    def num_trades(self):
        return 2


@pytest.fixture
def fake_compute(board, monkeypatch):
    calls = []
    monkeypatch.setattr(lb_service, "get_strategy", lambda entry: _FakeLLMStrategy(calls))
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *a, **k: {"AAPL": object()})
    monkeypatch.setattr(lb_service, "feed_provenance", lambda bars: {"market_data_feed": "sip"})
    return calls


def test_deploying_a_sample_writes_its_own_row_and_caches_by_id(board, fake_compute):
    primary = _seed_primary(board)

    first = lb_service.deploy_model_run(_ENTRY, sample=2)
    assert first["cached"] is False
    assert first["run_id"] == lb_service._sample_run_id(_ENTRY, _START, _END, 2)
    row = board.db.get_run(first["run_id"])
    assert row["mode"] == lb_service.LEADERBOARD_SAMPLE_MODE
    assert row["metadata"]["model_id"] == "deepseek/deepseek-v4-pro"
    # The primary row is untouched.
    assert board.db.get_run(primary)["mode"] == lb_service.LEADERBOARD_MODE

    again = lb_service.deploy_model_run(_ENTRY, sample=2)
    assert again["cached"] is True
    assert len(fake_compute) == 1, "a cached sample must not be billed again"

    lb_service.deploy_model_run(_ENTRY, sample=2, force_refresh=True)
    assert len(fake_compute) == 2


def test_a_primary_deploy_is_not_satisfied_by_a_sample(board, fake_compute):
    """The primary lookup keys on mode; a sample row is not a cached primary."""
    lb_service.deploy_model_run(_ENTRY, sample=1)
    result = lb_service.deploy_model_run(_ENTRY)
    assert result["cached"] is False
    assert result["run_id"] == lb_service._run_id(_ENTRY, _START, _END)


def test_baselines_cannot_be_sampled(board, fake_compute):
    with pytest.raises(ValueError, match="not an LLM entry"):
        lb_service.deploy_model_run("djia_index", sample=1)
    assert fake_compute == []


# --------------------------------------------------------------------------
# frontend label
# --------------------------------------------------------------------------

pytestmark_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _label(entry_js, long=False):
    script = "\n".join(
        [
            fn_body("function formatLeaderboardSamples(", _LEADERBOARD_JS),
            f"console.log(JSON.stringify(formatLeaderboardSamples({entry_js}, "
            f"{str(long).lower()})));",
        ]
    )
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytestmark_node
def test_label_for_a_sampled_row():
    entry = "{samples: {count: 3, min_return: -0.0125, max_return: 0.021}}"
    assert _label(entry) == "median of 3 · -1.25% to +2.10%"
    assert _label(entry, long=True) == "Median of 3 runs · -1.25% to +2.10%"


@pytestmark_node
def test_label_for_a_single_run_says_so():
    assert _label("{samples: {count: 1}}") == "1 run"
    assert _label("{samples: {count: 1}}", long=True) == "Single run · not repeated"


@pytestmark_node
def test_no_label_for_a_baseline():
    assert _label("{}") == ""
    assert _label("{samples: {count: 0}}") == ""
