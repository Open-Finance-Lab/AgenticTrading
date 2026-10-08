"""Repeat runs of an LLM leaderboard entry publish a median and a range (#602).

One model run is one draw: three DeepSeek V4 Pro reruns with identical inputs
and pinned sampling diverged on the first bar and finished between -1.25% and
-0.37% (#539). A board that ranks one curve per model ranks the dice.

Repeats live under their own mode (``leaderboard_sample``), so the primary row
and every lookup keyed on ``leaderboard`` are untouched; deleting the samples
restores the board. These cases pin that isolation, the median choice, and the
pooling rule (only runs that recorded the same config are one experiment).
"""

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.tests._frontend_source import fn_body
from dashboard.backend.tests.test_leaderboard_curve_integrity import (  # noqa: F401
    _DISPLAY_CAPITAL,
    _END,
    _FRONTEND,
    _LEADERBOARD_JS,
    _OTHER_CAPITAL,
    _SESSION,
    _START,
    _entry,
    board,
)

_ENTRY = "deepseek_v4_pro"
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
# The sampling pin leaderboard.json sets for _ENTRY, read rather than restated:
# a fixture that hard-coded the pre-#605 values (none) drifted from the entry
# the day it was re-pinned, and every "same experiment" case here read as stale.
_PINNED = next(
    s for s in lb_service.load_leaderboard_config()["strategies"] if s["id"] == _ENTRY
)
_TEMPERATURE = _PINNED.get("temperature")
_REASONING_EFFORT = _PINNED.get("reasoning_effort")


def _config(**overrides):
    md = {
        "entry_id": _ENTRY,
        "model_id": "deepseek/deepseek-v4-pro",
        "integration": "commonstack",
        "temperature": _TEMPERATURE,
        "reasoning_effort": _REASONING_EFFORT,
        "strategy_prompt": None,
        "llm_max_output_tokens": 2000,
        "initial_capital": _DISPLAY_CAPITAL,
        "market_data_feed": "sip",
    }
    md.update(overrides)
    return md


def _curve(start, final, days):
    """``days`` daily points from Apr 15: ``start`` until the last, then ``final``."""
    return [
        {"timestamp": f"2026-04-{15 + i:02d}T14:00:00+00:00",
         "equity": final if i == days - 1 else start,
         "cash": final if i == days - 1 else start,
         "positions_value": 0, "daily_return": 0}
        for i in range(days)
    ]


def _seed_sample(
    board,
    n,
    total_return,
    *,
    metadata="default",
    entry=_ENTRY,
    capital=_DISPLAY_CAPITAL,
    session_id=_SESSION,
    days=2,
    curve=None,
):
    """One repeat row; ``curve`` replaces the flat ``_curve`` path (see ``_path``)."""
    run_id = lb_service._sample_run_id(entry, _START, _END, n)
    final = capital * (1 + total_return)
    board.db.insert_run(
        run_id=run_id,
        session_id=session_id,
        agent_name="DeepSeek V4 Pro",
        mode=lb_service.LEADERBOARD_SAMPLE_MODE,
        start_date=_START,
        end_date=_END,
        initial_equity=capital,
        final_equity=final,
        total_return=total_return,
        sharpe_ratio=1.0,
        max_drawdown=-0.01,
        num_trades=3,
        llm_model=entry,
        llm_calls=10,
        metadata=_config(initial_capital=capital) if metadata == "default" else metadata,
    )
    board._curves[run_id] = curve if curve is not None else _curve(capital, final, days)
    return run_id


def _path(capital, levels):
    """A stored curve from ``{april_day: multiple of capital}``; None is a NULL.

    The interior is what a band is about, and ``_curve`` is flat until its last
    point, so every member of a band case is given its own path here.
    """
    return [
        {"timestamp": f"2026-04-{day:02d}T14:00:00+00:00",
         "equity": None if m is None else capital * m,
         "cash": None if m is None else capital * m,
         "positions_value": 0, "daily_return": 0}
        for day, m in sorted(levels.items())
    ]


def _seed_primary(board, total_return=0.0749, *, metadata=None, days=2):
    """The primary row; ``metadata=None`` is a July-vintage row that recorded none."""
    run_id = board.seed_run(
        _ENTRY,
        initial_equity=_DISPLAY_CAPITAL,
        equities=[_DISPLAY_CAPITAL] * (days - 1)
        + [_DISPLAY_CAPITAL * (1 + total_return)],
        total_return=total_return,
    )
    if metadata is not None:
        board.set_metadata(run_id, metadata)
    return run_id


def _seed_baseline(board, entry_id="djia_index", days=2):
    return board.seed_run(
        entry_id,
        initial_equity=_DISPLAY_CAPITAL,
        equities=[_DISPLAY_CAPITAL] * (days - 1) + [_DISPLAY_CAPITAL * 1.01],
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


@pytest.mark.parametrize("first,second", [(0.02, -0.01), (-0.01, 0.02)])
def test_an_even_count_breaks_the_tie_on_the_draw_not_the_return(board, first, second):
    """A real run, never an average -- and not always the worse of the two.

    The lower middle published the minimum of two runs every time, which ranked
    the entry below what it measured. The earlier draw leans neither way.
    """
    _seed_primary(board)
    earlier = _seed_sample(board, 1, first)
    _seed_sample(board, 2, second)

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)
    assert entry["run_id"] == earlier
    assert entry["cumulative_return"] == pytest.approx(first)
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
    temperature = _TEMPERATURE
    reasoning_effort = _REASONING_EFFORT
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
    assert again["config_drift"] == []
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
            fn_body("function boardSignedPercent(", _LEADERBOARD_JS),
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


@pytestmark_node
def test_an_even_count_is_labelled_middle_not_median():
    """Two middle runs and no median run: the label must not claim one."""
    entry = "{samples: {count: 2, min_return: -0.01, max_return: 0.02}}"
    assert _label(entry) == "middle of 2 · -1.00% to +2.00%"
    assert _label(entry, long=True) == "Middle of 2 runs · -1.00% to +2.00%"


@pytestmark_node
def test_the_range_shares_the_board_percent_format():
    """A zero bound renders like every other board percent: unsigned."""
    entry = "{samples: {count: 3, min_return: 0, max_return: 0.021}}"
    assert _label(entry) == "median of 3 · 0.00% to +2.10%"


def test_the_detail_panel_formats_the_runs_label_once():
    assert _LEADERBOARD_JS.count("formatLeaderboardSamples(entry, true)") == 1


def test_the_sample_label_cannot_widen_the_compact_return_column():
    """The compact table is 10px with a 110px Return column; a fixed 11px
    nowrap sub-label was larger than the return it annotates and wider than
    the column."""
    css = (_FRONTEND / "styles.css").read_text(encoding="utf-8")
    block = css[css.index(".leaderboard-sample-range {"):]
    block = block[: block.index("}")]
    assert "nowrap" not in block
    assert re.search(r"font-size:\s*[\d.]+em\b", block), block


# --------------------------------------------------------------------------
# which pool publishes: config match before size
# --------------------------------------------------------------------------


def test_repeats_of_a_replaced_model_never_outvote_a_fresh_primary(board, capsys):
    """A config edit, then a fresh primary: three old repeats must not hide it."""
    primary = _seed_primary(board, 0.03, metadata=_config())
    for n, r in enumerate((-0.01, 0.0, 0.01), start=1):
        _seed_sample(board, n, r, metadata=_config(model_id="deepseek/deepseek-v3"))

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["run_id"] == primary
    assert entry["samples"] == {"count": 1}
    out = capsys.readouterr().out
    assert "3 repeat run(s)" in out and "model_id" in out
    assert "--samples N --force" in out


def test_a_matching_pool_beats_a_larger_stale_one(board):
    _seed_primary(board)
    for n, r in enumerate((-0.03, -0.02, -0.01), start=1):
        _seed_sample(board, n, r, metadata=_config(model_id="deepseek/deepseek-v3"))
    earlier = _seed_sample(board, 4, 0.01)
    _seed_sample(board, 5, 0.02)

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["samples"]["count"] == 2
    assert entry["samples"]["min_return"] == pytest.approx(0.01)
    assert entry["run_id"] == earlier


def test_repeats_at_a_stale_seed_do_not_hide_a_primary_at_the_board_seed(board):
    """Before: the $10k median beat the $100k primary, then the capital guard
    dropped it, and the entry vanished from the board."""
    _seed_baseline(board)
    primary = _seed_primary(board, 0.02, metadata=_config())
    for n, r in enumerate((-0.01, 0.0, 0.01), start=1):
        _seed_sample(board, n, r, capital=_OTHER_CAPITAL)

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry is not None
    assert entry["run_id"] == primary


def test_an_outlier_published_from_repeats_names_the_repeat_remedy(board, capsys):
    """A fresh primary cannot fix it, so the warning must not prescribe one."""
    _seed_baseline(board, "djia_index")
    _seed_baseline(board, "spy_index")
    for n, r in enumerate((-0.01, 0.0, 0.01), start=1):
        _seed_sample(board, n, r, capital=_OTHER_CAPITAL)

    payload = lb_service.get_leaderboard()

    assert _entry(payload, _ENTRY) is None
    lines = [
        ln for ln in capsys.readouterr().out.splitlines()
        if _ENTRY in ln and "omitted" in ln
    ]
    assert len(lines) == 1, lines
    assert "--samples N --force" in lines[0]
    assert "deploy_model_run(force_refresh=True)" not in lines[0]


def test_a_clamped_or_fallback_tape_is_a_different_experiment(board):
    _seed_primary(board)
    _seed_sample(board, 1, -0.01, metadata=_config(end_clamped=False, sip_fallback_to_iex=False))
    _seed_sample(board, 2, 0.01, metadata=_config(end_clamped=False, sip_fallback_to_iex=False))
    _seed_sample(board, 3, 0.40, metadata=_config(end_clamped=True, sip_fallback_to_iex=False))

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["samples"]["count"] == 2
    assert entry["samples"]["max_return"] == pytest.approx(0.01)


def test_newest_means_last_written_on_either_backend():
    """Postgres keeps the first-seen ``created_at`` on a re-insert, so a
    ``--force`` rerun is newer only by ``updated_at``, which both backends set."""
    entry = {"id": _ENTRY, "model_id": "deepseek/deepseek-v4-pro", "integration": "commonstack"}

    def row(n, ceiling, created, updated):
        return {
            "run_id": f"lb_x_s{n}",
            "total_return": 0.01 * n,
            "initial_equity": _DISPLAY_CAPITAL,
            "metadata": _config(llm_max_output_tokens=ceiling),
            "created_at": created,
            "updated_at": updated,
        }

    older = [row(n, 2000, "2026-09-01 00:00:00", "2026-09-01 00:00:00") for n in (1, 2)]
    rerun = [row(n, 4096, "2026-08-01 00:00:00", "2026-10-01 00:00:00") for n in (3, 4)]

    pool, _ = lb_service._pooled_samples(older + rerun, entry, _DISPLAY_CAPITAL)

    assert {r["run_id"] for r in pool} == {"lb_x_s3", "lb_x_s4"}


# --------------------------------------------------------------------------
# the primary as one more draw
# --------------------------------------------------------------------------


def test_a_primary_of_the_same_experiment_counts_as_a_draw(board):
    primary = _seed_primary(board, 0.005, metadata=_config())
    _seed_sample(board, 1, -0.01)
    _seed_sample(board, 2, 0.02)

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["samples"]["count"] == 3
    assert entry["run_id"] == primary


def test_a_primary_over_different_days_is_not_a_draw(board):
    """Same recorded config, different traded days: an earlier engine's row."""
    primary = _seed_primary(board, 0.005, metadata=_config(), days=3)
    _seed_sample(board, 1, -0.01)

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["run_id"] == primary
    assert entry["samples"] == {"count": 1}


# --------------------------------------------------------------------------
# the board's window and its one scan
# --------------------------------------------------------------------------


def test_rows_that_traded_different_days_are_reported(board, capsys):
    _seed_baseline(board, days=2)
    for n, r in enumerate((-0.01, 0.0, 0.01), start=1):
        _seed_sample(board, n, r, days=3)

    lb_service.get_leaderboard()
    lb_service.get_leaderboard()

    lines = [
        ln for ln in capsys.readouterr().out.splitlines()
        if "traded different days" in ln
    ]
    assert len(lines) == 1, lines
    assert "2026-04-15 → 2026-04-16: djia_index" in lines[0]
    assert f"2026-04-15 → 2026-04-17: {_ENTRY}" in lines[0]


def test_rows_over_the_same_days_raise_no_window_warning(board, capsys):
    _seed_baseline(board)
    _seed_sample(board, 1, -0.01)
    _seed_sample(board, 2, 0.01)

    lb_service.get_leaderboard()

    assert "traded different days" not in capsys.readouterr().out


def test_the_board_reads_the_session_once(board, monkeypatch):
    """Primaries and repeats come off one scan, not one per entry plus one."""
    _seed_primary(board)
    _seed_baseline(board)
    _seed_sample(board, 1, -0.01)
    _seed_sample(board, 2, 0.01)
    calls = []
    real = board.db.get_runs_by_session
    monkeypatch.setattr(
        board.db, "get_runs_by_session", lambda sid: calls.append(sid) or real(sid)
    )

    lb_service.get_leaderboard()

    assert calls == [_SESSION]


# --------------------------------------------------------------------------
# the automated daily paths count a sample-only entry as present
# --------------------------------------------------------------------------


def _daily_config():
    cfg = dict(lb_service.load_leaderboard_config())
    cfg.update(session_id=_SESSION, start_date=_START, end_date=_END, period="daily")
    return cfg


def test_an_entry_published_from_samples_alone_is_not_pending(board):
    _seed_sample(board, 1, 0.01)

    status = lb_service._daily_models_status(_daily_config())

    assert _ENTRY not in status["pending_entry_ids"]
    assert status["models_cached"] == 1


def test_the_daily_refresh_does_not_bill_a_primary_for_a_sampled_entry(board, monkeypatch):
    _seed_sample(board, 1, 0.01)
    cfg = _daily_config()
    monkeypatch.setattr(lb_service, "resolve_leaderboard_config", lambda period="contest": cfg)
    monkeypatch.setattr(lb_service, "_daily_refresh_state", lambda: {})
    monkeypatch.setattr(lb_service, "_save_daily_refresh_state", lambda state: None)
    deployed = []
    monkeypatch.setattr(
        lb_service,
        "deploy_model_run",
        lambda entry_id, **kw: deployed.append(entry_id) or {"entry_id": entry_id},
    )

    lb_service.refresh_daily_leaderboard(deploy_models=True)
    assert _ENTRY not in deployed
    assert len(deployed) == len(lb_service.llm_leaderboard_entries(cfg)) - 1

    deployed.clear()
    lb_service.refresh_daily_leaderboard(deploy_models=True, force_refresh=True)
    assert _ENTRY in deployed, "an explicit force still re-runs it"


# --------------------------------------------------------------------------
# deploy_model_run(sample=n): drift and ownership
# --------------------------------------------------------------------------


def test_a_cached_sample_under_a_replaced_config_is_reported(board, fake_compute, capsys):
    _seed_sample(board, 1, 0.01, metadata=_config(model_id="deepseek/deepseek-v3"))

    result = lb_service.deploy_model_run(_ENTRY, sample=1)

    assert result["cached"] is True
    assert result["config_drift"] == ["model_id"]
    assert fake_compute == [], "reported, never re-billed without --force"
    out = capsys.readouterr().out
    assert "sample 1" in out and "model_id" in out and "--force" in out


def test_a_cached_sample_at_a_stale_seed_is_reported(board, fake_compute):
    _seed_sample(board, 1, 0.01, capital=_OTHER_CAPITAL)

    result = lb_service.deploy_model_run(_ENTRY, sample=1)

    assert result["config_drift"] == ["initial_capital"]


def test_a_sample_of_another_board_is_neither_reused_nor_moved(board, fake_compute):
    other = _seed_sample(board, 1, 0.01, session_id="leaderboard-daily")

    for force in (False, True):
        with pytest.raises(ValueError, match="already belongs to session 'leaderboard-daily'"):
            lb_service.deploy_model_run(_ENTRY, sample=1, force_refresh=force)

    assert board.db.get_run(other)["session_id"] == "leaderboard-daily"
    assert fake_compute == []


# --------------------------------------------------------------------------
# the deploy CLI reports the board's answer
# --------------------------------------------------------------------------


def _load_deploy_script(monkeypatch):
    import dotenv

    # Never read a developer's dashboard/.env into the suite's environment.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.syspath_prepend(str(_SCRIPTS_DIR))
    path = _SCRIPTS_DIR / "deploy_leaderboard_model.py"
    spec = importlib.util.spec_from_file_location("deploy_leaderboard_model_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli_args(samples=3):
    return argparse.Namespace(
        entry=_ENTRY,
        samples=samples,
        force=False,
        start=None,
        end=None,
        allow_fallback=False,
        period="contest",
    )


def test_the_cli_reports_what_the_board_publishes(board, monkeypatch):
    _seed_primary(board)
    for n, r in enumerate((-0.01, 0.0, 0.01), start=1):
        _seed_sample(board, n, r)
    _seed_sample(board, 4, 0.50, metadata=_config(reasoning_effort="disabled"))
    script = _load_deploy_script(monkeypatch)

    line = script._publication_line(_cli_args())

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)
    assert entry["run_id"] in line
    assert "median of 3 runs" in line
    assert "+1.00%" in line and "+50.00%" not in line


def test_the_cli_stops_on_a_failed_sample_and_still_reports(board, monkeypatch, capsys):
    """A RuntimeError used to escape the loop as a traceback."""
    only = _seed_sample(board, 1, -0.01)
    script = _load_deploy_script(monkeypatch)
    calls = []

    def fake_deploy(entry_id, **kwargs):
        calls.append(kwargs["sample"])
        if kwargs["sample"] == 2:
            raise RuntimeError("No equity curve produced")
        return {"run_id": only, "cached": True, "total_return": -0.01, "config_drift": []}

    monkeypatch.setattr(script, "deploy_model_run", fake_deploy)

    assert script._deploy_samples(_cli_args()) == 1
    assert calls == [1, 2]
    out = capsys.readouterr().out
    assert "Sample 2 failed" in out
    assert f"Board publishes a single run: {only}" in out


# --------------------------------------------------------------------------
# the chart band: the pooled runs' min/max envelope behind the median
# --------------------------------------------------------------------------

_CAP = _DISPLAY_CAPITAL


def _index_of_day(entry, day):
    """Where the ``2026-04-<day>`` bar lands on the entry's aligned axis.

    Matched on the bar's hour: the 15th also carries the midnight open tick.
    """
    return next(
        i for i, pt in enumerate(entry["equity_curve"])
        if pt["timestamp"].startswith(f"2026-04-{day:02d}T14")
    )


def _seed_three_paths(board):
    """Three repeats whose interiors differ: a dip, a flat median, a spike."""
    _seed_sample(board, 1, -0.0125, curve=_path(_CAP, {15: 1.0, 16: 0.97, 17: 0.9875}))
    median = _seed_sample(board, 2, -0.0048, curve=_path(_CAP, {15: 1.0, 16: 1.0, 17: 0.9952}))
    _seed_sample(board, 3, 0.0210, curve=_path(_CAP, {15: 1.0, 16: 1.04, 17: 1.021}))
    return median


def _assert_band_brackets_the_median(entry):
    band = entry["sample_band"]
    assert len(band["lower"]) == len(band["upper"]) == len(entry["equity_curve"])
    for lo, pt, hi in zip(band["lower"], entry["equity_curve"], band["upper"]):
        if None not in (lo, pt["equity"], hi):
            assert lo <= pt["equity"] <= hi


def test_a_sampled_entry_draws_a_band_around_its_median(board):
    median = _seed_three_paths(board)

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["run_id"] == median
    band = entry["sample_band"]
    assert band["runs"] == 3
    # A sibling, never inside the samples block the table labels.
    assert "sample_band" not in entry["samples"]
    _assert_band_brackets_the_median(entry)
    # Opens at the board's capital, like every curve's open tick.
    assert entry["equity_curve"][0]["equity"] == pytest.approx(_CAP)
    assert band["lower"][0] == pytest.approx(_CAP)
    assert band["upper"][0] == pytest.approx(_CAP)
    # Closes on the range the label prints.
    assert band["lower"][-1] == pytest.approx(_CAP * (1 + entry["samples"]["min_return"]))
    assert band["upper"][-1] == pytest.approx(_CAP * (1 + entry["samples"]["max_return"]))
    # And the interior is the members' paths, not a line between the endpoints.
    mid = _index_of_day(entry, 16)
    assert entry["equity_curve"][mid]["equity"] == pytest.approx(_CAP)
    assert band["lower"][mid] == pytest.approx(_CAP * 0.97)
    assert band["upper"][mid] == pytest.approx(_CAP * 1.04)


def test_a_single_run_entry_has_no_band(board):
    _seed_primary(board)
    _seed_sample(board, 1, -0.02)

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["samples"] == {"count": 1}
    assert "sample_band" not in entry


def test_a_lone_sample_has_no_band(board):
    _seed_sample(board, 1, 0.01)
    assert "sample_band" not in _entry(lb_service.get_leaderboard(), _ENTRY)


def test_baselines_have_no_band(board):
    _seed_baseline(board)
    _seed_three_paths(board)
    payload = lb_service.get_leaderboard()
    assert "sample_band" not in _entry(payload, "djia_index")
    assert "sample_band" in _entry(payload, _ENTRY)


def test_a_sample_of_another_config_does_not_widen_the_band(board):
    _seed_three_paths(board)
    _seed_sample(
        board, 4, 0.50,
        metadata=_config(reasoning_effort="disabled"),
        curve=_path(_CAP, {15: 1.0, 16: 2.0, 17: 1.5}),
    )

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    band = entry["sample_band"]
    assert band["runs"] == 3
    assert max(v for v in band["upper"] if v is not None) == pytest.approx(_CAP * 1.04)


def test_a_primary_that_joins_the_pool_contributes_its_curve(board):
    primary = _seed_primary(board, 0.03, metadata=_config(), days=3)
    board._curves[primary] = _path(_CAP, {15: 1.0, 16: 1.06, 17: 1.03})
    _seed_sample(board, 1, -0.01, curve=_path(_CAP, {15: 1.0, 16: 0.99, 17: 0.99}))
    median = _seed_sample(board, 2, 0.0, curve=_path(_CAP, {15: 1.0, 16: 1.0, 17: 1.0}))

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["run_id"] == median
    assert entry["samples"]["count"] == 3
    band = entry["sample_band"]
    assert band["runs"] == 3
    assert band["upper"][_index_of_day(entry, 16)] == pytest.approx(_CAP * 1.06)
    assert band["upper"][-1] == pytest.approx(_CAP * 1.03)


def test_a_member_at_a_refused_seed_leaves_the_pool(board, capsys):
    """The rule that omits a median run at another seed (issue #365, see
    test_an_outlier_published_from_repeats_names_the_repeat_remedy) is applied
    where the pool is formed: a repeat recorded at another seed is not a draw
    of the experiment, so it is out of the median and the "lo to hi" label as
    well as the band -- never counted by one and refused by another."""
    _seed_baseline(board)
    low = _seed_sample(board, 1, -0.01, curve=_path(_CAP, {15: 1.0, 16: 0.99, 17: 0.99}))
    _seed_sample(board, 2, 0.0, curve=_path(_CAP, {15: 1.0, 16: 1.0, 17: 1.0}))
    # Recorded the board's config but was run at another seed.
    stray = _seed_sample(
        board, 3, 0.01,
        capital=_OTHER_CAPITAL,
        metadata=_config(),
        curve=_path(_OTHER_CAPITAL, {15: 1.0, 16: 1.5, 17: 1.01}),
    )

    lb_service.get_leaderboard()
    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    # Two left: the even-count tie goes to the lower draw number.
    assert entry["run_id"] == low
    assert entry["samples"]["count"] == 2
    assert entry["samples"]["max_return"] == pytest.approx(0.0)
    band = entry["sample_band"]
    assert band["runs"] == 2
    assert max(v for v in band["upper"] if v is not None) == pytest.approx(_CAP)
    lines = [
        ln for ln in capsys.readouterr().out.splitlines()
        if "left out of the median" in ln
    ]
    assert len(lines) == 1, lines
    assert stray in lines[0] and _ENTRY in lines[0]


def test_the_cli_reports_the_vetted_pool(board):
    """The deploy CLI's closing line goes through the same vetting."""
    _seed_three_paths(board)
    _seed_sample(board, 4, 0.03, capital=_OTHER_CAPITAL, metadata=_config())
    cfg = dict(lb_service.load_leaderboard_config())
    cfg.update(session_id=_SESSION, start_date=_START, end_date=_END)

    described = lb_service.describe_entry_publication(_ENTRY, config=cfg)
    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert described["samples"] == entry["samples"]
    assert described["samples"]["count"] == 3


def test_a_band_always_covers_every_run_the_label_counts(board):
    """A 4-run pool losing one member at the pool is a 3-run label AND a 3-run
    band -- the two cannot disagree about how many runs they stand on."""
    _seed_three_paths(board)
    _seed_sample(
        board, 4, 0.03,
        capital=_OTHER_CAPITAL,
        metadata=_config(),
        curve=_path(_OTHER_CAPITAL, {15: 1.0, 16: 1.05, 17: 1.03}),
    )

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["samples"]["count"] == 3
    assert entry["sample_band"]["runs"] == 3
    assert entry["samples"]["max_return"] == pytest.approx(0.021)


def test_a_pool_left_with_one_run_draws_no_band(board):
    _seed_baseline(board)
    only = _seed_sample(board, 1, 0.0)
    _seed_sample(board, 2, 0.01, capital=_OTHER_CAPITAL, metadata=_config())

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["run_id"] == only
    assert entry["samples"]["count"] == 1
    assert "sample_band" not in entry


def test_a_repeat_over_other_days_leaves_the_pool(board, capsys):
    """As-of alignment would carry an early-ender's last value flat across the
    rest of the window -- a path nobody measured -- so it is not a draw."""
    _seed_sample(board, 1, -0.01, curve=_path(_CAP, {15: 1.0, 16: 0.99, 17: 0.99}))
    median = _seed_sample(board, 2, 0.0, curve=_path(_CAP, {15: 1.0, 16: 1.0, 17: 1.0}))
    _seed_sample(board, 3, 0.01, curve=_path(_CAP, {15: 1.0, 16: 1.02, 17: 1.01}))
    short = _seed_sample(board, 4, 0.30, curve=_path(_CAP, {15: 1.0, 16: 1.30}))

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["run_id"] == median
    assert entry["samples"]["count"] == 3
    assert entry["samples"]["max_return"] == pytest.approx(0.01)
    assert entry["sample_band"]["runs"] == 3
    assert max(v for v in entry["sample_band"]["upper"] if v is not None) == (
        pytest.approx(_CAP * 1.02)
    )
    out = capsys.readouterr().out
    assert short in out and "covers 2026-04-15 → 2026-04-16" in out


def test_a_repeat_with_no_stored_value_leaves_the_pool(board, capsys):
    _seed_sample(board, 1, -0.01, curve=_path(_CAP, {15: 1.0, 16: 0.99, 17: 0.99}))
    _seed_sample(board, 2, 0.01, curve=_path(_CAP, {15: 1.0, 16: 1.01, 17: 1.01}))
    empty = _seed_sample(board, 3, 0.5, curve=_path(_CAP, {15: None, 16: None, 17: None}))

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["samples"]["count"] == 2
    assert entry["samples"]["max_return"] == pytest.approx(0.01)
    assert entry["sample_band"]["runs"] == 2
    assert empty in capsys.readouterr().out


def test_a_null_member_point_is_a_gap_never_a_zero(board):
    _seed_sample(board, 1, -0.01, curve=_path(_CAP, {15: 1.0, 16: None, 17: 0.99}))
    _seed_sample(board, 2, 0.0, curve=_path(_CAP, {15: 1.0, 16: 1.0, 17: 1.0}))
    _seed_sample(board, 3, 0.01, curve=_path(_CAP, {15: 1.0, 16: 1.02, 17: 1.01}))

    payload = lb_service.get_leaderboard()
    entry = _entry(payload, _ENTRY)

    band = entry["sample_band"]
    mid = _index_of_day(entry, 16)
    assert band["lower"][mid] == pytest.approx(_CAP)
    assert band["upper"][mid] == pytest.approx(_CAP * 1.02)
    json.dumps(payload, allow_nan=False)


def test_a_point_no_member_has_is_null_and_the_payload_serializes(board):
    for n, r in enumerate((-0.01, 0.0, 0.01), start=1):
        _seed_sample(board, n, r, curve=_path(_CAP, {15: 1.0, 16: None, 17: 1 + r}))

    payload = lb_service.get_leaderboard()
    entry = _entry(payload, _ENTRY)

    mid = _index_of_day(entry, 16)
    assert entry["sample_band"]["lower"][mid] is None
    assert entry["sample_band"]["upper"][mid] is None
    _assert_band_brackets_the_median(entry)
    json.dumps(payload, allow_nan=False)


def test_members_align_by_timestamp_not_position(board):
    """A skipped bar carries the last value forward, the way the board aligns
    every curve -- never the next point by position."""
    median = _seed_sample(board, 1, -0.001, curve=_path(_CAP, {15: 1.0, 16: 1.0, 17: 0.999}))
    _seed_sample(board, 2, 0.001, curve=_path(_CAP, {15: 1.0, 16: 1.0, 17: 1.001}))
    _seed_sample(board, 3, -0.10, curve=_path(_CAP, {15: 1.10, 17: 0.90}))  # skips the 16th

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["run_id"] == median
    band = entry["sample_band"]
    assert band["runs"] == 3
    # Matched from the end by position, the short member would be shifted one
    # bar late: the 15th would read its open tick, not its 1.10.
    first = _index_of_day(entry, 15)
    assert band["upper"][first] == pytest.approx(_CAP * 1.10)
    mid = _index_of_day(entry, 16)
    assert band["lower"][mid] == pytest.approx(_CAP)
    assert band["upper"][mid] == pytest.approx(_CAP * 1.10)
    assert band["lower"][-1] == pytest.approx(_CAP * 0.90)
    assert band["upper"][-1] == pytest.approx(_CAP * 1.001)
    _assert_band_brackets_the_median(entry)


def test_every_drawn_band_point_covers_every_run(board):
    """A member's NULL carries its last real value like a skipped bar. Dropping
    it from min/max instead drew an N-1 run envelope under ``runs: N``."""
    _seed_sample(board, 1, -0.05, curve=_path(_CAP, {15: 1.0, 16: 0.95, 17: None, 18: 0.95}))
    median = _seed_sample(board, 2, 0.0, curve=_path(_CAP, {15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0}))
    _seed_sample(board, 3, 0.01, curve=_path(_CAP, {15: 1.0, 16: 1.01, 17: 1.01, 18: 1.01}))

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["run_id"] == median
    band = entry["sample_band"]
    assert band["lower"][_index_of_day(entry, 17)] == pytest.approx(_CAP * 0.95)


def test_a_pool_is_read_in_one_batch_and_never_twice(board, monkeypatch):
    """The curves read to vet a pool are the ones the board publishes and
    bands: one batch per pooled entry, and the median is not re-read."""
    primary = _seed_primary(board)
    _seed_baseline(board)
    first = _seed_sample(board, 1, -0.01)
    median = _seed_sample(board, 2, 0.0)
    last = _seed_sample(board, 3, 0.01)
    batches, singles, in_batch = [], [], []
    real_batch, real_single = board.db.get_equity_curves, board.db.get_equity_curve

    def batch(ids):
        batches.append(list(ids))
        in_batch.append(True)  # SQLite's batch loops over the single read
        try:
            return real_batch(ids)
        finally:
            in_batch.pop()

    def single(run_id):
        if not in_batch:
            singles.append(run_id)
        return real_single(run_id)

    monkeypatch.setattr(board.db, "get_equity_curves", batch)
    monkeypatch.setattr(board.db, "get_equity_curve", single)

    entry = _entry(lb_service.get_leaderboard(), _ENTRY)

    assert entry["run_id"] == median and entry["run_id"] != primary
    assert [sorted(c) for c in batches] == [sorted([first, median, last])]
    assert not {first, median, last} & set(singles)
