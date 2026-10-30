"""Who is on the Live board in a given month.

An LLM entry trades on the Live board when its id is in ``LIVE_MODEL_IDS`` and
its optional ``live_from`` month has opened. A not-yet-live entry must be
neither deployed (that bills a month-to-date replay) nor rendered (that shows a
pending row for a model nobody is running).
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import dashboard.backend.domain.leaderboard.live as live
import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.tests.test_live_leaderboard import (  # noqa: F401
    SEED,
    _insert_live_run,
    _isolated_live_session,
    _pt,
)

_ET = ZoneInfo("America/New_York")
_AUG27 = datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
_SEP3 = datetime(2026, 9, 3, 17, 39, tzinfo=_ET)
# After every live_from in leaderboard.json, so no entry is gated out.
_EVERY_JOIN_OPEN = "2099-12-01"
_DROP = object()


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    """The once-per-process warning record would hide a second test's line."""
    monkeypatch.setattr(live, "_warned_live_config", set())


def _config_with(entry_id, **keys):
    """leaderboard.json with ``keys`` set on one entry (``_DROP`` removes one)."""
    cfg = lb_service.load_leaderboard_config()
    strategies = []
    for strategy in cfg["strategies"]:
        strategy = dict(strategy)
        if strategy["id"] == entry_id:
            for key, value in keys.items():
                if value is _DROP:
                    strategy.pop(key, None)
                else:
                    strategy[key] = value
        strategies.append(strategy)
    return {**cfg, "strategies": strategies}


def _use_config(monkeypatch, cfg):
    monkeypatch.setattr(lb_service, "load_leaderboard_config", lambda: cfg)


def _live_ids(as_of):
    return [e["id"] for e in live.live_llm_entries(live.live_freeze_config(as_of))]


def test_an_entry_joins_in_its_live_from_month(monkeypatch):
    _use_config(monkeypatch, _config_with("deepseek_v4_pro", live_from="2026-09-01"))
    assert "deepseek_v4_pro" not in _live_ids(_AUG27)
    assert "deepseek_v4_pro" in _live_ids(_SEP3)


def test_month_start_names_the_board_month_for_the_base_config(monkeypatch):
    cfg = _config_with("deepseek_v4_pro", live_from="2026-09-01")
    before = live.live_board_strategies(cfg, month_start="2026-08-01")
    after = live.live_board_strategies(cfg, month_start="2026-09-01")
    assert "deepseek_v4_pro" not in {s["id"] for s in before}
    assert "deepseek_v4_pro" in {s["id"] for s in after}
    # Baselines are never gated.
    assert "spy_index" in {s["id"] for s in before}


def test_the_get_gates_on_the_live_month_not_the_competition_window(monkeypatch):
    """The base config's start_date is the April Competition window. Gating on
    it would hide every entry whose live_from is after April."""
    _use_config(monkeypatch, _config_with("deepseek_v4_pro", live_from="2026-05-01"))
    payload = live.get_live_leaderboard(as_of=_AUG27)
    assert "deepseek_v4_pro" in {e["entry_id"] for e in payload["entries"]}
    assert "deepseek_v4_pro" in payload["live_status"]["roster"]


def test_a_not_yet_live_entry_is_neither_rendered_nor_deployed(monkeypatch, tmp_path):
    _use_config(monkeypatch, _config_with("deepseek_v4_pro", live_from="2026-09-01"))
    # A row left over from an earlier run must not put it back on the board.
    _insert_live_run(
        "lb_deepseek_v4_pro_20260801_20260826", "deepseek_v4_pro", "2026-08-26",
        [_pt("2026-08-26T19:00:00+00:00", SEED)],
    )
    payload = live.get_live_leaderboard(as_of=_AUG27)
    assert "deepseek_v4_pro" not in {e["entry_id"] for e in payload["entries"]}
    assert "deepseek_v4_pro" not in payload["live_status"]["roster"]

    deployed = []
    monkeypatch.setattr(live, "_LIVE_REFRESH_STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(
        live,
        "deploy_live_model_increment",
        lambda entry, freeze_cfg, **kwargs: deployed.append(entry["id"])
        or {"entry_id": entry["id"]},
    )
    live.refresh_live_leaderboard(deploy_models=True, as_of=_AUG27)
    assert "deepseek_v4_pro" not in deployed


@pytest.mark.parametrize(
    "value",
    ["2026-09", "2026-09-15", "2026-13-01", "Sept 2026", "", 20260901, ["2026-09-01"]],
)
def test_a_malformed_live_from_warns_once_and_leaves_the_entry_live(
    monkeypatch, capsys, value
):
    _use_config(monkeypatch, _config_with("deepseek_v4_pro", live_from=value))
    assert "deepseek_v4_pro" in _live_ids(_AUG27)
    assert "deepseek_v4_pro" in _live_ids(_AUG27)
    out = capsys.readouterr().out
    line = "WARNING: live.config_invalid entry=deepseek_v4_pro key=live_from"
    assert out.count(line) == 1
    assert "expected=YYYY-MM-01" in out


def test_a_null_live_from_is_absent_and_silent(monkeypatch, capsys):
    _use_config(monkeypatch, _config_with("deepseek_v4_pro", live_from=None))
    assert "deepseek_v4_pro" in _live_ids(_AUG27)
    assert "WARNING" not in capsys.readouterr().out


def test_live_llm_entries_follow_live_model_ids_order(monkeypatch):
    """Deploy order is roster order, not file order (the file lists Haiku first)."""
    roster = ("qwen3_7_plus", "deepseek_v4_pro", "claude_haiku_4_5")
    monkeypatch.setattr(live, "LIVE_MODEL_IDS", roster)
    entries = live.live_llm_entries(
        lb_service.load_leaderboard_config(), month_start=_EVERY_JOIN_OPEN
    )
    assert [e["id"] for e in entries] == list(roster)


def test_the_deploy_cli_says_why_it_refuses_a_not_yet_live_entry(monkeypatch, capsys):
    from argparse import Namespace

    from dashboard.backend.tests.test_leaderboard_samples import _load_deploy_script

    _use_config(monkeypatch, _config_with("deepseek_v4_pro", live_from="2026-09-01"))
    freeze_cfg = live.live_freeze_config(_AUG27)
    monkeypatch.setattr(live, "live_freeze_config", lambda *args, **kwargs: freeze_cfg)
    monkeypatch.setattr(
        live,
        "deploy_live_model_increment",
        lambda *args, **kwargs: pytest.fail("the CLI deployed a not-yet-live entry"),
    )
    script = _load_deploy_script(monkeypatch)
    args = Namespace(
        entry="deepseek_v4_pro", start=None, end=None, force=False, allow_fallback=False
    )
    assert script._deploy_live(args) is None
    assert "its live_from month has not opened" in capsys.readouterr().out
