"""Phase 3C1 — custom-algorithm service move + characterization.

Verifies the algo service moved to the canonical
``dashboard.backend.domain.backtesting.algo_service`` package while the old root
module remains a re-export shim, that project-relative paths still resolve to the
same ``dashboard/`` directory after the move, and that chat / orchestration /
config-forwarding / error-propagation behavior is unchanged. Real subprocess
backtests are not launched (they require live Alpaca + Anthropic credentials).
"""

import ast
import json
from pathlib import Path

import pytest

from dashboard.backend.domain.backtesting import algo_service as svc

_PUBLIC = [
    "process_chat",
    "get_default_blocks",
    "get_algo_status",
    "execute_algo",
    "get_all_submissions",
    "get_real_submissions",
    "get_submissions_for_session",
]


# ---------------------------------------------------------------------------
# Canonical import
# ---------------------------------------------------------------------------

def test_canonical_module_imports():
    assert svc.process_chat.__module__ == (
        "dashboard.backend.domain.backtesting.algo_service"
    )
    for name in _PUBLIC:
        assert callable(getattr(svc, name)), name


# ---------------------------------------------------------------------------
# Paths still anchor to the dashboard/ project directory after the move
# ---------------------------------------------------------------------------

def test_project_dir_resolves_to_dashboard():
    assert svc._PROJECT_DIR.name == "dashboard"
    assert (svc._PROJECT_DIR / "backend").is_dir()
    assert svc.DATA_DIR == svc._PROJECT_DIR / "data"
    assert svc.CONFIG_DIR == svc._PROJECT_DIR / "data" / "algo_configs"
    assert svc.DEFAULTS_FILE == svc._PROJECT_DIR / "config" / "defaults.json"


# ---------------------------------------------------------------------------
# Import boundary: no API routers, no scripts imports
# ---------------------------------------------------------------------------

def test_service_does_not_import_api_or_scripts():
    tree = ast.parse(Path(svc.__file__).read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    for m in mods:
        assert not m.startswith("dashboard.backend.api"), m
        assert not m.startswith("dashboard.scripts"), m
        assert m != "fastapi" and not m.startswith("fastapi."), m


# ---------------------------------------------------------------------------
# Default blocks / helpers
# ---------------------------------------------------------------------------

def test_get_default_blocks_is_isolated_copy():
    blocks = svc.get_default_blocks()
    assert blocks == svc.DEFAULT_BLOCKS
    blocks["info_retrieval"] = "mutated"
    assert svc.DEFAULT_BLOCKS["info_retrieval"] != "mutated"


def test_default_team_name_resolution():
    assert svc._default_team_name({"info_retrieval": "watch elon musk"}) == "Elon Musk Twitter Algo"
    assert svc._default_team_name({"info_retrieval": "follow trump tweets"}) == "Trump Twitter Algo"
    assert svc._default_team_name({"info_retrieval": "generic strategy"}) == "My Trading Algo"


def test_blocks_match_defaults():
    assert svc._blocks_match_defaults(svc.get_default_blocks()) is True
    edited = svc.get_default_blocks()
    edited["trading_algorithm"] = "totally different strategy"
    assert svc._blocks_match_defaults(edited) is False


def test_get_algo_status_unknown_session():
    out = svc.get_algo_status("no-such-session")
    assert out == {"running": False, "message": "No active job for this session"}


# ---------------------------------------------------------------------------
# Chat (rule-based fallback path, no live LLM)
# ---------------------------------------------------------------------------

def test_process_chat_fallback_without_client(monkeypatch):
    monkeypatch.setattr(svc, "_get_anthropic_client", lambda: None)
    out = svc.process_chat("Switch from Trump to Elon Musk")
    assert set(out.keys()) == {"reply", "blocks", "updated_blocks"}
    assert "musk" in out["blocks"]["info_retrieval"].lower()
    assert "info_retrieval" in out["updated_blocks"]


# ---------------------------------------------------------------------------
# Orchestration: guard rails + config forwarding (no real subprocess)
# ---------------------------------------------------------------------------

def test_execute_algo_rejects_default_blocks(monkeypatch):
    monkeypatch.setattr(svc, "algo_status", {"running": False, "session_id": None})
    with pytest.raises(RuntimeError, match="Edit the four strategy blocks"):
        svc.execute_algo(svc.get_default_blocks(), session_id="sess-1")


def test_execute_algo_requires_anthropic_key(monkeypatch):
    monkeypatch.setattr(svc, "algo_status", {"running": False, "session_id": None})
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    edited = svc.get_default_blocks()
    edited["trading_algorithm"] = "custom plan"
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        svc.execute_algo(edited, session_id="sess-1")


def test_execute_algo_writes_config_and_forwards(monkeypatch, tmp_path):
    captured = {}

    def _fake_thread(*args, **kwargs):
        captured["args"] = kwargs.get("args")

        class _T:
            def start(self):
                captured["started"] = True

        return _T()

    monkeypatch.setattr(svc, "algo_status", {"running": False, "session_id": None})
    monkeypatch.setattr(svc, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(svc, "CONFIG_DIR", tmp_path / "data" / "algo_configs")
    monkeypatch.setattr(svc, "_has_alpaca_credentials", lambda: True)
    monkeypatch.setattr(svc.threading, "Thread", _fake_thread)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    edited = svc.get_default_blocks()
    edited["trading_algorithm"] = "custom plan"

    out = svc.execute_algo(
        edited,
        session_id="sess-1",
        team_name="My Team",
        start_date="2026-05-04",
        end_date="2026-05-12",
    )

    assert out["status"] == "running"
    assert out["team_name"] == "My Team"
    assert out["start_date"] == "2026-05-04"
    assert out["end_date"] == "2026-05-12"
    assert "multi-step runs can take several minutes" in out["message"]
    assert "3–10 minutes" not in out["message"]
    assert captured.get("started") is True

    # Config file is written into the isolated CONFIG_DIR with forwarded values.
    config_files = list((tmp_path / "data" / "algo_configs").glob("*.json"))
    assert len(config_files) == 1
    cfg = json.loads(config_files[0].read_text(encoding="utf-8"))
    assert cfg["team_name"] == "My Team"
    assert cfg["start_date"] == "2026-05-04"
    assert cfg["end_date"] == "2026-05-12"
    assert cfg["blocks"]["trading_algorithm"] == "custom plan"

    # The background worker receives the same config path + identifiers.
    fwd = captured["args"]
    assert fwd[1] == "sess-1"  # session_id
    assert fwd[2] == "My Team"  # team_name
    assert Path(fwd[0]).parent == tmp_path / "data" / "algo_configs"


# ---------------------------------------------------------------------------
# Default date range: a settings file that yields no range must not fail
# silently
# ---------------------------------------------------------------------------
#
# A bare `except: pass` here made "the file is absent" and "the file is on disk
# but unusable" byte-identical: both fall through to the hardcoded pair,
# silently running every backtest over the wrong window with no error, no
# metric and a green suite -- the fail-closed-but-not-fail-visible shape the
# news adapter already taught this repo.
#
# There are three ways to reach that fallback with the file present, and each
# must announce itself: unparseable bytes, valid JSON of the wrong *shape*, and
# valid JSON of the right shape carrying no usable range. The last two are the
# likelier failures in practice (a key rename outlives a corrupt write), and a
# fix that covers only the first leaves the hole where it actually opens.
#
# Asserted on stdout, not caplog: `logger.info()` emits nothing under the
# deployed uvicorn, so print() is the convention here.

FALLBACK_DATES = ("2026-05-04", "2026-05-12")


def _resolve_dates(tmp_path, monkeypatch, capsys, text):
    """Point the service at a defaults.json holding ``text``; return result + stdout."""
    path = tmp_path / "defaults.json"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(svc, "DEFAULTS_FILE", path)
    result = svc._default_backtest_dates()
    return result, capsys.readouterr().out, path


def test_unparseable_defaults_file_is_reported(tmp_path, monkeypatch, capsys):
    dates, out, path = _resolve_dates(tmp_path, monkeypatch, capsys, "{not json")

    assert dates == FALLBACK_DATES
    assert str(path) in out  # the operator has to learn *which* file


def test_defaults_file_without_a_usable_range_is_reported(tmp_path, monkeypatch, capsys):
    """Parses fine, right shape, but the keys were renamed upstream."""
    dates, out, path = _resolve_dates(
        tmp_path,
        monkeypatch,
        capsys,
        json.dumps({"defaultSettings": {"from": "2026-01-02", "to": "2026-01-09"}}),
    )

    assert dates == FALLBACK_DATES
    assert str(path) in out


def test_defaults_file_that_is_not_an_object_is_reported(tmp_path, monkeypatch, capsys):
    """Valid JSON, wrong shape: `.get` on a list raises, which is neither a
    JSONDecodeError nor an OSError, so it escaped the handler entirely."""
    dates, out, path = _resolve_dates(tmp_path, monkeypatch, capsys, "[]")

    assert dates == FALLBACK_DATES
    assert str(path) in out


def test_the_failure_modes_are_distinguishable(tmp_path, monkeypatch, capsys):
    """The whole point of the notice: two different faults must not read the
    same, or the message is back to hiding which one happened."""
    _, corrupt, _ = _resolve_dates(tmp_path, monkeypatch, capsys, "{not json")
    _, no_range, _ = _resolve_dates(
        tmp_path, monkeypatch, capsys, json.dumps({"defaultSettings": {}})
    )

    assert corrupt and no_range
    assert corrupt != no_range


def test_a_usable_defaults_file_says_nothing(tmp_path, monkeypatch, capsys):
    """The notice must mark the anomaly, not narrate the happy path."""
    dates, out, _ = _resolve_dates(
        tmp_path,
        monkeypatch,
        capsys,
        json.dumps({"defaultSettings": {"startDate": "2026-01-02", "endDate": "2026-01-09"}}),
    )

    assert dates == ("2026-01-02", "2026-01-09")
    assert out == ""
