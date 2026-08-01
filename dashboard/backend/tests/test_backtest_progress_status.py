"""Progress-file freshness, so the UI can tell 'working' from 'stuck'.

The status payload already carried step/total_steps; what it could not answer
was whether those numbers were current. A run whose subprocess wedges keeps
reporting its last step forever, which reads identically to steady progress.
"""

import json
import time

from dashboard.backend.api.routers import backtests


def test_progress_carries_the_file_mtime(tmp_path, monkeypatch):
    progress_file = tmp_path / "backtest_progress_test.json"
    progress_file.write_text(json.dumps({"step": 7, "total_steps": 240}), encoding="utf-8")
    monkeypatch.setitem(backtests.backtest_status, "progress_file", str(progress_file))

    payload = backtests._read_backtest_progress()

    assert payload["step"] == 7
    assert payload["total_steps"] == 240
    assert payload["progress_updated_at"] == progress_file.stat().st_mtime
    assert payload["progress_updated_at"] <= time.time() + 1


def test_missing_progress_file_still_returns_none(tmp_path, monkeypatch):
    """Unchanged behaviour: the status payload omits `progress` entirely rather
    than shipping a half-populated object."""
    monkeypatch.setitem(
        backtests.backtest_status, "progress_file", str(tmp_path / "nope.json")
    )
    assert backtests._read_backtest_progress() is None


def test_malformed_progress_file_still_returns_none(tmp_path, monkeypatch):
    progress_file = tmp_path / "broken.json"
    progress_file.write_text("{not json", encoding="utf-8")
    monkeypatch.setitem(backtests.backtest_status, "progress_file", str(progress_file))
    assert backtests._read_backtest_progress() is None


def test_non_dict_progress_file_still_returns_none(tmp_path, monkeypatch):
    progress_file = tmp_path / "list.json"
    progress_file.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setitem(backtests.backtest_status, "progress_file", str(progress_file))
    assert backtests._read_backtest_progress() is None
