"""The Live Leaderboard's budget-halt banner line and "Last nightly run" status.

Runs the whole of js/live-leaderboard.js under node against a fake DOM: the
file is top-level declarations plus two ``window.`` assignments, so it loads
without a browser. The ledger adds only ``live_status.budget_halted`` and
per-entry ``last_attempt = {on, outcome, kind}`` to the payload; neither
carries a dollar figure or error text.
"""

import json
import re
import shutil
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import run_node

_LIVE_JS = (
    Path(__file__).resolve().parents[2] / "frontend" / "js" / "live-leaderboard.js"
).read_text(encoding="utf-8")

_needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

# escapeHtml lives in app.js; this stub escapes the same characters.
_DOM_PRELUDE = """
globalThis.window = {};
const elements = {};
globalThis.document = {
  getElementById(id) {
    if (!elements[id]) {
      elements[id] = { textContent: '', innerHTML: '', hidden: true, classList: { toggle() {} } };
    }
    return elements[id];
  },
  querySelector() { return null; },
  querySelectorAll() { return []; },
  addEventListener() {},
};
globalThis.escapeHtml = (value) => String(value)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
"""

_HALT_LINE = (
    "Model runs are paused: the monthly model budget is used up. "
    "Baselines keep updating."
)

# A board with prints, current, and models cached: the existing banner is hidden.
_HEALTHY = {
    "has_prints": True,
    "snapshot_stale": False,
    "models_cached": 1,
    "frozen_through": "2026-08-27",
}


def _run_live_js(tail: str):
    return run_node(_DOM_PRELUDE + _LIVE_JS + "\n" + tail)


def _attempt(outcome, kind=None, on="2026-08-27"):
    return {
        "entry_id": "deepseek_v4_pro",
        "is_model": True,
        "last_attempt": {"on": on, "outcome": outcome, "kind": kind},
    }


def _banner_after(status):
    payload = {"window": {"label": "2026-08-01 — 2026-08-31"}, "entries": [], "live_status": status}
    return _run_live_js(
        f"updateLiveHeader({json.dumps(payload)});"
        "const b = document.getElementById('livePhaseBanner');"
        "process.stdout.write(JSON.stringify({hidden: b.hidden, text: b.textContent}));"
    )


@_needs_node
def test_last_nightly_run_copy_names_the_day_outcome_and_reason():
    entries = [
        _attempt("ok"),
        _attempt("failed", "market_data_gap"),
        _attempt("budget_halted", "budget"),
        _attempt("failed", "timeout"),
        _attempt("failed", "not-a-kind"),
        {"entry_id": "spy_index", "last_attempt": None},
        {"entry_id": "old_payload"},
    ]
    out = _run_live_js(
        f"const entries = {json.dumps(entries)};"
        "process.stdout.write(JSON.stringify({"
        "copy: entries.map(liveLastAttemptCopy), badge: entries.map(liveLastAttemptBadge)}));"
    )
    assert out["copy"] == [
        "Aug 27 · completed",
        "Aug 27 · failed: market data was missing for a session",
        "Aug 27 · paused: monthly model budget used up",
        "Aug 27 · failed: a request timed out",
        "Aug 27 · failed: an internal error",
        "",
        "",
    ]
    assert out["badge"] == [
        "", "Last run failed", "Paused", "Last run failed", "Last run failed", "", "",
    ]


@_needs_node
def test_a_budget_halt_shows_a_banner_line_on_a_healthy_board():
    assert _banner_after({**_HEALTHY, "budget_halted": True}) == {
        "hidden": False,
        "text": _HALT_LINE,
    }


@_needs_node
def test_the_budget_line_joins_an_existing_banner_message():
    banner = _banner_after({
        **_HEALTHY,
        "snapshot_stale": True,
        "snapshot_end": "2026-08-26",
        "budget_halted": True,
    })
    assert banner["hidden"] is False
    assert banner["text"].endswith(" " + _HALT_LINE)
    # The stale-snapshot message is kept, not replaced.
    assert banner["text"][: -len(_HALT_LINE)].strip()


@pytest.mark.parametrize("flag", [False, None, "true", 1])
@_needs_node
def test_no_budget_line_without_a_real_true(flag):
    status = dict(_HEALTHY)
    if flag is not None:
        status["budget_halted"] = flag
    assert _banner_after(status) == {"hidden": True, "text": ""}


@_needs_node
def test_the_table_and_detail_show_a_failed_night():
    entry = {
        **_attempt("failed", "network"),
        "model": "DeepSeek V4 Pro",
        "team_badge": "Model",
        "status": "frozen",
        "portfolio_value": 100500,
        "cumulative_return": 0.005,
        "snapshot_end": "2026-08-26",
    }
    out = _run_live_js(
        f"livePayload = {{entries: [{json.dumps(entry)}]}};"
        "populateLiveTable();"
        f"renderLiveDetail({json.dumps(entry)});"
        "process.stdout.write(JSON.stringify({"
        "row: document.getElementById('liveLeaderboardTableBody').innerHTML,"
        "detail: document.getElementById('liveSelectedDetail').innerHTML}));"
    )
    assert '<span class="team-badge">Last run failed</span>' in out["row"]
    assert re.search(
        r'<span class="team-detail-label">Last nightly run</span>'
        r'<span class="team-detail-value">Aug 27 · failed: a connection failed</span>',
        out["detail"],
    )


@_needs_node
def test_a_model_with_no_attempt_this_month_shows_no_run_row_or_badge():
    entry = {
        "entry_id": "deepseek_v4_pro",
        "model": "DeepSeek V4 Pro",
        "is_model": True,
        "status": "pending",
        "last_attempt": None,
    }
    out = _run_live_js(
        f"livePayload = {{entries: [{json.dumps(entry)}]}};"
        "populateLiveTable();"
        f"renderLiveDetail({json.dumps(entry)});"
        "process.stdout.write(JSON.stringify({"
        "row: document.getElementById('liveLeaderboardTableBody').innerHTML,"
        "detail: document.getElementById('liveSelectedDetail').innerHTML}));"
    )
    assert "Last nightly run" not in out["detail"]
    assert "Last run failed" not in out["row"] and "Paused" not in out["row"]


def test_the_board_never_renders_spend_figures():
    for key in ("est_cost_usd", "spent_usd", "cap_usd", "remaining_usd"):
        assert key not in _LIVE_JS
