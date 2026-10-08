"""The Run Backtest modal's dates come from /config/defaults, never from app.html.

A date baked into the markup went stale the week it was written, and it was
what a cold start submitted while the defaults were still loading. These run
the real app.js functions under node against stub inputs and a stub fetch.
"""

import re
import shutil

import pytest

from dashboard.backend.tests._frontend_source import APP_HTML, fn_body, js_let, run_node


def test_the_modal_date_inputs_ship_without_a_date():
    for input_id in ("startDate", "endDate"):
        tag = re.search(rf'<input[^>]*id="{input_id}"[^>]*>', APP_HTML).group(0)
        assert "value=" not in tag, tag


_HARNESS = "\n".join(
    [
        js_let("defaultsInFlight"),
        fn_body("function loadDefaults("),
        fn_body("function applyDefaultDate("),
        fn_body("async function ensureDefaultBacktestDates("),
        fn_body("async function fetchAndApplyDefaults("),
    ]
)

_PRELUDE = """
const inputs = {
  startDate: { value: '', dataset: {} },
  endDate: { value: '', dataset: {} },
};
globalThis.document = { getElementById: (id) => inputs[id] || null };
globalThis.window = {};
globalThis.console = { log() {}, warn() {} };
const API_BASE = '';
function selectPreset() {}
let fetches = 0;
let answer = { ok: true, status: 200, json: async () => ({
  defaultSettings: { startDate: '2026-09-28', endDate: '2026-10-02', assetList: [] },
}) };
globalThis.fetch = async () => { fetches += 1; return answer; };
"""

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@needs_node
def test_defaults_fill_empty_inputs():
    out = run_node(
        _PRELUDE
        + _HARNESS
        + """
(async () => {
  await loadDefaults();
  process.stdout.write(JSON.stringify([inputs.startDate.value, inputs.endDate.value]));
})();
"""
    )
    assert out == ["2026-09-28", "2026-10-02"]


@needs_node
def test_a_late_response_never_overwrites_a_date_the_user_chose():
    out = run_node(
        _PRELUDE
        + _HARNESS
        + """
(async () => {
  await loadDefaults();
  inputs.startDate.value = '2026-08-03';   // the user edits one date
  answer = { ok: true, status: 200, json: async () => ({
    defaultSettings: { startDate: '2026-10-05', endDate: '2026-10-09', assetList: [] },
  }) };
  await loadDefaults();                    // a later (rolled) answer lands
  process.stdout.write(JSON.stringify([inputs.startDate.value, inputs.endDate.value]));
})();
"""
    )
    # The edited date stays; the untouched one follows the server's new week.
    assert out == ["2026-08-03", "2026-10-09"]


@needs_node
def test_submit_waits_for_the_defaults_and_retries_a_failed_boot_fetch():
    out = run_node(
        _PRELUDE
        + _HARNESS
        + """
(async () => {
  const good = answer;
  answer = { ok: false, status: 503, statusText: 'cold start' };
  await loadDefaults();                    // boot fetch fails: inputs stay empty
  const afterBoot = [inputs.startDate.value, inputs.endDate.value];
  answer = good;
  await ensureDefaultBacktestDates();      // the submit path retries
  await ensureDefaultBacktestDates();      // filled now: no further request
  process.stdout.write(JSON.stringify({
    afterBoot, afterSubmit: [inputs.startDate.value, inputs.endDate.value], fetches,
  }));
})();
"""
    )
    assert out == {
        "afterBoot": ["", ""],
        "afterSubmit": ["2026-09-28", "2026-10-02"],
        "fetches": 2,
    }


@needs_node
def test_concurrent_callers_share_one_request():
    out = run_node(
        _PRELUDE
        + _HARNESS
        + """
(async () => {
  await Promise.all([loadDefaults(), ensureDefaultBacktestDates(), loadDefaults()]);
  process.stdout.write(JSON.stringify(fetches));
})();
"""
    )
    assert out == 1


def test_run_backtest_waits_for_the_dates_before_reading_them():
    body = fn_body("async function runBacktest(")
    assert body.index("await ensureDefaultBacktestDates()") < body.index(
        "startDateInput.value"
    )
