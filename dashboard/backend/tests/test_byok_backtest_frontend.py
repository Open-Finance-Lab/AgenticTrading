"""Static contracts for BYOK execution controls in Run Backtest."""

from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP_HTML = (FRONTEND / "app.html").read_text(encoding="utf-8")
APP_JS = (FRONTEND / "app.js").read_text(encoding="utf-8")


def _assert_contains(source: str, value: str) -> None:
    if value not in source:
        raise AssertionError(f"Missing frontend contract: {value}")


def _function_body(name: str) -> str:
    start = APP_JS.index(f"function {name}(")
    next_function = APP_JS.find("\nfunction ", start + 1)
    return APP_JS[
        start:
        next_function if next_function >= 0 else len(APP_JS)
    ]


def test_run_backtest_modal_has_execution_controls():
    _assert_contains(APP_HTML, 'id="runBacktestBillingGroup"')
    _assert_contains(APP_HTML, 'data-billing-mode="byok"')
    _assert_contains(APP_HTML, 'data-billing-mode="platform_credits"')
    _assert_contains(APP_HTML, 'id="runBacktestProviderSelect"')
    _assert_contains(APP_HTML, 'id="modelSelect"')
    _assert_contains(APP_HTML, "Model for this run")


def test_pending_byok_selection_is_validated_and_consumed():
    _assert_contains(APP_JS, "atlPendingByokBacktest")
    _assert_contains(APP_JS, "sessionStorage.getItem")
    _assert_contains(APP_JS, "sessionStorage.removeItem")
    _assert_contains(APP_JS, "expires_at")


def test_pipeline_llm_payload_sends_explicit_execution_lane():
    body = _function_body("runBacktest")
    _assert_contains(body, "payload.billing_mode")
    _assert_contains(body, "payload.provider_id")
    _assert_contains(body, "payload.model")
    _assert_contains(body, "Choose an AI billing method, provider, and model.")
