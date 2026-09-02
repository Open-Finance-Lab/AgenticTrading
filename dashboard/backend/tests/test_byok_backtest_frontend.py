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
    _assert_contains(APP_HTML, 'id="runBacktestProviderControl"')
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


def test_atl_credits_hides_provider_and_omits_provider_payload():
    _assert_contains(APP_JS, "function syncRunBacktestProviderVisibility()")
    visibility = _function_body("syncRunBacktestProviderVisibility")
    _assert_contains(visibility, "runBacktestBillingMode !== 'byok'")
    body = _function_body("runBacktest")
    _assert_contains(body, "selectedBillingMode === 'platform_credits'")
    _assert_contains(body, "if (selectedBillingMode === 'byok')")
    assert body.index("payload.billing_mode = selectedBillingMode") < body.index(
        "payload.provider_id = selectedProviderId"
    )
    _assert_contains(
        APP_JS,
        "ATL Credits automatically use OpenRouter first, then CommonStack if needed.",
    )


def test_atl_model_options_are_merged_without_duplicate_ids():
    body = _function_body("syncRunBacktestModelOptions")
    _assert_contains(body, "availableRunBacktestProviders('platform_credits')")
    _assert_contains(body, "const seenModels = new Set()")
    _assert_contains(body, "seenModels.has(normalizedId)")


def test_completed_run_config_prefers_backend_execution_evidence():
    body = _function_body("renderBacktestRunConfig")
    _assert_contains(body, "run?.llm_execution")
    _assert_contains(body, "const completedExecution = !running ? llmExecution : null")
    _assert_contains(body, "completedExecution?.model_id")
    _assert_contains(body, "completedExecution?.billing_mode")
    _assert_contains(body, "completedExecution?.provider_id")
    _assert_contains(body, "Usage unavailable")


def test_legacy_saved_model_falls_back_to_a_one_run_execution_lane():
    body = _function_body("loadRunBacktestExecutionOptions")
    pending_at = body.index("if (pending)")
    exact_at = body.index("findRunBacktestExecutionModel(option, agent?.model_name)")
    fallback_at = body.index("const fallbackModel = provider?.models?.[0] || null")
    unavailable_at = body.rindex("setRunBacktestExecutionUnavailable(")

    assert pending_at < exact_at < fallback_at < unavailable_at
    _assert_contains(body, "for (const billingMode of ['byok', 'platform_credits'])")
    _assert_contains(body, "Saved model is unavailable; this run will use")
    _assert_contains(body, "providerId: provider.provider_id")
    _assert_contains(body, "modelId: fallbackModel.model_id")


def test_confirmed_empty_execution_inventory_offers_api_key_recovery():
    _assert_contains(APP_HTML, 'id="runBacktestApiKeysBtn"')
    _assert_contains(APP_HTML, '>Go to API Keys</button>')
    body = _function_body("loadRunBacktestExecutionOptions")
    _assert_contains(body, "showApiKeysRecovery: true")
    assert body.index("catch (_error)") < body.index("showApiKeysRecovery: true")


def test_execution_options_request_failure_does_not_diagnose_a_missing_key():
    body = _function_body("loadRunBacktestExecutionOptions")
    catch_start = body.index("catch (_error)")
    catch_end = body.index("if (pending)", catch_start)
    catch_body = body[catch_start:catch_end]
    _assert_contains(catch_body, "Backtest execution options could not be loaded.")
    assert "showApiKeysRecovery: true" not in catch_body


def test_api_key_recovery_clears_pending_state_and_navigates_to_credits():
    body = _function_body("goToApiKeys")
    _assert_contains(body, "clearPendingByokBacktest()")
    _assert_contains(body, "closeRunBacktestModal()")
    _assert_contains(body, "navigateToPage('credits')")
    _assert_contains(body, "window.CreditsPage?.openApiKeys({ focus: true })")
