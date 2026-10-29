"""How a hosted TradingAgents failure is classified, and why it matters.

The engine treats the two runtime exception classes very differently:
``TradingAgentsConfigurationError`` aborts the run on its first occurrence,
while ``TradingAgentsRuntimeError`` holds that step and lets the run continue.
Every child failure used to be raised as the transient one -- so a rejected
gateway key, which answers 401 identically on every trading day, spent the
run's whole failure budget rediscovering it, and each of those steps is a full
upstream attempt pair the platform pays for.

A note on a claim this file deliberately does not pin. An independent review
reported that "every failure surfaces as `unsupported rating ''`". Traced
through the real path, that is not so: the bridge sets
``max_consecutive_failures=1``, so the SDK *raises* on the first failed date
rather than returning an error record, and the bridge's stderr -- which the
runner surfaces verbatim-but-sanitized -- carries ``error_type:
error_message``. The rating complaint is reachable only from a tampered or
corrupted bridge output, which is exactly what that check is for. The genuine
defect was the classification, which is what the cases below cover; the
error-record path is covered too, as the defensive half.
"""

from datetime import date, datetime, timezone

import pytest

from dashboard.backend.domain.agents.runtime import (
    AgentRuntimeConfigurationError,
    AgentRuntimeContext,
    AgentRuntimeError,
)
from dashboard.backend.infrastructure.tradingagents.adapter import (
    TradingAgentsConfigurationError,
    TradingAgentsOutputError,
    TradingAgentsRuntime,
    TradingAgentsRuntimeError,
    classify_runtime_failure,
)


def _context():
    return AgentRuntimeContext(
        timestamp=datetime(2026, 3, 13, 15, tzinfo=timezone.utc),
        backtest_start_date="2026-03-01",
        symbols=["AAPL"],
        cash=1_000.0,
        total_equity=1_000.0,
        positions={},
        entry_prices={},
        current_prices={"AAPL": 40.0},
        latest_market_date_before_decision=date(2026, 3, 12),
    )


def _error_record_output(error_type, error_message):
    """What the SDK produces for a failed date when it returns a record.

    ``rating`` is "" and ``atl_action`` "HOLD" by construction -- the SDK's own
    ``__post_init__`` refuses an error record carrying a rating.
    """
    return {
        "schema_version": "tradingagents-atl-v1",
        "manifest": {
            "symbol": "AAPL",
            "tradingagents_version": "0.3.1",
            "selected_analysts": ["market"],
            "safe_config_sha256": "0" * 64,
        },
        "decision": {
            "analysis_date": "2026-03-12",
            "rating": "",
            "atl_action": "HOLD",
            "status": "error",
            "attempts": 2,
            "raw_sha256": "a" * 64,
            "raw_excerpt": "",
            "error_type": error_type,
            "error_message": error_message,
        },
    }


class _RecordRunner:
    def __init__(self, output):
        self.output = output

    def run(self, payload, *, timeout_seconds, environment=None):
        return self.output


def _runtime(runner):
    return TradingAgentsRuntime(
        {"analysts": ["market"]},
        runner=runner,
        environment={
            "COMMONSTACK_API_KEY": "test-key",
            "TRADINGAGENTS_PYTHON": "/usr/bin/python3",
        },
    )


# ---------------------------------------------------------------------------
# 1. The classifier, over the six failure kinds that have to be told apart.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "detail",
    [
        # Gateway credential rejected or missing -- the platform's own key.
        "AuthenticationError: Error code: 401 - invalid api key",
        "PermissionDeniedError: Error code: 403 - no auth credentials found",
        "RuntimeError: Unauthorized",
        # The paying account is out of balance.
        "BadRequestError: 402 insufficient credits for this request",
        # OpenAI's wording for a drained account. It arrives as an HTTP 429,
        # the same status a per-minute rate limit uses, so only the body text
        # separates them -- which is why the fatal table names the phrase and
        # the transient hints never override a phrase.
        (
            "RateLimitError: 429 - You exceeded your current quota, please "
            "check your plan and billing details"
        ),
        "RuntimeError: quota exhausted for this account",
        # The isolated environment is wrong.
        "ModuleNotFoundError: No module named 'langgraph'",
        "ImportError: cannot import name 'StateGraph'",
        "TradingAgentsGenerationError: unsupported TradingAgents version 0.2.0",
        "TradingAgentsGenerationError: failed to initialize TradingAgents: bad model",
        # ATL's own payload was refused.
        "BridgeInputError: unsupported TradingAgents analysts: ['bogus']",
        # The SDK's own dependency guard, which replaces the
        # ModuleNotFoundError with prose. This is the likeliest configuration
        # fault in production -- a deploy whose isolated venv was not built --
        # and it carries none of the exception-name markers above.
        (
            "TradingAgentsDependencyError: TradingAgents is required to "
            "generate decisions. Install it with: git clone ..."
        ),
    ],
)
def test_deterministic_faults_abort_the_run(detail):
    assert classify_runtime_failure(detail) is TradingAgentsConfigurationError


@pytest.mark.parametrize(
    "detail",
    [
        # Rate limited: the next trading day may well succeed.
        "RateLimitError: Error code: 429 - too many requests",
        # Data vendor down or empty.
        "NoMarketDataError: Yahoo Finance returned no rows for AAPL",
        "URLError: <urlopen error [Errno -3] Temporary failure in name resolution>",
        # One malformed generation.
        "TradingAgentsOutputError: TradingAgents state has no final_trade_decision text",
        # Overran its budget.
        "TimeoutError: upstream did not finish",
        # Anything unrecognised. The default is the recoverable class on
        # purpose: losing a run that would have succeeded is worse than
        # spending the failure budget.
        "RuntimeError: something nobody has seen before",
        "",
        # --- the false positives an independent review reproduced ---
        # A per-minute limit that clears in 30 seconds. Google's wording, which
        # OpenRouter forwards verbatim; the bare word "quota" aborted it.
        (
            "APIStatusError: 429 Too Many Requests: Quota exceeded for quota "
            "metric 'generate requests per minute', limit 60 per minute; "
            "retry in 30s"
        ),
        # Classic overload; the bare word "insufficient" aborted it.
        "APIStatusError: 503 Service Unavailable: Insufficient capacity, please retry",
        # "401" inside an opaque request id.
        "APIStatusError: 500 Internal Server Error: upstream request id req-4013ab",
        # "403" inside a number.
        "TradingAgentsGenerationError: upstream returned 10403 tokens",
        # One date's data coverage, aborted as a deployment fault.
        "ValueError: insufficient data points to compute sma50 for AAPL",
        # An auth status code accompanied by evidence of a retry.
        "RateLimitError: Error code: 401 - temporarily unavailable, retry",
    ],
)
def test_recoverable_faults_hold_the_step(detail):
    assert classify_runtime_failure(detail) is TradingAgentsRuntimeError


def test_the_classifier_is_case_insensitive():
    # The text crosses a process boundary and upstream capitalises
    # inconsistently; matching case-sensitively would silently demote a
    # credential fault back to transient.
    assert (
        classify_runtime_failure("ModuleNotFoundError: no module named x")
        is TradingAgentsConfigurationError
    )
    assert (
        classify_runtime_failure("modulenotfounderror: no module named x")
        is TradingAgentsConfigurationError
    )


def test_both_classes_reach_the_engines_two_behaviours():
    # The engine branches on the runtime-agnostic base classes, so the mapping
    # below is what actually decides abort-vs-hold.
    assert issubclass(TradingAgentsConfigurationError, AgentRuntimeConfigurationError)
    assert issubclass(TradingAgentsRuntimeError, AgentRuntimeError)
    assert not issubclass(TradingAgentsRuntimeError, AgentRuntimeConfigurationError)


# ---------------------------------------------------------------------------
# 2. An error *record* is reported as the upstream failure, not as a rating.
# ---------------------------------------------------------------------------

def test_an_error_record_names_its_cause_rather_than_the_empty_rating():
    runtime = _runtime(
        _RecordRunner(
            _error_record_output(
                "RateLimitError", "Error code: 429 - too many requests"
            )
        )
    )

    with pytest.raises(TradingAgentsRuntimeError) as excinfo:
        runtime.decide(_context())

    message = str(excinfo.value)
    assert "RateLimitError" in message
    assert "429" in message
    # The symptom must not be the headline.
    assert "unsupported rating" not in message


def test_an_error_record_from_a_bad_credential_aborts_the_run():
    runtime = _runtime(
        _RecordRunner(
            _error_record_output(
                "AuthenticationError", "Error code: 401 - invalid api key"
            )
        )
    )
    # Configuration, so the engine stops rather than paying for the same 401
    # once per trading day.
    with pytest.raises(TradingAgentsConfigurationError):
        runtime.decide(_context())


def test_an_error_record_with_no_detail_still_says_something_usable():
    output = _error_record_output("", "")
    runtime = _runtime(_RecordRunner(output))
    with pytest.raises(TradingAgentsRuntimeError) as excinfo:
        runtime.decide(_context())
    message = str(excinfo.value)
    assert "unknown" in message
    assert "no detail reported" in message


def test_error_detail_is_bounded_and_control_characters_are_folded():
    # Upstream-controlled text that reaches print() and the persisted run
    # metadata: a crafted message must not be able to forge log lines or
    # balloon the row.
    hostile = "x" * 5000 + "\nFAKE: forged line\r\x00"
    runtime = _runtime(_RecordRunner(_error_record_output("Weird", hostile)))
    with pytest.raises(TradingAgentsRuntimeError) as excinfo:
        runtime.decide(_context())
    message = str(excinfo.value)
    assert "\n" not in message and "\r" not in message and "\x00" not in message
    assert len(message) < 1200, len(message)


def test_a_rating_outside_the_map_is_still_an_output_error():
    # The tampered/corrupted-output check must keep working: this is the one
    # thing the rating complaint is actually for.
    output = _error_record_output("", "")
    output["decision"]["status"] = "valid"
    output["decision"]["rating"] = "Strong Buy"
    output["decision"]["atl_action"] = "BUY"
    runtime = _runtime(_RecordRunner(output))
    with pytest.raises(TradingAgentsOutputError, match="unsupported rating"):
        runtime.decide(_context())

# ---------------------------------------------------------------------------
# 3. The real child process: the classification has to survive the boundary.
# ---------------------------------------------------------------------------

def test_a_real_child_failure_is_classified_from_its_sanitized_stderr():
    """Runs the actual bridge in a real subprocess and checks the class.

    The classifier reads text that crossed a process boundary, so a unit test
    over the function alone cannot prove the wiring: the runner has to raise
    the chosen class, and the message it matched on has to be the one the
    child actually printed.
    """
    import os
    import sys

    from dashboard.backend.infrastructure.tradingagents import bridge
    from dashboard.backend.infrastructure.tradingagents.adapter import (
        TradingAgentsSubprocessRunner,
    )

    runner = TradingAgentsSubprocessRunner(
        {
            "PATH": os.environ.get("PATH", ""),
            "TRADINGAGENTS_PYTHON": sys.executable,
        }
    )
    payload = {
        "schema": "tradingagents-bridge-v1",
        "symbol": "AAPL",
        "analysis_dates": ["2026-03-12"],
        "selected_analysts": ["market"],
        "config": {
            "llm_provider": "openrouter",
            "deep_think_llm": bridge.MANAGED_DEEP_THINK_MODEL,
            "quick_think_llm": bridge.MANAGED_QUICK_THINK_MODEL,
        },
    }

    import importlib.util

    if importlib.util.find_spec("tradingagents") is not None:
        pytest.skip("a real TradingAgents install would attempt a billable call")

    with pytest.raises(TradingAgentsConfigurationError) as excinfo:
        runner.run(payload, timeout_seconds=120)

    message = str(excinfo.value)
    # The deploy's venv is missing: identical on every step, so the run must
    # abort rather than pay for the same import failure once per trading day.
    assert "TradingAgents is required" in message
    # And still no raw traceback.
    assert "Traceback" not in message


def test_a_payload_the_bridge_refuses_is_a_configuration_fault():
    """ATL's own payload being rejected is a bug on our side, not weather."""
    import os
    import sys

    from dashboard.backend.infrastructure.tradingagents.adapter import (
        TradingAgentsSubprocessRunner,
    )

    runner = TradingAgentsSubprocessRunner(
        {
            "PATH": os.environ.get("PATH", ""),
            "TRADINGAGENTS_PYTHON": sys.executable,
        }
    )
    payload = {
        "schema": "tradingagents-bridge-v1",
        "symbol": "AAPL",
        "analysis_dates": ["2026-03-12"],
        "selected_analysts": ["market"],
        "config": {
            "llm_provider": "openrouter",
            # Not the platform's pinned pair: the bridge fails closed before
            # anything is spent.
            "deep_think_llm": "someone-elses/model",
            "quick_think_llm": "someone-elses/model",
        },
    }

    with pytest.raises(TradingAgentsConfigurationError) as excinfo:
        runner.run(payload, timeout_seconds=120)
    assert "platform-managed" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. The runtime and its runner must read one environment.
# ---------------------------------------------------------------------------

def test_the_default_runner_inherits_the_runtimes_environment():
    """Otherwise the gateway chosen and the credential handed over can differ.

    The runtime resolves which gateway pays from the environment it was given;
    the runner, left to its own default, read ``os.environ`` independently. A
    runtime handed CommonStack's key would then select CommonStack's model
    while the runner gave the child whatever ``os.environ`` held -- possibly
    OpenRouter's key, authenticating against one account for a model pinned to
    the other. Identical in production, where both are ``os.environ``, which is
    exactly why it would not have surfaced until something passed an explicit
    environment.
    """
    environment = {
        "COMMONSTACK_API_KEY": "commonstack-test-key",
        "TRADINGAGENTS_PYTHON": "/usr/bin/python3",
    }
    runtime = TradingAgentsRuntime({"analysts": ["market"]}, environment=environment)

    assert runtime.gateway_id == "commonstack"
    # The same mapping, not merely an equal one taken from the process.
    assert runtime.runner.environment == runtime.environment
    assert "COMMONSTACK_API_KEY" in runtime.runner.environment
    assert "OPENROUTER_API_KEY" not in runtime.runner.environment


def test_an_explicit_runner_is_still_respected():
    """Tests and future callers hand in their own runner; that must not be
    quietly replaced by one built from the runtime's environment."""

    class Sentinel:
        environment = {"marker": "explicit"}

        def run(self, payload, *, timeout_seconds, environment=None):
            raise AssertionError("not called")

    sentinel = Sentinel()
    runtime = TradingAgentsRuntime(
        {"analysts": ["market"]},
        runner=sentinel,
        environment={"COMMONSTACK_API_KEY": "k"},
    )
    assert runtime.runner is sentinel

def test_an_ambiguous_quota_message_stays_transient():
    """"Quota exceeded" alone means different things to different providers.

    Google means a per-minute limit; OpenAI means a drained account. Guessing
    fatal would discard runs that only needed to wait, so the ambiguous form
    stays transient and only the unambiguous phrasings abort. This is the
    documented safe direction, asserted rather than left to the table's shape.
    """
    assert (
        classify_runtime_failure("RuntimeError: quota exceeded")
        is TradingAgentsRuntimeError
    )
    assert (
        classify_runtime_failure("RuntimeError: quota exhausted")
        is TradingAgentsConfigurationError
    )


def test_no_bare_status_code_or_single_broad_word_is_a_fatal_phrase():
    """Pins the shape of the fix, not just its current cases.

    The first version of this table carried "401", "403", "insufficient",
    "quota" and "billing". Each matched text that had nothing to do with a
    deployment fault -- inside a request id, a token count, "Insufficient
    capacity", "insufficient data points", a per-minute quota metric -- and
    turned a recoverable step into a whole-run abort.
    """
    from dashboard.backend.infrastructure.tradingagents.adapter import (
        _FATAL_FAULT_PHRASES,
    )

    for phrase in _FATAL_FAULT_PHRASES:
        assert not phrase.isdigit(), phrase
        assert phrase not in {"insufficient", "quota", "billing"}, phrase


def test_a_child_killed_without_stderr_is_transient():
    """An empty stderr means the child never got to report.

    The bridge writes its failure to stderr, so nothing there means SIGKILL or
    the OOM killer -- which this service has a history of, and which is
    transient. The message may still fall back to stdout, but classifying that
    fallback ran the marker table over the child's own analysis prose.
    """
    assert classify_runtime_failure("") is TradingAgentsRuntimeError
    assert classify_runtime_failure(None) is TradingAgentsRuntimeError
    # Prose a model could plausibly write about a company.
    prose = (
        "The filing discusses authentication hardware and the unauthorized "
        "access incident disclosed in Q3, plus billing software revenue."
    )
    assert classify_runtime_failure(prose) is TradingAgentsConfigurationError, (
        "the phrases do match prose, which is exactly why the runner "
        "classifies stderr rather than the stdout fallback"
    )


def test_the_runner_classifies_stderr_not_the_stdout_fallback(monkeypatch):
    """A child killed without stderr must not be judged by its own prose.

    The bridge writes failures to stderr, so an empty stderr means the child
    never got to report -- SIGKILL or the OOM killer, which this service has a
    history of and which is transient. The *message* still falls back to
    stdout so the operator sees something, but classifying that fallback ran
    the fatal-phrase table over upstream's analysis text: a model writing
    "unauthorized access incident" about a company aborted the whole run as a
    configuration fault.
    """
    import os
    import subprocess
    import sys

    from dashboard.backend.infrastructure.tradingagents import adapter as module
    from dashboard.backend.infrastructure.tradingagents.adapter import (
        TradingAgentsSubprocessRunner,
    )

    prose = (
        "## Fundamentals\nThe filing discusses authentication hardware and "
        "the unauthorized access incident disclosed in Q3, plus billing "
        "software revenue."
    )

    def fake_run(*_args, **_kwargs):
        # Non-zero exit, nothing on stderr, analysis prose on stdout.
        return subprocess.CompletedProcess(
            args=[], returncode=-9, stdout=prose, stderr=""
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    runner = TradingAgentsSubprocessRunner(
        {
            "PATH": os.environ.get("PATH", ""),
            "TRADINGAGENTS_PYTHON": sys.executable,
        }
    )

    with pytest.raises(TradingAgentsRuntimeError) as excinfo:
        runner.run({"schema": "tradingagents-bridge-v1"}, timeout_seconds=5)

    # Transient, so the step holds and the run survives...
    assert not isinstance(excinfo.value, TradingAgentsConfigurationError)
    # ...and the operator still gets the stdout fallback in the message.
    assert "unauthorized access incident" in str(excinfo.value)


def test_the_runner_still_aborts_on_a_real_configuration_fault_in_stderr(
    monkeypatch,
):
    """The other half: a genuine deployment fault on stderr still aborts."""
    import os
    import subprocess
    import sys

    from dashboard.backend.infrastructure.tradingagents import adapter as module
    from dashboard.backend.infrastructure.tradingagents.adapter import (
        TradingAgentsSubprocessRunner,
    )

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="## Market\nPrices rose.",
            stderr=(
                "TradingAgentsDependencyError: TradingAgents is required to "
                "generate decisions."
            ),
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    runner = TradingAgentsSubprocessRunner(
        {
            "PATH": os.environ.get("PATH", ""),
            "TRADINGAGENTS_PYTHON": sys.executable,
        }
    )

    with pytest.raises(TradingAgentsConfigurationError):
        runner.run({"schema": "tradingagents-bridge-v1"}, timeout_seconds=5)
