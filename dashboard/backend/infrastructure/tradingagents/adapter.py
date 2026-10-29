"""Thin, subprocess-isolated adapter for ``TauricResearch/TradingAgents``.

The upstream package runs in its own virtual environment behind ``bridge.py``.
This module translates ATL portfolio state into one bridge invocation per
trading day, then translates the returned five-tier rating through ATL's
existing strict action schema and executable-action constraints. It never
executes or mutates trades.

**Where this differs from the AI Hedge Fund adapter.** Upstream there returns an
action *with a quantity and a confidence*, so its adapter only transcribes.
TradingAgents returns a rating and nothing else, so position sizing and
confidence are produced here -- the sizing model is the one the SDK's client-side
replay already used (a target weight of equity), and the confidence is the
conviction tier that ``RATING_TO_ACTION`` discards when it folds five ratings
into three directions. Both are platform-owned: an agent that could set its own
position weight would be choosing its own leverage.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from dashboard.backend.domain.agents.runtime import (
    TRADINGAGENTS_RUNTIME_TYPE,
    AgentRuntimeConfigurationError,
    AgentRuntimeContext,
    AgentRuntimeError,
    describe_tradingagents_lookahead,
    normalize_runtime_config,
)
from dashboard.backend.infrastructure.llm.validator import (
    MAX_ORDER_SHARES,
    actions_to_executable,
    parse_actions_payload,
)
from dashboard.backend.infrastructure.tradingagents.bridge import (
    DEFAULT_MANAGED_GATEWAY,
    MANAGED_GATEWAYS,
    MANAGED_MAX_DEBATE_ROUNDS,
    MANAGED_MAX_RISK_DISCUSS_ROUNDS,
)
from dashboard.backend.paths import REPO_ROOT

# A TradingAgents decision is a multi-agent debate, and the generator retries a
# failed date once before giving up -- so the budget covers two full passes.
# AI Hedge Fund's 300s default sizes a single upstream pass and is far too tight
# here; a parent that killed the child mid-debate would bill for the tokens
# already spent and keep nothing.
DEFAULT_TIMEOUT_SECONDS = 600
MIN_TIMEOUT_SECONDS = 1
MAX_TIMEOUT_SECONDS = 900

# Share of total equity one position may reach. Deployment-owned rather than
# agent config: this is leverage, not strategy.
DEFAULT_MAX_POSITION_WEIGHT = 0.2

DEFAULT_DECISION_INTERVAL = "daily"

_LOCAL_RUNTIME_DIR = REPO_ROOT / ".tradingagents-venv"
_SDK_SOURCE_DIR = REPO_ROOT / "packaging" / "agentictrading" / "src"
_MAX_DIAGNOSTIC_TEXT_LENGTH = 500

# Upstream's five-tier rating, and ATL's three directions. Mirrors
# ``RATING_TO_ACTION`` in the SDK's ``_tradingagents_core``, which the bridge
# already applied -- repeated here so a corrupted or tampered bridge output
# whose rating and action disagree is refused rather than executed.
_RATING_TO_ACTION = {
    "buy": "BUY",
    "overweight": "BUY",
    "hold": "HOLD",
    "underweight": "SELL",
    "sell": "SELL",
}

# The conviction the three-way fold throws away. TradingAgents emits no
# confidence of its own, and ``actions_to_executable`` drops anything under
# 0.3, so a flat constant would make "overweight" and "buy" indistinguishable
# to every downstream consumer.
_RATING_CONFIDENCE = {
    "buy": 0.85,
    "sell": 0.85,
    "overweight": 0.55,
    "underweight": 0.55,
    "hold": 0.55,
}

# The pinned upstream process only receives values needed for networking,
# locale/runtime behavior, and the two data/model providers it is allowed to
# reach. ATL database URLs, broker credentials and other model providers' keys
# do not cross the process boundary.
_SUBPROCESS_ENV_KEYS = frozenset(
    {
        "PATH",
        # Windows loads the socket service provider through these; without them
        # the child cannot open a connection at all (WinError 10106). Neither
        # is a secret, and both are absent on the Linux deploy target.
        "SystemRoot",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "TZ",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        # Whichever platform account pays for this deployment's hosted
        # decisions. Both are forwarded because the gateway is chosen at
        # runtime; only the selected one is ever actually used.
        "OPENROUTER_API_KEY",
        "COMMONSTACK_API_KEY",
        # Optional: upstream's macro tools read FRED. Absent, those tools simply
        # do not resolve; it is never required for a run to start.
        "FRED_API_KEY",
        # Where upstream writes its full state log. The child's HOME is this
        # call's TemporaryDirectory, so the default
        # ``$HOME/.tradingagents/logs`` is deleted the moment the call returns
        # -- the trajectory is lost by construction. This is upstream's own
        # override and is deployment-owned: ``results_dir`` stays out of
        # ``_ALLOWED_CONFIG_KEYS``, so a payload still cannot redirect writes.
        "TRADINGAGENTS_RESULTS_DIR",
    }
)


class TradingAgentsRuntimeError(AgentRuntimeError):
    """Base exception for runtime execution failures.

    Deliberately *transient* by default: the backtest engine holds the step and
    keeps the run alive for these, because a timeout or a bad upstream response
    on one day says nothing about the next.
    """


class TradingAgentsConfigurationError(
    TradingAgentsRuntimeError, AgentRuntimeConfigurationError
):
    """Raised when the deployment itself cannot host the runtime.

    Distinct from the transient base so the engine can abort immediately. A
    missing interpreter or a malformed platform setting fails identically on
    every step, and retrying it once per trading day only converts one clear
    error into a long, silent, all-holds run.
    """


class TradingAgentsOutputError(TradingAgentsRuntimeError):
    """Raised when the bridge returns data that cannot safely become an order."""


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _platform_setting(
    environment: Optional[Mapping[str, str]], name: str
) -> str:
    source = os.environ if environment is None else environment
    return str(source.get(name, "") or "").strip()


def _sdk_source_dir(environment: Optional[Mapping[str, str]] = None) -> Path:
    """Return the directory the child imports ``agentictrading`` from.

    ``TRADINGAGENTS_ROOT`` is this runtime's ``AI_HEDGE_FUND_ROOT``: a
    deployment-owned directory placed on the child's PYTHONPATH. Unset, it is
    the repository's own SDK sources, which is what both a checkout and the
    container image provide -- so a deployment that vendors the SDK elsewhere
    can say so without the adapter hard-coding an image layout.
    """
    configured = _platform_setting(environment, "TRADINGAGENTS_ROOT")
    return _resolve_path(configured) if configured else _SDK_SOURCE_DIR


def resolve_step_timeout_seconds(
    environment: Optional[Mapping[str, str]] = None,
) -> int:
    """Return the per-decision subprocess budget.

    Shared with the API layer so the caller that sizes the *outer* backtest
    subprocess timeout and the runtime that sizes each *inner* upstream call
    read the same number. Two independent readings of
    ``TRADINGAGENTS_TIMEOUT_SECONDS`` is how the parent ends up killing the
    child mid-run.
    """
    raw = _platform_setting(environment, "TRADINGAGENTS_TIMEOUT_SECONDS")
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError as exc:
        raise TradingAgentsConfigurationError(
            "TRADINGAGENTS_TIMEOUT_SECONDS must be an integer"
        ) from exc
    if not MIN_TIMEOUT_SECONDS <= value <= MAX_TIMEOUT_SECONDS:
        raise TradingAgentsConfigurationError(
            "TRADINGAGENTS_TIMEOUT_SECONDS must be between "
            f"{MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS}"
        )
    return value


def resolve_max_position_weight(
    environment: Optional[Mapping[str, str]] = None,
) -> float:
    """Return the share of equity one TradingAgents position may reach.

    Fails rather than falling back to the default on a malformed value: an
    operator who set ``2`` meant something, and silently trading a fifth of the
    book instead would be a position-sizing decision made by a typo.
    """
    raw = _platform_setting(environment, "TRADINGAGENTS_MAX_POSITION_WEIGHT")
    if not raw:
        return DEFAULT_MAX_POSITION_WEIGHT
    try:
        value = float(raw)
    except ValueError as exc:
        raise TradingAgentsConfigurationError(
            "TRADINGAGENTS_MAX_POSITION_WEIGHT must be a number"
        ) from exc
    if not math.isfinite(value) or not 0 < value <= 1:
        raise TradingAgentsConfigurationError(
            "TRADINGAGENTS_MAX_POSITION_WEIGHT must be greater than 0 and at most 1"
        )
    return value


def runtime_unavailable_reason(
    environment: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return why the hosted runtime cannot run here, or ``None`` if it can.

    Lets the API reject a hosted backtest at request time with an actionable
    503 instead of accepting it and failing inside a background subprocess
    minutes later. ``render.yaml`` is documentation rather than the deploy
    mechanism for this service, so "the isolated venv was never built" is a
    live production possibility, not a hypothetical.
    """
    try:
        TradingAgentsSubprocessRunner(environment)._python_executable()
    except TradingAgentsRuntimeError as exc:
        return str(exc)
    if not Path(__file__).with_name("bridge.py").is_file():
        return "TradingAgents bridge script is missing from this deployment"
    if not (_sdk_source_dir(environment) / "agentictrading").is_dir():
        return "TradingAgents bridge cannot reach the agentictrading SDK sources"
    return None


def _redact_runtime_error(value: Any, environment: Mapping[str, str]) -> str:
    message = str(value)
    for key, secret in environment.items():
        upper = key.upper()
        if not secret or not (
            upper.endswith("_API_KEY")
            or upper.endswith("_TOKEN")
            or "CREDENTIAL" in upper
        ):
            continue
        message = message.replace(secret, "[REDACTED]")
    # Collapse newlines and control characters. This text is upstream-controlled
    # and reaches both print() -- the only log channel that survives in the
    # deployed config -- and the persisted run metadata, so a crafted traceback
    # could otherwise forge log lines.
    message = "".join(
        " " if character in "\r\n" or ord(character) < 0x20 else character
        for character in message
    )
    return message[-1000:]


# Failures that are identical on every trading day, matched against the
# *already sanitized* child stderr. The distinction is a spend control, not
# cosmetics: the engine holds a step and retries once per trading day for a
# transient failure, and each of those retries is a full upstream attempt pair
# that the platform pays for. A bad gateway key answered 401 on every one of
# them before this table existed.
#
# Matched on substrings rather than exception classes because the text crosses
# a process boundary: the bridge prints `sanitize_error_message(exc)`, so the
# class name survives only when upstream put it there. Both halves are checked.
# Phrases that identify a failure as *deterministic* beyond reasonable doubt,
# matched against the already-sanitized child stderr. The distinction is a
# spend control, not cosmetics: the engine holds a step and retries once per
# trading day for a transient failure, and each retry is a full upstream
# attempt pair the platform pays for. A rejected gateway key answered 401 on
# every one of them before this table existed.
#
# ⚠ Every entry has to be unambiguous, and the first version of this table was
# not. It carried bare "401", "403", "insufficient", "quota" and "billing",
# which an independent review showed abort healthy runs:
#
#   "429 Too Many Requests: Quota exceeded for quota metric ... retry in 30s"
#       -- a per-minute limit that clears in 30 seconds, matched "quota"
#   "503 Service Unavailable: Insufficient capacity, please retry"
#       -- classic overload, matched "insufficient"
#   "500 Internal Server Error: upstream request id req-4013ab"
#       -- matched "401" inside an opaque id
#   "upstream returned 10403 tokens"            -- matched "403" inside a number
#   "insufficient data points to compute sma50" -- one date's data coverage
#
# Each of those threw away a run that would have recovered, which is the exact
# opposite of the safe direction this function's docstring states. So the
# phrases below name the *condition*, never a bare code or a single broad word.
_FATAL_FAULT_PHRASES = (
    # Gateway credential rejected or absent. The platform's own key, so no
    # amount of waiting fixes it.
    "unauthorized",
    "forbidden",
    "authentication",
    "invalid api key",
    "invalid_api_key",
    "incorrect api key",
    "no auth credentials",
    "permission denied",
    # The paying account is drained. A state of the account rather than of the
    # request, so retrying spends what is left discovering that. Spelled out,
    # because "insufficient" alone also matches capacity and data-coverage
    # failures, and "quota" alone also matches per-minute rate limits.
    "insufficient credit",
    "insufficient balance",
    "insufficient funds",
    "insufficient_quota",
    "quota exhausted",
    "exceeded your current quota",
    "billing hard limit",
    # The isolated environment is wrong: a missing dependency, an unexpected
    # upstream version, or a graph that cannot be constructed at all.
    "modulenotfounderror",
    "importerror",
    "unsupported tradingagents version",
    "failed to initialize tradingagents",
    # The SDK's own dependency guard, which replaces the ModuleNotFoundError
    # with prose and so carries none of the markers above. This is the single
    # likeliest configuration fault in production -- a deploy whose isolated
    # venv was not built -- and it was the case that proved a marker table
    # keyed on exception names alone is not enough.
    "tradingagents is required",
    "tradingagentsdependencyerror",
    # The bridge refused ATL's own payload. A bug or a bad config on our side,
    # and byte-identical on the next step.
    "bridgeinputerror",
)

# A bare 401/403 with no accompanying word is still worth treating as fatal --
# some gateways return little else -- but only with word boundaries, so a
# request id or a token count cannot match, and only when nothing in the text
# says the failure is temporary.
_AUTH_STATUS_RE = re.compile(r"\b40[13]\b")

# Evidence that a failure is temporary. Consulted only to veto the bare status
# codes above: it deliberately does NOT override the phrases, because HTTP 429
# carries both a per-minute rate limit and a drained account, and the body text
# is the only thing that tells them apart.
_TRANSIENT_HINTS = (
    "rate limit",
    "ratelimit",
    "too many requests",
    "per minute",
    "per-minute",
    "retry",
    "try again",
    "overloaded",
    "capacity",
    "timed out",
    "timeout",
    "temporarily",
    "service unavailable",
    "502",
    "503",
    "504",
)


def classify_runtime_failure(detail: str) -> type:
    """The exception class a child failure should be raised as.

    ``TradingAgentsConfigurationError`` aborts the run on the first step;
    ``TradingAgentsRuntimeError`` holds the step and lets the run continue.
    Defaults to the transient class on purpose: a misclassified configuration
    fault costs the failure budget plus one step, while a misclassified
    transient fault throws away a run that would have recovered.

    An ambiguous message therefore stays transient. "Quota exceeded" with no
    further detail is the live example -- Google means a per-minute limit by
    it and OpenAI means a drained account -- and guessing fatal there would
    discard runs that only needed to wait.
    """
    haystack = str(detail or "").lower()
    if any(phrase in haystack for phrase in _FATAL_FAULT_PHRASES):
        return TradingAgentsConfigurationError
    if _AUTH_STATUS_RE.search(haystack) and not any(
        hint in haystack for hint in _TRANSIENT_HINTS
    ):
        return TradingAgentsConfigurationError
    return TradingAgentsRuntimeError


# Which platform credential pays for each gateway, and the env var upstream
# actually reads for it. CommonStack has no upstream provider id, so its key has
# to arrive as OPENAI_COMPATIBLE_API_KEY -- the generic endpoint's variable --
# or the child authenticates as nobody and the run dies on a 401.
_GATEWAY_CREDENTIALS: Dict[str, Tuple[str, str]] = {
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),
    "commonstack": ("COMMONSTACK_API_KEY", "OPENAI_COMPATIBLE_API_KEY"),
}

MANAGED_GATEWAY_ENV_VAR = "TRADINGAGENTS_GATEWAY"

# Autodetect order. CommonStack first, matching the rest of ATL: chat and the
# leaderboard both prefer it when its key is present, and
# ATL_PLATFORM_PROVIDER_ORDER defaults to commonstack,openrouter so its prepaid
# balance is spent before OpenRouter's.
_GATEWAY_AUTODETECT_ORDER = ("commonstack", "openrouter")


def select_managed_gateway(
    environment: Optional[Mapping[str, str]] = None,
) -> str:
    """Return the id of the gateway this deployment pays with.

    Deployment-owned, never agent-owned: an explicit override wins, otherwise
    the first gateway whose platform credential is actually set.
    """
    env = os.environ if environment is None else environment
    configured = str(env.get(MANAGED_GATEWAY_ENV_VAR) or "").strip().lower()
    if configured:
        if configured not in MANAGED_GATEWAYS:
            raise TradingAgentsConfigurationError(
                f"{MANAGED_GATEWAY_ENV_VAR} must be one of "
                f"{sorted(MANAGED_GATEWAYS)}"
            )
        return configured
    for gateway in _GATEWAY_AUTODETECT_ORDER:
        source, _ = _GATEWAY_CREDENTIALS[gateway]
        if str(env.get(source) or "").strip():
            return gateway
    return DEFAULT_MANAGED_GATEWAY


def managed_gateway_credential_env(gateway: str) -> str:
    """Return the name of the platform env var that funds ``gateway``.

    Exposed so the API's readiness gate can ask which credential this
    deployment actually needs instead of hard-coding one provider's variable.
    """
    return _GATEWAY_CREDENTIALS[gateway][0]


class TradingAgentsSubprocessRunner:
    """Invoke the bridge through its isolated interpreter and a JSON file."""

    def __init__(self, environment: Optional[Mapping[str, str]] = None):
        self.environment = dict(os.environ if environment is None else environment)

    def _python_executable(self) -> Path:
        configured = self.environment.get("TRADINGAGENTS_PYTHON", "").strip()
        candidate = (
            _resolve_path(configured)
            if configured
            else _LOCAL_RUNTIME_DIR / "bin" / "python"
        )
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise TradingAgentsConfigurationError(
                "TradingAgents runtime is not installed; configure "
                "TRADINGAGENTS_PYTHON with its isolated interpreter"
            )
        return candidate

    def _subprocess_environment(
        self, home_directory: Optional[str] = None
    ) -> Dict[str, str]:
        environment = {
            key: value
            for key, value in self.environment.items()
            if key in _SUBPROCESS_ENV_KEYS
        }
        # Do not let an SDK discover credentials in the service account's home
        # directory or let a dependency's python-dotenv walk into a checkout.
        environment["HOME"] = home_directory or tempfile.gettempdir()
        environment["PYTHON_DOTENV_DISABLED"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        # The bridge reuses the SDK's TradingAgentsDecisionGenerator rather than
        # reimplementing it, and the SDK is not a backend dependency -- so the
        # child reaches it through the source tree this deployment provides.
        environment["PYTHONPATH"] = str(_sdk_source_dir(self.environment))
        # Hand over only the credential that funds the selected gateway, under
        # the name upstream reads for it. Every other gateway's key is dropped
        # first: upstream maps its own ``openrouter`` provider to
        # OPENROUTER_API_KEY, so leaving that key in a CommonStack child's
        # environment would let a stray code path spend the wrong account.
        source, upstream_name = _GATEWAY_CREDENTIALS[
            select_managed_gateway(self.environment)
        ]
        secret = str(self.environment.get(source) or "").strip()
        for other_source, other_upstream in _GATEWAY_CREDENTIALS.values():
            environment.pop(other_source, None)
            environment.pop(other_upstream, None)
        if secret:
            environment[upstream_name] = secret
        return environment

    def run(self, payload: Dict[str, Any], *, timeout_seconds: int) -> Dict[str, Any]:
        python_executable = self._python_executable()
        bridge = Path(__file__).with_name("bridge.py")

        with tempfile.TemporaryDirectory(prefix="atl_tradingagents_") as temp_dir:
            input_path = Path(temp_dir) / "input.json"
            output_path = Path(temp_dir) / "output.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            command = [
                str(python_executable),
                str(bridge),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
            environment = self._subprocess_environment(temp_dir)
            try:
                result = subprocess.run(
                    command,
                    # Avoid implicitly loading ATL's local .env; the child needs
                    # nothing from the repository working directory.
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env=environment,
                )
            except subprocess.TimeoutExpired as exc:
                raise TradingAgentsRuntimeError(
                    f"TradingAgents runtime timed out after {timeout_seconds} seconds"
                ) from exc
            except OSError as exc:
                raise TradingAgentsRuntimeError(
                    f"TradingAgents runtime could not start: {exc}"
                ) from exc

            if result.returncode != 0:
                detail = _redact_runtime_error(
                    result.stderr or result.stdout or "unknown error", environment
                )
                # Classified rather than uniformly transient. The text is
                # already credential-free (the bridge prints
                # sanitize_error_message, and _redact_runtime_error strips
                # this call's own secrets on top), so the cause can be carried
                # to the operator without exposing upstream's raw traceback.
                #
                # ⚠ Classified from **stderr only**, while the *message* may
                # fall back to stdout. The bridge writes its failure to stderr,
                # so an empty stderr means the child was killed without getting
                # to report -- SIGKILL or the OOM killer, which this service has
                # a history of (CLAUDE.md records six oomKilled events, and up
                # to MAX_ACTIVE_DASHBOARD_BACKTESTS children share 2GB). That is
                # transient. Classifying the stdout fallback instead ran the
                # marker table over the child's own analysis prose, where a
                # model writing "authentication" or "unauthorized" about a
                # company would have aborted the run as a configuration fault.
                raise classify_runtime_failure(result.stderr)(
                    f"TradingAgents runtime failed with code "
                    f"{result.returncode}: {detail}"
                )
            try:
                output = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise TradingAgentsOutputError(
                    "TradingAgents runtime did not produce valid JSON output"
                ) from exc
            if not isinstance(output, dict):
                raise TradingAgentsOutputError(
                    "TradingAgents runtime output must be a JSON object"
                )
            return output


class TradingAgentsRuntime:
    """Hosted runtime that preserves upstream analysis behind an ATL boundary."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        runner: Optional[TradingAgentsSubprocessRunner] = None,
        environment: Optional[Mapping[str, str]] = None,
    ):
        self.config = normalize_runtime_config(
            TRADINGAGENTS_RUNTIME_TYPE, config or {}
        )
        self.environment = dict(os.environ if environment is None else environment)
        # The runner reads the *same* environment this runtime resolved its
        # gateway from. Left to its own default it read os.environ
        # independently, so the two could disagree: the runtime could select
        # CommonStack from the environment it was handed while the runner gave
        # the child OpenRouter's key from the process environment, and the run
        # would authenticate against one account for a model pinned to the
        # other. Identical in production, where both are os.environ -- which
        # is exactly why it would not have shown up until something passed an
        # explicit environment.
        self.runner = runner or TradingAgentsSubprocessRunner(self.environment)
        # Hosted model/provider selection is platform-owned. Read from the
        # bridge's own table so the value ATL reports and the value the bridge
        # will accept cannot drift apart.
        self.gateway_id = select_managed_gateway(self.environment)
        self.gateway = MANAGED_GATEWAYS[self.gateway_id]
        self.model_name = self.gateway["deep_think_llm"]
        self.timeout_seconds = resolve_step_timeout_seconds(self.environment)
        self.max_position_weight = resolve_max_position_weight(self.environment)
        # Two counters, not one. ``calls`` is what the platform paid for -- it
        # ticks the moment upstream returns, before the output is mapped, so a
        # response that fails validation is still billed. ``decisions`` is what
        # the model actually drove. CLAUDE.md records the same distinction for
        # the pipeline path (llm_calls vs llm_decisions) and why conflating
        # them lets a non-decision count as coverage.
        self.calls = 0
        self.decisions = 0
        self._last_decision_day: Optional[str] = None
        self._decision_audit_rows: List[Dict[str, Any]] = []
        # Steps that produced no decision at all and why. Distinct from a
        # decision that produced no order: a run with no decisions must not
        # read as a run whose model chose to hold.
        self.skipped_steps: List[Dict[str, str]] = []
        # Which call, if any, the pending execution count belongs to. Without
        # it a mapping failure -- which bills a call but appends no row --
        # sent the engine's count to the *previous* day's row.
        self._audit_row_call: Optional[int] = None

    @property
    def decision_audit_rows(self) -> List[Dict[str, Any]]:
        """Return bounded successful-call traces ready for ATL persistence."""
        return [
            {
                **row,
                "actions_submitted": [
                    dict(trace) for trace in row.get("actions_submitted", [])
                ],
            }
            for row in self._decision_audit_rows
        ]

    @property
    def sizing_filter_counts(self) -> Dict[str, int]:
        """How many decisions produced no order, by reason.

        The reason matters more than the count: a window of zero trades reads
        as "the model held" on every surface ATL has, and the most likely
        cause is not the model at all but ``price_above_position_budget`` --
        the position budget (equity x TRADINGAGENTS_MAX_POSITION_WEIGHT) being
        smaller than one share. At the shipped capital bounds that budget is
        $200-$600, which is below a single share of a good half of the Dow.
        """
        counts: Dict[str, int] = {}
        for row in self._decision_audit_rows:
            for trace in row.get("actions_submitted", []):
                reason = trace.get("filter_reason")
                if reason:
                    counts[str(reason)] = counts.get(str(reason), 0) + 1
        return counts

    def record_latest_execution(self, executed_count: int) -> None:
        """Attach ATL's actual execution count to *this call's* audit row.

        Keyed on the call the row was appended for, not on ``[-1]``. A call
        whose output failed validation bills the call and appends no row, so
        the engine still reports an execution count for it -- which the old
        ``[-1]`` write applied to the previous trading day, silently replacing
        a real count with this step's.
        """
        if self._audit_row_call is None or not self._decision_audit_rows:
            return
        latest = self._decision_audit_rows[-1]
        if latest.get("step_index") != self._audit_row_call:
            return
        latest["actions_executed"] = max(0, int(executed_count))
        # One write per call: a second call for the same step would be a bug
        # upstream of here, and overwriting silently is how the first one hid.
        self._audit_row_call = None

    # ------------------------------------------------------------------
    # Request
    # ------------------------------------------------------------------

    def _require_as_of_safe_analysts(self) -> None:
        """Refuse a roster whose data sources are not bounded by the run's date.

        Fail-closed rather than fail-quiet. The alternatives were both worse:
        dropping the offending analyst would execute a configuration the user
        never chose while reporting their own roster back to them, and running
        it with a contamination flag on the audit row would still publish an
        equity curve that looks like a result. Upstream exposes no hook to
        remove a single tool from an analyst, so the analyst is the smallest
        thing that can be refused.
        """
        message = describe_tradingagents_lookahead(self.config.get("analysts"))
        if message:
            raise TradingAgentsConfigurationError(message)

    def _require_single_symbol(self, context: AgentRuntimeContext) -> str:
        symbols = [str(symbol).strip().upper() for symbol in context.symbols if symbol]
        if len(symbols) != 1:
            raise TradingAgentsConfigurationError(
                "TradingAgents currently supports single-symbol runs; this run "
                f"has {len(symbols)} symbols"
            )
        return symbols[0]

    def _bridge_payload(
        self, context: AgentRuntimeContext, symbol: str
    ) -> Dict[str, Any]:
        cutoff = context.latest_market_date_before_decision
        return {
            "schema": "tradingagents-bridge-v1",
            "symbol": symbol,
            # Upstream accepts dates, not an intraday as-of timestamp. Use ATL's
            # latest available market trading date strictly before this
            # decision's date; never infer the cutoff by subtracting a day.
            "analysis_dates": [cutoff.isoformat()],
            "selected_analysts": list(self.config.get("analysts") or []),
            "config": {
                "llm_provider": self.gateway["llm_provider"],
                "deep_think_llm": self.gateway["deep_think_llm"],
                "quick_think_llm": self.gateway["quick_think_llm"],
                # Sent explicitly so the manifest records them; upstream would
                # otherwise apply its own defaults and leave no trace.
                "max_debate_rounds": MANAGED_MAX_DEBATE_ROUNDS,
                "max_risk_discuss_rounds": MANAGED_MAX_RISK_DISCUSS_ROUNDS,
            },
        }

    # ------------------------------------------------------------------
    # Response
    # ------------------------------------------------------------------

    @staticmethod
    def _decision_fields(output: Any) -> Tuple[Dict[str, Any], str, str, float]:
        decision = output.get("decision") if isinstance(output, dict) else None
        if not isinstance(decision, dict):
            raise TradingAgentsOutputError(
                'TradingAgents output must include a "decision" object'
            )
        action = str(decision.get("atl_action") or "").strip().upper()
        rating = str(decision.get("rating") or "").strip().lower()
        status = str(decision.get("status") or "").strip().lower()
        if status == "error":
            # An upstream failure the SDK turned into a record rather than an
            # exception. It carries rating="" by construction, so reporting it
            # as an unsupported rating would name the symptom and hide every
            # cause -- a rejected credential, a dead data vendor and a
            # malformed generation would read identically.
            error_type = TradingAgentsRuntime._bounded_text(
                decision.get("error_type")
            ) or "unknown"
            error_message = TradingAgentsRuntime._bounded_text(
                decision.get("error_message")
            ) or "no detail reported"
            detail = f"{error_type}: {error_message}"
            raise classify_runtime_failure(detail)(
                f"TradingAgents analysis failed ({detail})"
            )
        if rating not in _RATING_TO_ACTION:
            raise TradingAgentsOutputError(
                f"TradingAgents returned an unsupported rating {rating!r}"
            )
        if action != _RATING_TO_ACTION[rating]:
            raise TradingAgentsOutputError(
                f"TradingAgents rating {rating!r} does not map to action {action!r}"
            )
        return decision, action, rating, _RATING_CONFIDENCE[rating]

    @staticmethod
    def _bounded_text(value: Any, *, limit: int = _MAX_DIAGNOSTIC_TEXT_LENGTH) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.split())[:limit]

    @staticmethod
    def _bounded_int(value: Any) -> Optional[int]:
        """Accept only a real int; a crafted manifest must not inject a type."""
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _bounded_str_list(value: Any, *, limit: int = 8) -> List[str]:
        """The manifest is upstream-controlled, so cap the list and drop
        anything that is not a plain string."""
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)][:limit]

    @classmethod
    def _diagnostics(
        cls, output: Dict[str, Any], decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Allow-list the audit fields; the debate transcript never appears.

        The bridge already withholds ``raw_final_trade_decision``; naming the
        kept fields explicitly means a future bridge that started returning it
        could not leak it into ATL run metadata by accident.
        """
        manifest = output.get("manifest")
        manifest = manifest if isinstance(manifest, dict) else {}
        return {
            "status": cls._bounded_text(decision.get("status"), limit=16),
            # Through the helper, not an inline isinstance: `isinstance(True,
            # int)` is True, so the inline form persisted `attempts: true` as
            # 1 -- the one audit field that bypassed the guard written for
            # exactly that.
            "attempts": cls._bounded_int(decision.get("attempts")),
            "raw_sha256": cls._bounded_text(decision.get("raw_sha256"), limit=64),
            "raw_excerpt": cls._bounded_text(decision.get("raw_excerpt")),
            "tradingagents_version": cls._bounded_text(
                manifest.get("tradingagents_version"), limit=32
            ),
            "safe_config_sha256": cls._bounded_text(
                manifest.get("safe_config_sha256"), limit=64
            ),
            # Both models, taken from the manifest the bridge returned rather
            # than from ATL's own config -- this records what upstream actually
            # ran. Upstream gives 9 of its 11 LLM calls to the quick model, so
            # a row naming only the deep one cannot support comparing one
            # experiment's models against another's.
            "quick_think_llm": cls._bounded_text(
                manifest.get("quick_think_llm"), limit=64
            ),
            "deep_think_llm": cls._bounded_text(
                manifest.get("deep_think_llm"), limit=64
            ),
            # The run parameters themselves, not only their hash:
            # ``safe_config_sha256`` can say two runs differed but never what
            # differed, and the debate rounds are what set the call count.
            "llm_provider": cls._bounded_text(
                manifest.get("llm_provider"), limit=32
            ),
            "selected_analysts": cls._bounded_str_list(
                manifest.get("selected_analysts")
            ),
            "max_debate_rounds": cls._bounded_int(
                manifest.get("max_debate_rounds")
            ),
            "max_risk_discuss_rounds": cls._bounded_int(
                manifest.get("max_risk_discuss_rounds")
            ),
            # The exact revision, which ``tradingagents_version`` cannot give:
            # it reports the tag, and the pin is six commits past it.
            "upstream_commit": cls._bounded_text(
                output.get("upstream_commit"), limit=40
            ),
        }

    def _target_shares(self, context: AgentRuntimeContext, price: float) -> int:
        """Shares that put this position at the platform's target weight.

        This is the *target position*, deliberately unclamped. The per-order
        ceiling belongs on the order (see ``_sized_order``): clamping the
        target instead capped the position itself at MAX_ORDER_SHARES forever,
        and once the position reached that cap the next day reported
        ``already_at_target_weight`` -- false, since the position could be a
        few percent of the 20% target it was still trying to reach.
        """
        budget = float(context.total_equity) * self.max_position_weight
        if not math.isfinite(budget) or budget <= 0:
            return 0
        return int(math.floor(budget / price))

    def _sized_order(
        self, *, action: str, context: AgentRuntimeContext, symbol: str
    ) -> Tuple[int, Optional[str]]:
        """Return ``(shares, filter_reason)`` for one mapped direction.

        TradingAgents supplies neither number. The model is the SDK replay's:
        buy up to a target weight of equity, sell the whole position.
        """
        held = int(context.positions.get(symbol, 0) or 0)
        price = float(context.current_prices.get(symbol, 0) or 0)
        if action == "HOLD":
            return 0, "model_hold"
        if not math.isfinite(price) or price <= 0:
            return 0, "missing_or_invalid_price"
        if action == "SELL":
            if held <= 0:
                return 0, "sell_without_position"
            return held, None
        target = self._target_shares(context, price)
        if target <= 0:
            # Not a model decision, and the single likeliest cause of a
            # flat curve: equity x max_position_weight is below one share.
            return 0, "price_above_position_budget"
        shares = target - held
        if shares <= 0:
            return 0, "already_at_target_weight"
        if shares > MAX_ORDER_SHARES:
            # The per-order ceiling every other order path in this repo
            # applies. This adapter computes the quantity itself rather than
            # receiving it from upstream, so nothing else bounds it:
            # ``parse_actions_payload`` *rejects* an oversized order, which
            # this adapter raised as a transient output error -- and the
            # computation is deterministic, so the engine spent its whole
            # per-run failure budget, and real money, rediscovering the same
            # arithmetic once per trading day. Reachable for a sub-dollar
            # ticker, and more easily if an operator raises
            # TRADINGAGENTS_MAX_POSITION_WEIGHT. Topped up over later days
            # rather than refused, so the order IS emitted -- a filter_reason
            # here would suppress it, because the caller treats any reason as
            # "no order". The cap is recorded in the sizing block instead.
            return MAX_ORDER_SHARES, None
        return shares, None

    def _sizing_diagnostics(
        self, context: AgentRuntimeContext, symbol: str
    ) -> Dict[str, Any]:
        """The arithmetic behind a sizing decision, recorded either way.

        Recorded on every trace rather than only on a refusal: "the budget
        could not buy one share" is only legible beside the budget and the
        price, and a reader looking at a window of zero trades needs to be
        able to tell this from a model that chose to hold.
        """
        price = float(context.current_prices.get(symbol, 0) or 0)
        equity = float(context.total_equity)
        budget = equity * self.max_position_weight
        return {
            "total_equity": round(equity, 2),
            "max_position_weight": self.max_position_weight,
            "position_budget": round(budget, 2) if math.isfinite(budget) else None,
            "price": round(price, 4) if math.isfinite(price) else None,
            "held_shares": int(context.positions.get(symbol, 0) or 0),
            "target_shares": self._target_shares(context, price) if price > 0 else 0,
            "max_order_shares": MAX_ORDER_SHARES,
            # Whether the per-order ceiling shortened this order. The position
            # is topped up over later days, so an order smaller than the
            # target implies is not a bug -- but it is not obvious either.
            "order_capped": bool(
                (self._target_shares(context, price) if price > 0 else 0)
                - int(context.positions.get(symbol, 0) or 0)
                > MAX_ORDER_SHARES
            ),
        }

    def _trace(
        self,
        *,
        context: AgentRuntimeContext,
        symbol: str,
        rating: str,
        action: str,
        shares: int,
        confidence: float,
        filter_reason: Optional[str],
        diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        cutoff = context.latest_market_date_before_decision
        return {
            "decision_date": context.timestamp.date().isoformat(),
            "data_cutoff_date": cutoff.isoformat() if cutoff is not None else "",
            "ticker": symbol,
            "upstream_action": rating,
            "upstream_quantity": None,
            "confidence": round(confidence * 100.0, 4),
            "mapped_atl_action": action,
            "sized_quantity": shares,
            "order_emitted": False,
            "filter_reason": filter_reason,
            "sizing": self._sizing_diagnostics(context, symbol),
            "diagnostics": diagnostics,
        }

    def output_to_atl_actions_with_trace(
        self, output: Dict[str, Any], context: AgentRuntimeContext
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Map one bridge decision and return a credential-free audit trace."""
        symbol = self._require_single_symbol(context)
        decision, action, rating, confidence = self._decision_fields(output)
        shares, filter_reason = self._sized_order(
            action=action, context=context, symbol=symbol
        )
        trace = self._trace(
            context=context,
            symbol=symbol,
            rating=rating,
            action=action,
            shares=shares,
            confidence=confidence,
            filter_reason=filter_reason,
            diagnostics=self._diagnostics(output, decision),
        )
        if filter_reason is not None or shares <= 0:
            return [], [trace]

        parsed, error = parse_actions_payload(
            {
                "actions": [
                    {
                        "symbol": symbol,
                        "action": action.lower(),
                        "position_size": shares,
                        "confidence": confidence,
                        "reasoning": (
                            f"TradingAgents rating={rating}; "
                            f"target_weight={self.max_position_weight:g}"
                        ),
                    }
                ]
            }
        )
        if error or parsed is None:
            raise TradingAgentsOutputError(
                f"TradingAgents decision failed ATL validation: {error}"
            )
        actions = actions_to_executable(
            parsed,
            cash=float(context.cash),
            positions={key: int(value) for key, value in context.positions.items()},
            current_prices={
                key: float(value) for key, value in context.current_prices.items()
            },
        )
        if actions:
            trace["order_emitted"] = True
        else:
            # actions_to_executable applies the final affordability and
            # confidence filters. Name the one that can still fire here rather
            # than leaving the trace reading like an upstream hold.
            trace["filter_reason"] = (
                "insufficient_cash"
                if action == "BUY" and shares * float(
                    context.current_prices.get(symbol, 0) or 0
                ) > float(context.cash)
                else "rejected_by_executable_filters"
            )
        for emitted in actions:
            reason = str(emitted.get("reason") or "")
            emitted["reason"] = reason.replace("[External]", "[TradingAgents]", 1)
        return actions, [trace]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def decide(self, context: AgentRuntimeContext) -> Dict[str, List[Dict[str, Any]]]:
        # Before anything else, including the repeat-day short circuit: a
        # roster that reads future data is wrong on every step, and a run that
        # holds its way through the window would still publish an equity curve
        # built on nothing. The API refuses this roster at request time; this
        # is the backstop for the CLI path, which has no router in front of it.
        self._require_as_of_safe_analysts()
        decision_day = context.timestamp.date().isoformat()
        if (
            DEFAULT_DECISION_INTERVAL == "daily"
            and decision_day == self._last_decision_day
        ):
            return {"actions": []}
        # Refuse an unsupported universe before the day is consumed, so the
        # engine aborts on a deployment-shaped error rather than holding.
        symbol = self._require_single_symbol(context)
        if context.latest_market_date_before_decision is None:
            # The first ATL trading date has no strictly earlier market date in
            # the loaded run. Holding is safer than inventing a calendar cutoff
            # -- but it is recorded, because a one-day backtest consists of
            # exactly this step and would otherwise complete "successfully"
            # having never asked the model anything.
            self._last_decision_day = decision_day
            self.skipped_steps.append(
                {
                    "timestamp": context.timestamp.isoformat(),
                    "reason": "no_market_date_before_decision",
                }
            )
            return {"actions": []}

        payload = self._bridge_payload(context, symbol)
        # Consume the day *before* the call, not after it. Left until the
        # success path, a failed decision leaves the day unmarked and the next
        # hourly bar retries upstream -- so one bad day costs bars-per-day full
        # timeouts and API spend for no new information.
        self._last_decision_day = decision_day
        # Cleared *before* the call, not after it. The engine calls
        # record_latest_execution whenever the call counter moved, and the
        # counter now also moves on a mid-flight failure -- so a stale pointer
        # here would send this step's execution count to the previous trading
        # day's row, which is the bug record_latest_execution exists to prevent.
        self._audit_row_call = None
        try:
            output = self.runner.run(payload, timeout_seconds=self.timeout_seconds)
        except TradingAgentsConfigurationError:
            # Nothing was spent: the interpreter is missing, the payload was
            # refused before `generate()`, or the gateway rejected the
            # credential. Re-raise without counting a billed call.
            raise
        except TradingAgentsRuntimeError:
            # The child ran and was killed or failed mid-flight -- a timeout
            # after ~10 of 11 model calls is the *designed-in* outcome at the
            # shipped 600s budget, because one decision is two upstream passes
            # (580-830s). Those tokens are spent and discarded, so the call has
            # to be counted: with the increment only on the success path, a
            # five-day run whose every day timed out persisted llm_calls = 0
            # and llm_model = "rule-based" -- reporting zero platform spend for
            # four near-complete billed debates.
            self.calls += 1
            raise
        self.calls += 1
        call_index = self.calls - 1
        # Cleared first: if the mapping below raises, this call has no row, and
        # the engine's execution count for it must land nowhere rather than on
        # the previous day's.
        self._audit_row_call = None
        actions, traces = self.output_to_atl_actions_with_trace(output, context)
        self.decisions += 1
        self._audit_row_call = call_index
        cutoff = context.latest_market_date_before_decision
        self._decision_audit_rows.append(
            {
                "step_index": call_index,
                "timestamp": context.timestamp.isoformat(),
                "decision_source": TRADINGAGENTS_RUNTIME_TYPE,
                "actions_submitted": traces,
                "actions_executed": 0,
                "context_ref": cutoff.isoformat() if cutoff is not None else None,
            }
        )
        return {"actions": actions}
