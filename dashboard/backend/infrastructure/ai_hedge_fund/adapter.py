"""Thin, subprocess-isolated adapter for ``virattt/ai-hedge-fund``.

The upstream package runs in its own virtual environment. This module only
translates ATL portfolio state into upstream's public ``run_hedge_fund`` input,
then translates the returned decisions through ATL's existing strict action
schema and executable-action constraints. It never executes or mutates trades.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from dashboard.backend.domain.agents.runtime import (
    AI_HEDGE_FUND_ANALYSTS,
    AI_HEDGE_FUND_RUNTIME_TYPE,
    AgentRuntimeConfigurationError,
    AgentRuntimeContext,
    AgentRuntimeError,
    normalize_runtime_config,
)
from dashboard.backend.infrastructure.llm.validator import (
    MAX_ORDER_SHARES,
    actions_to_executable,
    parse_actions_payload,
)
from dashboard.backend.infrastructure.llm.providers import openrouter
from dashboard.backend.paths import REPO_ROOT

DEFAULT_MODEL_NAME = openrouter.DEFAULT_MODEL
DEFAULT_MODEL_PROVIDER = "OpenRouter"
DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_TIMEOUT_SECONDS = 300
MIN_TIMEOUT_SECONDS = 1
MAX_TIMEOUT_SECONDS = 900
DEFAULT_DECISION_INTERVAL = "daily"
_LOCAL_RUNTIME_DIR = REPO_ROOT / ".ai-hedge-fund-venv"
_MIN_EXECUTABLE_CONFIDENCE = 0.3
_MAX_DIAGNOSTIC_TEXT_LENGTH = 240
_ANALYST_SIGNAL_IDS = frozenset(
    f"{analyst}_agent" for analyst in AI_HEDGE_FUND_ANALYSTS
)

# The pinned upstream process only receives values needed for networking,
# locale/runtime behavior, its market-data client, and its supported model
# provider. In particular, ATL database URLs, alternate model-provider keys,
# and unrelated service secrets do not cross the process boundary.
_SUBPROCESS_ENV_KEYS = frozenset(
    {
        "PATH",
        "TMPDIR",
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
        "FINANCIAL_DATASETS_API_KEY",
        "OPENROUTER_API_KEY",
    }
)


class AiHedgeFundRuntimeError(AgentRuntimeError):
    """Base exception for runtime execution failures.

    Deliberately *transient* by default: the backtest engine holds the step and
    keeps the run alive for these, because a timeout or a bad upstream response
    on one day says nothing about the next.
    """


class AiHedgeFundConfigurationError(
    AiHedgeFundRuntimeError, AgentRuntimeConfigurationError
):
    """Raised when the deployment itself cannot host the runtime.

    Distinct from the transient base so the engine can abort immediately. A
    missing interpreter or a malformed platform setting fails identically on
    every step, and retrying it once per trading day only converts one clear
    error into a long, silent, all-holds run.
    """


class AiHedgeFundOutputError(AiHedgeFundRuntimeError):
    """Raised when upstream returns data that cannot safely become ATL orders."""


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_step_timeout_seconds(
    environment: Optional[Mapping[str, str]] = None,
) -> int:
    """Return the per-decision subprocess budget.

    Shared with the API layer so the caller that sizes the *outer* backtest
    subprocess timeout and the runtime that sizes each *inner* upstream call
    read the same number. Two independent readings of
    ``AI_HEDGE_FUND_TIMEOUT_SECONDS`` is how the parent ends up killing the
    child mid-run.
    """
    source = os.environ if environment is None else environment
    raw = str(source.get("AI_HEDGE_FUND_TIMEOUT_SECONDS", "") or "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError as exc:
        raise AiHedgeFundConfigurationError(
            "AI_HEDGE_FUND_TIMEOUT_SECONDS must be an integer"
        ) from exc
    if not MIN_TIMEOUT_SECONDS <= value <= MAX_TIMEOUT_SECONDS:
        raise AiHedgeFundConfigurationError(
            "AI_HEDGE_FUND_TIMEOUT_SECONDS must be between "
            f"{MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS}"
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
        AiHedgeFundSubprocessRunner(environment)._python_executable()
    except AiHedgeFundRuntimeError as exc:
        return str(exc)
    if not Path(__file__).with_name("bridge.py").is_file():
        return "AI Hedge Fund bridge script is missing from this deployment"
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


class AiHedgeFundSubprocessRunner:
    """Invoke upstream through its isolated interpreter and a JSON file bridge."""

    def __init__(self, environment: Optional[Mapping[str, str]] = None):
        self.environment = dict(os.environ if environment is None else environment)

    def _python_executable(self) -> Path:
        configured = self.environment.get("AI_HEDGE_FUND_PYTHON", "").strip()
        candidate = (
            _resolve_path(configured)
            if configured
            else _LOCAL_RUNTIME_DIR / "bin" / "python"
        )
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise AiHedgeFundConfigurationError(
                "AI Hedge Fund runtime is not installed; configure "
                "AI_HEDGE_FUND_PYTHON with its isolated interpreter"
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
        # directory or let upstream python-dotenv walk into a checkout's .env.
        environment["HOME"] = home_directory or tempfile.gettempdir()
        environment["PYTHON_DOTENV_DISABLED"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        return environment

    def run(self, payload: Dict[str, Any], *, timeout_seconds: int) -> Dict[str, Any]:
        python_executable = self._python_executable()
        bridge = Path(__file__).with_name("bridge.py")
        project_root_raw = self.environment.get("AI_HEDGE_FUND_ROOT", "").strip()
        project_root = _resolve_path(project_root_raw) if project_root_raw else None
        if project_root is not None and not project_root.is_dir():
            raise AiHedgeFundConfigurationError(
                "AI_HEDGE_FUND_ROOT does not point to an upstream checkout"
            )

        with tempfile.TemporaryDirectory(prefix="atl_ai_hedge_fund_") as temp_dir:
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
            if project_root is not None:
                environment["PYTHONPATH"] = str(project_root)
            try:
                result = subprocess.run(
                    command,
                    # Avoid implicitly loading ATL's local .env. An optional
                    # read-only upstream checkout is importable via the child's
                    # controlled PYTHONPATH and does not need to be the cwd.
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env=environment,
                )
            except subprocess.TimeoutExpired as exc:
                raise AiHedgeFundRuntimeError(
                    f"AI Hedge Fund runtime timed out after {timeout_seconds} seconds"
                ) from exc
            except OSError as exc:
                raise AiHedgeFundRuntimeError(
                    f"AI Hedge Fund runtime could not start: {exc}"
                ) from exc

            if result.returncode != 0:
                detail = _redact_runtime_error(
                    result.stderr or result.stdout or "unknown error", environment
                )
                raise AiHedgeFundRuntimeError(
                    f"AI Hedge Fund runtime failed with code {result.returncode}: {detail}"
                )
            try:
                output = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise AiHedgeFundOutputError(
                    "AI Hedge Fund runtime did not produce valid JSON output"
                ) from exc
            if not isinstance(output, dict):
                raise AiHedgeFundOutputError(
                    "AI Hedge Fund runtime output must be a JSON object"
                )
            return output


class AiHedgeFundRuntime:
    """Hosted runtime that preserves upstream analysis behind an ATL boundary."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        runner: Optional[AiHedgeFundSubprocessRunner] = None,
        environment: Optional[Mapping[str, str]] = None,
    ):
        self.config = normalize_runtime_config(
            AI_HEDGE_FUND_RUNTIME_TYPE, config or {}
        )
        self.runner = runner or AiHedgeFundSubprocessRunner()
        self.environment = dict(os.environ if environment is None else environment)
        # Hosted model/provider selection is platform-owned. Reuse ATL's
        # registered OpenRouter model instead of accepting agent config or an
        # AI-Hedge-Fund-specific model override.
        self.model_name = DEFAULT_MODEL_NAME
        self.lookback_days = self._platform_int(
            "AI_HEDGE_FUND_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS, 1, 3650
        )
        self.timeout_seconds = resolve_step_timeout_seconds(self.environment)
        self.calls = 0
        self._last_decision_day: Optional[str] = None
        self._decision_audit_rows: list[Dict[str, Any]] = []

    @property
    def decision_audit_rows(self) -> list[Dict[str, Any]]:
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

    def record_latest_execution(self, executed_count: int) -> None:
        """Attach ATL's actual execution count to the latest runtime call."""
        if self._decision_audit_rows:
            self._decision_audit_rows[-1]["actions_executed"] = max(
                0, int(executed_count)
            )

    def _platform_int(
        self, name: str, default: int, minimum: int, maximum: int
    ) -> int:
        raw = self.environment.get(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise AiHedgeFundConfigurationError(
                f"{name} must be an integer"
            ) from exc
        if not minimum <= value <= maximum:
            raise AiHedgeFundConfigurationError(
                f"{name} must be between {minimum} and {maximum}"
            )
        return value

    def _upstream_payload(self, context: AgentRuntimeContext) -> Dict[str, Any]:
        # Upstream accepts dates, not an intraday as-of timestamp. Use ATL's
        # latest available market trading date strictly before this decision's
        # date; never infer the cutoff by subtracting a calendar day.
        end_date = context.latest_market_date_before_decision
        if end_date is None:
            raise AiHedgeFundRuntimeError(
                "AI Hedge Fund requires an ATL market date before the decision date"
            )
        if end_date >= context.timestamp.date():
            raise AiHedgeFundRuntimeError(
                "AI Hedge Fund data cutoff must be before the decision date"
            )
        start_date = end_date - timedelta(days=self.lookback_days)
        positions = {
            symbol: {
                "long": int(context.positions.get(symbol, 0) or 0),
                "short": 0,
                "long_cost_basis": float(context.entry_prices.get(symbol, 0) or 0),
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for symbol in context.symbols
        }
        return {
            "tickers": list(context.symbols),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "portfolio": {
                "cash": float(context.cash),
                "equity": float(context.total_equity),
                "margin_requirement": 0.0,
                "margin_used": 0.0,
                "positions": positions,
                "realized_gains": {
                    symbol: {"long": 0.0, "short": 0.0}
                    for symbol in context.symbols
                },
            },
            "show_reasoning": False,
            "selected_analysts": list(self.config.get("analysts") or []),
            "model_name": self.model_name,
            "model_provider": DEFAULT_MODEL_PROVIDER,
        }

    @staticmethod
    def _common_decision_fields(
        symbol: str, decision: Any
    ) -> tuple[str, Optional[int], float, str]:
        if not isinstance(decision, dict):
            raise AiHedgeFundOutputError(f"Decision for {symbol} must be an object")
        action = str(decision.get("action") or "").strip().lower()
        if action not in {"buy", "sell", "hold", "short", "cover"}:
            raise AiHedgeFundOutputError(
                f"Decision for {symbol} has unsupported action {action!r}"
            )
        quantity = decision.get("quantity", 0)
        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, int)
            or not 0 <= quantity <= MAX_ORDER_SHARES
        ):
            quantity = None
        confidence_raw = decision.get("confidence", 0)
        if isinstance(confidence_raw, bool) or not isinstance(
            confidence_raw, (int, float)
        ):
            raise AiHedgeFundOutputError(
                f"Decision for {symbol} has invalid confidence"
            )
        confidence_pct = float(confidence_raw)
        if not 0 <= confidence_pct <= 100:
            raise AiHedgeFundOutputError(
                f"Decision for {symbol} has invalid confidence"
            )
        reasoning_raw = decision.get("reasoning", "")
        if not isinstance(reasoning_raw, str):
            raise AiHedgeFundOutputError(
                f"Decision for {symbol} has invalid reasoning"
            )
        reasoning = reasoning_raw.strip() or "AI Hedge Fund decision"
        if len(reasoning) < 5:
            reasoning = f"AI Hedge Fund: {reasoning}"
        return action, quantity, confidence_pct / 100.0, reasoning[:500]

    @staticmethod
    def _trace(
        *,
        decision_date: str,
        data_cutoff_date: str,
        ticker: str,
        upstream_action: str,
        upstream_quantity: Optional[int],
        confidence: float,
        mapped_atl_action: Optional[str],
        order_emitted: bool,
        filter_reason: Optional[str],
    ) -> Dict[str, Any]:
        """Return the fixed, bounded audit representation of one decision."""
        return {
            "decision_date": decision_date,
            "data_cutoff_date": data_cutoff_date,
            "ticker": ticker,
            "upstream_action": upstream_action,
            "upstream_quantity": upstream_quantity,
            "confidence": round(max(0.0, min(100.0, confidence * 100.0)), 4),
            "mapped_atl_action": mapped_atl_action,
            "order_emitted": bool(order_emitted),
            "filter_reason": filter_reason,
        }

    @staticmethod
    def _bounded_text(value: Any, *, limit: int = _MAX_DIAGNOSTIC_TEXT_LENGTH) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.split())[:limit]

    @staticmethod
    def _bounded_number(
        value: Any,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        if minimum is not None:
            number = max(minimum, number)
        if maximum is not None:
            number = min(maximum, number)
        return round(number, 6)

    @staticmethod
    def _bounded_integer(value: Any) -> Optional[int]:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return max(0, min(value, 1_000_000))

    @classmethod
    def _sanitize_analyst_signals(
        cls, analyst_signals: Any, symbol: str
    ) -> Dict[str, Dict[str, Any]]:
        """Keep only the pinned analysts' bounded signal/confidence summaries."""
        if not isinstance(analyst_signals, dict):
            return {}
        summaries: Dict[str, Dict[str, Any]] = {}
        for agent_id in sorted(_ANALYST_SIGNAL_IDS):
            by_ticker = analyst_signals.get(agent_id)
            payload = by_ticker.get(symbol) if isinstance(by_ticker, dict) else None
            if not isinstance(payload, dict):
                continue
            signal = cls._bounded_text(payload.get("signal"), limit=64)
            confidence = cls._bounded_number(
                payload.get("confidence"), minimum=0.0, maximum=100.0
            )
            summaries[agent_id] = {
                "signal": signal,
                "confidence": confidence,
            }
        return summaries

    @classmethod
    def _sanitize_risk_output(
        cls, analyst_signals: Any, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Allow-list the exact risk fields emitted by the pinned runtime."""
        if not isinstance(analyst_signals, dict):
            return None
        by_ticker = analyst_signals.get("risk_management_agent")
        risk = by_ticker.get(symbol) if isinstance(by_ticker, dict) else None
        if not isinstance(risk, dict):
            return None

        sanitized: Dict[str, Any] = {}
        for key in ("remaining_position_limit", "current_price"):
            if key in risk:
                sanitized[key] = cls._bounded_number(risk.get(key))

        volatility = risk.get("volatility_metrics")
        if isinstance(volatility, dict):
            volatility_summary = {
                key: cls._bounded_number(volatility.get(key))
                for key in (
                    "daily_volatility",
                    "annualized_volatility",
                    "volatility_percentile",
                )
                if key in volatility
            }
            if "data_points" in volatility:
                volatility_summary["data_points"] = cls._bounded_integer(
                    volatility.get("data_points")
                )
            sanitized["volatility_metrics"] = volatility_summary

        correlation = risk.get("correlation_metrics")
        if isinstance(correlation, dict):
            correlation_summary: Dict[str, Any] = {}
            for key in (
                "avg_correlation_with_active",
                "max_correlation_with_active",
            ):
                if key in correlation:
                    value = correlation.get(key)
                    correlation_summary[key] = (
                        None if value is None else cls._bounded_number(value)
                    )
            top_correlated = correlation.get("top_correlated_tickers")
            if isinstance(top_correlated, list):
                bounded_correlations = []
                for item in top_correlated[:3]:
                    if not isinstance(item, dict):
                        continue
                    bounded_correlations.append(
                        {
                            "ticker": cls._bounded_text(item.get("ticker"), limit=16),
                            "correlation": cls._bounded_number(item.get("correlation")),
                        }
                    )
                correlation_summary["top_correlated_tickers"] = bounded_correlations
            sanitized["correlation_metrics"] = correlation_summary

        reasoning = risk.get("reasoning")
        if isinstance(reasoning, dict):
            reasoning_summary: Dict[str, Any] = {}
            for key in (
                "portfolio_value",
                "current_position_value",
                "base_position_limit_pct",
                "correlation_multiplier",
                "combined_position_limit_pct",
                "position_limit",
                "remaining_limit",
                "available_cash",
            ):
                if key in reasoning:
                    reasoning_summary[key] = cls._bounded_number(reasoning.get(key))
            for key in ("risk_adjustment", "error"):
                if key in reasoning:
                    reasoning_summary[key] = cls._bounded_text(reasoning.get(key))
            sanitized["reasoning"] = reasoning_summary
        return sanitized

    @classmethod
    def _bounded_diagnostics(
        cls,
        *,
        output: Dict[str, Any],
        symbol: str,
        decision: Dict[str, Any],
        action: str,
        quantity: Optional[int],
        confidence: float,
    ) -> Optional[Dict[str, Any]]:
        analyst_signals = output.get("analyst_signals")
        if not isinstance(analyst_signals, dict):
            return None
        return {
            "analyst_signals": cls._sanitize_analyst_signals(analyst_signals, symbol),
            "risk_management_agent": cls._sanitize_risk_output(analyst_signals, symbol),
            "portfolio_manager": {
                "action": action,
                "quantity": quantity,
                "confidence": round(max(0.0, min(100.0, confidence * 100.0)), 4),
                "reasoning_summary": cls._bounded_text(decision.get("reasoning")),
            },
        }

    @staticmethod
    def _no_order_reason(
        *,
        action: str,
        quantity: int,
        confidence: float,
        context: AgentRuntimeContext,
        symbol: str,
    ) -> str:
        """Mirror ATL executable-action checks in their established order.

        Falls back to ``"unknown"`` rather than raising. This is an audit-trail
        label, and the enumeration hand-mirrors ``actions_to_executable``: the
        day a filter is added there, raising here would abort a whole backtest
        over a missing string. An unlabelled trace is a far cheaper defect than
        a lost run.
        """
        if confidence < _MIN_EXECUTABLE_CONFIDENCE:
            return "below_minimum_confidence"
        price = float(context.current_prices.get(symbol, 0) or 0)
        if not math.isfinite(price) or price <= 0:
            return "missing_or_invalid_price"
        if action == "buy":
            if quantity <= 0:
                return "zero_quantity"
            if quantity * price > float(context.cash):
                return "insufficient_cash"
        elif action == "sell":
            if int(context.positions.get(symbol, 0) or 0) <= 0:
                return "sell_without_position"
        return "unknown"

    @classmethod
    def output_to_atl_actions_with_trace(
        cls,
        output: Dict[str, Any],
        context: AgentRuntimeContext,
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        """Map upstream output and return credential-free per-ticker traces."""
        decisions = output.get("decisions") if isinstance(output, dict) else None
        if not isinstance(decisions, dict):
            raise AiHedgeFundOutputError(
                'AI Hedge Fund output must include a "decisions" object'
            )

        decision_date = context.timestamp.date().isoformat()
        cutoff = context.latest_market_date_before_decision
        data_cutoff_date = cutoff.isoformat() if cutoff is not None else ""
        allowed = {str(symbol).strip().upper() for symbol in context.symbols}
        normalized = []
        trace_by_symbol: Dict[str, Dict[str, Any]] = {}
        seen = set()
        for raw_symbol, raw_decision in decisions.items():
            symbol = str(raw_symbol or "").strip().upper()
            if symbol not in allowed or symbol in seen:
                raise AiHedgeFundOutputError(
                    f"AI Hedge Fund returned invalid symbol {raw_symbol!r}"
                )
            seen.add(symbol)
            action, quantity, confidence, reasoning = cls._common_decision_fields(
                symbol, raw_decision
            )
            trace = cls._trace(
                decision_date=decision_date,
                data_cutoff_date=data_cutoff_date,
                ticker=symbol,
                upstream_action=action,
                upstream_quantity=quantity,
                confidence=confidence,
                mapped_atl_action=action if action in {"buy", "sell"} else "hold",
                order_emitted=False,
                filter_reason=None,
            )
            diagnostics = cls._bounded_diagnostics(
                output=output,
                symbol=symbol,
                decision=raw_decision,
                action=action,
                quantity=quantity,
                confidence=confidence,
            )
            if diagnostics is not None:
                trace["diagnostics"] = diagnostics
            trace_by_symbol[symbol] = trace

            if quantity is None:
                trace["mapped_atl_action"] = None
                trace["filter_reason"] = "invalid_quantity"
                continue
            if action == "hold":
                trace["filter_reason"] = "upstream_hold"
                continue
            if action == "short":
                trace["filter_reason"] = "short_unsupported_long_only"
                continue
            if action == "cover":
                trace["filter_reason"] = "cover_unsupported_long_only"
                continue
            normalized.append(
                {
                    "symbol": symbol,
                    "action": action,
                    "position_size": quantity,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }
            )

        parsed, error = parse_actions_payload({"actions": normalized})
        if error or parsed is None:
            raise AiHedgeFundOutputError(
                f"AI Hedge Fund decisions failed ATL validation: {error}"
            )
        actions = actions_to_executable(
            parsed,
            cash=float(context.cash),
            positions={key: int(value) for key, value in context.positions.items()},
            current_prices={
                key: float(value) for key, value in context.current_prices.items()
            },
        )
        emitted_symbols = {str(action.get("symbol") or "") for action in actions}
        for decision in parsed:
            trace = trace_by_symbol[decision.symbol]
            if decision.symbol in emitted_symbols:
                trace["order_emitted"] = True
            else:
                trace["filter_reason"] = cls._no_order_reason(
                    action=str(getattr(decision.action, "value", decision.action)),
                    quantity=int(decision.position_size),
                    confidence=float(decision.confidence),
                    context=context,
                    symbol=decision.symbol,
                )
        for action in actions:
            reason = str(action.get("reason") or "")
            action["reason"] = reason.replace(
                "[External]", "[AI Hedge Fund]", 1
            )
        return actions, list(trace_by_symbol.values())

    @classmethod
    def output_to_atl_actions(
        cls,
        output: Dict[str, Any],
        context: AgentRuntimeContext,
    ) -> list[Dict[str, Any]]:
        actions, _traces = cls.output_to_atl_actions_with_trace(output, context)
        return actions

    def decide(self, context: AgentRuntimeContext) -> Dict[str, list[Dict[str, Any]]]:
        decision_day = context.timestamp.date().isoformat()
        if DEFAULT_DECISION_INTERVAL == "daily" and decision_day == self._last_decision_day:
            return {"actions": []}
        if context.latest_market_date_before_decision is None:
            # The first ATL trading date has no strictly earlier market date in
            # the loaded run. Holding is safer than inventing a calendar cutoff.
            self._last_decision_day = decision_day
            return {"actions": []}

        payload = self._upstream_payload(context)
        # Consume the day *before* the call, not after it. Left until the
        # success path, a failed decision leaves the day unmarked and the next
        # hourly bar retries upstream -- so one bad day costs bars-per-day full
        # timeouts and API spend for no new information.
        self._last_decision_day = decision_day
        output = self.runner.run(payload, timeout_seconds=self.timeout_seconds)
        self.calls += 1
        actions, traces = self.output_to_atl_actions_with_trace(output, context)
        cutoff = context.latest_market_date_before_decision
        self._decision_audit_rows.append(
            {
                "step_index": self.calls - 1,
                "timestamp": context.timestamp.isoformat(),
                "decision_source": AI_HEDGE_FUND_RUNTIME_TYPE,
                "actions_submitted": traces,
                "actions_executed": 0,
                "context_ref": cutoff.isoformat() if cutoff is not None else None,
            }
        )
        return {"actions": actions}
