"""Thin, subprocess-isolated adapter for ``virattt/ai-hedge-fund``.

The upstream package runs in its own virtual environment. This module only
translates ATL portfolio state into upstream's public ``run_hedge_fund`` input,
then translates the returned decisions through ATL's existing strict action
schema and executable-action constraints. It never executes or mutates trades.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from dashboard.backend.domain.agents.runtime import (
    AI_HEDGE_FUND_RUNTIME_TYPE,
    AgentRuntimeContext,
    normalize_runtime_config,
)
from dashboard.backend.infrastructure.llm.validator import (
    MAX_ORDER_SHARES,
    actions_to_executable,
    parse_actions_payload,
)
from dashboard.backend.paths import REPO_ROOT

DEFAULT_MODEL_NAME = "gpt-4.1"
DEFAULT_MODEL_PROVIDER = "OpenAI"
DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_DECISION_INTERVAL = "daily"
_LOCAL_RUNTIME_DIR = REPO_ROOT / ".ai-hedge-fund-venv"

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
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
    }
)


class AiHedgeFundRuntimeError(RuntimeError):
    """Base exception for runtime configuration or execution failures."""


class AiHedgeFundOutputError(AiHedgeFundRuntimeError):
    """Raised when upstream returns data that cannot safely become ATL orders."""


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


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
            raise AiHedgeFundRuntimeError(
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
            raise AiHedgeFundRuntimeError(
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
        self.model_name = (
            self.environment.get("AI_HEDGE_FUND_MODEL_NAME", "").strip()
            or DEFAULT_MODEL_NAME
        )
        if len(self.model_name) > 100:
            raise AiHedgeFundRuntimeError(
                "AI_HEDGE_FUND_MODEL_NAME must be at most 100 characters"
            )
        self.lookback_days = self._platform_int(
            "AI_HEDGE_FUND_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS, 1, 3650
        )
        self.timeout_seconds = self._platform_int(
            "AI_HEDGE_FUND_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS, 1, 900
        )
        self.calls = 0
        self._last_decision_day: Optional[str] = None

    def _platform_int(
        self, name: str, default: int, minimum: int, maximum: int
    ) -> int:
        raw = self.environment.get(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise AiHedgeFundRuntimeError(f"{name} must be an integer") from exc
        if not minimum <= value <= maximum:
            raise AiHedgeFundRuntimeError(
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
    ) -> tuple[str, int, float, str]:
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
            raise AiHedgeFundOutputError(
                f"Decision for {symbol} has invalid quantity"
            )
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

    @classmethod
    def output_to_atl_actions(
        cls,
        output: Dict[str, Any],
        context: AgentRuntimeContext,
    ) -> list[Dict[str, Any]]:
        decisions = output.get("decisions") if isinstance(output, dict) else None
        if not isinstance(decisions, dict):
            raise AiHedgeFundOutputError(
                'AI Hedge Fund output must include a "decisions" object'
            )

        allowed = {str(symbol).strip().upper() for symbol in context.symbols}
        normalized = []
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
            # ATL's MVP is long-only. These valid upstream recommendations are
            # explicit holds here, so no short-side order reaches validation or
            # execution.
            if action in {"hold", "short", "cover"}:
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
        for action in actions:
            reason = str(action.get("reason") or "")
            action["reason"] = reason.replace(
                "[External]", "[AI Hedge Fund]", 1
            )
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

        output = self.runner.run(
            self._upstream_payload(context),
            timeout_seconds=self.timeout_seconds,
        )
        self.calls += 1
        actions = self.output_to_atl_actions(output, context)
        self._last_decision_day = decision_day
        return {"actions": actions}
