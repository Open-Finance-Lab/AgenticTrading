#!/usr/bin/env python3
"""Run local TradingAgents decisions through an ATL backtest.

TradingAgents and its model/data credentials stay on this machine. ATL receives
only the normalized decisions needed for simulation and result tracking.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from agentictrading import ATLAPIError, ATLClient
from agentictrading.integrations.tradingagents import (
    ArtifactValidationError,
    TradingAgentsATLRunner,
    TradingAgentsATLRunOutcome,
    TradingAgentsDecisionGenerator,
    TradingAgentsDependencyError,
    TradingAgentsGenerationError,
    TradingAgentsReplayIncompleteError,
    TradingAgentsReplayValidationError,
    TradingAgentsVersionError,
    default_decision_artifact_path,
    load_decision_artifact,
    sanitize_error_message,
    save_decision_artifact,
)

DEFAULT_ANALYSTS = ("market", "social", "news", "fundamentals")

# The us-equity-hourly-v1 environment fixes starting capital at $1,000 and caps
# any single position at 25% of equity, and the SDK rejects any other initial
# cash. One share therefore has to cost under ~$250 for a BUY to be executable
# at all, which rules out roughly a third of the DJIA-30 universe.
ENVIRONMENT_POSITION_CAP_NOTE = (
    "The us-equity-hourly-v1 environment starts with $1,000 and caps one "
    "position at 25% of equity, so a share priced above ~$250 can never be "
    "bought; see the per-step rationale for the exact price and budget."
)


class CLIConfigurationError(ValueError):
    """Raised when command arguments or required environment values are absent."""


@dataclass(frozen=True)
class ATLSettings:
    api_key: str
    base_url: str
    agent_version_id: str


@dataclass(frozen=True)
class CommandResult:
    artifact_path: Path
    artifact_sha256: str
    outcome: TradingAgentsATLRunOutcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate local TradingAgents decisions and replay them in ATL, "
            "or replay an existing decision artifact without new LLM calls."
        )
    )
    parser.add_argument("--symbol", required=True, help="One ATL-supported US stock")
    parser.add_argument(
        "--analysis-date",
        dest="analysis_dates",
        action="append",
        default=[],
        metavar="YYYY-MM-DD",
        help="TradingAgents analysis date; repeat for each desired date",
    )
    parser.add_argument(
        "--decisions-file",
        type=Path,
        help="Replay an existing artifact and skip TradingAgents entirely",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Artifact output path when generating decisions",
    )
    parser.add_argument("--start-date", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--llm-provider", help="TradingAgents LLM provider override")
    parser.add_argument("--deep-think-llm", help="TradingAgents deep model override")
    parser.add_argument("--quick-think-llm", help="TradingAgents quick model override")
    parser.add_argument("--temperature", type=float, help="Optional model temperature")
    parser.add_argument(
        "--selected-analyst",
        dest="selected_analysts",
        action="append",
        choices=DEFAULT_ANALYSTS,
        help="Analyst to enable; repeat as needed (default: all four)",
    )
    parser.add_argument(
        "--base-url",
        help="ATL API URL; defaults to ATL_BASE_URL",
    )
    parser.add_argument(
        "--agent-version-id",
        help="Existing ATL AgentVersion id; defaults to ATL_AGENT_VERSION_ID",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--max-wait-seconds", type=float, default=300.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.decisions_file and args.analysis_dates:
        raise CLIConfigurationError(
            "--decisions-file cannot be combined with --analysis-date"
        )
    if not args.decisions_file and not args.analysis_dates:
        raise CLIConfigurationError(
            "at least one --analysis-date is required when generating decisions"
        )
    if args.decisions_file and args.output:
        raise CLIConfigurationError(
            "--output is only valid while generating decisions"
        )
    if args.poll_interval <= 0:
        raise CLIConfigurationError("--poll-interval must be greater than 0")
    if args.max_wait_seconds is not None and args.max_wait_seconds < 0:
        raise CLIConfigurationError("--max-wait-seconds cannot be negative")


def resolve_atl_settings(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str],
) -> ATLSettings:
    api_key = str(environ.get("ATL_API_KEY") or "").strip()
    if not api_key:
        raise CLIConfigurationError("ATL_API_KEY is required")
    base_url = str(args.base_url or environ.get("ATL_BASE_URL") or "").strip()
    if not base_url:
        raise CLIConfigurationError("ATL_BASE_URL or --base-url is required")
    agent_version_id = str(
        args.agent_version_id or environ.get("ATL_AGENT_VERSION_ID") or ""
    ).strip()
    if not agent_version_id:
        raise CLIConfigurationError(
            "ATL_AGENT_VERSION_ID or --agent-version-id is required"
        )
    return ATLSettings(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        agent_version_id=agent_version_id,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_GENERATION_CONFIG_KEYS = (
    "llm_provider",
    "deep_think_llm",
    "quick_think_llm",
    "temperature",
)


def _generation_config(args: argparse.Namespace) -> dict:
    """Collect the TradingAgents overrides the caller actually supplied.

    The argparse dest names are the upstream config keys, so there is nothing
    to translate — only omitted (``None``) values to drop.
    """
    return {
        key: getattr(args, key)
        for key in _GENERATION_CONFIG_KEYS
        if getattr(args, key) is not None
    }


def run_from_args(
    args: argparse.Namespace,
    *,
    environ: Optional[Mapping[str, str]] = None,
    client_factory: Callable[..., Any] = ATLClient,
    runner_factory: Callable[[Any], Any] = TradingAgentsATLRunner,
    generator_factory: Callable[[], Any] = TradingAgentsDecisionGenerator,
) -> CommandResult:
    validate_args(args)
    settings = resolve_atl_settings(args, environ=environ or os.environ)
    symbol = str(args.symbol).strip().upper()
    if not symbol:
        raise CLIConfigurationError("--symbol cannot be empty")
    client = client_factory(
        base_url=settings.base_url,
        api_key=settings.api_key,
    )
    runner = runner_factory(client)
    runner.validate_symbol(symbol)

    if args.decisions_file:
        artifact_path = Path(args.decisions_file).expanduser()
        artifact = load_decision_artifact(artifact_path)
        artifact_sha256 = _file_sha256(artifact_path)
    else:
        artifact = generator_factory().generate(
            symbol=symbol,
            analysis_dates=tuple(args.analysis_dates),
            config=_generation_config(args),
            selected_analysts=tuple(args.selected_analysts or DEFAULT_ANALYSTS),
        )
        artifact_path = Path(
            args.output or default_decision_artifact_path(symbol)
        ).expanduser()
        artifact_sha256 = save_decision_artifact(artifact, artifact_path)

    artifact_symbol = str(artifact.manifest.get("symbol") or "").upper()
    if artifact_symbol != symbol:
        raise CLIConfigurationError(
            f"--symbol {symbol} does not match artifact symbol {artifact_symbol}"
        )

    outcome = runner.run_backtest(
        artifact=artifact,
        artifact_sha256=artifact_sha256,
        agent_version_id=settings.agent_version_id,
        start_date=args.start_date,
        end_date=args.end_date,
        poll_interval=args.poll_interval,
        max_wait_seconds=args.max_wait_seconds,
    )
    return CommandResult(
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        outcome=outcome,
    )


def warn_about_unexecutable_signals(outcome: TradingAgentsATLRunOutcome) -> None:
    """Report when a flat result came from arithmetic rather than the model.

    A run whose every BUY was priced out finishes green at 0% and looks exactly
    like a strategy that chose to stay in cash, so the difference has to be
    stated rather than left in the per-step rationale.
    """
    replay = outcome.replay
    if replay.price_too_high_holds and not replay.buy_orders:
        print(
            f"WARNING: all {replay.price_too_high_holds} BUY signal(s) were "
            "dropped because one share costs more than the position cap "
            f"allows. {ENVIRONMENT_POSITION_CAP_NOTE} This run's return "
            "reflects that limit, not the TradingAgents strategy.",
            file=sys.stderr,
        )
    elif replay.price_too_high_holds:
        print(
            f"NOTE: {replay.price_too_high_holds} BUY signal(s) were priced "
            f"out of the position cap. {ENVIRONMENT_POSITION_CAP_NOTE}",
            file=sys.stderr,
        )
    if replay.autoheld_steps:
        print(
            f"NOTE: ATL auto-held {replay.autoheld_steps} step(s) before "
            "accepting a decision. Those records were not consumed; they "
            "executed on a later step or are listed as unprocessed above.",
            file=sys.stderr,
        )


def print_summary(result: CommandResult) -> None:
    outcome = result.outcome
    replay = outcome.replay
    print(f"Decision artifact: {result.artifact_path}")
    print(f"Artifact SHA-256: {result.artifact_sha256}")
    print(f"ATL run: {outcome.run_id}")
    if outcome.compare_url:
        print(f"Result: {outcome.compare_url}")
    print(f"Total return: {outcome.result.metrics.get('total_return')}")
    print(
        "Replay: "
        f"buy={replay.buy_orders}, sell={replay.sell_orders}, "
        f"model_hold={replay.model_holds}, error_hold={replay.error_holds}, "
        f"passive_hold={replay.passive_holds}, "
        f"constraint_hold={replay.constraint_holds}, "
        f"price_too_high={replay.price_too_high_holds}, "
        f"superseded={replay.superseded}, "
        f"unprocessed={len(replay.unprocessed_dates)}"
    )
    print(
        f"Execution: fills={outcome.fills_count}, "
        f"rejections={len(outcome.rejections)}, "
        f"timeout_holds={outcome.timeout_holds}, "
        f"autoheld_steps={outcome.autoheld_steps}"
    )
    warn_about_unexecutable_signals(outcome)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_from_args(args)
    except TradingAgentsReplayIncompleteError as exc:
        print(f"ERROR: {sanitize_error_message(exc)}", file=sys.stderr)
        print(f"Incomplete ATL run id: {exc.run_id}", file=sys.stderr)
        return 2
    except (
        CLIConfigurationError,
        ArtifactValidationError,
        TradingAgentsDependencyError,
        TradingAgentsGenerationError,
        TradingAgentsReplayValidationError,
        TradingAgentsVersionError,
        ATLAPIError,
    ) as exc:
        print(f"ERROR: {sanitize_error_message(exc)}", file=sys.stderr)
        run_id = getattr(exc, "run_id", None)
        if run_id:
            # The run is still open server-side until the reaper sweeps it, so
            # name it rather than making the operator hunt for the orphan.
            print(f"Aborted ATL run id: {run_id}", file=sys.stderr)
        return 1
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
