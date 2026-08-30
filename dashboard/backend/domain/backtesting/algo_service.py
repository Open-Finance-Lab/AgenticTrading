"""
My Trading Algo service: real LLM chat + Alpaca hourly backtest execution.

Canonical location (Phase 3C1). Moved verbatim from
``dashboard/backend/algo_service.py``, which is now a thin compatibility
re-export shim. Public functions, constants, the ``algo_status`` registry,
signatures, return schemas, exceptions, logging, file handling, result
serialization, and backtest orchestration are unchanged.

The only mechanical adjustment is that paths previously derived from
``Path(__file__).resolve().parent.parent`` (the ``dashboard/`` directory when this
module lived at ``dashboard/backend/algo_service.py``) are now anchored to the
same ``dashboard/`` directory via ``_PROJECT_DIR``, so every resolved data,
config, scripts, credentials, and venv path is byte-for-byte identical.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Resolves to the ``dashboard/`` directory — identical to the previous
# ``Path(__file__).resolve().parent.parent`` when this module lived at
# ``dashboard/backend/algo_service.py``.
_PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = _PROJECT_DIR / "data"
SUBMISSIONS_FILE = DATA_DIR / "algo_submissions.json"
CONFIG_DIR = DATA_DIR / "algo_configs"
DEFAULTS_FILE = _PROJECT_DIR / "config" / "defaults.json"

DEFAULT_BLOCKS: dict[str, str] = {
    "info_retrieval": "Monitor Trump's Twitter / X feed; capture tweets and sentiment signals",
    "signal_transfer": "AI auto-selects target stocks (single name or basket); map tickers from tweet semantics",
    "trading_algorithm": "No execution algo: buy whatever Trump mentions (immediate market follow)",
    "stop_loss_take_profit": "Stop loss: exit if position down 5%; take profit: hold after +20%; daily stop: exit if down 5% intraday",
}

BLOCK_LABELS = {
    "info_retrieval": "Info Retrieval",
    "signal_transfer": "Signal Transfer",
    "trading_algorithm": "Trading Algorithm",
    "stop_loss_take_profit": "Stop Loss / Take Profit",
}

USER_ALGO_COLORS = ["#fbbf24", "#34d399", "#f472b6", "#818cf8", "#fb7185", "#2dd4bf"]

LLM_MODEL = "claude-haiku-4-5-20251001"

algo_status: dict[str, Any] = {
    "running": False,
    "submission_id": None,
    "session_id": None,
    "team_name": None,
    "progress": "",
    "error": None,
    "result": None,
}


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_submissions() -> list[dict[str, Any]]:
    _ensure_data_dir()
    if not SUBMISSIONS_FILE.exists():
        return []
    try:
        with open(SUBMISSIONS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def get_default_blocks() -> dict[str, str]:
    return deepcopy(DEFAULT_BLOCKS)


def _report_unusable_defaults(reason: str) -> None:
    """Announce that the settings file is present but yielded no date range.

    A bare `pass` made "the file is absent" and "the file is on disk but
    unusable" produce byte-identical behaviour: both fall through to the
    hardcoded pair, silently running every backtest over the wrong window.
    ``reason`` separates the ways that happens, so the notice says which fault
    occurred rather than only that one did.

    print(), not logging -- under the deployed uvicorn, logger calls emit
    nothing.
    """
    print(
        f"[algo_service] {DEFAULTS_FILE} {reason}; "
        "falling back to the built-in default date range"
    )


def _default_backtest_dates() -> tuple[str, str]:
    if DEFAULTS_FILE.exists():
        try:
            cfg = json.loads(DEFAULTS_FILE.read_text(encoding="utf-8"))
            settings = cfg.get("defaultSettings", {})
            start = settings.get("startDate")
            end = settings.get("endDate")
            if start and end:
                return start, end
            # Parsed, right shape, no range: an upstream key rename or a
            # blanked value. Likelier than a corrupt write, and a bare
            # fall-through hid it just as completely.
            _report_unusable_defaults(
                "has no usable defaultSettings.startDate/endDate"
            )
        except (json.JSONDecodeError, OSError, AttributeError) as exc:
            # AttributeError included deliberately: valid JSON that is not an
            # object (`[]`, a bare string) makes `.get` raise, which is neither
            # a decode nor an OS error, so it used to escape this handler and
            # 500 the caller. The try block is three lines wide, so this cannot
            # swallow an unrelated attribute bug.
            _report_unusable_defaults(f"could not be read as a settings object ({exc})")
    return "2026-05-04", "2026-05-12"


def _resolve_python_exe() -> str:
    project_dir = _PROJECT_DIR
    venv_dir = project_dir / ".venv"
    if venv_dir.exists():
        win_py = venv_dir / "Scripts" / "python.exe"
        if win_py.exists():
            return str(win_py)
        unix_py = venv_dir / "bin" / "python3"
        if unix_py.exists():
            return str(unix_py)
    return sys.executable


def _get_anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)
    except ImportError:
        return None


def _parse_chat_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start < 0:
        raise ValueError("LLM did not return JSON")
    return json.loads(cleaned[start:end])


def process_chat(message: str, blocks: dict[str, str] | None = None) -> dict[str, Any]:
    """Real LLM chat: update strategy blocks from natural language."""
    current = deepcopy(blocks or DEFAULT_BLOCKS)
    client = _get_anthropic_client()

    if not client:
        return _process_chat_fallback(message, current)

    system = f"""You are a trading strategy assistant for Agentic Trading Lab.
The user configures a 4-block pipeline. Update blocks based on their message.

Current blocks:
{json.dumps(current, ensure_ascii=False, indent=2)}

Rules:
- Always respond in English (including block text and reply).
- When user asks to change a source (e.g. Trump -> Elon Musk), update ALL affected blocks consistently.
- Be honest: naive strategies (blind copy-trading without risk control) often lose money in backtests.
- Return ONLY valid JSON, no markdown fences.

JSON schema:
{{
  "reply": "conversational reply to user",
  "blocks": {{
    "info_retrieval": "...",
    "signal_transfer": "...",
    "trading_algorithm": "...",
    "stop_loss_take_profit": "..."
  }},
  "updated_blocks": ["info_retrieval", ...]
}}"""

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=1500,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": message.strip()}],
        )
        parsed = _parse_chat_json(response.content[0].text)
        new_blocks = parsed.get("blocks", current)
        for key in BLOCK_LABELS:
            if key in new_blocks and isinstance(new_blocks[key], str):
                current[key] = new_blocks[key].strip()
        updated = parsed.get("updated_blocks") or [
            k for k in BLOCK_LABELS if current.get(k) != (blocks or DEFAULT_BLOCKS).get(k)
        ]
        return {
            "reply": parsed.get("reply", "Strategy modules updated."),
            "blocks": current,
            "updated_blocks": updated,
        }
    except Exception as exc:
        print(f"LLM chat error: {exc!r}")
        fallback = _process_chat_fallback(message, current)
        fallback["reply"] += "\n\n(LLM unavailable — used rule-based fallback.)"
        return fallback


def _process_chat_fallback(message: str, current: dict[str, str]) -> dict[str, Any]:
    """Minimal fallback when API key missing."""
    lower = message.lower()
    updated: list[str] = []
    if any(k in lower for k in ("musk", "elon", "trump", "twitter", "x.com")):
        if "musk" in lower or "elon" in lower:
            current["info_retrieval"] = "Monitor Elon Musk's Twitter / X feed; capture tweets and sentiment signals"
            current["trading_algorithm"] = "No execution algo: buy whatever Musk mentions (immediate market follow)"
        else:
            current["info_retrieval"] = "Monitor Trump's Twitter / X feed; capture tweets and sentiment signals"
            current["trading_algorithm"] = "No execution algo: buy whatever Trump mentions (immediate market follow)"
        updated.extend(["info_retrieval", "trading_algorithm"])
    return {"reply": "Strategy updated (rule-based fallback).", "blocks": current, "updated_blocks": updated}


def _default_team_name(blocks: dict[str, str]) -> str:
    text = blocks.get("info_retrieval", "").lower()
    if "musk" in text:
        return "Elon Musk Twitter Algo"
    if "trump" in text:
        return "Trump Twitter Algo"
    return "My Trading Algo"


def get_algo_status(session_id: str) -> dict[str, Any]:
    if algo_status.get("session_id") != session_id:
        return {"running": False, "message": "No active job for this session"}
    out = dict(algo_status)
    out.pop("session_id", None)
    return out


def _run_backtest_background(
    config_path: Path,
    session_id: str,
    team_name: str,
    submission_id: str,
    start_date: str,
    end_date: str,
) -> None:
    global algo_status
    project_dir = _PROJECT_DIR
    script = project_dir / "scripts" / "backtest_custom_algo.py"
    python_exe = _resolve_python_exe()
    env = os.environ.copy()

    algo_status["progress"] = "Fetching hourly bars from Alpaca…"

    try:
        proc = subprocess.run(
            [
                python_exe,
                str(script),
                "--config", str(config_path),
                "--session-id", session_id,
                "--team-name", team_name,
                "--submission-id", submission_id,
                "--start", start_date,
                "--end", end_date,
            ],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
        )

        print("=== CUSTOM ALGO BACKTEST ===")
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)

        result_path = config_path.with_suffix(".result.json")
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-800:]
            algo_status["error"] = f"Backtest failed (code {proc.returncode}): {tail}"
            return

        if not result_path.exists():
            algo_status["error"] = "Backtest finished but result file not found"
            return

        result = json.loads(result_path.read_text(encoding="utf-8"))
        algo_status["result"] = {
            "submission_id": result["submission_id"],
            "run_id": result.get("run_id"),
            "team_name": result["team_name"],
            "metrics": {
                "portfolio_value": result["portfolio_value"],
                "cumulative_return": result["cumulative_return"],
                "sharpe_ratio": result["sharpe_ratio"],
                "win_loss_ratio": result["win_loss_ratio"],
                "max_drawdown": result["max_drawdown"],
            },
            "num_trades": result.get("num_trades", 0),
            "llm_calls": result.get("llm_calls", 0),
            "start_date": result.get("start_date"),
            "end_date": result.get("end_date"),
            "days": result.get("days"),
            "equity_curve": result.get("equity_curve"),
            "color": result.get("color"),
            "message": (
                f"Real backtest complete ({result.get('start_date')} → {result.get('end_date')}, "
                f"{result.get('llm_calls', 0)} LLM decisions, {result.get('num_trades', 0)} trades). "
                f"Return {result['cumulative_return'] * 100:.2f}%."
            ),
        }
        algo_status["progress"] = "Done"
    except subprocess.TimeoutExpired:
        algo_status["error"] = "Backtest timed out (over 60 minutes)"
    except Exception as exc:
        algo_status["error"] = str(exc)
    finally:
        algo_status["running"] = False


def _has_alpaca_credentials() -> bool:
    if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
        return True
    creds_path = _PROJECT_DIR / "credentials" / "alpaca.json"
    return creds_path.exists()


def execute_algo(
    blocks: dict[str, str],
    session_id: str,
    team_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Start async real backtest. Returns immediately with job id."""
    global algo_status

    if algo_status["running"] and algo_status.get("session_id") == session_id:
        raise RuntimeError("A backtest is already running for this session. Please wait.")

    if _blocks_match_defaults(blocks):
        raise RuntimeError(
            "Edit the four strategy blocks (or use chat) before Execute. "
            "The built-in example and Leaderboard mock teams do not trigger a real backtest."
        )

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required for real LLM backtests")

    if not _has_alpaca_credentials():
        raise RuntimeError(
            "Alpaca credentials are required for market data. "
            "Set ALPACA_API_KEY / ALPACA_SECRET_KEY or create credentials/alpaca.json"
        )

    # Resolved once, not once per bound: the resolver now prints on a bad
    # settings file, and calling it twice reported the same fault twice.
    default_start, default_end = _default_backtest_dates()
    start, end = start_date or default_start, end_date or default_end
    name = (team_name or _default_team_name(blocks)).strip() or "My Trading Algo"
    submission_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    _ensure_data_dir()
    config_path = CONFIG_DIR / f"{submission_id}.json"
    config_path.write_text(
        json.dumps(
            {"blocks": deepcopy(blocks), "start_date": start, "end_date": end, "team_name": name},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    algo_status = {
        "running": True,
        "submission_id": submission_id,
        "session_id": session_id,
        "team_name": name,
        "progress": "Submitted; starting real backtest (Alpaca + LLM)…",
        "error": None,
        "result": None,
    }

    thread = threading.Thread(
        target=_run_backtest_background,
        args=(config_path, session_id, name, submission_id, start, end),
        daemon=True,
    )
    thread.start()

    return {
        "submission_id": submission_id,
        "team_name": name,
        "status": "running",
        "start_date": start,
        "end_date": end,
        "message": (
            f"Real backtest started ({start} → {end}). "
            "Uses Alpaca hourly bars + Claude hourly decisions; multi-step runs can take several minutes."
        ),
    }


def _blocks_match_defaults(blocks: dict[str, str]) -> bool:
    defaults = get_default_blocks()
    return all(blocks.get(k, "").strip() == defaults.get(k, "").strip() for k in defaults)


def get_real_submissions() -> list[dict[str, Any]]:
    """Only user-executed algos that completed a real Alpaca + LLM backtest."""
    return [s for s in _load_submissions() if s.get("data_source") == "real_backtest"]


def get_all_submissions() -> list[dict[str, Any]]:
    return get_real_submissions()


def get_submissions_for_session(session_id: str) -> list[dict[str, Any]]:
    return [s for s in get_real_submissions() if s.get("session_id") == session_id]
