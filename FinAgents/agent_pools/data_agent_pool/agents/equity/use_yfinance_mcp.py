"""Example showing how to connect the local yfinance MCP server to an OpenAI agent.

Prerequisites
--------------
- Install the OpenAI Agents Python SDK (``pip install openai``) and any other
    dependencies listed in this repository.
- Ensure the environment variable ``OPENAI_API_KEY`` is set.
- The script assumes that ``yfinance_server.py`` lives in the same directory.

Run the example with::

        python use_yfinance_mcp.py --symbol AAPL --metric marketCap --period 6mo

Pass ``--start``/``--end`` to specify explicit date ranges. Set the
``YFINANCE_AGENT_PROMPT`` environment variable to fully override the request.
When the managed prompt is used, the MCP tool saves the CSV itself and the
agent replies with JSON describing what it completed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from agents.model_settings import ModelSettings

BASE_DIR = Path(__file__).parent
SERVER_SCRIPT = BASE_DIR / "yfinance_server.py"


def build_prompt(
    symbol: str,
    metric: str,
    period: str | None,
    start: str | None,
    end: str | None,
    output_path: str,
) -> str:
    if start or end:
        range_text = "from {}".format(start) if start else ""
        if end:
            range_text = f"{range_text} to {end}" if range_text else f"until {end}"
        history_args = {"symbol": symbol, "output_path": output_path}
        if start:
            history_args["start"] = start
        if end:
            history_args["end"] = end
    else:
        range_text = f"for the last {period}" if period else "for the default period"
        history_args = {
            "symbol": symbol,
            "period": period or "1mo",
            "output_path": output_path,
        }

    metric_args = {"symbol": symbol, "metric": metric}

    prompt_lines = [
        "You have access to yfinance tools via MCP.",
        f"1. Call get_historical_data with arguments: {json.dumps(history_args)}",
        f"2. Call get_stock_metric with arguments: {json.dumps(metric_args)}",
        "3. Respond with a single JSON object that includes the keys:",
        "   - csv_path (string) matching the path used in step 1",
        "   - metric_name (string) and metric_value (number or string)",
        "   - row_count (integer)",
        "   - tasks_completed (array of short strings describing what you did)",
        "4. Do not add markdown, commentary, or extra text—return JSON only.",
        f"The dataset must cover {symbol} {range_text}."
    ]

    return "\n".join(prompt_lines)


def read_csv_preview(csv_path: Path, max_lines: int = 5) -> list[str]:
    preview: list[str] = []
    try:
        with csv_path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                preview.append(line.rstrip("\n"))
                if idx + 1 >= max_lines:
                    break
    except FileNotFoundError:
        return []
    return preview


async def run_agent_with_yfinance_server(args: argparse.Namespace) -> None:
    """Launch the local MCP server over stdio and run a simple agent query."""
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "OPENAI_API_KEY must be set so the agent can call OpenAI models."
        )

    if not SERVER_SCRIPT.exists():
        raise FileNotFoundError(
            f"Couldn't locate yfinance server script at {SERVER_SCRIPT!s}."
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    range_fragment = (
        f"{(args.start or 'start').replace('-', '')}_{(args.end or 'end').replace('-', '')}"
        if args.start or args.end
        else (args.period or "1mo").replace(" ", "")
    )
    target_csv_path = output_dir / f"{args.symbol.lower()}_{args.metric.lower()}_{range_fragment}_{timestamp}.csv"

    # Let callers override the question via an environment variable for quick tinkering.
    prompt_override = os.environ.get("YFINANCE_AGENT_PROMPT")
    prompt = prompt_override or build_prompt(
        args.symbol,
        args.metric,
        args.period,
        args.start,
        args.end,
        str(target_csv_path),
    )

    async with MCPServerStdio(
        name="Local yfinance MCP server",
        params={
            "command": sys.executable,
            "args": [str(SERVER_SCRIPT)],
        },
        cache_tools_list=True,
    ) as server:
        agent = Agent(
            name="EquityDataAssistant",
            instructions=(
                "You can access yfinance data through the attached MCP server. "
                "Follow the user's checklist exactly, calling the requested tools "
                "before answering. Respond with JSON only."
            ),
            mcp_servers=[server],
            model_settings=ModelSettings(model="gpt-4.1-mini", tool_choice="required"),
        )

        result = await Runner.run(agent, prompt)
        final_output = result.final_output.strip()

        if prompt_override:
            print("\n=== Agent Output ===\n")
            print(final_output)
            return

        try:
            payload = json.loads(final_output)
        except json.JSONDecodeError as exc:
            raise ValueError("Agent response was not valid JSON.") from exc

        csv_path = Path(payload.get("csv_path", str(target_csv_path))).expanduser()
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV file reported by agent was not found: {csv_path}"
            )

        tasks_completed = payload.get("tasks_completed") or [
            "Fetched historical OHLCV data via get_historical_data",
            f"Retrieved {args.metric} metric via get_stock_metric",
            f"CSV saved to {csv_path}",
        ]
        if isinstance(tasks_completed, str):
            tasks_completed = [tasks_completed]

        metric_name = payload.get("metric_name", args.metric)
        metric_value = payload.get("metric_value")
        row_count = payload.get("row_count")

        print("\n=== Completed Tasks ===")
        for step in tasks_completed:
            print(f"- {step}")

        if metric_value is not None:
            print("\n=== Metric ===")
            print(f"{metric_name}: {metric_value}")

        print("\n=== CSV Location ===")
        print(csv_path)

        preview_lines = read_csv_preview(csv_path)
        if preview_lines:
            print("\n=== CSV Preview ===")
            print("\n".join(preview_lines))

        if row_count is not None:
            print(f"\nRows saved: {row_count}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query yfinance MCP data via an OpenAI agent.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol to query")
    parser.add_argument(
        "--metric",
        default="marketCap",
        help="Metric field name to include (e.g., marketCap, trailingPE)",
    )
    parser.add_argument(
        "--period",
        default="1mo",
        help="Relative period if no explicit start/end provided (e.g., 1mo, 6mo, 1y)",
    )
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Directory to store the generated CSV file",
    )

    args = parser.parse_args(argv)

    if (args.start and not args.start.strip()) or (args.end and not args.end.strip()):
        raise ValueError("Start and end dates must be non-empty strings when provided.")

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(run_agent_with_yfinance_server(args))


if __name__ == "__main__":
    main()
