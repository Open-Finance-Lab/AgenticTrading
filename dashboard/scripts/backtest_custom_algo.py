#!/usr/bin/env python3
"""
Run a custom 4-block trading algo backtest on real Alpaca hourly data + LLM decisions.

Usage:
    python scripts/backtest_custom_algo.py \\
        --config data/algo_run_config.json \\
        --session-id <uuid> \\
        --team-name "Trump Twitter Algo"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Direct-execution bootstrap: make the repo root importable so canonical
# `dashboard.backend.*` imports resolve (no-op when run as part of the package).
from _bootstrap import ensure_repo_root

ensure_repo_root()

from dashboard.backend.infrastructure.llm.prompts import create_custom_algo_prompt, parse_risk_rules
from dashboard.backend.database import db

from backtest_hourly_agent import (  # noqa: E402
    DJIA_30,
    INITIAL_CAPITAL,
    LLM_MODEL_NAME,
    PortfolioManager,
    HourlyBacktester,
    HAS_ANTHROPIC,
)


class CustomAlgoPortfolioManager(PortfolioManager):
    """Portfolio manager with user strategy blocks + enforced stop-loss rules."""

    def __init__(self, initial_capital: float, strategy_blocks: dict[str, str]):
        super().__init__(initial_capital)
        self.strategy_blocks = strategy_blocks
        self.risk_rules = parse_risk_rules(strategy_blocks.get("stop_loss_take_profit", ""))
        self.day_start_equity: dict = {}

    def apply_risk_exits(self, portfolio_state: dict, market_data: dict, timestamp) -> List[dict]:
        """Force exits when stop-loss / daily stop rules are hit."""
        actions: List[dict] = []
        ts = pd.Timestamp(timestamp)
        day_key = ts.date()

        if day_key not in self.day_start_equity:
            self.day_start_equity[day_key] = portfolio_state["total_equity"]

        day_equity = portfolio_state["total_equity"]
        day_start = self.day_start_equity[day_key]
        if day_start > 0:
            day_dd_pct = ((day_equity - day_start) / day_start) * 100
            if day_dd_pct <= -self.risk_rules["daily_stop_pct"]:
                for pos in portfolio_state["positions"]:
                    if pos["symbol"] in self.positions and self.positions[pos["symbol"]] > 0:
                        actions.append({
                            "symbol": pos["symbol"],
                            "action": "sell",
                            "shares": self.positions[pos["symbol"]],
                            "reason": f"Daily stop: portfolio down {day_dd_pct:.1f}%",
                        })
                return actions

        for pos in portfolio_state["positions"]:
            symbol = pos["symbol"]
            pnl = pos["pnl_pct"]
            if pnl <= -self.risk_rules["stop_loss_pct"]:
                actions.append({
                    "symbol": symbol,
                    "action": "sell",
                    "shares": self.positions.get(symbol, 0),
                    "reason": f"Stop-loss: position down {pnl:.1f}%",
                })

        return actions

    def make_trading_decision_with_custom_algo(
        self,
        portfolio_state: dict,
        llm_client,
        mode: str = "custom_algo",
    ) -> dict:
        if not HAS_ANTHROPIC or not llm_client:
            return self.make_trading_decision(portfolio_state)

        import json as json_lib
        from backtest_hourly_agent import timedelta

        timestamp = portfolio_state.get("timestamp", datetime.now())
        timestamp_str = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)

        holdings = {}
        for position in portfolio_state["positions"]:
            holdings[position["symbol"]] = {
                "shares": position["shares"],
                "entry_price": round(position["entry_price"], 2),
                "current_price": round(position["current_price"], 2),
                "pnl_pct": round(position["pnl_pct"], 2),
            }

        recent_trades = []
        cutoff_time = timestamp - timedelta(hours=24)
        for trade in self.trades:
            if trade["timestamp"] > cutoff_time:
                recent_trades.append({
                    "symbol": trade["symbol"],
                    "side": trade["side"],
                    "shares": trade["shares"],
                    "price": round(float(trade["price"]), 2),
                })

        market_snapshot = {
            "timestamp": timestamp_str,
            "strategy_blocks": self.strategy_blocks,
            "portfolio": {
                "cash": round(portfolio_state["cash"], 2),
                "total_equity": round(portfolio_state["total_equity"], 2),
                "num_positions": len(portfolio_state["positions"]),
            },
            "current_holdings": holdings,
            "recent_trades": recent_trades,
            "top_signals": {},
        }

        signals = portfolio_state["market_signals"]
        rsi_sorted = sorted(
            [(sym, sig.get("rsi", 50)) for sym, sig in signals.items()],
            key=lambda x: abs(x[1] - 50),
            reverse=True,
        )
        for symbol, _ in rsi_sorted[:12]:
            signal = signals[symbol]
            market_snapshot["top_signals"][symbol] = {
                "price": float(signal.get("price") or 0),
                "rsi": float(signal.get("rsi") or 50),
                "sma20": float(signal.get("sma20") or 0),
                "sma50": float(signal.get("sma50") or 0),
            }

        prompt = create_custom_algo_prompt(market_snapshot, self.strategy_blocks)

        try:
            response = llm_client.messages.create(
                model=LLM_MODEL_NAME,
                max_tokens=2048,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
        except Exception as exc:
            print(f"   LLM error: {exc}, falling back to rules")
            return self.make_trading_decision(portfolio_state)

        from backtest_hourly_agent import fix_json_formatting

        try:
            cleaned = raw
            if "```" in cleaned:
                cleaned = cleaned.replace("```json", "").replace("```", "")
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start < 0:
                return self.make_trading_decision(portfolio_state)
            cleaned = fix_json_formatting(cleaned[start:end])
            parsed = json_lib.loads(cleaned)
        except Exception as exc:
            print(f"   JSON parse error: {exc}")
            return self.make_trading_decision(portfolio_state)

        actions = []
        signals = portfolio_state["market_signals"]

        for item in parsed.get("actions", [])[:8]:
            symbol = item.get("symbol")
            action_type = str(item.get("action", "hold")).lower()
            confidence = float(item.get("confidence", 0.5))
            reasoning = str(item.get("reasoning", ""))[:200]

            if confidence < 0.3 or action_type == "hold" or symbol not in DJIA_30:
                continue

            signal = signals.get(symbol, {})
            price = float(signal.get("price") or 0)
            if price <= 0:
                continue

            if action_type == "buy":
                shares = int(item.get("position_size") or 0)
                if shares <= 0:
                    risk_amount = portfolio_state["total_equity"] * 0.02 * confidence
                    shares = int(risk_amount / price)
                cost = shares * price
                if shares > 0 and cost <= portfolio_state["cash"]:
                    actions.append({
                        "symbol": symbol,
                        "action": "buy",
                        "shares": shares,
                        "reason": reasoning,
                    })
            elif action_type == "sell":
                shares = int(item.get("position_size") or self.positions.get(symbol, 0))
                if symbol in self.positions and shares > 0:
                    actions.append({
                        "symbol": symbol,
                        "action": "sell",
                        "shares": min(shares, self.positions[symbol]),
                        "reason": reasoning,
                    })

        return {"actions": actions}


class CustomAlgoBacktester(HourlyBacktester):
    """Hourly backtester driven by user 4-block strategy."""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        session_id: str,
        strategy_blocks: dict[str, str],
        team_name: str,
        submission_id: str,
    ):
        super().__init__(start_date, end_date, session_id, use_llm=True, mode="custom_algo")
        self.strategy_blocks = strategy_blocks
        self.team_name = team_name
        self.submission_id = submission_id

    def run_custom_algo_backtest(self) -> dict:
        print(f"Custom algo backtest: {self.team_name}")
        print(f"Period: {self.start_date} -> {self.end_date}")

        manager = CustomAlgoPortfolioManager(INITIAL_CAPITAL, self.strategy_blocks)
        llm_calls = 0
        llm_model = "rule-based"

        all_timestamps = set()
        for df in self.all_data.values():
            all_timestamps.update(df.index)
        all_timestamps = sorted(all_timestamps)

        import pytz

        et_tz = pytz.timezone("US/Eastern")
        market_hours = []
        for ts in all_timestamps:
            ts_et = ts.astimezone(et_tz)
            hour, minute = ts_et.hour, ts_et.minute
            is_open = (hour > 9 and hour < 16) or (hour == 9 and minute >= 30) or (hour == 16 and minute == 0)
            if ts_et.weekday() < 5 and is_open:
                market_hours.append(ts)

        all_timestamps = market_hours
        print(f"   {len(all_timestamps)} market hours to simulate")

        price_cache: Dict = {}
        for symbol, df in self.all_data.items():
            price_cache[symbol] = {}
            last_price = None
            for timestamp in all_timestamps:
                if timestamp in df.index:
                    last_price = df.loc[timestamp, "close"]
                if last_price is not None:
                    price_cache[symbol][timestamp] = last_price

        for i, timestamp in enumerate(all_timestamps):
            market_data = {}
            for symbol in DJIA_30:
                if symbol in self.all_data and timestamp in self.all_data[symbol].index:
                    market_data[symbol] = self.all_data[symbol].loc[timestamp]

            state = manager.get_portfolio_state(market_data, price_cache, timestamp)
            state["timestamp"] = timestamp

            risk_actions = manager.apply_risk_exits(state, market_data, timestamp)
            if risk_actions:
                manager.execute_actions(risk_actions, market_data, timestamp)
                state = manager.get_portfolio_state(market_data, price_cache, timestamp)
                state["timestamp"] = timestamp

            if self.use_llm and self.llm_client:
                decision = manager.make_trading_decision_with_custom_algo(state, self.llm_client)
                llm_calls += 1
                if llm_calls == 1:
                    llm_model = LLM_MODEL_NAME
            else:
                decision = manager.make_trading_decision(state)

            manager.execute_actions(decision["actions"], market_data, timestamp)
            manager.update_equity(market_data, price_cache, timestamp)

            if (i + 1) % 20 == 0 or i == len(all_timestamps) - 1:
                eq = manager.equity_history[-1]["equity"]
                ret = (eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                print(f"   Hour {i + 1}/{len(all_timestamps)}: ${eq:,.0f} ({ret:+.2f}%)")

        equity_curve = manager.get_equity_curve()
        for entry in equity_curve:
            if hasattr(entry["timestamp"], "isoformat"):
                entry["timestamp"] = entry["timestamp"].isoformat()

        daily_curve, daily_days = _aggregate_daily_equity(equity_curve)

        initial_eq = equity_curve[0]["equity"] if equity_curve else INITIAL_CAPITAL
        final_eq = equity_curve[-1]["equity"] if equity_curve else INITIAL_CAPITAL
        total_return = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL

        run_id = f"algo_{self.submission_id}"
        db.insert_run(
            run_id=run_id,
            session_id=self.session_id,
            agent_name=self.team_name,
            mode="backtest",
            start_date=self.start_date,
            end_date=self.end_date,
            initial_equity=initial_eq,
            final_equity=final_eq,
            total_return=total_return,
            sharpe_ratio=self._calc_sharpe(equity_curve),
            max_drawdown=self._calc_max_dd(equity_curve),
            num_trades=len(manager.trades),
            llm_model=llm_model,
        )
        db.insert_equity_points(run_id, equity_curve)

        wl = len([t for t in manager.trades if t.get("side") == "BUY"]) / max(1, len([t for t in manager.trades if t.get("side") == "SELL"]))

        result = {
            "submission_id": self.submission_id,
            "session_id": self.session_id,
            "team_name": self.team_name,
            "team_badge": "MY ALGO",
            "model": f"Custom Algo + {llm_model}",
            "blocks": deepcopy(self.strategy_blocks),
            "days": daily_days,
            "equity_curve": daily_curve,
            "portfolio_value": round(final_eq, 2),
            "cumulative_return": round(total_return, 4),
            "sharpe_ratio": round(self._calc_sharpe(equity_curve), 2),
            "win_loss_ratio": round(wl, 2),
            "max_drawdown": round(self._calc_max_dd(equity_curve), 4),
            "num_trades": len(manager.trades),
            "llm_calls": llm_calls,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": "Live",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "data_source": "real_backtest",
            "run_id": run_id,
        }
        return result


def _aggregate_daily_equity(hourly: list[dict]) -> tuple[list[float], list[str]]:
    """Last equity per calendar day for leaderboard chart."""
    by_day: dict[str, float] = {}
    for point in hourly:
        ts = point["timestamp"]
        if isinstance(ts, str):
            day = ts[:10]
        else:
            day = ts.strftime("%Y-%m-%d")
        by_day[day] = float(point["equity"])
    days = sorted(by_day.keys())
    return [by_day[d] for d in days], days


def _save_submission(result: dict) -> None:
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "algo_submissions.json"
    submissions = []
    if path.exists():
        try:
            submissions = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            submissions = []
    submissions = [s for s in submissions if s.get("submission_id") != result["submission_id"]]
    colors = ["#fbbf24", "#34d399", "#f472b6", "#818cf8", "#fb7185", "#2dd4bf"]
    result["color"] = colors[len(submissions) % len(colors)]
    submissions.append(result)
    path.write_text(json.dumps(submissions, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Custom 4-block algo backtest")
    parser.add_argument("--config", required=True, help="JSON config file path")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    blocks = config.get("blocks", {})
    start = args.start or config.get("start_date", "2026-05-04")
    end = args.end or config.get("end_date", "2026-05-12")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY required for custom algo backtest")
        return 1

    backtester = CustomAlgoBacktester(
        start, end, args.session_id, blocks, args.team_name, args.submission_id,
    )
    print("Loading Alpaca hourly data...")
    backtester.load_data()
    print("Calculating indicators...")
    backtester.calculate_indicators()

    result = backtester.run_custom_algo_backtest()
    _save_submission(result)

    out_path = Path(args.config).with_suffix(".result.json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. Return: {result['cumulative_return'] * 100:+.2f}%")
    print(f"Result saved: {out_path}")
    return 0


def _close_pools() -> None:
    """Close shared Postgres pools before this short-lived CLI exits."""
    pool_module = sys.modules.get("dashboard.backend.db_pool")
    if pool_module is not None:
        pool_module.close_all_pools()


if __name__ == "__main__":
    from dashboard.backend.infrastructure.market_data.alpaca_bars import (
        MarketDataUnavailableError,
    )

    try:
        try:
            sys.exit(main())
        except MarketDataUnavailableError as exc:
            # Library code raises; the CLI boundary owns the process exit.
            print(f"ERROR: {exc}")
            sys.exit(1)
    finally:
        _close_pools()
