#!/usr/bin/env python3
"""
Hourly Magnificent 7 backtest: Alpaca bars → LLM agent (CommonStack) → plot.
  pip install alpaca-py openai pandas pytz matplotlib requests
  export ALPACA_API_KEY=... ALPACA_SECRET_KEY=... COMMONSTACK_API_KEY=...
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import pandas as pd
import pytz
import requests

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
DJIA_INDEX = "^DJI"
NASDAQ_100_INDEX = "^NDX"
_YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

START, END = "2026-06-01", "2026-06-03"
INITIAL_CASH = 100_000.0
BASE_URL = os.getenv("COMMONSTACK_BASE_URL", "https://api.commonstack.ai/v1").rstrip("/")
MODEL = os.getenv("COMMONSTACK_MODEL", "deepseek/deepseek-v4-pro")

ET = pytz.timezone("US/Eastern")

SYSTEM_PROMPT = (
    "You are a Magnificent 7 hourly trading agent. Return ONLY valid JSON with an "
    '"actions" array. Each action: action (buy|sell|hold), symbol, '
    "confidence (0-1), reasoning, position_size (integer shares). "
    "Only sell symbols you own. Use symbols from the snapshot only."
)


def fetch_hourly_bars(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")

    bars = StockHistoricalDataClient(key, secret).get_stock_bars(
        StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Hour, start=start, end=end)
    ).df

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        if sym not in bars.index.get_level_values(0):
            continue
        df = bars.xs(sym)[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        out[sym] = df.sort_index()
    return out


def is_market_hour(ts: pd.Timestamp) -> bool:
    t = ts.astimezone(ET)
    if t.weekday() >= 5:
        return False
    m = t.hour * 60 + t.minute
    return 9 * 60 + 30 <= m <= 16 * 60


def market_timestamps(data: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
    return [ts for ts in sorted({ts for df in data.values() for ts in df.index}) if is_market_hour(ts)]


def fetch_index_hourly(symbol: str, start: str, end: str) -> List[Tuple[dt.datetime, float]]:
    """Hourly index levels from Yahoo Finance (^DJI, ^NDX, ...)."""
    start_e = int(dt.datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())
    end_e = int(dt.datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp()) + 86400

    resp = requests.get(
        _YAHOO_CHART.format(symbol=symbol),
        params={"period1": start_e, "period2": end_e, "interval": "1h"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    resp.raise_for_status()

    results = (resp.json().get("chart") or {}).get("result") or []
    if not results:
        return []

    res = results[0]
    timestamps = res.get("timestamp") or []
    closes = ((res.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []

    points: List[Tuple[dt.datetime, float]] = []
    for ts, close in zip(timestamps, closes):
        if close is None or not (start_e <= ts < end_e):
            continue
        points.append((dt.datetime.fromtimestamp(ts, dt.timezone.utc), float(close)))
    return points


def compute_index_baseline(index_symbol: str, timestamps: List[pd.Timestamp]) -> pd.Series:
    """Scale a real index level to INITIAL_CASH at the first aligned bar."""
    points = fetch_index_hourly(index_symbol, START, END)
    if not points:
        raise SystemExit(f"No Yahoo Finance data for {index_symbol}.")

    idx = pd.DatetimeIndex([p[0] for p in points], tz="UTC")
    levels = pd.Series([p[1] for p in points], index=idx).sort_index()
    levels = levels[[is_market_hour(ts) for ts in levels.index]]

    ts_idx = pd.DatetimeIndex(timestamps)
    if ts_idx.tz is None:
        ts_idx = ts_idx.tz_localize("UTC")
    else:
        ts_idx = ts_idx.tz_convert("UTC")

    aligned = levels.reindex(ts_idx, method="nearest", tolerance=pd.Timedelta("30min"))
    if aligned.isna().any():
        aligned = aligned.ffill().bfill()
    base = float(aligned.iloc[0])
    return pd.Series((INITIAL_CASH * (aligned / base)).values, index=pd.DatetimeIndex(timestamps))


@dataclass
class Portfolio:
    cash: float = INITIAL_CASH
    positions: Dict[str, int] = field(default_factory=dict)
    trades: List[dict] = field(default_factory=list)
    value_curve: List[dict] = field(default_factory=list)

    def mark_value(self, prices: Dict[str, float], ts: pd.Timestamp) -> None:
        pos_val = sum(self.positions.get(s, 0) * prices.get(s, 0) for s in self.positions)
        self.value_curve.append({"timestamp": ts, "value": self.cash + pos_val})

    def execute(self, actions: List[dict], prices: Dict[str, float], ts: pd.Timestamp) -> None:
        for act in actions:
            sym = act.get("symbol")
            side = str(act.get("action", "hold")).lower()
            shares = int(act.get("position_size") or 0)
            if not sym or sym not in prices or shares <= 0:
                continue
            price = prices[sym]
            if side == "buy" and shares * price <= self.cash:
                self.cash -= shares * price
                self.positions[sym] = self.positions.get(sym, 0) + shares
                self.trades.append({"ts": ts, "symbol": sym, "side": "BUY", "shares": shares, "price": price})
            elif side == "sell" and self.positions.get(sym, 0) > 0:
                qty = min(shares, self.positions[sym])
                self.cash += qty * price
                self.positions[sym] -= qty
                if self.positions[sym] == 0:
                    del self.positions[sym]
                self.trades.append({"ts": ts, "symbol": sym, "side": "SELL", "shares": qty, "price": price})


def make_llm_client():
    key = os.getenv("COMMONSTACK_API_KEY")
    if not key:
        raise SystemExit("Set COMMONSTACK_API_KEY.")
    from openai import OpenAI

    url = BASE_URL if BASE_URL.endswith("/v1") else f"{BASE_URL}/v1"
    return OpenAI(api_key=key, base_url=url)


def ask_agent(client, snapshot: dict) -> List[dict]:
    prompt = (
        f"Manage this Magnificent 7 portfolio. Buy/sell/hold this hour.\n"
        f"- Use only snapshot symbols/prices; sell only owned symbols.\n"
        f"- position_size = integer shares; each buy < 10% of cash; max 5 actions.\n\n"
        f"Snapshot:\n{json.dumps(snapshot, indent=2)}\n\n"
        f'Return ONLY: {{"actions": [{{"action": "buy|sell|hold", "symbol": "AAPL", '
        f'"confidence": 0.7, "reasoning": "...", "position_size": 10}}]}}'
    )
    text = client.chat.completions.create(
        model=MODEL,
        max_tokens=800,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    ).choices[0].message.content or ""

    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}") + 1
    if start < 0 or end <= 0:
        return []
    try:
        data = json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        return []
    return [
        {"symbol": a.get("symbol"), "action": a.get("action", "hold"), "position_size": int(a.get("position_size") or 0)}
        for a in data.get("actions", [])
    ]


def _to_et(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(ET)


def gapless_market_axis(timestamps: List[pd.Timestamp]) -> tuple[list[float], list[pd.Timestamp]]:
    """Map real market datetimes to gapless matplotlib x coords (1 market hour = 1h wide)."""
    ts_et = [_to_et(pd.DatetimeIndex([t]))[0] for t in timestamps]
    if not ts_et:
        return [], []
    hour = 1.0 / 24.0
    origin = mdates.date2num(ts_et[0])
    x = [origin + i * hour for i in range(len(ts_et))]
    return x, ts_et


def plot_results(
    portfolio: Portfolio,
    baselines: Dict[str, pd.Series],
    path: str = "backtest_equity.png",
) -> None:
    if not portfolio.value_curve:
        return

    timestamps = [p["timestamp"] for p in portfolio.value_curve]
    x, ts_et = gapless_market_axis(timestamps)
    values = [p["value"] for p in portfolio.value_curve]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, values, label="Agent")
    for label, series in baselines.items():
        ax.plot(x, series.values, label=label, linestyle="--")
    ax.set_title("Trading Performance")
    ax.set_ylabel("Portfolio value ($)")
    ax.set_xlabel("Date & time (ET)")

    day_ticks, day_labels = [], []
    i = 0
    while i < len(ts_et):
        j = i
        while j < len(ts_et) and ts_et[j].date() == ts_et[i].date():
            j += 1
        day_ticks.append((x[i] + x[j - 1]) / 2)
        day_labels.append(ts_et[i].strftime("%Y-%m-%d"))
        i = j

    hour_labels = [t.strftime("%H:%M") for t in ts_et]
    ax.xaxis.set_major_locator(FixedLocator(day_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(day_labels))
    ax.xaxis.set_minor_locator(FixedLocator(x))
    ax.xaxis.set_minor_formatter(FixedFormatter(hour_labels))
    ax.tick_params(axis="x", which="major", length=6, pad=10)
    ax.tick_params(axis="x", which="minor", length=3, labelsize=7, pad=2)
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=0, ha="center")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot → {path}")
    plt.close(fig)


def run_backtest() -> dict:
    print(f"Fetching Magnificent 7 hourly bars ({START} → {END})...")
    agent_data = fetch_hourly_bars(MAG7, START, END)
    timestamps = market_timestamps(agent_data)
    print(f"Loaded {len(agent_data)} symbols, {len(timestamps)} market hours.")
    print(f"Fetching index baselines from Yahoo ({DJIA_INDEX}, {NASDAQ_100_INDEX})...")

    client = make_llm_client()
    portfolio = Portfolio()

    for i, ts in enumerate(timestamps, 1):
        prices = {sym: float(df.loc[ts, "close"]) for sym, df in agent_data.items() if ts in df.index}
        if not prices:
            continue
        snapshot = {
            "timestamp": ts.isoformat(),
            "cash": round(portfolio.cash, 2),
            "holdings": {s: {"shares": q, "price": prices[s]} for s, q in portfolio.positions.items() if s in prices},
            "prices": {s: round(p, 2) for s, p in sorted(prices.items())},
        }
        portfolio.execute(ask_agent(client, snapshot), prices, ts)
        portfolio.mark_value(prices, ts)
        if i % 10 == 0 or i == len(timestamps):
            val = portfolio.value_curve[-1]["value"]
            print(f"  step {i}/{len(timestamps)}  value ${val:,.0f}  trades {len(portfolio.trades)}")

    if not portfolio.value_curve:
        return {"return_pct": 0.0, "trades": 0}

    end_val = portfolio.value_curve[-1]["value"]
    result = {
        "initial_cash": INITIAL_CASH,
        "final_value": round(end_val, 2),
        "return_pct": round((end_val - INITIAL_CASH) / INITIAL_CASH * 100, 3),
        "trades": len(portfolio.trades),
    }
    print("\n=== Results ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    plot_results(
        portfolio,
        {
            "DJIA index": compute_index_baseline(DJIA_INDEX, timestamps),
            "Nasdaq-100": compute_index_baseline(NASDAQ_100_INDEX, timestamps),
        },
    )
    return result


if __name__ == "__main__":
    run_backtest()
