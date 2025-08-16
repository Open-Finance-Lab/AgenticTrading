#!/usr/bin/env python3
import os
from pathlib import Path
import re
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(os.getenv("ALPHA_POOL_DATA_DIR", "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/data/cache"))
DEFAULT_CSV = os.getenv("ALPHA_POOL_DATA_CSV")
OUTPUT_DIR = DATA_DIR

DJIA_TICKERS = {
    "AAPL","MSFT","JNJ","JPM","WMT","KO","CAT","HON","HD","MMM","AMZN","GS","BAC",
    "MCD","MRK","MS","PEP","WFC","CVX","PG","DIS","V","INTC","NIKE"  # subset present in cache
}

def find_csv_for_symbol(symbol: str) -> Path | None:
    candidates = sorted(DATA_DIR.glob(f"{symbol}_*_1d.csv"))
    if candidates:
        return candidates[-1]
    return None


def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Infer columns
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("timestamp") or list(df.columns)[0]
    price_col = cols.get("close") or cols.get("adj_close") or list(df.columns)[-1]
    out = df[[date_col, price_col]].dropna().copy()
    out.columns = ["date", "close"]
    out["date"] = pd.to_datetime(out["date"])  # parse
    out.set_index("date", inplace=True)
    return out


def compute_momentum_factor(closes: pd.Series, lookback: int = 20) -> pd.Series:
    return closes.pct_change(lookback)


def to_signal_series(factor: pd.Series, cap: float = 1.0) -> pd.Series:
    # Normalize by rolling std to avoid lookahead; simple z-score with expanding std
    rolling = factor.rolling(60, min_periods=20)
    norm = (factor - rolling.mean()) / (rolling.std().replace(0, np.nan))
    sig = norm.clip(-cap, cap).fillna(0.0)
    return sig


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def load_index_proxy(tickers: set[str]) -> pd.Series:
    # Equal-weighted proxy from available tickers
    series_list = []
    names = []
    for t in tickers:
        p = find_csv_for_symbol(t)
        if p and p.exists():
            df = load_prices(p)
            ret = df["close"].pct_change()
            series_list.append(ret)
            names.append(t)
    if not series_list:
        return pd.Series(dtype=float)
    df = pd.concat(series_list, axis=1)
    df.columns = names
    bench = df.mean(axis=1)
    return bench


def main():
    # Select symbol and CSV
    symbol = os.getenv("ALPHA_POOL_SYMBOL", "AAPL")
    csv_path = Path(DEFAULT_CSV) if DEFAULT_CSV else find_csv_for_symbol(symbol)
    if not csv_path or not csv_path.exists():
        raise FileNotFoundError(f"CSV for {symbol} not found. Set ALPHA_POOL_DATA_CSV or place CSV in {DATA_DIR}")

    prices_df = load_prices(csv_path)
    closes = prices_df["close"]

    # Window selection by env (no hardcoding)
    start = os.getenv("ALPHA_POOL_START_DATE")
    end = os.getenv("ALPHA_POOL_END_DATE")
    if start:
        closes = closes[closes.index >= pd.to_datetime(start)]
    if end:
        closes = closes[closes.index <= pd.to_datetime(end)]

    # Strategy: momentum 20d
    lookback = int(os.getenv("ALPHA_POOL_LOOKBACK", "20"))
    factor = compute_momentum_factor(closes, lookback)
    signal = to_signal_series(factor)

    # Strategy returns: position at t uses signal at t-1 to avoid lookahead
    pos = signal.shift(1).fillna(0.0).clip(-1, 1)
    ret = closes.pct_change().fillna(0.0)
    strat_returns = pos * ret

    # Buy & Hold
    bnh_returns = ret.copy()

    # DJIA proxy
    djia_returns = load_index_proxy(DJIA_TICKERS)
    # Align
    index = strat_returns.index.intersection(bnh_returns.index)
    if not djia_returns.empty:
        index = index.intersection(djia_returns.index)
    strat_returns = strat_returns.loc[index]
    bnh_returns = bnh_returns.loc[index]
    if not djia_returns.empty:
        djia_returns = djia_returns.loc[index]

    # Metrics
    def sharpe(x: pd.Series):
        if x.std() == 0 or len(x) < 2:
            return 0.0
        return (x.mean() / x.std()) * np.sqrt(252)

    metrics = {
        "strategy_total_return": float(equity_curve(strat_returns).iloc[-1] - 1),
        "strategy_sharpe": float(sharpe(strat_returns)),
        "bnh_total_return": float(equity_curve(bnh_returns).iloc[-1] - 1),
        "bnh_sharpe": float(sharpe(bnh_returns)),
    }
    if not djia_returns.empty:
        metrics["djia_total_return"] = float(equity_curve(djia_returns).iloc[-1] - 1)
        metrics["djia_sharpe"] = float(sharpe(djia_returns))

    # Visualization
    plt.figure(figsize=(10,6))
    equity_df = pd.DataFrame({
        f"{symbol}_momentum": equity_curve(strat_returns),
        f"{symbol}_buy&hold": equity_curve(bnh_returns),
    })
    if not djia_returns.empty:
        equity_df["DJIA_proxy"] = equity_curve(djia_returns)
    equity_df.plot(ax=plt.gca())
    plt.title(f"Equity Curves {symbol}")
    plt.legend()
    out_png = OUTPUT_DIR / f"momentum_equity_{symbol}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    # Persist report JSON
    report = {
        "symbol": symbol,
        "start": str(equity_df.index.min()) if not equity_df.empty else None,
        "end": str(equity_df.index.max()) if not equity_df.empty else None,
        "lookback": lookback,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    out_json = OUTPUT_DIR / f"momentum_report_{symbol}.json"
    out_json.write_text(json.dumps(report, indent=2))

    # Submit factor and backtest via demo (persist jsonl)
    try:
        from demo_decoupled_system import EnhancedAlphaPoolDemo
        demo = EnhancedAlphaPoolDemo(test_mode=True)
        # Register strategy then submit
        cfg_res = demo._loop.run_until_complete(demo.develop_strategy_configuration()) if hasattr(demo, "_loop") else None
    except Exception:
        demo = None
        cfg_res = None
    # Fallback: write a submission jsonl
    submission = {
        "strategy_id": cfg_res.get("strategy_configuration", {}).get("strategy_id") if cfg_res else f"strategy_{symbol}",
        "factor": "momentum_20d",
        "metrics": metrics,
        "submitted_at": datetime.now().isoformat()
    }
    (OUTPUT_DIR / "submissions.jsonl").open("a").write(json.dumps(submission) + "\n")

    print(json.dumps({"status": "success", "report": str(out_json), "chart": str(out_png), "metrics": metrics}, indent=2))

if __name__ == "__main__":
    main()
