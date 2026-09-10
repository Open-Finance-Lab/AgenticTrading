"""Alpaca **live-money** broker adapter.

Separate from :mod:`alpaca_paper` on purpose: a live Alpaca account uses a
different key pair from a paper account (Alpaca rejects paper keys against
the live endpoint and vice versa), so the two clients must never share
credentials or a base URL. This module owns provider calls only; the risk
gate and execution gate live in
``dashboard.backend.execution.alpaca_live_service``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

from dashboard.backend.infrastructure.market_data.alpaca_bars import resolve_alpaca_data_feed
from dashboard.backend.paths import CREDENTIALS_DIR


class AlpacaLiveCredentialsError(RuntimeError):
    """Raised when live credentials are missing or malformed."""


@dataclass
class LiveOrderResult:
    order_id: Optional[str]
    status: Optional[str]
    symbol: str
    side: str
    qty: float
    raw: Dict[str, Any]


class AlpacaLiveTradingClient:
    """Interface to Alpaca's **live** trading API (real money, real account).

    Credentials are intentionally read from ``ALPACA_LIVE_API_KEY`` /
    ``ALPACA_LIVE_SECRET_KEY`` (or ``credentials/alpaca_live.json``) rather
    than the ``ALPACA_API_KEY`` pair used by :mod:`alpaca_paper`, so a paper
    setup can never be silently pointed at a real account by reusing the same
    env vars.
    """

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key = api_key
        self.secret_key = secret_key
        if not self.api_key or not self.secret_key:
            self._load_from_credentials()
        if not self.api_key or not self.secret_key:
            raise AlpacaLiveCredentialsError(
                "Alpaca live credentials not configured. Set ALPACA_LIVE_API_KEY / "
                "ALPACA_LIVE_SECRET_KEY in .env, or create credentials/alpaca_live.json "
                "(see credentials/alpaca_live.json.example)."
            )

        self._trading = TradingClient(self.api_key, self.secret_key, paper=False)
        self._data = StockHistoricalDataClient(self.api_key, self.secret_key)

    def _load_from_credentials(self) -> None:
        self.api_key = self.api_key or os.getenv("ALPACA_LIVE_API_KEY")
        self.secret_key = self.secret_key or os.getenv("ALPACA_LIVE_SECRET_KEY")
        if self.api_key and self.secret_key:
            return

        creds_path = CREDENTIALS_DIR / "alpaca_live.json"
        if creds_path.exists():
            with open(creds_path, "r") as f:
                creds = json.load(f)
                self.api_key = self.api_key or creds.get("api_key") or creds.get("apiKey")
                self.secret_key = self.secret_key or creds.get("secret_key") or creds.get("secretKey")

    def get_account(self) -> Optional[Dict[str, Any]]:
        try:
            account = self._trading.get_account()
            return {
                "cash": float(account.cash),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "account_number": account.account_number,
                "account_status": str(account.status),
                "pattern_day_trader": bool(account.pattern_day_trader),
                "trading_blocked": bool(account.trading_blocked),
            }
        except Exception as e:
            print(f"Exception in AlpacaLiveTradingClient.get_account: {e}")
            return None

    def get_positions(self) -> Dict[str, float]:
        """Return ``{symbol: qty}`` for every held position (long only is expected)."""
        try:
            positions = self._trading.get_all_positions()
            return {p.symbol.upper(): float(p.qty) for p in positions}
        except Exception as e:
            print(f"Exception in AlpacaLiveTradingClient.get_positions: {e}")
            return {}

    def get_positions_detailed(self) -> List[Dict[str, Any]]:
        """Return full per-position detail (qty, prices, P&L) for display purposes.

        Separate from :meth:`get_positions` (which the risk gate depends on for
        its ``{symbol: qty}`` shape) so enriching this one for the UI can never
        change behavior on the trading path.
        """
        try:
            positions = self._trading.get_all_positions()
            return [
                {
                    "symbol": p.symbol.upper(),
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price) if p.current_price else None,
                    "market_value": float(p.market_value) if p.market_value else None,
                    "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl else None,
                    "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else None,
                    "side": str(p.side),
                }
                for p in positions
            ]
        except Exception as e:
            print(f"Exception in AlpacaLiveTradingClient.get_positions_detailed: {e}")
            return []

    def get_quotes(self, symbols: List[str]) -> Dict[str, float]:
        """Return ``{symbol: ask_price or bid_price}`` for the given symbols. Missing
        or unpriceable symbols are simply absent from the result — callers must
        treat an absent symbol as "no usable quote", never as price 0."""
        if not symbols:
            return {}
        # Resolved (and validated) outside the try/except below: an unsupported
        # ALPACA_DATA_FEED must raise loudly, not be swallowed as "no quotes".
        feed = resolve_alpaca_data_feed(DataFeed)
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=feed)
            quotes = self._data.get_stock_latest_quote(request)
        except Exception as e:
            print(f"Exception in AlpacaLiveTradingClient.get_quotes: {e}")
            return {}

        prices: Dict[str, float] = {}
        for symbol, quote in quotes.items():
            price = getattr(quote, "ask_price", None) or getattr(quote, "bid_price", None)
            if price and price > 0:
                prices[symbol.upper()] = float(price)
        return prices

    def submit_market_order(self, symbol: str, qty: float, side: str) -> LiveOrderResult:
        """Submit a live market order. Callers must have already run this order
        through the risk gate in ``alpaca_live_service`` — this method performs
        no clamping or sizing of its own."""
        side_lower = side.lower()
        if side_lower == "buy":
            order_side = OrderSide.BUY
        elif side_lower == "sell":
            order_side = OrderSide.SELL
        else:
            raise ValueError(f"Unknown order side {side!r}; expected 'buy' or 'sell'.")
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = self._trading.submit_order(order_data=request)
        return LiveOrderResult(
            order_id=str(order.id) if order.id else None,
            status=str(order.status) if order.status else None,
            symbol=symbol,
            side=side,
            qty=qty,
            raw={
                "id": str(order.id),
                "status": str(order.status),
                "symbol": order.symbol,
                "qty": str(order.qty),
                "side": str(order.side),
                "submitted_at": str(order.submitted_at),
                "filled_qty": str(order.filled_qty) if order.filled_qty is not None else None,
                "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price is not None else None,
                "filled_at": str(order.filled_at) if order.filled_at is not None else None,
            },
        )
