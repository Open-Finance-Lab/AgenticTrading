"""
Market data fetcher - connects to Alpaca API for live quotes.

Canonical location (Phase 3D1). Moved verbatim from
``dashboard/backend/market_data.py`` (original module removed in Phase 4A).
Provider URLs, endpoints, request methods/headers/params/timeouts/retries,
symbol/timestamp handling, returned schemas, fallback behavior, caching, and
logging are unchanged; only the module location moved.
"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

from dashboard.backend.paths import CREDENTIALS_DIR

TICKER_CACHE_TTL_SECONDS = 30
# How long a stale payload may still be served while a background thread
# refreshes it. Beyond this the data is too old to show and the request pays
# the inline fetch again.
TICKER_STALE_SERVE_SECONDS = 900
# After a fetch comes back empty (provider outage), stop hammering the provider
# for this long. Without it a sustained outage puts every single request back on
# the multi-second inline path -- exactly the behaviour this cache exists to
# remove -- because a failed fetch never advances the cache timestamp.
TICKER_FAILED_FETCH_BACKOFF_SECONDS = 60
# /ticker is session-exempt and takes an arbitrary ``symbols=`` string, so every
# map keyed on that string is reachable by unauthenticated input. Bound them, or
# they grow without limit (and each distinct stale key can own a refresh thread).
TICKER_MAX_CACHE_KEYS = 256
_ticker_cache: Dict[str, Tuple[float, List[Dict]]] = {}
_ticker_refresh_lock = threading.Lock()
_ticker_refresh_inflight: set = set()
_ticker_fetch_locks: Dict[str, threading.Lock] = {}
_ticker_failed_fetch_at: Dict[str, float] = {}

# Symbols are interpolated into Alpaca URL paths, so anything outside a real
# ticker's alphabet (letters, digits, "." and "-", e.g. BRK.B / BF-B) could
# steer the request to a different endpoint under the server's API key.
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9.\-]{1,12}$")


def _is_valid_symbol(symbol: str) -> bool:
    return bool(_SYMBOL_RE.match(symbol or ""))


def _extract_price_from_quote(quote: dict) -> Optional[float]:
    """Derive a display price from an Alpaca quote payload."""
    try:
        ap = float(quote.get("ap")) if quote.get("ap") else None
    except (ValueError, TypeError):
        ap = None
    try:
        bp = float(quote.get("bp")) if quote.get("bp") else None
    except (ValueError, TypeError):
        bp = None
    try:
        last_trade = float(quote.get("p")) if quote.get("p") else None
    except (ValueError, TypeError):
        last_trade = None

    if ap is not None and ap > 0 and bp is not None and bp > 0:
        return (ap + bp) / 2
    if ap is not None and ap > 0:
        return ap
    if bp is not None and bp > 0:
        return bp
    if last_trade is not None and last_trade > 0:
        return last_trade
    return None


def _extract_prev_close_from_bars(bars: List[dict]) -> Optional[float]:
    """Pick the most recent completed daily close from sorted bars."""
    if not bars:
        return None

    bars_sorted = sorted(bars, key=lambda bar: bar.get("t", ""), reverse=True)
    target_bar = bars_sorted[1] if len(bars_sorted) > 1 else bars_sorted[0]
    try:
        prev_close = float(target_bar.get("c", 0))
    except (ValueError, TypeError):
        return None
    return prev_close if prev_close > 0 else None

class AlpacaMarketData:
    """Fetch live market quotes from Alpaca API."""
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, 
                 paper: bool = True):
        """
        Initialize with Alpaca credentials.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading endpoint (default True)
        """
        # Always initialize these first
        self.api_key = None
        self.secret_key = None
        
        # Try to load from credentials
        if api_key is None:
            self._load_from_credentials()
        else:
            self.api_key = api_key
            self.secret_key = secret_key
        
        self.paper = paper
        self.base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.data_api_url = "https://data.alpaca.markets"
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
    
    def _load_from_credentials(self):
        """Load credentials from environment variables or credentials file"""
        # Try environment variables first (for Render, Docker, etc.)
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if self.api_key and self.secret_key:
            return
        
        # Fall back to credentials file (for local development)
        creds_path = CREDENTIALS_DIR / "alpaca.json"
        
        if creds_path.exists():
            with open(creds_path, 'r') as f:
                creds = json.load(f)
                self.api_key = creds.get('api_key') or creds.get('apiKey')
                self.secret_key = creds.get('secret_key') or creds.get('secretKey')
                self.paper = creds.get('paper', True)
        else:
            raise FileNotFoundError(f"Credentials file not found: {creds_path}")
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get latest quote for a symbol from Alpaca Data API.
        
        Returns:
            Dict with keys: symbol, price, changePercent, timestamp
            changePercent is percentage change vs previous day's close
        """
        if not _is_valid_symbol(symbol):
            print(f"❌ Rejected invalid symbol for quote fetch: {symbol!r}")
            return None
        try:
            # Use Alpaca Data API endpoint
            url = f"{self.data_api_url}/v2/stocks/{symbol}/quotes/latest"
            
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract quote from response
                if "quote" in data:
                    quote = data["quote"]
                    quote_timestamp = quote.get("t", "unknown")
                    
                    # STEP 1: Get current price using bid/ask midpoint
                    try:
                        # Extract and convert to float
                        ap = quote.get("ap")
                        bp = quote.get("bp")
                        p = quote.get("p")
                        
                        # Convert all to float, skip if conversion fails
                        try:
                            ap = float(ap) if ap else None
                        except (ValueError, TypeError):
                            ap = None
                        
                        try:
                            bp = float(bp) if bp else None
                        except (ValueError, TypeError):
                            bp = None
                        
                        try:
                            p = float(p) if p else None
                        except (ValueError, TypeError):
                            p = None
                        
                        # Calculate current price with fallback logic
                        current_price = None
                        price_source = None
                        
                        if ap is not None and ap > 0 and bp is not None and bp > 0:
                            current_price = (ap + bp) / 2
                            price_source = "midpoint(ap,bp)"
                        elif ap is not None and ap > 0:
                            current_price = ap
                            price_source = "ask(ap)"
                        elif bp is not None and bp > 0:
                            current_price = bp
                            price_source = "bid(bp)"
                        elif p is not None and p > 0:
                            current_price = p
                            price_source = "last_trade(p)"
                        
                        if current_price is None or current_price <= 0:
                            print(f"❌ {symbol}: Could not determine current price (ap={ap}, bp={bp}, p={p})")
                            return None
                        
                        print(f"✅ {symbol}: current_price={current_price} source={price_source} ts={quote_timestamp}")
                    except Exception as e:
                        print(f"❌ {symbol}: Error calculating current price: {e}")
                        return None
                    
                    # STEP 2: Get previous close from historical daily bars
                    prev_close = self._get_previous_close(symbol)
                    
                    # STEP 3: Calculate % change
                    if prev_close and prev_close > 0:
                        change_percent = ((current_price - prev_close) / prev_close) * 100
                        print(f"✅ {symbol}: change_percent={change_percent:.2f}% (current={current_price} - prev_close={prev_close})")
                    else:
                        # If we can't get previous close, return None for change_percent
                        change_percent = None
                        print(f"⚠️ {symbol}: No previous_close available, showing '--' for % change")
                    
                    return {
                        "symbol": symbol,
                        "price": round(current_price, 2),
                        "changePercent": round(change_percent, 2) if change_percent is not None else None,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    print(f"❌ {symbol}: No 'quote' key in Alpaca response")
                    return None
            else:
                print(f"❌ {symbol}: Error fetching quote: {response.status_code} - {response.text[:200]}")
                return None
        
        except Exception as e:
            print(f"❌ {symbol}: Exception fetching quote: {e}")
            return None
    
    def _get_previous_close(self, symbol: str) -> Optional[float]:
        """
        Get previous day's close price from historical daily bars.
        
        Returns the close price (field 'c') from the most recent completed trading day.
        
        Strategy:
        1. Try IEX feed (free tier has access)
        2. If IEX fails, try SIP with delayed end time (15+ mins ago)
        3. If both fail, return None
        """
        if not _is_valid_symbol(symbol):
            print(f"❌ Rejected invalid symbol for previous-close fetch: {symbol!r}")
            return None
        try:
            from datetime import datetime, timedelta
            
            # Fetch last 5 trading days to ensure we get the most recent completed day
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            # ===== ATTEMPT 1: Try IEX feed (free tier) =====
            url_iex = f"{self.data_api_url}/v2/stocks/{symbol}/bars?timeframe=1Day&start={start_date}&end={end_date}&limit=5&feed=iex"
            print(f"  Fetching {symbol} previous_close from IEX: {url_iex}")
            
            response = requests.get(url_iex, headers=self.headers, timeout=5)
            print(f"  IEX response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if "bars" in data and len(data["bars"]) > 0:
                    bars = data["bars"]
                    
                    # Sort by timestamp descending (most recent first)
                    bars_sorted = sorted(bars, key=lambda x: x.get("t", ""), reverse=True)
                    
                    if len(bars_sorted) > 1:
                        # Use 2nd most recent (in case most recent is incomplete)
                        try:
                            bar_timestamp = bars_sorted[1].get("t", "unknown")
                            prev_close = float(bars_sorted[1].get("c", 0))
                            print(f"✅ {symbol}: previous_close={prev_close} (from IEX feed, ts={bar_timestamp})")
                            return prev_close if prev_close > 0 else None
                        except (ValueError, TypeError) as e:
                            print(f"⚠️ {symbol}: Could not parse IEX bar data: {e}")
                    elif len(bars_sorted) == 1:
                        try:
                            bar_timestamp = bars_sorted[0].get("t", "unknown")
                            prev_close = float(bars_sorted[0].get("c", 0))
                            print(f"✅ {symbol}: previous_close={prev_close} (from IEX feed - 1 bar only, ts={bar_timestamp})")
                            return prev_close if prev_close > 0 else None
                        except (ValueError, TypeError) as e:
                            print(f"⚠️ {symbol}: Could not parse IEX bar data: {e}")
            
            # ===== ATTEMPT 2: Try SIP with delayed end time (15+ mins ago) =====
            # Only query SIP data up to 15 minutes ago to avoid "recent SIP data" restriction
            delayed_end = datetime.now() - timedelta(minutes=15)
            delayed_end_str = delayed_end.strftime("%Y-%m-%d")
            
            url_sip = f"{self.data_api_url}/v2/stocks/{symbol}/bars?timeframe=1Day&start={start_date}&end={delayed_end_str}&limit=5"
            print(f"  Fetching {symbol} previous_close from delayed SIP (end={delayed_end_str}): {url_sip}")
            
            response = requests.get(url_sip, headers=self.headers, timeout=5)
            print(f"  SIP response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if "bars" in data and len(data["bars"]) > 0:
                    bars = data["bars"]
                    
                    # Sort by timestamp descending (most recent first)
                    bars_sorted = sorted(bars, key=lambda x: x.get("t", ""), reverse=True)
                    
                    if len(bars_sorted) > 1:
                        try:
                            bar_timestamp = bars_sorted[1].get("t", "unknown")
                            prev_close = float(bars_sorted[1].get("c", 0))
                            print(f"✅ {symbol}: previous_close={prev_close} (from delayed SIP, ts={bar_timestamp})")
                            return prev_close if prev_close > 0 else None
                        except (ValueError, TypeError) as e:
                            print(f"⚠️ {symbol}: Could not parse SIP bar data: {e}")
                    elif len(bars_sorted) == 1:
                        try:
                            bar_timestamp = bars_sorted[0].get("t", "unknown")
                            prev_close = float(bars_sorted[0].get("c", 0))
                            print(f"✅ {symbol}: previous_close={prev_close} (from delayed SIP - 1 bar only, ts={bar_timestamp})")
                            return prev_close if prev_close > 0 else None
                        except (ValueError, TypeError) as e:
                            print(f"⚠️ {symbol}: Could not parse SIP bar data: {e}")
            
            # ===== BOTH FAILED: Return None =====
            print(f"⚠️ {symbol}: Could not fetch previous_close (IEX and SIP both unavailable)")
            return None
                    
        except Exception as e:
            print(f"⚠️ Error getting previous close for {symbol}: {e}")
        
        return None
    
    def get_latest_quotes_batch(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch latest prices for many symbols in one Alpaca request."""
        if not symbols:
            return {}

        symbols_param = ",".join(symbols)
        url = (
            f"{self.data_api_url}/v2/stocks/quotes/latest"
            f"?symbols={symbols_param}&feed=iex"
        )
        print(f"  Batch latest quotes: {symbols_param}")

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                print(f"❌ Batch quotes failed: {response.status_code} - {response.text[:200]}")
                return {}

            data = response.json()
            prices: Dict[str, float] = {}
            for symbol, quote in (data.get("quotes") or {}).items():
                price = _extract_price_from_quote(quote)
                if price:
                    prices[symbol.upper()] = price
                    print(f"✅ {symbol}: current_price={price} source=batch_iex")
            return prices
        except Exception as exc:
            print(f"❌ Batch quotes exception: {exc}")
            return {}

    def get_previous_closes_batch(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch previous closes for many symbols in one Alpaca request."""
        if not symbols:
            return {}

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        symbols_param = ",".join(symbols)
        url = (
            f"{self.data_api_url}/v2/stocks/bars"
            f"?symbols={symbols_param}&timeframe=1Day&start={start_date}"
            f"&end={end_date}&limit=5&feed=iex"
        )
        print(f"  Batch previous closes: {symbols_param}")

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                print(f"❌ Batch bars failed: {response.status_code} - {response.text[:200]}")
                return {}

            data = response.json()
            closes: Dict[str, float] = {}
            for symbol, bars in (data.get("bars") or {}).items():
                prev_close = _extract_prev_close_from_bars(bars)
                if prev_close:
                    closes[symbol.upper()] = prev_close
                    print(f"✅ {symbol}: previous_close={prev_close} source=batch_iex")
            return closes
        except Exception as exc:
            print(f"❌ Batch bars exception: {exc}")
            return {}

    def _build_quote(self, symbol: str, price: float, prev_close: Optional[float]) -> Dict:
        change_percent = None
        if prev_close and prev_close > 0:
            change_percent = ((price - prev_close) / prev_close) * 100
        return {
            "symbol": symbol,
            "price": round(price, 2),
            "changePercent": round(change_percent, 2) if change_percent is not None else None,
            "timestamp": datetime.now().isoformat(),
        }

    def get_quotes_batch(self, symbols: List[str]) -> List[Dict]:
        """
        Get quotes for multiple symbols using batched Alpaca requests.
        Falls back to per-symbol fetching if batch endpoints fail.
        """
        if not symbols:
            return []

        with ThreadPoolExecutor(max_workers=2) as executor:
            latest_future = executor.submit(self.get_latest_quotes_batch, symbols)
            prev_close_future = executor.submit(self.get_previous_closes_batch, symbols)
            latest_prices = latest_future.result()
            prev_closes = prev_close_future.result()

        quotes: List[Dict] = []
        missing_symbols = []

        for symbol in symbols:
            price = latest_prices.get(symbol)
            if price is None:
                missing_symbols.append(symbol)
                continue
            quotes.append(self._build_quote(symbol, price, prev_closes.get(symbol)))

        if missing_symbols:
            print(f"⚠️ Batch fetch missed {missing_symbols}; falling back to per-symbol quotes")
            for symbol in missing_symbols:
                quote = self.get_quote(symbol)
                if quote:
                    quotes.append(quote)

        return quotes
    
    def get_crypto_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get crypto quote (BTC, ETH, etc.) from a free public API.
        Alpaca free tier doesn't support crypto quotes, so we use CoinGecko.
        """
        try:
            if symbol not in ["BTC", "ETH", "XRP", "SOL"]:
                return None
            
            # Map symbols to CoinGecko IDs
            gecko_ids = {
                "BTC": "bitcoin",
                "ETH": "ethereum",
                "XRP": "ripple",
                "SOL": "solana",
            }
            
            gecko_id = gecko_ids.get(symbol)
            if not gecko_id:
                return None
            
            # STEP 1: Fetch from CoinGecko API (free, no auth needed)
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={gecko_id}&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true"
            print(f"  Fetching {symbol} from CoinGecko: {url}")
            
            # Retry up to 2 times on rate limit
            max_retries = 2
            for attempt in range(max_retries):
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    break
                elif response.status_code == 429:
                    print(f"⚠️ {symbol}: Rate limited (429) - attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1)  # Wait before retry
                    else:
                        print(f"⚠️ {symbol}: Could not fetch CoinGecko quote after {max_retries} attempts: rate limited")
                        return None
                else:
                    print(f"⚠️ {symbol}: Could not fetch CoinGecko quote: {response.status_code}")
                    return None
            
            data = response.json()
            
            # CoinGecko returns: {gecko_id: {usd: price, usd_24h_change: percent}}
            if gecko_id not in data:
                print(f"⚠️ {symbol}: No data from CoinGecko")
                return None
            
            crypto_data = data[gecko_id]
            
            try:
                current_price = float(crypto_data.get("usd", 0))
                change_percent = float(crypto_data.get("usd_24h_change", None))
                quote_timestamp = datetime.now().isoformat()
            except (ValueError, TypeError) as e:
                print(f"⚠️ {symbol}: Could not parse CoinGecko data: {e}")
                return None
            
            if current_price <= 0:
                print(f"⚠️ {symbol}: Invalid current price from CoinGecko: {current_price}")
                return None
            
            print(f"✅ {symbol}: current_price={current_price} source=CoinGecko ts={quote_timestamp}")
            
            if change_percent is not None:
                print(f"✅ {symbol}: change_percent={change_percent:.2f}% (24h change from CoinGecko)")
            else:
                print(f"⚠️ {symbol}: No 24h change data available")
            
            # Format price (show thousands as k for consistency)
            if current_price >= 1000:
                formatted_price = f"{current_price/1000:.2f}k"
            else:
                formatted_price = f"{current_price:.2f}"
            
            return {
                "symbol": symbol,
                "price": formatted_price,
                "changePercent": round(change_percent, 2) if change_percent is not None else None,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            print(f"❌ {symbol}: Exception fetching crypto from Alpaca: {e}")
        
        return None


def _yahoo_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")


def _pick_first_number(*values) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def get_yahoo_quotes_batch(symbols: List[str]) -> List[Dict]:
    """Fetch stock quotes from Yahoo Finance (no API key required)."""
    if not symbols:
        return []

    try:
        import yfinance as yf
    except ImportError as exc:
        print(f"❌ yfinance not installed: {exc}")
        return []

    yahoo_symbols = [_yahoo_symbol(symbol) for symbol in symbols]
    symbol_map = dict(zip(yahoo_symbols, symbols))
    quotes: List[Dict] = []
    missing: List[str] = []

    try:
        tickers = yf.Tickers(" ".join(yahoo_symbols))
        for yahoo_symbol, original_symbol in symbol_map.items():
            ticker = tickers.tickers.get(yahoo_symbol)
            if ticker is None:
                missing.append(original_symbol)
                continue

            try:
                fast = ticker.fast_info
                price = _pick_first_number(
                    getattr(fast, "last_price", None),
                    getattr(fast, "regular_market_price", None),
                    fast.get("last_price") if isinstance(fast, dict) else None,
                    fast.get("regular_market_price") if isinstance(fast, dict) else None,
                )
                prev_close = _pick_first_number(
                    getattr(fast, "previous_close", None),
                    getattr(fast, "regular_market_previous_close", None),
                    fast.get("previous_close") if isinstance(fast, dict) else None,
                    fast.get("regular_market_previous_close") if isinstance(fast, dict) else None,
                )
            except Exception as exc:
                print(f"⚠️ {original_symbol}: fast_info failed ({exc})")
                missing.append(original_symbol)
                continue

            if price is None:
                missing.append(original_symbol)
                continue

            change_percent = None
            if prev_close:
                change_percent = ((price - prev_close) / prev_close) * 100

            quotes.append({
                "symbol": original_symbol,
                "price": round(price, 2),
                "changePercent": round(change_percent, 2) if change_percent is not None else None,
                "timestamp": datetime.now().isoformat(),
            })
            print(f"✅ {original_symbol}: price={price:.2f} change={change_percent} source=yahoo")
    except Exception as exc:
        print(f"❌ Yahoo batch fetch failed: {exc}")
        missing = list(symbols)

    if missing:
        try:
            history = yf.download(
                [_yahoo_symbol(symbol) for symbol in missing],
                period="5d",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            print(f"❌ Yahoo history fallback failed: {exc}")
            return quotes

        for original_symbol in missing:
            yahoo_symbol = _yahoo_symbol(original_symbol)
            try:
                if len(missing) == 1:
                    closes = history["Close"].dropna()
                else:
                    closes = history[yahoo_symbol]["Close"].dropna()
                if closes.empty:
                    continue
                price = float(closes.iloc[-1])
                prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else None
                change_percent = ((price - prev_close) / prev_close * 100) if prev_close else None
                quotes.append({
                    "symbol": original_symbol,
                    "price": round(price, 2),
                    "changePercent": round(change_percent, 2) if change_percent is not None else None,
                    "timestamp": datetime.now().isoformat(),
                })
                print(f"✅ {original_symbol}: price={price:.2f} change={change_percent} source=yahoo_history")
            except Exception as exc:
                print(f"⚠️ {original_symbol}: Yahoo history parse failed ({exc})")

    order = {symbol: index for index, symbol in enumerate(symbols)}
    quotes.sort(key=lambda quote: order.get(quote["symbol"], 999))
    return quotes


def _fetch_quotes_uncached(symbols: List[str]) -> List[Dict]:
    """Fetch quotes from the providers with no cache involved (slow: seconds)."""
    # Separate stocks and crypto
    stocks = [s for s in symbols if s not in ["BTC", "ETH", "XRP", "SOL"]]
    crypto = [s for s in symbols if s in ["BTC", "ETH", "XRP", "SOL"]]

    quotes = []

    # Fetch stocks from Yahoo Finance (fast, no Alpaca credentials needed)
    if stocks:
        quotes.extend(get_yahoo_quotes_batch(stocks))
        if len(quotes) < len(stocks):
            fetched = {quote["symbol"] for quote in quotes}
            missing_stocks = [symbol for symbol in stocks if symbol not in fetched]
            if missing_stocks:
                try:
                    market_data = AlpacaMarketData()
                    quotes.extend(market_data.get_quotes_batch(missing_stocks))
                except Exception as e:
                    print(f"Error initializing Alpaca fallback: {e}")

    # Fetch crypto
    if crypto:
        try:
            market_data = AlpacaMarketData()
            for symbol in crypto:
                quote = market_data.get_crypto_quote(symbol)
                if quote:
                    quotes.append(quote)
        except Exception as e:
            print(f"Error fetching crypto: {e}")

    return quotes


def _evict_ticker_keys_locked() -> None:
    """Bound the symbol-keyed maps. Call with ``_ticker_refresh_lock`` held."""
    overflow = len(_ticker_cache) - TICKER_MAX_CACHE_KEYS
    if overflow > 0:
        # Oldest first; entries are (timestamp, payload).
        oldest = sorted(_ticker_cache.items(), key=lambda kv: kv[1][0])[:overflow]
        for key, _ in oldest:
            _ticker_cache.pop(key, None)
            _ticker_failed_fetch_at.pop(key, None)
    for key in list(_ticker_fetch_locks):
        if key not in _ticker_cache and key not in _ticker_refresh_inflight:
            _ticker_fetch_locks.pop(key, None)


def _in_failure_backoff(cache_key: str) -> bool:
    failed_at = _ticker_failed_fetch_at.get(cache_key)
    return (
        failed_at is not None
        and time.time() - failed_at < TICKER_FAILED_FETCH_BACKOFF_SECONDS
    )


def _fetch_and_store(cache_key: str, symbols: List[str]) -> List[Dict]:
    quotes = _fetch_quotes_uncached(symbols)
    existing = _ticker_cache.get(cache_key)
    if quotes or not (existing and existing[1]):
        # An empty fetch (provider outage) must never overwrite a payload that
        # is still inside the serve-stale window.
        _ticker_cache[cache_key] = (time.time(), quotes)
    if quotes:
        _ticker_failed_fetch_at.pop(cache_key, None)
    else:
        # Remember the failure so callers back off instead of retrying the slow
        # provider on every request for as long as the outage lasts.
        _ticker_failed_fetch_at[cache_key] = time.time()
    with _ticker_refresh_lock:
        _evict_ticker_keys_locked()
    return quotes


def _refresh_ticker_in_background(cache_key: str, symbols: List[str]) -> None:
    with _ticker_refresh_lock:
        if cache_key in _ticker_refresh_inflight:
            return
        _ticker_refresh_inflight.add(cache_key)

    def worker():
        try:
            _fetch_and_store(cache_key, symbols)
        except Exception as exc:
            # !r on the key, not str: it is built from unauthenticated
            # ``symbols`` input, and repr escapes any embedded newline that
            # would otherwise forge a second log line.
            print(f"Ticker background refresh failed for {cache_key!r}: {exc!r}")
        finally:
            with _ticker_refresh_lock:
                _ticker_refresh_inflight.discard(cache_key)

    # cache_key is unauthenticated input, so it is truncated rather than pasted
    # whole into the thread name.
    thread = threading.Thread(
        target=worker, name=f"ticker-refresh-{cache_key[:40]}", daemon=True
    )
    try:
        thread.start()
    except (RuntimeError, OSError) as exc:
        # The key was marked inflight before the thread existed, so a failed
        # start would otherwise bar it from ever refreshing again. Swallowing
        # the error also keeps the caller's good stale payload -- letting this
        # propagate would surface as an empty ticker instead.
        print(f"Ticker background refresh could not start for {cache_key!r}: {exc!r}")
        with _ticker_refresh_lock:
            _ticker_refresh_inflight.discard(cache_key)


def _reset_ticker_refresh_state_for_tests() -> None:
    with _ticker_refresh_lock:
        _ticker_refresh_inflight.clear()
        _ticker_fetch_locks.clear()
        _ticker_failed_fetch_at.clear()


def get_market_quotes(symbols: List[str]) -> List[Dict]:
    """
    Convenience function to fetch quotes for stocks and crypto.

    Stale-while-revalidate: a fresh cache entry (< TICKER_CACHE_TTL_SECONDS)
    is returned as-is; a stale-but-recent one (< TICKER_STALE_SERVE_SECONDS)
    is returned immediately while one background thread refreshes it; only a
    genuinely cold cache pays the multi-second provider fetch in-request, and
    concurrent cold callers share a single fetch.

    Usage:
        quotes = get_market_quotes(["AAPL", "NVDA", "MSFT", "BTC"])
    """
    cache_key = ",".join(sorted(symbol.strip().upper() for symbol in symbols if symbol.strip()))
    if not cache_key:
        return _fetch_quotes_uncached(symbols)

    cached = _ticker_cache.get(cache_key)
    if cached:
        age = time.time() - cached[0]
        if age < TICKER_CACHE_TTL_SECONDS:
            return cached[1]
        if cached[1]:
            if age < TICKER_STALE_SERVE_SECONDS:
                if not _in_failure_backoff(cache_key):
                    _refresh_ticker_in_background(cache_key, list(symbols))
                return cached[1]
            if _in_failure_backoff(cache_key):
                # Aged out of the serve window *and* the provider is failing.
                # Retrying inline per request would restore the multi-second
                # blocking fetch this cache exists to avoid, and would hand the
                # caller an empty ticker while usable data sits right here.
                return cached[1]

    # Cold cache (or stale beyond the serve window / empty payload): fetch
    # inline, but let concurrent callers share one provider round-trip.
    with _ticker_refresh_lock:
        fetch_lock = _ticker_fetch_locks.setdefault(cache_key, threading.Lock())
    with fetch_lock:
        cached = _ticker_cache.get(cache_key)
        if cached and cached[1] and time.time() - cached[0] < TICKER_CACHE_TTL_SECONDS:
            return cached[1]  # another caller fetched while we waited
        quotes = _fetch_and_store(cache_key, symbols)
        if quotes:
            return quotes
        # The fetch came back empty. _fetch_and_store deliberately preserved any
        # good-but-expired payload, so prefer that over handing back nothing.
        stale = _ticker_cache.get(cache_key)
        return stale[1] if stale and stale[1] else quotes
