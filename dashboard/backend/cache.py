"""
Simple in-memory cache with TTL (time-to-live) for paper trading data.
Reduces redundant API calls to Alpaca.
"""

import time
from typing import Optional, Dict, Any, Callable, Set
from threading import Lock


class CachedData:
    """Cached item with expiration."""
    
    def __init__(self, data: Any, ttl_seconds: int = 30):
        self.data = data
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds
    
    def get(self) -> Optional[Any]:
        """Get data if not expired, None otherwise."""
        if self.is_expired():
            return None
        return self.data


class PaperTradingCache:
    """Cache for paper trading API responses."""
    
    def __init__(self):
        self.cache: Dict[str, CachedData] = {}
        self.lock = Lock()
        # Single-flight coordination for get_or_fetch. Kept separate from
        # `lock` (which guards `cache`) so a leader never holds `lock` across
        # the blocking fetch_fn.
        self._leader_lock = Lock()
        self._inflight: Set[str] = set()

    def get_or_fetch(
        self,
        key: str,
        fetch_fn: Callable[[], Any],
        *,
        ttl_seconds: int,
        negative_ttl_seconds: Optional[int] = None,
        is_negative: Optional[Callable[[Any], bool]] = None,
        default: Any = None,
    ) -> Any:
        """Single-flight read-through cache.

        Returns the cached value if fresh. Otherwise exactly ONE caller (the
        leader) runs ``fetch_fn`` and caches the result; concurrent callers
        (followers) neither run ``fetch_fn`` nor block on it — they return the
        freshest value available (``default`` on a cold cache). This bounds both
        ``fetch_fn`` invocations AND blocked threadpool workers to ~1 per key
        per refresh, so a burst of cold-cache requests can't pile up behind one
        slow (~10s) upstream call inside FastAPI's bounded threadpool.

        If ``is_negative(result)`` is truthy, the result is cached for
        ``negative_ttl_seconds`` (a short outage-throttle window) instead of
        ``ttl_seconds``. A ``None`` result is treated as absent (not cached).
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        with self._leader_lock:
            cached = self.get(key)  # re-check: a leader may have just filled it
            if cached is not None:
                return cached
            if key in self._inflight:
                return default  # follower: do not fetch, do not block
            self._inflight.add(key)
        try:
            value = fetch_fn()
            if value is not None:
                ttl = ttl_seconds
                if (negative_ttl_seconds is not None and is_negative is not None
                        and is_negative(value)):
                    ttl = negative_ttl_seconds
                self.set(key, value, ttl_seconds=ttl)
            return value if value is not None else default
        finally:
            with self._leader_lock:
                self._inflight.discard(key)

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        with self.lock:
            if key in self.cache:
                data = self.cache[key].get()
                if data is not None:
                    return data
                else:
                    # Expired, remove from cache
                    del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 30):
        """Store value with TTL."""
        with self.lock:
            self.cache[key] = CachedData(value, ttl_seconds)
    
    def invalidate(self, key: str):
        """Remove cached entry."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
    
    def clear_all(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self.lock:
            expired = sum(1 for item in self.cache.values() if item.is_expired())
            return {
                "total_items": len(self.cache),
                "expired_items": expired,
                "active_items": len(self.cache) - expired
            }


# Singleton instance
paper_trading_cache = PaperTradingCache()


# Cache key constants
CACHE_KEY_ACCOUNT = "paper:account"
CACHE_KEY_POSITIONS = "paper:positions"
CACHE_KEY_TRADES = "paper:trades"
CACHE_KEY_PORTFOLIO_HISTORY = "paper:portfolio_history"
CACHE_KEY_BASELINES = "paper:baselines"

# TTL settings (in seconds)
TTL_ACCOUNT = 30          # Account updates less frequently
TTL_POSITIONS = 30        # Positions update on trades
TTL_TRADES = 60           # Trade history changes infrequently
TTL_PORTFOLIO_HISTORY = 120  # Portfolio history very stable
TTL_BASELINES = 3600      # Baselines update daily (1 hour cache)

# Alias: the class is a generic TTL cache; paper trading was merely its first
# consumer — new non-paper consumers use this name.
shared_ttl_cache = paper_trading_cache

CACHE_KEY_NEWS_SIGNALS = "news:signals"  # namespaced like the `paper:` keys

# Deliberately above the frontend's 300s poll so a client's own re-poll lands
# inside the cached window instead of always missing at the boundary.
TTL_NEWS = 420
# Short negative-cache TTL: an outage must not serialize every request through
# _fetch_lock at ~10s/attempt, but the panel must also recover within ~30s of
# the producer returning, not stay "unavailable" for the full 420s.
TTL_NEWS_UNAVAILABLE = 30
