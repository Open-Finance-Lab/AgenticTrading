"""Panel proxy for FinSearch news signals — keeps FINGPT_API_KEY server-side."""
import copy
import json
import logging
import os
import threading

from fastapi import APIRouter

from dashboard.backend.cache import (CACHE_KEY_NEWS_SIGNALS, TTL_NEWS,
                                     TTL_NEWS_UNAVAILABLE, shared_ttl_cache)
from dashboard.backend.infrastructure.llm.validator import DJIA_30
from dashboard.backend.integrations import news_sentiment
from dashboard.backend.paths import CONFIG_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/news", tags=["news"])

# FinSearch's news pipeline is paused (2026-10-11), so the panel serves the last
# live payload, frozen, flagged `demo: true` for the "Demo news" line. Proxying
# live would only show a feed going stale with no explanation.
#
# NEWS_FEED_FROZEN is read per request so the feed resumes from the Render
# dashboard with no deploy: restart the FinSearch timers, then set it to `0`.
# Unset means frozen while the pipeline is paused. A value outside the boolean
# vocabulary also reads as frozen — the unfrozen path calls an upstream that may
# be dead, while the frozen path is labelled and local — and warns once.
NEWS_FEED_FROZEN_ENV = "NEWS_FEED_FROZEN"
NEWS_FEED_FROZEN_DEFAULT = True
NEWS_DEMO_SNAPSHOT_PATH = CONFIG_DIR / "news_demo_snapshot.json"

_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}
_warned_values: set[str] = set()

# The keys the panel reads; a snapshot without them is not a payload.
_SNAPSHOT_REQUIRED_KEYS = ("status", "feed", "signals")

_snapshot_lock = threading.Lock()
_snapshot_cache: dict | None = None


def news_feed_frozen() -> bool:
    raw = (os.getenv(NEWS_FEED_FROZEN_ENV) or "").strip().lower()
    if not raw:
        return NEWS_FEED_FROZEN_DEFAULT
    if raw in _FALSEY:
        return False
    if raw not in _TRUTHY and raw not in _warned_values:
        _warned_values.add(raw)
        logger.warning("news: %s=%r is not a boolean; serving the frozen snapshot",
                       NEWS_FEED_FROZEN_ENV, raw)
    return True


def _load_demo_snapshot() -> dict | None:
    try:
        payload = json.loads(NEWS_DEMO_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.error("news: demo snapshot %s unreadable; serving unavailable",
                     NEWS_DEMO_SNAPSHOT_PATH, exc_info=True)
        return None
    if not isinstance(payload, dict) or not all(k in payload for k in _SNAPSHOT_REQUIRED_KEYS):
        logger.error("news: demo snapshot %s is not a panel payload (needs %s); "
                     "serving unavailable", NEWS_DEMO_SNAPSHOT_PATH,
                     ", ".join(_SNAPSHOT_REQUIRED_KEYS))
        return None
    # The loader owns both frozen-feed markers so the file cannot disagree with
    # them: `demo` drives the "Demo news" line, and a frozen "Updated 0.2h ago"
    # would be false.
    return {**payload, "demo": True, "staleness_hours": None}


def _demo_snapshot() -> dict:
    """A private copy of the frozen payload, or UNAVAILABLE if it cannot load.

    Only a successful load is cached: a transient read error must not pin
    `unavailable` for the life of the process. The copy is deep because the
    payload's `feed`/`signals` are nested and shared across requests.
    """
    global _snapshot_cache
    if _snapshot_cache is None:
        with _snapshot_lock:
            if _snapshot_cache is None:
                _snapshot_cache = _load_demo_snapshot()
    if _snapshot_cache is None:
        return dict(news_sentiment.UNAVAILABLE_PAYLOAD)
    return copy.deepcopy(_snapshot_cache)


def _reset_demo_snapshot_cache() -> None:
    global _snapshot_cache
    with _snapshot_lock:
        _snapshot_cache = None


def _fetch_panel() -> dict:
    # get_latest_panel_payload already fails closed to UNAVAILABLE_PAYLOAD; the
    # except is a belt-and-braces guard so a bug can't leak a 500 to the panel.
    try:
        return news_sentiment.get_latest_panel_payload(list(DJIA_30))
    except Exception:
        return dict(news_sentiment.UNAVAILABLE_PAYLOAD)


# Deliberately sync `def`: FastAPI runs it in the threadpool, so the blocking
# requests.get inside the adapter cannot stall the event loop. Do NOT convert
# to `async def` without moving the HTTP call off the loop.
@router.get("/signals")
def latest_news_signals():
    if news_feed_frozen():
        return _demo_snapshot()
    # Single-flight lives in the cache primitive: one leader hits the producer,
    # concurrent callers return the cached value (or a brief UNAVAILABLE) WITHOUT
    # blocking a worker behind the ~10s upstream call — so a cold-cache burst of
    # Home polls can't starve the shared threadpool that also serves the
    # deadline-sensitive /api/v1/runs/* endpoints. A short negative-cache TTL
    # (vs TTL_NEWS) also lets the panel recover within ~30s of the producer
    # coming back instead of pinning "unavailable" for the full 420s.
    return shared_ttl_cache.get_or_fetch(
        CACHE_KEY_NEWS_SIGNALS,
        _fetch_panel,
        ttl_seconds=TTL_NEWS,
        negative_ttl_seconds=TTL_NEWS_UNAVAILABLE,
        is_negative=lambda payload: payload.get("status") == "unavailable",
        default=dict(news_sentiment.UNAVAILABLE_PAYLOAD),
    )
