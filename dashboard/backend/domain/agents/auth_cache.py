"""In-process TTL cache for API-key auth + last_used_at debounce (T4).

Sits ABOVE the repository so the SQLite/Postgres twins stay dumb. Only the
per-request hot paths (v1 resolve_agent_by_key, v2 auth_scopes.resolve_agent)
route through here; management endpoints resolve directly for always-fresh
auth. Revocation/rotation propagates within <=TTL for entries nobody
invalidates; the delete/rotate paths call invalidate_agent() for immediate
effect (required: rotate_api_key blind-UPDATEs by agent id, so the OLD hash —
the cache key — is never in scope there; hence the reverse index).
"""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Any, Dict, Optional, Set, Tuple

from dashboard.backend.domain.agents import repository as _repo
from dashboard.backend.domain.agents.repository import _hash_api_key

# Read once at import; 0 disables. +-20% per-entry jitter keeps 100 agents'
# entries from expiring in lockstep and stampeding the DB/pool.
AGENT_AUTH_CACHE_TTL_SECONDS = float(os.getenv("AGENT_AUTH_CACHE_TTL_SECONDS", "10"))
LAST_USED_WRITE_INTERVAL_SECONDS = 60.0

_now = time.monotonic  # test seam


def _jitter() -> float:
    return random.uniform(0.8, 1.2)


_lock = threading.Lock()
_by_hash: Dict[str, Tuple[float, Dict[str, Any]]] = {}   # hash -> (expires_at, agent)
_hashes_by_agent: Dict[str, Set[str]] = {}               # agent_id -> {hashes}
_last_write: Dict[str, float] = {}                       # hash -> last touch time
# Bumped on every invalidate_agent(). A resolve snapshots it before the unlocked
# DB read and re-checks it before filling the cache, so a rotate/delete landing
# mid-resolve cannot have its stale pre-rotation snapshot written back AFTER
# invalidate_agent already ran — invalidate_agent could not see a hash the
# resolve had not yet inserted, so without this guard the revoked key would
# authenticate straight from the cache until the TTL elapsed.
_invalidation_epoch = 0


def _copy_agent(agent: Dict[str, Any]) -> Dict[str, Any]:
    """Independent copy so neither the caller nor the cache can mutate the
    other's view. Only ``scopes`` is a mutable nested value; the rest scalars."""
    out = dict(agent)
    scopes = out.get("scopes")
    if isinstance(scopes, list):
        out["scopes"] = list(scopes)
    return out


def resolve_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Cached resolve. Falls through to the store on miss/expiry; debounces
    the store's last_used_at write to once per LAST_USED_WRITE_INTERVAL."""
    if not api_key or not api_key.strip():
        return None
    ttl = AGENT_AUTH_CACHE_TTL_SECONDS
    if ttl <= 0:
        # Late-bound module attribute so per-test store swaps apply.
        return _repo.agent_store.resolve_api_key(api_key)

    key_hash = _hash_api_key(api_key.strip())
    now = _now()
    with _lock:
        hit = _by_hash.get(key_hash)
        if hit is not None and hit[0] > now:
            return _copy_agent(hit[1])
        should_touch = (now - _last_write.get(key_hash, float("-inf"))
                        >= LAST_USED_WRITE_INTERVAL_SECONDS)
        epoch_at_miss = _invalidation_epoch

    agent = _repo.agent_store.resolve_api_key(api_key, touch=should_touch)
    if agent is None:
        return None  # misses are not cached: an invalid key always re-checks

    with _lock:
        # Skip the fill if an invalidation raced our unlocked DB read: the
        # snapshot may already be stale (the key was just rotated/deleted). This
        # request read a valid row and may proceed, but caching it would
        # resurrect a revoked key for the whole TTL.
        if _invalidation_epoch == epoch_at_miss:
            _by_hash[key_hash] = (now + ttl * _jitter(), _copy_agent(agent))
            _hashes_by_agent.setdefault(agent["agent_id"], set()).add(key_hash)
            if should_touch:
                _last_write[key_hash] = now
    return agent


def invalidate_agent(agent_id: str) -> None:
    """Immediate eviction for delete/rotate (old hash found via reverse index)."""
    global _invalidation_epoch
    with _lock:
        _invalidation_epoch += 1
        for key_hash in _hashes_by_agent.pop(agent_id, set()):
            _by_hash.pop(key_hash, None)
            _last_write.pop(key_hash, None)


def _reset_for_tests() -> None:
    global _invalidation_epoch
    with _lock:
        _by_hash.clear()
        _hashes_by_agent.clear()
        _last_write.clear()
        _invalidation_epoch = 0
