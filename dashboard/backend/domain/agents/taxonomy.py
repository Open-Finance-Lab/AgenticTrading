"""Shared agent category whitelist ("shelves"). Slugs are stored; display names live in the frontend.

Two coercers, deliberately asymmetric, because they guard different boundaries:

* :func:`coerce_category` guards *caller-supplied* input (the HTTP bodies and the
  domain service). An unrecognized value there is a typo or a stale client, and
  silently dropping it would leave an agent unshelved with no error — so it raises.
* :func:`normalize_category` guards the *inbound legacy* boundary (the marketplace
  catalog this platform ships, whose values predate this vocabulary). Rejecting
  there would break cloning, so unrecognized values degrade to ``None``.

``AGENT_CATEGORIES`` is derived from ``AgentCategory`` rather than declared
alongside it: the ``Literal`` is what Pydantic validates against and what FastAPI
publishes into ``openapi.json``, so making it the single source keeps the runtime
whitelist and the published contract from drifting apart.

The ``Literal``'s *declaration order* is load-bearing too: it is the market
order inside My Agents' Stocks shelf, mirroring ``MARKET_LABELS`` in
``dashboard/frontend/app.js``. Reorder the members and the Community listing
reorders with it (see :func:`category_sort_rank`).

These are *markets*, not asset classes. My Agents shelves by what an agent
trades (Stocks, Crypto, Futures) and only Stocks is live, so "crypto" and
"futures" are deliberately absent here: those shelves are locked, inert rows in
the frontend with nothing assignable to them. Adding a member would make them
selectable in Configure and cloneable from Community while no bar source,
``MarketProfile`` or engine support exists behind either.
"""
from typing import Literal, Optional, Tuple, get_args

AgentCategory = Literal["us_stocks", "cn_ashares"]

#: Market display order. ``get_args`` preserves the ``Literal``'s declaration
#: order, so this is the one place the ordering is stated -- alphabetical slug
#: order is *not* the intended order ("cn_ashares" would lead).
AGENT_CATEGORY_ORDER: Tuple[str, ...] = get_args(AgentCategory)

AGENT_CATEGORIES = frozenset(AGENT_CATEGORY_ORDER)

# Bounds the value this module echoes into its own error text, which is what
# reaches the log. It does NOT bound the whole 422 body: Pydantic attaches the
# raw input to every validation error under ``detail[].input``, so a caller that
# posts a megabyte still gets a megabyte back. That is app-wide behaviour of the
# default RequestValidationError handler, not specific to this field.
_MAX_ECHO_LENGTH = 50


def coerce_category(value: object) -> Optional[str]:
    """Canonicalize a caller-supplied category, or raise ``ValueError``.

    Empty and whitespace-only input clears the shelf rather than 422-ing: an
    unselected HTML ``<select>`` posts ``""``, not ``null``, and that is the shape
    the Configure form sends. Case and surrounding whitespace are folded. Anything
    else outside the whitelist raises.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"category must be a string or null, got {type(value).__name__}"
        )
    slug = value.strip().lower()
    if not slug:
        return None
    if slug not in AGENT_CATEGORIES:
        raise ValueError(
            f"unknown category {slug[:_MAX_ECHO_LENGTH]!r}; "
            f"allowed: {sorted(AGENT_CATEGORIES)}"
        )
    return slug


def normalize_category(value: object) -> Optional[str]:
    """Lenient counterpart to :func:`coerce_category` — unknown values become ``None``."""
    try:
        return coerce_category(value)
    except ValueError:
        return None


def category_sort_rank(value: object) -> int:
    """Display rank for a category, for ordering shelves and shelved listings.

    Sorting on the raw slug is wrong: the slugs are not alphabetical in shelf
    order, so ``sorted(...)`` would put the A-share shelf first. Unshelved
    (``None``) and unrecognized values rank last rather than first, so a
    template that has not been categorized yet never leads the listing.
    """
    slug = normalize_category(value)
    if slug is None:
        return len(AGENT_CATEGORY_ORDER)
    return AGENT_CATEGORY_ORDER.index(slug)
