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
"""
from typing import Literal, Optional, get_args

AgentCategory = Literal["prompting_llms", "us_stocks", "cn_ashares"]

AGENT_CATEGORIES = frozenset(get_args(AgentCategory))

# Bounds the value echoed back in the 422 detail. A caller can post a megabyte
# here; reflecting it verbatim would put that megabyte in the error body and the
# request log.
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
