"""Shared agent category whitelist ("shelves"). Slugs are stored; display names live in the frontend."""
from typing import Optional

AGENT_CATEGORIES = frozenset({"prompting_llms", "us_stocks", "cn_ashares"})


def normalize_category(value: object) -> Optional[str]:
    if value is None:
        return None
    slug = str(value).strip().lower()
    return slug if slug in AGENT_CATEGORIES else None
