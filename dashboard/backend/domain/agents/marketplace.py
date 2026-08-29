"""Open agent templates for the Agent Supermarket.

Templates are defined in ``dashboard/config/marketplace.json`` so baseline open
agents can be added without schema migrations. The listing is public; cloning
creates a user-owned built-in agent with the template's pipeline copied in.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

from dashboard.backend.domain.agents.taxonomy import (
    category_sort_rank,
    normalize_category,
)
from dashboard.backend.paths import CONFIG_DIR

_MARKETPLACE_PATH = CONFIG_DIR / "marketplace.json"


def _public_template(raw: Dict[str, Any]) -> Dict[str, Any]:
    # This "category" and an agent's "category" used to be two different
    # vocabularies under one key name: templates carried display strings
    # ("Foundation"/"Advanced"/"Hosted", defaulted to "General"), agents carried
    # ``taxonomy.AgentCategory`` slugs or NULL. The catalog has since been
    # recategorized onto the slugs, and the projection normalizes through the
    # taxonomy so the two are now provably one vocabulary -- the frontend can
    # map template and agent categories through a single label table.
    #
    # The "General" default is gone deliberately: it would be a fourth
    # out-of-vocabulary value that ``normalize_category`` silently maps to None,
    # which reads as a real shelf on the card but filters as unshelved.
    # ``None`` (rendered as a generic label by the frontend) is the honest shape.
    pipeline = raw.get("pipeline")
    step_count = len(pipeline) if isinstance(pipeline, list) else 0
    runtime_type = str(raw.get("runtime_type") or "pipeline")
    repo_url = str(raw.get("repo_url") or "").strip()
    public = {
        "template_id": raw["template_id"],
        "name": raw["name"],
        "model_name": raw.get("model_name") or "local-model",
        "description": raw.get("description"),
        "category": normalize_category(raw.get("category")),
        "tags": list(raw.get("tags") or []),
        "author": raw.get("author") or "Community",
        "runtime_type": runtime_type,
        "step_count": step_count,
        "mode": (
            "runtime"
            if runtime_type != "pipeline"
            else ("simple" if step_count <= 1 else "pipeline")
        ),
    }
    if repo_url.startswith(("https://github.com/", "http://github.com/")):
        public["repo_url"] = repo_url
    return public


@lru_cache(maxsize=1)
def _load_catalog() -> Dict[str, Dict[str, Any]]:
    if not _MARKETPLACE_PATH.is_file():
        return {}
    try:
        payload = json.loads(_MARKETPLACE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    templates = payload.get("templates") if isinstance(payload, dict) else None
    if not isinstance(templates, list):
        return {}
    catalog: Dict[str, Dict[str, Any]] = {}
    for item in templates:
        if not isinstance(item, dict):
            continue
        template_id = str(item.get("template_id") or "").strip()
        name = str(item.get("name") or "").strip()
        if not template_id or not name:
            continue
        catalog[template_id] = item
    return catalog


def list_marketplace_templates() -> List[Dict[str, Any]]:
    """Return public marketplace cards grouped by shelf, then sorted by name.

    Ordered by ``category_sort_rank`` rather than by the slug itself: the slugs
    are not alphabetical in shelf order, so a plain ``sorted`` on the raw value
    leads the Community listing with the A-share shelf. Uncategorized templates
    sort last.
    """
    items = [_public_template(raw) for raw in _load_catalog().values()]
    return sorted(
        items,
        key=lambda t: (category_sort_rank(t.get("category")), str(t.get("name") or "")),
    )


def get_marketplace_template(template_id: str) -> Optional[Dict[str, Any]]:
    """Return the full template record (including pipeline) or None."""
    return _load_catalog().get(template_id)


def reload_marketplace_catalog() -> None:
    """Clear the in-process catalog cache (tests)."""
    _load_catalog.cache_clear()
