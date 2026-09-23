"""Open agent templates for the Agent Supermarket.

Templates are defined in ``dashboard/config/marketplace.json`` so competition
models and hosted agents can be added without schema migrations. The listing
is public; cloning creates a user-owned built-in agent with the template's
pipeline copied in.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

from dashboard.backend.domain.agents.taxonomy import normalize_category
from dashboard.backend.paths import CONFIG_DIR

_MARKETPLACE_PATH = CONFIG_DIR / "marketplace.json"

# Community supermarket rows. Declared order is display order: LLMs first,
# then Agents. Unknown / omitted values fall through ``_normalize_shelf``.
MARKETPLACE_SHELVES = ("llms", "open", "research")


def _normalize_shelf(raw: Dict[str, Any]) -> str:
    """Return ``llms``, ``open`` or ``research``.

    Explicit ``shelf`` on the catalog row wins. Otherwise a non-pipeline
    runtime (today: AI Hedge Fund) is an open agent, so a future hosted
    project does not have to remember the field to land on the right row.
    Research agents (design N2/PR2) are always explicit — they are external
    services, not runtimes this process hosts.
    """
    explicit = str(raw.get("shelf") or "").strip().lower()
    if explicit in MARKETPLACE_SHELVES:
        return explicit
    runtime_type = str(raw.get("runtime_type") or "pipeline")
    return "open" if runtime_type != "pipeline" else "llms"


def shelf_is_research(raw: Dict[str, Any]) -> bool:
    """True when a catalog row is a research-agent service template."""
    return _normalize_shelf(raw) == "research"


def research_service_config(raw: Dict[str, Any]) -> Dict[str, str]:
    """Resolve the service base URL for a research template row.

    The URL comes from the environment named by ``service_base_url_env``
    (deployment-owned) with the catalog's ``service_base_url_default`` as the
    local-dev fallback — same pattern as every other credential in the app.
    """
    import os

    research = raw.get("research") or {}
    env_name = str(research.get("service_base_url_env") or "").strip()
    base = os.getenv(env_name, "") if env_name else ""
    return {
        "base_url": (base or str(research.get("service_base_url_default") or "")).rstrip("/"),
        "agent_id": str(research.get("agent_id") or ""),
    }


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
        "shelf": _normalize_shelf(raw),
        "card_subtitle": str(raw.get("card_subtitle") or "").strip() or None,
        "mode": (
            "runtime"
            if runtime_type != "pipeline"
            else ("simple" if step_count <= 1 else "pipeline")
        ),
    }
    if repo_url.startswith(("https://github.com/", "http://github.com/")):
        public["repo_url"] = repo_url
    # Research rows (N2/PR2) project their service pointer and delivery facts
    # instead of runtime/step fields, which mean nothing for an external
    # Deep Research service.
    research = raw.get("research")
    if shelf_is_research(raw):
        public["mode"] = "research"
        public["model_name"] = raw.get("model_name") or "deep-research"
        public["research"] = {
            "agent_id": research.get("agent_id"),
            "estimated_runtime_seconds": int(research.get("estimated_runtime_seconds") or 300),
            "max_runtime_seconds": int(research.get("max_runtime_seconds") or 1800),
            "output_formats": list(research.get("output_formats") or []),
        }
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
    """Return public marketplace cards grouped by supermarket shelf.

    ``MARKETPLACE_SHELVES`` declaration order is the page order (LLMs, then
    Agents). Within a shelf the catalog's insertion order is preserved
    (stable sort) so Community can list models in the leaderboard roster
    order without a second sort key.
    """
    items = [_public_template(raw) for raw in _load_catalog().values()]
    # _public_template always sets "shelf" via _normalize_shelf, which only
    # returns members of MARKETPLACE_SHELVES -- no fallback branch needed.
    return sorted(items, key=lambda t: MARKETPLACE_SHELVES.index(t["shelf"]))


def get_marketplace_template(template_id: str) -> Optional[Dict[str, Any]]:
    """Return the full template record (including pipeline) or None."""
    return _load_catalog().get(template_id)


def reload_marketplace_catalog() -> None:
    """Clear the in-process catalog cache (tests)."""
    _load_catalog.cache_clear()
