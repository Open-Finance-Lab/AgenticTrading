"""Guards for the My Agents shelf sections (2026-08-05).

My Agents used to split agents into two buckets: "Foundation Agents" (whose
subtitle called the product "A prompting game", the #1 trust-killer in every
persona walkthrough) and "External Agents". This task replaces both with four
static shelf sections keyed to the backend category taxonomy
(`dashboard.backend.domain.agents.taxonomy.AGENT_CATEGORIES`), plus the
unchanged external bucket, and adds the canonical no-real-money sentence next
to the capital controls. None of this is enforceable at runtime -- app.html
has no JS test harness -- so, per this suite's frontend convention
(`_frontend_source`), these are asserted against the shipped source directly.
"""

import re

from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, fn_body, js_const, js_string_const


def _strip_html_comments(html: str) -> str:
    """`html` with its `<!-- -->` comments removed.

    The skeleton-loader comment inside each shelf's grid explains *why* the
    markup looks the way it does, in prose that could echo a phrase this
    file asserts is gone. Stripping comments first keeps `in`/`not in`
    assertions reading the live markup, not commentary about it.
    """
    return re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)


_HTML = _strip_html_comments(APP_HTML)

_HEADERS_AND_SUBTITLES = [
    (
        "Prompting LLMs",
        "Prompt state-of-the-art LLMs to backtest on real market data.",
    ),
    (
        "U.S. Stock Trading",
        "Ready-made strategies for U.S. blue-chip stocks, tested hour by hour on real market data.",
    ),
    (
        "China A-Share Trading",
        "Strategies for Chinese A-share stocks, following that market's own next-day (T+1) trading rules.",
    ),
    (
        "For Developers: Connected Agents",
        "Run your own trading program against our backtests. Requires an access key.",
    ),
]

_CANONICAL_NO_REAL_MONEY_SENTENCE = (
    "Every test here uses simulated money. Real money is involved only if "
    "you explicitly connect a brokerage account and turn on live trading."
)

# Shelf id suffix -> the backend category slug it corresponds to. The
# external bucket isn't in AGENT_CATEGORIES (it's a separate builtin/external
# axis, not a trading-domain category) but rides the same id convention.
_SHELF_SUFFIX_TO_CATEGORY_SLUG = {
    "PromptingLlms": "prompting_llms",
    "UsStocks": "us_stocks",
    "CnAshares": "cn_ashares",
    "External": "external",
}


def test_all_four_shelf_headers_and_subtitles_are_present():
    for header, subtitle in _HEADERS_AND_SUBTITLES:
        assert header in _HTML, f"missing shelf header: {header!r}"
        assert subtitle in _HTML, f"missing shelf subtitle: {subtitle!r}"


def test_foundation_agents_heading_is_gone():
    assert "Foundation Agents" not in _HTML


def test_the_prompting_game_trust_killer_subtitle_is_gone():
    assert "A prompting game" not in _HTML


def test_canonical_no_real_money_sentence_is_present_verbatim():
    assert _CANONICAL_NO_REAL_MONEY_SENTENCE in _HTML


def test_four_agents_category_sections_with_distinct_shelf_ids():
    """Each shelf gets its own grid/empty-state/footer id so C3's render loop
    can address them uniformly: `agentsGrid<Shelf>` / `agentsGridFooter<Shelf>`
    / `agentsEmpty<Shelf>`, where `<Shelf>` is the PascalCase form of the
    backend category slug (`prompting_llms` -> `PromptingLlms`, etc.); the
    untouched `external` bucket keeps its pre-existing `External` suffix.
    """
    assert _HTML.count('class="agents-category"') == 4

    for shelf_suffix, category_slug in _SHELF_SUFFIX_TO_CATEGORY_SLUG.items():
        assert f'data-category="{category_slug}"' in _HTML, category_slug
        assert f'id="agentsGrid{shelf_suffix}"' in _HTML, shelf_suffix
        assert f'id="agentsGridFooter{shelf_suffix}"' in _HTML, shelf_suffix
        assert f'id="agentsEmpty{shelf_suffix}"' in _HTML, shelf_suffix


def test_no_leftover_single_builtin_bucket_ids():
    """The old single 'builtin' bucket is gone, not just renamed -- a
    leftover id would silently double-register an element C3's loop no
    longer expects to find.
    """
    assert 'id="agentsGridBuiltin"' not in _HTML
    assert 'id="agentsGridFooterBuiltin"' not in _HTML
    assert 'id="agentsEmptyBuiltin"' not in _HTML


# --- C3: shelf rendering (app.js) -------------------------------------------
#
# C2 built the four static sections above; C3 wires app.js's render loop to
# them. These guards pin the render-loop config and the two renamed strings
# it touches (the default agent's display name, and the retired two-bucket
# empty-state copy) directly against the shipped source, per this suite's
# frontend convention -- app.html/app.js have no JS test harness.


def _strip_js_comments(source: str) -> str:
    """`source` with `//` and `/* */` comments removed.

    `renderAgentCategories`'s doc comments describe the very bucket split
    this task retires ("distinguish 'no agents at all' ... foundation"), so a
    raw `not in` assertion over the function body would read the rationale
    prose instead of the live branches and could pass against unmigrated
    code, or fail against a correctly migrated function that still explains
    its history in a comment.
    """
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL))


def test_agent_shelves_config_has_the_four_shelf_keys():
    """`AGENT_SHELVES` is the declarative config the render loop iterates --
    each of the four shelves must be represented by its exact key.
    """
    config = js_const("AGENT_SHELVES")
    for key in ("prompting_llms", "us_stocks", "cn_ashares", "external"):
        assert f"key: '{key}'" in config, key


def test_no_foundation_agents_copy_is_gone_from_the_render_loop():
    """The old two-bucket ('Foundation'/'External') empty-state copy must not
    survive inside the category render loop -- a leftover string here would
    mean the shelf split is cosmetic (HTML only) and the JS still thinks in
    the old two buckets.
    """
    body = _strip_js_comments(fn_body("function renderAgentCategories("))
    assert "No foundation agents" not in body


def test_my_foundation_agent_display_name_is_gone():
    body = _strip_js_comments(fn_body("async function ensureDefaultFoundationAgent("))
    assert "My Foundation Agent" not in body


def test_my_trading_agent_display_name_is_present():
    """Display-name rename only -- `ensureDefaultFoundationAgent`'s function
    name and the guard-key plumbing it calls are untouched (see the prefix
    pin below); only the string handed to the create-agent API call changes.
    """
    body = _strip_js_comments(fn_body("async function ensureDefaultFoundationAgent("))
    assert "My Trading Agent" in body


def test_default_agent_provision_guard_prefix_is_byte_identical():
    """A changed prefix silently re-provisions a duplicate starter agent for
    every existing user (the guard key no longer matches what was already
    stored), so this pins the literal rather than merely checking presence.
    """
    assert js_string_const("DEFAULT_AGENT_PROVISION_GUARD_PREFIX") == "default-agent-provisioned:"
