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


# --- C4: Community category chips + CTA verb + label map --------------------
#
# C1 categorized the marketplace catalog; C3 built the four My Agents shelves
# and left a data-community-category hook on two of their empty-state links
# that, until this task, only opened Community without reading the category.
# This task adds the chip row that filters Community by that same taxonomy,
# unifies the "Add to My Agents" CTA (PR #253 already made it canonical
# everywhere except one AI-Hedge-Fund-scoped ternary), routes the card
# submeta through shared label tables instead of raw slugs, and finishes
# wiring C3's hook.

_MARKETPLACE_RENDER_FN = "function renderMarketplaceGrid()"


def _community_view_html() -> str:
    """The `communityView` page's markup, isolated from the rest of app.html.

    `id="accountView"` is the next page-view div after it in source order
    (same marker `test_frontend_marketplace_placement.py` uses), so this is a
    safe end bound without a full HTML parser.
    """
    start = _HTML.index('id="communityView"')
    end = _HTML.index('id="accountView"', start)
    return _HTML[start:end]


def test_copy_to_my_agents_cta_is_gone():
    """"Copy to My Agents" was scoped to the AI Hedge Fund template only.
    PR #253 made "Add to My Agents" canonical everywhere else, so this one
    holdout ternary must go, not gain a permanent sibling.
    """
    body = _strip_js_comments(fn_body(_MARKETPLACE_RENDER_FN))
    assert "Copy to My Agents" not in body


def test_add_to_my_agents_cta_is_a_single_unconditional_string():
    """The CTA must not branch per-template -- a ternary whose two branches
    happen to read the same today is still two code paths that can drift
    apart again tomorrow. Assert the direct, unconditional assignment and
    that the now-dead branch variable is gone with it.
    """
    body = _strip_js_comments(fn_body(_MARKETPLACE_RENDER_FN))
    assert "cloneLabel = 'Add to My Agents'" in body
    assert "isAiHedgeFundTemplate" not in body


def test_shelf_labels_map_is_derived_from_agent_shelves_not_duplicated():
    """SHELF_LABELS must be *built from* AGENT_SHELVES' `title` field, not a
    second hand-typed copy of the same three strings -- a hand-typed copy can
    silently drift from the shelf headers it's supposed to mirror.
    """
    decl = js_const("SHELF_LABELS")
    assert "AGENT_SHELVES" in decl
    for title in ("Prompting LLMs", "U.S. Stock Trading", "China A-Share Trading"):
        assert title not in decl, f"{title!r} is hardcoded in SHELF_LABELS instead of derived"


def test_marketplace_submeta_never_renders_a_raw_category_or_model_slug():
    """The card submeta line must route the category and model name through
    shared label tables, never `template.category`/`template.model_name`
    raw. Scoped to just the submeta template-literal line -- the
    provider-label lookup table legitimately contains the same
    'anthropic/'/'nvidia/' prefix strings elsewhere in the function, so a
    whole-function check would false-positive on the table doing its job.
    """
    body = _strip_js_comments(fn_body(_MARKETPLACE_RENDER_FN))
    submeta_line = next(line for line in body.splitlines() if "agent-card-submeta" in line)
    assert "template.category" not in submeta_line
    assert "template.model_name" not in submeta_line
    assert not re.search(r"nvidia/|anthropic/", submeta_line)


def test_fallback_description_copy_is_updated():
    body = _strip_js_comments(fn_body(_MARKETPLACE_RENDER_FN))
    assert "Open agent template." not in body
    assert "No description provided yet." in body


def test_render_marketplace_category_chips_covers_all_plus_the_three_categories():
    """The chip row is built from AGENT_SHELVES (minus 'external', which
    isn't a template category) rather than a second hardcoded list, plus an
    'all' chip that isn't in AGENT_SHELVES at all.
    """
    body = _strip_js_comments(fn_body("function renderMarketplaceCategoryChips()"))
    assert "AGENT_SHELVES" in body
    assert "'external'" in body
    assert "'all'" in body


def test_marketplace_category_chip_container_is_present_in_community_view():
    assert 'id="marketplaceCategoryChips"' in _community_view_html()


def test_community_link_hook_reads_the_dataset_category():
    """C3 left this handler only opening Community; the category it read off
    the clicked link's dataset was unused. Comments already named the
    identifier this test checks for (as a note-to-self for this task), so
    the assertion runs on the comment-stripped body -- otherwise it would
    pass against the leftover comment instead of real code.
    """
    body = _strip_js_comments(fn_body("function initNavigation()"))
    assert "communityLink.dataset.communityCategory" in body


def test_community_link_hook_routes_the_category_through_navigate_to_page():
    """Fixed 2026-08-05 (review round 1): the hook originally called
    setMarketplaceCategoryFilter directly, then navigateToPage('community')
    with no options. That worked for the one visit it fired on, but left
    marketplaceCategoryFilter sticky module state -- a later, unrelated
    Community visit through the plain nav tab silently inherited whatever
    category a previous empty-shelf link had set. navigateToPage is now the
    one place that resets the filter on entry (see the next test), so the
    category must ride the same call as an option rather than be set
    beforehand and immediately overwritten.
    """
    body = _strip_js_comments(fn_body("function initNavigation()"))
    assert (
        "navigateToPage('community', { communityCategory: communityLink.dataset.communityCategory })"
        in body
    )


def test_navigate_to_page_resets_chip_filter_on_plain_community_entry():
    """A category set by one Community visit must not leak into a later,
    unrelated visit made through the plain nav tab -- the most common entry
    path. navigateToPage is the one choke point every Community entry
    funnels through (it already redirects the retired Playground marketplace
    subtab here), so the reset belongs there: 'all' unless an explicit
    `communityCategory` option says otherwise. Signature passed to fn_body
    stops at the opening paren, not `(page, options = {})` -- that default
    value's own `{}` would otherwise be mistaken for the function body by
    fn_body's brace matcher (its docstring calls out this exact pattern).
    """
    body = _strip_js_comments(fn_body("function navigateToPage("))
    assert (
        "marketplaceCategoryFilter = SHELF_LABELS[options.communityCategory] "
        "? options.communityCategory : 'all';"
    ) in body


def test_community_page_carries_the_no_real_money_sentence_once():
    """C3 already put this sentence on My Agents' capital controls -- a
    separate, unrelated instance. This checks Community gets its own, and
    exactly one (a second copy on the same page would be visual noise, and
    the brief says "once per page").
    """
    community_html = _community_view_html()
    assert community_html.count(_CANONICAL_NO_REAL_MONEY_SENTENCE) == 1
