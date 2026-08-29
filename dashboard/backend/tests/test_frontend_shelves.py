"""Guards for the My Agents shelf sections (2026-08-05, updated 2026-08-28).

My Agents used to split agents into two buckets: "Foundation Agents" (whose
subtitle called the product "A prompting game", the #1 trust-killer in every
persona walkthrough) and "External Agents". Asset-class shelves (Stocks /
Crypto / Futures) replaced that, then this task split the live Stocks row into
**LLMs** (pipeline / instruction agents) and **Open Agents** (hosted
runtimes such as AI Hedge Fund), keeping the locked Crypto/Futures rows and
the developer Connected Agents shelf. None of this is enforceable at runtime
-- app.html has no JS test harness -- so, per this suite's frontend convention
(`_frontend_source`), these are asserted against the shipped source directly.
"""

import re
from pathlib import Path

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

_LIVE_SHELVES = [
    (
        "LLMs",
        "Write a trading instruction for an AI model and backtest it on market data.",
    ),
    (
        "Open Agents",
        "Open-source trading agents like AI Hedge Fund. Add them from Community, then customize and backtest.",
    ),
    (
        "For Developers: Connected Agents",
        "Run your own trading program against our backtests. Requires an access key.",
    ),
]

_LOCKED_SHELVES = [
    (
        "crypto",
        "Crypto",
        "Round-the-clock crypto backtesting isn't built yet. Nothing here can be run.",
    ),
    (
        "futures",
        "Futures",
        "Futures contracts aren't built yet. Nothing here can be run.",
    ),
]

_RETIRED_SECTION_HEADERS = (
    "Prompting LLMs",
    "Prompted Models",
    "U.S. Stock Trading",
    "China A-Share Trading",
    "Stocks",
    "Foundation Agents",
)

_CANONICAL_NO_REAL_MONEY_SENTENCE = (
    "Every test here uses simulated money. Real money is involved only if "
    "you explicitly connect a brokerage account and turn on live trading."
)

# Live shelf id suffix -> the AGENT_SHELVES key it corresponds to. Locked rows
# have no ids at all.
_SHELF_SUFFIX_TO_KEY = {
    "Prompted": "prompted",
    "Open": "open",
    "External": "external",
}


def test_live_shelf_headers_and_subtitles_are_present():
    for header, subtitle in _LIVE_SHELVES:
        assert header in _HTML, f"missing shelf header: {header!r}"
        assert subtitle in _HTML, f"missing shelf subtitle: {subtitle!r}"


def test_geographic_section_headers_are_retired():
    """These named the old axes -- geography, asset class, and how the agent
    decides -- and must not survive as section headers anywhere on the page.
    """
    for header in _RETIRED_SECTION_HEADERS:
        assert f">{header}</h3>" not in _HTML, header


def test_foundation_agents_heading_is_gone():
    assert "Foundation Agents" not in _HTML


def test_the_prompting_game_trust_killer_subtitle_is_gone():
    assert "A prompting game" not in _HTML


def test_canonical_no_real_money_sentence_is_present_verbatim():
    assert _CANONICAL_NO_REAL_MONEY_SENTENCE in _HTML


def test_live_sections_with_distinct_shelf_ids():
    """Each live shelf gets its own grid/footer/empty/count id so the render
    loop can address them uniformly: `agentsGrid<Shelf>` /
    `agentsGridFooter<Shelf>` / `agentsEmpty<Shelf>` / `agentsCount<Shelf>`,
    where `<Shelf>` is shelfIdSuffix's PascalCase form of the AGENT_SHELVES key.
    """
    for suffix, key in _SHELF_SUFFIX_TO_KEY.items():
        assert f'data-category="{key}"' in _HTML, key
        assert f'id="agentsGrid{suffix}"' in _HTML, suffix
        assert f'id="agentsGridFooter{suffix}"' in _HTML, suffix
        assert f'id="agentsEmpty{suffix}"' in _HTML, suffix
        assert f'id="agentsCount{suffix}"' in _HTML, suffix


def test_retired_shelf_ids_are_gone():
    """A leftover id would silently double-register an element the render loop
    no longer expects to find.
    """
    for suffix in ("Builtin", "PromptingLlms", "UsStocks", "CnAshares", "Stocks"):
        assert f'id="agentsGrid{suffix}"' not in _HTML, suffix
        assert f'id="agentsGridFooter{suffix}"' not in _HTML, suffix
        assert f'id="agentsEmpty{suffix}"' not in _HTML, suffix


def test_locked_shelves_are_rendered_inert_not_empty():
    """Crypto and Futures have no bar source, no MarketProfile and no engine
    support. They must read as "not built yet", never as "built and broken", so
    they carry aria-disabled and the locked class -- and, critically, none of
    the grid/footer/empty/count/chip elements the render loop addresses. A grid
    element here would make renderAgentCategories' missing-grid guard the only
    thing standing between a stray id and a page that renders nothing.
    """
    for slug, title, subtitle in _LOCKED_SHELVES:
        section_at = _HTML.index(f'data-category="{slug}"')
        section = _HTML[section_at : _HTML.index("</section>", section_at)]
        open_tag = _HTML[max(0, section_at - 120) : section_at + 120]
        assert 'class="agents-category agents-category--locked"' in open_tag, slug
        assert 'aria-disabled="true"' in open_tag, slug
        assert f">{title}</h3>" in section, title
        assert subtitle in section, subtitle
        assert "Not yet available" in section, slug
        assert "agents-grid" not in section, slug
        assert "agentsCount" not in section, slug
        assert "agentsEmpty" not in section, slug


def test_market_chip_container_is_inside_the_prompted_shelf():
    """The chips filter Prompted Models, and they ride #agentsCategories'
    existing delegated click handler -- so they must live inside that container,
    not in the page toolbar above it.
    """
    prompted_at = _HTML.index('data-category="prompted"')
    prompted = _HTML[prompted_at : _HTML.index("</section>", prompted_at)]
    assert 'id="agentsMarketChips"' in prompted
    assert _HTML.index('id="agentsCategories"') < prompted_at


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


def test_agent_shelves_config_holds_only_the_live_shelves():
    """`AGENT_SHELVES` drives rendering, so it lists only shelves that have a
    grid to render into. Crypto and Futures are locked, inert rows in app.html
    with no grid/footer/empty element -- listing them here would force a
    `locked` filter at all five iteration sites, and one missed filter trips
    renderAgentCategories' "some grid is missing" guard, silently aborting the
    whole My Agents render.
    """
    config = js_const("AGENT_SHELVES")
    for key in ("prompted", "open", "external"):
        assert f"key: '{key}'" in config, key
    assert "title: 'LLMs'" in config
    assert "title: 'Prompted Models'" not in config
    for absent in ("stocks", "crypto", "futures", "prompting_llms", "us_stocks", "cn_ashares"):
        assert f"key: '{absent}'" not in config, absent


def test_agent_shelf_key_splits_prompted_from_open():
    """Hosted runtimes (AI Hedge Fund) must not land on Prompted Models.
    runtime_type is always truthy, so the hosted check must be an inequality
    against 'pipeline', never a truthiness test.
    """
    body = _strip_js_comments(fn_body("function agentShelfKey("))
    assert "return 'external'" in body
    assert "return 'open'" in body
    assert "return 'prompted'" in body
    assert "!== 'pipeline'" in body
    assert "return 'stocks'" not in body


def test_market_labels_is_the_only_declaration_of_the_market_names():
    """One map, four consumers: the Prompted Models market chips, the Community
    category chips, the agent-card submeta, and the Configure picker. A second
    hand-typed copy would let one surface drift from the others -- which is
    exactly what the retired SHELF_LABELS existed to prevent, so its name must
    be gone too, not merely unused.
    """
    decl = js_const("MARKET_LABELS")
    assert "us_stocks: 'U.S.'" in decl
    assert "cn_ashares: 'China A-Share'" in decl
    # The bare identifier only. `window.AGENT_SHELF_LABELS` -- the export name
    # agent-editor.js reads, deliberately unchanged so the editor needs no
    # rewiring -- contains "SHELF_LABELS" as a substring, so a plain `not in`
    # here could never pass.
    assert not re.search(r"(?<![A-Z_])SHELF_LABELS", _strip_js_comments(APP_JS))


def test_my_agents_card_submeta_drops_duplicate_model_and_hosted_ai():
    """The card <h3> already names the model. Repeating it on the submeta
    line with 'Hosted AI' was noise. Market stays when known -- the All chip
    still mixes U.S. and China A-Share on this shelf.
    """
    body = _strip_js_comments(fn_body("function renderAgentCards("))
    assert "agentTypeLabel" not in body
    assert "Hosted AI" not in body
    assert "Your own code" not in body
    assert "formatAgentModelLabel(agent.model_name)" not in body
    assert "MARKET_LABELS[agentMarketKey(agent)]" in body
    assert "function agentTypeLabel(" not in APP_JS


def test_render_marketplace_category_chips_is_built_from_the_shared_label_map():
    """The chip row is built from MARKET_LABELS rather than a second hardcoded
    list, plus an 'all' chip that isn't a category at all. It is no longer built
    from AGENT_SHELVES: Community filters templates by *market*, and
    Prompted Models holds both markets, so the shelf list and the chip
    list are different things -- built from AGENT_SHELVES this row would emit a
    single, meaningless "Prompted Models" chip that matches no template.
    """
    body = _strip_js_comments(fn_body("function renderMarketplaceCategoryChips()"))
    assert "MARKET_LABELS" in body
    assert "'all'" in body
    for label in ("U.S.", "China A-Share"):
        assert label not in body, f"{label!r} hardcoded instead of read from MARKET_LABELS"


def test_navigate_to_page_resets_chip_filter_on_plain_community_entry():
    """A category set by one Community visit must not leak into a later,
    unrelated visit made through the plain nav tab -- the most common entry
    path. navigateToPage is the one choke point every Community entry funnels
    through, so the reset belongs there: 'all' unless an explicit
    `communityCategory` option says otherwise. Signature passed to fn_body stops
    at the opening paren, not `(page, options = {})` -- that default value's own
    `{}` would otherwise be mistaken for the function body by fn_body's brace
    matcher.
    """
    body = _strip_js_comments(fn_body("function navigateToPage("))
    assert (
        "marketplaceCategoryFilter = MARKET_LABELS[options.communityCategory] "
        "? options.communityCategory : 'all';"
    ) in body


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


def test_default_agent_is_named_deepseek_v4_pro():
    """Display-name only -- `ensureDefaultFoundationAgent`'s function
    name and the guard-key plumbing it calls are untouched (see the prefix
    pin below); only the string handed to the create-agent API call changes.
    """
    body = _strip_js_comments(fn_body("async function ensureDefaultFoundationAgent("))
    assert "STARTER_AGENTS" in body
    assert "My Trading Agent" not in body
    assert "My Foundation Agent" not in body
    assert js_string_const("DEFAULT_STARTER_AGENT_NAME") == "DeepSeek V4 Pro"


def test_prompted_card_title_follows_the_model_when_name_is_a_catalog_label():
    """The right-hand Claude card was stored as name=DeepSeek V4 Pro. The
    subtitle (model) was right; the <h3> was not. Bound titles must come from
    agentDisplayName, not the raw stored name.
    """
    body = _strip_js_comments(fn_body("function renderAgentCards("))
    assert "agentDisplayName(agent)" in body
    assert "escapeHtml(agent.name)" not in body
    display = _strip_js_comments(fn_body("function agentDisplayName("))
    assert "catalogModelLabels()" in display
    assert "formatAgentModelLabel" in display
    align = _strip_js_comments(fn_body("async function alignStarterAgentNames("))
    assert "agentDisplayName(agent)" in align
    assert "API.patch" in align


def test_ensure_default_fills_missing_starter_models():
    """An account that already has DeepSeek must still receive GPT-5.5 and
    Claude — skipping on `builtins.length` left those cards missing after the
    one-card era.
    """
    body = _strip_js_comments(fn_body("async function ensureDefaultFoundationAgent("))
    assert "STARTER_AGENTS.filter" in body
    assert "openai/gpt-5.5" in js_const("STARTER_AGENTS")
    assert "anthropic/claude-sonnet-4-6" in js_const("STARTER_AGENTS")


def test_signed_in_provision_guard_includes_account_created_at():
    """User ids recycle on ephemeral SQLite. Keying only on `u:${id}` made a
    brand-new account inherit the previous tenant's 'already provisioned'
    stamp and skip the starter card.
    """
    body = _strip_js_comments(fn_body("function defaultAgentProvisionGuardKey("))
    assert "user.created_at" in body
    assert "u:${user.id}:${created}" in body


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


def test_community_page_carries_the_no_real_money_sentence_once():
    """C3 already put this sentence on My Agents' capital controls -- a
    separate, unrelated instance. This checks Community gets its own, and
    exactly one (a second copy on the same page would be visual noise, and
    the brief says "once per page").
    """
    community_html = _community_view_html()
    assert community_html.count(_CANONICAL_NO_REAL_MONEY_SENTENCE) == 1


def _strip_js_comments_from(source: str) -> str:
    """`_strip_js_comments`, but for a file other than app.js."""
    without_block = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", without_block, flags=re.MULTILINE)


_EDITOR_JS_PATH = (
    Path(__file__).resolve().parents[2] / "frontend" / "js" / "agent-editor.js"
)
_EDITOR_JS = _strip_js_comments_from(_EDITOR_JS_PATH.read_text(encoding="utf-8"))


def test_configure_can_move_an_agent_between_shelves():
    """Shelving is only usable if something can set the column.

    Cloning from Community stamps a category, but an agent created directly
    with Add Agent gets NULL and would be pinned to Prompting LLMs forever --
    two of the four shelves unreachable, which reads as broken rather than
    empty. The backend PATCH support exists; this is its caller.
    """
    assert 'id="agentEditorCategorySelect"' in _HTML
    assert 'id="agentEditorCategoryField"' in _HTML
    assert "function fillCategorySelect(agent)" in _EDITOR_JS
    # Sent on save, and reachable from the state object the save path reads.
    assert "payload.category = category" in _EDITOR_JS
    assert "state.category," in _EDITOR_JS


def test_configure_shelf_options_are_not_a_second_hardcoded_list():
    """The <select>'s options come from app.js's MARKET_LABELS -- so renaming a
    market updates the picker, the Community chips, the My Agents market chips
    and the card submeta from one edit. The literal in agent-editor.js is a
    load-failure floor, not the source of truth.

    The exported NAME is deliberately still AGENT_SHELF_LABELS: only what it
    points at changed, so agent-editor.js needed no rewiring."""
    assert "window.AGENT_SHELF_LABELS = MARKET_LABELS;" in APP_JS
    assert "window.AGENT_SHELF_LABELS" in _EDITOR_JS
    assert "SHELF_LABELS_FALLBACK" in _EDITOR_JS


def test_clearing_the_shelf_is_saveable_not_a_no_op():
    """"" is a real choice (the backend folds it to NULL), so a falsy check
    would silently drop it and make "Not set" unselectable once a shelf had
    been picked. Only null/undefined may mean "leave the field alone"."""
    assert "if (category !== null && category !== undefined) payload.category = category;" in _EDITOR_JS


# --- render loop: market chips, count pill, Community button ----------------


def test_empty_shelf_community_cta_is_a_button_not_a_bare_anchor():
    """This is the primary path off an empty shelf, and it shipped as
    `<a href="#">` against a class with no CSS rule anywhere -- so it inherited
    plain link styling and did not read as actionable. The dataset hook the
    delegated handler reads is unchanged; only the element and its class are.
    """
    body = _strip_js_comments(fn_body("function communityShelfButtonHtml("))
    assert '<button type="button"' in body
    assert "agents-empty-community-btn" in body
    assert "data-community-category=" in body
    assert 'href="#"' not in body
    assert "agents-empty-community-link" not in _strip_js_comments(APP_JS)


def test_market_chips_filter_the_grid_but_never_the_count_pill():
    """The pill reports what the shelf HOLDS, read from the unfiltered roster
    (`allAgents`), not what is on screen. A number that moved while you typed in
    the search box or clicked a chip would read as agents disappearing.
    """
    body = _strip_js_comments(fn_body("function renderAgentCategories("))
    assert "allAgents.filter(shelf.match).length" in body
    assert "agentMarketKey(a) === agentMarketFilter" in body
    assert "shelf.key === 'prompted'" in body


def test_market_chip_selection_resets_pagination():
    """The page index is per-shelf, so a page-3 position under 'All' would land
    past the end of a narrower market's single page -- an empty grid with a
    "Page 3 of 1" footer. applyAgentFilters() resets it; applyAgentFilters(false)
    would not.
    """
    body = _strip_js_comments(fn_body("function setAgentMarketFilter("))
    assert "applyAgentFilters()" in body


def test_prompted_empty_state_distinguishes_search_chip_and_truly_empty():
    """Three distinct cases, deliberately worded apart. Collapsing them would
    tell a user who is mid-search, or who clicked a market chip, that they own
    no agents at all -- and Prompted Models is the onboarding surface now (the
    auto-provisioned DeepSeek card lands here), so its true-empty copy is
    the create-your-first voice, not the add-from-Community voice.
    """
    body = _strip_js_comments(fn_body("function promptedEmptyHtml("))
    assert "No agents match your search." in body
    assert "You don't have any agents yet." in body
    assert "communityShelfButtonHtml" in body


def test_open_agents_empty_state_points_at_community():
    """Open Agents is the hosted-runtime shelf, so its empty copy should send
    the user to Community (AI Hedge Fund) rather than the create-a-prompted-model
    onboarding voice.
    """
    body = _strip_js_comments(fn_body("function openAgentsEmptyHtml("))
    assert "No agents match your search." in body
    assert "AI Hedge Fund" in body
    assert "communityShelfButtonHtml" in body
    assert "You don't have any agents yet." not in body


def test_agent_card_identity_opens_the_editor():
    """The card name is the 'click in to edit instructions' path, not only
    the Configure button. Bound on the identity block so Run Backtest and the
    overflow menu keep their own handlers.
    """
    body = _strip_js_comments(fn_body("function renderAgentCards("))
    assert "querySelector('.agent-card-identity')" in body
    assert "window.AgentEditor.open(agent)" in body
    assert "Open to edit instructions" in body

_STYLES_PATH = Path(__file__).resolve().parents[2] / "frontend" / "styles.css"
_STYLES = re.sub(
    r"/\*.*?\*/", "", _STYLES_PATH.read_text(encoding="utf-8"), flags=re.DOTALL
)


def _rule(selector: str) -> str:
    """The declaration block for `selector`'s first rule, comments stripped.

    Scoped to one block so a check for "this shelf has a border" cannot be
    satisfied by an unrelated rule elsewhere in a 9,000-line stylesheet.
    """
    at = _STYLES.index(selector + " {")
    return _STYLES[at : _STYLES.index("}", at)]


def test_shelf_sections_are_real_panels_not_loose_prose():
    """The shipped shelves were five declarations -- a margin, an <h3>, a <p> --
    sitting above bordered cards, so the headers read as page copy rather than
    as a container the cards belong to. A panel needs all of a border, a radius,
    a background and padding to separate from the page beneath it.
    """
    rule = _rule(".agents-category")
    assert "border:" in rule
    assert "border-radius:" in rule
    assert "background:" in rule
    assert "padding:" in rule


def test_locked_shelves_are_visually_disabled_not_just_empty():
    """Dashed and muted, so the row reads as "not built yet" rather than as a
    live shelf that failed to load its cards.
    """
    rule = _rule(".agents-category--locked")
    assert "border-style: dashed" in rule
    assert "opacity:" in rule


def test_nothing_inside_a_locked_shelf_is_clickable():
    """A hover or pointer affordance on a row with nothing behind it converts
    caution into distrust -- the exact failure the locked treatment exists to
    avoid.
    """
    assert ".agents-category--locked * { pointer-events: none; }" in _STYLES


def test_community_cta_button_has_a_rule_of_its_own():
    """The class it replaces (`agents-empty-community-link`) had no rule
    anywhere in this file, which is why the CTA rendered as a plain hyperlink.
    A hover state is what makes it read as pressable.
    """
    assert ".agents-empty-community-btn {" in _STYLES
    assert ".agents-empty-community-btn:hover {" in _STYLES
    assert "agents-empty-community-link" not in _STYLES


def test_market_chips_reuse_the_community_chip_rules():
    """The same taxonomy must look the same on both surfaces, so the chips share
    `.marketplace-category-chip` rather than getting a forked copy that can
    drift. `.agents-market-chips` exists only to space the row inside the shelf
    head.
    """
    assert ".marketplace-category-chip {" in _STYLES
    assert not re.search(r"\.agents-market-chip\s*\{", _STYLES)


def test_agent_card_identity_is_visibly_clickable():
    """Clicking the name opens Configure; without a pointer cursor the
    identity block looks like static chrome.
    """
    rule = _rule(".agent-card--status .agent-card-identity")
    assert "cursor: pointer" in rule
