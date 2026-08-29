"""Guards for the Community model-vendor facet.

The vendor axis is a pure derivation from `model_name` -- no column, no
migration. MODEL_VENDORS is its single source of truth: chip order, display
label and open/closed licence all come from one table, so a badge cannot drift
from the vendor it describes. A wrong badge is a factual claim about someone
else's product.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dashboard.backend.domain.model_providers.execution_catalog import (
    ATL_EXECUTION_MODELS,
)
from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, fn_body, js_const

_CATALOG = json.loads(
    (Path(__file__).resolve().parents[3] / "dashboard/config/marketplace.json").read_text(
        encoding="utf-8"
    )
)["templates"]

EXPECTED_VENDORS = [
    ("anthropic", "anthropic/", "Claude", "closed"),
    ("openai", "openai/", "GPT", "closed"),
    ("google", "google/", "Gemini", "closed"),
    ("deepseek", "deepseek/", "DeepSeek", "open"),
    ("qwen", "qwen/", "Qwen", "open"),
    ("nvidia", "nvidia/nemotron", "NVIDIA Nemotron", "open"),
    ("meta", "meta-llama/", "Llama", "open"),
    ("xai", "x-ai/", "Grok", "closed"),
]


def _vendor_rows():
    return re.findall(
        r"key:\s*'([^']+)',\s*prefix:\s*'([^']+)',\s*label:\s*'([^']+)',\s*licence:\s*'([^']+)'",
        js_const("MODEL_VENDORS"),
    )


def test_vendor_table_is_pinned_including_licence():
    assert _vendor_rows() == EXPECTED_VENDORS


def test_every_catalog_model_matches_a_vendor_prefix():
    """The highest-value guard: a template on an unmatched prefix renders as
    "AI-powered" with no chip and no badge, which is otherwise invisible."""
    prefixes = [row[1] for row in _vendor_rows()]
    for template in _CATALOG:
        model = template["model_name"].lower()
        assert any(model.startswith(p) for p in prefixes), (
            f"{template['template_id']} runs {model!r}, which matches no MODEL_VENDORS prefix"
        )


def test_every_supported_model_matches_a_vendor_prefix():
    prefixes = [row[1] for row in _vendor_rows()]
    for slug in re.findall(r"slug:\s*'([^']+)'", js_const("SUPPORTED_MODELS")):
        assert any(slug.lower().startswith(p) for p in prefixes), slug


def test_backend_execution_catalog_matches_supported_models():
    frontend_slugs = re.findall(
        r"slug:\s*'([^']+)'",
        js_const("SUPPORTED_MODELS"),
    )

    assert frontend_slugs == [
        model.catalog_id for model in ATL_EXECUTION_MODELS
    ]


def test_supported_model_vendor_fields_agree_with_the_vendor_table():
    """SUPPORTED_MODELS carries its own `vendor` key; it must not drift."""
    by_prefix = {row[1]: row[0] for row in _vendor_rows()}
    pairs = re.findall(
        r"slug:\s*'([^']+)',\s*label:\s*'[^']+',\s*vendor:\s*'([^']+)'",
        js_const("SUPPORTED_MODELS"),
    )
    for slug, vendor in pairs:
        expected = next(k for p, k in by_prefix.items() if slug.lower().startswith(p))
        assert vendor == expected, f"{slug} is tagged {vendor!r}, table says {expected!r}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_provider_label_output_is_unchanged():
    """These six strings ship on cards today. The refactor must not touch them."""
    script = f"""
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
{fn_body("function formatModelProviderLabel")}
const cases = ['anthropic/claude-haiku-4-5', 'nvidia/nemotron-3-nano-30b-a3b',
               'deepseek/deepseek-v4-pro', 'openai/gpt-5.5',
               'google/gemini-3.1-pro-preview', 'qwen/qwen3.7-plus',
               'totally/unknown', ''];
console.log(JSON.stringify(cases.map(formatModelProviderLabel)));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "Powered by Claude",
        "Powered by NVIDIA Nemotron",
        "Powered by DeepSeek",
        "Powered by GPT",
        "Powered by Gemini",
        "Powered by Qwen",
        "AI-powered",
        "AI-powered",
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_unknown_vendor_resolves_to_empty_string():
    """Same contract as agentMarketKey: unknown stays visible under All and is
    excluded only by an explicit chip -- never hidden, never defaulted."""
    script = f"""
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
{fn_body("function agentVendorKey")}
{fn_body("function modelVendorLicence")}
console.log(JSON.stringify([
  modelVendorKey('totally/unknown'), modelVendorKey(null), modelVendorKey('local-model'),
  agentVendorKey({{model_name: 'qwen/qwen3.7-plus'}}), agentVendorKey(null),
  modelVendorLicence('deepseek/deepseek-v4-pro'),
  modelVendorLicence('anthropic/claude-haiku-4-5'),
  modelVendorLicence('totally/unknown'),
]));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["", "", "", "qwen", "", "open", "closed", ""]


def test_vendor_chip_container_exists_in_the_community_view():
    community = APP_HTML[
        APP_HTML.index('<div id="communityView"') : APP_HTML.index('<div id="accountView"')
    ]
    assert 'id="marketplaceVendorChips"' in community
    assert 'id="marketplaceCategoryChips"' in community
    assert community.index('id="marketplaceCategoryChips"') < community.index(
        'id="marketplaceVendorChips"'
    ), "market row must render above the vendor row"


def test_vendor_chips_are_derived_not_hardcoded():
    """Chips come from MODEL_VENDORS intersected with the loaded catalog, so a
    vendor with no templates never ships an empty chip."""
    body = fn_body("function renderMarketplaceVendorChips")
    assert "MODEL_VENDORS" in body
    assert "marketplaceTemplates" in body
    for literal in ("'anthropic'", "'openai'", "'deepseek'", "'qwen'"):
        assert literal not in body, f"{literal} hardcoded in the chip builder"


def test_vendor_chips_are_built_once_then_toggled():
    """renderMarketplaceGrid runs on every search keystroke; rebuilding innerHTML
    per keystroke would blow away the focused chip."""
    body = fn_body("function renderMarketplaceVendorChips")
    assert "existing.length !== chips.length" in body


def test_three_empty_states_stay_distinguishable():
    body = fn_body("function marketplaceEmptyHtml")
    assert "No templates match your search." in body
    assert "No templates match both filters" in body
    assert "marketplace-clear-filters" in body


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_empty_state_precedence():
    script = f"""
function escapeHtml(s) {{ return String(s); }}
const MARKET_LABELS = {{ us_stocks: 'U.S.', cn_ashares: 'China A-Share' }};
{js_const("MODEL_VENDORS")}
{fn_body("function marketplaceEmptyHtml")}
const out = [
  marketplaceEmptyHtml({{searching: true, categoryFilter: 'us_stocks', vendorFilter: 'qwen'}}),
  marketplaceEmptyHtml({{searching: false, categoryFilter: 'us_stocks', vendorFilter: 'qwen'}}),
  marketplaceEmptyHtml({{searching: false, categoryFilter: 'us_stocks', vendorFilter: 'all'}}),
  marketplaceEmptyHtml({{searching: false, categoryFilter: 'all', vendorFilter: 'all'}}),
];
console.log(JSON.stringify(out));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    search_empty, both, one_chip, none_at_all = json.loads(result.stdout)
    # A typed query wins: clearing the chips would not bring anything back.
    assert search_empty == "No templates match your search."
    assert "both filters" in both and "marketplace-clear-filters" in both
    assert "U.S." in one_chip and "both filters" not in one_chip
    assert none_at_all == "No templates match your search."


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_unknown_vendor_survives_the_all_chip_and_only_that_chip():
    script = f"""
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
const templates = [
  {{template_id: 'a', model_name: 'qwen/qwen3.7-plus'}},
  {{template_id: 'b', model_name: 'totally/unknown'}},
];
function visible(filter) {{
  return templates
    .filter((t) => filter === 'all' || modelVendorKey(t.model_name) === filter)
    .map((t) => t.template_id);
}}
console.log(JSON.stringify([visible('all'), visible('qwen')]));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [["a", "b"], ["a"]]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_shipped_filter_ands_market_and_vendor_not_or():
    """Lifts the REAL getFilteredMarketplaceTemplates (not a reimplementation) --
    the six chip/empty-state tests above never execute this function, so a
    regression that drops the vendor filter or ORs it with the market filter
    passed the whole suite until this test existed."""
    script = f"""
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
const document = {{ getElementById: () => null }};
const marketplaceTemplates = [
  {{ template_id: 't1', category: 'us_stocks', model_name: 'anthropic/claude-haiku-4-5' }},
  {{ template_id: 't2', category: 'us_stocks', model_name: 'qwen/qwen3.7-plus' }},
  {{ template_id: 't3', category: 'cn_ashares', model_name: 'anthropic/claude-haiku-4-5' }},
  {{ template_id: 't4', category: 'cn_ashares', model_name: 'qwen/qwen3.7-plus' }},
  {{ template_id: 't5', category: 'us_stocks', model_name: 'totally/unknown' }},
];
let marketplaceCategoryFilter = 'all';
let marketplaceVendorFilter = 'all';

{fn_body("function getFilteredMarketplaceTemplates")}

function ids() {{ return getFilteredMarketplaceTemplates().map((t) => t.template_id); }}

const results = {{}};
marketplaceCategoryFilter = 'us_stocks'; marketplaceVendorFilter = 'all';
results.marketOnly = ids();

marketplaceCategoryFilter = 'all'; marketplaceVendorFilter = 'qwen';
results.vendorOnly = ids();

marketplaceCategoryFilter = 'us_stocks'; marketplaceVendorFilter = 'qwen';
results.both = ids();

marketplaceCategoryFilter = 'all'; marketplaceVendorFilter = 'anthropic';
results.vendorExplicit = ids();

console.log(JSON.stringify(results));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    # Market filter alone: both known-vendor templates plus the unknown-vendor
    # one -- unknown must stay visible when no vendor chip narrows it.
    assert set(data["marketOnly"]) == {"t1", "t2", "t5"}
    # Vendor filter alone: both markets, only the matching vendor.
    assert set(data["vendorOnly"]) == {"t2", "t4"}
    # Both together must be the INTERSECTION (t2 only), not the union
    # (which would also include t1, t4, t5).
    assert data["both"] == ["t2"], "market+vendor must AND, not OR"
    # An explicit vendor chip excludes the unknown-vendor template -- it is
    # visible only under vendor 'all'.
    assert set(data["vendorExplicit"]) == {"t1", "t3"}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_shipped_vendor_chip_order_follows_model_vendors_not_catalog_order():
    """Lifts the REAL renderMarketplaceVendorChips against a synthetic DOM and a
    catalog whose insertion order is deliberately scrambled relative to
    MODEL_VENDORS. test_vendor_chips_are_derived_not_hardcoded only substring
    -checks the function's source text, so it never actually executes this and
    can't tell catalog order from MODEL_VENDORS order."""
    script = f"""
function escapeHtml(s) {{ return String(s); }}
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}

function makeContainer() {{
  let buttons = [];
  return {{
    querySelectorAll() {{ return buttons; }},
    set innerHTML(html) {{
      buttons = [];
      const re = /data-marketplace-vendor="([^"]*)"/g;
      let m;
      while ((m = re.exec(html))) {{
        buttons.push({{
          dataset: {{ marketplaceVendor: m[1] }},
          classList: {{ toggle() {{}} }},
          setAttribute() {{}},
        }});
      }}
    }},
  }};
}}
const container = makeContainer();
const document = {{ getElementById: (id) => (id === 'marketplaceVendorChips' ? container : null) }};

// MODEL_VENDORS order is anthropic, openai, google, deepseek, qwen, nvidia,
// meta, xai. This catalog is inserted qwen, anthropic, deepseek -- scrambled
// on purpose -- plus one unknown-vendor template and no openai template.
const marketplaceTemplates = [
  {{ template_id: 'a', model_name: 'qwen/qwen3.7-plus' }},
  {{ template_id: 'b', model_name: 'anthropic/claude-haiku-4-5' }},
  {{ template_id: 'c', model_name: 'deepseek/deepseek-v4-pro' }},
  {{ template_id: 'd', model_name: 'totally/unknown' }},
];
let marketplaceVendorFilter = 'all';

{fn_body("function renderMarketplaceVendorChips")}
renderMarketplaceVendorChips();

console.log(JSON.stringify(container.querySelectorAll().map((b) => b.dataset.marketplaceVendor)));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    keys = json.loads(result.stdout)
    # 'openai' has no template so it must not get a chip; order must follow
    # MODEL_VENDORS (anthropic, deepseek, qwen), not the catalog's insertion
    # order (qwen, anthropic, deepseek).
    assert keys == ["all", "anthropic", "deepseek", "qwen"]


def test_only_open_weight_models_get_a_badge():
    """Closed models get NOTHING. A "Closed" label reads as a warning about
    someone else's product; absence is not a negative claim."""
    grid = fn_body("function renderMarketplaceGrid")
    assert "modelVendorLicence" in grid
    assert "Open-source model" in grid
    assert "Closed-source" not in APP_JS
    assert "Proprietary" not in APP_JS


def test_licence_badge_has_a_style_rule():
    from dashboard.backend.tests._frontend_source import css_blocks

    assert css_blocks(".marketplace-licence-badge"), "badge has no styles.css rule"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_shipped_grid_badges_only_open_weight_cards():
    """Lifts the REAL renderMarketplaceGrid (not a reimplementation) against a
    synthetic catalog with one open-weight, one closed-weight and one
    unknown-vendor template. The two tests above only substring-check
    renderMarketplaceGrid's source text, so a polarity bug (badging closed
    models instead of open ones) or a "computed but never rendered" bug
    (licenceBadge assigned but not interpolated into the tag row) would both
    pass them silently. This test executes the shipped card template and
    checks the actual HTML each template produces."""
    script = f"""
{fn_body("function escapeHtml")}
{fn_body("function agentRobotIcon")}
const MARKET_LABELS = {{ us_stocks: 'U.S.', cn_ashares: 'China A-Share' }};
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
{fn_body("function modelVendorLicence")}
{fn_body("function formatModelProviderLabel")}

// Collaborators renderMarketplaceGrid calls that render other UI regions --
// stubbed as no-ops so this test stays scoped to the card template.
function renderMarketplaceCategoryChips() {{}}
function renderMarketplaceVendorChips() {{}}

function getFilteredMarketplaceTemplates() {{
  return [
    {{ template_id: 'open1', category: 'us_stocks', name: 'Open Template',
       model_name: 'deepseek/deepseek-v4-pro', tags: [], author: 'Community' }},
    {{ template_id: 'closed1', category: 'us_stocks', name: 'Closed Template',
       model_name: 'anthropic/claude-haiku-4-5', tags: [], author: 'Community' }},
    {{ template_id: 'unknown1', category: 'us_stocks', name: 'Unknown Template',
       model_name: 'totally/unknown', tags: [], author: 'Community' }},
  ];
}}

const cardHtml = [];
const grid = {{
  innerHTML: '',
  appendChild(card) {{ cardHtml.push(card.innerHTML); }},
  querySelectorAll() {{ return []; }},
}};
const document = {{
  getElementById(id) {{ return id === 'marketplaceGrid' ? grid : null; }},
  createElement() {{ return {{ className: '', innerHTML: '' }}; }},
}};

{fn_body("function renderMarketplaceGrid")}
renderMarketplaceGrid();

console.log(JSON.stringify(cardHtml.map((html) => html.includes('marketplace-licence-badge'))));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    # In catalog order: open-weight DeepSeek, closed-weight Claude, unknown vendor.
    assert json.loads(result.stdout) == [True, False, False]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_closed_card_differs_from_open_card_by_exactly_the_badge():
    """"Absence is not a negative claim" is a stronger promise than "no badge
    with this exact class and text" -- the two tests above enumerate two
    guessed forbidden strings ("Closed-source", "Proprietary"), which cannot
    rule out a differently-worded, differently-classed marker on closed cards
    (e.g. a `.marketplace-vendor-note` span reading "Vendor-locked model").

    This renders the REAL renderMarketplaceGrid twice, over two templates that
    are identical except for `model_name` (one open-weight, one closed-weight),
    and diffs the emitted HTML directly: stripping the open-source badge out of
    the open card's HTML must yield the closed card's HTML exactly, modulo the
    one legitimately-differing "Powered by X" label. Any extra marker on the
    closed card -- of any name or wording -- breaks that equality.
    """
    script = f"""
{fn_body("function escapeHtml")}
{fn_body("function agentRobotIcon")}
const MARKET_LABELS = {{ us_stocks: 'U.S.', cn_ashares: 'China A-Share' }};
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
{fn_body("function modelVendorLicence")}
{fn_body("function formatModelProviderLabel")}

function renderMarketplaceCategoryChips() {{}}
function renderMarketplaceVendorChips() {{}}

let currentTemplates;
function getFilteredMarketplaceTemplates() {{ return currentTemplates; }}
let document;

{fn_body("function renderMarketplaceGrid")}

function renderOneCard(modelName) {{
  const cardHtml = [];
  const grid = {{
    innerHTML: '',
    appendChild(card) {{ cardHtml.push(card.innerHTML); }},
    querySelectorAll() {{ return []; }},
  }};
  document = {{
    getElementById(id) {{ return id === 'marketplaceGrid' ? grid : null; }},
    createElement() {{ return {{ className: '', innerHTML: '' }}; }},
  }};
  // Identical in every field except model_name -- the only legitimate
  // difference between the two cards is the licence and the model label.
  // A shared tag keeps the tag-row wrapper div present on BOTH cards, so the
  // only thing that can differ inside it is the badge span itself -- with no
  // tags, the wrapper disappears entirely on the closed card (no badge, no
  // tags -> nothing to wrap) and that wrapper's absence would itself count as
  // a "difference", masking whether an extra marker was also added.
  currentTemplates = [
    {{ template_id: 'x', category: 'us_stocks', name: 'Same Template',
       model_name: modelName, tags: ['sample'], author: 'Community' }},
  ];
  renderMarketplaceGrid();
  return cardHtml[0];
}}

const openHtml = renderOneCard('deepseek/deepseek-v4-pro');
const closedHtml = renderOneCard('anthropic/claude-haiku-4-5');

const BADGE = '<span class="marketplace-licence-badge">Open-source model</span>';
const openHasBadge = openHtml.includes(BADGE);
const closedHasBadge = closedHtml.includes('marketplace-licence-badge');
// Strip only the exact badge span (proves it was actually there) and
// normalise only the one known-legitimate difference (the model label) --
// a blanket strip would hide any other marker instead of catching it.
// Global replace: the label is in both the submeta title attribute and the
// visible text; String.replace(string) would only hit the first.
const openWithoutBadge = openHtml.split(BADGE).join('')
  .split('Powered by DeepSeek').join('POWERED_BY_MODEL');
const closedNormalized = closedHtml
  .split('Powered by Claude').join('POWERED_BY_MODEL');

console.log(JSON.stringify({{
  openHasBadge,
  closedHasBadge,
  equalAfterNormalizing: openWithoutBadge === closedNormalized,
}}));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["openHasBadge"] is True
    assert data["closedHasBadge"] is False
    assert data["equalAfterNormalizing"] is True, (
        "closed card must be byte-identical to the open card minus the badge "
        "(modulo the model label) -- any other marker on the closed card is a "
        "negative claim about someone else's product"
    )


def test_primary_clone_cta_is_unchanged():
    """The conversion click keeps its label and its one-click behaviour."""
    grid = fn_body("function renderMarketplaceGrid")
    assert "const cloneLabel = 'Add to My Agents';" in grid
    assert "marketplace-clone-btn" in grid


def test_model_choice_is_a_secondary_affordance():
    grid = fn_body("function renderMarketplaceGrid")
    assert "marketplace-clone-model-btn" in grid
    assert "Choose model" in grid
    assert "SUPPORTED_MODELS" in grid


def test_clone_sends_the_chosen_model():
    body = fn_body("async function cloneMarketplaceTemplate")
    assert "model_name" in body


def test_clone_menu_changes_only_the_model():
    """A second half-Configure inside a clone menu is how two editing surfaces
    start drifting apart. Name, capital and pipeline stay in Configure."""
    grid = fn_body("function renderMarketplaceGrid")
    menu_start = grid.index("marketplace-model-menu")
    menu = grid[menu_start : menu_start + 800]
    for forbidden in ("cash_allocation", "backtest_allocation", "pipeline", "rename"):
        assert forbidden not in menu


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_model_picker_gated_on_runtime_type_not_truthiness():
    """`runtime_type` is always present and always truthy -- server-defaulted
    to 'pipeline' for every ordinary template (marketplace.py:
    `str(raw.get("runtime_type") or "pipeline")`). A gate written as
    `template.runtime_type ? '' : (...)` is therefore false for EVERY
    template and would hide the picker everywhere while looking correct on a
    substring-only test. The AI Hedge Fund runtime hardcodes its own model
    (infrastructure/ai_hedge_fund/adapter.py) and never reads the stored
    value, so offering a picker there would let a user "choose" a model that
    is silently ignored.

    This executes the REAL renderMarketplaceGrid over one ordinary
    (runtime_type: 'pipeline') template and one hosted (runtime_type:
    'ai_hedge_fund') template and checks which cards actually get a picker."""
    script = f"""
{fn_body("function escapeHtml")}
{fn_body("function agentRobotIcon")}
const MARKET_LABELS = {{ us_stocks: 'U.S.' }};
{js_const("MODEL_VENDORS")}
{js_const("SUPPORTED_MODELS")}
{fn_body("function modelVendorKey")}
{fn_body("function modelVendorLicence")}
{fn_body("function formatModelProviderLabel")}
{fn_body("function normalizeBacktestModelId")}

function renderMarketplaceCategoryChips() {{}}
function renderMarketplaceVendorChips() {{}}

const marketplaceTemplates = [
  {{ template_id: 'ordinary', category: 'us_stocks', name: 'Ordinary Template',
     model_name: 'anthropic/claude-haiku-4-5', tags: [], author: 'Community',
     runtime_type: 'pipeline' }},
  {{ template_id: 'hosted', category: 'us_stocks', name: 'Hosted Template',
     model_name: 'nvidia/nemotron-3-nano-30b-a3b', tags: [], author: 'Community',
     runtime_type: 'ai_hedge_fund' }},
];
function getFilteredMarketplaceTemplates() {{ return marketplaceTemplates; }}
let marketplaceCloneInFlight = false;

const cardHtml = [];
const grid = {{
  innerHTML: '',
  appendChild(card) {{ cardHtml.push(card.innerHTML); }},
  querySelectorAll() {{ return []; }},
}};
const document = {{
  getElementById(id) {{ return id === 'marketplaceGrid' ? grid : null; }},
  createElement() {{ return {{ className: '', innerHTML: '' }}; }},
}};

{fn_body("function renderMarketplaceGrid")}
renderMarketplaceGrid();

console.log(JSON.stringify(cardHtml.map((html) => ({{
  hasModelBtn: html.includes('marketplace-clone-model-btn'),
  hasModelMenu: html.includes('marketplace-model-menu'),
  hasPrimaryBtn: html.includes('marketplace-clone-btn'),
}}))));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    ordinary, hosted = json.loads(result.stdout)
    # Ordinary template: both the picker button AND its menu render.
    assert ordinary["hasModelBtn"] is True
    assert ordinary["hasModelMenu"] is True
    assert ordinary["hasPrimaryBtn"] is True
    # Hosted template: the primary "Add to My Agents" CTA still renders, but
    # neither the picker button nor its menu markup does -- a hidden button
    # with live menu markup is dead weight, not a fix.
    assert hosted["hasModelBtn"] is False
    assert hosted["hasModelMenu"] is False
    assert hosted["hasPrimaryBtn"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_clone_posts_the_chosen_model_and_omits_it_by_default():
    """Executes the REAL cloneMarketplaceTemplate (not a reimplementation).
    test_clone_sends_the_chosen_model only checks that the string
    "model_name" appears somewhere in the function's source -- that would
    pass even if the value were read but never sent, or appeared only in a
    comment. This captures the actual POST body under a stubbed API.post for
    both call shapes: a chosen model (the secondary menu's path) and the
    parameter omitted (the primary CTA's path, whose behaviour must be
    byte-for-byte unchanged: `{{}}`, never a model_name)."""
    script = f"""
const posted = [];
const API = {{
  post: async (url, body) => {{
    posted.push(body);
    return {{ agent: {{ agent_id: 'a1' }} }};
  }},
}};
const API_BASE = '';
function applyActiveAgent(agent) {{}}
async function loadAgents() {{}}
function switchPlaygroundTab(tab) {{}}
const window = {{}};

{fn_body("async function cloneMarketplaceTemplate")}

(async () => {{
  // Real templates always carry a model_name (marketplace.py defaults it to
  // "local-model"), so it is present here too -- a mutation that falls back
  // to template.model_name instead of truly omitting the key must produce a
  // visibly wrong body, not one that happens to serialize the same as {{}}.
  await cloneMarketplaceTemplate({{ template_id: 't1', model_name: 'anthropic/claude-haiku-4-5' }}, 'openai/gpt-5.5');
  await cloneMarketplaceTemplate({{ template_id: 't1', model_name: 'anthropic/claude-haiku-4-5' }});
  console.log(JSON.stringify(posted));
}})();
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    posted = json.loads(result.stdout)
    assert posted == [{"model_name": "openai/gpt-5.5"}, {}], (
        "a chosen model must be sent as {model_name: ...}; an omitted model "
        "must post an empty body exactly as before -- the primary CTA's "
        "one-click behaviour must not change"
    )


def test_duplicate_action_only_on_agents_that_have_run():
    """"Run on another model" is a follow-on offer, not a first action."""
    body = fn_body("function renderAgentCardActions")
    assert "agent-duplicate-model-btn" in body
    assert "'backtested'" in body and "'paper'" in body


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_duplicate_action_gated_on_type_status_and_runtime():
    """test_duplicate_action_only_on_agents_that_have_run only checks that the
    literal strings 'backtested' and 'paper' appear somewhere in the
    function's source -- that stays green even if the gate is built wrong:
    an inverted runtime check, a dropped status check, a dropped agent_type
    check, or a dropped runtime check entirely all leave those substrings in
    place. This lifts the REAL renderAgentCardActions and checks which
    agent/status combinations actually render the button.

    Must fail under each of:
    - M1 (inverted runtime predicate): `!agent.runtime_type` instead of
      `=== 'pipeline'`. runtime_type is always present and always truthy
      (server-defaulted to 'pipeline' for every ordinary agent -- see
      domain/agents/repository.py), so `!agent.runtime_type` is false for
      EVERY ordinary agent and hides the button everywhere. A presence
      assertion on the ordinary pipeline case is the only thing that catches
      this -- an absence-only test cannot.
    - M2 (runtime gate dropped): the button appears on an ai_hedge_fund agent.
    - M3 (status gate dropped): the button appears on a draft agent.
    - M4 (agent_type gate dropped): the button appears on an external agent.
    """
    script = f"""
{fn_body("function escapeHtml")}
{fn_body("function renderAgentCardActions")}

function hasBtn(agent, statusKey) {{
  return renderAgentCardActions(agent, statusKey).includes('agent-duplicate-model-btn');
}}

const pipeline = {{ agent_id: 'a1', agent_type: 'builtin', runtime_type: 'pipeline' }};
const hosted = {{ agent_id: 'a2', agent_type: 'builtin', runtime_type: 'ai_hedge_fund' }};
const external = {{ agent_id: 'a3', agent_type: 'external', runtime_type: 'pipeline' }};

console.log(JSON.stringify({{
  pipelineBacktested: hasBtn(pipeline, 'backtested'),
  pipelinePaper: hasBtn(pipeline, 'paper'),
  pipelineDraft: hasBtn(pipeline, 'draft'),
  hostedBacktested: hasBtn(hosted, 'backtested'),
  externalBacktested: hasBtn(external, 'backtested'),
}}));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    # Ordinary built-in agent that has actually run: the button must be there.
    assert data["pipelineBacktested"] is True
    assert data["pipelinePaper"] is True
    # Not yet run: no follow-on offer yet.
    assert data["pipelineDraft"] is False
    # Hosted runtime hardcodes its own model -- offering a picker would be a
    # false statement about which model actually runs.
    assert data["hostedBacktested"] is False
    # External agents authenticate via API key; duplicating one would mint a
    # new key through a hook that has no reason to do that.
    assert data["externalBacktested"] is False


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_duplicate_offers_every_model_except_the_current_one():
    script = f"""
{js_const("SUPPORTED_MODELS")}
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
{fn_body("function duplicateModelChoices")}
console.log(JSON.stringify([
  duplicateModelChoices({{model_name: 'qwen/qwen3.7-plus'}}).map((m) => m.slug),
  duplicateModelChoices({{model_name: 'local-model'}}).map((m) => m.slug).length,
]));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    without_qwen, legacy_count = json.loads(result.stdout)
    assert "qwen/qwen3.7-plus" not in without_qwen
    assert len(without_qwen) == 5
    # A legacy/hosted model isn't in the list, so nothing is filtered out.
    assert legacy_count == 6


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_duplicate_name_uses_the_vendor_label():
    script = f"""
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
{fn_body("function duplicateAgentName")}
console.log(duplicateAgentName({{name: 'Momentum Alpha'}}, 'deepseek/deepseek-v4-pro'));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Momentum Alpha (DeepSeek)"


def test_duplicate_does_not_start_a_backtest():
    """Auto-firing spends LLM credits on a click the user did not frame as run."""
    body = fn_body("async function submitDuplicateAgent")
    for forbidden in ("runBacktest(", "openRunBacktestModal("):
        assert forbidden not in body


def test_entering_community_resets_the_vendor_filter():
    """A vendor left selected on one visit must not leak into the next.

    `marketplaceCategoryFilter` already resets here, under a comment explaining
    exactly this hazard. The vendor filter was added later and initially did not,
    so returning to Community via the nav tab stayed filtered -- and the My Agents
    empty-shelf deep link (which rides in with a category) then ANDed against the
    stale vendor and landed the user on an empty grid.

    Scoped to the `page === 'community'` branch on purpose: a reset anywhere else
    in the function would not fix the leak, so it must not satisfy this guard.
    """
    body = fn_body("function navigateToPage")
    start = body.index("page === 'community'")
    branch = body[start : body.index("page === 'account'", start)]
    assert re.search(r"marketplaceCategoryFilter\s*=", branch), (
        "the category reset vanished from the community branch"
    )
    assert re.search(r"marketplaceVendorFilter\s*=\s*'all'", branch), (
        "entering Community must reset marketplaceVendorFilter to 'all'; "
        "without it the vendor chip leaks across visits and strands the "
        "empty-shelf deep links on an empty grid"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_duplicate_name_never_exceeds_the_backend_cap():
    """DuplicateAgentBody.name is max_length=100. An over-long generated name
    fails validation, and API.request JSON.stringify's the non-string `detail`,
    so the raw Pydantic array -- including the user's own agent name -- renders
    in the modal's error line. Trim the base, never the vendor suffix."""
    script = f"""
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
{fn_body("function duplicateAgentName")}
const long = 'Q'.repeat(95);
const out = [
  duplicateAgentName({{name: long}}, 'deepseek/deepseek-v4-pro'),
  duplicateAgentName({{name: 'Momentum Alpha'}}, 'deepseek/deepseek-v4-pro'),
  duplicateAgentName({{name: 'Z'.repeat(200)}}, 'totally/unknown'),
];
console.log(JSON.stringify(out.map((s) => [s.length, s.endsWith(')')])));
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    for length, ends_with_vendor in json.loads(result.stdout):
        assert length <= 100, f"generated a {length}-char name; backend caps at 100"
        assert ends_with_vendor, "the vendor suffix was trimmed instead of the base name"


def test_model_picker_accessible_name_contains_its_visible_label():
    """WCAG 2.5.3 Label in Name (Level A): a voice-control user saying
    "click Choose model" must be able to activate the button."""
    grid = fn_body("function renderMarketplaceGrid")
    match = re.search(r'aria-label="([^"]*)"[^>]*>Choose model', grid)
    assert match, "the model-picker button lost its aria-label or its visible text"
    assert match.group(1).lower().startswith("choose model"), (
        f"accessible name {match.group(1)!r} does not contain the visible label 'Choose model'"
    )
