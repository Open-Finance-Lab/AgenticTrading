"""Guards for the Community model-vendor table.

The vendor axis is a pure derivation from `model_name` -- no column, no
migration. MODEL_VENDORS is its single source of truth: the pinned key/label
rows, the LLM tile's company submeta and the (currently unrendered) licence
metadata all come from one table, so an entry cannot drift from the vendor it
describes. A wrong entry is a factual claim about someone else's product.
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
    """Model-vendor chips were removed: the LLM shelf already names each model."""
    community = APP_HTML[
        APP_HTML.index('<div id="communityView"') : APP_HTML.index('<div id="accountView"')
    ]
    assert 'id="marketplaceVendorChips"' not in community
    assert 'id="marketplaceCategoryChips"' in community


def test_vendor_chip_machinery_is_fully_gone():
    """The chip row's container left app.html with this redesign; the renderer,
    its filter state and the both-filters empty state must not outlive it as
    silent no-ops."""
    for remnant in (
        "renderMarketplaceVendorChips",
        "marketplaceVendorFilter",
        "setMarketplaceVendorFilter",
        "marketplace-clear-filters",
    ):
        assert remnant not in APP_JS, remnant


def test_empty_states_stay_distinguishable():
    """Search-empty vs market-empty keep distinct copy (the vendor chips are
    gone, and the old both-filters branch went with them)."""
    body = fn_body("function marketplaceEmptyHtml")
    assert "No templates match your search." in body
    assert "templates yet." in body
    assert "vendorFilter" not in body


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_empty_state_precedence():
    script = f"""
function escapeHtml(s) {{ return String(s); }}
const MARKET_LABELS = {{ us_stocks: 'U.S.', cn_ashares: 'China A-Share' }};
{fn_body("function marketplaceEmptyHtml")}
const out = [
  marketplaceEmptyHtml({{searching: true, categoryFilter: 'us_stocks'}}),
  marketplaceEmptyHtml({{searching: false, categoryFilter: 'us_stocks'}}),
  marketplaceEmptyHtml({{searching: false, categoryFilter: 'all'}}),
];
console.log(JSON.stringify(out));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    search_empty, market_only, none_at_all = json.loads(result.stdout)
    # A typed query wins: clearing the chip would not bring anything back.
    assert search_empty == "No templates match your search."
    assert "U.S." in market_only and market_only.endswith("templates yet.")
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
def test_marketplace_market_chips_only_emit_markets_present_in_the_catalog():
    """China A-Share must not ship as a permanent empty-state chip. Adding a
    cn_ashares template is what brings the chip back; the helper must not
    hardcode either market name."""
    script = f"""
const MARKET_LABELS = {{ us_stocks: 'U.S.', cn_ashares: 'China A-Share' }};
{fn_body("function marketplaceMarketChips")}
const keys = (templates) => marketplaceMarketChips(templates).map((c) => c.key);
const labels = (templates) => marketplaceMarketChips(templates).map((c) => c.label);
console.log(JSON.stringify({{
  usOnly: keys([{{category: 'us_stocks'}}, {{category: 'us_stocks'}}]),
  mixed: keys([{{category: 'cn_ashares'}}, {{category: 'us_stocks'}}]),
  empty: keys([]),
  unknown: keys([{{category: 'crypto'}}]),
  usLabels: labels([{{category: 'us_stocks'}}]),
}}));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["usOnly"] == ["all", "us_stocks"]
    assert data["mixed"] == ["all", "us_stocks", "cn_ashares"]
    assert data["empty"] == ["all"]
    assert data["unknown"] == ["all"]
    assert data["usLabels"] == ["All", "U.S."]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_shipped_filter_ands_market_and_vendor_not_or():
    """Vendor chips are gone; the market filter still applies to both shelves."""
    script = f"""
const document = {{ getElementById: () => null }};
const marketplaceTemplates = [
  {{ template_id: 't1', category: 'us_stocks', model_name: 'anthropic/claude-haiku-4-5' }},
  {{ template_id: 't2', category: 'us_stocks', model_name: 'qwen/qwen3.7-plus' }},
  {{ template_id: 't3', category: 'cn_ashares', model_name: 'anthropic/claude-haiku-4-5' }},
  {{ template_id: 't4', category: 'cn_ashares', model_name: 'qwen/qwen3.7-plus' }},
  {{ template_id: 't5', category: 'us_stocks', model_name: 'totally/unknown' }},
];
let marketplaceCategoryFilter = 'all';

{fn_body("function getFilteredMarketplaceTemplates")}

function ids() {{ return getFilteredMarketplaceTemplates().map((t) => t.template_id); }}

const results = {{}};
marketplaceCategoryFilter = 'us_stocks';
results.marketUs = ids();
marketplaceCategoryFilter = 'cn_ashares';
results.marketCn = ids();
marketplaceCategoryFilter = 'all';
results.marketAll = ids();
console.log(JSON.stringify(results));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert set(data["marketUs"]) == {"t1", "t2", "t5"}
    assert set(data["marketCn"]) == {"t3", "t4"}
    assert set(data["marketAll"]) == {"t1", "t2", "t3", "t4", "t5"}


def test_only_open_weight_models_get_a_badge():
    """LLM tiles no longer claim licence. Hosted repo cards use an explicit
    Open Source mark; closed models and strategy templates get nothing."""
    card = fn_body("function buildMarketplaceCardHtml")
    assert "Open Source" in card
    assert "Open-source model" not in card
    assert ">Open</span>" not in card
    assert "Closed-source" not in APP_JS
    assert "Proprietary" not in APP_JS


_CARD_HELPERS = f"""
{fn_body("function escapeHtml")}
{fn_body("function agentRobotIcon")}
const MARKET_LABELS = {{ us_stocks: 'U.S.', cn_ashares: 'China A-Share' }};
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
{fn_body("function formatModelCompanyLabel")}
let marketplaceLeaderboardEntries = [];
let marketplaceContestMeta = {{ start_date: null, end_date: null, display_capital: null, total_entries: null }};
{fn_body("function templateMarketplaceShelf")}
{fn_body("function findMarketplaceLeaderboardEntry")}
{js_const("MARKETPLACE_MONTHS")}
{fn_body("function marketplaceBenchmarkEntry")}
{fn_body("function downsampleMarketplaceCurve")}
{fn_body("function marketplaceIndexedPctSeries")}
{fn_body("function formatMarketplaceMd")}
{fn_body("function formatMarketplaceWindowRange")}
{fn_body("function formatMarketplaceCapital")}
{fn_body("function marketplaceNicePctTicks")}
{fn_body("function marketplaceLinePath")}
{fn_body("function buildMarketplaceCompareChartHtml")}
{fn_body("function marketplacePerformanceFor")}
{fn_body("function formatMarketplaceReturnPct")}
{fn_body("function marketplaceRepoLabel")}
{fn_body("function compareMarketplaceTemplatesByRank")}
{fn_body("function buildMarketplaceCardHtml")}
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_shipped_grid_badges_only_open_weight_cards():
    """Open Source is a repo-card mark, not an Agents-shelf or open-weight claim."""
    script = f"""
{_CARD_HELPERS}
const cards = [
  buildMarketplaceCardHtml({{ template_id: 'llm', shelf: 'llms', category: 'us_stocks',
    name: 'DeepSeek V4 Pro', model_name: 'deepseek/deepseek-v4-pro' }}),
  buildMarketplaceCardHtml({{ template_id: 'llm2', shelf: 'llms', category: 'us_stocks',
    name: 'Claude Haiku 4.5', model_name: 'anthropic/claude-haiku-4-5' }}),
  buildMarketplaceCardHtml({{ template_id: 'ai-hedge-fund', shelf: 'open', category: 'us_stocks',
    name: 'AI Hedge Fund', model_name: 'nvidia/nemotron-3-nano-30b-a3b',
    card_subtitle: 'Open-source multi-agent system',
    repo_url: 'https://github.com/virattt/ai-hedge-fund' }}),
  buildMarketplaceCardHtml({{ template_id: 'balanced-starter', shelf: 'open', category: 'us_stocks',
    name: 'Balanced Starter', model_name: 'anthropic/claude-haiku-4-5',
    description: 'A simple starter agent that diversifies across strong stocks.' }}),
];
console.log(JSON.stringify(cards.map((html) => html.includes('Open Source'))));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [False, False, True, False]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_closed_card_differs_from_open_card_by_exactly_the_badge():
    """Two LLM cards that differ only by model_name must differ only by company."""
    script = f"""
{_CARD_HELPERS}
function renderOneCard(modelName) {{
  return buildMarketplaceCardHtml({{
    template_id: 'x', shelf: 'llms', category: 'us_stocks',
    name: 'Same Template', model_name: modelName,
  }});
}}
const openHtml = renderOneCard('deepseek/deepseek-v4-pro');
const closedHtml = renderOneCard('anthropic/claude-haiku-4-5');
const openNorm = openHtml.split('DeepSeek').join('COMPANY');
const closedNorm = closedHtml.split('Anthropic').join('COMPANY');
console.log(JSON.stringify({{
  openHasBadge: openHtml.includes('Open Source'),
  closedHasBadge: closedHtml.includes('Open Source'),
  equalAfterNormalizing: openNorm === closedNorm,
}}));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["openHasBadge"] is False
    assert data["closedHasBadge"] is False
    assert data["equalAfterNormalizing"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_card_uses_overall_rank_and_return_without_placeholders():
    """Contest stats: trophy rank, Return, model vs DJIA cumulative-return chart."""
    script = f"""
{_CARD_HELPERS}
marketplaceLeaderboardEntries = [
  {{ is_model: true, model: 'Claude Haiku 4.5', entry_id: 'claude_haiku_4_5',
     rank: 2, cumulative_return: 0.084,
     equity_curve: [
       {{ timestamp: '2026-04-15T00:00:00+00:00', equity: 10000 }},
       {{ timestamp: '2026-04-30T00:00:00+00:00', equity: 10400 }},
       {{ timestamp: '2026-05-15T00:00:00+00:00', equity: 10840 }}
     ] }},
  {{ is_model: false, model: 'DJIA', entry_id: 'djia_index', rank: 8,
     cumulative_return: 0.022,
     equity_curve: [
       {{ timestamp: '2026-04-15T00:00:00+00:00', equity: 10000 }},
       {{ timestamp: '2026-04-30T00:00:00+00:00', equity: 10100 }},
       {{ timestamp: '2026-05-15T00:00:00+00:00', equity: 10224 }}
     ] }},
];
marketplaceContestMeta = {{
  start_date: '2026-04-15', end_date: '2026-05-15',
  display_capital: 10000, total_entries: 12,
}};
const ranked = buildMarketplaceCardHtml({{
  template_id: 'claude-haiku-4-5', shelf: 'llms', category: 'us_stocks',
  name: 'Claude Haiku 4.5', model_name: 'anthropic/claude-haiku-4-5',
}});
const unranked = buildMarketplaceCardHtml({{
  template_id: 'ai-hedge-fund', shelf: 'open', category: 'us_stocks',
  name: 'AI Hedge Fund', model_name: 'nvidia/nemotron-3-nano-30b-a3b',
  card_subtitle: 'Open-source multi-agent system',
  description: 'A team of AI investors that analyzes the market, develops trading ideas, and tests them through backtesting.',
  repo_url: 'https://github.com/virattt/ai-hedge-fund',
}});
console.log(JSON.stringify({{
  rankedHasMedian: ranked.includes('Median Return'),
  rankedHasLeaderboardReturn: ranked.includes('Leaderboard Return'),
  rankedHasOverallRank: ranked.includes('Overall Rank'),
  rankedHasCompetition: ranked.includes('Competition result'),
  rankedHasTotalReturn: ranked.includes('Total Return'),
  rankedHasReturnLabel: ranked.includes('>Return<'),
  rankedHasHashTwo: ranked.includes('#2 of 12'),
  rankedHasPct: ranked.includes('+8.4%'),
  rankedHasChart: ranked.includes('mp-compare-chart'),
  rankedHasAgentLegend: ranked.includes('Claude Haiku 4.5'),
  rankedHasDjiaLegend: ranked.includes('>DJIA<') || ranked.includes('DJIA</span>'),
  rankedHasDjiaBenchmark: ranked.includes('DJIA Benchmark'),
  rankedHasAgentPortfolio: ranked.includes('Agent Portfolio'),
  rankedHasMeta: ranked.includes('DJIA 30'),
  rankedHasWindow: ranked.includes('Apr 15–May 15'),
  rankedHasCapital: ranked.includes('$10K'),
  rankedHasHead: ranked.includes('mp-competition-head'),
  rankedMetaBeforeBody: ranked.indexOf('DJIA 30') < ranked.indexOf('mp-competition-body')
    && ranked.indexOf('Competition result') < ranked.indexOf('mp-competition-body'),
  rankedHasComingSoon: ranked.includes('Performance data coming soon'),
  unrankedHasCompetition: unranked.includes('Competition result'),
  unrankedHasSpark: unranked.includes('mp-compare-chart'),
  unrankedHasRepo: unranked.includes('virattt/ai-hedge-fund'),
  unrankedHasSubtitle: unranked.includes('Open-source multi-agent system'),
  unrankedHasOverallRank: unranked.includes('Overall Rank'),
  unrankedHasDescription: unranked.includes('A team of AI investors that analyzes the market'),
}}));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["rankedHasMedian"] is False
    assert data["rankedHasLeaderboardReturn"] is False
    assert data["rankedHasOverallRank"] is False
    assert data["rankedHasCompetition"] is True
    assert data["rankedHasTotalReturn"] is False
    assert data["rankedHasReturnLabel"] is True
    assert data["rankedHasHashTwo"] is True
    assert data["rankedHasPct"] is True
    assert data["rankedHasChart"] is True
    assert data["rankedHasAgentLegend"] is True
    assert data["rankedHasDjiaLegend"] is True
    assert data["rankedHasDjiaBenchmark"] is False
    assert data["rankedHasAgentPortfolio"] is False
    assert data["rankedHasMeta"] is True
    assert data["rankedHasWindow"] is True
    assert data["rankedHasCapital"] is True
    assert data["rankedHasHead"] is True
    assert data["rankedMetaBeforeBody"] is True
    assert data["rankedHasComingSoon"] is False
    assert data["unrankedHasCompetition"] is False
    assert data["unrankedHasSpark"] is False
    assert data["unrankedHasRepo"] is True
    assert data["unrankedHasSubtitle"] is True
    assert data["unrankedHasOverallRank"] is False
    assert data["unrankedHasDescription"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_llm_shelf_sorts_by_leaderboard_rank():
    script = f"""
{_CARD_HELPERS}
marketplaceLeaderboardEntries = [
  {{ is_model: true, model: 'Claude Haiku 4.5', rank: 11, cumulative_return: 0.0 }},
  {{ is_model: true, model: 'DeepSeek V4 Pro', rank: 1, cumulative_return: 0.07 }},
  {{ is_model: true, model: 'Qwen3.7 Plus', rank: 5, cumulative_return: 0.02 }},
];
const cards = [
  {{ template_id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5' }},
  {{ template_id: 'deepseek-v4-pro', name: 'DeepSeek V4 Pro' }},
  {{ template_id: 'qwen3-7-plus', name: 'Qwen3.7 Plus' }},
  {{ template_id: 'unknown-llm', name: 'Unlisted' }},
];
const sorted = cards.slice().sort(compareMarketplaceTemplatesByRank).map((t) => t.template_id);
console.log(JSON.stringify(sorted));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "deepseek-v4-pro",
        "qwen3-7-plus",
        "claude-haiku-4-5",
        "unknown-llm",
    ]


def test_marketplace_card_has_no_fixed_min_height():
    from dashboard.backend.tests._frontend_source import css_blocks

    blocks = css_blocks(".marketplace-card")
    assert blocks, "marketplace-card has no styles.css rule"
    assert all("min-height" not in block for block in blocks)


def test_llm_grid_sorts_by_leaderboard_rank():
    grid = fn_body("function renderMarketplaceGrid")
    assert "compareMarketplaceTemplatesByRank" in grid
    assert "shelf.key === 'llms'" in grid


def test_primary_clone_cta_is_unchanged():
    """The conversion click keeps its label and its one-click behaviour."""
    card = fn_body("function buildMarketplaceCardHtml")
    assert "const cloneLabel = 'Add to My Agents';" in card
    assert "marketplace-clone-btn" in card


def test_model_choice_is_not_on_the_card():
    """Tiles are the model. Switching models happens in Configure after Add."""
    card = fn_body("function buildMarketplaceCardHtml")
    grid = fn_body("function renderMarketplaceGrid")
    for body in (card, grid):
        assert "marketplace-clone-model-btn" not in body
        assert "Choose model" not in body
        assert "marketplace-model-menu" not in body


def test_clone_sends_the_chosen_model():
    body = fn_body("async function cloneMarketplaceTemplate")
    assert "model_name" in body


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_model_picker_gated_on_runtime_type_not_truthiness():
    """Neither shelf card offers Choose model; both keep Add to My Agents."""
    script = f"""
{_CARD_HELPERS}
const ordinary = buildMarketplaceCardHtml({{
  template_id: 'ordinary', shelf: 'llms', category: 'us_stocks',
  name: 'Ordinary Template', model_name: 'anthropic/claude-haiku-4-5',
  runtime_type: 'pipeline',
}});
const hosted = buildMarketplaceCardHtml({{
  template_id: 'hosted', shelf: 'open', category: 'us_stocks',
  name: 'Hosted Template', model_name: 'nvidia/nemotron-3-nano-30b-a3b',
  runtime_type: 'ai_hedge_fund',
}});
console.log(JSON.stringify([ordinary, hosted].map((html) => ({{
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
    assert ordinary["hasModelBtn"] is False
    assert ordinary["hasModelMenu"] is False
    assert ordinary["hasPrimaryBtn"] is True
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


def test_entering_community_resets_the_category_filter():
    """A chip left selected on one visit must not leak into the next.

    Scoped to the `page === 'community'` branch on purpose: a reset anywhere
    else in the function would not fix the leak, so it must not satisfy this
    guard. (The vendor filter that used to reset here left with the chip row.)
    """
    body = fn_body("function navigateToPage")
    start = body.index("page === 'community'")
    branch = body[start : body.index("page === 'account'", start)]
    assert re.search(r"marketplaceCategoryFilter\s*=", branch), (
        "the category reset vanished from the community branch"
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

