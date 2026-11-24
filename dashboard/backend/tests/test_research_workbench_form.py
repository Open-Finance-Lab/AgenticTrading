"""Execute the real research workbench form renderer against agent manifests."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
APP = (ROOT / 'frontend' / 'app.js').read_text()
CATALOG = json.loads((ROOT / 'config' / 'marketplace.json').read_text())['templates']
WORKBENCH = APP[APP.index('async function openResearchWorkbench('):APP.index('async function loadResearchRuns(')]
FETCH = APP[APP.index('const researchCatalogRuntime = new Map();'):APP.index('function researchStatusLabel(')]

# The acquirer manifest's shape: three required fields, three optional short
# settings, and longtext questions whose default is the service's blank
# placeholder ("Not specified").
ACQUIRER_MANIFEST = {
    'name': 'Acquisition Analysis (Acquirer Side)',
    'output_formats': ['markdown', 'evidence_json', 'docx', 'pdf'],
    'settings_schema': {'fields': [
        {'id': 'acquirer_company', 'label': 'Acquirer Company', 'type': 'text', 'required': True, 'default': None},
        {'id': 'target_company', 'label': 'Target Company', 'type': 'text', 'required': True, 'default': None},
        {'id': 'research_cutoff_date', 'label': 'Research Cut-off Date', 'type': 'date', 'required': True, 'default': None},
        {'id': 'valuation_date', 'label': 'Valuation Date', 'type': 'date', 'required': False, 'default': None},
        {'id': 'report_language', 'label': 'Report Language', 'type': 'text', 'required': False, 'default': 'English'},
        {'id': 'report_currency', 'label': 'Report Currency', 'type': 'text', 'required': False, 'default': 'USD'},
        {'id': 'deal_background', 'label': 'What is the deal background?', 'type': 'longtext', 'required': False,
         'default': 'Not specified', 'placeholder': 'Describe the opportunity.', 'description': 'Describe the opportunity.'},
        {'id': 'acquisition_objectives', 'label': "What are the acquirer's objectives?", 'type': 'longtext',
         'required': False, 'default': 'Not specified'},
    ]},
}


def _render(manifest, catalog_agents):
    if not shutil.which('node'):
        pytest.skip('Node required')
    script = r'''
const nodes = {};
const make = (id) => (nodes[id] = {id, textContent: '', innerHTML: '', hidden: false, className: '', dataset: {}, addEventListener() {}});
['researchAgentName', 'researchAgentDesc', 'researchAgentMeta', 'researchFormFields',
 'researchResultArea', 'researchRunsList', 'researchFormError', 'researchRunForm'].forEach(make);
const document = {getElementById: (id) => nodes[id] || make(id), querySelectorAll: () => []};
const MANIFEST = /*__MANIFEST_JSON__*/;
const AGENTS = /*__AGENTS_JSON__*/;
const API = {get: async (url) => url.endsWith('/agents') ? {agents: AGENTS} : MANIFEST};
const RESEARCH_API = '/api/v1/research';
const escapeHtml = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
const loadResearchRuns = async () => {};
const stopResearchPolling = () => {};
const showResearchWorkbench = () => {};
let researchWorkbenchTemplateId = null;
/*__FETCH__*/
/*__WORKBENCH__*/
(async () => {
  await fetchResearchAgents();
  await openResearchWorkbench('acquisition-analysis-acquirer');
  console.log(JSON.stringify({fields: nodes.researchFormFields.innerHTML, meta: nodes.researchAgentMeta.innerHTML}));
})().catch((e) => { console.error(e); process.exit(1); });
'''
    # Splice code first and JSON last, on sentinels, so fixture text can
    # never be mistaken for an insertion point.
    script = (script.replace('/*__FETCH__*/', FETCH).replace('/*__WORKBENCH__*/', WORKBENCH)
              .replace('/*__MANIFEST_JSON__*/', json.dumps(manifest))
              .replace('/*__AGENTS_JSON__*/', json.dumps(catalog_agents)))
    return json.loads(subprocess.check_output(['node', '-e', script], text=True))


def _section(html, start, end):
    i = html.index(start)
    j = html.index(end, i) if end in html[i:] else len(html)
    return html[i:j]


def test_questions_show_open_and_settings_fold():
    html = _render(ACQUIRER_MANIFEST, [])['fields']
    questions = _section(html, '<fieldset class="research-questions">', '</fieldset>')
    advanced = _section(html, '<details class="research-advanced">', '</details>')
    assert 'rf_deal_background' in questions and 'rf_acquisition_objectives' in questions
    assert 'Advanced options (3)' in advanced
    for setting in ('rf_valuation_date', 'rf_report_language', 'rf_report_currency'):
        assert setting in advanced and setting not in questions
    # Required fields come first, outside both groups.
    assert html.index('rf_acquirer_company') < html.index('research-questions')


def test_placeholder_default_is_hint_text_not_a_value():
    html = _render(ACQUIRER_MANIFEST, [])['fields']
    # The agent's own placeholder wins; with none, the blank default is named.
    # A blank answer still submits the service's default.
    assert ('<textarea id="rf_deal_background" rows="3" placeholder="Describe the opportunity."'
            ' data-blank-default="Not specified"></textarea>') in html
    assert ('<textarea id="rf_acquisition_objectives" rows="3" placeholder="Optional. Left blank: Not specified"'
            ' data-blank-default="Not specified"></textarea>') in html
    # Guidance already shown as the placeholder is not repeated underneath.
    assert html.count('Describe the opportunity.') == 1
    # A real default on a short setting is still pre-filled.
    assert 'id="rf_report_language" type="text" value="English"' in html


def test_required_longtext_keeps_its_placeholder_default_as_a_value():
    fields = [{'id': 'brief', 'label': 'Brief', 'type': 'longtext', 'required': True, 'default': 'Not specified'}]
    html = _render({**ACQUIRER_MANIFEST, 'settings_schema': {'fields': fields}}, [])['fields']
    # Blank would fail the backend's required check, so it stays pre-filled.
    assert '<textarea id="rf_brief" rows="3" placeholder="">Not specified</textarea>' in html
    assert 'Optional' not in html.split('research-questions')[0]


def test_runtime_falls_back_to_the_catalog_estimate():
    agents = [{'template_id': 'acquisition-analysis-acquirer', 'research': {'estimated_runtime_seconds': 600}}]
    assert '~10 min per run' in _render(ACQUIRER_MANIFEST, agents)['meta']
    own = {**ACQUIRER_MANIFEST, 'estimated_runtime_seconds': 1200}
    assert '~20 min per run' in _render(own, agents)['meta']
    # A malformed manifest estimate does not beat the catalog's.
    bad = {**ACQUIRER_MANIFEST, 'estimated_runtime_seconds': '10m'}
    assert '~10 min per run' in _render(bad, agents)['meta']


@pytest.mark.parametrize('template_id', ['acquisition-analysis-acquirer', 'acquisition-analysis-target-company'])
def test_acquisition_agents_are_listed_on_the_research_shelf(template_id):
    row = next(t for t in CATALOG if t['template_id'] == template_id)
    assert row['shelf'] == 'research'
    assert row['research']['agent_id'] == template_id
    # Hosted by the same research service as the shell-screening agent.
    assert row['research']['service_base_url_env'] == 'RESEARCH_SHELL_SERVICE_URL'
    assert row['repo_url'] == 'https://github.com/Lin-Feihan/acquisition-analysis-agent'
