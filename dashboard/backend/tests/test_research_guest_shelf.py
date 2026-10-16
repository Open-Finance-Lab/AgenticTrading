"""Execute the real shelf renderer for guest, owner, and network failure states."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP = (Path(__file__).parents[2] / 'frontend' / 'app.js').read_text()
RENDER = APP[APP.index('async function renderResearchShelf()'):APP.index('/** The Community research card', APP.index('async function renderResearchShelf()'))]


@pytest.mark.parametrize('mode', ['guest', 'owner', 'failure'])
def test_research_shelf_access_states(mode):
    if not shutil.which('node'):
        pytest.skip('Node required')
    script = r'''
const mode = MODE;
const section = {style:{display:'none'}};
let handler, login = false, workbench = false;
const card = {dataset:{templateId:'research'},querySelector: s => s === '.research-open-btn' ? {addEventListener: (_,fn)=>handler=fn} : null};
const grid = {innerHTML:'',closest:()=>section,querySelectorAll:()=>[card]};
const empty = {}, count = {};
const document = {getElementById:id=>({agentsGridResearch:grid,agentsEmptyResearch:empty,agentsCountResearch:count}[id])};
const agent = {template_id:'research',name:'Research demo',research:{},added:true};
const fetchResearchAgents = async()=> {if(mode==='owner') return [agent,{...agent,name:'Unadded',added:false}]; throw {status:mode==='guest'?401:503};};
let catalogCalls=0;
const API = {get:async()=>{catalogCalls++;return {templates:[agent,{name:'Trading',template_id:'trading'}]};}};
const API_BASE='';
const templateMarketplaceShelf = a=>a.research?'research':'open';
const escapeHtml=s=>String(s??'');
const openAuthModal=()=>login=true;
const openResearchWorkbench=()=>workbench=true;
const communityShelfButtonHtml=()=>'';
RENDER
(async()=>{await renderResearchShelf();if(handler)handler();console.log(JSON.stringify({html:grid.innerHTML,display:section.style.display,login,workbench,catalogCalls,error:empty.textContent}));})().catch(e=>{console.error(e);process.exit(1)});
'''.replace('MODE', json.dumps(mode)).replace('RENDER', RENDER)
    result = json.loads(subprocess.check_output(['node', '-e', script], text=True))
    assert result['display'] == ''
    if mode == 'guest':
        assert 'Research demo' in result['html']
        assert 'Sign in to use' in result['html']
        assert 'research-remove-btn' not in result['html']
        assert 'Trading' not in result['html']
        assert result['login'] and not result['workbench']
        assert result['catalogCalls'] == 1
    elif mode == 'owner':
        assert 'Open workbench' in result['html']
        assert 'Unadded' not in result['html']
        assert result['workbench'] and not result['login']
        assert result['catalogCalls'] == 0
    else:
        assert 'temporarily unavailable' in result['error']
        assert result['catalogCalls'] == 0
