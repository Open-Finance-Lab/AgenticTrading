"""Execute the shipped picker logic across category/custom/market switches."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from dashboard.backend.infrastructure.market_data.strategy_universe import representative_presets
from dashboard.backend.tests._frontend_source import js_const, fn_body


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
HTML = (Path(__file__).resolve().parents[2] / "frontend/app.html").read_text(encoding="utf-8")


def run_picker(script, setup=""):
    code = "\n".join([
        "const assert = require('node:assert/strict');",
        js_const("ASSET_UNIVERSES"),
        js_const("IFIND_ASHARE_SOURCE"),
        "let selectedUniverse = 'djia'; let representativePoolsPromise = null;",
        "let source = 'alpaca'; let builtin = true; let renders = 0; let requests = 0;",
        "const error = {hidden:true}; const options = ['ordinary','fund','all'].map(value => ({value,disabled:true}));",
        "const elements = {backtestUniverseLoadError:error,backtestUniverseSelect:{options},",
        "  marketDataSourceSelect:{get value(){return source}},builtinTab:{classList:{contains:()=>builtin}}};",
        "const document = {getElementById:id=>elements[id], querySelectorAll:()=>[{dataset:{ticker:'CUSTOM'}}]};",
        "const API_BASE = ''; const fixtures = " + json.dumps(representative_presets()) + ";",
        "const API = {get:async()=>{requests++; return {representative_presets:fixtures}}};",
        "function renderSelectedUniversePreview(){renders++;}",
        "function notifyAssetUniverseChanged(){}",
        "function getIFindUniverseProfile(){return {assets:[{symbol:'600519.SH'}]}}",
        fn_body("async function loadRepresentativeStockPools("),
        fn_body("function getSelectedStockPoolRequest("),
        fn_body("function selectPreset("),
        fn_body("function getSelectedAssets("),
        fn_body("async function runBacktest("),
        setup,
        "(async()=>{" + script + "})().catch(e=>{console.error(e);process.exit(1)});",
    ])
    result = subprocess.run(["node", "-"], input=code, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_submission_freezes_category_and_does_not_mix_it_with_explicit_assets():
    run_picker("""
      await loadRepresentativeStockPools();
      for(const pool of ['ordinary','fund','all']) {
        selectPreset(pool);
        await runBacktest();
        const {url,payload}=posted.at(-1);
        assert.equal(payload.stock_pool,pool);
        assert.equal(payload.pool_mode,'representative30');
        assert.equal('assets' in payload,false);
        assert.equal(new URL(url,'http://localhost').searchParams.has('assets'),false);
        assert.equal(payload.decision_source,'rule_based');
        assert.equal(payload.initial_capital,10000);
      }
      selectPreset('fund'); builtin=false;
      await runBacktest();
      assert.deepEqual(posted.at(-1).payload.assets,['CUSTOM']);
      assert.equal('stock_pool' in posted.at(-1).payload,false);
      builtin=true; selectPreset('djia');
      await runBacktest();
      assert.deepEqual(posted.at(-1).payload.assets,ASSET_UNIVERSES.djia.assets);
      assert.equal('pool_mode' in posted.at(-1).payload,false);
      assert.equal(posted.length,5);
    """, setup="""
      const RULE_BASED_DECISION_SOURCE='rule_based', LLM_DECISION_SOURCE='llm';
      source='vnpy_simulation';
      elements.startDate={value:'2026-05-04'}; elements.endDate={value:'2026-05-05'};
      const runBacktestModalAgent={agent_id:'agent-test',name:'Test',runtime_type:'pipeline'};
      const window={}; let currentMode=''; const posted=[];
      const activateAgent=async()=>{};
      const resolveBacktestCapital=()=>10000;
      const formatPromptFromPipeline=()=>'';
      const describeUniverseFromAssets=()=>'Custom';
      const renderBacktestDataSourceBadge=()=>{};
      // Closing can reset UI state; the request must use the captured selection.
      const closeRunBacktestModal=()=>{selectedUniverse='djia'};
      const prepareLiveBacktestView=()=>{};
      const markAgentBacktestRunning=()=>'pending-test';
      const navigateToPage=()=>{};
      const applyAgentFilters=()=>{};
      const updateBacktestRunProgress=()=>{};
      const formatBacktestError=()=>'';
      const showBacktestLaunchFailure=()=>{};
      API.post=async(url,payload)=>{posted.push({url,payload});return {success:false}};
    """)


def test_loaded_categories_use_backend_rosters_and_only_representative_mode():
    run_picker("""
      await Promise.all([loadRepresentativeStockPools(),loadRepresentativeStockPools()]);
      assert.equal(requests,1);
      for(const pool of ['ordinary','fund','all']) {
        selectPreset(pool);
        assert.deepEqual(getSelectedAssets(),fixtures.find(p=>p.stock_pool===pool).symbols);
        assert.deepEqual(getSelectedStockPoolRequest(),{stock_pool:pool,pool_mode:'representative30'});
        assert.equal(getSelectedAssets().length,30);
      }
      assert(options.every(option=>!option.disabled));
      builtin=false;
      assert.equal(getSelectedStockPoolRequest(),null);
      assert.deepEqual(getSelectedAssets(),['CUSTOM']);
      builtin=true; source='ifind_ashare';
      assert.equal(getSelectedStockPoolRequest(),null);
      assert.deepEqual(getSelectedAssets(),['600519.SH']);
      source='alpaca'; selectPreset('djia');
      assert.equal(getSelectedStockPoolRequest(),null);
    """)


def test_failed_catalog_load_can_retry_without_enabling_invalid_options():
    run_picker("""
      API.get=async()=>({representative_presets:[]});
      assert.equal(await loadRepresentativeStockPools(),false);
      assert.equal(error.hidden,false);
      assert(options.every(option=>option.disabled));
      assert.equal(ASSET_UNIVERSES.fund,undefined);
      API.get=async()=>({representative_presets:fixtures});
      assert.equal(await loadRepresentativeStockPools(),true);
      assert.equal(error.hidden,true);
      selectPreset('fund'); assert.equal(getSelectedAssets().length,30);
    """)


def test_picker_excludes_full_universe_and_capitalization_controls():
    from html.parser import HTMLParser

    class Options(HTMLParser):
        in_picker = False
        values = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "select":
                self.in_picker = attrs.get("id") == "backtestUniverseSelect"
            elif tag == "option" and self.in_picker:
                self.values.append(attrs["value"])

        def handle_endtag(self, tag):
            if tag == "select":
                self.in_picker = False

    parser = Options()
    parser.feed(HTML)
    assert parser.values == ["djia", "mag7", "ordinary", "fund", "all"]
    assert 'id="poolModeSelect"' not in HTML
