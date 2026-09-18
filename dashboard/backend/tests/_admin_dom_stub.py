"""Node harness for the /admin page modules.

Each module is an IIFE that only registers listeners at load, so the whole file
runs under `node -e` after this stub, and a test then calls the exported
renderers with a committed fixture. The stub is a minimal DOM: enough for
`createElement`/`textContent`/`append`/`setAttribute`/`classList`/`style`, and a
`toJSON()` that flattens a subtree so a test can assert on what a renderer
built. It has no `innerHTML` on purpose -- a renderer that reaches for it
throws here and fails the test before the source-shape pin in
test_admin_page_modules.py ever runs.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "admin_analytics"

requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def source(name: str) -> str:
    return (FRONTEND / "js" / name).read_text(encoding="utf-8")


def fixture(name: str) -> str:
    """The fixture as a JSON literal ready to paste into a scenario."""
    return (FIXTURES / name).read_text(encoding="utf-8")


def target_fixture(name: str) -> str:
    """The target-shape copy (today's payload plus the §9 fields; see the PR C plan's
    "Interim contract"). PR D folds these into the committed fixtures and deletes both
    the directory and this helper."""
    return (FIXTURES / "target" / name).read_text(encoding="utf-8")


DOM_STUB = r"""
const nav = [];
const docListeners = {};
const winListeners = {};
class Text {
  constructor(text) { this.textContent = String(text); this.nodeType = 3; }
  toJSON() { return { text: this.textContent }; }
}
class Node {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.nodeType = 1;
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.style = { setProperty(key, value) { this[key] = String(value); } };
    this._text = '';
    this._classes = new Set();
    this.hidden = false;
    this.disabled = false;
    this.listeners = {};
    this.classList = {
      add: (...names) => names.forEach((name) => this._classes.add(name)),
      remove: (...names) => names.forEach((name) => this._classes.delete(name)),
      toggle: (name, on) => { const next = on === undefined ? !this._classes.has(name) : Boolean(on); if (next) this._classes.add(name); else this._classes.delete(name); return next; },
      contains: (name) => this._classes.has(name),
    };
  }
  get className() { return [...this._classes].join(' '); }
  set className(value) { this._classes = new Set(String(value).split(/\s+/).filter(Boolean)); }
  // `_text` is the text assigned directly to this node (`.textContent = '…'`);
  // `children` are nodes appended afterward. A real DOM's `.textContent =`
  // setter creates one actual text-node child, so a later `appendChild` adds
  // *alongside* it rather than displacing it -- an icon button built as
  // `el(tag, cls, label)` then `appendChild(icon)` (admin-providers.js's
  // `appendIcon`) still reports the label in its textContent. Concatenating
  // here rather than switching on `children.length` is what keeps that true;
  // for every existing caller `_text` and `children` are never both non-empty
  // at once, so this is additive and changes no prior result.
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(''); }
  set textContent(value) { this.children = []; this._text = String(value); }
  get firstChild() { return this.children[0] || null; }
  get isConnected() { return true; }
  appendChild(child) {
    if (child.tagName === '#FRAGMENT') { child.children.slice().forEach((item) => this.appendChild(item)); child.children = []; return child; }
    this.children.push(child); child.parentNode = this; return child;
  }
  append(...items) { items.forEach((item) => this.appendChild(typeof item === 'string' ? new Text(item) : item)); }
  replaceChildren(...items) { this.children = []; this.append(...items); }
  removeChild(child) { this.children = this.children.filter((item) => item !== child); return child; }
  setAttribute(key, value) { this.attributes[key] = String(value); if (key === 'class') this.className = value; }
  getAttribute(key) { return key in this.attributes ? this.attributes[key] : null; }
  removeAttribute(key) { delete this.attributes[key]; }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  dispatchEvent() { return true; }
  focus() { globalThis.focused = this; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  toJSON() {
    const style = Object.fromEntries(Object.entries(this.style).filter(([, value]) => typeof value !== 'function'));
    return {
      tag: this.tagName,
      class: this.className || undefined,
      attrs: Object.keys(this.attributes).length ? this.attributes : undefined,
      dataset: Object.keys(this.dataset).length ? this.dataset : undefined,
      style: Object.keys(style).length ? style : undefined,
      hidden: this.hidden || undefined,
      text: this.children.length ? undefined : this._text,
      children: this.children.length ? this.children.map((child) => child.toJSON()) : undefined,
    };
  }
}
const elements = {};
globalThis.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init && init.detail; } };
globalThis.Event = class { constructor(type) { this.type = type; } };
globalThis.document = {
  createElement: (tag) => new Node(tag),
  createElementNS: (_ns, tag) => new Node(tag),
  createTextNode: (text) => new Text(text),
  createDocumentFragment: () => new Node('#fragment'),
  getElementById: (id) => elements[id] || null,
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener(name, fn) { (docListeners[name] ||= []).push(fn); },
  dispatchEvent(event) { (docListeners[event.type] || []).forEach((fn) => fn(event)); return true; },
  activeElement: null,
  documentElement: new Node('html'),
};
globalThis.window = {
  location: { href: 'https://atl.example/admin', hostname: 'atl.example', pathname: '/admin', search: '', hash: '',
    replace(url) { nav.push(['replace', url]); }, assign(url) { nav.push(['assign', url]); } },
  history: { state: null, replaceState(state, _title, url) { nav.push(['replaceState', String(url)]); }, pushState(state, _title, url) { nav.push(['pushState', String(url)]); } },
  addEventListener(name, fn) { (winListeners[name] ||= []).push(fn); },
  scrollTo() {},
};
globalThis.fetchQueue = [];
globalThis.fetchCalls = [];
globalThis.fetch = (url, options) => {
  fetchCalls.push([url, options]);
  const next = fetchQueue.shift() || { ok: true, status: 200, body: {} };
  return Promise.resolve({ ok: next.ok, status: next.status, json: () => Promise.resolve(next.body) });
};
function register(id, node) { elements[id] = node; return node; }
function flatten(node, out = []) { out.push(node); (node.children || []).forEach((child) => flatten(child, out)); return out; }
function byClass(node, name) { return flatten(node).filter((item) => item._classes && item._classes.has(name)); }
function byTag(node, tag) { return flatten(node).filter((item) => item.tagName === tag.toUpperCase()); }
function texts(nodes) { return nodes.map((item) => item.textContent); }
function panelStub() {
  const panel = new Node('section');
  const parts = { headline: new Node('strong'), body: new Node('div'), status: new Node('p'), error: new Node('p'), errorText: new Node('span'), retry: new Node('button') };
  parts.headline.textContent = '—';
  parts.error.appendChild(parts.errorText); parts.error.appendChild(parts.retry);
  panel.querySelector = (selector) => ({ '[data-headline]': parts.headline, '[data-body]': parts.body, '[data-status]': parts.status, '[data-error]': parts.error, '[data-error] span': parts.errorText, '[data-retry]': parts.retry }[selector] || null);
  return { panel, parts };
}
"""


def run_node(*parts: str, timeout: int = 30) -> object:
    """Run the stub, the given sources/scenario in order, and parse the last line as JSON."""
    script = "\n".join([DOM_STUB, *parts])
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=timeout
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])
