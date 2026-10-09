/** Group recorded events by explicit decision/step IDs, with optional raw detail. */
(function () {
  'use strict';
  function groupEvents(events) {
    const stepDecisions = new Map();
    events.forEach(e => { if (e.step_id && e.decision_id) {
      if (!stepDecisions.has(e.step_id)) stepDecisions.set(e.step_id, new Set());
      stepDecisions.get(e.step_id).add(e.decision_id);
    } });
    const groups = new Map();
    events.forEach(e => {
      const matches = stepDecisions.get(e.step_id);
      const decision = e.decision_id || (matches?.size === 1 ? [...matches][0] : null);
      const key = decision ? `decision:${decision}` : e.step_id ? `step:${e.step_id}` : `event:${e.sequence_no}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(e);
    });
    return [...groups].map(([key, events]) => ({ key, events }));
  }
  function summary(event) {
    const p = event.payload || {};
    const actions = [p.fills, p.actions, p.executed].find(a => Array.isArray(a) && a.length) || [];
    const orders = Array.isArray(actions) ? actions.slice(0, 3).map(a => [a.side || a.action, a.symbol, a.shares == null ? '' : `${a.shares} shares`].filter(Boolean).join(' ')).join('; ') : '';
    return orders || p.error_code || (typeof p.message === 'string' ? p.message.slice(0, 160) : '') || window.AdminShell.humanize(event.event_type || 'event');
  }
  function render(events, openState = {}) {
    const el = window.AdminShell.el;
    const list = el('ol', 'trace-timeline');
    if (!events?.length) { list.appendChild(el('li', 'panel-empty', 'No events recorded yet.')); return list; }
    groupEvents(events).forEach(group => {
      const li = el('li', 'trace-event-group');
      const details = el('details', 'trace-group');
      details.setAttribute('data-trace-expand', group.key);
      const failed = group.events.some(e => /failed|error/.test(e.event_type) || e.payload?.error_code);
      details.open = Object.hasOwn(openState, group.key) ? openState[group.key] : failed;
      const decision = group.events.find(e => e.event_type === 'decision_recorded');
      const last = group.events.at(-1);
      const outcome = group.events.find(e => e.event_type === 'execution_result') || decision || last;
      const heading = el('summary', 'trace-group-summary');
      heading.appendChild(el('strong', '', `${decision ? 'Decision' : window.AdminShell.humanize(group.events[0].event_type)} · ${summary(outcome)}`));
      const marketTime = decision?.payload?.decision_at;
      heading.appendChild(el('span', 'muted', `${marketTime ? 'Market ' + window.AdminShell.formatTimestamp(marketTime) : 'Recorded ' + window.AdminShell.formatTimestamp(last.occurred_at)} · ${group.events.length} events${failed ? ' · Failed' : ''}`));
      details.appendChild(heading);
      group.events.forEach(e => {
        const item = el('div', 'trace-event');
        item.setAttribute('id', `trace-event-${e.sequence_no}`); item.setAttribute('tabindex', '-1');
        item.appendChild(el('h3', '', `#${e.sequence_no} ${window.AdminShell.humanize(e.event_type)}`));
        item.appendChild(el('p', 'muted', `Recorded ${window.AdminShell.formatTimestamp(e.occurred_at)} · ${e.actor_type || 'Unknown actor'}`));
        item.appendChild(el('p', '', summary(e)));
        const reasons = e.payload?.reasoning_summaries || [e.payload?.reasoning_summary];
        if (Array.isArray(reasons)) reasons.filter(r => typeof r === 'string' && r).forEach(r => item.appendChild(el('p', 'trace-reason', r)));
        const raw = el('details', 'trace-technical');
        const rawKey = `raw:${e.sequence_no}`;
        raw.setAttribute('data-trace-expand', rawKey); raw.open = Boolean(openState[rawKey]);
        raw.appendChild(el('summary', '', 'Technical details (JSON)'));
        const json = el('pre', 'trace-event-payload');
        json.textContent = JSON.stringify(e, null, 2);
        raw.appendChild(json);
        item.appendChild(raw); details.appendChild(item);
      });
      li.appendChild(details); list.appendChild(li);
    });
    return list;
  }
  function capture(host) {
    const state = {};
    host.querySelectorAll('[data-trace-expand]').forEach(n => { state[n.getAttribute('data-trace-expand')] = n.open; });
    return state;
  }
  function reveal(sequence) {
    const target = document.getElementById(`trace-event-${sequence}`);
    if (!target) return;
    let parent = target.parentElement;
    while (parent) { if (parent.tagName === 'DETAILS') parent.open = true; parent = parent.parentElement; }
    target.scrollIntoView?.({block: 'center', behavior: 'smooth'}); target.focus?.({preventScroll: true});
  }
  window.AdminTraceTimeline = { groupEvents, summary, render, capture, reveal };
})();
