/** Admin Grant Credits console. Secrets never belong on this surface. */
(function () {
  'use strict';

  const { formatCredits, formatCreditsMicro } = window.CreditFormat;
  const ADMIN_CREDITS_USERS_PAGE_SIZE = 25;

  const state = {
    initialized: false,
    users: [],
    usersOffset: 0,
    usersTotal: 0,
    usersLimit: ADMIN_CREDITS_USERS_PAGE_SIZE,
    usersRequestSeq: 0,
    activityCursor: null,
    activityItems: [],
    pendingGrantReason: null,
  };
  const DEFAULT_ASSIGN_REASON = 'Approved Grant allocation.';
  const DEFAULT_RECLAIM_REASON = 'Admin reclaim of unused Grant Credits.';
  const POOL_RING_CIRCUMFERENCE = 2 * Math.PI * 78;

  function element(id) {
    return document.getElementById(id);
  }

  function isAdmin() {
    return window.getStoredAuthUser && window.getStoredAuthUser()?.role === 'admin';
  }

  function request(path, options = {}) {
    if (!window.API || typeof window.API.request !== 'function') {
      return Promise.reject(new Error('Admin API is not ready yet.'));
    }
    return window.API.request(path, options);
  }

  function setStatus(message, tone = '') {
    const status = element('adminCreditsStatus');
    if (!status) return;
    status.textContent = message || '';
    status.className = `credits-status${tone ? ` is-${tone}` : ''}`;
  }

  function uuid() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
      return window.crypto.randomUUID();
    }
    throw new Error('This browser cannot create a safe request ID.');
  }

  function parseCreditsMicro(value) {
    const text = String(value || '').trim();
    if (!/^\d+(?:\.\d{1,6})?$/.test(text)) {
      throw new Error('Enter a positive Credits amount with up to 6 decimal places.');
    }
    const [whole, fraction = ''] = text.split('.');
    const micro = BigInt(whole) * 1000000n + BigInt(fraction.padEnd(6, '0') || '0');
    if (micro <= 0n || micro > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error('Credits amount is outside the supported range.');
    }
    return Number(micro);
  }

  function parseSignedCreditsMicro(value) {
    const text = String(value ?? '').trim();
    if (!/^[+-]?\d+(?:\.\d{1,6})?$/.test(text)) {
      throw new Error('Enter a non-zero Credits adjustment with up to 6 decimal places.');
    }
    const negative = text.startsWith('-');
    const unsigned = text.replace(/^[+-]/, '');
    const [whole, fraction = ''] = unsigned.split('.');
    const micro = BigInt(whole) * 1000000n + BigInt(fraction.padEnd(6, '0') || '0');
    if (micro <= 0n || micro > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error('Credits adjustment is outside the supported range.');
    }
    return Number(negative ? -micro : micro);
  }

  function formatTime(value) {
    const date = new Date(value);
    if (!value || Number.isNaN(date.getTime())) return 'Unknown time';
    return date.toLocaleString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  }

  function appendIcon(button, iconId) {
    const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    icon.setAttribute('aria-hidden', 'true');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', `#${iconId}`);
    icon.appendChild(use);
    button.prepend(icon);
  }

  function textNode(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = text;
    return node;
  }

  function clear(node) {
    if (!node) return;
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function handleAccessLost(error) {
    if (error?.status !== 401 && error?.status !== 403) return false;
    setStatus('Admin access is no longer available.', 'error');
    if (typeof navigateToPage === 'function') navigateToPage('home');
    return true;
  }

  function setPoolRingSegment(id, ratio, offset = 0) {
    const ring = element(id);
    if (!ring) return 0;
    const clampedRatio = Math.min(1, Math.max(0, Number(ratio) || 0));
    const visibleLength = POOL_RING_CIRCUMFERENCE * clampedRatio;
    ring.style.strokeDasharray = `${visibleLength} ${POOL_RING_CIRCUMFERENCE - visibleLength}`;
    ring.style.strokeDashoffset = String(offset);
    return visibleLength;
  }

  function renderPool(pool) {
    if (!pool) return;
    const availableMicro = pool.pool_available_micro;
    const allocatedMicro = pool.allocated_to_users_micro;
    const validAvailable = Number.isSafeInteger(availableMicro);
    const validAllocated = Number.isSafeInteger(allocatedMicro);
    const totalMicro = validAvailable && validAllocated
      && Number.isSafeInteger(availableMicro + allocatedMicro)
      ? availableMicro + allocatedMicro
      : null;
    const availableText = formatCreditsMicro(availableMicro);
    const allocatedText = formatCreditsMicro(allocatedMicro);
    const totalText = formatCreditsMicro(totalMicro);
    const chartAvailable = validAvailable && availableMicro > 0 ? availableMicro : 0;
    const chartAllocated = validAllocated && allocatedMicro > 0 ? allocatedMicro : 0;
    const chartTotal = chartAvailable + chartAllocated;
    const availableRatio = chartTotal > 0 ? chartAvailable / chartTotal : 0;
    const allocatedRatio = chartTotal > 0 ? chartAllocated / chartTotal : 0;
    const allocatedLength = setPoolRingSegment('adminCreditsPoolRingAllocated', allocatedRatio);
    setPoolRingSegment('adminCreditsPoolRingAvailable', availableRatio, -allocatedLength);
    element('adminCreditsPoolTotal').textContent = totalText;
    element('adminCreditsPoolAvailable').textContent = `${availableText} Credits`;
    element('adminCreditsAllocated').textContent = `${allocatedText} Credits`;
    const summary = element('adminCreditsPoolSummary');
    if (summary) {
      summary.setAttribute(
        'aria-label',
        `Grant Pool: ${totalText} Credits total, ${availableText} available, ${allocatedText} allocated`,
      );
    }
  }

  async function loadPool() {
    const data = await request('/api/admin/credits/grant-pool');
    renderPool(data?.pool);
  }

  function userName(user) {
    return user.display_name || user.email || `User #${user.id}`;
  }

  function renderUsersPager() {
    const range = element('adminCreditsUsersRange');
    const previous = element('adminCreditsUsersPrevBtn');
    const next = element('adminCreditsUsersNextBtn');
    const { usersOffset, usersTotal, usersLimit } = state;
    const shown = Math.min(usersLimit, Math.max(0, usersTotal - usersOffset));
    if (range) {
      range.textContent = usersTotal
        ? `Showing ${usersOffset + 1}–${usersOffset + shown} of ${usersTotal}`
        : 'Showing 0 of 0';
    }
    if (previous) previous.disabled = usersOffset <= 0;
    if (next) next.disabled = usersOffset + usersLimit >= usersTotal;
  }

  function renderUsers(users, total = 0) {
    const body = element('adminCreditsUsersBody');
    const count = element('adminCreditsUserCount');
    if (!body) return;
    state.users = Array.isArray(users) ? users : [];
    state.usersTotal = Number.isFinite(Number(total)) ? Math.max(0, Number(total)) : state.users.length;
    if (count) count.textContent = `${state.users.length} shown`;
    renderUsersPager();
    clear(body);
    if (!state.users.length) {
      const row = document.createElement('tr');
      const cell = textNode('td', 'admin-empty', 'No matching accounts.');
      cell.colSpan = 8;
      row.appendChild(cell);
      body.appendChild(row);
      return;
    }

    state.users.forEach((user) => {
      const balance = user.balance || {};
      const row = document.createElement('tr');
      const account = document.createElement('td');
      account.className = 'admin-credits-account';
      account.appendChild(textNode('strong', '', userName(user)));
      account.appendChild(textNode('span', '', user.email || `User #${user.id}`));
      const roleCell = document.createElement('td');
      roleCell.className = 'admin-credits-role';
      const role = user.role === 'admin' ? 'admin' : 'user';
      const currentUser = window.getStoredAuthUser?.();
      const isSelf = currentUser && Number(currentUser.id) === Number(user.id);
      if (isSelf) {
        roleCell.appendChild(textNode('span', 'admin-role-locked', `${role} (you)`));
      } else {
        const roleSelect = document.createElement('select');
        roleSelect.className = 'admin-credits-role-select';
        roleSelect.setAttribute('aria-label', `Role for ${userName(user)}`);
        ['user', 'admin'].forEach((optionValue) => {
          const option = document.createElement('option');
          option.value = optionValue;
          option.textContent = optionValue;
          option.selected = optionValue === role;
          roleSelect.appendChild(option);
        });
        roleSelect.addEventListener('change', () => mutateUserRole(user, roleSelect));
        roleCell.appendChild(roleSelect);
      }
      const statusCell = document.createElement('td');
      const status = balance.account_status || 'active';
      const reason = balance.restriction_reason;
      const outstanding = Number(balance.outstanding_credits_micro || 0);
      statusCell.appendChild(textNode('strong', status === 'active' ? 'admin-status-active' : 'admin-status-restricted', status));
      if (status !== 'active') {
        const detail = reason === 'llm_overage'
          ? `Owes ${formatCreditsMicro(outstanding)} Credits`
          : 'Refund review';
        statusCell.appendChild(textNode('span', 'admin-credits-status-detail', detail));
      }
      const amountCell = document.createElement('td');
      const amount = document.createElement('input');
      amount.type = 'text';
      amount.inputMode = 'decimal';
      amount.placeholder = '0.000000';
      amount.autocomplete = 'off';
      amount.className = 'admin-credits-amount-input';
      amount.setAttribute('aria-label', `Grant Credits amount for ${userName(user)}`);
      amountCell.appendChild(amount);

      const actions = document.createElement('td');
      actions.className = 'admin-credits-row-actions';
      const assign = textNode('button', 'credits-key-action', 'Assign');
      assign.type = 'button';
      assign.title = `Assign Grant Credits to ${userName(user)}`;
      appendIcon(assign, 'icon-check-circle');
      assign.addEventListener('click', () => openAssignReasonDialog(user, amount));
      const reclaim = textNode('button', 'credits-key-action is-danger', 'Reclaim');
      reclaim.type = 'button';
      reclaim.title = `Reclaim Grant Credits from ${userName(user)}`;
      appendIcon(reclaim, 'icon-minus');
      reclaim.disabled = Number(balance.grant_available_micro || 0) <= 0;
      reclaim.addEventListener('click', () => mutateUserGrant(user, 'reclaim', amount));
      actions.append(assign, reclaim);
      if (status !== 'active' && reason !== 'llm_overage') {
        const reinstate = textNode('button', 'credits-key-action', 'Reinstate');
        reinstate.type = 'button';
        reinstate.title = `Reinstate ${userName(user)}`;
        reinstate.addEventListener('click', () => reinstateUser(user));
        actions.appendChild(reinstate);
      }

      row.append(
        account,
        roleCell,
        statusCell,
        textNode('td', 'admin-credits-number', formatCredits(balance.display_grant_credits)),
        textNode('td', 'admin-credits-number', formatCredits(balance.display_purchased_credits)),
        textNode('td', 'admin-credits-number', formatCredits(balance.display_total_credits)),
        amountCell,
        actions,
      );
      body.appendChild(row);
    });
  }

  async function reinstateUser(user) {
    if (!window.confirm(`Reinstate ${userName(user)} after refund review?`)) return;
    try {
      setStatus('Restoring account access…', 'pending');
      await request(`/api/admin/credits/accounts/${Number(user.id)}/reinstate`, { method: 'POST' });
      setStatus(`${userName(user)} is active again.`, 'success');
      await refresh();
    } catch (error) {
      if (!handleAccessLost(error)) setStatus(error.message || 'Account could not be reinstated.', 'error');
    }
  }

  async function loadUsers({ offset = state.usersOffset } = {}) {
    const query = String(element('adminCreditsUserQuery')?.value || '').trim();
    state.usersOffset = Math.max(0, Number.isFinite(Number(offset)) ? Number(offset) : 0);
    const seq = ++state.usersRequestSeq;
    const params = new URLSearchParams({
      limit: String(state.usersLimit),
      offset: String(state.usersOffset),
    });
    if (query) params.set('query', query);
    const data = await request(`/api/admin/credits/users?${params.toString()}`);
    if (seq !== state.usersRequestSeq) return;
    const responseOffset = Number(data?.offset);
    const responseLimit = Number(data?.limit);
    if (Number.isFinite(responseOffset)) state.usersOffset = Math.max(0, responseOffset);
    if (Number.isFinite(responseLimit) && responseLimit > 0) state.usersLimit = responseLimit;
    const responseTotal = Math.max(0, Number(data?.total) || 0);
    if (!Array.isArray(data?.users) && responseTotal > 0) {
      renderUsers([], responseTotal);
      return;
    }
    if (!data?.users?.length && state.usersOffset >= responseTotal && state.usersOffset > 0) {
      const lastPageOffset = Math.floor((responseTotal - 1) / state.usersLimit) * state.usersLimit;
      if (lastPageOffset !== state.usersOffset) {
        await loadUsers({ offset: lastPageOffset });
        return;
      }
    }
    renderUsers(data?.users, data?.total);
  }

  function operationLabel(entryType) {
    return {
      fund: 'Fund pool',
      reduce: 'Reduce pool',
      assign: 'Assign to user',
      reclaim: 'Reclaim from user',
    }[entryType] || entryType || 'Grant operation';
  }

  function operationSign(entryType) {
    return entryType === 'fund' || entryType === 'reclaim' ? '+' : '−';
  }

  function renderActivity(items, append = false) {
    const body = element('adminCreditsActivityBody');
    if (!body) return;
    if (!append) clear(body);
    if (!items.length && !append) {
      const row = document.createElement('tr');
      const cell = textNode('td', 'admin-empty', 'No Grant activity yet.');
      cell.colSpan = 6;
      row.appendChild(cell);
      body.appendChild(row);
      return;
    }
    items.forEach((entry) => {
      const row = document.createElement('tr');
      const account = entry.user_id ? `User #${entry.user_id}` : 'Grant Pool';
      const amount = `${operationSign(entry.entry_type)}${formatCredits(entry.display_credits)}`;
      row.append(
        textNode('td', 'admin-credits-time', formatTime(entry.created_at)),
        textNode('td', '', operationLabel(entry.entry_type)),
        textNode('td', '', account),
        textNode('td', `admin-credits-activity-amount is-${operationSign(entry.entry_type) === '+' ? 'positive' : 'negative'}`, amount),
        textNode('td', '', entry.source || '—'),
        textNode('td', 'admin-credits-reason', entry.reason || '—'),
      );
      body.appendChild(row);
    });
  }

  async function loadActivity({ append = false } = {}) {
    const params = new URLSearchParams({ limit: '50' });
    if (append && state.activityCursor) params.set('cursor', String(state.activityCursor));
    const data = await request(`/api/admin/credits/activity?${params.toString()}`);
    const items = Array.isArray(data?.items) ? data.items : [];
    renderActivity(items, append);
    state.activityCursor = data?.next_cursor || null;
    const more = element('adminCreditsActivityMoreBtn');
    if (more) more.hidden = !state.activityCursor;
  }

  async function refresh() {
    if (!isAdmin()) return;
    setStatus('Refreshing Grant Pool…', 'pending');
    state.activityCursor = null;
    const results = await Promise.allSettled([loadPool(), loadUsers(), loadActivity()]);
    const failed = results.find((result) => result.status === 'rejected');
    if (failed) {
      if (handleAccessLost(failed.reason)) return;
      setStatus(failed.reason?.message || 'Grant Credits data could not be loaded.', 'error');
      return;
    }
    setStatus('Grant Pool is up to date.', 'success');
  }

  async function mutatePool(form) {
    const button = form.querySelector('button[type="submit"]');
    try {
      const signedAmountMicro = parseSignedCreditsMicro(element('adminGrantPoolAmount')?.value);
      const operation = signedAmountMicro > 0 ? 'fund' : 'reduce';
      const reason = element('adminGrantPoolReason')?.value.trim();
      if (!reason) throw new Error('Reason is required.');
      if (button) button.disabled = true;
      setStatus(`${operation === 'fund' ? 'Funding' : 'Reducing'} Grant Pool…`, 'pending');
      await request(`/api/admin/credits/grant-pool/${operation}`, {
        method: 'POST',
        body: JSON.stringify({
          client_request_id: uuid(),
          amount_micro: Math.abs(signedAmountMicro),
          source: 'admin-console',
          reason,
        }),
      });
      element('adminGrantPoolAmount').value = '';
      setStatus(`Grant Pool ${operation === 'fund' ? 'funded' : 'reduced'}.`, 'success');
      await refresh();
    } catch (error) {
      if (!handleAccessLost(error)) setStatus(error.message || 'Grant Pool mutation failed.', 'error');
    } finally {
      if (button) button.disabled = false;
    }
  }

  function openAssignReasonDialog(user, amountInput) {
    let amountMicro;
    try {
      amountMicro = parseCreditsMicro(amountInput.value);
    } catch (error) {
      setStatus(error.message || 'Enter a valid Grant Credits amount.', 'error');
      return;
    }
    const dialog = element('adminGrantReasonDialog');
    const reasonInput = element('adminGrantReason');
    if (!dialog || !reasonInput || typeof dialog.showModal !== 'function') {
      mutateUserGrant(user, 'assign', amountInput, {
        amountMicro,
        reason: DEFAULT_ASSIGN_REASON,
      });
      return;
    }
    state.pendingGrantReason = { user, amountInput, amountMicro };
    element('adminGrantReasonSummary').textContent = `${userName(user)} · ${formatCreditsMicro(amountMicro)} Credits`;
    element('adminGrantReasonStatus').textContent = '';
    reasonInput.value = DEFAULT_ASSIGN_REASON;
    if (!dialog.open) dialog.showModal();
    window.requestAnimationFrame?.(() => reasonInput.focus());
  }

  function closeAssignReasonDialog() {
    state.pendingGrantReason = null;
    const dialog = element('adminGrantReasonDialog');
    if (dialog?.open) dialog.close();
  }

  async function confirmAssignReason(event) {
    event.preventDefault();
    const pending = state.pendingGrantReason;
    const reasonInput = element('adminGrantReason');
    const reasonStatus = element('adminGrantReasonStatus');
    const reason = String(reasonInput?.value || '').trim();
    if (!pending) {
      closeAssignReasonDialog();
      return;
    }
    if (!reason) {
      if (reasonStatus) reasonStatus.textContent = 'Reason is required.';
      reasonInput?.focus();
      return;
    }
    state.pendingGrantReason = null;
    const dialog = element('adminGrantReasonDialog');
    if (dialog?.open) dialog.close();
    await mutateUserGrant(pending.user, 'assign', pending.amountInput, {
      amountMicro: pending.amountMicro,
      reason,
    });
  }

  async function mutateUserGrant(user, operation, amountInput, options = {}) {
    let amountMicro;
    try {
      amountMicro = options.amountMicro || parseCreditsMicro(amountInput.value);
      const reason = operation === 'assign'
        ? String(options.reason || '').trim()
        : DEFAULT_RECLAIM_REASON;
      if (!reason) throw new Error('Allocation reason is required.');
      if (operation === 'reclaim' && !window.confirm(`Reclaim Grant Credits from ${userName(user)}?`)) return;
      amountInput.disabled = true;
      setStatus(`${operation === 'assign' ? 'Assigning' : 'Reclaiming'} Grant Credits…`, 'pending');
      await request(`/api/admin/credits/grants/${operation}`, {
        method: 'POST',
        body: JSON.stringify({
          client_request_id: uuid(),
          user_id: Number(user.id),
          amount_micro: amountMicro,
          source: 'admin-console',
          reason,
        }),
      });
      amountInput.value = '';
      setStatus(`Grant Credits ${operation === 'assign' ? 'assigned' : 'reclaimed'} for ${userName(user)}.`, 'success');
      await refresh();
    } catch (error) {
      if (!handleAccessLost(error)) setStatus(error.message || 'Grant mutation failed.', 'error');
    } finally {
      amountInput.disabled = false;
    }
  }

  async function mutateUserRole(user, roleSelect) {
    const previousRole = user.role === 'admin' ? 'admin' : 'user';
    const nextRole = roleSelect.value === 'admin' ? 'admin' : 'user';
    if (nextRole === previousRole) return;
    const subject = userName(user);
    const action = nextRole === 'admin' ? 'Promote' : 'Demote';
    const confirmed = window.confirm(
      `${action} ${subject} to ${nextRole}?\n\nThis changes their Admin access immediately.`,
    );
    if (!confirmed) {
      roleSelect.value = previousRole;
      return;
    }
    roleSelect.disabled = true;
    try {
      setStatus(`${action}ing ${subject}…`, 'pending');
      const data = await request(`/api/admin/users/${Number(user.id)}`, {
        method: 'PATCH',
        body: JSON.stringify({ role: nextRole }),
      });
      const updatedUser = data?.user || {};
      user.role = updatedUser.role === 'admin' ? 'admin' : nextRole;
      roleSelect.value = user.role;
      setStatus(`${subject} is now ${user.role}.`, 'success');
      const currentUser = window.getStoredAuthUser?.();
      if (currentUser && Number(currentUser.id) === Number(user.id) && typeof applyUpdatedUser === 'function') {
        applyUpdatedUser({
          ...currentUser,
          ...updatedUser,
          role: user.role,
        });
      }
    } catch (error) {
      roleSelect.value = previousRole;
      if (!handleAccessLost(error)) setStatus(error.message || 'Role update failed.', 'error');
    } finally {
      roleSelect.disabled = false;
    }
  }

  function bindEvents() {
    element('adminCreditsRefreshBtn')?.addEventListener('click', refresh);
    element('adminGrantPoolForm')?.addEventListener('submit', (event) => {
      event.preventDefault();
      mutatePool(event.currentTarget);
    });
    element('adminCreditsUserSearch')?.addEventListener('submit', (event) => {
      event.preventDefault();
      loadUsers({ offset: 0 }).catch((error) => {
        if (!handleAccessLost(error)) setStatus(error.message || 'User search failed.', 'error');
      });
    });
    element('adminCreditsUsersPrevBtn')?.addEventListener('click', () => {
      loadUsers({ offset: state.usersOffset - state.usersLimit }).catch((error) => {
        if (!handleAccessLost(error)) setStatus(error.message || 'Previous account page failed.', 'error');
      });
    });
    element('adminCreditsUsersNextBtn')?.addEventListener('click', () => {
      loadUsers({ offset: state.usersOffset + state.usersLimit }).catch((error) => {
        if (!handleAccessLost(error)) setStatus(error.message || 'Next account page failed.', 'error');
      });
    });
    element('adminGrantReasonForm')?.addEventListener('submit', confirmAssignReason);
    element('adminGrantReasonClose')?.addEventListener('click', closeAssignReasonDialog);
    element('adminGrantReasonCancel')?.addEventListener('click', closeAssignReasonDialog);
    element('adminCreditsActivityMoreBtn')?.addEventListener('click', () => {
      loadActivity({ append: true }).catch((error) => {
        if (!handleAccessLost(error)) setStatus(error.message || 'Activity could not be loaded.', 'error');
      });
    });
  }

  function onEnter() {
    if (!isAdmin()) return;
    if (!state.initialized) {
      state.initialized = true;
      bindEvents();
    }
    window.AdminTabs?.onEnter();
    refresh().catch((error) => {
      if (!handleAccessLost(error)) setStatus(error.message || 'Grant Credits data could not be loaded.', 'error');
    });
  }

  function syncAuth(user) {
    if (user?.role === 'admin') return;
    state.users = [];
    state.usersOffset = 0;
    state.usersTotal = 0;
    state.activityItems = [];
    setStatus('');
  }

  window.AdminCredits = { onEnter, syncAuth };
  document.addEventListener('DOMContentLoaded', () => {
    if (document.documentElement.dataset.navPage === 'admin') onEnter();
  });
})();
