/** Exact ATL Credit formatting. One Credit is 1,000,000 micro-Credits. */
(function () {
  'use strict';

  const UNAVAILABLE = '—';

  function groupWhole(value) {
    return value.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  function formatCredits(value) {
    const text = value == null ? '' : String(value);
    const match = /^(-?)(\d+)(?:\.(\d{1,6}))?$/.exec(text);
    if (!match) return UNAVAILABLE;
    const fraction = (match[3] || '').padEnd(6, '0');
    const isZero = /^0+$/.test(match[2]) && !/[1-9]/.test(fraction);
    const sign = match[1] && !isZero ? '-' : '';
    return `${sign}${groupWhole(match[2])}.${fraction}`;
  }

  function formatCreditsMicro(value) {
    const text = typeof value === 'number' && Number.isSafeInteger(value)
      ? String(value)
      : (typeof value === 'string' && /^-?\d+$/.test(value) ? value : '');
    if (!text) return UNAVAILABLE;
    const micro = BigInt(text);
    const sign = micro < 0n ? '-' : '';
    const absolute = micro < 0n ? -micro : micro;
    const whole = groupWhole(String(absolute / 1000000n));
    const fraction = String(absolute % 1000000n).padStart(6, '0');
    return `${sign}${whole}.${fraction}`;
  }

  window.CreditFormat = Object.freeze({ formatCredits, formatCreditsMicro });
})();
