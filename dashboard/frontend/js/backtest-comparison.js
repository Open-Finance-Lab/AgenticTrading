(function () {
  'use strict';

  const SERIES = Object.freeze({
    agent: { label: 'Your Agent', color: '#4FC3F7' },
    djia: { label: 'DJIA', color: '#F5C04A' },
    nasdaq: { label: 'Nasdaq-100', color: '#9AA4B2' },
    buyhold: { label: 'Buy & Hold', color: '#34D399' },
  });
  const METRICS = Object.freeze([
    { key: 'finalValue', label: 'Final Value', kind: 'currency' },
    { key: 'totalReturn', label: 'Total Return', kind: 'percent' },
    { key: 'maxDrawdown', label: 'Max Drawdown', kind: 'percent' },
    { key: 'sharpe', label: 'Sharpe Ratio', kind: 'ratio' },
  ]);
  const INDEX_IDS = Object.freeze({
    'index:^DJI': 'djia',
    'index:^NDX': 'nasdaq',
  });

  function finiteValues(values) {
    if (!Array.isArray(values)) return [];
    return values.reduce((clean, value) => {
      if (value === null || value === undefined || value === '') return clean;
      const numeric = Number(value);
      if (Number.isFinite(numeric)) clean.push(numeric);
      return clean;
    }, []);
  }

  function periodsPerYearFromTimestamps(timestamps) {
    // Chart valuation frequency can differ from agent decision frequency.
    // Use the most common intraday interval, ignoring overnight gaps.
    const hourly = 252 * 6.5;
    if (!Array.isArray(timestamps) || timestamps.length < 2) return hourly;
    const gaps = new Map();
    for (let i = 1; i < timestamps.length; i += 1) {
      const before = Date.parse(timestamps[i - 1]);
      const after = Date.parse(timestamps[i]);
      if (!Number.isFinite(before) || !Number.isFinite(after)) continue;
      const minutes = (after - before) / 60000;
      if (minutes > 0 && minutes <= 390)
        gaps.set(minutes, (gaps.get(minutes) || 0) + 1);
    }
    if (!gaps.size) return hourly;
    const interval = [...gaps].sort((a, b) => b[1] - a[1] || a[0] - b[0])[0][0];
    return hourly * 60 / interval;
  }

  function calculateMetrics(values, periodsPerYear = 252 * 6.5) {
    const clean = finiteValues(values);
    const result = {
      finalValue: clean.length ? clean[clean.length - 1] : null,
      totalReturn: null,
      maxDrawdown: null,
      sharpe: null,
    };
    if (clean.length < 2 || clean[0] <= 0) return result;

    result.totalReturn = (clean[clean.length - 1] / clean[0] - 1) * 100;
    let peak = clean[0];
    let maxDrawdown = 0;
    for (const value of clean) {
      peak = Math.max(peak, value);
      maxDrawdown = Math.min(maxDrawdown, value / peak - 1);
    }
    result.maxDrawdown = maxDrawdown * 100;

    const returns = [];
    for (let index = 1; index < clean.length; index += 1) {
      if (clean[index - 1] !== 0) {
        returns.push(clean[index] / clean[index - 1] - 1);
      }
    }
    if (returns.length < 2) return result;

    const mean = returns.reduce((sum, value) => sum + value, 0) / returns.length;
    const variance = returns.reduce(
      (sum, value) => sum + (value - mean) ** 2,
      0,
    ) / returns.length;
    const deviation = Math.sqrt(variance);
    if (deviation > 0 && Number.isFinite(periodsPerYear) && periodsPerYear > 0) {
      result.sharpe = mean / deviation * Math.sqrt(periodsPerYear);
    }
    return result;
  }

  function classify(entry, payload, run) {
    if (entry?.run_id === payload?.agent_run_id) return 'agent';
    if (INDEX_IDS[entry?.run_id]) return INDEX_IDS[entry.run_id];
    if (entry?.run_id === run?.baseline_buyhold_run_id) return 'buyhold';

    const label = String(entry?.label || '').toLowerCase();
    if (label === 'djia index' || label === 'djia') return 'djia';
    if (label === 'nasdaq-100' || label === 'nasdaq 100') return 'nasdaq';
    if (label === 'buy-and-hold' || label === 'buy & hold') return 'buyhold';
    return null;
  }

  function buildModel(payload, run) {
    const desired = run?.data_source === 'ifind_ashare'
      ? ['agent', 'buyhold']
      : ['agent', 'djia', 'nasdaq', 'buyhold'];
    const indexBaselinesOk = payload?.index_baselines_ok !== false;
    const found = new Map();
    const periodsPerYear = periodsPerYearFromTimestamps(payload?.timestamps);

    for (const entry of payload?.series || []) {
      const key = classify(entry, payload, run);
      if (key && !found.has(key)) found.set(key, entry);
    }

    const columns = desired.map((key) => {
      const candidate = found.get(key) || null;
      const entry = !indexBaselinesOk && (key === 'djia' || key === 'nasdaq')
        ? null
        : candidate;
      return {
        key,
        label: SERIES[key].label,
        color: entry?.color || SERIES[key].color,
        available: Boolean(entry),
        runId: entry?.run_id || null,
        values: entry?.values || [],
        metrics: entry ? calculateMetrics(entry.values, periodsPerYear) : calculateMetrics([]),
        dashed: Boolean(entry?.dashed),
      };
    });

    const bestByMetric = Object.fromEntries(METRICS.map(({ key }) => {
      const finite = columns.filter(
        (column) => Number.isFinite(column.metrics[key]),
      );
      if (!finite.length) return [key, []];
      const best = Math.max(...finite.map((column) => column.metrics[key]));
      return [
        key,
        finite
          .filter((column) => column.metrics[key] === best)
          .map((column) => column.key),
      ];
    }));
    const agent = columns.find((column) => column.key === 'agent');
    const agentDeltas = Object.fromEntries(columns
      .filter((column) => column.key !== 'agent')
      .map((column) => [
        column.key,
        Number.isFinite(agent?.metrics.totalReturn)
          && Number.isFinite(column.metrics.totalReturn)
          ? agent.metrics.totalReturn - column.metrics.totalReturn
          : null,
      ]));

    return {
      columns,
      bestByMetric,
      agentDeltas,
      indexBaselinesOk,
    };
  }

  window.BacktestComparison = Object.freeze({
    METRICS,
    buildModel,
    calculateMetrics,
  });
})();
