/**
 * Kestrel Sovereign Console - Metrics Component
 * Dashboard with KPI cards, charts, and error table
 */

import API from './api.js';

// ============================================================================
// DOM References & State
// ============================================================================

let refreshBtn = null;
let refreshSelect = null;
let autoRefreshInterval = null;
let timelineChart = null;
let durationChart = null;
let distributionChart = null;

// ============================================================================
// Initialization
// ============================================================================

export function initMetrics() {
    refreshBtn = document.getElementById('btn-refresh-metrics');
    refreshSelect = document.getElementById('metrics-refresh-interval');

    refreshBtn?.addEventListener('click', () => loadMetrics());
    refreshSelect?.addEventListener('change', () => {
        stopAutoRefresh();
        const panel = document.getElementById('panel-metrics');
        if (panel?.classList.contains('active')) {
            startAutoRefresh();
        }
    });

    setupAutoRefresh();
}

// ============================================================================
// Auto-refresh
// ============================================================================

function setupAutoRefresh() {
    const panel = document.getElementById('panel-metrics');
    if (!panel) return;

    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            if (mutation.attributeName === 'class') {
                if (panel.classList.contains('active')) {
                    startAutoRefresh();
                } else {
                    stopAutoRefresh();
                }
            }
        }
    });

    observer.observe(panel, { attributes: true });

    if (panel.classList.contains('active')) {
        startAutoRefresh();
    }
}

function getRefreshIntervalMs() {
    const val = parseInt(refreshSelect?.value || '30', 10);
    return val > 0 ? val * 1000 : 0;
}

function startAutoRefresh() {
    if (autoRefreshInterval) return;
    const ms = getRefreshIntervalMs();
    if (ms <= 0) return;

    autoRefreshInterval = setInterval(() => loadMetrics(), ms);
    console.log(`Metrics auto-refresh started (${ms / 1000}s)`);
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        console.log('Metrics auto-refresh stopped');
    }
}

// ============================================================================
// Load Metrics Data
// ============================================================================

export async function loadMetrics() {
    try {
        const [summary, eventsResp] = await Promise.all([
            API.request('/api/observability/summary?minutes=60'),
            API.request('/api/observability/events?limit=200'),
        ]);

        renderKPICards(summary);
        renderTimelineChart(eventsResp.events || []);
        renderDurationChart(eventsResp.events || []);
        renderDistributionChart(summary.events_by_type || {});
        renderErrors(summary.recent_errors || []);
    } catch (e) {
        console.error('Failed to load metrics:', e);
        const kpiContainer = document.getElementById('metrics-kpi-cards');
        if (kpiContainer) {
            kpiContainer.innerHTML = `
                <div style="
                    grid-column: 1 / -1;
                    text-align: center;
                    padding: 2rem;
                    color: var(--text-secondary);
                ">
                    <p>Unable to load metrics data</p>
                    <p style="font-size: 0.8rem; color: var(--text-tertiary);">${escapeHtml(e.message)}</p>
                </div>
            `;
        }
    }
}

// ============================================================================
// KPI Cards
// ============================================================================

function renderKPICards(summary) {
    const container = document.getElementById('metrics-kpi-cards');
    if (!container) return;

    const totalEvents = summary.total_events || 0;
    const errorCount = summary.error_count || 0;
    const errorRate = totalEvents > 0 ? ((errorCount / totalEvents) * 100).toFixed(1) : '0.0';
    const avgDuration = summary.avg_tool_duration_ms != null
        ? `${Math.round(summary.avg_tool_duration_ms)}ms`
        : 'N/A';
    const toolResponses = summary.tool_responses_count || 0;

    const cards = [
        { label: 'Total Events', value: totalEvents.toLocaleString(), color: 'var(--accent-color)' },
        { label: 'Error Rate', value: `${errorRate}%`, color: errorCount > 0 ? 'var(--error)' : 'var(--success)' },
        { label: 'Avg Tool Duration', value: avgDuration, color: 'var(--warning)' },
        { label: 'Tool Responses', value: toolResponses.toLocaleString(), color: 'var(--accent-color)' },
    ];

    container.innerHTML = cards.map(card => `
        <div style="
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem 1.25rem;
            box-shadow: var(--card-shadow);
        ">
            <div style="
                font-size: 0.8rem;
                color: var(--text-tertiary);
                margin-bottom: 0.5rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            ">${card.label}</div>
            <div style="
                font-size: 1.75rem;
                font-weight: 700;
                color: ${card.color};
                line-height: 1;
            ">${card.value}</div>
        </div>
    `).join('');
}

// ============================================================================
// Charts
// ============================================================================

function getChartColors() {
    const style = getComputedStyle(document.documentElement);
    return {
        text: style.getPropertyValue('--text-secondary').trim() || '#475569',
        textTertiary: style.getPropertyValue('--text-tertiary').trim() || '#94a3b8',
        border: style.getPropertyValue('--border-color').trim() || '#e2e8f0',
        accent: style.getPropertyValue('--accent-color').trim() || '#3b82f6',
        success: style.getPropertyValue('--success').trim() || '#22c55e',
        warning: style.getPropertyValue('--warning').trim() || '#f59e0b',
        error: style.getPropertyValue('--error').trim() || '#ef4444',
    };
}

function renderTimelineChart(events) {
    const canvas = document.getElementById('metrics-timeline-chart');
    if (!canvas || typeof Chart === 'undefined') return;

    // Group events into 5-minute buckets
    const buckets = {};
    const typeSet = new Set();

    for (const event of events) {
        if (!event.timestamp) continue;
        const date = new Date(event.timestamp);
        // Round down to 5-minute bucket
        date.setMinutes(Math.floor(date.getMinutes() / 5) * 5, 0, 0);
        const key = date.toISOString();
        const type = event.event_type || 'unknown';
        typeSet.add(type);

        if (!buckets[key]) buckets[key] = {};
        buckets[key][type] = (buckets[key][type] || 0) + 1;
    }

    const sortedKeys = Object.keys(buckets).sort();
    const types = Array.from(typeSet);
    const colors = getChartColors();

    const typeColors = {
        tool_call: colors.accent,
        tool_response: colors.success,
        agent_response: '#8b5cf6',
        llm_call: colors.warning,
        error: colors.error,
        metric: colors.textTertiary,
    };

    const datasets = types.map(type => ({
        label: type,
        data: sortedKeys.map(k => buckets[k][type] || 0),
        borderColor: typeColors[type] || colors.textTertiary,
        backgroundColor: (typeColors[type] || colors.textTertiary) + '33',
        fill: true,
        tension: 0.3,
        pointRadius: 2,
    }));

    const labels = sortedKeys.map(k => {
        const d = new Date(k);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    });

    if (timelineChart) {
        timelineChart.destroy();
    }

    timelineChart = new Chart(canvas, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    ticks: { color: colors.textTertiary, maxTicksLimit: 8 },
                    grid: { color: colors.border + '40' },
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: colors.textTertiary, stepSize: 1 },
                    grid: { color: colors.border + '40' },
                },
            },
            plugins: {
                legend: {
                    labels: { color: colors.text, boxWidth: 12, padding: 8 },
                },
            },
        },
    });
}

function renderDurationChart(events) {
    const canvas = document.getElementById('metrics-duration-chart');
    if (!canvas || typeof Chart === 'undefined') return;

    // Aggregate durations by tool name
    const toolDurations = {};
    for (const event of events) {
        if (event.event_type !== 'tool_response' || !event.duration_ms) continue;
        const name = event.tool_name || 'unknown';
        if (!toolDurations[name]) toolDurations[name] = [];
        toolDurations[name].push(event.duration_ms);
    }

    const toolNames = Object.keys(toolDurations).sort();
    const avgDurations = toolNames.map(name => {
        const durations = toolDurations[name];
        return Math.round(durations.reduce((a, b) => a + b, 0) / durations.length);
    });

    const colors = getChartColors();

    if (durationChart) {
        durationChart.destroy();
    }

    if (toolNames.length === 0) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = colors.textTertiary;
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No tool duration data yet', canvas.width / 2, canvas.height / 2);
        return;
    }

    durationChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: toolNames,
            datasets: [{
                label: 'Avg Duration (ms)',
                data: avgDurations,
                backgroundColor: colors.accent + 'aa',
                borderColor: colors.accent,
                borderWidth: 1,
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: toolNames.length > 6 ? 'y' : 'x',
            scales: {
                x: {
                    ticks: { color: colors.textTertiary },
                    grid: { color: colors.border + '40' },
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: colors.textTertiary },
                    grid: { color: colors.border + '40' },
                },
            },
            plugins: {
                legend: { display: false },
            },
        },
    });
}

function renderDistributionChart(eventsByType) {
    const canvas = document.getElementById('metrics-distribution-chart');
    if (!canvas || typeof Chart === 'undefined') return;

    const labels = Object.keys(eventsByType);
    const data = Object.values(eventsByType);
    const colors = getChartColors();

    const palette = [
        colors.accent,
        colors.success,
        '#8b5cf6',
        colors.warning,
        colors.error,
        colors.textTertiary,
        '#ec4899',
        '#14b8a6',
    ];

    if (distributionChart) {
        distributionChart.destroy();
    }

    if (labels.length === 0) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = colors.textTertiary;
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No event data yet', canvas.width / 2, canvas.height / 2);
        return;
    }

    distributionChart = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data,
                backgroundColor: labels.map((_, i) => palette[i % palette.length] + 'cc'),
                borderColor: labels.map((_, i) => palette[i % palette.length]),
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: colors.text, boxWidth: 12, padding: 8 },
                },
            },
        },
    });
}

// ============================================================================
// Recent Errors Table
// ============================================================================

function renderErrors(errors) {
    const container = document.getElementById('metrics-errors-list');
    if (!container) return;

    if (errors.length === 0) {
        container.innerHTML = `
            <p style="color: var(--text-secondary); font-size: 0.85rem; text-align: center; padding: 1rem;">
                No errors recorded in the last hour
            </p>
        `;
        return;
    }

    container.innerHTML = errors.map(err => {
        const time = err.timestamp ? formatTime(err.timestamp) : '';
        const errType = err.error_type || 'error';
        const msg = err.error_message || 'Unknown error';

        return `
            <div style="
                padding: 0.5rem 0.75rem;
                border-left: 3px solid var(--error);
                background: rgba(239, 68, 68, 0.05);
                margin-bottom: 0.5rem;
                border-radius: 0 4px 4px 0;
                font-size: 0.8125rem;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                    <span style="font-weight: 500; color: var(--error);">${escapeHtml(errType)}</span>
                    <span style="font-size: 0.75rem; color: var(--text-tertiary);">${time}</span>
                </div>
                <div style="color: var(--text-secondary);">${escapeHtml(msg)}</div>
            </div>
        `;
    }).join('');
}

// ============================================================================
// Helpers
// ============================================================================

function formatTime(timestamp) {
    try {
        const date = new Date(timestamp);
        const now = new Date();
        if (date.toDateString() === now.toDateString()) {
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }
        return date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch (e) {
        return timestamp;
    }
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
