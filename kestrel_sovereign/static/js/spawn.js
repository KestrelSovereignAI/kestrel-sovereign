/**
 * Kestrel Sovereign Console - Spawn Panel
 * Active children, delegation chains, budget meters, TTL countdowns, spawn history
 */

import API from './api.js';

// ============================================================================
// DOM References & State
// ============================================================================

let refreshBtn = null;
let refreshSelect = null;
let autoRefreshInterval = null;
let budgetChart = null;

// ============================================================================
// Initialization
// ============================================================================

export function initSpawn() {
    refreshBtn = document.getElementById('btn-refresh-spawn');
    refreshSelect = document.getElementById('spawn-refresh-interval');

    refreshBtn?.addEventListener('click', () => loadSpawn());
    refreshSelect?.addEventListener('change', () => {
        stopAutoRefresh();
        const panel = document.getElementById('panel-spawn');
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
    const panel = document.getElementById('panel-spawn');
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
    const val = parseInt(refreshSelect?.value || '10', 10);
    return val > 0 ? val * 1000 : 0;
}

function startAutoRefresh() {
    if (autoRefreshInterval) return;
    const ms = getRefreshIntervalMs();
    if (ms <= 0) return;

    autoRefreshInterval = setInterval(() => loadSpawn(), ms);
    console.log(`Spawn auto-refresh started (${ms / 1000}s)`);
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        console.log('Spawn auto-refresh stopped');
    }
}

// ============================================================================
// Load Spawn Data
// ============================================================================

export async function loadSpawn() {
    try {
        const data = await API.request('/api/spawn/children');
        renderChildren(data.children || []);
        renderDelegationChain(data.delegation_chain || {});
        renderBudgetChart(data.children || []);
        renderHistory(data.history || []);
    } catch (e) {
        console.error('Failed to load spawn data:', e);
        const container = document.getElementById('spawn-children-list');
        if (container) {
            container.innerHTML = `
                <div style="
                    text-align: center;
                    padding: 2rem;
                    color: var(--text-secondary);
                ">
                    <p>Unable to load spawn data</p>
                    <p style="font-size: 0.8rem; color: var(--text-tertiary);">${escapeHtml(e.message)}</p>
                </div>
            `;
        }
    }
}

// ============================================================================
// Active Children Table
// ============================================================================

function renderChildren(children) {
    const container = document.getElementById('spawn-children-list');
    if (!container) return;

    if (children.length === 0) {
        container.innerHTML = `
            <div style="
                text-align: center;
                padding: 2rem;
                color: var(--text-secondary);
                font-size: 0.875rem;
            ">
                No active child agents. Use the chat to spawn agents with the spawn_agent tool.
            </div>
        `;
        return;
    }

    const rows = children.map(child => {
        const statusStyle = getStatusStyle(child.status);
        const didDisplay = child.did ? truncateDid(child.did) : 'N/A';
        const ttlDisplay = formatTtl(child.ttl_remaining);
        const budgetPct = child.budget_allocated > 0
            ? Math.round((child.budget_spent / child.budget_allocated) * 100)
            : 0;

        return `
            <div style="
                padding: 1rem;
                border: 1px solid var(--border-color);
                border-radius: 8px;
                margin-bottom: 0.75rem;
                background: var(--bg-secondary);
            ">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-weight: 600; font-size: 0.95rem;">${escapeHtml(child.name)}</span>
                        <span style="
                            display: inline-block;
                            padding: 0.15rem 0.5rem;
                            border-radius: 12px;
                            font-size: 0.75rem;
                            font-weight: 600;
                            ${statusStyle}
                        ">${escapeHtml(child.status)}</span>
                    </div>
                    <span style="
                        font-family: monospace;
                        font-size: 0.8rem;
                        color: var(--text-tertiary);
                    " title="${escapeHtml(child.did)}">${escapeHtml(didDisplay)}</span>
                </div>

                ${child.purpose ? `
                    <div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
                        ${escapeHtml(child.purpose)}
                    </div>
                ` : ''}

                <div style="display: flex; gap: 1.5rem; align-items: center; flex-wrap: wrap;">
                    <!-- TTL Countdown -->
                    <div style="display: flex; align-items: center; gap: 0.35rem; font-size: 0.8rem; color: var(--text-secondary);">
                        <span style="font-size: 0.9rem;">&#x23F1;</span>
                        <span>TTL: <strong style="color: ${child.ttl_remaining < 60 ? 'var(--error)' : child.ttl_remaining < 300 ? 'var(--warning)' : 'var(--text-primary)'}">${ttlDisplay}</strong></span>
                    </div>

                    <!-- Budget Meter -->
                    ${child.budget_allocated > 0 ? `
                        <div style="flex: 1; min-width: 150px;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 0.25rem;">
                                <span>Budget</span>
                                <span>${budgetPct}% used</span>
                            </div>
                            <div style="
                                width: 100%;
                                height: 8px;
                                background: var(--bg-tertiary);
                                border-radius: 4px;
                                overflow: hidden;
                            ">
                                <div style="
                                    width: ${budgetPct}%;
                                    height: 100%;
                                    background: ${budgetPct > 90 ? 'var(--error)' : budgetPct > 70 ? 'var(--warning)' : 'var(--accent-color)'};
                                    border-radius: 4px;
                                    transition: width 0.3s ease;
                                "></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--text-tertiary); margin-top: 0.15rem;">
                                <span>Spent: ${child.budget_spent.toFixed(4)}</span>
                                <span>Remaining: ${child.budget_remaining.toFixed(4)}</span>
                            </div>
                        </div>
                    ` : `
                        <div style="font-size: 0.8rem; color: var(--text-tertiary);">No budget</div>
                    `}
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = rows;
}

// ============================================================================
// Delegation Chain Visualization
// ============================================================================

function renderDelegationChain(chain) {
    const container = document.getElementById('spawn-delegation-chain');
    if (!container) return;

    if (!chain.name) {
        container.innerHTML = '<div style="color: var(--text-secondary); font-size: 0.85rem; padding: 0.5rem;">No delegation chain</div>';
        return;
    }

    container.innerHTML = renderChainNode(chain, 0);
}

function renderChainNode(node, depth) {
    const indent = depth * 1.5;
    const statusColor = node.status === 'running' ? 'var(--success)' : 'var(--text-tertiary)';
    const connector = depth > 0 ? '<span style="color: var(--text-tertiary); margin-right: 0.35rem;">&#x2514;&#x2500;</span>' : '';
    const didShort = node.did ? truncateDid(node.did) : '';

    let html = `
        <div style="margin-left: ${indent}rem; padding: 0.35rem 0; display: flex; align-items: center; gap: 0.5rem;">
            ${connector}
            <span style="
                width: 8px; height: 8px;
                border-radius: 50%;
                background: ${statusColor};
                display: inline-block;
                flex-shrink: 0;
            "></span>
            <span style="font-weight: ${depth === 0 ? '600' : '500'}; font-size: 0.875rem;">${escapeHtml(node.name)}</span>
            ${node.purpose ? `<span style="font-size: 0.75rem; color: var(--text-tertiary);">— ${escapeHtml(node.purpose)}</span>` : ''}
            ${didShort ? `<span style="font-family: monospace; font-size: 0.7rem; color: var(--text-tertiary);">${escapeHtml(didShort)}</span>` : ''}
        </div>
    `;

    if (node.children && node.children.length > 0) {
        for (const child of node.children) {
            html += renderChainNode(child, depth + 1);
        }
    }

    return html;
}

// ============================================================================
// Budget Chart (Chart.js)
// ============================================================================

function renderBudgetChart(children) {
    const canvas = document.getElementById('spawn-budget-chart');
    if (!canvas || typeof Chart === 'undefined') return;

    const withBudget = children.filter(c => c.budget_allocated > 0);

    if (withBudget.length === 0) {
        if (budgetChart) {
            budgetChart.destroy();
            budgetChart = null;
        }
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const colors = getChartColors();
        ctx.fillStyle = colors.textTertiary;
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No budget data — spawn agents with a budget to see meters', canvas.width / 2, canvas.height / 2);
        return;
    }

    const labels = withBudget.map(c => c.name);
    const spent = withBudget.map(c => c.budget_spent);
    const remaining = withBudget.map(c => c.budget_remaining);
    const colors = getChartColors();

    if (budgetChart) {
        budgetChart.destroy();
    }

    budgetChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Spent',
                    data: spent,
                    backgroundColor: colors.warning + 'cc',
                    borderColor: colors.warning,
                    borderWidth: 1,
                    borderRadius: 4,
                },
                {
                    label: 'Remaining',
                    data: remaining,
                    backgroundColor: colors.success + 'cc',
                    borderColor: colors.success,
                    borderWidth: 1,
                    borderRadius: 4,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    stacked: true,
                    ticks: { color: colors.textTertiary },
                    grid: { color: colors.border + '40' },
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    ticks: { color: colors.textTertiary },
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

// ============================================================================
// Spawn History
// ============================================================================

function renderHistory(history) {
    const container = document.getElementById('spawn-history-list');
    if (!container) return;

    if (history.length === 0) {
        container.innerHTML = `
            <p style="color: var(--text-secondary); font-size: 0.85rem; text-align: center; padding: 1rem;">
                No spawn events recorded
            </p>
        `;
        return;
    }

    container.innerHTML = history.map(entry => {
        const isSpawn = entry.event === 'spawned';
        const borderColor = isSpawn ? 'var(--success)' : 'var(--warning)';
        const icon = isSpawn ? '&#x25B6;' : '&#x25A0;';
        const time = entry.started_at ? formatTime(entry.started_at) : '';

        return `
            <div style="
                padding: 0.5rem 0.75rem;
                border-left: 3px solid ${borderColor};
                background: ${isSpawn ? 'rgba(34, 197, 94, 0.05)' : 'rgba(245, 158, 11, 0.05)'};
                margin-bottom: 0.5rem;
                border-radius: 0 4px 4px 0;
                font-size: 0.8125rem;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                    <span style="font-weight: 500;">
                        <span>${icon}</span>
                        ${escapeHtml(entry.child_name)}
                        <span style="color: var(--text-tertiary); font-weight: 400;"> — ${escapeHtml(entry.event)}</span>
                    </span>
                    <span style="font-size: 0.75rem; color: var(--text-tertiary);">${time}</span>
                </div>
                ${entry.status && entry.status !== 'running' ? `
                    <div style="font-size: 0.75rem; color: var(--text-secondary);">
                        Status: ${escapeHtml(entry.status)}
                        ${entry.budget_consumed ? ` · Budget consumed: ${entry.budget_consumed.toFixed(4)}` : ''}
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

// ============================================================================
// Helpers
// ============================================================================

function getStatusStyle(status) {
    switch (status) {
        case 'running':
            return 'background: rgba(34, 197, 94, 0.15); color: var(--success);';
        case 'completed':
            return 'background: rgba(59, 130, 246, 0.15); color: var(--accent-color);';
        case 'terminated':
        case 'timed_out':
            return 'background: rgba(245, 158, 11, 0.15); color: var(--warning);';
        case 'failed':
            return 'background: rgba(239, 68, 68, 0.15); color: var(--error);';
        default:
            return 'background: var(--bg-tertiary); color: var(--text-secondary);';
    }
}

function truncateDid(did) {
    if (!did || did.length <= 24) return did;
    return did.substring(0, 16) + '...' + did.substring(did.length - 6);
}

function formatTtl(seconds) {
    if (seconds <= 0) return 'Expired';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

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

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
