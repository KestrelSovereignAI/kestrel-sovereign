/**
 * Kestrel Sovereign Console - Feature Store Panel
 * Card grid browser for feature packages: browse, install, enable/disable, configure
 */

import API from './api.js';
import { Modal, Toast, escapeHtml } from './ui.js';

// ============================================================================
// State
// ============================================================================

let allFeatures = [];
let currentFilter = 'all'; // 'all' | 'installed' | 'available'
let searchQuery = '';

// ============================================================================
// Initialization
// ============================================================================

export function initFeatureStore() {
    const filterBtns = document.querySelectorAll('#feature-store-filters .feature-filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFilter = btn.dataset.filter;
            renderFeatureGrid();
        });
    });

    const searchInput = document.getElementById('feature-search');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            searchQuery = e.target.value.toLowerCase().trim();
            renderFeatureGrid();
        });
    }
}

// ============================================================================
// Data Loading
// ============================================================================

export async function loadFeatureStore() {
    const container = document.getElementById('feature-grid');
    if (!container) return;

    container.innerHTML = '<div class="loading">Loading features...</div>';

    try {
        const data = await API.request('/api/features');
        allFeatures = data.features || [];
        renderFeatureGrid();
    } catch (error) {
        console.error('Failed to load features:', error);
        container.innerHTML = `
            <div style="color: var(--error); padding: 2rem; text-align: center;">
                Failed to load features.
                <button onclick="FeatureStore.reload()" class="btn btn-secondary" style="margin-left: 0.5rem;">
                    Retry
                </button>
            </div>
        `;
    }
}

// ============================================================================
// Rendering
// ============================================================================

function getFilteredFeatures() {
    let filtered = allFeatures;

    if (currentFilter === 'installed') {
        filtered = filtered.filter(f => f.status === 'enabled' || f.status === 'disabled' || f.status === 'installed');
    } else if (currentFilter === 'available') {
        filtered = filtered.filter(f => f.status === 'available');
    }

    if (searchQuery) {
        filtered = filtered.filter(f => {
            const name = (f.name || '').toLowerCase();
            const desc = (f.description || '').toLowerCase();
            const tags = (f.tags || []).join(' ').toLowerCase();
            const skillNames = (f.skills || []).map(s => s.name || '').join(' ').toLowerCase();
            return name.includes(searchQuery) ||
                   desc.includes(searchQuery) ||
                   tags.includes(searchQuery) ||
                   skillNames.includes(searchQuery);
        });
    }

    return filtered;
}

function renderFeatureGrid() {
    const container = document.getElementById('feature-grid');
    if (!container) return;

    const features = getFilteredFeatures();

    if (features.length === 0) {
        const msg = searchQuery
            ? 'No features match your search.'
            : currentFilter === 'installed'
                ? 'No installed features.'
                : currentFilter === 'available'
                    ? 'No available features.'
                    : 'No features found.';
        container.innerHTML = `
            <div style="color: var(--text-secondary); padding: 2rem; text-align: center; grid-column: 1 / -1;">
                ${msg}
            </div>
        `;
        return;
    }

    container.innerHTML = features.map(f => renderFeatureCard(f)).join('');
}

function renderFeatureCard(feature) {
    const status = feature.status || 'available';
    const badge = getStatusBadge(status);
    const icon = feature.icon || '📦';
    const name = escapeHtml(feature.name || 'Unknown');
    const description = escapeHtml(feature.description || 'No description');
    const tags = feature.tags || [];
    const skills = feature.skills || [];
    const isCore = feature.core || false;

    const tagHtml = tags.map(t => `
        <span style="
            display: inline-block;
            padding: 0.125rem 0.5rem;
            background: var(--bg-tertiary);
            border-radius: 10px;
            font-size: 0.7rem;
            color: var(--text-secondary);
        ">${escapeHtml(t)}</span>
    `).join('');

    const skillCountHtml = skills.length > 0
        ? `<div style="
            display: flex;
            align-items: center;
            gap: 0.25rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        ">
            <span title="Skills provided">⚡</span>
            <span>${skills.length} skill${skills.length !== 1 ? 's' : ''}</span>
        </div>`
        : '';

    const actionBtn = renderActionButton(feature);

    return `
        <div class="feature-card" data-feature-name="${escapeHtml(feature.name)}" onclick="FeatureStore.showDetail('${escapeHtml(feature.name)}')" style="
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            cursor: pointer;
            transition: box-shadow 0.2s, border-color 0.2s;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        " onmouseover="this.style.boxShadow='0 4px 12px rgba(0,0,0,0.1)';this.style.borderColor='var(--accent-color)'"
           onmouseout="this.style.boxShadow='var(--card-shadow)';this.style.borderColor='var(--border-color)'">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">${icon}</span>
                    <div>
                        <div style="font-weight: 600; font-size: 0.95rem;">${name}</div>
                        ${isCore ? '<span style="font-size: 0.65rem; color: var(--accent-color); font-weight: 600;">CORE</span>' : ''}
                    </div>
                </div>
                ${badge}
            </div>
            <p style="
                margin: 0;
                font-size: 0.825rem;
                color: var(--text-secondary);
                line-height: 1.4;
                flex: 1;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
            ">${description}</p>
            ${skillCountHtml}
            <div style="display: flex; flex-wrap: wrap; gap: 0.25rem; margin-top: 0.25rem;">
                ${tagHtml}
            </div>
            <div style="margin-top: auto; padding-top: 0.5rem;" onclick="event.stopPropagation()">
                ${actionBtn}
            </div>
        </div>
    `;
}

function getStatusBadge(status) {
    const badges = {
        enabled: { color: 'var(--success)', bg: 'rgba(34,197,94,0.1)', text: 'Enabled' },
        disabled: { color: 'var(--text-tertiary)', bg: 'var(--bg-tertiary)', text: 'Disabled' },
        installed: { color: 'var(--accent-color)', bg: 'rgba(59,130,246,0.1)', text: 'Installed' },
        available: { color: 'var(--text-secondary)', bg: 'transparent', text: 'Available' },
    };
    const b = badges[status] || badges.available;
    return `
        <span style="
            display: inline-flex;
            padding: 0.2rem 0.6rem;
            border-radius: 10px;
            font-size: 0.7rem;
            font-weight: 600;
            color: ${b.color};
            background: ${b.bg};
            border: 1px solid ${status === 'available' ? 'var(--border-color)' : 'transparent'};
            white-space: nowrap;
        ">${b.text}</span>
    `;
}

function renderActionButton(feature) {
    const status = feature.status || 'available';

    if (status === 'enabled') {
        return `<button class="feature-action-btn" onclick="FeatureStore.disableFeature('${escapeHtml(feature.name)}')" style="
            width: 100%;
            padding: 0.4rem;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            font-size: 0.8rem;
            cursor: pointer;
        ">Disable</button>`;
    }

    if (status === 'disabled') {
        return `<button class="feature-action-btn" onclick="FeatureStore.enableFeature('${escapeHtml(feature.name)}')" style="
            width: 100%;
            padding: 0.4rem;
            border: none;
            border-radius: 6px;
            background: var(--accent-color);
            color: white;
            font-size: 0.8rem;
            cursor: pointer;
        ">Enable</button>`;
    }

    if (status === 'installed') {
        return `<button class="feature-action-btn" onclick="FeatureStore.enableFeature('${escapeHtml(feature.name)}')" style="
            width: 100%;
            padding: 0.4rem;
            border: none;
            border-radius: 6px;
            background: var(--accent-color);
            color: white;
            font-size: 0.8rem;
            cursor: pointer;
        ">Enable</button>`;
    }

    // available
    return `<button class="feature-action-btn" onclick="FeatureStore.installFeature('${escapeHtml(feature.name)}')" style="
        width: 100%;
        padding: 0.4rem;
        border: none;
        border-radius: 6px;
        background: var(--success);
        color: white;
        font-size: 0.8rem;
        cursor: pointer;
    ">Install</button>`;
}

// ============================================================================
// Actions
// ============================================================================

async function installFeature(name) {
    try {
        Toast.info(`Installing ${name}...`);
        const result = await API.request(`/api/features/${encodeURIComponent(name)}/install`, {
            method: 'POST',
        });
        Toast.success(result.message || `${name} installed`);
        await loadFeatureStore();
    } catch (error) {
        console.error('Install failed:', error);
        Toast.error(`Failed to install ${name}: ${error.message}`);
    }
}

async function enableFeature(name) {
    try {
        await API.request(`/api/features/${encodeURIComponent(name)}/enable`, {
            method: 'POST',
        });
        Toast.success(`${name} enabled`);
        await loadFeatureStore();
    } catch (error) {
        console.error('Enable failed:', error);
        Toast.error(`Failed to enable ${name}: ${error.message}`);
    }
}

async function disableFeature(name) {
    try {
        await API.request(`/api/features/${encodeURIComponent(name)}/disable`, {
            method: 'POST',
        });
        Toast.success(`${name} disabled`);
        await loadFeatureStore();
    } catch (error) {
        console.error('Disable failed:', error);
        Toast.error(`Failed to disable ${name}: ${error.message}`);
    }
}

async function removeFeature(name) {
    const confirmed = await Modal.confirm(
        'Remove Feature',
        `Are you sure you want to remove <strong>${escapeHtml(name)}</strong>? This will uninstall the package.`
    );
    if (!confirmed) return;

    try {
        Toast.info(`Removing ${name}...`);
        const result = await API.request(`/api/features/${encodeURIComponent(name)}/remove`, {
            method: 'POST',
        });
        Toast.success(result.message || `${name} removed`);
        await loadFeatureStore();
    } catch (error) {
        console.error('Remove failed:', error);
        Toast.error(`Failed to remove ${name}: ${error.message}`);
    }
}

// ============================================================================
// Detail Panel
// ============================================================================

async function showDetail(name) {
    try {
        const detail = await API.request(`/api/features/${encodeURIComponent(name)}`);
        renderDetailModal(detail);
    } catch (error) {
        console.error('Failed to load feature detail:', error);
        Toast.error(`Failed to load details for ${name}`);
    }
}

function renderDetailModal(detail) {
    const name = detail.name || 'Unknown';
    const icon = detail.icon || '📦';
    const description = detail.description || detail.tool_description || 'No description';
    const status = detail.status || 'available';
    const badge = getStatusBadge(status);
    const tags = detail.tags || [];
    const skills = detail.skills || [];
    const tools = detail.tools || [];
    const hooks = detail.hooks || [];
    const configSchema = detail.config_schema;
    const gitUrl = detail.git;
    const installInstructions = detail.install_instructions;
    const isCore = detail.core || false;

    // Tools section
    const toolsHtml = tools.length > 0
        ? `<div style="margin-top: 1rem;">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 0.9rem;">Tools (${tools.length})</h4>
            ${tools.map(t => `
                <div style="
                    padding: 0.5rem 0.75rem;
                    background: var(--bg-tertiary);
                    border-radius: 6px;
                    margin-bottom: 0.375rem;
                    font-size: 0.825rem;
                ">
                    <div style="font-weight: 500; font-family: var(--font-mono);">${escapeHtml(t.name)}</div>
                    <div style="color: var(--text-secondary); font-size: 0.75rem; margin-top: 0.125rem;">${escapeHtml(t.description || '')}</div>
                </div>
            `).join('')}
        </div>`
        : '';

    // Skills section
    const skillsHtml = skills.length > 0
        ? `<div style="margin-top: 1rem;">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 0.9rem;">Skills (${skills.length})</h4>
            ${skills.map(s => `
                <div style="
                    padding: 0.5rem 0.75rem;
                    background: var(--bg-tertiary);
                    border-radius: 6px;
                    margin-bottom: 0.375rem;
                    font-size: 0.825rem;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div>
                        <div style="font-weight: 500;">⚡ ${escapeHtml(s.name)}</div>
                        <div style="color: var(--text-secondary); font-size: 0.75rem; margin-top: 0.125rem;">${escapeHtml(s.description || '')}</div>
                    </div>
                    ${s.category ? `<span style="
                        font-size: 0.65rem;
                        padding: 0.125rem 0.4rem;
                        background: var(--bg-primary);
                        border-radius: 8px;
                        color: var(--text-tertiary);
                    ">${escapeHtml(s.category)}</span>` : ''}
                </div>
            `).join('')}
        </div>`
        : '';

    // Hooks section
    const hooksHtml = hooks.length > 0
        ? `<div style="margin-top: 1rem;">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 0.9rem;">Hooks (${hooks.length})</h4>
            ${hooks.map(h => `
                <div style="
                    font-size: 0.8rem;
                    padding: 0.375rem 0.75rem;
                    background: var(--bg-tertiary);
                    border-radius: 6px;
                    margin-bottom: 0.25rem;
                    font-family: var(--font-mono);
                ">${escapeHtml(h.event)}: ${escapeHtml(h.name)}</div>
            `).join('')}
        </div>`
        : '';

    // Config schema section
    const configHtml = configSchema
        ? `<div style="margin-top: 1rem;">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 0.9rem;">Configuration</h4>
            <button onclick="FeatureStore.showConfigForm('${escapeHtml(name)}')" class="btn btn-secondary" style="
                font-size: 0.8rem;
                padding: 0.375rem 0.75rem;
            ">Edit Configuration</button>
        </div>`
        : '';

    // Tags
    const tagsHtml = tags.length > 0
        ? `<div style="display: flex; flex-wrap: wrap; gap: 0.25rem; margin-top: 0.5rem;">
            ${tags.map(t => `<span style="
                display: inline-block;
                padding: 0.125rem 0.5rem;
                background: var(--bg-tertiary);
                border-radius: 10px;
                font-size: 0.7rem;
                color: var(--text-secondary);
            ">${escapeHtml(t)}</span>`).join('')}
        </div>`
        : '';

    // Links
    const linksHtml = [];
    if (gitUrl) {
        linksHtml.push(`<a href="${escapeHtml(gitUrl)}" target="_blank" rel="noopener" style="
            color: var(--accent-color); font-size: 0.825rem; text-decoration: none;
        ">View on GitHub ↗</a>`);
    }
    if (installInstructions) {
        linksHtml.push(`<code style="
            font-size: 0.75rem; padding: 0.25rem 0.5rem;
            background: var(--bg-tertiary); border-radius: 4px;
        ">${escapeHtml(installInstructions)}</code>`);
    }

    // Action buttons for modal footer
    const buttons = [];
    if (status === 'enabled' && !isCore) {
        buttons.push({
            label: 'Disable',
            type: 'secondary',
            onClick: () => { Modal.hide(); disableFeature(name); }
        });
        buttons.push({
            label: 'Remove',
            type: 'danger',
            onClick: () => { Modal.hide(); removeFeature(name); }
        });
    } else if (status === 'disabled' || status === 'installed') {
        buttons.push({
            label: 'Enable',
            type: 'primary',
            onClick: () => { Modal.hide(); enableFeature(name); }
        });
        if (!isCore) {
            buttons.push({
                label: 'Remove',
                type: 'danger',
                onClick: () => { Modal.hide(); removeFeature(name); }
            });
        }
    } else if (status === 'available') {
        buttons.push({
            label: 'Install',
            type: 'primary',
            onClick: () => { Modal.hide(); installFeature(name); }
        });
    }
    buttons.push({
        label: 'Close',
        type: 'secondary',
        onClick: () => Modal.hide()
    });

    Modal.show({
        title: `${icon} ${escapeHtml(name)}`,
        content: `
            <div style="max-height: 60vh; overflow-y: auto;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    ${badge}
                    ${isCore ? '<span style="font-size: 0.7rem; color: var(--accent-color); font-weight: 600;">CORE</span>' : ''}
                </div>
                <p style="margin: 0 0 0.75rem 0; color: var(--text-secondary); font-size: 0.875rem; line-height: 1.5;">
                    ${escapeHtml(description)}
                </p>
                ${tagsHtml}
                ${toolsHtml}
                ${skillsHtml}
                ${hooksHtml}
                ${configHtml}
                ${linksHtml.length > 0 ? `
                    <div style="margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center;">
                        ${linksHtml.join('')}
                    </div>
                ` : ''}
            </div>
        `,
        buttons,
    });
}

// ============================================================================
// Config Form
// ============================================================================

async function showConfigForm(name) {
    try {
        const data = await API.request(`/api/features/${encodeURIComponent(name)}/config`);
        const schema = data.config_schema;
        const currentConfig = data.config || {};

        if (!schema || !schema.properties) {
            Toast.info('This feature has no configurable options.');
            return;
        }

        const properties = schema.properties;
        const required = schema.required || [];

        const fieldsHtml = Object.entries(properties).map(([key, prop]) => {
            const value = currentConfig[key] !== undefined ? currentConfig[key] : prop.default;
            const isRequired = required.includes(key);
            const label = prop.title || key;
            const desc = prop.description || '';

            let inputHtml = '';
            if (prop.type === 'boolean') {
                inputHtml = `
                    <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                        <input type="checkbox" data-config-key="${escapeHtml(key)}" ${value ? 'checked' : ''} style="
                            width: 1rem; height: 1rem; cursor: pointer;
                        ">
                        <span style="font-size: 0.825rem;">${escapeHtml(label)}</span>
                    </label>
                `;
            } else if (prop.enum) {
                inputHtml = `
                    <label style="display: block; font-size: 0.825rem; font-weight: 500; margin-bottom: 0.25rem;">
                        ${escapeHtml(label)}${isRequired ? ' *' : ''}
                    </label>
                    <select data-config-key="${escapeHtml(key)}" style="
                        width: 100%; padding: 0.5rem; border: 1px solid var(--border-color);
                        border-radius: 6px; background: var(--bg-primary); color: var(--text-primary);
                        font-size: 0.825rem;
                    ">
                        ${prop.enum.map(opt => `<option value="${escapeHtml(String(opt))}" ${String(value) === String(opt) ? 'selected' : ''}>${escapeHtml(String(opt))}</option>`).join('')}
                    </select>
                `;
            } else if (prop.type === 'integer' || prop.type === 'number') {
                inputHtml = `
                    <label style="display: block; font-size: 0.825rem; font-weight: 500; margin-bottom: 0.25rem;">
                        ${escapeHtml(label)}${isRequired ? ' *' : ''}
                    </label>
                    <input type="number" data-config-key="${escapeHtml(key)}" value="${value !== undefined ? value : ''}"
                        ${prop.minimum !== undefined ? `min="${prop.minimum}"` : ''}
                        ${prop.maximum !== undefined ? `max="${prop.maximum}"` : ''}
                        style="
                            width: 100%; padding: 0.5rem; border: 1px solid var(--border-color);
                            border-radius: 6px; background: var(--bg-primary); color: var(--text-primary);
                            font-size: 0.825rem;
                        ">
                `;
            } else {
                // Default: string input
                inputHtml = `
                    <label style="display: block; font-size: 0.825rem; font-weight: 500; margin-bottom: 0.25rem;">
                        ${escapeHtml(label)}${isRequired ? ' *' : ''}
                    </label>
                    <input type="text" data-config-key="${escapeHtml(key)}" value="${escapeHtml(String(value || ''))}"
                        placeholder="${escapeHtml(desc)}"
                        style="
                            width: 100%; padding: 0.5rem; border: 1px solid var(--border-color);
                            border-radius: 6px; background: var(--bg-primary); color: var(--text-primary);
                            font-size: 0.825rem;
                        ">
                `;
            }

            return `
                <div style="margin-bottom: 0.75rem;">
                    ${inputHtml}
                    ${desc && prop.type !== 'boolean' ? `<div style="font-size: 0.7rem; color: var(--text-tertiary); margin-top: 0.25rem;">${escapeHtml(desc)}</div>` : ''}
                </div>
            `;
        }).join('');

        Modal.show({
            title: `Configure ${escapeHtml(name)}`,
            content: `
                <form id="feature-config-form" style="max-height: 60vh; overflow-y: auto;">
                    ${fieldsHtml}
                </form>
            `,
            buttons: [
                {
                    label: 'Cancel',
                    type: 'secondary',
                    onClick: () => Modal.hide()
                },
                {
                    label: 'Save',
                    type: 'primary',
                    onClick: () => saveConfig(name, properties)
                },
            ],
        });
    } catch (error) {
        console.error('Failed to load config:', error);
        Toast.error(`Failed to load configuration for ${name}`);
    }
}

async function saveConfig(name, properties) {
    const form = document.getElementById('feature-config-form');
    if (!form) return;

    const config = {};
    for (const [key, prop] of Object.entries(properties)) {
        const el = form.querySelector(`[data-config-key="${key}"]`);
        if (!el) continue;

        if (prop.type === 'boolean') {
            config[key] = el.checked;
        } else if (prop.type === 'integer') {
            config[key] = parseInt(el.value, 10);
        } else if (prop.type === 'number') {
            config[key] = parseFloat(el.value);
        } else {
            config[key] = el.value;
        }
    }

    try {
        await API.request(`/api/features/${encodeURIComponent(name)}/config`, {
            method: 'PATCH',
            body: JSON.stringify({ config }),
        });
        Modal.hide();
        Toast.success('Configuration saved');
    } catch (error) {
        console.error('Failed to save config:', error);
        Toast.error(`Failed to save configuration: ${error.message}`);
    }
}

// ============================================================================
// Global API (for onclick handlers in rendered HTML)
// ============================================================================

window.FeatureStore = {
    reload: loadFeatureStore,
    showDetail,
    showConfigForm,
    installFeature,
    enableFeature,
    disableFeature,
    removeFeature,
};
