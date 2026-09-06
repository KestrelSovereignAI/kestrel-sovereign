/**
 * Kestrel Sovereign Console - Feature Store Panel
 * Card grid browser for feature packages: browse, install, enable/disable, configure
 */

import API from './api.js';
import { Modal, Toast, escapeHtml } from './ui.js';

// Render a feature icon. `name` is typically a kicon key (e.g. "fingerprint",
// "shield"). Falls back to the raw value when it's a literal emoji, and to a
// generic package glyph when the name is unknown to the icon library.
function renderFeatureIcon(name, size) {
    const fontSize = size || '1.5rem';
    const isKiconName = typeof name === 'string' && /^[a-z][a-z0-9-]*$/.test(name);
    if (isKiconName && typeof window.kicon === 'function' && window.KI_PATHS && window.KI_PATHS[name]) {
        return `<span style="font-size: ${fontSize}; line-height: 1; display: inline-flex;">${window.kicon(name, fontSize)}</span>`;
    }
    if (isKiconName && typeof window.kicon === 'function' && window.KI_PATHS && window.KI_PATHS['cabinet']) {
        return `<span style="font-size: ${fontSize}; line-height: 1; display: inline-flex;">${window.kicon('cabinet', fontSize)}</span>`;
    }
    return `<span style="font-size: ${fontSize};">${escapeHtml(name || '')}</span>`;
}

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
    // #879: short-circuit when the host opts out of the feature store.
    // Without this guard, the click handlers and search input would still
    // be wired against panel DOM that initNavigation() may have removed.
    if (!API.hasCapability('featureStore')) return;
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
    // #879: deep-link defense — even if app.js skipped initFeatureStore(),
    // a direct loadFeatureStore() call from a deep-link panel switcher must
    // not fire the /api/features fetch when the host disabled the feature.
    if (!API.hasCapability('featureStore')) return;
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
    const iconHtml = renderFeatureIcon(feature.icon);
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
            <span title="Skills provided">${window.kicon('lightning')}</span>
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
                    ${iconHtml}
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
    const isCore = feature.core || false;

    if (status === 'enabled' && isCore) {
        // A core (baseline) package cannot be disabled per agent — the server
        // answers 409 and points at kestrel.toml (#3234). The detail modal
        // already hides the action for core rows; the card matches it.
        return '';
    }

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
        // The package can install successfully and still have moved
        // kestrel-sovereign underneath the host (#2949). The server restored
        // core before answering — but a swap that happened at all is not a
        // green toast. (A swap it could NOT restore fails the request outright
        // and lands in the catch below.)
        if (result.status === 'installed_with_core_drift') {
            Toast.warning(result.message);
        } else {
            Toast.success(result.message || `${name} installed`);
        }
        await loadFeatureStore();
    } catch (error) {
        console.error('Install failed:', error);
        Toast.error(`Failed to install ${name}: ${error.message}`);
    }
}

async function enableFeature(name) {
    try {
        const resp = await API.request(`/api/features/${encodeURIComponent(name)}/enable`, {
            method: 'POST',
        });
        // #2041: re-derive capabilities from the new enabled set and emit
        // capabilities:changed so the UI re-gates without a page reload.
        if (resp && resp.capabilities) API.applyServerCapabilities(resp.capabilities);
        Toast.success(`${name} enabled`);
        await loadFeatureStore();
    } catch (error) {
        console.error('Enable failed:', error);
        Toast.error(`Failed to enable ${name}: ${error.message}`);
    }
}

async function disableFeature(name) {
    try {
        const resp = await API.request(`/api/features/${encodeURIComponent(name)}/disable`, {
            method: 'POST',
        });
        // #2041: a disabled feature flips its capability false; re-derive and
        // emit so the registry tears its contributions down (no reload).
        if (resp && resp.capabilities) API.applyServerCapabilities(resp.capabilities);
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
    const loadingModal = Modal.show({
        title: 'Feature Details',
        content: '<p style="margin:0;color:var(--text-secondary)">Loading…</p>',
    });
    let responseReceived = false;
    try {
        const detail = await API.request(`/api/features/${encodeURIComponent(name)}`);
        if (!loadingModal.isCurrent()) return;
        responseReceived = true;
        renderDetailModal(detail, loadingModal);
    } catch (error) {
        if (!responseReceived && !loadingModal.isCurrent()) return;
        loadingModal.close();
        console.error('Failed to load feature detail:', error);
        Toast.error(`Failed to load details for ${name}`);
    }
}

function renderDetailModal(detail, owner) {
    const name = detail.name || 'Unknown';
    const iconHtml = renderFeatureIcon(detail.icon, '1.25rem');
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
        ">View on GitHub ${window.kicon('arrow-up-right')}</a>`);
    }
    if (installInstructions) {
        linksHtml.push(`<code style="
            font-size: 0.75rem; padding: 0.25rem 0.5rem;
            background: var(--bg-tertiary); border-radius: 4px;
        ">${escapeHtml(installInstructions)}</code>`);
    }

    // Action buttons for modal footer. Every callback closes the exact detail
    // lifecycle that installed it; it can never dismiss a later permission or
    // configuration dialog.
    let detailModal;
    const buttons = [];
    if (status === 'enabled' && !isCore) {
        buttons.push({
            label: 'Disable',
            type: 'secondary',
            onClick: () => {
                try { detailModal.close(); } finally { disableFeature(name); }
            }
        });
        buttons.push({
            label: 'Remove',
            type: 'danger',
            onClick: () => {
                try { detailModal.close(); } finally { removeFeature(name); }
            }
        });
    } else if (status === 'disabled' || status === 'installed') {
        buttons.push({
            label: 'Enable',
            type: 'primary',
            onClick: () => {
                try { detailModal.close(); } finally { enableFeature(name); }
            }
        });
        if (!isCore) {
            buttons.push({
                label: 'Remove',
                type: 'danger',
                onClick: () => {
                    try { detailModal.close(); } finally { removeFeature(name); }
                }
            });
        }
    } else if (status === 'available') {
        buttons.push({
            label: 'Install',
            type: 'primary',
            onClick: () => {
                try { detailModal.close(); } finally { installFeature(name); }
            }
        });
    }
    buttons.push({
        label: 'Close',
        type: 'secondary',
        onClick: () => detailModal.close()
    });

    detailModal = owner.replace({
        title: `${iconHtml} ${escapeHtml(name)}`,
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
    return detailModal;
}

// ============================================================================
// Config Form
// ============================================================================

// A field is a secret when the schema marks it writeOnly or format=password.
// Standard JSON Schema keywords — see CONFIG_SCHEMA_UI_HINTS.md.
function isSecretField(prop) {
    return prop && (prop.writeOnly === true || prop.format === 'password');
}

// Render a single config field's input. `secretSet` indicates whether a
// write-only secret already has a stored value (so we can hint "unchanged").
function renderConfigField(key, prop, currentConfig, required, secretsSet) {
    const value = currentConfig[key] !== undefined ? currentConfig[key] : prop.default;
    const isRequired = required.includes(key);
    const label = prop.title || key;
    const desc = prop.description || '';
    const labelHtml = `<label style="display: block; font-size: 0.825rem; font-weight: 500; margin-bottom: 0.25rem;">${escapeHtml(label)}${isRequired ? ' *' : ''}</label>`;
    const fieldStyle = `width: 100%; padding: 0.5rem; border: 1px solid var(--border-color); border-radius: 6px; background: var(--bg-primary); color: var(--text-primary); font-size: 0.825rem;`;

    let inputHtml = '';

    if (prop.readOnly === true) {
        // Computed / status field — display only, never submitted.
        const display = value === undefined || value === null || value === '' ? '—' : String(value);
        inputHtml = `${labelHtml}
            <div data-config-key="${escapeHtml(key)}" data-config-readonly="1" style="
                ${fieldStyle} background: var(--bg-tertiary); color: var(--text-secondary);
            ">${escapeHtml(display)}</div>`;
    } else if (isSecretField(prop)) {
        // Write-only secret: never pre-filled; only submitted when changed.
        const isSet = !!(secretsSet && secretsSet[key]);
        const placeholder = isSet ? '•••••••• (unchanged — leave blank to keep)' : (desc || 'Enter a value');
        inputHtml = `${labelHtml}
            <input type="password" autocomplete="new-password" data-config-key="${escapeHtml(key)}" data-config-secret="1"
                value="" placeholder="${escapeHtml(placeholder)}" style="${fieldStyle}">`;
    } else if (prop.type === 'boolean') {
        inputHtml = `
            <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                <input type="checkbox" data-config-key="${escapeHtml(key)}" ${value ? 'checked' : ''} style="
                    width: 1rem; height: 1rem; cursor: pointer;
                ">
                <span style="font-size: 0.825rem;">${escapeHtml(label)}</span>
            </label>`;
    } else if (prop.enum) {
        inputHtml = `${labelHtml}
            <select data-config-key="${escapeHtml(key)}" style="${fieldStyle}">
                ${prop.enum.map(opt => `<option value="${escapeHtml(String(opt))}" ${String(value) === String(opt) ? 'selected' : ''}>${escapeHtml(String(opt))}</option>`).join('')}
            </select>`;
    } else if (prop.type === 'integer' || prop.type === 'number') {
        inputHtml = `${labelHtml}
            <input type="number" data-config-key="${escapeHtml(key)}" value="${value !== undefined ? value : ''}"
                ${prop.minimum !== undefined ? `min="${prop.minimum}"` : ''}
                ${prop.maximum !== undefined ? `max="${prop.maximum}"` : ''}
                style="${fieldStyle}">`;
    } else if (prop.format === 'textarea') {
        inputHtml = `${labelHtml}
            <textarea data-config-key="${escapeHtml(key)}" rows="4" placeholder="${escapeHtml(desc)}"
                style="${fieldStyle} resize: vertical;">${escapeHtml(String(value || ''))}</textarea>`;
    } else {
        inputHtml = `${labelHtml}
            <input type="text" data-config-key="${escapeHtml(key)}" value="${escapeHtml(String(value || ''))}"
                placeholder="${escapeHtml(desc)}" style="${fieldStyle}">`;
    }

    const showDesc = desc && prop.type !== 'boolean' && !isSecretField(prop);
    return `
        <div style="margin-bottom: 0.75rem;">
            ${inputHtml}
            ${showDesc ? `<div style="font-size: 0.7rem; color: var(--text-tertiary); margin-top: 0.25rem;">${escapeHtml(desc)}</div>` : ''}
        </div>`;
}

// Resolve sections from an optional `x-kestrel-ui.sections` hint. Fields not
// named by any section are collected into a trailing unlabelled group so a
// partial section list still renders every property.
function resolveSections(properties, ui) {
    const allKeys = Object.keys(properties);
    const declared = Array.isArray(ui.sections) ? ui.sections : null;
    if (!declared || declared.length === 0) {
        return [{ title: '', description: '', fields: allKeys }];
    }
    const sections = declared.map(s => ({
        title: s.title || '',
        description: s.description || '',
        fields: (s.fields || []).filter(k => properties[k] !== undefined),
    }));
    const grouped = new Set(sections.flatMap(s => s.fields));
    const leftover = allKeys.filter(k => !grouped.has(k));
    if (leftover.length) sections.push({ title: '', description: '', fields: leftover });
    return sections;
}

// Render action buttons declared via `x-kestrel-ui.actions`. Each action is
// {label, method, path, confirm?} and targets the feature's own router.
function renderConfigActions(actions) {
    if (!Array.isArray(actions) || actions.length === 0) return '';
    const btns = actions.map((a, i) => {
        const meta = escapeHtml(JSON.stringify({
            label: a.label || 'Run',
            method: (a.method || 'POST').toUpperCase(),
            path: a.path || '',
            confirm: a.confirm || '',
        }));
        return `<button type="button" class="btn btn-secondary" data-config-action="${meta}"
            onclick="FeatureStore.runConfigAction(this)" style="font-size: 0.8rem; padding: 0.375rem 0.75rem;">
            ${escapeHtml(a.label || 'Run')}
        </button>`;
    }).join('');
    return `
        <div style="margin-top: 1rem; border-top: 1px solid var(--border-color); padding-top: 0.75rem;">
            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">${btns}</div>
            <div id="feature-config-action-result" style="font-size: 0.75rem; margin-top: 0.5rem;"></div>
        </div>`;
}

async function runConfigAction(btn) {
    let action;
    try {
        action = JSON.parse(btn.getAttribute('data-config-action'));
    } catch (e) {
        return;
    }
    if (!action.path) return;
    if (action.confirm) {
        const ok = await Modal.confirm(action.label || 'Confirm', escapeHtml(action.confirm));
        if (!ok) return;
    }

    const resultEl = btn.closest('.modal-container')
        ?.querySelector('#feature-config-action-result');
    const setResult = (text, color) => {
        if (resultEl) {
            resultEl.textContent = text;
            resultEl.style.color = color;
        }
    };

    const original = btn.textContent;
    btn.disabled = true;
    setResult('Running…', 'var(--text-secondary)');
    try {
        const opts = { method: action.method };
        if (action.method !== 'GET' && action.method !== 'HEAD') {
            opts.body = JSON.stringify({});
        }
        const result = await API.request(action.path, opts);
        // Honor a {ok, message} convention; otherwise treat 2xx as success.
        const ok = result && typeof result.ok === 'boolean' ? result.ok : true;
        const message = (result && result.message) || (ok ? 'Success' : 'Failed');
        if (ok) {
            setResult(`✓ ${message}`, 'var(--success)');
            Toast.success(message);
        } else {
            setResult(`✗ ${message}`, 'var(--error)');
            Toast.error(message);
        }
    } catch (error) {
        console.error('Config action failed:', error);
        setResult(`✗ ${error.message}`, 'var(--error)');
        Toast.error(`Action failed: ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = original;
    }
}

async function showConfigForm(name) {
    const loadingModal = Modal.show({
        title: `Configure ${escapeHtml(name)}`,
        content: '<p style="margin:0;color:var(--text-secondary)">Loading…</p>',
    });
    let responseReceived = false;
    try {
        const data = await API.request(`/api/features/${encodeURIComponent(name)}/config`);
        if (!loadingModal.isCurrent()) return;
        responseReceived = true;
        const schema = data.config_schema;
        const currentConfig = data.config || {};
        const secretsSet = data.secrets_set || {};

        if (!schema || !schema.properties) {
            loadingModal.close();
            Toast.info('This feature has no configurable options.');
            return;
        }

        const properties = schema.properties;
        const required = schema.required || [];
        const ui = schema['x-kestrel-ui'] || {};

        const sections = resolveSections(properties, ui);
        const sectionsHtml = sections.map(sec => {
            const fields = sec.fields
                .map(key => renderConfigField(key, properties[key], currentConfig, required, secretsSet))
                .join('');
            const header = sec.title
                ? `<h4 style="margin: 0 0 0.25rem 0; font-size: 0.85rem; color: var(--text-primary);">${escapeHtml(sec.title)}</h4>`
                : '';
            const subdesc = sec.description
                ? `<p style="margin: 0 0 0.5rem 0; font-size: 0.72rem; color: var(--text-tertiary);">${escapeHtml(sec.description)}</p>`
                : '';
            return `<div style="margin-bottom: 1rem;">${header}${subdesc}${fields}</div>`;
        }).join('');

        const actionsHtml = renderConfigActions(ui.actions);

        let configModal;
        configModal = loadingModal.replace({
            title: `Configure ${escapeHtml(name)}`,
            content: `
                <div style="max-height: 60vh; overflow-y: auto;">
                    <form id="feature-config-form">
                        ${sectionsHtml}
                    </form>
                    ${actionsHtml}
                </div>
            `,
            buttons: [
                {
                    label: 'Cancel',
                    type: 'secondary',
                    onClick: () => configModal.close()
                },
                {
                    label: 'Save',
                    type: 'primary',
                    onClick: () => saveConfig(name, properties, configModal)
                },
            ],
        });
    } catch (error) {
        if (!responseReceived && !loadingModal.isCurrent()) return;
        loadingModal.close();
        console.error('Failed to load config:', error);
        Toast.error(`Failed to load configuration for ${name}`);
    }
}

async function saveConfig(name, properties, owner) {
    const form = owner.querySelector('#feature-config-form');
    if (!form) return;

    const config = {};
    for (const [key, prop] of Object.entries(properties)) {
        const el = form.querySelector(`[data-config-key="${key}"]`);
        if (!el) continue;

        // Read-only/computed fields are never submitted.
        if (el.getAttribute('data-config-readonly') === '1') continue;

        if (isSecretField(prop)) {
            // Write-only: only include when the user typed a new value, so an
            // untouched secret is omitted and the stored value is preserved.
            if (el.value !== '') config[key] = el.value;
            continue;
        }

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
        owner.close();
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
    runConfigAction,
    installFeature,
    enableFeature,
    disableFeature,
    removeFeature,
};
