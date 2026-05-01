/**
 * Kestrel Sovereign Console - Sovereignty Module
 * Sovereignty panel with export/import functionality
 *
 * Related modules:
 * - files.js: Local file browser
 * - ipfs.js: IPFS status and connectivity
 */

import API from './api.js';
import { state, Toast, Modal, formatBytes } from './ui.js';

// Re-export from related modules
export { loadLocalFiles } from './files.js';
export { loadIpfsStatus } from './ipfs.js';

// ============================================================================
// UI Helpers
// ============================================================================

function showLoading(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.innerHTML = '<div class="loading">Loading</div>';
}

function showError(elementId, message) {
    const el = document.getElementById(elementId);
    if (el) el.innerHTML = `<div style="color: var(--error); padding: 1rem;">${message}</div>`;
}

// ============================================================================
// Sovereignty Panel - Exports
// ============================================================================

export async function loadExports() {
    // #879: deep-link defense — no /api/sovereignty fetch when disabled.
    if (!API.hasCapability('sovereignty')) return;
    showLoading('export-list');
    try {
        const data = await API.getSovereigntyExports();
        state.exports = data;
        renderExports(data);
    } catch (e) {
        showError('export-list', `Failed to load exports: ${e.message}`);
    }
}

function getTierColor(tier) {
    const tierLower = (tier || '').toLowerCase();
    switch (tierLower) {
        case 'filecoin':
            return { bg: '#dcfce7', text: '#15803d' };
        case 'ipfs':
            return { bg: '#dbeafe', text: '#1d4ed8' };
        case 'local_only':
        case 'local':
            return { bg: '#fef3c7', text: '#92400e' };
        default:
            return { bg: '#e2e8f0', text: '#475569' };
    }
}

function renderExports(data) {
    const list = document.getElementById('export-list');
    const combined = [...(data.exports || []), ...(data.backups || [])];
    const seen = new Set();
    const allExports = combined.filter(e => {
        const key = e.cid || e.node_id;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });

    if (allExports.length === 0) {
        list.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">No exports yet. Click "Export to IPFS" to create your first backup.</p>';
        return;
    }

    list.innerHTML = allExports.map((exp, index) => {
        const tierColor = getTierColor(exp.storage_tier);
        const cid = exp.cid || '';
        const truncatedCid = cid.length > 20 ? cid.slice(0, 10) + '...' + cid.slice(-8) : cid;
        const timestamp = exp.created_at ? new Date(exp.created_at).toLocaleString() : 'Unknown';
        const isIPFS = exp.storage_tier === 'IPFS' || exp.storage_tier === 'ipfs';
        const isFilecoin = exp.storage_tier === 'FILECOIN' || exp.storage_tier === 'filecoin';

        return `
            <div class="export-card" data-export-index="${index}" style="
                border: 1px solid var(--border-color);
                border-radius: 12px;
                background: var(--bg-secondary);
                overflow: hidden;
                transition: border-color 0.2s;
            ">
                <div class="export-header" style="
                    padding: 1rem;
                    cursor: pointer;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                " onclick="toggleExportDetails(${index})">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="
                            padding: 0.25rem 0.625rem;
                            border-radius: 6px;
                            font-size: 0.7rem;
                            font-weight: 600;
                            background: ${tierColor.bg};
                            color: ${tierColor.text};
                        ">
                            ${(exp.storage_tier || 'IPFS').toUpperCase()}
                        </span>
                        <span style="font-size: 0.875rem; color: var(--text-secondary);">
                            ${exp.encrypted ? '\u{1F510}' : '\u{1F513}'} ${timestamp}
                        </span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        ${cid ? `
                            <button onclick="event.stopPropagation(); copyToClipboard('${cid}')" class="btn-icon" style="
                                background: var(--bg-tertiary);
                                border: none;
                                border-radius: 6px;
                                padding: 0.375rem 0.625rem;
                                cursor: pointer;
                                font-size: 0.8rem;
                                color: var(--text-primary);
                                display: flex;
                                align-items: center;
                                gap: 0.375rem;
                                transition: background 0.2s;
                            " title="Copy CID">
                                \u{1F4CB} Copy
                            </button>
                        ` : ''}
                        <span class="expand-icon" style="
                            font-size: 1rem;
                            transition: transform 0.2s;
                            color: var(--text-tertiary);
                        ">\u25BC</span>
                    </div>
                </div>

                ${cid ? `
                    <div style="
                        padding: 0 1rem 0.75rem 1rem;
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                    ">
                        <span style="
                            font-family: var(--font-mono);
                            font-size: 0.75rem;
                            background: var(--bg-tertiary);
                            padding: 0.375rem 0.625rem;
                            border-radius: 6px;
                            color: var(--text-secondary);
                            flex: 1;
                            overflow: hidden;
                            text-overflow: ellipsis;
                            white-space: nowrap;
                        " title="${cid}">${truncatedCid}</span>
                        ${(isIPFS || isFilecoin) && cid ? `
                            <a href="https://ipfs.io/ipfs/${cid}" target="_blank" rel="noopener" style="
                                font-size: 0.75rem;
                                color: var(--accent-color);
                                text-decoration: none;
                                display: flex;
                                align-items: center;
                                gap: 0.25rem;
                            " title="View on IPFS Gateway">
                                \u{1F517} View
                            </a>
                        ` : ''}
                    </div>
                ` : ''}

                <div class="export-details" data-details-index="${index}" style="
                    display: none;
                    border-top: 1px solid var(--border-color);
                    padding: 1rem;
                    background: var(--bg-tertiary);
                ">
                    <div style="display: grid; gap: 0.75rem; font-size: 0.8rem;">
                        <div style="display: grid; grid-template-columns: 120px 1fr; gap: 0.5rem;">
                            <span style="color: var(--text-tertiary);">Full CID:</span>
                            <span style="font-family: var(--font-mono); word-break: break-all; color: var(--text-primary);">${cid || 'N/A'}</span>
                        </div>
                        <div style="display: grid; grid-template-columns: 120px 1fr; gap: 0.5rem;">
                            <span style="color: var(--text-tertiary);">Storage Tier:</span>
                            <span style="color: var(--text-primary);">${(exp.storage_tier || 'IPFS').toUpperCase()}</span>
                        </div>
                        <div style="display: grid; grid-template-columns: 120px 1fr; gap: 0.5rem;">
                            <span style="color: var(--text-tertiary);">Encrypted:</span>
                            <span style="color: var(--text-primary);">${exp.encrypted ? 'Yes (AES-256-GCM)' : 'No'}</span>
                        </div>
                        <div style="display: grid; grid-template-columns: 120px 1fr; gap: 0.5rem;">
                            <span style="color: var(--text-tertiary);">Created:</span>
                            <span style="color: var(--text-primary);">${timestamp}</span>
                        </div>
                        ${exp.shard_count ? `
                            <div style="display: grid; grid-template-columns: 120px 1fr; gap: 0.5rem;">
                                <span style="color: var(--text-tertiary);">Shards:</span>
                                <span style="color: var(--text-primary);">${exp.shard_count}</span>
                            </div>
                        ` : ''}
                        ${exp.total_size ? `
                            <div style="display: grid; grid-template-columns: 120px 1fr; gap: 0.5rem;">
                                <span style="color: var(--text-tertiary);">Total Size:</span>
                                <span style="color: var(--text-primary);">${formatBytes(exp.total_size)}</span>
                            </div>
                        ` : ''}
                        ${exp.local_path ? `
                            <div style="display: grid; grid-template-columns: 120px 1fr; gap: 0.5rem;">
                                <span style="color: var(--text-tertiary);">Local Path:</span>
                                <span style="font-family: var(--font-mono); font-size: 0.75rem; word-break: break-all; color: var(--text-secondary);">${exp.local_path}</span>
                            </div>
                        ` : ''}
                    </div>

                    <div style="margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        ${cid ? `
                            <button onclick="copyToClipboard('${cid}')" class="btn btn-secondary" style="font-size: 0.75rem; padding: 0.375rem 0.75rem;">
                                \u{1F4CB} Copy Full CID
                            </button>
                        ` : ''}
                        ${(isIPFS || isFilecoin) && cid ? `
                            <a href="https://ipfs.io/ipfs/${cid}" target="_blank" rel="noopener" class="btn btn-secondary" style="
                                font-size: 0.75rem;
                                padding: 0.375rem 0.75rem;
                                text-decoration: none;
                                display: inline-flex;
                                align-items: center;
                                gap: 0.25rem;
                            ">
                                \u{1F310} IPFS Gateway
                            </a>
                            <a href="https://explore.ipld.io/#/explore/${cid}" target="_blank" rel="noopener" class="btn btn-secondary" style="
                                font-size: 0.75rem;
                                padding: 0.375rem 0.75rem;
                                text-decoration: none;
                                display: inline-flex;
                                align-items: center;
                                gap: 0.25rem;
                            ">
                                \u{1F50D} IPLD Explorer
                            </a>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

window.toggleExportDetails = function(index) {
    const details = document.querySelector(`.export-details[data-details-index="${index}"]`);
    const card = document.querySelector(`.export-card[data-export-index="${index}"]`);
    const icon = card?.querySelector('.expand-icon');

    if (details && icon) {
        const isHidden = details.style.display === 'none';
        details.style.display = isHidden ? 'block' : 'none';
        icon.style.transform = isHidden ? 'rotate(180deg)' : 'rotate(0)';

        if (card) {
            card.style.borderColor = isHidden ? 'var(--accent-color)' : 'var(--border-color)';
        }
    }
};

// ============================================================================
// Export Modal
// ============================================================================

function showExportModal() {
    Modal.show({
        title: 'Export Agent Data',
        content: `
            <p style="margin: 0 0 1.25rem 0; color: var(--text-secondary); line-height: 1.6;">
                Export your agent's data to create a backup. You own this data and can restore it anytime.
            </p>

            <div style="margin-bottom: 1.25rem;">
                <label style="display: block; margin-bottom: 0.5rem; font-weight: 500; font-size: 0.875rem;">
                    Storage Tier
                </label>
                <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                    <label style="
                        display: flex;
                        align-items: center;
                        gap: 0.75rem;
                        padding: 0.75rem;
                        background: var(--bg-tertiary);
                        border-radius: 8px;
                        cursor: pointer;
                        transition: background 0.2s;
                    " onmouseover="this.style.background='var(--bg-primary)'" onmouseout="this.style.background='var(--bg-tertiary)'">
                        <input type="radio" name="export-tier" value="LOCAL_ONLY" style="accent-color: var(--accent-color);">
                        <div>
                            <div style="font-weight: 500;">Local Only</div>
                            <div style="font-size: 0.8rem; color: var(--text-secondary);">Store in local cache (free)</div>
                        </div>
                    </label>
                    <label style="
                        display: flex;
                        align-items: center;
                        gap: 0.75rem;
                        padding: 0.75rem;
                        background: var(--bg-tertiary);
                        border-radius: 8px;
                        cursor: pointer;
                        transition: background 0.2s;
                    " onmouseover="this.style.background='var(--bg-primary)'" onmouseout="this.style.background='var(--bg-tertiary)'">
                        <input type="radio" name="export-tier" value="IPFS" checked style="accent-color: var(--accent-color);">
                        <div>
                            <div style="font-weight: 500;">IPFS</div>
                            <div style="font-size: 0.8rem; color: var(--text-secondary);">Decentralized storage (recommended)</div>
                        </div>
                    </label>
                    <label style="
                        display: flex;
                        align-items: center;
                        gap: 0.75rem;
                        padding: 0.75rem;
                        background: var(--bg-tertiary);
                        border-radius: 8px;
                        cursor: pointer;
                        transition: background 0.2s;
                    " onmouseover="this.style.background='var(--bg-primary)'" onmouseout="this.style.background='var(--bg-tertiary)'">
                        <input type="radio" name="export-tier" value="FILECOIN" style="accent-color: var(--accent-color);">
                        <div>
                            <div style="font-weight: 500;">Filecoin</div>
                            <div style="font-size: 0.8rem; color: var(--text-secondary);">Long-term archival storage</div>
                        </div>
                    </label>
                </div>
            </div>

            <div style="margin-bottom: 0.5rem;">
                <label style="
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    cursor: pointer;
                ">
                    <input type="checkbox" id="export-encrypt" checked style="
                        width: 18px;
                        height: 18px;
                        accent-color: var(--accent-color);
                    ">
                    <div>
                        <span style="font-weight: 500;">Encrypt backup</span>
                        <span style="font-size: 0.8rem; color: var(--text-secondary); margin-left: 0.5rem;">(recommended)</span>
                    </div>
                </label>
            </div>
        `,
        buttons: [
            { label: 'Cancel', type: 'secondary', onClick: () => Modal.hide() },
            { label: 'Export', type: 'primary', onClick: async () => {
                const tierInput = document.querySelector('input[name="export-tier"]:checked');
                const encryptInput = document.getElementById('export-encrypt');

                const tier = tierInput?.value || 'IPFS';
                const encrypt = encryptInput?.checked ?? true;

                Modal.hide();

                try {
                    Toast.info('Starting export...');
                    const result = await API.exportSovereignty(tier, encrypt);
                    Toast.success(result.message || 'Export completed successfully!');
                    loadExports();
                } catch (e) {
                    Toast.error(`Export failed: ${e.message}`);
                }
            }}
        ]
    });
}

// ============================================================================
// Import Modal
// ============================================================================

function showImportModal() {
    Modal.show({
        title: 'Import from CID',
        content: `
            <p style="margin: 0 0 1.25rem 0; color: var(--text-secondary); line-height: 1.6;">
                Restore your agent data from an IPFS Content Identifier (CID).
            </p>

            <div style="margin-bottom: 1.25rem;">
                <label style="display: block; margin-bottom: 0.5rem; font-weight: 500; font-size: 0.875rem;">
                    IPFS CID
                </label>
                <div style="display: flex; gap: 0.5rem;">
                    <input type="text" id="import-cid-input"
                        placeholder="Qm... or bafy..."
                        style="
                            flex: 1;
                            padding: 0.75rem 1rem;
                            border: 1px solid var(--border-color);
                            border-radius: 8px;
                            font-size: 0.9rem;
                            font-family: var(--font-mono);
                            background: var(--bg-primary);
                            color: var(--text-primary);
                            outline: none;
                            transition: border-color 0.2s;
                        "
                        onfocus="this.style.borderColor='var(--accent-color)'"
                        onblur="this.style.borderColor='var(--border-color)'"
                    />
                    <button id="paste-cid-btn" style="
                        padding: 0.75rem;
                        border: 1px solid var(--border-color);
                        border-radius: 8px;
                        background: var(--bg-tertiary);
                        color: var(--text-primary);
                        cursor: pointer;
                        font-size: 1rem;
                        transition: all 0.2s;
                    " title="Paste from clipboard">\u{1F4CB}</button>
                </div>
            </div>

            <div style="
                padding: 0.75rem;
                background: var(--bg-tertiary);
                border-radius: 8px;
                font-size: 0.8rem;
                color: var(--text-secondary);
            ">
                <strong style="color: var(--text-primary);">Tip:</strong> The CID was provided when you exported your data.
                It looks like <code style="background: var(--bg-primary); padding: 0.125rem 0.375rem; border-radius: 3px;">QmXy...</code> or
                <code style="background: var(--bg-primary); padding: 0.125rem 0.375rem; border-radius: 3px;">bafyb...</code>
            </div>
        `,
        buttons: [
            { label: 'Cancel', type: 'secondary', onClick: () => Modal.hide() },
            { label: 'Import', type: 'primary', onClick: async () => {
                const cidInput = document.getElementById('import-cid-input');
                const cid = cidInput?.value?.trim();

                if (!cid) {
                    Toast.warning('Please enter a CID');
                    return;
                }

                if (!cid.startsWith('Qm') && !cid.startsWith('bafy')) {
                    Toast.warning('CID should start with "Qm" or "bafy"');
                    return;
                }

                Modal.hide();

                try {
                    Toast.info('Starting import...');
                    const result = await API.importSovereignty(cid);
                    Toast.success(result.message || 'Import completed successfully!');
                    loadExports();
                } catch (e) {
                    Toast.error(`Import failed: ${e.message}`);
                }
            }}
        ]
    });

    setTimeout(() => {
        const pasteBtn = document.getElementById('paste-cid-btn');
        const cidInput = document.getElementById('import-cid-input');

        if (pasteBtn && cidInput) {
            pasteBtn.addEventListener('click', async () => {
                try {
                    const text = await navigator.clipboard.readText();
                    cidInput.value = text.trim();
                    cidInput.focus();
                    Toast.success('Pasted from clipboard');
                } catch (e) {
                    Toast.error('Could not access clipboard');
                }
            });

            cidInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    document.querySelector('.modal-btn-primary')?.click();
                }
            });
        }
    }, 50);
}

// Attach button handlers
export function initSovereigntyButtons() {
    document.getElementById('btn-export-ipfs')?.addEventListener('click', () => {
        showExportModal();
    });

    document.getElementById('btn-import')?.addEventListener('click', () => {
        showImportModal();
    });
}
