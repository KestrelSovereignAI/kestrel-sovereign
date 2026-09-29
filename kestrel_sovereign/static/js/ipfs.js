/**
 * Kestrel Sovereign Console - IPFS Module
 * IPFS Status and connectivity
 */

import API from './api.js';
import { state, renderTextError } from './ui.js';

// ============================================================================
// IPFS Status
// ============================================================================

state.ipfsStatus = null;
state.ipfsStatusVisible = false;

export async function loadIpfsStatus() {
    try {
        const data = await API.getIpfsStatus();
        // The agent-local status carries this agent's view only. The node's
        // identity, version and full pin set are the host's (#3226): fetch
        // them from the sovereign-gated node route when the server says this
        // caller would be admitted, and draw nothing that would only 403.
        let node = null;
        if (data.can_view_node === true) {
            try {
                node = await API.getIpfsNode();
            } catch (e) {
                console.warn('IPFS node view unavailable:', e.message);
            }
        }
        state.ipfsStatus = data;
        state.ipfsNode = node;
        renderIpfsStatus(data, node);
    } catch (e) {
        const container = document.getElementById('ipfs-status-container');
        renderTextError(container, `Failed to load IPFS status: ${e.message}`);
    }
}

function renderPinList(title, pins, total = null) {
    if (!pins || pins.length === 0) return '';
    const count = typeof total === 'number' ? total : pins.length;
    return `
        <div style="background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <h4 style="margin: 0 0 0.75rem 0; font-size: 0.9rem;">${title} (${count})</h4>
            <div style="max-height: 150px; overflow-y: auto;">
                ${pins.slice(0, 10).map(pin => `
                    <div style="font-size: 0.7rem; padding: 0.25rem 0; border-bottom: 1px solid var(--border-color);">
                        <code style="word-break: break-all;">${pin.cid}</code>
                    </div>
                `).join('')}
                ${count > 10 ? `<p style="font-size: 0.7rem; color: var(--text-secondary); margin: 0.5rem 0 0 0;">...and ${count - 10} more</p>` : ''}
            </div>
        </div>
    `;
}

function renderIpfsStatus(data, node = null) {
    const container = document.getElementById('ipfs-status-container');
    if (!container) return;

    const localNode = data.local_node;
    const nodeIdentity = node && node.local_node ? node.local_node : null;
    const nodePins = node && Array.isArray(node.pinned_content) ? node.pinned_content : null;
    const gateways = data.gateways || [];

    container.innerHTML = `
        <div style="background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <h4 style="margin: 0; font-size: 0.9rem;">\u{1F4BB} Local IPFS Node</h4>
                <span style="
                    background: ${localNode.available ? 'var(--success)' : 'var(--error)'};
                    color: white;
                    font-size: 0.7rem;
                    padding: 0.25rem 0.5rem;
                    border-radius: 10px;
                ">${localNode.available ? '\u25CF Connected' : '\u25CB Offline'}</span>
            </div>
            ${localNode.available && nodeIdentity ? `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.75rem;">
                    <div>
                        <span style="color: var(--text-secondary);">Version:</span>
                        <span>${nodeIdentity.version || 'Unknown'}</span>
                    </div>
                    <div>
                        <span style="color: var(--text-secondary);">Agent:</span>
                        <span>${nodeIdentity.agent_version || 'Unknown'}</span>
                    </div>
                    <div style="grid-column: 1 / -1;">
                        <span style="color: var(--text-secondary);">Peer ID:</span>
                        <code style="font-size: 0.65rem; word-break: break-all;">${nodeIdentity.peer_id || 'N/A'}</code>
                    </div>
                </div>
            ` : localNode.available ? `
                <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 0;">
                    Reachable. Node identity and the full pin set are host details, shown to the sovereign only.
                </p>
            ` : `
                <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 0;">
                    ${localNode.error || 'No local IPFS node detected.<br>Install <a href="https://docs.ipfs.tech/install/ipfs-desktop/" target="_blank" style="color: var(--accent-color);">IPFS Desktop</a> or run <code>ipfs daemon</code>'}
                </p>
            `}
        </div>

        ${renderPinList('\u{1F4CC} My Pinned Exports', data.pinned_content)}
        ${nodePins ? renderPinList('\u{1F5C4} All Pins on This Node', nodePins, node.pinned_total) : ''}

        <div style="background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem;">
            <h4 style="margin: 0 0 0.75rem 0; font-size: 0.9rem;">\u{1F310} Public Gateways</h4>
            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                ${gateways.map(gw => `
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; background: var(--bg-tertiary); border-radius: 4px;">
                        <span style="font-size: 0.8rem;">${gw.name}</span>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            ${gw.latency_ms ? `<span style="font-size: 0.7rem; color: var(--text-secondary);">${gw.latency_ms}ms</span>` : ''}
                            <span style="
                                width: 8px;
                                height: 8px;
                                border-radius: 50%;
                                background: ${gw.available ? 'var(--success)' : 'var(--error)'};
                            "></span>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>

        <button onclick="loadIpfsStatus()" class="btn btn-secondary" style="width: 100%; margin-top: 1rem; padding: 0.5rem;">
            \u{1F504} Refresh Status
        </button>
    `;
}

window.toggleIpfsStatus = function() {
    const container = document.getElementById('ipfs-status-section');
    const toggleBtn = document.getElementById('toggle-ipfs-status');

    if (!container || !toggleBtn) return;

    state.ipfsStatusVisible = !state.ipfsStatusVisible;

    if (state.ipfsStatusVisible) {
        container.style.display = 'block';
        toggleBtn.textContent = '\u{1F310} Hide IPFS Status';
        if (!state.ipfsStatus) {
            loadIpfsStatus();
        }
    } else {
        container.style.display = 'none';
        toggleBtn.textContent = '\u{1F310} IPFS Status';
    }
};
