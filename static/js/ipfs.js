/**
 * Kestrel Sovereign Console - IPFS Module
 * IPFS Status and connectivity
 */

import API from './api.js';
import { state } from './ui.js';

// ============================================================================
// IPFS Status
// ============================================================================

state.ipfsStatus = null;
state.ipfsStatusVisible = false;

export async function loadIpfsStatus() {
    try {
        const data = await API.getIpfsStatus();
        state.ipfsStatus = data;
        renderIpfsStatus(data);
    } catch (e) {
        const container = document.getElementById('ipfs-status-container');
        if (container) {
            container.innerHTML = `<p style="color: var(--error); padding: 1rem;">Failed to load IPFS status: ${e.message}</p>`;
        }
    }
}

function renderIpfsStatus(data) {
    const container = document.getElementById('ipfs-status-container');
    if (!container) return;

    const localNode = data.local_node;
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
            ${localNode.available ? `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.75rem;">
                    <div>
                        <span style="color: var(--text-secondary);">Version:</span>
                        <span>${localNode.version || 'Unknown'}</span>
                    </div>
                    <div>
                        <span style="color: var(--text-secondary);">Agent:</span>
                        <span>${localNode.agent_version || 'Unknown'}</span>
                    </div>
                    <div style="grid-column: 1 / -1;">
                        <span style="color: var(--text-secondary);">Peer ID:</span>
                        <code style="font-size: 0.65rem; word-break: break-all;">${localNode.peer_id || 'N/A'}</code>
                    </div>
                </div>
            ` : `
                <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 0;">
                    ${localNode.error || 'No local IPFS node detected.<br>Install <a href="https://docs.ipfs.tech/install/ipfs-desktop/" target="_blank" style="color: var(--accent-color);">IPFS Desktop</a> or run <code>ipfs daemon</code>'}
                </p>
            `}
        </div>

        ${data.pinned_content && data.pinned_content.length > 0 ? `
            <div style="background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <h4 style="margin: 0 0 0.75rem 0; font-size: 0.9rem;">\u{1F4CC} Pinned Content (${data.pinned_content.length})</h4>
                <div style="max-height: 150px; overflow-y: auto;">
                    ${data.pinned_content.slice(0, 10).map(pin => `
                        <div style="font-size: 0.7rem; padding: 0.25rem 0; border-bottom: 1px solid var(--border-color);">
                            <code style="word-break: break-all;">${pin.cid}</code>
                        </div>
                    `).join('')}
                    ${data.pinned_content.length > 10 ? `<p style="font-size: 0.7rem; color: var(--text-secondary); margin: 0.5rem 0 0 0;">...and ${data.pinned_content.length - 10} more</p>` : ''}
                </div>
            </div>
        ` : ''}

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
