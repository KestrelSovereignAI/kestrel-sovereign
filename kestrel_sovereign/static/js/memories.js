/**
 * Kestrel Sovereign Console - Memories Module
 * Constitution and Memories panels
 */

import API from './api.js';
import { state, Toast, Modal, truncate, escapeHtml, truncateId } from './ui.js';

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
// Constitution Panel
// ============================================================================

export async function loadConstitution() {
    // #879: deep-link defense — no /api/constitution fetch when disabled.
    if (!API.hasCapability('constitution')) return;
    showLoading('constitution-content');
    try {
        const data = await API.getConstitution();
        state.constitution = data;

        document.getElementById('constitution-hash').textContent = data.hash ? truncate(data.hash, 16) : '';
        document.getElementById('constitution-hash').title = data.hash || '';

        if (data.text) {
            const html = typeof marked !== 'undefined' ? marked.parse(data.text) : data.text.replace(/\n/g, '<br>');
            document.getElementById('constitution-content').innerHTML = html;
        } else {
            document.getElementById('constitution-content').innerHTML = '<p style="color: var(--text-secondary);">Constitution text not available.</p>';
        }
    } catch (e) {
        showError('constitution-content', `Failed to load constitution: ${e.message}`);
    }
}

// ============================================================================
// Memories Panel
// ============================================================================

export async function loadMemories(nodeType = null) {
    // #879: deep-link defense — no /api/memories fetch when disabled.
    if (!API.hasCapability('memory')) return;
    showLoading('memory-list');
    try {
        const data = await API.getMemories(nodeType);
        state.memories = data;
        renderMemories(data.nodes);
    } catch (e) {
        showError('memory-list', `Failed to load memories: ${e.message}`);
    }
}

function renderMemories(nodes) {
    const list = document.getElementById('memory-list');
    if (!nodes || nodes.length === 0) {
        list.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">No memories found.</p>';
        return;
    }

    list.innerHTML = nodes.map(node => `
        <div class="memory-item">
            <div>
                <span class="type-badge">${node.node_type}</span>
                <span style="margin-left: 0.75rem; font-size: 0.875rem;">${truncate(node.label || node.node_id, 40)}</span>
            </div>
            <div style="display: flex; gap: 0.5rem;">
                <button class="btn btn-secondary" style="padding: 0.25rem 0.5rem; font-size: 0.75rem;" onclick="viewMemory('${node.node_id}')" title="View details">\u{1F441}</button>
                ${!['agent', 'document'].includes(node.node_type) ? `<button class="btn btn-danger" style="padding: 0.25rem 0.5rem; font-size: 0.75rem;" onclick="deleteMemory('${node.node_id}')" title="Delete">\u{1F5D1}</button>` : ''}
            </div>
        </div>
    `).join('');
}

window.viewMemory = async function(nodeId) {
    try {
        const detail = await API.getMemoryDetail(nodeId);
        showMemoryModal(detail);
    } catch (e) {
        Toast.error(`Failed to load memory details: ${e.message}`);
    }
};

function showMemoryModal(detail) {
    const existing = document.getElementById('memory-modal');
    if (existing) existing.remove();

    const node = detail.node;
    const relationships = detail.relationships;
    const content = detail.content;

    const modal = document.createElement('div');
    modal.id = 'memory-modal';
    modal.style.cssText = `
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0,0,0,0.7); z-index: 1000;
        display: flex; align-items: center; justify-content: center;
        padding: 1rem;
    `;

    modal.innerHTML = `
        <div style="
            background: var(--bg-primary, #1a1a2e);
            border-radius: 12px;
            max-width: 800px;
            width: 100%;
            max-height: 90vh;
            overflow-y: auto;
            padding: 1.5rem;
            color: var(--text-primary, #e0e0e0);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h2 style="margin: 0; font-size: 1.25rem;">
                    <span style="
                        display: inline-block;
                        padding: 0.25rem 0.5rem;
                        border-radius: 4px;
                        font-size: 0.75rem;
                        font-weight: 600;
                        background: ${node.node_type === 'agent' ? '#3b82f6' : node.node_type === 'document' ? '#10b981' : '#8b5cf6'};
                        color: white;
                        margin-right: 0.5rem;
                    ">${node.node_type.toUpperCase()}</span>
                    ${node.label || 'Unnamed'}
                </h2>
                <button onclick="document.getElementById('memory-modal').remove()" style="
                    background: none; border: none; font-size: 1.5rem;
                    cursor: pointer; color: var(--text-secondary, #888);
                ">&times;</button>
            </div>

            <div style="margin-bottom: 1rem;">
                <div style="font-size: 0.75rem; color: var(--text-tertiary, #666); margin-bottom: 0.25rem;">Node ID</div>
                <div style="
                    font-family: monospace;
                    font-size: 0.8rem;
                    background: var(--bg-tertiary, #2a2a3e);
                    padding: 0.5rem;
                    border-radius: 4px;
                    word-break: break-all;
                ">${node.node_id}</div>
            </div>

            <div style="margin-bottom: 1rem;">
                <div style="font-size: 0.75rem; color: var(--text-tertiary, #666); margin-bottom: 0.25rem;">Properties</div>
                <pre style="
                    font-family: monospace;
                    font-size: 0.8rem;
                    background: var(--bg-tertiary, #2a2a3e);
                    padding: 0.75rem;
                    border-radius: 4px;
                    overflow-x: auto;
                    margin: 0;
                ">${JSON.stringify(node.properties, null, 2)}</pre>
            </div>

            ${relationships.incoming.length > 0 || relationships.outgoing.length > 0 ? `
                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 0.75rem; color: var(--text-tertiary, #666); margin-bottom: 0.5rem;">Relationships</div>
                    <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                        ${relationships.incoming.map(r => `
                            <div style="
                                display: flex; align-items: center; gap: 0.5rem;
                                background: var(--bg-tertiary, #2a2a3e);
                                padding: 0.5rem;
                                border-radius: 4px;
                                font-size: 0.8rem;
                            ">
                                <span style="color: #10b981;">\u2190</span>
                                <span style="font-family: monospace; word-break: break-all;">${truncateId(r.source)}</span>
                                <span style="
                                    background: #4f46e5;
                                    color: white;
                                    padding: 0.125rem 0.375rem;
                                    border-radius: 3px;
                                    font-size: 0.7rem;
                                ">${r.type}</span>
                                <span>this</span>
                            </div>
                        `).join('')}
                        ${relationships.outgoing.map(r => `
                            <div style="
                                display: flex; align-items: center; gap: 0.5rem;
                                background: var(--bg-tertiary, #2a2a3e);
                                padding: 0.5rem;
                                border-radius: 4px;
                                font-size: 0.8rem;
                            ">
                                <span>this</span>
                                <span style="
                                    background: #4f46e5;
                                    color: white;
                                    padding: 0.125rem 0.375rem;
                                    border-radius: 3px;
                                    font-size: 0.7rem;
                                ">${r.type}</span>
                                <span style="color: #10b981;">\u2192</span>
                                <span style="font-family: monospace; word-break: break-all;">${truncateId(r.target)}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            ${content ? `
                <div>
                    <div style="font-size: 0.75rem; color: var(--text-tertiary, #666); margin-bottom: 0.5rem;">Content</div>
                    <div style="
                        background: var(--bg-tertiary, #2a2a3e);
                        padding: 1rem;
                        border-radius: 4px;
                        max-height: 300px;
                        overflow-y: auto;
                        font-size: 0.85rem;
                        line-height: 1.5;
                        white-space: pre-wrap;
                        font-family: monospace;
                    ">${escapeHtml(content)}</div>
                </div>
            ` : ''}
        </div>
    `;

    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });

    document.body.appendChild(modal);
}

window.deleteMemory = async function(nodeId) {
    const confirmed = await Modal.confirm(
        'Delete Memory',
        `Are you sure you want to delete memory <code style="font-size: 0.8rem;">${truncate(nodeId, 20)}</code>? This action cannot be undone.`
    );
    if (!confirmed) return;
    try {
        await API.deleteMemory(nodeId);
        Toast.success('Memory deleted');
        loadMemories(document.getElementById('memory-filter').value || null);
    } catch (e) {
        Toast.error(`Failed to delete: ${e.message}`);
    }
};

// Memory filter event - attach after DOM ready
export function initMemoryFilter() {
    document.getElementById('memory-filter')?.addEventListener('change', (e) => {
        loadMemories(e.target.value || null);
    });
}
