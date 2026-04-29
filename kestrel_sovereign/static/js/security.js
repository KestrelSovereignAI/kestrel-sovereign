/**
 * Kestrel Security Module - Permission management and approval queue UI
 */

import { Modal, Toast } from './ui.js';
import API from './api.js';
import { subscribeSSE } from './chat.js';

export const Security = {
    pendingApprovals: new Map(),
    permissionTree: [],
    _initialized: false,
    // Modal is a singleton — if two approval_request events arrive in quick
    // succession, showing both concurrently would stack overlays in the DOM
    // and leave the older one stuck. Serialize them through a queue. #748.
    _approvalQueue: [],
    _approvalDraining: false,
    _seenApprovalIds: new Set(),

    // === Initialization ===

    async init() {
        if (this._initialized) return;

        // Set up SSE handler for approval requests
        this._setupSSEHandler();

        // Load initial data when Security panel is opened
        const securityTab = document.querySelector('[data-panel="security"]');
        if (securityTab) {
            securityTab.addEventListener('click', () => this.loadSecurityPanel());
        }

        this._initialized = true;
        console.log('Security module initialized');
    },

    _setupSSEHandler() {
        // Route approval_request events from the notification SSE stream into
        // the approval modal. subscribeSSE handles reconnects so the handler
        // survives network drops. See #748.
        subscribeSSE('approval_request', (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleApprovalRequest(data);
            } catch (err) {
                console.error('Failed to parse approval_request event:', err);
            }
        });
    },

    // === Approval Request Handling ===

    handleApprovalRequest(data) {
        console.log('Approval request received:', data);

        // Dedupe: the SSE stream can redeliver the same event on reconnect,
        // and the UI must not prompt twice for the same approval_id.
        if (this._seenApprovalIds.has(data.id) || this.pendingApprovals.has(data.id)) {
            return;
        }
        this._seenApprovalIds.add(data.id);

        this.pendingApprovals.set(data.id, data);
        this.updatePendingBadge(this.pendingApprovals.size);

        this._approvalQueue.push(data);
        // Fire-and-forget drain; errors are handled inside the drain loop.
        this._drainApprovalQueue();
    },

    async _drainApprovalQueue() {
        if (this._approvalDraining) return;
        this._approvalDraining = true;
        try {
            while (this._approvalQueue.length > 0) {
                const data = this._approvalQueue.shift();
                let decision;
                try {
                    decision = await this.showApprovalModal(data);
                } catch (err) {
                    console.error('Approval modal error:', err);
                    decision = { approved: false, scope: 'once' };
                }

                try {
                    await this.submitApproval(data.id, decision.approved, decision.scope);
                } catch (err) {
                    console.error('Failed to submit approval:', err);
                }

                this.pendingApprovals.delete(data.id);
                this.updatePendingBadge(this.pendingApprovals.size);

                const pendingContainer = document.getElementById('pending-approvals');
                if (pendingContainer) {
                    await this.loadPendingApprovals();
                }
            }
        } finally {
            this._approvalDraining = false;
        }
    },

    async showApprovalModal(data) {
        return new Promise((resolve) => {
            const argsHtml = data.args && Object.keys(data.args).length > 0
                ? `<pre class="args-preview" style="
                    background: var(--bg-tertiary);
                    padding: 0.75rem;
                    border-radius: 8px;
                    font-size: 0.8rem;
                    overflow-x: auto;
                    max-height: 150px;
                    margin-top: 0.75rem;
                ">${this._escapeHtml(JSON.stringify(data.args, null, 2))}</pre>`
                : '';

            Modal.show({
                title: kicon('lock') + ' Permission Required',
                content: `
                    <div class="security-approval">
                        <p style="margin: 0 0 0.5rem 0;">
                            <strong>Feature:</strong> ${this._escapeHtml(data.feature)}
                        </p>
                        <p style="margin: 0 0 0.5rem 0;">
                            <strong>Tool:</strong> ${this._escapeHtml(data.tool)}
                        </p>
                        ${argsHtml}
                        <p style="margin-top: 1rem; color: var(--text-secondary); font-size: 0.875rem;">
                            Choose how to handle this permission:
                        </p>
                    </div>
                `,
                buttons: [
                    {
                        label: `${kicon('x-mark')} Deny`,
                        type: 'danger',
                        onClick: () => {
                            Modal.hide();
                            resolve({ approved: false, scope: 'once' });
                        }
                    },
                    {
                        label: 'This Time',
                        type: 'secondary',
                        onClick: () => {
                            Modal.hide();
                            resolve({ approved: true, scope: 'once' });
                        }
                    },
                    {
                        label: 'This Session',
                        type: 'primary',
                        onClick: () => {
                            Modal.hide();
                            resolve({ approved: true, scope: 'session' });
                        }
                    },
                    {
                        label: `${kicon('checkmark')} Always`,
                        type: 'primary',
                        onClick: () => {
                            Modal.hide();
                            resolve({ approved: true, scope: 'always' });
                        }
                    }
                ],
                onClose: () => resolve({ approved: false, scope: 'once' })
            });
        });
    },

    async submitApproval(approvalId, approved, scope) {
        try {
            const response = await API.request('/api/security/approve', {
                method: 'POST',
                body: JSON.stringify({
                    approval_id: approvalId,
                    approved,
                    scope
                })
            });

            if (response.success) {
                Toast.success(approved
                    ? `Approved (${scope})`
                    : 'Denied'
                );
            } else {
                Toast.error('Failed to submit decision');
            }

            return response;
        } catch (error) {
            // 404 here almost always means the server-side request_approval
            // call already returned (timed out at 5min default, or was
            // cancelled). The user sees a raw stack otherwise — replace it
            // with a clear message and skip the redundant error toast.
            const msg = (error && error.message) || '';
            if (msg.includes('not found') || msg.includes('expired')) {
                Toast.warning('This approval already expired on the server — the caller moved on.');
                console.warn('Approval expired before user submitted decision:', approvalId);
            } else {
                console.error('Failed to submit approval:', error);
                Toast.error('Failed to submit decision');
            }
            return { success: false };
        }
    },

    // === Permission Tree UI ===

    async loadSecurityPanel() {
        await Promise.all([
            this.loadPermissionTree(),
            this.loadPendingApprovals(),
            this.loadAuditLog()
        ]);
    },

    async loadPermissionTree() {
        try {
            const response = await API.request('/api/security/permissions/tree');
            this.permissionTree = response.tree || [];
            this.renderPermissionTree();
        } catch (error) {
            console.error('Failed to load permission tree:', error);
            const container = document.getElementById('permission-tree');
            if (container) {
                container.innerHTML = `
                    <p style="color: var(--error);">
                        Failed to load permissions.
                        <button onclick="Security.loadPermissionTree()" class="btn btn-secondary">
                            Retry
                        </button>
                    </p>
                `;
            }
        }
    },

    renderPermissionTree() {
        const container = document.getElementById('permission-tree');
        if (!container) return;

        if (this.permissionTree.length === 0) {
            container.innerHTML = `
                <p class="empty-state" style="color: var(--text-secondary); padding: 1rem;">
                    No permissions configured yet. Tools will be registered when first invoked.
                </p>
            `;
            return;
        }

        container.innerHTML = this.permissionTree.map(feature => `
            <div class="permission-feature" data-feature="${this._escapeHtml(feature.name)}" style="
                border: 1px solid var(--border-color);
                border-radius: 8px;
                margin-bottom: 0.75rem;
                overflow: hidden;
            ">
                <div class="feature-header" onclick="Security.toggleFeature('${this._escapeHtml(feature.name)}')" style="
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 0.75rem 1rem;
                    background: var(--bg-tertiary);
                    cursor: pointer;
                ">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span class="expand-icon">▼</span>
                        <span class="feature-name" style="font-weight: 500;">${this._escapeHtml(feature.name)}</span>
                    </div>
                    <div class="rollup-control" onclick="event.stopPropagation()">
                        ${this.renderRollupControl(feature)}
                    </div>
                </div>
                <div class="feature-tools" style="padding: 0.5rem 1rem;">
                    ${feature.tools.map(tool => `
                        <div class="permission-tool" data-tool="${this._escapeHtml(tool.name)}" style="
                            display: flex;
                            align-items: center;
                            justify-content: space-between;
                            padding: 0.5rem 0;
                            border-bottom: 1px solid var(--border-color);
                        ">
                            <span class="tool-name" style="font-family: var(--font-mono); font-size: 0.875rem;">
                                ${this._escapeHtml(tool.name)}
                            </span>
                            <select class="permission-select"
                                    onchange="Security.setToolPermission('${this._escapeHtml(feature.name)}', '${this._escapeHtml(tool.name)}', this.value)"
                                    style="
                                        padding: 0.25rem 0.5rem;
                                        border: 1px solid var(--border-color);
                                        border-radius: 4px;
                                        background: var(--bg-primary);
                                        color: var(--text-primary);
                                        font-size: 0.8rem;
                                        cursor: pointer;
                                    ">
                                <option value="allow" ${tool.level === 'allow' ? 'selected' : ''}>${kicon('check-box')} Allow</option>
                                <option value="ask" ${tool.level === 'ask' ? 'selected' : ''}>${kicon('empty-box')} Ask</option>
                                <option value="deny" ${tool.level === 'deny' ? 'selected' : ''}>${kicon('x-box')} Deny</option>
                            </select>
                        </div>
                    `).join('')}
                </div>
            </div>
        `).join('');
    },

    renderRollupControl(feature) {
        const state = feature.rollup_state;
        const icons = {
            'allow_all': kicon('check-box'),
            'deny_all': kicon('x-box'),
            'ask_all': kicon('empty-box'),
            'session_all': kicon('half-circle'),
            'mixed': kicon('half-circle')
        };
        const labels = {
            'allow_all': 'Allow All',
            'deny_all': 'Deny All',
            'ask_all': 'Ask All',
            'session_all': 'Session All',
            'mixed': 'Mixed'
        };

        return `
            <button class="rollup-btn"
                    onclick="Security.cycleFeaturePermission('${this._escapeHtml(feature.name)}', '${state}')"
                    title="Click to change"
                    style="
                        padding: 0.25rem 0.75rem;
                        border: 1px solid var(--border-color);
                        border-radius: 4px;
                        background: var(--bg-primary);
                        color: var(--text-primary);
                        font-size: 0.8rem;
                        cursor: pointer;
                    ">
                ${icons[state] || '?'} ${labels[state] || 'Mixed'}
            </button>
        `;
    },

    async setToolPermission(featureName, toolName, level) {
        try {
            // #766: DENY toggles run through the demo-isolation rail.
            // The header carries the operator's intent into the audit log.
            const headers = level === 'deny'
                ? { 'X-Kestrel-Allow-Destructive': 'user-initiated-ui-deny' }
                : {};
            await API.request('/api/security/permissions', {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    feature: featureName,
                    tool: toolName,
                    level
                })
            });

            Toast.success(`Set ${toolName} to ${level}`);
            await this.loadPermissionTree(); // Refresh to update rollup
        } catch (error) {
            console.error('Failed to set permission:', error);
            Toast.error('Failed to set permission');
        }
    },

    async cycleFeaturePermission(featureName, currentState) {
        // Cycle: mixed → ask → allow → deny → ask
        const nextLevel = {
            'mixed': 'ask',
            'ask_all': 'allow',
            'allow_all': 'deny',
            'deny_all': 'ask',
            'session_all': 'ask'
        }[currentState] || 'ask';

        try {
            // #766: bulk DENY runs through the demo-isolation rail.
            const headers = nextLevel === 'deny'
                ? { 'X-Kestrel-Allow-Destructive': 'user-initiated-ui-deny' }
                : {};
            await API.request('/api/security/permissions/feature', {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    feature: featureName,
                    level: nextLevel
                })
            });

            Toast.success(`Set ${featureName} to ${nextLevel}`);
            await this.loadPermissionTree();
        } catch (error) {
            console.error('Failed to set feature permission:', error);
            Toast.error('Failed to set permission');
        }
    },

    toggleFeature(featureName) {
        const el = document.querySelector(`[data-feature="${featureName}"]`);
        if (!el) return;

        const tools = el.querySelector('.feature-tools');
        const icon = el.querySelector('.expand-icon');

        if (tools.style.display === 'none') {
            tools.style.display = 'block';
            icon.textContent = '▼';
        } else {
            tools.style.display = 'none';
            icon.textContent = '▶';
        }
    },

    // === Pending Approvals UI ===

    async loadPendingApprovals() {
        const container = document.getElementById('pending-approvals');
        if (!container) return;

        try {
            const response = await API.request('/api/security/pending');
            const pending = response.pending || [];

            this.updatePendingBadge(pending.length);

            if (pending.length === 0) {
                container.innerHTML = `
                    <p class="empty-state" style="color: var(--text-secondary); padding: 0.5rem 0;">
                        No pending approvals
                    </p>
                `;
                return;
            }

            container.innerHTML = pending.map(req => `
                <div class="pending-item" data-id="${req.id}" style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 0.75rem;
                    background: var(--bg-tertiary);
                    border-radius: 8px;
                    margin-bottom: 0.5rem;
                ">
                    <div class="pending-info">
                        <strong style="font-size: 0.9rem;">${this._escapeHtml(req.feature)}.${this._escapeHtml(req.tool)}</strong>
                        <span class="pending-time" style="
                            display: block;
                            font-size: 0.75rem;
                            color: var(--text-secondary);
                        ">${this._formatRelativeTime(req.timestamp)}</span>
                    </div>
                    <div class="pending-actions" style="display: flex; gap: 0.5rem;">
                        <button onclick="Security.quickDeny('${req.id}')"
                                class="btn btn-danger" style="padding: 0.25rem 0.5rem; font-size: 0.8rem;">
                            Deny
                        </button>
                        <button onclick="Security.quickApprove('${req.id}')"
                                class="btn btn-primary" style="padding: 0.25rem 0.5rem; font-size: 0.8rem;">
                            Allow
                        </button>
                    </div>
                </div>
            `).join('');
        } catch (error) {
            console.error('Failed to load pending approvals:', error);
            container.innerHTML = `
                <p style="color: var(--error);">Failed to load pending approvals</p>
            `;
        }
    },

    async quickApprove(requestId) {
        await this.submitApproval(requestId, true, 'once');
        await this.loadPendingApprovals();
    },

    async quickDeny(requestId) {
        await this.submitApproval(requestId, false, 'once');
        await this.loadPendingApprovals();
    },

    // === Audit Log UI ===

    async loadAuditLog() {
        const container = document.getElementById('security-audit-log');
        if (!container) return;

        try {
            const response = await API.request('/api/security/audit?limit=50');
            const logs = response.logs || [];

            if (logs.length === 0) {
                container.innerHTML = `
                    <p class="empty-state" style="color: var(--text-secondary); padding: 0.5rem 0;">
                        No audit log entries yet
                    </p>
                `;
                return;
            }

            container.innerHTML = `
                <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                    <thead>
                        <tr style="border-bottom: 1px solid var(--border-color);">
                            <th style="text-align: left; padding: 0.5rem; color: var(--text-secondary);">Time</th>
                            <th style="text-align: left; padding: 0.5rem; color: var(--text-secondary);">Tool</th>
                            <th style="text-align: left; padding: 0.5rem; color: var(--text-secondary);">Decision</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${logs.map(log => `
                            <tr style="border-bottom: 1px solid var(--border-color);">
                                <td style="padding: 0.5rem; color: var(--text-secondary);">
                                    ${this._formatRelativeTime(log.timestamp)}
                                </td>
                                <td style="padding: 0.5rem; font-family: var(--font-mono);">
                                    ${this._escapeHtml(log.feature)}.${this._escapeHtml(log.tool)}
                                </td>
                                <td style="padding: 0.5rem;">
                                    ${this._renderDecisionBadge(log.decision, log.user_choice)}
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        } catch (error) {
            console.error('Failed to load audit log:', error);
            container.innerHTML = `
                <p style="color: var(--error);">Failed to load audit log</p>
            `;
        }
    },

    _renderDecisionBadge(decision, userChoice) {
        const badges = {
            'auto_allowed': { icon: kicon('check-box'), color: 'var(--success)', text: 'Auto-allowed' },
            'auto_denied': { icon: kicon('x-box'), color: 'var(--error)', text: 'Auto-denied' },
            'user_approved': { icon: kicon('checkmark'), color: 'var(--success)', text: `Approved${userChoice ? ` (${userChoice})` : ''}` },
            'user_denied': { icon: kicon('x-mark'), color: 'var(--error)', text: 'Denied' },
            'timeout': { icon: kicon('hourglass'), color: 'var(--warning)', text: 'Timeout' }
        };

        const badge = badges[decision] || { icon: '?', color: 'var(--text-secondary)', text: decision };

        return `
            <span style="
                display: inline-flex;
                align-items: center;
                gap: 0.25rem;
                color: ${badge.color};
            ">
                ${badge.icon} ${badge.text}
            </span>
        `;
    },

    // === Pending Badge ===

    updatePendingBadge(count) {
        const badge = document.getElementById('security-pending-badge');
        if (badge) {
            badge.textContent = count;
            badge.style.display = count > 0 ? 'inline-flex' : 'none';
        }
    },

    // === Utility Functions ===

    _escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    _formatRelativeTime(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp);
        const now = new Date();
        const diff = Math.floor((now - date) / 1000);

        if (diff < 0) return 'just now';
        if (diff < 60) return `${diff}s ago`;
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
        return date.toLocaleDateString();
    },

    // === Session Control Methods ===

    async resetSession() {
        try {
            const confirmed = await new Promise((resolve) => {
                Modal.show({
                    title: 'Reset Session Permissions',
                    content: `
                        <p style="margin: 0; color: var(--text-secondary);">
                            This will clear all session-scoped permissions.
                            Tools that were approved for "This Session" will require approval again.
                        </p>
                    `,
                    buttons: [
                        { label: 'Cancel', type: 'secondary', onClick: () => { Modal.hide(); resolve(false); } },
                        { label: 'Reset', type: 'primary', onClick: () => { Modal.hide(); resolve(true); } }
                    ],
                    onClose: () => resolve(false)
                });
            });

            if (!confirmed) return;

            await API.request('/api/security/reset-session', { method: 'POST' });
            Toast.success('Session permissions cleared');
            await this.loadPermissionTree();
        } catch (error) {
            console.error('Failed to reset session:', error);
            Toast.error('Failed to reset session permissions');
        }
    },

    async cancelAllPending() {
        if (this.pendingApprovals.size === 0) {
            Toast.info('No pending approvals to cancel');
            return;
        }

        try {
            const confirmed = await new Promise((resolve) => {
                Modal.show({
                    title: 'Cancel All Pending',
                    content: `
                        <p style="margin: 0; color: var(--text-secondary);">
                            This will deny all ${this.pendingApprovals.size} pending approval request(s).
                            The waiting tool executions will be cancelled.
                        </p>
                    `,
                    buttons: [
                        { label: 'Keep Waiting', type: 'secondary', onClick: () => { Modal.hide(); resolve(false); } },
                        { label: 'Cancel All', type: 'danger', onClick: () => { Modal.hide(); resolve(true); } }
                    ],
                    onClose: () => resolve(false)
                });
            });

            if (!confirmed) return;

            const response = await API.request('/api/security/cancel-all', { method: 'POST' });
            Toast.success(`Cancelled ${response.cancelled} pending request(s)`);
            this.pendingApprovals.clear();
            this.updatePendingBadge(0);
            await this.loadPendingApprovals();
        } catch (error) {
            console.error('Failed to cancel pending:', error);
            Toast.error('Failed to cancel pending requests');
        }
    }
};

// Make Security available globally for onclick handlers
window.Security = Security;
