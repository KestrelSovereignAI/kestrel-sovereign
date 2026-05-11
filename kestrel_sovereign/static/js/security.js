/**
 * Kestrel Security Module - Permission management and approval queue UI
 */

import { Modal, Toast } from './ui.js';
import API from './api.js';
import { subscribeSSE } from './chat.js';

export const Security = {
    pendingApprovals: new Map(),
    permissionTree: [],
    globalAutoMode: false,
    _initialized: false,
    // Modal is a singleton — if two approval_request events arrive in quick
    // succession, showing both concurrently would stack overlays in the DOM
    // and leave the older one stuck. Serialize them through a queue. #748.
    _approvalQueue: [],
    _approvalDraining: false,
    _seenApprovalIds: new Set(),
    // Ids the server has already evicted (timeout/cancelled). When a
    // withdrawal SSE arrives for an id that's currently in flight we mark
    // it here so the modal-resolution path knows to skip the POST. #877.
    _withdrawnApprovalIds: new Set(),
    // Resolver for the active modal — set in showApprovalModal. When a
    // withdrawal SSE lands for the modal currently on screen, we close it
    // immediately rather than wait for the user. #877.
    _activeApprovalId: null,
    _activeApprovalResolver: null,

    // === Initialization ===

    async init() {
        if (this._initialized) return;

        // #879: short-circuit when the host opted out of *both* security
        // sub-surfaces.  Approval-request modals are still useful even when
        // the audit/permissions panels are hidden (an embedded host might
        // want approval prompts for tool calls without surfacing a full
        // security panel), so the SSE handler is wired regardless.
        const auditOn = API.hasCapability('audit');
        const permsOn = API.hasCapability('permissions');

        // Set up SSE handler for approval requests
        this._setupSSEHandler();

        // Load initial data when Security panel is opened.  Skip the click
        // wiring entirely when both sub-caps are off — the panel itself was
        // removed by initNavigation() so the tab doesn't exist.
        if (auditOn || permsOn) {
            const securityTab = document.querySelector('[data-panel="security"]');
            if (securityTab) {
                securityTab.addEventListener('click', () => this.loadSecurityPanel());
            }
        }

        if (permsOn) {
            this.loadAutoMode();
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
        // Withdrawal events: the server is telling us the queue entry is
        // gone (timeout or task cancellation). Close any modal showing this
        // id; clear local queue state. Without this the user clicks
        // ""Approve"" on a stale modal and sees a 404. #877.
        subscribeSSE('approval_withdrawn', (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleApprovalWithdrawn(data);
            } catch (err) {
                console.error('Failed to parse approval_withdrawn event:', err);
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
        // If the server already withdrew this id (rare race: withdrawal SSE
        // beat the request SSE on a slow network), don't prompt at all.
        if (this._withdrawnApprovalIds.has(data.id)) {
            return;
        }
        this._seenApprovalIds.add(data.id);

        this.pendingApprovals.set(data.id, data);
        this.updatePendingBadge(this.pendingApprovals.size);

        this._approvalQueue.push(data);
        // Fire-and-forget drain; errors are handled inside the drain loop.
        this._drainApprovalQueue();
    },

    handleApprovalWithdrawn(data) {
        console.log('Approval withdrawn:', data.id, 'reason:', data.reason);
        this._withdrawnApprovalIds.add(data.id);

        // If this id is the modal currently on screen, close it now and
        // skip the POST that would otherwise 404.
        if (this._activeApprovalId === data.id && this._activeApprovalResolver) {
            const resolver = this._activeApprovalResolver;
            this._activeApprovalResolver = null;
            this._activeApprovalId = null;
            try {
                Modal.hide();
            } catch (err) {
                // Modal may already be closing — non-fatal.
            }
            resolver({ approved: false, scope: 'once', _withdrawn: true });
        }

        // Drop any queued-but-not-yet-shown entry.
        this._approvalQueue = this._approvalQueue.filter(e => e.id !== data.id);
        if (this.pendingApprovals.delete(data.id)) {
            this.updatePendingBadge(this.pendingApprovals.size);
        }
    },

    async _drainApprovalQueue() {
        if (this._approvalDraining) return;
        this._approvalDraining = true;
        try {
            while (this._approvalQueue.length > 0) {
                const data = this._approvalQueue.shift();
                // If a withdrawal SSE landed before we got around to showing
                // this entry's modal, don't show it at all.
                if (this._withdrawnApprovalIds.has(data.id)) {
                    this.pendingApprovals.delete(data.id);
                    this.updatePendingBadge(this.pendingApprovals.size);
                    continue;
                }
                let decision;
                try {
                    decision = await this.showApprovalModal(data);
                } catch (err) {
                    console.error('Approval modal error:', err);
                    decision = { approved: false, scope: 'once' };
                }

                // Withdrawal short-circuits the POST: the server already
                // evicted the queue entry, so submitting would 404. The
                // resolver flagged this with ``_withdrawn: true``. #877.
                if (!decision._withdrawn) {
                    try {
                        await this.submitApproval(data.id, decision.approved, decision.scope, {
                            suppressToast: decision.suppressToast
                        });
                    } catch (err) {
                        console.error('Failed to submit approval:', err);
                    }
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
            // Track the active modal so a withdrawal SSE can close it
            // proactively. The wrapper resolver clears the tracking before
            // forwarding the decision so a late withdrawal can't double-close.
            const wrappedResolve = (decision) => {
                if (this._activeApprovalId === data.id) {
                    this._activeApprovalId = null;
                    this._activeApprovalResolver = null;
                }
                resolve(decision);
            };
            this._activeApprovalId = data.id;
            this._activeApprovalResolver = wrappedResolve;

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
                        <p style="margin: 0.5rem 0 0; color: var(--warning); font-size: 0.8125rem;">
                            Auto Mode approves this request and future non-denied requests while this session is active.
                            Constitutional, honesty, and security hooks can still flag or block.
                        </p>
                    </div>
                `,
                buttons: [
                    {
                        label: `${kicon('x-mark')} Deny`,
                        type: 'danger',
                        onClick: () => {
                            Modal.hide();
                            wrappedResolve({ approved: false, scope: 'once' });
                        }
                    },
                    {
                        label: 'This Time',
                        type: 'secondary',
                        onClick: () => {
                            Modal.hide();
                            wrappedResolve({ approved: true, scope: 'once' });
                        }
                    },
                    {
                        label: 'This Session',
                        type: 'primary',
                        onClick: () => {
                            Modal.hide();
                            wrappedResolve({ approved: true, scope: 'session' });
                        }
                    },
                    {
                        label: `${kicon('checkmark')} Always`,
                        type: 'primary',
                        onClick: () => {
                            Modal.hide();
                            wrappedResolve({ approved: true, scope: 'always' });
                        }
                    },
                    {
                        label: `${kicon('shield')} Enable Auto Mode`,
                        type: 'primary',
                        onClick: async () => {
                            try {
                                const response = await this.setGlobalAutoMode(true);
                                Modal.hide();
                                Toast.warning(response.warning);
                                wrappedResolve({
                                    approved: true,
                                    scope: 'once',
                                    suppressToast: true
                                });
                            } catch (error) {
                                console.error('Failed to enable Auto mode from approval modal:', error);
                                Toast.error('Failed to enable Auto mode');
                            }
                        }
                    }
                ],
                onClose: () => wrappedResolve({ approved: false, scope: 'once' })
            });
        });
    },

    async submitApproval(approvalId, approved, scope, options = {}) {
        try {
            const response = await API.request('/api/security/approve', {
                method: 'POST',
                body: JSON.stringify({
                    approval_id: approvalId,
                    approved,
                    scope
                })
            });

            if (response.success && !options.suppressToast) {
                Toast.success(approved
                    ? `Approved (${scope})`
                    : 'Denied'
                );
            } else {
                Toast.error('Failed to submit decision');
            }

            return response;
        } catch (error) {
            // 404 here means the server already evicted the queue entry —
            // either the calling task was cancelled (#877) or the request
            // timed out (5-minute default). Either way the user's decision
            // has nowhere to land. The withdrawal SSE handler is the
            // primary path for closing the modal proactively; this branch
            // only fires when the SSE didn't beat the user's click.
            const msg = (error && error.message) || '';
            if (msg.includes('not found') || msg.includes('expired')) {
                Toast.warning('Approval was withdrawn by the agent before you could decide.');
                console.warn('Approval withdrawn by agent before decision submitted:', approvalId);
            } else {
                console.error('Failed to submit approval:', error);
                Toast.error('Failed to submit decision');
            }
            return { success: false };
        }
    },

    // === Permission Tree UI ===

    async loadSecurityPanel() {
        // #879: only fan out fetches for the sub-sections the host enabled.
        // Pending approvals are tied to permissions tree (the queue is the
        // gating UI for permission grants), so they ride along with it.
        const tasks = [];
        if (API.hasCapability('permissions')) {
            tasks.push(this.loadAutoMode());
            tasks.push(this.loadPermissionTree());
            tasks.push(this.loadPendingApprovals());
        }
        if (API.hasCapability('audit')) {
            tasks.push(this.loadAuditLog());
        }
        await Promise.all(tasks);
    },

    async loadPermissionTree() {
        // #879: deep-link defense — no /api/security fetch when disabled.
        if (!API.hasCapability('permissions')) return;
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
                                <option value="auto" ${tool.level === 'auto' ? 'selected' : ''}>${kicon('shield')} Auto</option>
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
            'auto_all': kicon('shield'),
            'deny_all': kicon('x-box'),
            'ask_all': kicon('empty-box'),
            'session_all': kicon('half-circle'),
            'mixed': kicon('half-circle')
        };
        const labels = {
            'allow_all': 'Allow All',
            'auto_all': 'Auto All',
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
            const response = await API.request('/api/security/permissions', {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    feature: featureName,
                    tool: toolName,
                    level
                })
            });

            if (response.warning) {
                Toast.warning(response.warning);
            } else {
                Toast.success(`Set ${toolName} to ${level}`);
            }
            await this.loadPermissionTree(); // Refresh to update rollup
        } catch (error) {
            console.error('Failed to set permission:', error);
            Toast.error('Failed to set permission');
        }
    },

    async cycleFeaturePermission(featureName, currentState) {
        // Cycle: mixed -> ask -> auto -> allow -> deny -> ask
        const nextLevel = {
            'mixed': 'ask',
            'ask_all': 'auto',
            'auto_all': 'allow',
            'allow_all': 'deny',
            'deny_all': 'ask',
            'session_all': 'ask'
        }[currentState] || 'ask';

        try {
            // #766: bulk DENY runs through the demo-isolation rail.
            const headers = nextLevel === 'deny'
                ? { 'X-Kestrel-Allow-Destructive': 'user-initiated-ui-deny' }
                : {};
            const response = await API.request('/api/security/permissions/feature', {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    feature: featureName,
                    level: nextLevel
                })
            });

            if (response.warning) {
                Toast.warning(response.warning);
            } else {
                Toast.success(`Set ${featureName} to ${nextLevel}`);
            }
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
        // #879: deep-link defense — no /api/security/audit fetch when disabled.
        if (!API.hasCapability('audit')) return;
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
            'auto_mode_allowed': { icon: kicon('shield'), color: 'var(--warning)', text: 'Auto mode' },
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

    async loadAutoMode() {
        if (!API.hasCapability('permissions')) return;
        try {
            const response = await API.request('/api/security/auto-mode');
            this.globalAutoMode = Boolean(response.enabled);
            this.renderAutoModeButton();
        } catch (error) {
            console.error('Failed to load Auto mode:', error);
        }
    },

    async setGlobalAutoMode(enabled) {
        const response = await API.request('/api/security/auto-mode', {
            method: 'POST',
            body: JSON.stringify({ enabled })
        });
        this.globalAutoMode = Boolean(response.enabled);
        this.renderAutoModeButton();
        return response;
    },

    renderAutoModeButton() {
        document.querySelectorAll('.security-auto-mode-toggle').forEach((button) => {
            button.classList.toggle('btn-primary', this.globalAutoMode);
            button.classList.toggle('btn-secondary', !this.globalAutoMode);
            button.innerHTML = this.globalAutoMode
                ? `${kicon('shield')} Auto Mode: On`
                : `${kicon('shield')} Auto Mode: Off`;
            button.title = this.globalAutoMode
                ? 'Global Auto is on for this session. Explicit Deny still blocks.'
                : 'Turn on session-scoped global Auto for non-denied tools.';
        });
    },

    async toggleGlobalAutoMode(options = {}) {
        try {
            const nextEnabled = !this.globalAutoMode;
            const confirmEnable = options.confirmEnable !== false;
            if (nextEnabled && confirmEnable) {
                const confirmed = await new Promise((resolve) => {
                    Modal.show({
                        title: 'Enable Global Auto Mode',
                        content: `
                            <p style="margin: 0 0 0.75rem 0; color: var(--text-secondary);">
                                Auto Mode skips approval popups for every non-denied tool while this session is active.
                            </p>
                            <p style="margin: 0; color: var(--warning); font-size: 0.875rem;">
                                Constitutional, honesty, and security hooks still get the first chance to flag or block.
                                This is not a guarantee that every risk has been detected.
                            </p>
                        `,
                        buttons: [
                            { label: 'Cancel', type: 'secondary', onClick: () => { Modal.hide(); resolve(false); } },
                            { label: `${kicon('shield')} Enable Auto`, type: 'primary', onClick: () => { Modal.hide(); resolve(true); } }
                        ],
                        onClose: () => resolve(false)
                    });
                });
                if (!confirmed) return;
            }

            const response = await this.setGlobalAutoMode(nextEnabled);
            Toast[this.globalAutoMode ? 'warning' : 'success'](
                this.globalAutoMode ? response.warning : 'Global Auto mode disabled'
            );
            await this.loadPermissionTree();
        } catch (error) {
            console.error('Failed to toggle Auto mode:', error);
            Toast.error('Failed to toggle Auto mode');
        }
    },

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
            this.globalAutoMode = false;
            this.renderAutoModeButton();
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
