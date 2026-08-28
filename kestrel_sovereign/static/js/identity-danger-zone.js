/**
 * Identity panel — Danger Zone (issue #2208)
 *
 * The Identity panel owns a visually separated "danger zone" section at the
 * bottom for destructive actions (today: delete agent; future: other
 * irreversible operations). Deletion is deliberately hard to reach: it lives
 * here, is danger-styled, and is gated behind a type-the-name confirm modal.
 *
 * Per the sharing principle (embedding hosts own as little UI as possible),
 * this section is HOST-MAPPABLE. A host injects the actual delete handler via
 * `KESTREL_UI_CONFIG.dangerZone.delete.handler` (Frinz maps it to "delete
 * companion"); the standalone console falls back to Kestrel-native deletion
 * (`DELETE /api/agents/{name}`) when the host is running in multi-agent mode.
 * If neither a host handler nor a native capability is available, the section
 * is hidden entirely.
 *
 * This module is intentionally dependency-light and pure enough to unit-test:
 * `resolveDeleteAction` decides visibility + which handler fires without
 * touching the DOM, and `renderIdentityDangerZone` does the rendering + the
 * type-the-name confirm wiring.
 */

/**
 * Read the danger-zone config from the embed config object (defaults to the
 * global `KESTREL_UI_CONFIG`). Always returns a plain object.
 */
export function readDangerZoneConfig(config) {
    const cfg = config
        || (typeof globalThis !== 'undefined' && globalThis.KESTREL_UI_CONFIG)
        || {};
    const dz = cfg && typeof cfg.dangerZone === 'object' && cfg.dangerZone
        ? cfg.dangerZone
        : {};
    return dz;
}

/**
 * Decide whether the delete action should appear and, if so, which handler
 * runs when the type-the-name gate is passed.
 *
 * Resolution order:
 *   1. `delete.enabled === false`            → hidden (explicit host opt-out).
 *   2. `delete.handler` is a function        → host-mapped delete.
 *   3. caller-scoped native delete authority + a
 *      resolvable agent name                 → Kestrel-native delete.
 *   4. otherwise                             → hidden.
 *
 * @param {object} opts
 * @param {object} opts.identity   - identity payload (needs `name`).
 * @param {object} opts.api        - API client (needs `hasCapability`, and
 *                                    `deleteAgent` for the native path).
 * @param {object} [opts.dz]       - resolved dangerZone config.
 * @returns {{show: boolean, label: string, description: string,
 *            confirmName: string, handler: (function|null),
 *            native: boolean} }
 */
export function resolveDeleteAction({ identity, api, dz }) {
    const del = dz && typeof dz.delete === 'object' && dz.delete ? dz.delete : {};
    const name = (identity && identity.name) || '';

    const hidden = {
        show: false,
        label: '',
        description: '',
        confirmName: name,
        handler: null,
        native: false,
    };

    if (del.enabled === false) return hidden;

    const hostHandler = typeof del.handler === 'function' ? del.handler : null;
    // Native deletion targets the multi-agent manager's `DELETE /api/agents/{name}`
    // endpoint. Its GET /api/agents payload advertises this CALLER's authority;
    // feature presence alone is not permission. Fail closed until that payload
    // has classified the current authenticated principal.
    //
    // The manager is keyed by the ROUTING key (the selected agent name from
    // `/api/agents`, tracked by `api.getHostAgent()`), NOT the identity panel's
    // editable display name — a renamed agent would 404 the wrong target. Fall
    // back to the display name only when no routing key is tracked.
    const agentKey = (
        api && typeof api.getHostAgent === 'function' && api.getHostAgent()
    ) || name;
    const canNative = !!(
        api
        && typeof api.canManageHostAgentLifecycle === 'function'
        && api.canManageHostAgentLifecycle('delete')
        && agentKey
        && typeof api.deleteAgent === 'function'
    );

    if (!hostHandler && !canNative) return hidden;

    const native = !hostHandler && canNative;
    const handler = hostHandler
        ? () => hostHandler(identity)
        : () => {
            // Authority is caller-scoped and can be revoked while the panel or
            // confirm modal remains mounted. Re-check at the irreversible
            // boundary rather than trusting the render-time snapshot.
            if (!api.canManageHostAgentLifecycle('delete')) {
                throw new Error('Delete authority is no longer available. Refresh and try again.');
            }
            return api.deleteAgent(agentKey);
        };

    // The name the user must type to arm the button. Hosts may override it (e.g.
    // a companion display name that differs from the Kestrel agent name).
    const confirmName = (typeof del.confirmName === 'string' && del.confirmName)
        || name;

    return {
        show: true,
        label: (typeof del.label === 'string' && del.label) || 'Delete agent',
        description: (typeof del.description === 'string' && del.description)
            || 'Permanently remove this agent. This action cannot be undone.',
        confirmName,
        handler,
        native,
        onDeleted: typeof del.onDeleted === 'function' ? del.onDeleted : null,
    };
}

function esc(str) {
    return String(str == null ? '' : str).replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}

/**
 * Render the danger-zone section into `container` and wire the type-the-name
 * confirm modal. If the delete action resolves to hidden, the container is
 * cleared (and left empty) so a re-render can toggle it off cleanly.
 *
 * @param {object} opts
 * @param {HTMLElement} opts.container - element to render into.
 * @param {object} opts.identity       - identity payload.
 * @param {object} opts.api            - API client.
 * @param {object} opts.Modal          - Modal helper (`show` + ownership handle).
 * @param {object} [opts.Toast]        - Toast helper (`success`/`error`).
 * @param {object} [opts.config]       - embed config (defaults to global).
 * @returns {boolean} whether the section was rendered.
 */
export function renderIdentityDangerZone({ container, identity, api, Modal, Toast, config } = {}) {
    if (!container) return false;
    const dz = readDangerZoneConfig(config);
    const action = resolveDeleteAction({ identity, api, dz });

    if (!action.show) {
        container.innerHTML = '';
        return false;
    }

    container.innerHTML = `
        <div class="identity-danger-zone" data-testid="identity-danger-zone">
            <div class="identity-danger-zone-header">
                <span class="ki ki-warning" aria-hidden="true"></span>
                <h3 data-label-key="danger_zone_title">Danger Zone</h3>
            </div>
            <div class="identity-danger-zone-row">
                <div class="identity-danger-zone-copy">
                    <div class="identity-danger-zone-action-label">${esc(action.label)}</div>
                    <div class="identity-danger-zone-desc">${esc(action.description)}</div>
                </div>
                <button type="button" class="btn identity-danger-zone-btn" id="danger-zone-delete-btn">
                    ${esc(action.label)}
                </button>
            </div>
        </div>
    `;

    const btn = container.querySelector('#danger-zone-delete-btn');
    if (btn) {
        btn.addEventListener('click', () => {
            // Re-resolve after any intervening authentication or discovery
            // refresh. A stale button must not open an actionable modal.
            const currentAction = resolveDeleteAction({ identity, api, dz });
            if (!currentAction.show) {
                container.innerHTML = '';
                return;
            }
            _openConfirmModal({ action: currentAction, identity, Modal, Toast });
        });
    }
    return true;
}

/**
 * Type-the-name confirm modal. The Delete button stays disabled until the
 * typed value exactly matches the required name.
 */
function _openConfirmModal({ action, identity, Modal, Toast }) {
    if (!Modal || typeof Modal.show !== 'function') return;
    const required = action.confirmName || '';
    const inputId = 'danger-zone-confirm-input';
    const displayName = (identity && identity.name) || required || 'this agent';

    let confirmModal;
    confirmModal = Modal.show({
        title: action.label,
        content: `
            <p style="margin: 0 0 1rem 0; color: var(--text-secondary); line-height: 1.6;">
                This will permanently delete <strong>${esc(displayName)}</strong> and cannot be undone.
            </p>
            <p style="margin: 0 0 0.5rem 0; color: var(--text-secondary); font-size: 0.875rem;">
                Type <strong>${esc(required)}</strong> to confirm:
            </p>
            <input type="text" id="${inputId}" autocomplete="off" spellcheck="false"
                placeholder="${esc(required)}"
                style="
                    width: 100%;
                    padding: 0.75rem 1rem;
                    border: 1px solid var(--border-color);
                    border-radius: 8px;
                    font-size: 1rem;
                    background: var(--bg-primary);
                    color: var(--text-primary);
                    outline: none;
                " />
        `,
        buttons: [
            { label: 'Cancel', type: 'secondary', onClick: () => confirmModal.close() },
            {
                label: action.label,
                type: 'danger',
                onClick: () => {
                    const input = confirmModal.querySelector(`#${inputId}`);
                    const value = input ? input.value : '';
                    if (value !== required) {
                        // Guard: the button can be clicked before it is armed;
                        // never fire the handler on a mismatch.
                        if (input) input.style.borderColor = 'var(--error)';
                        return;
                    }
                    try {
                        confirmModal.close();
                    } finally {
                        _runDelete({ action, displayName, Toast });
                    }
                },
            },
        ],
    });

    // Arm/disarm the danger button live as the user types (mirrors the
    // prompt() wiring in ui.js — the modal is already in the DOM by now).
    setTimeout(() => {
        if (!confirmModal.isCurrent()) return;
        const input = confirmModal.querySelector(`#${inputId}`);
        const dangerBtn = confirmModal.querySelector('.modal-btn-danger');
        if (!input || !dangerBtn) return;
        const sync = () => {
            const armed = input.value === required;
            dangerBtn.disabled = !armed;
            dangerBtn.style.opacity = armed ? '1' : '0.5';
            dangerBtn.style.cursor = armed ? 'pointer' : 'not-allowed';
            input.style.borderColor = 'var(--border-color)';
        };
        sync();
        input.addEventListener('input', sync);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.isComposing && input.value === required) {
                e.preventDefault();
                e.stopPropagation();
                try {
                    confirmModal.close();
                } finally {
                    _runDelete({ action, displayName, Toast });
                }
            }
        });
        input.focus();
    }, 50);
}

async function _runDelete({ action, displayName, Toast }) {
    try {
        await action.handler();
        if (Toast && typeof Toast.success === 'function') {
            Toast.success(`Deleted ${displayName}`);
        }
        if (action.onDeleted) {
            try { action.onDeleted(); } catch (_) { /* host callback best-effort */ }
        } else if (action.native && typeof window !== 'undefined' && window.location) {
            // Standalone native delete: the current agent no longer exists — send
            // the user back to the host root so they don't sit on a dead panel.
            setTimeout(() => { window.location.href = '/'; }, 600);
        }
    } catch (e) {
        if (Toast && typeof Toast.error === 'function') {
            Toast.error(`Delete failed: ${e.message || e}`);
        } else {
            console.error('[danger-zone] delete failed:', e);
        }
    }
}

export default renderIdentityDangerZone;
