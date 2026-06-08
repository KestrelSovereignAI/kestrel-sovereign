/**
 * Kestrel Sovereign Console - UI Components
 * Modal, Toast, and common utilities
 */

// ============================================================================
// Animation Styles
// ============================================================================

const styleSheet = document.createElement('style');
styleSheet.textContent = `
    @keyframes modalFadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes modalSlideIn {
        from { opacity: 0; transform: scale(0.95) translateY(-10px); }
        to { opacity: 1; transform: scale(1) translateY(0); }
    }
    @keyframes toastSlideIn {
        from { opacity: 0; transform: translateX(100%); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes toastSlideOut {
        from { opacity: 1; transform: translateX(0); }
        to { opacity: 0; transform: translateX(100%); }
    }
    .modal-btn:hover {
        filter: brightness(1.1);
    }
`;
document.head.appendChild(styleSheet);

// ============================================================================
// Constants
// ============================================================================

export const PRIVACY_MODES = {
    ephemeral: { icon: kicon('lock'), label: 'EPHEMERAL', color: '#dc2626', description: 'Nothing stored, local LLM only' },
    isolated: { icon: kicon('lock-key'), label: 'ISOLATED', color: '#ea580c', description: 'Temporary storage, deleted on session end' },
    anonymous: { icon: kicon('mask'), label: 'ANONYMOUS', color: '#ca8a04', description: 'Stored without PII, encrypted backups' },
    normal: { icon: kicon('document'), label: 'NORMAL', color: '#16a34a', description: 'Standard persistence with all features' },
    public: { icon: kicon('globe'), label: 'PUBLIC', color: '#2563eb', description: 'Can be shared and exported publicly' },
};

// Commands are loaded dynamically from /api/commands
// This is a fallback list used until the API responds
export let AGENT_COMMANDS = [
    { cmd: '!help', description: 'Show available commands' },
    { cmd: '!model', description: 'Show current model' },
    { cmd: '!model-set', description: 'Set model', args: '<provider/model>' },
];

/**
 * Load commands dynamically from /api/commands endpoint.
 * Called after agent selection (multi_agent) or during standalone init.
 * Uses API module for proper auth and agent routing.
 */
export async function loadCommands(apiModule) {
    if (!apiModule) return;
    // #879: slash-commands hydrate the chat input autocomplete — when the
    // host has its own chat surface, no /api/commands fetch.
    if (typeof apiModule.hasCapability === 'function' && !apiModule.hasCapability('chat')) {
        return;
    }
    try {
        const data = await apiModule.request('/api/commands');
        if (data.commands && data.commands.length > 0) {
            AGENT_COMMANDS = data.commands;
            console.log(`Loaded ${data.count} commands from API`);
        }
    } catch (e) {
        // Non-critical — fallback commands are used
    }
}

// ============================================================================
// State
// ============================================================================

// Per-agent chat pane cache. Each agent owns a detached <div> that
// stays alive across agent switches so streams keep painting into
// their dispatch agent's pane whether visible or not. The visible
// pane is just whichever fragment is currently mounted into
// #chat-container. Keyed by host-agent name; null is the standalone-
// mode key. Pane shape:
//   {
//     element: HTMLDivElement,         // detached pane container
//     generation: number,               // bumped on within-agent context change
//     activeTurnId: number,              // #1573: monotonic per-turn token. Bumped when a
//                                        //   turn is dispatched; the streaming loop captures
//                                        //   it and only paints/recreates/tears down the
//                                        //   pane's stream bubble while it still owns the
//                                        //   turn. Stops a prior (e.g. interrupted) turn's
//                                        //   still-unwinding loop from welding its text into
//                                        //   the next turn's bubble.
//     streamingMsgDiv: HTMLDivElement|null,
//     fullContent: string,
//     thinkingItems: Array,             // UI-only thought bubbles for current stream
//     sessionId: string|null,
//     scrollPos: number,
//     hasUnrenderedMermaid: boolean,    // mermaid render deferred until mount
//     pendingRevise: boolean,            // Wave 5C: server fired a `revising` SSE event;
//                                        //   next chunk replaces (not appends) the bubble
//     reviseConsumedRequestId: string|null,
//                                        // Wave 5E idempotency: which request_id has had
//                                        //   its revise applied. Both signals (SSE + in-band
//                                        //   sentinel) check this so a delayed signal for an
//                                        //   already-consumed request is a no-op.
//     composerMode: 'interrupt'|'queue', // #1257: per-pane send-while-busy mode.
//                                        //   'interrupt' (default) = Phase 1 behavior
//                                        //   (stop the in-flight turn, dispatch now).
//                                        //   'queue' = stash the message and dispatch
//                                        //   it when the in-flight turn finishes.
//     queuedMessage: string|null,        // #1257: the single pending message in queue
//                                        //   mode. Re-Enter replaces it. Dispatched from
//                                        //   the completing turn's finally; cleared by
//                                        //   Stop and by a conversation switch.
//   }
const chatPanes = new Map();
let mountedChatAgent;  // undefined sentinel — null is a valid key

export const state = {
    currentPanel: 'chat',
    identity: null,
    constitution: null,
    memories: null,
    exports: null,
    storage: null,
    wallet: null,
    privacyMode: 'normal',
    // Per-agent waiting state (Set of host-agent names — null key for
    // standalone mode). Replaced the single `isWaiting` bool so a stream
    // on Agent A doesn't make Agent B's input look "Thinking" too.
    waitingAgents: new Set(),
    conversations: [],
    showDecrypted: true,
    encryptedAtRest: false,
    useStreaming: true,
    selectedModel: null,
    selectedProvider: null,

    chatPanes,

    get mountedChatAgent() { return mountedChatAgent; },
    set mountedChatAgent(v) { mountedChatAgent = v; },
};

// `state.currentSessionId` is now agent-scoped: it reads/writes the
// sessionId of whichever agent's pane is currently mounted. Treating
// it as a per-pane property is what lets each agent retain its
// session across switches without nuking it on selectAgent.
//
// Reads when no pane is mounted return null so callers that probe
// during early init (loadConversations auto-load, history sidebar
// rendering, footer status) keep their previous "no session yet"
// behavior. Writes when no pane is mounted are a no-op rather than
// creating a phantom pane keyed on `undefined`.
Object.defineProperty(state, 'currentSessionId', {
    get() {
        if (mountedChatAgent === undefined) return null;
        const pane = chatPanes.get(mountedChatAgent);
        return pane ? pane.sessionId : null;
    },
    set(value) {
        if (mountedChatAgent === undefined) return;
        const pane = chatPanes.get(mountedChatAgent);
        if (pane) pane.sessionId = value;
    },
    enumerable: true,
    configurable: false,
});

/**
 * Lazily create the per-agent chat pane element. Pane elements live
 * detached until mounted into #chat-container. Returned object is the
 * canonical per-agent chat state — message DOM, generation, session,
 * scroll, and the mermaid-deferral flag are all tracked here.
 */
export function getOrCreateChatPane(agentName) {
    let pane = chatPanes.get(agentName);
    if (pane) return pane;
    const element = document.createElement('div');
    element.className = 'chat-container-pane';
    if (agentName !== null && agentName !== undefined) {
        element.dataset.agent = String(agentName);
    }
    pane = {
        element,
        generation: 0,
        activeTurnId: 0,
        streamingMsgDiv: null,
        fullContent: '',
        thinkingItems: [],
        sessionId: null,
        scrollPos: 0,
        hasUnrenderedMermaid: false,
        pendingRevise: false,
        reviseConsumedRequestId: null,
        composerMode: 'interrupt',
        queuedMessage: null,
    };
    chatPanes.set(agentName, pane);
    return pane;
}

// ============================================================================
// Toast Notifications
// ============================================================================

export const Toast = {
    _container: null,

    _getContainer() {
        if (!this._container) {
            this._container = document.createElement('div');
            this._container.id = 'toast-container';
            this._container.style.cssText = `
                position: fixed;
                bottom: 1.5rem;
                right: 1.5rem;
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
                z-index: 3000;
                max-width: 400px;
            `;
            document.body.appendChild(this._container);
        }
        return this._container;
    },

    show(message, type = 'info', duration = 4000) {
        const container = this._getContainer();
        const toast = document.createElement('div');

        const colors = {
            success: { bg: '#059669', icon: kicon('checkmark') },
            error: { bg: '#dc2626', icon: kicon('x-mark') },
            warning: { bg: '#d97706', icon: kicon('warning') },
            info: { bg: '#2563eb', icon: kicon('info') }
        };
        const { bg, icon } = colors[type] || colors.info;

        toast.className = 'toast-item';
        toast.style.cssText = `
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.875rem 1.25rem;
            background: ${bg};
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
            font-size: 0.9rem;
            font-weight: 500;
            animation: toastSlideIn 0.3s ease-out;
            cursor: pointer;
        `;

        toast.innerHTML = `
            <span style="font-size: 1.1rem;">${icon}</span>
            <span style="flex: 1;">${message}</span>
            <button style="
                background: rgba(255,255,255,0.2);
                border: none;
                border-radius: 4px;
                color: white;
                padding: 0.25rem 0.5rem;
                cursor: pointer;
                font-size: 0.8rem;
            ">&times;</button>
        `;

        toast.addEventListener('click', () => this._removeToast(toast));
        container.appendChild(toast);

        if (duration > 0) {
            setTimeout(() => this._removeToast(toast), duration);
        }
    },

    _removeToast(toast) {
        if (!toast || !toast.parentNode) return;
        toast.style.animation = 'toastSlideOut 0.2s ease-in forwards';
        setTimeout(() => toast.remove(), 200);
    },

    success(message, duration) { this.show(message, 'success', duration); },
    error(message, duration) { this.show(message, 'error', duration); },
    warning(message, duration) { this.show(message, 'warning', duration); },
    info(message, duration) { this.show(message, 'info', duration); }
};

// ============================================================================
// Modal System
// ============================================================================

export const Modal = {
    _currentModal: null,
    _resolvePromise: null,

    show(options) {
        this.hide();
        const { title, content, buttons = [], onClose } = options;

        const overlay = document.createElement('div');
        overlay.id = 'modal-overlay';
        overlay.className = 'modal-overlay';
        overlay.style.cssText = `
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.6); z-index: 2000;
            display: flex; align-items: center; justify-content: center;
            padding: 1rem; backdrop-filter: blur(4px);
            animation: modalFadeIn 0.2s ease-out;
        `;

        const modal = document.createElement('div');
        modal.className = 'modal-container';
        modal.style.cssText = `
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 480px;
            width: 100%;
            max-height: 90vh;
            overflow: hidden;
            animation: modalSlideIn 0.2s ease-out;
        `;

        modal.innerHTML = `
            <div class="modal-header" style="
                padding: 1rem 1.5rem;
                border-bottom: 1px solid var(--border-color);
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <h3 style="margin: 0; font-size: 1.125rem; font-weight: 600;">${title}</h3>
                <button class="modal-close-btn" style="
                    background: none;
                    border: none;
                    font-size: 1.5rem;
                    cursor: pointer;
                    color: var(--text-secondary);
                    padding: 0;
                    line-height: 1;
                    transition: color 0.2s;
                ">&times;</button>
            </div>
            <div class="modal-body" style="padding: 1.5rem;">
                ${content}
            </div>
            ${buttons.length > 0 ? `
                <div class="modal-footer" style="
                    padding: 1rem 1.5rem;
                    border-top: 1px solid var(--border-color);
                    display: flex;
                    justify-content: flex-end;
                    gap: 0.75rem;
                ">
                    ${buttons.map((btn, i) => `
                        <button class="modal-btn modal-btn-${btn.type || 'secondary'}" data-btn-index="${i}" style="
                            padding: 0.625rem 1.25rem;
                            border: none;
                            border-radius: 8px;
                            font-size: 0.875rem;
                            font-weight: 500;
                            cursor: pointer;
                            transition: all 0.2s;
                            ${btn.type === 'primary' ? 'background: var(--accent-color); color: white;' : ''}
                            ${btn.type === 'danger' ? 'background: var(--error); color: white;' : ''}
                            ${btn.type === 'secondary' || !btn.type ? 'background: var(--bg-tertiary); color: var(--text-primary);' : ''}
                        ">${btn.label}</button>
                    `).join('')}
                </div>
            ` : ''}
        `;

        modal.querySelector('.modal-close-btn').addEventListener('click', () => {
            this.hide();
            if (onClose) onClose();
        });

        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                this.hide();
                if (onClose) onClose();
            }
        });

        modal.querySelectorAll('.modal-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const index = parseInt(btn.dataset.btnIndex);
                if (buttons[index] && buttons[index].onClick) {
                    buttons[index].onClick();
                }
            });
        });

        const escHandler = (e) => {
            if (e.key === 'Escape') {
                this.hide();
                if (onClose) onClose();
                document.removeEventListener('keydown', escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);

        overlay.appendChild(modal);
        document.body.appendChild(overlay);
        this._currentModal = overlay;

        const firstInput = modal.querySelector('input, select, textarea');
        if (firstInput) setTimeout(() => firstInput.focus(), 50);
    },

    hide() {
        if (this._currentModal) {
            this._currentModal.remove();
            this._currentModal = null;
        }
        if (this._resolvePromise) {
            this._resolvePromise(null);
            this._resolvePromise = null;
        }
    },

    confirm(title, message) {
        return new Promise((resolve) => {
            this.show({
                title,
                content: `<p style="margin: 0; color: var(--text-secondary); line-height: 1.6;">${message}</p>`,
                buttons: [
                    { label: 'Cancel', type: 'secondary', onClick: () => { this.hide(); resolve(false); } },
                    { label: 'Confirm', type: 'primary', onClick: () => { this.hide(); resolve(true); } }
                ],
                onClose: () => resolve(false)
            });
        });
    },

    prompt(title, placeholder = '', defaultValue = '') {
        return new Promise((resolve) => {
            const inputId = 'modal-prompt-input-' + Date.now();
            this.show({
                title,
                content: `
                    <input type="text" id="${inputId}"
                        placeholder="${placeholder}"
                        value="${defaultValue}"
                        style="
                            width: 100%;
                            padding: 0.75rem 1rem;
                            border: 1px solid var(--border-color);
                            border-radius: 8px;
                            font-size: 1rem;
                            background: var(--bg-primary);
                            color: var(--text-primary);
                            outline: none;
                            transition: border-color 0.2s;
                        "
                        onfocus="this.style.borderColor='var(--accent-color)'"
                        onblur="this.style.borderColor='var(--border-color)'"
                    />
                `,
                buttons: [
                    { label: 'Cancel', type: 'secondary', onClick: () => { this.hide(); resolve(null); } },
                    { label: 'OK', type: 'primary', onClick: () => {
                        const value = document.getElementById(inputId)?.value || '';
                        this.hide();
                        resolve(value);
                    }}
                ],
                onClose: () => resolve(null)
            });

            setTimeout(() => {
                const input = document.getElementById(inputId);
                if (input) {
                    input.addEventListener('keydown', (e) => {
                        if (e.key === 'Enter') {
                            this.hide();
                            resolve(input.value);
                        }
                    });
                }
            }, 50);
        });
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

export function formatBytes(bytes, decimals = 2) {
    if (bytes === 0 || bytes === null || bytes === undefined) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

export function truncate(str, maxLength = 30) {
    if (!str || str.length <= maxLength) return str;
    return str.slice(0, maxLength - 3) + '...';
}

export async function copyToClipboard(text, showNotification = true) {
    try {
        await navigator.clipboard.writeText(text);
        if (showNotification) {
            Toast.success('Copied to clipboard');
        }
        return true;
    } catch (e) {
        if (showNotification) {
            Toast.error('Failed to copy to clipboard');
        }
        return false;
    }
}

export function showLoading(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.innerHTML = '<div class="loading">Loading</div>';
}

export function showError(elementId, message) {
    const el = document.getElementById(elementId);
    if (el) el.innerHTML = `<div style="color: var(--error); padding: 1rem;">${message}</div>`;
}

export function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

export function truncateId(id) {
    if (!id || id.length <= 12) return id;
    return id.slice(0, 6) + '...' + id.slice(-4);
}

// Make utilities globally available for onclick handlers
window.copyToClipboard = copyToClipboard;
