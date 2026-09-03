/**
 * Kestrel Sovereign Console - UI Components
 * Modal, Toast, and common utilities
 */

// ============================================================================
// Overlay root (#2233)
// ============================================================================
// Body-level UI (modals, toasts — and anything else that floats) mounts into
// a configurable root instead of document.body. Embeds that serve kestrel's
// CSS scoped to their mount roots (e.g. Frinz's @scope-wrapped
// chat-scoped.css) point this at an element INSIDE a scope root so overlay
// content is styled; standalone keeps the document.body default. Fixed
// positioning is unaffected by the parent, so overlays still cover the
// viewport wherever the root lives.
let _overlayRoot = null;
export function setOverlayRoot(el) { _overlayRoot = el || null; }
export function getOverlayRoot() {
    return (_overlayRoot && _overlayRoot.isConnected) ? _overlayRoot : document.body;
}

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
    anonymous: { icon: kicon('mask'), label: 'ANONYMOUS', color: '#ca8a04', description: 'Local only, stored with PII redacted' },
    normal: { icon: kicon('document'), label: 'NORMAL', color: '#16a34a', description: 'Standard persistence with all features' },
    public: { icon: kicon('globe'), label: 'PUBLIC', color: '#2563eb', description: 'Can be shared and exported publicly' },
    deidentified: { icon: kicon('beaker'), label: 'DEIDENTIFIED', color: '#7c3aed', description: 'Research sharing with Safe Harbor evidence' },
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
export async function loadCommands(apiModule, expectedAgent = apiModule?.getHostAgent?.()) {
    if (!apiModule) return;
    // #879: slash-commands hydrate the chat input autocomplete — when the
    // host has its own chat surface, no /api/commands fetch.
    if (typeof apiModule.hasCapability === 'function' && !apiModule.hasCapability('chat')) {
        return;
    }
    try {
        const data = await apiModule.request('/api/commands');
        if (apiModule.getHostAgent?.() !== expectedAgent) return;
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
//     followLive: boolean,              // #2909: stick-to-bottom engaged for this
//                                        //   conversation. Saved on detach and
//                                        //   restored on mount alongside scrollPos,
//                                        //   so returning to a scrolled-up pane does
//                                        //   not silently re-engage tail-following.
//     unseenTail: boolean,              // #2909: content appended below the reader
//                                        //   while following was disengaged, and they
//                                        //   have not scrolled back to it — i.e. this
//                                        //   conversation is showing "Jump to latest".
//                                        //   Travels with followLive across a switch.
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
    // A local stream abort is not proof that the backend turn stopped. Keep a
    // separate fail-closed latch until /stop confirms STOPPED or
    // ALREADY_COMPLETE; the aborted stream's finally may clear waitingAgents.
    unconfirmedStopAgents: new Set(),
    // Preserve the exact turn boundary across failed Stop retries. The stream
    // client's finally clears its live request-id projection after aborting,
    // but a retry must never silently widen into an agent-wide Stop.
    unconfirmedStopRequestIds: new Map(),
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
        followLive: true,
        unseenTail: false,
        draftText: '',
        micArmed: false,
        hasUnrenderedMermaid: false,
        pendingRevise: false,
        reviseConsumedRequestId: null,
        composerMode: 'interrupt',
        queuedMessage: null,
        // The user asked for a new conversation and its session is still
        // being minted. Such a pane is CLAIMED even though it is empty and
        // carries no session id yet — #714's auto-load must not read that
        // emptiness as "cold" and fill it with history. See chat.js
        // `setPaneAwaitingNewSession`.
        awaitingNewSession: false,
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
            getOverlayRoot().appendChild(this._container);
        }
        // Re-home a container created before setOverlayRoot ran (or whose root
        // was torn down) so late configuration still takes effect.
        if (this._container.parentNode !== getOverlayRoot()) {
            getOverlayRoot().appendChild(this._container);
        }
        return this._container;
    },

    _show(message, type = 'info', duration = 4000, { trustedHtml = false } = {}) {
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

        const iconEl = document.createElement('span');
        iconEl.style.fontSize = '1.1rem';
        // Icons come from the console's fixed-name icon registry, never from a
        // response or user-supplied string.
        iconEl.innerHTML = icon;

        const messageEl = document.createElement('span');
        messageEl.style.flex = '1';
        if (trustedHtml) {
            messageEl.innerHTML = String(message ?? '');
        } else {
            messageEl.textContent = String(message ?? '');
        }

        const closeButton = document.createElement('button');
        closeButton.style.cssText = `
            background: rgba(255,255,255,0.2);
            border: none;
            border-radius: 4px;
            color: white;
            padding: 0.25rem 0.5rem;
            cursor: pointer;
            font-size: 0.8rem;
        `;
        closeButton.type = 'button';
        closeButton.ariaLabel = 'Dismiss notification';
        closeButton.textContent = '\u00d7';
        toast.appendChild(iconEl);
        toast.appendChild(messageEl);
        toast.appendChild(closeButton);

        toast.addEventListener('click', () => this._removeToast(toast));
        container.appendChild(toast);

        if (duration > 0) {
            setTimeout(() => this._removeToast(toast), duration);
        }
    },

    show(message, type = 'info', duration = 4000) {
        return this._show(message, type, duration);
    },

    // Explicit escape hatch for small, pre-sanitized console-owned fragments
    // such as the upgrade CTA. API errors and ordinary caller strings must use
    // show()/error()/warning(), whose message path is always textContent.
    showTrustedHtml(trustedHtml, type = 'info', duration = 4000) {
        return this._show(trustedHtml, type, duration, { trustedHtml: true });
    },

    _removeToast(toast) {
        if (!toast || !toast.parentNode) return;
        toast.style.animation = 'toastSlideOut 0.2s ease-in forwards';
        setTimeout(() => toast.remove(), 200);
    },

    success(message, duration) { this.show(message, 'success', duration); },
    error(message, duration) { this.show(message, 'error', duration); },
    warning(message, duration) { this.show(message, 'warning', duration); },
    info(message, duration) { this.show(message, 'info', duration); },
    // Red toast for high-consequence persistent state (e.g. Always Auto). Same
    // color as 'error' but semantically a deliberate danger acknowledgement,
    // not a failure.
    danger(message, duration) { this.show(message, 'error', duration); }
};

// ============================================================================
// Modal System
// ============================================================================

const MODAL_FOCUSABLE_SELECTOR = [
    'a[href]',
    'area[href]',
    'button:not([disabled])',
    'input:not([disabled]):not([type="hidden"])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    'details > summary:first-of-type',
    'iframe',
    '[contenteditable]:not([contenteditable="false"])',
    '[tabindex]:not([tabindex="-1"])',
].join(', ');

let modalId = 0;

function modalElementIsHidden(element, modal) {
    for (let current = element; current && current !== modal; current = current.parentElement) {
        if (current.hidden
            || current.getAttribute('aria-hidden') === 'true'
            || current.hasAttribute('inert')) {
            return true;
        }

        const style = window.getComputedStyle(current);
        if (style.display === 'none'
            || style.visibility === 'hidden'
            || style.visibility === 'collapse') {
            return true;
        }

        if (current.tagName === 'DETAILS' && !current.open) {
            const summary = Array.from(current.children)
                .find((child) => child.tagName === 'SUMMARY');
            if (!summary?.contains(element)) return true;
        }
    }
    return false;
}

function modalFocusableElements(modal) {
    // Derive DOM order independently from the selector-list result. Browsers
    // return selector-list matches in tree order, but jsdom has regressed this
    // for mixed combinator/simple selector lists; the trap must not depend on
    // selector-engine grouping to identify its first and last controls.
    const domOrder = new Map(
        Array.from(modal.querySelectorAll('*')).map((element, index) => [element, index]),
    );
    const candidates = Array.from(modal.querySelectorAll(MODAL_FOCUSABLE_SELECTOR))
        .map((element) => ({ element, domIndex: domOrder.get(element) }))
        .filter(({ element }) => {
            return element.tabIndex >= 0
                && !element.matches(':disabled')
                && !modalElementIsHidden(element, modal);
        });

    // A named radio group contributes one sequential tab stop: its checked
    // control, or the first available control when none is checked. Counting
    // every radio can make the focus trap believe a non-tabbable radio is its
    // final boundary and allow Tab to escape into the page behind the dialog.
    const tabbable = candidates.filter(({ element }) => {
        if (!(element instanceof window.HTMLInputElement)
            || element.type !== 'radio'
            || !element.name) {
            return true;
        }
        const group = candidates
            .map((candidate) => candidate.element)
            .filter((candidate) => candidate instanceof window.HTMLInputElement
                && candidate.type === 'radio'
                && candidate.name === element.name
                && candidate.form === element.form
                && candidate.getRootNode() === element.getRootNode());
        return element === (group.find((candidate) => candidate.checked) || group[0]);
    });

    return tabbable
        .sort((left, right) => {
            const leftTabIndex = left.element.tabIndex;
            const rightTabIndex = right.element.tabIndex;
            if (leftTabIndex > 0 && rightTabIndex > 0) {
                return leftTabIndex - rightTabIndex || left.domIndex - right.domIndex;
            }
            if (leftTabIndex > 0) return -1;
            if (rightTabIndex > 0) return 1;
            return left.domIndex - right.domIndex;
        })
        .map(({ element }) => element);
}

function deepActiveElement(root = document) {
    let active = root?.activeElement || null;
    while (active?.shadowRoot?.activeElement) {
        active = active.shadowRoot.activeElement;
    }
    return active;
}

function modalEventIsInside(event, modal) {
    const eventPath = typeof event.composedPath === 'function' ? event.composedPath() : [];
    return eventPath.includes(modal) || modal.contains(event.target);
}

function eventOriginatesFromRoot(event, root) {
    if (!root || root === document) return false;
    const eventPath = typeof event.composedPath === 'function' ? event.composedPath() : [];
    if (eventPath.includes(root)) return true;

    // A closed ShadowRoot is deliberately omitted from composedPath() outside
    // that tree. Its host remains visible, and activeElement on the retained
    // root tells us whether the host is a retargeted inner event or the actual
    // external target. The root's own capture guard will handle the former.
    return Boolean(root.host
        && eventPath.includes(root.host)
        && root.activeElement);
}

function addModalListener(lifecycle, target, type, listener, options) {
    target.addEventListener(type, listener, options);
    lifecycle.removeListeners.push(() => target.removeEventListener(type, listener, options));
}

function modalAttribute(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function reportModalActionError(error) {
    if (typeof window.reportError === 'function') {
        window.reportError(error);
        return;
    }
    window.setTimeout(() => {
        throw error;
    }, 0);
}

function invokeModalCloseCallback(callback) {
    const result = callback();
    if (result && typeof result.then === 'function') {
        Promise.resolve(result).catch(reportModalActionError);
    }
}

function createModalHandle(controller, lifecycle) {
    return Object.freeze({
        isCurrent() {
            return controller._lifecycle === lifecycle && !lifecycle.closed;
        },
        querySelector(selector) {
            if (controller._lifecycle !== lifecycle || lifecycle.closed) return null;
            return lifecycle.modal.querySelector(selector);
        },
        close() {
            if (controller._lifecycle !== lifecycle || lifecycle.closed) return false;
            controller._close(lifecycle);
            return true;
        },
        replace(options) {
            if (controller._lifecycle !== lifecycle || lifecycle.closed) return null;
            return controller.show(options);
        },
    });
}

function createInactiveModalHandle() {
    return Object.freeze({
        isCurrent: () => false,
        querySelector: () => null,
        close: () => false,
        replace: () => null,
    });
}

// Close contract: X, overlay, Escape, direct hide(), and replacement all run
// the same synchronous teardown. DOM/listeners are removed and opener focus is
// restored before onClose runs, and onClose runs at most once. Action buttons
// remain caller-controlled for compatibility; Modal.confirm()/prompt() record
// their result before hide(), while Promise observers run after teardown. A
// throwing or rejecting action is the exceptional case: its lifecycle is torn
// down before the original error is reported, so broken caller code cannot
// strand the dialog or its root/document focus guards. Handles also scope DOM
// queries to their own live modal, so shadow mounts and replacements cannot be
// confused through global selectors.
export const Modal = {
    _currentModal: null,
    _lifecycle: null,
    _showSequence: 0,

    show(options) {
        const showSequence = ++this._showSequence;
        this.hide();

        // An onClose callback may synchronously open a different modal while
        // the previous one is being replaced. The newest show() request wins;
        // never append a second overlay from the superseded request. The
        // unmounted request is still cancelled through its close callback so
        // confirm(), prompt(), and approval promises cannot remain pending.
        if (showSequence !== this._showSequence) {
            if (typeof options?.onClose === 'function') invokeModalCloseCallback(options.onClose);
            return createInactiveModalHandle();
        }

        const { title, content, buttons = [], onClose } = options;
        const titleId = `modal-title-${++modalId}`;
        const overlayRoot = getOverlayRoot();
        const opener = deepActiveElement(overlayRoot.getRootNode())
            || deepActiveElement(document);

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
        modal.tabIndex = -1;
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.setAttribute('aria-labelledby', titleId);
        modal.style.cssText = `
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 480px;
            width: 100%;
            max-height: 90vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
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
                <h3 id="${titleId}" style="margin: 0; font-size: 1.125rem; font-weight: 600;">${title || 'Dialog'}</h3>
                <button type="button" class="modal-close-btn" aria-label="Close dialog" style="
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
            <div class="modal-body" style="
                padding: 1.5rem;
                overflow-y: auto;
                min-height: 0;
                flex: 1 1 auto;
            ">
                ${content}
            </div>
            ${buttons.length > 0 ? `
                <div class="modal-footer" style="
                    padding: 1rem 1.5rem;
                    border-top: 1px solid var(--border-color);
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: flex-end;
                    gap: 0.75rem;
                    flex: 0 0 auto;
                ">
                    ${buttons.map((btn, i) => `
                        <button type="button" class="modal-btn modal-btn-${btn.type || 'secondary'}" data-btn-index="${i}"${btn.disabled ? ' disabled' : ''}${btn.title ? ` title="${modalAttribute(btn.title)}"` : ''} style="
                            padding: 0.625rem 1.25rem;
                            border: none;
                            border-radius: 8px;
                            font-size: 0.875rem;
                            font-weight: 500;
                            min-width: 0;
                            white-space: nowrap;
                            cursor: ${btn.disabled ? 'not-allowed' : 'pointer'};
                            transition: all 0.2s;
                            ${btn.disabled ? 'opacity: 0.5;' : ''}
                            ${btn.type === 'primary' ? 'background: var(--accent-color); color: white;' : ''}
                            ${btn.type === 'danger' ? 'background: var(--error); color: white;' : ''}
                            ${btn.type === 'secondary' || !btn.type ? 'background: var(--bg-tertiary); color: var(--text-primary);' : ''}
                        ">${btn.label}</button>
                    `).join('')}
                </div>
            ` : ''}
        `;

        const lifecycle = {
            overlay,
            modal,
            root: null,
            opener,
            onClose,
            removeListeners: [],
            closed: false,
            onCloseInvoked: false,
            lastFocused: null,
            redirectingFocus: false,
        };
        const handle = createModalHandle(this, lifecycle);

        const keydownHandler = (e) => {
            if (this._lifecycle !== lifecycle || lifecycle.closed) return;
            // A modal owns every key event from its subtree. Controls still
            // receive the event first, but document-level application
            // shortcuts must never act behind the active dialog.
            e.stopPropagation();

            if (e.key === 'Escape') {
                if (e.isComposing) return;
                if (e.defaultPrevented) return;
                e.preventDefault();
                this._close(lifecycle);
                return;
            }

            if (e.key !== 'Tab') return;
            if (e.defaultPrevented) return;

            const focusable = modalFocusableElements(modal);
            if (focusable.length === 0) {
                e.preventDefault();
                modal.focus();
                return;
            }

            const first = focusable[0];
            const last = focusable[focusable.length - 1];
            const active = deepActiveElement(lifecycle.root);

            if (!focusable.includes(active)) {
                e.preventDefault();
                (e.shiftKey ? last : first).focus();
            } else if (e.shiftKey && active === first) {
                e.preventDefault();
                last.focus();
            } else if (!e.shiftKey && active === last) {
                e.preventDefault();
                first.focus();
            }
        };
        const rootKeydownHandler = (e) => {
            if (this._lifecycle !== lifecycle || lifecycle.closed) return;
            if (modalEventIsInside(e, modal)) return;

            // Removing or disabling the focused control can move focus to the
            // document body without firing focusin. Contain that orphaned key
            // event, then reuse the dialog's normal Escape/Tab behavior.
            keydownHandler(e);
            if (this._lifecycle !== lifecycle || lifecycle.closed || e.key === 'Tab') return;

            const focusable = modalFocusableElements(modal);
            const target = focusable.includes(lifecycle.lastFocused)
                ? lifecycle.lastFocused
                : (focusable[0] || modal);
            target.focus();
            lifecycle.lastFocused = target;
        };
        const rootFocusinHandler = (e) => {
            if (this._lifecycle !== lifecycle || lifecycle.closed || lifecycle.redirectingFocus) return;
            const eventPath = typeof e.composedPath === 'function' ? e.composedPath() : [];
            if (modalEventIsInside(e, modal)) {
                const active = deepActiveElement(lifecycle.root);
                lifecycle.lastFocused = modal.contains(active)
                    ? active
                    : (eventPath.find((target) => target?.nodeType === Node.ELEMENT_NODE
                        && modal.contains(target)) || e.target);
                return;
            }

            const focusable = modalFocusableElements(modal);
            const target = focusable.includes(lifecycle.lastFocused)
                ? lifecycle.lastFocused
                : (focusable[0] || modal);
            lifecycle.redirectingFocus = true;
            try {
                target.focus();
            } finally {
                lifecycle.redirectingFocus = false;
            }
        };
        const documentKeydownHandler = (e) => {
            if (eventOriginatesFromRoot(e, lifecycle.root)) return;
            rootKeydownHandler(e);
        };
        const documentFocusinHandler = (e) => {
            if (eventOriginatesFromRoot(e, lifecycle.root)) return;
            rootFocusinHandler(e);
        };

        try {
            addModalListener(lifecycle, modal.querySelector('.modal-close-btn'), 'click', () => {
                this._close(lifecycle);
            });

            addModalListener(lifecycle, overlay, 'click', (e) => {
                if (e.target === overlay) {
                    this._close(lifecycle);
                }
            });

            modal.querySelectorAll('.modal-btn').forEach(btn => {
                addModalListener(lifecycle, btn, 'click', () => {
                    const index = parseInt(btn.dataset.btnIndex, 10);
                    // Disabled buttons (e.g. a tier-gated approval scope, #2232) are
                    // rendered but inert — the user should SEE the option, not act
                    // on it. Guard here as well as via the `disabled` attribute.
                    if (buttons[index] && buttons[index].disabled) return;
                    if (!buttons[index] || !buttons[index].onClick) return;

                    try {
                        const result = buttons[index].onClick();
                        if (result && typeof result.then === 'function') {
                            Promise.resolve(result).catch((error) => {
                                try {
                                    this._close(lifecycle);
                                } catch (closeError) {
                                    reportModalActionError(new AggregateError(
                                        [error, closeError],
                                        'Modal action and close both failed',
                                    ));
                                    return;
                                }
                                reportModalActionError(error);
                            });
                        }
                    } catch (error) {
                        // A caller-controlled action normally decides whether
                        // the dialog stays open. Once it fails, however, no
                        // caller code remains to perform that close. Preserve
                        // the original error while guaranteeing teardown;
                        // _close() is idempotent when the action hid first.
                        try {
                            this._close(lifecycle);
                        } catch (closeError) {
                            throw new AggregateError(
                                [error, closeError],
                                'Modal action and close both failed',
                            );
                        }
                        throw error;
                    }
                });
            });

            addModalListener(lifecycle, overlay, 'keydown', keydownHandler);

            overlay.appendChild(modal);
            overlayRoot.appendChild(overlay);
            lifecycle.root = modal.getRootNode();
            this._currentModal = overlay;
            this._lifecycle = lifecycle;
            addModalListener(lifecycle, lifecycle.root, 'keydown', rootKeydownHandler, true);
            addModalListener(lifecycle, lifecycle.root, 'focusin', rootFocusinHandler, true);
            if (lifecycle.root !== document) {
                addModalListener(lifecycle, document, 'keydown', documentKeydownHandler, true);
                addModalListener(lifecycle, document, 'focusin', documentFocusinHandler, true);
            }

            // Prefer an explicitly-autofocused/content control, then an action,
            // with the always-present close button as the final fallback.
            const focusable = modalFocusableElements(modal);
            const modalBody = modal.querySelector('.modal-body');
            const autofocus = Array.from(modal.querySelectorAll('[autofocus]'))
                .find((element) => !element.matches(':disabled')
                    && !modalElementIsHidden(element, modal));
            const fallbackFocus = focusable.find((element) => modalBody?.contains(element))
                || focusable.find((element) => element.classList.contains('modal-btn'))
                || focusable.find((element) => element.classList.contains('modal-close-btn'))
                || modal;
            let initialFocus = autofocus
                || focusable.find((element) => modalBody?.contains(element))
                || focusable.find((element) => element.classList.contains('modal-btn'))
                || focusable.find((element) => element.classList.contains('modal-close-btn'))
                || modal;
            initialFocus.focus();
            if (!modal.contains(deepActiveElement(lifecycle.root)) && initialFocus !== fallbackFocus) {
                initialFocus = fallbackFocus;
                initialFocus.focus();
            }
            lifecycle.lastFocused = modal.contains(deepActiveElement(lifecycle.root))
                ? deepActiveElement(lifecycle.root)
                : initialFocus;
        } catch (error) {
            lifecycle.onClose = null;
            try {
                this._close(lifecycle);
            } catch (cleanupError) {
                throw new AggregateError([error, cleanupError], 'Modal setup and cleanup both failed');
            }
            throw error;
        }
        return handle;
    },

    hide() {
        if (this._lifecycle) {
            this._close(this._lifecycle);
        }
    },

    _close(lifecycle) {
        if (!lifecycle || lifecycle.closed) return;
        lifecycle.closed = true;

        const errors = [];

        for (const removeListener of lifecycle.removeListeners) {
            try {
                removeListener();
            } catch (error) {
                errors.push(error);
            }
        }
        lifecycle.removeListeners.length = 0;
        try {
            lifecycle.overlay.remove();
        } catch (error) {
            errors.push(error);
        }

        if (this._lifecycle === lifecycle) {
            this._lifecycle = null;
            this._currentModal = null;
        }

        // Teardown and focus restoration are complete before onClose runs.
        // Teardown stages are isolated so one failure cannot skip onClose.
        // Collected caller errors still fail visibly once every stage ran.
        if (lifecycle.opener
            && lifecycle.opener !== document.body
            && lifecycle.opener.isConnected
            && typeof lifecycle.opener.focus === 'function') {
            try {
                lifecycle.opener.focus();
            } catch (error) {
                errors.push(error);
            }
        }

        if (!lifecycle.onCloseInvoked && typeof lifecycle.onClose === 'function') {
            lifecycle.onCloseInvoked = true;
            try {
                invokeModalCloseCallback(lifecycle.onClose);
            } catch (error) {
                errors.push(error);
            }
        }

        if (errors.length === 1) throw errors[0];
        if (errors.length > 1) throw new AggregateError(errors, 'Modal close failed');
    },

    confirm(title, message) {
        return new Promise((resolve) => {
            let handle;
            handle = this.show({
                title,
                content: `<p style="margin: 0; color: var(--text-secondary); line-height: 1.6;">${message}</p>`,
                buttons: [
                    { label: 'Cancel', type: 'secondary', onClick: () => { resolve(false); handle.close(); } },
                    { label: 'Confirm', type: 'primary', onClick: () => { resolve(true); handle.close(); } }
                ],
                onClose: () => resolve(false)
            });
        });
    },

    prompt(title, placeholder = '', defaultValue = '') {
        return new Promise((resolve) => {
            const inputId = `modal-prompt-input-${++modalId}`;
            let handle;
            handle = this.show({
                title,
                content: `
                    <input type="text" id="${inputId}"
                        aria-label="${modalAttribute(title || 'Response')}"
                        placeholder="${modalAttribute(placeholder)}"
                        value="${modalAttribute(defaultValue)}"
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
                    { label: 'Cancel', type: 'secondary', onClick: () => { resolve(null); handle.close(); } },
                    { label: 'OK', type: 'primary', onClick: () => {
                        const value = handle.querySelector(`#${inputId}`)?.value || '';
                        resolve(value);
                        handle.close();
                    }}
                ],
                onClose: () => resolve(null)
            });

            const lifecycle = handle.isCurrent() ? this._lifecycle : null;
            const input = handle.querySelector(`#${inputId}`);
            if (input && !lifecycle.closed) {
                addModalListener(lifecycle, input, 'keydown', (e) => {
                    if (e.key === 'Enter' && !e.isComposing) {
                        e.stopPropagation();
                        if (e.defaultPrevented) return;
                        e.preventDefault();
                        resolve(input.value);
                        handle.close();
                    }
                });
            }
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
    renderTextError(document.getElementById(elementId), message);
}

export function renderTextError(target, message) {
    if (!target) return;
    const error = document.createElement('div');
    error.style.cssText = 'color: var(--error); padding: 1rem;';
    error.textContent = String(message ?? '');
    target.replaceChildren(error);
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
