/**
 * Kestrel Sovereign Console - Message-bubble kebab menu (#2410)
 *
 * One shared kebab ("⋯") builder for chat message bubbles, replacing the twin
 * hover-revealed red delete/purge circles that used to fight for the bubble's
 * top-right corner. Both chat.js (signal-wake chips) and identity.js (the two
 * conversation loaders) render bubbles, so the builder lives here rather than
 * in a single loader.
 *
 * Reuses the conversation-card kebab primitive (`kebab_menu.js`): a single
 * hover/focus-revealed button whose menu is built lazily on open, plus the same
 * `contextmenu` (right-click) accelerator conversation rows have so right-click
 * is never the only path.
 *
 * The kebab is also the single sanctioned surface for feature-contributed
 * per-message actions: features register `chat-message-actions` item providers
 * (see `ui-ext/contract.js`); they contribute menu ITEMS, never overlay DOM.
 */

import { createKebabButton, openMenuAt, positionFromEvent } from './kebab_menu.js';
import { UI } from './ui-ext/registry.js';
import API from './api.js';

/** Resolve the owning agent name from the bubble node's pane, if available. */
function resolveAgent(node) {
    if (!node || typeof node.closest !== 'function') return undefined;
    const pane = node.closest('[data-agent]');
    return pane ? pane.dataset.agent : undefined;
}

/**
 * Build the ordered menu items for one message bubble: base trash + permanent
 * delete, with any gated/ordered `chat-message-actions` feature items folded in
 * ABOVE the destructive separator. Feature providers are error-isolated by
 * `UI.collectItems`, so a throwing contribution never drops the base items.
 */
export function messageMenuItems(msg, node, api = API) {
    const id = msg && msg.id;
    const trash = {
        label: 'Move to trash', labelKey: 'msg_menu_trash', action: 'trash',
        onSelect: () => {
            if (typeof window.deleteMessage === 'function') window.deleteMessage(id, node);
        },
    };
    const purge = {
        label: 'Delete permanently', labelKey: 'msg_menu_delete_permanent', action: 'purge',
        danger: true, separatorBefore: true,
        onSelect: () => {
            if (typeof window.purgeMessage === 'function') window.purgeMessage(id, node);
        },
    };

    let extra = [];
    if (UI && typeof UI.collectItems === 'function') {
        // Same `api` handle every render-based slot ctx carries (chat.js,
        // agent_list.js) — providers gate on `ctx.api.hasCapability(...)`.
        // Callers on an embedded/custom-deps surface thread their mounted
        // client through; the module singleton is only the default.
        extra = UI.collectItems('chat-message-actions', {
            messageId: id,
            role: msg && msg.role,
            metadata: msg && msg.metadata,
            agent: resolveAgent(node),
            api,
        }) || [];
    }

    // Base trash first, feature items next (above the destructive separator),
    // permanent delete last.
    return [trash, ...extra, purge];
}

/**
 * Build the single hover-revealed kebab button for a message bubble. Wires the
 * `contextmenu` accelerator on the bubble node (matching conversation rows) so
 * right-click opens the same menu. `getItems` is called on every open so
 * feature items reflect current state.
 */
export function buildMessageKebab(msg, node, api = API) {
    const kebab = createKebabButton(() => messageMenuItems(msg, node, api), {
        className: 'msg-kebab-btn',
        ariaLabel: 'Message actions',
    });
    if (node && typeof node.addEventListener === 'function') {
        node.addEventListener('contextmenu', (e) => {
            if (e && typeof e.preventDefault === 'function') e.preventDefault();
            openMenuAt(messageMenuItems(msg, node, api), positionFromEvent(e));
        });
    }
    return kebab;
}
