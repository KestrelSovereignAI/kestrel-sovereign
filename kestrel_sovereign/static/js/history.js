/**
 * Kestrel Sovereign Console - History Module
 * Chat History Browser
 *
 * #2149: the conversation-LIST render + rename that used to live here has been
 * consolidated into the shared ``conversations.js`` component.
 *
 * #2171: the standalone console collapsed to a SINGLE conversation surface —
 * the ``#conversations-pane`` sidebar (owned by identity.js ``loadConversations``).
 * The history slideout this module used to mount the shared component into is
 * gone. This module now keeps only the message-level chat rendering (loading a
 * conversation's messages into the chat pane, restart-status repaint, typed-part
 * interleave) plus the new-conversation flow.
 */

import API from './api.js';
import { state, Toast } from './ui.js';
import {
    updateContextStatus,
    wipeAgentChatPane,
} from './chat.js';

// ============================================================================
// Chat History — conversation-message rendering (#2149 / #2171)
// ============================================================================

state.conversations = null;
state.currentSessionId = null;

// #2171: the conversation LIST now lives solely in the ``#conversations-pane``
// sidebar, rendered by identity.js ``loadConversations``. This function is kept
// (panels.js still re-exports it) as the single "the list is stale, refresh it"
// entry point for the message-level flows in this module (new conversation,
// encryption-view toggle). It signals the sidebar owner via the shared stale
// event rather than mounting a second surface of its own.
export async function loadConversationHistory() {
    window.dispatchEvent(new CustomEvent('kestrel:conversations-stale'));
}

// #2380: the module-scope ``window.loadConversation`` that used to live here was
// a legacy near-duplicate of the canonical loader in identity.js — no
// stale-agent guards (#1358), no #2222 highlight wiring — and, because
// explorers.js statically imports history.js AFTER panels.js re-exports
// identity.js, it clobbered the global and every standalone-console row click
// ran the guard-less, highlight-less loader (the reported "selected conversation
// never highlights" bug). The loader is deleted; its two unique behaviors (the
// encrypted-at-rest banner and the #1816 restart-status trail interleave) were
// ported into the canonical identity.js loader. This module keeps only its
// message-level flows (new conversation, encryption-view toggle, delete/purge).

window.toggleEncryptionView = async function() {
    state.showDecrypted = !state.showDecrypted;

    await loadConversationHistory();

    if (state.currentSessionId) {
        // force: this IS a same-session reload — the canonical loader's
        // same-session no-op (#2380) must not skip the re-render that swaps
        // between decrypted and raw-ciphertext views.
        await window.loadConversation(state.currentSessionId, { force: true });
    }

    Toast.info(state.showDecrypted ? '\u{1F513} Now viewing decrypted content' : '\u{1F510} Now viewing raw encrypted data');
};

window.startNewConversation = async function() {
    try {
        // #2222: when the shared conversations pane is mounted for this host,
        // route through its component-owned new-conversation action so the New
        // tile appears instantly, becomes the CURRENT conversation (active
        // highlight + subsequent messages land in it), and the host-side chat
        // wiping / state / context-footer update runs exactly once (via the
        // pane's `onNewConversation`). The pane's optimistic prepend replaces
        // the pre-#2199 `loadConversationHistory()` list refresh here.
        const viaPane = (typeof window.newConversationViaPane === 'function')
            ? window.newConversationViaPane()
            : null;
        if (viaPane) {
            // The pane's newConversation() handles its own failure (Toast.error
            // + resolves null), so only claim success when it actually minted a
            // session — otherwise the user would see both the error toast and a
            // contradictory success toast.
            const result = await viaPane;
            if (result && result.session_id) {
                Toast.success('New conversation started');
            }
            return;
        }

        // Fallback (no pane mounted — e.g. a host without the conversations
        // surface): do the chat-side flow directly.
        const result = await API.newConversation();

        // Wipe the visible agent's pane and bump that agent's pane-
        // local generation so any stream still running against the
        // previous (now-replaced) session gates out. Other agents are
        // unaffected. Then write currentSessionId via the property,
        // which writes into the visible agent's pane.
        wipeAgentChatPane(API.getHostAgent(), `
            <div style="text-align: center; padding: 2rem; color: var(--text-secondary);">
                <span style="font-size: 2rem;">\u{2728}</span>
                <p style="margin-top: 0.5rem;">New conversation started. Say hello!</p>
            </div>
        `);
        state.currentSessionId = result.session_id;

        // Signal the shared pane owner (identity.js) to reconcile its list.
        await loadConversationHistory();

        // Refresh the context status for the new (empty) session so the
        // footer stops reporting stale message count / utilization from the
        // previous conversation.
        updateContextStatus();

        Toast.success('New conversation started');
    } catch (e) {
        Toast.error(`Failed to start new conversation: ${e.message}`);
    }
};

// #1914: a typed-part message renders as several sibling bubbles that all
// carry the same ``data-message-id``. Fade+remove every node for the id (not
// just the clicked one) so a delete doesn't orphan the part/extra-prose
// bubbles. Falls back to the passed node when none are tagged (single-bubble
// messages predating this path / non-id bubbles).
function fadeRemoveMessageNodes(messageId, fallbackDiv) {
    let nodes = [];
    if (messageId && typeof document.querySelectorAll === 'function') {
        // Escape the id for an attribute selector (quotes/backslashes).
        const sel = String(messageId).replace(/(["\\])/g, '\\$1');
        nodes = Array.from(
            document.querySelectorAll(`.message[data-message-id="${sel}"]`),
        );
    }
    if (!nodes.length && fallbackDiv) nodes = [fallbackDiv];
    for (const node of nodes) {
        node.style.transition = 'opacity 0.2s, transform 0.2s';
        node.style.opacity = '0';
        node.style.transform = 'scale(0.95)';
        setTimeout(() => node.remove(), 200);
    }
}

window.deleteMessage = async function(messageId, messageDiv) {
    // Soft-delete (#763) — moves the message to Trash, recoverable from
    // the trash sub-view (#765).
    if (!confirm('Move this message to Trash? You can restore it from the trash view.')) return;

    try {
        await API.deleteMessage(messageId);
        fadeRemoveMessageNodes(messageId, messageDiv);
        Toast.info('Message moved to trash');
    } catch (e) {
        Toast.error(`Failed to delete message: ${e.message}`);
    }
};

window.purgeMessage = async function(messageId, messageDiv) {
    // Permanent delete (#765) — hard SQL DELETE, no recovery.
    if (!confirm(
        `Delete this message PERMANENTLY?\n\n`
        + `This is a hard delete and CANNOT be restored. Soft-delete first `
        + `(the regular ✕) is the recoverable path.`
    )) return;

    try {
        await API.purgeMessage(messageId, 'user-initiated-ui');
        fadeRemoveMessageNodes(messageId, messageDiv);
        Toast.info('Message permanently deleted');
    } catch (e) {
        Toast.error(`Failed to permanently delete: ${e.message}`);
    }
};
