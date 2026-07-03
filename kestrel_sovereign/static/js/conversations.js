/**
 * Kestrel Sovereign Console - Conversation List Component (#2149)
 *
 * ONE conversation-list surface. Before this module the console shipped two
 * parallel, divergent list UIs — the `#conversations-pane` sidebar
 * (identity.js) and the history slideout (history.js) — each with its own
 * render, its own rename implementation, and a mutually-incomplete action set.
 * This module owns the shared list logic so both surfaces (and embedding hosts
 * like Frinz) render identical rows with the full affordance set:
 *
 *   - timeline-grouped rows with preview + per-row kebab (⋯) overflow menu
 *   - Rename (inline edit — the ONE rename path; the old prompt() is retired)
 *   - Archive / Unarchive (a first-class tidy-away state, #2149 backend)
 *   - Move to Trash, and a danger-styled Delete Permanently (confirm-gated,
 *     preserving the #765 slow-down)
 *   - right-click (contextmenu) opens the SAME menu as an accelerator
 *   - views/filters: Active, Archived, Trash (trash keeps restore/purge)
 *
 * Two consumption shapes:
 *   - `buildConversationRow` / `renderConversationList` — the shared row +
 *     list primitives the sidebar reuses while keeping its own
 *     request-sequencing / agent-pinning guards and auto-load (identity.js).
 *   - `mountConversations(containerEl, config)` — a self-contained, embeddable
 *     surface (same contract family as chat's `mount()` / the panel host's
 *     `mountPanels()`), used by the history slideout and by embedders.
 */

import API from './api.js';
import { state, Toast, escapeHtml as sharedEscapeHtml } from './ui.js';
import { groupTrashBySession, trashGroupKey } from './trash_grouping.js';
import {
    createKebabButton,
    openMenuAt,
    positionFromEvent,
    closeKebabMenu,
} from './kebab_menu.js';

// #1816: DB timestamps are UTC but conversation rows serialize naive (no tz
// suffix). Date.parse() reads a naive string as LOCAL time, so pin tz-less
// strings to UTC ('Z') before parsing — otherwise timeline ordering drifts by
// the viewer's local offset. Preserved verbatim from history.js.
export function timelineTs(value) {
    const s = String(value || '');
    if (!s) return 0;
    const hasTz = /[zZ]$|[+-]\d{2}:?\d{2}$/.test(s);
    const t = Date.parse(hasTz ? s : `${s}Z`);
    return Number.isNaN(t) ? 0 : t;
}

function formatDateLabel(dateStr) {
    const date = new Date(dateStr);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    if (date.toDateString() === today.toDateString()) return 'Today';
    if (date.toDateString() === yesterday.toDateString()) return 'Yesterday';
    return dateStr;
}

function esc(fn, value) {
    const e = fn || sharedEscapeHtml;
    return typeof e === 'function' ? e(value) : String(value == null ? '' : value);
}

// ============================================================================
// Inline rename (the single rename implementation, #2149)
// ============================================================================
//
// Moved from identity.js's beginRenameConversation. Swaps the preview text for
// an inline input seeded with the current display name; commit on Enter/blur,
// cancel on Escape. All state is closure-local. The prompt()-based rename in
// history.js is retired.
export function beginInlineRename(previewEl, conv, opts = {}) {
    const api = opts.api || API;
    const originalText = previewEl.textContent;
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'conversation-rename-input';
    input.value = conv.name || '';
    input.placeholder = originalText || 'Conversation name';
    input.maxLength = 120;
    // Keystrokes/clicks inside the input must not bubble to the row's
    // click → load handler while the user is typing.
    input.addEventListener('click', (e) => e.stopPropagation());

    let finalized = false;

    async function commit() {
        if (finalized) return;
        finalized = true;
        const newName = input.value;
        const storedName = conv.name || '';
        if (newName.trim() === storedName.trim()) {
            previewEl.textContent = originalText;
            return;
        }
        try {
            const result = await api.renameConversation(conv.session_id, newName);
            const applied = result && result.name;
            conv.name = applied || null;
            previewEl.textContent = applied || conv.preview || 'New conversation';
            Toast.info(applied ? 'Conversation renamed' : 'Conversation name cleared');
            if (typeof opts.onCommitted === 'function') opts.onCommitted(conv);
        } catch (e) {
            previewEl.textContent = originalText;
            Toast.error(`Rename failed: ${e.message}`);
        }
    }

    function cancel() {
        if (finalized) return;
        finalized = true;
        previewEl.textContent = originalText;
    }

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
        else if (e.key === 'Escape') { e.preventDefault(); cancel(); previewEl.textContent = originalText; }
    });
    input.addEventListener('blur', () => { if (!finalized) commit(); });

    previewEl.textContent = '';
    previewEl.appendChild(input);
    if (typeof input.focus === 'function') input.focus();
    if (typeof input.select === 'function') input.select();
}

// ============================================================================
// Shared row primitive
// ============================================================================

/**
 * Build a single `.conversation-item` row node. Shared by the sidebar and the
 * mount so there is exactly one row render. Options:
 *   active         — whether to mark the row active
 *   agentKey       — value for data-agent-key (sidebar agent-pinning, #1358)
 *   escapeHtml     — escape helper (defaults to the shared one)
 *   showKebab      — render the ⋯ overflow button (default true)
 *   onSelect       — (conv, rowEl) => void; row click / Enter
 *   buildMenuItems — (conv, rowEl) => items[]; kebab + contextmenu menu
 *   onRename       — (previewEl, conv, rowEl) => void; dblclick affordance
 */
export function buildConversationRow(conv, opts = {}) {
    const escapeFn = opts.escapeHtml || sharedEscapeHtml;
    const row = document.createElement('div');
    row.className = `conversation-item${opts.active ? ' active' : ''}`;
    row.dataset.sessionId = conv.session_id;
    if (opts.agentKey !== undefined && opts.agentKey !== null) {
        row.dataset.agentKey = opts.agentKey;
    }
    row.tabIndex = 0;

    const displayName = (conv.name || '').trim();
    const preview = conv.preview || 'New conversation';
    const displayText = displayName || preview;

    const metaRow = document.createElement('div');
    metaRow.className = 'conversation-meta-row';

    const time = document.createElement('span');
    time.className = 'conversation-time';
    const startedRaw = conv.started_at || conv.last_message_at || conv.deleted_at
        || conv.archived_at;
    time.textContent = startedRaw ? new Date(startedRaw).toLocaleString() : '';
    metaRow.appendChild(time);

    if (typeof conv.message_count === 'number') {
        const count = document.createElement('span');
        count.className = 'conversation-msg-count';
        count.textContent = `${conv.message_count} msgs`;
        metaRow.appendChild(count);
    }

    const buildMenuItems = typeof opts.buildMenuItems === 'function'
        ? opts.buildMenuItems
        : () => [];

    if (opts.showKebab !== false) {
        const kebab = createKebabButton(() => buildMenuItems(conv, row), {
            className: 'conv-kebab-btn',
            ariaLabel: `Actions for ${displayText}`,
        });
        metaRow.appendChild(kebab);
    }
    row.appendChild(metaRow);

    const previewEl = document.createElement('div');
    previewEl.className = 'conversation-preview';
    previewEl.textContent = displayText;
    previewEl.title = displayName
        ? `${displayName} — ${conv.preview || ''}`
        : (conv.preview || 'Double-click to rename');
    // Double-click begins inline rename — a discoverable accelerator kept from
    // the sidebar (#716). The kebab's Rename item is the primary path.
    const renameTrigger = typeof opts.onRename === 'function'
        ? opts.onRename
        : (pEl, c) => beginInlineRename(pEl, c, { api: opts.api });
    previewEl.addEventListener('dblclick', (e) => {
        e.stopPropagation();
        renameTrigger(previewEl, conv, row);
    });
    row.appendChild(previewEl);

    if (typeof opts.onSelect === 'function') {
        row.addEventListener('click', () => opts.onSelect(conv, row));
        row.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') opts.onSelect(conv, row);
        });
    }

    // Right-click opens the SAME menu as the kebab (accelerator; never the only
    // path — the kebab and dblclick cover touch/discoverability).
    row.addEventListener('contextmenu', (e) => {
        if (e && typeof e.preventDefault === 'function') e.preventDefault();
        openMenuAt(buildMenuItems(conv, row), positionFromEvent(e));
    });

    return row;
}

/**
 * Render a full list into `container` with optional timeline grouping. Shared
 * by the mount; the sidebar renders flat rows itself (its rows must be direct
 * children of `#conversations-list` for the agent-pinning tests).
 */
export function renderConversationList(container, conversations, opts = {}) {
    container.innerHTML = '';
    const group = opts.group !== false;
    if (!group) {
        for (const conv of conversations) {
            container.appendChild(buildConversationRow(conv, opts));
        }
        return;
    }
    const grouped = new Map();
    for (const conv of conversations) {
        const raw = conv.started_at || conv.last_message_at || conv.deleted_at
            || conv.archived_at;
        const key = raw ? new Date(raw).toLocaleDateString() : 'Unknown';
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(conv);
    }
    for (const [dateKey, convs] of grouped) {
        const groupEl = document.createElement('div');
        groupEl.className = 'date-group';
        const label = document.createElement('div');
        label.className = 'date-group-label';
        label.textContent = formatDateLabel(dateKey);
        groupEl.appendChild(label);
        for (const conv of convs) {
            groupEl.appendChild(buildConversationRow(conv, opts));
        }
        container.appendChild(groupEl);
    }
}

// ============================================================================
// Default menu-item builders (used by the mount; the sidebar supplies its own
// so it can route trash/purge through identity.js's pane-aware handlers)
// ============================================================================

function confirmTrash() {
    return typeof confirm !== 'function' || confirm(
        'Move this conversation to Trash? You can restore it from the trash '
        + 'view, or delete it permanently from there.',
    );
}

function confirmPurge() {
    return typeof confirm !== 'function' || confirm(
        'Delete this conversation PERMANENTLY?\n\n'
        + 'This is a hard delete — every message in the session will be removed '
        + 'and CANNOT be restored. Move to Trash first is the recoverable path.',
    );
}

function animateOut(rowEl) {
    if (!rowEl) return;
    rowEl.style.transition = 'opacity 0.2s, transform 0.2s';
    rowEl.style.opacity = '0';
    rowEl.style.transform = 'scale(0.97)';
    setTimeout(() => rowEl.remove(), 200);
}

// ============================================================================
// mountConversations — embeddable surface
// ============================================================================

/**
 * Mount the full conversation-list surface into `containerEl`. Returns a
 * handle: `{ element, refresh, retarget(agentName), setView(view), destroy }`.
 * `retarget` lets an embedding host repoint the list when the active agent
 * switches (same contract family as chat's mount).
 */
export function mountConversations(containerEl, config = {}) {
    if (!containerEl) throw new Error('mountConversations requires a container element');
    const api = config.api || API;
    const onSelect = typeof config.onSelect === 'function'
        ? config.onSelect
        : (conv) => {
            if (typeof window !== 'undefined' && typeof window.loadConversation === 'function') {
                window.loadConversation(conv.session_id);
            }
        };
    const getActiveSessionId = typeof config.getActiveSessionId === 'function'
        ? config.getActiveSessionId
        : () => (state ? state.currentSessionId : null);

    let view = config.view || 'active';
    let searchTerm = '';
    let agentName = config.agentName;
    let lastConversations = [];

    containerEl.classList && containerEl.classList.add('conversations-component');

    const root = document.createElement('div');
    root.className = 'conversations-root';

    const controls = document.createElement('div');
    controls.className = 'conversations-controls';

    // View switcher (Active / Archived / Trash) — the generalized trash toggle.
    const viewBar = document.createElement('div');
    viewBar.className = 'conversations-view-bar';
    const VIEWS = [
        { key: 'active', label: 'Active', labelKey: 'conv_view_active' },
        { key: 'archived', label: 'Archived', labelKey: 'conv_view_archived' },
        { key: 'trash', label: 'Trash', labelKey: 'conv_view_trash' },
    ];
    const viewButtons = new Map();
    for (const v of VIEWS) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'conversations-view-btn';
        b.dataset.view = v.key;
        b.dataset.labelKey = v.labelKey;
        b.textContent = v.label;
        b.addEventListener('click', () => setView(v.key));
        viewButtons.set(v.key, b);
        viewBar.appendChild(b);
    }
    controls.appendChild(viewBar);

    // Search / filter box.
    const search = document.createElement('input');
    search.type = 'search';
    search.className = 'conversations-search';
    search.placeholder = 'Search conversations…';
    search.dataset.labelAttrPlaceholder = 'conv_search_placeholder';
    if (config.showSearch !== false) {
        search.addEventListener('input', () => {
            searchTerm = String(search.value || '').trim().toLowerCase();
            renderCurrent();
        });
        controls.appendChild(search);
    }
    root.appendChild(controls);

    const stats = document.createElement('div');
    stats.className = 'conversations-stats';
    if (config.showStats !== false) root.appendChild(stats);

    const listEl = document.createElement('div');
    listEl.className = 'conversations-list-body';
    root.appendChild(listEl);

    containerEl.innerHTML = '';
    containerEl.appendChild(root);

    function syncViewButtons() {
        for (const [key, btn] of viewButtons) {
            btn.classList.toggle('active', key === view);
        }
    }

    function filtered(conversations) {
        if (!searchTerm) return conversations;
        return conversations.filter((c) => {
            const hay = `${c.name || ''} ${c.preview || ''}`.toLowerCase();
            return hay.includes(searchTerm);
        });
    }

    function renderStats(conversations) {
        if (config.showStats === false) return;
        const sessions = conversations.length;
        const messages = conversations.reduce(
            (sum, c) => sum + (Number(c.message_count) || 0), 0,
        );
        stats.textContent = `${sessions} conversation${sessions === 1 ? '' : 's'}`
            + (messages ? ` · ${messages} messages` : '');
    }

    function menuItemsFor(conv, rowEl) {
        if (view === 'trash') {
            return [
                {
                    label: 'Restore', labelKey: 'conv_menu_restore', action: 'restore',
                    onSelect: () => doRestore(conv, rowEl),
                },
                {
                    label: 'Delete Permanently', labelKey: 'conv_menu_delete_permanent',
                    action: 'purge', danger: true, separatorBefore: true,
                    onSelect: () => doPurge(conv, rowEl),
                },
            ];
        }
        const archived = view === 'archived';
        return [
            {
                label: 'Rename', labelKey: 'conv_menu_rename', action: 'rename',
                onSelect: () => {
                    const previewEl = rowEl.querySelector
                        ? rowEl.querySelector('.conversation-preview')
                        : null;
                    if (previewEl) beginInlineRename(previewEl, conv, { api, onCommitted: refresh });
                },
            },
            archived
                ? {
                    label: 'Unarchive', labelKey: 'conv_menu_unarchive', action: 'unarchive',
                    onSelect: () => doUnarchive(conv, rowEl),
                }
                : {
                    label: 'Archive', labelKey: 'conv_menu_archive', action: 'archive',
                    onSelect: () => doArchive(conv, rowEl),
                },
            {
                label: 'Move to Trash', labelKey: 'conv_menu_trash', action: 'trash',
                onSelect: () => doTrash(conv, rowEl),
            },
            {
                label: 'Delete Permanently', labelKey: 'conv_menu_delete_permanent',
                action: 'purge', danger: true, separatorBefore: true,
                onSelect: () => doPurge(conv, rowEl),
            },
        ];
    }

    async function doArchive(conv, rowEl) {
        try {
            await api.archiveConversation(conv.session_id);
            animateOut(rowEl);
            Toast.info('Conversation archived');
            refresh();
        } catch (e) { Toast.error(`Failed to archive: ${e.message}`); }
    }
    async function doUnarchive(conv, rowEl) {
        try {
            await api.unarchiveConversation(conv.session_id);
            animateOut(rowEl);
            Toast.info('Conversation unarchived');
            refresh();
        } catch (e) { Toast.error(`Failed to unarchive: ${e.message}`); }
    }
    async function doTrash(conv, rowEl) {
        if (!confirmTrash()) return;
        try {
            await api.deleteConversation(conv.session_id);
            animateOut(rowEl);
            Toast.info('Conversation moved to trash');
            refresh();
        } catch (e) { Toast.error(`Failed to delete: ${e.message}`); }
    }
    async function doPurge(conv, rowEl) {
        if (!confirmPurge()) return;
        try {
            await api.purgeConversation(conv.session_id, 'user-initiated-ui');
            animateOut(rowEl);
            Toast.info('Conversation permanently deleted');
            refresh();
        } catch (e) { Toast.error(`Failed to permanently delete: ${e.message}`); }
    }
    async function doRestore(conv, rowEl) {
        try {
            await api.restoreConversation(conv.session_id);
            animateOut(rowEl);
            Toast.success('Conversation restored');
            refresh();
        } catch (e) { Toast.error(`Failed to restore: ${e.message}`); }
    }

    function rowOpts() {
        const activeId = getActiveSessionId();
        return {
            api,
            group: config.group !== false,
            escapeHtml: config.escapeHtml,
            onSelect: (conv) => onSelect(conv),
            buildMenuItems: menuItemsFor,
            // active flag applied per-row below via renderCurrent
            _activeId: activeId,
        };
    }

    function renderCurrent() {
        const convs = filtered(lastConversations);
        renderStats(convs);
        if (!convs.length) {
            listEl.innerHTML = '';
            const empty = document.createElement('p');
            empty.className = 'empty-state';
            empty.textContent = view === 'trash'
                ? 'Nothing in trash.'
                : (view === 'archived' ? 'No archived conversations.' : 'No conversations yet.');
            listEl.appendChild(empty);
            return;
        }
        const activeId = getActiveSessionId();
        const opts = rowOpts();
        // Wrap buildConversationRow so each row gets its active flag.
        listEl.innerHTML = '';
        const group = opts.group;
        const build = (conv) => buildConversationRow(conv, {
            ...opts,
            active: conv.session_id === activeId,
        });
        if (!group) {
            for (const conv of convs) listEl.appendChild(build(conv));
            return;
        }
        const grouped = new Map();
        for (const conv of convs) {
            const raw = conv.started_at || conv.last_message_at || conv.deleted_at
                || conv.archived_at;
            const key = raw ? new Date(raw).toLocaleDateString() : 'Unknown';
            if (!grouped.has(key)) grouped.set(key, []);
            grouped.get(key).push(conv);
        }
        for (const [dateKey, list] of grouped) {
            const groupEl = document.createElement('div');
            groupEl.className = 'date-group';
            const label = document.createElement('div');
            label.className = 'date-group-label';
            label.textContent = formatDateLabel(dateKey);
            groupEl.appendChild(label);
            for (const conv of list) groupEl.appendChild(build(conv));
            listEl.appendChild(groupEl);
        }
    }

    async function loadData() {
        const decrypt = state ? state.showDecrypted : true;
        if (view === 'trash') {
            const data = await api.listTrash(500);
            const { sessions } = groupTrashBySession(data.messages || []);
            return (sessions || []).map((s) => ({
                session_id: s.session_id,
                preview: s.preview,
                message_count: s.count,
                deleted_at: s.deleted_at,
                started_at: s.deleted_at,
            }));
        }
        const data = await api.getConversations(decrypt, view);
        return data.conversations || [];
    }

    async function refresh() {
        syncViewButtons();
        try {
            lastConversations = await loadData();
            renderCurrent();
        } catch (e) {
            listEl.innerHTML = '';
            const err = document.createElement('p');
            err.className = 'conversations-error';
            err.textContent = `Failed to load conversations: ${e.message}`;
            listEl.appendChild(err);
        }
    }

    function setView(next) {
        if (next === view) return;
        view = next;
        searchTerm = '';
        search.value = '';
        refresh();
    }

    function retarget(nextAgentName) {
        agentName = nextAgentName;
        // Routing is handled by the API layer (API.setHostAgent); just reload.
        refresh();
    }

    function destroy() {
        closeKebabMenu();
        containerEl.innerHTML = '';
    }

    syncViewButtons();
    if (config.autoLoad !== false) refresh();

    return { element: root, refresh, retarget, setView, destroy, get view() { return view; } };
}
