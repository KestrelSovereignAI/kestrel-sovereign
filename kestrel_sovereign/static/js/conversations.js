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
 * Three consumption shapes:
 *   - `buildConversationRow` / `renderConversationList` — the shared row +
 *     list primitives.
 *   - `mountConversations(containerEl, config)` — a self-contained, embeddable
 *     LIST surface (same contract family as chat's `mount()` / the panel host's
 *     `mountPanels()`), used by the history slideout and by embedders.
 *   - `mountConversationsPane(containerEl, config)` — the full collapsible PANE
 *     unit: `mountConversations` PLUS the pane chrome (a chevron that fully
 *     hides the pane, drag-resize with min/max + localStorage persistence, and a
 *     search/view-bar/stats disclosure) with `open()/close()/toggle()` +
 *     `onToggle`. The standalone console (#2199) and any embedder consume THIS
 *     one implementation; a host provides only a container + config.
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
import { storeGet, storeSet } from './ui_state.mjs';

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

// #2165: compact meta-row time. In narrow sidebars (~250px) the verbose
// `toLocaleString()` timestamp ("7/4/2026, 11:05:05 AM") wrapped onto two
// lines and shoved the msg-count/kebab out of place. Collapse it:
//   - same day  → time only ("11:05 AM")
//   - this year → "Jul 4"
//   - older     → "7/4/25"
// Reuses timelineTs's #1816 UTC pinning so naive (tz-less) DB strings parse as
// UTC, not local. Returns { text, title } — the full timestamp rides the title
// attribute for hover.
export function formatConversationTime(value) {
    const s = String(value || '');
    if (!s) return { text: '', title: '' };
    const ms = timelineTs(s);
    if (!ms) return { text: '', title: s };
    const date = new Date(ms);
    const now = new Date();
    const full = date.toLocaleString();
    let text;
    if (date.toDateString() === now.toDateString()) {
        text = date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    } else if (date.getFullYear() === now.getFullYear()) {
        text = date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    } else {
        text = date.toLocaleDateString([], { year: '2-digit', month: 'numeric', day: 'numeric' });
    }
    return { text, title: full };
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

// Append `text` to `el` as text nodes with each case-insensitive occurrence of
// `term` wrapped in <mark>. DOM-node construction only — snippet text is
// decrypted message content and must never travel through innerHTML.
export function appendHighlighted(el, text, term) {
    const s = String(text == null ? '' : text);
    const t = String(term == null ? '' : term);
    if (!t) {
        el.appendChild(document.createTextNode(s));
        return;
    }
    const lower = s.toLowerCase();
    const needle = t.toLowerCase();
    let pos = 0;
    for (;;) {
        const idx = lower.indexOf(needle, pos);
        if (idx < 0) break;
        if (idx > pos) el.appendChild(document.createTextNode(s.slice(pos, idx)));
        const mark = document.createElement('mark');
        mark.textContent = s.slice(idx, idx + needle.length);
        el.appendChild(mark);
        pos = idx + needle.length;
    }
    if (pos < s.length) el.appendChild(document.createTextNode(s.slice(pos)));
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
        // stopPropagation: the row's own keydown handler selects/loads the
        // conversation on Enter — committing a rename must not also trigger
        // that (and Escape must not bubble into row/global handlers either).
        if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); input.blur(); }
        else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); cancel(); previewEl.textContent = originalText; }
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
    const started = formatConversationTime(startedRaw);
    time.textContent = started.text;
    if (started.title) time.title = started.title;
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

    // Full-text search hit context: a decrypted excerpt around the first
    // matching message, with the matched term highlighted. Built with text
    // nodes + <mark> (never innerHTML) so message content can't inject markup.
    if (conv.match_snippet) {
        const snip = document.createElement('div');
        snip.className = 'conversation-match-snippet';
        appendHighlighted(snip, conv.match_snippet, opts.highlightTerm || '');
        row.appendChild(snip);
    }

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
 * handle: `{ element, refresh, retarget(agentName), setView(view),
 * setActiveSessionId(id), newConversation(), destroy }`.
 * `retarget` lets an embedding host repoint the list when the active agent
 * switches (same contract family as chat's mount). `setActiveSessionId` and
 * `newConversation` are the #2222 active-highlight + new-tile surface:
 *   - `setActiveSessionId(sessionId)` overrides the active-row highlight (what
 *     a host calls when the current conversation changes).
 *   - `newConversation()` calls `onNewConversation` (defaulting to
 *     `api.newConversation()`), optimistically prepends a tile for the returned
 *     `session_id`, marks it active, and fires a reconciling `refresh()`.
 *
 * Sidebar-oriented hooks (the standalone conversations pane in identity.js is a
 * consumer as of #2199 — it no longer reimplements fetch/refresh/seq-guard):
 *   - `showViewBar: false` hides the Active/Archived/Trash switcher so an
 *     embedder can drive the view from its own chrome (the sidebar's
 *     `#trash-toggle-btn` calls `setView`).
 *   - `onSelect(conv, { agentName })` — the second arg carries the agent the
 *     list was loaded for, so the consumer can pin the load (#1358).
 *   - `onLoaded(conversations, { view, agentName })` — fires after each
 *     non-stale load; the sidebar uses it for its #714 auto-load-most-recent.
 *   - `onMutated(action, conv)` — fires after a successful archive / unarchive
 *     / trash / purge / restore; the sidebar uses it to coordinate chat state
 *     (start a fresh session when the currently-open conversation is deleted).
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

    // Active-conversation highlight (#2222). `getActiveSessionId` is the host's
    // source of truth, read at render time. `setActiveSessionId` lets a host —
    // or the component's own new-conversation action — override the highlight
    // synchronously, which is what paints a just-created tile active BEFORE any
    // list refetch resolves. A NULLISH id CLEARS the override (defer to the
    // config getter): "no active session" must never be a sticky pin, or a
    // host seeding the highlight while an agent has no current session
    // (identity.js retarget) would mask the session chat.js learns on the
    // first message — the list would never highlight it (codex P2 on #2224).
    let activeIdOverride = null; // nullish = defer to getActiveSessionId()
    function currentActiveId() {
        return activeIdOverride != null ? activeIdOverride : getActiveSessionId();
    }

    let view = config.view || 'active';
    let searchTerm = '';
    // Server-side full-text search overlay. While `searchTerm` is set, the
    // instant client-side name/preview filter paints first; a debounced
    // `GET /api/conversations?q=` then replaces the rows with content-level
    // hits (decrypted + grouped server-side) carrying match snippets.
    // `searchResults === null` means no server response applies to the
    // current term. Trash view keeps the client filter only — trash rows are
    // assembled client-side from /api/trash and have no search endpoint.
    let searchResults = null;
    let searchSeq = 0;
    let searchDebounce = null;
    const SEARCH_DEBOUNCE_MS = 250;
    let agentName = config.agentName;
    let lastConversations = [];
    let refreshSeq = 0;
    // The agent the CURRENTLY-PAINTED rows were rendered for (#2199 P2-3).
    // retarget() flips `agentName` synchronously and then fires an async
    // reload; until that reload repaints, the visible rows still belong to
    // the OLD agent. A kebab Archive/Trash/Purge fired in that window would
    // send the previous agent's session_id against the NEW agent's route.
    // Mutation handlers compare against this to drop such stale actions.
    //
    // Two anchors, one per data source: `renderedForAgent` tracks
    // `lastConversations` (pinned in refresh()'s success block) and
    // `searchForAgent` tracks `searchResults` (pinned in runServerSearch()'s
    // success block). The search overlay and the plain list reload race each
    // other during a retarget; whichever source is painted decides which
    // anchor the stale guard reads — a single anchor would either reject
    // actions on fresh search rows or accept them on stale list rows.
    let renderedForAgent = agentName;
    let searchForAgent = agentName;
    function isStaleRow() {
        const anchor = usingSearchOverlay() ? searchForAgent : renderedForAgent;
        return anchor !== agentName;
    }

    // Whether the server-search overlay currently owns the painted rows.
    function usingSearchOverlay() {
        return !!searchTerm && searchResults !== null && view !== 'trash';
    }

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
    if (config.showViewBar !== false) controls.appendChild(viewBar);

    // Search / filter box.
    const search = document.createElement('input');
    search.type = 'search';
    search.className = 'conversations-search';
    search.placeholder = 'Search conversations…';
    search.dataset.labelAttrPlaceholder = 'conv_search_placeholder';
    if (config.showSearch !== false) {
        search.addEventListener('input', () => {
            searchTerm = String(search.value || '').trim().toLowerCase();
            searchResults = null;
            searchSeq++; // invalidate any in-flight server search
            if (searchDebounce) clearTimeout(searchDebounce);
            renderCurrent(); // instant name/preview filter while the server search runs
            if (searchTerm && view !== 'trash' && typeof api.searchConversations === 'function') {
                searchDebounce = setTimeout(runServerSearch, SEARCH_DEBOUNCE_MS);
            }
        });
        controls.appendChild(search);
    }

    // Debounced content-level search against the server (full text, decrypted
    // and grouped there). Seq-guarded like refresh(): a stale response — an
    // older keystroke, or one that raced a view switch/retarget — must never
    // paint over the rows the current term owns.
    async function runServerSearch() {
        const seq = ++searchSeq;
        const term = searchTerm;
        // The agent this search is issued for. The seq guard drops responses
        // that predate a retarget, so a response that lands here belongs to
        // the agent the API layer was routed to at issue time.
        const forAgent = agentName;
        try {
            const decrypt = state ? state.showDecrypted : true;
            const data = await api.searchConversations(term, view, decrypt);
            if (seq !== searchSeq || term !== searchTerm) return;
            searchResults = data.conversations || [];
            searchForAgent = forAgent;
            renderCurrent();
        } catch (e) {
            // Keep the client-side filtered view; searching-as-you-type must
            // degrade quietly, not toast on every keystroke.
            if (seq === searchSeq) searchResults = null;
        }
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

    function notifyMutated(action, conv) {
        // Every lifecycle mutation removes the row from the CURRENT view
        // (archive/trash/purge leave active; unarchive leaves archived;
        // restore leaves trash), so an active server-search overlay must drop
        // the hit too — otherwise the mutation's refresh() repaints the stale
        // `searchResults` row with actions for a view it no longer belongs to
        // (codex P2). refresh() then re-runs the server search for the
        // authoritative post-mutation hit list.
        if (searchResults && conv && conv.session_id) {
            searchResults = searchResults.filter((c) => c.session_id !== conv.session_id);
        }
        if (typeof config.onMutated === 'function') config.onMutated(action, conv);
    }

    // Drop a mutation whose row belongs to an agent we've since retargeted
    // away from (#2199 P2-3). The API layer is already routed to the NEW host
    // agent, so acting on a stale row's session_id would hit the wrong route.
    function dropIfStale() {
        if (!isStaleRow()) return false;
        Toast.info('Switched agents — reloading conversations');
        return true;
    }

    async function doArchive(conv, rowEl) {
        if (dropIfStale()) return;
        try {
            await api.archiveConversation(conv.session_id);
            animateOut(rowEl);
            Toast.info('Conversation archived');
            notifyMutated('archive', conv);
            refresh();
        } catch (e) { Toast.error(`Failed to archive: ${e.message}`); }
    }
    async function doUnarchive(conv, rowEl) {
        if (dropIfStale()) return;
        try {
            await api.unarchiveConversation(conv.session_id);
            animateOut(rowEl);
            Toast.info('Conversation unarchived');
            notifyMutated('unarchive', conv);
            refresh();
        } catch (e) { Toast.error(`Failed to unarchive: ${e.message}`); }
    }
    async function doTrash(conv, rowEl) {
        if (dropIfStale()) return;
        if (!confirmTrash()) return;
        try {
            await api.deleteConversation(conv.session_id);
            animateOut(rowEl);
            Toast.info('Conversation moved to trash');
            notifyMutated('trash', conv);
            refresh();
        } catch (e) { Toast.error(`Failed to delete: ${e.message}`); }
    }
    async function doPurge(conv, rowEl) {
        if (dropIfStale()) return;
        if (!confirmPurge()) return;
        try {
            // Orphan trash entries (individually-deleted messages with no
            // session, #2199 P2-1) purge at the message level.
            if (conv._trashMessageId) {
                await api.purgeMessage(conv._trashMessageId, 'user-initiated-ui');
            } else {
                await api.purgeConversation(conv.session_id, 'user-initiated-ui');
            }
            animateOut(rowEl);
            Toast.info('Conversation permanently deleted');
            notifyMutated('purge', conv);
            refresh();
        } catch (e) { Toast.error(`Failed to permanently delete: ${e.message}`); }
    }
    async function doRestore(conv, rowEl) {
        if (dropIfStale()) return;
        try {
            if (conv._trashMessageId) {
                await api.restoreMessage(conv._trashMessageId);
            } else {
                await api.restoreConversation(conv.session_id);
            }
            animateOut(rowEl);
            Toast.success('Conversation restored');
            notifyMutated('restore', conv);
            refresh();
        } catch (e) { Toast.error(`Failed to restore: ${e.message}`); }
    }

    function rowOpts() {
        const activeId = currentActiveId();
        // Snapshot the agent at render time so each row's pin is immutable
        // (#1358). retarget() mutates `agentName` synchronously and then fires
        // an async refresh; the old rows stay clickable until the new list
        // resolves. Reading the live `agentName` at click time would let a
        // stale row dispatch under the NEW agent during that fetch window.
        // Capturing here means a stale row always carries the agent it was
        // rendered for, so loadConversation's currentAgentMatches gate drops it.
        const renderAgent = agentName;
        return {
            api,
            group: config.group !== false,
            escapeHtml: config.escapeHtml,
            // Orphan trash rows (#2199 P2-1) have no session to open — their
            // only affordances are the kebab's restore/purge.
            onSelect: (conv) => {
                if (conv._trashMessageId) return;
                onSelect(conv, { agentName: renderAgent });
            },
            buildMenuItems: menuItemsFor,
            // Highlight term for match snippets (server search results only).
            highlightTerm: searchTerm,
            // active flag applied per-row below via renderCurrent
            _activeId: activeId,
        };
    }

    function renderCurrent() {
        // NB: `renderedForAgent` is pinned in refresh()'s seq-guarded success
        // block, NOT here — renderCurrent() also runs on every search keystroke
        // to re-filter `lastConversations` without a reload. Anchoring the stale
        // guard here would let a mid-retarget keystroke advance `renderedForAgent`
        // to the new agent while still painting the OLD agent's rows, defeating
        // the #2199 P2-3 guard. Pin to the agent the *data* belongs to instead.
        // While a term is active and the server search has answered, its
        // content-level hits own the list (they are a superset of the client
        // name/preview filter and carry match snippets). Until then — and in
        // the trash view, which has no server search — the instant client
        // filter paints.
        const convs = usingSearchOverlay() ? searchResults : filtered(lastConversations);
        renderStats(convs);
        if (!convs.length) {
            listEl.innerHTML = '';
            const empty = document.createElement('p');
            empty.className = 'empty-state';
            empty.textContent = searchTerm
                ? 'No matching conversations.'
                : (view === 'trash'
                    ? 'Nothing in trash.'
                    : (view === 'archived' ? 'No archived conversations.' : 'No conversations yet.'));
            listEl.appendChild(empty);
            return;
        }
        const activeId = currentActiveId();
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
            const { sessions, orphans } = groupTrashBySession(data.messages || []);
            const sessionRows = (sessions || []).map((s) => ({
                session_id: s.session_id,
                preview: s.preview,
                message_count: s.count,
                deleted_at: s.deleted_at,
                started_at: s.deleted_at,
            }));
            // Individually-deleted messages (no session_id) still need to be
            // restore/purge-actionable, so surface them as single-message rows
            // rather than dropping them (#2199 P2-1). They carry _trashMessageId
            // so the restore/purge handlers act at the message level.
            const orphanRows = (orphans || []).map((m) => ({
                _trashMessageId: m.id,
                preview: `${m.role || 'msg'}: ${(m.content || '').slice(0, 80) || '(empty)'}`,
                message_count: 1,
                deleted_at: m.deleted_at,
                started_at: m.deleted_at,
            }));
            return [...sessionRows, ...orphanRows]
                .sort((a, b) => (b.deleted_at || '').localeCompare(a.deleted_at || ''));
        }
        const data = await api.getConversations(decrypt, view);
        return data.conversations || [];
    }

    async function refresh() {
        syncViewButtons();
        // Revalidate an active content-search overlay alongside the list
        // reload — a mutation, retarget or view event may have changed which
        // sessions match. Seq-guarded, so whichever response is newest wins.
        if (searchTerm && view !== 'trash' && typeof api.searchConversations === 'function') {
            runServerSearch();
        }
        // Sequence token: a view switch or agent retarget while a previous
        // refresh is still in flight must win — otherwise the older response
        // lands late and renders the wrong rows under the new view (e.g. active
        // rows shown with Archived actions). Same guard identity.js's list had
        // (conversationListRequestSeq, #1358).
        const seq = ++refreshSeq;
        try {
            const data = await loadData();
            if (seq !== refreshSeq) return; // stale — a newer refresh owns the list
            lastConversations = data;
            // Pin the stale-row anchor to the agent this data was loaded for
            // (#2199 P2-3). A pure re-filter repaint (search keystroke) leaves
            // this pointing at the old agent until a retarget's reload lands, so
            // isStaleRow() stays true across the whole in-flight window.
            renderedForAgent = agentName;
            renderCurrent();
            if (typeof config.onLoaded === 'function') {
                config.onLoaded(data, { view, agentName });
            }
        } catch (e) {
            if (seq !== refreshSeq) return;
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
        searchResults = null;
        searchSeq++;
        if (searchDebounce) clearTimeout(searchDebounce);
        refresh();
    }

    function retarget(nextAgentName) {
        agentName = nextAgentName;
        // Routing is handled by the API layer (API.setHostAgent); just reload.
        // A pending/answered server search belongs to the OLD agent — drop it
        // so stale hits never paint post-switch; refresh() re-runs the search
        // for the new agent.
        searchResults = null;
        searchSeq++;
        if (searchDebounce) clearTimeout(searchDebounce);
        refresh();
    }

    // Override the active-conversation highlight and repaint (#2222). Hosts call
    // this whenever the current conversation changes (select, agent switch, new
    // conversation) so the pane's highlight and the host's notion of "current"
    // stay unified — the pane no longer reads a divergent identity.js value.
    function setActiveSessionId(sessionId) {
        activeIdOverride = sessionId;
        renderCurrent();
    }

    // Start a new conversation and surface it immediately (#2222). Calls the
    // canonical API (or a host-supplied `onNewConversation` that also does the
    // host-side chat wiring), then OPTIMISTICALLY prepends a tile for the minted
    // session_id and marks it active — no wait for a list refetch. A reconciling
    // refresh() follows; the server session-marker row keeps the tile present.
    async function newConversation() {
        const startNew = typeof config.onNewConversation === 'function'
            ? config.onNewConversation
            : () => api.newConversation();
        let result;
        try {
            result = await startNew();
        } catch (e) {
            Toast.error(`Failed to start new conversation: ${e.message}`);
            return null;
        }
        const sessionId = result && result.session_id;
        if (!sessionId) return result || null;
        const optimistic = {
            session_id: sessionId,
            preview: 'New conversation',
            started_at: (result && result.started_at) || null,
            message_count: 0,
        };
        lastConversations = [
            optimistic,
            ...lastConversations.filter((c) => c.session_id !== sessionId),
        ];
        activeIdOverride = sessionId;
        renderCurrent();
        // Reconcile with the authoritative server list without blocking the
        // optimistic paint above.
        refresh();
        return result;
    }

    function destroy() {
        if (searchDebounce) clearTimeout(searchDebounce);
        searchSeq++; // orphan any in-flight server search
        closeKebabMenu();
        containerEl.innerHTML = '';
    }

    syncViewButtons();
    if (config.autoLoad !== false) refresh();

    return {
        element: root,
        refresh,
        retarget,
        setView,
        setActiveSessionId,
        newConversation,
        destroy,
        get view() { return view; },
    };
}

// ============================================================================
// mountConversationsPane — the full collapsible pane unit (#2199)
// ============================================================================

// Best-effort localStorage persistence now lives in the shared ui_state.mjs
// module (#2298) — `storeGet`/`storeSet` are imported above. The raw-string
// contract is unchanged, so the pane's on-disk format is byte-for-byte
// identical (no stored state migrates or breaks).

/**
 * Mount the full collapsible conversations PANE — the embeddable list surface
 * (`mountConversations`) PLUS the surrounding pane chrome. This is the ONE pane
 * implementation shared by the standalone console (identity.js) and any embed;
 * a host provides only a container + config and gets the complete unit:
 *
 *   - a `<` chevron that fully HIDES the pane (`display:none`, no leftover rail;
 *     #2216) with `open()` / `close()` / `toggle()` and an `onToggle(collapsed)`
 *     callback so a host toolbar button (the `ki-history` chat-header trigger)
 *     can reopen it;
 *   - a drag-resize handle with min/max width + `localStorage` persistence;
 *   - a disclosure toggle that collapses the search / view-bar / stats block
 *     (default open, persisted);
 *   - collapse/resize/filters state all persisted under `config.storageKey`.
 *
 * Chrome is ADOPT-or-BUILD: when the container already looks like a pane (the
 * console's static `#conversations-pane` with its header / resize handle), those
 * elements are reused; when the container is bare (the embedder contract — a
 * host hands over just a `<div>`), the full chrome is built inside it. No
 * two-pane layout is assumed (a third left pane is planned, #2203).
 *
 * Config (all optional except where a list needs them):
 *   - api, onSelect, onNewConversation, agentName, getActiveSessionId,
 *     onLoaded, onMutated, autoLoad, showViewBar, showSearch, showStats,
 *     group — forwarded verbatim to `mountConversations`.
 *   - collapsed        — initial collapsed state (overridden by persistence).
 *   - storageKey       — persistence namespace (default 'kestrel:conversations-pane').
 *   - title            — pane header title (default 'History').
 *   - onToggle(bool)   — fired after every collapse/expand with the new state.
 *   - minWidth/maxWidth — resize clamps (default 200 / 500, matching the CSS).
 *
 * Returns a handle:
 *   `{ element, conversations, refresh, retarget, setView, view,
 *      open, close, toggle, collapsed, setFiltersOpen, filtersOpen, destroy }`
 * where `conversations` is the inner `mountConversations` handle.
 */
export function mountConversationsPane(containerEl, config = {}) {
    if (!containerEl) throw new Error('mountConversationsPane requires a container element');
    const doc = containerEl.ownerDocument
        || (typeof document !== 'undefined' ? document : null);
    if (!doc) throw new Error('mountConversationsPane requires a document');

    const storageKey = config.storageKey || 'kestrel:conversations-pane';
    const KEY_WIDTH = `${storageKey}:width`;
    const KEY_COLLAPSED = `${storageKey}:collapsed`;
    const KEY_FILTERS = `${storageKey}:filters`;
    const minWidth = Number.isFinite(config.minWidth) ? config.minWidth : 200;
    const maxWidth = Number.isFinite(config.maxWidth) ? config.maxWidth : 500;

    // The container IS the pane element; tag it so it inherits the pane-sidebar
    // chrome CSS whether it was already a pane (adopt) or a bare div (build).
    if (containerEl.classList) {
        containerEl.classList.add('pane-sidebar', 'conversations-pane');
    }
    const paneEl = containerEl;

    // --- Header (adopt existing .pane-header, else build one) ---------------
    let header = paneEl.querySelector('.pane-header');
    let builtHeader = false;
    if (!header) {
        header = doc.createElement('div');
        header.className = 'pane-header';
        const h3 = doc.createElement('h3');
        h3.className = 'conversations-pane-title';
        h3.textContent = config.title || 'History';
        header.appendChild(h3);
        paneEl.insertBefore(header, paneEl.firstChild);
        builtHeader = true;
    }

    // Filters disclosure toggle — lives in the header, hides/shows the
    // search/view-bar/stats block. Default open, state persisted.
    const filtersToggle = doc.createElement('button');
    filtersToggle.type = 'button';
    filtersToggle.className = 'conversations-filters-toggle btn-icon';
    filtersToggle.setAttribute('aria-label', 'Toggle search and filters');
    filtersToggle.title = 'Search & filters';
    filtersToggle.innerHTML = (typeof window !== 'undefined' && typeof window.kicon === 'function')
        ? window.kicon('search')
        : '<span class="ki ki-search" aria-hidden="true"></span>';

    // --- Collapse rail (adopt existing .collapse-btn, else build one) -------
    let collapseBtn = header.querySelector('.collapse-btn');
    if (!collapseBtn) {
        collapseBtn = doc.createElement('button');
        collapseBtn.type = 'button';
        collapseBtn.className = 'collapse-btn';
        collapseBtn.title = 'Collapse';
        collapseBtn.setAttribute('aria-label', 'Collapse conversations pane');
        collapseBtn.innerHTML = (typeof window !== 'undefined' && typeof window.kicon === 'function')
            ? window.kicon('chevron-left')
            : '<span class="ki ki-chevron-left" aria-hidden="true"></span>';
        header.appendChild(collapseBtn);
    }
    // The disclosure toggle sits just before the collapse chevron.
    header.insertBefore(filtersToggle, collapseBtn);

    // --- List body (adopt existing #conversations-list / .pane-content) -----
    let body = paneEl.querySelector('#conversations-list')
        || paneEl.querySelector('.conversations-pane-body');
    if (!body) {
        body = doc.createElement('div');
        body.className = 'pane-content conversations-pane-body';
        // Insert before any existing resize handle so the handle stays last.
        const existingHandle = paneEl.querySelector('.resize-handle');
        if (existingHandle) paneEl.insertBefore(body, existingHandle);
        else paneEl.appendChild(body);
    }

    // --- Resize handle (adopt existing .resize-handle, else build one) ------
    let resizeHandle = paneEl.querySelector('.resize-handle');
    if (!resizeHandle) {
        resizeHandle = doc.createElement('div');
        resizeHandle.className = 'resize-handle conversations-resize-handle';
        paneEl.appendChild(resizeHandle);
    }

    // --- Mount the shared list surface into the body -----------------------
    const listHandle = mountConversations(body, {
        api: config.api,
        onSelect: config.onSelect,
        onNewConversation: config.onNewConversation,
        agentName: config.agentName,
        getActiveSessionId: config.getActiveSessionId,
        onLoaded: config.onLoaded,
        onMutated: config.onMutated,
        autoLoad: config.autoLoad,
        showViewBar: config.showViewBar,
        showSearch: config.showSearch,
        showStats: config.showStats,
        group: config.group,
        escapeHtml: config.escapeHtml,
        view: config.view,
    });

    // --- New-conversation button (adopt existing, else build) --------------
    // #2222: the New button is component-owned so embed hosts — which never run
    // identity.js's DOMContentLoaded wiring — still get it. The standalone
    // console's static `#new-conversation-sidebar-btn` is adopted in place; a
    // built header gets a fresh `ki-plus` button. Same adopt-or-build pattern as
    // the collapse chevron. Clicking it drives the component's own
    // new-conversation action (optimistic tile + active highlight).
    let newBtn = header.querySelector('#new-conversation-sidebar-btn')
        || header.querySelector('.new-conversation-btn');
    let builtNewBtn = false;
    if (!newBtn) {
        newBtn = doc.createElement('button');
        newBtn.type = 'button';
        newBtn.className = 'new-conversation-btn btn-icon';
        newBtn.title = 'New Conversation';
        newBtn.setAttribute('aria-label', 'New conversation');
        newBtn.innerHTML = (typeof window !== 'undefined' && typeof window.kicon === 'function')
            ? window.kicon('plus')
            : '<span class="ki ki-plus" aria-hidden="true"></span>';
        // Sit just after the title, before the filters/collapse cluster.
        const titleEl = header.querySelector('.conversations-pane-title')
            || header.querySelector('h3');
        if (titleEl && titleEl.nextSibling) header.insertBefore(newBtn, titleEl.nextSibling);
        else header.insertBefore(newBtn, header.firstChild);
        builtNewBtn = true;
    }
    const onNewClick = () => { listHandle.newConversation(); };
    newBtn.addEventListener('click', onNewClick);

    // The controls (view bar + search) and stats blocks the disclosure governs.
    const controlsEl = listHandle.element.querySelector('.conversations-controls');
    const statsEl = listHandle.element.querySelector('.conversations-stats');

    // ---- Collapse state ---------------------------------------------------
    // #2216: the pane has exactly TWO states — open (full pane) and fully
    // HIDDEN (`display:none`, zero width — NO collapsed chevron rail). close()
    // hides the pane entirely; open() shows it. The `.collapsed` class is kept
    // in lock-step with `display:none` as the single state marker so every
    // reader (identity.js, tests, the KEY_COLLAPSED persistence — '1' = closed)
    // sees one consistent representation. The chevron CLOSES; the chat-header
    // history trigger reopens.
    function isCollapsed() {
        return !!(paneEl.classList && paneEl.classList.contains('collapsed'));
    }
    function applyCollapsed(collapsed, persist) {
        if (!paneEl.classList) return;
        paneEl.classList.toggle('collapsed', collapsed);
        // Fully hide when closed; restore CSS-driven display when open. Inline
        // display is the load-bearing hide (CSS `.conversations-pane.collapsed`
        // backs it up); clearing it lets `.pane-sidebar { display:flex }` apply.
        paneEl.style.display = collapsed ? 'none' : '';
        if (persist) storeSet(KEY_COLLAPSED, collapsed ? '1' : '0');
        if (typeof config.onToggle === 'function') config.onToggle(collapsed);
    }
    function open() { if (isCollapsed()) applyCollapsed(false, true); }
    function close() { if (!isCollapsed()) applyCollapsed(true, true); }
    function toggle() { applyCollapsed(!isCollapsed(), true); }

    // The chevron `<` CLOSES the pane (fully hides it) — it never reopens; the
    // chat-header history trigger owns reopening (#2216).
    const onCollapseClick = () => close();
    collapseBtn.addEventListener('click', onCollapseClick);

    // Initial state: a persisted value wins; otherwise the first-run default is
    // CLOSED/hidden (#2216 — an explicit `config.collapsed: false` can still
    // start it open). Always applied so the pane's `display` reflects the state
    // from mount, whichever branch wins.
    const persistedCollapsed = storeGet(KEY_COLLAPSED);
    const startCollapsed = persistedCollapsed !== null
        ? persistedCollapsed === '1'
        : (config.collapsed !== undefined ? !!config.collapsed : true);
    applyCollapsed(startCollapsed, false);

    // ---- Filters disclosure ----------------------------------------------
    function applyFilters(open_, persist) {
        if (controlsEl) controlsEl.style.display = open_ ? '' : 'none';
        if (statsEl) statsEl.style.display = open_ ? '' : 'none';
        filtersToggle.classList.toggle('active', open_);
        filtersToggle.setAttribute('aria-expanded', open_ ? 'true' : 'false');
        if (persist) storeSet(KEY_FILTERS, open_ ? '1' : '0');
    }
    let filtersOpen = storeGet(KEY_FILTERS);
    filtersOpen = filtersOpen !== null ? filtersOpen === '1' : true; // default open
    applyFilters(filtersOpen, false);
    const onFiltersClick = () => {
        filtersOpen = !filtersOpen;
        applyFilters(filtersOpen, true);
    };
    filtersToggle.addEventListener('click', onFiltersClick);

    // ---- Resize (min/max + persistence) -----------------------------------
    // Restore a persisted width (clamped) before wiring the drag.
    const persistedWidth = parseInt(storeGet(KEY_WIDTH), 10);
    if (Number.isFinite(persistedWidth)) {
        paneEl.style.width = `${Math.max(minWidth, Math.min(maxWidth, persistedWidth))}px`;
    }
    let startX = 0;
    let startWidth = 0;
    function onMouseMove(e) {
        const diff = e.clientX - startX;
        const w = Math.max(minWidth, Math.min(maxWidth, startWidth + diff));
        paneEl.style.width = `${w}px`;
    }
    function onMouseUp() {
        doc.removeEventListener('mousemove', onMouseMove);
        doc.removeEventListener('mouseup', onMouseUp);
        if (doc.body) {
            doc.body.style.cursor = '';
            doc.body.style.userSelect = '';
        }
        storeSet(KEY_WIDTH, String(paneEl.offsetWidth || parseInt(paneEl.style.width, 10) || startWidth));
    }
    function onResizeDown(e) {
        startX = e.clientX;
        startWidth = paneEl.offsetWidth || parseInt(paneEl.style.width, 10) || minWidth;
        if (doc.body) {
            doc.body.style.cursor = 'col-resize';
            doc.body.style.userSelect = 'none';
        }
        doc.addEventListener('mousemove', onMouseMove);
        doc.addEventListener('mouseup', onMouseUp);
    }
    resizeHandle.addEventListener('mousedown', onResizeDown);

    function destroy() {
        collapseBtn.removeEventListener('click', onCollapseClick);
        filtersToggle.removeEventListener('click', onFiltersClick);
        newBtn.removeEventListener('click', onNewClick);
        resizeHandle.removeEventListener('mousedown', onResizeDown);
        doc.removeEventListener('mousemove', onMouseMove);
        doc.removeEventListener('mouseup', onMouseUp);
        try { listHandle.destroy(); } catch (_) { /* best-effort */ }
        // Remove only chrome this mount built; adopted chrome is left in place.
        if (builtHeader && header.parentNode) header.parentNode.removeChild(header);
        if (builtNewBtn) newBtn.remove();
        filtersToggle.remove();
    }

    return {
        element: paneEl,
        conversations: listHandle,
        refresh: (...a) => listHandle.refresh(...a),
        retarget: (...a) => listHandle.retarget(...a),
        setView: (...a) => listHandle.setView(...a),
        setActiveSessionId: (...a) => listHandle.setActiveSessionId(...a),
        newConversation: (...a) => listHandle.newConversation(...a),
        get view() { return listHandle.view; },
        open,
        close,
        toggle,
        get collapsed() { return isCollapsed(); },
        setFiltersOpen(next) { filtersOpen = !!next; applyFilters(filtersOpen, true); },
        get filtersOpen() { return filtersOpen; },
        destroy,
    };
}
