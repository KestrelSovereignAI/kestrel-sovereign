/**
 * Kestrel Sovereign Console - Agent List Component (#2278 / design #2166)
 *
 * ONE agent/companion list surface, in the SAME contract family as chat's
 * `mount()` and the conversations module (`mountConversations` /
 * `mountConversationsPane`, #2149 / #2199 / #2216 / #2222). Before this module
 * the standalone console hand-rolled the agent loop inside identity.js's
 * `loadAgents`, and every embedding host (Frinz's companion list) maintained a
 * bespoke copy. This module owns the shared list logic — fetch-via-adapter,
 * the owned card shell, selection routing, the per-card `agent-card-actions`
 * slot, refresh, and the active-selection highlight — while card RENDERING is a
 * config hook so a host can supply its own skin.
 *
 * The division is exactly the conversations pane's owned-rows-vs-host-callbacks
 * split: the component owns chrome + orchestration; presentation is pluggable.
 *
 *   - `mountAgentList(containerEl, config)` — the embeddable LIST surface.
 *   - `createDefaultAgentAdapter(api)` — the default `/api/agents` adapter for
 *     the standalone console (avatar via `avatar_hash` → `/api/files/<hash>`).
 *
 * The machine-readable contract these build to is
 * `docs/proposals/agent-list-component/contract.js` (AgentListItem,
 * AgentListAdapter, AgentCardRenderer, AgentListConfig, AgentListHandle).
 */

import API from './api.js';
import { escapeHtml as sharedEscapeHtml } from './ui.js';
import { UI } from './ui-ext/registry.js';
import { storeGet, storeSet } from './ui_state.mjs';

// ============================================================================
// Default adapter — the standalone console's `/api/agents` data source
// ============================================================================

/**
 * The default {@link AgentListAdapter} over `API.getAgents()` (design question
 * 1: standalone reads `/api/agents`). It normalizes each agent-card record onto
 * the {@link AgentListItem} shape and resolves the portrait URL from
 * `avatar_hash` → `/api/files/<hash>` (the component never builds an avatar URL
 * itself). `mode` / `serverDemoMode` mirror the response fields so a host can
 * read them back after `listAgents()` resolves (identity.js drives its
 * demo-banner + misconfig-gated auto-select off them).
 */
export function createDefaultAgentAdapter(api = API) {
    const adapter = {
        mode: 'multi_agent',
        serverDemoMode: false,
        // False until a /api/agents payload has actually been parsed —
        // consumers gating SAFETY decisions (demo rail, create-agent flow)
        // must fail CLOSED while this is false rather than trust the
        // defaults above (codex P1 on #2358).
        classificationLoaded: false,
        lastPayload: null,
        async listAgents() {
            const data = await api.getAgents();
            adapter.lastPayload = data;
            adapter.mode = data.mode === 'standalone' ? 'standalone' : 'multi_agent';
            adapter.serverDemoMode = data.server_demo_mode === true;
            adapter.classificationLoaded = true;
            const agents = data.agents || [];
            return agents.map((a) => ({
                name: a.name,
                id: a.id || a.did || a.name,
                displayName: a.name,
                description: a.description,
                avatarUrl: a.avatar_hash ? `/api/files/${a.avatar_hash}` : undefined,
                status: a.status,
                isDemo: a.is_demo === true,
                raw: a,
            }));
        },
    };
    return adapter;
}

// ============================================================================
// Default card renderer — the CONSOLE ROW (matches today's `.agent-item`)
// ============================================================================

// Build the default console-row body: the per-agent thinking pulse, the
// name/description block, and the stop control — the exact affordance set the
// standalone console shipped in identity.js. Returns a DocumentFragment so the
// children become DIRECT children of the `.agent-item` flex row (a wrapping
// div would break the row layout), letting the component prepend the
// component-owned status dot and append the actions anchor around it. The
// status dot is component-owned (§3.2), so this renderer does NOT draw it.
function makeConsoleRenderer({ onStop }) {
    return (item) => {
        const doc = typeof document !== 'undefined' ? document : null;
        const frag = doc.createDocumentFragment();
        const name = item.displayName || item.name || 'Unnamed Agent';

        const thinkingDot = doc.createElement('span');
        thinkingDot.className = 'agent-thinking-dot';
        thinkingDot.title = `${item.name || 'Agent'} is thinking`;
        frag.appendChild(thinkingDot);

        const info = doc.createElement('div');
        info.className = 'agent-info';
        const nameEl = doc.createElement('div');
        nameEl.className = 'agent-name';
        nameEl.textContent = name;
        const desc = doc.createElement('div');
        desc.className = 'agent-description';
        desc.textContent = item.description || 'No description';
        info.appendChild(nameEl);
        info.appendChild(desc);
        frag.appendChild(info);

        // Per-agent stop control. Rendered always but only VISIBLE while the
        // row carries `.agent-thinking` (CSS gate); a click aborts that exact
        // agent's stream via the host `onStop` hook. stopPropagation so it does
        // not also fire the row's selection handler.
        const stopBtn = doc.createElement('button');
        stopBtn.className = 'agent-stop-btn';
        stopBtn.title = `Stop ${item.name || 'agent'}`;
        stopBtn.setAttribute('aria-label', `Stop ${item.name || 'agent'}`);
        stopBtn.innerHTML = '&times;';
        stopBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (typeof onStop === 'function') onStop(item.name);
        });
        frag.appendChild(stopBtn);

        return frag;
    };
}

// ============================================================================
// mountAgentList — the embeddable list surface
// ============================================================================

/**
 * Mount the agent/companion list surface into `containerEl`. Returns a handle:
 * `{ element, refresh, select, setActiveName, getActive, destroy }`.
 *
 * The component NEVER fetches directly — it calls `config.adapter.listAgents()`
 * (defaulting to the `/api/agents` adapter). Selecting a card drives the shared
 * host-agent selection path — `API.setHostAgent(name)` in multi-agent mode ONLY
 * (standalone must not install a route prefix; see the identity.js `loadAgents`
 * note) — then invokes the host `onSelect`. Card RENDERING is `config.renderCard`
 * (default = the console row); the component owns the outer shell, the status
 * dot, and the per-card `agent-card-actions` slot anchor.
 */
export function mountAgentList(containerEl, config = {}) {
    if (!containerEl) throw new Error('mountAgentList requires a container element');
    const doc = containerEl.ownerDocument
        || (typeof document !== 'undefined' ? document : null);
    if (!doc) throw new Error('mountAgentList requires a document');

    const api = config.api || API;
    const adapter = config.adapter || createDefaultAgentAdapter(api);
    const escapeFn = typeof config.escapeHtml === 'function' ? config.escapeHtml : sharedEscapeHtml;
    const hostRenderCard = typeof config.renderCard === 'function' ? config.renderCard : null;
    const usingDefaultRenderer = !hostRenderCard;
    const showStatusDot = config.showStatusDot !== false; // default true = console behavior
    const isThinking = typeof config.isThinking === 'function' ? config.isThinking : () => false;
    const onStop = typeof config.onStop === 'function' ? config.onStop : null;
    const onSelect = typeof config.onSelect === 'function'
        ? config.onSelect
        : (adapter && typeof adapter.onSelect === 'function' ? adapter.onSelect : null);
    const renderCard = hostRenderCard || makeConsoleRenderer({ onStop });

    let items = [];
    let activeName = config.selectedName || null;
    let refreshSeq = 0;

    if (containerEl.classList) containerEl.classList.add('agent-list-component');
    const root = doc.createElement('div');
    root.className = 'agent-list-root';
    containerEl.innerHTML = '';
    containerEl.appendChild(root);

    function mode() { return (adapter && adapter.mode) || 'multi_agent'; }
    function isStandalone() { return mode() === 'standalone'; }
    function findItem(name) { return items.find((i) => i && i.name === name) || null; }

    // Portrait/signed-URL hosts can mint the avatar URL per render via an adapter
    // `avatarUrl(item)` override; otherwise the adapter precomputed it in
    // `listAgents`. The component never builds an avatar URL itself.
    function resolveAvatar(item) {
        if (adapter && typeof adapter.avatarUrl === 'function') {
            try { return adapter.avatarUrl(item); } catch (_) { /* fall through */ }
        }
        return item ? item.avatarUrl : undefined;
    }

    function buildCard(item) {
        const selected = item.name != null && item.name === activeName;
        const offline = item.status === 'offline';
        const thinking = !!isThinking(item.name);

        const shell = doc.createElement('div');
        const classes = [];
        // The default renderer IS the console row, so tag the shell `.agent-item`
        // so identity.js/chat.js selectors (refreshAgentThinkingDot's
        // `.agent-item[data-agent-name]`, the CSS state rules) keep matching. A
        // host renderer gets a clean `.agent-card` shell without the row layout.
        if (usingDefaultRenderer) classes.push('agent-item');
        classes.push('agent-card');
        if (selected) classes.push('selected');
        if (offline) classes.push('offline');
        if (thinking) classes.push('agent-thinking');
        shell.className = classes.join(' ');
        // Always carry the real agent name — thinking-dot / stop lookups depend
        // on it in every mode (see chat.js refreshAgentThinkingDot).
        shell.dataset.agentName = item.name || '';

        // The per-card `agent-card-actions` slot anchor (design question 3),
        // component-created and handed to the renderer as `ctx.actionsAnchor`.
        const actionsAnchor = doc.createElement('div');
        actionsAnchor.dataset.slot = 'agent-card-actions';
        actionsAnchor.className = 'agent-card-actions';

        // Component-owned status dot — a config flag (`showStatusDot`, default
        // true = console behavior); a host renderCard may omit it entirely.
        let statusDot = null;
        if (showStatusDot) {
            statusDot = doc.createElement('span');
            statusDot.className = `agent-status-dot ${offline ? 'offline' : 'online'}`;
        }

        // Selection: only online agents in multi-agent mode are clickable —
        // standalone must NOT install a host-agent prefix (it 404s the
        // un-prefixed routes; the identity.js note at loadAgents), and offline
        // agents are not selectable. `select()` is still available programmatically.
        const selectable = !offline && !isStandalone();
        if (selectable) {
            shell.addEventListener('click', () => select(item.name));
        }

        if (adapter && typeof adapter.avatarUrl === 'function') {
            item.avatarUrl = resolveAvatar(item);
        }

        const ctx = {
            selected,
            standalone: isStandalone(),
            escapeHtml: escapeFn,
            actionsAnchor,
        };
        const body = renderCard(item, ctx);

        // Component owns the shell layout: status dot first, then the renderer's
        // body, then the actions anchor — UNLESS a host renderer already placed
        // the anchor inside its own layout (portrait cards position it under the
        // portrait), in which case the component leaves it where the host put it.
        if (statusDot) shell.appendChild(statusDot);
        if (body) shell.appendChild(body);
        if (!actionsAnchor.parentNode) shell.appendChild(actionsAnchor);

        // NOTE: the `agent-card-actions` slot is rendered by `renderList` AFTER
        // this shell is appended to the live `root`, NOT here. Slot code
        // (e.g. voice's mountAgentVoiceControls → refreshAgentVoiceCard) locates
        // its row via `document.querySelector`, which returns null against a
        // detached card and silently skips the initial state paint. Mounting the
        // slot only once the card is in the document restores the pre-#2278
        // ordering (card live → slot mounts).
        return { shell, actionsAnchor, item };
    }

    // Render the per-card actions slot INTO the anchor via the existing ui-ext
    // registry (unchanged from identity.js today). MUST run only after the card
    // is attached to the live DOM so slot code that does `document.querySelector`
    // for its row resolves. `standalone` comes from `adapter.mode`; `agentName`
    // is the item's name (the voice session key). The registry tears the anchor's
    // contributions down on the next list rebuild.
    function mountActionsSlot(actionsAnchor, item) {
        try {
            UI.renderSlot('agent-card-actions', {
                element: actionsAnchor,
                api,
                agentName: item.name,
                standalone: isStandalone(),
            });
        } catch (_) { /* a missing/misbehaving slot must not break the row */ }
    }

    function renderList() {
        root.innerHTML = '';
        if (!items.length) {
            const empty = doc.createElement('p');
            empty.className = 'agent-list-empty empty-state';
            empty.textContent = config.emptyText || 'No agents available';
            root.appendChild(empty);
            return;
        }
        // Append every card to the live DOM FIRST, then mount each card's actions
        // slot — so slot code that queries `document` for its row resolves.
        for (const item of items) {
            const { shell, actionsAnchor } = buildCard(item);
            root.appendChild(shell);
            mountActionsSlot(actionsAnchor, item);
        }
    }

    // Repaint the active-selection highlight only — no rebuild, so per-card slot
    // anchors and listeners survive (mirrors the conversations pane's
    // setActiveSessionId, #2222).
    function renderHighlight() {
        const cards = root.querySelectorAll('.agent-card');
        cards.forEach((c) => {
            c.classList.toggle('selected', c.dataset.agentName === activeName);
        });
    }

    // In multi-agent mode, select the first online agent when none is selected
    // (matches identity.js). Gated by `autoSelectFirst`; the standalone console
    // keeps its own demo-misconfig gate host-side (via `onLoaded`).
    function maybeAutoSelect() {
        if (!config.autoSelectFirst) return;
        if (isStandalone()) return;
        if (activeName) return;
        const firstOnline = items.find((i) => i && i.status !== 'offline');
        if (firstOnline) select(firstOnline.name);
    }

    async function refresh() {
        // Seq-guard like the conversations list so a stale response never wins.
        const seq = ++refreshSeq;
        try {
            const next = await adapter.listAgents();
            if (seq !== refreshSeq) return;
            items = Array.isArray(next) ? next : [];
            renderList();
            if (typeof config.onLoaded === 'function') {
                config.onLoaded(items, { mode: mode() });
            }
            maybeAutoSelect();
        } catch (e) {
            if (seq !== refreshSeq) return;
            root.innerHTML = '';
            const err = doc.createElement('p');
            err.className = 'agent-list-error';
            err.textContent = config.errorText || 'Failed to load agents';
            root.appendChild(err);
            if (typeof config.onError === 'function') config.onError(e);
        }
    }

    // Drive the shared host-agent selection path: pin routing
    // (`API.setHostAgent`) in multi-agent mode ONLY, repaint the active
    // highlight, then invoke the host `onSelect` (chat mount / product state).
    function select(name) {
        activeName = name;
        if (!isStandalone() && api && typeof api.setHostAgent === 'function') {
            api.setHostAgent(name);
        }
        renderHighlight();
        if (typeof onSelect === 'function') {
            onSelect(findItem(name) || { name }, { standalone: isStandalone() });
        }
    }

    // Override the active highlight WITHOUT firing selection (host reconciling
    // its own notion of current agent).
    function setActiveName(name) {
        activeName = name;
        renderHighlight();
    }

    function destroy() {
        refreshSeq++; // orphan any in-flight refresh
        containerEl.innerHTML = '';
    }

    if (config.autoLoad !== false) refresh();

    return {
        element: root,
        refresh,
        select,
        setActiveName,
        getActive: () => findItem(activeName),
        destroy,
    };
}

// ============================================================================
// mountAgentListPane — the full collapsible pane unit (design #2166 §4)
// ============================================================================

// Best-effort localStorage persistence now lives in the shared ui_state.mjs
// module (#2298) — `storeGet`/`storeSet` are imported above. The raw-string
// contract is unchanged, so the pane's collapse/width on-disk format is
// byte-for-byte identical (no stored state migrates or breaks).

/**
 * Mount the full collapsible agent/companion PANE — the embeddable list surface
 * (`mountAgentList`) PLUS the surrounding pane chrome. This is the agent
 * analogue of `mountConversationsPane`, sharing the SAME chrome contract (#2199
 * / #2216) so the standalone console and any embed consume ONE pane
 * implementation; a host provides only a container + config and gets:
 *
 *   - a `<` chevron that fully HIDES the pane (`display:none`, no leftover rail;
 *     #2216) with `open()` / `close()` / `toggle()` and an `onToggle(collapsed)`
 *     callback so a host toolbar trigger can reopen it;
 *   - a drag-resize handle with min/max width + `localStorage` persistence
 *     (width under `:width`, collapsed state under `:collapsed`), guarded to a
 *     no-op when localStorage is unavailable;
 *   - a component-owned "+ New" header action wired via `config.onNew` (Frinz →
 *     Add-a-Companion; standalone console → new-agent / spawn flow, an
 *     affordance it gains here, same-everywhere rule).
 *
 * Chrome is ADOPT-or-BUILD: when the container already looks like a pane (the
 * console's static `#agents-pane` with its `.pane-header` / `.resize-handle`),
 * those elements are reused; when the container is bare (the embedder contract —
 * a host hands over just a `<div>`), the full chrome is built inside it.
 * `destroy()` removes only chrome this mount built; adopted chrome is left.
 *
 * Config (all optional except where the list needs them):
 *   - api, adapter, renderCard, showStatusDot, isThinking, onStop, onSelect,
 *     onLoaded, onError, autoLoad, autoSelectFirst, selectedName, escapeHtml,
 *     emptyText, errorText — forwarded verbatim to `mountAgentList`.
 *   - onNew()          — the "+ New" header action (Add-a-Companion / new agent).
 *                        The New button is only built/adopted when this is a fn.
 *   - newLabel         — accessible label / tooltip for the New button.
 *   - collapsed        — initial collapsed state (overridden by persistence).
 *   - storageKey       — persistence namespace (default 'kestrel:agents-pane').
 *   - title            — pane header title (default 'Agents').
 *   - onToggle(bool)   — fired after every collapse/expand with the new state.
 *   - minWidth/maxWidth — resize clamps (default 200 / 500, matching the CSS).
 *
 * Returns a handle:
 *   `{ element, list, refresh, select, setActiveName, getActive,
 *      open, close, toggle, collapsed, destroy }`
 * where `list` is the inner `mountAgentList` handle.
 */
export function mountAgentListPane(containerEl, config = {}) {
    if (!containerEl) throw new Error('mountAgentListPane requires a container element');
    const doc = containerEl.ownerDocument
        || (typeof document !== 'undefined' ? document : null);
    if (!doc) throw new Error('mountAgentListPane requires a document');

    const storageKey = config.storageKey || 'kestrel:agents-pane';
    const KEY_WIDTH = `${storageKey}:width`;
    const KEY_COLLAPSED = `${storageKey}:collapsed`;
    const minWidth = Number.isFinite(config.minWidth) ? config.minWidth : 200;
    const maxWidth = Number.isFinite(config.maxWidth) ? config.maxWidth : 500;

    // The container IS the pane element; tag it so it inherits the pane-sidebar
    // chrome CSS whether it was already a pane (adopt) or a bare div (build).
    if (containerEl.classList) {
        containerEl.classList.add('pane-sidebar', 'agent-list-pane');
    }
    const paneEl = containerEl;

    // --- Header (adopt existing .pane-header, else build one) ---------------
    let header = paneEl.querySelector('.pane-header');
    let builtHeader = false;
    if (!header) {
        header = doc.createElement('div');
        header.className = 'pane-header';
        const h3 = doc.createElement('h3');
        h3.className = 'agent-list-pane-title';
        h3.textContent = config.title || 'Agents';
        header.appendChild(h3);
        paneEl.insertBefore(header, paneEl.firstChild);
        builtHeader = true;
    }

    // --- Collapse rail (adopt existing .collapse-btn, else build one) -------
    let collapseBtn = header.querySelector('.collapse-btn');
    if (!collapseBtn) {
        collapseBtn = doc.createElement('button');
        collapseBtn.type = 'button';
        collapseBtn.className = 'collapse-btn';
        collapseBtn.title = 'Collapse';
        collapseBtn.setAttribute('aria-label', 'Collapse agents pane');
        collapseBtn.innerHTML = (typeof window !== 'undefined' && typeof window.kicon === 'function')
            ? window.kicon('chevron-left')
            : '<span class="ki ki-chevron-left" aria-hidden="true"></span>';
        header.appendChild(collapseBtn);
    }

    // --- List body (adopt existing #agents-list / .pane-content) -----------
    let body = paneEl.querySelector('#agents-list')
        || paneEl.querySelector('.agent-list-pane-body');
    if (!body) {
        body = doc.createElement('div');
        body.className = 'pane-content agent-list-pane-body';
        // Insert before any existing resize handle so the handle stays last.
        const existingHandle = paneEl.querySelector('.resize-handle');
        if (existingHandle) paneEl.insertBefore(body, existingHandle);
        else paneEl.appendChild(body);
    }

    // --- Resize handle (adopt existing .resize-handle, else build one) ------
    let resizeHandle = paneEl.querySelector('.resize-handle');
    let builtResizeHandle = false;
    if (!resizeHandle) {
        resizeHandle = doc.createElement('div');
        resizeHandle.className = 'resize-handle agent-list-resize-handle';
        paneEl.appendChild(resizeHandle);
        builtResizeHandle = true;
    }

    // --- Mount the shared list surface into the body -----------------------
    const listHandle = mountAgentList(body, {
        api: config.api,
        adapter: config.adapter,
        renderCard: config.renderCard,
        showStatusDot: config.showStatusDot,
        isThinking: config.isThinking,
        onStop: config.onStop,
        onSelect: config.onSelect,
        onLoaded: config.onLoaded,
        onError: config.onError,
        autoLoad: config.autoLoad,
        autoSelectFirst: config.autoSelectFirst,
        selectedName: config.selectedName,
        escapeHtml: config.escapeHtml,
        emptyText: config.emptyText,
        errorText: config.errorText,
    });

    // --- "+ New" header action (adopt existing, else build) ----------------
    // Component-owned so embed hosts — which never run the console's
    // DOMContentLoaded wiring — still get it, and so the standalone console
    // GAINS a new-agent affordance it lacks today (same-everywhere rule). The
    // action is entirely host-defined via `onNew`, so the button is only
    // built/adopted when `onNew` is a function. The console's static
    // `#new-agent-sidebar-btn` (if present) is adopted in place; otherwise a
    // fresh `ki-plus` button is built. Same adopt-or-build pattern as the
    // conversations pane's New button.
    const hasNew = typeof config.onNew === 'function';
    let newBtn = null;
    let builtNewBtn = false;
    let onNewClick = null;
    if (hasNew) {
        newBtn = header.querySelector('#new-agent-sidebar-btn')
            || header.querySelector('.new-agent-btn');
        if (!newBtn) {
            newBtn = doc.createElement('button');
            newBtn.type = 'button';
            newBtn.className = 'new-agent-btn btn-icon';
            newBtn.title = config.newLabel || 'New Agent';
            newBtn.setAttribute('aria-label', config.newLabel || 'New agent');
            newBtn.innerHTML = (typeof window !== 'undefined' && typeof window.kicon === 'function')
                ? window.kicon('plus')
                : '<span class="ki ki-plus" aria-hidden="true"></span>';
            // Sit just after the title, before the collapse chevron.
            const titleEl = header.querySelector('.agent-list-pane-title')
                || header.querySelector('h3');
            if (titleEl && titleEl.nextSibling) header.insertBefore(newBtn, titleEl.nextSibling);
            else header.insertBefore(newBtn, header.firstChild);
            builtNewBtn = true;
        }
        onNewClick = () => { config.onNew(); };
        newBtn.addEventListener('click', onNewClick);
    }

    // ---- Collapse state ---------------------------------------------------
    // #2216 two-state: open (full pane) and fully HIDDEN (`display:none`, zero
    // width — NO collapsed chevron rail). The `.collapsed` class is kept in
    // lock-step with `display:none` as the single state marker. Unlike the
    // conversations pane, the agents pane is the primary navigation surface, so
    // its first-run default is OPEN (a persisted value or explicit
    // `config.collapsed` still wins).
    function isCollapsed() {
        return !!(paneEl.classList && paneEl.classList.contains('collapsed'));
    }
    function applyCollapsed(collapsed, persist) {
        if (!paneEl.classList) return;
        paneEl.classList.toggle('collapsed', collapsed);
        paneEl.style.display = collapsed ? 'none' : '';
        if (persist) storeSet(KEY_COLLAPSED, collapsed ? '1' : '0');
        if (typeof config.onToggle === 'function') config.onToggle(collapsed);
    }
    function open() { if (isCollapsed()) applyCollapsed(false, true); }
    function close() { if (!isCollapsed()) applyCollapsed(true, true); }
    function toggle() { applyCollapsed(!isCollapsed(), true); }

    // The chevron `<` CLOSES the pane (fully hides it); a host trigger reopens.
    const onCollapseClick = () => close();
    collapseBtn.addEventListener('click', onCollapseClick);

    // Initial state: a persisted value wins; otherwise `config.collapsed`;
    // otherwise OPEN. Always applied so the pane's `display` reflects the state
    // from mount, whichever branch wins.
    const persistedCollapsed = storeGet(KEY_COLLAPSED);
    const startCollapsed = persistedCollapsed !== null
        ? persistedCollapsed === '1'
        : (config.collapsed !== undefined ? !!config.collapsed : false);
    applyCollapsed(startCollapsed, false);

    // ---- Resize (min/max + persistence) -----------------------------------
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
        if (newBtn && onNewClick) newBtn.removeEventListener('click', onNewClick);
        resizeHandle.removeEventListener('mousedown', onResizeDown);
        doc.removeEventListener('mousemove', onMouseMove);
        doc.removeEventListener('mouseup', onMouseUp);
        try { listHandle.destroy(); } catch (_) { /* best-effort */ }
        // Remove only chrome this mount built; adopted chrome is left in place.
        if (builtHeader && header.parentNode) header.parentNode.removeChild(header);
        if (builtNewBtn && newBtn) newBtn.remove();
        // The built resize handle too (codex P2): a leaked absolutely-positioned
        // .resize-handle overlays the container edge and gets ADOPTED by the
        // next mount into the same container, doubling listeners over time.
        if (builtResizeHandle && resizeHandle) resizeHandle.remove();
    }

    return {
        element: paneEl,
        list: listHandle,
        refresh: (...a) => listHandle.refresh(...a),
        select: (...a) => listHandle.select(...a),
        setActiveName: (...a) => listHandle.setActiveName(...a),
        getActive: (...a) => listHandle.getActive(...a),
        open,
        close,
        toggle,
        get collapsed() { return isCollapsed(); },
        destroy,
    };
}
