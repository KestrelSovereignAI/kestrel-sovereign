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
        lastPayload: null,
        async listAgents() {
            const data = await api.getAgents();
            adapter.lastPayload = data;
            adapter.mode = data.mode === 'standalone' ? 'standalone' : 'multi_agent';
            adapter.serverDemoMode = data.server_demo_mode === true;
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
