// ============================================================================
// Mountable agent-list component — interface stubs (design ticket #2166)
// ============================================================================
//
// This file is the authoritative, machine-readable companion to
// `docs/proposals/agent-list-component/DESIGN.md`. It contains JSDoc typedefs
// ONLY — no runtime logic, no DOM, no exports of behavior. The implementation
// ticket(s) that follow build `mountAgentList` / `mountAgentListPane` against
// these types, exactly as the UI-extension-slot registry (ticket 02 of epic
// #2038) was built against `static/js/ui-ext/contract.js`.
//
// Nothing here runs. Importing this module has no side effects.
//
// Stability: the names and shapes below are a CONTRACT. Downstream tickets
// implement against them and Frinz migrates onto them; changing a typedef is
// an API change, not a tidy-up.
//
// Precedent this deliberately mirrors:
//   - `mountConversations(containerEl, config)` / `mountConversationsPane(...)`
//     (`static/js/conversations.js`, #2149 / #2199 / #2216 / #2222) — the
//     list-surface + collapsible-pane split, adopt-or-build chrome, owned
//     rows vs host callbacks. The agent list is the SAME contract family for
//     the agent/companion surface.
//   - `AgentCardActionsContext` (`static/js/ui-ext/contract.js`) — the
//     per-card `agent-card-actions` slot this list hosts unchanged.
// ============================================================================

// ----------------------------------------------------------------------------
// Data source — the host adapter (design question 1)
// ----------------------------------------------------------------------------

/**
 * A single agent/companion as the list consumes it. This is the NORMALIZED
 * shape every adapter produces; it is intentionally a subset of the standalone
 * `/api/agents` agent-card (see `endpoints/models.py get_agents`) so a host
 * adapter (Frinz's `/api/companions`) can map its own records onto it without
 * the component reaching into host-specific fields.
 *
 * The component reads ONLY these keys. `raw` carries the untouched source
 * record so a host `renderCard` can read product-specific fields the normalized
 * shape omits (Frinz mood, relationship tier, …) without widening this type.
 *
 * @typedef {Object} AgentListItem
 * @property {string}  name         - stable identity + selection key. This is
 *           the value passed to `API.setHostAgent(name)` and carried as the
 *           `agentName` of the per-card `agent-card-actions` slot context.
 * @property {string} [id]          - opaque host id (DID / agent_id) when the
 *           name is not the primary key host-side; defaults to `name`.
 * @property {string} [displayName] - label shown on the card; defaults to `name`.
 * @property {string} [description] - one-line subtitle (console row style).
 * @property {string} [avatarUrl]   - resolved absolute/relative URL for the
 *           portrait. Adapters own resolution (avatar_hash → `/api/files/...`
 *           standalone; Frinz → its CDN/selfie URL). The component never builds
 *           an avatar URL itself.
 * @property {('online'|'offline'|'error'|string)} [status] - presence; drives
 *           the status dot and whether the card is selectable. Absent ⇒ online.
 * @property {boolean} [isDemo]      - carried through for the #868 demo banner;
 *           the component does not act on it (the host owns banner policy).
 * @property {*}       [raw]         - untouched source record for host renderCard.
 */

/**
 * The host data adapter. The component NEVER fetches directly — it calls the
 * adapter, so the same list serves the standalone console (`/api/agents`) and
 * an embed (Frinz `/api/companions`) with no branching inside the component.
 *
 * A default adapter over `API.getAgents()` ships for the standalone console
 * (design question 1: standalone reads `/api/agents`); a host supplies its own
 * to back the list with a different source.
 *
 * @typedef {Object} AgentListAdapter
 * @property {() => Promise<AgentListItem[]>} listAgents
 *           - resolve the current agents/companions, already normalized to
 *             {@link AgentListItem}. Rejection surfaces as the list error state.
 * @property {(item: AgentListItem) => void} [onSelect]
 *           - OPTIONAL host hook fired IN ADDITION to the component's own
 *             selection path (see {@link AgentListConfig.onSelect}). Prefer the
 *             config-level `onSelect`; this exists so an adapter can bundle its
 *             data source and selection wiring as one object.
 * @property {(item: AgentListItem) => string} [avatarUrl]
 *           - OPTIONAL override for portrait URL resolution when it cannot be
 *             precomputed in `listAgents` (e.g. signed URLs minted per render).
 * @property {('multi_agent'|'standalone'|string)} [mode]
 *           - source mode, mirrors `/api/agents .mode`. Governs auto-select and
 *             whether selection installs a host-agent route prefix (standalone
 *             must NOT — see identity.js note). Absent ⇒ inferred 'multi_agent'.
 */

// ----------------------------------------------------------------------------
// Card presentation — the pluggable renderer (design question 2 + clarification)
// ----------------------------------------------------------------------------

/**
 * Context passed to a host `renderCard(item, ctx)`. The component owns the
 * card's OUTER element, selection wiring, the status dot, and the
 * `agent-card-actions` slot anchor; `renderCard` fills the card BODY. This is
 * the same division as the conversations pane's owned rows vs host callbacks:
 * presentation is pluggable, behavior is shared.
 *
 * @typedef {Object} AgentCardRenderContext
 * @property {boolean}  selected    - true when this card is the active selection.
 * @property {boolean}  standalone  - true in single-agent mode (mirrors the
 *           `agent-card-actions` slot's `standalone`).
 * @property {(value: string) => string} escapeHtml
 *           - shared HTML-escape helper; card bodies built via string HTML MUST
 *             route untrusted fields through it (avatar URLs, names).
 * @property {HTMLElement} actionsAnchor
 *           - the `agent-card-actions` slot container the component has already
 *             created and appended. A host `renderCard` that builds its own
 *             layout MUST place this element where per-card action buttons
 *             belong (the actions row under the portrait). The component renders
 *             the slot INTO it either way; the host only positions it.
 */

/**
 * A card renderer. Receives the normalized item + context; returns the card's
 * body element (or mutates a passed element — impl ticket picks one; the
 * signature below is the design intent: return an element the component appends
 * inside the card shell it owns).
 *
 * Default (omitted) = the CONSOLE ROW style (status dot + name + description +
 * stop button), matching today's `.agent-item` markup in identity.js. Frinz
 * supplies a PORTRAIT-CARD renderer: large portrait, name below, and an actions
 * area under the portrait. The shared list layout BUDGETS for an actions row by
 * default (design question 2) so buttons under a portrait have real vertical
 * room — the current Frinz cards are too tight and that is a component
 * responsibility, not a per-host CSS patch.
 *
 * @typedef {(item: AgentListItem, ctx: AgentCardRenderContext) => HTMLElement} AgentCardRenderer
 */

// ----------------------------------------------------------------------------
// The list surface — mountAgentList(containerEl, config)
// ----------------------------------------------------------------------------

/**
 * Config for the embeddable LIST surface — the agent/companion analogue of
 * `mountConversations`. Owns: fetch-via-adapter, render (owned shell + pluggable
 * card body), selection, the per-card `agent-card-actions` slot, refresh, and
 * the active-selection highlight. Does NOT own pane chrome (that is
 * {@link AgentListPaneConfig}).
 *
 * @typedef {Object} AgentListConfig
 * @property {AgentListAdapter} [adapter]
 *           - data source (design question 1). Omitted ⇒ the default adapter
 *             over `API.getAgents()` (standalone console).
 * @property {AgentCardRenderer} [renderCard]
 *           - per-card body renderer (design question 2 + clarification).
 *             Omitted ⇒ console row style.
 * @property {(item: AgentListItem, meta: { standalone: boolean }) => void} [onSelect]
 *           - fired when a card is chosen. The component ALWAYS drives the
 *             shared host-agent selection path first (`API.setHostAgent(name)`
 *             in multi-agent mode; see {@link AgentListHandle.select} and
 *             design question 4), then invokes this so the host can mount chat
 *             / update product state. Selection is the SAME path the chat mount
 *             uses — a host does not reimplement routing.
 * @property {string} [selectedName]
 *           - initial active selection (name). Overridden by user selection.
 * @property {boolean} [autoLoad=true]
 *           - fetch + render on mount. False lets a host drive `refresh()`.
 * @property {boolean} [autoSelectFirst]
 *           - in multi-agent mode, select the first online agent when none is
 *             selected (matches identity.js, gated off in demo-misconfig).
 * @property {(items: AgentListItem[], meta: { mode: string }) => void} [onLoaded]
 *           - fired after each successful load (host demo-banner / stats hook).
 * @property {(value: string) => string} [escapeHtml]
 *           - HTML-escape override (defaults to the shared `ui.js` helper).
 */

/**
 * Handle returned by `mountAgentList`. Mirrors the `mountConversations` handle.
 *
 * @typedef {Object} AgentListHandle
 * @property {HTMLElement} element - the list root the component owns.
 * @property {() => Promise<void>} refresh
 *           - re-fetch via the adapter and repaint (seq-guarded like the
 *             conversations list so a stale response never wins).
 * @property {(name: string) => void} select
 *           - programmatically select an agent by name: drives the shared
 *             selection path (`API.setHostAgent` + `onSelect`) and repaints the
 *             active highlight. The click handler calls this.
 * @property {(name: string) => void} setActiveName
 *           - override the active-selection highlight WITHOUT firing selection
 *             (host reconciling its own notion of current agent; mirrors the
 *             conversations pane's `setActiveSessionId`, #2222).
 * @property {() => (AgentListItem|null)} getActive - current selection or null.
 * @property {() => void} destroy - teardown listeners + empty the container.
 */

// ----------------------------------------------------------------------------
// The pane — mountAgentListPane(containerEl, config) (scope additions)
// ----------------------------------------------------------------------------

/**
 * Config for the full collapsible PANE — the agent/companion analogue of
 * `mountConversationsPane`. It is `mountAgentList` PLUS the SAME chrome contract
 * as the conversations pane (#2199), one implementation for standalone + embeds:
 *
 *   - a `<` chevron collapse rail (fully hides the pane; a host toolbar trigger
 *     reopens) with `open()/close()/toggle()` + `onToggle(collapsed)`;
 *   - a drag-resize handle with min/max width + `localStorage` persistence;
 *   - a component-owned "+ New" action in the pane header, mapped by the host
 *     via `onNew` (Frinz → Add-a-Companion; standalone console → new-agent /
 *     spawn flow, an affordance it does NOT have today and SHOULD gain here,
 *     same-everywhere rule);
 *   - collapse / resize / (view) state persisted under `storageKey`.
 *
 * Chrome is ADOPT-or-BUILD exactly like the conversations pane: an existing
 * `.pane-header` / `.resize-handle` in the container is reused; a bare `<div>`
 * (the embedder contract) gets the full chrome built inside it. Extends
 * {@link AgentListConfig} — all list config is forwarded verbatim to the inner
 * `mountAgentList`.
 *
 * @typedef {AgentListConfig & {
 *   title?: string,
 *   collapsed?: boolean,
 *   storageKey?: string,
 *   minWidth?: number,
 *   maxWidth?: number,
 *   onToggle?: (collapsed: boolean) => void,
 *   onNew?: () => (void | Promise<void>),
 *   newLabel?: string,
 * }} AgentListPaneConfig
 */

/**
 * @typedef {Object} AgentListPaneHandle
 * @property {HTMLElement} element - the pane element the component owns.
 * @property {AgentListHandle} list - the inner `mountAgentList` handle.
 * @property {() => Promise<void>} refresh - forwards to the inner list.
 * @property {(name: string) => void} select - forwards to the inner list.
 * @property {(name: string) => void} setActiveName - forwards to the inner list.
 * @property {() => void} open   - reveal the pane.
 * @property {() => void} close  - fully hide the pane (chevron behavior).
 * @property {() => void} toggle - toggle collapsed state (persisted).
 * @property {boolean} collapsed - current collapsed state (getter).
 * @property {() => void} destroy - teardown chrome this mount built + inner list.
 */

// No exports: this module is types-only. Importing it has no runtime effect.
export {};
