// ============================================================================
// UI extension slot contract — interface stubs (ticket 01, epic #2038)
// ============================================================================
//
// This file is the authoritative, machine-readable companion to
// `docs/proposals/ui-extension-slots/SLOTS.md`. It contains JSDoc typedefs
// ONLY — no runtime logic, no registry, no exports of behavior. Ticket 02
// (`02-slot-registry.md`) implements `UI.register` / `UI.renderSlot` and the
// event bus against these types; ticket 04 migrates voice onto them.
//
// Editors/IDEs pick these typedefs up through the ambient `@typedef`s below;
// implementation modules reference them with `@param {UIContribution}` etc.
// Nothing here runs. Importing this module has no side effects.
//
// Stability: the names and shapes below are a CONTRACT. Downstream tickets
// implement against them; changing a typedef is an API change, not a tidy-up.
// ============================================================================

/**
 * The set of stable zone ids a contribution may target. Each value is the
 * canonical string used as the first argument to `UI.register(slot, ...)` and
 * `UI.renderSlot(slot, ctx)`. See SLOTS.md for per-zone anchor semantics,
 * context guarantees, and retrigger events.
 *
 * `chat-message-renderers`, `nav-tabs`, and `panel-root` are listed for
 * completeness but are governed by ticket 06's separate registries
 * (`registerPartRenderer` lineage + panel contributions), not the positional
 * inline-widget contract below. They are named here so the id space is closed.
 *
 * @typedef {(
 *   | 'chat-input-actions'
 *   | 'agent-card-actions'
 *   | 'input-footer-status'
 *   | 'modal-root'
 *   | 'panel-section'
 *   | 'chat-message-renderers'
 *   | 'nav-tabs'
 *   | 'panel-root'
 * )} SlotId
 */

/**
 * Event names the bus emits to retrigger a contribution's gate + render.
 * A contribution opts into a subset via its `events` array; the registry
 * re-evaluates `gate(ctx)` and (if still truthy) re-runs `render(el, ctx)`
 * — calling any prior teardown first — when one of its events fires.
 *
 * @typedef {(
 *   | 'agent:switch'
 *   | 'session:change'
 *   | 'tools_updated'
 *   | 'capabilities:changed'
 *   | 'panel:shown'
 * )} SlotEvent
 */

/**
 * The minimal API surface a contribution can rely on from any context object.
 * This is the live `API` client (`api_client.mjs`); only the members the slot
 * contract guarantees are documented here. Zone-specific context objects
 * extend `SlotContext` with additional keys (see the per-zone typedefs).
 *
 * @typedef {Object} SlotApi
 * @property {(key: string) => boolean} hasCapability
 *           - dot-path capability check (e.g. `hasCapability('voice')`,
 *             `hasCapability('keys.agent')`). The gate's primary input.
 */

/**
 * Base context passed to every `render(el, ctx)` and `gate(ctx)`. Concrete
 * zones widen this; consumers should read only the keys their zone guarantees
 * (documented per-zone in SLOTS.md and the typedefs below).
 *
 * @typedef {Object} SlotContext
 * @property {SlotApi}      api     - the live API client (capability checks).
 * @property {HTMLElement} [element]
 *           - the mount element for the zone instance. Present in render ctx;
 *             contributions mount INTO the `el` argument of `render`, not into
 *             `ctx.element` directly (the registry owns the per-contribution
 *             container). `ctx.element` is the zone anchor for reference.
 */

/**
 * Context for `chat-input-actions`. One global instance (the single chat
 * input row). Anchor: insert-before `#send-button` within its parent, ordered
 * by `order`. Mirrors voice's mic button (`voice/ui.js` `mountButton`).
 *
 * @typedef {SlotContext} ChatInputActionsContext
 */

/**
 * Context for `agent-card-actions`. Rendered once PER agent card. Carries the
 * card's agent identity and the standalone-vs-multi-agent mode so a
 * contribution can scope its controls (voice keys standalone sessions on
 * `null`, distinct from the row's `data-agent-name`). Anchor: append into the
 * card's actions container. Re-rendered on `agent:switch` / `session:change`.
 *
 * @typedef {Object} AgentCardActionsContext
 * @property {SlotApi}  api
 * @property {HTMLElement} [element]
 * @property {string}   agentName   - the card's agent name (the voice session
 *           key; `null`-equivalent sentinel in standalone mode).
 * @property {boolean}  standalone  - true when running single-agent (no
 *           multi-agent "Other agents" sidebar); false in multi-agent host.
 */

/**
 * Context for `input-footer-status`. One global instance (`.input-footer`).
 * For status chips/badges (voice path badge + privacy banner). Anchor:
 * insert at the left of the footer (before `firstChild`) so it does not fight
 * right-aligned context-status. Re-rendered on `session:change`.
 *
 * @typedef {SlotContext} InputFooterStatusContext
 */

/**
 * Context for `modal-root`. Feature-owned overlay dialogs mounted to a
 * dedicated overlay root (today voice appends to `document.body`). A modal is
 * typically built once and toggled; the trigger to SHOW it is an event
 * subscription on the bus, not the slot render itself (see SLOTS.md
 * "event-driven, panel-independent UI").
 *
 * @typedef {SlotContext} ModalRootContext
 */

/**
 * Context for `panel-section` — a section contributed INTO an existing panel
 * (finer grain than a whole panel). `panelId` identifies the host panel
 * (e.g. `'resources'`, `'security'`); the contribution's `gate` typically
 * checks a sub-capability (`keys.agent`, `wallet`, …) the way
 * `resources.js` gates its sub-sections today. Re-rendered on `panel:shown`
 * and `capabilities:changed`.
 *
 * @typedef {Object} PanelSectionContext
 * @property {SlotApi}  api
 * @property {HTMLElement} [element]
 * @property {string}   panelId     - id of the host panel the section mounts into.
 */

/**
 * A single UI contribution. A feature calls `UI.register(slot, contribution)`
 * (ticket 02) once at module load; the registry mounts it into every instance
 * of `slot`, sorted by `order`, gated by `gate`, and re-renders on `events`.
 *
 * `render` MAY return a teardown function. The registry calls it before any
 * re-render of the same contribution and before unmount, so contributions
 * that attach listeners or live state clean up deterministically. This is the
 * capability `registerHeaderAction` lacks (it rebuilds all buttons on every
 * call, destroying live state) — and is why the slot registry supersedes it.
 *
 * @typedef {Object} UIContribution
 * @property {SlotId}   slot       - zone id (see {@link SlotId}).
 * @property {number}  [order=100] - ascending sort within the zone.
 * @property {string}  [id]        - stable id for update/removal & dedupe.
 *           Re-registering the same `id` replaces the prior contribution.
 * @property {(ctx: SlotContext) => boolean} [gate]
 *           - shown only when truthy; re-evaluated on each event in `events`.
 *             Defaults to always-shown when omitted.
 * @property {(el: HTMLElement, ctx: SlotContext) => (void | (() => void))} render
 *           - mounts into `el`; MAY return a teardown fn called before
 *             re-render/unmount. `el` is the registry-owned per-contribution
 *             container, not the shared zone anchor.
 * @property {SlotEvent[]} [events]
 *           - event names that retrigger gate+render for this contribution.
 */

/**
 * The registry surface ticket 02 exposes (documented here so contributions
 * and core call sites share one shape). NOT implemented in this file.
 *
 * @typedef {Object} SlotRegistry
 * @property {(slot: SlotId, contribution: Omit<UIContribution, 'slot'> & { slot?: SlotId }) => void} register
 *           - register a contribution into a zone.
 * @property {(slot: SlotId, ctx: SlotContext) => void} renderSlot
 *           - core calls this when building a zone instance; iterates
 *             registered contributions, sorts by `order`, applies gates,
 *             mounts survivors into per-contribution containers.
 * @property {(event: SlotEvent, detail?: object) => void} emit
 *           - publish a bus event; retriggers contributions subscribed via
 *             their `events` array.
 */

// ----------------------------------------------------------------------------
// OUT OF SLOT SCOPE — documented here so the boundary is explicit and closed.
// These are NOT zones and NOT contributions; they are named so future authors
// don't try to model them as slots. See SLOTS.md §"Out of slot scope".
// ----------------------------------------------------------------------------

/**
 * Shared-singleton claim/release contract (ticket 09). Voice SEIZES the model
 * selector during a realtime session (`acquireSelectorOwnership` /
 * `releaseSelectorOwnership` in `voice/ui.js`; the epic calls this
 * `acquireVoiceLock`/`releaseVoiceLock` conceptually). This is a negotiation
 * over an exclusive shared control, NOT a mount point — the slot model cannot
 * express it and must not try. Stubbed here only to mark the escape hatch.
 *
 * @typedef {Object} SingletonClaim
 * @property {string}  resource    - id of the contended singleton (e.g. 'model-selector').
 * @property {() => boolean} acquire - seize the resource; false if already claimed.
 * @property {() => void}    release - relinquish the resource.
 */

// No exports: this module is types-only. Importing it has no runtime effect.
export {};
