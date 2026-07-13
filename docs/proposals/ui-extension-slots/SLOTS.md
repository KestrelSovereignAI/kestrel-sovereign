---
type: Architecture Spec
title: UI Extension Slot Taxonomy & Contribution Contract
description: Empirically-derived list of UI extension slots and the contribution contract, design spike for epic #2038 (ticket 01).
status: proposed
---

# SLOTS.md — UI extension slot taxonomy & contribution contract

**Status:** Spec (design spike, ticket 01 of epic #2038)
**Implements against:** nothing yet — this is the contract tickets 02 and 04
build to. No runtime behavior ships with this document.
**Companion stub:** [`static/js/ui-ext/contract.js`](../../../kestrel_sovereign/static/js/ui-ext/contract.js)
(JSDoc typedefs, no logic).

This document is the authoritative, **empirically-derived** list of UI zones a
feature may contribute into, and the contract every contribution obeys. It is
derived from voice's existing injection points — not invented — and is
pressure-tested against **every** voice touch point in the epic table
([epic #2038](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2038))
below. Where the slot model cannot express a voice
behavior, that gap is recorded as a finding (it reshapes ticket 02), never as a
"TBD".

The doctrine driving this is *survey-platform-first*: two partial extension APIs
already exist in `chat.js`. We reconcile with them (build-on vs supersede)
before defining anything new.

---

## 1. Pre-existing extension APIs — build-on vs supersede

### 1.1 `registerPartRenderer(type, fn)` — BUILD ON (ticket 06)

[chat.js:3583](../../../kestrel_sovereign/static/js/chat.js). A **solid**
type-dispatch renderer registry: error-isolated, realm-safe (duck-typed
`nodeType` Node check, #1650), with a documented fallback to escaped text when
no renderer is registered. The standalone console registers none; hosts (e.g.
Frinz's selfie `<img>`) register their own.

- **Decision:** keep as-is; it is the foundation for ticket 06's chat-renderer
  registry. The `chat-message-renderers` zone id in this taxonomy is a *pointer*
  to this path, not a new positional contract.
- **Trust model to carry forward:** the host owns the markup and its
  sanitization — `appendMessagePart` renders a string via `innerHTML` with **no
  core sanitization**. Installed features are already a trust boundary (their
  Python runs arbitrary code), so same-origin feature JS is no less trusted.
  Documented, not changed, here.
- **Known gap (for ticket 06, not this ticket):** it is a **separate path** from
  positional tool cards (`renderToolCardsHtml`), which have **no** renderer hook
  today. A feature cannot yet customize how *its own tool card* renders inline.

### 1.2 `registerHeaderAction(action)` — SUPERSEDE, preserve contract (ticket 02)

[chat.js:684](../../../kestrel_sovereign/static/js/chat.js), public embedder API
(#1623/#1627). It exists but has **no internal feature adoption** — voice
deliberately bypasses it. Its limitations are the **acceptance criteria** the
slot registry must satisfy and then supersede:

| Limitation of `registerHeaderAction` | Becomes a slot-registry requirement |
|---|---|
| Header-only position (one zone) | Multiple named zones with explicit anchor semantics |
| `renderHeaderActions()` rebuilds **all** buttons every call, destroying live state | `render` returns a teardown fn; re-render is per-contribution and deterministic |
| `onClick`-only action model | Arbitrary `render(el, ctx)` mounting any DOM + listeners |
| No per-agent / session context | Per-zone context object (`agentName`, `standalone`, `api`, …) |
| No gate; caller decides whether to call | Declarative `gate(ctx)`, re-evaluated on bus events |

- **Decision:** ticket 02 supersedes it **while preserving its public contract**.
  Embedders depend on it, so ticket 02 ships `registerHeaderAction` as a **shim**
  over the new registry (a contribution into a `chat-header-actions`-style zone),
  not a deletion. This taxonomy does not enumerate `chat-header-actions` as a
  first-class feature zone because no *feature* needs it — it is the embedder
  back-compat surface.

### 1.3 Explicitly out of slot scope (named here so the boundary is closed)

- **Host embed mode (`chrome`).** `chrome:false` hides the whole nav/sidebar for
  chat-only embeds ([app.js:48](../../../kestrel_sovereign/static/js/app.js),
  capability flag in [api_client.mjs](../../../kestrel_sovereign/static/js/api_client.mjs)).
  A host-level mode toggle, handled by host config / the singleton mechanism —
  **not a zone**.
- **Event-driven, panel-independent UI.** Notifications SSE + approval modals
  fire on a *union* of capabilities even when their panel is hidden
  ([chat.js:1242](../../../kestrel_sovereign/static/js/chat.js)). The *mount* is
  `modal-root`; the *trigger* is an event subscription on the bus (ticket 02) —
  not a slot render.

---

## 2. Zone taxonomy

Nine zone ids form a closed id space. Five are **inline positional** zones
governed by the `UIContribution` contract in §3. One (`chat-message-actions`,
§2.6) is **item-provider-shaped** — contributions supply menu items, not DOM.
Three (`chat-message-renderers`, `nav-tabs`, `panel-root`) are listed for
completeness and id-space closure but are governed by ticket 06's separate
registries; their rows note the owning contract.

Each inline zone defines: **stable id · context contract · anchor semantics ·
retrigger events**.

### 2.1 `chat-input-actions`

- **id:** `chat-input-actions`
- **Cardinality:** one global instance (the single chat input row).
- **Context:** `{ api, element }` (`ChatInputActionsContext`). No per-agent
  identity — the input row is shared across the active pane.
- **Anchor semantics:** **insert-before** `#send-button` within its parent,
  ordered by `order` ascending. Row reads `textarea | <contributions> | send`.
  Voice mounts its mic exactly here
  ([voice/ui.js:454](../../../kestrel_sovereign/static/js/voice/ui.js)
  `insertBefore`).
- **Retrigger events:** `capabilities:changed` (appear/disappear when a feature
  is enabled/disabled). `session:change` if the action reflects live session
  state.

### 2.2 `agent-card-actions`

- **id:** `agent-card-actions`
- **Cardinality:** one instance **per agent card**.
- **Context:** `{ api, element, agentName, standalone }`
  (`AgentCardActionsContext`). `agentName` is the card's identity (and the voice
  session key — `null`-equivalent sentinel in standalone mode, distinct from the
  row's `data-agent-name`). `standalone` distinguishes single-agent vs the
  multi-agent host's "Other agents" sidebar.
- **Anchor semantics:** **append** into the card's actions container. Voice
  appends its 🎧/🎤 controls
  ([voice/ui.js:297](../../../kestrel_sovereign/static/js/voice/ui.js)
  `appendChild`), invoked from
  [identity.js:758](../../../kestrel_sovereign/static/js/identity.js).
- **Retrigger events:** `agent:switch`, `session:change` (controls reflect live
  per-agent session state — this is exactly voice's `onAgentSwitch` /
  `refreshAgentVoiceCard` need), `capabilities:changed`.

### 2.3 `input-footer-status`

- **id:** `input-footer-status`
- **Cardinality:** one global instance (`.input-footer`).
- **Context:** `{ api, element }` (`InputFooterStatusContext`).
- **Anchor semantics:** **insert at the left** of the footer (before
  `firstChild`) so contributions don't fight the right-aligned context-status
  text. Voice inserts its path badge + privacy banner here
  ([voice/ui.js:483](../../../kestrel_sovereign/static/js/voice/ui.js)).
- **Retrigger events:** `session:change` (status chips reflect live session
  state), `capabilities:changed`.

### 2.4 `modal-root`

- **id:** `modal-root`
- **Cardinality:** shared overlay root; many modals may register.
- **Context:** `{ api, element }` (`ModalRootContext`).
- **Anchor semantics:** **append** to a dedicated overlay root (today voice
  appends to `document.body`,
  [voice/ui.js:505](../../../kestrel_sovereign/static/js/voice/ui.js)). The
  registry should provide a stable `#modal-root` rather than `document.body` so
  teardown is scoped. A modal is built once and toggled hidden/visible.
- **Retrigger events:** none for the *mount* (the modal persists hidden). The
  **show** trigger is an event subscription on the bus (e.g. a custom event the
  feature listens for), per §1.3 "event-driven UI".

### 2.5 `panel-section`

- **id:** `panel-section`
- **Cardinality:** one instance per `(panelId, section)`; many sections per
  panel.
- **Context:** `{ api, element, panelId }` (`PanelSectionContext`). `panelId`
  names the host panel (`'resources'`, `'security'`, …).
- **Anchor semantics:** **append** into the host panel's content container,
  ordered by `order`. Empirically required: the Resources panel already composes
  sub-sections gated individually by `keys.agent` / `keys.user` /
  `keys.platform` / `wallet`
  ([resources.js:29-64](../../../kestrel_sovereign/static/js/resources.js)). A
  feature adding "its section" to Resources/Security/etc. needs this finer grain
  than a whole-panel contribution.
- **Retrigger events:** `panel:shown` (panels load lazily on tab click — render
  when first shown), `capabilities:changed` (a sub-section's gate mirrors
  `resources.js`'s per-sub-capability checks).

### 2.6 `chat-message-actions` (item-provider slot, #2410)

- **id:** `chat-message-actions`
- **Cardinality:** one lazy evaluation per message-bubble kebab (⋯) *open* —
  the builder (`static/js/message_kebab.js`) calls `UI.collectItems` each time a
  bubble's menu opens, so items reflect current state and features registered
  after page load still appear.
- **Context:** `{ messageId, role, metadata, agent }`
  (`ChatMessageActionsContext`). Scoped to one message bubble.
- **Shape:** **item-provider**, NOT DOM. A contribution supplies
  `items: (ctx) => [{ label, danger?, separatorBefore?, onSelect }]`
  (`MessageActionsContribution` / `MessageActionsItem`). It is never mounted by
  `renderSlot`; `UI.collectItems('chat-message-actions', ctx)` gathers items
  across contributions — gated by `gate(ctx)`, ordered by `order`, and each
  provider **error-isolated** so a throwing contribution never drops the base
  Move-to-trash / Delete-permanently items. Feature items land **above** the
  destructive separator. This is the single sanctioned surface for per-message
  actions: features contribute menu items, never overlay buttons on bubbles.

### 2.7 Zones owned by ticket 06 (id-space closure only)

| id | Owning contract | Note |
|---|---|---|
| `chat-message-renderers` | `registerPartRenderer` (§1.1) | Custom renderers keyed by tool name / content type. Separate, type-dispatch path — not positional. |
| `nav-tabs` | Panel contribution (ticket 06) | A whole nav tab; today hardcoded in [app.js:53](../../../kestrel_sovereign/static/js/app.js) + `index.html`. |
| `panel-root` | Panel contribution (ticket 06) | The panel body a `nav-tabs` entry reveals. Paired with `nav-tabs`. |

---

## 3. The contribution contract

```js
/**
 * @typedef {Object} UIContribution
 * @property {string}   slot      - zone id
 * @property {number}   [order]   - ascending sort within the zone (default 100)
 * @property {string}   [id]      - stable id for update/removal & dedupe
 * @property {(ctx) => boolean} [gate] - shown only when truthy; re-evaluated on event
 * @property {(el: HTMLElement, ctx: object) => (void | () => void)} render
 *           - mounts into el; MAY return a teardown fn called before re-render/unmount
 * @property {string[]} [events] - event names that retrigger gate+render for this contribution
 */
```

Authoritative typedefs live in
[`static/js/ui-ext/contract.js`](../../../kestrel_sovereign/static/js/ui-ext/contract.js).
Key semantics:

- **`order`** — ascending within a zone, default `100`. Multiple features in one
  zone get deterministic order and must not assume exclusive ownership of an
  anchor (cross-cutting risk in the epic).
- **`id`** — stable; re-registering the same `id` **replaces** the prior
  contribution (update/removal/dedupe).
- **`gate(ctx)`** — capability-based, re-evaluated on each event in `events`.
  Mirrors voice's `API.hasCapability('voice')` guards. Default: always shown.
- **`render(el, ctx)` → teardown** — mounts into the registry-owned
  per-contribution container `el` (not the shared zone anchor). The optional
  returned teardown fn is called **before** re-render and **before** unmount —
  the exact capability `registerHeaderAction` lacks (it destroys live state on
  every rebuild).
- **`events`** — subset of the §4 vocabulary that retriggers gate+render.

---

## 4. Event vocabulary (the bus, ticket 02)

The slot registry is paired with an event bus so contributions re-render without
each feature reinventing voice's bespoke refresh plumbing. Vocabulary, each
cross-referenced to the voice trigger it replaces:

| Event | Meaning | Voice equivalent today |
|---|---|---|
| `agent:switch` | Active agent changed | `onAgentSwitch(prev, next)` ([identity.js:817](../../../kestrel_sovereign/static/js/identity.js) → `onVoiceAgentSwitch`) |
| `session:change` | Live session state changed (privacy, path, armed) | `applyActiveSessionState` / `refreshAgentVoiceCard` |
| `tools_updated` | Backend emitted progressive tool disclosure (#1315) | `subscribeSSE('tools_updated', …)` ([voice/ui.js:258](../../../kestrel_sovereign/static/js/voice/ui.js)) |
| `capabilities:changed` | Enabled-feature/capability set changed | (none — voice assumes static caps; new requirement) |
| `panel:shown` | A lazily-loaded panel became visible | (none — core lazy-loads panels on tab click; `panel-section` needs it) |

---

## 5. Voice walkthrough — every injection point mapped (no residue)

Each row of the epic's voice surface-area table
([epic #2038](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2038))
mapped to a zone or to the ticket-09 escape hatch.
**No "TBD".**

| Voice injection point | Today's coupling | Maps to | Resolution |
|---|---|---|---|
| mic button before `#send-button` | `insertBefore` ([voice/ui.js:454](../../../kestrel_sovereign/static/js/voice/ui.js)) | **`chat-input-actions`** (insert-before `#send-button`) | ✅ expressible |
| 🎧 listen / 🎤 talk per agent card | `appendChild` ([voice/ui.js:297](../../../kestrel_sovereign/static/js/voice/ui.js)), called from [identity.js:758](../../../kestrel_sovereign/static/js/identity.js) | **`agent-card-actions`** (append; ctx `agentName`, `standalone`) | ✅ expressible |
| path badge + privacy banner | `.input-footer` insert ([voice/ui.js:483](../../../kestrel_sovereign/static/js/voice/ui.js)) | **`input-footer-status`** (insert-left) | ✅ expressible |
| voice picker dialog | append `document.body` ([voice/ui.js:505](../../../kestrel_sovereign/static/js/voice/ui.js)) | **`modal-root`** (append to overlay root) | ✅ expressible |
| realtime tool cards / bubbles | delegates to `addMessage` / `finalizeStreamingMessage` | **`chat-message-renderers`** via `registerPartRenderer` (ticket 06) | ✅ existing path (§1.1) |
| stateful card refresh | `onAgentSwitch` / `refreshAgentVoiceCard` | **events** `agent:switch` + `session:change` retrigger `agent-card-actions` | ✅ expressible (bus, §4) |
| `tools_updated` realtime tool-set push | `subscribeSSE('tools_updated', …)` ([voice/ui.js:258](../../../kestrel_sovereign/static/js/voice/ui.js)) | **event** `tools_updated` (§4) | ✅ expressible (bus) |
| model-selector seizure during realtime | `acquireSelectorOwnership` / `releaseSelectorOwnership` ([voice/ui.js:649](../../../kestrel_sovereign/static/js/voice/ui.js)) | **ticket-09 escape hatch** (shared-singleton claim/release) | ⛔ **NOT a slot** — see §6 |

### Finding (reshapes downstream, not deferred)

The model-selector seizure is the single voice coupling the slot model **cannot**
express, and confirms the epic's hypothesis. A slot mounts *additive* UI into a
zone; it has no notion of *seizing an exclusive shared control* owned by core
(the model selector) and releasing it on session end / agent switch. Modeling it
as a slot would require a contribution to reach outside its own `el` and mutate a
core singleton — violating the inversion-of-dependency thesis. This is correctly
ticket 09's claim/release contract, deliberately sequenced *after* voice
migration (04) so its real shape is discovered, not guessed. Recording it here
satisfies the acceptance criterion that any inexpressible behavior is a finding,
not residue.

> **Naming note:** the epic refers to this conceptually as
> `acquireVoiceLock`/`releaseVoiceLock`; the live implementation in
> `voice/ui.js` is `acquireSelectorOwnership`/`releaseSelectorOwnership` with a
> per-session `ownsSelector` flag. Ticket 09 should name the generalized
> primitive without the `voice`/`selector` specificity.

---

## 6. Out of slot scope (escape hatches & host modes)

| Concern | Why not a slot | Handled by |
|---|---|---|
| Model-selector seizure (`ownsSelector`) | Exclusive claim over a core singleton, not additive mounting | Ticket 09 (claim/release API) |
| Host embed mode (`chrome:false`) | Host-level whole-chrome toggle | Host config / capability flag ([api_client.mjs](../../../kestrel_sovereign/static/js/api_client.mjs)) |
| Show-modal trigger (vs mount) | The *mount* is `modal-root`; the *trigger* is an event | Event bus (ticket 02) |
| Whole nav panel | Coarser than inline; own contract | `nav-tabs` + `panel-root` (ticket 06) |
| Custom inline tool-card markup | Type dispatch, not positional | `registerPartRenderer` / ticket 06 |

---

## 7. Acceptance-criteria self-check

- [x] Every zone has **id · context contract · anchor semantics · retrigger
  events** (§2).
- [x] Written walkthrough maps **each** voice injection point to a zone or the
  ticket-09 escape hatch with **no "TBD"** (§5); the one inexpressible coupling
  is recorded as a finding, not deferred silently (§5 Finding, §6).
- [x] Interface stub committed:
  [`static/js/ui-ext/contract.js`](../../../kestrel_sovereign/static/js/ui-ext/contract.js),
  typedefs only, no runtime logic.
- [x] Pre-existing APIs inventoried with explicit build-on (`registerPartRenderer`)
  vs supersede-while-preserving-contract (`registerHeaderAction`) decisions (§1).

## Explicitly NOT in this ticket

- No registry implementation (ticket 02).
- No voice changes (ticket 04).
