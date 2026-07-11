---
type: Design Proposal
title: Mountable, skinnable agent-list component
description: Design for extracting the multi-agent sidebar's agent list into a mountable, host-skinnable component (list surface + collapsible pane), so embedding hosts stop maintaining bespoke agent/companion lists. Design spike for issue #2166.
status: proposed
---

# Mountable, skinnable agent-list component

**Status:** Design spike (issue #2166) — this document + interface stubs, **not**
implementation. Implementation tickets follow from the accepted design.
**Implements against:** nothing yet — this is the contract the follow-up
ticket(s) build to. No runtime behavior ships with this document.
**Companion stub:** [`contract.js`](./contract.js) (JSDoc typedefs, no logic),
mirroring [`static/js/ui-ext/contract.js`](../../../kestrel_sovereign/static/js/ui-ext/contract.js).

---

## 1. Motivation

Frinz's companion list is functionally Kestrel's agent list with a product skin:
portrait-forward cards, per-card actions, selection driving the chat mount. It
is the largest remaining bespoke chat-adjacent surface in the Frinz embed. The
UI-extension-slot contract already defines an `agent-card-actions` slot
([SLOTS.md §2.2](../ui-extension-slots/SLOTS.md)) — but there is no mountable
list to host it in an embed, so Frinz maintains its own list, its own
collapse/resize pane machinery, and its own "+ New" affordance.

This design extracts the agent list into the **same contract family** already
proven for chat (`mount()`) and conversations (`mountConversations` /
`mountConversationsPane`, #2149 / #2199 / #2216 / #2222): the component owns
chrome + orchestration; the host supplies a container + config; presentation is
pluggable.

**Doctrine:** *survey-platform-first.* We do not invent a new pane model — we
reuse the conversations-pane one verbatim (chevron collapse, drag-resize,
persistence, adopt-or-build chrome, owned rows vs host callbacks). One
implementation serves the standalone console and every embed.

---

## 2. The split — one list surface, one pane, presentation pluggable

Two mount points, matching the conversations module exactly:

| Function | Owns | Host supplies |
|---|---|---|
| `mountAgentList(el, config)` | fetch-via-adapter, owned card shell, selection path, per-card `agent-card-actions` slot, refresh, active highlight | data adapter, `renderCard`, `onSelect` |
| `mountAgentListPane(el, config)` | all of the above **plus** chevron collapse rail, drag-resize + persistence, "+ New" header action, `onToggle` | everything above **plus** `onNew`, `title`, `storageKey` |

The division is the clarification the user settled on the ticket: **the shared
component owns chrome + orchestration** (collapse/resize/persistence, "+ New"
hook, selection, refresh) **while card RENDERING is a config hook**
(`renderCard(item, ctx) → element`, default = console row style; Frinz supplies
its portrait-card renderer). This is the same owned-rows-vs-host-callbacks
division as the conversations pane — presentation pluggable, behavior shared.

---

## 3. Design questions settled

### 3.1 Data source — the host adapter (question 1)

The component NEVER fetches directly. It calls an `AgentListAdapter.listAgents()`
that returns items already normalized to `AgentListItem`:

- **Standalone console:** a default adapter over `API.getAgents()`
  (`/api/agents`, see [`endpoints/models.py`](../../../kestrel_sovereign/endpoints/models.py)
  `get_agents`). It maps the agent-card fields (`name`, `description`,
  `status`, `is_demo`) and resolves `avatarUrl` from `avatar_hash` →
  `/api/files/<hash>`.
- **Embed (Frinz):** its own adapter over `/api/companions`, mapping companion
  records onto the same normalized shape and resolving its own portrait URLs.

`AgentListItem` is deliberately a **subset** of the standalone agent-card so a
host maps onto it without the component reaching into host-specific fields. The
untouched source record rides along as `raw` so a host `renderCard` can read
product fields (mood, relationship tier) the normalized shape omits — without
widening the shared type. Avatar URL resolution is the **adapter's** job (or an
optional `avatarUrl(item)` override for per-render signed URLs); the component
never builds an avatar URL itself.

`adapter.mode` mirrors `/api/agents .mode` and governs auto-select + whether
selection installs a host-agent route prefix (standalone must NOT — using a
host-agent prefix in standalone 404s `/api/conversations`; see the identity.js
note at `loadAgents`).

### 3.2 Card presentation — pluggable renderer, not theme-vars-only (question 2)

We chose the **host-provided `renderCard(item, ctx)` hook** over a base-markup +
theme-variables approach, because the user clarified hosts render cards
*structurally* differently (Frinz portrait cards vs the console row), not just
recolored. Theme vars cannot express "portrait on top, name below, actions row
under it" vs "dot + name + description in a row".

- **Default (`renderCard` omitted):** the CONSOLE ROW style — status dot, name,
  description, stop button — matching today's `.agent-item` markup in
  identity.js.
- **Frinz:** a portrait-card renderer — large portrait, name below, actions area
  under the portrait.

The component owns the card's OUTER shell, selection wiring, the status dot, and
the `agent-card-actions` slot anchor (passed to the renderer as
`ctx.actionsAnchor`); `renderCard` fills the body and positions the actions
anchor. **The shared list layout budgets for an actions row by default** — the
ticket calls out that current Frinz cards are too tight for buttons under
portraits, and that the layout "must budget for an actions row by default". That
is a component responsibility (base CSS reserves vertical room for the actions
area), not a per-host CSS patch.

### 3.3 Slot integration — the existing `agent-card-actions` registry (question 3)

Unchanged from today. Per card, the component creates the slot anchor and calls
`UI.renderSlot('agent-card-actions', { element, api, agentName, standalone })`
exactly as identity.js does now (identity.js `loadAgents`), so voice's per-card
controls (SLOTS.md §2.2) keep working with no change. `standalone` comes from
`adapter.mode === 'standalone'`; `agentName` is the item's `name` (the voice
session key). The component tears the anchor down on list rebuild (the slot
registry's teardown contract handles listener cleanup). Card rendering being
pluggable does not touch the slot: the host renderer only *positions* the
anchor; the component *renders into* it.

### 3.4 Selection contract — the shared host-agent path (question 4)

Selecting a card fires the **same host-agent selection path the chat mount
uses**. In multi-agent mode the component calls `API.setHostAgent(item.name)`
(pinning routing) before invoking the host `onSelect` — a host does not
reimplement routing. This mirrors `window.selectAgent` in identity.js, but the
routing primitive (`setHostAgent`) is what the component owns; capability
refresh / chat mount / `agent:switch` bus emit stay host-side product wiring
(see §3.5). In standalone mode selection does NOT install a route prefix.

`select(name)` (programmatic) and `setActiveName(name)` (highlight-only, no
fire) are both on the handle — the latter mirrors the conversations pane's
`setActiveSessionId` (#2222) so a host reconciles its own notion of "current
agent" without a selection round-trip.

Collapse/expand of the list panel is the pane's chevron affordance (§4), the
same two-state hide (`open`/`close`/`toggle` + `onToggle`) as
`mountConversationsPane`.

### 3.5 Migration sketch for Frinz (question 5)

**Component owns (deleted from Frinz on adoption):**

- the companions-list fetch + render loop,
- `panes.js` companions-pane machinery (collapse rail, drag-resize, persisted
  width/collapsed state) — deleted exactly as Frinz's bespoke history sidebar
  was on conversations-pane adoption,
- the "+ New" button chrome (becomes the component's header action, wired via
  `onNew`),
- selection → `setHostAgent` → chat-mount plumbing (the routing half).

**Stays host-side (product logic passed as config):**

- `companionCreator` / adoption entry points — invoked FROM `onNew`; the flow
  itself (wizard, template pick, payment) is Frinz product logic the component
  never sees,
- the portrait-card `renderCard` (Frinz's skin),
- the `/api/companions` adapter (`listAgents` + avatar URL resolution),
- what happens AFTER selection beyond routing (mounting Frinz's chat surface,
  updating product state) — Frinz's `onSelect`.

Standalone console migration is symmetric: identity.js's `loadAgents` loop and
the static `#agents-pane` chrome are replaced by a `mountAgentListPane` call
with the default `/api/agents` adapter and default row renderer. Per the
same-everywhere rule, the console GAINS a "+ New" affordance it lacks today,
wired via `onNew` to a new-agent / spawn flow.

---

## 4. Pane chrome — identical contract to the conversations pane (#2199)

The scope additions on the ticket make the agent-list pane share the SAME chrome
contract as the conversations pane, one implementation for standalone + embeds.
`mountAgentListPane` reuses the `mountConversationsPane` design point-for-point
(see [`conversations.js`](../../../kestrel_sovereign/static/js/conversations.js)
`mountConversationsPane`):

1. **Chevron (`<`) collapse rail** — two states, open and fully hidden
   (`display:none`, no leftover rail, the #2216 behavior). `open()` / `close()`
   / `toggle()` + `onToggle(collapsed)` so a host toolbar trigger reopens it.
2. **Drag-resize** — min/max width clamp, width persisted to `localStorage`
   under `storageKey` (`:width`), collapsed state under `:collapsed`.
3. **Component-owned "+ New" action** in the pane header — a config hook
   (`onNew`) each host maps: Frinz → Add-a-Companion; standalone console →
   new-agent / spawn flow.
4. **Adopt-or-build chrome** — an existing `.pane-header` / `.resize-handle` in
   the container is reused; a bare `<div>` (the embedder contract) gets the full
   chrome built inside it. `destroy()` removes only chrome this mount built.

Persistence degrades to no-ops when `localStorage` is unavailable (embed hosts /
jsdom tests), exactly like the conversations pane's guarded `paneStorage()`.

---

## 5. Interface stub

The machine-readable contract is [`contract.js`](./contract.js) — JSDoc typedefs
only, no runtime, importing it has no side effects. Key types:

- `AgentListItem` — normalized agent/companion record the list consumes.
- `AgentListAdapter` — `listAgents()` + identity/avatar resolution (the host
  data-source seam).
- `AgentCardRenderer` / `AgentCardRenderContext` — the pluggable card hook and
  its context (including the `actionsAnchor` the host positions).
- `AgentListConfig` / `AgentListHandle` — the list surface.
- `AgentListPaneConfig` / `AgentListPaneHandle` — the collapsible pane (extends
  the list config with chrome: `title`, `collapsed`, `storageKey`,
  `minWidth`/`maxWidth`, `onToggle`, `onNew`).

The names and shapes there are a CONTRACT — implementation tickets build to them
and Frinz migrates onto them; changing a typedef is an API change.

---

## 6. Out of scope (named so the boundary is closed)

- **The `agent-card-actions` slot registry itself** — owned by epic #2038; this
  component *hosts* the slot, it does not redefine it.
- **New-agent / companion-creation flows** — the component exposes the `onNew`
  hook; the flows behind it are host product logic.
- **Server endpoints** — no `/api/agents` or `/api/companions` change is
  proposed. Adapters normalize existing responses.
- **A2A / agent lifecycle** (spawn, stop semantics beyond the row stop button,
  presence transport) — the list reflects `status`; it does not own presence.

---

## 7. Follow-up implementation tickets (sketch, not part of this spike)

1. `mountAgentList` — list surface + default `/api/agents` adapter + default row
   renderer + selection path + `agent-card-actions` slot integration; migrate
   identity.js `loadAgents` onto it.
2. `mountAgentListPane` — pane chrome (reuse the conversations-pane primitives),
   "+ New" via `onNew`; add the standalone console's new-agent affordance.
3. Frinz migration — `/api/companions` adapter + portrait `renderCard`; delete
   Frinz's bespoke companions list + `panes.js` companions-pane machinery.
