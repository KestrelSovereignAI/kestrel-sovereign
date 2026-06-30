# 02 — Client-side slot registry + event bus

**Type:** Feature (core frontend)
**Depends on:** 01
**Risk:** Medium

## Goal

Implement the runtime that contributions register into and that core renders zones
through. Two collaborating primitives: a **slot registry** and an **event bus**.

This registry **supersedes the abandoned `registerHeaderAction`**
([chat.js:684](../../../kestrel_sovereign/static/js/chat.js)). That API failed for
voice — and the exact reasons it failed are the requirements here. The registry MUST:

- support anchors **other than the chat header** (voice needs the chat *input row*,
  left of `#send-button`; Resources needs a panel section) — anchor semantics are
  per-zone, not a single fixed slot;
- update a single contribution **without rebuilding/destroying its siblings**
  (`registerHeaderAction` did `slot.innerHTML = ''` on every call, dropping live
  state and listeners — fatal for voice's high-frequency button-state updates);
- carry **per-context state** (e.g. `agentName`) so the same contribution renders
  differently per agent card;
- preserve a contribution's DOM element/closures across re-gates where possible
  (stable identity), tearing down only on actual removal.

If the registry cannot do all four, it has not improved on `registerHeaderAction` and
voice will bypass it again.

## Design

### Slot registry (`static/js/ui-ext/registry.js`)

```js
export const UI = {
  register(contribution) { /* validate against contract, store, dedupe by id */ },
  unregister(slot, id) { /* tear down just this contribution */ },
  renderSlot(slot, ctx) { /* first pass: sort by order, gate, render all; track per-contribution teardowns */ },
  refreshContribution(slot, id) { /* re-gate + re-render ONE contribution; siblings untouched */ },
  refreshSlot(slot) { /* re-evaluate the whole zone vs last-known ctx (used sparingly) */ },
};
```

- Teardown and re-render are **per-contribution**, keyed by `(slot, element, id)` —
  **not** wholesale on the zone. An event affecting contribution A must re-gate/
  re-render *only* A and call *only* A's teardown; sibling B's DOM, listeners, and
  closures are untouched. (This is the `registerHeaderAction` failure inverted: it
  blew away the whole slot every call.) A full first-pass `renderSlot` mounts all
  contributions; subsequent event-driven updates are granular. This is also what makes
  voice's `refreshAgentVoiceCard` fall out for free — its button updates in place.
- Contributions are sorted by `order`; ties broken by registration order (stable).
- A contribution that throws in `render` is isolated (logged, skipped) so one
  feature cannot blank a whole zone.
- The last `ctx` per (slot, element) is retained so an event-driven `refreshSlot`
  can re-render without the caller re-supplying context.

### Event bus (`static/js/ui-ext/bus.js`)

- Minimal pub/sub: `on(event, fn)`, `emit(event, payload)`, `off`.
- Core emits the vocabulary defined in ticket 01. The registry subscribes: when an
  event fires, every contribution that declared that event in `events` (or whose
  zone is bound to it) is re-gated and re-rendered.
- Bridges the existing `subscribeSSE` channel
  ([voice/ui.js:258](../../../kestrel_sovereign/static/js/voice/ui.js)) so server
  push events (`tools_updated`) become bus events — features stop subscribing to SSE
  directly.

### Core render-site wiring

Add `UI.renderSlot(...)` calls at each zone's natural render site:
- agent-card creation in
  [identity.js:721-758](../../../kestrel_sovereign/static/js/identity.js)
- chat input row / `.input-footer` construction (chat.js / index.html hydration)
- a single `modal-root` mount target appended once at boot

Core must emit a generic `agent:switch` event at the point where it currently calls
`onVoiceAgentSwitch` ([identity.js:817](../../../kestrel_sovereign/static/js/identity.js)).
**This ticket is strictly additive: emit the generic event *alongside* the existing
`onVoiceAgentSwitch`/`mountAgentVoiceControls` calls — do NOT remove them here.** The
core→voice calls are deleted only in ticket 04, after voice subscribes to the bus.
Removing them in this ticket would sever agent-switch/session refresh from voice
until 04 lands (a multi-PR window of broken voice). Same rule for every other
core→feature call: add the generic emit, leave the legacy call until 04.

## Tasks

1. Implement registry + bus with the ticket-01 contract.
2. Wire `renderSlot` into each zone render site (still no feature registered — zones
   render empty).
3. Add bus `emit` calls **next to** existing `onVoiceAgentSwitch`-style core→feature
   calls (additive — the legacy calls stay until ticket 04).
4. Unit tests: ordering, gate re-evaluation on event, teardown-on-rerender, error
   isolation, dedupe-by-id, empty-zone no-op.

## Acceptance criteria

- All existing UI behaves identically (zones render empty; voice still works via its
  current hardcoded path **which remains fully intact** — not yet migrated, that's
  ticket 04). Generic bus events fire in parallel with the legacy voice calls.
- `npm run test:js` green; new tests cover the registry/bus invariants above —
  including a test that updating one contribution does **not** tear down a sibling.
- No feature-name strings (`voice`, etc.) appear in registry/bus code.
- `registerHeaderAction` is either reimplemented as a thin shim over
  `UI.register('chat-input-actions'|'chat-header-actions', ...)` or marked deprecated
  with a migration note — there must be exactly one action-registration mechanism,
  not two. (Document the decision; do not leave both as independent live paths.)

## Risks / decisions to resolve

- **Render-site retrofitting.** Some zones (input row) are partly in `index.html`
  static markup, partly hydrated in JS. Decide per-zone whether the anchor is a
  static `<div data-slot="...">` placeholder or a JS-located anchor. Prefer static
  placeholders where the markup exists — they are the stable contract.
- **Double-render guards.** Agent cards re-render on list refresh; ensure teardown
  fires so contributions don't leak DOM/listeners (voice currently manages this by
  hand).
