# 02 — Client-side slot registry + event bus

**Type:** Feature (core frontend)
**Depends on:** 01
**Risk:** Medium

## Goal

Implement the runtime that contributions register into and that core renders zones
through. Two collaborating primitives: a **slot registry** and an **event bus**.

## Design

### Slot registry (`static/js/ui-ext/registry.js`)

```js
export const UI = {
  register(contribution) { /* validate against contract, store, dedupe by id */ },
  unregister(slot, id) { /* ... */ },
  renderSlot(slot, ctx) { /* sort by order, gate, render, track teardowns */ },
  refreshSlot(slot) { /* re-render last-known ctx for a zone */ },
};
```

- `renderSlot` is idempotent: re-invoking with the same element clears prior
  contribution mounts (calling their teardown fns) before re-rendering. This is what
  makes voice's `refreshAgentVoiceCard` fall out for free.
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
- `npm run test:js` green; new tests cover the registry/bus invariants above.
- No feature-name strings (`voice`, etc.) appear in registry/bus code.

## Risks / decisions to resolve

- **Render-site retrofitting.** Some zones (input row) are partly in `index.html`
  static markup, partly hydrated in JS. Decide per-zone whether the anchor is a
  static `<div data-slot="...">` placeholder or a JS-located anchor. Prefer static
  placeholders where the markup exists — they are the stable contract.
- **Double-render guards.** Agent cards re-render on list refresh; ensure teardown
  fires so contributions don't leak DOM/listeners (voice currently manages this by
  hand).
