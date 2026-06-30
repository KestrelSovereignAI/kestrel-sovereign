# 01 — Extract slot taxonomy + slot registry contract (design spike)

**Type:** Design spike (produces a committed spec doc + interface stubs, no behavior)
**Depends on:** none
**Risk:** Low

## Goal

Produce the authoritative list of UI **zones** and the **contribution contract**,
derived empirically from voice's existing injection points rather than invented.
Output is a committed `SLOTS.md` spec + JSDoc/TS-style interface stubs that tickets
02 and 04 implement against.

## Why this is first

If the zone list is wrong, every downstream ticket churns. The voice recon already
enumerated the couplings; this ticket converts that into a stable, named contract
and *pressure-tests it against every voice touch point* before any code is written.

## Tasks

1. Enumerate zones from the voice surface area. Starting set (validate/extend, do
   not blindly accept):
   - `chat-input-actions` — action buttons in the chat input row, anchored relative
     to `#send-button`.
   - `agent-card-actions` — per-agent-card controls; context includes the agent
     identity and standalone-vs-multi-agent mode.
   - `input-footer-status` — status chips/badges in `.input-footer`.
   - `modal-root` — feature-owned overlay dialogs.
   - `chat-message-renderers` — custom renderers keyed by tool name / content type
     (separate contract; see ticket 06).
   - `nav-tabs` + `panel-root` — whole-panel contributions (ticket 06).
   - `panel-section` — a section contributed *into* an existing panel (finer grain
     than a whole panel). Empirically required: the Resources panel already composes
     sub-sections gated individually by `keys.agent`/`keys.user`/`keys.platform`/
     `wallet` ([resources.js:29-64](../../../kestrel_sovereign/static/js/resources.js)).
     A feature adding "its section" to Resources/Security/etc. needs this.
2. **Inventory the two pre-existing extension APIs and decide build-on vs supersede**
   (survey-platform-first — do this before designing anything new):
   - `registerPartRenderer(type, fn)`
     ([chat.js:3583](../../../kestrel_sovereign/static/js/chat.js)) — SOLID type
     dispatch; the basis for the ticket-06 renderer registry. Note its current trust
     model (host-trusted `innerHTML`, no core sanitization) and that it is a
     **separate path** from positional tool cards (`renderToolCardsHtml`), which have
     **no** hook today.
   - `registerHeaderAction(action)`
     ([chat.js:684](../../../kestrel_sovereign/static/js/chat.js)) — a documented
     embedder API (#1623/#1627) with no internal feature adoption; voice bypasses it.
     Its limitations (header-only position, full rebuild on every call, `onClick`-only,
     no per-agent context) become **explicit requirements** for the ticket-02 registry,
     which supersedes it **while preserving its public contract** (embedders depend on
     it).
   - Document `chrome`/embed mode and event-driven panel-independent UI as **out of
     slot scope** (host mode + event bus respectively).
3. For each zone define: stable string id, the **context object** passed to
   `render(el, ctx)`, the DOM anchor semantics (insert-before / append / replace),
   and which **events** can trigger re-render.
4. Define the contribution contract:
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
5. Define the per-zone **context contract** explicitly (what keys each zone
   guarantees). Cross-check against what voice reads today (`agentName`,
   standalone flag, session state, `api`).
6. Define the **event vocabulary** the bus must emit (`agent:switch`,
   `session:change`, `tools_updated`, `capabilities:changed`, `panel:shown`).
   Cross-reference voice's existing triggers: `onAgentSwitch`
   ([identity.js:817](../../../kestrel_sovereign/static/js/identity.js)),
   `refreshAgentVoiceCard`, the `tools_updated` SSE subscription
   ([voice/ui.js:258](../../../kestrel_sovereign/static/js/voice/ui.js)).
7. Identify couplings that the slot model **cannot** express and document them as
   explicitly out-of-scope-for-slots, to be handled by ticket 09:
   - model-selector lock (`acquireVoiceLock`/`releaseVoiceLock`).

## Acceptance criteria

- `docs/proposals/ui-extension-slots/SLOTS.md` committed: every zone has id,
  context contract, anchor semantics, retrigger events.
- A written walkthrough mapping **each** voice injection point from the epic table
  to a zone or to the ticket-09 escape hatch — with **no** "TBD" or unexplained
  residue. If any voice behavior cannot be expressed, that is a finding that
  reshapes ticket 02, not a thing to defer.
- Interface stub file (`static/js/ui-ext/contract.js` or `.d.ts`) with the
  typedefs, no runtime logic.

## Explicitly NOT in this ticket

- No registry implementation (ticket 02).
- No voice changes (ticket 04).
