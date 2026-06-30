# 04 — Migrate voice UI onto the slot registry (the proof)

**Type:** Refactor (zero user-visible change)
**Depends on:** 02, 03
**Risk:** High (it touches the most-coupled feature) — but **the point of the epic**

## Goal

Re-express **every** voice UI injection through the slot registry + event bus +
derived capability, and **delete the core→voice coupling**. This is the validation
gate for the entire epic: if voice cannot be expressed without escape hatches, the
contracts from 01/02 are wrong and must change *here*, before any external feature
depends on them.

## The couplings to remove

| Coupling to delete | Replaced by |
|---|---|
| named call `initVoiceUI()` from [app.js:77](../../../kestrel_sovereign/static/js/app.js) | replaced by a **bare side-effect import** `import './voice/ui.js'` whose module body self-registers via `UI.register(...)`. The import line stays in app.js through this ticket (voice assets are still in-tree); ticket 05 removes it once the manifest loader imports voice as an out-of-tree module. **Do not delete the import in this ticket — only the named `initVoiceUI()` call.** |
| `mountAgentVoiceControls(item, ...)` call in [identity.js:758](../../../kestrel_sovereign/static/js/identity.js) | `UI.renderSlot('agent-card-actions', ctx)` (core) + voice registration |
| `onVoiceAgentSwitch(...)` call in [identity.js:817](../../../kestrel_sovereign/static/js/identity.js) | core `bus.emit('agent:switch', ...)`; voice subscribes |
| mic button `insertBefore(#send-button)` [voice/ui.js:454](../../../kestrel_sovereign/static/js/voice/ui.js) | `chat-input-actions` zone registration |
| footer badge/banner [voice/ui.js:483](../../../kestrel_sovereign/static/js/voice/ui.js) | `input-footer-status` zone registration |
| picker modal → `document.body` [voice/ui.js:505](../../../kestrel_sovereign/static/js/voice/ui.js) | `modal-root` zone |
| direct `subscribeSSE('tools_updated')` [voice/ui.js:258](../../../kestrel_sovereign/static/js/voice/ui.js) | bus event bridged in ticket 02 |
| realtime tool cards / bubbles via `addMessage` | `chat-message-renderers` (ticket 06) — if 06 not yet landed, voice keeps calling `addMessage` directly and this row defers to 06 |

## Tasks

1. Convert each voice mount function into a `UI.register({slot, gate, order,
   render, events})` call. `gate: ctx => ctx.api.hasCapability('voice')` everywhere
   voice currently checks it manually
   ([voice/ui.js:224/298](../../../kestrel_sovereign/static/js/voice/ui.js)).
2. Replace voice's bespoke refresh (`refreshAgentVoiceCard`) with reliance on
   registry re-render driven by `session:change` / `agent:switch` bus events.
3. Delete the core→voice **named calls** in app.js (`initVoiceUI()`) and identity.js
   (`mountAgentVoiceControls`, `onVoiceAgentSwitch`). Keep the bare
   `import './voice/ui.js'` side-effect import in app.js so the module still loads and
   self-registers — that import is removed by ticket 05's manifest loader, not here.
4. Leave the model-selector lock **as-is** for now — it is ticket 09's subject. Note
   it explicitly in the PR so it is not mistaken for an oversight.
5. Manual + Kestrel Eye verification: mic button, agent-card 🎧/🎤, footer badge,
   picker modal, push-to-talk, agent switch refresh, multi-agent panes — all behave
   identically.

## Acceptance criteria

- **Zero user-visible change.** Voice works exactly as before.
- `identity.js` no longer references voice by name (no `mountAgentVoiceControls` /
  `onVoiceAgentSwitch`). `app.js` retains exactly **one** voice reference — the bare
  `import './voice/ui.js'` side-effect import — removed in ticket 05. The
  model-selector lock in chat.js is the other allowed remaining reference, pending
  ticket 09. No *named-call* coupling remains in either file.
- Every row in the coupling table above is closed or explicitly deferred to a named
  ticket with rationale.
- `npm run test:js` green.

## Why high-risk and how we de-risk

Voice has subtle stateful behavior (per-agent sessions, locks, SSE). De-risk by:
- Migrating zone-by-zone behind the *already-shipped* registry (02), each zone a
  separate reviewable commit.
- Keeping the voice *logic* (session management, WebRTC, etc.) untouched — only the
  **mount/refresh/gate** plumbing moves. This is a plumbing refactor, not a rewrite.
- Vision-verified E2E (Kestrel Eye) as the regression gate, not just unit tests.

## Findings loop

If any voice behavior resists the slot model, **do not** add a voice-specific escape
hatch. File the gap against ticket 01/02 (reshape the contract) or ticket 09 (if it
is a singleton-negotiation, not a mount). The whole value of doing voice first is to
surface these now.
