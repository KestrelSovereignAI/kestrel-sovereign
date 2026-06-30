# Epic: Slot-based UI & CLI extension surface for features

**Status:** Proposal
**Owner:** TBD
**Labels:** `roadmap`, `architecture`, `frontend`, `features`

## Problem

Feature packages (`kestrel_sovereign.features` entry point, subclassing `Feature`
in [features/base.py](../../../kestrel_sovereign/features/base.py)) already have a
strong **backend** extension story: tools, HTTP routers (auto-mounted in
[server.py:209](../../../kestrel_sovereign/server.py), hooks, background tasks, DB
tables, config schema, lifecycle. What a feature **cannot** do today is contribute
**client-side surface**:

1. Inline widgets injected into *existing* core UI (buttons, badges, status chips).
2. Whole nav panels.
3. Custom rendering of its own tool output in the chat stream.
4. `kestrel` CLI subcommands.

The canonical proof that this is a real gap is **voice**. Voice is described as a
feature, but at the UI layer it is not pluggable at all — its frontend
(`static/js/voice/`) is **baked into the core package**, imported directly by
[app.js:77](../../../kestrel_sovereign/static/js/app.js), and core modules reach
*back into it*: [identity.js:758](../../../kestrel_sovereign/static/js/identity.js)
calls `mountAgentVoiceControls(item, ...)` when rendering every agent card. The only
thing that makes voice feel optional is one good seam: the capability flag
(`API.hasCapability('voice')`).

A second feature that wants a mic-like button on the agent card, or a chip in the
chat input bar, has **no way to do it** without editing core files. That does not
scale, and it is the opposite of the "discovered, not hand-curated" doctrine the
backend already follows.

## What voice actually needs (the surface area)

Voice is not a "panel." It injects inline widgets into a small, finite set of
*pre-existing* core surfaces. This list is the specification — we extract it from
voice rather than inventing it:

| Zone | Voice's use | Today's hardcoded coupling |
|---|---|---|
| `chat-input-actions` | mic button before `#send-button` | [voice/ui.js:454](../../../kestrel_sovereign/static/js/voice/ui.js) `insertBefore` |
| `agent-card-actions` | 🎧 listen / 🎤 talk per agent card | [voice/ui.js:297](../../../kestrel_sovereign/static/js/voice/ui.js) `appendChild`, invoked from [identity.js:758](../../../kestrel_sovereign/static/js/identity.js) |
| `input-footer-status` | path badge + privacy banner | [voice/ui.js:483](../../../kestrel_sovereign/static/js/voice/ui.js) |
| `modal-root` | voice picker dialog | [voice/ui.js:505](../../../kestrel_sovereign/static/js/voice/ui.js) → `document.body` |
| `chat-message-renderers` | realtime tool cards / bubbles | delegates to `addMessage` / `finalizeStreamingMessage` |
| `nav-tabs` + panel | (the existing panel case) | hardcoded list in [app.js:53](../../../kestrel_sovereign/static/js/app.js) + `index.html` |

Two further coupling categories that are **not** slots and must be handled
distinctly:

- **Stateful refresh.** Card buttons reflect live session state, so voice has
  `onAgentSwitch` ([identity.js:817](../../../kestrel_sovereign/static/js/identity.js))
  and `refreshAgentVoiceCard`. Slots must carry context and re-render on events.
- **Shared-singleton negotiation.** Voice *seizes* the model selector during a
  realtime session (`acquireVoiceLock`/`releaseVoiceLock`,
  [chat.js:3743](../../../kestrel_sovereign/static/js/chat.js)). This is a
  claim/release contract, not a mount point.

## Design thesis

Invert the dependency. Core stops importing features; core declares **named zones**
and renders each by iterating registered **contributions**. A feature *registers*
into a zone instead of core *calling into* the feature.

```js
UI.register('agent-card-actions', {
  gate: ctx => ctx.api.hasCapability('voice'),
  order: 10,
  render: (el, ctx) => { /* mount controls for ctx.agentName */ },
});
```

Core, when building an agent card, calls
`UI.renderSlot('agent-card-actions', {element, agentName, api})`. The slot registry
is paired with an **event bus** (`agent:switch`, `session:change`, `tools_updated`)
so contributions re-render without each feature reinventing voice's bespoke refresh
plumbing.

Gating stays capability-based, but the capability set is **derived from enabled
features** (backend already knows via `/api/features`) and surfaced through
`window.KESTREL_UI_CONFIG.capabilities` — not hardcoded as a default in
[api_client.mjs:46](../../../kestrel_sovereign/static/js/api_client.mjs).

Out-of-tree delivery: a feature serves its JS under `/features/{name}/static/`,
declares an entry module in a `/api/ui/contributions` manifest (built from a new
`Feature.get_ui_contributions()`), and `app.js` dynamically imports those modules at
boot; each calls `UI.register(...)`.

## Proof obligation (non-negotiable)

We do **not** ship the registry and hope features fit. We build the taxonomy +
registry + event bus, then **re-express voice entirely through it** and delete the
core→voice coupling. Voice is the hardest existing case; if the slots express it
with zero escape hatches, the design is validated; if they cannot, we found the gap
before exposing it to feature authors. This migration is a refactor with **zero
user-visible change** — the safest way to introduce a framework.

## Tickets

| # | Ticket | Depends on | Risk |
|---|---|---|---|
| 01 | [Extract slot taxonomy + slot registry contract (design spike)](01-slot-taxonomy.md) | — | Low |
| 02 | [Client-side slot registry + event bus](02-slot-registry.md) | 01 | Med |
| 03 | [Capability set derived from enabled features](03-capability-derivation.md) | — | Low |
| 04 | [Migrate voice UI onto the slot registry (the proof)](04-voice-migration.md) | 02, 03 | High |
| 05 | [Manifest-driven out-of-tree UI asset loading](05-manifest-loading.md) | 02, 03 | Med |
| 06 | [Panel contributions + chat-renderer registry](06-panels-and-renderers.md) | 02, 05 | Med |
| 07 | [Config-schema UI hints (quick win)](07-config-schema-hints.md) | — | Low |
| 08 | [CLI extension via entry-point group](08-cli-entrypoint-group.md) | — | Low |
| 09 | [Shared-singleton claim/release collaboration API](09-singleton-claims.md) | 02, 04 | Med |

## Sequencing rationale

- **07 and 08 are independent quick wins** — ship them in parallel with the design
  spike; they need no registry and deliver immediate value.
- **01 → 02 → 03 → 04** is the spine. 04 (voice migration) is the gate: nothing in
  05/06/09 is exposed to external feature authors until voice proves the contracts.
- **09** is deliberately *after* 04 because the model-selector lock is the one
  coupling we expect the slot model *not* to absorb; we want to discover its real
  shape during the voice migration, not guess it up front.

## Out of scope

- Server-rendered template injection (Jinja). The frontend is a static SPA; all
  contribution is client-side JS + JSON manifest.
- A marketplace / remote-loaded UI from untrusted origins. Features are
  pip-installed (already an arbitrary-code-execution trust boundary); their JS runs
  same-origin and is no less trusted than their Python. Remote/untrusted UI is a
  separate security epic.

## Cross-cutting risks

- **Trust / XSS.** Feature JS runs in the main origin with full DOM + session
  access. Mitigation: served from per-feature paths, no remote `eval`, CSP review,
  explicit "installed = trusted" statement (see ticket 05).
- **Isolated-venv features.** Out-of-process features
  ([isolated_runtime.py](../../../kestrel_sovereign/features/isolated_runtime.py))
  forward their router via `ProxyFeature`; `get_ui_contributions()` must be
  forwardable the same way or they cannot contribute UI (see ticket 05).
- **Render ordering / DnD.** Multiple features in one zone need deterministic order
  and must not assume exclusive ownership of an anchor (see ticket 02).
