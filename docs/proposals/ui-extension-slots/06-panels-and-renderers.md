# 06 — Panel contributions + chat-renderer registry

**Type:** Feature (frontend)
**Depends on:** 02, 05
**Risk:** Medium

## Goal

Apply the same registry pattern outward to the two remaining contribution kinds:
whole **nav panels** and custom **chat-message renderers**. This subsumes the
original "panel" idea and the chat-rendering idea into the slot framework rather than
building them as separate mechanisms.

## Part A — Panel contributions

Today nav tabs are a static list in `index.html` and a hardcoded import +
`setLazyLoaders({...})` in [app.js:53](../../../kestrel_sovereign/static/js/app.js).
Convert to data-driven:

- `nav-tabs` zone: a contribution declares `{label, icon, panelId, order, gate}`.
- `panel-root` zone: the panel body, lazily rendered on first activation (preserve
  the existing lazy-load semantics from `setLazyLoaders`).
- **Data-drive at least one existing core panel** (e.g. Metrics or Spawn) through the
  registry so there is a single code path, not a plugin sidecar bolted next to a
  hardcoded list. **Scope boundary:** here the panel stays in the core `static/` tree —
  this proves the *registry/nav path*. Relocating a panel out into a feature package
  (with capability derivation + manifest delivery) is ticket 10. 06 = data-drive in
  place; 10 = extract out-of-tree. Do not conflate them.

## Part B — Chat-renderer registry

**Build on the existing `registerPartRenderer`, do not create a parallel system.**
The codebase already has a working type-dispatch renderer registry
([chat.js:3583](../../../kestrel_sovereign/static/js/chat.js)): error-isolated,
realm-safe, with a documented escaped-text fallback, consumed by `appendMessagePart`
in the streaming path via the `PART_SENTINEL` protocol. That is the foundation. This
ticket **generalizes it**, with two reconciliations the audit surfaced:

1. **Two distinct trust levels — keep them distinct.**
   - *Host/embedder parts* (existing `registerPartRenderer`, e.g. Frinz's selfie
     `<img>`): host-trusted markup written via `innerHTML`, **host owns
     sanitization**. This is the documented, intentional contract for code the
     embedder ships. Leave it as-is.
   - *Feature tool-output renderers* (new, this ticket): render **tool/model payload**
     — data that is NOT author-trusted. These MUST go through **core
     sanitization** and therefore use the inert-string contract below. A feature
     renderer is not the same trust class as an embedder part; do not let it use the
     host-trusted `innerHTML` path.
2. **Tool cards are a SEPARATE, positional path with no hook today.** Tool output is
   rendered by `renderToolCardsHtml` (collapsible cards segmented by stream position),
   which is **not** `registerPartRenderer`-driven and exposes **no** renderer hook.
   Rendering custom *tool* output therefore requires adding a new dispatch hook inside
   the tool-card path — it cannot be done by registering a "part". Scope that hook
   here explicitly; do not pretend parts cover it.

- Hook into the existing chat rendering dispatch (the sanitize / mermaid / katex /
  tool-sentinel pipeline) rather than creating a parallel one — survey it first
  (`static/shared/markdown/*`, chat.js tool-card rendering) and map onto the existing
  dispatch point.
- Contract (feature tool-output renderers): `registerRenderer({match: {tool?|contentType?}, render(payload, ctx) =>
  string, mount?(rootEl, ctx) => (void | () => void)})`. `render` returns **inert
  markup only — a plain HTML string** (never live DOM, never a `Node`/`Element`/
  `DocumentFragment`). Core parses and sanitizes that string itself, then creates the
  live DOM. The renderer is **not** handed the live target element in `render`. This
  makes sanitization enforceable *by construction*.
  - Rationale: any contract that lets the renderer touch live DOM defeats
    sanitization. `render(el, ...)` lets it assign `el.innerHTML`. Returning a
    `Node`/`DocumentFragment` is just as unsafe — DOMPurify sanitizes *attributes and
    elements*, but cannot remove listeners already attached via `addEventListener` on
    nodes the renderer built. Only an **inert string** that core parses guarantees the
    output passed through sanitization with no pre-attached behavior. The contract
    therefore accepts a string and nothing else.
  - If a renderer needs imperative behavior (canvas/chart, click handlers), it does so
    **exclusively** in the optional `mount(rootEl, ctx)` hook, which core invokes
    *after* it has parsed+sanitized the markup and created `rootEl`. `mount` operates
    only on `rootEl` (the element core built from the sanitized string) and its
    descendants — never on arbitrary core DOM. `mount` may return a teardown fn.
- Renderers are gated by capability like every other contribution.
- **Sanitization is non-negotiable and structural:** all returned markup flows through
  the same DOMPurify path as core
  ([static/shared/markdown](../../../kestrel_sovereign/static/shared/markdown)). The
  contract shape (return, don't mutate) is what enforces it.

## Tasks

1. `nav-tabs` + `panel-root` + `panel-section` zones + contribution shape; migrate one
   core panel. (`panel-section` is required by the Resources composite-panel pattern —
   sub-sections gated per sub-capability,
   [resources.js:29-64](../../../kestrel_sovereign/static/js/resources.js).)
2. Chat-renderer registry: extend `registerPartRenderer` for the host-trusted-part
   case (keep its contract) AND add the inert-string feature tool-output renderer +
   the new tool-card dispatch hook. One registry, two documented trust contracts.
3. Update the voice tool-card path (deferred row from ticket 04) to use the renderer
   registry — closing that ticket-04 deferral.
4. Tests: panel show/hide + lazy load; renderer dispatch by tool/content type;
   sanitization applied to renderer output.

## Acceptance criteria

- The migrated core panel behaves identically via the new mechanism.
- A feature can add a nav panel and a custom tool renderer with no core edits.
- Voice tool cards now render via the registry; ticket-04's deferred row is closed.
- All renderer output is sanitized; an adversarial renderer returning a `<script>`
  payload, an inline `onerror=` handler, or a `javascript:` href is neutralized
  (test all three). Because the contract is return-based, there is no code path by
  which a renderer can write unsanitized markup to the DOM.

## Risk

- The chat rendering pipeline is intricate (streaming, sentinels, idempotency).
  Mis-hooking it can reorder or duplicate messages. Hook at the *existing* dispatch
  seam; do not add a second rendering path.
