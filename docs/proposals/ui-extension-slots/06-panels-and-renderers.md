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
- **Migrate at least one existing core panel** (e.g. Metrics or Spawn) onto the
  mechanism so there is a single code path, not a plugin sidecar bolted next to a
  hardcoded list. (Do not migrate all core panels in this ticket — one is the proof.)

## Part B — Chat-renderer registry

Voice renders realtime tool cards by delegating to core `addMessage` /
`finalizeStreamingMessage` / `renderToolCardsHtml`. Generalize: a feature registers a
renderer keyed by **tool name** or **content type**, and the chat pipeline dispatches
to it.

- Hook into the existing chat rendering dispatch (the sanitize / mermaid / katex /
  tool-sentinel pipeline) rather than creating a parallel one — survey it first
  (`static/shared/markdown/*`, chat.js tool-card rendering) and map onto the existing
  dispatch point.
- Contract: `registerRenderer({match: {tool?|contentType?}, render(payload, ctx) =>
  string | {html: string} | DocumentFragment})`. The renderer **returns** markup/a
  fragment; it is **not** handed the live target element. Core sanitizes the returned
  value and inserts it. This makes sanitization enforceable *by construction* — a
  renderer cannot write to the DOM before DOMPurify runs.
  - Rationale: a `render(el, ...)` contract that hands over the live element cannot
    guarantee sanitization — a buggy/adversarial renderer could assign `el.innerHTML`
    or attach inline handlers before core ever sees the output. Return-based output is
    the only contract where the sanitization guarantee actually holds.
  - If a renderer genuinely needs imperative DOM (e.g. a canvas/chart), it returns a
    placeholder element via the sanitized fragment and attaches behavior in a
    follow-up `mount(rootEl, ctx)` callback that operates **only** on the element core
    created from the sanitized markup — never on arbitrary core DOM.
- Renderers are gated by capability like every other contribution.
- **Sanitization is non-negotiable and structural:** all returned markup flows through
  the same DOMPurify path as core
  ([static/shared/markdown](../../../kestrel_sovereign/static/shared/markdown)). The
  contract shape (return, don't mutate) is what enforces it.

## Tasks

1. `nav-tabs` + `panel-root` zones + contribution shape; migrate one core panel.
2. Chat-renderer registry hooked into the existing rendering dispatch.
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
