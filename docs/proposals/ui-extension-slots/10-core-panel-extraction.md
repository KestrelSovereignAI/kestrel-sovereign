# 10 — Extract a core panel into a feature package (second proof / north star)

**Type:** Refactor + feature extraction (zero user-visible change)
**Depends on:** 02 (registry), 03 (capability derivation), 05 (manifest delivery), 06 (panel + panel-section zones)
**Risk:** High — but it is the proof that the monolith→feature direction is real

## Why this ticket exists

The epic's thesis is that core panels (Security, Tasks, Spawn, Sovereignty, Memory,
Metrics, …) live in the core static tree only because Kestrel **started as a
monolith**, not because that is the right architecture. The target is that each such
panel is the UI of a *feature package*, contributed through the same extension
mechanism as any third-party feature. Voice (ticket 04) proves an **inline-injecting**
feature fits the slots. This ticket proves a **whole panel** can leave core and live
in a feature package — closing the loop and de-risking every future extraction.

## Scope: ONE panel, end to end

Pick a single, relatively self-contained panel as the pilot. Candidate criteria:
- Backed by an existing feature package (so the UI has a natural home).
- Limited cross-panel coupling and no shared-singleton negotiation (avoid the
  model-selector-lock class — that is voice/ticket 09 territory).
- Exercises both `panel-root` and (ideally) `panel-section`.

Likely candidates: **Spawn**, **Metrics**, or **Tasks** — each maps to a backend
feature and is mostly read-plus-actions. **Security/Approvals is explicitly NOT the
pilot** (it has event-driven modals gated by a capability union and is the highest
cross-coupling — extract it later, once the pattern is boring). Final pick is part of
the ticket's first task, justified against the criteria.

## Tasks

1. Choose the pilot panel; document why against the criteria above.
2. Move its frontend (panel JS, templates, CSS) out of core `static/` into the
   backing feature package; serve it via `get_ui_contributions()` (ticket 05).
3. Register its nav tab + panel body via the `nav-tabs`/`panel-root` zones (ticket 06)
   instead of the hardcoded `index.html` list + `setLazyLoaders` in
   [app.js:53](../../../kestrel_sovereign/static/js/app.js). Remove its entries from
   those hardcoded lists.
4. Derive its capability from the feature's enabled state (ticket 03); remove its
   static `CAPABILITY_KEYS` default and its `PANEL_CAPABILITIES`
   ([identity.js:75-88](../../../kestrel_sovereign/static/js/identity.js)) entry.
5. Verify: panel behaves identically; disabling the feature removes the tab/panel and
   flips the capability; re-enabling restores it — all without reload.

## Acceptance criteria

- **Zero user-visible change** when the feature is enabled.
- The pilot panel's JS no longer exists in the core `static/` tree; it loads via the
  manifest like any out-of-tree feature.
- `grep` for the pilot panel id in `index.html`, `app.js` (`setLazyLoaders`/imports),
  and `PANEL_CAPABILITIES` returns nothing — core no longer hardcodes it.
- Disable/enable the backing feature adds/removes the panel at runtime.
- A short **extraction playbook** is written from this experience, so the remaining
  panels can follow it. The playbook (not just the one migration) is the deliverable —
  it is what makes the north star tractable.

## Explicitly out of scope

- Extracting all panels. This is one pilot + a playbook. A big-bang extraction is a
  separate, later body of work and must not block this proof.
- Any panel requiring shared-singleton negotiation (ticket 09) or the chrome/embed
  host-mode toggle.

## Findings loop

If the pilot extraction needs a capability the registry/manifest/derivation does not
provide, that is a gap in 02/03/05/06 to fix there — not a special case here. As with
voice, the value of doing one for real is to surface those gaps before committing to
extracting the rest.
