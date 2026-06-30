---
type: Architecture Spec
title: Panel extraction playbook
description: Step-by-step procedure for extracting a core Sovereign Console panel
  into a feature package via get_ui_contributions(), using the Spawn panel
  extraction (#2048) as the worked reference so remaining panels can follow it.
resource: /docs/architecture/features/PANEL_EXTRACTION_PLAYBOOK.md
tags:
- docs
- architecture
- architecture-spec
- features
timestamp: '2026-06-30T00:00:00Z'
status: current
owner: architecture
canonical: true
generated: false
privacy: public
---

# Panel extraction playbook

Epic #2038 (UI extension slots) makes the Sovereign Console's nav panels a
data-driven, feature-contributable surface. Ticket 10 — the "north star" —
extracted the **Spawn** panel out of core `static/` and into the in-tree
`features/spawn` package, so the panel now leaves core exactly the way a
third-party feature panel would. **The playbook, not just the one migration,
is the deliverable**: this document is the repeatable procedure for extracting
any remaining core panel (Tasks, Sovereignty, Resources, Security, …).

Spawn (#2048) is the worked reference throughout. File/line anchors are to the
state after that extraction.

## What "extracted" means

Before: the panel's nav tab and `#panel-<id>` body lived in
`kestrel_sovereign/static/index.html`; its loader was wired into
`app.js`'s `setLazyLoaders({...})`; its capability gate lived in
`identity.js`'s `PANEL_CAPABILITIES`; its JS lived in core `static/js/`.

After: **none** of those core entries exist. The feature package owns the panel
JS, declares it through `Feature.get_ui_contributions()`, and the frontend boot
loader (`ui-ext/feature-loader.js`, `loadFeatureUIContributions()`) `import()`s
it. The module self-registers its nav tab + panel body through the panel
registry (`ui-ext/panels.js`, `registerPanel()`), and its capability is derived
from the feature's enabled state (#2041). Disabling the feature removes the
tab/panel and flips the capability at runtime — no page reload, no server
restart.

## Steps

### 1. Move the panel JS into the feature package `static/` dir

Move `kestrel_sovereign/static/js/<panel>.js` →
`kestrel_sovereign/features/<feature>/static/<panel>.js`.

Two import rules for the relocated, out-of-tree asset:

- Imports of **core** singletons must use the host-served absolute path, not a
  relative one, so the asset shares the SAME registry/bus/API singletons core
  uses. Spawn imports `'/js/api.js'`, `'/js/ui-ext/panels.js'`,
  `'/js/ui-ext/bus.js'` — NOT `'../...'`. A relative import would resolve under
  `/features/<feature>/static/` and load a second module instance with its own
  registry, so `registerPanel()` would write to a registry core never reads.
- The panel body markup (the old `#panel-<id>` innerHTML from `index.html`)
  moves into the module as a string the `render` callback injects. Preserve
  `data-label-key` attributes so the i18n layer still translates the
  dynamically-injected DOM (the inline text is only the English fallback).

### 2. Self-register the panel through the panel registry

In the moved module, call `registerPanel({...})` (from
`/js/ui-ext/panels.js`):

```js
registerPanel({
    panelId: 'spawn',
    label: 'Spawn',
    labelKey: 'tab_spawn',
    before: 'features',                 // preserve the original nav position
    gate: () => API.hasCapability('spawn'),
    render: renderSpawnPanel,           // lazily fills the body on first activation
});
```

`render` runs once, the first time the tab is activated (mirrors the old
`setLazyLoaders` lazy semantics). For a feature panel the registry creates the
`#panel-<id>` container and `.panel-content` wrapper; `render(bodyEl)` fills the
wrapper.

Drive data refresh off the registry's lifecycle events, not the old per-click
dispatch: `bus.on('panel:shown', ...)` to (re)load when the panel becomes
visible.

### 3. Declare the assets from `get_ui_contributions()`

Add the hook to the feature class. The `static_dir` is the package's `static/`
dir; `modules` are the entry ES modules the boot loader `import()`s; the
`capability` is what the contribution gates on:

```python
def get_ui_contributions(self):
    from kestrel_sovereign.features.base import UIContributions
    static_dir = str((Path(__file__).parent / "static").resolve())
    return UIContributions(
        modules=["spawn.js"],
        static_dir=static_dir,
        capability="spawn",
    )
```

The server mounts `static_dir` at `/features/{name}/static/` and
`compute_ui_manifest()` (`ui_contributions.py`) serves the resolved module URLs
from `GET /api/ui/contributions`.

### 4. Derive the capability (#2041)

The capability the panel gates on is derived from the feature's enabled state —
no static map. Set `UIContributions.capability` explicitly (Spawn pins
`"spawn"`); if omitted it defaults to the feature's registry name. The frontend
`API.hasCapability(...)` reflects the live, enabled-filtered capability set so
the gate re-evaluates when the feature is enabled/disabled at runtime.

### 5. Delete the core entries

Remove every trace of the panel from core:

- **`static/index.html`** — delete the `<button class="nav-tab" data-panel="…">`
  and the `<div id="panel-…">` body.
- **`static/js/app.js`** — delete the panel's entry from the
  `setLazyLoaders({...})` call and any import of its old loader.
- **`static/js/identity.js`** — delete the panel's `PANEL_CAPABILITIES` entry.
  Registry-contributed panels carry `data-panel-registry="true"` and are
  re-gated by the registry's own `gate`, so they MUST NOT also appear in
  `PANEL_CAPABILITIES` (that would double-gate them).

After this step, grep the core tree for the panel id — only comments should
remain.

### 6. Handle runtime enable: mount assets for disabled-at-startup features

The 404 trap (#2048): assets were mounted only for features ENABLED at startup.
A feature that starts disabled and is enabled later from the Feature Store would
surface its contribution in `/api/ui/contributions` while `/features/…/static/…`
was never mounted — the dynamic `import()` 404s and the tab never appears until
a restart.

Fix: `_mount_feature_ui_assets` (`server.py`) mounts every feature that declares
a `static_dir`, **including disabled ones**
(`feature_static_mounts(agent, include_disabled=True)`). The manifest still lists
only enabled features, so a disabled feature's mount is dormant until enabled —
then it serves immediately, no restart. The feature static mounts are auth-exempt
exactly like core `/static` and `/js/` (browsers can't attach `X-API-Key` to
`<link>`/`import()`).

### 7. Handle runtime disable: tear the panel's work down

The teardown leak (#2048): when a registry-owned panel gates off while it is the
ACTIVE panel, the registry detaches the `#panel-<id>` node. A bare detach fires
no `active`-class mutation, so panel code that keys teardown off losing `active`
(Spawn's auto-refresh `MutationObserver`) never stops — the interval keeps
issuing hidden `/api/spawn/children` requests.

Fix: `_syncNav` (`ui-ext/panels.js`) runs the deactivation path BEFORE detaching
— it strips the `active` class (drives the observer path) and emits a
`panel:hidden` bus event (the deterministic path). The panel module subscribes
to that to stop its work:

```js
bus.on('panel:hidden', (payload) => {
    if (payload && payload.panelId === 'spawn') stopAutoRefresh();
});
```

Any panel that starts background work (intervals, observers, sockets) on
activation MUST stop it on `panel:hidden`.

### 8. Handle multi-agent host mode: pin assets to the selected agent

The wrong-agent trap (#2048): in multi-agent host mode the UI is served by the
host (`host.py`), a thin proxy that owns no feature static mounts. The host runs
agents as separate processes that may have **heterogeneous** feature sets — the
selected agent can have a feature enabled (and its `static_dir` mounted) while
the first-configured agent does not. The per-agent manifest is fetched
host-agent-prefixed, but the module/css URLs it carries are root-relative
(`/features/{slug}/static/…`). A host route that proxied those to a *fixed* first
agent would 404 whenever that agent lacks the feature — and serves the wrong code
in the general case.

Fix (the simpler robust option of the two in #2048): **pin feature-static URLs to
the selected agent on the frontend**, not on the host. `pinFeatureAssetUrl`
(`ui-ext/feature-loader.js`) rewrites a `/features/…` URL to
`/api/agents/{selected}/features/…` via `API.buildAgentUrl`, so the existing
generic `/api/agents/{id}/{path}` proxy forwards it to the exact agent whose
manifest declared the contribution. The host exempts that path from API-key auth
via `FEATURE_STATIC_ASSET_RE` (header-less `import()` again). Core-bundled
`/js/…` assets are NOT pinned — the host serves those directly. In standalone
mode `buildAgentUrl` is a no-op, so the URL stays root-relative and the server's
own `/features/{slug}/static/` mount serves it — one manifest, both modes.

Why pin on the client rather than serve on the host: serving on the host would
force it to discover and mount every installed feature's `static_dir` (it has no
agent), duplicating `_mount_feature_ui_assets` and assuming a homogeneous
install. Pinning reuses the proxy that already exists and routes to the agent
that genuinely serves the asset.

### 9. Tests

Mirror the Spawn coverage:

- **Runtime-enable serving** — assert a disabled feature's `static_dir` is still
  mounted/served (`tests/integration/test_feature_ui_runtime_enable.py`,
  `tests/unit/test_ui_contributions_runtime_enable.py`).
- **Multi-agent host delivery** — assert the host serves an agent-pinned
  feature-static asset header-less, and keeps feature *API* routes protected
  (`tests/integration/test_host_proxy_integration.py`); assert the loader pins
  `/features/…` URLs but not `/js/…`
  (`tests/frontend/feature_ui_contributions_loader.test.mjs`).
- **Teardown on disable** — assert the panel stops its work (interval cleared /
  `panel:hidden` fired) when its gate flips off
  (`tests/frontend/ui_ext_panels_teardown.test.mjs`).

## Checklist

- [ ] Panel JS moved to `features/<feature>/static/`, core imports via `/js/...`
- [ ] `registerPanel({...})` with a capability `gate` + `render`
- [ ] Data refresh wired to `panel:shown`; teardown wired to `panel:hidden`
- [ ] `get_ui_contributions()` returns `static_dir` + `modules` + `capability`
- [ ] Removed from `index.html`, `app.js` `setLazyLoaders`, `PANEL_CAPABILITIES`
- [ ] Server mounts disabled features' assets (`include_disabled=True`)
- [ ] Multi-agent host: frontend pins `/features/…` URLs to the selected agent
- [ ] Tests for runtime-enable serving + host delivery + teardown-on-disable

## References

- Epic #2038 (UI extension slots); ticket 06 (panel registry); ticket 10 (#2048,
  Spawn extraction — the north star).
- Capability derivation: #2041.
- Panel registry: `kestrel_sovereign/static/js/ui-ext/panels.js`.
- Manifest + mounts: `kestrel_sovereign/ui_contributions.py`,
  `kestrel_sovereign/server.py` (`_mount_feature_ui_assets`).
- Reference extraction: `kestrel_sovereign/features/spawn/`.
