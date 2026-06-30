# 05 — Manifest-driven out-of-tree UI asset loading

**Type:** Feature (backend + frontend + Feature base class)
**Depends on:** 02, 03
**Risk:** Medium

## Goal

Let a **pip-installed, out-of-tree** feature ship its own frontend JS/CSS and have it
loaded at boot — so a feature in its own repo can register slot contributions without
its assets living in the core `static/` tree (which is how voice does it today, and
why voice isn't truly pluggable).

## Design

### `Feature.get_ui_contributions()`

New optional method on
[features/base.py](../../../kestrel_sovereign/features/base.py), mirroring the
existing `get_router()` pattern:

```python
def get_ui_contributions(self) -> Optional[UIContributions]:
    """Static assets + entry modules this feature contributes to the web UI.

    Returns a manifest descriptor or None. Mirrors get_router(): the server
    discovers it after all features load.
    """
    return None
```

`UIContributions` declares: the feature's capability key (default = feature name,
keeps ticket 03 in sync), a static asset directory, and an ordered list of ES module
entry paths (resolved relative to the served static mount).

### Server

1. For each feature with a static dir, mount it at `/features/{name}/static/`
   (mirror [server.py:514-523](../../../kestrel_sovereign/server.py) static mounting;
   reuse the `_mount_feature_routers` discovery pattern at
   [server.py:209](../../../kestrel_sovereign/server.py)).
2. Expose `GET /api/ui/contributions` returning the merged manifest:
   `[{feature, capability, modules: ["/features/voice/static/ui.js"], css: [...]}]`,
   filtered to **enabled** features.

### Frontend boot

`app.js` boot sequence becomes:
1. fetch config + capabilities (ticket 03)
2. init registry + bus (ticket 02)
3. fetch `/api/ui/contributions`, dynamically `import()` each enabled feature's
   modules in declared order
4. each module calls `UI.register(...)`
5. first `renderSlot` pass

### Isolated-venv features

Out-of-process features
([isolated_runtime.py](../../../kestrel_sovereign/features/isolated_runtime.py))
forward their router via `ProxyFeature` ([line ~175](../../../kestrel_sovereign/features/isolated_runtime.py)).
`get_ui_contributions()` must be **forwardable** through the same proxy handshake, or
isolated features cannot contribute UI. Either:
- (a) the manifest is returned over the SDK init handshake and the host serves the
  assets it received, or
- (b) the host proxies `/features/{name}/static/` to the isolated service.
Decide and document; (a) is simpler and keeps asset serving in one place.

## Tasks

1. Add `get_ui_contributions()` + `UIContributions` type to the Feature base class.
2. Server: per-feature static mount + `/api/ui/contributions` endpoint (enabled-only).
3. Frontend: dynamic-import boot loader honoring declared order + capability gate.
4. Extend `ProxyFeature` to forward UI contributions for isolated-venv features.
5. Provide a **reference example feature** (smallest possible: one button in
   `chat-input-actions`) living *outside* the core static tree, proving end-to-end
   out-of-tree contribution.

## Acceptance criteria

- The reference feature, pip-installed from its own location, mounts a working
  button via the manifest with **no** edits to core `static/` or `app.js`.
- Disabled features contribute nothing (manifest is enabled-filtered).
- Isolated-venv reference path validated (at least one of (a)/(b) working).

## Security (must address, do not hand-wave)

- Feature JS runs **same-origin** with full DOM + session access. State explicitly in
  docs: **installed = trusted** (pip already grants arbitrary code execution; UI JS
  is no greater privilege than the feature's Python).
- No remote module URLs in the manifest — only paths under the host's own
  `/features/.../static/`. Reject absolute/cross-origin module URLs server-side.
- CSP review: ensure dynamic `import()` of same-origin module paths is permitted and
  cross-origin is not.
- This ticket does **not** enable untrusted/marketplace UI (out of scope per epic).
