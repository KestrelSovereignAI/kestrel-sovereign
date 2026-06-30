# kestrel-feature-ui-slot-example

Reference feature proving **manifest-driven, out-of-tree UI asset loading**
(epic #2038, ticket #2043). It is the smallest possible end-to-end example: a
pip-installed feature, living entirely outside the core `static/` tree, that
mounts one working button into the chat input row with **no edits to core
`static/` or `app.js`**.

## How it works

1. `UISlotExampleFeature.get_ui_contributions()` returns a `UIContributions`
   whose `static_dir` points at this package's own `static/` directory and
   whose `modules` lists `ui.js`.
2. On agent start the host mounts that directory at
   `/features/ui-slot-example/static/` and the manifest at
   `GET /api/ui/contributions` advertises
   `/features/ui-slot-example/static/ui.js`.
3. The console boot loader (`app.js` → `loadFeatureUIContributions`) fetches the
   manifest and dynamically `import()`s the module.
4. `ui.js` calls `UI.register(...)` to add a button to the `chat-input-actions`
   slot.

Disable or uninstall the feature and it contributes nothing — the manifest is
enabled-filtered.

## Try it

```bash
# from a Kestrel project venv
uv pip install -e examples/kestrel-feature-ui-slot-example
kestrel restart           # remount assets + reload feature entry-points
```

Open the Sovereign Console and look for the ★ button next to the chat send
button.

## Security

`installed = trusted`. This feature's JS runs same-origin with full DOM and
session access — but installing any Python package already grants arbitrary code
execution, so its UI JS is no greater privilege. The host rejects remote /
cross-origin module URLs in the manifest; only same-origin asset paths under the
host's own `/features/.../static/` (or core-served `/js/...`) are served.
