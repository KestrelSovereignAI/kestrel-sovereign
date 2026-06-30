---
type: Architecture Spec
title: Config-schema UI hints
description: Optional JSON Schema UI hints that let a feature's config_schema render
  sections, status readouts, masked secrets, and action buttons in the Feature Store
  with no per-feature frontend code.
resource: /docs/architecture/features/CONFIG_SCHEMA_UI_HINTS.md
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

# Config-schema UI hints

Most features that "want UI" actually just want a **settings form + a status
readout**. A feature exposes that by returning a JSON Schema from
`Feature.config_schema`; the Sovereign Console's Feature Store renders a form
from it (no per-feature frontend code). This document defines the optional **UI
hints** that schema can carry so the rendered form gains sections, status
readouts, masked secrets, and action buttons.

Implemented for [#2045](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2045)
(epic #2038). The dogfood adopter is `WebSearchFeature`
(`kestrel_sovereign/features/web_search/feature.py`).

## Design rule: standard keywords first

Prefer **standard JSON Schema keywords**. Only the things JSON Schema has no
equivalent for (sections, action buttons) live under a single namespaced
`x-kestrel-ui` object. There is **no** `x-kestrel-widget` keyword.

| Need | How |
|------|-----|
| Widget — multi-line text | `"format": "textarea"` |
| Widget — secret / password masking | `"format": "password"` (usually with `"writeOnly": true`) |
| Read-only / computed status field | `"readOnly": true` |
| Secret (write-only) | `"writeOnly": true` |
| Select from a fixed list | standard `"enum": [...]` |
| Sections / grouping | `x-kestrel-ui.sections` |
| Action buttons | `x-kestrel-ui.actions` |

A schema that uses **none** of these renders exactly as before — the hints are
fully backward compatible.

## Field-level keywords (standard)

- **`readOnly: true`** — rendered as a static value display, not an input. Its
  value comes from the feature's `get_config()` (so it can show live state, e.g.
  "Connected"). It is never submitted on save.
- **`writeOnly: true`** / **`format: "password"`** — treated as a *secret*. See
  *Secret semantics* below.
- **`format: "textarea"`** — multi-line text input.
- **`format: "password"`** — masked input.

## `x-kestrel-ui` (the one namespaced object)

A single optional object at the **schema root**:

```jsonc
"x-kestrel-ui": {
  "sections": [
    { "title": "Credentials", "description": "…", "fields": ["api_key"] },
    { "title": "Status", "fields": ["status"] }
  ],
  "actions": [
    { "label": "Test connection", "method": "GET", "path": "/api/features/web_search/test" }
  ]
}
```

### `sections`

Ordered list of `{ title, description?, fields: [propertyName, …] }`. Fields are
rendered in the order listed, grouped under the section title. Any property
**not** named by a section is appended in a trailing unlabelled group, so a
partial section list never hides a field.

### `actions`

Ordered list of buttons, each `{ label, method, path, confirm? }`:

- **`path`** is the full API path of an endpoint the **feature's own router**
  (`Feature.get_router()`) exposes — not a new mechanism.
- **`method`** defaults to `POST`; `GET`/`HEAD` send no body.
- **`confirm`** (optional) shows a confirmation dialog before firing.

The endpoint may return `{ "ok": boolean, "message": string }`; the UI shows the
message inline and as a toast (success/error keyed off `ok`). Any 2xx without an
`ok` field is treated as success.

## Secret semantics (write-only)

Secrets use **write-only** handling end to end — stored secret values are never
sent to the browser:

1. **GET `/api/features/{name}/config`** strips every secret field from `config`
   and instead reports presence in a `secrets_set` map
   (`{ "api_key": true }`). The plaintext value is never returned.
2. The form renders a masked (`type="password"`) input, empty, with a
   placeholder indicating whether a value is already stored.
3. **On save**, an untouched secret field is *omitted* from the PATCH. A secret
   is included only when the user typed a new value.
4. **PATCH `/api/features/{name}/config`** re-injects the stored value for any
   secret field omitted from the body before validation/save, so leaving a
   secret blank preserves it. The PATCH response also strips secrets.

For this to hold, a feature's `set_config()` should **merge** incoming values
over stored config rather than replacing wholesale (so an omitted secret keeps
its stored value). `WebSearchFeature` does this.

## Worked example — `WebSearchFeature`

```python
@property
def config_schema(self):
    return {
        "type": "object",
        "properties": {
            "api_key": {
                "type": "string", "title": "Tavily API Key",
                "writeOnly": True, "format": "password",
            },
            "status": {
                "type": "string", "title": "Provider Status", "readOnly": True,
            },
        },
        "x-kestrel-ui": {
            "sections": [
                {"title": "Credentials", "fields": ["api_key"]},
                {"title": "Status", "fields": ["status"]},
            ],
            "actions": [
                {"label": "Test connection", "method": "GET",
                 "path": "/api/features/web_search/test"},
            ],
        },
    }
```

This yields a sectioned form with a masked, write-only API-key field, a live
"Provider Status" readout, and a working **Test connection** button that hits the
feature's own router — with no custom frontend code.
