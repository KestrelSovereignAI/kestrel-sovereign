---
type: Architecture Spec
title: UI Theme File Schema
description: 'Theme files define the user-facing labels for the Kestrel UI under the
  theme + i18n system established by epic #986. Each file pairs one theme with one
  locale.'
resource: /docs/architecture/ui_theme_schema.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
generated: false
privacy: public
---

# UI Theme File Schema

Theme files define the user-facing labels for the Kestrel UI under the theme + i18n system established by epic #986. Each file pairs one theme with one locale.

## File layout

```
kestrel_sovereign/themes/<theme>/<locale>.toml
```

Currently shipped:

- `kestrel_sovereign/themes/legacy/en.toml` — labels as rendered on `main` today (zero-change UX)
- `kestrel_sovereign/themes/falconry/en.toml` — falconry vocabulary (Mews, Eyrie, Hatchery, etc.)
- `kestrel_sovereign/themes/plain/en.toml` — plain English suitable for outside contributors

Theme names and locale codes are case-sensitive lowercase. Locale codes follow ISO 639-1 (`en`, `es`, `fr`, …) optionally with a region tag (`en-US`).

## Required fields

Every theme file MUST contain:

```toml
schema_version = "1"
theme = "<theme-name>"
locale = "<locale-code>"

[labels]
<key> = "<value>"
# ...
```

- `schema_version` — string. Currently `"1"`. Bumped if the schema changes incompatibly.
- `theme` — string. Must match the parent directory name.
- `locale` — string. Must match the file basename (without `.toml`).
- `[labels]` — table mapping snake_case keys to display strings.

## Key conventions

- **snake_case**, ASCII only.
- **Surface-prefixed** where natural: `tab_*`, `sidebar_*`, `btn_*`, `loading_*`, `chat_*`, `memories_*`, `tasks_*`, `sovereignty_*`, `resources_*`, `metrics_*`, `spawn_*`, `features_*`, `security_*`.
- Cross-cutting keys may be flat: `btn_refresh`, `filter_all`, `loading_generic`.
- Keys never carry HTML, whitespace, or formatting characters. Values may include HTML entities (e.g. `&#8679;` for the ⇧ glyph).

## Identical key sets across a theme

For a given locale, every theme file at that locale MUST have the same key set. Adding a new key requires adding it to every theme file.

Rationale: the loader (#989) resolves `(theme, locale)` to a flat label map. Different key sets across themes would make UI behavior depend on the active theme in ways that surprise users — a label visible in one theme would silently disappear in another.

## Fallback semantics

Defined and tested by the loader in #989. In summary:

- A missing key in a non-legacy theme falls back to the same key in `legacy/<locale>.toml`.
- A missing locale falls back to `en`.
- A missing theme is a 404, not a fallback.
- Every fallback emits a structured warning so the gap is visible in logs.

## Authoring guidance

When adding a new label:

1. Decide the classification (`theme` / `locale` / `mech` / `data`) per [`ui_label_inventory.md`](./ui_label_inventory.md).
2. `mech` and `data` strings don't get keys at all.
3. `locale` strings get the same value across all three theme files at the same locale (English values are identical across `legacy/en.toml`, `falconry/en.toml`, `plain/en.toml`).
4. `theme` strings get values that match the theme's character — see the inventory's illustrative mapping for the keys that vary.

When the legacy theme value should be unambiguously equivalent to the on-screen text on `main` (the legacy theme is the rendering on first load, before any user picks a theme).

## Validation checklist (manual until #989 ships)

- [ ] `schema_version`, `theme`, `locale` fields match file location
- [ ] `[labels]` key set matches `legacy/en.toml` exactly
- [ ] All values are non-empty strings
- [ ] No HTML or template syntax in keys
- [ ] Values that contain HTML entities are escaped correctly in TOML strings
