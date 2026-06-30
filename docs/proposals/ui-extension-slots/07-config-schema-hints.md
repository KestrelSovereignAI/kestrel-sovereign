# 07 — Config-schema UI hints (quick win)

**Type:** Feature (frontend + light backend)
**Depends on:** none (independent quick win; ship in parallel with the design spike)
**Risk:** Low

## Goal

Most features that "want UI" actually just want a **settings form + a status
readout**. That already exists: `config_schema`
([features/base.py:345](../../../kestrel_sovereign/features/base.py)) renders a
JSON-Schema form in the Feature Store panel
([static/js/feature-store.js](../../../kestrel_sovereign/static/js/feature-store.js),
config endpoints at
[endpoints/features.py:349-415](../../../kestrel_sovereign/endpoints/features.py)).
Extend it with light UI hints so ~70% of feature UI needs are met with **zero
per-feature frontend code** — the highest-leverage, lowest-risk move, independent of
the slot framework.

## Scope

Extend the config-schema rendering to honor optional UI hints (within JSON Schema's
own conventions where possible to avoid inventing a dialect):

- **Sections / grouping** of fields.
- **Read-only / computed status fields** (display feature state, not just editable
  config) — e.g. via a `readOnly` + a values endpoint.
- **Action buttons** that POST to the feature's own router (`get_router()`) — e.g.
  "Test connection", "Reset queue". The button declares
  `{label, method, path, confirm?}`; the action targets the feature's existing
  endpoints, not a new mechanism.
- **Widget hints** (`format`/`x-kestrel-widget`: textarea, secret/password masking,
  select-from-endpoint).

## Tasks

1. Define the hint vocabulary (documented; prefer standard JSON Schema keywords +
   one namespaced `x-kestrel-ui` object for the rest).
2. Extend the Feature Store config renderer to honor hints (sections, readonly,
   widgets, action buttons).
3. Secret masking for sensitive config (do not render API keys in plaintext).
4. One real feature adopts hints as the dogfood example.

## Acceptance criteria

- A feature gets a sectioned settings form with a masked secret field and a working
  "Test connection" action button — with **no** custom frontend code, only
  `config_schema` + an existing router endpoint.
- Plain schemas (no hints) render exactly as today (backward compatible).

## Why separate from the slot framework

This needs none of 01–06. It is the pragmatic floor that covers the common case
immediately, so the slot framework can focus on the genuinely-inline cases (voice-
style buttons) without being the only path to *any* feature UI.
