# 03 — Capability set derived from enabled features

**Type:** Feature (backend + frontend glue)
**Depends on:** none (parallelizable with 01/02)
**Risk:** Low

## Problem

Frontend capability gating is the one good seam voice already uses
(`API.hasCapability('voice')`), but the capability set is **hardcoded**: `voice:
true` is a static default in
[api_client.mjs:27-47](../../../kestrel_sovereign/static/js/api_client.mjs), and the
host can only override via `window.KESTREL_UI_CONFIG.capabilities`
([api.js](../../../kestrel_sovereign/static/js/api.js)). Every new feature that
wants gating today requires a **core edit** to `CAPABILITY_KEYS`. That defeats the
"discovered, not curated" doctrine and blocks out-of-tree features.

## Goal

Derive the capability set from **enabled features**, which the backend already knows
([endpoints/features.py](../../../kestrel_sovereign/endpoints/features.py),
`/api/features`). A feature being enabled → its capability is true → its UI
registrations activate. Disabling a feature at runtime flips the capability and the
registry tears its contributions down.

## Design

1. Backend: extend the bootstrap config injected into `window.KESTREL_UI_CONFIG`
   (where it is rendered/served) to include a `capabilities` map computed from
   enabled features. A feature declares its UI capability key (default: its feature
   name) — see ticket 05's `get_ui_contributions()` for where this is declared so
   the two stay in sync.
2. Keep `CAPABILITY_KEYS` for **core/static** capabilities (chrome, conversations,
   etc.) but stop treating feature capabilities as static defaults — feature keys
   come from the server-computed map, merged over the static defaults.
3. Emit a `capabilities:changed` bus event (ticket 02) when the set changes at
   runtime (feature enable/disable via the Feature Store panel), so the registry
   re-gates without a page reload.
4. `hasCapability(path)` resolution
   ([api_client.mjs:703](../../../kestrel_sovereign/static/js/api_client.mjs))
   unchanged in shape — only its data source changes.

## Tasks

1. Add server-side capability computation from the enabled-feature set.
2. Inject into `window.KESTREL_UI_CONFIG.capabilities` at page render.
3. Merge precedence — **asymmetric, because a disabled feature has no assets to gate
   into:**
   - **Core/static capabilities** (chrome, conversations, …): explicit host override >
     static default (unchanged from today).
   - **Feature-derived capabilities:** server-derived *disabled* is **authoritative**
     and cannot be overridden true. A host override may only force a feature
     capability **off** (force-true on a disabled feature is ignored, with a console
     warning). Rationale: ticket 05 filters `/api/ui/contributions` to *enabled*
     features, so a forced-true-but-disabled feature would gate its UI "available"
     while its modules were never loaded — a guaranteed-broken state. Disabled means
     disabled, end to end.
   - Effective value = `serverEnabled && (hostOverride !== false)`.
4. Wire enable/disable endpoints
   ([features.py:247/268](../../../kestrel_sovereign/endpoints/features.py)) to push
   `capabilities:changed`.
5. Tests: feature enabled → capability true; disabled → false + teardown; host
   override still wins.

## Acceptance criteria

- Removing the hardcoded `voice: true` default does **not** change behavior when the
  voice feature is enabled (it is now derived).
- A new feature gains a working `hasCapability(<name>)` with **zero** edits to
  `api_client.mjs`.
- Runtime enable/disable flips UI without reload.
- A host override cannot force a *disabled* feature's capability true (verified: gate
  stays false, contributions stay torn down, manifest still excludes it). Host
  override force-*off* on an enabled feature does work.

## Risk

- **Ordering at boot.** Capabilities must be known before `app.js` runs feature
  registrations (ticket 05 manifest load). Define the boot sequence:
  config+capabilities → registry init → dynamic feature module import → first render.
