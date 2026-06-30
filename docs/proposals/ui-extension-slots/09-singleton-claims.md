# 09 — Shared-singleton claim/release collaboration API

**Type:** Feature (frontend)
**Depends on:** 02, 04
**Risk:** Medium

## Goal

Generalize the **one coupling the slot model deliberately does not absorb**: a
feature temporarily *seizing control of a shared core widget*. The canonical case is
voice taking over the model selector during a realtime session
(`acquireVoiceLock`/`releaseVoiceLock`,
[chat.js:3743](../../../kestrel_sovereign/static/js/chat.js);
[shared/model-selector/index.js:728-852](../../../kestrel_sovereign/static/shared/model-selector/index.js)).

This is **not** a mount point — it is a negotiation: "while my session is live, this
shared control is mine, then I give it back." Forcing it into the slot abstraction
would rot the slot model, so it gets its own small contract.

## Why after ticket 04

The exact shape of this contract should be **discovered during the voice migration**,
not guessed up front. Ticket 04 leaves the model-selector lock as-is precisely so we
see its real requirements (store prior selection, inject an unpickable option,
restore on release, survive a model-list reload via `reapplyActiveSelectorLock`
[voice/ui.js:438](../../../kestrel_sovereign/static/js/voice/ui.js)) before
generalizing.

## Design (to be refined against ticket-04 findings)

A claim registry on shared widgets:

```js
// A shared widget exposes a claimable surface:
ModelSelector.claims.acquire(claimId, {
  onAcquire: (widget) => { /* set unpickable, store prior, etc. */ },
  onRelease: (widget) => { /* restore */ },
  onRefresh: (widget) => { /* re-apply after the widget rebuilds its options */ },
});
ModelSelector.claims.release(claimId);
```

- Single-holder per widget; a second acquire is rejected or queued (decide; voice is
  the only holder today, so single-holder-reject is the safe default).
- `onRefresh` covers the "widget rebuilt its options, re-assert my claim" case voice
  handles with `reapplyActiveSelectorLock`.
- Claims are released automatically if the claiming feature is disabled
  (`capabilities:changed` from ticket 03).

## Tasks

1. Extract a generic claim/release/refresh contract from the model-selector lock.
2. Refactor the model selector to expose it; reimplement voice's lock on top.
3. Remove the last voice-by-name reference from `chat.js` (the
   [chat.js:3743](../../../kestrel_sovereign/static/js/chat.js)
   `hasCapability('voice')` branch becomes a generic claim driven by voice's own
   session events).
4. Tests: acquire/release/refresh, auto-release on disable, double-acquire policy.

## Acceptance criteria

- Voice's model-selector takeover behaves identically, now via the generic claim API.
- `grep -ri voice kestrel_sovereign/static/js/chat.js` returns nothing — closing the
  final core→voice coupling left open by ticket 04.
- A second feature could claim a different shared widget using the same contract
  (demonstrated by a test, even if no second feature exists yet).

## Scope discipline

Generalize **only** what voice actually needs plus the obvious single extension
(multiple distinct widgets). Do **not** build a general capability-arbitration
framework. If only the model selector is ever claimed, the contract stays small.
