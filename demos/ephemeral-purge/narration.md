# EPHEMERAL Purge Vignette — Narration

## Why this feature exists

EPHEMERAL is a contract: **nothing stored, local LLM only**.  The privacy
wrapper (`PrivacyEnforcingStorage`) sits in front of every storage call and
gates the persistent path.  But a single gate is fragile:

* A new feature added next week may bypass the wrapper.
* A bug in `_set_privacy_mode_with_effects_locked` leaves a window.
* An integration writes directly to the underlying DB.

Issue **#767** adds defense-in-depth.  Whenever the agent transitions
*out* of EPHEMERAL — back to NORMAL, ANONYMOUS, ISOLATED, or PUBLIC — it
runs `_purge_ephemeral_leaks()`.  That hard-deletes any rows owned by this
agent that should not exist after an EPHEMERAL stint:

* `conversation_history` rows the wrapper let through.
* `graph_nodes` rows whose `properties.agent_id` matches.
* Any associated `graph_edges`.

The purge writes to `security_audit_log` with row counts so an operator
can audit what was scrubbed.

## The beats

### Beat 1 — NORMAL baseline
The privacy badge in the chat header shows the current mode.  Persistence
is the default contract.

### Beat 2 — Flip to EPHEMERAL
The badge opens a selector.  Picking EPHEMERAL flips the mode; the wrapper
now silently drops persistent writes.

### Beat 3 — A conversation that leaves no trace
A message sent in EPHEMERAL renders in the chat but doesn't persist.

### Beat 4 — Exit EPHEMERAL — the purge fires
Switching back to NORMAL is the trigger.  `_purge_ephemeral_leaks()` runs
and any rows the wrapper let through are hard-deleted.

### Beat 5 — Bookend
The Security panel surfaces the purge audit row.

## Running the vignette

```bash
kestrel demo run ephemeral-purge
kestrel-eye review --config demos/ephemeral-purge/eye.toml
```
