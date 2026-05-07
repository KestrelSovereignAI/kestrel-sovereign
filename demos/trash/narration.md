# Trash Vignette — Narration

## Why this feature exists

On **2026-04-24** a Playwright harness pointed at the live `localhost:8888`
server wiped three agents' conversation histories. The harness called
destructive APIs and the storage layer obeyed — there was no soft-delete, no
recoverable trash, no audit trail.

This vignette demonstrates the recovery surface that came out of that
incident:

* **#763** — soft-delete by default. `delete_message` and friends now set
  `deleted_at` instead of `DELETE FROM`. The row is gone *from the user's
  view* but still in the database.
* **#765** — Trash sub-view. The conversations pane has a 🗑 toggle that
  flips between active and trashed items. Each trashed item shows what it
  was, when it was deleted, and offers Restore / Purge.
* **#766** — destructive-op rail. Hard-purge requires the
  `X-Kestrel-Allow-Destructive` header and writes to `security_audit_log`.

## The beats

### Beat 1 — A conversation worth keeping
A user sends a real chat message; both messages land in
`conversation_history`. We need *something* to delete.

### Beat 2 — Soft-delete a message
Hovering a message reveals two icons: ✕ (soft-delete) and ⊘ (purge). The
user clicks ✕. The message vanishes from chat. The row still exists with
`deleted_at` set — recoverable.

### Beat 3 — Find it in Trash
The 🗑 toggle on the conversations pane flips into Trash mode. The deleted
message is here, with a preview, a "deleted at" timestamp, and Restore /
Purge buttons.

### Beat 4 — Restore
Restore clears `deleted_at`. The row is live again. We pop back to chat;
the message is there, indistinguishable from one that was never deleted.

### Beat 5 — Hard-purge
The user soft-deletes again, then hits ⊘ — the purge affordance. The UI
raises a `confirm()` dialog (auto-accepted in this demo) and the request
carries `X-Kestrel-Allow-Destructive: user-initiated-ui`. The row is
deleted outright — no recovery.

### Beat 6 — Bookend
Trash is empty (or down by one). The destructive-op rail recorded the
purge to `security_audit_log` so an operator can audit what was destroyed.

## Running the vignette

```bash
kestrel demo run trash
```

Outputs in `demos/trash/demo-output/` — screenshots, generated narration
transcript, Playwright video recording.
