# Demo-Isolation Rail Vignette — Narration

## Why this feature exists

On **2026-04-24** a Playwright harness pointed at the live `localhost:8888`
server and called destructive APIs against three agents.  Their conversation
histories were wiped.  The convention layer (`kestrel demo run`) prevents this
*by discipline* — but discipline is not enforcement.

Issue **#766** is the server-side enforcement layer.  Every destructive
endpoint sits behind a FastAPI dependency, `enforce_destructive_op`, that
checks two things:

1. Is this server in **demo mode** (every loaded agent is demo-scoped)?
2. Is the target agent **live** or demo-scoped?

A live target requires the `X-Kestrel-Allow-Destructive` header carrying a
free-text reason.  Without the header, the call is refused with 403 and an
audit row is written to `security_audit_log`.

## The beats

### Beat 1 — A live conversation
Seed a small conversation so the rail has something real to protect.

### Beat 2 — Refusal
A misbehaving script issues `DELETE /api/conversations/messages/<id>` without
the header.  Server returns **403**.  An audit row is written with the
caller's IP, the endpoint, the redacted headers, and the decision
`refused-no-destructive-header`.

### Beat 3 — Allowed
Same call, this time with `X-Kestrel-Allow-Destructive: demo-isolation-vignette`.
Server returns **200**.  An audit row is written with the same shape but the
decision `allowed-with-header` and the reason string captured.

### Beat 4 — Bookend
The Security panel surfaces the audit log so an operator can see what fired
and why.

## Running the vignette

```bash
kestrel demo run demo-isolation
kestrel-eye review --config demos/demo-isolation/eye.toml
```
