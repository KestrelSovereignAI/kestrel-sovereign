---
type: Audit Ledger
title: Auth Surface Matrix
description: The live auth model is enforced centrally in [`server.py`](../../server.py),
  with endpoint semantics layered on top.
resource: /docs/audit/AUTH_SURFACE_MATRIX.md
tags:
- docs
- audit
- audit-ledger
timestamp: '2026-11-02T00:00:00Z'
status: active
owner: documentation
canonical: false
generated: false
privacy: public
---

# Auth Surface Matrix

The live auth model is enforced centrally in [`server.py`](../../server.py), with endpoint semantics layered on top.

| Class | Description | Representative routes |
|---|---|---|
| `Public` | No API key or session required; `/health` returns aggregate readiness only | `/health`, `/webhooks/stripe/crypto` |
| `Public-Localhost` | Public only from localhost-like callers when bootstrap is enabled | `/api/auth/key` |
| `OAuth public entrypoints` | Browser sign-in flow entrypoints | `/auth/login`, `/auth/callback`, `/auth/logout` |
| `APIKeyOrSession` | Protected routes that accept API key, bearer JWT, or OAuth session auth | `/health/detailed`, most `/agent/*` and `/api/*` routes |
| `APIKeyOrSession+SSEQuery` | Protected routes that also permit `?api_key=` because `EventSource` cannot send custom headers | `/agent/stream`, `/agent/notifications/sse` |
| `OAuthSessionSemantic` | Route may pass middleware via API key but still only returns success data with a real browser session | `/auth/me` |
| `Browser-Conditional` | Root page behavior depends on UI mode and OAuth-required mode | `/` |

## Key findings

- The auth model is not just public versus protected.
- Most protected routes are effectively `API key OR OAuth session`.
- `/health` is the public load-balancer contract: only `status` and
  `agent_initialized`; `/health/detailed` is operator-authenticated.
- `/auth/me` is not middleware-public; it is semantically session-backed.
- `/api/auth/key` is intentionally narrow and must remain localhost-scoped.

## What "localhost-like" means for `/api/auth/key`

Verified for #3804. The `test_bootstrap_key_source_scoping` unit module pins
the gate's admissions and refusals described here, with the two exposures
tracked in issues as strict `xfail` cases.

- **Who is admitted.** `request.client.host` must exactly match `127.0.0.1`,
  `localhost`, `::1`, `172.17.0.1`, or an entry in
  `KESTREL_BOOTSTRAP_ALLOWED_HOSTS`
  ([`bootstrap_access.py`](../../kestrel_sovereign/security/bootstrap_access.py)).
- **That host is not always the socket peer.** Every launcher leaves uvicorn's
  proxy-header defaults on: `X-Forwarded-For` is read only from a `127.0.0.1`
  peer, and only when exactly one such header line is present, and its last
  untrusted entry becomes the client. A remote peer cannot claim a local
  address, and a loopback proxy that names its remote client is refused.
  Setting `FORWARDED_ALLOW_IPS` or `UVICORN_FORWARDED_ALLOW_IPS` to `*` or a
  remote range would let any such client name itself `127.0.0.1`.
- **Where it is enabled.** Only in a standalone process. It answers 404 under
  `KESTREL_REQUIRE_OAUTH`, in a host-managed peer process, and on a fleet host
  that has published its registry. A standalone agent started with
  `kestrel start <name>` keeps the project's `KESTREL_API_KEY`, the same key the
  fleet host requires, and its bootstrap returns it.
- **It does not close.** The same key is returned on every call, limited only
  to 5 per minute per client host. Turning it off requires one of the three
  conditions above.

Known exposures, each of which returns the key to a caller other than the
operator's local browser:

- A same-host reverse proxy or tunnel that forwards without any forwarding
  header. Such a request is indistinguishable from a local browser; do not put
  a standalone agent behind one.
- A loopback proxy whose forwarding evidence uvicorn does not read: two
  `X-Forwarded-For` lines (HAProxy `option forwardfor` appends its own), an
  RFC 7239 `Forwarded` header alone, or a proxy connecting from `::1`. Tracked
  in #3905.
- A fleet host whose startup failed before its registry was published: the
  endpoint then treats the process as standalone and returns the fleet's
  out-of-band key. Tracked in #3904.
- `172.17.0.1` is the Docker bridge gateway. Traffic that Docker's userland
  proxy relays to a published port, including from sibling containers, also
  arrives from that address. This is inferred from Docker's networking model,
  not measured.

The health boundary and regression history are documented in
[`HEALTH_ENDPOINT_AUTHORIZATION.md`](../architecture/security/HEALTH_ENDPOINT_AUTHORIZATION.md).
