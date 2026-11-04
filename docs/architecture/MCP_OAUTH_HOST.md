---
type: Architecture Spec
title: MCP OAuth host boundary
description: Optional host identity, request authority, and connection leases for native MCP OAuth.
resource: /docs/architecture/MCP_OAUTH_HOST.md
tags: [docs, architecture, mcp, oauth]
timestamp: '2026-11-04T00:00:00Z'
status: draft
owner: architecture
canonical: true
generated: false
privacy: public
---

# MCP OAuth host boundary

This Core contract is part of [epic #3679](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3679).
It is dormant without a separately installed feature runtime. Core does not
acquire vendor grants, refresh tokens, mount OAuth tools, or provide the Connect
panel in this change. Published MCP feature 0.12.0 does not attach this runtime.
Production vendor acceptance remains an epic release gate.

## Installation identity

The web client profile requires the Console's explicit public HTTPS origin:

```sh
KESTREL_MCP_OAUTH_CONSOLE_ORIGIN=https://console.example.com
KESTREL_SERVE_UI=true
```

An unconfigured host has no MCP OAuth identity. Invalid identity or budget
configuration fails at startup. The public Console origin must serve callback
assets. A host-managed child may have `KESTREL_SERVE_UI=false` while inheriting
the public parent's identity; the parent serves the callback page and assets.
`KESTREL_MCP_OAUTH_POLICY` optionally supplies a strict JSON object of budgets
and `client_name`. The origin belongs only in its dedicated variable. Unknown
fields, secrets, replacement client IDs, and conflicting origins are rejected.
The policy schema in `kestrel_sovereign/mcp_oauth.py` owns all defaults and limits.

`GET /oauth/mcp/client-metadata.json` serves the public Client ID Metadata
Document with `application_type=web`, public token authentication (`none`),
the exact callback, and code/refresh grants. It contains no vendor scopes or
secrets. The configured origin determines identity; request Host headers do not.

`GET /oauth/mcp/callback` serves a fixed public page. This navigation grants no
authority. Its module captures the callback, strips the query before imports
or API requests, consumes a same-tab routing hint, and submits to the original
agent's authenticated API. The API client supplies the actual credential lane
and cookie CSRF protection. No hint means no API request. Failures never display
callback parameters or upstream exceptions and never replay completion.

The feature panel must store `sessionStorage["kestrel.mcp-oauth." + state]` with
`{agentPrefix, operationId, returnTo: "/"}` before same-tab navigation. Prefer
the canonical `/api/agent-routes/{base64url}` prefix for arbitrary agent names.
The hint is routing data, never authority. Public responses prohibit caching,
framing, referrers, and third-party script/network sources. Access logs and
pre-span URL capture classify callbacks and operation routes as private. Native
SDK OAuth diagnostics lose message arguments and exception traces before logging;
the feature must report safe typed outcome codes instead.

## Host and feature contract

The original agent exposes `create_mcp_oauth_host(feature, is_current=...)` and
`create_mcp_oauth_router(feature, resolve_host=...)`. Attach exactly one runtime
implementing `begin`, `complete`, `cancel`, `connect`, `disconnect`, `list`,
`retire_authorization`, `withdraw_authorizations`, and `close_authorizations`.
The feature composes the returned router with its existing router.

The runtime signatures are:

```python
async def begin(context, check, request, identity, *, authorization_handle): ...
async def complete(handle, context, check, callback): ...
async def cancel(handle, check): ...
async def connect(context, check, connection_id, *, refresh): ...
async def disconnect(context, check, connection_id): ...
async def list(context): ...
async def retire_authorization(handle): ...
async def close_authorizations(): ...
def withdraw_authorizations(): ...  # synchronous, returns None
```

`identity` contains the host policy fields and `client_document`. `begin` returns
exactly `{handle, authorization_url, state, issuer, resource, scopes}`, echoing the
original handle. Core owns expiry; the runtime does not return `expires_at`.
Complete and Connect return `{connection_id, server_url, issuer, resource, scopes,
state, connected}`; List returns a list of those same status dictionaries. The
canonical strict schemas and state vocabulary live in `mcp_oauth.py`. Disconnect
returns the local-only receipt described below. `check()` raises on withdrawn
authority and must be called at each feature effect boundary. Invalid feature
receipts are service failures; failures after a mutating feature call return an
unknown-outcome code that explicitly prohibits replay.

The runtime calls `host.client_identity(context)` inside the original host
operation to obtain the same current identity and budgets during List or Connect
after a restart. It returns a fresh copy and requires that exact context and
task; no prior Begin, private Core fields, or duplicated environment parser is
needed. A connection retains its approved budget snapshot for its own lifetime.

All private routes retain native request authentication, owner/privacy checks,
and transport custody through the final ASGI send:

| Route under `/api/mcp/oauth` | Action |
| --- | --- |
| `GET /connections` | List this principal's connection metadata |
| `POST /authorizations` | Begin with explicit server URL and scope ceiling |
| `POST /authorizations/{id}/callback` | Consume one owner-bound callback |
| `DELETE /authorizations/{id}` | Cancel before exchange |
| `POST /connections/{id}/connect` | Connect or explicitly refresh |
| `DELETE /connections/{id}` | Disconnect locally |

Core creates an `OperatorContext` from the current native authentication lane.
Role, lane, and subject partition the principal ID; descriptors and saved rows
cannot mint contexts. Cookie mutations require native CSRF checks. Query keys,
anonymous callers, and peer identity do not provide interactive authority.

Pending authorization keeps the original caller, agent privacy epoch, deadline,
opaque feature handle, and a state digest in memory. Quotas reserve before the
feature is awaited. Core allocates that original handle before awaiting `begin`
and passes it as the keyword-only `authorization_handle`. The runtime registers
its operation under that exact object before spawning or awaiting owned work,
and echoes it in its result. Core can therefore retire the original worker even
when returned metadata is invalid or omits/substitutes the handle. Returned URLs
must preserve host identity, issuer origin,
resource, actual requested scopes, and S256 PKCE. Another principal is refused
without consuming or cancelling the owner's operation. A bad state retires
the original handle. Consumption precedes exchange; a failed exchange cannot be
replayed. Cancel consumes before awaiting the feature's join. `maintain()`
retires expired or withdrawn handles only when their request is no longer in
flight; a browser deadline never interrupts an admitted token exchange. Concurrent
retirement observers serialize on the original handle and retain the record
until the feature acknowledges physical closure. The feature's retirement method
must safely reenter the same original worker after a failed observation and
acknowledge an unknown handle idempotently when no work was registered. Core
records whether the handle was handed to `begin`; a refused pre-handoff check
requires no feature retirement. Failed Cancel immediately withdraws its entry
for maintenance rather than waiting for the browser deadline.

Only the original task inside Connect or Complete may issue a connection lease.
The lease binds the original agent, feature incarnation, privacy epoch, principal,
connection, and credential slot. Interactive credential use needs a fresh matching
native authentication. Background lifecycle traffic uses a distinct, finite,
revocable credential proof and rechecks the actual credential lane (including
key rotation, JWT validity, and allowed-email withdrawal). Ending a request never
makes that original request binding reusable.

`credential_operation()` retains canonical privacy and host authority gates for
credential storage and refresh. Same-task nesting reuses its original checked
operation to avoid waiting behind a writer that is draining that very operation.
No resource-response generator should hold this span across a network yield.

## Ownership and limits

The feature owns native SDK acquisition, pinned HTTP, encrypted custody, durable
exchange/refresh claims, per-principal status filtering, dispatch checks, and
physical MCP/tool/App retirement. Status reads must not decrypt credentials.
The host supplies the original native database handle; database identity is not
permission to use a grant. The feature schedules `maintain()` without durable
writes and owns/join its authorization tasks. Foreground Begin does not wait on
unrelated maintenance: a failed retirement remains owned and counts toward the
finite quotas while the independent maintenance owner reports/reconciles it.
Each sweep rechecks eligibility under each original retirement lock, retaining
failed originals and attempting independent cleanup. Errors retain their native
exception chain. A real sweep cancellation remains cancellation after the other
finite feature-owned joins; a later ordinary error cannot replace that control.
Every failed `begin` must retain
and retire its own workers, including before it can return a valid handle.

Privacy transitions synchronously withdraw OAuth leases before installing another
mode. Shutdown stops acceptance and awaits original authorization worker closure.
A failed close retains host custody and prevents replacement. Local disconnect's
receipt is exactly `local_tokens_deleted=true, remote_revocation=not_attempted`;
it makes no authorization-server revocation claim.

The focused Core tests exercise real HTTP authentication, signed cookies, CSRF,
private transport, SQLite privacy transitions, quotas, state/owner/replay checks,
and credential lifetimes with a host-contract consumer. They do not establish
vendor consent, native OAuth acquisition/refresh, or browser end-to-end acceptance.
Those require the feature's native AS/resource peers and real connector tests.
