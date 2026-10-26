---
type: Architecture Spec
title: Original Restrictions for Native Synchronous HTTP
description: Optional original-caller checks at supported native synchronous HTTPX TLS and delivery boundaries.
resource: /docs/architecture/EXECUTION_SYNC_HTTP.md
tags:
- docs
- architecture
- execution-custody
timestamp: '2026-10-26T00:00:00Z'
status: active
owner: architecture
canonical: true
generated: false
privacy: public
---

# Original restrictions for native synchronous HTTP, API version 1

`kestrel_sovereign.execution_sync_http` is an optional source prerequisite for
hosts that need a synchronous HTTP client to retain original authorization
through native I/O. It does not register a provider, acquire capacity or mint
authority. Existing asynchronous APIs and execution custody remain unchanged.
The installed Core must actually export `NATIVE_SYNC_HTTP_ADMISSION_API_VERSION
== 1`; a source candidate with unchanged package metadata is not proof that the
same published version contains this API.

## Retaining the original operation

Trusted host code obtains a synchronous checker from its original authenticated
operation. It retains that SAME owner/grant/privacy/lifecycle lease outside:

```python
with bind_native_sync_http_execution_admission(existing_client, original_check):
    result = await await_native_sync_http(
        existing_client, lambda client: client.post(original_url, content=original_bytes).json()
    )
```

The checker is an additional restriction. Context IDs, attribution, checkpoints
and copied request metadata cannot construct authority. Existing canonical
execution-custody snapshots remain mandatory at every boundary. Any original
async native validator is explicitly refused, including in copied contexts;
this module does not skip it, run it in another loop or replace it with a fresh
checker. The host must not supply an invented callback as a grant substitute.

The binding installs guards on the EXISTING native client/pool once, before any
connection is used. That client remains guarded: subsequent requests require
an explicit live original restriction. A copied binding stays denied after
lexical exit even when another caller has a new binding on the same client.
This is an intentional opt-in boundary; it must not be installed on a client
whose unrelated callers expect to run without such a binding.

## Narrow native support

| Surface | Supported contract |
| --- | --- |
| Native libraries | HTTPX 0.28.1 and HTTPcore 1.0.9, exact native Client/HTTPTransport/ConnectionPool/SyncBackend |
| Transport | Direct verified HTTPS, negotiated TLS1.3, HTTP/1.1 |
| Timeouts | Explicit finite positive connect/read/write/pool values, at most 30 seconds each, including per-request overrides |
| Replay | No automatic redirects, auth flows or native connection retries |
| Other configurations | Proxy, H2, UDS, cookies, replaced/custom transport/backend/dispatch methods, previously used pools and other versions are refused |

The boundary creates no second client, pool, HTTP parser, SSL verifier or
partial-write loop. The native TLS stream installs a guard on its actual
`SSLSocket.send` method and delegates to that original bound method. Every
iteration of HTTPcore's existing partial-write loop therefore checks the
original request immediately before its physical send, after connect/TLS work.
The request's checker inventory is captured privately before observers; it
never comes from mutable request extensions or the first connection owner's
scope. Reused pooled connections consult the CURRENT original request.

Private body iteration checks after actual native reads and before yielding.
Delivery is checked after physical response close and after the final thread
join. A response observer cannot replace these original restrictions. These
checks cannot retract bytes already accepted by the peer.

## Native ownership, cancellation and unknown outcomes

`await_native_sync_http` uses Core's existing owned-operation machinery to run
and physically join the synchronous thread. Caller cancellation retires that
operation's I/O and shuts down its exact currently borrowed sockets; it does
NOT cancel the `to_thread` task and discard its running native thread. Repeated
caller cancellation still joins the same original owner. A socket borrow
retires before its connection returns to the pool. Its independent retirement
flag/lock prevents an already queued old cancellation from closing a later
borrower's connection.

The callback consumes/closes its responses in that thread and returns a result.
Unconsumed responses are closed and refused before returning; async callbacks
and returned awaitables are refused. Do not detach work or wrap a raw
`to_thread` call and let its original binding exit early. Native phase timeouts
are not an end-to-end action deadline: the host's original checker must still
enforce its own expiry and cost/policy lifetime.

After any possible native send, errors or denied delivery raise
`NativeSyncHTTPOutcomeUnknown`, an execution-control denial. Original request,
action, attempt and resource IDs remain in their existing host custody; no
automatic retry is authorized. The exception retains `original_error` and may
retain `native_result` as PRIVATE original facts for that custody owner. Its
error text/repr do not emit them. Never serialize exception dictionaries or
use those facts as permission to expose data or perform more I/O. Physical
cleanup does not require a new grant.

## Evidence and remaining integration gates

Native tests use the repository's owned temporary TLS certificates and actual
loopback sockets. They cover post-handshake revocation before HTTP bytes,
per-partial-send denial, body/close/final-join revocation, original execution
custody, copied/absent/async-validator refusal, real pooled reuse, queued old
cancellation versus a new borrower, and joined thread/response retirement.
No cloud endpoint, provider, remote worker or paid capacity is exercised.

The existing asynchronous admission hook checks before awaiting native
transport dispatch. A separately recorded native TLS reproducer demonstrates
that this does not itself provide a post-handshake physical-write restriction.
This module does not change or claim to fix that existing async boundary.

Runpod's exact prepared-body path, SDK acquisition-custody adapter and Core
encrypted request-row integration remain separate reviewed work. No existing
GPU caller, release metadata, provider registration or paid-create path is
changed by this optional prerequisite.
