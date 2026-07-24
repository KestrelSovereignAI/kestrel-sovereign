---
type: Architecture Note
title: Scoped peer directories
description: Authorization boundary for automatic peer routing in local and hosted runtimes.
resource: /docs/architecture/core/SCOPED_PEER_DIRECTORY.md
tags:
- architecture
- multi-agent
- a2a
- security
status: active
privacy: public
---

# Scoped peer directories

`PeersFeature` treats a peer as an entry in the caller's **automatic peer
directory**, not as an arbitrary agent address. A peer name or slug is first
resolved to a stable agent identity inside that directory. The resolved identity
is then used for synchronous invocation, A2A task delivery, result retrieval,
and task-result subscription.

## Provider boundary

Hosted runtimes inject two dependencies into `KestrelAgent`:

- `peer_directory_router`: a `PeerDirectoryRouter` implementation;
- `peer_requester`: a `PeerRequester` carrying the agent's stable requester
  identity and an opaque authorization scope.

The scope is created by the host after its own authentication and authorization
work. Kestrel neither parses it nor accepts a tool-supplied user ID, tenant ID,
or replacement scope. An injected router without this requester context fails
feature initialization.

Providers must authorize every one of these operations, not only the initial
listing:

1. `list_peers` and `resolve_peer` return only automatic peers in the scope.
2. `invoke`, `send_a2a_task`, and `get_a2a_task` reauthorize the resolved peer.
3. `subscribe_a2a_task` reauthorizes again because scope can change after a
   question is sent.

A provider must not accept a `PeerIdentity` just because it was returned by an
earlier lookup. It can be stale or retained after access is revoked.

## Tenant isolation

Two scopes may each have a peer called `companion`; their stable identities and
internal routing keys can differ. A scoped router resolves `companion` only in
the requester's scope. Unknown, ambiguous, cross-scope, and DID-like automatic
shortcut inputs all return the same no-peer result, without exposing whether
another scope owns that name or identity.

This boundary applies only to automatic peers. A2A remains the transport
protocol, including its existing signed envelope semantics. Explicit external
or cross-user A2A trust/addressing is a separate future capability and must not
reuse the automatic peer shortcut.

## Local compatibility

Without an injected router, Kestrel installs `LocalHostPeerDirectory`. It calls
the existing local multi-agent host endpoints, resolves the host's
`routing_name` to its stable `id`, and then performs the existing invoke, A2A,
result, or SSE subscription route. Local operators therefore retain the same
multi-agent behavior while hosted runtimes get a single authorization seam.
