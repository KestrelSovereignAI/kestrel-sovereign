---
type: Architecture Spec
title: Native MCP and A2A interoperability
description: Protocol ownership, security invariants, and release gates for epic 3679.
resource: /docs/architecture/NATIVE_MCP_A2A.md
tags:
- docs
- architecture
- mcp
- a2a
timestamp: '2026-10-17T00:00:00Z'
status: draft
owner: architecture
canonical: true
generated: false
privacy: public
---

# Native MCP and A2A interoperability

Implementation contract for [#3679](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3679).
This is a design, not a claim that the migration has shipped. Phase 0 evidence
and open gates are recorded in [the spike report](NATIVE_PROTOCOL_SPIKES.md).

## Protocol lock and ownership

Use the published [MCP 2026-07-28 specification](https://modelcontextprotocol.io/specification/2026-07-28)
and [A2A 1.0 specification](https://a2a-protocol.org/v1.0/specification/).
Use official Python SDKs at every network boundary. The initial candidates are
`mcp==2.3.0` and `a2a-sdk==1.2.2`; exact versions belong in dependency locks,
and dependency ranges must admit the releases we actually test. Old dependency
caps are subject to migration, not reasons to downgrade the protocol target.

| Owner | Changes |
| --- | --- |
| `kestrel-feature-mcp` | Official Client, Streamable HTTP, remote connections, OAuth SDK integration, registry discovery, MCP result preservation, tasks |
| `kestrel-sovereign` | Dependency cascade, agent credential service, governed dispatch, typed UI parts, approvals/waits, A2A handlers/client/card, persistence migrations |
| `kestrel-sovereign-sdk` | Remove draft wire models; shared host/feature credential and typed-part contracts; update all consumers |
| Host manifest packages | Survey published constraints and editable sources; release incompatible consumers before the host floor moves |

Core's `a2a/types.py` and `a2a/agent_card.py` are re-exports. The actual
handwritten models are in `kestrel_sdk/a2a/`. Removing only the core files
would leave the private wire intact. Inventory both namespaces and their
consumers before the coordinated SDK/core release. Keep the SDK lightweight:
the official A2A dependency belongs to consumers of A2A, not every feature.

No second protocol handler, compatibility route, or old-to-new network
translation layer is introduced. Internal database rows remain storage models;
their conversion to official SDK objects is a persistence boundary, not a
second wire protocol. Existing rows receive an explicit, versioned migration.

## MCP host

The call path remains:

`model or app → host governed dispatch → dynamic tool handle → MCP Client → server`.

Connection kinds are gateway, container, and remote. One connection owner
holds the official `Client` and performs tool discovery/calls. Reuse its
existing owned-task teardown and timeout behavior. Remote mode requires no
Docker daemon. An unavailable or unsupported transport fails explicitly.
Retire the legacy HTTP+SSE client, `/sse` defaults, and container fixtures that
only speak it. SSE response bodies within Streamable HTTP are still standard.

Use the SDK's modern protocol mode (`2026-07-28`) and extension APIs. Do not
manually serialize `_meta`, call `initialize`, retain `Mcp-Session-Id`, or
silently negotiate the draft protocol. Test captured HTTP requests to establish
that this is the protocol actually sent. Gateway/container releases must pass
the same modern-protocol test before their upgrade ships.

| Host duty | Implementation and evidence |
| --- | --- |
| Authorization discovery | Official SDK RFC 9728 discovery, RFC 8414 and OIDC metadata; validate the resource and issuer before use |
| Client identity | Operator-configured stable HTTPS CIMD URL; public client, `token_endpoint_auth_method=none`; no embedded client secret |
| Authorization | PKCE S256 and one-use state bound to agent DID, connection, issuer, redirect URI, resource, scopes and expiry |
| Resource binding | RFC 8707 resource parameter on authorization and token requests; never forward a token to another resource/origin |
| Callback validation | SDK RFC 9207 issuer checks; missing advertised issuer, mismatched state/issuer, duplicate and expired callbacks fail before token exchange |
| Refresh | Agent-scoped encrypted token bundle; serialize refresh/rotation for the same connection across requests; commit rotated credentials atomically |
| Discovery | Official Registry alongside Docker/local catalogs; explicit source and namespaced identity, opaque pagination, no arbitrary executable installation from search results |
| Apps | Preserve tool metadata, structured content and `ui://` resource association through a typed part; isolated renderer and governed app calls |
| Tasks/input | Versioned extensions via SDK; durable wait provider and pending input record, no conversation lock held across human response |

### Credential storage and connect UI

Reuse `security/service_key_storage.py::ServiceKeyStorage`, whose storage is
already agent-scoped and encrypted through `security/agent_encryption.py`.
Do not use the host/user/sponsor master-key stores for per-agent tokens, and
do not invent a plaintext token file. Extend this credential service with a
typed bundle and atomic generation check for refresh-token rotation; the MCP
feature sees a host-provided protocol rather than importing core.

Key the connection by agent DID, canonical resource, authorization issuer and
client identity. Store access/refresh tokens, expiry and granted scopes in the
encrypted bundle. Connection configuration and UI responses contain references
and status, never tokens. Disconnect removes credentials, unmounts handles and
invalidates pending connect attempts. Volatile privacy modes may retain only
in-memory credentials and must report that restart requires consent again.

The console starts a connect attempt and returns immediately with a status and
authorization link. The callback redeems only that attempt and mounts tools
only after validation and token persistence. Refresh failure moves the existing
connection to an explicit reconnect-required state, not environment-token
fallback. Authorization codes, tokens and PKCE verifiers are excluded from
access logs, tool results, traces and persisted conversation content.

RFC 8252 loopback redirects are appropriate when the browser and Kestrel run on
the same computer. A remotely hosted console needs its own HTTPS callback;
loopback in the operator's browser cannot reach that host. The shared CIMD must
declare the supported redirects, and arbitrary host callbacks cannot be assumed
acceptable to vendors. Phase 0 must distinguish local and hosted deployments.
If a vendor needs a pre-registered client, show that requirement explicitly.
Do not silently switch to deprecated dynamic registration or borrow another
product's client identity to make the spike pass.

### Registry and Apps

Registry discovery uses the official registry's currently published API schema
and a pinned contract fixture; its API version is independent of the MCP wire
version. Preserve source, server name, version and remote endpoint in catalog
identity. Local explicit configuration wins only through a documented override,
not an accidental name collision. Pagination failures and malformed entries
produce actionable errors; they must not become an empty successful catalog.

An MCP App resource is loaded from the same authorized connection as its tool.
The host validates MIME type and resource association, then renders it in an
iframe with sandbox/CSP restrictions and the official Apps bridge. Never inject
vendor HTML into the console DOM. No host credentials enter the iframe. Bind
each bridge message to the iframe source, session nonce, agent, connection and
origin; allow only the Apps methods the host implements. A requested tool must
be mounted and allowed for that app. Its invocation re-enters the same host
dispatch used by model calls, including approvals and the resulting
`a2a_tool_dispatches` row. Unmount/disconnect invalidates outstanding bridges.

### Tasks and operator input

An MCP task is identified by agent, connection and remote task ID. Its durable
wait registration survives restart and maps terminal status into the existing
wait-provider contract. Advertise only supported extension versions. Poll,
cancel and result retrieval use the same authorized connection and policy
checks as the original call. Cancellation does not invent a completed result.

Input-required responses create a pending approval/input item and release the
turn lock before waiting. Resume a new owned continuation after an authenticated
operator response. Reject stale responses and input for a different agent or
task. This must be exercised against the real turn scheduler, reproducing
[#3439](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3439)'s lock-contention scenario.

## A2A 1.0 cutover

Use official SDK types and serializers. A2A 1.0 uses protobuf JSON conventions:
Python's `context_id`/`supported_interfaces` serialize as `contextId`/
`supportedInterfaces`; `Part` uses the content member, not the 0.2 `kind`
discriminator. Test the serialized HTTP, not just Python attributes.

Use per-agent routing under `/api/agents/{agent_id}`. Publish the card at
`/api/agents/{agent_id}/.well-known/agent-card.json`. Its supported interfaces
advertise the agent's absolute JSON-RPC and HTTP+JSON base URLs. Let the SDK
define method names, resource paths, error mappings and event serialization.
The authenticated console projection is a separate view of this same card.

`SDK request handler → verified principal + recipient policy → TaskManager → stores/worker`.

Integrate the SDK request-handler and executor interfaces with the existing
TaskManager lifecycle. Do not create a second task lifecycle or store. Both
bindings enter the same authorization path, including get/list/cancel/subscribe
and push-notification configuration. Multi-tenant routing binds the recipient
before signature verification; a signed audience cannot select another tenant.

### Semantic mapping

Extension identifiers below are the canonical URL of this document followed by
the indicated anchor:
`https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/architecture/NATIVE_MCP_A2A.md`.
Declare URI, versioned schema and whether required on the signed Agent Card;
negotiate them using the standard A2A extensions mechanism. URI-keyed metadata
holds extension payloads. Unknown required extensions fail before task creation.

| Existing semantic | A2A core or extension | Invariant |
| --- | --- | --- |
| Task lifecycle, history, artifacts and streams | SDK Task, Message, Artifact and event models | Preserve terminal transitions and durable IDs |
| `sessionId` | Core `contextId` | Migrate rows once; no draft field on the wire |
| Sender DID and replay nonce | `#signed-sender-v1` | Verify signer, audience, operation, message digest, timestamp and nonce before side effects |
| Recipient authorization | `#recipient-policy-v1` plus A2A security schemes | Verified identity is input to recipient policy, never permission by itself |
| Question/answer correlation | `#question-correlation-v1` | Response references an outstanding question owned by this sender/recipient pair |
| Causation | `#causation-v1` | Attribution/loop control only; never grants authority |
| Outbound task store | Internal persistence using core task/context IDs | Preserve dispatch evidence, final outcome and restart reconciliation |
| Cancellation authority | Core CancelTask plus recipient policy | Authenticate creator/recipient authority on every mutation |
| Wake signals | Internal transition events | One committed transition drives existing wake hooks |

### Signed sender v1

The extension contains sender DID, recipient DID, operation, timestamp, nonce
and the sender proof. Bind the complete official request payload (including
non-text parts, configuration and all behavior-affecting extension metadata)
using RFC 8785 canonicalization, excluding only the proof field itself.
Do not reuse the draft envelope's projection of selected text fields. Preserve
hybrid sender verification through the existing identity and verification-policy
services. Define algorithm identifiers and canonical-byte test vectors in the
extension implementation before release; no ad hoc algorithm is called JWS.
Reserve replay nonces through the existing shared replay store, with documented
retry/idempotency behavior. Reads/cancels also bind method, target and audience.

The Agent Card has a separate standard JWS signature over the SDK card's JCS
representation, excluding signatures as the spec requires. Use the agent's
Ed25519 verification method and a DID-bound `kid`; the existing hybrid message
proof is not substituted for an interoperable Agent Card signature. Verify the
card against a trusted DID resolution result before trusting advertised URLs
or capabilities. A card containing its own untrusted key is not a trust root.
Card signature verification does not authenticate an incoming caller.

### Recipient policy v1

Reuse `RecipientA2ASenderAuthorizer`, scoped requester contexts and durable
cancellation authority. An external peer may authenticate with a standard
advertised security scheme without implementing the DID extension only when
the recipient explicitly permits that principal and the extension is optional
on that card. A recipient requiring DID proof must advertise it as required;
do not lower that policy to pass an interop test. Standard-only peer tests use
an explicitly provisioned principal and a compatible card policy.

### Question correlation v1

Carry question ID, reply expectation and answer reference in extension
metadata. Use core task/context/message identifiers for transport identity.
Retain `async_pending_a2a_question_store.py`, recipient binding and
`a2a_question_answered` wake behavior. Duplicate, unsolicited and wrong-peer
answers cannot satisfy another agent's pending wait.

### Causation v1

Carry the typed CausationFrame chain in extension metadata, covered by sender
proof. Apply existing depth/loop validation. Caller-supplied causation cannot
authorize reads, cancel a task, govern descendants or bypass approval.

### Persistence, outbound and retirement

Migrate serialized draft messages/events in active tasks and historical rows
to a versioned internal representation. Preserve context identity, question
correlation, signatures as historical evidence and outbound destination keys.
Do not re-sign old requests as though they were received on the new wire.
Test SQLite and PostgreSQL migration/restart, including nonterminal work.

`send_a2a_message` and `send_a2a_question` resolve and verify a peer card, select
one advertised binding and use the SDK client. Persist the outbound dispatch
before submission and record the returned task identity without losing failure
evidence. Local routing enters the same official-type submission contract.

Ship inbound and outbound together. Delete `POST /api/agent/tasks/send`, old
task read/cancel/subscribe routes and their transport-key path patterns, all
draft wire models and all callers. Do not repurpose the operator API key as
peer authority. Drain or explicitly stop old in-flight work during the host
upgrade; mixed draft/new workers are unsupported. Rollback requires the saved
pre-migration database and a coherent old host/SDK set, not a parallel handler.

Push notifications require authenticated configuration changes, recipient-bound
ownership and outbound URL validation against SSRF (including redirect/DNS
changes). Signing a card does not authorize arbitrary callback egress.

## Review, verification and release order

1. Review this design in its worktree; complete dependency/import and three
   real-vendor OAuth spikes. Record capability advertisement separately from
   successful authorization, refresh and tool calls.
2. Upgrade the dependency set and MCP transports together, including gateway
   and container tests. Resolve the full host feature manifest before release.
3. Add remote connections/OAuth/registry, then Apps and Tasks. Remote write
   scopes remain gated on [#596](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/596).
4. Release the coordinated SDK/core A2A cutover with storage migration and
   every consumer updated. Run both A2A bindings through the TCK and a
   different-framework bidirectional task/card-verification exercise.
5. Implement the server direction under [#1990](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1990),
   reusing MCP 2.x, per-agent routing, protected-resource metadata, credential
   governance and the normal memory/turn persistence path.

Each PR requires a completed full-diff worktree review before creation, with
findings fixed and meaningful regression tests. Pin the conformance runner
versions/commits and record which protocol scenarios they actually support.
Do not label a skipped 2026 scenario or a mocked interop peer a conformance pass.
Acceptance includes governed dispatch database evidence, Apps bridge denial
tests, OAuth restart/rotation, cross-agent token isolation, signature tampering,
replay, recipient denial, question forgery, cancellation races and lock-free
operator input. Final acceptance also searches for every retired wire surface.
