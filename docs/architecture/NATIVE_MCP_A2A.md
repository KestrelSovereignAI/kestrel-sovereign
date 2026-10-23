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
and [A2A 1.0.1 specification](https://a2a-protocol.org/v1.0.1/specification/),
the latest released A2A revision checked on 2026-10-19.
Use official Python SDK implementations at every network boundary. MCP was tested at 2.3.0.
Kestrel ships a reviewed, modified copy of the official A2A Python SDK in
`kestrel_sdk.native_a2a`, owned by the shared SDK 0.42 `a2a` extra. Its signing, HTTP authorization, borrowed-client
ownership and response-decoding fixes preserve the A2A 1.0.1 wire protocol.
Vendoring is authorized for this cutover; an upstream PyPI release is no longer
a prerequisite. This is a modified SDK distribution, not an official 2.0 release.
Earlier spike evidence used public 1.2.3, rechecked on 2026-10-20.
The packaged `VENDORED.json` records the public base commit, exact reviewed tree,
every original/relocated file digest, the Apache license and a complete source/test
patch. The SDK's `uv run python -m scripts.vendor_a2a --check` verifies custody offline and reverses
the module relocation to verify the reviewed source. Python package names change;
protobuf descriptors, JSON fields, signatures and HTTP bindings do not.
Core does not install or import the external `a2a` package. The isolated namespace
prevents an independently installed SDK from replacing Kestrel's implementation.
The shared SDK declares the runtime dependencies in its optional `a2a` extra.
Core consumes that extra and resolves it in its lock; base SDK consumers remain
lightweight. Feature skill producers and the host use the same native types.
Clean wheel and sdist installs must exercise the same wire/oracle tests as source.
Full conformance, interoperability and host acceptance remain release gates.
Return to the upstream distribution when a released SDK passes these same checks;
remove the vendored implementation and switch all callers together.
Exact versions belong in dependency locks,
and dependency ranges must admit the releases we actually test. Old dependency
caps are subject to migration, not reasons to downgrade the protocol target.

| Owner | Changes |
| --- | --- |
| `kestrel-feature-mcp` | Official Client, Streamable HTTP, remote connections, OAuth SDK integration, registry discovery, MCP result preservation, tasks |
| `kestrel-sovereign` | Dependency cascade, agent credential service, governed dispatch, typed UI parts, approvals/waits, A2A handlers/client/card, persistence migrations |
| `kestrel-sovereign-sdk` | Remove draft wire models; shared host/feature credential and typed-part contracts; update all consumers |
| Host manifest packages | Survey published constraints and editable sources; release incompatible consumers before the host floor moves |

The coordinated Core cutover requires **SDK 0.42.1 or newer**. SDK
[PR #87](https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk/pull/87)
and [PR #88](https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk/pull/88)
add the shared computer-use, artifact, resource-scope and workspace-transfer
contracts. Both are included at SDK commit
`c01a6e933665bf7778c656a9b13dd8b79e371e36`. The immutable `v0.40.0` tag
at `eff4c1f86bee3a5cf5c40b4d36c95b3ee29e02a8` predates these contracts.
SDK 0.42.0 is published and provides the shared vendored A2A implementation.
SDK 0.42.1 ([PR #93](https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk/pull/93))
is also published. It corrects native HTTP Major.Minor version negotiation while
ignoring specification patch numbers. The retained optional gRPC handler's
version negotiation and the alternative HTTP version query carrier remain
tracked limitations in SDK [#91](https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk/issues/91)
and [#92](https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk/issues/92).
The query carrier needs correction before claiming complete Core conformance;
Core does not serve gRPC.
Core's public lock resolves its reviewed PyPI wheel and source distribution;
installed-package custody checks verify the native implementation. Consumer
constraints and acceptance tests must still verify the complete host feature
set. Core's native-A2A integration, conformance and deployment gates remain
required.

The cutover worktree removes `kestrel_sovereign/a2a/types.py` and
`kestrel_sovereign/a2a/agent_card.py`, which forwarded the draft models from
SDK 0.39. SDK 0.40 removed those handwritten models; Core's coordinated
adoption uses 0.42 and retires the old imports and serialized callers. Keep the SDK lightweight:
the official A2A dependency belongs to consumers of A2A, not every feature.
The MCP feature owns the direct `mcp` requirement. Core owns the compatible
lock and Phoenix dependencies; the dependency spike adds protocol packages as
experimental resolver inputs, not as new production dependencies of core.

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

Use the official SDK's standard `Client(mode="auto")` negotiation at connect,
including `server/discover`. Publish the selected revision in connection/tool
status before making the connection available. The SDK chooses a shared
published revision; a peer sharing none fails connection setup with its supported
versions preserved in the error. Do not implement a Kestrel negotiation fallback
or wire adapter. Existing connectors on a published handshake revision remain
usable through the official SDK; this does not preserve the private A2A format.
Gate extension support on the negotiated revision and capabilities.

For `2026-07-28` peers, use the modern extension APIs and per-request metadata.
Do not manually serialize `_meta` or retain session IDs in Kestrel. Capture HTTP
to verify no `initialize` or `Mcp-Session-Id` on modern connections, and assert
`Mcp-Method` plus tool/resource/prompt-specific `Mcp-Name`. On that revision,
retire `notifications/initialized`, `ping`, `logging/setLevel`,
`notifications/roots/list_changed`, resource subscribe/unsubscribe, standalone
HTTP GET streams and SSE resumption (`Last-Event-ID`/event IDs). New subscriptions
use core `subscriptions/listen`. Roots, Sampling and Logging are deprecated.
The SDK owns any handshake/session behavior needed for a negotiated earlier
standard. Test modern-peer conformance and gateway interoperability separately;
a successful older-revision call is not a modern conformance result.

Do not automatically replay a failed `tools/call` after stream loss. The
protocol requires a fresh request ID for a re-issued request, but gives no
blanket exactly-once guarantee. Mark the governed dispatch outcome unknown if
execution may have started. A retry requires policy approval and an explicit
server-supported idempotency mechanism or a tool known to be safe to repeat;
link any retry dispatch to the original rather than inventing a success.

| Host duty | Implementation and evidence |
| --- | --- |
| Authorization discovery | Official SDK RFC 9728 discovery, RFC 8414 and OIDC metadata; validate the resource and issuer before use |
| Client identity | Operator-configured stable HTTPS CIMD URL; public client, `token_endpoint_auth_method=none`; no embedded client secret |
| Authorization | PKCE S256 and one-use state bound to agent DID, connection, issuer, redirect URI, resource, scopes and expiry |
| Resource binding | RFC 8707 resource parameter on authorization and token requests; never forward a token to another resource/origin |
| Callback validation | SDK RFC 9207 issuer checks with exact comparison (table below); state, expiry and one-use callback checks before exchange |
| Refresh | Agent-scoped encrypted token bundle; serialize refresh/rotation for the same connection across requests; commit rotated credentials atomically |
| Discovery | Official Registry alongside Docker/local catalogs; explicit source and namespaced identity, opaque pagination, no arbitrary executable installation from search results |
| Apps | Preserve tool metadata, structured content and `ui://` resource association through a typed part; isolated renderer and governed app calls |
| Tasks | Capable official extension/SDK release (support gap #3684), tool-call result handling, durable wait provider |
| Operator input | Core MRTR (`InputRequiredResult`), governed input surface, no lock held across an interactive response |

### Authorization response issuer checks

Apply these RFC 9207 cases before sending a code to the token endpoint:

| `authorization_response_iss_parameter_supported` | Response `iss` | Action |
| --- | --- | --- |
| true | present | Compare exactly with the validated recorded issuer |
| true | absent | Reject |
| false or absent | present | Compare exactly with the validated recorded issuer |
| false or absent | absent | Continue with the other callback checks |

Compare the form-decoded string without case folding, default-port elision,
trailing-slash changes or additional percent-decoding/normalization. The same
checks apply to error callbacks before trusting or displaying error details.

### Credential storage and connect UI

Reuse the host credential subsystem's agent-derived encryption and database
ownership, surveyed in `kestrel_sovereign/security/service_key_storage.py` and
`kestrel_sovereign/security/agent_encryption.py`. Its existing API-key
`ServiceKeyStorage.store_key` is insert-only: replacing a live key belongs to
`kestrel_sovereign/features/keys/feature.py::rotate_service_key` and requires
constitutional approval (F196). OAuth refresh must not bypass that contract.

Add a typed OAuth credential capability in this subsystem, with separate
`agent_oauth_grants` and `agent_oauth_credentials` rows. Reuse the existing
agent encryption implementation with an OAuth-specific purpose; extract common
encrypted persistence mechanics if needed, without cloning crypto or changing
API-key write policy. Do not use host/user/sponsor master-key stores or
plaintext token files. The feature consumes a host-provided credential protocol
and cannot call an unrestricted `store_key(replace=True)` endpoint.

The authenticated connect flow creates a durable, scoped grant only after the
operator authorizes that agent, client identity, issuer, resource and scopes.
Token refresh is an operation authorized by that existing grant: atomically
compare the stored generation, verify that the grant remains active, and rotate
only that connection's encrypted token bundle. Refresh cannot widen scopes,
change the bound identities/resource or replace an API key. Disconnect revokes
the grant and removes its tokens atomically. The host checks the durable grant;
a caller-supplied grant ID or causation claim confers no authority. Test forged,
wrong-agent, revoked and racing-refresh grants alongside the existing F196
API-key tests. This keeps unattended refresh possible without an approval prompt
for every access-token expiry and keeps API-key replacement approval intact.

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

Enforce this before using `OAuthClientProvider`: discover and validate resource/
issuer metadata through the SDK, and reject CIMD mode unless that issuer
advertises CIMD. Pre-seed its issuer/connection-scoped `TokenStorage` client-info
record using the SDK's `create_client_info_from_metadata_url`, with the selected
CIMD URL as `client_id` and exact redirect URIs. `get_client_info` must return
that typed record; the SDK then has no need to enter its default DCR branch.
Operator-registered identities use their own explicit record instead. Do not
construct an unseeded provider and rely on a debug log to detect registration.
Acceptance must prove no registration POST for absent CIMD support, changed
metadata or invalid-client responses. If a selected SDK cannot enforce this,
upgrade/expose the SDK policy before release; do not patch a private auth flow.

### Registry and Apps

Registry discovery uses the official registry's currently published API schema
and a pinned contract fixture; its API version is independent of the MCP wire
version. Preserve source, server name, version and remote endpoint in catalog
identity. Local explicit configuration wins only through a documented override,
not an accidental name collision. Pagination failures and malformed entries
produce actionable errors; they must not become an empty successful catalog.

Initially disable the SDK response cache. When enabling it, honor each result's
`ttlMs` and `cacheScope`; private resources/listings stay scoped to the exact
agent, connection and authorization identity. Never share private `ui://`
responses across agents. Disconnect and identity changes invalidate the cache.

An MCP App resource is loaded from the same authorized connection as its tool.
The host validates MIME type and resource association, then renders it in an
iframe with sandbox/CSP restrictions and the official Apps bridge. Treat the
resource's `_meta.ui.csp` and `_meta.ui.permissions` as requests, not grants.
Intersect requested network/resource/frame/base origins with host-owned,
connection-scoped allowlists, and reject unsupported requests clearly. Device
permissions (camera, microphone, geolocation, clipboard) require explicit host
policy and user authorization; a vendor declaration cannot widen them. Default
egress/device policy denies access. Apply CSP and Permissions Policy before
loading content, including nested frames and navigation. See the official
[Apps CSP contract](https://apps.extensions.modelcontextprotocol.io/api/documents/csp-and-cors.html)
and [resource metadata schema](https://apps.extensions.modelcontextprotocol.io/api/variables/app.McpUiResourceMetaSchema.html).
Never inject
vendor HTML into the console DOM. No host credentials enter the iframe. Bind
each bridge message to the iframe source, session nonce, agent, connection and
origin; allow only the Apps methods the host implements. A requested tool must
be mounted and allowed for that app. Its invocation re-enters the same host
dispatch used by model calls, including approvals and the resulting
`a2a_tool_dispatches` row. Keep one governed registry of connection tools with
separate model/app projections. Honor `_meta.ui.visibility`: model-only tools
are excluded from app calls, app-only tools stay out of model schemas but remain
registered for governed calls from that server's own app. Omission defaults to
both surfaces; vendor visibility never grants authority beyond host policy.
See [tool visibility](https://apps.extensions.modelcontextprotocol.io/api/interfaces/app.McpUiToolMeta.html).
Unmount/disconnect invalidates outstanding bridges.

### Tasks and operator input

An MCP task is identified by agent, connection and remote task ID. Its durable
wait registration survives restart and maps terminal status into the existing
wait-provider contract. Advertise only supported extension versions. Poll,
cancel and result retrieval use the same authorized connection and policy
checks as the original call. Cancellation does not invent a completed result.
Declaring `io.modelcontextprotocol/tasks` allows a server to return
a `resultType: "task"` result instead of a final result on `tools/call`;
there is no per-call task opt-in. The 2026-07-28 extension supports task creation
only on tool calls. A task result on `resources/read` or `prompts/get` is an
invalid response, not a task to register. Keep the extension registration
extensible for future supported methods without claiming them today. Do not
advertise it until the durable wait provider and tool-call result path handle
both shapes. Register the task before releasing the originating dispatch.

The Phase 0 candidate `mcp==2.3.0` does **not** ship the 2026 Tasks extension.
Its exported `CreateTaskResult` is the old 2025 core type and must not be used
for this extension; the required native `tasks/get`, `tasks/update` and
`tasks/cancel` implementation is absent. Its tool-call-only result claims and
resource/prompt parsers follow the extension's current method set; they are not
support gaps. A6 is gated on
[#3684](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3684): select
an actual official extension/SDK package and revision supporting the published
tool-call task contract, or contribute that support to the official SDK, then raise dependency
bounds as needed. The candidate is not a ceiling to retain. No partial
advertisement, private wire types or generic-result adapter fills this gap.
Until that gate passes, do not advertise Tasks; surface a server's required-
capability error clearly instead of accepting work whose result cannot be read.

When `tasks/get` reports `input_required`, classify its `inputRequests` using
the same machine/operator rules and policy below. Respond through
`tasks/update(taskId, inputResponses)`, then resume observing status. This path
does not reissue the original operation or use MRTR `requestState`. Deduplicate
input keys for the task lifetime; bind responses to agent, connection, task and
outstanding key. Operator waits release the turn lock, survive restart and
respect unattended-turn denial. Test both input paths against the published
[Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/2026-07-28/tasks).

MCP's Multi Round-Trip Requests (MRTR) are core protocol, independent of Tasks.
A normal tool/resource/prompt request can return `InputRequiredResult` with
`resultType: "input_required"`, `inputRequests` and opaque `requestState`.
A1's basic client has no operator-input callbacks and can use the SDK's automatic
state-only driver. Before A6 advertises governed input, switch its governed
operations to the documented manual SDK session APIs, for example
`client.session.call_tool(..., allow_input_required=True)`, and the corresponding
resource/prompt APIs. Do not call the high-level auto-driver on those operations:
the host must receive each typed intermediate result to persist a continuation.
The host owns deadline, round limit and bounded backoff for manual reissues.
Empty `inputRequests` retries unchanged state without creating a human wait;
machine-handled requests likewise do not become operator prompts.

Elicitation is the supported operator request family; roots and sampling stay
off. Configure the SDK elicitation callback only once governed handling exists.
In SDK 2.3 that callback advertises both form and URL modes, so handle both:
forms enter the host's governed input surface; URL-mode requests initially get
an explicit audited standard decline, with no browser navigation. Enabling URL
flows later requires explicit host origin/egress policy and operator action.
For earlier negotiated protocols the callback declines operator-required input;
it does not pretend an old inline request has restartable MRTR state.
Unsolicited roots/sampling requests fail with the SDK's capability error.

On modern interactive turns, persist the request binding, release the turn lock
and return control. A new owned continuation reissues the **original operation**
with SDK `inputResponses`, byte-unchanged `requestState`, and a fresh request ID.
Durable storage survives host restart; remote state may not survive server
restart, key rotation or expiry. Rejected/expired state makes the continuation
explicitly non-resumable; retain an unknown dispatch outcome if execution cannot
be established. Never drop state and submit a fresh side effect automatically.
Results without `resultType` remain `complete` as the SDK specifies.

For operator-required input, signal-driven turns and
`NON_INTERACTIVE_SESSION_IDS` keep the shipped #3439 behavior:
`allow_blocking=False` and an auditable `no_approver` denial. This does not deny
state-only MRTR retries. Unattended turns do not acquire interactive authority
through a continuation. Reject stale/wrong-agent/wrong-operation human responses
and keep secrets out of conversations. Exercise state-only and operator-required
paths against the real turn scheduler, including #3439's lock-contention case.

## A2A 1.0 cutover

Use official SDK types and serializers. A2A 1.0 uses protobuf JSON conventions:
Python's `context_id`/`supported_interfaces` serialize as `contextId`/
`supportedInterfaces`; `Part` uses the content member, not the 0.2 `kind`
discriminator. Test the serialized HTTP, not just Python attributes.

A host explicitly selects its public discovery agent and serves that agent's
signed card at the standard authority-root `/.well-known/agent-card.json`.
Other per-agent public cards are reached by **direct configuration**, at
`/api/agent-routes/{encoded_routing_name}/agent-card.json`; these nested URLs are
not called well-known discovery. Encode the manager's registration key using
`kestrel_sovereign/multi_agent/route_name.py::encode_agent_route_name`. `agent.agent_id` is a DID,
and a mutable display name is also unsuitable for routing.

Both bindings use the manager's stable routing key to select the agent. The
AgentInterface entries carry the exact `tenant` value to be sent on every SDK
request. Validate that tenant against the route-selected recipient for send,
get/list/cancel/subscribe and push configuration before authorization or side
effects; do not silently route based on a caller-supplied tenant. A mismatch
fails with the binding's standard error. The sender copies the selected
interface tenant exactly; omit it when the selected interface advertises none.
`AgentInterface.tenant` is a no-presence protobuf scalar: its empty value cannot
be distinguished from unset and is omitted from canonical card JSON.

JSON-RPC carries tenant in the SDK request body. HTTP+JSON uses the SDK's
`/{tenant}` route alias and can also carry tenant in the JSON body (the SDK
REST client sends both on send/stream/cancel/push writes). The dispatcher does
not reconcile them: its call-context tenant comes from the path while the
parsed request keeps the body value. Validate both before handing the request
to authorization, task storage or a side-effecting handler. Every present REST
body tenant must exactly equal the decoded path tenant AND the value advertised
for the already-selected recipient/interface. On a bare REST route the path
represents the advertised default/empty tenant, so any non-empty body tenant
fails. This applies to standard-only peers as well as DID-extension peers.
The route-selected agent remains the sole recipient; neither tenant carrier
can select or override it. Reject a mismatch with the binding's standard error,
even with a valid peer transport credential. Serve tenant-qualified aliases
only for the exact non-empty advertised tenant and base routes for its default.
Choose path-safe opaque values and let the SDK encode/decode them. Derive
transport admission from the same binding registration, including both aliases
and REST tenant-qualified paths; do not maintain a separate pattern list.

The public card endpoints need narrowly matched, prefix-aware exemptions in
host authentication middleware, which runs before agent prefix stripping.
Exempt only the configured public cards; publish only explicitly public fields.
Authenticated details use the standard `extendedAgentCard` capability and
`GetExtendedAgentCard`, enforced through the same recipient identity/policy
path. A console list projection is not a replacement for either SDK method.
Let the SDK define method names, binding resource paths, errors and events.

`SDK request handler → verified principal + recipient policy → TaskManager → stores/worker`.

The native HTTP registration mounts both SDK bindings below
`/api/agent-routes/{encoded_routing_name}/a2a`: JSON-RPC uses `/rpc`, and
HTTP+JSON uses the mount root with the selected interface's tenant alias.
The route registry publishes the exact SDK route objects that host peer
admission matches, including HTTP methods and complete nested endpoints.
It does not maintain a second transport path pattern. Only the advertised
tenant is served; an empty tenant uses the SDK's base routes. Replacing a
registration retains its successor when an older owner withdraws.

The native guard captures the exact published recipient policy, task manager,
task store and hybrid identity. Each request separately retains the sender's
host identity witness around verification and authorization. Waiting requests
release the topology lease, then recheck their original policy and identities
before the first or subsequent stream event and final read; this does not spend
their replay nonce again. A replacement with the same DID and keys is still a
different host generation. Missing proof remains HTTP 401, recipient refusal
remains HTTP 403, and database failures become the standard internal error with
the original retained for host diagnostics. An operator key cannot promote a
native peer request to sovereign authority.

Host onboarding now prepares a signed native card privately. The manager's
synchronous registration commit publishes that card and both SDK bindings under
its topology writer; policy revocation withdraws exactly that registration's
routes. Public card reads retain the matching reader through profile I/O and
recheck the original policy and identity. Live names and skills are refreshed
before signing; the manager's routing key and interface URLs remain stable.
The console's `/api/agents` response uses public display profiles independently
of this protocol publication.

Public hosting requires `KESTREL_BASE_URL`. Without it, operator APIs remain
available and no peer card or binding is published; a listen address is never
substituted for a public URL. `KESTREL_A2A_DISCOVERY_AGENT` explicitly selects
the registration key served at the authority-root well-known path. Other
recipients expose only their directly configured card URLs. Blank or invalid
configuration refuses publication.

This is a private implementation candidate, exercised with unpublished SDK
builds containing the reviewed signing and HTTP-authorization repairs. Push
delivery and draft-route retirement are implemented. Released dependency
adoption, TCK and cross-framework interoperability remain acceptance work.
The offline reviewed migration is implemented; it has not been run on
operator data. The isolated HTTP tests do not establish release acceptance.

The native task manager records the handler's admitted Task as its baseline.
Lifecycle and cancellation writes apply that handler's delta to the latest
recipient- and tenant-scoped row while holding its database lock. Independent
artifact chunks, progress history and metadata survive finalization. A handler
cannot remove published artifacts, rewrite committed history or silently
replace data changed by another writer; conflicting edits fail before a write.
The store returns the native payload it committed, including server-owned
authority fields. A lost commit acknowledgement is reconciled through its
private operation token and an authorized canonical read. If that read is
unavailable, execution retains the infrastructure error instead of returning
the attempted snapshot or running the handler again.

Local decorated-tool results use the SDK's standard inline raw artifact with
`application/json`. JSON encoding rejects non-finite values before the task
claims completion and retains integer types and large identifiers that a
protobuf Value would round to binary64. Commands and workflows share one
projection for native data values and inline JSON bytes; that projection never
fetches a file URL. Primary-store and official SDK JSON round trips retain the
same embedded bytes.

Integrate the SDK request-handler and executor interfaces with the existing
TaskManager lifecycle. Do not create a second task lifecycle or store. Both
bindings enter the same authorization path, including get/list/cancel/subscribe
and push-notification configuration. Multi-tenant routing binds the recipient
before signature verification; a signed audience cannot select another tenant.

### Native context ownership

A peer-supplied `contextId` does not authorize access to local chat-session
state. The native context ownership ledger is keyed by recipient, selected
tenant and context ID, and binds the context to its verified creator. Reserve
it after sender verification and recipient authorization, before admitting a
task. Both bindings use the same ledger. A different sender cannot acquire an
existing context by submitting another message with its ID.

For a continuation, obtain the context from the authorized existing task and
reject an inconsistent message context. Read operations check ownership
without creating a context. Session history remains a projection; it cannot
grant context ownership. Populate the ledger during the one-time migration
from authoritative task principals, and fail preflight on ambiguous ownership
rather than choosing an owner from caller metadata or historical display names.

### Admitted task input and execution ownership

Outbound follow-ups use `send_a2a_task(continue_dispatch_id=...)` with the latest
acknowledged local dispatch. Each message gets a distinct immutable dispatch ID
and message ID, with durable parent and root dispatch references. The root owns
the peer's task ID. Follow-ups retain that task, context, recipient, tenant,
signing identity and selected interface; a new task response is refused. Only
one follow-up can claim a parent until that attempt is proven not sent. Lost
responses are reconciled through authorized history reads without replaying the
message. Every ancestor must still acknowledge active work and have no pending
cancellation when a follow-up is reserved and immediately before peer HTTP.
A question follow-up retains the original question's correlation and deadline;
it cannot become independent work by using another tool or delivery verb.
The native core store bundle is revision 5. Earlier private candidate schemas
are refused before startup writes; the runtime does not upgrade them in place.

Incoming `SendMessage` continuations append the original message to the same
creator-owned task. An omitted context is inferred from that task; a conflicting
context or a terminal task is refused. The full official request, message ID
and deterministic request digest commit in `a2a_task_inputs` with the task and
its server-owned `input_revision`. Retrying an identical retained message does
not admit another input or wake; reusing its ID for different content is refused.

Each admitted input has a distinct, unambiguous durable wake identity. Recovery
of a WORKING task first checks the original wake for that same input. Privacy
recovery cannot substitute a newer input into an earlier event. A source-store
read that proves withdrawal returns an explicit terminal no-op, which settles
through the dispatcher's existing lease-conditional delivery path. Missing
recovery data remains an error and never authorizes an acknowledgement.

The source revalidates the input revision before cognition and monitors it
through execution. All recipient lifecycle, terminal, artifact and decline
writes made by that signal also include the input revision in the SQL authority
predicate. If a newer input wins first, the old turn cannot publish over it.
The dispatcher physically joins a withdrawn execution before settling its
receipt. Explicit creator cancellation retains its existing authority.

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
| Cooperative peer Stop | SendMessage plus `#peer-stop-v1` | Keep dispatcher policy and durable Stop receipts; create no cognition task |
| Wake signals | Internal transition events | One committed transition drives existing wake hooks |

The operator console reads `/api/agent/task-inbox`, independently of the A2A
binding namespaces. Its task detail includes the complete official Task JSON;
its summary and the task tools share one ownership-based projection. Claimed
sender/skill metadata never replaces the committed creator or URI-keyed hint.
Generic outbound waits use `a2a:<dispatch_id>` and the native outbound ledger;
questions retain their separate durable resumption rail.

Same-host peers use real HTTP against the configured published URL for Agent
Card discovery and SDK operations. Sender egress and receiving recipient policy
run in separate request contexts. A source turn's context variables or database
execution custody must not become recipient authority. Blocking disconnects
close only their response subscription; the accepted task remains durable.

### Recipient policy v1

Native admission authenticates the caller and applies recipient policy before
creating a context or task. The committed task's creator, recipient and tenant
columns are the authority for later reads and wakes. A native wake snapshot
carries these admission facts together with its context and lifecycle revision;
it is not an independent authentication proof. Every public producer must pass
the verified caller and host-selected recipient scope to the same admission
path. Caller metadata such as `sender`, `sender_verified`, or a claimed creator
never changes those facts or the signal's trust.

Immediate callbacks, restart reads and privacy-elided signal rehydration use
the same snapshot. Reject missing principals and a recipient/context mismatch.

Wake eligibility is server-owned durable policy, independent of peer metadata.
Incoming requests permit input and completion cognition. Local synchronous
skills permit neither; their caller already receives the result. Local
asynchronous skills permit completion cognition only. Restart reconciliation,
signal builders, privacy rehydration and execution guards honor the same policy,
and local skill tasks reject subsequent incoming messages. Recovery does not
replay a physical local tool invocation after process loss.

The native payload-encoding revision does not prove the whole store bundle is
installed. TaskManager records core, archive and feedback schema completion
separately, after their canonical initializers succeed in the same shared
transaction. A repeated boot checks those durable revisions under the schema
lease and skips completed DDL; adding an optional store initializes that unit.
Unknown or incomplete revisions refuse startup. Schema failures roll back the
bundle and its completion records; startup does not hide missing migrations.

Peer context IDs map to a host-chat projection that includes recipient and
tenant; a peer cannot select an existing human chat session by naming it.

Reuse `RecipientA2ASenderAuthorizer`, scoped requester contexts and durable
cancellation authority. An external peer may authenticate with a standard
advertised security scheme without implementing the DID extension only when
the recipient explicitly permits that principal and the extension is optional
on that card. A recipient requiring DID proof must advertise it as required;
do not lower that policy to pass an interop test. Standard-only peer tests use
an explicitly provisioned principal and a compatible card policy.

### Question correlation v1

The metadata key is this document's `#question-correlation-v1` URI. Its payload
has exactly `version`, `kind`, and `question_id`:

```json
{"version": 1, "kind": "question", "question_id": "sender-owned-question-id"}
```

`kind` is `question` or `answer`; `question_id` is a non-empty opaque string
chosen and persisted by the question sender before dispatch. An answer uses
that same ID. These fields describe correlation and never authorize settlement:
the receiver must match the outstanding question's durable sender, recipient
and tenant before recording an answer or emitting a question-answered wake.
The native intake/outbound integration must enforce that ledger check; parsing
or displaying this payload alone does not satisfy the correlation gate.

The sender reserves an immutable native dispatch and its primary-store question
intent before the SDK can admit HTTP. Both the initial send and the final HTTP
dispatch gate require that exact live intent and deadline. Only a committed,
authorized native response with matching question, recipient, tenant and response
digest can settle it. A new SendMessage cannot introduce an `answer`; answers
belong to the original task's agent-role status Message. Taskless Messages retain
their real response body and never acquire an invented remote task ID.

The native question ledger commits terminal receipt, retained answer and wake
outbox together. Its resident observer uses only GetTask/ListTasks/CancelTask
through the official client and the retained stable route. A live original
sender is never reconciled as a crashed submission. Unknown sends or cancellation
attempts are observed without replay. `KESTREL_A2A_QUESTION_POLL_SECONDS` configures
the observation interval (positive finite seconds, default 1).

Question read, deadline and wake polling share `DatabasePollRetryPolicy` with
the push resident. Its default budget is three attempts per pass; successful
passes reset that budget. `KESTREL_A2A_QUESTION_DATABASE_RETRY_ATTEMPTS` configures
the question budget, with the observation interval between attempts. Only
recognized native database contention and availability errors qualify. Permanent
SQL failures, untyped database wrappers, original authority loss, acknowledged
or uncertain commits and native cleanup debt remain failures. Recovery retains
the same resident owners and never replays an original peer send.

Wakes retain deterministic sender-scoped event IDs across restart. The outbox
advances only on a canonical durable signal admission receipt; the dispatcher
owns cognition and retry after that commit. Deadline and late-answer waves are
distinct, ordered events. A deadline alone cannot assert withdrawal. Privacy-
elided durable events rehydrate from this same sender-owned outbox. The asking
turn's local chat session remains separate from the peer's A2A context.

### Delivery intent v1

The optional `#delivery-intent-v1` extension carries exactly
`{"version": 1, "intent": "notification"}` or
`{"version": 1, "intent": "assignment"}` under its URI metadata key.
Notifications request no reply; assignments request task processing. Question
correlation already defines question intent, so combining these extensions is
invalid. Neither intent grants execution authority. Every known URI payload
requires explicit extension activation; metadata cannot activate itself.

### Peer Stop v1

Hosts providing cooperative Stop advertise the optional `#peer-stop-v1`
extension on their signed Agent Card. Clients activate it through
`A2A-Extensions` and send `{"version": 1, "intent": {...}}` under its URI
metadata key. The intent uses the canonical Stop schema: `scope`, `target`,
`reason`, `cascade` and `correlation_id`. The user Message contains exactly one
text Part with the same reason. It cannot carry a task ID or combine Stop with
a skill hint, delivery intent or question correlation. Signed sender admission
and the recipient's current policy precede every dispatch; causation retains
its usual cycle and depth checks.

Both SendMessage and SendStreamingMessage return one agent Message containing
the canonical durable Stop receipt as inline `application/json`. The response
keeps the requested context and creates no task, work item or cognition wake.
The existing dispatcher decides rate limits, cycle policy, cascade refusals,
the fleet circuit breaker and effect ownership. Stop cancels work cooperatively;
it does not terminate a process or confer Hold authority.

Only a transport failure can trigger a bounded Stop redelivery. Each delivery
uses a new native Message ID, dispatch record and sender nonce while preserving
the same intent, context and Stop correlation ID. The recipient's durable Stop
receipt prevents repeating the effect. Ordinary task and question sends retain
their single-attempt dispatch contract. Exhausted or interrupted delivery keeps
its uncertain dispatch evidence and never reports a confirmed Stop.

### Skill hint v1

The optional `#skill-hint-v1` extension carries exactly
`{"version": 1, "skill_id": "published-skill-id"}` under its URI metadata key.
The selected signed card must advertise that skill. The hint describes intent
for the receiving agent; it neither invokes a provider directly nor grants tool
permissions. Both sender and receiver admit input against the selected skill's
declared input modes. A skill with no explicit modes inherits the card defaults;
without a hint the card defaults apply.

`send_a2a_task` carries input handoffs in standard `Message.parts`, including
text, filenames and arbitrary JSON `data` values. Artifacts are outputs of a
task. The inbox exposes the complete original native Message, preserving those
parts. Kestrel currently advertises `text/plain` and `application/json`; it
rejects unsupported or malformed declared MIME types before creating a task.
Omitted MIME types use the content-kind default; malformed values never use a
default. A JSON-only skill can receive `parts=[{"data": {"request": "hello"}}]`
with the optional `message` omitted. Nonempty text is preserved and must also
satisfy that skill's contract; the sender never converts or discards it to make
the request fit. Input must contain message text or at least one standard Part.
File URLs are content references, not permission to fetch them.

### Causation v1

The metadata key is this document's `#causation-v1` URI. Its payload has exactly
`version` and `chain`. The chain contains canonical SDK `CausationFrame` values:

```json
{"version": 1, "chain": [{"agent_id": "did:example:sender", "source": "cron.followup", "signal_id": "source-event-id", "turn_id": null, "depth": 1, "emitted_at": "2026-10-20T00:00:00+00:00"}]}
```

Frame identity/source strings must be non-empty, depth is a non-negative
integer, and timestamps require a timezone. `turn_id` may be omitted or null.
Reject malformed declared extensions before task/context admission; do not
drop broken frames and restart a chain. Protobuf Struct's integral numeric
forms such as `1.0` retain the same numeric meaning as `1`; booleans are not
versions or depths. Causation supports attribution and loop detection only.
It never chooses the creator, recipient or authorization policy.

### Signed sender v1

The proof travels in exactly one `Kestrel-A2A-Sender-Proof` HTTP header for
every native operation, including reads and subscriptions whose request types
have no metadata. Activate the extension through `A2A-Extensions`. Its compact
encoding is `base64url(claims).base64url(classical_signature).base64url(pq_signature)`;
all segments are unpadded and canonical. This format is an extension proof,
not JWS. Reject duplicates, non-ASCII headers and headers exceeding the host's
configured size limit before resolving a DID. The initial limit is 8192 bytes.

Claims are RFC 8785 bytes with these exact fields: `version` (1), `protocol`
(`"1.0"`), `extension` (this extension URI), `sender`, `recipient`, `operation`
(the SDK service operation name), `request_digest` (lowercase SHA-256 hex of
the complete official ProtoJSON request), `timestamp` (integer Unix seconds),
`nonce` (32 lowercase hex characters), `extensions` (sorted unique activated
extension URIs), and `signers` (the ordered `{alg,kid}` entries). V1 uses
`ed25519` followed by `ml-dsa-65`; signatures cover the exact claims bytes,
including their signer identifiers. The separate proof header excludes no
field from the native request digest. Claim JSON must be canonical, with no
unknown fields or duplicate keys. Validate required native request fields and
the actual operation/input-type pair before hashing.

The default verification window permits 300 seconds of clock skew in either
direction. Spend a verified nonce in the mandatory shared replay store for
twice that window, covering a future-dated proof's full validity. A replay
store failure or uncertain reservation propagates; there is no process-local
degradation or nonce rollback. These limits are host policy configuration.
The proof establishes identity only. Recipient authorization and live
identity-epoch validation must still precede task creation or other effects.
This schema is an implementation contract; Core route integration, signed
cards, conformance and interoperability remain release gates below.

The maintained foreign-peer exercise is
`tests/interop/test_a2a_go.py`, with its public Go SDK dependency and independent
hybrid extension implementation in `tests/interop/a2a_go_peer`. The
`a2a-go-interop` CI job runs it after unit tests, including on the release ref.
It initializes an isolated SQLite Kestrel agent and verifies signed discovery,
task completion in both directions over both bindings, durable creator and
outbound ownership, wrong-recipient refusal, and rejection of an unrelated
Agent Card trust anchor. Its deterministic business handler completes canonical
TaskManager tasks; it does not establish LLM quality or a deployed public
frontdoor.

`tests/interop/test_a2a_tck.py` runs a Kestrel-maintained A2A 1.0.1 TCK snapshot
against the same real native publication and TaskManager. Its public upstream
base, pinned specification, complete source/test patch and license are recorded
in `tests/interop/a2a_tck/`. The fixture checks those inputs and the complete
patched Git tree, creates its own checkout and locked environment, and verifies
the runner's unit tests. It never borrows an operator checkout or environment.
The reviewed corrections enforce native error identities and normative levels,
validate Message fields in direct replies and tasks, and exercise subscription
closure through an actual WORKING-to-CANCELED transition. Its explicit HTTP auth
provider adds Kestrel sender proofs and activated-extension headers without
rewriting protocol requests, responses or assertions.

CI runs MUST checks followed by the full runner on JSON-RPC and HTTP+JSON. Both
runs require the exact reviewed MUST observation matrix in
`tests/interop/a2a_tck/COVERAGE.json`: 72 passed requirements, 20 conditional skips,
7 manual obligations and 16 requirements without automated probes. The raw
automated MUST score is 81.8%; this is a coverage gate, not complete conformance.
The profile records each missing probe explicitly. Authentication and authorization
scoping, in-task authorization, version negotiation, binding equivalence and gRPC
TLS still need additional evidence; some can be automated. Missing or changed
requirements, a new skip, any MUST failure, runner error, missing report or lost
Message/terminal-stream pass on either binding fail the gate. A profile change
requires review against the pinned runner source.

Raw JSON, HTML and JUnit reports retain all manual obligations, optional-capability
skips and SHOULD/MAY failures. The full run permits only failed checks that map
solely to advisory requirements; fixture errors and unmapped failures remain
fatal. This is a pinned, corrected runner, not an unmodified upstream TCK or a
claim that automated checks establish every requirement. Run it locally with
`uv run pytest tests/interop/test_a2a_tck.py`; no external TCK-source override is
accepted. Adoption of published dependencies and the full host feature manifest
remain separate release requirements.

The extension contains sender DID, recipient DID, operation, timestamp, nonce
and the sender proof. Bind the complete official request payload (including
non-text parts, configuration and all behavior-affecting extension metadata)
using RFC 8785 canonicalization, excluding only the proof field itself. Both
signer and verifier first serialize with the official protobuf JSON presence
rules: omit unset optional fields and implicit defaults, retain required fields
and explicitly set optional defaults. Include the interface `tenant` with the
same presence as the official request; test equivalent forms and deliberate
presence changes. Do not hash an arbitrary Pydantic `model_dump()` projection.
Do not reuse the draft envelope's projection of selected text fields. Preserve
hybrid sender verification through the existing identity and verification-policy
services. Define algorithm identifiers and canonical-byte test vectors in the
extension implementation before release; no ad hoc algorithm is called JWS.
Reserve replay nonces through the existing shared replay store, with documented
retry/idempotency behavior. Reads/cancels also bind method, target and audience.

The Agent Card has a separate standard JWS signature over the SDK card's JCS
representation. Before JCS, apply A2A 1.0 section 5.7/8.4.1 presence rules:
omit optional-and-unset fields and implicit defaults, retain required fields
and explicitly set optional defaults, and exclude `signatures`. The normative
[section 8.4.1](https://a2a-protocol.org/v1.0.1/specification/#841-canonicalization-requirements)
is the canonical-byte oracle, not a same-helper signing round trip.

**Signed-card release gate:** SDK 0.42.0's reviewed
`kestrel_sdk.native_a2a` implementation must pass that oracle, the checked-in
canonical-byte/signature vectors and the independent Go verifier. Core consumes
the published SDK's `a2a` extra; it does not wait for an upstream major release.
The [SDK 0.42.0 change](https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk/pull/90)
vendors the corrected official implementation with pinned source, its complete
patch, license and source/artifact custody checks. The same implementation
produces feature skills and handles Core's native requests.

The unpatched public `a2a-sdk==1.2.3` does not implement that oracle: its default
`MessageToDict` serialization omits required no-presence fields at their defaults
(including `description` and required repeated fields), then recursive empty
cleanup removes explicitly present empties such as optional
`documentationUrl: ""` and required `capabilities: {}`. Both its signer and verifier
share these deviations. Removing only the cleanup still loses required defaults;
printing every protobuf default includes implicit optional values that the
normative rules omit. The vendored repair respects each field's required
annotation and presence. There is no parallel Kestrel canonicalizer, relaxed
verification or removal of valid empty fields.

The normative canonical-byte/signature vectors cover required empties, absent
versus explicitly empty optional values, explicit false capabilities and the
implicit empty interface tenant. Both vendored SDK helpers and an independent
verifier must pass before releasing signed cards and the coordinated A2A
cutover. Plain `model_dump()` plus JCS is insufficient.

Upstream required a library major release because correcting signed bytes
changes its compatibility contract. The contributed
[PR #1360](https://github.com/a2aproject/a2a-python/pull/1360) targets that release.
This is upstream release context, not an outstanding dependency for Kestrel's
published vendored implementation. The protocol remains A2A 1.0.1. Full Core
conformance, interoperability, feature acceptance and CI remain separate gates.

Use the agent's Ed25519 verification method and a DID-bound `kid`; the existing hybrid message
proof is not substituted for an interoperable Agent Card signature. Verify the
card against a trusted DID resolution result before trusting advertised URLs
or capabilities. A card containing its own untrusted key is not a trust root.
Card signature verification does not authenticate an incoming caller.

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

The sender owns an immutable local dispatch ID and original message ID. The
remote task ID is nullable until acknowledged; accepting a remote ID never
rekeys the dispatch or another dispatch's retained route. Bind the directory's
stable recipient, verified card identity, selected interface and tenant before
submission. Record submission intent durably before entering the SDK send.
After an uncertain acknowledgement, retain `UNKNOWN` and reconcile only by
authorized reads of the original context/message. An absent result is not
permission to replay a send. Taskless native Message responses retain their
actual type rather than acquiring an invented task ID. Questions use the
sender-owned correlation ID and the same durable recipient/tenant binding.
Reconciliation requests full task artifacts. A terminal question answer and its
durable wake retain the complete response, including artifact-only details, so
a later result read agrees with the acknowledged answer.

History reconciliation has a finite `NativeReconciliationPolicy`. The common
hosted and standalone peer-session factory loads its settings from environment
variables with the `KESTREL_A2A_RECONCILIATION_` prefix:

| Suffix | Default | Bound |
|---|---|---|
| `PAGE_SIZE` | SDK default, 50 | Tasks per page; validated against the SDK's protocol maximum |
| `MAX_PAGES` | 20 | Pages in one complete history scan |
| `MAX_RESPONSE_BYTES` | 8 MiB | Cumulative decoded protobuf response bytes |
| `MAX_PAGE_TOKEN_BYTES` | 4096 | UTF-8 bytes in a pagination token |
| `TIMEOUT_SECONDS` | 30 | Elapsed duration of the peer-history scan |

All settings require positive integers. The scan retains only the canonical
matching witness and checks ambiguity across every page before admitting a
receipt. Exceeding a bound or repeating a token raises a peer protocol refusal,
preserves the original `UNKNOWN` dispatch and lets question recovery continue
with its next retained question. The elapsed bound joins the original read's
HTTP retirement; unrelated timeouts and original execution-control errors retain
their identity. An incomplete scan grants no permission to resend the message.
The resident currently observes due questions serially. Local deadlines and
wake handoff have an independent lane, but several slow reads can delay another
peer's answer or remote withdrawal. [#3800](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3800)
tracks bounded fair observation using retained recipient and tenant identities.

Native message, tracked-task and blocking `ask_agent` tools share the SDK
producer. Result lookup accepts the immutable local `dispatch_id`, reads the
sender's canonical primary ledger and resolves its stable peer under current
authorization. It never routes by a caller-supplied remote task ID. Reads record
the observed native Task by compare-and-swap and cannot regress a terminal
outcome. The outbound audit exposes the selected tenant, delivery state, remote
task ID and cancellation intent without storing message bodies.

Failure before the original native HTTP dispatcher is admitted records
`not-sent`; possible delivery records `unknown`. Neither permits an implicit
SDK replay. Cancellation records a separate intent on the same dispatch before
entering the SDK. A lost cancellation acknowledgement preserves the known
remote identity and requires read reconciliation. An active read cannot prove
that cancellation did not happen; a terminal read settles the retained outcome.
Only a cancellation proven never dispatched permits a new explicit attempt.

Ship inbound and outbound together. Delete `POST /api/agent/tasks/send`, old
task read/cancel/subscribe routes, all draft wire models and all callers.
Replace the retired entries in `kestrel_sovereign/a2a/transport_auth.py` with
classification from the actual installed SDK route objects, in the peer lane
of host authentication before operator-key handling and agent-prefix stripping.
Include only the REST tenant aliases advertised by that registration. Public
card exemptions use its exact card routes and methods. Do not retain a second
method/path pattern list or add old per-agent path aliases.
Adjacent operator endpoints must remain denied. Do not repurpose the operator
API key as peer authority. Drain or explicitly stop old in-flight work during the host
upgrade; mixed draft/new workers are unsupported. Rollback requires the saved
pre-migration database and a coherent old host/SDK set, not a parallel handler.

### Retired surface coverage

The cutover deletes the private task worker, local-submission and envelope
services, draft TaskStore and pending-question store, legacy outbound ledger,
and private send/read/cancel/subscribe/Stop routes. Historical schema/migration
code remains solely for the explicit offline migration; it is not a serving
adapter. The tests for the removed protocol are replaced by these contracts:

| Retired contract | Native checks in `tests/unit/` |
| --- | --- |
| Private routes, draft model imports, host callback seams | `test_a2a_native_retired_surface.py`, `test_a2a_native_host.py` |
| Unsigned sender tiers, draft envelope signing and recipient reads/mutations | `test_a2a_native_sender.py`, `test_a2a_native_security.py`, `test_a2a_native_http.py`, `test_a2a_native_host.py` |
| Draft task/session lifecycle and local submission | `test_a2a_native_task_manager.py`, `test_a2a_native_context_ownership.py`, `test_scoped_agent_did_guard.py` |
| Legacy outbound audit, questions, replay, expiry and owed wakes | `test_a2a_native_outbound_store.py`, `test_a2a_native_question_store.py`, `test_a2a_native_question_supervisor.py`, `test_a2a_native_host.py` |
| Private SSE forwarding, cancellation and HTTP resource custody | `test_a2a_native_client.py`, `test_execution_custody_review49_native.py`, `test_execution_custody_review54_native.py`, `test_execution_custody_production_routes.py` |
| Draft Stop transport | `test_a2a_native_peer_stop.py`, `test_peer_stop_signals.py` |
| Pending-question service storage and store shutdown | `test_execution_custody_a2a_merge_native.py`, `test_a2a_native_shutdown.py`, `storage/test_sqlite_timestamp_defaults.py` |

The integration replacements are `test_native_a2a_authority_backends.py`
(principal/recipient/tenant isolation, independent-connection CAS and cancellation
races, and read-only refusal of draft databases) and
`test_native_a2a_subscription_original_control.py` (real PostgreSQL release
receipts, original generator retirement, and loopback SDK subscriptions on both
bindings). The old integration tests for draft store auto-upgrades, PostgreSQL
TaskStore wrappers and private SSE framing are retired with those implementations.
Active wait, identity, context, cancellation and storage-parity integration
callers use native snapshots and the primary ledger.

This maps safety coverage, not a conformance or deployment claim. The final
release still requires the published dependencies and acceptance gates below.

### Native push delivery

Push configuration, HTTP request retirement and notification acknowledgement
are separate facts. `PushConfigStore` binds callback credentials to the native
task's creator, recipient and tenant. The shared AEAD service-key hierarchy
protects the complete configuration, with its task, owner and generation in
associated data. Reads return independent official protobuf values.

The worker reauthorizes the stored creator against the exact publication's
live recipient policy before reading the task, before egress, and at final
HTTP admission after DNS and lease renewal. Its in-flight monitor repeats that
check and joins the original HTTP owner when access is revoked. A request
already admitted may have reached its callback; revocation cannot retract it.
Publication withdrawal or replacement cannot silently grant the old worker a
new authority. These checks authenticate no new sender and replay no request
nonce. Native database errors retain their original polling disposition.

A configuration ID names one current value; an omitted ID defaults to the task
ID. Repeating an identical normalized configuration preserves its generation,
consumer and initial snapshot. Changing it revokes the old configuration and
consumer, then inserts a new immutable generation in the same transaction.
The old ciphertext and attempts remain until their actual HTTP owners retire.
The resident retires ready revoked generations in finite indexed batches, so
replacement does not require a later client Delete to destroy old credentials.

An active configuration remains retrievable with its task, and its retired
attempt receipts remain until configuration deletion. `retention_days` bounds
durable notifications; it does not currently purge active configurations or
those receipts. [#3799](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3799)
tracks a coordinated bounded retention policy that preserves uncertain COMMIT
outcomes and unresolved original HTTP owners.
It also tracks denied callback claims: these currently retain their notification
until a later claim or explicit lifecycle cleanup. Publication unavailability
must remain distinct from confirmed creator revocation when settling those
claims and counting actual HTTP retries.

`NativePushTransport` sends an official `StreamResponse` using the SDK's media
type and version constants. It validates HTTPS egress, pins the approved IPs,
verifies TLS and refuses redirects. Its native response/client owners retain
physical close through cancellation. A total exchange deadline also bounds a
peer that keeps making partial header progress inside each socket deadline.

Deletion first revokes new admission and inventories every original generation
under the task lock, then waits for their actual HTTP owners to retire. A setter
that commits after revocation creates a fresh generation outside that inventory;
the earlier deletion cannot erase it. An expired lease is not physical retirement.
Unretired requests block both the deletion acknowledgement and task cleanup;
confirmed deletion clears credentials and completed request rows atomically.
The sender inventories its attempt before an admission COMMIT can become
uncertain, so a lost acknowledgement before HTTP begins is recoverable without
inventing a network request or replacing the original database outcome.

Each attempt also persists its original physical process witness before HTTP
admission: kernel boot UUID, PID namespace, PID and raw kernel process
birth identity. Capture, dispatch checks and ordinary retirement require that
same process identity. Hostname and epoch timestamp are diagnostic metadata;
changing either cannot revoke ownership, replace the original retirement
receipt, or establish or hide an exit. Strict observation
uses Linux boot-relative process ticks or Darwin's kernel process lifetime ID.
Only the original kernel/namespace and a changed birth identity or a
kernel-confirmed absent PID can establish that the original process exited.
Opaque reads and denied process observations retain the attempt.

Delete observes its exact revoked-generation inventory; the resident observes
finite indexed batches with a rotating inventory hint so live owners cannot
starve later exited owners. Observation records physical retirement only: it
neither acknowledges a notification nor grants another HTTP request. Native
startup requires encryption and this kernel identity capability before push DDL
or publication. Automatic observation supports Linux and macOS; unavailable
kernel interfaces refuse push startup explicitly.

An observer in another kernel or PID namespace cannot prove the original
owner exited. Such deletion remains pending, including after a reboot or
container replacement without an authoritative containment receipt.
[#3796](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3796)
tracks that separate receipt capability; no lease, heartbeat, host name or
force-clear flag substitutes for it. The general launcher/serving-record
migration to the shared kernel birth identity contract is tracked in
[#3797](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3797).

`TaskChangeStore` records metadata-only references through SQLite and PostgreSQL
triggers in each original task transaction. Capture requires an active callback
generation and its exact active durable consumer; ordinary tasks and disabled
push recipients accumulate no unused journal. The optional push initializer
installs capture after its configuration, subscription and consumer stores.
The current core schema bundle is revision 5 and push is revision 4. Earlier
native capture or generation layouts require offline migration, including partial old bundles.
Offline conversion suspends the canonical trigger inventory while rewriting
history and restores it before commit, without producing new work for historical
tasks. Task cleanup removes its journal references.

`NativePushOutbox` commits encrypted configuration, durable consumer registration
and an initial snapshot reference together. A creator-authorized task row lock
orders that registration against task changes. The existing `DurableSignalStore`
owns fan-out, claims, leases, retry and acknowledgement. Forwarding commits the
signal event, initial deliveries and journal receipt together; stable source IDs
preserve original deliveries after replay. Push consumers persist a future-only
contract: registration and event fan-out share the original source handoff,
while the explicit snapshot supplies the current state. Claims never backfill
retained history for these consumers. Other durable consumers retain their
explicit replay contract. A recipient/tenant-derived source scope and immutable
journal boundary prevent cross-tenant delivery and sending older queued changes
to a new subscription.

`NativePushWorker` reads each notification from the latest canonical task under
the subscription's creator authority. It renews the original delivery token and
checks it again at actual HTTP dispatch. Acknowledgement follows physical HTTP
retirement. A successful terminal-task POST acknowledges its original lease and
deactivates that consumer in the same transaction. Configuration reads remain
available until creator deletion. Each resident pass forwards, claims and retires
finite batches. Its read-only due inventory uses live delivery indexes; idle
subscriptions and retained event payloads never enter a claim loop. Retention
deletes only terminal events from the exact private push source and removes
journal references to those exact deleted IDs in one transaction. Pending,
retrying and leased deliveries retain their events. Retry diagnostics contain
error types, excluding callback secrets.
Failed callback exchanges use exponential backoff from `RETRY_SECONDS`, capped
by `RETRY_MAX_SECONDS` (default 300 seconds). `DELIVERY_RETRY_ATTEMPTS` bounds
each notification's persisted delivery attempts (default 8). Exhaustion records
an explicit durable `FAILED` disposition after the original HTTP owner retires;
it does not acknowledge delivery. Restarting a worker cannot reopen that failed
notification. A later task-change notification has its own budget, and the
configuration remains available to its authorized creator. These operational
limits do not override uncertain commits, lost authority or physical cleanup
debt. All settings use the `KESTREL_A2A_PUSH_` environment prefix.
The due queue orders pending work by its initial eligibility, retries by their
next attempt time, and expired leases by lease expiration. A failed callback
therefore goes behind older waiting work. At most `BATCH_SIZE` subscription
children run concurrently; each retains its own claim, HTTP owner and ACK/retry
transition. One slow callback cannot serialize every healthy callback in that
batch. Batch cancellation joins all accepted children before returning.
Owned database transitions finish before cancellation is reported, preserving
actual commit failures. Delete revokes configuration and consumer admission in
one transaction, joins local request owners, then requires the durable retirement
receipt. An owner whose retirement cannot be proven leaves deletion pending
with credentials retained; an expired lease cannot establish retirement. Active configurations
also block task cleanup before their first HTTP request.

After HTTP resources close, the original transport owner retains that fact while
retrying only its idempotent retirement write through known database outages.
The same attempt and task span every retry burst; no POST is repeated. Delete
and observer cancellation join that original writer rather than interrupting
its native COMMIT. The retirement CAS has an explicit transaction on both
backends, including embeddings without execution custody. A missing COMMIT
reply keeps its uncertain outcome and prevents delivery acknowledgement;
permanent SQL defects stop immediately. Failed HTTP cleanup still prevents any
retirement write. Hosted workers and transports share the configured database
retry policy and delay.

`NativePushOutbox.prepare` validates a copied callback before task row locks and
returns a frozen, secret-bearing `PreparedPushConfig`. Its one
`register_prepared` path binds that value to the recipient/tenant and the
server-admitted task ID without DNS inside the transaction. Ordinary config
creation delegates through the same preparation and registration path.

`TaskManager` binds one ready managed service to its exact canonical task store.
Inline callback validation finishes before either admission transaction. New
context ownership, task/input persistence and encrypted registration commit
before the cognition wake. A continuation registers only when its new input is
actually appended; an exact message replay cannot recreate a deleted callback.
A failed registration rolls back the input revision and task change together.
Both paths retain the original complete wire digest while cognition receives
only the immutable callback-free projection. Domain refusals remain typed
native conflicts through the enclosing transaction.

Shutdown joins the managed service before closing any primary store. A stopped
resident produces explicit service unavailability with its original cause,
rather than injecting another task's cancellation into the request. Journal
forwarding, claims and lease/ACK/retry transitions use the same owned database
handoff. Ordinary withdrawal cannot interrupt their original COMMIT ACK;
genuine lost acknowledgements retain their native control evidence. Deletion
recognizes its own config/lease revocation from the monitor while retaining
actual custody and storage failures.

The optional push schema bundle is revision 4. Configuration, delivery-attempt
and subscription DDL commit with its completion marker under the native schema
lease. A completed boot skips data-table DDL. The shared encryption key must be
available before any push schema write, including on a repeated boot.

`NativePushService` retains the original resident task admission before its
host publisher can return or fail. Startup waits for readiness while observing
the original owner's failure. Refused publication closes the fresh coroutine;
accepted publication failure joins that exact resident. Shutdown joins the
original worker through observer cancellation and physical HTTP close.

Hosted native publication starts this service after the agent's READY boundary,
using the original runtime task publisher and the exact initialized primary
signal ledger. The signed card advertises push only after the resident is ready.
A failed or stopped configured service refuses publication and push operations;
it does not silently downgrade the capability. Embeddings supply an explicit
`NativePushPolicy`; operator-hosted publication enables its default policy when
`KESTREL_BASE_URL` is set. `KESTREL_A2A_PUSH_ENABLED=false` explicitly disables
push. The `KESTREL_A2A_PUSH_` settings `RETENTION_DAYS`, `LEASE_SECONDS`,
`TIMEOUT_SECONDS`, `POLL_SECONDS`, `RENEWAL_SECONDS`, `RETRY_SECONDS`, `BATCH_SIZE`,
and `DATABASE_RETRY_ATTEMPTS` override the one policy's defaults. Values must be positive integers,
and renewal must precede lease expiration. Required encryption is checked before
push schema initialization.

Push database polling uses that finite attempt budget and `RETRY_SECONDS`
between attempts. Known availability failures delay another burst on the same
resident after its budget expires; a longer database outage does not replace or
permanently kill that owner. The question supervisor uses the same polling
contract. Initialization remains strict and never publishes a failed startup as
ready. PostgreSQL interface/client errors qualify only under the backend's
proof that the original session closed before COMMIT acceptance and its original
rollback owner finished; the enclosing pool reset must still finish. An accepted
COMMIT with a missing acknowledgement keeps its uncertain outcome.
The PostgreSQL driver floor is asyncpg 0.32.0, which also fixes pool release
after protocol aborts and mid-operation server-error preservation.

An ordinary fully retired worker failure stays on the original Task and still
refuses `require_ready`; shutdown logs it and may release primary storage.
Task inputs without callbacks depend on their own task store and authority,
so this optional service's ordinary historical failure does not block them.
Original execution control is checked again at task admission even without a
callback. Observer cancellation, original commit evidence and physical cleanup
debt still refuse storage release until their original owners settle.

Both official HTTP bindings expose standard Create/Get/List/Delete push
configuration operations under the same signed request and task-creator policy
as task access. List cursors bind the recipient, tenant, principal and task.
Delete revokes future dispatch and waits for original local HTTP owners; another
runtime's unretired attempt remains an explicit internal refusal with credentials
retained for retirement. Peer Stop returns a Message and refuses inline task
callbacks. No network request may hold a database transaction.

Callback DNS runs after signed-sender and recipient authorization, with the
host lifecycle lease released. New input, continuation input and explicit
configuration creation reacquire the original request's authority for their
finite commit. A withdrawn registration, changed principal or stopped push
service refuses that commit. The immutable input projection is captured before
DNS; callback credentials remain outside task-input storage.

Callback-scoped log filters redact the URL, normalized HTTP URL, notification
token and authentication credentials. Retained error messages and explicit or
implicit exception ancestry are redacted without replacing the original
exceptions, commit evidence, request objects or actual headers. Original pending
cleanup owners retain their redaction scope until they finish; unrelated
concurrent requests keep their own logging context.

`prepare_native_task_input` copies the complete official request and hashes its
canonical JSON before extracting inline callback credentials. Initial and
continuation inputs persist only the callback-free executable projection and
the original wire digest. Reusing a message ID with changed callback credentials
conflicts even when its executable content is unchanged. Snapshot readers
refuse callback material or a missing wire identity. The signing producer and
task-input projection share one pure digest implementation, avoiding an
agent/storage import cycle. Earlier input shapes require explicit offline
migration; runtime startup refuses to reinterpret them.

Initial and continuation callback projections register atomically with task
admission, encrypted configuration and the durable consumer. A registration
refusal rolls back the task or input transition and its wake. Exact continuation
replays do not register again. Callback credentials never enter cognition prompts
or plaintext task-input JSON. Signing a card does not authorize arbitrary
callback egress.

`NativeA2APublication` owns the common signed card, handler and SDK routes.
Its host-owned guard factory supplies request authority and the matching
lifecycle reader; the embedding host retains its writer during publication and
withdrawal. AgentManager and the standalone server's startup path use this
contract after real agent READY. Standalone publication requires an explicit
trusted DID resolver and a recipient-scoped peer directory/requester; a DID
signature alone never grants access. A single agent uses its canonical stable
DID as the routing key unless `KESTREL_A2A_DISCOVERY_AGENT` selects one.
`SingleAgentNativeAuthority` provides the original-runtime and scoped-policy
guard for that embedding. Its
writer closes admission, joins current readers, and withdraws the exact route
binding before the agent's task manager may close its primary storage. Replacing
an agent field cannot erase the captured runtime's original revocation.

Launcher-managed processes retain the explicit parent `KESTREL_HOST_URL` for
authenticated roster reads. When `KESTREL_BASE_URL` is configured, the launcher
sets each child's discovery key to its exact registration name. That public
front door must route each child's native card and A2A paths to that child;
neither its listen port nor the parent roster URL substitutes for the public
origin. The process session resolves the peer's signing DID from immutable
launcher-attested identity material and rechecks the live scoped roster before
discovery and operations. It uses the same SDK session setup as AgentManager,
verifies the signed card, and refuses an interface that changes the trusted
origin, route or tenant. The roster credential never enters native discovery
or task requests. Explicitly injected native session factories retain ownership.

Real standalone lifespan tests exercise inception, agent initialization, signed
discovery, push service startup and shutdown on isolated storage without a
metered model call. The runtime task publisher marks resident service ownership
for the restart idle gate. Finite background work keeps its separate admission
and still defers restart, even when it has the same name as a resident.
Process-session tests drive the actual startup factory,
SDK HTTP routes and durable creator/outbound rows over a loopback listener,
including a proxy prefix and revoked roster. These are implementation checks;
they do not establish deployment or foreign-framework acceptance.

## Review, verification and release order

1. Review this design in its worktree; complete dependency/import and three
   real-vendor OAuth spikes. Record capability advertisement separately from
   successful authorization, refresh and tool calls.
2. Complete the MCP feature transport change and publish its 0.5 release
   admitting `mcp>=2.3,<3` before releasing the host dependency update. Published
   feature 0.4.1's `mcp<2` cannot resolve with Phoenix 20's FastMCP dependency.
   Stage and test both changes together from worktrees, then release feature
   before host. Resolve the full host feature manifest before host release.
   Docker gateway 0.44.1 works via the official SDK's standard negotiation at
   `2025-11-25`. Its newer-protocol capability gap remains tracked in
   [#3683](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3683).
   Verify each negotiated revision and gate extensions; no retirement decision
   is required for ordinary tool access.
3. Add remote connections/OAuth/registry, then Apps and Tasks. Remote write
   scopes remain gated on [#596](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/596).
4. Complete the native-A2A callers and storage migration, then publish/test
   the coordinated Core dependency cutover with **SDK >=0.42.1**. Re-survey
   consumers and update their actual callers and constraints: the recorded
   survey found Core ceilings below 0.54 in Talon/Features and SDK ceilings
   below 0.40 in Telegram, Talon, Features and editable Claws/Eye/Flight.
   Verify clean installed SDK 0.42 contracts and the full host feature manifest;
   an isolated source install does not establish Core compatibility. Reconcile
   this branch with current main before its final full-diff review, preserving
   the architecture/artifact-safety prerequisites and the encrypted-store
   recovery code and documentation merged in
   [#3730](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/3730).
   Run both A2A bindings through the TCK and a different-framework bidirectional
   task/card-verification exercise. Computer-use runtime, artifact viewer/editor
   and provider acceptance remain tracked in
   [#3703](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3703)
   and [#3711](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3711).
5. Implement the server direction under [#1990](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1990),
   reusing MCP 2.x, per-agent routing, protected-resource metadata, credential
   governance and the normal memory/turn persistence path.

Each PR requires a completed full-diff worktree review before creation, with
findings fixed and meaningful regression tests. Pin the conformance runner
versions/commits and record which protocol scenarios they actually support.
Do not label a skipped 2026 scenario or a mocked interop peer a conformance pass.
Acceptance includes governed dispatch database evidence, Apps bridge denial
tests (including vendor CSP/permission escalation), tool-call task result
handling with a real capable SDK and rejection on unsupported methods,
`tasks/get` → `tasks/update`
input/restart, form elicitation and explicit URL denial, no implicit DCR,
OAuth restart/rotation, cross-agent token isolation, signature tampering,
replay, recipient denial, question forgery, cancellation races and lock-free
operator input, exact issuer comparison (including present unadvertised `iss`),
manual MRTR ownership/state expiry, model/app visibility isolation, MRTR
operation rebinding, unknown outcomes after stream loss, private cache
isolation, tenant mismatch, peer transport admission for both bindings/aliases,
public discovery without an operator key, no-common-revision failure and
REST path/body tenant disagreement for signed and standard-only principals,
protobuf-presence signature vectors against the normative oracle and an SDK
release that passes #3685. Final acceptance also searches for every retired wire surface.
