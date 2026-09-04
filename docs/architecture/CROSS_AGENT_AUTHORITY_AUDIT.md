---
type: Architecture Spec
title: Cross-Agent Authority Audit
description: Classification and enforcement inventory for every core surface that can observe, address, mutate, or control another agent.
resource: /docs/architecture/CROSS_AGENT_AUTHORITY_AUDIT.md
tags:
- docs
- architecture
- architecture-spec
- multi-agent
- security
timestamp: '2026-08-28T00:00:00Z'
status: active
owner: architecture
canonical: true
generated: false
privacy: public
---

# Cross-Agent Authority Audit

## Result

Core has two relationship axes: causation and authority. Causation explains why
work happened. Authority is permission to control another agent. A causation
frame, trace parent, scheduler source, display label, peer status, co-hosting,
or guessed identifier never grants authority.

The design requires a verified, parent-signed spawn mandate receipt bound to the
final child DID. The live implementation has not reached that model: `_do_spawn`
signs while `child_did` is unset, mutates that signed field after creation, and
does not re-sign or verify at the later control boundaries. Its `spawned_by`
graph edge, `_parent_children` map, and `_child_mandates` map currently act as
unverified authority inputs. Restart rehydration is tracked in
[#3133](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3133);
final-DID signature verification and authoritative descendant lookup are tracked
in [#3142](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3142).

A universally available capability is the absence of a required relation, not
a third relationship axis. Peer communication and bounded cooperative Stop may
be universal only when the policy says so explicitly. Generic tool ASK/AUTO
state is operational consent, not constitutional authority. There is no
`admin agent`: host administration belongs to the sovereign key, or to a
narrow, revocable, signed delegation.

## Authority classes

| Class | Meaning | Required evidence |
|---|---|---|
| Self | An agent reads or mutates only its own runtime or durable namespace. | The trusted runtime binds the caller DID; caller-supplied identity is not accepted. |
| Universal policy | No relationship is required because the capability is explicitly universal and bounded. | The named policy plus its limits, receipts, rate limits, and target authentication. |
| Spawn mandate | A parent controls a child within the signed mandate. | A valid parent signature over the final child DID and constraints, verified at the mutation boundary. |
| Sovereign/delegated | Host or fleet administration. | Sovereign-key caller context, or a narrow signed delegation whose scope includes the exact operation and target. |

## Audit matrix

| Surface/action | Entry doors | Target | Required class | Enforcement seam and result |
|---|---|---|---|---|
| Discover peers | `list_peers`; `GET /api/agents` | Agents in the requester's automatic directory | Universal policy (read-only) | `PeerDirectoryRouter.list_peers` scopes feature discovery. Host discovery is authenticated but intentionally contains public agent cards; mutation authority does not follow from visibility. |
| Synchronous peer message | `ask_agent` | One directory-resolved peer | Universal policy (communication) | The router resolves in trusted `PeerRequester.authorization_scope` and must reauthorize `invoke`. The request creates causation, not control authority. |
| Asynchronous peer message/question/task | `send_a2a_message`, `send_a2a_question`, `send_a2a_task`; `POST /api/agent/tasks/send` | One directory-resolved recipient | Universal policy (communication) | Outbound routing reauthorizes the stable peer identity; inbound hosted delivery requires a verified sender/scoped authorizer. Signed-envelope verification authenticates the sender but does not create hierarchy. |
| Invoke a routed agent | `POST /api/agent/invoke`; `POST /api/agent/stream`; `POST /api/bridge/invoke`; `POST /api/bridge/stream`; `POST /v1/chat/completions` | The agent pinned by trusted request routing | Outside agent hierarchy; host-authenticated external ingress | The host authenticates the API-key/JWT/session caller and `get_agent(request)` consumes the middleware-pinned target. Bridge sender/session fields and invocation provenance describe the request; they confer no peer or hierarchy authority. |
| General webhook ingress | `POST /webhooks/{webhook_name}` | The request-bound agent when agent-prefixed; otherwise the first enabled receiver matching the name | Outside agent hierarchy; explicitly configured external ingress | Agent-prefixed routing binds the target from trusted request state. The unprefixed multi-agent route aggregates receivers and does not reject duplicate names, so iteration order can choose the target. Defect: [#3216](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3216). The supported `auth_type="none"`, `rate_limit=0` combination accepts every reachable request without throttling; `allow_unauthenticated` acknowledges that choice but adds no gate. A bounded operator policy therefore requires a real auth mode and a positive rate limit. Payload fields create no agent authority. |
| Rasa webhook ingress | `POST /webhooks/rest/webhook` | The host-bound agent | Outside agent hierarchy; sovereign-configured external ingress | Rasa requires its sovereign-configured shared secret, applies a fixed request rate, and invokes the host-bound agent. The payload sender becomes session/provenance data only and creates no agent authority. |
| Bootstrap host API credential | `GET /api/auth/key` | The host-wide API authentication boundary | Public-localhost provisioning exception | The endpoint is disabled unless bootstrap policy enables it, restricts callers to loopback/Docker gateway/explicit allowed hosts, and is rate-limited. It returns the host API key, which authenticates broad API access but is not the sovereign signing key and cannot satisfy the stronger #3149 lifecycle gate. This classification is reconciled with `docs/audit/AUTH_SURFACE_MATRIX.md`. |
| Read host UI state / issue browser tokens | `GET /api/host/ui/contributions`; `GET /api/host/csrf`; `POST /api/host/phoenix/session` | Shared host UI manifest, CSRF token, or Phoenix embed session | Host-authenticated external operation | Central API-key/JWT/session middleware protects these app-level routes. The CSRF token is a double-submit value rather than standalone authority; the Phoenix route mints a short-lived path-scoped cookie only after host authentication and backend reachability. |
| Read observability summaries/metrics | `GET /api/observability/summary`; `GET /api/observability/metrics/{metric_name}` | The routed agent's events only | Self | A per-agent SQLite store happens to isolate the data, but shared PostgreSQL queries omit the trusted agent predicate and the metrics route accepts an arbitrary optional `agent_name`. Defect: [#3215](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3215). |
| Inspect/configure routed feature state | Feature catalog/detail/config/skills routes; feature enable/disable/config mutation | The request-routed agent | Self or sovereign/delegated operator policy | `get_agent(request)` binds the target runtime. These namespace matches are classified explicitly so they cannot conceal a future cross-agent implementation; they currently do not grant one agent authority over another. |
| Install/remove feature package | `POST /api/features/{name}/install`; `POST /api/features/{name}/remove` | Shared host interpreter and all loaded users of the package | Sovereign/delegated | The handlers currently require only an authenticated routed agent despite their sovereign-only docstrings. Defect: [#3214](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3214). |
| Read outbound peer result/audit | `get_peer_task_result`, `list_outbound_a2a_tasks` | A task created by the caller | Self (creator) | Outbound records retain creator/recipient binding. Shared-store reads and HTTP/SSE still need durable principal predicates: [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| Read task inbox/status/result | `check_task_status`, `list_my_tasks`, `get_task_result`, built-in `!tasks`; task GET/list/SSE endpoints | Today, any row found by an unscoped ID/full-table query; intended recipient inbox or creator-owned result | Self (recipient or creator, according to operation) | The current tool, command, HTTP, and SSE reads omit a durable recipient/creator predicate on shared PostgreSQL, so the required Self class is not enforced. Defect: [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| Respond/fail/complete or attach artifact | `respond_to_a2a_task`, `attach_artifact_to_a2a_task` | An incoming A2A task | Self (recipient) | Enforced by [#3144](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3144): each mutation binds the trusted caller DID and includes the recipient in the durable atomic predicate. |
| Cancel A2A task | `cancel_task`; `POST /api/agent/tasks/{task_id:path}/cancel` | A non-terminal task | Self (creator or recipient) | Enforced by [#3134](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3134): durable creator/recipient authorization and an atomic cancellation predicate are shared by the tool and signed peer route. Causation/sender display metadata is not consulted. |
| Create child | `spawn_agent` | A new child | Spawn mandate | The live path signs before the final child DID is known and then mutates `child_did`, invalidating the signature. Defect: [#3142](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3142). A created child does not grant reciprocal authority. |
| List/read child work | `list_children`, `get_child_result`; `GET /api/spawn/children` | The caller's children | Spawn mandate or self-owned feature state | Results are held by the calling feature instance, but child enumeration currently trusts process-local graph/cache projections without verifying a final-DID receipt. Defects: [#3133](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3133), [#3142](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3142). |
| Delegate work to child | `delegate_task` | A direct child | Spawn mandate | The live boundary checks only `manager.get_children()` / `_parent_children`; `verify_mandate()` has no production caller. Defect: [#3142](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3142). |
| Terminate/offboard child | `terminate_child` | A direct child/descendant runtime tree | Spawn mandate | The live boundary checks the same unverified cache relation. Signed authority must be added alongside lifecycle/custody/refund gates; `ALWAYS_ASK` is consent, not proof. Defect: [#3142](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3142). |
| Cooperative Stop | Stop authority service and future peer signal rail; current `POST /api/agent/stop` is local only | Turn, agent, subtree, host, or fleet | Universal policy for Stop; spawn mandate/sovereign for Hold | Typed Stop work is tracked by [#3139](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3139) and [#3141](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3141). Peer Stop must inherit signal cycle detection and load-bearing rate limits; repeated Stop must not become Hold through the back door. |
| File/execute whole-host restart or update | `request_restart`, `restart_coordinator` | Every co-hosted agent and possibly their code checkout | Sovereign/delegated | The current feature admits an ordinary agent tool call; generic ASK can be auto-promoted. Defect: [#3148](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3148). |
| Read/cancel/ack restart request | `list_restart_requests`, `list_restart_status_events`, `cancel_restart_request`, `acknowledge_restart_escalation`; restart status endpoint | A durable restart request/event | Self for requester detail/mutation; explicitly public host-coordination fields may be universal read-only | Enforced by [#3146](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3146): list/event reads bind the requesting agent, and cancel/ack mutations include `requested_by_agent` in the durable predicate. |
| Scheduler watcher wake | `github_pr_watch`/`ecosystem_discovery_watch` arguments executed through schedules | Owning agent only | Self | Enforced by [#3147](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3147): the runtime binds the scheduler owner's DID and ignores legacy caller-provided `notify` as a routing or causation identity. |
| Host agent create/withdraw/offboard | `POST /api/agents`, `DELETE /api/agents/{agent_name}`, `!create-agent` | Host registry, peer runtime, hosted namespace, trusted identity directory | Sovereign/delegated | Enforced by [#3149](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3149): every host lifecycle/provisioning door requires a live sovereign caller context at the handler boundary; ordinary OAuth/JWT authentication is insufficient. |
| Talon coding/repository orchestration | External `kestrel-feature-talon`/`kestrel-talon` process | Repository work, issues, PRs | Outside agent hierarchy | Talon is an operator-enabled external feature/process. Its coordinator state, reviewer state, and worktree lineage are not Kestrel agent authority or causation relations. |

`OrchestrationStore` is currently a backend library with no core agent-facing
consumer. Its unscoped `task_id` mutation methods are therefore not a live
cross-agent door. Any future exposure must add durable principals and enter the
machine-checked inventory below before shipping.

## Machine-checked tool inventory

The contract test discovers every `@tool` whose public name names an agent
relation (`agent`, `peer`, `a2a`, `child`, `descendant`, or `delegate`), a work
object (`task`), a control verb (`cancel`, `interrupt`, `stop`, `hold`,
`terminate`, `offboard`, or `withdraw`), or host/fleet/restart scope. Every
match must remain classified here, including false positives, so a newly named
cross-agent door cannot silently appear merely because it omits `agent`.

| Surface ID | Classification |
|---|---|
| `kestrel_sovereign/features/bootstrap/feature.py::rename_agent` | Self-only display-name mutation; not a peer door. |
| `kestrel_sovereign/features/bootstrap/feature.py::restart_discovery` | Self-only bootstrap-state retry; not a host restart. |
| `kestrel_sovereign/features/deploy/feature.py::deploy_agent` | External deployment profile control, not a co-hosted-agent relation; separately ASK-gated. |
| `kestrel_sovereign/features/peers/feature.py::ask_agent` | Universal peer communication through the scoped directory. |
| `kestrel_sovereign/features/peers/feature.py::get_peer_task_result` | Creator-owned routed read; #3145. |
| `kestrel_sovereign/features/peers/feature.py::list_outbound_a2a_tasks` | Self-owned outbound audit. |
| `kestrel_sovereign/features/peers/feature.py::list_peers` | Universal scoped discovery. |
| `kestrel_sovereign/features/peers/feature.py::send_a2a_message` | Universal bounded peer communication. |
| `kestrel_sovereign/features/peers/feature.py::send_a2a_question` | Universal bounded peer communication. |
| `kestrel_sovereign/features/peers/feature.py::send_a2a_task` | Universal bounded peer communication. |
| `kestrel_sovereign/features/restart_coordinator/feature.py::acknowledge_restart_escalation` | Requester-owned mutation; #3146. |
| `kestrel_sovereign/features/restart_coordinator/feature.py::cancel_restart_request` | Requester-owned mutation; #3146. |
| `kestrel_sovereign/features/restart_coordinator/feature.py::list_restart_requests` | Requester/explicit host-coordination read; #3146. |
| `kestrel_sovereign/features/restart_coordinator/feature.py::list_restart_status_events` | Requester/explicit host-coordination read; #3146. |
| `kestrel_sovereign/features/restart_coordinator/feature.py::request_restart` | Sovereign or narrow signed delegation; #3148. |
| `kestrel_sovereign/features/restart_coordinator/feature.py::restart_coordinator` | Sovereign executor/registered cron action; #3148. |
| `kestrel_sovereign/features/spawn/feature.py::delegate_task` | Unverified process-local child map; defect #3142. |
| `kestrel_sovereign/features/spawn/feature.py::get_child_result` | Self-owned result state keyed by the caller's prior delegated task. |
| `kestrel_sovereign/features/spawn/feature.py::list_children` | Unverified process-local child map; defects #3133/#3142. |
| `kestrel_sovereign/features/spawn/feature.py::spawn_agent` | Signature is invalidated when the final child DID is assigned; defect #3142. |
| `kestrel_sovereign/features/spawn/feature.py::terminate_child` | Unverified process-local child map plus lifecycle gates; defect #3142. |
| `kestrel_sovereign/features/tasks/feature.py::attach_artifact_to_a2a_task` | Recipient-owned mutation; #3144. |
| `kestrel_sovereign/features/tasks/feature.py::cancel_task` | Creator/recipient-owned mutation; #3134. |
| `kestrel_sovereign/features/tasks/feature.py::check_task_status` | Unscoped task-ID read; defect [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| `kestrel_sovereign/features/tasks/feature.py::get_task_result` | Unscoped task-ID read; defect [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| `kestrel_sovereign/features/tasks/feature.py::list_my_tasks` | Unscoped shared-store inbox read; defect [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| `kestrel_sovereign/features/tasks/feature.py::respond_to_a2a_task` | Recipient-owned mutation; #3144. |
| `kestrel_sovereign/features/todo/feature.py::todo_link_task` | Self-owned todo metadata link; not an A2A task control. |

## Machine-checked built-in command inventory

Built-in commands do not carry feature `@tool` decorators. The contract test
therefore discovers command names containing the same cross-agent terms
directly from `BUILTIN_COMMAND_SPECS`, so a command-handler mutation cannot add
a host-control door behind the feature-tool inventory.

| Surface ID | Classification |
|---|---|
| `kestrel_sovereign/command_handler.py::!create-agent` | Sovereign/delegated host identity provisioning; #3149. |
| `kestrel_sovereign/command_handler.py::!tasks` | Unscoped shared-store task listing; defect [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |

## Machine-checked HTTP inventory

| Surface ID | Classification |
|---|---|
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/invoke` | Host-authenticated external invocation of the request-routed agent; provenance is not hierarchy authority. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/stream` | Host-authenticated external invocation of the request-routed agent; provenance is not hierarchy authority. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agent/tasks` | Unscoped shared-store inbox read; defect [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agent/tasks/{task_id}` | Unscoped task-ID read; defect [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agent/tasks/{task_id}/subscribe` | Unscoped task-ID subscription; defect [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/tasks/{task_id:path}/cancel` | Creator/recipient-owned mutation; the signed peer envelope authenticates the actor, live recipient scope is rechecked, and the durable transition uses the atomic #3134 authority predicate. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/stop` | Current routed-agent/self Stop; peer Stop must use the typed authority rail. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/tasks/send` | Scoped, authenticated A2A delivery. |
| `kestrel_sovereign/endpoints/features.py::GET /api/features` | Request-routed agent feature catalog; not a cross-agent grant. |
| `kestrel_sovereign/endpoints/features.py::GET /api/features/installed` | Request-routed agent feature inventory; not a cross-agent grant. |
| `kestrel_sovereign/endpoints/features.py::GET /api/features/{name}` | Request-routed agent feature detail; not a cross-agent grant. |
| `kestrel_sovereign/endpoints/features.py::POST /api/features/{name}/install` | Shared-host package mutation lacks sovereign/delegated enforcement; defect #3214. |
| `kestrel_sovereign/endpoints/features.py::POST /api/features/{name}/enable` | Request-routed agent runtime mutation; not a cross-agent grant. |
| `kestrel_sovereign/endpoints/features.py::POST /api/features/{name}/disable` | Request-routed agent runtime mutation; not a cross-agent grant. |
| `kestrel_sovereign/endpoints/features.py::POST /api/features/{name}/remove` | Shared-host package mutation lacks sovereign/delegated enforcement; defect #3214. |
| `kestrel_sovereign/endpoints/features.py::GET /api/features/{name}/config` | Request-routed agent config read; secrets remain write-only. |
| `kestrel_sovereign/endpoints/features.py::PATCH /api/features/{name}/config` | Request-routed agent config mutation; not a cross-agent grant. |
| `kestrel_sovereign/endpoints/features.py::GET /api/features/{name}/skills` | Request-routed agent skill discovery; not a cross-agent grant. |
| `kestrel_sovereign/endpoints/models.py::POST /v1/chat/completions` | Host-authenticated external invocation of the request-routed agent; message/user fields are not hierarchy authority. |
| `kestrel_sovereign/endpoints/models.py::DELETE /api/agents/{agent_name}` | Sovereign/delegated host lifecycle; #3149. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents` | Authenticated read-only host discovery. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents` | Sovereign/delegated host lifecycle; #3149. |
| `kestrel_sovereign/endpoints/observability.py::GET /api/observability/summary` | Shared PostgreSQL read lacks a trusted agent predicate; defect #3215. |
| `kestrel_sovereign/endpoints/observability.py::GET /api/observability/metrics/{metric_name}` | Shared PostgreSQL read accepts an untrusted optional agent filter; defect #3215. |
| `kestrel_sovereign/endpoints/restart_events.py::GET /api/restart/status-events` | Requester/explicit host-coordination read; #3146. |
| `kestrel_sovereign/endpoints/rasa_shim.py::POST /webhooks/rest/webhook` | Sovereign-configured, authenticated ingress to the host-bound agent; payload sender is not authority. |
| `kestrel_sovereign/endpoints/spawn.py::GET /api/spawn/children` | Read-only child status projected from unverified process-local relationships; defects #3133/#3142. |
| `kestrel_sovereign/features/bridge/router.py::POST /api/bridge/invoke` | Host-authenticated external invocation of the request-routed agent; gateway metadata is not authority. |
| `kestrel_sovereign/features/bridge/router.py::POST /api/bridge/stream` | Host-authenticated external streaming invocation of the request-routed agent; gateway metadata is not authority. |
| `kestrel_sovereign/features/bridge/router.py::GET /api/bridge/capabilities` | Host-authenticated capability read from the request-routed agent; no cross-agent grant. |
| `kestrel_sovereign/features/bridge/router.py::GET /api/bridge/health` | Host-authenticated bridge status read for the request-routed agent; no cross-agent grant. |
| `kestrel_sovereign/features/bridge/router.py::POST /api/bridge/session` | Host-authenticated session creation within the request-routed agent's Bridge feature. |
| `kestrel_sovereign/features/web_search/feature.py::GET /api/features/web_search/test` | Request-routed agent connectivity test; not a cross-agent grant. |
| `kestrel_sovereign/features/webhooks/receiver.py::POST /webhooks/{webhook_name}` | Agent-prefixed requests are scoped; the unprefixed route has ambiguous duplicate ownership (defect [#3216](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3216)). `auth_type="none"` plus `rate_limit=0` is intentionally open and unlimited; `allow_unauthenticated` only acknowledges it. No mode creates agent hierarchy. |
| `kestrel_sovereign/server.py::GET /api/auth/key` | Public-localhost, rate-limited host API-key bootstrap; reconciled with the checked-in auth-surface ledger. |
| `kestrel_sovereign/server.py::GET /api/host/ui/contributions` | Host-authenticated read of the shared UI manifest. |
| `kestrel_sovereign/server.py::GET /api/host/csrf` | Host-authenticated issuance of a double-submit CSRF token; the token is not standalone authority. |
| `kestrel_sovereign/server.py::POST /api/host/phoenix/session` | Host-authenticated minting of a short-lived, path-scoped Phoenix embed cookie. |

## Review rule

Adding a cross-agent surface requires updating this matrix in the same change.
The implementation must bind the trusted caller identity before the operation,
authorize and mutate atomically where storage is shared, return the same public
shape for missing and unauthorized targets, and include a mutation test that
fails if the authority predicate or its wiring is removed.
