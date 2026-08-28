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

The authoritative parent-to-child relation is a verified, parent-signed spawn
mandate receipt bound to the final child DID. The manager's `spawned_by` graph
edge and `_parent_children` map are indexes/caches; neither is authority without
receipt verification. Work to restore that evidence after restart is tracked in
[#3133](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3133),
and the durable descendant query is
[#3142](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3142).

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
| External webhook ingress | `POST /webhooks/{webhook_name}`; Rasa `POST /webhooks/rest/webhook` | The request-bound agent or uniquely configured receiver | Universal policy (bounded ingress) | The route binds the target from trusted request state/receiver registration and the configured receiver authenticates and rate-limits the payload. Rasa uses its sovereign-configured shared secret and the host-bound agent; payload sender fields create no agent authority. |
| Read outbound peer result/audit | `get_peer_task_result`, `list_outbound_a2a_tasks` | A task created by the caller | Self (creator) | Outbound records retain creator/recipient binding. Shared-store reads and HTTP/SSE still need durable principal predicates: [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| Read task inbox/status/result | `check_task_status`, `list_my_tasks`, `get_task_result`; task GET/list/SSE endpoints | Recipient inbox or creator-owned result | Self (recipient or creator, according to operation) | Current task-ID/full-table reads are not consistently principal-scoped on shared PostgreSQL. Defect: [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| Respond/fail/complete or attach artifact | `respond_to_a2a_task`, `attach_artifact_to_a2a_task` | An incoming A2A task | Self (recipient) | Current mutations use task ID without an atomic recipient predicate. Defect: [#3144](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3144). |
| Cancel A2A task | `cancel_task` | A non-terminal task | Self (creator or recipient) | Durable creator/recipient authorization and an atomic cancellation predicate are implemented by [#3134](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3134). Causation/sender display metadata is not consulted. |
| Create child | `spawn_agent` | A new child | Spawn mandate | The parent constructs/signs the mandate; the host binds and re-signs it to the final child DID. A created child does not grant the child reciprocal authority. |
| List/read child work | `list_children`, `get_child_result`; `GET /api/spawn/children` | The caller's children | Spawn mandate or self-owned feature state | Child enumeration must derive from verified receipts ([#3133](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3133), [#3142](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3142)); results are held by the calling feature instance. |
| Delegate work to child | `delegate_task` | A direct child | Spawn mandate | The caller DID must be the verified mandate parent at dispatch time. `_parent_children` is only a lookup cache. |
| Terminate/offboard child | `terminate_child` | A direct child/descendant runtime tree | Spawn mandate | Signed parent authority is required in addition to lifecycle/custody/refund gates. `ALWAYS_ASK` tightens consent but is not the authority proof. |
| Cooperative Stop | Stop authority service and future peer signal rail; current `POST /api/agent/stop` is local only | Turn, agent, subtree, host, or fleet | Universal policy for Stop; spawn mandate/sovereign for Hold | Typed Stop work is tracked by [#3139](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3139) and [#3141](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3141). Peer Stop must inherit signal cycle detection and load-bearing rate limits; repeated Stop must not become Hold through the back door. |
| File/execute whole-host restart or update | `request_restart`, `restart_coordinator` | Every co-hosted agent and possibly their code checkout | Sovereign/delegated | The current feature admits an ordinary agent tool call; generic ASK can be auto-promoted. Defect: [#3148](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3148). |
| Read/cancel/ack restart request | `list_restart_requests`, `list_restart_status_events`, `cancel_restart_request`, `acknowledge_restart_escalation`; restart status endpoint | A durable restart request/event | Self for requester detail/mutation; explicitly public host-coordination fields may be universal read-only | Cancel/ack currently address by ID without `requested_by_agent`; list visibility also needs an explicit classification. Defect: [#3146](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3146). |
| Scheduler watcher wake | `github_pr_watch`/`ecosystem_discovery_watch` arguments executed through schedules | Owning agent only | Self | Caller-provided `notify` can currently stamp a peer DID while the owning dispatcher wakes itself, corrupting causation. Defect: [#3147](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3147). |
| Host agent create/withdraw/offboard | `POST /api/agents`, `DELETE /api/agents/{agent_name}` | Host registry, peer runtime, hosted namespace | Sovereign/delegated | Global authentication also admits ordinary OAuth/JWT callers; the handlers do not require sovereign context. Defect: [#3149](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3149). |
| Talon coding/repository orchestration | External `kestrel-feature-talon`/`kestrel-talon` process | Repository work, issues, PRs | Outside agent hierarchy | Talon is an operator-enabled external feature/process. Its coordinator state, reviewer state, and worktree lineage are not Kestrel agent authority or causation relations. |

`OrchestrationStore` is currently a backend library with no core agent-facing
consumer. Its unscoped `task_id` mutation methods are therefore not a live
cross-agent door. Any future exposure must add durable principals and enter the
machine-checked inventory below before shipping.

## Machine-checked tool inventory

The contract test discovers every `@tool` whose public name contains
`agent`, `peer`, `a2a`, `child`, `restart`, or `task`. Every match must remain
classified here, including false positives, so a newly named cross-agent door
cannot silently appear.

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
| `kestrel_sovereign/features/spawn/feature.py::delegate_task` | Verified parent-to-child mandate. |
| `kestrel_sovereign/features/spawn/feature.py::get_child_result` | Self-owned result state, addressed within verified child set. |
| `kestrel_sovereign/features/spawn/feature.py::list_children` | Verified parent-to-child mandate query. |
| `kestrel_sovereign/features/spawn/feature.py::spawn_agent` | Creates a signed parent-to-child mandate. |
| `kestrel_sovereign/features/spawn/feature.py::terminate_child` | Verified parent-to-child mandate plus lifecycle gates. |
| `kestrel_sovereign/features/tasks/feature.py::attach_artifact_to_a2a_task` | Recipient-owned mutation; #3144. |
| `kestrel_sovereign/features/tasks/feature.py::cancel_task` | Creator/recipient-owned mutation; #3134. |
| `kestrel_sovereign/features/tasks/feature.py::check_task_status` | Principal-scoped read; #3145. |
| `kestrel_sovereign/features/tasks/feature.py::get_task_result` | Principal-scoped read; #3145. |
| `kestrel_sovereign/features/tasks/feature.py::list_my_tasks` | Recipient inbox read; #3145. |
| `kestrel_sovereign/features/tasks/feature.py::respond_to_a2a_task` | Recipient-owned mutation; #3144. |
| `kestrel_sovereign/features/todo/feature.py::todo_link_task` | Self-owned todo metadata link; not an A2A task control. |

## Machine-checked HTTP inventory

| Surface ID | Classification |
|---|---|
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/invoke` | Universal peer communication through the scoped directory; creates causation, not authority. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agent/tasks` | Recipient inbox read; #3145. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agent/tasks/{task_id}` | Principal-scoped read; #3145. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agent/tasks/{task_id}/subscribe` | Principal-scoped subscription; #3145. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/stop` | Current routed-agent/self Stop; peer Stop must use the typed authority rail. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/tasks/send` | Scoped, authenticated A2A delivery. |
| `kestrel_sovereign/endpoints/models.py::DELETE /api/agents/{agent_name}` | Sovereign/delegated host lifecycle; #3149. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents` | Authenticated read-only host discovery. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents` | Sovereign/delegated host lifecycle; #3149. |
| `kestrel_sovereign/endpoints/restart_events.py::GET /api/restart/status-events` | Requester/explicit host-coordination read; #3146. |
| `kestrel_sovereign/endpoints/rasa_shim.py::POST /webhooks/rest/webhook` | Sovereign-configured, authenticated ingress to the host-bound agent; payload sender is not authority. |
| `kestrel_sovereign/endpoints/spawn.py::GET /api/spawn/children` | Read-only child status projected from verified relationships. |
| `kestrel_sovereign/features/webhooks/receiver.py::POST /webhooks/{webhook_name}` | Bounded ingress to the request-scoped or uniquely configured receiver; webhook auth does not create agent hierarchy. |

## Review rule

Adding a cross-agent surface requires updating this matrix in the same change.
The implementation must bind the trusted caller identity before the operation,
authorize and mutate atomically where storage is shared, return the same public
shape for missing and unauthorized targets, and include a mutation test that
fails if the authority predicate or its wiring is removed.
