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
| Operate on routed-agent state | Remaining `/api/agent/*` status, notification, health, heartbeat, privacy, attachment, and channel-link routes | The agent pinned by trusted request routing | Self or host-authenticated external operation | On a multi-agent host every singular route is addressable through `/api/agents/{name}/...`; the trusted routing middleware pins the runtime before the handler runs. The complete namespace is machine-inventoried below so a future cross-agent operation cannot hide behind an innocuous suffix. |
| General webhook ingress | `POST /webhooks/{webhook_name}` | The request-bound agent when agent-prefixed; otherwise the first enabled receiver matching the name | Outside agent hierarchy; explicitly configured external ingress | Agent-prefixed routing binds the target from trusted request state. The unprefixed multi-agent route aggregates receivers and does not reject duplicate names, so iteration order can choose the target. Defect: [#3216](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3216). The supported `auth_type="none"`, `rate_limit=0` combination accepts every reachable request without throttling; `allow_unauthenticated` acknowledges that choice but adds no gate. A bounded operator policy therefore requires a real auth mode and a positive rate limit. Payload fields create no agent authority. |
| Rasa webhook ingress | `POST /webhooks/rest/webhook` | The host-bound agent | Outside agent hierarchy; sovereign-configured external ingress | Rasa requires its sovereign-configured shared secret, applies a fixed request rate, and invokes the host-bound agent. The payload sender becomes session/provenance data only and creates no agent authority. |
| Bootstrap host API credential | `GET /api/auth/key` | The host-wide API authentication boundary | Public-localhost provisioning exception | The endpoint is disabled unless bootstrap policy enables it, restricts callers to loopback/Docker gateway/explicit allowed hosts, and is rate-limited. It returns the host API key, which authenticates broad API access but is not the sovereign signing key and cannot satisfy the stronger #3149 lifecycle gate. This classification is reconciled with `docs/audit/AUTH_SURFACE_MATRIX.md`. |
| Read host UI state / issue browser tokens | `GET /api/host/ui/contributions`; `GET /api/host/csrf`; `POST /api/host/phoenix/session` | Shared host UI manifest, CSRF token, or Phoenix embed session | Host-authenticated external operation | Central API-key/JWT/session middleware protects these app-level routes. The CSRF token is a double-submit value rather than standalone authority; the Phoenix route mints a short-lived path-scoped cookie only after host authentication and backend reachability. |
| Read host/fleet health or process metrics | `GET /health`; `GET /health/detailed`; `GET /metrics` | Public aggregate readiness, authenticated per-agent fleet diagnostics, or public process-wide Prometheus telemetry | Outside agent hierarchy; deployment/operator observation policy | `/health` intentionally exposes only aggregate readiness and `/metrics` is an explicitly public scraper surface. Global authentication protects `/health/detailed`, whose multi-agent response names agents and their checks. None of these observations creates control authority. |
| Use the Phoenix trace proxy | All registered methods on `/phoenix` and `/phoenix/{path:path}` | The shared host Phoenix trace store | Outside agent hierarchy; host-authenticated external operation | Global authentication accepts a host session/API key or the short-lived path-scoped embed cookie. The route proxies fleet-scoped trace data but does not grant one agent authority over another. Symbolic `api_route(methods=...)` declarations are expanded by the contract scanner. |
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
| Core operator CLI | Every command in the canonical `kestrel` dispatch table, including `ask`, `create`, `start`, `terminate`, `restart`, and `update` | A named agent, the local fleet/host, agent data, or external infrastructure according to the command | Outside agent hierarchy; local sovereign/operator process authority | CLI process access is an operator boundary, not agent causation or peer authority. The complete core dispatch table is classified below rather than filtered by names; otherwise authority-bearing verbs such as `ask`, `create`, and `update` evade discovery. Remote API calls still pass the destination host's authentication checks. |
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
The scanner also follows the generic `execute_skill` and `_create_schedule`
dispatch calls, because a meta-tool can reach an authority-bearing target even
when its own public name contains none of those words.

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
| `kestrel_sovereign/features/scheduler/feature.py::schedule_add` | Self-owned indirect dispatcher: it persists only a registered tool name, and the scheduled execution re-enters the downstream tool's runtime permission gate; scheduling conveys no relation authority. |
| `kestrel_sovereign/features/scheduler/feature.py::schedule_add_deadline` | Self-owned one-shot indirect dispatcher with the same registered-name and downstream runtime permission gates; scheduling conveys no relation authority. |
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
| `kestrel_sovereign/features/tasks/feature.py::run_workflow` | Self-owned indirect dispatcher through `TaskManager.execute_skill`; each selected feature tool retains its PRE_TOOL_USE and target-specific authority checks, so the workflow supplies sequence, not relation authority. |
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

## Machine-checked core CLI inventory

The contract test reads every key from the canonical core CLI dispatch table.
It deliberately classifies the whole table, including commands unrelated to
cross-agent control: selecting only agent-shaped names would miss verbs such as
`ask`, `create`, and `update`. Feature-contributed entry-point groups remain
outside core and must define their own operator policy.

| Surface ID | Classification |
|---|---|
| `kestrel_sovereign/cli.py::kestrel agent` | Local operator control of Docker-isolated agent lifecycle/chat; outside agent hierarchy. |
| `kestrel_sovereign/cli.py::kestrel ask` | Local operator invocation of a named running agent; the remote host still authenticates the request. |
| `kestrel_sovereign/cli.py::kestrel auth` | Local operator provider-authentication setup; no cross-agent grant. |
| `kestrel_sovereign/cli.py::kestrel config` | Local operator host configuration read/mutation; outside agent hierarchy. |
| `kestrel_sovereign/cli.py::kestrel constitution` | Local operator constitutional verification/maintenance; no peer authority. |
| `kestrel_sovereign/cli.py::kestrel create` | Local sovereign/operator provisioning of a new agent identity and registration. |
| `kestrel_sovereign/cli.py::kestrel demo` | Local operator demo-agent lifecycle; outside production agent hierarchy. |
| `kestrel_sovereign/cli.py::kestrel deploy` | Local operator control of external deployment infrastructure. |
| `kestrel_sovereign/cli.py::kestrel docker` | Local operator control of Docker images/runtimes. |
| `kestrel_sovereign/cli.py::kestrel doctor` | Local operator host/fleet diagnostic read. |
| `kestrel_sovereign/cli.py::kestrel embeddings` | Local operator audit/reindex of selected agent storage. |
| `kestrel_sovereign/cli.py::kestrel feature` | Local operator feature package/runtime management; not agent hierarchy. |
| `kestrel_sovereign/cli.py::kestrel health` | Deprecated local operator diagnostic alias for `doctor`. |
| `kestrel_sovereign/cli.py::kestrel identity` | Local operator identity-export custody maintenance. |
| `kestrel_sovereign/cli.py::kestrel ipfs` | Local operator control of external storage infrastructure. |
| `kestrel_sovereign/cli.py::kestrel list` | Local operator read of the configured agent fleet. |
| `kestrel_sovereign/cli.py::kestrel logs` | Local operator read of a named agent or host log. |
| `kestrel_sovereign/cli.py::kestrel migrate-config` | Local operator host configuration migration. |
| `kestrel_sovereign/cli.py::kestrel migrate-encryption` | Local operator mutation of selected agent storage. |
| `kestrel_sovereign/cli.py::kestrel migrate-llm-config` | Local operator host model-configuration migration. |
| `kestrel_sovereign/cli.py::kestrel release` | Local operator release-evidence/repository operation; no agent grant. |
| `kestrel_sovereign/cli.py::kestrel restart` | Local operator restart of a named agent or the host/fleet. |
| `kestrel_sovereign/cli.py::kestrel runpod` | Local operator control of external RunPod infrastructure. |
| `kestrel_sovereign/cli.py::kestrel serve` | Local operator control of the shared local model server. |
| `kestrel_sovereign/cli.py::kestrel setup` | Local operator host/agent bootstrap and recovery. |
| `kestrel_sovereign/cli.py::kestrel shell` | Local operator interactive invocation of a named agent. |
| `kestrel_sovereign/cli.py::kestrel skills` | Local operator read/install of feature skills. |
| `kestrel_sovereign/cli.py::kestrel start` | Local operator start of a named agent or the host/fleet. |
| `kestrel_sovereign/cli.py::kestrel status` | Local operator host/fleet process-status read. |
| `kestrel_sovereign/cli.py::kestrel storage` | Local operator storage diagnostics or migrations, optionally fleet-wide. |
| `kestrel_sovereign/cli.py::kestrel terminate` | Local operator termination of a named agent or the host/fleet. |
| `kestrel_sovereign/cli.py::kestrel tool-dispatches` | Local operator read of a selected agent's tool-dispatch log. |
| `kestrel_sovereign/cli.py::kestrel tool-log` | Alias for the local operator tool-dispatch log read. |
| `kestrel_sovereign/cli.py::kestrel update` | Local operator source/install/feature reconciliation followed by named-agent or fleet restart. |
| `kestrel_sovereign/cli.py::kestrel verify-install` | Local operator installation integrity check. |

## Machine-checked HTTP inventory

| Surface ID | Classification |
|---|---|
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/invoke` | Host-authenticated external invocation of the request-routed agent; provenance is not hierarchy authority. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/stream` | Host-authenticated external invocation of the request-routed agent; provenance is not hierarchy authority. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agent/context-status` | Host-authenticated status read from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agent/health/status` | Host-authenticated health read from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agent/heartbeat/status` | Host-authenticated heartbeat-state read from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agent/info` | Host-authenticated identity/capability read from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agent/notifications` | Host-authenticated notification read from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agent/notifications/sse` | Host-authenticated notification stream from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agent/privacy-mode` | Host-authenticated privacy-state read from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agent/reflection/status` | Host-authenticated reflection-state read from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/attachments` | Host-authenticated attachment mutation on the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/health/trigger` | Host-authenticated health check on the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/heartbeat/trigger` | Host-authenticated LLM heartbeat invocation of the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/privacy-mode` | Host-authenticated privacy transition on the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/privacy-mode/cancel` | Host-authenticated cancellation of the request-routed agent's pending privacy transition. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agent/privacy-mode/confirm` | Host-authenticated confirmation of the request-routed agent's pending privacy transition. |
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
| `kestrel_sovereign/endpoints/files.py::GET /api/agent/channels/{channel_type}/link-qr.png` | Host-authenticated channel-link QR read from the request-routed agent. |
| `kestrel_sovereign/endpoints/metrics.py::GET /metrics` | Explicitly public, process-wide Prometheus telemetry; observation grants no control authority. |
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
| `kestrel_sovereign/server.py::GET /health` | Intentionally public aggregate readiness; no agent names or control authority. |
| `kestrel_sovereign/server.py::GET /health/detailed` | Host-authenticated fleet diagnostics, including named per-agent health. |
| `kestrel_sovereign/server.py::GET /phoenix` | Host-authenticated proxy read from the fleet-scoped Phoenix trace store. |
| `kestrel_sovereign/server.py::GET /phoenix/{path:path}` | Host-authenticated proxy read from the fleet-scoped Phoenix trace store. |
| `kestrel_sovereign/server.py::POST /phoenix` | Host-authenticated proxy mutation of the fleet-scoped Phoenix trace store. |
| `kestrel_sovereign/server.py::POST /phoenix/{path:path}` | Host-authenticated proxy mutation of the fleet-scoped Phoenix trace store. |
| `kestrel_sovereign/server.py::PUT /phoenix` | Host-authenticated proxy mutation of the fleet-scoped Phoenix trace store. |
| `kestrel_sovereign/server.py::PUT /phoenix/{path:path}` | Host-authenticated proxy mutation of the fleet-scoped Phoenix trace store. |
| `kestrel_sovereign/server.py::PATCH /phoenix` | Host-authenticated proxy mutation of the fleet-scoped Phoenix trace store. |
| `kestrel_sovereign/server.py::PATCH /phoenix/{path:path}` | Host-authenticated proxy mutation of the fleet-scoped Phoenix trace store. |
| `kestrel_sovereign/server.py::DELETE /phoenix` | Host-authenticated proxy mutation of the fleet-scoped Phoenix trace store. |
| `kestrel_sovereign/server.py::DELETE /phoenix/{path:path}` | Host-authenticated proxy mutation of the fleet-scoped Phoenix trace store. |
| `kestrel_sovereign/server.py::HEAD /phoenix` | Host-authenticated proxy metadata read from the fleet-scoped Phoenix trace store. |
| `kestrel_sovereign/server.py::HEAD /phoenix/{path:path}` | Host-authenticated proxy metadata read from the fleet-scoped Phoenix trace store. |

## Machine-checked request-routed alias inventory

The host routing middleware accepts
`/api/agents/{selected_agent_name}/{remaining_path}` and rewrites the remainder before
FastAPI dispatch. The table therefore spells the synthesized alias for every
decorated core HTTP route, including handlers that do not consume `Request`
and routes whose canonical spelling contains no agent-shaped word. The
canonical root is excluded because the router regex requires a non-empty
remainder. Authentication sees the
prefixed path before routing: every alias below requires host authentication
except the deliberately self-authenticating webhook family. The repeated
agent-local classifications are intentional false-positive dispositions; an
exact-set contract makes any new request-bound route fail until it is added.

Classification codes: **A** = sovereign/operator-authenticated,
target-agent-local read or mutation, with routing selecting the target but
granting no hierarchy authority;
**H** = host/fleet handler whose explicit operator policy ignores agent
selection as authority; **W** = webhook-self-authenticated ingress; **D-n** =
known focused defect n.

| Surface ID | Classification |
|---|---|
| `kestrel_sovereign/endpoints/agent.py::GET /api/agents/{selected_agent_name}/api/agent/context-status` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agents/{selected_agent_name}/api/agent/health/status` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agents/{selected_agent_name}/api/agent/heartbeat/status` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agents/{selected_agent_name}/api/agent/info` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agents/{selected_agent_name}/api/agent/notifications` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agents/{selected_agent_name}/api/agent/notifications/sse` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agents/{selected_agent_name}/api/agent/privacy-mode` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agents/{selected_agent_name}/api/agent/reflection/status` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agents/{selected_agent_name}/api/agent/tasks` | D-3145 — shared-task read lacks principal scoping. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agents/{selected_agent_name}/api/agent/tasks/{task_id}` | D-3145 — shared-task read lacks principal scoping. |
| `kestrel_sovereign/endpoints/agent.py::GET /api/agents/{selected_agent_name}/api/agent/tasks/{task_id}/subscribe` | D-3145 — shared-task read lacks principal scoping. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agents/{selected_agent_name}/api/agent/attachments` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agents/{selected_agent_name}/api/agent/health/trigger` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agents/{selected_agent_name}/api/agent/heartbeat/trigger` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agents/{selected_agent_name}/api/agent/invoke` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agents/{selected_agent_name}/api/agent/privacy-mode` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agents/{selected_agent_name}/api/agent/privacy-mode/cancel` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agents/{selected_agent_name}/api/agent/privacy-mode/confirm` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agents/{selected_agent_name}/api/agent/stop` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agents/{selected_agent_name}/api/agent/stream` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agents/{selected_agent_name}/api/agent/tasks/send` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/agent.py::POST /api/agents/{selected_agent_name}/api/agent/tasks/{task_id:path}/cancel` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /api/agents/{selected_agent_name}/auth/callback` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /api/agents/{selected_agent_name}/auth/login` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /api/agents/{selected_agent_name}/auth/logout` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /api/agents/{selected_agent_name}/auth/me` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /api/agents/{selected_agent_name}/auth/verify` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/auth_oauth.py::POST /api/agents/{selected_agent_name}/auth/token` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/commands.py::GET /api/agents/{selected_agent_name}/api/commands` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::DELETE /api/agents/{selected_agent_name}/api/conversations/messages/{message_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::DELETE /api/agents/{selected_agent_name}/api/conversations/{session_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::GET /api/agents/{selected_agent_name}/api/conversations` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::GET /api/agents/{selected_agent_name}/api/conversations/{session_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::GET /api/agents/{selected_agent_name}/api/conversations/{session_id}/transcript` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::GET /api/agents/{selected_agent_name}/api/sessions` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::GET /api/agents/{selected_agent_name}/api/trash` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::PATCH /api/agents/{selected_agent_name}/api/conversations/{session_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::POST /api/agents/{selected_agent_name}/api/conversations/messages/{message_id}/purge` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::POST /api/agents/{selected_agent_name}/api/conversations/messages/{message_id}/restore` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::POST /api/agents/{selected_agent_name}/api/conversations/new` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::POST /api/agents/{selected_agent_name}/api/conversations/{session_id}/archive` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::POST /api/agents/{selected_agent_name}/api/conversations/{session_id}/purge` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::POST /api/agents/{selected_agent_name}/api/conversations/{session_id}/restore` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/conversations.py::POST /api/agents/{selected_agent_name}/api/conversations/{session_id}/unarchive` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/database.py::GET /api/agents/{selected_agent_name}/api/db/tables` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/database.py::GET /api/agents/{selected_agent_name}/api/db/tables/{table_name}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/features.py::GET /api/agents/{selected_agent_name}/api/features` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/features.py::GET /api/agents/{selected_agent_name}/api/features/installed` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/features.py::GET /api/agents/{selected_agent_name}/api/features/{name}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/features.py::GET /api/agents/{selected_agent_name}/api/features/{name}/config` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/features.py::GET /api/agents/{selected_agent_name}/api/features/{name}/skills` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/features.py::GET /api/agents/{selected_agent_name}/api/skills` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/features.py::GET /api/agents/{selected_agent_name}/api/skills/{skill_id}/schema` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/features.py::GET /api/agents/{selected_agent_name}/api/ui/capabilities` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/features.py::GET /api/agents/{selected_agent_name}/api/ui/contributions` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/features.py::PATCH /api/agents/{selected_agent_name}/api/features/{name}/config` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/features.py::POST /api/agents/{selected_agent_name}/api/features/{name}/disable` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/features.py::POST /api/agents/{selected_agent_name}/api/features/{name}/enable` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/features.py::POST /api/agents/{selected_agent_name}/api/features/{name}/install` | D-3214 — shared-host package mutation lacks sovereign/delegated enforcement. |
| `kestrel_sovereign/endpoints/features.py::POST /api/agents/{selected_agent_name}/api/features/{name}/remove` | D-3214 — shared-host package mutation lacks sovereign/delegated enforcement. |
| `kestrel_sovereign/endpoints/files.py::GET /api/agents/{selected_agent_name}/api/agent/channels/{channel_type}/link-qr.png` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/files.py::GET /api/agents/{selected_agent_name}/api/files/{content_hash}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/files.py::HEAD /api/agents/{selected_agent_name}/api/files/{content_hash}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/github.py::GET /api/agents/{selected_agent_name}/api/github/repos` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/github.py::GET /api/agents/{selected_agent_name}/api/github/{path:path}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/memories.py::DELETE /api/agents/{selected_agent_name}/api/memories/{node_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/memories.py::GET /api/agents/{selected_agent_name}/api/identity-chain` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/memories.py::GET /api/agents/{selected_agent_name}/api/memories` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/memories.py::GET /api/agents/{selected_agent_name}/api/memories/{node_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/metrics.py::GET /api/agents/{selected_agent_name}/metrics` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/endpoints/models.py::DELETE /api/agents/{selected_agent_name}/api/agents/{agent_name}` | H — host/fleet discovery or lifecycle; #3149 policy enforces mutations. |
| `kestrel_sovereign/endpoints/models.py::DELETE /api/agents/{selected_agent_name}/api/keys/user/{provider}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::DELETE /api/agents/{selected_agent_name}/api/keys/{provider}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/agents` | H — host/fleet discovery or lifecycle; #3149 policy enforces mutations. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/constitution` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/embedding/models` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/embedding/reindex/{job_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/embedding/settings` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/identity` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/ipfs/status` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/keys` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/keys/available-sources` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/keys/platform` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/keys/user` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/keys/{provider}/usage` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/model/current` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/models` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/wallet` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/v1/models` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::PATCH /api/agents/{selected_agent_name}/api/identity` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::PATCH /api/agents/{selected_agent_name}/api/keys/{provider}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/api/agents` | H — host/fleet discovery or lifecycle; #3149 policy enforces mutations. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/api/embedding/reindex` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/api/embedding/route-model` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/api/embedding/settings` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/api/embedding/space/verify` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/api/identity/avatar` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/api/identity/avatar/generate` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/api/keys` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/api/keys/user` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/api/keys/user/verify` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/api/model/set` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/v1/chat/completions` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::PUT /api/agents/{selected_agent_name}/api/embedding/route-model` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::PUT /api/agents/{selected_agent_name}/api/embedding/settings` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/observability.py::GET /api/agents/{selected_agent_name}/api/observability/metrics/{metric_name}` | D-3215 — shared PostgreSQL read lacks a trusted selected-agent predicate. |
| `kestrel_sovereign/endpoints/observability.py::GET /api/agents/{selected_agent_name}/api/observability/summary` | D-3215 — shared PostgreSQL read lacks a trusted selected-agent predicate. |
| `kestrel_sovereign/endpoints/rasa_shim.py::POST /api/agents/{selected_agent_name}/webhooks/rest/webhook` | W — target-selected ingress; source auth/rate limit enforce; no hierarchy grant. |
| `kestrel_sovereign/endpoints/restart_events.py::GET /api/agents/{selected_agent_name}/api/restart/status-events` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/saved_items.py::DELETE /api/agents/{selected_agent_name}/api/saved-items/{item_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/saved_items.py::GET /api/agents/{selected_agent_name}/api/saved-items` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/saved_items.py::GET /api/agents/{selected_agent_name}/api/saved-items/by-schema/{schema_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/saved_items.py::GET /api/agents/{selected_agent_name}/api/saved-items/by-tag/{tag}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/saved_items.py::GET /api/agents/{selected_agent_name}/api/saved-items/schemas` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/saved_items.py::GET /api/agents/{selected_agent_name}/api/saved-items/stats` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/saved_items.py::GET /api/agents/{selected_agent_name}/api/saved-items/tags` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/saved_items.py::GET /api/agents/{selected_agent_name}/api/saved-items/{item_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/saved_items.py::PATCH /api/agents/{selected_agent_name}/api/saved-items/{item_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/saved_items.py::POST /api/agents/{selected_agent_name}/api/saved-items` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/saved_items.py::POST /api/agents/{selected_agent_name}/api/saved-items/search` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/saved_items.py::POST /api/agents/{selected_agent_name}/api/saved-items/structured` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/saved_items.py::POST /api/agents/{selected_agent_name}/api/saved-items/{item_id}/pin` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/security.py::DELETE /api/agents/{selected_agent_name}/api/security/auto-approve/rules/{rule_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/security.py::GET /api/agents/{selected_agent_name}/api/security/audit` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/security.py::GET /api/agents/{selected_agent_name}/api/security/auto-approve/audit` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/security.py::GET /api/agents/{selected_agent_name}/api/security/auto-approve/rules` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/security.py::GET /api/agents/{selected_agent_name}/api/security/auto-mode` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/security.py::GET /api/agents/{selected_agent_name}/api/security/pending` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/security.py::GET /api/agents/{selected_agent_name}/api/security/permissions/tree` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/security.py::POST /api/agents/{selected_agent_name}/api/security/approve` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/security.py::POST /api/agents/{selected_agent_name}/api/security/auto-mode` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/security.py::POST /api/agents/{selected_agent_name}/api/security/cancel-all` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/security.py::POST /api/agents/{selected_agent_name}/api/security/cancel/{request_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/security.py::POST /api/agents/{selected_agent_name}/api/security/permissions` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/security.py::POST /api/agents/{selected_agent_name}/api/security/permissions/feature` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/security.py::POST /api/agents/{selected_agent_name}/api/security/reset-session` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/sovereignty.py::GET /api/agents/{selected_agent_name}/api/sovereignty/exports` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/sovereignty.py::GET /api/agents/{selected_agent_name}/api/sovereignty/files` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/sovereignty.py::GET /api/agents/{selected_agent_name}/api/sovereignty/files/{filename}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/sovereignty.py::GET /api/agents/{selected_agent_name}/api/sovereignty/files/{filename}/preview` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/sovereignty.py::GET /api/agents/{selected_agent_name}/api/storage/stats` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/sovereignty.py::POST /api/agents/{selected_agent_name}/api/sovereignty/export` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/sovereignty.py::POST /api/agents/{selected_agent_name}/api/sovereignty/import` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/spawn.py::GET /api/agents/{selected_agent_name}/api/spawn/children` | D-3142 — unverified process-local child relation. |
| `kestrel_sovereign/endpoints/ui.py::GET /api/agents/{selected_agent_name}/api/ui/theme` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/ui.py::GET /api/agents/{selected_agent_name}/api/ui/themes` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/features/bridge/router.py::GET /api/agents/{selected_agent_name}/api/bridge/capabilities` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/features/bridge/router.py::GET /api/agents/{selected_agent_name}/api/bridge/health` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/features/bridge/router.py::POST /api/agents/{selected_agent_name}/api/bridge/invoke` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/features/bridge/router.py::POST /api/agents/{selected_agent_name}/api/bridge/session` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/features/bridge/router.py::POST /api/agents/{selected_agent_name}/api/bridge/stream` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/features/web_search/feature.py::GET /api/agents/{selected_agent_name}/api/features/web_search/test` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/features/webhooks/receiver.py::POST /api/agents/{selected_agent_name}/webhooks/{webhook_name}` | W — target-selected ingress; source auth/rate limit enforce; no hierarchy grant. |
| `kestrel_sovereign/server.py::DELETE /api/agents/{selected_agent_name}/phoenix` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::DELETE /api/agents/{selected_agent_name}/phoenix/{path:path}` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/api/auth/key` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/api/host/csrf` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/api/host/ui/contributions` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/assets/{path:path}` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/health` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/health/detailed` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/phoenix` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/phoenix/{path:path}` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::HEAD /api/agents/{selected_agent_name}/assets/{path:path}` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::HEAD /api/agents/{selected_agent_name}/phoenix` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::HEAD /api/agents/{selected_agent_name}/phoenix/{path:path}` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::PATCH /api/agents/{selected_agent_name}/phoenix` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::PATCH /api/agents/{selected_agent_name}/phoenix/{path:path}` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::POST /api/agents/{selected_agent_name}/api/host/phoenix/session` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::POST /api/agents/{selected_agent_name}/phoenix` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::POST /api/agents/{selected_agent_name}/phoenix/{path:path}` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::PUT /api/agents/{selected_agent_name}/phoenix` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::PUT /api/agents/{selected_agent_name}/phoenix/{path:path}` | H — explicit operator/public policy; selected-agent context is not authority. |

## Review rule

Adding a cross-agent surface requires updating this matrix in the same change.
The implementation must bind the trusted caller identity before the operation,
authorize and mutate atomically where storage is shared, return the same public
shape for missing and unauthorized targets, and include a mutation test that
fails if the authority predicate or its wiring is removed.
