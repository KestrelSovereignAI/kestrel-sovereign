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
| Invoke a routed agent | `POST /api/agent/invoke`; `POST /api/agent/stream`; deprecated `/agent/invoke` and `/agent/stream`; `POST /api/bridge/invoke`; `POST /api/bridge/stream`; `POST /v1/chat/completions` | The agent pinned by trusted request routing | Outside agent hierarchy; host-authenticated external ingress | The host authenticates the API-key/JWT/session caller and `get_agent(request)` consumes the middleware-pinned target. Bridge sender/session fields and invocation provenance describe the request; they confer no peer or hierarchy authority. |
| Operate on routed-agent state | Remaining `/api/agent/*` status, notification, health, heartbeat, privacy, attachment, and channel-link routes, plus their deprecated `/agent/*` spellings | The agent pinned by trusted request routing | Self or host-authenticated external operation | On a multi-agent host every singular route is addressable through `/api/agents/{name}/...`; the trusted routing middleware pins the runtime before the handler runs. The complete canonical and compatibility namespaces are machine-inventoried below so a future cross-agent operation cannot hide behind an innocuous suffix. |
| General webhook ingress | `POST /webhooks/{webhook_name}` | The request-bound agent when agent-prefixed; otherwise the first enabled receiver matching the name | Outside agent hierarchy; explicitly configured external ingress | Agent-prefixed routing binds the target from trusted request state. The unprefixed multi-agent route aggregates receivers and does not reject duplicate names, so iteration order can choose the target. Defect: [#3216](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3216). The supported `auth_type="none"`, `rate_limit=0` combination accepts every reachable request without throttling; `allow_unauthenticated` acknowledges that choice but adds no gate. A bounded operator policy therefore requires a real auth mode and a positive rate limit. Payload fields create no agent authority. |
| Rasa webhook ingress | `POST /webhooks/rest/webhook`; synthesized `/api/agents/{name}/webhooks/rest/webhook` alias | The host-default agent, even when the request prefix names a different agent | Outside agent hierarchy; sovereign-configured external ingress | Rasa requires its sovereign-configured shared secret and applies a fixed request rate, but the handler reads `app.state.agent` rather than the request-routed target. The prefixed alias can therefore invoke the wrong agent or fail when no default exists: [#3220](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3220). The payload sender becomes session/provenance data only and creates no agent authority. |
| Bootstrap host API credential | `GET /api/auth/key` | The host-wide API authentication boundary | Public-localhost provisioning exception that yields runtime sovereign authority | The endpoint is disabled unless bootstrap policy enables it, restricts callers to loopback/Docker gateway/explicit allowed hosts, and is rate-limited. It returns the host API key. Runtime authentication maps that key to `CallerRole.SOVEREIGN`, so it satisfies the current #3149 host-lifecycle gate and can create or withdraw hosted agents. The API key remains distinct from the constitutional sovereign signing key; bootstrap policy is nevertheless a path to host administration. This classification is reconciled with `docs/audit/AUTH_SURFACE_MATRIX.md`. |
| Authenticate a host user / inspect credentials | `/auth/login`, `/auth/callback`, `/auth/logout`, `/auth/token`, `/auth/me`, `/auth/verify` | The host-wide OAuth session or JWT authentication boundary | Outside agent hierarchy; configured human authentication | Login, callback, logout, and token are self-authenticating entrypoints; token issuance is allowlist/password checked and rate-limited. `me` and `verify` first require central host authentication and apply their endpoint semantics. A synthesized `/api/agents/{name}/auth/*` prefix is host-authenticated by the outer middleware and does not make these host credential handlers target-agent-local or confer relation authority. This classification is reconciled with `docs/audit/AUTH_SURFACE_MATRIX.md`. |
| Read host UI state / issue browser tokens | `GET /api/host/ui/contributions`; `GET /api/host/csrf`; `POST /api/host/phoenix/session` | Shared host UI manifest, CSRF token, or Phoenix embed session | Host-authenticated external operation | Central API-key/JWT/session middleware protects these app-level routes. The CSRF token is a double-submit value rather than standalone authority; the Phoenix route mints a short-lived path-scoped cookie only after host authentication and backend reachability. |
| Read API documentation or mounted UI assets | FastAPI-generated `/openapi.json`, `/docs*`, and `/redoc`; core and feature `app.mount(...)` boundaries | Host API schema/documentation and static UI asset trees | Outside agent hierarchy; host publication policy | Generated routes and programmatic mounts are entry doors even though they have no decorator. The contract inventories FastAPI constructor defaults, concrete core mount prefixes, and a stable expression marker for runtime-computed feature mount paths. A selected-agent prefix only rewrites to the same host/static tree and grants no agent relation authority. |
| Use the host GitHub credential | `GET /api/github/repos`; `GET /api/github/{path:path}` | Repositories visible to the process-wide GitHub token and host configuration | Outside agent hierarchy; host-authenticated external operation | Global authentication protects both routes. The handlers do not bind `get_agent`; repository allowlisting constrains the process-wide token. A synthesized selected-agent prefix therefore remains host-scoped and grants no agent relation authority. |
| Manage authenticated-user or platform service keys | `/api/keys/user*`; `GET /api/keys/platform` | The authenticated user's BYOK namespace, or the platform-global key catalog | Outside agent hierarchy; authenticated-user or host policy | User-key handlers key storage on the request-context `request.state.user_id` principal; the selected agent supplies only PostgreSQL connectivity. Missing platform/user context makes the routes unavailable. The platform catalog is host-global, so neither family becomes agent-local when reached through a selected-agent prefix. |
| Browse sovereignty export cache | `GET /api/sovereignty/files`; `GET /api/sovereignty/files/{filename}`; `GET /api/sovereignty/files/{filename}/preview` | The routed agent's export artifacts, or the shared host cache for an explicit operator surface | Self for agent artifacts; sovereign/delegated for host-wide observation | The handlers read the process-global `STORAGE_CACHE_DIR` and authorize only by filename/path containment; they never bind an artifact owner from the trusted routed agent. Listing also returns host paths and `.meta` content. Defect: [#3225](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3225). |
| Inspect local IPFS node and pins | `GET /api/ipfs/status` | Agent-owned pins, or shared daemon health/pins for an explicit operator surface | Self for agent pins; sovereign/delegated for host-wide observation | The handler resolves an agent but queries the process-global IPFS daemon and returns its unscoped recursive pin set. Optional agent adapter metadata does not constrain those CIDs. Defect: [#3226](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3226). |
| Read host/fleet health or process metrics | `GET /health`; `GET /health/detailed`; `GET /metrics` | Public aggregate readiness, authenticated per-agent fleet diagnostics, or public process-wide Prometheus telemetry | Outside agent hierarchy; deployment/operator observation policy | `/health` intentionally exposes only aggregate readiness and `/metrics` is an explicitly public scraper surface. Global authentication protects `/health/detailed`, whose multi-agent response names agents and their checks. None of these observations creates control authority. |
| Use the Phoenix trace proxy | All registered methods on `/phoenix` and `/phoenix/{path:path}` | The shared host Phoenix trace store | Outside agent hierarchy; host-authenticated external operation | Global authentication accepts a host session/API key or the short-lived path-scoped embed cookie. The route proxies fleet-scoped trace data but does not grant one agent authority over another. Symbolic `api_route(methods=...)` declarations are expanded by the contract scanner. |
| Read observability summaries/metrics | `GET /api/observability/summary`; `GET /api/observability/metrics/{metric_name}` | The routed agent's events only | Self | A per-agent SQLite store happens to isolate the data, but shared PostgreSQL queries omit the trusted agent predicate and the metrics route accepts an arbitrary optional `agent_name`. Defect: [#3215](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3215). |
| Read consent history/statistics | `consent_log`; `consent_stats` | The calling agent's consent records and aggregates | Self | Writes stamp the trusted agent DID, but every detailed and aggregate read currently omits that predicate on shared PostgreSQL. A caller can observe other agents' consent reflections and fleet-contaminated statistics. Defect: [#3229](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3229). |
| Anchor/status/verify audit history | `audit_anchor`; `audit_anchor_status`; `audit_verify` | The calling agent's audit entries and anchor rows | Self | Anchor writes stamp the trusted agent DID, but last-anchor, count, and enumeration reads omit it on shared PostgreSQL. Foreign anchors can suppress local anchoring, contaminate status, and be verified against the wrong audit log. Defect: [#3230](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3230). |
| Inspect/configure routed feature state | Feature catalog/detail/config/skills routes; feature enable/disable/config mutation | The request-routed agent | Self or sovereign/delegated operator policy | `get_agent(request)` binds the target runtime. These namespace matches are classified explicitly so they cannot conceal a future cross-agent implementation; they currently do not grant one agent authority over another. |
| Install/remove feature package | `POST /api/features/{name}/install`; `POST /api/features/{name}/remove` | Shared host interpreter and all loaded users of the package | Sovereign/delegated | The handlers currently require only an authenticated routed agent despite their sovereign-only docstrings. Defect: [#3214](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3214). |
| Manage shared local models | `pull_model`; `cleanup_models(dry_run=False)` | Shared local-model service and model storage used by every co-hosted agent | Sovereign/delegated | The tools currently use ordinary feature permission and mutate the shared Ollama/model store. Cleanup protects only models used by the calling agent rather than accounting for all hosted agents. Defect: [#3221](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3221). |
| Deploy/teardown shared agent hosting | `deploy_agent` with a multi-agent deployment profile | An external service hosting the entire configured agent fleet | Sovereign/delegated | The checked-in `dev` profile selects `deployment_mode="multi_agent"`, but an ordinary agent can invoke deployment or teardown after generic ASK consent, which may be auto-promoted and is not constitutional authority. Defect: [#3223](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3223). |
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

The contract test discovers every core feature `@tool`, including tools that
appear agent-local or external, plus every core runtime-generated `AgentTool`
execution boundary, the generic `Feature.to_orchestrator_tool` registration
boundary, and the registration/execution boundaries for non-feature dynamic
tools such as MCP tools. Cross-agent capability is a property of the
implementation and deployment, not a public-name convention: exact inventory
of the complete registered set prevents a shared-host mutation from hiding
behind an innocuous name. Direct writes to the runtime tool registry are also
discovered, including the ephemeral constitution-receipt tool. False positives
remain explicitly classified. The
inventory therefore also includes generic `execute_skill`, `execute_named_tool`, and
`_create_schedule` dispatchers; those meta-tools can reach an
authority-bearing target even when their own names contain no relation or
control words. Runtime-advertised provider names cannot be enumerated from the
core checkout, so their core forwarding boundaries are classified here and
each out-of-tree provider repository remains responsible for its own exact
method inventory and target-specific authority.

| Surface ID | Classification |
|---|---|
| `kestrel_sovereign/agent/orchestrator_engine.py::_dispatch_direct_tool` | Generic governed execution boundary for runtime-registered direct tools. PRE/POST_TOOL_USE and ordinary tool permission checks still apply, but the owning non-feature provider must inventory and enforce target-specific relation authority. |
| `kestrel_sovereign/agent/orchestrator_engine.py::execute_named_tool` | Generic transport-neutral dispatcher that can resolve runtime-registered direct tools as well as feature tools. Dispatch supplies governance hooks, not relation authority; the selected tool retains its target-specific gate. |
| `kestrel_sovereign/agent/tool_registry.py::register_dynamic_tools` | Generic publication boundary for arbitrary runtime tool names, including MCP providers. The registry defaults unknown tools to ASK but cannot infer target authority; provider-owned controls require their own exact inventory and enforcement. |
| `kestrel_sovereign/kestrel_agent.py::KestrelAgent._handle_constitution_receipt_tool` | Execution boundary for the ephemeral constitution-receipt canary. It records exact system-prompt receipt for one local cognition turn and grants no agent relation authority. |
| `kestrel_sovereign/kestrel_agent.py::KestrelAgent.register_constitution_receipt_tool` | Direct publication boundary for the ephemeral constitution-receipt canary. The dispatcher owns its expected value and lifetime; publication grants no peer, parent, or host authority. |
| `kestrel_sovereign/features/base.py::Feature.get_tools.DynamicTool.execute` | Generic runtime wrapper for the separately inventoried core `@tool` methods; it introduces no target authority and the wrapped method retains its target-specific gate. |
| `kestrel_sovereign/features/base.py::Feature.to_orchestrator_tool` | Generic high-level dispatcher registered once per visible feature, including `deploy_feature` and `restart_coordinator_feature`. PRE_SUBAGENT_CALL/PRE_TOOL_USE are operational consent gates, not relation authority; the selected downstream method retains its target-specific authority requirement and known defects #3223/#3148. |
| `kestrel_sovereign/features/isolated_runtime.py::IsolatedFeatureTool.execute` | Generic forwarding boundary for runtime-advertised out-of-tree tools. Core preserves ordinary tool governance but cannot infer relation authority from child metadata; the owning feature must inventory and enforce every target-specific authority boundary. |
| `kestrel_sovereign/features/bootstrap/feature.py::rename_agent` | Self-only display-name mutation; not a peer door. |
| `kestrel_sovereign/features/bootstrap/feature.py::restart_discovery` | Self-only bootstrap-state retry; not a host restart. |
| `kestrel_sovereign/features/deploy/feature.py::deploy_agent` | D-3223 — a checked-in multi-agent profile can deploy or teardown the fleet through generic ASK rather than sovereign/delegated authority. |
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
| `kestrel_sovereign/features/strategic_memory/feature.py::signal_dispatch` | Self-owned indirect dispatcher through the governed `workflow_run` tool; the contributed workflow and selected downstream controls retain their own consent, evidence, and target-authority gates. |
| `kestrel_sovereign/features/tasks/feature.py::attach_artifact_to_a2a_task` | Recipient-owned mutation; #3144. |
| `kestrel_sovereign/features/tasks/feature.py::cancel_task` | Creator/recipient-owned mutation; #3134. |
| `kestrel_sovereign/features/tasks/feature.py::check_task_status` | Unscoped task-ID read; defect [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| `kestrel_sovereign/features/tasks/feature.py::get_task_result` | Unscoped task-ID read; defect [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| `kestrel_sovereign/features/tasks/feature.py::list_my_tasks` | Unscoped shared-store inbox read; defect [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| `kestrel_sovereign/features/tasks/feature.py::respond_to_a2a_task` | Recipient-owned mutation; #3144. |
| `kestrel_sovereign/features/tasks/feature.py::run_workflow` | Self-owned indirect dispatcher through `TaskManager.execute_skill`; each selected feature tool retains its PRE_TOOL_USE and target-specific authority checks, so the workflow supplies sequence, not relation authority. |
| `kestrel_sovereign/features/todo/feature.py::todo_link_task` | Self-owned todo metadata link; not an A2A task control. |
| `kestrel_sovereign/features/attachments/feature.py::read_attachment` | Caller-bound attachment read; no co-hosted-agent target. |
| `kestrel_sovereign/features/audit_anchor/feature.py::audit_anchor` | D-3230: shared PostgreSQL last-anchor lookup omits the trusted agent predicate, so a foreign newer anchor can suppress this caller's audit anchoring. |
| `kestrel_sovereign/features/audit_anchor/feature.py::audit_anchor_status` | D-3230: shared PostgreSQL last-anchor and count reads are unscoped, exposing fleet-contaminated audit status. |
| `kestrel_sovereign/features/audit_anchor/feature.py::audit_verify` | D-3230: shared PostgreSQL anchor enumeration is unscoped and can verify a foreign anchor against this caller's audit log. |
| `kestrel_sovereign/features/bootstrap/feature.py::bootstrap_add` | Caller-owned bootstrap configuration or state; no co-hosted-agent target. |
| `kestrel_sovereign/features/bootstrap/feature.py::bootstrap_list` | Caller-owned bootstrap configuration or state; no co-hosted-agent target. |
| `kestrel_sovereign/features/bootstrap/feature.py::bootstrap_reload` | Caller-owned bootstrap configuration or state; no co-hosted-agent target. |
| `kestrel_sovereign/features/bootstrap/feature.py::bootstrap_remove` | Caller-owned bootstrap configuration or state; no co-hosted-agent target. |
| `kestrel_sovereign/features/bootstrap/feature.py::bootstrap_status` | Caller-owned bootstrap configuration or state; no co-hosted-agent target. |
| `kestrel_sovereign/features/bootstrap/feature.py::skip_discovery` | Caller-owned bootstrap configuration or state; no co-hosted-agent target. |
| `kestrel_sovereign/features/bridge/feature.py::bridge_connections` | Caller-bound external bridge connection or history; no co-hosted-agent control. |
| `kestrel_sovereign/features/bridge/feature.py::bridge_history` | Caller-bound external bridge connection or history; no co-hosted-agent control. |
| `kestrel_sovereign/features/bridge/feature.py::bridge_status` | Caller-bound external bridge connection or history; no co-hosted-agent control. |
| `kestrel_sovereign/features/channels/feature.py::channels_history` | Caller-configured external-channel operation; channel policy and ordinary tool consent remain enforcement, not agent hierarchy. |
| `kestrel_sovereign/features/channels/feature.py::channels_list` | Caller-configured external-channel operation; channel policy and ordinary tool consent remain enforcement, not agent hierarchy. |
| `kestrel_sovereign/features/channels/feature.py::channels_send` | Caller-configured external-channel operation; channel policy and ordinary tool consent remain enforcement, not agent hierarchy. |
| `kestrel_sovereign/features/cli/feature.py::cli_status` | Read-only inspection of the configured repository checkout; shared-host observation is operational access, not agent hierarchy. |
| `kestrel_sovereign/features/cli/feature.py::git_diff` | Read-only inspection of the configured repository checkout; shared-host observation is operational access, not agent hierarchy. |
| `kestrel_sovereign/features/cli/feature.py::git_log` | Read-only inspection of the configured repository checkout; shared-host observation is operational access, not agent hierarchy. |
| `kestrel_sovereign/features/cli/feature.py::git_merge_base` | Read-only inspection of the configured repository checkout; shared-host observation is operational access, not agent hierarchy. |
| `kestrel_sovereign/features/cli/feature.py::git_show_file` | Read-only inspection of the configured repository checkout; shared-host observation is operational access, not agent hierarchy. |
| `kestrel_sovereign/features/cli/feature.py::git_status` | Read-only inspection of the configured repository checkout; shared-host observation is operational access, not agent hierarchy. |
| `kestrel_sovereign/features/compute/feature.py::empty_trash` | Compute-workspace operation governed by configured sandbox, path, trash, and approval policy; any host reach is operational capability, not agent authority. |
| `kestrel_sovereign/features/compute/feature.py::execution_history` | Compute-workspace operation governed by configured sandbox, path, trash, and approval policy; any host reach is operational capability, not agent authority. |
| `kestrel_sovereign/features/compute/feature.py::get_compute_capabilities` | Compute-workspace operation governed by configured sandbox, path, trash, and approval policy; any host reach is operational capability, not agent authority. |
| `kestrel_sovereign/features/compute/feature.py::get_compute_policy` | Compute-workspace operation governed by configured sandbox, path, trash, and approval policy; any host reach is operational capability, not agent authority. |
| `kestrel_sovereign/features/compute/feature.py::list_scripts` | Compute-workspace operation governed by configured sandbox, path, trash, and approval policy; any host reach is operational capability, not agent authority. |
| `kestrel_sovereign/features/compute/feature.py::list_trash` | Compute-workspace operation governed by configured sandbox, path, trash, and approval policy; any host reach is operational capability, not agent authority. |
| `kestrel_sovereign/features/compute/feature.py::restore_from_trash` | Compute-workspace operation governed by configured sandbox, path, trash, and approval policy; any host reach is operational capability, not agent authority. |
| `kestrel_sovereign/features/compute/feature.py::run_script` | Compute-workspace operation governed by configured sandbox, path, trash, and approval policy; any host reach is operational capability, not agent authority. |
| `kestrel_sovereign/features/compute/feature.py::show_script` | Compute-workspace operation governed by configured sandbox, path, trash, and approval policy; any host reach is operational capability, not agent authority. |
| `kestrel_sovereign/features/compute/feature.py::write_script` | Compute-workspace operation governed by configured sandbox, path, trash, and approval policy; any host reach is operational capability, not agent authority. |
| `kestrel_sovereign/features/computer_use/feature.py::fs_edit` | Host-computer operation governed by filesystem/shell policy and consent; this strong operational capability grants no relation authority. |
| `kestrel_sovereign/features/computer_use/feature.py::fs_list` | Host-computer operation governed by filesystem/shell policy and consent; this strong operational capability grants no relation authority. |
| `kestrel_sovereign/features/computer_use/feature.py::fs_read` | Host-computer operation governed by filesystem/shell policy and consent; this strong operational capability grants no relation authority. |
| `kestrel_sovereign/features/computer_use/feature.py::fs_write` | Host-computer operation governed by filesystem/shell policy and consent; this strong operational capability grants no relation authority. |
| `kestrel_sovereign/features/computer_use/feature.py::shell` | Host-computer operation governed by filesystem/shell policy and consent; this strong operational capability grants no relation authority. |
| `kestrel_sovereign/features/consent/feature.py::consent_log` | D-3229: shared PostgreSQL detail reads omit the trusted agent predicate and disclose other agents' consent records. |
| `kestrel_sovereign/features/consent/feature.py::consent_stats` | D-3229: every shared PostgreSQL aggregate read is unscoped, so another agent's consent data contaminates caller-visible statistics. |
| `kestrel_sovereign/features/constitution.py::constitution` | No co-hosted-agent target identified; caller-local or external-operation policy remains enforcement. |
| `kestrel_sovereign/features/context/feature.py::compact_context` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/context/feature.py::context_stash` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/context/feature.py::context_stash_apply` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/context/feature.py::context_stash_drop` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/context/feature.py::context_stash_list` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/context/feature.py::context_stash_peek` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/context/feature.py::context_stash_pop` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/context/feature.py::context_stash_save` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/context/feature.py::context_status` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/context/feature.py::exclude_from_context` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/context/feature.py::hierarchical_compact` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/context/feature.py::mark_content` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/context/feature.py::recursive_query` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/context/feature.py::restore_excluded` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/context/feature.py::summarize_section` | Caller-owned context mutation or read; no co-hosted-agent target. |
| `kestrel_sovereign/features/delivery/feature.py::delivery_failed` | Caller-owned outbound delivery queue operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/delivery/feature.py::delivery_purge` | Caller-owned outbound delivery queue operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/delivery/feature.py::delivery_queue_list` | Caller-owned outbound delivery queue operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/delivery/feature.py::delivery_retry` | Caller-owned outbound delivery queue operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/delivery/feature.py::delivery_status` | Caller-owned outbound delivery queue operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/health/feature.py::health_check` | Caller runtime health or heartbeat state; no co-hosted-agent target. |
| `kestrel_sovereign/features/health/feature.py::health_history` | Caller runtime health or heartbeat state; no co-hosted-agent target. |
| `kestrel_sovereign/features/health/feature.py::health_interval` | Caller runtime health or heartbeat state; no co-hosted-agent target. |
| `kestrel_sovereign/features/health/feature.py::heartbeat_check` | Caller runtime health or heartbeat state; no co-hosted-agent target. |
| `kestrel_sovereign/features/health/feature.py::heartbeat_interval` | Caller runtime health or heartbeat state; no co-hosted-agent target. |
| `kestrel_sovereign/features/health/feature.py::heartbeat_status` | Caller runtime health or heartbeat state; no co-hosted-agent target. |
| `kestrel_sovereign/features/identity/feature.py::assess_substrate` | Caller identity, custody, lifecycle, or migration operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/identity/feature.py::export_identity` | Caller identity, custody, lifecycle, or migration operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/identity/feature.py::import_identity` | Caller identity, custody, lifecycle, or migration operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/identity/feature.py::lifecycle_status` | Caller identity, custody, lifecycle, or migration operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/identity/feature.py::migration_history` | Caller identity, custody, lifecycle, or migration operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/identity/feature.py::verify_identity` | Caller identity, custody, lifecycle, or migration operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/inference_lease/feature.py::inference_lease_acquire` | Owner-bound lease over shared inference capacity; lease ownership and provider policy apply, and no agent hierarchy is granted. |
| `kestrel_sovereign/features/inference_lease/feature.py::inference_lease_release` | Owner-bound lease over shared inference capacity; lease ownership and provider policy apply, and no agent hierarchy is granted. |
| `kestrel_sovereign/features/inference_lease/feature.py::inference_lease_status` | Owner-bound lease over shared inference capacity; lease ownership and provider policy apply, and no agent hierarchy is granted. |
| `kestrel_sovereign/features/keys/feature.py::add_service_key` | Caller-owned service-credential operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/keys/feature.py::delete_service_key` | Caller-owned service-credential operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/keys/feature.py::get_key_usage` | Caller-owned service-credential operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/keys/feature.py::list_providers` | Caller-owned service-credential operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/keys/feature.py::list_service_keys` | Caller-owned service-credential operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/keys/feature.py::remove_service_key` | Caller-owned service-credential operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/keys/feature.py::rotate_service_key` | Caller-owned service-credential operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::confirm_person_match` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::delete_conversation` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::delete_message_by_id` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::delete_messages` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::get_episodes` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::list_conversations` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::list_trashed_messages` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::mark_superseded` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::memory_consolidate` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::memory_index_backfill` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::memory_status` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::purge_conversation` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::purge_message_by_id` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::recall_action_items` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::recall_decisions` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::recall_emotional` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::recall_interactions` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::recall_recent` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::restore_conversation` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::restore_message_by_id` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::search_case_law` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::search_documents` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::search_memory` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory/feature.py::update_action_item` | Caller-owned memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory_agency/feature.py::forget_fact` | Caller-owned memory-agency namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory_agency/feature.py::memory_admin_unpin_all` | Caller-owned memory-agency namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory_agency/feature.py::memory_admin_unpin_oldest` | Caller-owned memory-agency namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory_agency/feature.py::memory_pin` | Caller-owned memory-agency namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory_agency/feature.py::memory_pin_stats` | Caller-owned memory-agency namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory_agency/feature.py::memory_pinned` | Caller-owned memory-agency namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory_agency/feature.py::memory_release` | Caller-owned memory-agency namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/memory_agency/feature.py::save_fact` | Caller-owned memory-agency namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/model/feature.py::cleanup_models` | D-3221 — deletes from shared model storage without sovereign/delegated authority or fleet-wide in-use accounting. |
| `kestrel_sovereign/features/model/feature.py::get_current_model` | Caller-agent model preference read or mutation; no co-hosted-agent target. |
| `kestrel_sovereign/features/model/feature.py::get_model_info` | Read-only observation of the shared local-model service or storage; no agent relation authority. |
| `kestrel_sovereign/features/model/feature.py::get_model_storage_info` | Read-only observation of the shared local-model service or storage; no agent relation authority. |
| `kestrel_sovereign/features/model/feature.py::list_models` | Read-only observation of the shared local-model service or storage; no agent relation authority. |
| `kestrel_sovereign/features/model/feature.py::pull_model` | D-3221 — mutates the shared local-model service without sovereign/delegated host authority. |
| `kestrel_sovereign/features/model/feature.py::set_model` | Caller-agent model preference read or mutation; no co-hosted-agent target. |
| `kestrel_sovereign/features/response_audit/feature.py::audit_disable` | Caller-owned response-audit configuration or status; no co-hosted-agent target. |
| `kestrel_sovereign/features/response_audit/feature.py::audit_enable` | Caller-owned response-audit configuration or status; no co-hosted-agent target. |
| `kestrel_sovereign/features/response_audit/feature.py::audit_status` | Caller-owned response-audit configuration or status; no co-hosted-agent target. |
| `kestrel_sovereign/features/save/feature.py::recall` | Caller-owned saved-memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/save/feature.py::recall_delete` | Caller-owned saved-memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/save/feature.py::recall_get` | Caller-owned saved-memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/save/feature.py::recall_list` | Caller-owned saved-memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/save/feature.py::save_excerpt` | Caller-owned saved-memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/save/feature.py::save_item` | Caller-owned saved-memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/save/feature.py::save_stash` | Caller-owned saved-memory namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/scheduler/feature.py::schedule_engagement` | Caller-owned schedule state; scheduling conveys no authority to a downstream target. |
| `kestrel_sovereign/features/scheduler/feature.py::schedule_history` | Caller-owned schedule state; scheduling conveys no authority to a downstream target. |
| `kestrel_sovereign/features/scheduler/feature.py::schedule_list` | Caller-owned schedule state; scheduling conveys no authority to a downstream target. |
| `kestrel_sovereign/features/scheduler/feature.py::schedule_pause` | Caller-owned schedule state; scheduling conveys no authority to a downstream target. |
| `kestrel_sovereign/features/scheduler/feature.py::schedule_record_outcome` | Caller-owned schedule state; scheduling conveys no authority to a downstream target. |
| `kestrel_sovereign/features/scheduler/feature.py::schedule_remove` | Caller-owned schedule state; scheduling conveys no authority to a downstream target. |
| `kestrel_sovereign/features/scheduler/feature.py::schedule_resume` | Caller-owned schedule state; scheduling conveys no authority to a downstream target. |
| `kestrel_sovereign/features/scheduler/feature.py::schedule_update` | Caller-owned schedule state; scheduling conveys no authority to a downstream target. |
| `kestrel_sovereign/features/security/feature.py::approve` | Caller-agent consent and permission state; no permission over a co-hosted agent. |
| `kestrel_sovereign/features/security/feature.py::deny` | Caller-agent consent and permission state; no permission over a co-hosted agent. |
| `kestrel_sovereign/features/security/feature.py::list_permissions` | Caller-agent consent and permission state; no permission over a co-hosted agent. |
| `kestrel_sovereign/features/security/feature.py::pending_approvals` | Caller-agent consent and permission state; no permission over a co-hosted agent. |
| `kestrel_sovereign/features/security/feature.py::security_audit` | Caller-agent consent and permission state; no permission over a co-hosted agent. |
| `kestrel_sovereign/features/security/feature.py::security_audit_search` | Caller-agent consent and permission state; no permission over a co-hosted agent. |
| `kestrel_sovereign/features/security/feature.py::set_permission` | Caller-agent consent and permission state; no permission over a co-hosted agent. |
| `kestrel_sovereign/features/skills/feature.py::skill_delete` | Caller-owned skill-library operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/skills/feature.py::skill_extract_candidates` | Caller-owned skill-library operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/skills/feature.py::skill_list` | Caller-owned skill-library operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/skills/feature.py::skill_save` | Caller-owned skill-library operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/skills/feature.py::skill_show` | Caller-owned skill-library operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/sovereignty/feature.py::check_sovereignty_status` | Caller-owned sovereignty data/status operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/sovereignty/feature.py::export_sovereignty` | Caller-owned sovereignty data/status operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/sovereignty/feature.py::import_sovereignty` | Caller-owned sovereignty data/status operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/state_of_mind.py::state_of_mind` | No co-hosted-agent target identified; caller-local or external-operation policy remains enforcement. |
| `kestrel_sovereign/features/strategic_memory/feature.py::backlog_hygiene` | Caller-owned strategic-memory operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/strategic_memory/feature.py::morning_signal` | Caller-owned strategic-memory operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/strategic_memory/feature.py::recall_blockers` | Caller-owned strategic-memory operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/strategic_memory/feature.py::recall_patterns` | Caller-owned strategic-memory operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/strategic_memory/feature.py::session_log` | Caller-owned strategic-memory operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/strategic_memory/feature.py::strategy_add_blocker` | Caller-owned strategic-memory operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/strategic_memory/feature.py::strategy_add_decision` | Caller-owned strategic-memory operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/strategic_memory/feature.py::strategy_add_pattern` | Caller-owned strategic-memory operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/strategic_memory/feature.py::strategy_reconcile_blockers` | Caller-owned strategic-memory operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/strategic_memory/feature.py::strategy_resolve_blocker` | Caller-owned strategic-memory operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/strategic_memory/feature.py::strategy_search` | Caller-owned strategic-memory operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/strategic_memory/feature.py::strategy_supersede_pattern` | Caller-owned strategic-memory operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/strategic_memory/feature.py::strategy_view` | Caller-owned strategic-memory operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/tasks/feature.py::list_available_skills` | Read-only caller-visible feature registry; no task or agent mutation. |
| `kestrel_sovereign/features/todo/feature.py::todo_add` | Caller-owned todo namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/todo/feature.py::todo_complete` | Caller-owned todo namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/todo/feature.py::todo_list` | Caller-owned todo namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/todo/feature.py::todo_rollup` | Caller-owned todo namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/todo/feature.py::todo_update` | Caller-owned todo namespace operation; no co-hosted-agent target. |
| `kestrel_sovereign/features/wait/feature.py::wait` | Current caller turn suspension; no co-hosted-agent control. |
| `kestrel_sovereign/features/web_search/feature.py::web_search` | External search operation under ordinary tool policy; no co-hosted-agent target. |
| `kestrel_sovereign/features/webhooks/feature.py::webhooks_history` | Caller-owned webhook receiver configuration/history; unprefixed target ambiguity remains #3216 and no mode grants agent hierarchy. |
| `kestrel_sovereign/features/webhooks/feature.py::webhooks_list` | Caller-owned webhook receiver configuration/history; unprefixed target ambiguity remains #3216 and no mode grants agent hierarchy. |
| `kestrel_sovereign/features/webhooks/feature.py::webhooks_register` | Caller-owned webhook receiver configuration/history; unprefixed target ambiguity remains #3216 and no mode grants agent hierarchy. |
| `kestrel_sovereign/features/webhooks/feature.py::webhooks_remove` | Caller-owned webhook receiver configuration/history; unprefixed target ambiguity remains #3216 and no mode grants agent hierarchy. |
| `kestrel_sovereign/features/wellness/feature.py::wellness_check` | Caller runtime wellness state; no co-hosted-agent target. |
| `kestrel_sovereign/features/wellness/feature.py::wellness_export` | Caller runtime wellness state; no co-hosted-agent target. |
| `kestrel_sovereign/features/wellness/feature.py::wellness_history` | Caller runtime wellness state; no co-hosted-agent target. |

## Machine-checked core signal source inventory

Every core `SourceRegistration` is an execution boundary even when it is not a
scheduler target and its name contains no agent-shaped word. The contract scans
every constructor under `kestrel_sovereign`, resolves generic factory names
from their string-valued call sites, discovers every `CRON_TASKS` entry, and
includes every literal bespoke-handler mapping supplied to
`build_cron_registrations`. A registration authenticates and constrains a
source; its signal and causation metadata do not grant authority over a peer or
the host. Target-specific authority remains mandatory at the invoked handler.

| Surface ID | Classification |
|---|---|
| `kestrel_sovereign/features/scheduler/feature.py::_handle_backup_snapshot` | Bespoke self-owned backup handler for `cron.backup_snapshot`; no co-hosted-agent target. |
| `kestrel_sovereign/features/scheduler/feature.py::_handle_sleep` | Bespoke caller-agent memory-maintenance handler for `cron.sleep`; no co-hosted-agent target. |
| `kestrel_sovereign/features/scheduler/feature.py::_run_bootstrap_timeout_check` | Bespoke caller-agent bootstrap watchdog for `cron.bootstrap_timeout_check`; no peer authority. |
| `kestrel_sovereign/features/scheduler/feature.py::_run_ecosystem_discovery_watch` | Bespoke caller-agent repository watcher for `cron.ecosystem_discovery_watch`; downstream provider policy remains enforcement. |
| `kestrel_sovereign/features/scheduler/feature.py::_run_github_pr_watch` | Bespoke caller-agent repository watcher for `cron.github_pr_watch`; #3147 binds any resulting wake to the scheduler owner. |
| `kestrel_sovereign/features/scheduler/feature.py::_run_trash_retention` | Bespoke caller-agent retention handler for `cron.trash_retention`; no co-hosted-agent target. |
| `kestrel_sovereign/features/scheduler/feature.py::_run_wait_reconcile` | Bespoke caller-agent wait reconciliation for `cron.wait_reconcile`; resulting wakes remain locally targeted. |
| `kestrel_sovereign/signals/sources/a2a.py::a2a.task_complete` | Bounded peer-completion wake for the local task creator. The signed or authenticated peer event supplies causation, not control authority; cycle detection and rate limits remain mandatory. |
| `kestrel_sovereign/signals/sources/a2a_question_answered.py::a2a.question_answered` | Local resumption wake for a question the caller sent. Durable task correlation binds the waiting session; peer identity and causation metadata grant no control authority. |
| `kestrel_sovereign/signals/sources/a2a_task_submitted.py::a2a.task_submitted` | Bounded inbound peer-work wake. Recipient routing authenticates the target and the signal inherits cycle detection and rate limits; submission grants communication, not lifecycle authority. |
| `kestrel_sovereign/signals/sources/channels.py::channel.message` | External-channel ingress for the locally configured receiver. Channel authentication and target routing remain the boundary; sender or thread metadata grants no agent authority. |
| `kestrel_sovereign/signals/sources/ecosystem_discovery.py::ecosystem.discovery_findings` | Caller-agent repository-watch observation. Provider access policy remains enforcement and discovered repository metadata grants no peer or host authority. |
| `kestrel_sovereign/signals/sources/github_pr_watch.py::github.pr_activity` | Caller-owned GitHub watch wake. #3147 binds the wake to the scheduler owner; repository activity and causation metadata grant no relation authority. |
| `kestrel_sovereign/signals/sources/heartbeat.py::heartbeat` | Self-owned periodic wake registered on the agent's dispatcher. Host timing metadata grants no peer or host authority. |
| `kestrel_sovereign/signals/sources/restart.py::restart.completed` | Requester-bound post-restart wake. The completion event conveys host-coordination state only; it cannot authorize the restart/update operation tracked by #3148. |
| `kestrel_sovereign/signals/sources/scheduler.py::cron.backup_snapshot` | Self-owned backup action; schedule ownership and the local dispatcher bind the target agent. |
| `kestrel_sovereign/signals/sources/scheduler.py::cron.bootstrap_timeout_check` | Self-owned bootstrap watchdog; no peer or host target. |
| `kestrel_sovereign/signals/sources/scheduler.py::cron.ecosystem_discovery_watch` | Self-owned repository watch; its provider and downstream tool policies remain enforcement. |
| `kestrel_sovereign/signals/sources/scheduler.py::cron.github_pr_watch` | Self-owned repository watch; #3147 rejects caller-supplied peer wake identity and binds the owner. |
| `kestrel_sovereign/signals/sources/scheduler.py::cron.memory_consolidate` | Self-owned memory maintenance dispatched to the matching feature tool. |
| `kestrel_sovereign/signals/sources/scheduler.py::cron.morning_signal` | Self-owned strategic-memory artifact dispatched to the matching feature tool. |
| `kestrel_sovereign/signals/sources/scheduler.py::cron.reflect` | Self-owned reflection artifact dispatched to the matching feature tool. |
| `kestrel_sovereign/signals/sources/scheduler.py::cron.restart_coordinator` | Whole-host restart/update target; sovereign/delegated authority is required and the current ordinary-agent admission defect is tracked by #3148. |
| `kestrel_sovereign/signals/sources/scheduler.py::cron.signal_dispatch` | Generic local scheduled-tool dispatch; the selected downstream tool retains target-specific authority checks. |
| `kestrel_sovereign/signals/sources/scheduler.py::cron.sleep` | Self-owned memory-maintenance action with a bespoke local handler. |
| `kestrel_sovereign/signals/sources/scheduler.py::cron.training_cycle` | Caller-agent training/model workflow; shared-provider policy remains independently required. |
| `kestrel_sovereign/signals/sources/scheduler.py::cron.trash_retention` | Self-owned memory-retention action with a bespoke local handler. |
| `kestrel_sovereign/signals/sources/scheduler.py::cron.wait_reconcile` | Self-owned wait reconciliation; resulting signals remain locally targeted. |
| `kestrel_sovereign/signals/sources/system_resumed.py::system.resumed` | Local host-resume maintenance event. It may re-anchor the caller's dispatcher but grants no peer, fleet, or lifecycle authority. |
| `kestrel_sovereign/signals/sources/wait.py::wait.complete` | Caller-owned wait resumption selected through the caller's registered provider. Provider result metadata and causation grant no relation authority. |
| `kestrel_sovereign/signals/sources/wallet.py::webhook.stripe.deposit_complete` | Authenticated external payment event routed to the configured local receiver. Webhook authenticity permits this ingress only and grants no peer or host control. |
| `kestrel_sovereign/signals/sources/workflow_rescue.py::a2a_repair_dispatch` | Provider-neutral workflow stage that records explicitly selected repair targets. Workflow consent and causation are not relation authority; any downstream A2A or lifecycle boundary must independently authorize its exact target. |
| `kestrel_sovereign/signals/sources/workflow_rescue.py::close_resolved_todos` | Evidence-gated workflow bookkeeping for the caller's todo namespace; no co-hosted-agent lifecycle authority follows from the pipeline. |
| `kestrel_sovereign/signals/sources/workflow_rescue.py::evidence_verify` | Read-only workflow evidence stage. Observed state grants no mutation or agent authority. |
| `kestrel_sovereign/signals/sources/workflow_rescue.py::fleet_stalled_sweep` | Read-only fleet-work observation with optional provider-bound discovery. Visibility grants no authority to repair, stop, or otherwise control an observed agent. |
| `kestrel_sovereign/signals/sources/workflow_rescue.py::governance_review` | Records intervention intent but explicitly does not authorize it. A later target-specific authority boundary remains mandatory. |
| `kestrel_sovereign/signals/sources/workflow_rescue.py::reopen_resolved_todos` | Compensation bookkeeping for the caller's todo workflow; no co-hosted-agent lifecycle authority follows from the pipeline. |

## Machine-checked dynamic router boundary inventory

Runtime-installed agent and host features can return routers whose decorators
live outside this checkout. The contract therefore inventories every
function-scoped `include_router` publication call in core. These generic doors
do not grant relation authority: the contributed route must still bind a
trusted principal and enforce its target-specific policy, and each out-of-tree
provider remains responsible for its exact route inventory.

| Surface ID | Classification |
|---|---|
| `kestrel_sovereign/host_features/runtime.py::mount_host_feature_routers.include_router[0]` | Generic host-feature router publication boundary. Host authentication and provider-owned target policy remain mandatory; mounting conveys no agent relation authority. |
| `kestrel_sovereign/server.py::_mount_feature_routers._collect_routers_from_agent.include_router[0]` | Generic agent-feature router publication boundary. Core adds the feature-state gate, while the contributed handler retains authentication and target-authority responsibility. |
| `kestrel_sovereign/server.py::_mount_feature_routers.include_router[0]` | Generic shared webhook-router publication boundary. Its live receiver selection and each receiver's configured source policy remain enforcement; mounting conveys no hierarchy authority. |

## Machine-checked built-in command inventory

Built-in commands do not carry feature `@tool` decorators. The contract test
therefore discovers every entry directly from `BUILTIN_COMMAND_SPECS`, including
apparently local commands. A neutrally named host-control door cannot hide
behind either the feature-tool inventory or a command-name heuristic.

| Surface ID | Classification |
|---|---|
| `kestrel_sovereign/command_handler.py::!status` | Caller runtime status read; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!help` | Static command-catalog read; no agent target. |
| `kestrel_sovereign/command_handler.py::!reload-context` | Caller bootstrap/context reload; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!heartbeat` | Caller heartbeat trigger; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!verify-constitution` | Caller constitutional-integrity read; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!reanchor-constitution` | Sovereign-gated mutation of the caller's constitutional anchor; no peer grant. |
| `kestrel_sovereign/command_handler.py::!safe-mode` | Sovereign-gated transition of the caller's safe-mode state; no peer grant. |
| `kestrel_sovereign/command_handler.py::!privacy` | Caller privacy-mode state/session control; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!set-privacy-mode` | Caller privacy-mode transition; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!confirm-privacy-mode` | Caller confirmation of its pending privacy-mode transition; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!get-privacy-mode` | Caller privacy-mode read; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!privacy-status` | Caller privacy-state read; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!privacy-save` | Caller isolated-session save; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!privacy-discard` | Caller isolated-session discard; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!backup` | Caller backup creation; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!promote-backup` | Caller isolated-session promotion and backup; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!sleep` | Caller memory consolidation and sovereignty export; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!consolidate` | Caller memory consolidation; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!compact` | Caller session-context compaction; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!create-agent` | Sovereign/delegated host identity provisioning; #3149. |
| `kestrel_sovereign/command_handler.py::!anchor` | Caller memory-state anchor; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!set-app-context` | Caller active-session app context; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!legacy-echo` | Caller legacy app-context echo path; no co-hosted-agent target. |
| `kestrel_sovereign/command_handler.py::!tasks` | Unscoped shared-store task listing; defect [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| `kestrel_sovereign/command_handler.py::!continue` | Resumes only the caller's stopped request; not peer Stop or mandate-only Hold. |

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

This inventory includes canonical HTTP routes, programmatic route/mount
registrations, FastAPI-generated documentation routes, and the live #871 `/agent/*`
compatibility spellings synthesized from every `/api/agent/*` declaration.
The compatibility middleware rewrites those deprecated entry doors before
downstream authentication and FastAPI dispatch, so they carry the same
authority classification as the canonical handler and cannot be omitted merely
because no second decorator declares them. In-tree `websocket` and
`websocket_route` declarations are represented as `WEBSOCKET`; the exact-set
contract therefore also fails on a new live WebSocket until it is classified.

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
| `kestrel_sovereign/endpoints/agent.py::POST /agent/invoke` | Deprecated compatibility ingress to the request-routed agent; same host-authenticated classification as `/api/agent/invoke`. |
| `kestrel_sovereign/endpoints/agent.py::POST /agent/stream` | Deprecated compatibility ingress to the request-routed agent; same host-authenticated classification as `/api/agent/stream`. |
| `kestrel_sovereign/endpoints/agent.py::GET /agent/context-status` | Deprecated compatibility status read from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::GET /agent/health/status` | Deprecated compatibility health read from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::GET /agent/heartbeat/status` | Deprecated compatibility heartbeat-state read from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::GET /agent/info` | Deprecated compatibility identity/capability read from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::GET /agent/notifications` | Deprecated compatibility notification read from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::GET /agent/notifications/sse` | Deprecated compatibility notification stream from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::GET /agent/privacy-mode` | Deprecated compatibility privacy-state read from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::GET /agent/reflection/status` | Deprecated compatibility reflection-state read from the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::POST /agent/attachments` | Deprecated compatibility attachment mutation on the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::POST /agent/health/trigger` | Deprecated compatibility health check on the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::POST /agent/heartbeat/trigger` | Deprecated compatibility LLM heartbeat invocation of the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::POST /agent/privacy-mode` | Deprecated compatibility privacy transition on the request-routed agent. |
| `kestrel_sovereign/endpoints/agent.py::POST /agent/privacy-mode/cancel` | Deprecated compatibility cancellation of the request-routed agent's pending privacy transition. |
| `kestrel_sovereign/endpoints/agent.py::POST /agent/privacy-mode/confirm` | Deprecated compatibility confirmation of the request-routed agent's pending privacy transition. |
| `kestrel_sovereign/endpoints/agent.py::GET /agent/tasks` | Deprecated compatibility shared-store inbox read; defect [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| `kestrel_sovereign/endpoints/agent.py::GET /agent/tasks/{task_id}` | Deprecated compatibility unscoped task-ID read; defect [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| `kestrel_sovereign/endpoints/agent.py::GET /agent/tasks/{task_id}/subscribe` | Deprecated compatibility unscoped task-ID subscription; defect [#3145](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3145). |
| `kestrel_sovereign/endpoints/agent.py::POST /agent/tasks/{task_id:path}/cancel` | Deprecated compatibility creator/recipient-owned mutation with the same #3134 predicate as the canonical handler. |
| `kestrel_sovereign/endpoints/agent.py::POST /agent/stop` | Deprecated compatibility current-agent/self Stop. |
| `kestrel_sovereign/endpoints/agent.py::POST /agent/tasks/send` | Deprecated compatibility scoped, authenticated A2A delivery. |
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /auth/login` | Self-authenticating OAuth entrypoint governed by the configured human identity policy; no agent relation authority. |
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /auth/callback` | Self-authenticating OAuth callback that mints a host-wide browser session after allowlist validation. |
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /auth/logout` | Self-authenticating host session clear; no selected-agent state is involved. |
| `kestrel_sovereign/endpoints/auth_oauth.py::POST /auth/token` | Rate-limited, allowlist/password-gated host JWT issuance; no agent relation authority. |
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /auth/me` | Host-authenticated credential inspection with additional browser-session semantics. |
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /auth/verify` | Host-authenticated bearer verification; no selected-agent state is involved. |
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
| `kestrel_sovereign/endpoints/files.py::GET /agent/channels/{channel_type}/link-qr.png` | Deprecated compatibility channel-link QR read from the request-routed agent. |
| `kestrel_sovereign/endpoints/github.py::GET /api/github/repos` | Host-authenticated access through the process-wide GitHub credential and repository allowlist; no agent principal is bound. |
| `kestrel_sovereign/endpoints/github.py::GET /api/github/{path:path}` | Host-authenticated, repository-scoped proxy through the process-wide GitHub credential; no agent principal is bound. |
| `kestrel_sovereign/endpoints/metrics.py::GET /metrics` | Explicitly public, process-wide Prometheus telemetry; observation grants no control authority. |
| `kestrel_sovereign/endpoints/models.py::DELETE /api/keys/user/{provider}` | Authenticated-user-scoped BYOK mutation keyed by `request.state.user_id`; the routed agent supplies PostgreSQL connectivity only. |
| `kestrel_sovereign/endpoints/models.py::GET /api/keys/platform` | Host-authenticated read of the platform-global key catalog; the routed agent supplies PostgreSQL connectivity only. |
| `kestrel_sovereign/endpoints/models.py::GET /api/keys/user` | Authenticated-user-scoped BYOK read keyed by `request.state.user_id`; the routed agent supplies PostgreSQL connectivity only. |
| `kestrel_sovereign/endpoints/models.py::POST /api/keys/user` | Authenticated-user-scoped BYOK mutation keyed by `request.state.user_id`; the routed agent supplies PostgreSQL connectivity only. |
| `kestrel_sovereign/endpoints/models.py::POST /api/keys/user/verify` | Authenticated-user-scoped passphrase verification keyed by `request.state.user_id`; the routed agent supplies PostgreSQL connectivity only. |
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
| `kestrel_sovereign/host_features/runtime.py::MOUNT <dynamic:mount_path>` | Generic host-feature static-asset mount boundary. Its concrete host path is runtime feature metadata; the expression marker prevents the programmatic registration from escaping the exact inventory. |
| `kestrel_sovereign/server.py::GET /api/auth/key` | Public-localhost, rate-limited host API-key bootstrap; reconciled with the checked-in auth-surface ledger. |
| `kestrel_sovereign/server.py::GET /api/host/ui/contributions` | Host-authenticated read of the shared UI manifest. |
| `kestrel_sovereign/server.py::GET /api/host/csrf` | Host-authenticated issuance of a double-submit CSRF token; the token is not standalone authority. |
| `kestrel_sovereign/server.py::POST /api/host/phoenix/session` | Host-authenticated minting of a short-lived, path-scoped Phoenix embed cookie. |
| `kestrel_sovereign/server.py::GET /docs` | FastAPI-generated host API documentation; publication grants no agent relation authority. |
| `kestrel_sovereign/server.py::GET /docs/oauth2-redirect` | FastAPI-generated documentation OAuth redirect; publication grants no agent relation authority. |
| `kestrel_sovereign/server.py::HEAD /docs` | FastAPI-generated host API documentation metadata; publication grants no agent relation authority. |
| `kestrel_sovereign/server.py::HEAD /docs/oauth2-redirect` | FastAPI-generated documentation OAuth redirect metadata; publication grants no agent relation authority. |
| `kestrel_sovereign/server.py::GET /health` | Intentionally public aggregate readiness; no agent names or control authority. |
| `kestrel_sovereign/server.py::GET /health/detailed` | Host-authenticated fleet diagnostics, including named per-agent health. |
| `kestrel_sovereign/server.py::GET /openapi.json` | FastAPI-generated host API schema; publication grants no agent relation authority. |
| `kestrel_sovereign/server.py::HEAD /openapi.json` | FastAPI-generated host API schema metadata; publication grants no agent relation authority. |
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
| `kestrel_sovereign/server.py::GET /redoc` | FastAPI-generated host ReDoc UI; publication grants no agent relation authority. |
| `kestrel_sovereign/server.py::HEAD /redoc` | FastAPI-generated host ReDoc metadata; publication grants no agent relation authority. |
| `kestrel_sovereign/server.py::MOUNT /js` | Programmatic mount of host UI assets; no agent relation authority. |
| `kestrel_sovereign/server.py::MOUNT /shared` | Programmatic mount of host UI assets; no agent relation authority. |
| `kestrel_sovereign/server.py::MOUNT /static` | Programmatic mount of host UI assets; no agent relation authority. |
| `kestrel_sovereign/server.py::MOUNT /utils` | Programmatic mount of host UI assets; no agent relation authority. |
| `kestrel_sovereign/server.py::MOUNT <dynamic:mount_path>` | Generic agent-feature static-asset mount boundary. Its concrete path is runtime feature metadata; the selected mount grants no agent relation authority. |

## Machine-checked request-routed alias inventory

The host routing middleware accepts
`/api/agents/{selected_agent_name}/{remaining_path}` and rewrites the remainder before
FastAPI dispatch. The table therefore spells the synthesized alias for every
declared core HTTP/WebSocket route and programmatic mount, including handlers that do not consume `Request`
and routes whose canonical spelling contains no agent-shaped word. The
canonical root is excluded because the router regex requires a non-empty
remainder. Authentication sees the
prefixed path before routing: every alias below requires host authentication
except the deliberately self-authenticating webhook family and the
agent-feature static-asset aliases. The latter are narrowly matched by
`FEATURE_STATIC_ASSET_RE` so browser module and stylesheet loads can omit an
API-key header; they publish static content and grant no agent authority. The
repeated agent-local classifications are intentional false-positive
dispositions; an exact-set contract makes any new request-bound route fail
until it is added.

Classification codes: **A** = sovereign/operator-authenticated,
target-agent-local read or mutation, with routing selecting the target but
granting no hierarchy authority;
**H** = host/fleet handler whose explicit operator policy ignores agent
selection as authority; **U** = authenticated-user namespace whose
`request.state.user_id` principal ignores agent selection; **W** = configured webhook ingress policy, whose auth
and rate limits apply only when configured (explicit open/unlimited modes are
not described as enforced); **S** = deliberately unauthenticated static-asset
publication whose selected-agent prefix grants no relation authority; **D-n** =
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
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /api/agents/{selected_agent_name}/auth/callback` | H — host OAuth callback; the selected-agent prefix grants no target-local or relation authority. |
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /api/agents/{selected_agent_name}/auth/login` | H — host OAuth login; the selected-agent prefix grants no target-local or relation authority. |
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /api/agents/{selected_agent_name}/auth/logout` | H — host session clear; the selected-agent prefix grants no target-local or relation authority. |
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /api/agents/{selected_agent_name}/auth/me` | H — host credential inspection; the selected-agent prefix grants no target-local or relation authority. |
| `kestrel_sovereign/endpoints/auth_oauth.py::GET /api/agents/{selected_agent_name}/auth/verify` | H — host bearer verification; the selected-agent prefix grants no target-local or relation authority. |
| `kestrel_sovereign/endpoints/auth_oauth.py::POST /api/agents/{selected_agent_name}/auth/token` | H — host JWT issuance; the selected-agent prefix grants no target-local or relation authority. |
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
| `kestrel_sovereign/endpoints/github.py::GET /api/agents/{selected_agent_name}/api/github/repos` | H — process-wide GitHub credential/configuration; the selected-agent prefix grants no target-local authority. |
| `kestrel_sovereign/endpoints/github.py::GET /api/agents/{selected_agent_name}/api/github/{path:path}` | H — process-wide GitHub credential/configuration; the selected-agent prefix grants no target-local authority. |
| `kestrel_sovereign/endpoints/memories.py::DELETE /api/agents/{selected_agent_name}/api/memories/{node_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/memories.py::GET /api/agents/{selected_agent_name}/api/identity-chain` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/memories.py::GET /api/agents/{selected_agent_name}/api/memories` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/memories.py::GET /api/agents/{selected_agent_name}/api/memories/{node_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/metrics.py::GET /api/agents/{selected_agent_name}/metrics` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/endpoints/models.py::DELETE /api/agents/{selected_agent_name}/api/agents/{agent_name}` | H — host/fleet discovery or lifecycle; #3149 policy enforces mutations. |
| `kestrel_sovereign/endpoints/models.py::DELETE /api/agents/{selected_agent_name}/api/keys/user/{provider}` | U — authenticated-user BYOK principal from `request.state.user_id`; agent selection only supplies PostgreSQL connectivity. |
| `kestrel_sovereign/endpoints/models.py::DELETE /api/agents/{selected_agent_name}/api/keys/{provider}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/agents` | H — host/fleet discovery or lifecycle; #3149 policy enforces mutations. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/constitution` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/embedding/models` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/embedding/reindex/{job_id}` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/embedding/settings` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/identity` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/ipfs/status` | D-3226 — shared IPFS daemon health and recursive pins lack agent ownership filtering or sovereign host-only enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/keys` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/keys/available-sources` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/keys/platform` | H — platform-global key catalog; agent selection only supplies PostgreSQL connectivity. |
| `kestrel_sovereign/endpoints/models.py::GET /api/agents/{selected_agent_name}/api/keys/user` | U — authenticated-user BYOK principal from `request.state.user_id`; agent selection only supplies PostgreSQL connectivity. |
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
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/api/keys/user` | U — authenticated-user BYOK principal from `request.state.user_id`; agent selection only supplies PostgreSQL connectivity. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/api/keys/user/verify` | U — authenticated-user BYOK principal from `request.state.user_id`; agent selection only supplies PostgreSQL connectivity. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/api/model/set` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::POST /api/agents/{selected_agent_name}/v1/chat/completions` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::PUT /api/agents/{selected_agent_name}/api/embedding/route-model` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/models.py::PUT /api/agents/{selected_agent_name}/api/embedding/settings` | A — target-local policy remains enforcement. |
| `kestrel_sovereign/endpoints/observability.py::GET /api/agents/{selected_agent_name}/api/observability/metrics/{metric_name}` | D-3215 — shared PostgreSQL read lacks a trusted selected-agent predicate. |
| `kestrel_sovereign/endpoints/observability.py::GET /api/agents/{selected_agent_name}/api/observability/summary` | D-3215 — shared PostgreSQL read lacks a trusted selected-agent predicate. |
| `kestrel_sovereign/endpoints/rasa_shim.py::POST /api/agents/{selected_agent_name}/webhooks/rest/webhook` | D-3220 — the prefix selects an agent in request state, but the handler ignores it and invokes the host-default agent (or fails when none exists); source auth/rate limit do not repair target binding. |
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
| `kestrel_sovereign/endpoints/sovereignty.py::GET /api/agents/{selected_agent_name}/api/sovereignty/files` | D-3225 — shared host export-cache listing lacks routed-agent ownership or sovereign host-only enforcement. |
| `kestrel_sovereign/endpoints/sovereignty.py::GET /api/agents/{selected_agent_name}/api/sovereignty/files/{filename}` | D-3225 — shared host export-cache download lacks routed-agent ownership or sovereign host-only enforcement. |
| `kestrel_sovereign/endpoints/sovereignty.py::GET /api/agents/{selected_agent_name}/api/sovereignty/files/{filename}/preview` | D-3225 — shared host export-cache preview lacks routed-agent ownership or sovereign host-only enforcement. |
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
| `kestrel_sovereign/features/webhooks/receiver.py::POST /api/agents/{selected_agent_name}/webhooks/{webhook_name}` | W — target-selected configured webhook policy; authentication and rate limiting apply only when configured, while explicit open/unlimited mode grants no hierarchy authority. |
| `kestrel_sovereign/host_features/runtime.py::MOUNT /api/agents/{selected_agent_name}/<dynamic:mount_path>` | H — runtime-computed host-feature asset mount; selected-agent context grants no target-local or relation authority. |
| `kestrel_sovereign/server.py::DELETE /api/agents/{selected_agent_name}/phoenix` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::DELETE /api/agents/{selected_agent_name}/phoenix/{path:path}` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/api/auth/key` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/api/host/csrf` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/api/host/ui/contributions` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/assets/{path:path}` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/docs` | H — FastAPI-generated host documentation; selected-agent context grants no target-local or relation authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/docs/oauth2-redirect` | H — FastAPI-generated documentation redirect; selected-agent context grants no target-local or relation authority. |
| `kestrel_sovereign/server.py::HEAD /api/agents/{selected_agent_name}/docs` | H — FastAPI-generated host documentation metadata; selected-agent context grants no target-local or relation authority. |
| `kestrel_sovereign/server.py::HEAD /api/agents/{selected_agent_name}/docs/oauth2-redirect` | H — FastAPI-generated documentation redirect metadata; selected-agent context grants no target-local or relation authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/health` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/health/detailed` | H — explicit operator/public policy; selected-agent context is not authority. |
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/openapi.json` | H — FastAPI-generated host API schema; selected-agent context grants no target-local or relation authority. |
| `kestrel_sovereign/server.py::HEAD /api/agents/{selected_agent_name}/openapi.json` | H — FastAPI-generated host API schema metadata; selected-agent context grants no target-local or relation authority. |
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
| `kestrel_sovereign/server.py::GET /api/agents/{selected_agent_name}/redoc` | H — FastAPI-generated host ReDoc UI; selected-agent context grants no target-local or relation authority. |
| `kestrel_sovereign/server.py::HEAD /api/agents/{selected_agent_name}/redoc` | H — FastAPI-generated host ReDoc metadata; selected-agent context grants no target-local or relation authority. |
| `kestrel_sovereign/server.py::MOUNT /api/agents/{selected_agent_name}/js` | H — mounted host UI assets; selected-agent context grants no target-local or relation authority. |
| `kestrel_sovereign/server.py::MOUNT /api/agents/{selected_agent_name}/shared` | H — mounted host UI assets; selected-agent context grants no target-local or relation authority. |
| `kestrel_sovereign/server.py::MOUNT /api/agents/{selected_agent_name}/static` | H — mounted host UI assets; selected-agent context grants no target-local or relation authority. |
| `kestrel_sovereign/server.py::MOUNT /api/agents/{selected_agent_name}/utils` | H — mounted host UI assets; selected-agent context grants no target-local or relation authority. |
| `kestrel_sovereign/server.py::MOUNT /api/agents/{selected_agent_name}/<dynamic:mount_path>` | S — runtime-computed, host-authentication-exempt agent-feature asset mount; selected-agent context grants no relation authority. |

## Review rule

Adding a cross-agent surface requires updating this matrix in the same change.
The implementation must bind the trusted caller identity before the operation,
authorize and mutate atomically where storage is shared, return the same public
shape for missing and unauthorized targets, and include a mutation test that
fails if the authority predicate or its wiring is removed.
