---
type: Research Note
title: 'Terminal settlement owners, step 0: which function writes the terminal state today'
description: Step 0 of issue 4108. For the four kinds of accepted input (A2A task,
  queued wake, wait completion, scheduler claim), the write that accepts each one and,
  for each way the carrying turn can end, the function that writes its terminal state
  on this commit, or "none found". Readings only, with no owners proposed and no fixes.
resource: /docs/investigations/4108-terminal-settlement-owners.md
tags:
- docs
- investigations
- research-note
- signals
- a2a
- scheduler
- waits
timestamp: '2026-11-15T00:00:00Z'
status: snapshot
owner: architecture
canonical: false
generated: false
privacy: internal
---

# Terminal settlement owners, step 0: which function writes the terminal state today

**Issue:** #4108, step 0 only (work order wo-4108-1; architect note rv-4108-0).
**Date:** 2026-11-15.

This note records what the code does today. It does not propose an owner for any
input and contains no fixes or recommendations. "Who the owner should be" is the
ruling's question. This note answers "who it is today".

How it was produced:

- Every cell comes from reading source code at the commit in section 1.
- No test, script or agent was run to produce a cell. `scripts/docs_verify.py`
  and its unit tests were run only to regenerate and check the doc indexes in
  `docs/generated/`.
- No database, configuration file or log was read or written.
- Six issue bodies (#3775, #4014, #4026, #4036, #4041, #4073) and the state of PR
  #4029 were read with `gh`, read-only. They are cited only where the code does
  not settle a point.

Each cell is marked:

- **measured (read)**: the terminal write, and every call from the way the turn
  ended to that write, was read in the cited functions at this commit. Here
  "measured" means read, not reproduced.
- **inferred**: at least one link was reasoned rather than read, or depends on
  runtime ordering that reading cannot settle. The unread link is named.
- **reported privately**: the settlement is a permission decision or a privacy
  obligation. Only the file and function are given (section 6).

Line numbers are at the commit below and are given so the reading can be repeated.

## 1. Commit and searches

**Read at:** `a256260f430d14904906c5140a46e2fc1c73b8df` (`origin/main` and this
branch's `HEAD` when the read began). The SDK in `uv.lock` is
`kestrel-sovereign-sdk` 0.54.0. One SDK fact was read from its installed copy:
`kestrel_sdk.signals.models.Status.COALESCED` has the value `"coalesced"`.

Searches, run from the repository root (`rg` is ripgrep):

| Id | Command | Used for |
|---|---|---|
| S1 | `rg -n '_announce_submitted_input\|on_task_submitted\|on_task_complete\|_deliver_a2a_wake_until_durable\|def _a2a_wake\|a2a\.task_submitted' kestrel_sovereign` | A2A wake producers |
| S2 | `rg -n 'apply_undelivered_wake_transition\|settle_terminal_delivery\|undelivered_cognition_wake_settlement\|_request_owed_settlement_pass\|unsettled_cognition_wakes\|settle_undelivered\|list_unsettled_terminal_deliveries' kestrel_sovereign` | A2A wake-settlement code from the unmerged #4001 branch (PR #4029, state CLOSED). **No match.** |
| S3 | `rg -n 'wake_obligation\|a2a_wake_obligations' kestrel_sovereign` | #4041 obligation table. **No match.** |
| S4 | `rg -n '\.fail_task\(\|\.complete_task\(\|task_manager\.update_status\(\|manager\.update_status\(\|\.cancel_task\(\|save_recipient_terminal_outcome\(\|cancel_if_authorized\(\|save_recipient_lifecycle\(' kestrel_sovereign` | Every caller of an A2A task terminal writer |
| S5 | `rg -n 'task_manager\.\|task_store\.' kestrel_sovereign/signals/dispatcher.py` | Task writes from the dispatcher. **No match.** |
| S6 | `rg -n 'task_manager\.[a-z_]+\(' -o kestrel_sovereign/agent/event_manager.py` | Task calls from the wake producer. Reads only: `get_task_cancellation_snapshot`, `get_task_snapshot`, `list_cognition_wake_candidates`, `questions_for_actor` |
| S7 | `rg -l 'TASK_STATE_FAILED\|TASK_STATE_CANCELED\|TASK_STATE_REJECTED' kestrel_sovereign` | Files that construct a terminal A2A state |
| S8 | `rg -n '_StoppedCognitionResult\(\|_ReconciliationCognitionResult\(\|_hold_durable_cognition_stop_completion\|stop_terminalized' kestrel_sovereign`, then `rg -n '_StoppedCognitionResult\|_ReconciliationCognitionResult' kestrel_sovereign/signals/dispatcher.py` | Stop and journal-fence results |
| S9 | `rg -n 'stop_acknowledged\|invocation_self_fenced' kestrel_sovereign` | Readers of the Stop and self-fence markers. Only the two writers in `dispatcher.py` |
| S10 | `rg -n 'raise InvocationCancelledError\|InvocationCancelledError\(' kestrel_sovereign` | Where a turn sees Stop |
| S11 | `rg -n 'max_attempts' kestrel_sovereign --glob '!**/durable.py'` | Retry caps per consumer |
| S12 | `rg -n 'def _track_runtime_task\|def _track_background_task\|_runtime_publication_ready' kestrel_sovereign` | The READY barrier |
| S13 | `rg -n 'persist_signal\(' kestrel_sovereign` | Queued-wake acceptance |
| S14 | `rg -n 'async def fail_cognition_delivery' kestrel_sovereign` | PostgreSQL wake terminalizer |
| S15 | `git log -S '_own_cognition_claim_body' --format='%h %ad %s' --date=short -- kestrel_sovereign/signals/dispatcher.py` | When the claim-owner guard first appears: `cb0fc9151 2026-10-23 fix: preserve original native signal outcomes through revocation (#3798)` |
| S16 | `rg -n -i 'stopp\|InvocationCancelled\|cancel' kestrel_sovereign/waits/reconciler.py` | Stop handling in the wait reconciler. One comment line, no code |
| S17 | `rg -n 'dispatch_signal\(\|enqueue_signal\(\|enqueue_durable\|Status\.COALESCED\|status\.value\|_StoppedCognition\|stop_acknowledged\|def execute_scheduled\|async def _execute_cron\|async def _run_scheduled' kestrel_sovereign/features/scheduler/feature.py` | How the scheduler dispatches a cron task |
| S18 | `rg -n 'MAX_\|max_attempt\|attempt_count >=\|unresolved_effect\|_recover\|lease_expired\|recovery' kestrel_sovereign/features/scheduler/runner.py` | Scheduler recovery |
| S19 | `rg -n 'attempt_count\s*(>=\|>\|<)\|max_attempts\|MAX_ATTEMPTS\|MAX_CLAIM\|retry_cap' kestrel_sovereign/features/scheduler` | Scheduler retry cap. **No match.** |
| S20 | `rg -n '_runtime_publication_ready\|_track_runtime_task\|ready\.wait\|SchedulerDispatchNotReady\(' kestrel_sovereign/features/scheduler`, then `rg -n 'runtime_task_factory\|occurrence_task_factory' kestrel_sovereign/features/scheduler/runner.py` | READY barrier on the scheduler path |
| S21 | `rg -n 'SchedulerDispatchNotReady' kestrel_sovereign` | Scheduler's feature-load deferral |
| S22 | `rg -n '#3999\|single-use operation\|operation fence\|admit_request_operation\(' kestrel_sovereign` | The restart fence #4023 names. **Not located** |
| S23 | `rg -n 'def cancel_current_request\|def stop_agent_work\|def _stop_agent_scope\|async def stop_all' kestrel_sovereign` | Entry of agent-scope Stop |
| S24 | `rg -n 'EPHEMERAL\|ISOLATED\|DEIDENTIFIED\|volatile' kestrel_sovereign/features/scheduler/feature.py`, `rg -n 'def _[a-z_]*(privacy\|volatile)[a-z_]*\|def _refuse[a-z_]*\|PAUSE_CAUSE\|pause_schedule=True\|HOOK_FAILED_STATUS =' kestrel_sovereign/features/scheduler` | Location of privacy and permission settlements (section 6) |
| S25 | `rg -n '^    async def \|^    def [a-z_]+\(\|^class \|^async def \|^def ' <file>` for `signals/dispatcher.py`, `signals/durable.py`, `a2a/task_manager.py`, `a2a/stores/unified/task_store.py`, `a2a/request_handler.py`, `waits/reconciler.py`, `storage/async_wait_signal_store.py`, `features/scheduler/runner.py` | Function maps of the files read |
| S26 | `rg -n 'task_store\.\|task_manager\.\|save_recipient\|cancel_if_authorized\|update_status\(\|UPDATE a2a_tasks\|a2a_tasks' <file>` for the S7 files not otherwise opened: `a2a/native_client.py`, `a2a/migrate_legacy.py`, `a2a/lifecycle.py`, `features/strategic_memory/run_history.py`, `features/base.py` | Recipient task writes in those files. **No match**, apart from two mentions of the tool name `list_outbound_a2a_tasks` in `run_history.py` |
| S27 | `rg -n '\._tick\(\|_admit_occurrence\(\|_run_occurrence\(' kestrel_sovereign` | Callers of the scheduler occurrence path. One each: `_loop` calls `_tick` (`runner.py:2235`), `_tick` calls `_admit_occurrence` (:2292), `_admit_occurrence` starts `_run_occurrence` (:2300). The `heartbeat.py` matches are a different `_tick` |
| S28 | `rg -n '_start_host_scheduler\|SchedulerRunner\(\|AgentManagerHostedSchedulerExecutor\(\|runtime_task_factory=\|occurrence_task_factory=\|_uses_shared_postgres_scheduler\(\) ->' kestrel_sovereign/server.py kestrel_sovereign/features/scheduler` | The two scheduler runners and their task factories. `runtime_task_factory=` and `occurrence_task_factory=` match only in `feature.py`, **not in `kestrel_sovereign/server.py`** |
| S29 | `rg -n 'prepare_scheduled\|_mark_effect_started\(\|_terminalize_inconsistent_execution_log\(' kestrel_sovereign/features/scheduler` | Where the effect-start mark is written (`runner.py:3967`, :4063, :4206) and read back (:3735) |
| S30 | `rg -n '_runtime_publication_ready' kestrel_sovereign/kestrel_agent.py`, `rg -n 'async def load_agent\|async def _initialize_agent\|await agent\.initialize\(' kestrel_sovereign/multi_agent/agent_manager.py`, `rg -n 'set_state\(\|BootPhaseState\.READY\|async def run_boot_sequence' kestrel_sovereign/agent/boot.py` | Where a cold-loaded agent becomes READY |
| S31 | `rg -n 'async def stop\(\|occurrence\.task for\|async def _shutdown_host_scheduler\|"host_scheduler_runner", runner, "stop"' kestrel_sovereign/features/scheduler/runner.py kestrel_sovereign/server.py` | What cancels an admitted occurrence of the host-owned runner |

## 2. When each input is accepted

| Mechanism | What it is in the code | The write that accepts it | Mark |
|---|---|---|---|
| A2A task | An inbound native task row (`a2a_tasks`) that a peer created with SendMessage. A cognition turn woken by an `a2a.task_submitted` wake carries it. | `TaskManager.create_task` (`kestrel_sovereign/a2a/task_manager.py:1678`), reached from `NativeA2ARequestHandler._submit` (`kestrel_sovereign/a2a/request_handler.py:373`, call at :430). Inside one `_task_input_admission` transaction it calls `TaskStore.create` (:1748) and `TaskStore.record_initial_input` (:1755) in `kestrel_sovereign/a2a/stores/unified/task_store.py`. It then registers `_announce_submitted_input` (:1771), which a code comment (:1775-1782) says fires on commit. A continuation goes through `TaskManager.continue_task` (:1628). | measured |
| Queued wake | A durable cognition delivery row (`durable_signal_deliveries`) for one of the three A2A consumers in `a2a_cognition_consumers()` (`kestrel_sovereign/signals/sources/__init__.py:11`): `a2a.question_answered`, `a2a.task_complete` and `a2a.task_submitted`. All three are registered with `max_attempts=0` in `KestrelAgent._boot_phase_a2a_observability_signals` (`kestrel_sovereign/kestrel_agent.py:3511-3519`). | `DurableSignalStore.persist_signal` (`kestrel_sovereign/signals/durable.py:3625`) commits the event and its matching delivery rows together. It is called from `SignalDispatcher._run` (`kestrel_sovereign/signals/dispatcher.py:4938` and :5022) when `SignalDispatcher.enqueue_durable_cognition` (:1482) runs. For a task wake the caller is `EventManagerMixin._deliver_a2a_wake_until_durable` (`kestrel_sovereign/agent/event_manager.py:223`, call at :332). That is a separate, later transaction from the task insert, retried in-process ("The task row committed before this coroutine was scheduled", :235). `EventManagerMixin.reconcile_a2a_cognition_wakes` (:712) re-runs it at boot. | measured |
| Wait completion | A terminal transition of a handle that a provider reports, for an explicit watch or a monitored provider. | The wait itself: `WaitSignalStore.start_watch` (`kestrel_sovereign/storage/async_wait_signal_store.py:754`), called from `register_wait_watch` (`kestrel_sovereign/waits/reconciler.py:1601`, call at :1646). Implicit providers (`active_handles`) have no registration row. The completion: `WaitReconciler._process_handle` (:952) calls `WaitSignalStore.record_pending` (:1146; store :325) for the transition's token, then `SignalDispatcher.enqueue_signal` (:1155), which is the non-durable route. | measured |
| Scheduler claim | A due occurrence of a `scheduled_tasks` row. A COGNITION cron task (`self_followup`) is carried by a turn: `SchedulerFeature._dispatch_scheduled_task` (`kestrel_sovereign/features/scheduler/feature.py:919`) awaits `SignalDispatcher.dispatch_signal` (:1150). ACTION and ARTIFACT tasks run a handler, with no turn. Two runners claim occurrences (S28). The agent-owned runner is built by `SchedulerFeature` (`feature.py:363`). The host-owned runner is built by `_start_host_scheduler` (`kestrel_sovereign/server.py:2856`, runner at :2929) with `AgentManagerHostedSchedulerExecutor` (`runner.py:1013`) when `_uses_shared_postgres_scheduler` (`kestrel_sovereign/server.py:2649`) is true, that is, `KESTREL_DB_BACKEND=postgres` and `KESTREL_DATABASE_URL` set. On such a host `SchedulerFeature` builds no runner (`feature.py:352-357`). The host-owned runner dispatches through the same `SchedulerFeature._dispatch_scheduled_task` (`HostedSchedulerExecutor._dispatch_resolved_agent`, `runner.py:967-977`). | Both runners: `SchedulerRunner._claim` (`kestrel_sovereign/features/scheduler/runner.py:3485`) does a conditional `UPDATE scheduled_tasks` that sets the claim token and lease (:3595) and an `INSERT INTO task_execution_log` with status `claimed` (:3683), in one transaction. | measured |

Tasks created by `TaskManager.execute_skill` (local skill execution, with
`notify_submitted=False` and `input_wake_eligible=False`, :816-822) are carried by a
handler, not a turn. They are not in the tables (section 5).

## 3. The ways a turn ends, as the code identifies them

| Way | How the code identifies it |
|---|---|
| completes | The route returns `Status.OK`. |
| fails terminally | (i) `process_input` raises an ordinary exception: `SignalDispatcher._dispatch_cognition` returns `FAILED` (`dispatcher.py:7340-7373`). (ii) The source withdraws the wake, or validation or cycle detection drops it: `DROPPED_VALIDATION` / `DROPPED_CYCLE`. |
| is stopped | `InvocationCancelledError` reaches `_dispatch_cognition`, which returns a `_StoppedCognitionResult` with status `COALESCED` and error `stop_acknowledged: …` (:7297-7311). The raise sites (S10) are `kestrel_sovereign/agent/invocation.py:593`, :957 and `TurnLifecycleMixin._active_turn_scope` (`kestrel_sovereign/agent/turn_lifecycle.py:996`). Agent-scope Stop enters at `RequestLifecycleMixin.cancel_current_request` (`kestrel_sovereign/agent/request_lifecycle.py:570`). The path from there to the raise was **not read**, so every Stop cell is inferred on that link. |
| is fenced | The code uses "fence" for two different things. (a) **Owner self-fence:** `InvocationSelfFencedError` gives `FAILED` with error `invocation_self_fenced: …` (:7283-7296). (b) **Tool-effect journal fence:** an error whose tool-effect failure has `reconciliation_required` gives `_ReconciliationCognitionResult` with status `FAILED` (:7343-7350). This is the path #4036 describes. |
| hits the retry cap | The mechanism's own attempt cap, where one exists. |
| cancelled before READY | `KestrelAgent._track_runtime_task` (`kestrel_agent.py:9668`) awaits `_runtime_publication_ready` before it enters the submitted coroutine. Its done-callback closes a coroutine that never started (:9724-9733). It is used by `SignalDispatcher._track_resident_durable_task` (`dispatcher.py:1797`), and for the scheduler supervisor by `SchedulerRunner.arm` (`runner.py:1814`, through `runtime_task_factory`, `feature.py:370-374`). `SignalDispatcher.enqueue_signal` (:1453) uses `KestrelAgent._track_background_task` (`kestrel_agent.py:9610`), which has no READY barrier. The host-owned scheduler runner (section 2) is built without either task factory (`kestrel_sovereign/server.py:2929-2947`; S28), so its supervisor and its occurrences are plain `asyncio.create_task` tasks (`runner.py:1374-1375`), and `runner.start()` (`kestrel_sovereign/server.py:2951`) arms polling at once. Its target agent becomes READY inside an occurrence that has already claimed: for a cold target, `AgentManagerHostedSchedulerExecutor.prepare_scheduled` (`runner.py:1093`) calls `_resolve_or_wake` (:1210, calls at :1125 and :1151), which calls `AgentManager.load_agent` (`kestrel_sovereign/multi_agent/agent_manager.py:6757`), then `_initialize_agent` (:2908, call at :6825), then `KestrelAgent.initialize` (`kestrel_agent.py:2655`, call at `agent_manager.py:3307`), then `run_boot_sequence` (`kestrel_sovereign/agent/boot.py:231`, call at `kestrel_agent.py:2706`). That sets READY only after every boot phase (`boot.py:255`), and READY sets `_runtime_publication_ready` (`kestrel_agent.py:2689-2699`). |
| dies with host | The process exits, gracefully or not, with the input in flight. Settlement, if any, comes from the next process. |

## 4. Who writes the terminal state today

### 4.1 Overview

"retry" means the write that happens is not terminal. The detail and the mark for
each cell are in 4.2 to 4.5.

| Mechanism | completes | fails terminally | is stopped | is fenced | retry cap | cancelled before READY | dies with host |
|---|---|---|---|---|---|---|---|
| A2A task | only the turn's own call to `respond_to_a2a_task` / `cancel_task` | none found | none found | none found | not reachable (no cap) | none found | none found |
| Queued wake | `ack_delivery` | (i) retry; (ii) `terminal_ackable` | `terminal_ackable` | (a) retry; (b) `failed` | not reachable (`max_attempts=0`) | retry (inferred) | retry |
| Wait completion | `record_delivery` locks the token | (i) retry; (ii) `record_delivery` locks the token | recorded as delivered (`coalesced_*`) | retry until the cap | `lock_at_max_attempts` | not applicable (no READY barrier) | retry (`lost_at_restart`) |
| Scheduler claim, agent-owned runner | `_finalize` (`success`) | `_finalize` (`failed`) | `_finalize` (`missed`, COGNITION tasks) | `_finalize` (`failed`); the scheduler's own lease loss writes nothing | none found (no cap) | no claim can exist | re-claim. SQLite: replay. PostgreSQL: replay if the host died before `_mark_effect_started` committed `executing`, `unresolved_effect` if after |
| Scheduler claim, host-owned runner (shared PostgreSQL) | as the agent-owned row | as the agent-owned row | as the agent-owned row | as the agent-owned row | none found (no cap) | none. A cold target boots after the claim. Cancelled during that boot, the claim and its `executing` log row stay until the lease expires, then the re-claim writes `unresolved_effect`. Cancelled before `executing` committed, the re-claim replays | re-claim: replay if the host died before `_mark_effect_started` committed `executing`, `unresolved_effect` if after |

### 4.2 A2A task

Terminal state: `a2a_tasks` state `completed`, `failed`, `canceled` or `rejected`.

| Way | Terminal write today | Mark |
|---|---|---|
| completes | Only if the turn itself makes it. `TaskFeature.respond_to_a2a_task` (`kestrel_sovereign/features/tasks/feature.py:870`) calls `TaskManager.update_status` (`task_manager.py:1874`, call at `feature.py:896`), which calls `TaskStore.save_recipient_lifecycle` (`task_store.py:712`, call at `task_manager.py:1945`), for `completed`, `failed` and `rejected`. For `canceled`: `TaskFeature.cancel_task` (`feature.py:958`) calls `TaskManager.cancel_task` (`task_manager.py:2169`), which calls `TaskStore.cancel_if_authorized`, reported privately. If the turn completes without either call, the task stays `submitted` or `working`. Its wake is acknowledged (4.3) and no task write follows: none found (S4, S5, S6). | measured |
| fails terminally | None found (S4, S5, S6). (i) The wake is NACKed for retry (4.3), so a later turn can still write the task. (ii) A withdrawal is returned only when the task is missing, already terminal, superseded or ineligible (`EventManagerMixin.validate_cognition_signal_execution`, `event_manager.py:834`; `rehydrate_durable_cognition_signal`, :605). A `DROPPED_CYCLE` drop settles only the wake (4.3) and leaves a live task unwritten. | measured |
| is stopped | None found (S4, S5, S6, S9). The wake is settled `terminal_ackable` (4.3) and the task stays `submitted` or `working`. This is the path #4014 traces. | inferred (Stop link, section 3); the dispatcher part is measured |
| is fenced | None found (S4, S5, S6). (a) The wake is NACKed for retry. (b) The wake is written `failed` (4.3) and the task stays `submitted`. This is the shape #4036 reports. | measured |
| hits the retry cap | Not reachable. The A2A consumers register `max_attempts=0` (`kestrel_agent.py:3511-3519`). `DurableSignalStore.persist_signal` copies the consumer's value into each delivery row (`durable.py:3856-3866`). `DurableSignalStore.nack_delivery` (`durable.py:5152`) applies the cap only when `max_attempts > 0`. No task write. | measured |
| cancelled before READY | None found for the task (S4, S5, S6). See the wake cell in 4.3. | inferred |
| dies with host | None found for the task (S4, S5, S6). At the next boot, `EventManagerMixin.reconcile_a2a_cognition_wakes` (`event_manager.py:712`) re-admits a wake for each `submitted` or `working` task from `TaskManager.list_cognition_wake_candidates` (`task_manager.py:2154`) that changed within its source's retention horizon (:750-764). A `working` task whose original wake is still live is skipped (:787-801). Whether the re-run turn is admitted is not settled here: #4023 reports a refusal by a single-use operation fence (#3999), and that fence was not located (S22). Retention, `TaskStore.cleanup_old` (`task_store.py:1938`), deletes only tasks that are already terminal. No sweeper that moves a live task to a terminal state was found (S7, S26). | inferred |

### 4.3 Queued wake (durable cognition delivery, A2A consumers)

Terminal state: `durable_signal_deliveries.status` `acknowledged`,
`terminal_ackable` or `failed`. Settlement runs in
`SignalDispatcher._execute_original_claimed_durable_cognition`
(`dispatcher.py:5837`) and in the claim guard
`SignalDispatcher._own_cognition_claim_body` (:5770).

| Way | Terminal write today | Mark |
|---|---|---|
| completes | `SignalDispatcher.ack_durable_delivery` (:2671, call at :6242) calls `DurableSignalStore.ack_delivery` (`durable.py:5090`), which writes `acknowledged`. If the ACK misses because the lease expired, `release_durable_delivery_after_task` (:2836, call at :6288) calls `DurableSignalStore.release_managed_delivery_after_task` (`durable.py:5286`), which writes `retry`. That write is not terminal, and the turn can run again. | measured |
| fails terminally | (i) An ordinary exception goes to `SignalDispatcher._settle_cognition_non_delivery` (:6367, call at :6335), then `nack_durable_delivery(terminal=False)` (:2688) and `DurableSignalStore.nack_delivery`, which writes `retry`. No terminal write follows, because `max_attempts=0`. (ii) A withdrawal (:5898-5911), `DROPPED_VALIDATION` or `DROPPED_CYCLE` (`_cognition_result_is_terminal`, :6360) goes to `_settle_cognition_non_delivery` with `terminal=True, terminal_ackable=True`, and the row is written `terminal_ackable`. | measured |
| is stopped | `_settle_cognition_non_delivery` calls `nack_durable_delivery(terminal=True, terminal_ackable=True)`, and the row is written `terminal_ackable`. If the lease has expired it falls back to `release_managed_delivery_after_task` with the same flags (:6395-6403). If that NACK raises, `_own_cognition_claim_body` (:5808-5812) calls `SignalDispatcher._terminalize_failed_cognition` (:6623), which writes `failed`. `_hold_durable_cognition_stop_completion` (:1569) reports the Stop as completed only when `terminal_ackable` was written (:1665-1673, :6338-6342). | inferred (Stop link, section 3); the dispatcher part is measured |
| is fenced | (a) `nack_delivery` writes `retry`. (b) `_settle_cognition_non_delivery` (:6372-6377) calls `_terminalize_failed_cognition`, then `SignalDispatcher._apply_cognition_control_terminal` (:6721). On PostgreSQL that is `PostgresBackend.fail_cognition_delivery` (`kestrel_sovereign/storage/db/postgres.py:2794`). On SQLite it is `DurableSignalStore.release_managed_delivery_after_task(terminal=True, terminal_ackable=False)`. Both write `failed`. | measured |
| hits the retry cap | Not reachable for these consumers (`max_attempts=0`, copied into each delivery row by `persist_signal`, `durable.py:3856-3866`). The cap expression in `DurableSignalStore.nack_delivery`, `release_managed_delivery_after_task` and `recover_abandoned_leases` (`durable.py:4906`) writes `failed` only for consumers registered with `max_attempts > 0` (S11). | measured |
| cancelled before READY | `SignalDispatcher._run_original_cognition_claim` (:5515) publishes the resident through `_track_resident_durable_task` and joins it (:5613-5638). A cancellation before READY leaves no routing task, so the exception reaches `_own_cognition_claim_body`. Its remaining branch (:5813-5814) calls `_release_retained_durable_cognition_task(observe_failure=True)`, then `release_durable_delivery_after_task`, then `release_managed_delivery_after_task`, which writes `retry`. That write is not terminal. This was not reproduced. #3775 (filed 2026-10-21) describes the joined-resident branch as doing no cleanup, and `_own_cognition_claim_body` first appears on 2026-10-23 (S15). Whether that guard covers #3775's case was not verified. | inferred |
| dies with host | The next owner's `SignalDispatcher._heartbeat_runtime_owner` (:3896) calls `_recover_abandoned_leases` (:3954, call at :3923), which calls `DurableSignalStore.recover_abandoned_leases`. That writes `retry` for leases whose owner is stopped or stale, or `failed` at a cap, which these consumers do not have. On a graceful exit, `_defer_unstarted_cognition_for_shutdown` (:5648) leaves unstarted claims to that recovery. Unactivated initial reservations go to `DurableSignalStore.recover_abandoned_initial_reservations`, whose body was not read. No terminal write. | measured for leases; inferred for initial reservations |

### 4.4 Wait completion

Terminal state: `wait_signal_state.last_signaled_outcome` set to the transition's
token, which stops any re-emit, together with `last_delivery_status`. The writes are
in `WaitReconciler._reconcile_once` (`waits/reconciler.py:516`, harvest phase) and
`WaitReconciler._process_handle` (:952). The wake goes through the non-durable
route: `SignalDispatcher.dispatch_signal` (`dispatcher.py:1256`), then `_run`, which
returns `_route_after_durable_persistence` (:6850), then `_route_under_locks`
(:7102), then `_dispatch_cognition`. `dispatch_signal` may pass the result through
`_snapshot_terminal_result` in `kestrel_sovereign/signals/outcome_custody.py`. That
function was not read, so the cells below that depend on the returned status
being kept are inferred.

| Way | Terminal write today | Mark |
|---|---|---|
| completes | Status `ok` is in `_PERSISTED_STATES` (:119). `WaitSignalStore.record_delivery(signaled_outcome=target, delivered_at=…)` (store :415; call :607) records status `ok_*` and locks the token. | measured |
| fails terminally | (i) `FAILED` is a soft failure. `record_delivery` without `signaled_outcome` (:696) writes no terminal state, and the next tick re-emits, which starts another turn. (ii) `DROPPED_VALIDATION` or `DROPPED_CYCLE` (:120) is a hard failure: `record_delivery(signaled_outcome=target)` (:650) locks the token. | measured |
| is stopped | A Stop returns status `COALESCED`, and `"coalesced"` is in `_PERSISTED_STATES` (:119). The harvest records it as persisted, with `record_delivery(signaled_outcome=target, delivered_at=…)` (:607) and status `coalesced_*`. The completion is locked as delivered although its turn was stopped, and no re-emit follows. The reconciler does not read the `stop_acknowledged` marker (S9, S16). | inferred (Stop link; `_snapshot_terminal_result`) |
| is fenced | (a) and (b) are both status `FAILED`, a soft failure (:696). No terminal write. The completion is re-emitted and a new turn runs for it, until the cap. | inferred (`_snapshot_terminal_result`) |
| hits the retry cap | `WaitReconciler._process_handle` (:1091-1124) calls `WaitSignalStore.lock_at_max_attempts` (store :638; `MAX_DELIVERY_ATTEMPTS = 10`, :112), which writes `max_attempts_exceeded` and locks the token. The lock is cleared only by `record_pending` (store :372, a new transition's dispatch) or by `record_delivery`. An agent-facing exit was not looked for in `features/wait/feature.py` beyond S11. #4073 reports that there is none. | measured for the lock |
| cancelled before READY | Not applicable. The wake is published by `SignalDispatcher.enqueue_signal` through `KestrelAgent._track_background_task`, which has no READY barrier (section 3). | measured |
| dies with host | On the next process's first tick, the harvest finds a durable pending row with no in-memory task and calls `record_delivery(delivery_status="lost_at_restart")` (:552-565) without a token. That write is not terminal, and the next tick re-emits. The attempt already counted by `record_pending` is not refunded. Within a running process, a dispatch task that was cancelled has its `CancelledError` returned by `owned_task_error` (`kestrel_sovereign/_async_ownership.py:1701`). The harvest re-raises it inside a block that catches only `Exception` (:574-581), so it leaves the tick before the row is classified, and the pending row stays for the restart case above. | measured for restart; inferred for the in-process case |

### 4.5 Scheduler claim

Terminal state: the occurrence's `task_execution_log.status` leaves `claimed` or
`executing`, for example `success`, `failed`, `missed`, `skipped_misfire` or
`cancelled`. A one-shot row also gets `scheduled_tasks.terminal_status`. The writer
is `SchedulerRunner._finalize` (`runner.py:4798`), a compare-and-set on the live
claim, called from `SchedulerRunner._execute_claim` (:3899).

Both runners (section 2) go through the same `_run_occurrence` (:2319), which
commits `_claim` (call at :2338) before it calls `_execute_claim` (:2342). They
differ in when the PostgreSQL effect-start mark is written. On PostgreSQL,
`SchedulerRunner._mark_effect_started` (:4433) changes the log row from `claimed`
to `executing` in its own transaction. On SQLite it returns early (:4447) and
writes nothing.

- Agent-owned runner: its executor is the bound method
  `SchedulerFeature._dispatch_scheduled_task` (`feature.py:363-366`), which has
  no `prepare_scheduled` attribute, so `_prepared_executor_method` (:4283)
  returns `None`. The only mark is at :4063, after the admission checks and
  immediately before the target is dispatched. Whether any deployment runs this
  runner on PostgreSQL was not checked.
- Host-owned runner: `AgentManagerHostedSchedulerExecutor` has
  `prepare_scheduled` (:1093), so the first mark is at :3966-3967, before
  preparation, that is, before a cold target is loaded. A second mark at :4063
  leaves `executing` in place (`status IN ('claimed', 'executing')`, :4466-4471).
  If preparation raises `SchedulerFeatureUnavailable` while the lease is live,
  `_mark_effect_started(preparation_deferred=True)` (:4206) writes `claimed`
  back and `_execute_claim` returns without `_finalize`.

The cells below apply to both runners unless they say otherwise.

| Way | Terminal write today | Mark |
|---|---|---|
| completes | `_execute_claim` calls `_finalize` (:4157), with the status from `SchedulerRunner._normalise_result` (:4754). For a COGNITION task, `OK` maps through `SchedulerFeature._translate_signal_result` (`feature.py:1188`) and `_normalise_result` to `success`. Outcomes in `_finalize` that come from a permission gate are reported privately. | inferred (`_snapshot_terminal_result`) for COGNITION tasks; measured for the runner part |
| fails terminally | `FAILED`, `DROPPED_VALIDATION` or `DROPPED_CYCLE` makes `_translate_signal_result` raise `RuntimeError` (:1239-1247). `_execute_claim` sets `failed` (:4123-4141) and calls `_finalize`. A preparation failure calls `_finalize(status="failed")` (:4245). A plain storage error ends `SchedulerRunner._run_occurrence` (:2356-2367) and leaves the claim to lease recovery, with no terminal write. | inferred (`_snapshot_terminal_result`) for COGNITION tasks; measured for the runner part |
| is stopped | For a COGNITION task, `COALESCED` maps through `_translate_signal_result` (:1256-1263) to `ScheduledTaskOutcome(status=MISSED_COGNITION_STATUS)`. `_normalise_result` keeps that status, and `_finalize` writes `missed`. For ACTION and ARTIFACT tasks there is no turn, and whether a Stop reaches a scheduled handler was not read. | inferred (Stop link; `_snapshot_terminal_result`) |
| is fenced | (a) and (b) both arrive as `FAILED`, and `_finalize` writes `failed`. The journal fence comes back as a returned result, not as a raised control error. The scheduler's own lease fence is different: when renewal is lost (`renewal_state.is_live()` false at :3981, :3991, :4071, :4150), `_execute_claim` returns without calling `_finalize`. A loss during the host-owned runner's preparation cancels the preparation inside `_run_dispatch_while_lease_live` (:4475), and `_execute_claim` returns at :4180-4181, also without `_finalize`. Lease recovery then applies, and which of its two outcomes follows depends on whether `executing` had been committed (dies with host). Execution-control errors latch readiness through `_latch_protocol_failure` without finalizing (:4124-4130, :4213-4218). | inferred for (a) and (b) (`_snapshot_terminal_result`); measured for the lease fence |
| hits the retry cap | None found (S19). `_claim` increments `attempt_count` on each re-claim of the same occurrence (:3601-3606), and no ceiling was found. | measured |
| cancelled before READY | **Agent-owned runner:** no claim can exist yet. A claim is made only in `_run_occurrence` (:2319), which is started only by `_admit_occurrence` (:2296), which is called only by `_tick` (:2256), which is called only by `_loop` (:2235) (S27). `_loop` runs in the worker task that `_supervise_loop` (:2133) creates (:2148). `_supervise_loop` is the task that `arm` (:1799) publishes with `runtime_task_factory` (:1814), which is `_track_runtime_task` when the agent has it (`feature.py:370-374`). `arm` is called from `SchedulerFeature.on_agent_ready` (`feature.py:401-405`). **Host-owned runner:** the claim already exists. The occurrence task has no READY barrier, and a cold target boots to READY inside preparation (section 3). That is after `_claim` committed (:2338) and, on this runner, after `_mark_effect_started` (:3966-3967) committed `executing`. `SchedulerRunner.stop` (:1828) cancels every admitted occurrence (:1843). It is reached from `_shutdown_host_scheduler` (`kestrel_sovereign/server.py:3035`, call at :3054) (S31). A cancellation during the cold boot leaves `AgentManagerHostedSchedulerExecutor.prepare_scheduled` (:1093) with no database write. `_run_dispatch_while_lease_live` (:4475) cancels and joins the preparation and lets the cancellation propagate (:4554-4590). `_execute_claim` re-raises it (:4182-4183) without `_finalize`, and its `finally` (:4254) only stops the lease renewal. `_run_occurrence` re-raises it (:2343-2354), and its done callback `_occurrence_finished` (:2572) writes nothing. The lease, the claim token and the `executing` log row stay until the lease expires. The next `_claim` of that occurrence then writes `unresolved_effect` (dies with host). A cancellation that lands before the :3967 mark commits leaves the log `claimed`, and the re-claim replays. Not read: what else can cancel a host-owned occurrence, and what a cancelled `load_agent` leaves of the target beyond the rollback in `run_boot_sequence` (`boot.py:256-264`). | measured |
| dies with host | The lease expires, and a later `_claim` re-claims the same execution id (:3548-3570, :3574). What the re-claim does depends on the log status the dead owner left, which `_claim` reads back at :3723-3741. **SQLite:** `_mark_effect_started` returns early (:4447), so the log is `claimed` at every point. The re-claim's `INSERT … ON CONFLICT` (:3683) keeps it `claimed`, the check at :3734 passes, and `_claim` returns the occurrence, which is executed again (replay). On replay the misfire check in `_execute_claim` (:4010-4026) can write `skipped_misfire` instead of dispatching. **PostgreSQL, host died after acceptance and before `executing` committed:** `_claim` committed the log row `claimed` (:3683). For the agent-owned runner this window runs to the mark at :4063, which follows the misfire check and the admission reads. For the host-owned runner it runs to the mark at :3966-3967, before preparation. A host-owned preparation deferred by `SchedulerFeatureUnavailable` that wrote `claimed` back (:4206) is in this window again. The log is still `claimed`, so the re-claim replays as on SQLite, with no terminal write until that replay's own `_finalize`. **PostgreSQL, host died after `executing` committed:** the re-claim finds `executing` (:3734) and calls `SchedulerRunner._terminalize_inconsistent_execution_log` (:3735; :3386). That writes `scheduled_tasks.terminal_status = unresolved_effect`, keeps the claim's execution id, clears the lease and disables the row (:3422-3439). The log row is left `executing`. | measured |

## 5. Not read

Every in-scope file and directory that was not opened, or was opened only in part.
None of them was unreadable. No cell above depends on them except where the cell
names them.

- `kestrel_sovereign/signals/outcome_custody.py`: not opened (`_snapshot_terminal_result`).
- `kestrel_sovereign/signals/delivery.py`: not opened (`await_terminal_delivery`, `detached_from_enclosing_turn`).
- `kestrel_sovereign/signals/store.py`: not opened (`signal_log`).
- `kestrel_sovereign/signals/sources/a2a.py`, `a2a_task_submitted.py`, `a2a_question_answered.py`: only searched (S1), not opened.
- `kestrel_sovereign/signals/sources/scheduler.py`, `self_followup.py`, `wait.py`, `wait_replay.py`, `channels.py`: not opened.
- `kestrel_sovereign/signals/dispatcher.py`: opened in part. Not read: `_run` (:4669-5218) except its return points and persistence calls; `_route_durable_terminal_delivery`; the shutdown functions (`_complete_durable_shutdown`, `_retire_runtime_owner`); `_release_expired_initial_handoffs`; `claim_durable_delivery_for_event`.
- `kestrel_sovereign/signals/durable.py`: opened in part. Not read: the bodies of `recover_abandoned_initial_reservations`, `claim_delivery`, `claim_delivery_for_event`, `purge_expired`.
- `kestrel_sovereign/execution_custody.py`: not opened.
- `kestrel_sovereign/tools/effect_journal.py`: not opened (how the journal fence and a restart treat a redelivered wake).
- `kestrel_sovereign/stop/` and `kestrel_sovereign/cli_stop.py`: not opened (how agent-scope Stop picks the turns it cancels, and whether it touches unclaimed ledger rows).
- `kestrel_sovereign/agent/invocation.py`, `agent/turn_lifecycle.py`, `agent/request_lifecycle.py`: only the lines cited in section 3.
- `kestrel_sovereign/agent/boot.py`: only `run_boot_sequence` (:231-270).
- `kestrel_sovereign/kestrel_agent.py`: beyond the functions cited, only `KestrelAgent.initialize` from :2655 to the boot call at :2706.
- `kestrel_sovereign/hold/`: not opened. Hold dispositions are not a column here.
- `kestrel_sovereign/waits/engine.py`: not opened.
- `kestrel_sovereign/features/wait/feature.py`: only searched (S11).
- `kestrel_sovereign/a2a/stores/unified/task_store.py`: opened in part. Not read: `_save_recipient_lifecycle_from_states`, `list_cognition_wake_candidates`, `cancel_if_authorized` (section 6).
- `kestrel_sovereign/a2a/question_supervisor.py`, `a2a/stores/unified/question_store.py`: not opened (settlement of questions on the sender side).
- `kestrel_sovereign/a2a/push_outbox.py`, `push_worker.py`, `push_service.py`, `a2a/native_client.py`, `features/peers/feature.py`: not opened (what the sender sees).
- `kestrel_sovereign/features/channels/feature.py`: not opened. The `channel.message` consumer uses the same ledger as 4.3, but its cells were not built.
- `kestrel_sovereign/features/restart_coordinator/feature.py`: not opened.
- `kestrel_sovereign/features/base.py`: not opened (`execute_as_subagent`, cited by #4036; default `handle_task`).
- `kestrel_sovereign/a2a/task_manager.py`: the skill-task paths were read, but no table was built for them. On those paths the writers are `TaskManager._persist_execution_outcome`, `_persist_sync_execution_outcome`, `fail_task` (:880), `_settle_execution_interruption` (:1097) and `_persist_handler_cancellation` (:1064). At close, `drain_execution_tasks(cancel=True)` (:1000) calls `cancel_task`.
- `kestrel_sovereign/features/scheduler/runner.py`: `HostedSchedulerExecutor` and `AgentManagerHostedSchedulerExecutor` (:930-1251) were read for this revision. Not read: `_supervise_loop` beyond its worker creation (:2133-2150).
- `kestrel_sovereign/server.py`: only `_uses_shared_postgres_scheduler` (:2649), `_prepare_shared_postgres_scheduler_protocol` (:2731-2853), `_start_host_scheduler` up to its startup-failure cleanup (:2856-3010) and `_shutdown_host_scheduler` (:3035-3068).
- `kestrel_sovereign/multi_agent/agent_manager.py`: only `load_agent` (:6757-6830) and the two lines that locate `_initialize_agent` (:2908) and its `agent.initialize()` call (:3307). The rest of `multi_agent/` not opened, including how `scheduler_authority_for` and `scheduler_lifecycle_lock` are implemented.
- Out of this repository, not read: the Workflows feature (workflow wait-resume consumers on the same ledger) and the Talon feature (its wait provider).
- The single-use operation fence #4023 attributes to #3999: not located (S22).

## 6. Settlements that are permission decisions or privacy obligations

These are reported privately: file and function only.

- `kestrel_sovereign/a2a/stores/unified/task_store.py`, `TaskStore.cancel_if_authorized` (4.2, completes, `canceled` branch).
- `kestrel_sovereign/a2a/task_manager.py`, `TaskManager.execute_skill` (skill tasks; section 5).
- `kestrel_sovereign/features/scheduler/runner.py`, `SchedulerRunner._finalize` (4.5, completes; the outcomes that come from a permission gate).
- `kestrel_sovereign/features/scheduler/feature.py`, `SchedulerFeature._dispatch_scheduled_task` (4.5; a refusal at fire time that depends on the privacy mode).
- `kestrel_sovereign/signals/dispatcher.py`, `SignalDispatcher._release_expired_initial_handoffs` (4.3; release of a wake whose payload a privacy mode elided).
