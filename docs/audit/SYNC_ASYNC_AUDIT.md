# Sync/Async Audit

Control document for the original issue `#300` and the refreshed current-runtime
audit in issue `#624`, focused on maintained runtime surfaces.

## Boundary model

- FastAPI request handlers should be `async def`.
- Command handlers should be `async def` when they directly invoke async work.
- Sync handlers may remain sync only when they are truly synchronous.
- Shared dispatch should use explicit awaitability checks, not ad hoc coroutine assumptions.

## First-pass findings

### Fixed

- `!verify-constitution` was previously treating async constitution verification and safe-mode entry as synchronous.
  - Fixed by making the handler explicitly async and awaiting both calls.
- `!privacy-save` was a sync handler returning a coroutine from `privacy_agent.save_isolated_session()`.
  - Fixed by making the handler explicitly async and awaiting the save.
- `KestrelAgent.set_privacy_mode()` was a sync method branching on loop state to decide whether consent ran via `create_task()` or `run_until_complete()`.
  - Fixed by making the transition explicitly async and awaiting it from both command and API paths.
- `KestrelAgent.create_trusted_agent()` was a sync method calling async graph storage without `await`.
  - Fixed by making trusted-agent creation explicitly async and awaiting it from the command path.
- `ContextBuilder.build_rag_context()` was a sync helper calling async `storage.search_chunks()`.
  - Fixed by making the helper explicitly async and updating its tests to use the real storage contract.
- `SecurityFeature.pending_approvals()` was mixing monotonic loop time with wall-clock request timestamps.
  - Fixed by computing approval age from UTC wall-clock time so pending request status is truthful.
- SSE/MCP/VastAI runtime timers were still using `get_event_loop()` inside active async code.
  - Fixed by normalizing those paths to monotonic elapsed-time checks that work cleanly across sync/async boundaries.
- Sovereignty file-browser endpoints were doing blocking filesystem work directly on async request paths.
  - Fixed by offloading directory scans and preview reads via `asyncio.to_thread()`.
- `StrategicMemory` GitHub helpers were using legacy loop executor plumbing for blocking network calls.
  - Fixed by making the offload explicit with `asyncio.to_thread()` and direct contract tests.
- `CommandHandler.handle()` was checking `hasattr(result, '__await__')` instead of using `inspect.isawaitable()`.
  - Fixed to use the standard awaitability check.
- `KestrelAgent.close()` exposed a misleading sync cleanup path over async storage shutdown.
  - Fixed by removing `close()` and standardizing on `await agent.shutdown()` for maintained cleanup.
- `CodeEditFeature` tool methods (`code_diff`, `code_commit`, `code_test`, `code_lint`, `code_rollback`) were calling `subprocess.run()` directly on the event loop.
  - Fixed by introducing `_run_subprocess()` wrapper using `asyncio.to_thread()` and replacing all call sites.
- `CodeEditFeature` async tool methods were still reading and writing files directly on the event loop.
  - Fixed by offloading read, search, edit, restart-signal, and log file operations via `asyncio.to_thread()`.
- `local_mps_adapter.py` `generate_image()` was still using `get_event_loop().time()` for elapsed timing.
  - Fixed by replacing it with `time.monotonic()`.
- `local_mps_adapter.py` still performed local filesystem and process work directly inside async training/generation methods.
  - Fixed by offloading training setup, process launch/termination, log reads, output scans, LoRA reads, cleanup, generation artifact checks, and artifact reads/writes via `asyncio.to_thread()`.
- The Kestrel GitHub bot accepted webhooks and launched event handlers with bare `asyncio.create_task()`, leaving in-flight tasks unowned during shutdown.
  - Fixed by tracking webhook event tasks, removing completed tasks from the set, and cancelling/awaiting any in-flight handlers during feature shutdown.
- `KestrelAgent._post_response_pipeline()` launched post-response memory enrichment as an unowned background task, which could still be touching storage while shutdown closed storage.
  - Fixed by adding agent-owned background task tracking and cancelling/awaiting those tasks before sync and storage shutdown.
- `MemoryRetriever.retrieve()` launched rehearsal-effect access-count writes with bare `asyncio.create_task()`, so storage writes could outlive retrieval and agent shutdown.
  - Fixed by making `MemoryRetriever` own those update tasks, adding deterministic drain/shutdown methods, and wiring `MemorySystem.shutdown()` into `KestrelAgent.shutdown()` before storage closes.
- `KeyRotationService.start_rotation()` and `resume_rotation()` launched long-running database mutation work with bare `asyncio.create_task()`.
  - Fixed by tracking rotation tasks in the service, adding drain/shutdown hooks, and preserving resume semantics for cancelled in-progress rotations.
- `TaskManager.execute_skill(sync=False)` launched A2A background skill execution without lifecycle ownership.
  - Fixed by tracking execution tasks, cancelling/awaiting them in `TaskManager.close()`, and saving a terminal `canceled` task state before stores close.
- `LLMService` model preference persistence used loop-created tasks with no owner, so preference writes could outlive service cleanup.
  - Fixed by tracking persistence tasks, draining them in `LLMService.close()`, and logging callback failures through a done callback.
- Voice websocket VAD processing uses `asyncio.create_task()` inside the request handler.
  - Classified as acceptable request-scoped concurrency: the websocket `finally` path signals the VAD queue, cancels the task, and awaits it before the request exits. Added a disconnect test proving the VAD generator cleans up.

### Resolved audit patterns

- Conflicting close/shutdown contracts: resolved. Agent cleanup contract is `await agent.shutdown()`. MCPToolManager's sync `close()` is correctly called from async `shutdown()`.
- Command/API parity: verified. `CommandHandler.handle()` properly dispatches both sync and async results.
- Blocking subprocess/file work on async paths: resolved. All maintained tool paths now offload via `asyncio.to_thread()`.
- Background-task and scheduler paths: refreshed. Scheduler, heartbeat, GitHub App webhook handling, agent post-response enrichment, memory rehearsal-effect writes, key rotation, A2A background execution, and LLM preference persistence use owned async task patterns with shutdown cleanup. Voice VAD is request-scoped and awaited by the websocket cleanup path.
- `get_event_loop()` elimination: zero remaining calls in maintained source (`kestrel_sovereign/` and `endpoints/`).

## Initial high-risk surfaces

- `kestrel_sovereign/command_handler.py`
- `kestrel_sovereign/kestrel_agent.py`
- `endpoints/`
- scheduler and heartbeat paths
- feature execution boundaries that bridge tools, runtime, and storage
- lifecycle paths that expose both `close()` and `shutdown()`

## Proof added so far

- `tests/unit/test_command_handler_constitution_contracts.py`
- `tests/unit/test_command_handler_privacy_contracts.py`
- `tests/unit/test_command_handler_async_boundary_contracts.py`
- `tests/unit/test_consent_caller_contracts.py`
- `tests/unit/test_agent_runtime_endpoint_contracts.py`
- `tests/unit/test_code_edit_feature.py` (subprocess offload contract)
- `tests/unit/test_local_mps_adapter_async_contracts.py`
- `tests/unit/test_strategic_memory_async_contracts.py`
- `tests/unit/test_sovereignty_endpoint_contracts.py`
- GitHub bot webhook lifecycle tests
- `tests/unit/test_security_feature.py`
- `tests/unit/test_context_builder.py`
- `tests/unit/test_kestrel_agent.py`
- `tests/unit/test_memory_wiring.py`
- `tests/unit/test_key_rotation.py`
- `tests/unit/test_a2a_task_manager.py`
- `tests/unit/test_llm_service.py`
- `tests/unit/test_voice_websocket.py`

## Final classification

The sync/async boundary audit is complete for the maintained core runtime surface covered by issue `#624`:

- Identified command/API awaitability violations have been fixed with root-cause changes.
- Identified blocking I/O on maintained async tool paths has been offloaded via `asyncio.to_thread()`.
- All `get_event_loop()` usage has been eliminated from maintained source code.
- Direct contract tests cover each corrected seam.
- Remaining maintained `create_task()` sites have been classified as owned lifecycle tasks or request-scoped tasks.

Accepted task-spawn patterns:
- Owned lifecycle loops: `TaskWorker`, storage sync service, delivery queue, spawn lifecycle TTL, heartbeat, scheduler runner, GitHub bot event handling, memory access updates, key rotation, A2A background execution, and LLM preference persistence.
- Request-scoped concurrency: voice websocket VAD processing, which is cancelled and awaited by the websocket cleanup path.

Deferred/extracted workflow code:
- Training adapter task spawns in `features/training/adapters/` are intentionally excluded from this core runtime closure because LoRA/training workflows are being extracted from core. They should be handled with the non-core feature extraction work rather than used to keep issue `#624` open.
