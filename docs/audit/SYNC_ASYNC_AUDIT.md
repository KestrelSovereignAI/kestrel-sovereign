# Sync/Async Audit

First-pass control document for issue `#300`, focused on maintained runtime surfaces.

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
- `CommandHandler.handle()` was checking `hasattr(result, '__await__')` instead of using `inspect.isawaitable()`.
  - Fixed to use the standard awaitability check.
- `KestrelAgent.close()` exposed a misleading sync cleanup path over async storage shutdown.
  - Fixed by removing `close()` and standardizing on `await agent.shutdown()` for maintained cleanup.

### Active patterns to audit further

- Conflicting close/shutdown contracts.
  - Audit remains relevant in other lifecycle surfaces, but the agent cleanup contract is now `await agent.shutdown()`.
- Command/API parity for behaviors that cross runtime boundaries.
- Endpoint paths that mix blocking local file or subprocess work with async request handling.
- Background-task and scheduler call paths that may assume a running event loop.

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

## Next audit targets

- inventory all sync methods that invoke async work indirectly
- identify blocking I/O on active request paths
- tighten event-loop ownership patterns in runtime and scheduler boundaries
