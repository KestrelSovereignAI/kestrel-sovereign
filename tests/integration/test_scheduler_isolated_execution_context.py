"""Core-to-SDK propagation of durable scheduler execution identity.

This uses the SDK's in-memory JSON-RPC transport rather than a proxy stub so
the test covers Core's scheduler context, the isolated proxy, and the public
SDK wire boundary together.
"""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
from typing import Any

import pytest

from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features.isolated_runtime import ProxyFeature
from kestrel_sovereign.features.scheduler.runner import (
    SchedulerExecution,
    _SchedulerExecutionScope,
    _current_execution,
)


_OVERSIZED_SCHEDULE_ID = "🦅" * 129  # 516 UTF-8 bytes: just over the SDK cap.
_OVERSIZED_SCHEDULE_SOURCE_ID = (
    "schedule-sha256:"
    + hashlib.sha256(_OVERSIZED_SCHEDULE_ID.encode("utf-8")).hexdigest()
)


class _MemoryReader:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def readline(self) -> bytes:
        return await self._queue.get()

    def feed(self, data: bytes) -> None:
        self._queue.put_nowait(data)

    def close(self) -> None:
        self._queue.put_nowait(b"")


class _MemoryWriter:
    def __init__(self, peer: _MemoryReader) -> None:
        self._peer = peer

    def write(self, data: bytes) -> None:
        for line in data.splitlines(keepends=True):
            self._peer.feed(line)

    async def drain(self) -> None:
        await asyncio.sleep(0)

    def close(self) -> None:
        self._peer.close()

    async def wait_closed(self) -> None:
        await asyncio.sleep(0)


class _SdkJsonRpcAdapter:
    """ProxyFeature client facade backed by the SDK's real JSON-RPC client."""

    def __init__(self, service: Any, client_type: Any) -> None:
        self._service = service
        self._client_type = client_type
        self._client: Any = None
        self._service_task: asyncio.Task[None] | None = None

    @property
    def capabilities(self) -> dict[str, Any]:
        return self._client.capabilities if self._client is not None else {}

    @property
    def supports_tool_execution_context(self) -> bool:
        return bool(
            self._client is not None
            and self._client.supports_tool_execution_context
        )

    async def start(self) -> None:
        host_reader = _MemoryReader()
        service_reader = _MemoryReader()
        host_writer = _MemoryWriter(service_reader)
        service_writer = _MemoryWriter(host_reader)
        self._service_task = asyncio.create_task(
            self._service.serve(service_reader, service_writer)
        )
        self._client = self._client_type(host_reader, host_writer)
        await self._client.initialize()
        await self._client.health()

    async def stop(self) -> None:
        if self._client is None:
            return
        try:
            if self._service_task is not None and not self._service_task.done():
                await self._client.shutdown()
                await self._service_task
        finally:
            await self._client.close()

    async def health(self) -> dict[str, Any]:
        return await self._client.health()

    async def list_tools(self) -> list[Any]:
        return await self._client.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any], *, context: Any = None) -> Any:
        return await self._client.call_tool(name, arguments, context=context)

    def on_event(self, handler: Any) -> None:
        self._client.on_event(handler)


def _execution(*, attempt: int, schedule_id: str = "daily-report") -> SchedulerExecution:
    return SchedulerExecution(
        id="occurrence-42",
        schedule_id=schedule_id,
        agent_id="agent-1",
        task_name="isolated_effect",
        args={},
        scheduled_for="2026-07-25T15:00:00+00:00",
        idempotency_key="durable-effect-42",
        attempt=attempt,
        owner="scheduler-replica",
    )


async def _execute_scheduled(tool: Any, execution: SchedulerExecution, **arguments: Any) -> Any:
    scope = _SchedulerExecutionScope(execution)
    token = _current_execution.set(scope)
    try:
        return await tool.execute(**arguments)
    finally:
        scope.revoke()
        _current_execution.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schedule_id", "expected_source_id"),
    [
        ("daily-report", "daily-report"),
        (_OVERSIZED_SCHEDULE_ID, _OVERSIZED_SCHEDULE_SOURCE_ID),
    ],
    ids=("native-schedule-id", "oversized-migrated-schedule-id"),
)
async def test_scheduler_context_crosses_real_sdk_json_rpc_and_is_revoked(
    monkeypatch, tmp_path, schedule_id, expected_source_id
):
    """Retries keep a stable effect key while handler context stays scoped."""

    try:
        from kestrel_sdk.isolated_feature import (
            IsolatedFeatureClient,
            IsolatedFeatureService,
            ToolMetadata,
            get_tool_execution_context,
        )
    except ImportError:
        pytest.skip("requires kestrel-sovereign-sdk 0.32.0 execution-context API")

    service = IsolatedFeatureService(name="execution-test", version="1.0.0")
    seen_async: list[tuple[dict[str, Any], Any]] = []
    seen_sync: list[tuple[dict[str, Any], Any]] = []
    child_started = asyncio.Event()
    inspect_child = asyncio.Event()
    child_done = asyncio.Event()
    child_context: list[Any] = []

    async def child() -> None:
        child_started.set()
        await inspect_child.wait()
        child_context.append(get_tool_execution_context())
        child_done.set()

    async def async_handler(arguments: dict[str, Any]) -> dict[str, Any]:
        context = get_tool_execution_context()
        assert context is not None
        seen_async.append((dict(arguments), context))
        asyncio.create_task(child())
        return {"attempt": context.attempt}

    def sync_handler(arguments: dict[str, Any]) -> dict[str, Any]:
        context = get_tool_execution_context()
        assert context is not None
        seen_sync.append((dict(arguments), context))
        return {"attempt": context.attempt}

    schema = {"type": "object", "properties": {}}
    service.register_tool(
        ToolMetadata(
            name="async_effect", description="async execution-context test", input_schema=schema
        ),
        async_handler,
    )
    service.register_tool(
        ToolMetadata(
            name="sync_effect", description="sync execution-context test", input_schema=schema
        ),
        sync_handler,
    )
    adapter = _SdkJsonRpcAdapter(service, IsolatedFeatureClient)

    runtime = InstalledFeatureRuntime(
        class_name="ExecutionFeature",
        entry_point="execution.feature:ExecutionFeature",
        distribution="execution-feature",
        runtime="isolated-venv",
        service="execution-service",
    )
    monkeypatch.setenv("KESTREL_FEATURE_EXECUTIONFEATURE_BIN", "/bin/execution-service")
    agent = SimpleNamespace(
        storage_path=str(tmp_path / "agent" / "kestrel_prime.db"),
        features={},
    )
    feature = ProxyFeature(agent, runtime, client_factory=lambda **_: adapter)
    await feature.initialize()

    try:
        tools = {tool.name: tool for tool in feature.get_tools()}
        user_arguments = {
            "execution_context": "untrusted-user-value",
            "payload": "keep this unchanged",
        }
        await _execute_scheduled(
            tools["async_effect"],
            _execution(attempt=1, schedule_id=schedule_id),
            **user_arguments,
        )
        await asyncio.wait_for(child_started.wait(), timeout=1)
        await _execute_scheduled(
            tools["sync_effect"],
            _execution(attempt=2, schedule_id=schedule_id),
            payload="retry",
        )

        # The public SDK context reaches both async and sync handlers.  Retry
        # attempt increments without changing the durable occurrence or effect
        # identity, and reserved metadata never mutates user arguments.
        assert seen_async[0][0] == user_arguments
        assert seen_async[0][1].invocation_id == "occurrence-42"
        assert seen_async[0][1].idempotency_key == "durable-effect-42"
        assert seen_async[0][1].attempt == 1
        assert seen_async[0][1].trigger.kind == "scheduler"
        assert seen_async[0][1].trigger.source_id == expected_source_id
        assert len(seen_async[0][1].trigger.source_id.encode("utf-8")) <= 512
        assert seen_sync[0][1].invocation_id == "occurrence-42"
        assert seen_sync[0][1].idempotency_key == "durable-effect-42"
        assert seen_sync[0][1].attempt == 2

        # SDK revocation reaches a task that inherited the handler context.
        inspect_child.set()
        await asyncio.wait_for(child_done.wait(), timeout=1)
        assert child_context == [None]
    finally:
        inspect_child.set()
        await feature.shutdown()
