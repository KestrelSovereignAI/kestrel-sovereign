"""Bridge serializer withdrawal survives actual source-owner retirement.

Real BridgeFeature SQL, RequestLifecycleMixin registration and the owned
iterator execute. The request/Agent view is an admitted component input, not
authenticated gateway/full Agent READY or a paid LLM stream.
"""

import asyncio
import inspect

import pytest
from fastapi import FastAPI, Request

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.features.bridge import router as bridge_router
from kestrel_sovereign.features.bridge.feature import BridgeFeature
from kestrel_sovereign.features.bridge.protocol import BridgeRequest
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["native-control", "native-cancel", "ordinary", "healthy"]
)
async def test_actual_bridge_stream_keeps_serializer_withdrawal(
    native_memory_owners, monkeypatch, kind
):
    n = native_memory_owners
    started, source_retired = asyncio.Event(), asyncio.Event()
    source_tasks, originals, withdrawals = [], [], []
    actual_terminal = bridge_router.execution_terminal_error

    def observed_terminal(*errors, **kwargs):
        if errors and isinstance(errors[0], asyncio.CancelledError):
            withdrawals.append(errors[0])
        return actual_terminal(*errors, **kwargs)

    monkeypatch.setattr(bridge_router, "execution_terminal_error", observed_terminal)

    class BridgeInput(RequestLifecycleMixin):
        def __init__(self):
            self.did = self.agent_id = n.did
            self.storage = self._raw_storage = n.pg_storage
            self._current_request_id = None
            self._active_request_ids = set()
            self._active_request_counts = {}
            self._active_request_started_at = {}
            self._cancelled_requests = set()
            self._request_completion_events = {}
            self.features = {}

        async def process_input_streaming(self, *args, **kwargs):
            source_tasks.append(asyncio.current_task())
            try:
                started.set()
                await asyncio.Event().wait()
                yield "unreachable after cancelled serializer"
            finally:
                try:
                    assert await n.pg_storage.db.fetchval("SELECT 1") == 1
                    if kind.startswith("native"):
                        try:
                            n.lease.require_live()
                        except BaseException as original:
                            originals.append(original)
                            if kind == "native-cancel":
                                raise asyncio.CancelledError(
                                    "actual source retirement observation withdrawn"
                                ) from original
                            raise
                    if kind == "ordinary":
                        raise OSError("ordinary source cleanup failure")
                finally:
                    source_retired.set()

    agent = BridgeInput()
    feature = BridgeFeature(agent)
    agent.features["BridgeFeature"] = feature
    app = FastAPI()
    app.state.agent = agent
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/bridge/stream",
            "raw_path": b"/api/bridge/stream",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 49300),
            "server": ("127.0.0.1", 0),
            "app": app,
        }
    )
    response = owner = None
    request_id = "actual-bridge-serializer"
    try:
        await feature.initialize()
        endpoint = next(
            route.endpoint
            for route in bridge_router.get_router().routes
            if route.path == "/api/bridge/stream"
        )
        response = await inspect.unwrap(endpoint)(
            request,
            BridgeRequest(message="component input", request_id=request_id),
        )

        async def serialize():
            return await anext(response.body_iterator)

        owner = create_owned_task(
            serialize(), name="actual-bridge-serializer", retain_originals=True
        )
        await asyncio.wait_for(started.wait(), 5)
        owner.cancel("actual serializer transport withdrawn")
        outcome = await asyncio.wait_for(await_owned_task(owner), 5)
        assert (
            source_retired.is_set()
            and source_tasks
            and all(task.done() for task in source_tasks)
        )
        assert outcome.error is not None
        chain = tuple(_explicit_control_chain(outcome.error))
        if kind == "healthy":
            assert isinstance(outcome.error, asyncio.CancelledError)
            assert outcome.error.args == ("actual serializer transport withdrawn",)
        else:
            assert withdrawals and withdrawals[0] in chain, (
                "bridge discarded original serializer cancellation at source retirement"
            )
        if kind.startswith("native"):
            assert originals and originals[0] in chain
            assert is_execution_control_error(outcome.error)
        assert request_id not in agent._active_request_ids
        assert not n.requests
    finally:
        if owner is not None:
            await await_owned_task(owner)
        if response is not None:
            try:
                await response.body_iterator.aclose()
            except BaseException as error:
                assert is_execution_control_error(error) or isinstance(
                    error, asyncio.CancelledError
                )
        await feature.shutdown()
