"""Core proxy admissions retain their actual original lifecycle attempt.

Task-factory/callback faults carry actual earlier local PG receipts. The child
facade is declared component input, not a launched subprocess or Agent READY.
"""
import asyncio
import inspect

import pytest
from unittest.mock import Mock

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.features.isolated_runtime import ProxyFeature
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.unit.test_isolated_feature_runtime import (
    FakeIsolatedClient, _cfg_runtime, _initialized_host_ingress_proxy,
)
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["initialize", "shared", "owned", "prepared"])
@pytest.mark.parametrize("fault", ["refuse", "publish-before", "publish-after"])
async def test_proxy_lifecycle_admission_retains_original_before_refusal(
    native_pg, tmp_path, monkeypatch, door, fault,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    if door == "initialize":
        monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
        feature = ProxyFeature(Mock(did="did:test:proxy-original", features={}),
                               _cfg_runtime(), client_factory=FakeIsolatedClient)
    else:
        feature, _ = await _initialized_host_ingress_proxy(
            monkeypatch, tmp_path, FakeIsolatedClient)
    accepted, refused, attempts = [], [], []
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    selected_code = (ProxyFeature._initialize_uninterrupted.__code__
                     if door == "initialize" else ProxyFeature._terminal_cleanup_uninterrupted.__code__)
    class PublicationTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not self.failed and getattr(callback, "__name__", "") == "close_unstarted":
                self.failed = True
                if fault == "publish-after":
                    super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is selected_code:
            attempts.append(operation)
            if fault == "refuse":
                refused.extend((coroutine, operation))
                raise source.error
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    async def run():
        if door == "initialize":
            await feature.initialize()
        elif door == "owned":
            feature._latch_terminal_lifecycle()
            async with feature._reload_lock:
                await feature._complete_terminal_cleanup(lifecycle_lock_held=True)
        elif door == "prepared":
            feature.prepare_shutdown_with_agent_deadline()
            await feature.shutdown_with_agent_deadline()
        else:
            await feature.shutdown()
    owner = create_owned_task(run(), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        outcome = await await_owned_task(owner)
        source.assert_retained(outcome.error)
        assert attempts, "actual proxy lifecycle admission was not reached"
        assert all(task.done() for task in accepted), "proxy abandoned accepted publication-failed owner"
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in refused), "proxy leaked refused lifecycle coroutine"
        assert feature._terminal_lifecycle_latched
        count = len(attempts)
        repeated = await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))
        source.assert_retained(repeated.error)
        assert len(attempts) == count, "native-unresolved proxy created a replacement cleanup attempt"
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        for coroutine in refused:
            coroutine.close()  # Measurement ended; dispose this exact fixture input.
        supervisor = feature._supervision_task
        if supervisor is not None:
            feature._supervision_task = None
            if not supervisor.done():
                supervisor.cancel()
            await await_owned_task(supervisor)
        client = feature._client
        if client is not None:
            await FakeIsolatedClient.stop(client)


@pytest.mark.asyncio
async def test_proxy_native_neutral_cleanup_never_retries_failed_original(
    native_pg, tmp_path, monkeypatch,
):
    sources, stops = [], []
    class Client(FakeIsolatedClient):
        async def stop(self):
            stops.append(asyncio.current_task())
            if not sources:
                sources.append(OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch)))
            raise sources[0].error
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, Client)
    first = await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))
    assert len(stops) == 1 and sources
    sources[0].assert_retained(first.error)
    original = feature._terminal_cleanup_task
    assert original is not None and original.done(), "neutral cleanup discarded its failed original owner"
    for call in (feature.shutdown, feature.initialize):
        repeated = await await_owned_task(create_owned_task(call(), retain_originals=True))
        sources[0].assert_retained(repeated.error)
        assert len(stops) == 1, "native-unresolved retirement retried exact stop"
        assert feature._terminal_cleanup_task is original
    assert feature._terminal_retirement_clients
