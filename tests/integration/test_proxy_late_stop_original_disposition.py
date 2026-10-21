"""A timed-out actual stop transfers late native failure to its same marker."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.features import isolated_runtime
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_isolated_feature_runtime import FakeIsolatedClient, _initialized_host_ingress_proxy
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("hold_callbacks", [False, True])
async def test_late_native_stop_never_becomes_another_physical_attempt(
    native_pg, tmp_path, monkeypatch, source_cancel, hold_callbacks,
):
    entered, release, stops, sources = asyncio.Event(), asyncio.Event(), [], []
    class Client(FakeIsolatedClient):
        async def stop(self):
            stops.append(asyncio.current_task())
            assert len(stops) == 1, "native-unresolved original stop was invoked again"
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            sources.append(OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel)))
            raise sources[0].error
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, Client)
    client = feature._client
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_OPERATION_TIMEOUT", .02)
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_CANCELLATION_GRACE", .02)
    first = create_owned_task(feature.shutdown(), retain_originals=True)
    operation, held = None, []
    try:
        await asyncio.wait_for(entered.wait(), 5)
        first_outcome = await await_owned_task(first)
        assert first_outcome.error is not None and len(stops) == 1
        (tracked,) = tuple(feature._terminal_lifecycle_tasks)
        operation = tracked.task
        if hold_callbacks:
            monkeypatch.setattr(operation, "_schedule_settlement_callback", held.append)
        settled = asyncio.Event()
        operation.add_done_callback(lambda completed: settled.set())
        release.set()
        if hold_callbacks:
            await await_owned_task(operation._host_task)
            assert feature._terminal_lifecycle_tasks
        else:
            await asyncio.wait_for(settled.wait(), 5)
            assert not feature._terminal_lifecycle_tasks
        assert sources and stops[0].done()
        later = await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))
        sources[0].assert_retained(later.error)
        assert len(stops) == 1, "late native completion lost the exact failed per-client operation"
        markers = [marker for marker in feature._terminal_stop_completed_client_markers if marker.client() is client]
        assert len(markers) == 1 and markers[0].operation is operation
        sources[0].assert_retained(markers[0].failure)
        assert await native_pg[1].fetchval(f'SELECT count(*) FROM "{native_pg[2]}".effects WHERE id=231') == 1
    finally:
        release.set()
        for callback in held:
            callback(operation)
        await await_owned_task(first)
        if operation is not None:
            await await_owned_task(operation._host_task)
