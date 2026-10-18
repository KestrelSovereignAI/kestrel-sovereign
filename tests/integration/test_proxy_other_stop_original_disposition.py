"""Replacement/supervisor must hand off the SAME failed physical stop."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_isolated_feature_runtime import FakeIsolatedClient, _initialized_host_ingress_proxy
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("supervisor", [False, True])
async def test_all_physical_stop_producers_keep_original_native_disposition(native_pg, tmp_path, monkeypatch, supervisor):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    stops = []
    class Client(FakeIsolatedClient):
        async def health(self):
            return False
        async def stop(self):
            stops.append(asyncio.current_task())
            raise source.error
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, Client)
    client = feature._client
    if supervisor:
        owner = create_owned_task(feature._supervise(), retain_originals=True)
        feature._supervision_task = owner
    else:
        owner = create_owned_task(feature.reload(), retain_originals=True)
    try:
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert len(stops) == 1, "quarantine repeated a native-unresolved stop"
        later = await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))
        source.assert_retained(later.error)
        assert len(stops) == 1 and any(item is client for item in feature._terminal_retirement_clients)
    finally:
        feature._supervision_task = None
        if not owner.done():
            owner.cancel()
        await await_owned_task(owner)
