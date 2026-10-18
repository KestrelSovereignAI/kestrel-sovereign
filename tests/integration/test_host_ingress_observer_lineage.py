"""A copied context alone cannot enroll an unrelated resident background task."""
import asyncio

import pytest
from kestrel_sdk.isolated_feature import HostIngressCapabilities

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_custody_snapshot
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.unit.test_isolated_feature_runtime import _HostIngressClient, _initialized_host_ingress_proxy


@pytest.mark.asyncio
async def test_completed_ingress_observer_does_not_fence_resident_background_task(native_pg, tmp_path, monkeypatch):
    backend, observer, schema = native_pg
    ready, release = asyncio.Event(), asyncio.Event()
    tasks, observations = [], []
    async def resident():
        ready.set()
        await release.wait()
        await backend.execute("INSERT INTO effects VALUES (207, 'independent resident lifetime')")
    class Client(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            (record,) = feature._host_ingress_admissions
            observations.append(record.observer)
            tasks.append(create_owned_task(resident(), retain_originals=True))
            await ready.wait()
            return {"accepted": True}
    def factory(**kwargs):
        return Client(ingress_capabilities=HostIngressCapabilities(names=("private-input",)), **kwargs)
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, factory)
    scope = ExecutionCustody(GenerationFence())
    try:
        with bind_execution_custody_snapshot((scope,)):
            assert await feature.call_host_ingress("private-input") == {"accepted": True}
        assert not feature._host_ingress_admissions
        release.set()
        result = await await_owned_task(tasks[0])
        assert result.error is None and not observations[0].native_failures
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=207') == 1
    finally:
        release.set()
        for task in tasks:
            await await_owned_task(task)
        await feature.shutdown()
