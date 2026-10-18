"""An actual stopped caller scope does not poison the shared feature runtime."""
import pytest
from contextlib import asynccontextmanager
from kestrel_sdk.isolated_feature import HostIngressCapabilities

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_custody_snapshot, current_execution_custody, is_execution_control_error
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.unit.test_isolated_feature_runtime import _HostIngressClient, _initialized_host_ingress_proxy


@pytest.mark.asyncio
async def test_ingress_rejects_original_stopped_scope_but_accepts_distinct_live_scope(native_pg, tmp_path, monkeypatch):
    def factory(**kwargs):
        return _HostIngressClient(ingress_capabilities=HostIngressCapabilities(names=("private-input",)), **kwargs)
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, factory)
    client = feature._client
    stopped = ExecutionCustody(GenerationFence())
    actual_admit = feature._traffic_gate.admit
    @asynccontextmanager
    async def admit(*args, **kwargs):
        async with actual_admit(*args, **kwargs) as value:
            if any(scope is stopped for scope in current_execution_custody()):
                stopped.revoke("actual original scoped work retired after admission")
            yield value
    monkeypatch.setattr(feature._traffic_gate, "admit", admit)
    async def call(scope):
        with bind_execution_custody_snapshot((scope,)):
            return await feature.call_host_ingress("private-input")
    try:
        first = await await_owned_task(create_owned_task(call(stopped), retain_originals=True))
        assert is_execution_control_error(first.error) and not client.ingress_calls
        second = await await_owned_task(create_owned_task(call(ExecutionCustody(GenerationFence())), retain_originals=True))
        assert second.error is None and second.result == {"accepted": True}
        assert len(client.ingress_calls) == 1
        assert not feature._host_ingress_admissions and feature._host_ingress_public_control is None
        assert vars(feature).get("_event_ingress_control") is None
    finally:
        await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))
