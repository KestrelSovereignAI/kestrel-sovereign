"""Foreign control-shaped errors are not proof of original native failure."""
import asyncio

import pytest
from kestrel_sdk.isolated_feature import HostIngressCapabilities

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError, ExecutionCustody, bind_execution_custody_snapshot,
    is_execution_control_error,
)
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.unit.test_isolated_feature_runtime import (
    _HostIngressClient, _initialized_host_ingress_proxy,
    _assert_host_ingress_error_is_detached, _traceback_locals_reach_any,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("native_body", [False, True])
@pytest.mark.parametrize("shape", ["attrs", "typed", "cause", "group"])
async def test_foreign_native_claim_does_not_authenticate_even_in_healthy_rollback(
    native_pg, tmp_path, monkeypatch, native_body, shape,
):
    backend, observer, schema = native_pg
    secret = "foreign-forged-ingress-secret-205"
    forged = (ExecutionCommitOutcomeError("unknown") if shape == "typed"
              else RuntimeError(secret))
    forged._kestrel_unresolved_commit_outcome = "unknown"
    forged._kestrel_native_retirement_debts = ((object(), None),)
    if shape == "cause":
        wrapper = RuntimeError(secret)
        wrapper.__cause__ = forged
        forged = wrapper
    elif shape == "group":
        forged = BaseExceptionGroup(secret, [RuntimeError(secret), forged])
    calls = []
    class Client(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            calls.append(asyncio.current_task())
            if len(calls) > 1:
                return {"accepted": True}
            if native_body:
                async with backend.transaction():
                    await backend.execute("INSERT INTO effects VALUES (206, 'must really roll back')")
                    raise forged
            raise forged
    def factory(**kwargs):
        return Client(ingress_capabilities=HostIngressCapabilities(names=("private-input",)), **kwargs)
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, factory)
    scope = ExecutionCustody(GenerationFence())
    async def call():
        with bind_execution_custody_snapshot((scope,)):
            return await feature.call_host_ingress("private-input", {"secret": secret})
    try:
        outcome = await await_owned_task(create_owned_task(call(), retain_originals=True))
        assert outcome.error is not None and not is_execution_control_error(outcome.error), "foreign attributes fabricated native uncertainty"
        _assert_host_ingress_error_is_detached(outcome.error, secret=secret, external=forged)
        assert not _traceback_locals_reach_any(outcome.error, (feature, feature._client, forged, secret))
        assert not feature._host_ingress_admissions, "healthy rollback manufactured a retained failed native owner"
        assert vars(feature).get("_event_ingress_control") is None
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=206') == 0
        later = await await_owned_task(create_owned_task(call(), retain_originals=True))
        assert later.error is None and later.result == {"accepted": True} and len(calls) == 2
    finally:
        await feature.shutdown()
