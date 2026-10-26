"""Concrete native facts require their original live invocation and settlement."""
from datetime import UTC, datetime, timedelta

import pytest
from kestrel_sdk.computer_use import ActionRequest, ComputerSession
from kestrel_sdk.resources import ResourceNotFound, ResourceOutcomeUnknown, ResourceScope
from kestrel_sovereign.visual_computer.dispatch import NativeActionNotDispatched, original_dispatch
from kestrel_sovereign.visual_computer.docker import DockerComputerDriver, PROVIDER


def originals():
    session=ComputerSession(session_id='s',scope=ResourceScope(tenant_id='t',principal_id='p'),
        lease_id='lease',driver_id=PROVIDER,surface='browser',capabilities=frozenset({'screenshot','click'}),
        state='ready',controller='agent',control_epoch=0,expires_at=datetime.now(UTC)+timedelta(seconds=30))
    request=ActionRequest(action_id='a',session_id='s',observation_id='f',control_epoch=0,
        not_after=datetime.now(UTC)+timedelta(seconds=10),action={'kind':'click','x':1,'y':1})
    return session,request


@pytest.mark.asyncio
async def test_native_non_dispatch_fact_cannot_be_copied_rebound_or_used_after_retirement(monkeypatch):
    session,request=originals();driver=DockerComputerDriver(docker_socket='/tmp/owned-fact.sock')
    driver._connections['s']=object()
    async def missing(*args,**kwargs):raise ResourceNotFound('owned preflight refusal')
    monkeypatch.setattr(driver,'_bound',missing)
    with original_dispatch(session,request) as origin:
        with pytest.raises(NativeActionNotDispatched) as caught:await driver.act(session,request)
        fact=caught.value;origin.require(fact)
        with pytest.raises(ResourceOutcomeUnknown):origin.require(NativeActionNotDispatched(fact._trace,fact.original))
    with pytest.raises(ResourceOutcomeUnknown):origin.require(fact)
    with original_dispatch(session,request) as next_origin:
        with pytest.raises(ResourceOutcomeUnknown):next_origin.require(fact)
        with pytest.raises(ResourceOutcomeUnknown,match='invocation_not_original'):
            await driver.act(session,request.model_copy(update={'observation_id':'changed'}))
    with pytest.raises(ResourceNotFound):await driver.act(session,request)


@pytest.mark.asyncio
async def test_done_native_task_without_physical_settlement_is_not_negative_proof(monkeypatch):
    session,request=originals();driver=DockerComputerDriver(docker_socket='/tmp/owned-fact.sock')
    driver._connections['s']=object()
    async def unsettled(*args,**kwargs):
        kwargs['_native_owner']['entered']=True
        raise ResourceOutcomeUnknown('native closer has not acknowledged settlement')
    monkeypatch.setattr(driver.boundary,'_invoke',unsettled)
    async def preflight(*args,**kwargs):return await driver.boundary.invoke(['container','inspect','c'*64])
    monkeypatch.setattr(driver,'_bound',preflight)
    with original_dispatch(session,request) as origin:
        with pytest.raises(ResourceOutcomeUnknown):await driver.act(session,request)
        assert origin.trace.proof is None


@pytest.mark.asyncio
async def test_possible_native_stdin_write_cannot_become_a_negative_receipt(tmp_path,monkeypatch):
    from pydantic import SecretStr
    from types import SimpleNamespace
    import sys
    session,request=originals();driver=DockerComputerDriver(docker_socket='/tmp/owned-fact.sock')
    driver._connections['s']=SimpleNamespace(credential=SecretStr('a'*64))
    async def bound(*args,**kwargs):return {'container_id':'c'*64}
    monkeypatch.setattr(driver,'_bound',bound)
    program=tmp_path/'owned-native';received=tmp_path/'received'
    program.write_text(f'#!{sys.executable}\nimport sys\nfrom pathlib import Path\npayload=sys.stdin.buffer.read();Path({str(received)!r}).write_bytes(payload)\nprint(\'{{"id":"wrong"}}\')\n')
    program.chmod(0o700);driver.boundary.executable=str(program)
    with original_dispatch(session,request) as origin:
        with pytest.raises(ResourceOutcomeUnknown):await driver.act(session,request)
        assert received.stat().st_size>0 and origin.trace.proof is None


@pytest.mark.asyncio
async def test_rpc_without_original_native_settlement_never_mints_negative_proof(monkeypatch):
    from pydantic import SecretStr
    from types import SimpleNamespace
    from kestrel_sdk.resources import ResourceConflict
    session,request=originals();driver=DockerComputerDriver(docker_socket='/tmp/owned-fact.sock')
    driver._connections['s']=SimpleNamespace(credential=SecretStr('a'*64))
    async def bound(*args,**kwargs):return {'container_id':'c'*64}
    async def arbitrary(*args,**kwargs):raise ResourceConflict('an arbitrary RPC refusal is not native settlement')
    monkeypatch.setattr(driver,'_bound',bound);monkeypatch.setattr(driver.boundary,'rpc',arbitrary)
    with original_dispatch(session,request) as origin:
        with pytest.raises(ResourceConflict):await driver.act(session,request)
        assert origin.trace.proof is None
