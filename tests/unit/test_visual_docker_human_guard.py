"""Human native dispatch retains the original lexical grant through every await."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC,datetime,timedelta
from types import SimpleNamespace
import pytest
from kestrel_sdk.computer_use import ComputerSession
from kestrel_sdk.resources import ResourceScope,ResourceUnsupported,ResourceOutcomeUnknown
from pydantic import SecretStr
from kestrel_sovereign.visual_computer.docker import DockerComputerDriver
from kestrel_sovereign.visual_computer.docker_human import human_operation_id
from tests.unit.test_visual_docker_dispatch_guard import Revoked,current_check


def session():
    return ComputerSession(session_id='session',scope=ResourceScope(tenant_id='fixture',principal_id='owner'),
        lease_id='lease',driver_id='local-docker-browser',surface='browser',capabilities=frozenset({'screenshot','type','takeover'}),
        state='paused',controller='human',control_epoch=1,expires_at=datetime.now(UTC)+timedelta(seconds=30))


@pytest.mark.asyncio
@pytest.mark.parametrize('operation',['open','observe','act','status','action_status','revoke'])
async def test_each_human_rpc_uses_exact_purpose_original_id_and_child(operation,monkeypatch):
    origin=asyncio.current_task();checks=[];bound=[]
    @asynccontextmanager
    async def guard(candidate,identifier):
        assert candidate==session_value and identifier==human_operation_id('control',operation,'input')
        assert asyncio.current_task() is not origin
        bound.append(True)
        yield lambda:checks.append(True)
    session_value=session()
    driver=DockerComputerDriver(docker_socket='/tmp/fixture.sock',human_guard=guard,human_status_guard=guard)
    async def rpc(sid,name,payload,**options):
        assert bound and name=='human' and payload['operation']==operation
        assert options['cleanup']==(operation in {'status','action_status','revoke'})
        options['current']();return {'proof':True}
    monkeypatch.setattr(driver,'_rpc',rpc)
    assert await driver.human_control()._call(session_value,'control',operation,{},'input')=={'proof':True}
    assert checks and not driver._workspace_borrowers


@pytest.mark.asyncio
@pytest.mark.parametrize('operation',['open','observe','act','status','action_status','revoke'])
async def test_grant_revocation_after_inspect_refuses_final_dispatch(operation,monkeypatch):
    active=[True];entered,proceed=asyncio.Event(),asyncio.Event();sent=[]
    @asynccontextmanager
    async def guard(*args):yield current_check(active)
    driver=DockerComputerDriver(docker_socket='/tmp/fixture.sock',human_guard=guard,human_status_guard=guard)
    driver._connections['session']=object()
    async def bound(*args,**kwargs):
        entered.set();await proceed.wait();return {'container_id':'c'*64}
    async def rpc(*args,**kwargs):sent.append(True);raise AssertionError('revoked human RPC reached native stdin')
    monkeypatch.setattr(driver,'_bound',bound);monkeypatch.setattr(driver.boundary,'rpc',rpc)
    task=asyncio.create_task(driver.human_control()._call(session(),'control',operation,{}))
    await entered.wait();active[0]=False;proceed.set()
    with pytest.raises(Revoked):await task
    assert not sent and not driver._workspace_borrowers


@pytest.mark.asyncio
async def test_post_acceptance_reply_loss_is_unknown_and_missing_guard_refuses(monkeypatch):
    driver=DockerComputerDriver(docker_socket='/tmp/fixture.sock')
    with pytest.raises(ResourceUnsupported):await driver.human_control()._call(session(),'control','observe',{})
    active=[True]
    @asynccontextmanager
    async def guard(*args):yield current_check(active)
    driver.human_guard=guard
    driver._connections['session']=SimpleNamespace(credential=SecretStr('fixture-only'))
    async def bound(*args,**kwargs):return {'container_id':'c'*64}
    async def rpc(cid,payload,**options):active[0]=False;return {'id':payload['id'],'result':{}}
    monkeypatch.setattr(driver,'_bound',bound);monkeypatch.setattr(driver.boundary,'rpc',rpc)
    with pytest.raises(ResourceOutcomeUnknown):await driver.human_control()._call(session(),'control','act',{})
