"""Retained workspace authority reaches the same native stdin guard as input."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC,datetime,timedelta
from types import SimpleNamespace

import pytest
from kestrel_sdk.computer_use import ComputerSession,ControlRequest
from kestrel_sdk.resources import ResourceScope,ResourceUnsupported,ResourceOutcomeUnknown
from pydantic import SecretStr
from kestrel_sovereign.visual_computer.docker import DockerComputerDriver,PROVIDER,computer_driver_factory
from tests.unit.test_visual_docker_dispatch_guard import Revoked,current_check


def session():
    return ComputerSession(session_id="session",scope=ResourceScope(tenant_id="fixture",principal_id="owner"),
        lease_id="lease",driver_id=PROVIDER,surface="browser",capabilities=frozenset({"files","screenshot"}),
        state="paused",controller="none",control_epoch=1,expires_at=datetime.now(UTC)+timedelta(seconds=30))


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,arguments",[
    ("prepare_export",{"request":{"transfer_id":"original"}}),("prepare_import",{"request":{"transfer_id":"original"}}),
    ("read_export",{"snapshot_id":"original","path":"draft.txt"}),
    ("write_import_file",{"transfer_id":"original","file":{},"content_base64":""}),
    ("commit_import",{"transfer_id":"original"}),("status",{"transfer_id":"original"})])
async def test_data_rpc_borrows_exact_original_id_in_owned_child(operation,arguments,monkeypatch):
    value=session();origin=asyncio.current_task();entered=[];checks=[]
    @asynccontextmanager
    async def guard(candidate,identifier):
        assert candidate==value and identifier=="original" and asyncio.current_task() is not origin
        entered.append(asyncio.current_task())
        try:yield lambda:checks.append(True)
        finally:entered.remove(asyncio.current_task())
    driver=computer_driver_factory(settings={"docker_socket":"/tmp/fixture.sock","workspace_guard":guard})
    async def rpc(session_id,name,payload,*,cleanup,current):
        assert session_id==value.session_id and name=="workspace" and payload["operation"]==operation and not cleanup
        assert entered==[asyncio.current_task()]
        current();return {"synthetic":True}
    monkeypatch.setattr(driver,"_rpc",rpc)
    assert await driver.workspace_transfer()._call(value,operation,arguments)=={"synthetic":True}
    assert checks and not entered and driver._workspace_borrowers=={}


@pytest.mark.asyncio
async def test_revoked_original_transfer_after_inspect_cannot_dispatch(monkeypatch):
    active,entered,proceed=[True],asyncio.Event(),asyncio.Event()
    @asynccontextmanager
    async def guard(candidate,identifier):yield current_check(active)
    driver=DockerComputerDriver(docker_socket="/tmp/fixture.sock",workspace_guard=guard)
    driver._connections["session"]=object();dispatched=[]
    async def bound(*args,**kwargs):
        entered.set();await proceed.wait();return {"container_id":"c"*64}
    async def rpc(*args,**kwargs):dispatched.append(args);raise AssertionError("revoked workspace reached native dispatch")
    monkeypatch.setattr(driver,"_bound",bound);monkeypatch.setattr(driver.boundary,"rpc",rpc)
    pending=asyncio.create_task(driver.workspace_transfer().commit_import(session(),"original"))
    await entered.wait();active[0]=False;proceed.set()
    with pytest.raises(Revoked):await pending
    assert dispatched==[] and driver._workspace_borrowers=={}


@pytest.mark.asyncio
async def test_revoked_private_reply_after_native_acceptance_remains_unknown(monkeypatch):
    active=[True]
    @asynccontextmanager
    async def guard(*args):yield current_check(active)
    driver=DockerComputerDriver(docker_socket="/tmp/fixture.sock",workspace_guard=guard)
    driver._connections["session"]=SimpleNamespace(credential=SecretStr("SYNTHETIC"))
    async def bound(*args,**kwargs):return {"container_id":"c"*64}
    async def rpc(cid,payload,**options):
        active[0]=False
        return {"id":payload["id"],"result":{"accepted_original_id":"original"}}
    monkeypatch.setattr(driver,"_bound",bound);monkeypatch.setattr(driver.boundary,"rpc",rpc)
    with pytest.raises(ResourceOutcomeUnknown,match="driver_effect_outcome_unknown"):
        await driver.workspace_transfer()._call(session(),"commit_import",{"transfer_id":"original"})
    assert driver._workspace_borrowers=={}


@pytest.mark.asyncio
@pytest.mark.parametrize("operation",["release_export","abort_import"])
async def test_retained_cleanup_never_borrows_new_data_authority(operation,monkeypatch):
    @asynccontextmanager
    async def guard(*args):
        raise AssertionError("cleanup acquired fresh data authority")
        yield
    driver=DockerComputerDriver(docker_socket="/tmp/fixture.sock",workspace_guard=guard)
    async def rpc(*args,cleanup,current):assert cleanup and current is None;return {"retired":True}
    monkeypatch.setattr(driver,"_rpc",rpc)
    assert await driver.workspace_transfer()._call(session(),operation,{"transfer_id":"original"})=={"retired":True}


@pytest.mark.asyncio
async def test_priority_control_physically_joins_workspace_borrower(monkeypatch):
    value=session();entered,closed=asyncio.Event(),asyncio.Event()
    @asynccontextmanager
    async def guard(*args):yield lambda:None
    driver=DockerComputerDriver(docker_socket="/tmp/fixture.sock",workspace_guard=guard)
    async def rpc(session_id,operation,payload,**options):
        if operation=="workspace":
            entered.set()
            try:await asyncio.Event().wait()
            finally:closed.set()
        assert operation=="control" and closed.is_set()
        return value.model_copy(update={"control_epoch":2,"controller":"human"}).model_dump(mode="json")
    monkeypatch.setattr(driver,"_rpc",rpc)
    pending=asyncio.create_task(driver.workspace_transfer().commit_import(value,"original"))
    await entered.wait()
    await driver.control(value,ControlRequest(request_id="takeover",session_id=value.session_id,expected_control_epoch=1,operation="takeover"))
    await asyncio.gather(pending,return_exceptions=True)
    assert pending.done() and closed.is_set() and driver._workspace_borrowers=={}


def test_factory_rejects_invalid_workspace_guard():
    with pytest.raises(ResourceUnsupported):
        computer_driver_factory(settings={"docker_socket":"/tmp/fixture.sock","workspace_guard":False})
