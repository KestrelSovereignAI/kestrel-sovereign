"""Fixture ownership/refusal guards; only the daemon proof establishes isolation."""
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading

import pytest

from kestrel_sdk.resources import ResourceNotFound
from kestrel_sovereign.visual_computer.docker import DockerCommandError
from scripts.visual_owned_peers import OwnedNetworkPeers, LABEL, CONTROL


def fixture():
    return {"proof_id":"a"*32,"server_id":"b"*64,"network_id":"c"*64,"targets":[
        {"family":"ipv4","host":"172.30.0.2","port":18080},
        {"family":"ipv6","host":"fd00:1234::2","port":18080}]}


def peer(tmp_path,invoke=None):
    async def unused(*a,**k):pytest.fail("unexpected native operation")
    value=OwnedNetworkPeers(invoke or unused,"sha256:"+"d"*64,"a"*32,tmp_path/"peers",image_label="owned-image")
    value.state.update(network_id="c"*64,fixture=fixture())
    return value


def test_refusal_result_must_identify_both_exact_owned_endpoints(tmp_path):
    value=peer(tmp_path)
    result={"network_fixture":fixture(),"refused_peers":[{**target,"refused":True} for target in fixture()["targets"]]}
    value.verify_refused(result)
    for change in ({"refused_peers":[]},{"refused_peers":[{**target,"refused":False} for target in fixture()["targets"]]},
                   {"network_fixture":{**fixture(),"server_id":"e"*64}}):
        with pytest.raises(RuntimeError,match="differs"):value.verify_refused({**result,**change})


@pytest.mark.parametrize("field,value",[("Id","e"*64),("Labels",{LABEL:"other"}),
    ("Internal",False),("EnableIPv6",False),("Driver","host")])
def test_network_data_admission_requires_exact_owned_internal_dual_stack(tmp_path,field,value):
    peers=peer(tmp_path)
    record={"Id":"c"*64,"Name":peers.network_name,"Labels":{LABEL:peers.identity},"Internal":True,"EnableIPv6":True,"Driver":"bridge"}
    peers._network(record)
    with pytest.raises(RuntimeError):peers._network({**record,field:value})


@pytest.mark.asyncio
async def test_lost_network_create_response_can_only_retire_the_original_owned_network(tmp_path):
    calls=[];peers=None;removed=False
    async def invoke(args,**options):
        nonlocal removed
        calls.append(args)
        if args[0]=='container':raise ResourceNotFound('absent')
        if args[:2]==['network','inspect']:
            if removed:raise ResourceNotFound('absent')
            return json.dumps([{'Id':'c'*64,'Name':peers.network_name,'Labels':{LABEL:peers.identity},'Containers':{}}]).encode()
        if args==['network','rm','c'*64]:removed=True;return b''
        pytest.fail(str(args))
    peers=peer(tmp_path,invoke);peers.state['network_id']=None
    await peers.close()
    assert ['network','rm','c'*64] in calls and peers.state['cleanup_confirmed']


@pytest.mark.asyncio
async def test_cleanup_rejects_replacement_owner_and_unknown_daemon_failure(tmp_path):
    calls=[];peers=None
    async def foreign(args,**options):
        calls.append(args)
        if args[0]=='container':raise ResourceNotFound('absent')
        return json.dumps([{'Id':'e'*64,'Name':peers.network_name,'Labels':{LABEL:'foreign'},'Containers':{}}]).encode()
    peers=peer(tmp_path,foreign)
    with pytest.raises(RuntimeError,match='identity'):await peers.close()
    assert not any('rm' in command for command in calls) and not peers.state['cleanup_confirmed']
    async def unavailable(*args,**options):raise DockerCommandError('unknown',diagnostics=b'Cannot connect: No such file or directory')
    peers.invoke=unavailable
    with pytest.raises(DockerCommandError):await peers.close()
    assert not peers.state['cleanup_confirmed']


def native_probe(monkeypatch):
    directory=Path(__file__).parents[2]/'docker/visual-browser'
    monkeypatch.syspath_prepend(str(directory))
    spec=importlib.util.spec_from_file_location('owned_network_probe',directory/'sandbox_probe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('target',[{'family':'ipv6','host':'2606:4700:4700::1111','port':18080},
    {'family':'ipv6','host':'::1','port':18080},{'family':'ipv6','host':'fd00:1234::2','port':0}])
def test_native_probe_validates_all_facts_before_any_connect(monkeypatch,target):
    module=native_probe(monkeypatch)
    monkeypatch.setattr(module.socket,'socket',lambda *a,**k:pytest.fail('invalid peer reached data plane'))
    value=fixture();value['targets'][1]=target
    with pytest.raises(RuntimeError):module.refuse_owned_peers(value)


def test_native_refusal_uses_only_the_supplied_owned_pair(monkeypatch):
    module=native_probe(monkeypatch);attempts=[]
    class RefusedSocket:
        def __init__(self,family):self.family=family
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def settimeout(self,value):assert value==.2
        def connect(self,address):attempts.append((self.family,address));raise OSError('network unreachable')
    monkeypatch.setattr(module.socket,'socket',RefusedSocket)
    result=module.refuse_owned_peers(fixture())
    assert [address for _,address in attempts]==[(target['host'],target['port']) for target in fixture()['targets']]
    assert all(target['refused'] for target in result['refused_peers'])


def test_positive_control_rejects_a_live_peer_with_the_wrong_original_nonce():
    with socket.socket(socket.AF_INET) as listener:
        listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(3)
        def serve():
            connection,_=listener.accept()
            with connection:connection.sendall(b'foreign-peer')
        thread=threading.Thread(target=serve,daemon=True);thread.start()
        selected={'targets':[{'family':'ipv4','host':'127.0.0.1','port':listener.getsockname()[1]}]}
        try:
            completed=subprocess.run([sys.executable,'-c',CONTROL,json.dumps(selected),'original-owned-nonce'],
                capture_output=True,timeout=5)
            assert completed.returncode!=0 and b'exact nonce' in completed.stderr
        finally:
            thread.join(timeout=4)
            assert not thread.is_alive()
