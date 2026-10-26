"""Real keyed initialized host + real governed Docker + shipped browser panel.

Set ARTIFACT_TEST_BROWSER_IMAGE to a locally built immutable browser image ID.
The test creates only its own local sessions and releases their exact leases.
"""
import asyncio
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
from uuid import uuid4

from cryptography.fernet import Fernet
import httpx
import pytest
import toml

from kestrel_sdk.computer_use import ActionRequest, Click, TypeText
from kestrel_sdk.payer_policy import PayerKind, PayerPolicy, PayerSpec
from tests.shared.native_shutdown_budget import shutdown_fixture_agent
from tests.shared.local_docker import LocalDockerUnavailable, local_docker_socket


async def _deny_native_materialization(client, agent, boundary, current, request, container_id, monkeypatch, phase, owner):
    """Revoke the original host policy after real inspect/native launch awaits."""
    visual=agent.features['VisualComputerFeature']
    driver=visual.runtime.drivers['local-docker-browser']
    target=ContextVar('artifact-test-original-materialization',default=False)
    rpc,bound,launch=driver._rpc,driver._bound,asyncio.create_subprocess_exec
    entered,proceed=asyncio.Event(),asyncio.Event();children=[];stdin_writes=[]
    async def selected_rpc(session_id, operation, arguments, **kwargs):
        token=target.set(operation=='workspace' and arguments.get('operation')=='commit_import')
        try:return await rpc(session_id,operation,arguments,**kwargs)
        finally:target.reset(token)
    async def delayed_bound(*args, **kwargs):
        value=await bound(*args,**kwargs)
        if target.get() and not kwargs.get('cleanup'):
            entered.set();await proceed.wait()
        return value
    async def delayed_launch(*args, **kwargs):
        process=await launch(*args,**kwargs)
        if target.get() and '/opt/kestrel-browser/rpc_client.py' in args:
            children.append(process)
            write=process.stdin.write
            def observed_write(data):
                stdin_writes.append(len(data));return write(data)
            process.stdin.write=observed_write
            entered.set();await proceed.wait()
        return process
    pending=None
    with monkeypatch.context() as patch:
        patch.setattr(driver,'_rpc',selected_rpc)
        if phase=='inspect':patch.setattr(driver,'_bound',delayed_bound)
        else:patch.setattr(asyncio,'create_subprocess_exec',delayed_launch)
        try:
            pending=asyncio.create_task(client.post('/api/artifacts/workspaces/materialize',json=request))
            async with asyncio.timeout(10):await entered.wait()
            # A new request can bind the new generation. This original one
            # must never be resurrected by transport inspection or launch.
            agent.features[owner]._generation+=1
            proceed.set();response=await pending
            assert response.status_code in {403,503},response.text
        finally:
            proceed.set()
            if pending is not None and not pending.done():
                pending.cancel();await asyncio.gather(pending,return_exceptions=True)
    assert not stdin_writes,'revoked private commit reached native stdin'
    assert all(child.returncode is not None for child in children),'native dispatch child was not physically reaped'
    key=sha256(request['workspace_id'].encode()).hexdigest()
    physical=json.loads(await boundary.invoke(['exec',container_id,'python','-c',
        "import json;from pathlib import Path;p=Path('/tmp/kestrel-browser/workspace-transfer/workspaces/"+key+"');print(json.dumps({'target_exists':p.exists()}))"]))
    assert not physical['target_exists'],'revoked host transfer physically installed a fresh workspace'
    status=await client.get('/api/artifacts/workspaces/sessions/'+current['session_id']+'/transfers/'+request['transfer_id'])
    assert status.status_code==200 and status.json()['state']=='aborted',status.text
    assert not visual.runtime.egress._calls
    return {'phase':phase,'retired_policy':owner,'transfer_id':request['transfer_id'],
        'http_status':response.status_code,'native_stdin_writes':stdin_writes,'physical_target':physical,
        'native_children_reaped':all(child.returncode is not None for child in children),'retained_cleanup':status.json()['state']}


@pytest.mark.asyncio
async def test_real_host_docker_source_destruction_panel_edit_and_pinned_materialization(tmp_path, monkeypatch):
    image=os.environ.get('ARTIFACT_TEST_BROWSER_IMAGE')
    if not image:
        pytest.skip('Real workspace journey requires ARTIFACT_TEST_BROWSER_IMAGE immutable local image ID')
    assert image.startswith('sha256:') and len(image)==71
    try:
        docker_socket=local_docker_socket()
    except LocalDockerUnavailable as error:
        pytest.skip(str(error))
    for name in tuple(os.environ):
        if name.startswith(('KESTREL_','OTEL_')) or name.endswith(('_API_KEY','_TOKEN','_SECRET')):
            monkeypatch.delenv(name,raising=False)
    from kestrel_sovereign.cli_demo import _build_smoke_env
    from kestrel_sovereign import server
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.visual_computer.runtime import VisualAction
    from kestrel_sovereign.visual_computer.docker import DockerBoundary
    import uvicorn
    home=tmp_path/'home';home.mkdir();data=tmp_path/'data';data.mkdir()
    isolated=_build_smoke_env(os.environ,home=home,data_dir=data,data_key=Fernet.generate_key().decode())
    for name,value in isolated.items():monkeypatch.setenv(name,value)
    key='workspace-disposable-host-proof-key';monkeypatch.setenv('KESTREL_API_KEY',key)
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED','1')
    config=Path(os.environ['KESTREL_DB_PATH'])/'kestrel.toml';config.parent.mkdir(parents=True,exist_ok=True)
    config.write_text(toml.dumps({'features':{'visual_computer':{'enabled':True,
        'granted_actions':[action.value for action in VisualAction],
        'providers':{'local-docker-browser':{'docker_socket':docker_socket,'image':'local-worker@'+image}},
        'drivers':{'local-docker-browser':{'docker_socket':docker_socket}}}}}))
    llm=LLMService();defaults=PayerPolicy.host_env_default()
    policy=PayerPolicy(**{field:PayerSpec(vendor=getattr(defaults,field).vendor,kind=PayerKind.NONE) for field in ('llm','storage','compute','tools','comms')})
    agent=KestrelAgent(did='did:test:artifact-workspace-docker',storage_path=str(tmp_path/'agent.db'),llm_service=llm,
        payer_policy=policy,allowed_features={'ArtifactsFeature','VisualComputerFeature'})
    app=server.app;lifespan=app.router.lifespan_context
    monkeypatch.setattr(app.state,'agent',agent,raising=False);monkeypatch.setattr(app.state,'agent_manager',None,raising=False)
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    runner=None;serving=None;created=[];containers=[];released=[]
    boundary=DockerBoundary(docker_socket)
    try:
        await agent.initialize()
        assert agent.features['VisualComputerFeature'].runtime is not None
        assert {'artifact_publish_workspace','artifact_materialize_workspace','artifact_workspace_status'} <= {item['function']['name'] for item in agent._build_all_tools()}
        server._mount_feature_routers(app)
        runner=uvicorn.Server(uvicorn.Config(app,log_config=None,access_log=False,lifespan='off'))
        serving=asyncio.create_task(runner.serve(sockets=[sock]))
        async with asyncio.timeout(15):
            while not runner.started:await asyncio.sleep(.01)
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}',headers={'X-API-Key':key},timeout=45) as client:
            async def start():
                now=datetime.now(UTC)
                response=await client.post('/api/visual-computer/requests',json={'request_id':uuid4().hex,
                    'provider_id':'local-docker-browser','profile_id':'offline-v1','image':'local-worker@'+image,
                    'surface':'browser','required_capabilities':['screenshot','click','type','takeover','files'],'location':'local',
                    'requested_at':now.isoformat(),'expires_at':(now+timedelta(minutes=5)).isoformat(),
                    'idle_ttl_seconds':120,'ready_timeout_seconds':25,'max_hourly_cost_usd':'0','max_total_cost_usd':'0'})
                assert response.status_code==200,response.text
                current=response.json();created.append(current['session_id'])
                status=(await client.get('/api/visual-computer/sessions/'+current['session_id'])).json()
                containers.append(status['lease']['provider_resource_id'])
                return current
            async def action(current,effect):
                observation=(await client.get('/api/visual-computer/sessions/'+current['session_id']+'/observation')).json()
                request=ActionRequest(action_id=uuid4().hex,session_id=current['session_id'],observation_id=observation['frame']['observation_id'],
                    control_epoch=current['control_epoch'],not_after=datetime.now(UTC)+timedelta(seconds=10),action=effect)
                response=await client.post('/api/visual-computer/sessions/'+current['session_id']+'/actions',json=request.model_dump(mode='json'))
                assert response.status_code==200,response.text
            async def pause(current):
                sid=current['session_id']
                response=await client.post('/api/visual-computer/sessions/'+sid+'/stop',json={'request_id':uuid4().hex,
                    'session_id':sid,'expected_control_epoch':current['control_epoch'],'operation':'pause'})
                assert response.status_code==200,response.text
                return (await client.get('/api/visual-computer/sessions/'+sid)).json()['session']
            def request(current,transfer_id,workspace='browser'):
                return {'transfer_id':transfer_id,'session_id':current['session_id'],'lease_id':current['lease_id'],
                    'control_epoch':current['control_epoch'],'workspace_id':workspace,'expires_at':current['expires_at'],
                    'limits':{'max_files':4,'max_file_bytes':131072,'max_total_bytes':262144}}
            try:
                first=await start();await action(first,Click(x=20,y=20));await action(first,TypeText(text='Broker durable source'))
                first=await pause(first)
                # Each refusal occurs AFTER the real worker reserves 2MiB.
                # More than eight failed IDs must not exhaust its16MiB quota.
                failed_exports=[]
                for index in range(9):
                    transfer_id='refused-export-'+str(index)
                    rejected={**request(first,transfer_id),'paths':['not-browser-document'],'entrypoint':'not-browser-document'}
                    rejected['limits']['max_total_bytes']=2*1024*1024
                    refusal=await client.post('/api/artifacts/workspaces/publish',json={'request':rejected,'title':'Refused source','publication_key':'refused-key-'+str(index)})
                    assert refusal.status_code!=200,refusal.text
                    status=await client.get('/api/artifacts/workspaces/sessions/'+first['session_id']+'/transfers/'+transfer_id)
                    assert status.status_code==200 and status.json()['state']=='released',status.text
                    failed_exports.append(transfer_id)
                physical=json.loads(await boundary.invoke(['exec',containers[0],'python','-c',
                    "import json;from pathlib import Path;p=Path('/tmp/kestrel-browser/workspace-transfer');r=[json.loads(f.read_text()) for f in (p/'records').iterdir()];print(json.dumps({'reserved':sum(v['reserved'] for v in r),'export_trees':len(list((p/'exports').iterdir()))}))"]))
                assert physical=={'reserved':0,'export_trees':0},physical
                print(json.dumps({'failed_exports_released':failed_exports,'physical_after_refusals':physical}))
                selected={**request(first,'export-original'),'paths':['document.html'],'entrypoint':'document.html'}
                exported=await client.post('/api/artifacts/workspaces/publish',json={'request':selected,
                    'title':'Broker durable output','publication_key':'broker-original'})
                assert exported.status_code==200,exported.text
                first_ref=exported.json()['receipt']['ref']
                release=await client.post('/api/visual-computer/sessions/'+first['session_id']+'/release')
                assert release.status_code==200 and release.json()['released'],release.text
                released.append(first['session_id'])
                present=await boundary.invoke(['container','ls','--all','--filter','id='+containers[0],'--format','{{.ID}}'])
                assert not present.strip(),'source physical container remains'
                source=await client.get(f"/api/artifacts/{first_ref['artifact_id']}/revisions/{first_ref['revision_id']}/file",params={'path':'document.html'})
                assert b'Broker durable source' in source.content
                child=await asyncio.create_subprocess_exec('node','tests/frontend/artifact_workspace.browser.cjs',cwd=Path(__file__).resolve().parents[2],
                    env={**os.environ,'ARTIFACT_TEST_URL':f'http://127.0.0.1:{port}','ARTIFACT_TEST_KEY':key},
                    stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
                try:stdout,stderr=await asyncio.wait_for(child.communicate(),60)
                except BaseException:
                    child.kill();await child.communicate();raise
                assert child.returncode==0,(stdout+stderr).decode()
                browser=json.loads(stdout);assert browser['verdict']=='PASS'
                second_ref={**first_ref,'revision_id':browser['revision_id']}
                second=(await client.get(f"/api/artifacts/{second_ref['artifact_id']}/revisions/{second_ref['revision_id']}")).json()
                destination=await pause(await start())
                dispatch=[]
                for phase in ('inspect','launch'):
                    for owner in ('ArtifactsFeature','VisualComputerFeature'):
                        name='denied-'+phase+'-'+owner.lower()
                        denied={**request(destination,name,name),'source':second_ref,'manifest':second['manifest']}
                        dispatch.append(await _deny_native_materialization(client,agent,boundary,destination,denied,
                            containers[-1],monkeypatch,phase,owner))
                incoming={**request(destination,'import-edited','fresh-edited'),'source':second_ref,'manifest':second['manifest']}
                installed=await client.post('/api/artifacts/workspaces/materialize',json=incoming)
                assert installed.status_code==200,installed.text
                assert installed.json()['materialization']['source']==second_ref
                outgoing={**request(destination,'export-third','fresh-edited'),'paths':['document.html'],'entrypoint':'document.html'}
                third=await client.post('/api/artifacts/workspaces/publish',json={'request':outgoing,'title':'Broker durable output',
                    'publication_key':'broker-third','artifact_id':second_ref['artifact_id'],'expected_revision_id':second_ref['revision_id']})
                assert third.status_code==200,third.text
                third_ref=third.json()['receipt']['ref']
                result=await client.get(f"/api/artifacts/{third_ref['artifact_id']}/revisions/{third_ref['revision_id']}/file",params={'path':'document.html'})
                assert result.content.decode()==browser['source_text'] and third_ref['artifact_id']==first_ref['artifact_id']
                print(json.dumps({'verdict':'PASS','path':'real host+Docker export → physical source destruction → shipped panel edit revision2 → exact fresh workspace materialization → export CAS revision3',
                    'image':image,'containers':containers,'revisions':[first_ref,second_ref,third_ref],'dispatch_revocation':dispatch,
                    'browser':{k:v for k,v in browser.items() if k!='source_text'}}))
            finally:
                for sid in created:
                    if sid not in released:
                        response=await client.post('/api/visual-computer/sessions/'+sid+'/release')
                        assert response.status_code==200 and response.json()['released'],response.text
                        released.append(sid)
                for cid in containers:
                    assert not (await boundary.invoke(['container','ls','--all','--filter','id='+cid,'--format','{{.ID}}'])).strip()
    finally:
        if runner is not None:runner.should_exit=True
        if serving is not None:await asyncio.wait_for(serving,15)
        sock.close();server._unmount_feature_routers(app);app.router.lifespan_context=lifespan
        await shutdown_fixture_agent(agent);await llm.close()
        assert serving is None or serving.done()
        assert sock.fileno()==-1 and set(released)==set(created)
