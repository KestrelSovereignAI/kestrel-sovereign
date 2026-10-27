"""Production initialized feature/HTTP/component; no model or genesis invoke."""
import asyncio
import base64
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import re
import socket
import subprocess
from uuid import uuid4

import httpx
import pytest


@pytest.mark.asyncio
async def test_initialized_manual_browser_and_safe_artifact_after_destroy(tmp_path, monkeypatch, caplog):
    image = os.environ.get('HUMAN_TEST_WORKER_IMAGE')
    destination = os.environ.get('HUMAN_TEST_EVIDENCE_DIR')
    if not image or not destination:
        pytest.skip('Explicit owned worker image and private evidence directory are required')
    assert re.fullmatch(r'sha256:[0-9a-f]{64}', image)
    project_root = Path(__file__).resolve().parents[2]
    output = Path(destination)
    assert output.is_dir()
    assert (project_root/'node_modules/playwright/package.json').is_file()
    from cryptography.fernet import Fernet
    from kestrel_sdk.payer_policy import PayerKind, PayerPolicy, PayerSpec
    from kestrel_sovereign.cli_demo import _build_smoke_env
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.features.visual_computer import feature as visual_feature
    from kestrel_sovereign.visual_computer.runtime import VisualAction
    from kestrel_sovereign.visual_computer.docker import PROVIDER
    from tests.shared.native_shutdown_budget import shutdown_fixture_agent
    import uvicorn

    (tmp_path/'home').mkdir();(tmp_path/'data').mkdir()
    for name, value in _build_smoke_env({}, home=tmp_path/'home', data_dir=tmp_path/'data', data_key=Fernet.generate_key().decode()).items():
        monkeypatch.setenv(name, value)
    key = 'human-browser-disposable-'+uuid4().hex
    monkeypatch.setenv('KESTREL_API_KEY', key)
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED','1')
    docker_socket='/Users/jason/.docker/run/docker.sock'
    config={'enabled':True,'human_control_enabled':True,'granted_actions':[a.value for a in VisualAction],
        'providers':{PROVIDER:{'docker_socket':docker_socket,'image':'local-worker@'+image,'profile_id':'offline-v1'}},
        'drivers':{PROVIDER:{'docker_socket':docker_socket}}}
    monkeypatch.setattr(visual_feature,'load_section',lambda section:{'visual_computer':config})
    policy0=PayerPolicy.host_env_default()
    policy=PayerPolicy(**{f:PayerSpec(vendor=getattr(policy0,f).vendor,kind=PayerKind.NONE) for f in ('llm','storage','compute','tools','comms')})
    llm=LLMService(agent_data_dir=tmp_path/'llm')
    model_attempts=[]
    async def forbidden_model(*args,**kwargs):
        model_attempts.append(True)
        raise AssertionError('manual proof must never call model generation')
    monkeypatch.setattr(llm,'generate_with_messages',forbidden_model)
    agent=KestrelAgent(did='did:test:human-browser',storage_path=str(tmp_path/'agent.db'),
        llm_service=llm,payer_policy=policy,allowed_features={'VisualComputerFeature','ArtifactsFeature'})
    from kestrel_sovereign import server
    app=server.app;lifespan=app.router.lifespan_context
    monkeypatch.setattr(app.state,'agent',agent,raising=False)
    monkeypatch.setattr(app.state,'agent_manager',None,raising=False)
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    runner=serving=None;sid=cid=None
    state={'model_or_genesis_invoke':False,'source_root':str(Path.cwd()),'image':image}
    def record():(output/'browser-host-custody.json').write_text(json.dumps(state,indent=2)+'\n')
    try:
        await agent.initialize()
        assert {'visual_computer_step','visual_plan','artifact_publish_text'} <= {t['function']['name'] for t in agent._build_all_tools()}
        assert not any('human' in t['function']['name'] for t in agent._build_all_tools())
        server._mount_feature_ui_assets(app);server._mount_feature_routers(app)
        runner=uvicorn.Server(uvicorn.Config(app,log_config=None,access_log=False,lifespan='off'))
        serving=asyncio.create_task(runner.serve(sockets=[sock]))
        async with asyncio.timeout(15):
            while not runner.started:await asyncio.sleep(.01)
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}',trust_env=False,headers={'X-API-Key':key},timeout=30) as client:
            now=datetime.now(UTC)
            admitted=await client.post('/api/visual-computer/requests',json={'request_id':uuid4().hex,'provider_id':PROVIDER,
                'profile_id':'offline-v1','image':'local-worker@'+image,'surface':'browser','location':'local',
                'required_capabilities':['screenshot','click','type','key','takeover','files'],
                'requested_at':now.isoformat(),'expires_at':(now+timedelta(minutes=10)).isoformat(),
                'idle_ttl_seconds':300,'ready_timeout_seconds':30,'max_hourly_cost_usd':'0','max_total_cost_usd':'0'})
            assert admitted.status_code==200,admitted.text
            session=admitted.json();sid=session['session_id'];base=f'/api/visual-computer/sessions/{sid}'
            status=await client.get(base);assert status.status_code==200,status.text
            cid=status.json()['lease']['provider_resource_id'];state.update(session_id=sid,container_id=cid);record()
            async def control(suffix,operation,epoch,**extra):
                res=await client.post(base+'/'+suffix,json={'request_id':uuid4().hex,'session_id':sid,'expected_control_epoch':epoch,'operation':operation,**extra})
                assert res.status_code==200,res.text
                return res.json()
            async def agent_input(action,epoch):
                viewed=await client.get(base+'/observation');assert viewed.status_code==200,viewed.text
                req={'action_id':uuid4().hex,'session_id':sid,'observation_id':viewed.json()['frame']['observation_id'],
                     'control_epoch':epoch,'not_after':(datetime.now(UTC)+timedelta(seconds=12)).isoformat(),'action':action}
                res=await client.post(base+'/actions',json=req);assert res.status_code==200,res.text
                assert res.json()['outcome']=='completed'
            await agent_input({'kind':'click','x':20,'y':20},0)
            await agent_input({'kind':'type','text':'Public fixture content'},0)
            await control('takeover','takeover',0)
            secret='PRIVATE-HUMAN-'+secrets.token_hex(8)
            async def browser(script, extra):
                child_env={k:v for k,v in os.environ.items() if k in {'HOME','PATH','LANG','LC_ALL','TMPDIR'}}
                if os.environ.get('HUMAN_TEST_BROWSER_CACHE'):
                    child_env['PLAYWRIGHT_BROWSERS_PATH']=os.environ['HUMAN_TEST_BROWSER_CACHE']
                child=await asyncio.create_subprocess_exec('node',str(project_root/'tests/frontend'/script),stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,env=child_env)
                try:
                    raw,err=await asyncio.wait_for(child.communicate(json.dumps({'origin':f'http://127.0.0.1:{port}',
                        'key':key,'secret':secret,'output':str(output),'stylesheet_sha256':sha256((project_root/'kestrel_sovereign/static/index.css').read_bytes()).hexdigest(),**extra}).encode()),45)
                except BaseException:
                    child.kill();await child.communicate();raise
                (output/(script+'.stdout')).write_bytes(raw);(output/(script+'.stderr')).write_bytes(err)
                assert child.returncode==0,err.decode()
                return json.loads(raw)
            before=(await agent._raw_storage.db.fetchone('SELECT COUNT(*) FROM conversation_history'))[0]
            manual=await browser('visual_human_product.browser.cjs',{})
            state.update(actual_component=manual,node_child_joined=True,secret_sha256=sha256(secret.encode()).hexdigest());record()
            status=await client.get(base);assert secret not in status.text
            assert (await client.get(base+'/observation')).status_code==409
            assert (await client.post(base+'/plan',json={'prompt':'must not see manual screen'})).status_code==409
            assert (await client.get(base+'/actions/'+manual['actions'][1]['action_id'])).status_code==404
            await control('stop','pause',1)
            export={'transfer_id':'private-'+uuid4().hex,'session_id':sid,'lease_id':session['lease_id'],'control_epoch':2,
                'workspace_id':'browser','expires_at':session['expires_at'],'limits':{'max_files':1,'max_file_bytes':131072,'max_total_bytes':131072},
                'paths':['document.html'],'entrypoint':'document.html'}
            res=await client.post('/api/artifacts/workspaces/publish',json={'request':export,'title':'must refuse private manual state','publication_key':'private-'+uuid4().hex})
            assert res.status_code in {403,409},res.text
            state['private_recording_denied_after_stop']=True
            await control('takeover','takeover',2)
            clean='clean-'+uuid4().hex
            opened=await client.post(base+'/human/'+clean);assert opened.status_code==200,opened.text
            lease=opened.json()
            async def human_input(effect):
                viewed=await client.get(base+'/human/'+clean+'/observation');assert viewed.status_code==200,viewed.text
                req={'control_id':clean,'consumer_id':lease['request']['consumer_id'],'action':{'action_id':uuid4().hex,
                    'session_id':sid,'observation_id':viewed.json()['frame']['observation_id'],'control_epoch':3,
                    'not_after':(datetime.now(UTC)+timedelta(seconds=10)).isoformat(),'action':effect}}
                res=await client.post(base+'/human/'+clean+'/actions',json=req);assert res.status_code==200,res.text
                assert res.json()['receipt']['outcome']=='completed'
            await human_input({'kind':'key','key':'Control+A'})
            await human_input({'kind':'type','text':'Safe to share'})
            denied=await client.post(base+'/handback',json={'request_id':uuid4().hex,'session_id':sid,'expected_control_epoch':3,'operation':'resume'})
            assert denied.status_code==409
            await control('handback','resume',3,share_screen=True)
            shared=await client.get(base+'/observation');assert shared.status_code==200,shared.text
            assert not shared.json()['frame']['observation_id'].startswith('human-')
            (output/'explicitly-shared-agent-screen.png').write_bytes(base64.b64decode(shared.json()['image_base64']))
            await control('stop','pause',4)
            export.update(transfer_id='safe-'+uuid4().hex,control_epoch=5)
            published=await client.post('/api/artifacts/workspaces/publish',json={'request':export,'title':'Explicitly shared Human document','publication_key':'safe-'+uuid4().hex})
            assert published.status_code==200,published.text
            ref=published.json()['receipt']['ref']
            released=await client.post(base+'/release');assert released.status_code==200,released.text
            assert released.json()['released'] is True
            absent=subprocess.run(['/usr/local/bin/docker','inspect','--type','container',cid],cwd=Path.cwd(),capture_output=True)
            (output/'browser-container-absence.stderr').write_bytes(absent.stderr)
            assert absent.returncode==1 and (('No such container: '+cid).encode() in absent.stderr or ('No such object: '+cid).encode() in absent.stderr)
            reopened=await client.get(f'/api/artifacts/{ref["artifact_id"]}/revisions/{ref["revision_id"]}/file',params={'path':'document.html'})
            assert reopened.status_code==200 and b'Safe to share' in reopened.content and secret.encode() not in reopened.content
            state['shared_right_panel_after_destroy']=await browser('visual_human_artifact_after_destroy.browser.cjs',{'ref':ref})
            assert (await agent._raw_storage.db.fetchone('SELECT COUNT(*) FROM conversation_history'))[0]==before
            assert model_attempts==[] and secret not in caplog.text
            state.update(conversation_records_added=0,model_attempts=0,private_text_absent_logs=True,
                private_native_input_absent_agent_status=True,explicit_share_confirmation=True,
                original_worker_physically_absent=True,artifact_ref=ref)
            record()
    finally:
        if serving is not None and not serving.done() and sid is not None:
            async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}',trust_env=False,headers={'X-API-Key':key},timeout=30) as cleanup:
                response=await cleanup.post(f'/api/visual-computer/sessions/{sid}/release')
                state['final_release_status']=response.status_code;record()
        if runner is not None:runner.should_exit=True
        if serving is not None:await asyncio.wait_for(serving,15)
        sock.close();server._unmount_feature_routers(app);server._unmount_feature_ui_assets(app);app.router.lifespan_context=lifespan
        await shutdown_fixture_agent(agent)
        await llm.close()
        state.update(server_joined=serving is None or serving.done(),socket_closed=sock.fileno()==-1)
        record()
