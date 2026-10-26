"""Public first-party bridge uses original policy and actual private transport."""
import asyncio
from hashlib import sha256

from fastapi import Request
import pytest
import pytest_asyncio
from starlette.responses import JSONResponse
from kestrel_sdk.artifacts import ArtifactCommit, ArtifactManifest
from kestrel_sdk.operator import OperatorAuthorizationError

from kestrel_sovereign.auth import CallerContext, caller_context_scope
from kestrel_sovereign.privacy import PrivacyMode
from tests.integration.test_artifact_host import host, repository, KEY, AUTH, path, publish


@pytest.mark.asyncio
async def test_public_operation_publishes_through_original_caller_and_context_retires(host):
    client, agents, _ = host
    feature=agents['kite'].features['ArtifactsFeature']
    assert feature.artifact_bridge_api_version==1
    with caller_context_scope(CallerContext.sovereign(credential=KEY)):
        async with feature.artifact_operation() as (service, context):
            assert agents['kite'].gate.locked()
            assert context.acting_agent_id==agents['kite'].did
            assert context.granted_capabilities=={'artifacts'}
            content=b'Canonical workflow source'
            staged=await service.stage(context,content,publication_key='public-bridge',path='workflow.md',media_type='text/markdown')
            assert staged.sha256==sha256(content).hexdigest()
            receipt=await service.commit(context,ArtifactCommit(idempotency_key='public-bridge',title='Workflow source',
                manifest=ArtifactManifest(files=(staged,),entrypoint=staged.path)))
            assert receipt.scope.agent_id==agents['kite'].did
        with pytest.raises(OperatorAuthorizationError):await service.get(context,receipt.ref)
    assert not agents['kite'].gate.locked()
    response=await client.get(path()+f'/{receipt.ref.artifact_id}/revisions/{receipt.ref.revision_id}/file',
        headers=AUTH,params={'path':'workflow.md'})
    assert response.content==content


@pytest.mark.asyncio
async def test_public_operation_never_mints_caller_privacy_generation_or_rotated_grants(host,monkeypatch):
    _, agents, _=host;agent=agents['kite'];feature=agent.features['ArtifactsFeature']
    with pytest.raises(OperatorAuthorizationError):
        async with feature.artifact_operation():pytest.fail('no caller was supplied')
    with caller_context_scope(CallerContext.sovereign(credential=KEY)):
        agent.set_mode(PrivacyMode.EPHEMERAL)
        with pytest.raises(OperatorAuthorizationError):
            async with feature.artifact_operation():pytest.fail('privacy forbids persistence')
        agent.set_mode(PrivacyMode.NORMAL)
        with pytest.raises(OperatorAuthorizationError):
            async with feature.artifact_operation() as (service, context):
                feature._generation+=1
                await service.list(context)
        with pytest.raises(OperatorAuthorizationError):
            async with feature.artifact_operation() as (service, context):
                monkeypatch.setenv('KESTREL_API_KEY','changed-current-bridge-key')
                await service.list(context)


@pytest_asyncio.fixture
async def bridge_host(host):
    from kestrel_sovereign import server
    client, agents, app=host;feature=agents['kite'].features['ArtifactsFeature']
    async def build(service,context):
        return JSONResponse((await service.list(context)).model_dump(mode='json'))
    @feature.get_router().get('/public-bridge-proof')
    async def read(request:Request):return feature.artifact_response(request,build)
    server._unmount_feature_routers(app);server._mount_feature_routers(app)
    yield client,agents,app


@pytest.mark.asyncio
async def test_public_response_uses_real_host_auth_and_no_store_delivery(bridge_host):
    from kestrel_sovereign import server
    client,_,_=bridge_host;ref,_=await publish(client)
    target=path()+'/public-bridge-proof'
    assert (await client.get(target)).status_code==401
    peer=await client.get(target,headers={**AUTH,server.A2A_TRANSPORT_KEY_HEADER:server.ensure_a2a_transport_key()})
    assert peer.status_code==403
    response=await client.get(target,headers=AUTH)
    assert response.status_code==200,response.text
    assert response.json()['items'][0]['ref']==ref
    assert 'no-store' in response.headers['cache-control']


@pytest.mark.asyncio
async def test_public_response_retains_privacy_and_current_generation_through_actual_send(bridge_host):
    _,agents,app=bridge_host;agent=agents['kite'];feature=agent.features['ArtifactsFeature']
    entered,release=asyncio.Event(),asyncio.Event();messages=[]
    async def receive():return {'type':'http.request','body':b'','more_body':False}
    async def send(message):
        if message['type']=='http.response.start' and message['status']==200:
            entered.set();await release.wait()
        messages.append(message)
    target=path()+'/public-bridge-proof'
    scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},'http_version':'1.1',
        'method':'GET','scheme':'http','path':target,'raw_path':target.encode(),'query_string':b'',
        'root_path':'','server':('test',80),'client':('test',1),'headers':[(b'x-api-key',KEY.encode())]}
    serving=asyncio.create_task(app(scope,receive,send));transition=None
    transitioned=asyncio.Event()
    async def privacy_transition():
        async with agent.gate.exclusive():transitioned.set()
    try:
        await asyncio.wait_for(entered.wait(),5)
        transition=asyncio.create_task(privacy_transition());await asyncio.sleep(.02)
        assert not transitioned.is_set()
        feature._generation+=1;release.set()
        with pytest.raises(OperatorAuthorizationError):await serving
        assert not any(message['type']=='http.response.body' and message.get('body') for message in messages)
        await asyncio.wait_for(transition,5)
        assert transitioned.is_set() and not agent.gate.locked()
    finally:
        release.set()
        for task in (serving,transition):
            if task is not None and not task.done():task.cancel()
        await asyncio.gather(*(task for task in (serving,transition) if task is not None),return_exceptions=True)
