"""Real host grants and erasure fencing through the derivative transport send."""
import asyncio
import base64

from kestrel_sdk.artifacts import ArtifactRef
from kestrel_sovereign.artifacts.preview import DockerArtifactRenderer, host_preview_pool
import pytest

from tests.integration.test_artifact_host import host, repository, KEY, AUTH, path
from tests.integration.test_artifact_preview_docker import renderer_image, png, pdf, ObservedRenderer, absent


async def publish_source(client, source, media_type, key='preview-source'):
    file = await client.post(path() + '/staging', headers=AUTH, json={'publication_key':key,
        'path':'source.bin','media_type':media_type,'content_base64':base64.b64encode(source).decode()})
    assert file.status_code == 200
    response = await client.post(path() + '/publications', headers=AUTH, json={'idempotency_key':key,
        'title':'Private source','manifest':{'files':[file.json()],'entrypoint':'source.bin'}})
    assert response.status_code == 200
    return response.json()['ref']


def url(ref):
    return path() + '/' + ref['artifact_id'] + '/revisions/' + ref['revision_id'] + '/preview'


@pytest.mark.asyncio
async def test_host_admission_refuses_before_reading_any_source_bytes(host, renderer_image, monkeypatch):
    client, agents, app = host
    service = agents['kite'].features['ArtifactsFeature']._service
    service.preview_renderer = ObservedRenderer(renderer_image)
    ref = await publish_source(client, png(), 'image/png')
    pool = host_preview_pool(app)
    held = [pool.reserve(), pool.reserve()]
    reads = []
    async def unexpected_read(*args, **kwargs):
        reads.append(args)
        raise AssertionError('a saturated preview pool must not read source bytes')
    monkeypatch.setattr(service.repository, 'read', unexpected_read)
    try:
        response = await client.get(url(ref), headers=AUTH)
        assert response.status_code == 503
        assert response.json()['error']['code'] == 'artifact_preview_unavailable'
        assert not reads and pool.active_count == 2
    finally:
        for job in held:
            pool.complete(job)


@pytest.mark.asyncio
async def test_registered_host_preview_is_owner_exact_and_refusal_preserves_source(host, renderer_image):
    client, agents, app = host
    renderer = ObservedRenderer(renderer_image)
    agents['kite'].features['ArtifactsFeature']._service.preview_renderer = renderer
    ref = await publish_source(client, png(), 'image/png')
    assert (await client.get(url(ref))).status_code == 401
    assert (await client.get(url(ref).replace('/kite/','/other/'), headers=AUTH)).status_code == 404
    response = await client.get(url(ref), headers=AUTH)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value['source'] == ref and value['media_type'] == 'image/png'
    assert (value['width'],value['height']) == (64,32)
    assert 'no-store' in response.headers['cache-control']
    assert app.state.artifact_preview_pool.active_count == 0
    broken = await publish_source(client,b'%PDF-1.7\nprivate invalid source','application/pdf',key='invalid-pdf')
    refused = await client.get(url(broken),headers=AUTH)
    assert refused.status_code in {422,503} and 'private invalid' not in refused.text
    source = await client.get(url(broken).replace('/preview','/file'),headers=AUTH,params={'path':'source.bin'})
    assert source.content == b'%PDF-1.7\nprivate invalid source'
    await absent(renderer,renderer.names)


@pytest.mark.asyncio
async def test_key_revocation_before_decoder_input_blocks_private_source_and_cleans_container(host,renderer_image,monkeypatch):
    client,agents,app = host
    entered,release = asyncio.Event(),asyncio.Event()
    class PausedRenderer(ObservedRenderer):
        async def _command(self, arguments, **kwargs):
            result = await super()._command(arguments, **kwargs)
            if 'inspect' in arguments and 'image' not in arguments and 'context' not in arguments:
                entered.set(); await release.wait()
            return result
    renderer=PausedRenderer(renderer_image)
    agents['kite'].features['ArtifactsFeature']._service.preview_renderer=renderer
    ref=await publish_source(client,png(),'image/png')
    pending=asyncio.create_task(client.get(url(ref),headers=AUTH))
    try:
        await asyncio.wait_for(entered.wait(),10)
        monkeypatch.setenv('KESTREL_API_KEY','rotated-renderer-test-key');release.set()
        response=await asyncio.wait_for(pending,10)
        assert response.status_code==403 and response.json()['error']['code']=='artifact_access_denied'
        assert not renderer.started.is_set()
        assert app.state.artifact_preview_pool.active_count==0
        await absent(renderer,renderer.names)
    finally:
        release.set()
        if not pending.done():pending.cancel()
        await asyncio.gather(pending,return_exceptions=True)


@pytest.mark.asyncio
async def test_canonical_delete_waits_for_the_actual_preview_transport_send(host,renderer_image):
    client,agents,app=host
    renderer=ObservedRenderer(renderer_image)
    service=agents['kite'].features['ArtifactsFeature']._service;service.preview_renderer=renderer
    ref=await publish_source(client,pdf(),'application/pdf')
    entered,release=asyncio.Event(),asyncio.Event();messages=[]
    async def receive():return {'type':'http.request','body':b'','more_body':False}
    async def send(message):
        if message['type']=='http.response.start' and message['status']==200:
            entered.set();await release.wait()
        messages.append(message)
    target=url(ref)
    scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},'http_version':'1.1',
        'method':'GET','scheme':'http','path':target,'raw_path':target.encode(),'query_string':b'',
        'root_path':'','server':('test',80),'client':('test',1),'headers':[(b'x-api-key',KEY.encode())]}
    serving=asyncio.create_task(app(scope,receive,send));deleting=None
    try:
        await asyncio.wait_for(entered.wait(),10)
        deleting=asyncio.create_task(client.delete(path()+'/'+ref['artifact_id'],headers=AUTH,
            params={'expected_revision_id':ref['revision_id']}))
        await asyncio.sleep(.05)
        assert not deleting.done(),'erasure must wait for the revision-bound derivative send'
        release.set();await asyncio.wait_for(serving,5)
        assert (await asyncio.wait_for(deleting,5)).status_code==204
        assert any(message['type']=='http.response.body' and message.get('body') for message in messages)
        assert (await client.get(url(ref),headers=AUTH)).status_code==404
        assert not service._erasure_fences and not service._preview_deliveries
        assert app.state.artifact_preview_pool.active_count==0
        await absent(renderer,renderer.names)
    finally:
        release.set()
        for task in (serving,deleting):
            if task is not None and not task.done():task.cancel()
        await asyncio.gather(*(task for task in (serving,deleting) if task is not None),return_exceptions=True)
