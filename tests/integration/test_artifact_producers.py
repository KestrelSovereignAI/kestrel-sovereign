"""Real registered producer tools and encrypted host/API interoperation."""

import asyncio

import pytest

from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sovereign.auth import AuthMethod, CallerContext, caller_context_scope
from kestrel_sovereign.privacy import PrivacyMode
from tests.integration.test_artifact_host import host as host, repository as repository, KEY, AUTH, path


def caller():
    return CallerContext.sovereign(credential=KEY)


def tools(agent):
    return {tool.name: tool for tool in agent.features['ArtifactsFeature'].get_tools()}


async def publish(agent, **changes):
    args = dict(title='Producer report', text='# Durable source', path='report.md', publication_key='producer')
    args.update(changes)
    return await tools(agent)['artifact_publish_text'].execute(**args)


@pytest.mark.asyncio
async def test_registered_producer_tool_creates_shared_card_and_reopens_via_real_host(host):
    client, agents, _ = host
    assert set(tools(agents['kite'])) == {'artifact_publish_text', 'artifact_list', 'artifact_get', 'artifact_publication_status',
        'artifact_publish_workspace', 'artifact_materialize_workspace', 'artifact_workspace_status'}
    with caller_context_scope(caller()):
        result = await publish(agents['kite'])
        assert result['status'] == 'ok' and result['data']['state'] == 'saved'
        ref = result['data']['receipt']['ref']
        assert result['parts'] == [{'type': 'artifact', 'data': {'ref': ref, 'title': 'Producer report'}}]
        again = await publish(agents['kite'])
        assert again['data']['receipt'] == result['data']['receipt']
        status = await tools(agents['kite'])['artifact_publication_status'].execute(publication_key='producer')
        assert status['data']['receipt']['ref'] == ref
        page = await tools(agents['kite'])['artifact_list'].execute(query='REPORT.MD')
        assert page['parts'][0]['data']['ref'] == ref
        opened = await tools(agents['kite'])['artifact_get'].execute(artifact_id=ref['artifact_id'], include_text=True)
        assert opened['parts'][0]['data']['ref'] == ref
        assert opened['data']['source_text'] == '# Durable source'
        cross_owner = await tools(agents['other'])['artifact_get'].execute(**{k:v for k,v in ref.items() if k != 'schema_version'})
        assert cross_owner['status'] == 'error'
    url = path() + f'/{ref["artifact_id"]}/revisions/{ref["revision_id"]}/file'
    response = await client.get(url, headers=AUTH, params={'path':'report.md'})
    assert response.content == b'# Durable source'
    assert response.headers['content-disposition'].startswith('attachment;')


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [None, CallerContext.anonymous(), CallerContext.authenticated('operator'),
                                 CallerContext.a2a_transport(), CallerContext.sovereign(AuthMethod.OAUTH_SESSION, credential=KEY),
                                 CallerContext.sovereign(credential='old-key')])
async def test_producer_never_promotes_absent_cookie_peer_or_stale_authority(host, value):
    _, agents, _ = host
    with caller_context_scope(value):
        result = await publish(agents['kite'])
    assert result['status'] == 'error' and result['data']['state'] == 'denied'
    repo = agents['kite'].features['ArtifactsFeature']._service.repository
    assert (await repo.db.fetchone('SELECT COUNT(*) FROM artifact_revisions'))[0] == 0


@pytest.mark.asyncio
async def test_producer_cas_conflict_unknown_commit_and_original_key_reconciliation(host, monkeypatch):
    _, agents, _ = host
    feature = agents['kite'].features['ArtifactsFeature']
    with caller_context_scope(caller()):
        first = await publish(agents['kite'])
        ref = first['data']['receipt']['ref']
        second = await publish(agents['kite'], text='second', publication_key='second', artifact_id=ref['artifact_id'], expected_revision_id=ref['revision_id'])
        assert second['data']['receipt']['ref']['revision_id'] != ref['revision_id']
        conflict = await publish(agents['kite'], text='stale', publication_key='stale', artifact_id=ref['artifact_id'], expected_revision_id=ref['revision_id'])
        assert conflict['data']['state'] == 'conflict'
        commit = feature._service.commit
        async def lose_ack(context, request):
            await commit(context, request)
            raise ResourceOutcomeUnknown('lost response')
        monkeypatch.setattr(feature._service, 'commit', lose_ack)
        unknown = await publish(agents['kite'], publication_key='lost')
        assert unknown['status'] == 'partial' and unknown['data'] == {'state':'unknown','publication_key':'lost'}
        status = await tools(agents['kite'])['artifact_publication_status'].execute(publication_key='lost')
        assert status['data']['state'] == 'saved'
        absent = await tools(agents['kite'])['artifact_publication_status'].execute(publication_key='not-seen')
        assert absent['status'] == 'partial' and absent['data']['state'] == 'unknown'


@pytest.mark.asyncio
async def test_endpoint_revocation_during_producer_stage_prevents_publication(host, monkeypatch):
    _, agents, _ = host
    service = agents['kite'].features['ArtifactsFeature']._service
    entered, release = asyncio.Event(), asyncio.Event()
    stage = service.stage
    async def blocked(*args, **kwargs):
        entered.set(); await release.wait(); return await stage(*args, **kwargs)
    monkeypatch.setattr(service, 'stage', blocked)
    with caller_context_scope(caller()):
        pending = asyncio.create_task(publish(agents['kite']))
        await entered.wait()
    release.set()  # Endpoint lifetime has ended; copied task authority is dead.
    result = await pending
    assert result['data']['state'] == 'denied'
    assert (await service.repository.db.fetchone('SELECT COUNT(*) FROM artifact_revisions'))[0] == 0


@pytest.mark.asyncio
async def test_producer_checks_privacy_and_live_feature_before_staging(host, monkeypatch):
    _, agents, _ = host
    agent = agents['kite']; feature = agent.features['ArtifactsFeature']
    with caller_context_scope(caller()):
        agent.set_mode(PrivacyMode.EPHEMERAL)
        assert (await publish(agent))['data']['state'] == 'denied'
        agent.set_mode(PrivacyMode.NORMAL)
        result = await publish(agent, path='../escape')
        assert result['data']['state'] == 'rejected'
        assert 'escape' not in result['error']
        await feature.on_disable()
        assert (await publish(agent))['data']['state'] == 'denied'
