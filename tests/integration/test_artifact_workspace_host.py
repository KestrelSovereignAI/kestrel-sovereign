"""Two actual host grants, real encrypted storage and a bounded SDK transport.

The controlled transport exercises host authority/receipt races. Real local
Docker persistence/materialization is proved separately without this double.
"""
import asyncio
import json
from datetime import UTC, datetime
from hashlib import sha256
from types import SimpleNamespace

import pytest
import pytest_asyncio
from pydantic import SecretBytes
from kestrel_sdk.artifacts import ArtifactFile, ArtifactManifest
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown
from kestrel_sdk.workspaces import WorkspaceSnapshot, WorkspaceTransferStatus, WorkspaceMaterialization, WorkspaceExportRequest

from tests.integration.test_artifact_host import host as host, repository as repository, AUTH, KEY, path, publish
from tests.unit.visual_computer.conftest import Adapter
from tests.unit.visual_computer.test_host import intent, path as visual_path
from kestrel_sovereign.features.visual_computer import feature as visual_module
from kestrel_sovereign.features.visual_computer import VisualComputerFeature
from kestrel_sovereign.visual_computer.runtime import VisualAction


class Transfer:
    def __init__(self):
        self.records = {}; self.files = {}; self.live = {'report.txt':b'workspace draft'}
        self.prepares = 0; self.reads = 0; self.installs = 0; self.cleanups = []
        self.after_read = None; self.lose_import_ack = False
    async def prepare_export(self, session, request):
        self.prepares += 1
        old = self.records.get(request.transfer_id)
        if old is not None:
            if old.request_sha256 != request.digest(): raise ResourceConflict('changed transfer')
            if old.snapshot is None: raise ResourceNotFound('retired export')
            return old.snapshot
        files = tuple(ArtifactFile(path=path, media_type='text/plain', size_bytes=len(self.live[path]), sha256=sha256(self.live[path]).hexdigest()) for path in request.paths)
        snapshot = WorkspaceSnapshot(request=request, snapshot_id=request.transfer_id,
            manifest=ArtifactManifest(files=files, entrypoint=request.entrypoint))
        self.files[request.transfer_id] = {file.path:self.live[file.path] for file in files}
        self.records[request.transfer_id] = WorkspaceTransferStatus(transfer_id=request.transfer_id, session_id=session.session_id,
            request_sha256=request.digest(), direction='export', state='ready', snapshot=snapshot)
        return snapshot
    async def read_export(self, session, snapshot_id, path):
        self.reads += 1
        value = self.files[snapshot_id][path]
        if self.after_read is not None: await self.after_read()
        return SecretBytes(value)
    async def release_export(self, session, transfer_id):
        self.cleanups.append(('release',session,transfer_id))
        old = self.records.get(transfer_id)
        if old is None: raise ResourceNotFound('no export')
        self.files.pop(transfer_id,None)
        status = old.model_copy(update={'state':'released','snapshot':None})
        self.records[transfer_id] = status
        return status
    async def prepare_import(self, session, request):
        old = self.records.get(request.transfer_id)
        if old is not None:
            if old.request_sha256 != request.digest(): raise ResourceConflict('changed transfer')
            return old
        self.files[request.transfer_id] = {}
        status = WorkspaceTransferStatus(transfer_id=request.transfer_id, session_id=session.session_id,
            request_sha256=request.digest(), direction='import', state='ready')
        self.records[request.transfer_id] = status
        self.request = request
        return status
    async def write_import_file(self, session, transfer_id, file, content):
        self.files[transfer_id][file.path] = content.get_secret_value()
        return self.records[transfer_id]
    async def commit_import(self, session, transfer_id):
        if self.records[transfer_id].materialization is not None:
            return self.records[transfer_id].materialization
        request = self.request
        self.installs += 1
        self.live = self.files.pop(transfer_id)
        receipt = WorkspaceMaterialization(transfer_id=transfer_id, session_id=session.session_id,
            lease_id=session.lease_id, control_epoch=session.control_epoch, workspace_id=request.workspace_id,
            source=request.source, manifest_sha256=request.manifest.digest(), request_sha256=request.digest(), completed_at=datetime.now(UTC))
        self.records[transfer_id] = self.records[transfer_id].model_copy(update={'state':'complete','materialization':receipt})
        if self.lose_import_ack:
            self.lose_import_ack = False
            raise ResourceOutcomeUnknown('lost materialization acknowledgment')
        return receipt
    async def abort_import(self, session, transfer_id):
        self.cleanups.append(('abort',session,transfer_id))
        old = self.records.get(transfer_id)
        if old is None: raise ResourceNotFound('no import')
        if old.state == 'complete': return old
        self.files.pop(transfer_id,None)
        status = old.model_copy(update={'state':'aborted'})
        self.records[transfer_id] = status
        return status
    async def status(self, session, transfer_id):return self.records.get(transfer_id)


@pytest_asyncio.fixture
async def workspace_host(host, monkeypatch):
    from kestrel_sovereign import server
    client, agents, app = host
    cfg = {'enabled':True,'granted_actions':[action.value for action in VisualAction],
        'providers':{'test-provider':{}},'drivers':{'test-driver':{}}}
    monkeypatch.setattr(visual_module,'load_section',lambda section:{'visual_computer':cfg})
    adapter = Adapter(); transport = Transfer()
    capabilities = adapter.capabilities
    async def with_files():
        value = await capabilities()
        return value.model_copy(update={'operations':value.operations|{'files'}})
    adapter.capabilities = with_files
    adapter.workspace_transfer = lambda:transport
    monkeypatch.setattr(visual_module,'construct_adapters',lambda **kwargs:({'test-provider':adapter},{'test-driver':adapter}))
    for agent in agents.values():
        feature = VisualComputerFeature(agent)
        await feature.initialize();agent.features[feature.name]=feature;await feature.on_enable()
    server._unmount_feature_routers(app);server._mount_feature_routers(app)
    try:yield SimpleNamespace(client=client,agents=agents,app=app,adapter=adapter,transfer=transport,cfg=cfg)
    finally:
        for agent in agents.values():await agent.features['VisualComputerFeature'].shutdown()


async def session(h, request_id='workspace-session'):
    request = intent();request['request_id']=request_id;request['required_capabilities'].append('files')
    admitted = await h.client.post(visual_path()+'/requests',headers=AUTH,json=request)
    assert admitted.status_code == 200,admitted.text
    value = admitted.json();sid=value['session_id']
    stopped = await h.client.post(visual_path()+f'/sessions/{sid}/stop',headers=AUTH,json={
        'request_id':'pause-'+request_id,'session_id':sid,'expected_control_epoch':0,'operation':'pause'})
    assert stopped.status_code == 200,stopped.text
    return (await h.client.get(visual_path()+f'/sessions/{sid}',headers=AUTH)).json()['session']


def export_request(value, transfer='export'):
    return {'transfer_id':transfer,'session_id':value['session_id'],'lease_id':value['lease_id'],
        'control_epoch':value['control_epoch'],'workspace_id':'browser','expires_at':value['expires_at'],
        'limits':{'max_files':4,'max_file_bytes':131072,'max_total_bytes':262144},
        'paths':['report.txt'],'entrypoint':'report.txt'}


def publication(value, **changes):
    return {'request':export_request(value),'title':'Workspace output','publication_key':'workspace-publication',**changes}


async def post(h, suffix, value):
    return await h.client.post(path()+'/workspaces/'+suffix,headers=AUTH,json=value)


@pytest.mark.asyncio
async def test_host_workspace_publish_survives_source_release_and_materializes_pinned_edit(workspace_host):
    h=workspace_host;first_session=await session(h)
    response=await post(h,'publish',publication(first_session));assert response.status_code==200,response.text
    first=response.json();ref=first['receipt']['ref'];assert first['state']=='saved' and first['cleanup_confirmed']
    assert (await h.client.post(visual_path()+f"/sessions/{first_session['session_id']}/release",headers=AUTH)).json()['released']
    h.transfer.live.clear()
    source=await h.client.get(path()+f"/{ref['artifact_id']}/revisions/{ref['revision_id']}/file",headers=AUTH,params={'path':'report.txt'})
    assert source.content==b'workspace draft'
    second,_=await publish(h.client,key='edited-workspace',ref=ref,content=b'edited after source destruction')
    revision=(await h.client.get(path()+f"/{second['artifact_id']}/revisions/{second['revision_id']}",headers=AUTH)).json()
    current=await session(h,'destination-session')
    incoming={key:value for key,value in export_request(current,'import').items() if key not in {'paths','entrypoint'}}
    incoming.update(source=second,manifest=revision['manifest'],workspace_id='fresh-target')
    materialized=await post(h,'materialize',incoming);assert materialized.status_code==200,materialized.text
    assert materialized.json()['materialization']['source']==second and h.transfer.live['report.txt']==b'edited after source destruction'
    outgoing=publication(current,publication_key='revision-three',artifact_id=second['artifact_id'],expected_revision_id=second['revision_id'])
    outgoing['request']=export_request(current,'export-three');outgoing['request']['workspace_id']='fresh-target'
    third=await post(h,'publish',outgoing);assert third.status_code==200,third.text
    assert third.json()['receipt']['ref']['artifact_id']==ref['artifact_id']
    assert third.json()['receipt']['ref']['revision_id']!=second['revision_id']


@pytest.mark.asyncio
async def test_two_grants_owner_epoch_and_relative_selection_are_independent(workspace_host):
    h=workspace_host;value=await session(h);body=publication(value)
    assert (await h.client.post(path()+'/workspaces/publish',json=body)).status_code==401
    assert (await h.client.post(path('other')+'/workspaces/publish',headers=AUTH,json=body)).status_code==404
    h.cfg['granted_actions'].remove('visual.transfer')
    assert (await post(h,'publish',body)).status_code==403
    h.cfg['granted_actions'].append('visual.transfer')
    body['request']['control_epoch']=0
    assert (await post(h,'publish',body)).status_code==409
    body=publication(value);body['request']['paths']=['/private/host'];body['request']['entrypoint']='/private/host'
    assert (await post(h,'publish',body)).status_code==400
    assert h.transfer.prepares==0 and h.transfer.reads==0


@pytest.mark.asyncio
async def test_lost_publication_ack_reconciles_without_resnapshot_or_duplicate(workspace_host,monkeypatch):
    h=workspace_host;value=await session(h);body=publication(value)
    service=h.agents['kite'].features['ArtifactsFeature']._service;commit=service.commit
    async def lost(*args,**kwargs):
        await commit(*args,**kwargs)
        raise ResourceOutcomeUnknown('lost canonical receipt')
    monkeypatch.setattr(service,'commit',lost)
    assert (await post(h,'publish',body)).status_code==503
    monkeypatch.setattr(service,'commit',commit)
    recovered=await post(h,'publish',body);assert recovered.status_code==200,recovered.text
    assert h.transfer.prepares==1
    changed=await post(h,'publish',{**body,'title':'changed request'})
    assert changed.status_code==409
    assert len((await h.client.get(path(),headers=AUTH)).json()['items'])==1


@pytest.mark.asyncio
async def test_lost_install_ack_returns_original_receipt_without_installing_twice(workspace_host):
    h=workspace_host;current=await session(h)
    ref,_=await publish(h.client)
    revision=(await h.client.get(path()+f"/{ref['artifact_id']}/revisions/{ref['revision_id']}",headers=AUTH)).json()
    request={key:value for key,value in export_request(current,'install').items() if key not in {'paths','entrypoint'}}
    request.update(source=ref,manifest=revision['manifest'],workspace_id='fresh-install')
    h.transfer.lose_import_ack=True
    assert (await post(h,'materialize',request)).status_code==503
    response=await post(h,'materialize',request);assert response.status_code==200,response.text
    assert h.transfer.installs==1
    status=await h.client.get(path()+f"/workspaces/sessions/{current['session_id']}/transfers/install",headers=AUTH)
    assert status.json()['state']=='complete'


@pytest.mark.asyncio
async def test_revocation_after_export_read_prevents_staging_but_retained_cleanup_runs(workspace_host):
    h=workspace_host;current=await session(h)
    async def revoke():h.cfg['granted_actions'].remove('visual.transfer')
    h.transfer.after_read=revoke
    assert (await post(h,'publish',publication(current))).status_code==403
    assert h.transfer.cleanups and h.transfer.cleanups[-1][1].control_epoch==current['control_epoch']
    assert (await h.client.get(path(),headers=AUTH)).json()['items']==[]


@pytest.mark.asyncio
async def test_visual_constraint_rechecks_inside_canonical_database_guard(workspace_host,monkeypatch):
    h=workspace_host;current=await session(h)
    repository=h.agents['kite'].features['ArtifactsFeature']._service.repository;stage=repository.stage
    async def revoke_at_db(*args,**kwargs):
        h.cfg['granted_actions'].remove('visual.transfer')
        return await asyncio.create_task(stage(*args,**kwargs))
    monkeypatch.setattr(repository,'stage',revoke_at_db)
    assert (await post(h,'publish',publication(current))).status_code==403
    assert (await h.client.get(path(),headers=AUTH)).json()['items']==[]
    assert h.transfer.cleanups


@pytest.mark.asyncio
async def test_conflicting_prepare_never_retires_an_existing_different_request(workspace_host):
    h=workspace_host;current=await session(h)
    request=WorkspaceExportRequest.model_validate_json(json.dumps(export_request(current)))
    admitted=h.adapter.sessions[current['session_id']]
    snapshot=await h.transfer.prepare_export(admitted,request)
    changed=publication(current);changed['request']['workspace_id']='different-workspace'
    response=await post(h,'publish',changed)
    assert response.status_code==409
    assert not h.transfer.cleanups
    assert h.transfer.records[request.transfer_id].snapshot==snapshot


@pytest.mark.asyncio
async def test_actual_producer_tools_use_original_caller_and_emit_shared_artifact_part(workspace_host):
    import json
    from kestrel_sovereign.auth import CallerContext, caller_context_scope
    h=workspace_host;current=await session(h)
    tools={tool.name:tool for tool in h.agents['kite'].features['ArtifactsFeature'].get_tools()}
    arguments={'request_json':json.dumps(export_request(current)), 'publication_json':json.dumps({
        'title':'Tool workspace output','publication_key':'tool-workspace'})}
    with caller_context_scope(CallerContext.sovereign(credential=KEY)):
        saved=await tools['artifact_publish_workspace'].execute(**arguments)
    assert saved['status']=='ok',saved
    assert saved['data']['state']=='saved'
    assert saved['parts'][0]['type']=='artifact'
    assert saved['parts'][0]['data']['ref']==saved['data']['receipt']['ref']
    denied=await tools['artifact_publish_workspace'].execute(**arguments)
    assert denied['status']=='error' and denied['data']['state']=='denied'


@pytest.mark.asyncio
async def test_artifact_transfer_grant_cannot_be_promoted_by_visual_context(workspace_host,monkeypatch):
    from dataclasses import replace
    from kestrel_sovereign.features.artifacts import feature as artifact_module
    original=artifact_module.OperatorContext
    def limited(**kwargs):
        context=original(**kwargs)
        return replace(context,granted_actions=context.granted_actions-{'artifacts.transfer'})
    h=workspace_host;current=await session(h)
    monkeypatch.setattr(artifact_module,'OperatorContext',limited)
    assert (await post(h,'publish',publication(current))).status_code==403
    assert h.transfer.prepares==0


@pytest.mark.asyncio
async def test_pinned_manifest_and_materialization_receipt_are_verified(workspace_host,monkeypatch):
    from copy import deepcopy
    h=workspace_host;current=await session(h)
    ref,_=await publish(h.client)
    revision=(await h.client.get(path()+f"/{ref['artifact_id']}/revisions/{ref['revision_id']}",headers=AUTH)).json()
    request={key:value for key,value in export_request(current,'verified-import').items() if key not in {'paths','entrypoint'}}
    request.update(source=ref,manifest=revision['manifest'],workspace_id='verified-target')
    changed=deepcopy(request);changed['manifest']['files'][0]['media_type']='text/markdown'
    assert (await post(h,'materialize',changed)).status_code==409
    commit=h.transfer.commit_import
    async def wrong_receipt(*args):
        result=await commit(*args)
        return result.model_copy(update={'request_sha256':'0'*64})
    monkeypatch.setattr(h.transfer,'commit_import',wrong_receipt)
    assert (await post(h,'materialize',request)).status_code==503
    assert h.transfer.installs==1


@pytest.mark.asyncio
async def test_visual_grant_is_retained_and_rechecked_through_actual_transport_delivery(workspace_host):
    from kestrel_sdk.operator import OperatorAuthorizationError
    h=workspace_host;current=await session(h);body=publication(current)
    entered,release=asyncio.Event(),asyncio.Event();messages=[]
    async def receive():return {'type':'http.request','body':json.dumps(body).encode(),'more_body':False}
    async def send(message):
        if message['type']=='http.response.start' and message['status']==200:
            entered.set();await release.wait()
        messages.append(message)
    target=path()+'/workspaces/publish'
    scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},'http_version':'1.1',
        'method':'POST','scheme':'http','path':target,'raw_path':target.encode(),'query_string':b'',
        'root_path':'','server':('test',80),'client':('test',1),
        'headers':[(b'x-api-key',KEY.encode()),(b'content-type',b'application/json')]}
    pending=asyncio.create_task(h.app(scope,receive,send))
    try:
        await asyncio.wait_for(entered.wait(),5)
        h.cfg['granted_actions'].remove('visual.transfer');release.set()
        with pytest.raises(OperatorAuthorizationError):await pending
        assert not any(message['type']=='http.response.body' and message.get('body') for message in messages)
        # Publication accepted; the original artifact key is sufficient to
        # reconcile durability after the visual grant or computer disappears.
        receipt=await h.client.get(path()+'/publications/workspace-publication',headers=AUTH)
        assert receipt.status_code==200
        assert not h.agents['kite'].features['ArtifactsFeature']._service._current_constraints
    finally:
        release.set()
        if not pending.done():pending.cancel()
        await asyncio.gather(pending,return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['prepare_export', 'read_export', 'prepare_import', 'write_import_file', 'commit_import', 'status'])
@pytest.mark.parametrize('revocation', [None, 'visual', 'artifacts'])
async def test_broker_native_borrow_uses_original_transfer_and_both_current_grants(workspace_host, monkeypatch, method, revocation):
    """Model a trusted transport's final dispatch after an awaited launch.

    Real native-process/worker dispatch is exercised in the Docker host test.
    This route test proves every broker data call opens the right lexical lane,
    and either original feature's withdrawal reaches the borrowed final check.
    """
    h=workspace_host;current=await session(h)
    runtime=h.agents['kite'].features['VisualComputerFeature'].runtime
    artifacts=h.agents['kite'].features['ArtifactsFeature']
    original=getattr(h.transfer, method);effects=[];checks=[]
    transfer_id='native-dispatch'
    async def guarded(admitted, *args):
        # The production driver owns this child. The broker origin itself is
        # deliberately ineligible to borrow transport authority.
        async def native():
            async with runtime.egress.transfer_operation(admitted, transfer_id) as check:
                check();checks.append(check)
                await asyncio.sleep(0)  # inspection/process launch boundary
                if revocation=='visual':h.cfg['granted_actions'].remove('visual.transfer')
                if revocation=='artifacts':artifacts._generation+=1
                check()
                effects.append(method)
                return await original(admitted, *args)
        return await asyncio.create_task(native())
    monkeypatch.setattr(h.transfer, method, guarded)
    if method in {'prepare_export','read_export'}:
        body=publication(current);body['request']['transfer_id']=transfer_id
        response=await post(h, 'publish', body)
    elif method=='status':
        response=await h.client.get(path()+f"/workspaces/sessions/{current['session_id']}/transfers/{transfer_id}",headers=AUTH)
    else:
        ref,_=await publish(h.client)
        revision=(await h.client.get(path()+f"/{ref['artifact_id']}/revisions/{ref['revision_id']}",headers=AUTH)).json()
        body={key:value for key,value in export_request(current,transfer_id).items() if key not in {'paths','entrypoint'}}
        body.update(source=ref,manifest=revision['manifest'],workspace_id='native-target')
        response=await post(h, 'materialize', body)
    assert checks, response.text
    assert response.status_code==(200 if revocation is None else 403), response.text
    assert effects==([method] if revocation is None else [])
    assert not runtime.egress._calls
    if revocation is not None:
        assert h.transfer.installs==0
        if method not in {'prepare_export','prepare_import','status'}:
            assert h.transfer.cleanups[-1][1].session_id==current['session_id']
            assert h.transfer.cleanups[-1][2]==transfer_id


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["refused", "unknown", "cancelled"])
async def test_every_attempted_export_prepare_retires_only_its_original_owner(workspace_host, monkeypatch, failure):
    from kestrel_sdk.resources import ResourceUnsupported
    h = workspace_host; current = await session(h)
    retained = []
    async def prepare(admitted, request):
        # Prepare may reserve private native bytes before any kind of failure.
        h.transfer.records[request.transfer_id] = WorkspaceTransferStatus(transfer_id=request.transfer_id, session_id=admitted.session_id, request_sha256=request.digest(), direction="export", state="preparing")
        retained.append((admitted, request.transfer_id))
        if failure == "cancelled":raise asyncio.CancelledError()
        if failure == "unknown":raise ResourceOutcomeUnknown("accepted prepare lost ACK")
        raise ResourceUnsupported("source cannot be exported")
    monkeypatch.setattr(h.transfer, "prepare_export", prepare)
    if failure == "cancelled":
        from kestrel_sovereign.auth import CallerContext, caller_context_scope
        tool = {item.name:item for item in h.agents['kite'].features['ArtifactsFeature'].get_tools()}['artifact_publish_workspace']
        with caller_context_scope(CallerContext.sovereign(credential=KEY)):
            with pytest.raises(asyncio.CancelledError):
                await tool.execute(request_json=json.dumps(export_request(current)), publication_json=json.dumps({'title':'Workspace output','publication_key':'workspace-publication'}))
    else:
        response = await post(h, "publish", publication(current))
        assert response.status_code in {500, 503}, response.text
    assert len(retained) == 1
    assert h.transfer.cleanups == [("release", retained[0][0], retained[0][1])], "prepare failure abandoned its reserved export"
    assert h.transfer.records[retained[0][1]].state == "released"
    assert h.transfer.reads == 0 and not h.transfer.files


@pytest.mark.asyncio
async def test_failed_export_does_not_read_status_or_retire_after_original_grant_loss(workspace_host, monkeypatch):
    from kestrel_sdk.resources import ResourceUnsupported
    h=workspace_host;current=await session(h);calls=[]
    async def refused(admitted, request):
        h.transfer.records[request.transfer_id]=WorkspaceTransferStatus(transfer_id=request.transfer_id,session_id=admitted.session_id,request_sha256=request.digest(),direction="export",state="preparing")
        h.cfg['granted_actions'].remove('visual.transfer')
        raise ResourceUnsupported('source refused after grant loss')
    status=h.transfer.status
    async def observed(*args):
        calls.append(True)
        return await status(*args)
    monkeypatch.setattr(h.transfer,'prepare_export',refused);monkeypatch.setattr(h.transfer,'status',observed)
    response=await post(h,'publish',publication(current))
    # Preserve the existing generic unsupported-source error mapping.
    assert response.status_code==500,response.text
    assert not calls and not h.transfer.cleanups, 'failed prepare acquired data/cleanup after original grant loss'
    assert h.transfer.records['export'].state=='preparing'


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["session", "direction", "digest", "absent", "unknown"])
async def test_failed_export_cannot_retire_unproven_original_request(workspace_host, monkeypatch, damage):
    from kestrel_sdk.resources import ResourceUnsupported
    h=workspace_host;current=await session(h)
    async def refused(admitted, request):
        h.transfer.records[request.transfer_id]=WorkspaceTransferStatus(transfer_id=request.transfer_id,session_id=admitted.session_id,request_sha256=request.digest(),direction="export",state="preparing")
        raise ResourceUnsupported('source refused')
    async def status(admitted, transfer_id):
        value=h.transfer.records[transfer_id]
        if damage=="absent":return None
        if damage=="unknown":raise ResourceOutcomeUnknown('status unavailable')
        return value.model_copy(update={"session_id":"another-session"} if damage=="session" else {"direction":"import"} if damage=="direction" else {"request_sha256":"0"*64})
    monkeypatch.setattr(h.transfer,'prepare_export',refused);monkeypatch.setattr(h.transfer,'status',status)
    response=await post(h,'publish',publication(current))
    assert response.status_code==500,response.text
    assert not h.transfer.cleanups, 'unproven failed-prepare ownership was retired'
    assert h.transfer.records['export'].state=='preparing'
