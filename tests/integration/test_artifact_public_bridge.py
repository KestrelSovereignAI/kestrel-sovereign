"""Public first-party bridge uses original policy and actual private transport."""
import asyncio
from contextlib import asynccontextmanager
from hashlib import sha256
from types import SimpleNamespace

from fastapi import Request
import pytest
import pytest_asyncio
from starlette.responses import JSONResponse, Response
from kestrel_sdk.artifacts import ArtifactCommit, ArtifactManifest
from kestrel_sdk.operator import OperatorAuthorizationError

from kestrel_sovereign.auth import CallerContext, caller_context_scope
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.features.artifacts import ArtifactOperation
from tests.integration.test_artifact_host import host as host, repository as repository, KEY, AUTH, path, publish


@pytest.mark.asyncio
async def test_public_operation_publishes_through_original_caller_and_context_retires(host):
    client, agents, _ = host
    feature=agents['kite'].features['ArtifactsFeature']
    assert feature.artifact_bridge_api_version==3
    with caller_context_scope(CallerContext.sovereign(credential=KEY)):
        async with feature.artifact_operation() as operation:
            service,context=operation.service,operation.context
            operation.check_current()
            assert isinstance(operation,ArtifactOperation) and operation.expires_at==context.expires_at
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
        with pytest.raises(OperatorAuthorizationError):operation.check_current()
    assert not agents['kite'].gate.locked()
    response=await client.get(path()+f'/{receipt.ref.artifact_id}/revisions/{receipt.ref.revision_id}/file',
        headers=AUTH,params={'path':'workflow.md'})
    assert response.content==content


@pytest.mark.asyncio
async def test_typed_operation_checker_ends_before_original_resolver_cleanup_finishes(host,monkeypatch):
    _,agents,_=host;feature=agents['kite'].features['ArtifactsFeature']
    original=feature._resolve_tool_context;retiring,finish=asyncio.Event(),asyncio.Event();retained=[]
    @asynccontextmanager
    async def delayed_cleanup():
        async with original() as context:
            try:yield context
            finally:
                retiring.set();await finish.wait()
    monkeypatch.setattr(feature,'_resolve_tool_context',delayed_cleanup)
    async def owner():
        async with feature.artifact_operation() as operation:
            retained.append(operation);operation.check_current()
    with caller_context_scope(CallerContext.sovereign(credential=KEY)):
        pending=asyncio.create_task(owner())
        try:
            await asyncio.wait_for(retiring.wait(),5)
            with pytest.raises(OperatorAuthorizationError):retained[0].check_current()
        finally:
            finish.set();await pending
    assert not agents['kite'].gate.locked()


@pytest.mark.asyncio
async def test_typed_original_checker_rolls_back_consumer_relation_before_native_commit(host):
    from kestrel_sdk.storage.database import TransactionError
    _,agents,_=host;feature=agents['kite'].features['ArtifactsFeature']
    db=agents['kite']._raw_storage.db
    await db.execute('CREATE TABLE consumer_reference_guard (id INTEGER PRIMARY KEY)')
    with caller_context_scope(CallerContext.sovereign(credential=KEY)):
        with pytest.raises((OperatorAuthorizationError,TransactionError)) as denied:
            async with feature.artifact_operation() as operation:
                async with db.transaction():
                    operation.check_current()
                    await db.execute('INSERT INTO consumer_reference_guard VALUES (1)')
                    feature._generation+=1
                    operation.check_current()  # Original pre-COMMIT boundary.
        error=denied.value
        while not isinstance(error,OperatorAuthorizationError) and error.__cause__ is not None:error=error.__cause__
        assert isinstance(error,OperatorAuthorizationError)
    assert (await db.fetchone('SELECT COUNT(*) FROM consumer_reference_guard'))[0]==0


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
            async with feature.artifact_operation() as operation:
                service,context=operation.service,operation.context
                operation.check_current()
                feature._generation+=1
                await service.list(context)
        with pytest.raises(OperatorAuthorizationError):
            async with feature.artifact_operation() as operation:
                service,context=operation.service,operation.context
                operation.check_current()
                monkeypatch.setenv('KESTREL_API_KEY','changed-current-bridge-key')
                await service.list(context)


@pytest_asyncio.fixture
async def bridge_host(host):
    from kestrel_sovereign import server
    client, agents, app=host;feature=agents['kite'].features['ArtifactsFeature']
    async def build(operation):
        service,context=operation.service,operation.context
        operation.check_current()
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


@pytest_asyncio.fixture
async def restricted_bridge_host(host):
    """Consumer-owned independent lease; it never supplies artifact grants."""
    from kestrel_sovereign import server
    client,agents,app=host;feature=agents['kite'].features['ArtifactsFeature']
    policy=SimpleNamespace(generation=0,active=False,denied=False,built=0,checks=[],last_check=None,
        close_before_response=False,inner_body_checked=asyncio.Event(),result_factory=lambda:None)
    @asynccontextmanager
    async def independent_operation():
        generation=policy.generation;policy.active=True
        def current():
            policy.checks.append(True)
            if not policy.active or policy.denied or generation!=policy.generation:
                raise OperatorAuthorizationError('original independent policy ended')
            return policy.result_factory()
        policy.last_check=current
        try:yield current
        finally:policy.active=False
    class IndependentResponse(Response):
        async def __call__(self,scope,receive,send):
            async with independent_operation() as current:
                # This frame must enclose the artifact response's ASGI call;
                # resolving a run lease inside build then exiting is invalid.
                async def build(operation):
                    service,context=operation.service,operation.context
                    operation.check_current()
                    policy.built+=1
                    class InnerCheckedResponse(JSONResponse):
                        async def __call__(self,s,r,output):
                            async def inner_send(message):
                                current()
                                if message['type']=='http.response.body':policy.inner_body_checked.set()
                                await output(message)
                            await super().__call__(s,r,inner_send)
                    return InnerCheckedResponse((await service.list(context)).model_dump(mode='json'))
                response=feature.artifact_response(Request(scope,receive),build,check_current=current)
                if not policy.close_before_response:
                    await response(scope,receive,send)
                    return
            await response(scope,receive,send)  # Deliberately retired consumer lease.
    @feature.get_router().get('/restricted-public-bridge-proof')
    async def read():return IndependentResponse()
    server._unmount_feature_routers(app);server._mount_feature_routers(app)
    yield client,agents,app,policy


@pytest.mark.asyncio
async def test_public_response_restriction_never_promotes_or_resurrects_an_independent_policy(restricted_bridge_host):
    client,agents,_,policy=restricted_bridge_host
    target=path()+'/restricted-public-bridge-proof'
    assert (await client.get(target)).status_code==401 and policy.built==0
    policy.denied=True
    assert (await client.get(target,headers=AUTH)).status_code==403 and policy.built==0
    policy.denied=False
    policy.close_before_response=True
    assert (await client.get(target,headers=AUTH)).status_code==403 and policy.built==0
    policy.close_before_response=False
    assert (await client.get(target,headers=AUTH)).status_code==200 and policy.built==1
    with pytest.raises(OperatorAuthorizationError):policy.last_check()
    assert not policy.active and not agents['kite'].gate.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("violation", ["revocation", "invalid-return"])
async def test_public_restriction_fences_canonical_database_read_in_owned_child(restricted_bridge_host,monkeypatch,violation):
    client,agents,_,policy=restricted_bridge_host
    service=agents['kite'].features['ArtifactsFeature']._service
    original=service.repository.page;returned=[]
    async def revoke_at_database(*args,**kwargs):
        if violation == "invalid-return":policy.result_factory=lambda:False
        else:policy.generation+=1
        value=await asyncio.create_task(original(*args,**kwargs))
        returned.append(value)
        return value
    monkeypatch.setattr(service.repository,'page',revoke_at_database)
    response=await client.get(path()+'/restricted-public-bridge-proof',headers=AUTH)
    assert response.status_code==403,response.text
    assert not returned,'independent revocation failed to fence the original database read'
    assert not policy.active and not service._current_constraints


@pytest.mark.asyncio
@pytest.mark.parametrize('revoke',[False,True,'invalid-return'])
async def test_public_restriction_reaches_actual_outer_buffered_transport(restricted_bridge_host,revoke):
    _,agents,app,policy=restricted_bridge_host
    entered,release=asyncio.Event(),asyncio.Event();messages=[]
    async def receive():return {'type':'http.request','body':b'','more_body':False}
    async def send(message):
        if message['type']=='http.response.start' and message['status']==200:
            entered.set();await release.wait()
        messages.append(message)
    target=path()+'/restricted-public-bridge-proof'
    scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},'http_version':'1.1',
        'method':'GET','scheme':'http','path':target,'raw_path':target.encode(),'query_string':b'',
        'root_path':'','server':('test',80),'client':('test',1),'headers':[(b'x-api-key',KEY.encode())]}
    serving=asyncio.create_task(app(scope,receive,send))
    try:
        await asyncio.wait_for(entered.wait(),5)
        await asyncio.wait_for(policy.inner_body_checked.wait(),5)
        assert policy.active and agents['kite'].gate.locked()
        # Inner response checks may already have acknowledged middleware's
        # queue. Only the outer registered checker can stop the actual body.
        if revoke == 'invalid-return':policy.result_factory=lambda:False
        elif revoke:policy.generation+=1
        release.set()
        if revoke:
            denied=False
            try:await serving
            except OperatorAuthorizationError:denied=True
            assert not any(m['type']=='http.response.body' and m.get('body') for m in messages), 'independent revocation failed to fence actual private body'
            assert denied
        else:
            await serving
            assert any(m['type']=='http.response.body' and m.get('body') for m in messages)
        assert not policy.active and not agents['kite'].gate.locked()
        with pytest.raises(OperatorAuthorizationError):policy.last_check()
    finally:
        release.set()
        if not serving.done():serving.cancel()
        await asyncio.gather(serving,return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [False, True, 0, "allowed", "coroutine", "future"])
async def test_public_restriction_must_synchronously_return_none_or_raise(restricted_bridge_host, value):
    import inspect
    client, _, _, policy = restricted_bridge_host
    results = []
    executed = []
    async def asynchronous_check():
        executed.append(True)
    def result():
        if value == "coroutine":
            returned = asynchronous_check()
        elif value == "future":
            returned = asyncio.get_running_loop().create_future()
            returned.set_result(None)
        else:
            returned = value
        results.append(returned)
        return returned
    policy.result_factory = result
    response = await client.get(path()+"/restricted-public-bridge-proof", headers=AUTH)
    assert response.status_code == 403 and policy.built == 0, response.text
    assert not executed, "restriction awaitable was executed"
    assert all(not inspect.iscoroutine(item) or inspect.getcoroutinestate(item) == inspect.CORO_CLOSED for item in results), "rejected coroutine was abandoned"
