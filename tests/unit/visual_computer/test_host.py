"""Actual host auth and feature routing with real SQLite and a bounded driver.

This verifies host registration/authority, not real Docker readiness. The parent
composition separately exercises the concrete driver and worker boundary.
"""
import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

from kestrel_sovereign.features.visual_computer import feature as feature_module
from kestrel_sovereign.features.visual_computer import VisualComputerFeature
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.integration.test_artifact_service import PolicyHolder

KEY = "visual-host-test-only-key"
AUTH = {"X-API-Key":KEY}


@pytest_asyncio.fixture
async def visual_host(harness, monkeypatch):
    from kestrel_sovereign import server
    monkeypatch.setenv("KESTREL_API_KEY", KEY)
    monkeypatch.setenv("KESTREL_A2A_TRANSPORT_ONLY", "false")
    config = {"enabled":True, "granted_actions":[a.value for a in VisualAction],
        "providers":{"test-provider":{}}, "drivers":{"test-driver":{}}}
    monkeypatch.setattr(feature_module, "load_section", lambda section:{"visual_computer":config})
    monkeypatch.setattr(feature_module, "construct_adapters", lambda **kwargs:({"test-provider":harness.adapter},{"test-driver":harness.adapter}))
    agents = {}
    for name in ("kite", "other"):
        agent = PolicyHolder()
        agent.did = "did:test:"+name
        agent._raw_storage = SimpleNamespace(db=harness.db)
        agent.features = {}
        agent._track_background_task = lambda coro, *, name:asyncio.create_task(coro, name=name)
        feature = VisualComputerFeature(agent)
        await feature.initialize()
        agent.features[feature.name] = feature
        await feature.on_enable()
        agents[name] = agent
    app = server.app
    original_agent = getattr(app.state,"agent",None)
    original_manager = getattr(app.state,"agent_manager",None)
    original_lifespan = app.router.lifespan_context
    app.state.agent = None
    app.state.agent_manager = SimpleNamespace(list_agents=lambda:dict(agents), get_agent=lambda name:agents.get(name))
    server._mount_feature_routers(app)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
            yield client, agents, app
    finally:
        for agent in agents.values():
            await agent.features["VisualComputerFeature"].shutdown()
        server._unmount_feature_routers(app)
        app.router.lifespan_context = original_lifespan
        app.state.agent, app.state.agent_manager = original_agent, original_manager


def path(name="kite"):
    return f"/api/agents/{name}/api/visual-computer"


def intent():
    now = datetime.now(UTC)
    return {"request_id":"host-request", "provider_id":"test-provider", "profile_id":"offline", "image":"worker@sha256:"+"a"*64,
        "surface":"browser", "required_capabilities":["screenshot","click","type","takeover"], "location":"local",
        "requested_at":now.isoformat(), "expires_at":(now+timedelta(minutes=10)).isoformat(),
        "idle_ttl_seconds":60, "ready_timeout_seconds":10, "max_hourly_cost_usd":"0", "max_total_cost_usd":"0"}


@pytest.mark.asyncio
async def test_real_host_auth_admission_owner_binding_and_stop(visual_host):
    client, agents, _ = visual_host
    assert (await client.get(path()+"/sessions")).status_code == 401
    assert (await client.get(path()+"/sessions",params={"api_key":KEY})).status_code == 401
    response = await client.post(path()+"/requests",headers=AUTH,json=intent())
    assert response.status_code == 200, response.text
    session = response.json()
    sid = session["session_id"]
    assert session["scope"] == {"schema_version":1,"tenant_id":"did:test:kite", "principal_id":"host-sovereign", "agent_id":"did:test:kite"}
    view = await client.get(path()+f"/sessions/{sid}",headers=AUTH)
    assert view.status_code == 200
    assert "private-secret" not in view.text and "private-endpoint" not in view.text
    assert (await client.get(path("other")+f"/sessions/{sid}",headers=AUTH)).status_code == 404
    frame = await client.get(path()+f"/sessions/{sid}/observation",headers=AUTH)
    assert frame.status_code == 200
    assert frame.headers["cache-control"].startswith("private, no-store")
    assert frame.json()["image_base64"]
    stop = {"request_id":"stop-original", "session_id":sid, "expected_control_epoch":0, "operation":"pause"}
    stopped = await client.post(path()+f"/sessions/{sid}/stop",headers=AUTH,json=stop)
    assert stopped.json() == {"session_id":sid,"stopped":True}
    assert (await client.post(path()+f"/sessions/{sid}/stop",headers=AUTH,json=stop)).json() == stopped.json()
    state = (await client.get(path()+f"/sessions/{sid}",headers=AUTH)).json()
    assert state["session"]["controller"] == "none" and state["lease"]["state"] == "ready"
    assert (await client.get(path()+f"/sessions/{sid}/observation",headers=AUTH)).status_code == 409


@pytest.mark.asyncio
async def test_explicit_grant_current_rotation_privacy_and_lifecycle(visual_host, monkeypatch):
    client, agents, _ = visual_host
    response = await client.post(path()+"/requests",headers=AUTH,json=intent())
    assert response.status_code == 200, response.text
    sid = response.json()["session_id"]
    feature = agents["kite"].features["VisualComputerFeature"]
    feature._cfg["granted_actions"].remove(VisualAction.OBSERVE.value)
    assert (await client.get(path()+f"/sessions/{sid}/observation",headers=AUTH)).status_code == 403
    feature._cfg["granted_actions"].append(VisualAction.OBSERVE.value)
    agents["kite"].set_mode(PrivacyMode.EPHEMERAL)
    assert (await client.get(path()+f"/sessions/{sid}",headers=AUTH)).status_code == 403
    stop = {"request_id":"stop", "session_id":sid, "expected_control_epoch":0, "operation":"pause"}
    assert (await client.post(path()+f"/sessions/{sid}/stop",headers=AUTH,json=stop)).status_code == 200
    assert (await client.post(path()+f"/sessions/{sid}/release",headers=AUTH)).json()["released"] is True
    agents["kite"].set_mode(PrivacyMode.NORMAL)
    monkeypatch.setenv("KESTREL_API_KEY", "rotated-visual-key")
    assert (await client.get(path()+"/sessions",headers=AUTH)).status_code == 401
    assert (await client.get(path()+"/sessions",headers={"X-API-Key":"rotated-visual-key"})).status_code == 200
    await feature.on_disable()
    assert (await client.get(path()+"/sessions",headers={"X-API-Key":"rotated-visual-key"})).status_code == 403
    await feature.on_enable()
    assert (await client.get(path()+"/sessions",headers={"X-API-Key":"rotated-visual-key"})).status_code == 200


@pytest.mark.asyncio
async def test_owner_payload_cookie_and_peer_never_mint_visual_grant(visual_host):
    from kestrel_sovereign import server
    client, _, _ = visual_host
    body = {**intent(),"scope":{"principal_id":"victim"}}
    invalid = await client.post(path()+"/requests",headers=AUTH,json=body)
    assert invalid.status_code == 400 and "victim" not in invalid.text
    peer = await client.get(path()+"/sessions",headers={**AUTH,server.A2A_TRANSPORT_KEY_HEADER:server.ensure_a2a_transport_key()})
    assert peer.status_code == 403
    oversized = await client.post(path()+"/requests",headers=AUTH,json={"private":"secret"*20000})
    assert oversized.status_code == 400 and "secret" not in oversized.text


@pytest.mark.asyncio
async def test_bound_context_rechecks_live_grant_and_exact_feature_generation(visual_host):
    from starlette.requests import Request
    from kestrel_sdk.operator import OperatorAuthorizationError
    from kestrel_sovereign.auth import CallerContext
    client, agents, app = visual_host
    admitted = await client.post(path()+"/requests",headers=AUTH,json=intent())
    sid = admitted.json()["session_id"]
    agent = agents["kite"]
    feature = agent.features["VisualComputerFeature"]
    caller = CallerContext.sovereign(credential=KEY)
    request = Request({"type":"http", "app":app, "headers":[(b"x-api-key",KEY.encode())], "state":{"caller":caller,"agent":agent}})
    async with feature._resolve_context(request) as context:
        feature._cfg["granted_actions"].remove(VisualAction.STATUS.value)
        with pytest.raises(OperatorAuthorizationError):
            await feature.runtime.status(context, sid)
        feature._cfg["granted_actions"].append(VisualAction.STATUS.value)
        feature._generation += 1
        with pytest.raises(OperatorAuthorizationError):
            await feature.runtime.status(context, sid)


@pytest.mark.asyncio
async def test_disabled_visual_config_exposes_no_tools_or_router(harness, monkeypatch):
    monkeypatch.setattr(feature_module,"load_section",lambda section:{})
    feature = VisualComputerFeature(SimpleNamespace())
    await feature.initialize()
    assert feature.get_tools() == [] and feature.get_router() is None and feature.get_ui_contributions() is None


@pytest.mark.asyncio
async def test_canonical_factory_receives_only_trusted_host_egress_guard(harness, monkeypatch):
    h = harness
    cfg={"enabled":True, "granted_actions":[VisualAction.ADMIT.value],
         "providers":{"local-docker-browser":{"image":"private-image"}},
         "drivers":{"local-docker-browser":{}}}
    monkeypatch.setattr(feature_module,"load_section",lambda section:{"visual_computer":cfg})
    captured={}
    def factory(**settings):
        captured.update(settings)
        return {"local-docker-browser":h.adapter},{"local-docker-browser":h.adapter}
    monkeypatch.setattr(feature_module,"construct_adapters",factory)
    agent=PolicyHolder()
    agent.did="did:test:kite"
    agent._raw_storage=SimpleNamespace(db=h.db)
    agent.features={}
    feature=VisualComputerFeature(agent)
    await feature.initialize()
    guard=captured["drivers"]["local-docker-browser"]["egress_guard"]
    assert callable(guard) and guard.__self__ is feature.runtime.egress
    assert len(captured["providers"]["local-docker-browser"]["credential_key"]) == 32
    assert "egress_guard" not in cfg["drivers"]["local-docker-browser"]


@pytest.mark.asyncio
async def test_actual_stop_route_does_not_wait_behind_privacy_writer(visual_host,harness):
    client,agents,_=visual_host
    admitted=await client.post(path()+"/requests",headers=AUTH,json=intent())
    sid=admitted.json()["session_id"]
    observation=await client.get(path()+f"/sessions/{sid}/observation",headers=AUTH)
    frame=observation.json()["frame"]
    entered=asyncio.Event()
    async def blocked():
        entered.set()
        await asyncio.Event().wait()
    harness.adapter.before_effect=blocked
    payload={"action_id":"blocked-action","session_id":sid,"observation_id":frame["observation_id"],"control_epoch":0,
        "not_after":(datetime.now(UTC)+timedelta(seconds=20)).isoformat(),"action":{"kind":"click","x":1,"y":1}}
    active=asyncio.create_task(client.post(path()+f"/sessions/{sid}/actions",headers=AUTH,json=payload))
    await entered.wait()
    async def transition():
        async with agents["kite"].gate.exclusive():
            pass
    writer=asyncio.create_task(transition())
    await asyncio.sleep(0)
    try:
        stop={"request_id":"priority-stop","session_id":sid,"expected_control_epoch":0,"operation":"pause"}
        response=await asyncio.wait_for(client.post(path()+f"/sessions/{sid}/stop",headers=AUTH,json=stop),0.5)
        assert response.status_code == 200 and response.json()["stopped"] is True
        assert not writer.done()
    finally:
        active.cancel()
        writer.cancel()
        await asyncio.gather(active,writer,return_exceptions=True)


@pytest.mark.asyncio
async def test_visual_plan_uses_original_caller_configured_model_and_host_router(visual_host,monkeypatch):
    import sys
    from kestrel_sovereign.auth import CallerContext,caller_context_scope
    from kestrel_sdk.operator import OperatorAuthorizationError
    client,agents,_=visual_host
    admitted=await client.post(path()+"/requests",headers=AUTH,json=intent())
    sid=admitted.json()["session_id"]
    feature=agents["kite"].features["VisualComputerFeature"]
    service=object()
    agents["kite"].llm_service=service
    feature._cfg["visual_model"]="explicit-visual-model"
    agents["kite"].privacy_config.processing="local"
    calls=[]
    class Planner:
        def __init__(self,runtime,llm_service):
            assert runtime is feature.runtime and llm_service is service
        async def plan(self,context,session_id,prompt,**kwargs):
            calls.append((context,session_id,prompt,kwargs))
            return {"frame":None,"action_request":None,"answer":"Finished"}
    monkeypatch.setitem(sys.modules,"kestrel_sovereign.visual_computer.inference",SimpleNamespace(VisualInference=Planner))
    with caller_context_scope(CallerContext.sovereign(credential=KEY)):
        answer=await feature.visual_plan(sid,"Summarize this screen")
    assert answer == {"frame":None,"action_request":None,"answer":"Finished"}
    context,session,prompt,kwargs=calls[0]
    assert context.tenant_id == "did:test:kite" and context.acting_agent_id == "did:test:kite"
    assert session == sid and kwargs == {"model_override":"explicit-visual-model","force_local_only":True}
    with caller_context_scope(CallerContext.authenticated("peer")),pytest.raises(OperatorAuthorizationError):
        await feature.visual_plan(sid,"Read")
    assert len(calls)==1
