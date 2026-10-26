"""Real router and OpenAI image transport; only the model server is scripted.

These are contract/race proofs, never evidence that a live model sees pixels.
"""
import asyncio
import base64
from datetime import UTC, datetime
from hashlib import sha256
import json
import inspect
from types import SimpleNamespace
from uuid import uuid4

import httpx
import openai
import pytest
import pytest_asyncio
from kestrel_sdk.computer_use import ActionRequest, ControlRequest, ScreenFrame, ScreenObservation
from kestrel_sdk.resources import ResourceConflict, ResourceUnsupported

from kestrel_sovereign.execution_custody import ExecutionAuthorityError, bind_native_execution_admission
from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
from kestrel_sovereign.llm import service as service_module
from kestrel_sovereign.llm.model_cache import get_shared_model_cache
from kestrel_sovereign.llm.model_metadata import ModelInfo
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.visual_computer.inference import VisualInference
from kestrel_sovereign.visual_computer.runtime import VisualAction

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


@pytest_asyncio.fixture
async def inference(harness, monkeypatch, tmp_path):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session = await h.runtime.admit(ctx, h.request(ctx))
    async def screenshot(session):
        return ScreenObservation(frame=ScreenFrame(observation_id=uuid4().hex, session_id=session.session_id,
            control_epoch=session.control_epoch, observed_at=datetime.now(UTC), width=1, height=1,
            image_sha256=sha256(PNG).hexdigest(), media_type="image/png"), image=PNG)
    h.adapter.observe = screenshot
    monkeypatch.setattr(service_module, "load_section", lambda section: {"route_priority":["openai:first","openai:second"]} if section == "llm" else {})
    monkeypatch.setattr(service_module, "load_config", lambda _: {})
    monkeypatch.setattr(service_module.ProviderRegistry, "initialize_providers", lambda _: [])
    service = service_module.LLMService(agent_data_dir=tmp_path)
    cache = get_shared_model_cache()
    cache.clear()
    # Scripted discovery evidence, separate from the actual image wire proof.
    cache.set([ModelInfo(id="gpt-4o", display_name="Synthetic vision", provider="openai", supports_vision=True)])
    posts = []
    state = SimpleNamespace(answer={"action":{"kind":"click", "x":0, "y":0}, "answer":""}, before_request=None, during_response=None)
    async def server(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object":"list", "data":[{"id":"gpt-4o", "object":"model", "created":1, "owned_by":"synthetic"}]})
        body = json.loads(request.content)
        posts.append((request.url.host, body))
        if state.during_response:
            await state.during_response()
        return httpx.Response(200, json={"id":"synthetic-completion", "object":"chat.completion", "created":1, "model":"gpt-4o",
            "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":json.dumps(state.answer)}}],
            "usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}})
    async def request_observer(request):
        if request.url.path.endswith("/chat/completions") and state.before_request:
            await state.before_request()
    clients = []
    for name in ("first", "second"):
        client = openai.AsyncOpenAI(api_key="synthetic", base_url=f"http://{name}.test/v1", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(server), event_hooks={"request":[request_observer]}))
        clients.append(client)
    service.providers = [{"name":f"openai:{name}","vendor":"openai","route":name,"model":"gpt-4o",
        "is_local":True,"adapter":OpenAIAdapter(),"client":client,"capabilities":{"supports_vision":True}}
        for name, client in zip(("first","second"),clients)]
    try:
        yield SimpleNamespace(h=h, ctx=ctx, session=session, service=service, planner=VisualInference(h.runtime,service), posts=posts, state=state)
    finally:
        await service.close()
        for client in clients:
            await client.close()
        cache.clear()


@pytest.mark.asyncio
async def test_actual_router_transports_image_and_proposal_uses_canonical_receipt(inference):
    x = inference
    with x.h.bind(x.ctx):
        result = await x.planner.plan(x.ctx,x.session.session_id,"Click the screenshot")
        request = ActionRequest.model_validate_json(json.dumps(result["action_request"]))
        receipt = await x.h.runtime.act(x.ctx,request)
        duplicate = await x.h.runtime.act(x.ctx,request)
    assert receipt == duplicate and x.h.adapter.effects == 1
    assert len(x.posts) == 1
    content = x.posts[0][1]["messages"][-1]["content"]
    url = next(item["image_url"]["url"] for item in content if item["type"] == "image_url")
    assert base64.b64decode(url.split(",",1)[1]) == PNG
    assert result["frame"]["observation_id"] == request.observation_id
    assert "image_base64" not in json.dumps(result)


@pytest.mark.asyncio
async def test_blind_route_is_not_dispatched_and_fallback_keeps_image(inference):
    x=inference
    x.service.providers[0]["capabilities"]["supports_vision"]=False
    with x.h.bind(x.ctx):
        await x.planner.plan(x.ctx,x.session.session_id,"inspect")
    assert [host for host,_ in x.posts] == ["second.test"]


@pytest.mark.asyncio
async def test_all_blind_routes_explicitly_refuse_without_model_submission(inference):
    x=inference
    for route in x.service.providers:route["capabilities"]["supports_vision"]=False
    with x.h.bind(x.ctx), pytest.raises(ResourceUnsupported):
        await x.planner.plan(x.ctx,x.session.session_id,"inspect")
    assert not x.posts and not x.h.adapter.effects


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["unknown_model", "unknown_transport"])
async def test_unknown_vision_evidence_never_becomes_family_level_fallback(inference,mode):
    x=inference
    if mode == "unknown_model":get_shared_model_cache().clear()
    else:
        for route in x.service.providers:route["capabilities"]["vision_input_mode"]="unknown"
    with x.h.bind(x.ctx),pytest.raises(ResourceUnsupported):
        await x.planner.plan(x.ctx,x.session.session_id,"inspect")
    assert not x.posts


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_request", "during_response"])
async def test_takeover_fence_blocks_native_dispatch_or_delivery_and_fallback(inference,phase):
    x=inference
    async def fence():
        x.h.runtime._fences.add(x.h.runtime._key(x.session.scope,x.session.session_id))
    setattr(x.state,phase,fence)
    with x.h.bind(x.ctx), pytest.raises(ExecutionAuthorityError):
        await x.planner.plan(x.ctx,x.session.session_id,"inspect")
    assert len(x.posts) == (0 if phase == "before_request" else 1)
    assert not x.h.adapter.effects


@pytest.mark.asyncio
async def test_local_only_never_submits_to_remote_routes(inference):
    x=inference
    for route in x.service.providers:route["is_local"]=False
    with x.h.bind(x.ctx),pytest.raises(ResourceUnsupported):
        await x.planner.plan(x.ctx,x.session.session_id,"inspect",force_local_only=True)
    assert not x.posts


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [
    {"action":{"kind":"click","x":1,"y":0}},
    {"action":{"kind":"shell","command":"echo unsafe"}},
    {"action":{"kind":"click","x":0,"y":0},"action_id":"model-selected"},
    {"action":{"kind":"navigate","url":"https://ungranted.test"}},
])
async def test_model_cannot_supply_authority_or_unavailable_input(inference,answer):
    x=inference;x.state.answer=answer
    with x.h.bind(x.ctx),pytest.raises(ResourceUnsupported):
        await x.planner.plan(x.ctx,x.session.session_id,"inspect")
    assert not x.h.adapter.effects


@pytest.mark.asyncio
async def test_cancellation_joins_model_call_and_never_acts(inference):
    x=inference;entered=asyncio.Event();closed=asyncio.Event()
    async def wait():
        entered.set()
        try:await asyncio.Event().wait()
        finally:closed.set()
    x.state.during_response=wait
    with x.h.bind(x.ctx):
        task=asyncio.create_task(x.planner.plan(x.ctx,x.session.session_id,"inspect"))
        await asyncio.wait_for(entered.wait(),5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
    assert closed.is_set() and not x.h.adapter.effects


@pytest.mark.asyncio
async def test_original_grant_revocation_blocks_native_submission(inference):
    x=inference
    async def revoke():x.h.revoked.add(id(x.ctx))
    x.state.before_request=revoke
    with x.h.bind(x.ctx),pytest.raises(ExecutionAuthorityError):
        await x.planner.plan(x.ctx,x.session.session_id,"inspect")
    assert not x.posts


@pytest.mark.asyncio
async def test_completion_proposes_no_input_and_no_private_pixels(inference):
    x=inference;x.state.answer={"action":None,"answer":"Finished the task."}
    with x.h.bind(x.ctx):result=await x.planner.plan(x.ctx,x.session.session_id,"inspect")
    assert result["action_request"] is None and result["answer"] == "Finished the task."
    assert not x.h.adapter.effects


@pytest.mark.asyncio
async def test_actual_takeover_finishes_before_model_response_and_old_proposal_is_denied(inference):
    x=inference;entered=asyncio.Event();release=asyncio.Event()
    async def wait():entered.set();await release.wait()
    x.state.during_response=wait
    with x.h.bind(x.ctx):
        task=asyncio.create_task(x.planner.plan(x.ctx,x.session.session_id,"inspect"))
        try:
            await asyncio.wait_for(entered.wait(),5)
            control=ControlRequest(request_id=uuid4().hex,session_id=x.session.session_id,
                expected_control_epoch=0,operation="takeover")
            taken=await asyncio.wait_for(x.h.runtime.control(x.ctx,x.session.session_id,control,action=VisualAction.TAKEOVER),1)
            assert taken.controller == "human" and not task.done()
            release.set()
            with pytest.raises(ExecutionAuthorityError):await task
        finally:
            release.set()
            if not task.done():task.cancel()
            await asyncio.gather(task,return_exceptions=True)
    assert not x.h.adapter.effects


@pytest.mark.asyncio
async def test_non_http_provider_attempt_cannot_run_before_original_native_admission(inference):
    x=inference;effects=[]
    async def deny():raise ExecutionAuthorityError("original context retired")
    async def effect():effects.append("submitted");return "not authorized"
    operation=effect()
    with bind_native_execution_admission(deny),pytest.raises(ExecutionAuthorityError):
        await x.service._run_provider_attempt(operation,"openai:first","gpt-4o",path="visual-test",
            invocation_context=LLMInvocationContext(redact_content=True))
    assert not effects and inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED


@pytest.mark.asyncio
async def test_private_driver_cannot_forge_bounded_image_metadata(inference):
    x=inference;original=x.h.adapter.observe
    async def forged(session):
        observation=await original(session)
        return observation.model_copy(update={"frame":observation.frame.model_copy(update={"width":4097})})
    x.h.adapter.observe=forged
    with x.h.bind(x.ctx),pytest.raises(ResourceUnsupported):
        await x.planner.plan(x.ctx,x.session.session_id,"inspect")
    assert not x.posts
