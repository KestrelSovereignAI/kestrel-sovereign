"""Opt-in actual keyed host, Docker pixels, local vision and durable output.

Only an explicitly named already installed LOCAL Ollama model is used. This
test provisions no cloud resource and never downloads model weights.
"""
import asyncio
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import socket
from uuid import uuid4

import httpx
import pytest
from cryptography.fernet import Fernet


@pytest.mark.asyncio
async def test_real_local_visual_product(tmp_path, monkeypatch):
    model = os.environ.get("KESTREL_VISUAL_TEST_LOCAL_MODEL")
    destination = os.environ.get("KESTREL_VISUAL_TEST_EVIDENCE")
    docker_socket = os.environ.get("KESTREL_VISUAL_TEST_DOCKER_SOCKET", "/var/run/docker.sock")
    if not model or not destination:
        pytest.skip("Requires explicit installed local vision model and private evidence directory")
    output = Path(destination)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    identity = uuid4().hex
    custody = {"identity":identity,"model":model,"scope":"real local keyed host and Docker visual product",
               "image_id":None,"session_id":None,"physical_cleanup_confirmed":False,"cognitive_task":False}
    def record():
        (output/"custody.json").write_text(json.dumps(custody,indent=2)+"\n")
    record()
    for name in tuple(os.environ):
        if name.startswith(("KESTREL_","OTEL_")) or name.endswith(("_API_KEY","_TOKEN","_SECRET")):
            monkeypatch.delenv(name,raising=False)
    from kestrel_sovereign.cli_demo import _build_smoke_env
    (tmp_path/"home").mkdir();(tmp_path/"data").mkdir()
    for name,value in _build_smoke_env(os.environ,home=tmp_path/"home",data_dir=tmp_path/"data",
                                      data_key=Fernet.generate_key().decode()).items():
        monkeypatch.setenv(name,value)
    key="visual-product-"+secrets.token_hex(24)
    monkeypatch.setenv("KESTREL_API_KEY",key)
    monkeypatch.setenv("KESTREL_DATA_KEY",Fernet.generate_key().decode())
    monkeypatch.setenv("KESTREL_TRACING_ENABLED","0")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED","1")
    monkeypatch.setenv("KESTREL_SKIP_DOTENV","1")
    async with httpx.AsyncClient(trust_env=False,timeout=5) as local:
        tags=(await local.get("http://127.0.0.1:11434/api/tags")).json()["models"]
        installed=next(item for item in tags if item["name"]==model)
        assert not installed.get("remote_host") and not installed.get("remote_model")
        shown=(await local.post("http://127.0.0.1:11434/api/show",json={"model":model})).json()
        assert "vision" in shown.get("capabilities",[]) and not shown.get("remote_host") and not shown.get("remote_model")
        custody["model_digest"]=installed["digest"]
        custody["model_capabilities"]=shown["capabilities"]
    from kestrel_sovereign.visual_computer.docker import DockerBoundary,LABEL,PROVIDER
    from scripts.verify_visual_browser_sandbox import CONTEXT,source_identity
    boundary=DockerBoundary(docker_socket)
    tag="kestrel-visual-host-proof:"+identity
    custody["source_before"]=source_identity()
    agent=llm=runner=serving=None
    app=lifespan=None
    sock=None
    try:
        build=await boundary.invoke(["build","--label",f"{LABEL}.proof={identity}","--tag",tag,str(CONTEXT)],timeout=600)
        (output/"build.log").write_bytes(build)
        image=json.loads(await boundary.invoke(["image","inspect",tag]))[0]
        assert image["Config"]["Labels"][LABEL+".proof"]==identity
        custody["image_id"]=image["Id"];record()
        from kestrel_sovereign import server
        from kestrel_sovereign.kestrel_agent import KestrelAgent
        from kestrel_sovereign.llm import service as llm_module
        from kestrel_sovereign.features.visual_computer import feature as visual_feature
        from kestrel_sovereign.visual_computer.runtime import VisualAction
        config={"enabled":True,"granted_actions":[a.value for a in VisualAction],"operation_seconds":120,
            "visual_model":f"ollama:local/{model}",
            "visual_request_options":{"raw":{"format":"json","think":False,"keep_alive":0,
                                              "options":{"num_ctx":8192,"temperature":0}}},
            "providers":{PROVIDER:{"docker_socket":docker_socket,"image":"local-worker@"+image["Id"],"profile_id":"offline-v1"}},
            "drivers":{PROVIDER:{"docker_socket":docker_socket}}}
        monkeypatch.setattr(visual_feature,"load_section",lambda section:{"visual_computer":config})
        # Configure the production registry, including its retained catalog,
        # with one fixed local endpoint. No model operation is replaced.
        load_section=llm_module.load_section
        llm_config={"route_priority":["ollama:local"],"vendors":{"ollama":{"is_cloud":False,
            "routes":{"local":{"adapter":"OllamaAdapter","host":"http://127.0.0.1:11434",
                                "model":model,"max_output_tokens":1024}}}}}
        monkeypatch.setattr(llm_module,"load_section",lambda section:llm_config if section=="llm" else load_section(section))
        llm=llm_module.LLMService(agent_data_dir=tmp_path)
        await llm.discover_all_models(use_cache=False)
        agent=KestrelAgent(did="did:test:visual-host-product",storage_path=str(tmp_path/"agent.db"),
            llm_service=llm,allowed_features={"VisualComputerFeature","ArtifactsFeature"})
        app=server.app;lifespan=app.router.lifespan_context
        monkeypatch.setattr(app.state,"agent",agent,raising=False)
        monkeypatch.setattr(app.state,"agent_manager",None,raising=False)
        await agent.initialize()
        assert {"visual_plan","visual_computer_action","artifact_publish_text"} <= {
            item["function"]["name"] for item in agent._build_all_tools()}
        server._mount_feature_routers(app)
        import uvicorn
        sock=socket.socket();sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
        runner=uvicorn.Server(uvicorn.Config(app,log_config=None,access_log=False,lifespan="off"))
        serving=asyncio.create_task(runner.serve(sockets=[sock]))
        async with asyncio.timeout(15):
            while not runner.started:await asyncio.sleep(.01)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}",trust_env=False,
                                     headers={"X-API-Key":key},timeout=130) as client:
            base="/api/visual-computer"
            now=datetime.now(UTC)
            value={"request_id":identity,"provider_id":PROVIDER,"profile_id":"offline-v1",
                "image":"local-worker@"+image["Id"],"surface":"browser","location":"local",
                "required_capabilities":["screenshot","click","type","key","takeover","files"],
                "requested_at":now.isoformat(),"expires_at":(now+timedelta(minutes=10)).isoformat(),
                "idle_ttl_seconds":300,"ready_timeout_seconds":30,"max_hourly_cost_usd":"0","max_total_cost_usd":"0"}
            response=await client.post(base+"/requests",json=value)
            assert response.status_code==200,response.text
            session=response.json();sid=session["session_id"];custody["session_id"]=sid;record()
            async def act(action):
                response=await client.get(base+f"/sessions/{sid}/observation");assert response.status_code==200,response.text
                frame=response.json()["frame"]
                request={"action_id":uuid4().hex,"session_id":sid,"observation_id":frame["observation_id"],
                    "control_epoch":session["control_epoch"],"not_after":(datetime.now(UTC)+timedelta(seconds=15)).isoformat(),"action":action}
                response=await client.post(base+f"/sessions/{sid}/actions",json=request)
                assert response.status_code==200,response.text
                return response.json()
            await act({"kind":"click","x":20,"y":20})
            code="HARBOR-"+secrets.token_hex(3).upper()
            await act({"kind":"type","text":"Screen code: "+code})
            # The task contains no code. Only actual screen pixels reveal it.
            response=await client.post(base+f"/sessions/{sid}/plan",json={"prompt":
                "Read the displayed screen code. Append exactly one space followed by that code at the current text caret, "
                'using a single action of kind "type", with the code in its "text" field. '
                'Do not scroll, click or replace existing text.'})
            assert response.status_code==200,response.text
            proposed=response.json();request=proposed["action_request"]
            (output/"proposal.json").write_text(json.dumps(proposed,indent=2)+"\n")
            assert request is not None and request["action"]["kind"]=="type",proposed
            assert request["action"]["text"].strip()==code,proposed
            response=await client.post(base+f"/sessions/{sid}/actions",json=request)
            assert response.status_code==200,response.text
            receipt=response.json()
            assert (await client.post(base+f"/sessions/{sid}/actions",json=request)).json()==receipt
            pause={"request_id":uuid4().hex,"session_id":sid,"expected_control_epoch":0,"operation":"pause"}
            response=await client.post(base+f"/sessions/{sid}/stop",json=pause);assert response.status_code==200,response.text
            # The registered broker owns the original workspace operation and
            # both visual/artifact policies through native export and commit.
            current=await client.get(base+f"/sessions/{sid}")
            assert current.status_code==200,current.text
            paused=current.json()["session"]
            export={"transfer_id":uuid4().hex,"session_id":sid,"lease_id":paused["lease_id"],
                "control_epoch":paused["control_epoch"],"workspace_id":"browser","expires_at":paused["expires_at"],
                "limits":{"max_files":1,"max_file_bytes":128*1024,"max_total_bytes":128*1024},
                "paths":["document.html"],"entrypoint":"document.html"}
            published=await client.post("/api/artifacts/workspaces/publish",json={"request":export,
                "publication_key":identity,"title":"Visual product document"})
            assert published.status_code==200,published.text
            ref=published.json()["receipt"]["ref"]
            saved=await client.get(f'/api/artifacts/{ref["artifact_id"]}/revisions/{ref["revision_id"]}/file',params={"path":"document.html"})
            assert saved.status_code==200,saved.text
            document=saved.content
            assert document.count(code.encode())==2
            (output/"document.html").write_bytes(document)
            custody.update(cognitive_task=True,document_sha256=sha256(document).hexdigest(),
                action_id=receipt["action_id"],observation_id=request["observation_id"],control_epoch=request["control_epoch"],artifact_ref=ref)
            response=await client.post(base+f"/sessions/{sid}/release")
            assert response.status_code==200,response.text
            reopened=await client.get(f'/api/artifacts/{ref["artifact_id"]}/revisions/{ref["revision_id"]}/file',params={"path":"document.html"})
            assert reopened.status_code==200 and reopened.content==document
            custody["artifact_survives_browser_release"]=True
            record()
        custody["source_unchanged"]=source_identity()==custody["source_before"]
        assert custody["source_unchanged"]
    finally:
        custody["physical_cleanup_confirmed"]=False
        custody["forced_cleanup_containers"]=[]
        try:
            try:
                if runner is not None and serving is not None and not serving.done() and custody["session_id"]:
                    try:
                        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}",trust_env=False,
                                                     headers={"X-API-Key":key},timeout=30) as cleanup:
                            response=await cleanup.post(f'/api/visual-computer/sessions/{custody["session_id"]}/release')
                            custody["final_host_release_status"]=response.status_code
                    except Exception as error:
                        custody["final_host_release_error"]=type(error).__name__
                if runner is not None:runner.should_exit=True
                if serving is not None:await asyncio.wait_for(serving,15)
                if sock is not None:sock.close()
                if app is not None:
                    server._unmount_feature_routers(app);app.router.lifespan_context=lifespan
                if agent is not None:
                    from tests.shared.native_shutdown_budget import shutdown_fixture_agent
                    await shutdown_fixture_agent(agent)
                if llm is not None:await llm.close()
            finally:
                if custody["image_id"] is not None:
                    remaining=await boundary.invoke(["ps","-aq","--filter","ancestor="+custody["image_id"]])
                    # Retire only this fixture, even after another cleanup
                    # failure. Forced test cleanup cannot prove host cleanup.
                    for cid in remaining.decode().split():
                        owned=await boundary.inspect(cid)
                        assert owned["Image"]==custody["image_id"] and owned["Config"]["Labels"][LABEL+".proof"]==identity
                        custody["forced_cleanup_containers"].append(owned["Id"])
                        await boundary.invoke(["container","rm","--force",owned["Id"]])
                    assert not (await boundary.invoke(["ps","-aq","--filter","ancestor="+custody["image_id"]])).strip()
                    await boundary.invoke(["image","rm",tag])
            assert not custody["forced_cleanup_containers"], "host left fixture containers requiring forced cleanup"
            custody["physical_cleanup_confirmed"]=True
        finally:
            record()
