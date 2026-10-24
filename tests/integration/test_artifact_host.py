"""Real host auth/routing and durable artifact service; no provider calls.

The runtime holders are controlled fixtures, not a full initialized Kite.
SQLite and PostgreSQL run the same owner, credential and lifecycle checks.
"""

import asyncio
import base64
import json
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from itsdangerous import TimestampSigner

from kestrel_sovereign.features.artifacts import ArtifactsFeature
from kestrel_sovereign.privacy import PrivacyMode
from tests.integration.test_artifact_repository import repository as _repository_fixture
from tests.integration.test_artifact_service import PolicyHolder

repository = _repository_fixture
KEY = "artifact-host-test-only-key"
AUTH = {"X-API-Key": KEY}


@pytest_asyncio.fixture
async def host(repository, monkeypatch):
    from kestrel_sovereign import server

    monkeypatch.setenv("KESTREL_API_KEY", KEY)
    monkeypatch.setenv("KESTREL_A2A_TRANSPORT_ONLY", "false")
    store, _ = repository
    agents = {}
    for name in ("kite", "other"):
        agent = PolicyHolder()
        agent.did = "did:test:" + name
        agent._raw_storage = SimpleNamespace(db=store.db)
        agent.features = {}
        agent._track_background_task = lambda coro, *, name: asyncio.create_task(coro, name=name)
        feature = ArtifactsFeature(agent)
        await feature.initialize()
        agent.features[feature.name] = feature
        await feature.on_enable()
        agents[name] = agent

    app = server.app
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    app.state.agent = None
    app.state.agent_manager = SimpleNamespace(
        list_agents=lambda: dict(agents), get_agent=lambda name: agents.get(name),
    )
    server._mount_feature_routers(app)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            yield client, agents, app
    finally:
        for agent in agents.values():
            await agent.features["ArtifactsFeature"].shutdown()
        server._unmount_feature_routers(app)
        app.state.agent = original_agent
        app.state.agent_manager = original_manager


def path(name="kite"):
    return f"/api/agents/{name}/api/artifacts"


async def publish(client, name="kite", key="publication", ref=None, content=b"private"):
    staged = await client.post(path(name) + "/staging", headers=AUTH, json={
        "publication_key": key, "path": "report.txt", "media_type": "text/plain",
        "content_base64": base64.b64encode(content).decode(),
    })
    assert staged.status_code == 200, staged.text
    body = {"idempotency_key": key, "title": "Report",
            "manifest": {"files": [staged.json()], "entrypoint": "report.txt"}}
    if ref:
        body.update(artifact_id=ref["artifact_id"], expected_revision_id=ref["revision_id"])
    response = await client.post(path(name) + "/publications", headers=AUTH, json=body)
    assert response.status_code == 200, response.text
    return response.json()["ref"], body


@pytest.mark.asyncio
async def test_registered_host_publish_edit_scope_and_reload(host):
    client, agents, _ = host
    first, body = await publish(client)
    second, _ = await publish(client, key="edit", ref=first, content=b"edited")
    head = path() + f'/{first["artifact_id"]}/head'
    assert (await client.get(head, headers=AUTH)).json()["ref"] == second
    assert (await client.get(head.replace("/kite/", "/other/"), headers=AUTH)).status_code == 404
    assert (await client.get(path("other"), headers=AUTH)).json()["items"] == []
    agent = agents["kite"]
    old = agent.features["ArtifactsFeature"]
    await old.shutdown()
    assert (await client.get(head, headers=AUTH)).status_code == 403
    replacement = ArtifactsFeature(agent)
    await replacement.initialize()
    agent.features[replacement.name] = replacement
    await replacement.on_enable()
    # Mounted router dispatches to the current instance; no second mount.
    assert (await client.get(head, headers=AUTH)).json()["ref"] == second
    replay = await client.post(path() + "/publications", headers=AUTH, json=body)
    assert replay.json()["ref"] == first
    data = await client.get(path() + f'/{second["artifact_id"]}/revisions/{second["revision_id"]}/file',
                            headers=AUTH, params={"path": "report.txt"})
    assert data.content == b"edited"
    assert "no-store" in data.headers["cache-control"]


@pytest.mark.asyncio
async def test_only_current_explicit_sovereign_credentials(host, monkeypatch):
    from kestrel_sovereign import server
    client, _, _ = host
    assert (await client.get(path())).status_code == 401
    assert (await client.get(path(), params={"api_key": KEY})).status_code == 401
    assert (await client.get(path(), headers={"Authorization": "Bearer " + KEY})).status_code == 200
    peer = await client.get(path(), headers={
        **AUTH, server.A2A_TRANSPORT_KEY_HEADER: server.ensure_a2a_transport_key(),
    })
    assert peer.status_code == 403
    payload = base64.b64encode(json.dumps({"user_email": "person@example.test"}).encode())
    cookie = TimestampSigner(server._SESSION_SECRET).sign(payload).decode()
    client.cookies.set("kestrel_session", cookie)
    for method in ("GET", "POST"):
        response = await client.request(method, path() + ("/staging" if method == "POST" else ""),
                                        headers={"X-CSRF-Token": "token"})
        assert response.status_code == 403  # CSRF cannot manufacture a grant.
    client.cookies.clear()
    monkeypatch.setenv("KESTREL_API_KEY", "rotated-artifact-test-key")
    assert (await client.get(path(), headers=AUTH)).status_code == 401
    assert (await client.get(path(), headers={"X-API-Key": "rotated-artifact-test-key"})).status_code == 200


@pytest.mark.asyncio
async def test_query_key_exception_cannot_authorize_artifact_receipts(host):
    client, _, _ = host
    ref, _ = await publish(client, key="link-qr.png")
    url = path() + "/publications/link-qr.png"
    for headers in ({}, {"X-API-Key": "wrong"}, {"Authorization": "Bearer wrong"}):
        denied = await client.get(url, params={"api_key": KEY}, headers=headers)
        assert denied.status_code == 403
        assert ref["artifact_id"] not in denied.text
    assert (await client.get(url, headers=AUTH)).json()["ref"] == ref


@pytest.mark.asyncio
async def test_privacy_transition_blocks_reads_and_writes_but_allows_erasure(host):
    client, agents, _ = host
    ref, _ = await publish(client)
    agent = agents["kite"]
    async with agent.gate.exclusive():
        agent.set_mode(PrivacyMode.EPHEMERAL)
    assert (await client.get(path(), headers=AUTH)).status_code == 403
    assert (await client.post(path() + "/staging", headers=AUTH, json={
        "publication_key": "blocked", "path": "a", "media_type": "text/plain", "content_base64": "YQ==",
    })).status_code == 403
    erased = await client.delete(path() + '/' + ref["artifact_id"], headers=AUTH,
                                 params={"expected_revision_id": ref["revision_id"]})
    assert erased.status_code == 204
    async with agent.gate.exclusive():
        agent.set_mode(PrivacyMode.NORMAL)
    assert (await client.get(path(), headers=AUTH)).json()["items"] == []


@pytest.mark.asyncio
async def test_link_destinations_are_not_implicitly_granted(host):
    client, _, _ = host
    ref, _ = await publish(client)
    response = await client.post(path() + "/links", headers=AUTH, json={
        "ref": ref, "context_kind": "conversation", "context_id": "another-owner",
    })
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_credential_revoked_during_upload_does_not_publish(host, monkeypatch):
    client, agents, _ = host
    feature = agents["kite"].features["ArtifactsFeature"]
    entered, release = asyncio.Event(), asyncio.Event()

    async def body():
        yield b'{"publication_key":"delayed",'
        entered.set()
        await release.wait()
        yield b'"path":"a","media_type":"text/plain","content_base64":"YQ=="}'

    pending = asyncio.create_task(client.post(path() + "/staging", headers={**AUTH, "Content-Type": "application/json"}, content=body()))
    await entered.wait()
    monkeypatch.setenv("KESTREL_API_KEY", "rotated-artifact-test-key")
    release.set()
    assert (await pending).status_code == 403
    assert (await feature._service.repository.db.fetchone("SELECT COUNT(*) FROM artifact_staging"))[0] == 0


@pytest.mark.asyncio
async def test_disable_and_reenable_cannot_revive_upload_context(host):
    client, agents, _ = host
    feature = agents["kite"].features["ArtifactsFeature"]
    entered, release = asyncio.Event(), asyncio.Event()

    async def body():
        yield b'{"publication_key":"delayed",'
        entered.set()
        await release.wait()
        yield b'"path":"a","media_type":"text/plain","content_base64":"YQ=="}'

    pending = asyncio.create_task(client.post(path() + "/staging", headers={**AUTH, "Content-Type": "application/json"}, content=body()))
    await entered.wait()
    await feature.on_disable()
    await feature.on_enable()
    release.set()
    assert (await pending).status_code == 403
    assert (await client.get(path(), headers=AUTH)).status_code == 200


def test_artifact_feature_is_discoverable():
    from kestrel_sovereign.features import discover_feature_class_by_name
    assert discover_feature_class_by_name("ArtifactsFeature") is ArtifactsFeature


@pytest.mark.asyncio
async def test_shutdown_joins_inflight_read_and_denies_private_delivery(host, monkeypatch):
    client, agents, _ = host
    ref, _ = await publish(client)
    feature = agents["kite"].features["ArtifactsFeature"]
    repository = feature._service.repository
    original = repository.read
    entered, release = asyncio.Event(), asyncio.Event()

    async def held_read(*args, **kwargs):
        result = await original(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(repository, "read", held_read)
    pending = asyncio.create_task(client.get(
        path() + f'/{ref["artifact_id"]}/revisions/{ref["revision_id"]}/file',
        headers=AUTH, params={"path": "report.txt"},
    ))
    await entered.wait()
    shutdown = asyncio.create_task(feature.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    assert feature._accepting is False
    release.set()
    response = await pending
    await shutdown
    assert response.status_code == 403
    assert response.json() == {"error": {"code": "artifact_access_denied"}}
    assert (await repository.db.fetchone("SELECT COUNT(*) FROM artifact_heads"))[0] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["shutdown", "credential", "privacy", "cancel"])
async def test_real_middleware_retains_custody_through_outer_send(host, monkeypatch, transition):
    client, agents, app = host
    ref, _ = await publish(client, content=b"private-outer-send")
    agent = agents["kite"]
    feature = agent.features["ArtifactsFeature"]
    entered, release = asyncio.Event(), asyncio.Event()
    messages = []
    url = path() + f'/{ref["artifact_id"]}/revisions/{ref["revision_id"]}/file'
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
             "http_version": "1.1", "method": "GET", "scheme": "http", "path": url,
             "raw_path": url.encode(), "query_string": b"path=report.txt", "root_path": "",
             "server": ("test", 80), "client": ("test", 1), "headers": [(b"x-api-key", KEY.encode())]}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            entered.set()
            await release.wait()
        messages.append(message)

    async def change_privacy():
        async with agent.gate.exclusive():
            agent.set_mode(PrivacyMode.EPHEMERAL)

    serving = asyncio.create_task(app(scope, receive, send))
    changing = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if transition == "credential":
            monkeypatch.setenv("KESTREL_API_KEY", "rotated-artifact-test-key")
        elif transition == "cancel":
            serving.cancel()
        else:
            changing = asyncio.create_task(feature.shutdown() if transition == "shutdown" else change_privacy())
            await asyncio.sleep(0)
            assert not changing.done(), "transition completed before outer transport released the response"
        release.set()
        outcome = await asyncio.wait_for(asyncio.gather(serving, return_exceptions=True), 5)
        if changing is not None:
            await asyncio.wait_for(changing, 5)
        body = b"".join(message.get("body", b"") for message in messages)
        if transition == "privacy":
            assert body == b"private-outer-send"
            assert agent.privacy_mode == PrivacyMode.EPHEMERAL
            assert outcome == [None]
        else:
            assert b"private-outer-send" not in body
            assert isinstance(outcome[0], BaseException)
        assert not feature._service.authority._bindings
        assert not feature._service.authority._operations
        assert not agent.gate.locked()
    finally:
        release.set()
        tasks = [serving] + ([changing] if changing is not None else [])
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_actual_host_reenables_same_instance(repository, tmp_path):
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    agent = KestrelAgent(did="did:test:reenable-artifacts", storage_path=str(tmp_path / "host.db"))
    agent.features = {}
    agent._raw_storage = SimpleNamespace(db=repository[0].db)
    feature = ArtifactsFeature(agent)
    try:
        await agent._register_feature(feature)
        service = feature._service
        first_sweep = feature._sweep_task
        await agent._unregister_feature_runtime(feature, unload=False)
        feature.enabled = False
        assert first_sweep.done()
        await agent._activate_feature_runtime(feature)
        assert agent.features[feature.name] is feature
        assert feature._service is service
        assert feature.enabled and feature._accepting
        assert feature._sweep_task is not first_sweep
        assert not feature._sweep_task.done()
    finally:
        await agent._unregister_feature_runtime(feature, unload=False)
    assert feature._sweep_task.done()


@pytest.mark.asyncio
async def test_active_host_sweeps_expired_staging_without_owner_return(host, monkeypatch):
    from datetime import timedelta
    client, agents, _ = host
    feature = agents["kite"].features["ArtifactsFeature"]
    repo = feature._service.repository
    await client.post(path() + "/staging", headers=AUTH, json={
        "publication_key": "abandoned", "path": "a.txt", "media_type": "text/plain", "content_base64": "YQ==",
    })
    assert (await repo.db.fetchone("SELECT COUNT(*) FROM artifact_staging"))[0] == 1
    await feature.on_disable()
    now = repo.clock()
    monkeypatch.setattr(repo, "clock", lambda: now + timedelta(hours=2))
    monkeypatch.setattr(feature, "_sweep_seconds", 0.01)
    await feature.on_enable()
    async with asyncio.timeout(3):
        while (await repo.db.fetchone("SELECT COUNT(*) FROM artifact_staging"))[0]:
            await asyncio.sleep(0.01)
    assert (await repo.db.fetchone("SELECT COUNT(*) FROM artifact_blobs"))[0] == 0
    await feature.on_disable()
    assert feature._sweep_task.done()


@pytest.mark.asyncio
async def test_keyless_agent_boots_without_advertising_artifacts(temp_db, monkeypatch):
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService
    from tests.shared.native_shutdown_budget import shutdown_fixture_agent
    monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)
    monkeypatch.delenv("KESTREL_DATA_KEY_FILE", raising=False)
    llm = LLMService()
    agent = KestrelAgent(did="did:test:keyless-artifact-boot", storage_path=str(temp_db),
                         llm_service=llm, privacy_mode=PrivacyMode.NORMAL)
    try:
        await agent.initialize()
        assert agent.features
        assert "ArtifactsFeature" not in agent.features
        assert any(item.feature_name == "ArtifactsFeature" and "KESTREL_DATA_KEY" in item.reason
                   for item in agent.rejected_feature_contributions)
    finally:
        await shutdown_fixture_agent(agent)
        await llm.close()
