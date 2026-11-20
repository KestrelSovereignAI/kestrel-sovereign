"""The real Kite command path owns durable admission before native fact writes."""

import asyncio
import json

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import FastAPI

from kestrel_sovereign.agent.invocation import request_provenance
from kestrel_sovereign.api_errors import register_api_error_handlers
from kestrel_sovereign.endpoints.agent import _kite_runtime_observation, router
from kestrel_sovereign.rate_limit import limiter
from starlette.middleware.sessions import SessionMiddleware
from tests.shared.booted_agent import booted_agent
from tests.unit.test_generation_gate import _Daemon, _final, _route
from tests.unit.test_inference_lease_coordinator import _Provider


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_kite_native_save_and_delete_have_original_journal_ownership(tmp_path, monkeypatch):
    monkeypatch.setenv('KESTREL_HOME', str(tmp_path))
    monkeypatch.setenv('KESTREL_DB_PATH', str(tmp_path))
    monkeypatch.setenv('KESTREL_KITE_RELEASE_EVIDENCE', '1')
    monkeypatch.setenv('KESTREL_KITE_RELEASE_EVIDENCE_ROOT', str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'kestrel.toml').write_text('[llm]\nroute_priority=[]\n')
    async with booted_agent(tmp_path, name='OwnedKiteJournal',
            provenance='test:3679-kite-native-journal', features={'MemoryAgencyFeature'}) as agent:
        provenance = request_provenance(source_kind='http_request', source_locator='/agent/invoke')
        with pytest.raises(RuntimeError, match='Kite runtime command did not return a successful typed result'):
            await _kite_runtime_observation(agent,
                request_id='owned-kite-delete-missing', provenance=provenance,
                request={'operation': 'delete', 'nonce': '3' * 64})
        operation, observation = await _kite_runtime_observation(agent,
            request_id='owned-kite-save', provenance=provenance,
            request={'operation': 'save', 'nonce': '1' * 64,
                     'value': 'kite-evidence-owned-native-journal-region'})
        assert (operation, observation) == ('save', {'fact_write_count': 1})
        operation, observation = await _kite_runtime_observation(agent,
            request_id='owned-kite-delete', provenance=provenance,
            request={'operation': 'delete', 'nonce': '2' * 64})
        assert (operation, observation) == ('delete', {'fact_delete_count': 1})
        assert await agent.tool_effect_store.database.fetchall(
            'SELECT state FROM tool_effect_claims') == [
                ('RETURNED_ERROR',), ('RETURNED_OK',), ('RETURNED_OK',)]
        assert await agent.tool_effect_store.database.fetchall(
            'SELECT state FROM tool_effect_operations') == [('CLOSED',)] * 3
        assert not agent._active_request_ids


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_kite_invoke_signs_only_the_real_completed_offer(tmp_path, monkeypatch):
    from kestrel_sovereign.features.inference_lease import feature
    from kestrel_sovereign.server import auth_middleware

    key = "isolated-kite-offer-test-api-key"
    monkeypatch.setenv("KESTREL_API_KEY", key)
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path))
    monkeypatch.delenv("KESTREL_KITE_RELEASE_EVIDENCE", raising=False)
    monkeypatch.setenv("KESTREL_KITE_INFERENCE_OFFER_EVIDENCE", "1")
    monkeypatch.setenv("KESTREL_KITE_RELEASE_EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kestrel.toml").write_text("[llm]\nroute_priority=[]\n")
    provider = _Provider("local-catalog", total="0.20", ready_seconds=30)
    monkeypatch.setattr(feature, "discover_inference_lease_providers",
        lambda: {provider.provider_name: provider})
    offer_args = {"request": {
        "model": "qwen3:8b", "max_hourly_cost_usd": "1", "max_total_cost_usd": "0.40",
        "allowed_regions": ["us-tx-1"], "request_id": "real-offer-request",
    }}
    plain_answer = "I claim I prepared an offer, but this sentence proves nothing."
    sent_calls = set()
    paid_args = None

    def answer(body):
        response = _final("stop", plain_answer)
        advertised = {tool["function"]["name"] for tool in body.get("tools", [])}
        choices = (
            ("inference_lease_offer", offer_args),
            ("inference_lease_feature", {"task": "Please execute the offer tool."}),
        )
        if paid_args is not None:
            choices = (("inference_lease_acquire_prepared", paid_args),)
        for name, arguments in choices:
            if name in advertised and name not in sent_calls:
                sent_calls.add(name)
                response = _final("stop")
                response["message"]["tool_calls"] = [
                    {"function": {"name": name, "arguments": arguments}},
                ]
                break
        return httpx.Response(200, json=response)

    async with booted_agent(tmp_path, name="Kite", is_test_instance=False,
            provenance="test:4193-kite-inference-offer", features={"InferenceLeaseFeature"}) as agent:
        daemon = _Daemon(answer)
        service = agent.llm_service
        route = _route(agent, "local", daemon)
        monkeypatch.setattr(service, "providers", [route])
        monkeypatch.setattr(service, "config", {**(service.config or {}), "route_priority": [route["name"]]})
        monkeypatch.setattr(service, "_mandate_preference", {})
        app = FastAPI()
        app.state.agent = agent
        app.state.limiter = limiter
        app.middleware("http")(auth_middleware)
        app.add_middleware(SessionMiddleware, secret_key=key)
        app.include_router(router)
        register_api_error_handlers(app)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://kite") as client:
            owned = {"X-API-Key": key}
            base = {"input": "Please execute the offer tool.", "request_id": "real-invoke",
                    "session_id": "real-offer-session",
                    "kite_inference_offer_nonce": "a" * 64}
            denied = await client.post("/api/agent/invoke", json=base)
            assert denied.status_code == 401
            public_key_path = "/api/agent/inference-offer-evidence/key"
            key_response = await client.get(public_key_path, headers=owned)
            assert key_response.status_code == 200, key_response.text
            public_key = key_response.json()["kite_evidence_public_key"]
            forwarded = await client.get(public_key_path, headers={**owned, "Forwarded": "for=192.0.2.1"})
            assert forwarded.status_code == 404
            legacy = await client.post("/api/agent/invoke", headers=owned, json={
                "kite_evidence": {"operation": "diagnostics", "nonce": "c" * 64},
                "request_id": "legacy-forbidden",
            })
            assert legacy.status_code == 404
            response = await client.post("/api/agent/invoke", json=base, headers=owned)
            assert response.status_code == 200, (response.text, sent_calls, [
                sorted(tool["function"]["name"] for tool in chat.get("tools", []))
                for chat in daemon.chats
            ])
            assert response.headers["X-Request-ID"] == base["request_id"]
            envelope = response.json()["kite_evidence"]
            signature = envelope.pop("signature")
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key)).verify(
                bytes.fromhex(signature),
                json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode(),
            )
            assert envelope["nonce"] == base["kite_inference_offer_nonce"]
            assert envelope["contract"] == "kite-http-evidence-v1"
            assert envelope["operation"] == "private_inference_offer"
            observation = envelope["observation"]
            assert observation["actor"] == "api_key"
            assert observation["invocation_id"] == base["request_id"]
            assert observation["owner_id"] == agent.agent_id
            assert observation["tool"] == "inference_lease_offer"
            assert observation["outcome"] == "RETURNED_OK"
            assert observation["offer"]["request"]["request_id"] == "real-offer-request"
            assert observation["offer"]["quote"]["provider_name"] == "local-catalog"
            assert observation["offer"]["offer_digest"].startswith("sha256:")
            assert await agent.tool_effect_store.database.fetchall(
                "SELECT state FROM tool_effect_claims"
            ) == [("RETURNED_OK",)]
            assert await agent.tool_effect_store.database.fetchall(
                "SELECT state FROM tool_effect_operations"
            ) == [("CLOSED",)]

            replay = await client.post("/api/agent/invoke", json={
                **base, "request_id": "different-invoke",
            }, headers=owned)
            assert replay.status_code == 409
            assert replay.json()["error"]["code"] == "kite_evidence_nonce_replayed"
            prose = await client.post("/api/agent/invoke", json={
                "input": plain_answer, "request_id": "prose-invoke",
                "session_id": "prose-offer-session",
                "kite_inference_offer_nonce": "b" * 64,
            }, headers=owned)
            assert prose.status_code == 409, prose.text
            assert prose.json()["error"]["code"] == "kite_inference_offer_unconfirmed"
            assert "kite_evidence" not in prose.json()

            next_event = asyncio.get_running_loop().create_future()

            async def on_event(name, event):
                if name == "approval_request" and not next_event.done():
                    next_event.set_result(event)

            agent.add_event_listener(on_event)
            paid_args = {"request_id": "real-offer-request",
                         "offer_digest": observation["offer"]["offer_digest"]}
            security = next(f for f in agent.features.values() if type(f).__name__ == "SecurityFeature")
            turn = asyncio.create_task(client.post("/api/agent/invoke", headers=owned, json={
                "input": "Request approval for the exact offer.",
                "request_id": "exact-acquisition-turn",
                "session_id": "real-offer-session",
            }))
            try:
                approval = await asyncio.wait_for(next_event, timeout=10)
                assert approval["invocation_id"] == "exact-acquisition-turn"
                assert approval["feature"] == "InferenceLeaseFeature"
                assert approval["tool"] == "inference_lease_acquire_prepared"
                assert approval["args"] == paid_args
                assert await security.approval_queue.submit_decision(approval["id"], approved=False)
                response = await turn
                assert response.status_code == 200, response.text
            finally:
                agent.remove_event_listener(on_event)
                if not turn.done():
                    turn.cancel()
                    await asyncio.gather(turn, return_exceptions=True)
    assert provider.acquire_calls == 0 and provider.release_calls == 0
