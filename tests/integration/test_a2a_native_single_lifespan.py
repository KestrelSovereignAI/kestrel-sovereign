"""The real standalone server publishes after real agent boot and retires routes."""

import asyncio

import httpx
import pytest
from a2a.types import AgentCard
from fastapi import FastAPI
from google.protobuf.json_format import ParseDict

from kestrel_sovereign import server
from kestrel_sovereign.a2a.did_registry import ProcessA2ADidResolver
from kestrel_sovereign.a2a.native_card import verify_native_agent_card
from kestrel_sovereign.features.peers.directory import PeerRequester
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from tests.shared.genesis_audit import deterministic_safe_genesis_auditor


@pytest.mark.parametrize("push", ["false", "true"])
async def test_actual_single_lifespan_publishes_after_boot_and_withdraws_before_close(
    tmp_path, monkeypatch, push,
):
    from kestrel_sovereign.llm import service as service_module

    root = tmp_path / "agent"
    root.mkdir()
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("KESTREL_DB_PATH", str(root))
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KESTREL_HOST_DB_PATH", str(tmp_path / "host.db"))
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_HOLD_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_API_KEY", "isolated-native-lifespan")
    monkeypatch.setenv("KESTREL_PHOENIX_ENABLED", "0")
    monkeypatch.setenv("KESTREL_BASE_URL", "https://single.native.test")
    monkeypatch.setenv("KESTREL_A2A_DISCOVERY_AGENT", "single")
    monkeypatch.setenv("KESTREL_A2A_PUSH_ENABLED", push)
    monkeypatch.delenv("KESTREL_MULTI_AGENT", raising=False)
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
    monkeypatch.delenv("LIGHTHOUSE_API_KEY", raising=False)
    # This exercises boot and HTTP publication without any metered model call.
    # Real service, identity, genesis lifecycle, primary DB and TaskManager;
    # only the optional provider route configuration is empty.
    monkeypatch.setattr(service_module, "load_section", lambda _name: {})
    credentials = await create_kestrel_identity_async(
        str(root),
        is_test_instance=True,
        genesis_auditor=deterministic_safe_genesis_auditor,
        genesis_audit_provenance="test:native_single_lifespan",
    )

    class ExplicitDirectory:
        async def authorize_inbound_sender(self, requester, sender_id):
            return False

    original_constructor = server.KestrelAgent
    candidates = []

    def construct(**kwargs):
        # Supply the embedding's actual scoped dependencies at construction;
        # initialize(), READY and every publication/retirement body run intact.
        agent = original_constructor(
            **kwargs,
            allowed_features=set(),
            peer_directory_router=ExplicitDirectory(),
            peer_requester=PeerRequester(credentials.agent_did, object()),
        )
        agent.a2a_did_resolver = ProcessA2ADidResolver(()).resolve
        candidates.append(agent)
        return agent

    monkeypatch.setattr(server, "KestrelAgent", construct)
    app = FastAPI()
    async with server.lifespan(app):
        assert len(candidates) == 1 and app.state.agent is candidates[0]
        agent = candidates[0]
        assert agent._runtime_publication_ready.is_set()
        publication = getattr(agent, "_native_a2a_publication", None)
        assert publication is not None and publication.is_published, (
            "actual standalone startup did not publish its native contract"
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="https://single.native.test"
        ) as client:
            response = await client.get("/.well-known/agent-card.json")
        assert response.status_code == 200
        assert response.json()["capabilities"].get("pushNotifications", False) is (
            push == "true"
        )
        verify_native_agent_card(
            ParseDict(response.json(), AgentCard()),
            signing_did=agent.identity.signing_did,
            trusted_document={
                "id": agent.identity.signing_did,
                "verificationMethod": agent.identity.new_verification_methods,
            },
        )
        from kestrel_sovereign._async_ownership import await_owned_task
        from kestrel_sovereign.features.restart_coordinator.feature import (
            RestartCoordinatorFeature,
        )

        restart = RestartCoordinatorFeature(agent)
        assert restart._agent_appears_idle()["idle"], (
            "published native infrastructure must permit an idle restart"
        )
        # Runtime publication identifies residents by ownership, including
        # services whose name is absent from the legacy bookkeeping allowlist.
        resident = agent._track_runtime_task(
            asyncio.Event().wait(), name="opaque-runtime-service"
        )
        work = None
        try:
            await asyncio.sleep(0)
            assert restart._agent_appears_idle()["idle"], (
                "runtime service ownership was mistaken for finite agent work"
            )
            work = agent._track_background_task(
                asyncio.Event().wait(), name="opaque-runtime-service"
            )
            assert not restart._agent_appears_idle()["idle"], (
                "ordinary work acquired a resident's infrastructure exemption"
            )
        finally:
            for task in (work, resident):
                if task is not None:
                    task.cancel()
                    outcome = await await_owned_task(task)
                    assert isinstance(outcome.error, asyncio.CancelledError)
        backend = agent.task_manager.task_store.backend
    assert not publication.is_published
    assert backend._connection is None
