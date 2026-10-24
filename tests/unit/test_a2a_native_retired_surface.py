"""The draft transport cannot remain a second peer protocol after cutover."""

import importlib.util

import httpx
import pytest
from fastapi import FastAPI

from kestrel_sovereign.a2a import transport_auth
from kestrel_sovereign.endpoints.agent import router
from kestrel_sovereign.features.peers.directory import (
    LocalHostPeerDirectory,
    PeerDirectoryRouter,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/tasks/send"),
        ("POST", "/peer/stop"),
        ("POST", "/tasks/opaque-task/read"),
        ("POST", "/tasks/opaque-task/cancel"),
        ("POST", "/tasks/opaque-task/subscribe"),
        ("GET", "/tasks/opaque-task/subscribe"),
    ],
)
async def test_draft_http_routes_are_absent(method, path):
    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://retired.test"
    ) as client:
        response = await client.request(method, "/api/agent" + path)
    assert response.status_code == 404


@pytest.mark.parametrize(
    "module",
    ["types", "agent_card", "task_worker", "stores.task_store", "envelope_signing", "outbound_store", "local_submission"],
)
def test_retired_draft_modules_are_absent(module):
    assert importlib.util.find_spec("kestrel_sovereign.a2a." + module) is None


@pytest.mark.parametrize(
    "method",
    ["invoke", "send_a2a_task", "get_a2a_task", "cancel_a2a_task",
     "subscribe_a2a_task", "stop_peer"],
)
def test_directory_exposes_the_native_session_contract_only(method):
    assert not hasattr(PeerDirectoryRouter, method)
    assert not hasattr(LocalHostPeerDirectory, method)


@pytest.mark.parametrize(
    "method",
    ["authorize_a2a_legacy_unsigned_sender", "_authorize_host_attested_local_a2a_route",
     "get_host_attested_local_a2a_task", "subscribe_host_attested_local_a2a_task",
     "cancel_host_attested_local_a2a_task", "stop_host_attested_local_peer"],
)
def test_host_cannot_bypass_native_peer_proof_with_a_draft_callback(method):
    assert not hasattr(AgentManager, method)


def test_roster_credential_does_not_admit_task_or_invoke_paths():
    assert transport_auth.is_peer_directory_request("GET", "/api/agents")
    for prefix in ("", "/api/agents/recipient"):
        for suffix in ("invoke", "peer/stop", "tasks/send", "tasks/id/read",
                       "tasks/id/cancel", "tasks/id/subscribe"):
            for method in ("GET", "POST"):
                assert not transport_auth.is_peer_directory_request(
                    method, prefix + "/api/agent/" + suffix
                )


def test_retired_draft_question_store_is_absent():
    assert importlib.util.find_spec("kestrel_sovereign.storage.async_pending_a2a_question_store") is None
