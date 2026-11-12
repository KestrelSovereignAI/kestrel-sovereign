"""The finite external model must distinguish task completion from a new effect request."""

import httpx
import pytest
from starlette.applications import Starlette

from tests.integration.mcp_permission_peer import add_owned_provider_routes
from kestrel_sovereign.agent.operator_signals import _fallback_user_notice


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_submission", [False, True])
@pytest.mark.parametrize("notification", [
    "[A2A_COMPLETE] An A2A task has finished. Acknowledge and move on.",
    "Background task completed. No new operation was requested.",
])
async def test_completion_wake_does_not_script_another_marker_effect(tmp_path, prior_submission, notification):
    app = Starlette()
    add_owned_provider_routes(app, "owned-model", tmp_path / "requests.jsonl")
    messages = []
    if prior_submission:
        messages.append({"role": "user", "content":
            "[A2A_TASK_SUBMITTED] Peer `did:peer` submitted native A2A task `original-task`"})
    messages.append({"role": "user", "content": notification})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="https://owned.example") as client:
        response = await client.post("/v1/chat/completions", json={"model": "owned-model",
            "messages": messages, "tools": [{"type": "function", "function": {"name": name}}
                for name in ["mcp__owned__record_marker", "respond_to_a2a_task"]]})
    assert response.status_code == 200, response.text
    choice, = response.json()["choices"]
    assert choice["finish_reason"] == "stop" and "tool_calls" not in choice["message"], (
        "The finite provider scripted a new external effect for an A2A completion notification"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("request_content", [
    "Record the owned marker using the connected MCP tool.",
    "[A2A_TASK_SUBMITTED] Peer `did:peer` submitted native A2A task `original-task`",
])
async def test_original_effect_request_survives_a_native_trailing_operator_notice(tmp_path, request_content):
    app = Starlette()
    add_owned_provider_routes(app, "owned-model", tmp_path / "requests.jsonl")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="https://owned.example") as client:
        response = await client.post("/v1/chat/completions", json={"model": "owned-model",
            "messages": [{"role": "user", "content": request_content},
                {"role": "user", "content": _fallback_user_notice("Owned governance state")}],
            "tools": [{"type": "function", "function": {"name": name}}
                for name in ["mcp__owned__record_marker", "respond_to_a2a_task"]]})
    assert response.status_code == 200, response.text
    choice, = response.json()["choices"]
    assert choice["finish_reason"] == "tool_calls", (
        "The finite provider mistook the native operator notice for the original effect request"
    )
    call, = choice["message"]["tool_calls"]
    assert call["function"]["name"] == "mcp__owned__record_marker"
