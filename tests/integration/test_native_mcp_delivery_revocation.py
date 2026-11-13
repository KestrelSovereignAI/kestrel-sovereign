"""Revocation acknowledges only after original private delivery retires."""

import asyncio

import pytest

from kestrel_sovereign.mcp_server.tool_caller import _REQUEST_KEY
from tests.integration.test_native_mcp_memory_tools import observe_native_call
from tests.integration.test_native_mcp_publication import (
    assistant_agent as _assistant_agent,
)
from tests.integration.test_native_mcp_publication import publication as _publication
from tests.integration.test_native_mcp_tool_context import issue_native_tokens

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
assistant_agent = _assistant_agent
publication = _publication


@pytest.mark.parametrize("assistant_agent", [{"features": ["MemoryFeature"]}], indirect=True)
async def test_native_revocation_cannot_acknowledge_before_blocked_private_delivery(publication, monkeypatch):
    owner, http = publication
    private = "owned-native-memory-blocked-during-actual-revocation"
    entered, release, revoking = asyncio.Event(), asyncio.Event(), asyncio.Event()
    order = []
    await owner.agent.privacy_agent.add_conversation("user", private, session_id="owned-revocation")

    class OriginalBlockedBody:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            async def original_send(message):
                if message["type"] == "http.response.body" and private.encode() in message.get("body", b""):
                    entered.set()
                    await release.wait()
                    await send(message)
                    order.append("original-private-body-complete")
                else:
                    await send(message)
            await self.app(scope, receive, original_send)

    original_load = owner.runtime.authorization.grants.load_for_revocation

    async def original_revocation_load(*args, **kwargs):
        revoking.set()
        return await original_load(*args, **kwargs)

    monkeypatch.setattr(owner.runtime.authorization.grants, "load_for_revocation", original_revocation_load)
    owner.app.add_middleware(OriginalBlockedBody)
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    original = asyncio.create_task(observe_native_call(owner, token, "recall_recent",
        {"limit": 5, "operation_id": "owned-physical-revocation", "conversation_id": "owned-revocation"},
        discover_first=True))
    entered_observer = asyncio.create_task(entered.wait())
    revocation = revoke_observer = None

    async def revoke():
        response = await http.post(owner.resource + "/oauth/revoke", data={"token": token,
            "client_id": "owned-provider", "client_secret": "owned-secret"})
        order.append("original-revocation-response")
        return response

    try:
        await asyncio.wait({original, entered_observer}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        reached_body = entered.is_set()
        if reached_body:
            revocation = asyncio.create_task(revoke())
            revoke_observer = asyncio.create_task(revoking.wait())
            await asyncio.wait({revocation, revoke_observer}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
            started_revoke = revoking.is_set()
            await asyncio.wait({revocation}, timeout=5)
            protected = not revocation.done() and not order
        else:
            started_revoke = protected = False
    finally:
        release.set()
        for observer in (entered_observer, revoke_observer):
            if observer is not None:
                observer.cancel()
        result, failure, responses = await original
        response = await revocation if revocation is not None else None
        await asyncio.gather(*[observer for observer in (entered_observer, revoke_observer)
            if observer is not None], return_exceptions=True)
    assert reached_body and started_revoke, "Actual native private delivery/revocation interleaving was not reached"
    assert protected, "Native revocation acknowledged while original private bytes remained deliverable"
    assert order == ["original-private-body-complete", "original-revocation-response"], (
        "Native private bytes were delivered after the original revocation acknowledgement")
    assert result is not None and not result.is_error and failure is None, (failure, responses)
    assert response is not None and response.status_code == 200
    assert await owner.runtime.authorization.verify_token(token) is None
    assert any(private.encode() in body for _status, body in responses)


@pytest.mark.parametrize("assistant_agent", [{"features": ["MemoryFeature"]}], indirect=True)
@pytest.mark.parametrize("publication", [{"operation_seconds": 0.1}], indirect=True)
async def test_native_delivery_checks_deadline_between_original_http_messages(publication):
    owner, http = publication
    private = "owned-native-private-result-after-response-start-gap"
    delayed = []
    await owner.agent.privacy_agent.add_conversation("user", private, session_id="owned-send-gap")

    class DelayAfterOriginalStart:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            async def original_send(message):
                await send(message)
                request = scope.get(_REQUEST_KEY)
                if (message["type"] == "http.response.start" and request is not None
                        and request._privacy_epoch is not None):
                    delayed.append(request)
                    await asyncio.sleep(owner.policy.operation_seconds * 2)
            await self.app(scope, receive, original_send)

    owner.runtime._transport.add_middleware(DelayAfterOriginalStart)
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    result, failure, responses = await observe_native_call(owner, token, "recall_recent",
        {"limit": 5, "operation_id": "owned-native-send-gap", "conversation_id": "owned-send-gap"},
        discover_first=True)
    assert len(delayed) == 1, "Original private response did not traverse the controlled inter-message delay"
    assert all(private.encode() not in body for _status, body in responses), (
        "Native private bytes crossed their deadline between original HTTP messages")
    assert result is None and failure is not None
