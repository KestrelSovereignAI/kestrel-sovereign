"""Scoped peer-router contracts for hosted multi-tenant runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, AsyncIterator, Mapping, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.peers.directory import (
    LocalHostPeerDirectory,
    PeerAccessDeniedError,
    PeerDirectoryConfigurationError,
    PeerIdentity,
    PeerNotFoundError,
    PeerRequester,
    PeerSubscriptionEvent,
)
from kestrel_sovereign.features.peers.feature import PeersFeature


@dataclass
class ScopedRouter:
    """In-memory hosted adapter with duplicate slugs in separate scopes."""

    scope_a: object
    scope_b: object
    invoked_on: list[str] = field(default_factory=list)
    sent_to: list[str] = field(default_factory=list)
    fetched_from: list[str] = field(default_factory=list)
    subscription_attempts: list[str] = field(default_factory=list)
    subscribed_to: list[str] = field(default_factory=list)
    deny_subscription: bool = False
    unexpected_resolution_failure: bool = False
    malformed_listing: bool = False
    malformed_invoke_result: bool = False

    def _directory(self, requester: PeerRequester) -> dict[str, PeerIdentity]:
        if requester.authorization_scope is self.scope_a:
            return {
                "companion": PeerIdentity(
                    agent_id="did:tenant-a:companion",
                    slug="companion",
                    routing_key="a-internal-42",
                    name="Companion",
                    status="online",
                ),
            }
        if requester.authorization_scope is self.scope_b:
            return {
                "companion": PeerIdentity(
                    agent_id="did:tenant-b:companion",
                    slug="companion",
                    routing_key="b-internal-99",
                    name="Companion",
                    status="online",
                ),
            }
        raise PeerAccessDeniedError("unknown authorization scope")

    def _authorize(self, requester: PeerRequester, peer: PeerIdentity) -> None:
        if peer not in self._directory(requester).values():
            raise PeerAccessDeniedError("peer is outside current scope")

    async def list_peers(self, requester: PeerRequester) -> Sequence[PeerIdentity]:
        if self.malformed_listing:
            return "not-a-peer-list"  # type: ignore[return-value]
        return list(self._directory(requester).values())

    async def resolve_peer(
        self, requester: PeerRequester, peer_name_or_slug: str,
    ) -> Optional[PeerIdentity]:
        if self.unexpected_resolution_failure:
            raise RuntimeError("tenant-b connection diagnostic")
        return self._directory(requester).get(peer_name_or_slug.casefold())

    async def invoke(
        self, requester: PeerRequester, peer: PeerIdentity, message: str,
    ) -> Mapping[str, Any]:
        self._authorize(requester, peer)
        self.invoked_on.append(peer.agent_id)
        if self.malformed_invoke_result:
            return ["not-an-object"]  # type: ignore[return-value]
        return {"response": f"{peer.agent_id}:{message}"}

    async def send_a2a_task(
        self,
        requester: PeerRequester,
        peer: PeerIdentity,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._authorize(requester, peer)
        self.sent_to.append(peer.agent_id)
        return {
            "id": payload["id"],
            "sessionId": payload["sessionId"],
            "status": {"state": "submitted"},
        }

    async def get_a2a_task(
        self, requester: PeerRequester, peer: PeerIdentity, task_id: str,
    ) -> Mapping[str, Any]:
        self._authorize(requester, peer)
        self.fetched_from.append(peer.agent_id)
        return {"id": task_id, "status": "completed", "message": "done"}

    async def subscribe_a2a_task(
        self,
        requester: PeerRequester,
        peer: PeerIdentity,
        task_id: str,
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[PeerSubscriptionEvent]:
        self._authorize(requester, peer)
        self.subscription_attempts.append(peer.agent_id)
        if self.deny_subscription:
            raise PeerAccessDeniedError("scope was revoked")
        self.subscribed_to.append(peer.agent_id)
        yield PeerSubscriptionEvent(
            event="status",
            data=(
                '{"id":"' + task_id + '","status":{"state":"completed",'
                '"message":{"parts":[{"text":"done"}]}}}'
            ),
        )


def _feature_for_scope(router: ScopedRouter, scope: object) -> PeersFeature:
    agent = SimpleNamespace(
        _agent_name="caller",
        did="did:tenant-a:caller",
        peer_directory_router=router,
        peer_requester=PeerRequester(
            identity="did:tenant-a:caller",
            authorization_scope=scope,
        ),
        identity=None,
        _provide_causation_chain=lambda: None,
        _get_current_turn_id=lambda: None,
        pending_a2a_questions=MagicMock(
            insert=AsyncMock(return_value=None),
            mark_resolved=AsyncMock(return_value=True),
        ),
        dispatcher=MagicMock(enqueue_signal=AsyncMock()),
        _track_background_task=lambda coro, *, name="": coro.close() or MagicMock(),
    )
    feature = PeersFeature(agent)
    return feature


@pytest.mark.asyncio
async def test_hosted_router_requires_host_injected_requester_identity_and_scope():
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b)
    feature = PeersFeature(SimpleNamespace(
        _agent_name="caller",
        did="did:tenant-a:caller",
        peer_directory_router=router,
        peer_requester=None,
    ))

    with pytest.raises(PeerDirectoryConfigurationError):
        await feature.initialize()


@pytest.mark.asyncio
async def test_duplicate_peer_names_resolve_to_the_callers_scoped_stable_identity():
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b)
    feature = _feature_for_scope(router, scope_a)
    await feature.initialize()

    peers = await feature.list_peers()
    assert peers.status is ToolResultStatus.OK
    assert peers.data["peers"] == [
        {"name": "Companion", "status": "online", "description": ""},
    ]

    invoked = await feature.ask_agent("companion", "tenant-a question")
    fetched = await feature.get_peer_task_result("companion", "result-a")

    result = await feature.send_a2a_task("companion", "tenant-a work")
    other_scope_feature = _feature_for_scope(router, scope_b)
    await other_scope_feature.initialize()
    other_scope_result = await other_scope_feature.send_a2a_task(
        "companion", "tenant-b work",
    )

    assert invoked.status is ToolResultStatus.OK
    assert fetched.status is ToolResultStatus.OK
    assert result.status is ToolResultStatus.OK
    assert other_scope_result.status is ToolResultStatus.OK
    assert router.invoked_on == ["did:tenant-a:companion"]
    assert router.fetched_from == ["did:tenant-a:companion"]
    assert router.sent_to == ["did:tenant-a:companion", "did:tenant-b:companion"]
    assert "did:tenant-b:companion" not in str(result.data)


@pytest.mark.asyncio
async def test_cross_scope_name_and_did_probes_are_indistinguishable_and_unrouted():
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b)
    feature = _feature_for_scope(router, scope_a)
    await feature.initialize()

    by_name = await feature.ask_agent("other-tenant-private-name", "probe")
    by_did = await feature.ask_agent("did:tenant-b:companion", "probe")
    result = await feature.get_peer_task_result("did:tenant-b:companion", "t-b")

    assert by_name.status is ToolResultStatus.ERROR
    assert by_did.status is ToolResultStatus.ERROR
    assert result.status is ToolResultStatus.ERROR
    assert by_name.error == by_did.error == "Peer is not available in the automatic directory"
    assert result.error == "Peer is not available in the automatic directory"
    assert router.sent_to == []
    assert router.fetched_from == []


@pytest.mark.asyncio
async def test_unexpected_router_resolution_error_does_not_disclose_provider_detail():
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b, unexpected_resolution_failure=True)
    feature = _feature_for_scope(router, scope_a)
    await feature.initialize()

    result = await feature.ask_agent("companion", "probe")

    assert result.status is ToolResultStatus.ERROR
    assert result.error == "Peer is not available in the automatic directory"
    assert "tenant-b connection diagnostic" not in (result.error or "")


@pytest.mark.asyncio
async def test_malformed_hosted_router_results_fail_without_crashing_or_leaking():
    scope_a, scope_b = object(), object()
    listing_router = ScopedRouter(scope_a, scope_b, malformed_listing=True)
    listing_feature = _feature_for_scope(listing_router, scope_a)
    await listing_feature.initialize()

    listed = await listing_feature.list_peers()

    invoke_router = ScopedRouter(scope_a, scope_b, malformed_invoke_result=True)
    invoke_feature = _feature_for_scope(invoke_router, scope_a)
    await invoke_feature.initialize()
    invoked = await invoke_feature.ask_agent("companion", "probe")

    assert listed.status is ToolResultStatus.ERROR
    assert listed.error == "Could not list peers"
    assert invoked.status is ToolResultStatus.ERROR
    assert invoked.error == "Could not message agent 'companion'"


@pytest.mark.asyncio
async def test_scope_and_recipient_substitution_cannot_retarget_a2a_send():
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b)
    feature = _feature_for_scope(router, scope_a)
    await feature.initialize()

    # The tool has no scope/user-id argument to replace.  Even though another
    # tenant owns the same slug, the router receives the injected scope and the
    # stable identity it resolved in that scope.
    result = await feature.send_a2a_task("companion", "do not retarget")

    assert result.status is ToolResultStatus.OK
    assert router.sent_to == ["did:tenant-a:companion"]
    assert feature.agent.peer_requester.authorization_scope is scope_a


@pytest.mark.asyncio
async def test_subscription_reauthorizes_after_resolution_and_hides_revocation_details():
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b, deny_subscription=True)
    feature = _feature_for_scope(router, scope_a)
    await feature.initialize()

    await feature._supervise_a2a_question(
        task_id="task-1",
        recipient="companion",
        original_question="status?",
        sess_id="session-1",
        deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
        causation_chain=None,
    )

    assert router.subscription_attempts == ["did:tenant-a:companion"]
    assert router.subscribed_to == []
    feature.agent.pending_a2a_questions.mark_resolved.assert_awaited_once_with("task-1")
    signal = feature.agent.dispatcher.enqueue_signal.await_args.args[0]
    assert signal.payload["state"] == "failed"
    assert signal.payload["reply_text"] == "Peer task subscription is no longer authorized."


@pytest.mark.asyncio
async def test_local_host_reauthorizes_before_routing_and_rejects_forged_route_key():
    """The default adapter must meet the same stale-identity contract.

    A direct adapter caller is not normally a tool caller, but accepting its
    stale or forged route key would make the default implementation weaker
    than the documented provider protocol.
    """
    directory_response = MagicMock(status_code=200)
    directory_response.raise_for_status.return_value = None
    directory_response.json.return_value = [{
        "id": "did:local:companion",
        "name": "companion",
        "routing_name": "actual-route",
    }]
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.get.return_value = directory_response

    adapter = LocalHostPeerDirectory(
        "http://host",
        client_factory=lambda *args, **kwargs: client,
    )
    requester = PeerRequester("did:local:caller", object())
    forged_peer = PeerIdentity(
        agent_id="did:local:companion",
        slug="companion",
        routing_key="forged-route",
    )

    with pytest.raises(PeerNotFoundError):
        await adapter.invoke(requester, forged_peer, "should not route")

    client.post.assert_not_awaited()
