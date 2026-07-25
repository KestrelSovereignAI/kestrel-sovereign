"""Scoped peer-router contracts for hosted multi-tenant runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, AsyncIterator, Mapping, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.a2a import outbound_store
from kestrel_sovereign.features.peers.directory import (
    LocalHostPeerDirectory,
    PeerAccessDeniedError,
    PeerDirectoryConfigurationError,
    PeerIdentity,
    PeerNotFoundError,
    PeerRequester,
    PeerSubscriptionEvent,
    PeerTransportError,
)
from kestrel_sovereign.features.peers.feature import PeersFeature
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend


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
    resolve_calls: list[str] = field(default_factory=list)
    resolve_by_agent_id_calls: list[str] = field(default_factory=list)
    inbound_authorization_calls: list[str] = field(default_factory=list)
    scope_a_entries: Optional[dict[str, PeerIdentity]] = None
    deny_subscription: bool = False
    unexpected_resolution_failure: bool = False
    malformed_listing: bool = False
    malformed_invoke_result: bool = False
    recipient_task_id: Optional[str] = None

    def _directory(self, requester: PeerRequester) -> dict[str, PeerIdentity]:
        if requester.authorization_scope is self.scope_a:
            if self.scope_a_entries is not None:
                return self.scope_a_entries
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
        self.resolve_calls.append(peer_name_or_slug)
        if self.unexpected_resolution_failure:
            raise RuntimeError("tenant-b connection diagnostic")
        return self._directory(requester).get(peer_name_or_slug.casefold())

    async def resolve_peer_by_agent_id(
        self, requester: PeerRequester, agent_id: str,
    ) -> Optional[PeerIdentity]:
        self.resolve_by_agent_id_calls.append(agent_id)
        matches = [
            peer for peer in self._directory(requester).values()
            if peer.agent_id == agent_id
        ]
        return matches[0] if len(matches) == 1 else None

    async def authorize_inbound_sender(
        self,
        requester: PeerRequester,
        sender_id: str,
    ) -> bool:
        self.inbound_authorization_calls.append(sender_id)
        return any(
            peer.agent_id == sender_id
            for peer in self._directory(requester).values()
        )

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
            "id": self.recipient_task_id or payload["id"],
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


async def _durable_feature_for_scope(
    router: ScopedRouter,
    scope: object,
    tmp_path,
    *,
    database_name: str,
) -> tuple[PeersFeature, SQLiteBackend]:
    """Build a hosted feature with the durable outbound-route store it needs."""
    feature = _feature_for_scope(router, scope)
    backend = SQLiteBackend(str(tmp_path / database_name))
    await backend.connect()
    feature.agent._raw_storage = SimpleNamespace(db=AsyncDatabase(backend))
    feature.agent.storage = None
    await feature.initialize()
    return feature, backend


@pytest.mark.asyncio
@pytest.mark.parametrize("inject_router", [True, False])
async def test_hosted_router_and_requester_must_be_injected_as_a_pair(inject_router):
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b)
    feature = PeersFeature(SimpleNamespace(
        _agent_name="caller",
        did="did:tenant-a:caller",
        peer_directory_router=router if inject_router else None,
        peer_requester=None if inject_router else PeerRequester(
            identity="did:tenant-a:caller", authorization_scope=scope_a,
        ),
    ))

    with pytest.raises(PeerDirectoryConfigurationError):
        await feature.initialize()
    with pytest.raises(PeerDirectoryConfigurationError):
        feature._peer_directory_context()


@pytest.mark.asyncio
async def test_duplicate_peer_names_resolve_to_the_callers_scoped_stable_identity(tmp_path):
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b)
    feature, backend_a = await _durable_feature_for_scope(
        router, scope_a, tmp_path, database_name="tenant-a.db",
    )
    other_scope_feature, backend_b = await _durable_feature_for_scope(
        router, scope_b, tmp_path, database_name="tenant-b.db",
    )
    try:
        peers = await feature.list_peers()
        assert peers.status is ToolResultStatus.OK
        assert peers.data["peers"] == [
            {
                "name": "Companion",
                "slug": "companion",
                "status": "online",
                "description": "",
            },
        ]

        invoked = await feature.ask_agent("companion", "tenant-a question")
        result = await feature.send_a2a_task("companion", "tenant-a work")
        fetched = await feature.get_peer_task_result(
            "companion", result.data["task_id"],
        )
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
    finally:
        await backend_a.close()
        await backend_b.close()


@pytest.mark.asyncio
async def test_same_name_peer_routes_by_stable_identity_while_self_is_rejected(
    tmp_path,
):
    """A scoped directory, not a display name, defines the self boundary."""
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b)
    same_name_peer = PeerIdentity(
        agent_id="did:tenant-a:other-caller",
        slug="caller",
        routing_key="other-caller-route",
        name="Caller",
    )
    router.scope_a_entries = {"caller": same_name_peer}
    feature, backend = await _durable_feature_for_scope(
        router, scope_a, tmp_path, database_name="same-name-peer.db",
    )
    try:
        invoked = await feature.ask_agent("CALLER", "hello")
        dispatched = await feature.send_a2a_task("caller", "do work")

        assert invoked.status is ToolResultStatus.OK
        assert dispatched.status is ToolResultStatus.OK
        assert router.invoked_on == [same_name_peer.agent_id]
        assert router.sent_to == [same_name_peer.agent_id]

        # The identical display name becomes self only when the directory's
        # stable identity equals the trusted requester identity.
        router.scope_a_entries = {
            "caller": PeerIdentity(
                agent_id=feature.agent.did,
                slug="caller",
                routing_key="caller-route",
                name="Caller",
            ),
        }
        self_message = await feature.ask_agent("caller", "hello")
        self_task = await feature.send_a2a_task("caller", "do work")

        assert self_message.status is ToolResultStatus.ERROR
        assert self_message.error == "Cannot send a message to yourself"
        assert self_task.status is ToolResultStatus.ERROR
        assert self_task.error == "Cannot send an A2A task to yourself"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_peer_listing_preserves_slug_and_uses_it_when_name_is_empty():
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b)
    router.scope_a_entries = {
        "slug-only": PeerIdentity(
            agent_id="did:tenant-a:slug-only",
            slug="slug-only",
            routing_key="a-slug-only",
            name="",
            status="online",
        ),
        "duplicate-one": PeerIdentity(
            agent_id="did:tenant-a:duplicate-one",
            slug="duplicate-one",
            routing_key="a-duplicate-one",
            name="Duplicate",
            status="online",
        ),
        "duplicate-two": PeerIdentity(
            agent_id="did:tenant-a:duplicate-two",
            slug="duplicate-two",
            routing_key="a-duplicate-two",
            name="Duplicate",
            status="online",
        ),
    }
    feature = _feature_for_scope(router, scope_a)
    await feature.initialize()

    result = await feature.list_peers()

    assert result.status is ToolResultStatus.OK
    assert result.data["peers"] == [
        {
            "name": "slug-only",
            "slug": "slug-only",
            "status": "online",
            "description": "",
        },
        {
            "name": "Duplicate",
            "slug": "duplicate-one",
            "status": "online",
            "description": "",
        },
        {
            "name": "Duplicate",
            "slug": "duplicate-two",
            "status": "online",
            "description": "",
        },
    ]


@pytest.mark.asyncio
async def test_cross_scope_name_and_did_probes_are_indistinguishable_and_unrouted():
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b)
    feature = _feature_for_scope(router, scope_a)
    await feature.initialize()

    by_name = await feature.ask_agent("other-tenant-private-name", "probe")
    resolves_before_did_probes = list(router.resolve_calls)
    by_did = await feature.ask_agent("did:tenant-b:companion", "probe")
    result = await feature.get_peer_task_result("did:tenant-b:companion", "t-b")

    assert by_name.status is ToolResultStatus.ERROR
    assert by_did.status is ToolResultStatus.ERROR
    assert result.status is ToolResultStatus.ERROR
    assert by_name.error == by_did.error == "Peer is not available in the automatic directory"
    assert result.error == "Peer is not available in the automatic directory"
    assert router.resolve_calls == resolves_before_did_probes
    assert router.sent_to == []
    assert router.fetched_from == []


@pytest.mark.asyncio
async def test_local_host_inbound_authorization_uses_live_stable_identity():
    adapter = LocalHostPeerDirectory("http://local-host")
    requester = PeerRequester("did:tenant-a:requester", object())
    adapter._directory_entries = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            PeerIdentity(
                agent_id="did:tenant-a:companion",
                slug="companion",
                routing_key="companion-route",
            )
        ]
    )

    assert (
        await adapter.authorize_inbound_sender(
            requester,
            "did:tenant-a:companion",
        )
        is True
    )
    assert (
        await adapter.authorize_inbound_sender(
            requester,
            "did:tenant-b:companion",
        )
        is False
    )
    assert adapter._directory_entries.await_count == 2


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
async def test_local_host_resolution_transport_failure_is_reported_as_unreachable():
    """Resolution outages stay distinct from authorization/name denials."""
    scope_a, scope_b = object(), object()
    feature = _feature_for_scope(ScopedRouter(scope_a, scope_b), scope_a)
    await feature.initialize()
    adapter = LocalHostPeerDirectory("http://local-host")
    adapter.resolve_peer = AsyncMock(
        side_effect=PeerTransportError("local host connection refused"),
    )
    feature._peer_router = adapter
    feature._peer_requester = PeerRequester(
        identity=feature.agent.did,
        authorization_scope=scope_a,
    )

    invoked = await feature.ask_agent("companion", "hello")
    dispatched = await feature.send_a2a_task("companion", "do work")

    expected = "Could not reach agent 'companion' — multi_agent host unreachable"
    assert invoked.status is ToolResultStatus.ERROR
    assert invoked.error == expected
    assert dispatched.status is ToolResultStatus.ERROR
    assert dispatched.error == expected
    assert adapter.resolve_peer.await_count == 2


@pytest.mark.asyncio
async def test_retained_task_result_resolution_transport_failure_is_reported_as_unreachable():
    """A stable-ID resolution outage must not be reported as a missing peer."""
    scope_a, scope_b = object(), object()
    feature = _feature_for_scope(ScopedRouter(scope_a, scope_b), scope_a)
    await feature.initialize()
    adapter = LocalHostPeerDirectory("http://local-host")
    adapter.resolve_peer_by_agent_id = AsyncMock(
        side_effect=PeerTransportError("local host connection refused"),
    )
    adapter.get_a2a_task = AsyncMock()
    feature._peer_router = adapter
    feature._peer_requester = PeerRequester(
        identity=feature.agent.did,
        authorization_scope=scope_a,
    )
    feature._outbound_recipient_agent_id = AsyncMock(
        return_value="did:tenant-a:companion",
    )

    result = await feature.get_peer_task_result("companion", "task-123")

    assert result.status is ToolResultStatus.ERROR
    assert result.error == "Could not reach peer 'companion' for task task-123"
    adapter.resolve_peer_by_agent_id.assert_awaited_once_with(
        feature._peer_requester, "did:tenant-a:companion",
    )
    adapter.get_a2a_task.assert_not_awaited()


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
async def test_scope_and_recipient_substitution_cannot_retarget_a2a_send(tmp_path):
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b)
    feature, backend = await _durable_feature_for_scope(
        router, scope_a, tmp_path, database_name="scope-substitution.db",
    )
    try:
        # The tool has no scope/user-id argument to replace.  Even though another
        # tenant owns the same slug, the router receives the injected scope and the
        # stable identity it resolved in that scope.
        result = await feature.send_a2a_task("companion", "do not retarget")

        assert result.status is ToolResultStatus.OK
        assert router.sent_to == ["did:tenant-a:companion"]
        assert feature.agent.peer_requester.authorization_scope is scope_a
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_hosted_identity_write_failure_cannot_route_replacement_after_restart(
    tmp_path,
):
    """A rejected hosted send leaves no name-based retained-route escape hatch."""
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b)
    feature, backend = await _durable_feature_for_scope(
        router, scope_a, tmp_path, database_name="identity-write-failure.db",
    )

    class FailingOutboundWriteDatabase:
        async def execute(self, *_args, **_kwargs):
            raise OSError("durable identity write failed")

    try:
        # Failure injection is deliberately before the router delivery call.
        # The previous implementation sent first, swallowed this error, then
        # allowed a later retained lookup to resolve the mutable name.
        feature._db = FailingOutboundWriteDatabase()
        rejected = await feature.send_a2a_task("companion", "sensitive work")

        assert rejected.status is ToolResultStatus.ERROR
        assert rejected.data["sent"] is False
        assert rejected.data["error_type"] == "peer_identity_persistence_failed"
        assert router.sent_to == []

        # Simulate a restart after the name has been reassigned.  The real
        # backing database is empty because the reserve write failed.
        router.scope_a_entries = {
            "companion": PeerIdentity(
                agent_id="did:tenant-a:replacement",
                slug="companion",
                routing_key="a-replacement",
                name="Companion",
            ),
        }
        restarted = _feature_for_scope(router, scope_a)
        restarted.agent._raw_storage = feature.agent._raw_storage
        restarted.agent.storage = None
        await restarted.initialize()

        fetched = await restarted.get_peer_task_result(
            "companion", rejected.data["task_id"],
        )
        await restarted._supervise_a2a_question(
            task_id=rejected.data["task_id"],
            recipient="companion",
            recipient_agent_id=None,
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        assert fetched.status is ToolResultStatus.ERROR
        assert fetched.error == "Peer is not available in the automatic directory"
        assert router.fetched_from == []
        assert router.subscription_attempts == []
        assert router.subscribed_to == []
        # Only the original send performed directory resolution.  The restart
        # did not ask the provider to resolve the replacement display name.
        assert router.resolve_calls == ["companion"]
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_hosted_route_store_read_failure_cannot_route_replacement(
    tmp_path,
    monkeypatch,
):
    """A ready hosted store must fail closed when a retained lookup raises."""
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b)
    feature, backend = await _durable_feature_for_scope(
        router, scope_a, tmp_path, database_name="hosted-route-read-failure.db",
    )
    try:
        submitted = await feature.send_a2a_task("companion", "sensitive work")
        assert submitted.status is ToolResultStatus.OK
        assert feature._outbound_route_store_ready is True

        router.scope_a_entries = {
            "companion": PeerIdentity(
                agent_id="did:tenant-a:replacement",
                slug="companion",
                routing_key="a-replacement",
                name="Companion",
            ),
        }
        monkeypatch.setattr(
            outbound_store,
            "get_outbound_task",
            AsyncMock(side_effect=OSError("durable route lookup failed")),
        )

        fetched = await feature.get_peer_task_result(
            "companion", submitted.data["task_id"],
        )
        await feature._supervise_a2a_question(
            task_id=submitted.data["task_id"],
            recipient="companion",
            recipient_agent_id=None,
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        assert fetched.status is ToolResultStatus.ERROR
        assert router.fetched_from == []
        assert router.subscription_attempts == []
        assert router.subscribed_to == []
        # The send is the only operation allowed to resolve the mutable name.
        assert router.resolve_calls == ["companion"]
        assert router.resolve_by_agent_id_calls == []
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_hosted_rekey_failure_fails_closed_with_the_peer_task_id(
    tmp_path, monkeypatch,
):
    """A failed hosted rekey must not publish an unbound peer task id."""
    scope_a, scope_b = object(), object()
    peer_task_id = "peer-assigned-task-id"
    router = ScopedRouter(
        scope_a,
        scope_b,
        recipient_task_id=peer_task_id,
    )
    feature, backend = await _durable_feature_for_scope(
        router, scope_a, tmp_path, database_name="hosted-rekey-failure.db",
    )
    monkeypatch.setattr(
        outbound_store,
        "rekey_outbound_task",
        AsyncMock(return_value=0),
    )

    try:
        rejected = await feature.send_a2a_task("companion", "sensitive work")

        assert rejected.status is ToolResultStatus.ERROR
        assert rejected.data["sent"] is True
        assert rejected.data["task_id"] != peer_task_id
        assert "sender_task_id" not in rejected.data
        assert rejected.data["error_type"] == "peer_identity_persistence_failed"
        assert router.sent_to == ["did:tenant-a:companion"]

        # The retained provisional record is a failed audit entry, not a
        # routable fallback to the mutable display name.
        fetched = await feature.get_peer_task_result(
            "companion", rejected.data["task_id"],
        )
        assert fetched.status is ToolResultStatus.ERROR
        assert router.fetched_from == []
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "marker_behavior",
    [
        pytest.param(0, id="zero-row-marker"),
        pytest.param(OSError("transient marker failure"), id="marker-exception"),
    ],
)
async def test_hosted_rekey_rejection_stays_unroutable_when_failure_marker_does_not_land(
    tmp_path,
    monkeypatch,
    marker_behavior,
):
    """A failed lifecycle marker cannot turn a reservation into a route.

    This is intentionally a restart test.  The original sender returns a
    delivery warning after a peer id collision; the restarted process must
    still reject both result fetch and subscription without consulting the
    replacement peer's mutable display name.
    """
    scope_a, scope_b = object(), object()
    router = ScopedRouter(
        scope_a,
        scope_b,
        recipient_task_id="peer-assigned-task-id",
    )
    feature, backend = await _durable_feature_for_scope(
        router, scope_a, tmp_path, database_name="marker-failure.db",
    )
    if isinstance(marker_behavior, BaseException):
        marker = AsyncMock(side_effect=marker_behavior)
    else:
        marker = AsyncMock(return_value=marker_behavior)
    monkeypatch.setattr(
        outbound_store,
        "rekey_outbound_task",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(outbound_store, "update_outbound_terminal_state", marker)

    try:
        rejected = await feature.send_a2a_task("companion", "sensitive work")
        assert rejected.status is ToolResultStatus.ERROR
        assert rejected.data["task_id"] != "peer-assigned-task-id"
        marker.assert_awaited_once()

        router.scope_a_entries = {
            "companion": PeerIdentity(
                agent_id="did:tenant-a:replacement",
                slug="companion",
                routing_key="a-replacement",
                name="Companion",
            ),
        }
        restarted = _feature_for_scope(router, scope_a)
        restarted.agent._raw_storage = feature.agent._raw_storage
        restarted.agent.storage = None
        await restarted.initialize()

        fetched = await restarted.get_peer_task_result(
            "companion", rejected.data["task_id"],
        )
        await restarted._supervise_a2a_question(
            task_id=rejected.data["task_id"],
            recipient="companion",
            recipient_agent_id="did:tenant-a:companion",
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        assert fetched.status is ToolResultStatus.ERROR
        assert router.fetched_from == []
        assert router.subscription_attempts == []
        assert router.subscribed_to == []
        assert router.resolve_calls == ["companion"]
        assert router.resolve_by_agent_id_calls == []
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_hosted_remote_task_id_collision_preserves_owner_across_retry_and_restart(
    tmp_path,
):
    """A peer-returned id cannot steal another task's retained route.

    The first peer owns ``shared-task-id``. A second peer returning that same
    id must fail closed: retrying or restarting cannot make fetches or
    subscriptions reach the second peer, while the first task remains routed
    through its original stable identity.
    """
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b, recipient_task_id="shared-task-id")
    first = PeerIdentity(
        agent_id="did:tenant-a:first",
        slug="first",
        routing_key="a-first",
        name="First",
    )
    second = PeerIdentity(
        agent_id="did:tenant-a:second",
        slug="second",
        routing_key="a-second",
        name="Second",
    )
    router.scope_a_entries = {"first": first, "second": second}
    feature, backend = await _durable_feature_for_scope(
        router, scope_a, tmp_path, database_name="remote-task-id-collision.db",
    )
    try:
        owner = await feature.send_a2a_task("first", "first task")
        collided = await feature.send_a2a_task("second", "second task")
        replayed_collision = await feature.send_a2a_task(
            "second", "second task retry",
        )

        assert owner.status is ToolResultStatus.OK
        assert owner.data["task_id"] == "shared-task-id"
        assert collided.status is ToolResultStatus.ERROR
        assert replayed_collision.status is ToolResultStatus.ERROR
        assert collided.data["task_id"] != "shared-task-id"
        assert replayed_collision.data["task_id"] != "shared-task-id"
        assert router.sent_to == [first.agent_id, second.agent_id, second.agent_id]

        # The original task id continues to belong only to the first peer.
        fetched_owner = await feature.get_peer_task_result(
            "first", "shared-task-id",
        )
        assert fetched_owner.status is ToolResultStatus.OK

        # Neither collision's provisional audit id is a retained route. This
        # prevents a result lookup from reaching the second peer with a task it
        # cannot safely identify.
        for rejected in (collided, replayed_collision):
            fetched_rejected = await feature.get_peer_task_result(
                "second", rejected.data["task_id"],
            )
            assert fetched_rejected.status is ToolResultStatus.ERROR

        # A recreated feature sees the same retained owner binding. The
        # surviving result fetch and replay subscription both stay on first.
        restarted = _feature_for_scope(router, scope_a)
        restarted.agent._raw_storage = feature.agent._raw_storage
        restarted.agent.storage = None
        await restarted.initialize()
        fetched_after_restart = await restarted.get_peer_task_result(
            "first", "shared-task-id",
        )
        assert fetched_after_restart.status is ToolResultStatus.OK
        await restarted._supervise_a2a_question(
            task_id="shared-task-id",
            recipient="first",
            recipient_agent_id=first.agent_id,
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        assert router.fetched_from == [first.agent_id, first.agent_id]
        assert router.subscribed_to == [first.agent_id]
        assert second.agent_id not in router.fetched_from
        assert second.agent_id not in router.subscribed_to
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_local_colliding_remote_task_id_is_delivered_but_untrackable_after_restart(
    tmp_path,
):
    """A local peer cannot redirect retained routes with a colliding task id.

    The first peer owns the returned remote id.  A second local peer returning
    that same id has received its work, but the sender must not expose the id,
    route a fetch/subscription to either peer for the failed dispatch, or
    resurrect that path after restart.  The first task remains routable through
    its original stable identity.
    """
    first = PeerIdentity(
        agent_id="did:local:first",
        slug="first",
        routing_key="first",
        name="First",
    )
    second = PeerIdentity(
        agent_id="did:local:second",
        slug="second",
        routing_key="second",
        name="Second",
    )
    peers = {"first": first, "second": second}
    fetched_from: list[str] = []
    subscribed_to: list[str] = []
    adapter = LocalHostPeerDirectory("http://local-host")
    adapter.resolve_peer = AsyncMock(
        side_effect=lambda _requester, name: peers.get(name.casefold()),
    )
    adapter.resolve_peer_by_agent_id = AsyncMock(
        side_effect=lambda _requester, agent_id: next(
            (peer for peer in peers.values() if peer.agent_id == agent_id),
            None,
        ),
    )
    adapter.send_a2a_task = AsyncMock(side_effect=lambda _requester, _peer, payload: {
        # Both recipients (including a duplicate response/retry from second)
        # return the same remote id.
        "id": "shared-peer-task-id",
        "sessionId": payload["sessionId"],
        "status": {"state": "submitted"},
    })

    async def _get_task(_requester, peer, task_id):
        fetched_from.append(peer.agent_id)
        return {"id": task_id, "status": "completed", "message": "done"}

    async def _subscribe_task(
        _requester, peer, task_id, *, timeout_seconds,
    ):
        del timeout_seconds
        subscribed_to.append(peer.agent_id)
        yield PeerSubscriptionEvent(
            event="status",
            data=(
                '{"id":"' + task_id + '","status":{"state":"completed",'
                '"message":{"parts":[{"text":"done"}]}}}'
            ),
        )

    adapter.get_a2a_task = AsyncMock(side_effect=_get_task)
    adapter.subscribe_a2a_task = _subscribe_task
    requester = PeerRequester("did:local:caller", object())

    def _agent() -> SimpleNamespace:
        return SimpleNamespace(
            _agent_name="caller",
            did="did:local:caller",
            peer_directory_router=adapter,
            peer_requester=requester,
            identity=None,
            _provide_causation_chain=lambda: None,
            _get_current_turn_id=lambda: None,
            pending_a2a_questions=MagicMock(
                mark_resolved=AsyncMock(return_value=True),
            ),
            dispatcher=MagicMock(enqueue_signal=AsyncMock()),
            _track_background_task=lambda coro, *, name="": coro.close() or MagicMock(),
        )

    backend = SQLiteBackend(str(tmp_path / "local-task-id-collision.db"))
    await backend.connect()
    agent = _agent()
    agent._raw_storage = SimpleNamespace(db=AsyncDatabase(backend))
    agent.storage = None
    feature = PeersFeature(agent)
    await feature.initialize()

    try:
        owner = await feature.send_a2a_task("first", "first task")
        collided = await feature.send_a2a_task("second", "second task")
        replayed_collision = await feature.send_a2a_task(
            "second", "second task retry",
        )

        assert owner.status is ToolResultStatus.OK
        assert owner.data["task_id"] == "shared-peer-task-id"
        for delivered_but_untrackable in (collided, replayed_collision):
            assert delivered_but_untrackable.status is ToolResultStatus.ERROR
            assert delivered_but_untrackable.data["sent"] is True
            assert (
                delivered_but_untrackable.data["task_id"]
                != "shared-peer-task-id"
            )
            assert (
                delivered_but_untrackable.data["error_type"]
                == "peer_identity_persistence_failed"
            )

            # The provisional id was never activated.  It must not fetch or
            # subscribe to second, even if a caller/replay carries second's
            # previously resolved stable identity.
            fetched = await feature.get_peer_task_result(
                "second", delivered_but_untrackable.data["task_id"],
            )
            assert fetched.status is ToolResultStatus.ERROR
            await feature._supervise_a2a_question(
                task_id=delivered_but_untrackable.data["task_id"],
                recipient="second",
                recipient_agent_id=second.agent_id,
                original_question="status?",
                sess_id="session-1",
                deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
                causation_chain=None,
            )

        assert adapter.send_a2a_task.await_count == 3
        assert fetched_from == []
        assert subscribed_to == []

        # Recreate the feature over the same durable sender store.  The
        # reservations must remain non-routable after restart, while the first
        # peer's canonical binding still owns the shared remote task id.
        restarted_agent = _agent()
        restarted_agent._raw_storage = agent._raw_storage
        restarted_agent.storage = None
        restarted = PeersFeature(restarted_agent)
        await restarted.initialize()
        for delivered_but_untrackable in (collided, replayed_collision):
            fetched = await restarted.get_peer_task_result(
                "second", delivered_but_untrackable.data["task_id"],
            )
            assert fetched.status is ToolResultStatus.ERROR
            await restarted._supervise_a2a_question(
                task_id=delivered_but_untrackable.data["task_id"],
                recipient="second",
                recipient_agent_id=second.agent_id,
                original_question="status?",
                sess_id="session-1",
                deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
                causation_chain=None,
            )

        fetched_owner = await restarted.get_peer_task_result(
            "first", "shared-peer-task-id",
        )
        assert fetched_owner.status is ToolResultStatus.OK
        await restarted._supervise_a2a_question(
            task_id="shared-peer-task-id",
            recipient="first",
            recipient_agent_id=first.agent_id,
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        assert fetched_from == [first.agent_id]
        assert subscribed_to == [first.agent_id]
        assert second.agent_id not in fetched_from
        assert second.agent_id not in subscribed_to
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_local_ambiguous_historical_route_never_falls_back_to_replacement(
    tmp_path,
):
    """A known-ambiguous local task id is not a missing legacy record.

    Local compatibility may resolve a genuinely absent pre-store task by its
    old display name.  A duplicate historical outbound route is instead
    positive evidence that no recipient can safely be selected.  This covers
    both a direct result fetch and a startup-style subscription replay after
    that display name has been reassigned.
    """
    replacement = PeerIdentity(
        agent_id="did:local:replacement",
        slug="companion",
        routing_key="replacement",
        name="Companion",
    )
    fetched_from: list[str] = []
    subscribed_to: list[str] = []
    adapter = LocalHostPeerDirectory("http://local-host")
    adapter.resolve_peer = AsyncMock(return_value=replacement)
    adapter.resolve_peer_by_agent_id = AsyncMock(return_value=replacement)

    async def _get_task(_requester, peer, _task_id):
        fetched_from.append(peer.agent_id)
        return {"status": "completed"}

    async def _subscribe_task(
        _requester, peer, task_id, *, timeout_seconds,
    ):
        del timeout_seconds
        subscribed_to.append(peer.agent_id)
        yield PeerSubscriptionEvent(
            event="status",
            data=(
                '{"id":"' + task_id + '","status":{"state":"completed",'
                '"message":{"parts":[{"text":"done"}]}}}'
            ),
        )

    adapter.get_a2a_task = AsyncMock(side_effect=_get_task)
    adapter.subscribe_a2a_task = _subscribe_task
    requester = PeerRequester("did:local:caller", object())

    def _agent() -> SimpleNamespace:
        return SimpleNamespace(
            _agent_name="caller",
            did="did:local:caller",
            peer_directory_router=adapter,
            peer_requester=requester,
            identity=None,
            _provide_causation_chain=lambda: None,
            _get_current_turn_id=lambda: None,
            pending_a2a_questions=MagicMock(
                mark_resolved=AsyncMock(return_value=True),
            ),
            dispatcher=MagicMock(enqueue_signal=AsyncMock()),
            _track_background_task=lambda coro, *, name="": coro.close() or MagicMock(),
        )

    backend = SQLiteBackend(str(tmp_path / "local-ambiguous-route.db"))
    await backend.connect()
    agent = _agent()
    agent._raw_storage = SimpleNamespace(db=AsyncDatabase(backend))
    agent.storage = None
    feature = PeersFeature(agent)
    await feature.initialize()
    task_id = "ambiguous-historical-task"

    try:
        for recipient, stable_id in (
            ("original", "did:local:original"),
            ("other", "did:local:other"),
        ):
            await outbound_store.record_outbound_dispatch(
                feature._db,
                agent_id="did:local:caller",
                task_id=task_id,
                recipient=recipient,
                recipient_agent_id=stable_id,
                verb="question",
                session_id="session-1",
                dispatch_tool="send_a2a_question",
                route_state=outbound_store.ROUTE_STATE_AMBIGUOUS,
            )

        fetched = await feature.get_peer_task_result("companion", task_id)
        assert fetched.status is ToolResultStatus.ERROR
        await feature._supervise_a2a_question(
            task_id=task_id,
            recipient="companion",
            recipient_agent_id=None,
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        restarted_agent = _agent()
        restarted_agent._raw_storage = agent._raw_storage
        restarted_agent.storage = None
        restarted = PeersFeature(restarted_agent)
        await restarted.initialize()
        fetched_after_restart = await restarted.get_peer_task_result(
            "companion", task_id,
        )
        assert fetched_after_restart.status is ToolResultStatus.ERROR
        await restarted._supervise_a2a_question(
            task_id=task_id,
            recipient="companion",
            recipient_agent_id=None,
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        assert fetched_from == []
        assert subscribed_to == []
        adapter.resolve_peer.assert_not_awaited()
        adapter.resolve_peer_by_agent_id.assert_not_awaited()
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_local_route_store_read_failure_cannot_route_replacement(
    tmp_path,
    monkeypatch,
):
    """A ready local store must fail closed when a retained lookup raises."""
    replacement = PeerIdentity(
        agent_id="did:local:replacement",
        slug="companion",
        routing_key="replacement",
        name="Companion",
    )
    fetched_from: list[str] = []
    subscribed_to: list[str] = []
    adapter = LocalHostPeerDirectory("http://local-host")
    adapter.resolve_peer = AsyncMock(return_value=replacement)
    adapter.resolve_peer_by_agent_id = AsyncMock(return_value=replacement)

    async def _get_task(_requester, peer, _task_id):
        fetched_from.append(peer.agent_id)
        return {"status": "completed"}

    async def _subscribe_task(
        _requester, peer, task_id, *, timeout_seconds,
    ):
        del timeout_seconds
        subscribed_to.append(peer.agent_id)
        yield PeerSubscriptionEvent(
            event="status",
            data=(
                '{"id":"' + task_id + '","status":{"state":"completed",'
                '"message":{"parts":[{"text":"done"}]}}}'
            ),
        )

    adapter.get_a2a_task = AsyncMock(side_effect=_get_task)
    adapter.subscribe_a2a_task = _subscribe_task
    requester = PeerRequester("did:local:caller", object())
    agent = SimpleNamespace(
        _agent_name="caller",
        did="did:local:caller",
        peer_directory_router=adapter,
        peer_requester=requester,
        identity=None,
        _provide_causation_chain=lambda: None,
        _get_current_turn_id=lambda: None,
        pending_a2a_questions=MagicMock(
            mark_resolved=AsyncMock(return_value=True),
        ),
        dispatcher=MagicMock(enqueue_signal=AsyncMock()),
        _track_background_task=lambda coro, *, name="": coro.close() or MagicMock(),
    )
    backend = SQLiteBackend(str(tmp_path / "local-route-read-failure.db"))
    await backend.connect()
    agent._raw_storage = SimpleNamespace(db=AsyncDatabase(backend))
    agent.storage = None
    feature = PeersFeature(agent)
    await feature.initialize()
    task_id = "retained-local-task"

    try:
        await outbound_store.record_outbound_dispatch(
            feature._db,
            agent_id="did:local:caller",
            task_id=task_id,
            recipient="companion",
            recipient_agent_id="did:local:original",
            verb="question",
            session_id="session-1",
            dispatch_tool="send_a2a_question",
        )
        assert feature._outbound_route_store_ready is True
        monkeypatch.setattr(
            outbound_store,
            "get_outbound_task",
            AsyncMock(side_effect=OSError("durable route lookup failed")),
        )

        fetched = await feature.get_peer_task_result("companion", task_id)
        await feature._supervise_a2a_question(
            task_id=task_id,
            recipient="companion",
            recipient_agent_id=None,
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        assert fetched.status is ToolResultStatus.ERROR
        assert fetched_from == []
        assert subscribed_to == []
        adapter.resolve_peer.assert_not_awaited()
        adapter.resolve_peer_by_agent_id.assert_not_awaited()
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_local_reservation_write_failure_never_exposes_colliding_peer_id(
    tmp_path,
    monkeypatch,
):
    """A failed local reservation is delivered but remains non-routable.

    The first peer owns the remote id.  The second send loses its initial
    reservation write and receives that same id from an older peer.  It must
    return only the sender provisional id, persist a denied reservation when
    the store recovers, and stay denied across result fetch, subscription, and
    restart.
    """
    first = PeerIdentity(
        agent_id="did:local:first",
        slug="first",
        routing_key="first",
        name="First",
    )
    second = PeerIdentity(
        agent_id="did:local:second",
        slug="second",
        routing_key="second",
        name="Second",
    )
    peers = {"first": first, "second": second}
    fetched_from: list[str] = []
    subscribed_to: list[str] = []
    adapter = LocalHostPeerDirectory("http://local-host")
    adapter.resolve_peer = AsyncMock(
        side_effect=lambda _requester, name: peers.get(name.casefold()),
    )
    adapter.resolve_peer_by_agent_id = AsyncMock(
        side_effect=lambda _requester, agent_id: next(
            (peer for peer in peers.values() if peer.agent_id == agent_id),
            None,
        ),
    )
    adapter.send_a2a_task = AsyncMock(side_effect=lambda _requester, _peer, payload: {
        "id": "shared-peer-task-id",
        "sessionId": payload["sessionId"],
        "status": {"state": "submitted"},
    })

    async def _get_task(_requester, peer, task_id):
        fetched_from.append(peer.agent_id)
        return {"id": task_id, "status": "completed", "message": "done"}

    async def _subscribe_task(
        _requester, peer, task_id, *, timeout_seconds,
    ):
        del timeout_seconds
        subscribed_to.append(peer.agent_id)
        yield PeerSubscriptionEvent(
            event="status",
            data=(
                '{"id":"' + task_id + '","status":{"state":"completed",'
                '"message":{"parts":[{"text":"done"}]}}}'
            ),
        )

    adapter.get_a2a_task = AsyncMock(side_effect=_get_task)
    adapter.subscribe_a2a_task = _subscribe_task
    requester = PeerRequester("did:local:caller", object())

    def _agent() -> SimpleNamespace:
        return SimpleNamespace(
            _agent_name="caller",
            did="did:local:caller",
            peer_directory_router=adapter,
            peer_requester=requester,
            identity=None,
            _provide_causation_chain=lambda: None,
            _get_current_turn_id=lambda: None,
            pending_a2a_questions=MagicMock(
                mark_resolved=AsyncMock(return_value=True),
            ),
            dispatcher=MagicMock(enqueue_signal=AsyncMock()),
            _track_background_task=lambda coro, *, name="": coro.close() or MagicMock(),
        )

    backend = SQLiteBackend(str(tmp_path / "local-reservation-failure.db"))
    await backend.connect()
    agent = _agent()
    agent._raw_storage = SimpleNamespace(db=AsyncDatabase(backend))
    agent.storage = None
    feature = PeersFeature(agent)
    await feature.initialize()
    original_record = outbound_store.record_outbound_dispatch
    failed_second_reservation = False

    async def _fail_second_reservation(*args, **kwargs):
        nonlocal failed_second_reservation
        if (
            kwargs["recipient"] == "second"
            and kwargs["route_state"] == outbound_store.ROUTE_STATE_RESERVED
            and not failed_second_reservation
        ):
            failed_second_reservation = True
            raise OSError("injected reservation-write failure")
        return await original_record(*args, **kwargs)

    monkeypatch.setattr(
        outbound_store, "record_outbound_dispatch", _fail_second_reservation,
    )

    try:
        owner = await feature.send_a2a_task("first", "first task")
        delivered_but_untrackable = await feature.send_a2a_task(
            "second", "second task",
        )

        assert owner.status is ToolResultStatus.OK
        assert owner.data["task_id"] == "shared-peer-task-id"
        assert delivered_but_untrackable.status is ToolResultStatus.ERROR
        assert delivered_but_untrackable.data["sent"] is True
        assert (
            delivered_but_untrackable.data["task_id"]
            != "shared-peer-task-id"
        )
        assert (
            delivered_but_untrackable.data["error_type"]
            == "peer_identity_persistence_failed"
        )

        failed_task_id = delivered_but_untrackable.data["task_id"]
        fetched = await feature.get_peer_task_result("second", failed_task_id)
        assert fetched.status is ToolResultStatus.ERROR
        await feature._supervise_a2a_question(
            task_id=failed_task_id,
            recipient="second",
            recipient_agent_id=second.agent_id,
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        restarted_agent = _agent()
        restarted_agent._raw_storage = agent._raw_storage
        restarted_agent.storage = None
        restarted = PeersFeature(restarted_agent)
        await restarted.initialize()
        fetched_after_restart = await restarted.get_peer_task_result(
            "second", failed_task_id,
        )
        assert fetched_after_restart.status is ToolResultStatus.ERROR
        await restarted._supervise_a2a_question(
            task_id=failed_task_id,
            recipient="second",
            recipient_agent_id=second.agent_id,
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        fetched_owner = await restarted.get_peer_task_result(
            "first", "shared-peer-task-id",
        )
        assert fetched_owner.status is ToolResultStatus.OK
        await restarted._supervise_a2a_question(
            task_id="shared-peer-task-id",
            recipient="first",
            recipient_agent_id=first.agent_id,
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        assert adapter.send_a2a_task.await_count == 2
        assert fetched_from == [first.agent_id]
        assert subscribed_to == [first.agent_id]
        assert second.agent_id not in fetched_from
        assert second.agent_id not in subscribed_to
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_local_all_reservation_writes_failed_never_fall_back_to_replacement(
    tmp_path,
    monkeypatch,
):
    """A ready local store with no route row is not a legacy no-store send.

    If the initial reservation plus every post-delivery denial write fails,
    restart cannot know which stable peer accepted the task.  It must refuse
    fetch and replay rather than resolving a same-name replacement peer.
    """
    original = PeerIdentity(
        agent_id="did:local:original",
        slug="companion",
        routing_key="original-route",
        name="Companion",
    )
    replacement = PeerIdentity(
        agent_id="did:local:replacement",
        slug="companion",
        routing_key="replacement-route",
        name="Companion",
    )
    peers = {"companion": original}
    fetched_from: list[str] = []
    subscribed_to: list[str] = []
    adapter = LocalHostPeerDirectory("http://local-host")
    adapter.resolve_peer = AsyncMock(
        side_effect=lambda _requester, name: peers.get(name.casefold()),
    )
    adapter.resolve_peer_by_agent_id = AsyncMock(
        side_effect=lambda _requester, agent_id: next(
            (peer for peer in peers.values() if peer.agent_id == agent_id),
            None,
        ),
    )
    adapter.send_a2a_task = AsyncMock(
        side_effect=lambda _requester, _peer, payload: {
            "id": payload["id"],
            "sessionId": payload["sessionId"],
            "status": {"state": "submitted"},
        },
    )

    async def _get_task(_requester, peer, _task_id):
        fetched_from.append(peer.agent_id)
        return {"status": "completed"}

    async def _subscribe_task(
        _requester, peer, task_id, *, timeout_seconds,
    ):
        del timeout_seconds
        subscribed_to.append(peer.agent_id)
        yield PeerSubscriptionEvent(
            event="status",
            data=(
                '{"id":"' + task_id + '","status":{"state":"completed",'
                '"message":{"parts":[{"text":"done"}]}}}'
            ),
        )

    adapter.get_a2a_task = AsyncMock(side_effect=_get_task)
    adapter.subscribe_a2a_task = _subscribe_task
    requester = PeerRequester("did:local:caller", object())

    def _agent() -> SimpleNamespace:
        return SimpleNamespace(
            _agent_name="caller",
            did="did:local:caller",
            peer_directory_router=adapter,
            peer_requester=requester,
            identity=None,
            _provide_causation_chain=lambda: None,
            _get_current_turn_id=lambda: None,
            pending_a2a_questions=MagicMock(
                mark_resolved=AsyncMock(return_value=True),
            ),
            dispatcher=MagicMock(enqueue_signal=AsyncMock()),
            _track_background_task=lambda coro, *, name="": coro.close() or MagicMock(),
        )

    backend = SQLiteBackend(str(tmp_path / "local-all-reservation-failures.db"))
    await backend.connect()
    agent = _agent()
    agent._raw_storage = SimpleNamespace(db=AsyncDatabase(backend))
    agent.storage = None
    feature = PeersFeature(agent)
    await feature.initialize()

    async def _always_fail_reservation(*_args, **_kwargs):
        raise OSError("injected persistent reservation-write failure")

    monkeypatch.setattr(
        outbound_store, "record_outbound_dispatch", _always_fail_reservation,
    )

    try:
        delivered_but_untrackable = await feature.send_a2a_task(
            "companion", "sensitive work",
        )

        assert feature._outbound_route_store_ready is True
        assert delivered_but_untrackable.status is ToolResultStatus.ERROR
        assert delivered_but_untrackable.data["sent"] is True
        assert (
            delivered_but_untrackable.data["error_type"]
            == "peer_identity_persistence_failed"
        )
        failed_task_id = delivered_but_untrackable.data["task_id"]
        assert await outbound_store.get_outbound_task(
            feature._db,
            agent_id="did:local:caller",
            task_id=failed_task_id,
        ) is None

        # The original automatic name is now assigned to another peer.  A
        # reconstructed feature must not turn the missing route row into a
        # mutable-name lookup for result fetch or subscription replay.
        peers["companion"] = replacement
        restarted_agent = _agent()
        restarted_agent._raw_storage = agent._raw_storage
        restarted_agent.storage = None
        restarted = PeersFeature(restarted_agent)
        await restarted.initialize()

        fetched = await restarted.get_peer_task_result(
            "companion", failed_task_id,
        )
        assert fetched.status is ToolResultStatus.ERROR
        await restarted._supervise_a2a_question(
            task_id=failed_task_id,
            recipient="companion",
            recipient_agent_id=None,
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        assert adapter.send_a2a_task.await_count == 1
        assert adapter.resolve_peer.await_count == 1
        assert adapter.resolve_peer.await_args.args == (requester, "companion")
        adapter.resolve_peer_by_agent_id.assert_not_awaited()
        assert fetched_from == []
        assert subscribed_to == []
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_local_route_store_initialization_failure_never_routes_replacement(
    tmp_path,
    monkeypatch,
):
    """A configured but uninitialized store is never legacy no-store mode.

    If startup fails while preparing retained outbound-route state, a later
    fetch or supervisor replay has no trustworthy binding for any old task.
    It must refuse before resolving the caller-supplied display name, including
    after that name has been reassigned to a replacement peer.
    """
    replacement = PeerIdentity(
        agent_id="did:local:replacement",
        slug="companion",
        routing_key="replacement-route",
        name="Companion",
    )
    fetched_from: list[str] = []
    subscribed_to: list[str] = []
    adapter = LocalHostPeerDirectory("http://local-host")
    adapter.resolve_peer = AsyncMock(return_value=replacement)
    adapter.resolve_peer_by_agent_id = AsyncMock(return_value=replacement)

    async def _get_task(_requester, peer, _task_id):
        fetched_from.append(peer.agent_id)
        return {"status": "completed"}

    async def _subscribe_task(
        _requester, peer, task_id, *, timeout_seconds,
    ):
        del timeout_seconds
        subscribed_to.append(peer.agent_id)
        yield PeerSubscriptionEvent(
            event="status",
            data=(
                '{"id":"' + task_id + '","status":{"state":"completed",'
                '"message":{"parts":[{"text":"done"}]}}}'
            ),
        )

    adapter.get_a2a_task = AsyncMock(side_effect=_get_task)
    adapter.subscribe_a2a_task = _subscribe_task
    requester = PeerRequester("did:local:caller", object())

    def _agent() -> SimpleNamespace:
        return SimpleNamespace(
            _agent_name="caller",
            did="did:local:caller",
            peer_directory_router=adapter,
            peer_requester=requester,
            identity=None,
            _provide_causation_chain=lambda: None,
            _get_current_turn_id=lambda: None,
            pending_a2a_questions=MagicMock(
                mark_resolved=AsyncMock(return_value=True),
            ),
            dispatcher=MagicMock(enqueue_signal=AsyncMock()),
            _track_background_task=lambda coro, *, name="": coro.close() or MagicMock(),
        )

    backend = SQLiteBackend(str(tmp_path / "local-route-store-init-failure.db"))
    await backend.connect()

    async def _fail_route_store_initialization(_db):
        raise OSError("injected outbound route store initialization failure")

    monkeypatch.setattr(
        outbound_store,
        "ensure_a2a_outbound_tasks_table",
        _fail_route_store_initialization,
    )

    agent = _agent()
    agent._raw_storage = SimpleNamespace(db=AsyncDatabase(backend))
    agent.storage = None
    feature = PeersFeature(agent)
    await feature.initialize()
    assert feature._outbound_route_store_ready is False

    try:
        task_id = "unbound-task-after-init-failure"
        fetched = await feature.get_peer_task_result("companion", task_id)
        assert fetched.status is ToolResultStatus.ERROR

        await feature._supervise_a2a_question(
            task_id=task_id,
            recipient="companion",
            recipient_agent_id=None,
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        # Simulate process restart: the persistent database remains configured
        # but route-store initialization keeps failing, and the current
        # automatic name now resolves only to the replacement.
        restarted_agent = _agent()
        restarted_agent._raw_storage = agent._raw_storage
        restarted_agent.storage = None
        restarted = PeersFeature(restarted_agent)
        await restarted.initialize()
        assert restarted._outbound_route_store_ready is False

        fetched_after_restart = await restarted.get_peer_task_result(
            "companion", task_id,
        )
        assert fetched_after_restart.status is ToolResultStatus.ERROR
        await restarted._supervise_a2a_question(
            task_id=task_id,
            recipient="companion",
            recipient_agent_id=None,
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        adapter.resolve_peer.assert_not_awaited()
        adapter.resolve_peer_by_agent_id.assert_not_awaited()
        assert fetched_from == []
        assert subscribed_to == []
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_subscription_reauthorizes_after_resolution_and_hides_revocation_details(
    tmp_path,
):
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b, deny_subscription=True)
    feature, backend = await _durable_feature_for_scope(
        router, scope_a, tmp_path, database_name="subscription-reauthorization.db",
    )
    try:
        submitted = await feature.send_a2a_task("companion", "status?")
        assert submitted.status is ToolResultStatus.OK

        await feature._supervise_a2a_question(
            task_id=submitted.data["task_id"],
            recipient="companion",
            recipient_agent_id="did:tenant-a:companion",
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        assert router.subscription_attempts == ["did:tenant-a:companion"]
        assert router.subscribed_to == []
        feature.agent.pending_a2a_questions.mark_resolved.assert_awaited_once_with(
            submitted.data["task_id"],
        )
        signal = feature.agent.dispatcher.enqueue_signal.await_args.args[0]
        assert signal.payload["state"] == "failed"
        assert (
            signal.payload["reply_text"]
            == "Peer task subscription is no longer authorized."
        )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_hosted_subscription_ignores_mismatched_pending_recipient_identity(
    tmp_path,
):
    """A pending row/input cannot override the activated outbound binding."""
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b)
    feature, backend = await _durable_feature_for_scope(
        router, scope_a, tmp_path, database_name="subscription-binding-source.db",
    )
    try:
        submitted = await feature.send_a2a_task("companion", "status?")
        assert submitted.status is ToolResultStatus.OK

        await feature._supervise_a2a_question(
            task_id=submitted.data["task_id"],
            recipient="companion",
            recipient_agent_id="did:tenant-a:replacement",
            original_question="status?",
            sess_id="session-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        assert router.subscription_attempts == []
        assert router.subscribed_to == []
        assert router.resolve_by_agent_id_calls == []
        signal = feature.agent.dispatcher.enqueue_signal.await_args.args[0]
        assert signal.payload["state"] == "failed"
        assert (
            signal.payload["reply_text"]
            == "Peer task subscription is no longer authorized."
        )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_retained_question_identity_survives_name_reassignment(tmp_path):
    """A replay/subscription must not re-resolve an old name onto a new peer."""
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b)
    original = PeerIdentity(
        agent_id="did:tenant-a:original",
        slug="companion",
        routing_key="a-original-v1",
        name="Companion",
    )
    router.scope_a_entries = {"companion": original}
    feature, backend = await _durable_feature_for_scope(
        router, scope_a, tmp_path, database_name="retained-question.db",
    )
    try:
        submitted = await feature.send_a2a_question("companion", "status?")
        assert submitted.status is ToolResultStatus.OK
        insert_args = feature.agent.pending_a2a_questions.insert.await_args.kwargs
        assert insert_args["recipient_agent_id"] == original.agent_id

        # The old display name now belongs to a replacement; the original peer was
        # renamed but remains authorized in this scope under its stable identity.
        replacement = PeerIdentity(
            agent_id="did:tenant-a:replacement",
            slug="companion",
            routing_key="a-replacement",
            name="Companion",
        )
        renamed_original = PeerIdentity(
            agent_id=original.agent_id,
            slug="original-renamed",
            routing_key="a-original-v2",
            name="Original renamed",
        )
        router.scope_a_entries = {
            "companion": replacement,
            "original-renamed": renamed_original,
        }

        # Result retrieval obtains the same persisted identity from outbound state;
        # the caller's historical display name must not select the replacement.
        fetched = await feature.get_peer_task_result(
            "companion", submitted.data["task_id"],
        )
        assert fetched.status is ToolResultStatus.OK
        assert router.fetched_from == [original.agent_id]

        await feature._supervise_a2a_question(
            task_id=submitted.data["task_id"],
            recipient="companion",
            recipient_agent_id=insert_args["recipient_agent_id"],
            original_question="status?",
            sess_id=submitted.data["session_id"],
            deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=5),
            causation_chain=None,
        )

        assert router.resolve_by_agent_id_calls == [
            original.agent_id, original.agent_id,
        ]
        assert router.subscribed_to == [original.agent_id]
        assert replacement.agent_id not in router.subscribed_to
    finally:
        await backend.close()


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
