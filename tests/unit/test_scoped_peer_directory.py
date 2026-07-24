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
async def test_hosted_rekey_failure_fails_closed_with_the_peer_task_id(
    tmp_path, monkeypatch,
):
    """A hosted result must identify the delivered task if its binding fails."""
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
        assert rejected.data["task_id"] == peer_task_id
        assert rejected.data["sender_task_id"] != peer_task_id
        assert rejected.data["error_type"] == "peer_identity_persistence_failed"
        assert router.sent_to == ["did:tenant-a:companion"]
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_local_rekey_failure_keeps_the_delivered_task_successful(
    tmp_path, monkeypatch, caplog,
):
    """Local-host audit rekeys are best effort after successful delivery."""
    peer = PeerIdentity(
        agent_id="did:local:companion",
        slug="companion",
        routing_key="companion",
        name="Companion",
    )
    adapter = LocalHostPeerDirectory("http://local-host")
    adapter.resolve_peer = AsyncMock(return_value=peer)
    adapter.send_a2a_task = AsyncMock(return_value={
        "id": "peer-assigned-task-id",
        "sessionId": "peer-session-id",
        "status": {"state": "submitted"},
    })
    requester = PeerRequester("did:local:caller", object())
    agent = SimpleNamespace(
        _agent_name="caller",
        did="did:local:caller",
        peer_directory_router=adapter,
        peer_requester=requester,
        identity=None,
        _provide_causation_chain=lambda: None,
        _get_current_turn_id=lambda: None,
    )
    feature = PeersFeature(agent)
    backend = SQLiteBackend(str(tmp_path / "local-rekey-failure.db"))
    await backend.connect()
    agent._raw_storage = SimpleNamespace(db=AsyncDatabase(backend))
    agent.storage = None
    await feature.initialize()
    monkeypatch.setattr(
        outbound_store,
        "rekey_outbound_task",
        AsyncMock(return_value=0),
    )

    try:
        delivered = await feature.send_a2a_task("companion", "local work")

        assert delivered.status is ToolResultStatus.OK
        assert delivered.data["sent"] is True
        assert delivered.data["task_id"] == "peer-assigned-task-id"
        adapter.send_a2a_task.assert_awaited_once()
        assert "could not rekey its best-effort outbound audit record" in caplog.text
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_subscription_reauthorizes_after_resolution_and_hides_revocation_details():
    scope_a, scope_b = object(), object()
    router = ScopedRouter(scope_a, scope_b, deny_subscription=True)
    feature = _feature_for_scope(router, scope_a)
    await feature.initialize()

    await feature._supervise_a2a_question(
        task_id="task-1",
        recipient="companion",
        recipient_agent_id="did:tenant-a:companion",
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
