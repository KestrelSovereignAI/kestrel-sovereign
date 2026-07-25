"""Recipient-scoped authorization after successful A2A DID verification."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

from kestrel_sovereign.a2a.inbound_authorization import (
    RecipientA2ASenderAuthorizer,
    has_a2a_inbound_scoped_policy,
    install_a2a_inbound_sender_authorizer,
    mark_a2a_inbound_scoped_policy,
)
from kestrel_sovereign.features.peers.directory import PeerRequester


class _Manager:
    def __init__(self, agents: dict[str, object]):
        self._agents = agents

    def list_agents(self):
        return dict(self._agents)


def _agent(
    agent_id: str,
    signing_did: str,
    *,
    router=None,
    requester=None,
):
    return SimpleNamespace(
        agent_id=agent_id,
        did=agent_id,
        identity=SimpleNamespace(signing_did=signing_did),
        peer_directory_router=router,
        peer_requester=requester,
        a2a_inbound_sender_authorizer=None,
    )


@dataclass
class _ScopedRouter:
    allowed: dict[object, set[str]]
    calls: list[tuple[PeerRequester, str]] = field(default_factory=list)

    async def authorize_inbound_sender(
        self,
        requester: PeerRequester,
        sender_id: str,
    ) -> bool:
        self.calls.append((requester, sender_id))
        return sender_id in self.allowed.get(requester.authorization_scope, set())


DID_SAME_USER = "did:web:example.test:agent:emma"
DID_OTHER_USER = "did:web:example.test:agent:claw"
DID_RECIPIENT = "did:web:example.test:agent:ivy"
DID_EXTERNAL = "did:web:remote.example:agent:phoenix"
SENDER_SAME_ID = "did:pkh:tenant-a:emma"
SENDER_OTHER_ID = "did:pkh:tenant-b:claw"
RECIPIENT_ID = "did:pkh:tenant-a:ivy"


def test_same_user_sender_allowed_and_other_user_sender_denied():
    scope = object()
    requester = PeerRequester(RECIPIENT_ID, scope)
    router = _ScopedRouter({scope: {SENDER_SAME_ID}})
    same_user = _agent(SENDER_SAME_ID, DID_SAME_USER)
    other_user = _agent(SENDER_OTHER_ID, DID_OTHER_USER)
    recipient = _agent(
        RECIPIENT_ID,
        DID_RECIPIENT,
        router=router,
        requester=requester,
    )
    authorizer = RecipientA2ASenderAuthorizer(
        _Manager(
            {
                "same": same_user,
                "other": other_user,
                "recipient": recipient,
            }
        ),
        recipient=recipient,
    )

    assert asyncio.run(authorizer.authorize(DID_SAME_USER)) is True
    assert asyncio.run(authorizer.authorize(DID_OTHER_USER)) is False
    assert router.calls == [
        (requester, SENDER_SAME_ID),
        (requester, SENDER_OTHER_ID),
    ]


def test_external_unloaded_sender_requires_explicit_directory_authorization():
    scope = object()
    requester = PeerRequester(RECIPIENT_ID, scope)
    router = _ScopedRouter({scope: set()})
    recipient = _agent(
        RECIPIENT_ID,
        DID_RECIPIENT,
        router=router,
        requester=requester,
    )
    authorizer = RecipientA2ASenderAuthorizer(
        _Manager({"recipient": recipient}),
        recipient=recipient,
    )

    assert asyncio.run(authorizer.authorize(DID_EXTERNAL)) is False
    assert router.calls == [(requester, DID_EXTERNAL)]

    router.allowed[scope].add(DID_EXTERNAL)
    assert asyncio.run(authorizer.authorize(DID_EXTERNAL)) is True
    assert router.calls[-1] == (requester, DID_EXTERNAL)


def test_removed_scoped_context_is_revocation_not_standalone_fallback():
    scope = object()
    requester = PeerRequester(RECIPIENT_ID, scope)
    router = _ScopedRouter({scope: {SENDER_SAME_ID}})
    sender = _agent(SENDER_SAME_ID, DID_SAME_USER)
    recipient = _agent(
        RECIPIENT_ID,
        DID_RECIPIENT,
        router=router,
        requester=requester,
    )
    authorizer = install_a2a_inbound_sender_authorizer(
        _Manager({"sender": sender, "recipient": recipient}),
        recipient=recipient,
    )
    assert asyncio.run(authorizer.authorize(DID_SAME_USER)) is True

    recipient.peer_directory_router = None
    recipient.peer_requester = None

    assert has_a2a_inbound_scoped_policy(recipient) is True
    assert authorizer.requires_verified_sender is True
    assert asyncio.run(authorizer.authorize(DID_SAME_USER)) is False
    assert router.calls == [(requester, SENDER_SAME_ID)]


def test_requester_identity_must_match_recipient_stable_identity():
    scope = object()
    mismatched = PeerRequester("did:pkh:tenant-b:ivy", scope)
    router = _ScopedRouter({scope: {SENDER_SAME_ID}})
    sender = _agent(SENDER_SAME_ID, DID_SAME_USER)
    recipient = _agent(
        RECIPIENT_ID,
        DID_RECIPIENT,
        router=router,
        requester=mismatched,
    )
    authorizer = RecipientA2ASenderAuthorizer(
        _Manager({"sender": sender, "recipient": recipient}),
        recipient=recipient,
    )

    assert authorizer.requires_verified_sender is True
    assert asyncio.run(authorizer.authorize(DID_SAME_USER)) is False
    assert router.calls == []


def test_scoped_router_missing_inbound_method_fails_closed():
    scope = object()
    recipient = _agent(
        RECIPIENT_ID,
        DID_RECIPIENT,
        router=SimpleNamespace(),
        requester=PeerRequester(RECIPIENT_ID, scope),
    )
    authorizer = RecipientA2ASenderAuthorizer(
        _Manager({"recipient": recipient}),
        recipient=recipient,
    )

    assert authorizer.requires_verified_sender is True
    assert asyncio.run(authorizer.authorize(DID_EXTERNAL)) is False


def test_true_standalone_authorizer_preserves_shared_api_key_compatibility():
    recipient = _agent(RECIPIENT_ID, DID_RECIPIENT)
    authorizer = RecipientA2ASenderAuthorizer(
        _Manager({"recipient": recipient}),
        recipient=recipient,
    )

    assert authorizer.requires_verified_sender is False
    assert asyncio.run(authorizer.authorize(DID_EXTERNAL)) is True


def test_another_recipient_scope_does_not_contaminate_standalone_policy():
    scope = object()
    scoped_router = _ScopedRouter({scope: set()})
    scoped_recipient = _agent(
        "did:pkh:tenant-b:recipient",
        "did:web:example.test:agent:tenant-b-recipient",
        router=scoped_router,
        requester=PeerRequester("did:pkh:tenant-b:recipient", scope),
    )
    standalone_recipient = _agent(RECIPIENT_ID, DID_RECIPIENT)
    manager = _Manager(
        {
            "scoped": scoped_recipient,
            "standalone": standalone_recipient,
        }
    )

    standalone_authorizer = RecipientA2ASenderAuthorizer(
        manager,
        recipient=standalone_recipient,
    )

    assert standalone_authorizer.requires_verified_sender is False
    assert asyncio.run(standalone_authorizer.authorize(DID_EXTERNAL)) is True


def test_registration_policy_marker_is_monotonic_without_live_authorizer():
    recipient = _agent(RECIPIENT_ID, DID_RECIPIENT)

    assert mark_a2a_inbound_scoped_policy(recipient, required=True) is True
    assert mark_a2a_inbound_scoped_policy(recipient, required=False) is True

    recipient.peer_directory_router = None
    recipient.peer_requester = None
    recipient.a2a_inbound_sender_authorizer = None
    assert has_a2a_inbound_scoped_policy(recipient) is True
