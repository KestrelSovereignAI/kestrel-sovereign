"""Contracts for non-cryptographic, host-attested local A2A delivery."""

from types import SimpleNamespace

import pytest

from kestrel_sovereign.a2a.local_submission import (
    HOST_ATTESTED_LOCAL_SUBMISSION_METADATA,
    require_host_attested_local_task_owner,
    submit_host_attested_local_a2a_task,
)
from kestrel_sovereign.features.peers.directory import (
    PeerAccessDeniedError,
    PeerIdentity,
    PeerRequester,
)


class _Router:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls = []

    async def authorize_inbound_sender(self, requester, sender_id):
        self.calls.append((requester, sender_id))
        return self.allowed

    async def resolve_peer_by_agent_id(self, requester, agent_id):
        if not self.allowed:
            return None
        return PeerIdentity(agent_id, "recipient", "recipient")


class _TaskManager:
    async def create_task(
        self, *, params, agent_name, artifacts, creator_agent_id=None
    ):
        return SimpleNamespace(
            metadata=dict(params.metadata),
            agent_name=agent_name,
            artifacts=artifacts,
            creator_agent_id=creator_agent_id,
        )


def _recipient(router):
    recipient_id = "did:frinz:recipient"
    return SimpleNamespace(
        did=recipient_id,
        peer_directory_router=router,
        peer_requester=PeerRequester(recipient_id, object()),
        task_manager=_TaskManager(),
    )


@pytest.mark.asyncio
async def test_local_submission_is_recipient_authorized_and_not_cryptographically_verified():
    router = _Router()
    recipient = _recipient(router)
    sender = PeerRequester("did:frinz:sender", object())
    params = SimpleNamespace(metadata={"sender": "did:frinz:sender", "signature": {"forged": True}})

    task = await submit_host_attested_local_a2a_task(
        recipient=recipient,
        sender_requester=sender,
        recipient_peer=PeerIdentity("did:frinz:recipient", "recipient", "recipient"),
        params=params,
    )

    assert router.calls == [(recipient.peer_requester, "did:frinz:sender")]
    assert task.metadata["sender_verified"] is False
    assert task.creator_agent_id == "did:frinz:sender"
    assert task.metadata["sender_trust"] == "host_attested_local"
    assert "signature" not in task.metadata
    assert task.metadata[HOST_ATTESTED_LOCAL_SUBMISSION_METADATA] == {
        "requester_id": "did:frinz:sender",
        "recipient_id": "did:frinz:recipient",
    }
    require_host_attested_local_task_owner(
        task,
        requester_id="did:frinz:sender",
        recipient_id="did:frinz:recipient",
    )
    with pytest.raises(PeerAccessDeniedError):
        require_host_attested_local_task_owner(
            task,
            requester_id="did:frinz:other-user",
            recipient_id="did:frinz:recipient",
        )


@pytest.mark.asyncio
async def test_local_submission_rejects_recipient_scope_denial_and_sender_spoofing():
    recipient = _recipient(_Router(allowed=False))
    sender = PeerRequester("did:frinz:sender", object())

    with pytest.raises(PeerAccessDeniedError, match="authorized"):
        await submit_host_attested_local_a2a_task(
            recipient=recipient,
            sender_requester=sender,
            recipient_peer=PeerIdentity("did:frinz:recipient", "recipient", "recipient"),
            params=SimpleNamespace(metadata={}),
        )

    recipient = _recipient(_Router())
    with pytest.raises(PeerAccessDeniedError, match="does not match"):
        await submit_host_attested_local_a2a_task(
            recipient=recipient,
            sender_requester=sender,
            recipient_peer=PeerIdentity("did:frinz:recipient", "recipient", "recipient"),
            params=SimpleNamespace(metadata={"sender": "did:frinz:other-user"}),
        )
