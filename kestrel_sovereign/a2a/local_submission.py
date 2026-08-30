"""Explicit host-attested local A2A submission.

This is intentionally separate from the wire-envelope verifier.  A host that
has already authenticated two locally routed tenants can attest that delivery
was authorized without fabricating a DID signature or marking the sender
cryptographically verified.  The helper records the exact sender/recipient
relationship on the task so a host router can bind later reads and
subscriptions to that same relationship.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kestrel_sovereign.features.peers.directory import (
    PeerAccessDeniedError,
    PeerDirectoryConfigurationError,
    PeerIdentity,
    PeerRequester,
)


HOST_ATTESTED_LOCAL_SUBMISSION_METADATA = "_kestrel_host_attested_local_submission"


@dataclass(frozen=True, slots=True)
class HostAttestedLocalTaskOwner:
    """The only relationship allowed to retrieve a local-attested task."""

    requester_id: str
    recipient_id: str


def _stable_agent_id(agent: Any) -> str | None:
    for attribute in ("agent_id", "did"):
        value = getattr(agent, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None


def _recipient_peer_context(recipient: Any) -> tuple[Any, PeerRequester, str]:
    """Return the host-injected recipient scope, never caller input."""

    router = getattr(recipient, "peer_directory_router", None)
    requester = getattr(recipient, "peer_requester", None)
    recipient_id = _stable_agent_id(recipient)
    if (
        router is None
        or not isinstance(requester, PeerRequester)
        or recipient_id is None
        or requester.identity != recipient_id
        or not callable(getattr(router, "authorize_inbound_sender", None))
    ):
        raise PeerDirectoryConfigurationError(
            "host-attested local A2A delivery requires a recipient-bound peer scope"
        )
    return router, requester, recipient_id


async def submit_host_attested_local_a2a_task(
    *,
    recipient: Any,
    sender_requester: PeerRequester,
    recipient_peer: PeerIdentity,
    params: Any,
    artifacts: list[Any] | None = None,
) -> Any:
    """Create one locally routed A2A task under explicit host attestation.

    The caller has already used its scoped directory to resolve the recipient.
    This helper independently asks the *recipient's* current scoped directory
    whether that exact sender is still allowed.  It then stamps immutable task
    ownership metadata.  It never calls the envelope verifier and deliberately
    records ``sender_verified=False``: local host attestation is not a
    cryptographic signature.
    """

    if not isinstance(sender_requester, PeerRequester):
        raise PeerDirectoryConfigurationError(
            "host-attested local A2A delivery requires a trusted sender requester"
        )
    if not isinstance(recipient_peer, PeerIdentity):
        raise PeerAccessDeniedError("host-attested local A2A recipient is invalid")
    router, recipient_requester, recipient_id = _recipient_peer_context(recipient)
    if recipient_peer.agent_id != recipient_id:
        raise PeerAccessDeniedError("host-attested local A2A recipient changed")
    if not callable(getattr(getattr(recipient, "task_manager", None), "create_task", None)):
        raise PeerAccessDeniedError("host-attested local A2A recipient is unavailable")

    resolve_sender_scope = getattr(router, "resolve_peer_by_agent_id", None)
    if not callable(resolve_sender_scope):
        raise PeerDirectoryConfigurationError(
            "host-attested local A2A delivery requires requester reauthorization"
        )
    # Validate both directions while using host-owned requester handles.  The
    # recipient's inbound policy alone proves only the sender DID is eligible;
    # this second lookup proves the original requester still owns a scope in
    # which this exact recipient is a peer.
    sender_visible_recipient = await resolve_sender_scope(
        sender_requester, recipient_id
    )
    if (
        not isinstance(sender_visible_recipient, PeerIdentity)
        or sender_visible_recipient.agent_id != recipient_id
    ):
        raise PeerAccessDeniedError(
            "host-attested local A2A requester is no longer authorized"
        )
    authorized = await router.authorize_inbound_sender(
        recipient_requester, sender_requester.identity
    )
    if authorized is not True:
        raise PeerAccessDeniedError(
            "host-attested local A2A sender is not authorized by the recipient"
        )

    # Re-read the recipient's host-owned scope after the await.  A companion
    # may have been retired, transferred, or repointed while its directory was
    # checking authorization; committing against the stale scope would turn a
    # valid answer into an unauthorized task.
    router_after, requester_after, recipient_after = _recipient_peer_context(recipient)
    if (
        router_after is not router
        or requester_after != recipient_requester
        or recipient_after != recipient_id
    ):
        raise PeerAccessDeniedError(
            "host-attested local A2A recipient authorization changed during delivery"
        )
    sender_visible_recipient = await resolve_sender_scope(
        sender_requester, recipient_id
    )
    if (
        not isinstance(sender_visible_recipient, PeerIdentity)
        or sender_visible_recipient.agent_id != recipient_id
    ):
        raise PeerAccessDeniedError(
            "host-attested local A2A requester authorization changed during delivery"
        )

    metadata = dict(getattr(params, "metadata", None) or {})
    claimed_sender = metadata.get("sender")
    if claimed_sender not in (None, "", sender_requester.identity):
        raise PeerAccessDeniedError("host-attested local A2A sender does not match payload")
    # A caller must not carry forward an arbitrary wire signature/trust verdict
    # into the local path.  The host attestation is explicit and lower-trust.
    metadata.pop("signature", None)
    metadata["sender"] = sender_requester.identity
    metadata["sender_verified"] = False
    metadata["sender_trust"] = "host_attested_local"
    metadata[HOST_ATTESTED_LOCAL_SUBMISSION_METADATA] = {
        "requester_id": sender_requester.identity,
        "recipient_id": recipient_id,
    }
    params.metadata = metadata

    manager = getattr(recipient, "_a2a_host_manager", None)
    lease_factory = getattr(manager, "a2a_execution_lease", None)
    if not callable(lease_factory):
        lease_factory = getattr(manager, "a2a_lifecycle_lease", None)
    if callable(lease_factory):
        # A hosted recipient can be retired while a local router is awaiting
        # scope checks.  Its manager's lifecycle lease makes the final task
        # admission obey the same publication boundary as Core's HTTP path.
        async with lease_factory():
            policy_for = getattr(manager, "a2a_hosted_policy_for", None)
            if callable(policy_for) and policy_for(recipient) is None:
                raise PeerAccessDeniedError(
                    "host-attested local A2A recipient is no longer published"
                )
            return await recipient.task_manager.create_task(
                params=params,
                agent_name=recipient_id,
                artifacts=artifacts or None,
                creator_agent_id=sender_requester.identity,
            )
    return await recipient.task_manager.create_task(
        params=params,
        agent_name=recipient_id,
        artifacts=artifacts or None,
        creator_agent_id=sender_requester.identity,
    )


def host_attested_local_task_owner(task: Any) -> HostAttestedLocalTaskOwner | None:
    """Return validated ownership evidence from a locally attested task."""

    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    owner = metadata.get(HOST_ATTESTED_LOCAL_SUBMISSION_METADATA)
    if not isinstance(owner, Mapping):
        return None
    requester_id = owner.get("requester_id")
    recipient_id = owner.get("recipient_id")
    if not all(isinstance(value, str) and value for value in (requester_id, recipient_id)):
        return None
    return HostAttestedLocalTaskOwner(
        requester_id=requester_id,
        recipient_id=recipient_id,
    )


def require_host_attested_local_task_owner(
    task: Any,
    *,
    requester_id: str,
    recipient_id: str,
) -> None:
    """Fail closed unless a task belongs to this exact routed relationship."""

    owner = host_attested_local_task_owner(task)
    if owner is None or owner != HostAttestedLocalTaskOwner(
        requester_id=requester_id,
        recipient_id=recipient_id,
    ):
        raise PeerAccessDeniedError("A2A task is not owned by this peer relationship")
