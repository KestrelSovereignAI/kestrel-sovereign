"""Bind native admission to the host's exact published policy and identities."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING

from starlette.exceptions import HTTPException

from kestrel_sovereign.a2a.did_registry import same_a2a_sender_identity_witness
from kestrel_sovereign.a2a.native_security import (
    NativeDIDRequestGuard,
    NativeRequestAuthority,
)
from kestrel_sovereign.a2a.native_sender import NativeProofPolicy, NativeVerifiedSender
from kestrel_sovereign.a2a.replay_store import SharedReplayNonceStore
from kestrel_sovereign.a2a.task_manager import TaskManager

if TYPE_CHECKING:
    from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.multi_agent.agent_manager import (
        A2AHostedPolicy,
        AgentManager,
    )


class _HostedNativeRequest:
    """One request's witness, never shared between concurrent verifications."""

    def __init__(
        self,
        manager: AgentManager,
        routing_key: str,
        policy: A2AHostedPolicy,
        task_manager: TaskManager,
        task_store: TaskStore,
        recipient_signing_did: str,
        recipient_witness: tuple,
    ):
        self.manager = manager
        self.routing_key = routing_key
        self.policy = policy
        self.task_manager = task_manager
        self.task_store = task_store
        self.scope = self.task_store.scope
        self.recipient_signing_did = recipient_signing_did
        self.recipient_witness = recipient_witness
        self.sender_did: str | None = None
        self.sender_witness: tuple | None = None

    def require_current(self, sender: NativeVerifiedSender | None) -> None:
        recipient = self.policy.recipient
        scope = self.scope
        if (
            self.manager.get_agent_name(scope.recipient_id) != self.routing_key
            or self.manager.get_agent(self.routing_key) is not recipient
            or self.manager.a2a_hosted_policy_for(recipient) is not self.policy
            or getattr(recipient, "task_manager", None) is not self.task_manager
            or self.task_manager.task_store is not self.task_store
            or self.policy.recipient_id != scope.recipient_id
            or not same_a2a_sender_identity_witness(
                self.recipient_witness,
                self.manager.a2a_sender_identity_witness(self.recipient_signing_did),
            )
            or self.policy.authorizer.has_valid_policy_scope(
                self.policy.router, self.policy.requester
            )
            is not True
        ):
            raise HTTPException(
                status_code=403, detail="A2A recipient registration changed"
            )
        if sender is None:
            return
        if self.sender_witness is None or self.sender_did != sender.sender:
            raise RuntimeError(
                "Native sender verification has no request-owned identity witness"
            )
        current = self.manager.a2a_sender_identity_witness(sender.sender)
        if (
            self.sender_witness[0] == "ambiguous"
            or not same_a2a_sender_identity_witness(self.sender_witness, current)
            or (
                self.sender_witness[0] == "local"
                and self.sender_witness[3] != sender.verification_document_fingerprint
            )
        ):
            raise HTTPException(status_code=403, detail="A2A sender identity changed")

    async def resolve(self, did: str):
        self.require_current(None)
        if self.sender_witness is not None:
            raise RuntimeError("A native request cannot replace its sender witness")
        self.sender_did = did
        self.sender_witness = self.manager.a2a_sender_identity_witness(did)
        document = self.policy.resolver.resolve(did)
        if inspect.isawaitable(document):
            document = await document
        self.require_current(None)
        return document

    async def authorize(self, did: str) -> str | None:
        return await self.policy.authorizer.authorize_principal_with_policy(
            did, router=self.policy.router, requester=self.policy.requester
        )


def create_hosted_native_guard(
    manager: AgentManager,
    *,
    routing_key: str,
    recipient: KestrelAgent,
    required_extensions: frozenset[str],
    proof_policy: NativeProofPolicy,
    clock: Callable[[], float],
) -> NativeDIDRequestGuard:
    """Capture one published generation while the host owns its lifecycle lease.

    There is no unsigned admission or operator-key principal. Host publication
    supplies a hybrid recipient identity and an explicit directory policy; a
    later policy generation requires a new guard and route registration.
    """
    task_manager = getattr(recipient, "task_manager", None)
    if not isinstance(task_manager, TaskManager):
        raise TypeError(
            "Native hosting requires the recipient's canonical task manager"
        )
    task_store = task_manager.task_store
    scope = task_store.scope
    policy = manager.a2a_hosted_policy_for(recipient)
    if (
        manager.get_agent_name(scope.recipient_id) != routing_key
        or manager.get_agent(routing_key) is not recipient
        or policy is None
        or policy.recipient_id != scope.recipient_id
        or not callable(getattr(policy.resolver, "resolve", None))
        or not callable(getattr(policy.authorizer, "has_valid_policy_scope", None))
        or not callable(
            getattr(policy.authorizer, "authorize_principal_with_policy", None)
        )
        or policy.authorizer.has_valid_policy_scope(policy.router, policy.requester)
        is not True
    ):
        raise ValueError("Native hosting requires an exact published recipient policy")
    identity = getattr(recipient, "identity", None)
    signing_did = getattr(identity, "signing_did", None)
    if not isinstance(signing_did, str) or not signing_did:
        raise ValueError(
            "Native hosting requires the recipient's hybrid signing identity"
        )
    recipient_witness = manager.a2a_sender_identity_witness(signing_did)
    if recipient_witness[0] != "local" or recipient_witness[1] is not recipient:
        raise ValueError(
            "Native hosting requires an unambiguous hybrid recipient identity"
        )

    def authority() -> NativeRequestAuthority:
        request = _HostedNativeRequest(
            manager,
            routing_key,
            policy,
            task_manager,
            task_store,
            signing_did,
            recipient_witness,
        )
        return NativeRequestAuthority(
            request.resolve, request.authorize, request.require_current
        )

    return NativeDIDRequestGuard(
        routing_key=routing_key,
        scope=scope,
        required_extensions=required_extensions,
        proof_policy=proof_policy,
        authority_factory=authority,
        replay_store=SharedReplayNonceStore(task_store.backend),
        lifecycle_lease=manager.a2a_execution_lease,
        clock=clock,
    )
