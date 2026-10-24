"""A2A identity/policy consumers preserve original PostgreSQL control.

The roster, requester capability and durable identity fields are explicit
private component inputs, not Agent READY/genesis or authenticated peer grants.
The directory never authorizes a sender: native failure or ordinary refusal
must create no task. Actual SQL, topology leases and immutable policy are used.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from fastapi import HTTPException, Request

from kestrel_sovereign.a2a.inbound_authorization import RecipientA2ASenderAuthorizer
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.endpoints.agent import _scoped_agent_did_or_503
from kestrel_sovereign.endpoints.task_inbox import list_inbox, read_inbox
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.features.peers.directory import PeerRequester
from kestrel_sovereign.features.storage_access import AgentIdentityUnavailable
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.utils.native_a2a import manager_for


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "door",
    [
        "manager-policy",
        "inbound-identity",
        "task-recipient",
        "task-scope",
        "task-list",
    ],
)
@pytest.mark.parametrize("kind", ["native-control", "ordinary"])
async def test_a2a_policy_identity_does_not_turn_original_control_into_refusal(
    native_memory_owners, tmp_path, door, kind
):
    n = native_memory_owners
    tasks = manager_for(n.backend, NativeRecipientScope(n.did), owns_backend=False)
    await tasks.initialize()
    original = OSError("original A2A policy native release acknowledgement lost")
    ordinary = OSError("ordinary policy callback refused after acknowledged SQL")
    sources, fired = [], []
    pool = n.backend._pool

    @asynccontextmanager
    async def native_release(native):
        async with native as connection:
            yield connection
        if not fired:
            fired.append(True)
            raise original

    async def actual_policy_read():
        if kind == "native-control":
            n.backend._pool = PostgresPoolAdapter(pool, native_release)
        try:
            await n.backend.execute(
                "INSERT INTO effects VALUES (1, 'actual original A2A directory effect')"
            )
        except BaseException as error:
            sources.append(error)
            raise
        finally:
            n.backend._pool = pool
        fired.append(True)
        raise ordinary

    @dataclass
    class RosterProjection:
        did: str
        _agent_name: str
        identity: object = None

        @property
        def agent_id(self):
            return self.did

    recipient = RosterProjection(n.did, "recipient")
    recipient.task_manager = tasks
    recipient._raw_storage = n.pg_storage
    sender = RosterProjection(n.did + ":legacy-component", "legacy-component")
    manager = AgentManager(base_data_dir=tmp_path / "private-policy-roster")
    # Private component roster input only: do not invoke registration or
    # bypass any genesis/readiness verifier to publish a fake live agent.
    manager._agents = {"recipient": recipient, "legacy": sender}
    manager._agent_names = {n.did: "recipient", sender.did: "legacy"}

    class NativeDirectory:
        async def authorize_inbound_sender(self, requester, sender_id):
            assert requester.identity == recipient.did and sender_id == sender.did
            await actual_policy_read()
            raise AssertionError("directory fault fixture must never grant admission")

    router = NativeDirectory()
    requester = PeerRequester(n.did, object())
    recipient.peer_directory_router = router
    recipient.peer_requester = requester
    authorizer = RecipientA2ASenderAuthorizer(manager, recipient=recipient)
    async with manager.a2a_lifecycle_lease():
        policy = manager.install_a2a_hosted_policy(
            recipient,
            resolver=None,
            authorizer=authorizer,
            router=router,
            requester=requester,
        )
    n.backend._execution_custody = ExecutionCustody(GenerationFence())
    failure = result = None
    try:
        if door == "manager-policy":
            try:
                async with manager.a2a_execution_lease():
                    result = await policy.authorizer.authorize_principal_with_policy(
                        sender.did, router=policy.router, requester=policy.requester,
                    )
            except BaseException as error:
                failure = error
        else:
            # The synchronous identity callback carries an original already
            # observed native receipt; it does not invent a driver exception.
            try:
                await actual_policy_read()
            except BaseException as receipt:
                retained = receipt

            class IdentityReadback:
                task_manager = tasks

                @property
                def did(self):
                    if kind == "native-control":
                        raise AgentIdentityUnavailable(
                            "declared typed identity readback carrier"
                        ) from retained
                    raise AgentIdentityUnavailable("ordinary missing identity")

            try:
                if door == "inbound-identity":
                    result = authorizer._stable_agent_id(IdentityReadback())
                elif door == "task-recipient":
                    result = await read_inbox(
                        Request({"type": "http", "state": {"agent": IdentityReadback()}}), "absent-task",
                    )
                elif door == "task-list":
                    result = await list_inbox(
                        Request({"type": "http", "state": {"agent": IdentityReadback()}}),
                        status=None,
                        limit=50,
                    )
                else:
                    result = _scoped_agent_did_or_503(
                        IdentityReadback(), "native policy read requires"
                    )
            except BaseException as error:
                failure = error

        assert fired, f"actual native policy operation was not reached: {failure!r}"
        assert (
            await n.observer.fetchval(f'SELECT count(*) FROM "{n.schema}".effects') == 1
        )
        assert (
            await n.observer.fetchval(f'SELECT count(*) FROM "{n.schema}".a2a_tasks')
            == 0
        )
        if kind == "native-control":
            assert sources and execution_commit_outcome(sources[0]) == "committed"
            assert original in tuple(_explicit_control_chain(sources[0]))
            assert failure is not None and is_execution_control_error(failure), (
                f"native A2A identity/policy control became ordinary refusal: "
                f"door={door} failure={failure!r} result={result!r}"
            )
            assert sources[0] in tuple(_explicit_control_chain(failure))
            assert execution_commit_outcome(failure) == "committed"
        elif door in {"task-recipient", "task-scope", "task-list"}:
            assert isinstance(failure, HTTPException)
            assert failure.status_code == 503
            assert not is_execution_control_error(failure)
        elif door == "manager-policy":
            assert failure is ordinary and result is None
            assert not is_execution_control_error(failure)
        else:
            assert failure is None and result is None
        assert n.requests == []
    finally:
        n.backend._pool = pool
        n.backend._execution_custody = None
        await tasks.close()
