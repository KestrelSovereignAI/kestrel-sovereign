"""Real hybrid proof and durable recipient admission around canonical storage."""

import asyncio
from dataclasses import FrozenInstanceError, replace

import pytest
from a2a.types import GetTaskRequest
from a2a.utils.errors import (
    ExtensionSupportRequiredError,
    InternalError,
    InvalidParamsError,
)
from starlette.exceptions import HTTPException

from kestrel_sovereign.a2a.native_sender import SENDER_PROOF_HEADER
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.storage.db.interface import QueryError
from tests.utils.native_a2a_security import (
    EXTENSIONS,
    NOW,
    PRINCIPAL,
    SENDER,
    send,
    signed_context,
)


@pytest.mark.parametrize("protocol", ["JSONRPC", "HTTP+JSON"])
async def test_verified_sender_policy_supplies_the_only_durable_creator(
    native_request_admission, protocol
):
    params = send(native_request_admission)
    before = params.SerializeToString(deterministic=True)
    context = signed_context(native_request_admission, params, protocol=protocol)
    async with native_request_admission.guard.admit(
        params, "SendMessage", context
    ) as granted:
        accepted = await native_request_admission.manager.create_task(
            granted.request, creator_agent_id=granted.principal_id
        )
        assert granted.principal_id == PRINCIPAL
    snapshot = await native_request_admission.manager.get_task_snapshot(
        accepted.id, recipient_agent_id=native_request_admission.scope.recipient_id
    )
    assert snapshot.ownership.creator_agent_id == PRINCIPAL
    assert snapshot.ownership.tenant == native_request_admission.scope.tenant
    assert native_request_admission.state.authorizations == [SENDER]
    assert params.SerializeToString(deterministic=True) == before
    with pytest.raises(HTTPException) as replay:
        async with native_request_admission.guard.admit(params, "SendMessage", context):
            pytest.fail("Replayed proof reached native_request_admission")
    assert replay.value.status_code == 401
    assert native_request_admission.state.authorizations == [SENDER]


@pytest.mark.parametrize("operation", ["GetTask", "SubscribeToTask", "CancelTask"])
async def test_each_native_read_or_mutation_authenticates_its_actual_operation(
    native_request_admission, operation
):
    from a2a.types import CancelTaskRequest, SubscribeToTaskRequest

    cls = {
        "GetTask": GetTaskRequest,
        "SubscribeToTask": SubscribeToTaskRequest,
        "CancelTask": CancelTaskRequest,
    }[operation]
    params = cls(id="existing", tenant=native_request_admission.scope.tenant)
    context = signed_context(native_request_admission, params, operation=operation)
    async with native_request_admission.guard.admit(
        params, operation, context
    ) as granted:
        assert granted.request == params and granted.principal_id == PRINCIPAL


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "malformed", "audience", "expired"]
)
async def test_bad_proof_never_reaches_recipient_policy_or_task_admission(
    native_request_admission, fault
):
    params = send(native_request_admission)
    headers = {
        "missing": lambda proof: [],
        "duplicate": lambda proof: [
            (SENDER_PROOF_HEADER, proof),
            (SENDER_PROOF_HEADER, proof),
        ],
        "malformed": lambda proof: [(SENDER_PROOF_HEADER, "invalid")],
    }.get(fault)
    changes = (
        {"recipient": "did:kestrel:other"}
        if fault == "audience"
        else {
            "timestamp": NOW
            - native_request_admission.guard.proof_policy.max_age_seconds
            - 1
        }
        if fault == "expired"
        else {}
    )
    context = signed_context(
        native_request_admission, params, headers=headers, proof_changes=changes
    )
    with pytest.raises(HTTPException) as failed:
        async with native_request_admission.guard.admit(params, "SendMessage", context):
            pytest.fail("Invalid proof reached native_request_admission")
    assert failed.value.status_code == 401
    assert "WWW-Authenticate" in failed.value.headers
    assert native_request_admission.state.authorizations == []
    assert (
        await native_request_admission.backend.fetch_all("SELECT * FROM a2a_contexts")
        == []
    )
    assert (
        await native_request_admission.backend.fetch_all("SELECT id FROM a2a_tasks")
        == []
    )


@pytest.mark.parametrize("stage", ["before", "verification", "authorization"])
async def test_retired_policy_cannot_commit_after_any_await(
    native_request_admission, stage
):
    params = send(native_request_admission)
    context = signed_context(native_request_admission, params)
    guard = native_request_admission.guard
    if stage == "before":
        native_request_admission.state.current = False
    elif stage == "verification":

        async def resolve(did):
            native_request_admission.state.current = False
            return native_request_admission.signing[1]

        authority = replace(guard.authority_factory(), resolver=resolve)
        guard = replace(guard, authority_factory=lambda: authority)
    else:

        async def authorize(did):
            native_request_admission.state.current = False
            return PRINCIPAL

        authority = replace(guard.authority_factory(), authorize_sender=authorize)
        guard = replace(guard, authority_factory=lambda: authority)
    with pytest.raises(HTTPException) as failed:
        async with guard.admit(params, "SendMessage", context):
            pytest.fail("Retired policy reached native_request_admission")
    assert failed.value.status_code == 403
    assert (
        await native_request_admission.backend.fetch_all("SELECT * FROM a2a_contexts")
        == []
    )


async def test_recipient_policy_denial_is_not_cryptographic_authority(
    native_request_admission,
):
    await native_request_admission.backend.execute("DELETE FROM native_test_principals")
    params = send(native_request_admission)
    with pytest.raises(HTTPException) as failed:
        async with native_request_admission.guard.admit(
            params, "SendMessage", signed_context(native_request_admission, params)
        ):
            pytest.fail("A valid signature bypassed recipient policy")
    assert failed.value.status_code == 403
    assert native_request_admission.state.authorizations == [SENDER]
    assert (
        await native_request_admission.backend.fetch_all("SELECT id FROM a2a_tasks")
        == []
    )


async def test_required_extensions_fail_before_nonce_reservation(
    native_request_admission,
):
    params = send(native_request_admission)
    with pytest.raises(ExtensionSupportRequiredError):
        async with native_request_admission.guard.admit(
            params,
            "SendMessage",
            signed_context(native_request_admission, params, activated=frozenset()),
        ):
            pytest.fail("Missing required extension reached native_request_admission")
    assert native_request_admission.state.authorizations == []
    assert (
        await native_request_admission.backend.fetch_all(
            "SELECT nonce FROM a2a_replay_nonces"
        )
        == []
    )


async def test_execution_control_propagates_without_becoming_an_authentication_error(
    native_request_admission,
):
    error = ExecutionAuthorityError("Policy native_request_admission retired")

    async def resolve(did):
        raise error

    authority = replace(
        native_request_admission.guard.authority_factory(), resolver=resolve
    )
    guard = replace(native_request_admission.guard, authority_factory=lambda: authority)
    params = send(native_request_admission)
    with pytest.raises(ExecutionAuthorityError) as failed:
        async with guard.admit(
            params, "SendMessage", signed_context(native_request_admission, params)
        ):
            pytest.fail("Retired execution reached native_request_admission")
    assert failed.value is error


async def test_lifecycle_writer_waits_for_the_authorized_commit_to_finish(
    native_request_admission,
):
    params = send(native_request_admission)
    context = signed_context(native_request_admission, params)
    wrote = asyncio.Event()

    async def retire():
        async with native_request_admission.lock:
            native_request_admission.state.current = False
            wrote.set()

    async with native_request_admission.guard.admit(
        params, "SendMessage", context
    ) as granted:
        writer = asyncio.create_task(retire())
        try:
            await asyncio.sleep(0)
            assert not wrote.is_set()
            accepted = await native_request_admission.manager.create_task(
                granted.request, creator_agent_id=granted.principal_id
            )
            assert not wrote.is_set()
        except BaseException:
            writer.cancel()
            raise
    await asyncio.wait_for(writer, 1)
    assert wrote.is_set()
    assert (
        await native_request_admission.manager.get_task_for_recipient(
            accepted.id, native_request_admission.scope.recipient_id
        )
        is not None
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("scope", None, TypeError),
        ("routing_key", "", ValueError),
        ("required_extensions", set(EXTENSIONS), TypeError),
        ("required_extensions", frozenset(), ValueError),
    ],
)
async def test_guard_registration_is_immutable_and_requires_its_declared_policy(
    native_request_admission, field, value, error
):
    with pytest.raises(error):
        replace(native_request_admission.guard, **{field: value})
    with pytest.raises(FrozenInstanceError):
        native_request_admission.guard.scope = NativeRecipientScope("did:kestrel:other")


@pytest.mark.parametrize(
    "field,value",
    [("recipient_id", "did:kestrel:other"), ("tenant", "different-tenant")],
)
async def test_scope_disagreement_is_rejected_before_nonce_or_policy_io(
    native_request_admission, field, value
):
    params = send(native_request_admission)
    context = signed_context(native_request_admission, params)
    guard = replace(
        native_request_admission.guard,
        scope=replace(native_request_admission.scope, **{field: value}),
    )
    with pytest.raises(InvalidParamsError, match="recipient"):
        async with guard.admit(params, "SendMessage", context):
            pytest.fail("Misrouted scope reached native_request_admission")
    assert native_request_admission.state.authorizations == []
    assert (
        await native_request_admission.backend.fetch_all(
            "SELECT nonce FROM a2a_replay_nonces"
        )
        == []
    )


async def test_request_digest_covers_changes_after_signing(native_request_admission):
    params = send(native_request_admission)
    context = signed_context(native_request_admission, params)
    params.message.parts[0].text = "changed after signing"
    with pytest.raises(HTTPException) as failed:
        async with native_request_admission.guard.admit(params, "SendMessage", context):
            pytest.fail("Changed request reached native_request_admission")
    assert failed.value.status_code == 401
    assert native_request_admission.state.authorizations == []
    assert (
        await native_request_admission.backend.fetch_all(
            "SELECT nonce FROM a2a_replay_nonces"
        )
        == []
    )


async def test_replay_storage_failure_stays_an_infrastructure_failure(
    native_request_admission,
):
    if native_request_admission.backend.backend_type == "sqlite":
        await native_request_admission.backend.execute_script("""
            CREATE TRIGGER refuse_native_test_nonce BEFORE INSERT ON a2a_replay_nonces
            BEGIN SELECT RAISE(ABORT, 'native proof storage refusal'); END;
        """)
    else:
        await native_request_admission.backend.execute(
            "ALTER TABLE a2a_replay_nonces ADD CONSTRAINT refuse_native_test_nonce CHECK (sender <> 'did:web:sender.test:agent')"
        )
    params = send(native_request_admission)
    with pytest.raises(InternalError, match="persistence is unavailable") as failed:
        async with native_request_admission.guard.admit(
            params, "SendMessage", signed_context(native_request_admission, params)
        ):
            pytest.fail("Failed replay reservation reached native_request_admission")
    assert isinstance(failed.value.__cause__, QueryError)
    assert native_request_admission.state.authorizations == []
    assert (
        await native_request_admission.backend.fetch_all(
            "SELECT nonce FROM a2a_replay_nonces"
        )
        == []
    )
    assert (
        await native_request_admission.backend.fetch_all("SELECT id FROM a2a_tasks")
        == []
    )


async def test_waiting_request_rechecks_policy_without_replaying_its_nonce(
    native_request_admission,
):
    params = send(native_request_admission)
    async with native_request_admission.guard.admit(
        params, "SendMessage", signed_context(native_request_admission, params)
    ) as admitted:
        pass
    for _ in range(2):
        async with native_request_admission.guard.hold_authority(admitted):
            assert native_request_admission.lock.locked()
    assert (
        len(
            await native_request_admission.backend.fetch_all(
                "SELECT nonce FROM a2a_replay_nonces"
            )
        )
        == 1
    )
    assert native_request_admission.state.authorizations == [SENDER] * 3
    native_request_admission.state.current = False
    with pytest.raises(HTTPException) as failed:
        async with native_request_admission.guard.hold_authority(admitted):
            pytest.fail("Retired waiting request retained authority")
    assert failed.value.status_code == 403


@pytest.mark.parametrize("change", ["principal", "recipient", "routing-key", "guard"])
async def test_waiting_request_cannot_change_its_original_authority(
    native_request_admission, change
):
    params = send(native_request_admission)
    async with native_request_admission.guard.admit(
        params, "SendMessage", signed_context(native_request_admission, params)
    ) as admitted:
        pass
    guard = native_request_admission.guard
    if change == "principal":
        await native_request_admission.backend.execute(
            "UPDATE native_test_principals SET principal = ?",
            ("did:kestrel:different",),
        )
    elif change == "recipient":
        guard = replace(
            guard, scope=NativeRecipientScope("did:kestrel:other", "tenant-alpha")
        )
    elif change == "routing-key":
        guard = replace(guard, routing_key="different-registration")
    else:
        guard = replace(guard)
    with pytest.raises(HTTPException) as failed:
        async with guard.hold_authority(admitted):
            pytest.fail("Waiting request changed authority")
    assert failed.value.status_code == 403


async def test_each_request_retains_its_own_authority_across_waits(
    native_request_admission,
):
    guard = native_request_admission.guard
    original = guard.authority_factory()
    checks = []

    def create_authority():
        calls = []
        checks.append(calls)

        def require_current(sender):
            original.require_current(sender)
            calls.append(sender)

        return replace(original, require_current=require_current)

    guard = replace(guard, authority_factory=create_authority)
    params = send(native_request_admission)
    async with guard.admit(
        params, "SendMessage", signed_context(native_request_admission, params)
    ) as first:
        pass
    async with guard.admit(
        params, "SendMessage", signed_context(native_request_admission, params)
    ) as second:
        pass
    assert first.authority is not second.authority
    assert len(checks) == 2 and [len(calls) for calls in checks] == [3, 3]
    async with guard.hold_authority(first):
        pass
    assert len(checks) == 2 and [len(calls) for calls in checks] == [5, 3]


async def test_malformed_authority_factory_fails_before_spending_a_nonce(
    native_request_admission,
):
    guard = replace(native_request_admission.guard, authority_factory=lambda: None)
    params = send(native_request_admission)
    with pytest.raises(TypeError, match="request-owned policy"):
        async with guard.admit(
            params, "SendMessage", signed_context(native_request_admission, params)
        ):
            pytest.fail("Invalid authority reached admission")
    assert (
        await native_request_admission.backend.fetch_all(
            "SELECT nonce FROM a2a_replay_nonces"
        )
        == []
    )


async def test_sender_proof_error_retains_its_execution_control_cause(
    native_request_admission,
):
    from kestrel_sovereign.a2a.native_sender import NativeSenderProofError

    control = ExecutionAuthorityError("Original native verification authority retired")
    refusal = NativeSenderProofError("Original verification refused")

    async def resolve(did):
        raise refusal from control

    authority = replace(
        native_request_admission.guard.authority_factory(), resolver=resolve
    )
    guard = replace(native_request_admission.guard, authority_factory=lambda: authority)
    params = send(native_request_admission)
    with pytest.raises(NativeSenderProofError) as failed:
        async with guard.admit(
            params, "SendMessage", signed_context(native_request_admission, params)
        ):
            pytest.fail("Retired verification reached admission")
    assert failed.value is refusal and failed.value.__cause__ is control


async def test_waiting_policy_read_failure_remains_an_internal_error(
    native_request_admission,
):
    guard = native_request_admission.guard
    params = send(native_request_admission)
    async with guard.admit(
        params, "SendMessage", signed_context(native_request_admission, params)
    ) as admitted:
        pass
    await native_request_admission.backend.execute("DROP TABLE native_test_principals")
    with pytest.raises(InternalError, match="persistence is unavailable") as failed:
        async with guard.hold_authority(admitted):
            pytest.fail("An unavailable policy store retained request authority")
    assert isinstance(failed.value.__cause__, QueryError)


async def test_persistence_boundary_preserves_execution_control(
    native_request_admission,
):
    control = ExecutionAuthorityError("Original policy execution was retired")
    failure = QueryError("Original retained database operation")

    async def authorize(did):
        raise failure from control

    original = native_request_admission.guard
    authority = replace(original.authority_factory(), authorize_sender=authorize)
    guard = replace(original, authority_factory=lambda: authority)
    params = send(native_request_admission)
    with pytest.raises(QueryError) as failed:
        async with guard.admit(
            params, "SendMessage", signed_context(native_request_admission, params)
        ):
            pytest.fail("Retired database authority admitted a request")
    assert failed.value is failure and failed.value.__cause__ is control
