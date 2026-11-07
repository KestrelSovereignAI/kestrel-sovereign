"""Native SQL/file effects and real Agent carrier ownership, without READY boot.

The separate process-host tests prove normal boot. These exercise the shared
execution component, not model cognition or the four dispatch producers.
"""

import asyncio

from cryptography.fernet import Fernet
import pytest
import pytest_asyncio

from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sovereign._async_ownership import create_owned_task
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.features.privacy.feature import PrivacyAgent
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.tool_effect_store import (
    ToolEffectAttemptClosed, ToolEffectReconciliationRequired, ToolEffectStore,
)
from kestrel_sovereign.tools.effect_journal import (
    ToolEffectJournalUnavailable, ToolEffectOutcomeUnconfirmed,
    ToolEffectPrivacyRefused, admit_tool_effect_turn, bind_tool_effect_turn,
    execute_tool_effect, logical_operation_key, require_persistent_tool_effects,
)
from kestrel_sovereign.turn_scope import capture_turn_scope
from tests.unit.test_cleanup_task_admission import _partial_native_agent
from kestrel_sovereign.testing.fixtures import native_tool_effect_agent as native_tool_effect_agent


pytestmark = pytest.mark.timeout(30)


@pytest_asyncio.fixture
async def owner(native_tool_effect_agent):
    yield await native_tool_effect_agent(did='did:test:unpublished-cleanup')


async def admit(owner, key="owned-op"):
    return await admit_tool_effect_turn(
        owner, namespace=["caller", "sovereign", "api_key", "owned-user"],
        client_key=key, request={"input": "owned-input", "session_id": None},
        invocation_id="owned-invocation",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('unavailable', ['policy', 'retired', 'foreign'])
async def test_journal_initialization_cannot_replace_an_unavailable_original(owner, tmp_path, unavailable):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    original = owner.tool_effect_store
    foreign_db = foreign = None
    try:
        if unavailable == 'policy':
            owner.privacy_agent = None
            message = 'initialized privacy policy'
        else:
            message = 'Original tool effect journal is unavailable'
            if unavailable == 'retired':
                await original.close()
            else:
                foreign_db = await AsyncDatabase.sqlite(str(tmp_path / 'foreign-journal.db'))
                foreign = ToolEffectStore(foreign_db, scope=NativeRecipientScope(owner.did))
                await foreign.initialize()
                owner.tool_effect_store = foreign
        retained = owner.tool_effect_store
        with pytest.raises(RuntimeError, match=message):
            await owner._initialize_tool_effect_store()
        assert owner.tool_effect_store is retained
        assert await original.database.fetchone('SELECT COUNT(*) FROM tool_effect_keys') == (1,)
    finally:
        owner.tool_effect_store = original
        if foreign is not None:
            await foreign.close()
        if foreign_db is not None:
            await foreign_db.close()


@pytest.mark.asyncio
async def test_claim_is_acknowledged_before_file_effect_and_replay_is_fenced(owner, tmp_path):
    turn = await admit(owner)
    path = tmp_path / "effect"

    async def effect():
        assert await owner.tool_effect_store.database.fetchone(
            "SELECT state FROM tool_effect_claims"
        ) == ("STARTED",)
        assert owner.tool_effect_store.database.owns_open_transaction is False
        path.write_text("one effect")
        return {"success": True}

    with bind_tool_effect_turn((turn, None)):
        assert await execute_tool_effect(owner, registration="file", arguments={}, execute=effect) == {"success": True}
    await turn.close()
    with pytest.raises(ToolEffectReconciliationRequired):
        await admit(owner)
    assert path.read_text() == "one effect"
    assert await owner.tool_effect_store.database.fetchone("SELECT state FROM tool_effect_claims") == ("RETURNED_OK",)


@pytest.mark.asyncio
async def test_failed_unknown_settlement_still_fences_new_effects_in_the_live_turn(owner, tmp_path):
    from kestrel_sovereign.execution_custody import execution_evidence_carriers

    turn = await admit(owner)
    db = owner.tool_effect_store.database
    witness = tmp_path / 'live-turn-effects'
    original = ResourceOutcomeUnknown('Original reply lost after owned effect')

    async def first_effect():
        witness.write_text('first effect')
        raise original

    async def must_not_execute():
        witness.write_text('second effect incorrectly executed')
        return {'success': True}

    await db.execute("CREATE TRIGGER reject_effect_settlement BEFORE UPDATE ON tool_effect_claims BEGIN SELECT RAISE(ABORT, 'owned settlement refusal'); END")
    try:
        with bind_tool_effect_turn((turn, None)):
            with pytest.raises((ResourceOutcomeUnknown, BaseExceptionGroup)) as failed:
                await execute_tool_effect(owner, registration='first', arguments={}, execute=first_effect)
            assert any(error is original for error in execution_evidence_carriers(failed.value))
            assert await db.fetchone('SELECT state,latched FROM tool_effect_operations') == ('OPEN', 0)
            assert await db.fetchone('SELECT state FROM tool_effect_claims') == ('STARTED',)
            await db.execute('DROP TRIGGER reject_effect_settlement')
            with pytest.raises(ToolEffectOutcomeUnconfirmed):
                await execute_tool_effect(owner, registration='second', arguments={}, execute=must_not_execute)
        assert witness.read_text() == 'first effect'
        assert await db.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (1,)
    finally:
        with pytest.raises(BaseExceptionGroup) as closed:
            await turn.close()
        # The DDL changed the admitted catalog generation. Terminal custody
        # refuses that destination and must retain the earlier lost reply.
        from kestrel_sovereign.execution_custody import is_execution_control_error
        assert is_execution_control_error(closed.value)
        assert any(error is original for error in execution_evidence_carriers(closed.value))
        assert await db.fetchone('SELECT state,latched FROM tool_effect_operations') == ('OPEN', 0)


@pytest.mark.asyncio
async def test_known_resource_precondition_is_returned_error_but_still_fences_replay(owner):
    turn = await admit(owner)
    original = ResourceConflict('Original resource revision changed')

    async def refuse():
        raise original

    with bind_tool_effect_turn((turn, None)):
        with pytest.raises(ResourceConflict) as observed:
            await execute_tool_effect(owner, registration='owned-revision', arguments={}, execute=refuse)
    assert observed.value is original
    await turn.close()
    assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_claims') == ('RETURNED_ERROR',)
    with pytest.raises(ToolEffectReconciliationRequired):
        await admit(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal_kind', ['unknown', 'cancelled', 'mixed-group'])
async def test_root_close_preserves_an_original_terminal_error_with_uncertain_effects(owner, tmp_path, terminal_kind):
    turn = await admit(owner)
    original_unknown = ResourceOutcomeUnknown('Original uncertain effect')
    witness = tmp_path / 'uncertain-original'

    async def effect():
        witness.write_text('one effect')
        raise original_unknown

    with bind_tool_effect_turn((turn, None)):
        with pytest.raises(ResourceOutcomeUnknown):
            await execute_tool_effect(owner, registration='owned-file', arguments={}, execute=effect)
    terminal = (original_unknown if terminal_kind == 'unknown' else
        asyncio.CancelledError('original caller withdrawal'))
    if terminal_kind == 'mixed-group':
        terminal = BaseExceptionGroup('original control group', [terminal, ValueError('original closer failure')])
    with pytest.raises(BaseException) as observed:
        await turn.close(terminal)
    assert observed.value is terminal
    assert witness.read_text() == 'one effect'
    assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_claims') == ('UNCONFIRMED',)


@pytest.mark.asyncio
async def test_registration_withdrawal_after_claim_ack_cannot_start_an_effect(owner, tmp_path, monkeypatch):
    from kestrel_sovereign.agent.tool_allowlist import bump_registry_generation

    turn = await admit(owner)
    original = owner.tool_effect_store.claim
    witness = tmp_path / 'must-not-execute'

    async def withdraw_after_ack(*args, **kwargs):
        claim = await original(*args, **kwargs)
        assert owner.tool_effect_store.database.owns_open_transaction is False
        bump_registry_generation(owner)
        return claim

    async def effect():
        witness.write_text('unpermitted effect')

    monkeypatch.setattr(owner.tool_effect_store, 'claim', withdraw_after_ack)
    with bind_tool_effect_turn((turn, None)):
        with pytest.raises(ResourceConflict, match='registration changed'):
            await execute_tool_effect(owner, registration='original-file-tool', arguments={}, execute=effect)
    await turn.close()
    assert not witness.exists()
    assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_claims') == ('NOT_EXECUTED',)
    assert (await admit(owner)).attempt.attempt_id != turn.attempt.attempt_id


@pytest.mark.asyncio
async def test_registration_validator_cannot_publish_an_unawaited_guard(owner, tmp_path):
    turn = await admit(owner)
    witness = tmp_path / 'must-not-execute'

    async def invalid_validation():
        pytest.fail('An asynchronous validation must not execute after admission')

    async def effect():
        witness.write_text('unpermitted effect')

    with bind_tool_effect_turn((turn, None)):
        with pytest.raises(TypeError, match='must complete synchronously'):
            await execute_tool_effect(owner, registration='original-file-tool', arguments={},
                execute=effect, validate_registration=invalid_validation)
    await turn.close()
    assert not witness.exists()
    assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_claims') == ('NOT_EXECUTED',)


@pytest.mark.asyncio
async def test_copied_turn_scope_is_revoked_after_root_exit(owner, tmp_path):
    turn = await admit(owner)
    with bind_tool_effect_turn((turn, None)):
        snapshot = capture_turn_scope(owner)
    await turn.close()

    async def effect():
        (tmp_path / "forbidden").write_text("effect")

    with snapshot.bind(), pytest.raises(ToolEffectJournalUnavailable, match="retired"):
        await execute_tool_effect(owner, registration="file", arguments={}, execute=effect)
    assert not (tmp_path / "forbidden").exists()
    assert await owner.tool_effect_store.database.fetchone("SELECT COUNT(*) FROM tool_effect_claims") == (0,)


@pytest.mark.asyncio
async def test_root_drain_waits_for_original_claimant_before_closing_attempt(owner, tmp_path):
    turn = await admit(owner)
    entered, release = asyncio.Event(), asyncio.Event()

    async def effect():
        entered.set()
        await release.wait()
        (tmp_path / "effect").write_text("one")

    with bind_tool_effect_turn((turn, None)):
        leaf = create_owned_task(execute_tool_effect(owner, registration="file", arguments={}, execute=effect))
    await entered.wait()
    closer = asyncio.create_task(turn.close())
    try:
        await asyncio.sleep(0)
        assert not closer.done()
        assert await owner.tool_effect_store.database.fetchone("SELECT state FROM tool_effect_operations") == ("OPEN",)
    finally:
        release.set()
        await asyncio.gather(leaf, closer)
    assert await owner.tool_effect_store.database.fetchone("SELECT state FROM tool_effect_operations") == ("CLOSED",)
    assert await owner.tool_effect_store.database.fetchone("SELECT state FROM tool_effect_claims") == ("RETURNED_OK",)


@pytest.mark.asyncio
async def test_lost_result_latches_attempt_and_root_refuses_success(owner, tmp_path):
    turn = await admit(owner)
    original = ResourceOutcomeUnknown("owned lost result")

    async def effect():
        (tmp_path / "effect").write_text("one")
        raise original

    with bind_tool_effect_turn((turn, None)):
        with pytest.raises(ResourceOutcomeUnknown) as error:
            await execute_tool_effect(owner, registration="file", arguments={}, execute=effect)
        assert error.value is original
        with pytest.raises(ToolEffectOutcomeUnconfirmed):
            await execute_tool_effect(owner, registration="file", arguments={}, execute=effect)
    with pytest.raises(ToolEffectOutcomeUnconfirmed, match="tool_effect_outcome_unconfirmed"):
        await turn.close()
    assert (tmp_path / "effect").read_text() == "one"
    assert await owner.tool_effect_store.database.fetchone("SELECT state FROM tool_effect_claims") == ("UNCONFIRMED",)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED])
async def test_volatile_privacy_has_no_admission_rows_and_refuses_effects(owner, mode):
    owner.privacy_agent = PrivacyAgent(owner.storage, mode)
    assert await admit(owner) is None
    with pytest.raises(ToolEffectPrivacyRefused):
        require_persistent_tool_effects(owner)
    assert await owner.tool_effect_store.database.fetchone("SELECT COUNT(*) FROM tool_effect_operations") == (0,)


@pytest.mark.asyncio
async def test_privacy_writer_cannot_pass_claim_through_settlement(owner):
    turn = await admit(owner)
    entered, release, writer_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def effect():
        entered.set()
        await release.wait()

    async def writer():
        async with owner._get_durable_persistence_gate().exclusive():
            assert await owner.tool_effect_store.database.fetchone("SELECT state FROM tool_effect_claims") == ("RETURNED_OK",)
            writer_entered.set()

    with bind_tool_effect_turn((turn, None)):
        leaf = create_owned_task(execute_tool_effect(owner, registration="file", arguments={}, execute=effect))
    await entered.wait()
    transition = asyncio.create_task(writer())
    await asyncio.sleep(0)
    assert not writer_entered.is_set()
    release.set()
    await leaf
    await transition
    await turn.close()


@pytest.mark.asyncio
async def test_root_exit_after_privacy_flip_writes_no_journal_metadata(owner):
    turn = await admit(owner)
    async with owner._get_durable_persistence_gate().exclusive():
        owner.privacy_agent = PrivacyAgent(owner.storage, PrivacyMode.EPHEMERAL)
    await turn.close()
    assert turn.live is False
    # The obsolete generation is revoked in memory. Its untouched OPEN row
    # can only be superseded by a new durable admission after leaving privacy.
    assert await owner.tool_effect_store.database.fetchone("SELECT state FROM tool_effect_operations") == ("OPEN",)


@pytest.mark.asyncio
async def test_returned_business_failure_is_an_effect_capable_terminal(owner):
    turn = await admit(owner)

    async def effect():
        return {"success": False, "error": "owned refusal after execution"}

    with bind_tool_effect_turn((turn, None)):
        await execute_tool_effect(owner, registration="file", arguments={}, execute=effect)
    await turn.close()
    assert await owner.tool_effect_store.database.fetchone("SELECT state FROM tool_effect_claims") == ("RETURNED_ERROR",)
    with pytest.raises(ToolEffectReconciliationRequired):
        await admit(owner)


@pytest.mark.parametrize("key", ["雪/opaque%25", ["a2a-task", 2], "x" * 256])
def test_wire_identifiers_encode_to_bounded_stable_store_keys(key):
    encoded = logical_operation_key(key)
    assert len(encoded) == 64 and encoded.isascii()
    assert encoded == logical_operation_key(key)
    assert encoded != logical_operation_key([key])


def test_noncanonical_key_is_refused_before_admission():
    with pytest.raises(ValueError, match="canonical JSON"):
        logical_operation_key(float("nan"))


@pytest.mark.asyncio
async def test_copied_journal_context_does_not_admit_an_execution_task(owner, tmp_path):
    import contextvars
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError

    turn = await admit(owner, key='copied-journal-context')
    witness = tmp_path / 'unadmitted-effect'

    async def effect():
        witness.write_text('unadmitted effect')

    with bind_tool_effect_turn((turn, None)):
        copied = contextvars.copy_context()
        task = asyncio.create_task(execute_tool_effect(owner, registration='file',
            arguments={}, execute=effect), context=copied)
    with pytest.raises(ExecutionAuthorityError, match='original native privacy lifetime'):
        await task
    await turn.close()
    assert not witness.exists()
    assert await owner.tool_effect_store.database.fetchone(
        'SELECT COUNT(*) FROM tool_effect_claims') == (0,)


@pytest.mark.asyncio
async def test_native_fixture_refuses_two_primary_database_owners(owner, native_tool_effect_agent):
    database = owner.tool_effect_store.database
    before = await database.fetchone('SELECT COUNT(*) FROM tool_effect_keys')
    with pytest.raises(ValueError, match='one primary database owner'):
        await native_tool_effect_agent(did='did:test:ambiguous-primary',
            connected_backend=database.backend, borrowed_database=database)
    assert database.backend.is_connected
    assert await database.fetchone('SELECT COUNT(*) FROM tool_effect_keys') == before


@pytest.mark.asyncio
async def test_prepared_native_registration_keeps_its_original_generation(owner, tmp_path):
    from kestrel_sovereign.agent.tool_allowlist import bump_registry_generation, registry_generation
    from kestrel_sovereign.tools.effect_journal import bind_tool_effect_registration
    from tests.utils.native_tool_effects import drive_native

    witness = tmp_path / 'stale-prepared-generation-effect'
    selected_generation = registry_generation(owner)
    async def effect():
        witness.write_text('stale prepared generation executed')
        return {'success': True}

    async def operation():
        with bind_tool_effect_registration(owner, selected_generation, lambda: None):
            bump_registry_generation(owner)
            try:
                await execute_tool_effect(owner, registration='prepared-file', arguments={}, execute=effect)
            except ResourceConflict:
                pass

    await drive_native(owner, operation)
    assert not witness.exists(), 'Prepared native registration refreshed its original generation'
    assert await owner.tool_effect_store.database.fetchone(
        'SELECT state FROM tool_effect_claims') == ('NOT_EXECUTED',)
