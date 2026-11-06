"""Native SQL/file effects and real Agent carrier ownership, without READY boot.

The separate process-host tests prove normal boot. These exercise the shared
execution component, not model cognition or the four dispatch producers.
"""

import asyncio

from cryptography.fernet import Fernet
import pytest
import pytest_asyncio

from kestrel_sdk.resources import ResourceOutcomeUnknown
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


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    agent.privacy_agent = PrivacyAgent(agent.storage)
    store = ToolEffectStore(db, scope=NativeRecipientScope(agent.did))
    await store.initialize()
    agent.tool_effect_store = store
    try:
        yield agent
    finally:
        await store.close()
        await agent.llm_service.close()
        await db.close()


async def admit(owner, key="owned-op"):
    return await admit_tool_effect_turn(
        owner, namespace=["caller", "sovereign", "api_key", "owned-user"],
        client_key=key, request={"input": "owned-input", "session_id": None},
        invocation_id="owned-invocation",
    )


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
        leaf = asyncio.create_task(execute_tool_effect(owner, registration="file", arguments={}, execute=effect))
    await entered.wait()
    closer = asyncio.create_task(turn.close())
    await asyncio.sleep(0)
    assert not closer.done()
    assert await owner.tool_effect_store.database.fetchone("SELECT state FROM tool_effect_operations") == ("OPEN",)
    release.set()
    await leaf
    await closer
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
        with pytest.raises(ToolEffectAttemptClosed):
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
        leaf = asyncio.create_task(execute_tool_effect(owner, registration="file", arguments={}, execute=effect))
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
