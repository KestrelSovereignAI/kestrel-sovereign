"""Native configured immutable-path symlink custody at review55 boundaries."""
from unittest.mock import Mock

import asyncio
import aiosqlite

import pytest

from kestrel_sovereign.features.isolated_runtime import (
    IsolatedRuntimePreparationError,
    ProxyFeature,
)
from tests.unit.test_isolated_feature_runtime import (
    FakeIsolatedClient,
    _configure_idle_lifecycle,
    _idle_test_runtime,
    _write_prebuilt_venv_shape,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact", ["BIN", "VENV"])
@pytest.mark.parametrize("entry", ["configured", "intermediate"])
async def test_required_operator_symlink_is_never_reclaimed(monkeypatch, tmp_path, artifact, entry):
    agent = Mock(did="did:test:isolated-runtime", features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    workspace = feature._prepare_runtime_workspace()
    target = tmp_path / "operator-artifact"
    if artifact == "BIN":
        target.write_text("#!/bin/sh\nexit 0\n")
        target.chmod(0o700)
    else:
        _write_prebuilt_venv_shape(target)
    required_entry = workspace / "operator-entry"
    required_entry.symlink_to(target, target_is_directory=artifact == "VENV")
    setting = required_entry
    if entry == "intermediate":
        setting = tmp_path / "external-operator-entry"
        setting.symlink_to(required_entry, target_is_directory=artifact == "VENV")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_" + artifact, str(setting))
    feature.resolve_runtime_paths()
    feature._idle_retired = True
    feature._client = None
    assert not feature.runtime_telemetry_snapshot().cleanup_eligible
    with pytest.raises(IsolatedRuntimePreparationError, match="immutable"):
        await feature.reclaim_idle_workspace()
    assert required_entry.is_symlink() and setting.resolve() == target.resolve()
    assert workspace.is_dir()
    # The exact configured entry remains a valid cold wake artifact.
    venv, executable = feature.resolve_runtime_paths()
    assert (executable if artifact == "BIN" else venv) == target.resolve()


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["error", "cancel"])
async def test_detached_stop_native_commit_control_survives_callback_consumption(tmp_path, carrier):
    from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
    from kestrel_sovereign.stop import StopCleanupRegistry
    from tests.unit.test_execution_custody_review43_native import carries
    async with aiosqlite.connect(tmp_path / "detached.db") as ledger:
        await ledger.execute("CREATE TABLE effects(id INTEGER)")
        original = OSError("native detached Stop acknowledgement") if carrier == "error" else asyncio.CancelledError("native detached Stop acknowledgement")
        original.__cause__ = ExecutionCommitOutcomeError("unknown")
        async def native_tail():
            await ledger.execute("INSERT INTO effects VALUES (1)")
            await ledger.commit()
            raise original
        registry = StopCleanupRegistry()
        task = create_owned_task(native_tail())
        registry.retain(task)
        await await_owned_task(task)
        await asyncio.sleep(0)
        assert not registry._tasks
        with pytest.raises(BaseException) as caught:
            await registry.drain()
        assert carries(caught.value, original)
        assert await (await ledger.execute("SELECT COUNT(*) FROM effects")).fetchone() == (1,)


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["error", "cancel"])
async def test_native_stop_receipt_commit_control_cannot_be_read_back_as_success(tmp_path, monkeypatch, carrier):
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
    from kestrel_sovereign.storage.async_storage import AsyncStorage
    from kestrel_sovereign.stop import StopReceiptStore, StopDoor
    from tests.unit.test_stop_receipts import _request, _outcomes
    from tests.unit.test_execution_custody_review43_native import carries
    async with AsyncStorage(str(tmp_path / "stop.db"), agent_id="did:test:stop55") as storage:
        store = StopReceiptStore(storage.db)
        await store.ensure_schema()
        request = _request()
        connection = storage.db.backend._connection
        native_commit = aiosqlite.Connection.commit
        original = OSError("native Stop receipt commit ACK") if carrier == "error" else asyncio.CancelledError("native Stop receipt commit ACK")
        original.__cause__ = ExecutionCommitOutcomeError("unknown")
        commits = []
        async def lost_ack(target):
            await native_commit(target)
            if target is connection and not commits:
                commits.append(True)
                raise original
        monkeypatch.setattr(aiosqlite.Connection, "commit", lost_ack)
        try:
            with pytest.raises(BaseException) as caught:
                await store.persist(request, _outcomes(request), door=StopDoor.AGENT)
            assert carries(caught.value, original)
            assert commits == [True]
            assert await storage.db.fetchval("SELECT COUNT(*) FROM stop_receipts") == 1
        finally:
            monkeypatch.undo()


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["error", "cancel"])
async def test_native_ephemeral_purge_retains_control_after_independent_sweeps(tmp_path, monkeypatch, carrier):
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage.async_storage import AsyncStorage
    from kestrel_sovereign.storage.async_graph_store import GraphNode
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
    from tests.integration.test_ephemeral_hard_purge import _now_iso_utc
    from tests.unit.test_execution_custody_review43_native import carries
    async with AsyncStorage(str(tmp_path / "purge.db"), agent_id="did:test:purge55") as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        watermark = wrapper._entered_ephemeral_at
        await asyncio.sleep(1.05)
        await storage.conversation.add_conversation("user", "native ephemeral leak")
        await storage.add_node(GraphNode(node_id="leaked", node_type="memory", label="private", properties={"agent_id": storage.agent_id, "created_at": _now_iso_utc()}))
        connection = storage.db.backend._connection
        native_commit = aiosqlite.Connection.commit
        original = OSError("native purge commit ACK") if carrier == "error" else asyncio.CancelledError("native purge commit ACK")
        original.__cause__ = ExecutionCommitOutcomeError("unknown")
        commits = []
        async def lost_ack(target):
            await native_commit(target)
            if target is connection and not commits:
                commits.append(True)
                raise original
        monkeypatch.setattr(aiosqlite.Connection, "commit", lost_ack)
        try:
            with pytest.raises(BaseException) as caught:
                await wrapper.purge_ephemeral_session("review55 native receipt")
            assert carries(caught.value, original)
            assert commits == [True]
            assert wrapper._entered_ephemeral_at == watermark
            assert await storage.db.fetchval("SELECT COUNT(*) FROM conversation_history") == 0
            assert await storage.db.fetchval("SELECT COUNT(*) FROM graph_nodes WHERE node_id='leaked'") == 0
            assert await storage.db.fetchval("SELECT COUNT(*) FROM conversation_history_changes") == 0
        finally:
            monkeypatch.undo()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["message", "orphan"])
@pytest.mark.parametrize("carrier", ["error", "group"])
async def test_actual_pin_delete_commit_cannot_be_optional_table_fallback(tmp_path, monkeypatch, route, carrier):
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage.async_storage import AsyncStorage
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
    from tests.unit.test_execution_custody_review43_native import carries
    async with AsyncStorage(str(tmp_path / "pins.db"), agent_id="did:test:pins55") as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
        await storage.db.execute_commit("CREATE TABLE memory_pins(message_id INTEGER, agent_id TEXT)")
        await storage.db.execute_commit("INSERT INTO memory_pins VALUES (1, ?)", (storage.agent_id,))
        original = OSError("native pin-delete acknowledgement")
        original.__cause__ = ExecutionCommitOutcomeError("unknown")
        if carrier == "group":
            original = ExceptionGroup("native pin deletion", [ValueError("ordinary"), original])
        connection = storage.db.backend._connection
        native_commit = aiosqlite.Connection.commit
        async def lost_ack(target):
            await native_commit(target)
            if target is connection:
                raise original
        monkeypatch.setattr(aiosqlite.Connection, "commit", lost_ack)
        try:
            with pytest.raises(BaseException) as caught:
                if route == "message":
                    await wrapper._delete_pin_for_message(1, storage.agent_id)
                else:
                    await wrapper._delete_orphaned_pins(storage.agent_id)
            assert carries(caught.value, original)
            assert await storage.db.fetchval("SELECT COUNT(*) FROM memory_pins") == 0
        finally:
            monkeypatch.undo()
