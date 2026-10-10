"""Native regression witnesses for the independent publication review."""

from copy import deepcopy
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_constitution_round9_custody import _isolated_schema
from tests.integration.test_constitution_refusal_races import _agent
from kestrel_sovereign.agent.constitution import ConstitutionMixin


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("reject_restriction", [False, True])
async def test_preliminary_exit_reports_replacement_commit_failure(db_backend, reject_restriction):
    async with _isolated_schema(db_backend) as backend:
        storage = AsyncStorage(backend=backend, agent_id="did:test:early-exit:" + uuid4().hex)
        await storage.initialize()
        trigger = "core_early_refusal_" + uuid4().hex
        pg = backend.backend_type == "postgres"
        installed = False
        try:
            agent = await _agent(storage)
            digest = await storage.store_file(resolve_governing_constitution_bytes(None), "constitution.md")
            await storage.add_node(GraphNode(node_id=agent.agent_id, node_type="agent", label="early refusal", properties={"constitution_hash": digest}))
            await agent._anchor_constitution_governance(digest)
            assert await agent.enter_safe_mode("prior restriction")
            before = await agent._constitution_state_store.load(agent.agent_id)
            events = await agent._constitution_state_store.list_events(agent.agent_id)
            await storage.delete_edge(agent.agent_id, digest, "governed_by")
            if reject_restriction:
                if pg:
                    await backend.execute_script(f"""
                        CREATE FUNCTION {trigger}() RETURNS trigger AS $f$
                        BEGIN RAISE EXCEPTION 'fixture replacement commit refused'; END; $f$ LANGUAGE plpgsql;
                        CREATE TRIGGER {trigger} BEFORE INSERT ON constitution_runtime_events
                        FOR EACH ROW EXECUTE FUNCTION {trigger}();
                    """)
                else:
                    await backend.execute_script(f"""
                        CREATE TRIGGER {trigger} BEFORE INSERT ON constitution_runtime_events
                        BEGIN SELECT RAISE(ABORT, 'fixture replacement commit refused'); END;
                    """)
                installed = True
            result = await agent.exit_safe_mode(authorization="explicit fixture owner")
            assert "integrity verification failed" in result and "governed_by" in result, result
            assert agent._safe_mode is True
            if reject_restriction:
                assert "replacement restriction could not be persisted" in result, result
                assert agent._constitution_state_persistence_pending is True
                assert await agent._constitution_state_store.load(agent.agent_id) == before
                assert await agent._constitution_state_store.list_events(agent.agent_id) == events
            else:
                assert "could not be persisted" not in result, result
                current = await agent._constitution_state_store.load(agent.agent_id)
                assert current.safe_mode is True and current.revision == before.revision + 1
                assert "governed_by" in current.safe_mode_reason
                assert (await agent._constitution_state_store.list_events(agent.agent_id))[:-1] == events
        finally:
            if installed:
                await backend.execute(f"DROP TRIGGER {trigger}" + (" ON constitution_runtime_events" if pg else ""))
                if pg:
                    await backend.execute(f"DROP FUNCTION {trigger}()")
            await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("current_kind", ["pending", "passed", "absent"])
@pytest.mark.parametrize("entry_point", ["explicit", "readiness"])
async def test_runtime_reconciles_failed_historical_genesis_without_reroll(db_backend, current_kind, entry_point):
    from kestrel_sovereign.constitution.genesis_audit import (
        GenesisAuditError, GenesisAuditRejectedError, evaluate_genesis_constitution,
        pending_genesis_audit,
    )
    from kestrel_sovereign.features.privacy.feature import PrivacyAgent
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    storage = AsyncStorage(backend=db_backend, agent_id="did:test:historical-genesis:" + uuid4().hex)
    await storage.initialize()
    try:
        agent = await _agent(storage)
        for name in ("_persist_governance_receipt_node", "_persist_genesis_audit_completion", "_persist_genesis_audit_pending_attempt", "perform_genesis_audit"):
            setattr(agent, name, getattr(ConstitutionMixin, name).__get__(agent))
        agent.privacy_agent = PrivacyAgent(PrivacyEnforcingStorage(storage, "normal"), "normal")
        content = resolve_governing_constitution_bytes(None)
        digest = await storage.store_file(content, "constitution.md")

        async def failed_auditor(prompt):
            return {"risk_level": 3, "reasoning": "Preserved synthetic historical rejection"}

        with pytest.raises(GenesisAuditRejectedError) as rejected:
            await evaluate_genesis_constitution(content, constitution_hash=digest, auditor=failed_auditor, provenance="test:historical-rejection")
        failed = rejected.value.record
        history = [{"receipt": deepcopy(failed), "superseded_at": "2026-10-09T00:00:00Z", "superseded_by_constitution_hash": "b" * 64, "provenance": "test:previous-signed-repair"}]
        properties = {"constitution_hash": digest, "genesis_audit_history": history}
        if current_kind == "pending":
            properties["genesis_audit"] = pending_genesis_audit(digest, provenance="test:old-return-repair")
        elif current_kind == "passed":
            properties["genesis_audit"] = dict(failed, status="passed", risk_level=1)
        await storage.add_node(GraphNode(node_id=agent.agent_id, node_type="agent", label="historical", properties=properties))
        await agent._anchor_constitution_governance(digest)
        before = deepcopy((await storage.get_node(agent.agent_id)).properties)
        calls = []

        async def must_not_reroll(prompt):
            calls.append(prompt)
            return {"risk_level": 1, "reasoning": "Forbidden reroll"}

        agent.get_audit_response = must_not_reroll
        method = (ConstitutionMixin.perform_genesis_audit if entry_point == "explicit" else ConstitutionMixin._ensure_genesis_audit_ready)
        with pytest.raises(GenesisAuditError):
            await method(agent)
        assert calls == [], "Historical completed rejection must precede provider admission"
        after = (await storage.get_node(agent.agent_id)).properties
        if current_kind == "passed":
            assert after == before, "Contradictory terminals must refuse without mutation"
        else:
            assert after["genesis_audit"] == failed
            assert after["genesis_audit_history"][:len(history)] == history
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("encrypted", [False, True])
async def test_avatar_refuses_corrupt_owned_physical_winner(db_backend, monkeypatch, encrypted):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!" if encrypted else "")
    storage = AsyncStorage(backend=db_backend, agent_id="did:test:avatar-winner:" + uuid4().hex)
    await storage.initialize()
    try:
        await storage.add_node(GraphNode(node_id=storage.agent_id, node_type="agent", label="avatar owner", properties={"genesis_audit": {"status": "failed"}}))
        image = b"synthetic avatar image bytes"
        digest = await storage.store_file(image, "owned prior image")
        await storage.db.execute_commit("UPDATE files SET content=? WHERE content_hash=?", (b"corrupt physical winner", digest))
        before_root = deepcopy((await storage.get_node(storage.agent_id)).properties)
        before_owner = await storage.db.fetchone("SELECT original_name,metadata FROM file_owners WHERE content_hash=? AND agent_id=?", (digest, storage.agent_id))
        refusal = None
        # Catch inside the REAL outer transaction and COMMIT, proving the
        # isolated avatar publication rolls back provisional references too.
        async with storage.transaction():
            try:
                await storage.files.store_avatar(image, storage.agent_id, "primary")
            except Exception as exc:
                refusal = exc
        assert refusal is not None, "Input digest is not proof of the actual durable winner"
        assert (await storage.get_node(storage.agent_id)).properties == before_root
        assert await storage.db.fetchone("SELECT original_name,metadata FROM file_owners WHERE content_hash=? AND agent_id=?", (digest, storage.agent_id)) == before_owner
        assert await storage.db.fetchall("SELECT node_id FROM graph_nodes WHERE node_type='avatar' AND node_id=?", (storage.files._avatar_node_id(storage.agent_id, "primary", digest),)) == []
        assert await storage.db.fetchall("SELECT target_id FROM graph_edges WHERE source_id=? AND label='has_avatar'", (storage.agent_id,)) == []
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("deleted_root", [False, True])
async def test_inception_cannot_replace_existing_or_consumed_deterministic_identity(db_backend, tmp_path, monkeypatch, deleted_root):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    monkeypatch.setenv("KESTREL_AUDIT_MODE", "skip")
    storage = AsyncStorage(backend=db_backend, agent_id="did:test:unused-fixture-owner:" + uuid4().hex)
    await storage.initialize()
    try:
        slug = "duplicate-" + uuid4().hex
        kwargs = dict(database=storage.db, is_test_instance=True, identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test", did_web_slug=slug, constitution_path=None)
        first = await create_kestrel_identity_async(output_dir=str(tmp_path / "first"), agent_name="First", **kwargs)
        root = await storage.db.fetchone("SELECT properties FROM graph_nodes WHERE node_id=?", (first.agent_did,))
        events = []
        if deleted_root:
            native = AsyncStorage(backend=db_backend, agent_id=first.agent_did)
            await native.initialize()
            agent = await _agent(native)
            assert await agent.enter_safe_mode("consumed identity lifetime")
            events = await agent._constitution_state_store.list_events(first.agent_did)
            await storage.db.execute_commit("DELETE FROM graph_nodes WHERE node_id=?", (first.agent_did,))
        refusal = None
        try:
            await create_kestrel_identity_async(output_dir=str(tmp_path / "second"), agent_name="Replacement", **kwargs)
        except Exception as exc:
            refusal = exc
        assert refusal is not None, "New key material must not recreate an existing deterministic DID"
        assert await storage.db.fetchone("SELECT properties FROM graph_nodes WHERE node_id=?", (first.agent_did,)) == (None if deleted_root else root)
        assert list((tmp_path / "first").glob("*")), "Never erase the original identity's keys"
        if deleted_root:
            assert await agent._constitution_state_store.list_events(first.agent_did) == events
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_stale_bootstrap_metadata_preserves_completed_governance(db_backend, tmp_path):
    from kestrel_sovereign.bootstrap.service import BootstrapService

    storage = AsyncStorage(backend=db_backend, agent_id="did:test:stale-bootstrap:" + uuid4().hex)
    await storage.initialize()
    try:
        await storage.add_node(GraphNode(node_id=storage.agent_id, node_type="agent", label="bootstrap", properties={"genesis_audit": {"status": "pending"}}))
        stale = await storage.get_node(storage.agent_id)
        committed = deepcopy(stale)
        committed.properties = {"genesis_audit": {"status": "failed", "reasoning": "Durable rejection"}, "genesis_audit_history": [{"receipt": {"status": "failed"}}], "constitution_hash": "a" * 64, "constitution_reanchor": {"artifact_hash": "b" * 64}}
        await storage.add_node(committed)
        before = deepcopy(committed.properties)
        service = BootstrapService(storage.db, storage.agent_id, "Bootstrap", None, tmp_path, storage=storage)
        await service.mark_stale_bootstrap(agent_node=stale, storage=storage)
        fresh = await storage.get_node(storage.agent_id)
        assert fresh.properties["bootstrap_status"] == service.STALE_BOOTSTRAP_STATUS
        assert {key: fresh.properties[key] for key in before} == before
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("writer", ["description", "rename", "bootstrap", "runtime-overlay", "offline-overlay"])
async def test_metadata_producers_serialize_newer_governance(db_backend, tmp_path, monkeypatch, writer):
    from kestrel_sovereign.bootstrap.service import BootstrapService, persist_agent_description
    from kestrel_sovereign.features.bootstrap.feature import rename_agent_core
    from kestrel_sovereign.setup.overlay_anchor import _anchor_overlay_in

    storage = AsyncStorage(backend=db_backend, agent_id="did:test:metadata-publisher:" + uuid4().hex)
    await storage.initialize()
    try:
        initial = GraphNode(node_id=storage.agent_id, node_type="agent", label="Initial", properties={"name": "Initial", "genesis_audit": {"status": "pending"}})
        await storage.add_node(initial)
        native_read = storage.graph.get_node
        published = False
        terminal = {"status": "failed", "constitution_hash": "a" * 64, "audited": True, "risk_level": 3, "reasoning": "Newer durable rejection", "completed_at": "2026-10-09T00:00:00Z"}
        history = [{"receipt": terminal, "provenance": "test:signed-repair"}]

        async def publish_newer():
            nonlocal published
            # Native publication uses the same owning graph lock domain. An
            # unlocked old read can be followed by this committed writer; a
            # locked producer instead serializes it after its own write.
            async with storage.transaction():
                await storage.lock_nodes_for_update([storage.agent_id])
                fresh = await native_read(storage.agent_id)
                fresh.properties.update(genesis_audit=deepcopy(terminal), genesis_audit_history=deepcopy(history), constitution_hash="a" * 64)
                await storage.add_node(fresh)
            published = True

        async def read_with_intervening_native_publication(identity):
            fresh = await native_read(identity)
            if not published and not storage.owns_open_transaction:
                await publish_newer()
            return fresh

        monkeypatch.setattr(storage.graph, "get_node", read_with_intervening_native_publication)
        if writer == "description":
            await persist_agent_description(storage.db, storage, storage.agent_id, "Updated bio")
        elif writer == "rename":
            agent = SimpleNamespace(agent_id=storage.agent_id, storage=storage, _raw_storage=storage, _agent_name="Initial", bootstrap_service=None, privacy_mode="normal")
            result = await rename_agent_core(agent, "Updated name")
            assert result.success, result
        elif writer == "bootstrap":
            service = BootstrapService(storage.db, storage.agent_id, "Initial", None, tmp_path, storage=storage)
            await service.mark_stale_bootstrap(storage=storage)
        elif writer == "runtime-overlay":
            agent = SimpleNamespace(agent_id=storage.agent_id, storage=storage, _constitution_overlay_sha="b" * 64, OVERLAY_HASH_PROPERTY=ConstitutionMixin.OVERLAY_HASH_PROPERTY)
            assert (await ConstitutionMixin.anchor_constitution_overlay(agent))[0]
        else:
            @asynccontextmanager
            async def open_storage():
                yield storage

            target = SimpleNamespace(agent_did=storage.agent_id, backend=db_backend.backend_type, open_storage=open_storage, describe=lambda: "session-owned native metadata fixture")
            result = await _anchor_overlay_in(target, agent_name="Initial", overlay_path=tmp_path / "CONSTITUTION.md", new_hash="b" * 64)
            assert result.error is None, result
        if not published:
            await publish_newer()
        fresh = await native_read(storage.agent_id)
        assert fresh.properties["genesis_audit"] == terminal
        assert fresh.properties["genesis_audit_history"] == history
        assert fresh.properties["constitution_hash"] == "a" * 64
        if writer == "description":
            assert fresh.properties["description"] == "Updated bio"
        elif writer == "rename":
            assert fresh.label == fresh.properties["name"] == "Updated name"
        elif writer == "bootstrap":
            assert fresh.properties["bootstrap_status"] == service.STALE_BOOTSTRAP_STATUS
        else:
            assert fresh.properties["constitution_overlay_hash"] == "b" * 64
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("change", ["historical-pass", "contradictory-admission", "contradictory-after-admission"])
async def test_turn_history_custody_preserves_pass_and_refuses_conflicts(db_backend, change):
    from tests.integration.test_constitution_turn_admission import _ready_turn
    from kestrel_sovereign.constitution.genesis_audit import pending_genesis_audit

    storage = AsyncStorage(backend=db_backend, agent_id="did:test:turn-history:" + uuid4().hex)
    await storage.initialize()
    try:
        agent, _ = await _ready_turn(storage)
        root = await storage.get_node(storage.agent_id)
        terminal = deepcopy(root.properties["genesis_audit"])
        calls = []

        async def must_not_audit(prompt):
            calls.append(prompt)
            pytest.fail("A terminal historical receipt must not call an auditor")

        agent.get_audit_response = must_not_audit
        agent._persist_governance_receipt_node = ConstitutionMixin._persist_governance_receipt_node.__get__(agent)
        if change == "historical-pass":
            root.properties["genesis_audit_history"] = [{"receipt": terminal}]
            root.properties["genesis_audit"] = pending_genesis_audit(root.properties["constitution_hash"], provenance="test:return-to-passed-content")
            await storage.add_node(root)
            assert await ConstitutionMixin._ensure_genesis_audit_ready(agent)
            assert (await storage.get_node(storage.agent_id)).properties["genesis_audit"] == terminal
        async with agent._turn_lifecycle():
            if change == "contradictory-admission":
                root.properties["genesis_audit_history"] = [{"receipt": dict(terminal, status="failed", risk_level=3)}]
                await storage.add_node(root)
            result = await ConstitutionMixin._genesis_audit_cognition_block(agent, "ordinary turn")
            if change == "contradictory-admission":
                assert result is not None and "BLOCKED" in result, result
            else:
                assert result is None, result
                if change == "contradictory-after-admission":
                    root = await storage.get_node(storage.agent_id)
                    root.properties["genesis_audit_history"] = [{"receipt": dict(terminal, status="failed", risk_level=3)}]
                    await storage.add_node(root)
                    result = await ConstitutionMixin._get_governing_constitution(agent)
                    assert result.startswith("Error:") and "history changed" in result, result
        assert calls == []
    finally:
        await storage.close()
