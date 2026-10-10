"""Populated upgrades, atomic lifetime allocation, and native refusal durability."""

import asyncio
import re
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin, SafeModeCause
from kestrel_sovereign.constitution.runtime_state import ConstitutionRuntimeStateStore
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from tests.integration.test_constitution_refusal_races import _agent
from tests.integration.test_constitution_turn_admission import _ready_turn
from tests.integration.test_constitution_exit_evidence import _assert_integrity_refusal


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("completion", ["not an instant", "1999-01-01T00:00:00Z"])
async def test_runtime_migration_refuses_contradictory_legacy_completion(db_backend, completion):
    from kestrel_sovereign.constitution.genesis_audit import GenesisAuditError

    async with _isolated_schema(db_backend) as backend:
        storage = AsyncStorage(backend=backend, agent_id="did:test:legacy-completion:" + uuid4().hex)
        await storage.initialize()
        try:
            agent = await _agent(storage)
            digest = await storage.store_file(resolve_governing_constitution_bytes(None), "constitution.md")
            receipt = {"constitution_hash": digest, "risk_level": 1, "timestamp": "2026-10-09T00:00:00Z", "completed_at": completion}
            await storage.add_node(GraphNode(node_id=agent.agent_id, node_type="agent", label="legacy receipt", properties={"constitution_hash": digest, "genesis_audit": receipt}))
            await agent._anchor_constitution_governance(digest)
            before = (await storage.get_node(agent.agent_id)).properties
            state = await agent._constitution_state_store.load(agent.agent_id)
            events = await agent._constitution_state_store.list_events(agent.agent_id)

            async def never_audit(*args, **kwargs):
                pytest.fail("Malformed completion evidence must never reroll a paid auditor")

            agent.get_audit_response = never_audit
            with pytest.raises(GenesisAuditError, match="completion"):
                await ConstitutionMixin.perform_genesis_audit(agent)
            assert (await storage.get_node(agent.agent_id)).properties == before
            assert await agent._constitution_state_store.load(agent.agent_id) == state
            assert await agent._constitution_state_store.list_events(agent.agent_id) == events
        finally:
            await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("reject_restriction", [False, True])
async def test_native_final_refusal_reports_replacement_commit_failure(db_backend, reject_restriction):
    async with _isolated_schema(db_backend) as backend:
        storage = AsyncStorage(backend=backend, agent_id="did:test:refusal-durability:" + uuid4().hex)
        await storage.initialize()
        trigger = "core_refusal_" + uuid4().hex
        pg = backend.backend_type == "postgres"
        installed = False
        try:
            agent = await _agent(storage)
            digest = await storage.store_file(resolve_governing_constitution_bytes(None), "constitution.md")
            await storage.add_node(GraphNode(node_id=agent.agent_id, node_type="agent", label="refusal", properties={"constitution_hash": digest}))
            await agent._anchor_constitution_governance(digest)
            assert await agent.enter_safe_mode("prior restriction")
            before = await agent._constitution_state_store.load(agent.agent_id)
            events = await agent._constitution_state_store.list_events(agent.agent_id)
            native_persist = agent._persist_constitution_runtime_state

            async def changed_then_persist(**kwargs):
                nonlocal installed
                if kwargs.get("event_type") == "safe_mode_exited":
                    # Change REAL SQL only after the real preliminary verifier.
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
                return await native_persist(**kwargs)

            agent._persist_constitution_runtime_state = changed_then_persist
            result = await agent.exit_safe_mode(authorization="explicit fixture owner")
            assert "integrity verification refused" in result, result
            assert "governed_by" in result, result
            assert agent._safe_mode is True
            if reject_restriction:
                assert "replacement restriction could not be persisted" in result, result
                assert agent._constitution_state_persistence_pending is True
                assert await agent._constitution_state_store.load(agent.agent_id) == before
                assert await agent._constitution_state_store.list_events(agent.agent_id) == events
            else:
                assert "could not be persisted" not in result, result
                await _assert_integrity_refusal(agent, before)
        finally:
            if installed:
                await backend.execute(f"DROP TRIGGER {trigger}" + (" ON constitution_runtime_events" if pg else ""))
                if pg:
                    await backend.execute(f"DROP FUNCTION {trigger}()")
            await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_final_attestation_translates_native_malformed_legacy_target(db_backend):
    # Ordinary JSON indexes already reject this corruption at write time.
    # Emulate a legacy database missing those indexes in an OWNED schema/file,
    # never by altering the shared PostgreSQL fixture's production-like schema.
    async with _isolated_schema(db_backend) as backend:
        storage = AsyncStorage(backend=backend, agent_id="did:test:legacy-json:" + uuid4().hex)
        await storage.initialize()
        try:
            if backend.backend_type == "postgres":
                indexes = await backend.fetch_all("SELECT indexname FROM pg_indexes WHERE schemaname=current_schema() AND tablename='graph_nodes' AND indexdef LIKE '%jsonb%'")
            else:
                indexes = await backend.fetch_all("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='graph_nodes' AND sql LIKE '%json%'")
            assert indexes
            for (name,) in indexes:
                assert re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name)
                await backend.execute(f"DROP INDEX {name}")
            agent = await _agent(storage)
            digest = await storage.store_file(resolve_governing_constitution_bytes(None), "constitution.md")
            await storage.add_node(GraphNode(node_id=agent.agent_id, node_type="agent", label="legacy", properties={"constitution_hash": digest}))
            await agent._anchor_constitution_governance(digest)
            assert await agent.enter_safe_mode("prior legacy restriction")
            before = await agent._constitution_state_store.load(agent.agent_id)
            native_persist = agent._persist_constitution_runtime_state
            failures = []
            native_mark = agent._mark_constitution_state_unavailable

            def observed_failure(exc, **kwargs):
                failures.append(exc)
                return native_mark(exc, **kwargs)

            async def corrupt_then_persist(**kwargs):
                if kwargs.get("event_type") == "safe_mode_exited":
                    await storage.db.execute_commit("UPDATE graph_nodes SET properties=? WHERE node_id=?", ("{malformed", digest))
                return await native_persist(**kwargs)

            agent._persist_constitution_runtime_state = corrupt_then_persist
            agent._mark_constitution_state_unavailable = observed_failure
            result = await agent.exit_safe_mode(authorization="explicit fixture owner")
            assert "integrity verification refused" in result and "Malformed" in result, result
            assert failures == [], "Available native SQL must not be misreported as a database outage"
            await _assert_integrity_refusal(agent, before)
        finally:
            await storage.close()


@asynccontextmanager
async def _isolated_schema(backend):
    """Never replace the shared PostgreSQL fixture's live runtime tables."""
    if backend.backend_type != "postgres":
        yield backend
        return
    schema = "core05326_migrate_" + uuid4().hex
    await backend.execute(f"CREATE SCHEMA {schema}")
    identity = await backend.fetch_one("SELECT oid FROM pg_namespace WHERE nspname=?", (schema,))
    scoped = PostgresBackend(
        backend._dsn + ("&" if "?" in backend._dsn else "?") + "search_path=" + schema + ",public",
        min_pool_size=1, max_pool_size=2, advisory_max_pool_size=1,
    )
    try:
        await scoped.connect()
        assert (await scoped.fetch_one("SELECT current_schema()"))[0] == schema
        yield scoped
    finally:
        await scoped.close()
        actual = await backend.fetch_one("SELECT oid FROM pg_namespace WHERE nspname=?", (schema,))
        assert actual == identity, "Refuse cleanup of a replaced schema"
        await backend.execute(f"DROP SCHEMA {schema} CASCADE")


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("legacy_shape", ["pre-generation", "empty-generation-v3"])
@pytest.mark.parametrize("restricted", [False, True])
@pytest.mark.parametrize("legacy_bootstrap", [False, True])
async def test_populated_runtime_upgrade_fences_generations_without_resetting_evidence(db_backend, legacy_shape, restricted, legacy_bootstrap):
    async with _isolated_schema(db_backend) as backend:
        store = ConstitutionRuntimeStateStore(backend)
        pg = backend.backend_type == "postgres"
        timestamp, boolean = store._timestamp_type(), store._boolean_type()
        generation_column = ", generation TEXT NOT NULL DEFAULT ''" if legacy_shape == "empty-generation-v3" else ""
        await backend.execute_script(f"""
            CREATE TABLE constitution_runtime_state (
                agent_id TEXT PRIMARY KEY, safe_mode {boolean} NOT NULL,
                safe_mode_reason TEXT, safe_mode_entered_at {timestamp},
                safe_mode_exited_at {timestamp}, safe_mode_exit_authorization TEXT,
                last_successful_audit_at {timestamp}, interaction_count INTEGER NOT NULL,
                bootstrap_pending {boolean} NOT NULL, schema_version INTEGER NOT NULL,
                updated_at {timestamp} NOT NULL, safe_mode_cause TEXT,
                revision INTEGER NOT NULL DEFAULT 0 {generation_column}
            );
            CREATE TABLE constitution_runtime_events (
                id {store._integer_primary_key_type()}, agent_id TEXT NOT NULL,
                event_type TEXT NOT NULL, reason TEXT, authorization_detail TEXT,
                occurred_at {timestamp} NOT NULL
            );
        """)
        identity = "did:test:populated-upgrade:" + uuid4().hex
        instant = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
        reason = "retained registry restriction" if restricted else None
        cause = SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value if restricted else None
        await backend.execute(
            "INSERT INTO constitution_runtime_state "
            "(agent_id,safe_mode,safe_mode_reason,safe_mode_entered_at,last_successful_audit_at,"
            "interaction_count,bootstrap_pending,schema_version,updated_at,safe_mode_cause,revision) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (identity, store._boolean_param(restricted), reason, store._timestamp_param(instant) if restricted else None,
             store._timestamp_param(instant), 14, store._boolean_param(legacy_bootstrap), 1,
             store._timestamp_param(instant), cause, 7),
        )
        await backend.execute(
            "INSERT INTO constitution_runtime_events (agent_id,event_type,reason,occurred_at) VALUES (?,?,?,?)",
            (identity, "legacy_event", "retained evidence", store._timestamp_param(instant)),
        )
        before_events = await store.list_events(identity)
        if legacy_shape == "empty-generation-v3":
            trigger = "constitution_runtime_revision_fence_v3"
            if pg:
                await backend.execute_script(f"""
                    CREATE FUNCTION {trigger}() RETURNS trigger AS $f$
                    BEGIN
                        IF TG_OP='INSERT' THEN
                            IF NEW.generation='' THEN RAISE EXCEPTION 'old generation fence'; END IF;
                        ELSIF NEW.revision<>OLD.revision+1 OR NEW.generation<>OLD.generation THEN
                            RAISE EXCEPTION 'old revision fence';
                        END IF;
                        RETURN NEW;
                    END; $f$ LANGUAGE plpgsql;
                    CREATE TRIGGER {trigger} BEFORE INSERT OR UPDATE ON constitution_runtime_state
                    FOR EACH ROW EXECUTE FUNCTION {trigger}();
                """)
            else:
                await backend.execute_script(f"""
                    CREATE TRIGGER {trigger} BEFORE UPDATE ON constitution_runtime_state
                    FOR EACH ROW WHEN NEW.revision<>OLD.revision+1 OR NEW.generation<>OLD.generation
                    BEGIN SELECT RAISE(ABORT, 'old revision fence'); END;
                    CREATE TRIGGER {trigger}_insert BEFORE INSERT ON constitution_runtime_state
                    FOR EACH ROW WHEN NEW.generation=''
                    BEGIN SELECT RAISE(ABORT, 'old generation fence'); END;
                """)
        await store.initialize()
        migrated = await store.load(identity)
        assert migrated.generation and migrated.revision == 8
        assert migrated.safe_mode is restricted
        assert migrated.safe_mode_reason == reason and migrated.safe_mode_cause == cause
        assert migrated.last_successful_audit_at == instant
        assert migrated.interaction_count == 14 and migrated.bootstrap_pending is legacy_bootstrap
        assert migrated.updated_at == instant
        assert migrated.safe_mode_exited_at is None and migrated.safe_mode_exit_authorization is None
        events = await store.list_events(identity)
        assert events[:-1] == before_events and events[-1]["event_type"] == "runtime_generation_migrated"
        await store.initialize()
        assert await store.load(identity) == migrated
        assert await store.list_events(identity) == events
        # A normal CAS can advance the upgraded lifetime, never rotate it or
        # recreate consumed bootstrap authority.
        counted = await store.write(replace(migrated, interaction_count=15), event_type="interaction_counted")
        assert counted.generation == migrated.generation and counted.bootstrap_pending is legacy_bootstrap
        with pytest.raises(Exception, match="fence"):
            await backend.execute("UPDATE constitution_runtime_state SET generation='',revision=revision+1 WHERE agent_id=?", (identity,))
        assert (await store.load(identity)).generation == migrated.generation
        if legacy_bootstrap:
            storage = AsyncStorage(backend=backend, agent_id=identity)
            await storage.initialize()
            try:
                await storage.add_node(GraphNode(node_id=identity, node_type="agent", label="legacy pending marker", properties={}))
                agent = await _agent(storage, is_new_identity=False)
                before = await store.load(identity)
                before_events = await store.list_events(identity)
                result = await ConstitutionMixin._get_governing_constitution(agent)
                assert result.startswith("Error:") and "signed repair" in result, result
                assert (await storage.get_node(identity)).properties == {}
                assert await store.load(identity) == before
                assert await store.list_events(identity) == before_events
            finally:
                await storage.close()
        elif not restricted:
            storage = AsyncStorage(backend=backend, agent_id=identity)
            await storage.initialize()
            try:
                agent, _ = await _ready_turn(storage, is_new_identity=False)
                assert await storage.get_node(agent.agent_id) is not None
                # Real CONVERSATION admission checks the migrated generation,
                # durable audit and receipt without reaching a paid provider.
                async with agent._turn_lifecycle():
                    result = await ConstitutionMixin._genesis_audit_cognition_block(agent, "ordinary turn")
                    assert result is None, (result, agent._safe_mode_reason, agent.agent_id, storage.agent_id)
            finally:
                await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("new_identity", [False, True])
async def test_stale_creation_decision_cannot_cross_a_surviving_lifetime(db_backend, monkeypatch, new_identity):
    import kestrel_sovereign.constitution.runtime_state as runtime

    storage = AsyncStorage(backend=db_backend, agent_id="did:test:creation-race:" + uuid4().hex)
    await storage.initialize()
    stale = None
    reached, release = asyncio.Event(), asyncio.Event()

    class PausedStore(ConstitutionRuntimeStateStore):
        async def has_lifetime_history(self, identity):
            result = await super().has_lifetime_history(identity)
            if asyncio.current_task() is stale and not self._backend.owns_open_transaction and not reached.is_set():
                assert result is False
                reached.set()
                await release.wait()
            return result

    monkeypatch.setattr(runtime, "ConstitutionRuntimeStateStore", PausedStore)
    try:
        stale = asyncio.create_task(_agent(storage, is_new_identity=new_identity))
        await asyncio.wait_for(reached.wait(), 5)
        other = await _agent(storage)
        assert await other.enter_safe_mode("intervening lifetime restriction", cause=SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value)
        prior = await other._constitution_state_store.load(storage.agent_id)
        events = await other._constitution_state_store.list_events(storage.agent_id)
        await storage.db.execute_commit("DELETE FROM constitution_runtime_state WHERE agent_id=?", (storage.agent_id,))
        release.set()
        fresh = await asyncio.wait_for(stale, 5)
        current = await fresh._constitution_state_store.load(storage.agent_id)
        assert current.safe_mode is True and fresh._safe_mode is True
        assert current.generation != prior.generation
        assert current.safe_mode_cause == SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value
        assert current.bootstrap_pending is False and current.last_successful_audit_at is None
        assert fresh._constitution_bootstrap_pending is False
        assert (await fresh._constitution_state_store.list_events(storage.agent_id))[:-1] == events
        await fresh._audit_constitution_on_startup()
        assert (await fresh._constitution_state_store.load(storage.agent_id)).safe_mode is True
    finally:
        release.set()
        if stale is not None:
            if not stale.done():
                stale.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await stale
        await storage.close()
