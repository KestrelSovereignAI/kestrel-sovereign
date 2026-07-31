"""Contract tests for the narrow #2752 legacy graph-fact migration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.storage.async_assertion_store import (
    _issue_assertion_tenant_capability,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.legacy_fact_migration import (
    LegacyGraphFactMigration,
)
from kestrel_sovereign.storage.sqla.migrations import (
    migrate_legacy_graph_fact_migration_state,
)


TENANT = "did:example:legacy-facts"


@pytest.mark.asyncio
async def test_bookkeeping_schema_is_common_transactional_ddl_for_postgres():
    db = MagicMock()
    db.execute = AsyncMock()

    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    db.transaction = MagicMock(return_value=Transaction())
    await migrate_legacy_graph_fact_migration_state(db)
    statements = [call.args[0] for call in db.execute.await_args_list]
    assert any("legacy_fact_migration_records" in statement for statement in statements)
    assert any("legacy_fact_migration_checkpoints" in statement for statement in statements)
    db.transaction.assert_called_once_with()


async def _storage(tmp_path) -> AsyncStorage:
    storage = AsyncStorage(
        str(tmp_path / "legacy-facts.db"),
        agent_id=TENANT,
        _assertion_tenant_capability=_issue_assertion_tenant_capability(TENANT),
    )
    await storage.initialize()
    return storage


async def _node(storage: AsyncStorage, node_id: str, properties: object, *, owner: str = TENANT) -> None:
    assert storage.db is not None
    raw = properties if isinstance(properties, str) else json.dumps(properties)
    await storage.db.execute(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) VALUES (?, 'fact', ?, ?)",
        (node_id, "legacy fact", raw),
    )
    await storage.db.execute(
        "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
        (node_id, owner),
    )


@pytest.mark.asyncio
async def test_plan_is_content_safe_and_never_trusts_properties_owner(tmp_path):
    storage = await _storage(tmp_path)
    try:
        await _node(
            storage,
            "fact-1",
            {
                "subject": "user",
                "predicate": "preferred_deploy_region",
                "value": "private-region-value",
                "created_at": "2026-01-01T00:00:00+00:00",
                "agent_id": "did:attacker:forged",
            },
        )
        migration = storage.legacy_graph_fact_migration()
        plan = await migration.plan()

        assert plan.scanned == plan.eligible == 1
        assert plan.by_agent == {TENANT: 1}
        assert plan.by_source == {"legacy_graph_node": 1}
        assert plan.content_hashes and "private-region-value" not in repr(plan.to_mapping())
        assert plan.compatibility_flag_enabled is False
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_migrates_idempotently_across_restart_and_rolls_back_without_legacy_delete(tmp_path):
    storage = await _storage(tmp_path)
    try:
        original = {
            "subject": "user",
            "predicate": "preferred_deploy_region",
            "value": "restart-safe-region",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        await _node(storage, "fact-1", original)
        first = await LegacyGraphFactMigration(storage).run(batch_size=1)
        assert first.migrated == 1
        assert first.complete

        # Simulate a fresh process using the same authenticated tenant.
        await storage.close()
        storage = await _storage(tmp_path)
        restarted = LegacyGraphFactMigration(storage)
        second = await restarted.run(batch_size=1)
        assert second.migrated == 0
        assert second.idempotent == 0

        assert storage.db is not None
        rows = await storage.db.fetchall(
            "SELECT outcome FROM legacy_fact_migration_records WHERE tenant_id = ?",
            (TENANT,),
        )
        assert [row[0] for row in rows] == ["migrated"]
        rollback = await restarted.rollback()
        assert rollback.migrated == 1
        # The original graph fact is an audit/compatibility input, never a
        # migration cleanup target.
        legacy = await storage.db.fetchone("SELECT properties FROM graph_nodes WHERE node_id = ?", ("fact-1",))
        assert json.loads(legacy[0]) == original
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_rejects_malformed_unsupported_and_shared_nodes_without_promotion(tmp_path):
    storage = await _storage(tmp_path)
    try:
        # Current graph writes enforce JSON validity; a legacy non-object is
        # still malformed for the fact adapter and exercises the same safe
        # rejection path without bypassing database integrity checks.
        await _node(storage, "bad-json", "[]")
        await _node(
            storage,
            "unsupported",
            {"subject": "user", "predicate": "unmapped", "value": "value", "created_at": "2026-01-01T00:00:00+00:00"},
        )
        await _node(
            storage,
            "shared",
            {"subject": "user", "predicate": "preferred_deploy_region", "value": "not-promoted", "created_at": "2026-01-01T00:00:00+00:00"},
        )
        assert storage.db is not None
        await storage.db.execute(
            "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
            ("shared", "did:example:other"),
        )
        result = await LegacyGraphFactMigration(storage).run(batch_size=3)

        assert result.migrated == 0
        assert result.rejected["malformed_properties"] == 1
        assert result.rejected["shared_or_ambiguous_ownership"] == 1
        assert result.rejected["unsupported predicate 'unmapped'; supported predicates: preferred_deploy_region"] == 1
        current = await storage.db.fetchone(
            "SELECT COUNT(*) FROM semantic_assertions WHERE tenant_id = ?", (TENANT,)
        )
        assert current[0] == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_feature_projection_invalidator_observes_only_accepted_assertions(tmp_path):
    storage = await _storage(tmp_path)
    seen: list[tuple[str, tuple[str, ...]]] = []
    try:
        await _node(
            storage,
            "fact-1",
            {"subject": "user", "predicate": "preferred_deploy_region", "value": "indexed-region", "created_at": "2026-01-01T00:00:00+00:00"},
        )

        async def invalidate(tenant: str, assertion_ids: tuple[str, ...]) -> None:
            seen.append((tenant, assertion_ids))

        result = await LegacyGraphFactMigration(storage, index_invalidator=invalidate).run()
        assert result.index_invalidation_requested is True
        assert seen[0][0] == TENANT
        assert len(seen[0][1]) == 1
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_crash_after_canonical_commit_replays_deterministic_operation_without_duplicate(tmp_path, monkeypatch):
    storage = await _storage(tmp_path)
    try:
        await _node(
            storage,
            "fact-crash",
            {"subject": "user", "predicate": "preferred_deploy_region", "value": "crash-region", "created_at": "2026-01-01T00:00:00+00:00"},
        )
        migration = LegacyGraphFactMigration(storage)
        original_record = migration._record
        calls = 0

        async def crash_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("simulated process interruption after canonical commit")
            await original_record(*args, **kwargs)

        monkeypatch.setattr(migration, "_record", crash_once)
        with pytest.raises(RuntimeError, match="simulated process interruption"):
            await migration.run(batch_size=1)

        # The first canonical transaction committed, but no migration record
        # did. A fresh runner repeats the immutable operation receipt rather
        # than creating another assertion/revision.
        recovered = await LegacyGraphFactMigration(storage).run(batch_size=1)
        assert recovered.idempotent == 1
        assert storage.db is not None
        count = await storage.db.fetchone(
            "SELECT COUNT(*) FROM semantic_assertions WHERE tenant_id = ?", (TENANT,)
        )
        assert count[0] == 1
    finally:
        await storage.close()
