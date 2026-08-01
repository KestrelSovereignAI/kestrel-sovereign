"""Contract tests for the narrow #2752 legacy graph-fact migration."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

import kestrel_sovereign.storage.legacy_fact_migration as migration_module
from kestrel_sovereign.storage.async_assertion_store import (
    _issue_assertion_tenant_capability,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.legacy_fact_migration import (
    LegacyGraphFactMigration,
    LegacyFactMigrationError,
)
from kestrel_sovereign.storage.sqla.migrations import (
    migrate_legacy_graph_fact_migration_state,
)
from kestrel_sovereign.storage.timestamps import utc_timestamp_parameter


TENANT = "did:example:legacy-facts"


@pytest.mark.parametrize(
    ("backend_type", "expected_type"),
    (("sqlite", str), ("postgres", datetime)),
)
def test_durable_utc_timestamp_parameter_has_an_explicit_backend_contract(
    backend_type, expected_type,
):
    value = "2026-01-01T06:00:00-06:00"

    parameter = utc_timestamp_parameter(backend_type, value)

    assert isinstance(parameter, expected_type)
    if backend_type == "sqlite":
        assert parameter == "2026-01-01T12:00:00+00:00"
    else:
        assert parameter == datetime(2026, 1, 1, 12)


@pytest.mark.parametrize(
    "value", ("2026-01-01T12:00:00", datetime(2026, 1, 1, 12))
)
def test_durable_utc_timestamp_parameter_rejects_ambiguous_instants(value):
    with pytest.raises(ValueError, match="timezone"):
        utc_timestamp_parameter("postgres", value)


@pytest.mark.asyncio
async def test_bookkeeping_schema_is_common_transactional_ddl_for_postgres():
    db = MagicMock()
    db.backend_type = "postgres"
    db.execute = AsyncMock()
    db.fetchone = AsyncMock(return_value=(1,))

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
    assert any("legacy_fact_migration_invalidations" in statement for statement in statements)
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


async def _assert_real_postgres_migration_contract(dsn: str) -> None:
    """Exercise the migration against an authority-provided PostgreSQL DSN."""
    tenant = "did:example:legacy-facts-postgres"
    original = {
        "subject": "user",
        "predicate": "preferred_deploy_region",
        "value": "postgres-content-must-stay-legacy",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    storage = None
    try:
        storage = AsyncStorage(
            backend="postgres",
            dsn=dsn,
            agent_id=tenant,
            _assertion_tenant_capability=_issue_assertion_tenant_capability(tenant),
        )
        await storage.initialize()
        await _node(storage, "fact-postgres", original, owner=tenant)

        assert storage.db is not None
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM graph_nodes WHERE node_id = ?",
            ("fact-postgres",),
        ) == 1
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM graph_node_owners WHERE node_id = ?",
            ("fact-postgres",),
        ) == 1
        assert await storage.db.fetchval(
            "SELECT agent_id FROM graph_node_owners WHERE node_id = ?",
            ("fact-postgres",),
        ) == tenant
        migration = storage.legacy_graph_fact_migration()
        assert len(await migration._rows_after(storage.db, tenant, None, 1)) == 1

        first = await migration.run(batch_size=1)
        assert first.migrated == 1
        assert first.complete is True

        recorded_at = await storage.db.fetchval(
            "SELECT created_at FROM legacy_fact_migration_records "
            "WHERE tenant_id = ? AND node_id = ?",
            (tenant, "fact-postgres"),
        )
        checkpoint_at = await storage.db.fetchval(
            "SELECT updated_at FROM legacy_fact_migration_checkpoints "
            "WHERE tenant_id = ? AND migration_name = ?",
            (tenant, migration_module.MIGRATION_NAME),
        )
        assert isinstance(recorded_at, datetime)
        assert recorded_at.tzinfo is None
        assert isinstance(checkpoint_at, datetime)
        assert checkpoint_at.tzinfo is None

        await storage.close()
        storage = AsyncStorage(
            backend="postgres",
            dsn=dsn,
            agent_id=tenant,
            _assertion_tenant_capability=_issue_assertion_tenant_capability(tenant),
        )
        await storage.initialize()
        restarted = await storage.legacy_graph_fact_migration().run(batch_size=1)
        assert restarted.processed == 0
        assert restarted.migrated == 0
        rollback = await storage.legacy_graph_fact_migration().rollback()
        assert rollback.migrated == 1
        legacy = await storage.db.fetchone(
            "SELECT properties FROM graph_nodes WHERE node_id = ?",
            ("fact-postgres",),
        )
        assert legacy is not None
        assert json.loads(legacy[0]) == original
    finally:
        if storage is not None:
            await storage.close()


@pytest.mark.asyncio
async def test_migration_bookkeeping_uses_typed_timestamps_on_real_disposable_postgres():
    """The migration stays restart-safe on asyncpg, not only on SQLite."""
    catalog_dsn = (
        os.environ.get("TEST_POSTGRES_URL")
        if os.environ.get("KESTREL_SEMANTIC_RELEASE_CATALOG_POSTGRES") == "1"
        else None
    )
    if catalog_dsn:
        await _assert_real_postgres_migration_contract(catalog_dsn)
        return
    if (
        os.environ.get("KESTREL_SEMANTIC_RELEASE_ISOLATED") != "1"
        or not os.environ.get("KESTREL_SEMANTIC_RELEASE_ISOLATED_POSTGRES_ADMIN_DSN")
    ):
        pytest.skip("isolated PostgreSQL release-evidence environment is required")
    from kestrel_sovereign.knowledge.release_evidence_postgres import (
        DisposablePostgresDatabase,
    )

    async with await DisposablePostgresDatabase.create() as database:
        await _assert_real_postgres_migration_contract(database.dsn)


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
        assert result.rejected["unsupported_semantic_mapping"] == 1
        current = await storage.db.fetchone(
            "SELECT COUNT(*) FROM semantic_assertions WHERE tenant_id = ?", (TENANT,)
        )
        assert current[0] == 0
        # Stable rejected shared ownership is an auditable terminal outcome,
        # not source drift requiring an impossible reset.
        assert (await LegacyGraphFactMigration(storage).run()).processed == 0
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
        rollback = await LegacyGraphFactMigration(
            storage, index_invalidator=invalidate
        ).rollback()
        assert rollback.index_invalidation_requested is True
        assert len(seen) == 2
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_stale_invalidation_callback_cannot_ack_rollback_generation(tmp_path):
    storage = await _storage(tmp_path)
    callback_started = asyncio.Event()
    release_stale_callback = asyncio.Event()
    assertion_was_present: list[bool] = []
    try:
        await _node(
            storage,
            "fact-invalidation-race",
            {
                "subject": "user",
                "predicate": "preferred_deploy_region",
                "value": "generation-race",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
        )

        async def invalidate(_tenant: str, assertion_ids: tuple[str, ...]) -> None:
            assertion_was_present.append(
                await storage.get_assertion(assertion_ids[0]) is not None
            )
            if len(assertion_was_present) == 1:
                callback_started.set()
                await release_stale_callback.wait()

        migration = LegacyGraphFactMigration(
            storage, index_invalidator=invalidate
        )
        run_task = asyncio.create_task(migration.run())
        await callback_started.wait()

        # Withdrawal advances the durable invalidation generation while the
        # original assertion-present callback is still in flight.
        rollback = await LegacyGraphFactMigration(storage).rollback()
        assert rollback.index_invalidation_requested is False
        release_stale_callback.set()
        result = await run_task

        assert result.index_invalidation_requested is True
        assert assertion_was_present == [True, False]
        assert storage.db is not None
        receipt = await storage.db.fetchone(
            "SELECT state, generation FROM legacy_fact_migration_invalidations "
            "WHERE tenant_id = ?",
            (TENANT,),
        )
        assert receipt == ("delivered", 2)
    finally:
        release_stale_callback.set()
        await storage.close()


@pytest.mark.asyncio
async def test_projection_invalidation_drains_every_bounded_page(
    tmp_path, monkeypatch
):
    storage = await _storage(tmp_path)
    seen: list[tuple[str, ...]] = []
    try:
        monkeypatch.setattr(migration_module, "_INVALIDATION_PAGE_SIZE", 2)
        monkeypatch.setattr(migration_module, "_MAX_INVALIDATION_PAGES", 3)
        assert storage.db is not None
        async with storage.db.transaction():
            for index in range(5):
                await storage.db.execute(
                    "INSERT INTO legacy_fact_migration_invalidations "
                    "(tenant_id, migration_name, assertion_id, state, created_at, delivered_at) "
                    "VALUES (?, ?, ?, 'pending', ?, NULL)",
                    (
                        TENANT,
                        migration_module.MIGRATION_NAME,
                        f"assertion:{index}",
                        "2026-01-01T00:00:00+00:00",
                    ),
                )

        async def invalidate(_tenant: str, assertion_ids: tuple[str, ...]) -> None:
            seen.append(assertion_ids)

        result = await LegacyGraphFactMigration(
            storage, index_invalidator=invalidate
        ).run()
        assert result.index_invalidation_requested is True
        assert [len(page) for page in seen] == [2, 2, 1]
        pending = await storage.db.fetchone(
            "SELECT COUNT(*) FROM legacy_fact_migration_invalidations "
            "WHERE tenant_id = ? AND state = 'pending'",
            (TENANT,),
        )
        assert pending == (0,)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_projection_invalidation_budget_surfaces_residual_for_retry(
    tmp_path, monkeypatch
):
    storage = await _storage(tmp_path)
    seen: list[tuple[str, ...]] = []
    try:
        monkeypatch.setattr(migration_module, "_INVALIDATION_PAGE_SIZE", 2)
        monkeypatch.setattr(migration_module, "_MAX_INVALIDATION_PAGES", 2)
        assert storage.db is not None
        async with storage.db.transaction():
            for index in range(5):
                await storage.db.execute(
                    "INSERT INTO legacy_fact_migration_invalidations "
                    "(tenant_id, migration_name, assertion_id, state, created_at, delivered_at) "
                    "VALUES (?, ?, ?, 'pending', ?, NULL)",
                    (
                        TENANT,
                        migration_module.MIGRATION_NAME,
                        f"assertion:{index}",
                        "2026-01-01T00:00:00+00:00",
                    ),
                )

        async def invalidate(_tenant: str, assertion_ids: tuple[str, ...]) -> None:
            seen.append(assertion_ids)

        migration = LegacyGraphFactMigration(
            storage, index_invalidator=invalidate
        )
        with pytest.raises(
            LegacyFactMigrationError,
            match="projection_invalidation_delivery_budget_exhausted",
        ):
            await migration.run()
        assert [len(page) for page in seen] == [2, 2]
        pending = await storage.db.fetchone(
            "SELECT COUNT(*) FROM legacy_fact_migration_invalidations "
            "WHERE tenant_id = ? AND state = 'pending'",
            (TENANT,),
        )
        assert pending == (1,)

        monkeypatch.setattr(migration_module, "_MAX_INVALIDATION_PAGES", 3)
        recovered = await migration.run()
        assert recovered.index_invalidation_requested is True
        assert [len(page) for page in seen] == [2, 2, 1]
        pending = await storage.db.fetchone(
            "SELECT COUNT(*) FROM legacy_fact_migration_invalidations "
            "WHERE tenant_id = ? AND state = 'pending'",
            (TENANT,),
        )
        assert pending == (0,)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_rejected_plan_never_echoes_legacy_predicate_or_value(tmp_path):
    storage = await _storage(tmp_path)
    try:
        private_predicate = "private-predicate-8e101"
        private_value = "private-value-a438c"
        await _node(
            storage,
            "unmappable",
            {
                "subject": "user",
                "predicate": private_predicate,
                "value": private_value,
                "created_at": "2026-01-01T00:00:00+00:00",
            },
        )
        plan = await LegacyGraphFactMigration(storage).plan()
        rendered = repr(plan.to_mapping())
        assert plan.rejected == {"unsupported_semantic_mapping": 1}
        assert private_predicate not in rendered
        assert private_value not in rendered
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_projection_invalidation_is_durable_and_retried_after_failure(tmp_path):
    storage = await _storage(tmp_path)
    calls: list[tuple[str, tuple[str, ...]]] = []
    try:
        await _node(
            storage,
            "fact-pending-index",
            {"subject": "user", "predicate": "preferred_deploy_region", "value": "retry-index", "created_at": "2026-01-01T00:00:00+00:00"},
        )

        async def fail_once(tenant: str, assertion_ids: tuple[str, ...]) -> None:
            calls.append((tenant, assertion_ids))
            if len(calls) == 1:
                raise RuntimeError("projection unavailable")

        migration = LegacyGraphFactMigration(storage, index_invalidator=fail_once)
        with pytest.raises(RuntimeError, match="projection unavailable"):
            await migration.run(batch_size=1)

        # The completed canonical page is checkpointed, but the external
        # projection receipt stays pending and is delivered on a later run.
        recovered = await migration.run(batch_size=1)
        assert recovered.processed == 0
        assert recovered.index_invalidation_requested is True
        assert len(calls) == 2
        assert storage.db is not None
        state = await storage.db.fetchone(
            "SELECT state FROM legacy_fact_migration_invalidations WHERE tenant_id = ?",
            (TENANT,),
        )
        assert state[0] == "delivered"
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_lone_surrogate_is_rejected_with_fixed_code(tmp_path):
    storage = await _storage(tmp_path)
    try:
        # json.dumps escapes this for SQLite; json.loads recreates the lone
        # surrogate, which must not escape as an exception or diagnostic.
        await _node(
            storage,
            "bad-unicode",
            {"subject": "user", "predicate": "preferred_deploy_region", "value": "\ud800", "created_at": "2026-01-01T00:00:00+00:00"},
        )
        plan = await LegacyGraphFactMigration(storage).plan()
        assert plan.rejected == {"invalid_unicode": 1}
        result = await LegacyGraphFactMigration(storage).run()
        assert result.rejected == {"invalid_unicode": 1}
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_late_lower_node_requires_review_then_safe_reset(tmp_path):
    storage = await _storage(tmp_path)
    try:
        await _node(
            storage,
            "migrated-middle",
            {"subject": "user", "predicate": "preferred_deploy_region", "value": "first", "created_at": "2026-01-01T00:00:00+00:00"},
        )
        migration = LegacyGraphFactMigration(storage)
        assert (await migration.run(batch_size=1)).complete
        await _node(
            storage,
            "added-before-cursor",
            {"subject": "user", "predicate": "preferred_deploy_region", "value": "late", "created_at": "2026-01-02T00:00:00+00:00"},
        )
        with pytest.raises(LegacyFactMigrationError, match="checkpoint_reset_required"):
            await migration.run(batch_size=1)
        review = await migration.reset_checkpoint_after_review()
        assert review.late_added_before_checkpoint == 1
        result = await migration.run(batch_size=2)
        assert result.migrated == 1
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_corrected_recorded_source_requires_operator_review_not_rewrite(tmp_path):
    storage = await _storage(tmp_path)
    try:
        node_id = "fact-corrected"
        await _node(
            storage,
            node_id,
            {"subject": "user", "predicate": "preferred_deploy_region", "value": "before", "created_at": "2026-01-01T00:00:00+00:00"},
        )
        migration = LegacyGraphFactMigration(storage)
        await migration.run()
        assert storage.db is not None
        await storage.db.execute(
            "UPDATE graph_nodes SET properties = ? WHERE node_id = ?",
            (json.dumps({"subject": "user", "predicate": "preferred_deploy_region", "value": "after", "created_at": "2026-01-01T00:00:00+00:00"}), node_id),
        )
        review = await migration.review_source_set()
        assert review.changed == 1
        with pytest.raises(LegacyFactMigrationError, match="source_review_required"):
            await migration.run()
        with pytest.raises(LegacyFactMigrationError, match="source_review_required"):
            await migration.reset_checkpoint_after_review()
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_bound_tenant_never_inventories_foreign_owner_rows(tmp_path):
    storage = await _storage(tmp_path)
    try:
        await _node(
            storage,
            "foreign-fact",
            {"subject": "user", "predicate": "preferred_deploy_region", "value": "foreign", "created_at": "2026-01-01T00:00:00+00:00"},
            owner="did:example:foreign",
        )
        migration = LegacyGraphFactMigration(storage)
        assert (await migration.plan()).scanned == 0
        assert (await migration.run()).migrated == 0
    finally:
        await storage.close()


def test_storage_package_import_does_not_eagerly_load_memory_agency_feature():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kestrel_sovereign.storage; "
            "assert 'kestrel_sovereign.features.memory_agency.semantic_facts' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_mutated_ownership_after_migration_is_terminal_review_failure(tmp_path):
    storage = await _storage(tmp_path)
    try:
        await _node(
            storage,
            "fact-owner-mutates",
            {"subject": "user", "predicate": "preferred_deploy_region", "value": "owner-first", "created_at": "2026-01-01T00:00:00+00:00"},
        )
        migration = LegacyGraphFactMigration(storage)
        await migration.run()
        assert storage.db is not None
        await storage.db.execute(
            "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
            ("fact-owner-mutates", "did:example:new-co-owner"),
        )
        review = await migration.review_source_set()
        assert review.changed == 1
        with pytest.raises(LegacyFactMigrationError, match="source_review_required"):
            await migration.run()
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_poisoned_rollback_record_becomes_terminal_refusal_not_repeat_wedge(tmp_path):
    storage = await _storage(tmp_path)
    try:
        await _node(
            storage,
            "fact-poison-rollback",
            {"subject": "user", "predicate": "preferred_deploy_region", "value": "poison", "created_at": "2026-01-01T00:00:00+00:00"},
        )
        migration = LegacyGraphFactMigration(storage)
        await migration.run()
        assert storage.db is not None
        await storage.db.execute(
            "UPDATE legacy_fact_migration_records SET source_occurrence_id = ? WHERE tenant_id = ?",
            ("poisoned-source", TENANT),
        )
        refused = await migration.rollback()
        assert refused.rejected == {"rollback_refused_provenance": 1}
        outcome = await storage.db.fetchone(
            "SELECT outcome FROM legacy_fact_migration_records WHERE tenant_id = ?",
            (TENANT,),
        )
        assert outcome[0] == "rollback_refused_provenance"
        assert (await migration.rollback()).processed == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_truncated_compatibility_metrics_never_claim_removal_safe(tmp_path):
    storage = await _storage(tmp_path)
    try:
        for index in range(501):
            await _node(
                storage,
                f"fact-{index:04d}",
                {"subject": "user", "predicate": "preferred_deploy_region", "value": f"v{index}", "created_at": "2026-01-01T00:00:00+00:00"},
            )
        metrics = await LegacyGraphFactMigration(storage).compatibility_metrics()
        assert metrics == {
            "enabled": False,
            "complete_inventory": False,
            "removal_safe": False,
            "reason": "inventory_truncated",
            "scanned": 500,
        }
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_empty_tenant_is_refused_before_ontology_probe():
    storage = MagicMock()
    storage._initialized = True
    storage.db = MagicMock()
    storage.semantic_assertion_binding.return_value = MagicMock(tenant_id="")
    with pytest.raises(LegacyFactMigrationError, match="invalid_migration_tenant"):
        await LegacyGraphFactMigration(storage)._ready()


@pytest.mark.asyncio
async def test_duplicate_claim_rows_append_distinct_provenance_and_rollback_together(tmp_path):
    storage = await _storage(tmp_path)
    try:
        properties = {
            "subject": "user",
            "predicate": "preferred_deploy_region",
            "value": "same-claim",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        await _node(storage, "fact-duplicate-a", properties)
        await _node(storage, "fact-duplicate-b", properties)
        await _node(storage, "fact-duplicate-c", properties)
        migration = LegacyGraphFactMigration(storage)
        result = await migration.run(batch_size=3)
        assert result.migrated == 3
        assert storage.db is not None
        records = await storage.db.fetchall(
            "SELECT assertion_id, source_occurrence_id, outcome FROM legacy_fact_migration_records "
            "WHERE tenant_id = ? ORDER BY node_id",
            (TENANT,),
        )
        assert len({row[0] for row in records}) == 1
        assert len({row[1] for row in records}) == 3
        assert [row[2] for row in records] == [
            "migrated",
            "source_appended",
            "source_appended",
        ]
        sources = await storage.list_assertion_sources(records[0][0])
        assert {source.source_occurrence_id for source in sources} == {
            row[1] for row in records
        }
        # The three record rows share one canonical assertion.  A two-row
        # request must expand to exactly the full source group.
        rollback = await migration.rollback(batch_size=2)
        assert rollback.processed == 3
        assert rollback.migrated == 3
        assert rollback.complete is True
        assert await storage.get_assertion(records[0][0]) is None
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_count", [1, 3])
async def test_rollback_recovers_crash_after_canonical_withdrawal(
    tmp_path, monkeypatch, source_count
):
    storage = await _storage(tmp_path)
    seen: list[tuple[str, tuple[str, ...]]] = []
    try:
        properties = {
            "subject": "user",
            "predicate": "preferred_deploy_region",
            "value": f"rollback-crash-{source_count}",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        for index in range(source_count):
            await _node(storage, f"fact-rollback-crash-{index}", properties)

        async def invalidate(tenant: str, assertion_ids: tuple[str, ...]) -> None:
            seen.append((tenant, assertion_ids))

        migration = LegacyGraphFactMigration(
            storage, index_invalidator=invalidate
        )
        await migration.run(batch_size=source_count)
        assert len(seen) == 1
        original_finalize = migration._finalize_rollback_group
        crashed = False

        async def crash_before_receipts(*args, **kwargs):
            nonlocal crashed
            if not crashed:
                crashed = True
                raise RuntimeError("crash after canonical withdrawal")
            await original_finalize(*args, **kwargs)

        monkeypatch.setattr(
            migration, "_finalize_rollback_group", crash_before_receipts
        )
        with pytest.raises(RuntimeError, match="crash after canonical withdrawal"):
            await migration.rollback(batch_size=1)

        assert storage.db is not None
        assertion_row = await storage.db.fetchone(
            "SELECT assertion_id FROM legacy_fact_migration_records "
            "WHERE tenant_id = ? LIMIT 1",
            (TENANT,),
        )
        assert await storage.get_assertion(assertion_row[0]) is None
        outcomes_before = await storage.db.fetchall(
            "SELECT outcome FROM legacy_fact_migration_records "
            "WHERE tenant_id = ? ORDER BY node_id",
            (TENANT,),
        )
        assert all(row[0] != "rolled_back" for row in outcomes_before)

        recovered = await LegacyGraphFactMigration(
            storage, index_invalidator=invalidate
        ).rollback(batch_size=1)
        assert recovered.processed == source_count
        assert recovered.idempotent == source_count
        assert recovered.complete is True
        assert len(seen) == 2
        outcomes_after = await storage.db.fetchall(
            "SELECT outcome FROM legacy_fact_migration_records "
            "WHERE tenant_id = ? ORDER BY node_id",
            (TENANT,),
        )
        assert outcomes_after == [("rolled_back",)] * source_count
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_duplicate_source_record_crash_replays_without_second_append(tmp_path, monkeypatch):
    storage = await _storage(tmp_path)
    try:
        properties = {
            "subject": "user",
            "predicate": "preferred_deploy_region",
            "value": "same-after-crash",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        await _node(storage, "fact-duplicate-crash-a", properties)
        await _node(storage, "fact-duplicate-crash-b", properties)
        migration = LegacyGraphFactMigration(storage)
        original_record = migration._record
        calls = 0

        async def fail_second_record(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("crash after source append")
            await original_record(*args, **kwargs)

        monkeypatch.setattr(migration, "_record", fail_second_record)
        with pytest.raises(RuntimeError, match="crash after source append"):
            await migration.run(batch_size=2)
        recovered = await LegacyGraphFactMigration(storage).run(batch_size=2)
        assert recovered.idempotent == 2
        assert storage.db is not None
        assertion_id = await storage.db.fetchone(
            "SELECT assertion_id FROM legacy_fact_migration_records WHERE tenant_id = ? LIMIT 1",
            (TENANT,),
        )
        assert len(await storage.list_assertion_sources(assertion_id[0])) == 2
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
