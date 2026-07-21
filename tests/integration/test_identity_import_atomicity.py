"""Atomic, exact identity replace contracts for SQLite and PostgreSQL."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest

from kestrel_sovereign.identity.graph_namespace import (
    namespace_imported_graph_node,
    namespace_imported_record,
)
from kestrel_sovereign.identity.identity_package import (
    AgentIdentityPackage,
    RelationshipRecord,
    SkillRecord,
)
from kestrel_sovereign.identity.importer import IdentityImporter
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import (
    record_graph_edge_owner,
    record_graph_node_owner,
)


_FAULT_BOUNDARIES = (
    "_clear_existing_data",
    "_import_episodes",
    "_import_saved_items",
    "_import_temporal_patterns",
    "_import_reflection_insights",
    "_import_relationships",
    "_import_skills",
    "_import_wallet_state",
    "_record_migration",
)


@dataclass(frozen=True)
class _SeededInventory:
    agent_id: str
    other_agent_id: str
    old_user_node: str
    old_skill_node: str
    old_migration_node: str


async def _database(db_backend) -> AsyncDatabase:
    db = AsyncDatabase(db_backend)
    await db._init_schema()
    db._initialized = True
    return db


async def _seed_old_inventory(db: AsyncDatabase) -> _SeededInventory:
    token = uuid4().hex
    agent_id = f"did:test:replace:{token}"
    other_agent_id = f"did:test:replace-other:{token}"
    prefix = agent_id[:20]
    old_user_node = f"{prefix}_old-user"
    old_skill_node = f"{prefix}_old-skill"
    old_migration_node = f"mig_old_{token}"

    await db.execute(
        "INSERT INTO memory_episodes (id, agent_id, title, summary) "
        "VALUES (?, ?, 'old episode', 'old')",
        (f"{prefix}_old-episode", agent_id),
    )
    await db.execute(
        "INSERT INTO memory_episodes (id, agent_id, title, summary) "
        "VALUES (?, ?, 'other episode', 'other')",
        (f"other_{token}", other_agent_id),
    )
    await db.execute(
        """INSERT INTO saved_items
           (id, agent_id, item_type, name, content, content_hash)
           VALUES (?, ?, 'note', 'old item', 'old content', 'old-hash')""",
        (f"{prefix}_old-item", agent_id),
    )
    await db.execute(
        """INSERT INTO temporal_patterns
           (id, agent_id, pattern_type, description)
           VALUES (?, ?, 'old-pattern', 'old pattern')""",
        (f"{prefix}_old-pattern", agent_id),
    )
    await db.execute(
        """INSERT INTO reflection_insights
           (id, agent_id, type, title, description)
           VALUES (?, ?, 'old-insight', 'old insight', 'old')""",
        (f"{prefix}_old-insight", agent_id),
    )
    await db.execute(
        """INSERT INTO wallet_state
           (agent_id, main_balance, audit_balance)
           VALUES (?, '99.0', '1.0')""",
        (agent_id,),
    )
    await db.execute(
        """INSERT INTO wallet_transactions
           (agent_id, transaction_type, currency, amount, memo, new_balance)
           VALUES (?, 'deposit', 'FIL', '99.0', 'old tx', '99.0')""",
        (agent_id,),
    )

    for node_id, node_type, label in (
        (agent_id, "agent", "Protected agent"),
        (old_user_node, "user", "Old user"),
        (old_skill_node, "skill", "Old skill"),
        (old_migration_node, "migration_record", "Old migration"),
    ):
        await db.execute(
            """INSERT INTO graph_nodes
               (node_id, node_type, label, properties)
               VALUES (?, ?, ?, '{}')""",
            (node_id, node_type, label),
        )
    for target_id, label in (
        (old_user_node, "knows"),
        (old_skill_node, "has_skill"),
        (old_migration_node, "migrated_via"),
    ):
        await db.execute(
            """INSERT INTO graph_edges
               (source_id, target_id, label, properties)
               VALUES (?, ?, ?, '{}')""",
            (agent_id, target_id, label),
        )

    for node_id in (agent_id, old_user_node, old_skill_node, old_migration_node):
        await record_graph_node_owner(db, node_id, agent_id)
    for target_id, label in (
        (old_user_node, "knows"),
        (old_skill_node, "has_skill"),
        (old_migration_node, "migrated_via"),
    ):
        await record_graph_edge_owner(db, agent_id, target_id, label, agent_id)

    return _SeededInventory(
        agent_id=agent_id,
        other_agent_id=other_agent_id,
        old_user_node=old_user_node,
        old_skill_node=old_skill_node,
        old_migration_node=old_migration_node,
    )


def _replacement_package(agent_id: str) -> AgentIdentityPackage:
    return AgentIdentityPackage(
        did=agent_id,
        agent_name="Replacement agent",
        created_at="2026-07-18T12:00:00Z",
        constitution_hash="",
        constitution_text="",
        episodes=[
            {
                "id": "new-episode",
                "title": "new episode",
                "summary": "new",
                "key_message_ids": [],
                "created_at": "2026-07-18T12:01:00Z",
            }
        ],
        saved_items=[
            {
                "id": "new-item",
                "item_type": "note",
                "name": "new item",
                "content": "new content",
                "content_hash": "new-hash",
                "tags": [],
                "metadata": {},
                "created_at": "2026-07-18T12:02:00Z",
                "updated_at": "2026-07-18T12:02:00Z",
            }
        ],
        temporal_patterns=[
            {
                "id": "new-pattern",
                "pattern_type": "new-pattern",
                "description": "new pattern",
                "trigger_conditions": {},
                "created_at": "2026-07-18T12:03:00Z",
                "updated_at": "2026-07-18T12:03:00Z",
            }
        ],
        reflection_insights=[
            {
                "id": "new-insight",
                "insight_type": "new-insight",
                "title": "new insight",
                "description": "new",
                "evidence": [],
                "created_at": "2026-07-18T12:04:00Z",
            }
        ],
        relationships=[
            RelationshipRecord(
                user_id="new-user",
                relationship_type="collaborates_with",
                relationship_notes="new relationship",
            )
        ],
        skills=[
            SkillRecord(
                skill_id="new-skill",
                skill_name="New skill",
                skill_type="tool",
            )
        ],
        wallet_balance="7.0",
        wallet_transaction_history=[
            {
                "id": "source-local-id",
                "transaction_type": "deposit",
                "currency": "FIL",
                "amount": "7.0",
                "memo": "new tx",
                "new_balance": "7.0",
                "created_at": "2026-07-18T12:05:00Z",
            }
        ],
        source_substrate="test:source",
    )


async def _snapshot(db: AsyncDatabase, inventory: _SeededInventory) -> dict:
    agent_id = inventory.agent_id
    rows: dict[str, list[tuple]] = {}
    for table in IdentityImporter.REPLACE_ROW_TABLES:
        rows[table] = await db.fetchall(
            f"SELECT * FROM {table} WHERE agent_id = ? ORDER BY 1",
            (agent_id,),
        )
    rows["graph_edges"] = await db.fetchall(
        "SELECT * FROM graph_edges WHERE source_id = ? ORDER BY target_id, label",
        (agent_id,),
    )
    rows["graph_nodes"] = await db.fetchall(
        """SELECT * FROM graph_nodes
           WHERE node_id = ? OR node_id IN
             (SELECT target_id FROM graph_edges WHERE source_id = ?)
           ORDER BY node_id""",
        (agent_id, agent_id),
    )
    rows["graph_node_owners"] = await db.fetchall(
        "SELECT * FROM graph_node_owners WHERE agent_id = ? ORDER BY node_id",
        (agent_id,),
    )
    rows["graph_edge_owners"] = await db.fetchall(
        "SELECT * FROM graph_edge_owners WHERE agent_id = ? "
        "ORDER BY source_id, target_id, label",
        (agent_id,),
    )
    return rows


async def _cleanup(db: AsyncDatabase, inventory: _SeededInventory) -> None:
    agent_ids = (inventory.agent_id, inventory.other_agent_id)
    for table in IdentityImporter.REPLACE_ROW_TABLES:
        for agent_id in agent_ids:
            await db.execute(
                f"DELETE FROM {table} WHERE agent_id = ?",
                (agent_id,),
            )

    for agent_id in agent_ids:
        await db.execute(
            "DELETE FROM graph_edge_owners WHERE agent_id = ?",
            (agent_id,),
        )
        await db.execute(
            "DELETE FROM graph_node_owners WHERE agent_id = ?",
            (agent_id,),
        )

    node_rows = await db.fetchall(
        """SELECT node_id FROM graph_nodes
           WHERE node_id = ? OR node_id IN
             (SELECT target_id FROM graph_edges WHERE source_id = ?)""",
        (inventory.agent_id, inventory.agent_id),
    )
    for row in node_rows:
        node_id = row[0]
        await db.execute(
            "DELETE FROM graph_edges WHERE source_id = ? OR target_id = ?",
            (node_id, node_id),
        )
        await db.execute(
            "DELETE FROM graph_nodes WHERE node_id = ?",
            (node_id,),
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_replace_rolls_back_every_component_boundary(db_backend):
    """Every required component, including audit evidence, is atomic."""
    db = await _database(db_backend)

    for method_name in _FAULT_BOUNDARIES:
        inventory = await _seed_old_inventory(db)
        before = await _snapshot(db, inventory)
        importer = IdentityImporter(db, target_agent_id=inventory.agent_id)
        original = getattr(importer, method_name)

        async def fail_after_write(*args, _original=original, **kwargs):
            await _original(*args, **kwargs)
            raise RuntimeError("injected component boundary failure")

        setattr(importer, method_name, fail_after_write)
        try:
            result = await importer.import_package(
                _replacement_package(inventory.agent_id),
                verify_signature=False,
                verify_constitution=False,
                allow_unsigned=True,
                merge_mode="replace",
            )

            assert result.success is False, method_name
            assert result.migration_id == "", method_name
            assert result.stats == {}, method_name
            assert "rolled back" in result.errors[0], method_name
            assert await _snapshot(db, inventory) == before, method_name
        finally:
            await _cleanup(db, inventory)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_replace_exact_inventory_and_preserves_audit_nodes(db_backend):
    db = await _database(db_backend)
    inventory = await _seed_old_inventory(db)
    package = _replacement_package(inventory.agent_id)

    try:
        result = await IdentityImporter(
            db, target_agent_id=inventory.agent_id
        ).import_package(
            package,
            verify_signature=False,
            verify_constitution=False,
            allow_unsigned=True,
            merge_mode="replace",
        )

        assert result.success is True
        assert result.migration_id

        expected_ids = {
            "memory_episodes": namespace_imported_record(
                inventory.agent_id, "new-episode"
            ),
            "saved_items": namespace_imported_record(
                inventory.agent_id, "new-item"
            ),
            "temporal_patterns": namespace_imported_record(
                inventory.agent_id, "new-pattern"
            ),
            "reflection_insights": namespace_imported_record(
                inventory.agent_id, "new-insight"
            ),
        }
        for table, expected_id in expected_ids.items():
            rows = await db.fetchall(
                f"SELECT id FROM {table} WHERE agent_id = ?",
                (inventory.agent_id,),
            )
            assert rows == [(expected_id,)]

        wallet = await db.fetchone(
            "SELECT main_balance, audit_balance FROM wallet_state WHERE agent_id = ?",
            (inventory.agent_id,),
        )
        assert tuple(wallet) == ("7.0", "0.0")
        wallet_rows = await db.fetchall(
            """SELECT transaction_type, currency, amount, memo, new_balance
               FROM wallet_transactions WHERE agent_id = ?""",
            (inventory.agent_id,),
        )
        assert wallet_rows == [("deposit", "FIL", "7.0", "new tx", "7.0")]

        new_user = namespace_imported_graph_node(
            inventory.agent_id, "new-user"
        )
        new_skill = namespace_imported_graph_node(
            inventory.agent_id, "new-skill"
        )
        graph_rows = await db.fetchall(
            "SELECT node_id, node_type FROM graph_nodes "
            "WHERE node_id IN (?, ?, ?, ?, ?)",
            (
                inventory.old_user_node,
                inventory.old_skill_node,
                inventory.old_migration_node,
                new_user,
                new_skill,
            ),
        )
        assert set(graph_rows) == {
            (inventory.old_migration_node, "migration_record"),
            (new_user, "user"),
            (new_skill, "skill"),
        }

        migration_rows = await db.fetchall(
            """SELECT target_id FROM graph_edges
               WHERE source_id = ? AND label = 'migrated_via'""",
            (inventory.agent_id,),
        )
        assert set(migration_rows) == {
            (inventory.old_migration_node,),
            (result.migration_id,),
        }
        other = await db.fetchone(
            "SELECT title FROM memory_episodes WHERE agent_id = ?",
            (inventory.other_agent_id,),
        )
        assert other[0] == "other episode"
    finally:
        await _cleanup(db, inventory)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_invalid_required_record_is_rejected_before_replace(db_backend):
    db = await _database(db_backend)
    inventory = await _seed_old_inventory(db)
    package = _replacement_package(inventory.agent_id)
    package.skills[0].skill_id = ""
    before = await _snapshot(db, inventory)

    try:
        result = await IdentityImporter(
            db, target_agent_id=inventory.agent_id
        ).import_package(
            package,
            verify_signature=False,
            verify_constitution=False,
            allow_unsigned=True,
            merge_mode="replace",
        )

        assert result.success is False
        assert "validation failed" in result.errors[0].lower()
        assert await _snapshot(db, inventory) == before
    finally:
        await _cleanup(db, inventory)
