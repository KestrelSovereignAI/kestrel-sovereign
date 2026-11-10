"""#4002: a stored ``core=5`` bundle is upgraded in place; older ones still refuse.

#3995 added ``a2a_tool_dispatches.dispatch_id`` and ``parent_dispatch_id`` to
the CORE stores without moving ``NativeStoreSchema.CORE``. ``TaskManager``
skips the CORE initializers for a database already at the runtime revision,
so no existing database received the columns, and the required dispatch audit
#3999 introduced failed every tool call.

CORE is now revision 6, and revision 5 has an additive online upgrade: under
the schema lease, ``TaskManager.initialize`` re-runs the idempotent CORE
initializers and advances the stored row. No row is converted; every other
stored revision still requires the offline migration.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from kestrel_sovereign.a2a.native_schema import (
    ADDITIVE_ONLINE_UPGRADES,
    NativeA2AMigrationRequired,
    NativeStoreSchema,
    publish_native_store_schema,
    require_native_a2a_schema,
)
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.observability_store import ToolDispatchEntry
from kestrel_sovereign.storage.db.interface import TransactionError
from tests.shared import native_store_schema
from tests.shared.agent_store_schema import load, snapshot_path
from tests.utils.native_a2a import manager_for
from tests.utils.native_push import configure_push_test_key

SCOPE = NativeRecipientScope("did:recipient")
DISPATCH_COLUMNS = {"dispatch_id", "parent_dispatch_id"}


async def _columns(backend) -> set[str]:
    if backend.backend_type == "sqlite":
        rows = await backend.fetch_all(
            "SELECT name FROM pragma_table_info('a2a_tool_dispatches')"
        )
    else:
        rows = await backend.fetch_all(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = 'a2a_tool_dispatches'"
        )
    return {row[0] for row in rows}


async def _revisions(backend) -> set[tuple[str, int]]:
    return set(
        await backend.fetch_all(
            "SELECT component, revision FROM a2a_native_store_schema"
        )
    )


async def _store_before_3995(factory, *, core_revision: int):
    """A completed CORE bundle exactly as Core 0.63.0 left it, plus one audit row.

    The current initializers build the bundle; the #3995 indexes and columns
    are then removed and the stored revision set back.
    """
    backend = factory()
    await backend.connect()
    manager = manager_for(backend, SCOPE, owns_backend=False)
    try:
        await manager.initialize()
        for index in ("idx_tool_dispatches_dispatch", "idx_tool_dispatches_parent"):
            await backend.execute(f"DROP INDEX IF EXISTS {index}")
        for column in ("parent_dispatch_id", "dispatch_id"):
            await backend.execute(
                f"ALTER TABLE a2a_tool_dispatches DROP COLUMN {column}"
            )
        await backend.execute(
            "UPDATE a2a_native_store_schema SET revision = ? WHERE component = 'core'",
            (core_revision,),
        )
        await backend.execute(
            "INSERT INTO a2a_tool_dispatches (agent_did, session_id, turn_id, "
            "tool_name, adapter, args_redacted, result_status, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("did:recipient", None, "turn-old", "list_peers", "direct", "{}", "success", 3),
        )
    finally:
        await manager.close()
        await backend.close()


def _released_dispatch_columns(tmp_path) -> list[str]:
    db_path = tmp_path / "released.db"
    load(db_path, snapshot_path("0.63.0"))
    connection = sqlite3.connect(db_path)
    try:
        return [
            row[1]
            for row in connection.execute("PRAGMA table_info(a2a_tool_dispatches)")
        ]
    finally:
        connection.close()


def _entry(dispatch_id: str) -> ToolDispatchEntry:
    return ToolDispatchEntry(
        agent_did="did:recipient",
        session_id=None,
        turn_id="turn-new",
        tool_name="wait_status",
        adapter="direct",
        args_redacted={"limit": 5},
        result_status="success",
        error_class=None,
        error_message=None,
        latency_ms=1,
        result_size_bytes=None,
        dispatch_id=dispatch_id,
    )


async def test_core_5_store_gains_the_dispatch_columns_at_initialization(
    native_a2a_backend_factory, tmp_path
):
    await _store_before_3995(native_a2a_backend_factory, core_revision=5)
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(backend, SCOPE, owns_backend=False)
    try:
        before = await _columns(backend)
        assert not before & DISPATCH_COLUMNS
        if backend.backend_type == "sqlite":
            # The aged table is the released one, column for column.
            assert sorted(before) == sorted(_released_dispatch_columns(tmp_path))
        # The read-only preflight the deploy gate runs accepts the store.
        await require_native_a2a_schema(backend)
        assert ("core", 5) in await _revisions(backend)

        await manager.initialize()

        assert await _columns(backend) >= DISPATCH_COLUMNS
        assert await _revisions(backend) == {NativeStoreSchema.CORE.value}
        await manager.observability_store.log_tool_dispatch(
            _entry("dispatch-new"), required=True
        )
        rows = await backend.fetch_all(
            "SELECT turn_id, tool_name, dispatch_id, parent_dispatch_id "
            "FROM a2a_tool_dispatches ORDER BY id"
        )
        # The row written before the upgrade is kept as it was.
        assert rows == [
            ("turn-old", "list_peers", None, None),
            ("turn-new", "wait_status", "dispatch-new", None),
        ]
        health = manager.observability_store.tool_dispatch_audit_health
        assert health.failures == 0 and health.last_success_at is not None
    finally:
        await manager.close()
        await backend.close()


@pytest.mark.parametrize("revision", [1, 2, 3, 4])
async def test_core_revisions_before_5_still_require_offline_migration(
    native_a2a_backend_factory, revision
):
    await _store_before_3995(native_a2a_backend_factory, core_revision=revision)
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(backend, SCOPE, owns_backend=False)
    try:
        with pytest.raises(NativeA2AMigrationRequired, match="not supported"):
            await manager.initialize()
        assert not await _columns(backend) & DISPATCH_COLUMNS
        assert await _revisions(backend) == {("core", revision)}
    finally:
        await manager.close()
        await backend.close()


async def test_concurrent_managers_upgrade_one_core_5_store_together(
    native_a2a_backend_factory,
):
    await _store_before_3995(native_a2a_backend_factory, core_revision=5)
    backends = [native_a2a_backend_factory(), native_a2a_backend_factory()]
    for backend in backends:
        await backend.connect()
    managers = [
        manager_for(backend, NativeRecipientScope(did), owns_backend=False)
        for backend, did in zip(backends, ("did:first", "did:second"))
    ]
    try:
        await asyncio.gather(*(manager.initialize() for manager in managers))
        assert await _revisions(backends[0]) == {NativeStoreSchema.CORE.value}
        assert await _columns(backends[0]) >= DISPATCH_COLUMNS
    finally:
        for manager, backend in zip(managers, backends):
            await manager.close()
            await backend.close()


async def test_core_upgrade_keeps_an_enabled_push_bundle(
    native_a2a_backend_factory, monkeypatch
):
    from kestrel_sovereign.a2a.native_schema import native_store_schema_ready
    from kestrel_sovereign.a2a.push_outbox import NativePushOutbox
    from kestrel_sovereign.a2a.stores.unified.push_config_store import PushConfigStore
    from kestrel_sovereign.signals.durable import DurableSignalStore

    configure_push_test_key(monkeypatch)

    async def push_bundle(backend, manager):
        signals = DurableSignalStore(backend)
        await signals.initialize()
        await NativePushOutbox(
            PushConfigStore(manager.task_store),
            signals,
            retention_days=30,
            lease_seconds=60,
        ).initialize()

    async def task_triggers(backend) -> set[str]:
        if backend.backend_type == "sqlite":
            rows = await backend.fetch_all(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND tbl_name = 'a2a_tasks'"
            )
        else:
            rows = await backend.fetch_all(
                "SELECT t.tgname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = current_schema() AND c.relname = 'a2a_tasks' "
                "AND NOT t.tgisinternal"
            )
        return {row[0] for row in rows}

    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(backend, SCOPE, owns_backend=False)
    try:
        await manager.initialize()
        await push_bundle(backend, manager)
        triggers = await task_triggers(backend)
        await backend.execute(
            "UPDATE a2a_native_store_schema SET revision = 5 WHERE component = 'core'"
        )
    finally:
        await manager.close()
        await backend.close()

    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(backend, SCOPE, owns_backend=False)
    try:
        await manager.initialize()
        # Push capture and the terminal fences are exactly as push left them.
        assert await task_triggers(backend) == triggers
        assert triggers & {"a2a_task_changes_update_v2", "a2a_task_changes_v2"}
        await push_bundle(backend, manager)
        assert await native_store_schema_ready(backend, NativeStoreSchema.PUSH)
        assert await _revisions(backend) == {
            NativeStoreSchema.CORE.value,
            NativeStoreSchema.PUSH.value,
        }
    finally:
        await manager.close()
        await backend.close()


async def test_publication_never_inserts_over_a_stored_revision(tmp_path):
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    backend = SQLiteBackend(str(tmp_path / "publish.db"))
    await backend.connect()
    try:
        assert await publish_native_store_schema(backend, NativeStoreSchema.PUSH) is None
        # The lease's owning transaction reports its body's error as the cause.
        with pytest.raises(TransactionError) as caught:
            await publish_native_store_schema(backend, NativeStoreSchema.PUSH)
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert "already published" in str(caught.value.__cause__)
        await backend.execute(
            "UPDATE a2a_native_store_schema SET revision = 3 WHERE component = 'push'"
        )
        with pytest.raises(TransactionError) as caught:
            await publish_native_store_schema(backend, NativeStoreSchema.PUSH)
        assert isinstance(caught.value.__cause__, NativeA2AMigrationRequired)
        assert "no online upgrade" in str(caught.value.__cause__)
        await backend.execute("INSERT INTO a2a_native_store_schema VALUES ('core', 5)")
        assert await publish_native_store_schema(backend, NativeStoreSchema.CORE) == 5
        assert await _revisions(backend) == {("push", 3), NativeStoreSchema.CORE.value}
    finally:
        await backend.close()


def test_online_upgrades_name_only_older_revisions_of_known_components():
    revisions = dict(component.value for component in NativeStoreSchema)
    for name, older in ADDITIVE_ONLINE_UPGRADES.items():
        assert name in revisions, name
        assert older and all(0 < revision < revisions[name] for revision in older)


async def test_store_ddl_cannot_change_without_its_schema_revision(
    tmp_path, monkeypatch
):
    """Startup skips a component's initializers once its revision is stored.

    A change to what a component creates, kept at the same revision, would
    reach only databases born on the new code. Bump the component's revision
    in ``NativeStoreSchema``, record how stored revisions cross it
    (``ADDITIVE_ONLINE_UPGRADES`` or ``data/offline_migrations.toml``), then
    regenerate the snapshot (see ``tests/shared/native_store_schema.py``).
    """
    configure_push_test_key(monkeypatch)
    current = await native_store_schema.capture(tmp_path)
    pinned = native_store_schema.load_snapshot()
    assert sorted(current) == sorted(pinned)
    for name, snapshot in current.items():
        if snapshot == pinned[name]:
            continue
        shape = {key: value for key, value in snapshot.items() if key != "revision"}
        pinned_shape = {
            key: value for key, value in pinned[name].items() if key != "revision"
        }
        if shape != pinned_shape and snapshot["revision"] == pinned[name]["revision"]:
            pytest.fail(
                f"The {name} store DDL changed but its revision is still "
                f"{snapshot['revision']}: existing databases skip the {name} "
                "initializers and will never receive the change (#4002). Bump "
                "the revision, record how stored revisions cross it, then "
                "regenerate tests/fixtures/native_store_schema.json."
            )
        pytest.fail(
            f"The {name} store schema revision moved without its snapshot: "
            "regenerate tests/fixtures/native_store_schema.json."
        )
