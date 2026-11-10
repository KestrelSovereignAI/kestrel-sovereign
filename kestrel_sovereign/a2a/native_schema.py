"""Explicit stored-protocol boundary; runtime startup never converts draft data."""

from contextlib import asynccontextmanager
from enum import Enum
from types import MappingProxyType

from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase


class NativeA2AMigrationRequired(RuntimeError):
    """The primary store must pass the offline A2A cutover before startup."""


class NativeStoreSchema(Enum):
    """Versioned schema units owned and atomically installed by TaskManager.

    These revisions describe completed store DDL, separately from the A2A
    payload encoding. A standalone TaskStore never proves the other stores.
    """

    CORE = ("core", 6)
    MEMORY = ("memory", 1)
    FEEDBACK = ("feedback", 1)
    PUSH = ("push", 4)


#: Stored revisions TaskManager finishes in place, under the schema lease.
#:
#: Every entry is additive DDL only: re-running the component's idempotent
#: store initializers carries the stored tables to the runtime revision and
#: converts no row (#4002). A revision absent here, and every data conversion,
#: stays an offline migration. A component whose DDL changes needs a new
#: revision AND a decision recorded here or in ``data/offline_migrations.toml``;
#: otherwise existing stores never receive the change.
ADDITIVE_ONLINE_UPGRADES = MappingProxyType(
    {
        # 6: a2a_tool_dispatches.dispatch_id and parent_dispatch_id (#3995).
        NativeStoreSchema.CORE.value[0]: frozenset({5}),
    }
)


def _upgrades_online(component: str, revision: int) -> bool:
    return revision in ADDITIVE_ONLINE_UPGRADES.get(component, frozenset())


async def _require_native_input_schema(backend) -> None:
    """An earlier native projection must not silently lose its wire identity."""
    if not await backend.table_exists("a2a_task_inputs"):
        return
    if backend.backend_type == "sqlite":
        row = await backend.fetch_one(
            "SELECT 1 FROM pragma_table_info('a2a_task_inputs') WHERE name = 'wire_request_digest'"
        )
    elif backend.backend_type == "postgres":
        row = await backend.fetch_one(
            "SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() "
            "AND table_name = 'a2a_task_inputs' AND column_name = 'wire_request_digest'"
        )
    else:
        raise ValueError("Native A2A requires a SQLite or PostgreSQL backend")
    if row is None:
        raise NativeA2AMigrationRequired(
            "Stored native A2A input projection schema requires offline migration"
        )


async def _require_native_store_revisions(backend) -> bool:
    if not await backend.table_exists("a2a_native_store_schema"):
        return False
    rows = await backend.fetch_all(
        "SELECT component, revision FROM a2a_native_store_schema"
    )
    expected = dict(component.value for component in NativeStoreSchema)
    if any(
        name not in expected
        or (revision != expected[name] and not _upgrades_online(name, revision))
        for name, revision in rows
    ):
        raise NativeA2AMigrationRequired(
            "Stored native A2A store schema revision is not supported by this runtime"
        )
    if rows and not any(name == NativeStoreSchema.CORE.value[0] for name, _ in rows):
        raise NativeA2AMigrationRequired(
            "Native A2A store schema has no completed core bundle"
        )
    return bool(rows)


async def native_store_schema_ready(backend, component: NativeStoreSchema) -> bool:
    """Read durable DDL completion while the caller holds the schema lease."""
    if not await backend.table_exists("a2a_native_store_schema"):
        return False
    row = await backend.fetch_one(
        "SELECT revision FROM a2a_native_store_schema WHERE component = ?",
        (component.value[0],),
    )
    return row == (component.value[1],)


async def publish_native_store_schema(
    backend, component: NativeStoreSchema
) -> int | None:
    """Record completion in the same transaction as the component's DDL.

    Only the TaskManager bundle initializer publishes this proof, after every
    store in the unit succeeded. Optional components share its outer transaction.
    A stored revision listed in ``ADDITIVE_ONLINE_UPGRADES`` is advanced in
    place rather than inserted over. Returns the revision it replaced, or
    ``None`` when the component had no stored revision.
    """
    name, revision = component.value
    async with native_schema_write(backend):
        await backend.execute_script("""
            CREATE TABLE IF NOT EXISTS a2a_native_store_schema (
                component TEXT PRIMARY KEY,
                revision INTEGER NOT NULL CHECK (revision > 0)
            )
        """)
        row = await backend.fetch_one(
            "SELECT revision FROM a2a_native_store_schema WHERE component = ?",
            (name,),
        )
        if row is None:
            await backend.execute(
                "INSERT INTO a2a_native_store_schema (component, revision) VALUES (?, ?)",
                component.value,
            )
            return None
        (stored,) = row
        if stored == revision:
            raise RuntimeError(
                f"Native A2A {name} store schema revision {revision} is already published"
            )
        if not _upgrades_online(name, stored):
            raise NativeA2AMigrationRequired(
                f"Stored native A2A {name} store schema revision {stored} has no "
                f"online upgrade to revision {revision}"
            )
        updated = await backend.execute(
            "UPDATE a2a_native_store_schema SET revision = ? "
            "WHERE component = ? AND revision = ?",
            (revision, name, stored),
        )
        if updated != 1:
            raise RuntimeError(
                f"Native A2A {name} store schema revision changed under the schema lease"
            )
        return stored


async def require_native_a2a_schema(backend) -> None:
    """Read-only preflight before any native store creates tables or writes rows."""
    for table in ("a2a_outbound_tasks", "pending_a2a_questions"):
        if (
            await backend.table_exists(table)
            and await backend.fetch_one(f"SELECT 1 FROM {table} LIMIT 1") is not None
        ):
            raise NativeA2AMigrationRequired(
                f"Draft A2A data remains in {table}; offline migration is required after draining and before native startup"
            )
    completed_stores = await _require_native_store_revisions(backend)
    await _require_native_input_schema(backend)
    if await backend.table_exists("a2a_protocol_schema"):
        row = await backend.fetch_one(
            "SELECT revision FROM a2a_protocol_schema WHERE singleton = 1"
        )
        if row != (1,):
            raise NativeA2AMigrationRequired(
                "Stored A2A protocol revision is not supported by this runtime"
            )
        return
    if completed_stores:
        raise NativeA2AMigrationRequired(
            "Completed native A2A stores have no protocol encoding revision"
        )
    if (
        await backend.table_exists("a2a_tasks")
        and await backend.fetch_one("SELECT 1 FROM a2a_tasks LIMIT 1") is not None
    ):
        raise NativeA2AMigrationRequired(
            "Unversioned A2A task data requires offline provenance migration; startup will not settle or reinterpret it"
        )


@asynccontextmanager
async def native_schema_write(backend):
    """Serialize the complete native schema bundle across independent agents."""
    store = UnifiedStoreBase(backend, owns_backend=False)
    async with store.write_transaction():
        if store.is_postgres:
            await backend.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
                ("a2a_native_protocol_schema_v1",),
            )
        yield


@asynccontextmanager
async def native_schema_initialization(backend):
    """Preflight without writes, then recheck under the shared DDL lease."""
    await require_native_a2a_schema(backend)
    conflict = None
    async with native_schema_write(backend):
        try:
            await require_native_a2a_schema(backend)
        except NativeA2AMigrationRequired as error:
            conflict = error
        else:
            yield
    if conflict is not None:
        raise conflict


async def publish_native_a2a_schema(backend) -> None:
    """Publish the data encoding after native schema creation or offline import."""
    async with native_schema_write(backend):
        await backend.execute_script("""
            CREATE TABLE IF NOT EXISTS a2a_protocol_schema (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                revision INTEGER NOT NULL CHECK (revision = 1)
            )
        """)
        await backend.execute(
            "INSERT INTO a2a_protocol_schema (singleton, revision) VALUES (1, 1) ON CONFLICT DO NOTHING"
        )
