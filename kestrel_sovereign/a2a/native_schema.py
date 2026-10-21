"""Explicit stored-protocol boundary; runtime startup never converts draft data."""

from contextlib import asynccontextmanager

from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase


class NativeA2AMigrationRequired(RuntimeError):
    """The primary store must pass the offline A2A cutover before startup."""


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
    if await backend.table_exists("a2a_protocol_schema"):
        row = await backend.fetch_one(
            "SELECT revision FROM a2a_protocol_schema WHERE singleton = 1"
        )
        if row != (1,):
            raise NativeA2AMigrationRequired(
                "Stored A2A protocol revision is not supported by this runtime"
            )
        return
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
