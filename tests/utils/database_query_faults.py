"""Native SQLite contention and PostgreSQL serialization fault injection."""

import sqlite3

from kestrel_sovereign.storage.db.interface import DatabaseBackend


async def prepare_transient_database_query_fault(backend: DatabaseBackend):
    """Return a producer of real driver errors in this owned test database.

    SQLite uses a second physical writer. PostgreSQL raises its documented
    serialization SQLSTATE from the server. Neither path fabricates a wrapper,
    a commit outcome or a successful database result.
    """
    await backend.execute(
        "CREATE TABLE native_resident_retry_probe (value INTEGER NOT NULL)"
    )

    async def fail_query():
        if backend.backend_type == "sqlite":
            timeout = await backend.fetch_val("PRAGMA busy_timeout")
            await backend.execute("PRAGMA busy_timeout = 1")
            blocker = sqlite3.connect(backend.db_path)
            try:
                blocker.execute("BEGIN IMMEDIATE")
                await backend.execute(
                    "UPDATE native_resident_retry_probe SET value = value"
                )
            finally:
                blocker.rollback()
                blocker.close()
                await backend.execute(f"PRAGMA busy_timeout = {int(timeout)}")
        elif backend.backend_type == "postgres":
            await backend.execute(
                "DO $native_retry$ BEGIN RAISE EXCEPTION 'native retry probe' "
                "USING ERRCODE = '40001'; END $native_retry$"
            )
        else:
            raise AssertionError("native retry tests require SQLite or PostgreSQL")
        raise AssertionError("the real database did not produce the required fault")

    return fail_query
