"""The canonical migration fixture retires its exact original real backend."""

from tests.conftest import sqlite_database_factory


async def test_sqlite_factory_joins_original_backend_before_returning_close(tmp_path):
    lifetime = sqlite_database_factory.__wrapped__()
    factory = await anext(lifetime)
    database = await factory(tmp_path / "original.db")
    backend = database.backend
    try:
        assert backend.is_connected
        await lifetime.aclose()
        assert not backend.is_connected, "SQLite factory left its original backend connected"
    finally:
        await lifetime.aclose()
        if backend.is_connected:
            await database.close()  # Failed test still owns this exact backend.
