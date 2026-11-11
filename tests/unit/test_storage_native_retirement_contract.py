"""Native lifecycle authority remains distinct from privacy-aware data access."""

import pytest

from kestrel_sovereign.privacy import PrivacyMode
from tests.utils.aiosqlite_workers import aiosqlite_worker
import sys
from kestrel_sovereign.storage import AsyncStorage, PrivacyEnforcingStorage
from kestrel_sovereign.storage.sqla import make_session_factory
from kestrel_sovereign.storage.db import ConnectionError
from tests.utils.aiosqlite_workers import (
    normal_close_schedule, close_sqla_factory_on_a_normal_schedule,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED])
async def test_native_storage_retirement_contract_preserves_privacy(tmp_path, mode):
    raw_storage = AsyncStorage(str(tmp_path / "retirement-contract.db"))
    storage = PrivacyEnforcingStorage(raw_storage, mode)
    assert raw_storage.native_retirement_owner is None
    assert storage.native_retirement_owner is None
    try:
        await raw_storage.initialize()
        assert raw_storage.native_retirement_owner is raw_storage
        assert storage.native_retirement_owner is storage
        assert storage.native_retirement_owner is not raw_storage.db
    finally:
        await _retire_native_storage(storage)


async def _retire_native_storage(storage):
    raw_storage = storage._storage
    try:
        if raw_storage.db is not None:
            await close_sqla_factory_on_a_normal_schedule(
                make_session_factory(raw_storage.db),
            )
    finally:
        # Independent primary cleanup must run after factory acquisition or
        # close fails. Python retains both original errors through __context__.
        backend = raw_storage._backend
        connection = backend._connection
        close = storage.close if raw_storage.db is not None else backend.close
        if connection is None:
            await close()
        else:
            async with normal_close_schedule() as finish:
                await finish(
                    connection, close(),
                    retained_closes=backend._retired_connection_closes,
                )


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_fails", [False, True])
async def test_storage_cleanup_retires_primary_after_factory_failure(
    tmp_path, monkeypatch, primary_fails,
):
    raw = AsyncStorage(str(tmp_path / "cleanup-failure.db"))
    storage = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
    await raw.initialize()
    connection = raw._backend._connection
    worker = aiosqlite_worker(connection)
    factory_failure = OSError("original factory close epilogue")
    primary_failure = OSError("original primary close epilogue")
    original_factory_close = close_sqla_factory_on_a_normal_schedule
    original_primary_close = raw.close

    async def close_factory_then_fail(factory):
        await original_factory_close(factory)
        raise factory_failure

    async def close_primary_then_fail():
        await original_primary_close()
        raise primary_failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(sys.modules[__name__], "close_sqla_factory_on_a_normal_schedule", close_factory_then_fail)
            if primary_fails:
                patch.setattr(raw, "close", close_primary_then_fail)
            with pytest.raises(OSError) as caught:
                await _retire_native_storage(storage)
        assert not worker.is_alive(), "factory failure skipped the independent primary close"
        if primary_fails:
            assert caught.value is primary_failure
            assert caught.value.__context__ is factory_failure
        else:
            assert caught.value is factory_failure
    finally:
        # Explicit test-owner reconciliation also retires a deliberately broken guard.
        if worker.is_alive():
            await _retire_native_storage(storage)


@pytest.mark.asyncio
async def test_storage_cleanup_retires_partial_native_initialization(tmp_path, monkeypatch):
    import aiosqlite

    raw = AsyncStorage(str(tmp_path / "partial-initialization.db"))
    storage = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
    original_fetchall = aiosqlite.Connection.execute_fetchall

    async def fail_original_pragma(connection, sql, *args, **kwargs):
        if sql == "PRAGMA journal_mode=WAL":
            return await original_fetchall(connection, "SELECT * FROM missing_initialization_probe")
        return await original_fetchall(connection, sql, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(aiosqlite.Connection, "execute_fetchall", fail_original_pragma)
            with pytest.raises(ConnectionError, match="missing_initialization_probe"):
                await raw.initialize()
        assert raw.db is None
        connection = raw._backend._connection
        assert connection is not None
        worker = aiosqlite_worker(connection)
        assert worker.is_alive()
    finally:
        await _retire_native_storage(storage)
    assert not worker.is_alive()
    assert raw._backend._connection is None
