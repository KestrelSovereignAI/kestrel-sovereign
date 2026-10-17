"""Regression tests for #3618 — concurrent first LLM calls open the usage DB once.

``UsageTrackingMixin._ensure_db_initialized`` had no lock. Two LLM calls on a
service whose usage DB was not open yet both saw ``_db_initialized == False``
and both ran ``AsyncDatabase.sqlite`` on the same ``llm_usage.db``: the startup
migrations ran against one file twice at once (``database is locked``, and a CI
hang in ``test_concurrent_requests_do_not_cross_attribute_on_fallback_path``),
and the second assignment to ``_usage_db`` dropped the first connection without
closing it.

The SQLite cases use the real ``AsyncDatabase`` on a temp dir, counting calls
into its factory. The PostgreSQL case replaces ``AsyncDatabase.postgres``: the
real factory runs the core startup sequence against the test database, which
the lock under test does not need.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.llm.usage_tracking import UsageTrackingMixin
from kestrel_sovereign.storage.async_database import AsyncDatabase

POSTGRES_URL = "postgresql://unused/usage"


@pytest.fixture(autouse=True)
def _clear_usage_env(monkeypatch):
    """Usage-DB selection reads these; a developer's environment must not decide."""
    from kestrel_sovereign.storage.db import sqlite

    # These tests assert connection sharing/ownership, not the one-second
    # physical-retirement deadline. A busy xdist worker can delay an otherwise
    # successful real close beyond that incidental default. Keep a finite local
    # test budget; native close/debt deadline tests and production are unchanged.
    monkeypatch.setattr(sqlite, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", 10.0)
    for var in ("KESTREL_DB_PATH", "KESTREL_DATABASE_URL", "DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)


async def _let_other_callers_run() -> None:
    """Yield until every other gathered caller has run up to its next await.

    Without the lock each of them would reach the database factory here.
    """
    for _ in range(10):
        await asyncio.sleep(0)


class _SQLiteOpenSpy:
    """Wraps the real ``AsyncDatabase.sqlite``: counts calls, keeps every DB."""

    def __init__(self, monkeypatch):
        self.calls = 0
        self.opened: list[AsyncDatabase] = []
        self._real = AsyncDatabase.sqlite
        monkeypatch.setattr(AsyncDatabase, "sqlite", self._open)

    async def _open(self, db_path, **kwargs):
        self.calls += 1
        await _let_other_callers_run()
        db = await self._real(db_path, **kwargs)
        self.opened.append(db)
        return db

    def live(self) -> list[AsyncDatabase]:
        return [db for db in self.opened if db.backend.is_connected]

    async def close_leftovers(self) -> None:
        for db in self.live():
            await db.close()


def _bare_sqlite_tracker(tmp_path) -> UsageTrackingMixin:
    """The issue's repro: built with ``__new__``, no ``_init_usage_tracking``."""
    tracker = UsageTrackingMixin.__new__(UsageTrackingMixin)
    tracker._db_initialized = False
    tracker._usage_db = None
    tracker._db_backend = "sqlite"
    tracker._usage_db_dir = str(tmp_path)
    tracker._usage_db_path = os.path.join(str(tmp_path), "llm_usage.db")
    return tracker


@pytest.mark.parametrize("callers", [2, 10])
async def test_concurrent_first_calls_open_the_sqlite_db_once(
    tmp_path, monkeypatch, callers
):
    spy = _SQLiteOpenSpy(monkeypatch)
    tracker = _bare_sqlite_tracker(tmp_path)
    try:
        await asyncio.gather(
            *(tracker._ensure_db_initialized() for _ in range(callers))
        )

        assert spy.calls == 1
        assert tracker._db_initialized
        assert spy.live() == [tracker._usage_db]

        await tracker.close_usage_db()
        assert spy.live() == []
    finally:
        await spy.close_leftovers()


async def test_concurrent_first_usage_writes_share_one_connection(
    tmp_path, monkeypatch
):
    """The path the CI hang took: concurrent LLM calls each recording usage."""
    spy = _SQLiteOpenSpy(monkeypatch)
    tracker = UsageTrackingMixin()
    tracker._init_usage_tracking(agent_data_dir=tmp_path)
    try:
        await asyncio.gather(
            *(tracker._track_model_usage("model-a", "provider-a", 1) for _ in range(10))
        )

        assert spy.calls == 1
        row = await tracker._usage_db.fetchone(
            "SELECT use_count, total_tokens FROM model_usage WHERE model_id = ?",
            ("model-a",),
        )
        assert tuple(row) == (10, 10)

        await tracker.close_usage_db()
        assert spy.live() == []
    finally:
        await spy.close_leftovers()


async def test_concurrent_first_calls_open_the_postgres_db_once(monkeypatch):
    calls: list[str] = []
    opened: list[AsyncMock] = []

    async def postgres(dsn, **kwargs):
        calls.append(dsn)
        await _let_other_callers_run()
        db = AsyncMock()
        opened.append(db)
        return db

    monkeypatch.setattr(AsyncDatabase, "postgres", postgres)
    tracker = UsageTrackingMixin()
    tracker._init_usage_tracking(database_url=POSTGRES_URL)

    await asyncio.gather(*(tracker._ensure_db_initialized() for _ in range(10)))

    assert calls == [POSTGRES_URL]
    assert tracker._db_initialized
    assert tracker._usage_db is opened[0]

    await tracker.close_usage_db()
    opened[0].close.assert_awaited_once()
    assert tracker._usage_db is None
    assert not tracker._db_initialized


async def test_failed_first_open_is_retried_once_by_later_calls(
    tmp_path, monkeypatch
):
    spy = _SQLiteOpenSpy(monkeypatch)
    tracker = _bare_sqlite_tracker(tmp_path)
    # SQLite cannot open a directory, so the first open really fails.
    blocker = tmp_path / "llm_usage.db"
    blocker.mkdir()
    try:
        await tracker._ensure_db_initialized()

        assert spy.calls == 1
        assert not tracker._db_initialized
        assert tracker._usage_db is None
        assert spy.opened == []

        blocker.rmdir()
        await asyncio.gather(*(tracker._ensure_db_initialized() for _ in range(10)))

        assert spy.calls == 2
        assert tracker._db_initialized
        assert spy.live() == [tracker._usage_db]
    finally:
        await tracker.close_usage_db()
        await spy.close_leftovers()


async def test_close_during_an_open_closes_the_connection_it_opens(
    tmp_path, monkeypatch
):
    """A close racing the open must not let the open publish a connection
    after the close found nothing to close."""
    real_sqlite = AsyncDatabase.sqlite
    entered = asyncio.Event()
    release = asyncio.Event()
    opened: list[AsyncDatabase] = []

    async def gated_sqlite(db_path, **kwargs):
        entered.set()
        await release.wait()
        db = await real_sqlite(db_path, **kwargs)
        opened.append(db)
        return db

    monkeypatch.setattr(AsyncDatabase, "sqlite", gated_sqlite)
    tracker = _bare_sqlite_tracker(tmp_path)
    opening = asyncio.create_task(tracker._ensure_db_initialized())
    try:
        await entered.wait()
        closing = asyncio.create_task(tracker.close_usage_db())
        await _let_other_callers_run()
        assert not closing.done()

        release.set()
        await asyncio.gather(opening, closing)

        assert len(opened) == 1
        assert not opened[0].backend.is_connected
        assert tracker._usage_db is None
        assert not tracker._db_initialized
    finally:
        release.set()
        await asyncio.gather(opening, return_exceptions=True)
        for db in opened:
            if db.backend.is_connected:
                await db.close()


async def test_initialized_fast_path_does_not_wait_for_the_lock(tmp_path):
    tracker = _bare_sqlite_tracker(tmp_path)
    tracker._db_initialized = True
    lock = tracker._get_usage_db_lock()
    await lock.acquire()
    ensure = tracker._ensure_db_initialized()
    try:
        # Completing on the first step means it never awaited the held lock.
        with pytest.raises(StopIteration):
            ensure.send(None)
    finally:
        ensure.close()
        lock.release()
