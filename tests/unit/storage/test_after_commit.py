"""Work deferred until the owning transaction commits (#2913).

``AsyncDatabase.call_after_commit`` runs a callback once the current task's
writes so far have committed: now when no transaction is open, after the
outermost transaction commits when one is, and never when it rolls back. A
savepoint that rolls back discards what was registered inside it; a joined
nested scope shares its parent's outcome.

The cases run on a real SQLite database everywhere, and on PostgreSQL when
``TEST_POSTGRES_URL`` is set.
"""

from __future__ import annotations

import asyncio
import logging
import os

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import TransactionError
from kestrel_sovereign.storage.db import after_commit as after_commit_module
from kestrel_sovereign.storage.db.after_commit import AfterCommitCallbacks


class _RollBack(Exception):
    """Raised inside a transaction so that it rolls back."""


async def _no_schema(db):
    """Schema initializer for a connection that needs no tables."""


# Parametrized with "postgres" so the #3381 guard fails the PostgreSQL case,
# rather than letting it skip, in a job that provides PostgreSQL.
@pytest.fixture(params=["sqlite", "postgres"])
async def db(request, tmp_path):
    if request.param == "postgres":
        url = os.environ.get("TEST_POSTGRES_URL")
        if not url:
            pytest.skip("TEST_POSTGRES_URL is not set")
        database = await AsyncDatabase.postgres(url, schema_initializer=_no_schema)
    else:
        database = await AsyncDatabase.sqlite(
            str(tmp_path / "after_commit.db"), schema_initializer=_no_schema
        )
    try:
        yield database
    finally:
        await database.close()


def _recorder(db, ran, label):
    """A callback that records *label* and whether a transaction was open."""

    def callback():
        ran.append((label, db.owns_open_transaction))

    return callback


async def test_with_no_transaction_open_it_runs_now(db):
    ran = []
    assert db.call_after_commit(_recorder(db, ran, "now")) is True
    assert ran == [("now", False)]


async def test_it_runs_after_the_transaction_commits(db):
    ran = []
    async with db.transaction():
        assert db.call_after_commit(_recorder(db, ran, "first")) is True
        db.call_after_commit(_recorder(db, ran, "second"))
        assert ran == []
    # In registration order, with the transaction closed.
    assert ran == [("first", False), ("second", False)]


async def test_it_never_runs_when_the_transaction_rolls_back(db):
    ran = []
    with pytest.raises(TransactionError):
        async with db.transaction():
            db.call_after_commit(_recorder(db, ran, "rolled back"))
            raise _RollBack()
    assert ran == []

    # Nothing registered by the rolled-back transaction leaks into the next.
    async with db.transaction():
        db.call_after_commit(_recorder(db, ran, "next"))
    assert ran == [("next", False)]


async def test_a_rolled_back_savepoint_discards_only_its_own_callbacks(db):
    ran = []
    async with db.transaction():
        db.call_after_commit(_recorder(db, ran, "outer"))
        with pytest.raises(_RollBack):
            async with db.transaction(savepoint=True):
                db.call_after_commit(_recorder(db, ran, "undone"))
                raise _RollBack()
        async with db.transaction(savepoint=True):
            db.call_after_commit(_recorder(db, ran, "released"))
        assert ran == []
    assert ran == [("outer", False), ("released", False)]


async def test_a_released_savepoint_waits_for_the_outer_commit(db):
    ran = []
    with pytest.raises(TransactionError):
        async with db.transaction():
            async with db.transaction(savepoint=True):
                db.call_after_commit(_recorder(db, ran, "released"))
            raise _RollBack()
    assert ran == []


async def test_a_joined_scope_shares_its_parents_outcome(tmp_path):
    """SQLite joins a plain nested scope: its writes commit with the parent's,
    even when the nested scope raised and the parent carried on."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "joined.db"))
    try:
        assert db.nested_transaction_strategy == "joined"
        ran = []
        async with db.transaction():
            with pytest.raises(_RollBack):
                async with db.transaction():
                    db.call_after_commit(_recorder(db, ran, "joined"))
                    raise _RollBack()
            assert ran == []
        assert ran == [("joined", False)]
    finally:
        await db.close()


async def test_a_task_that_does_not_own_the_transaction_runs_now(db):
    """A child task inherits the context, not the transaction."""
    ran = []

    async def child():
        db.call_after_commit(_recorder(db, ran, "child"))

    async with db.transaction():
        await asyncio.get_running_loop().create_task(child())
        assert ran == [("child", False)]
        db.call_after_commit(_recorder(db, ran, "owner"))
    assert ran == [("child", False), ("owner", False)]


async def test_a_failing_callback_is_logged_and_the_commit_stands(db, caplog):
    ran = []

    def fails():
        raise RuntimeError("callback failed")

    with caplog.at_level(logging.ERROR, logger=after_commit_module.logger.name):
        async with db.transaction():
            db.call_after_commit(fails)
            db.call_after_commit(_recorder(db, ran, "after the failure"))

    assert ran == [("after the failure", False)]
    [record] = [
        r for r in caplog.records if r.name == after_commit_module.logger.name
    ]
    assert record.levelno == logging.ERROR
    assert record.exc_info[1].args == ("callback failed",)


async def _swallow_a_failed_statement(db, ran):
    """Open a transaction, fail one statement inside it, carry on, register."""
    async with db.transaction():
        with pytest.raises(Exception, match="no_such_table_2913"):
            await db.execute("SELECT * FROM no_such_table_2913", ())
        db.call_after_commit(_recorder(db, ran, "after the failure"))


@pytest.mark.parametrize("session", ["pooled", "operational"])
async def test_a_swallowed_failure_runs_the_callback_only_if_the_commit_is_real(
    db, session
):
    """PostgreSQL aborts a transaction on any failed statement, and COMMIT on an
    aborted transaction rolls back without an error. A transaction holding
    after-commit work therefore fails instead, and the work never runs.
    SQLite undoes only the failed statement and commits the rest."""
    if session == "operational" and db.backend_type != "postgres":
        pytest.skip("operational sessions are PostgreSQL-only")
    ran = []
    if db.backend_type == "postgres":
        with pytest.raises(TransactionError, match="current transaction is aborted"):
            if session == "operational":
                async with db._backend.operational_session():  # noqa: SLF001 - backend transaction path
                    await _swallow_a_failed_statement(db, ran)
            else:
                await _swallow_a_failed_statement(db, ran)
        assert ran == []
    else:
        await _swallow_a_failed_statement(db, ran)
        assert ran == [("after the failure", False)]


async def test_a_transaction_on_a_pinned_operational_session_runs_after_commit(db):
    """PostgreSQL's pinned-connection path opens its transaction separately."""
    if db.backend_type != "postgres":
        pytest.skip("operational sessions are PostgreSQL-only")
    ran = []
    async with db._backend.operational_session():  # noqa: SLF001 - backend transaction path
        async with db.transaction():
            db.call_after_commit(_recorder(db, ran, "committed"))
            assert ran == []
        assert ran == [("committed", False)]

        with pytest.raises(TransactionError):
            async with db.transaction():
                db.call_after_commit(_recorder(db, ran, "rolled back"))
                raise _RollBack()
    assert ran == [("committed", False)]


async def test_a_pool_adapter_that_skips_backend_init_runs_after_commit(db):
    """``PoolBackendAdapter`` wraps a pool without ``PostgresBackend.__init__``;
    its transactions must keep working and defer like any other."""
    if db.backend_type != "postgres":
        pytest.skip("PoolBackendAdapter is PostgreSQL-only")
    from kestrel_sovereign.a2a.stores.postgres import PoolBackendAdapter

    adapter = PoolBackendAdapter(db._backend.operational_pool)  # noqa: SLF001 - shares the test's pool
    ran = []
    async with adapter.transaction():
        adapter.call_after_commit(lambda: ran.append("committed"))
        assert ran == []
    assert ran == ["committed"]


class _HooklessBackend:
    """A backend that cannot report its commits, only whether one is open."""

    backend_type = "sqlite"

    def __init__(self, owns):
        self.owns_open_transaction = owns


@pytest.mark.parametrize(
    ("owns", "runs"), [(False, True), (True, False), (None, False)]
)
def test_a_backend_without_the_hook_runs_only_with_no_transaction_open(owns, runs):
    ran = []
    db = AsyncDatabase(_HooklessBackend(owns))

    assert db.call_after_commit(lambda: ran.append("ran")) is runs
    assert ran == (["ran"] if runs else [])


def test_frames_merge_on_release_and_drop_on_rollback():
    ran = []
    callbacks = AfterCommitCallbacks()
    callbacks.register(lambda: ran.append("outer"))
    callbacks.open_savepoint()
    callbacks.register(lambda: ran.append("released"))
    callbacks.open_savepoint()
    callbacks.register(lambda: ran.append("undone"))
    callbacks.roll_back_savepoint()
    callbacks.release_savepoint()

    callbacks.run()
    assert ran == ["outer", "released"]
    # Run once: a second call has nothing left to run.
    callbacks.run()
    assert ran == ["outer", "released"]
