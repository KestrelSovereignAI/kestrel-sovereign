"""Real CLI-owned SQLite opening/retirement and an actual native lease receipt.

The PostgreSQL fixture owns the original retired advisory lease. The CLI owns
its own event loop and real SQLite connection in a separate thread. Faults
follow actual opening, reads and physical connection close, never fake ACKs.
These are component/CLI proofs, not Agent READY or live operator acceptance.
"""

import asyncio
import builtins
import inspect
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest

from kestrel_sovereign import cli_embeddings
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.storage.async_database import (
    AsyncDatabase,
    DatabaseRetirementError,
)
from kestrel_sovereign.storage.db.sqlite import _aiosqlite_worker_is_alive
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_cli_embeddings import _agent_data_dir


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage", ["import", "open", "close", "body-close", "refuse-close"]
)
@pytest.mark.parametrize("kind", ["native-control", "ordinary", "healthy"])
async def test_actual_cli_database_owner_preserves_open_and_retirement(
    native_memory_owners, tmp_path, monkeypatch, stage, kind
):
    n = native_memory_owners
    data_dir = await asyncio.to_thread(
        _agent_data_dir, tmp_path, monkeypatch, legacy_column=False
    )
    actual_factory = AsyncDatabase.sqlite
    actual_create = asyncio.create_task
    actual_import = builtins.__import__
    owners, native_connections, sources, body_sources = [], [], [], []
    physical_close = []
    actual_closes = {}

    def original_control(target):
        try:
            n.lease.require_live()
        except BaseException as original:
            target.append(original)
            raise

    def fail(target):
        if kind == "native-control":
            original_control(target)
        if kind == "ordinary":
            error = OSError("ordinary local CLI observation failed")
            target.append(error)
            raise error

    def import_database(name, globals=None, locals=None, fromlist=(), level=0):
        module = actual_import(name, globals, locals, fromlist, level)
        caller = inspect.currentframe().f_back
        if (
            stage == "import"
            and name == "kestrel_sovereign.storage.async_database"
            and caller.f_code.co_name == "_runner"
            and "AsyncDatabase" in fromlist
        ):
            # The actual module lookup succeeded. Deliver the original native
            # admission receipt at that boundary, not a fabricated module.
            fail(sources)
        return module

    monkeypatch.setattr(builtins, "__import__", import_database)

    async def factory(cls, *args, **kwargs):
        db = await actual_factory(*args, **kwargs)
        owners.append(db)
        native = db.backend._connection
        native_connections.append(native)
        actual_closes[id(db)] = db.close
        if stage == "open" and kind != "healthy":
            # The actual initialized candidate is retired before this
            # deliberate post-open admission failure is published.
            await db.close()
            physical_close.append(native)
            fail(sources)
        if stage in {"close", "body-close"}:
            native_close = native.close

            async def close():
                await native_close()
                physical_close.append(native)
                # Lost observation after physical native retirement: the
                # original actual Task/resource must remain on the receipt.
                fail(sources)

            monkeypatch.setattr(native, "close", close)
        if stage == "body-close" and kind != "healthy":
            actual_fetch = db.fetchall
            fired = False

            async def fetch(sql, *args, **kwargs):
                nonlocal fired
                result = await actual_fetch(sql, *args, **kwargs)
                if "GROUP BY embedding_profile_id" in sql and not fired:
                    fired = True
                    fail(body_sources)
                return result

            monkeypatch.setattr(db, "fetchall", fetch)
        return db

    monkeypatch.setattr(AsyncDatabase, "sqlite", classmethod(factory))

    def publish(operation, *args, **kwargs):
        if (
            stage == "refuse-close"
            and kind != "healthy"
            and kwargs.get("name") == "owned-database-retirement"
        ):
            fail(sources)
        return actual_create(operation, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", publish)
    args = SimpleNamespace(
        embeddings_command="audit",
        data_dir=str(data_dir),
        agent_name=None,
        table="saved_items",
        agent_id=n.did,
    )
    failure = result = None
    try:
        try:
            result = await asyncio.to_thread(cli_embeddings.run, args)
        except BaseException as error:
            failure = error
        if stage == "import" and kind != "healthy":
            assert not owners and not native_connections
        else:
            assert owners and native_connections
        if kind == "healthy":
            assert failure is None and result == 0
        elif stage in {"import", "open"} and kind == "ordinary":
            assert failure is None and result == 2
        elif stage in {"import", "open"}:
            assert sources and failure is sources[0], (
                f"native CLI opening became ordinary exit: {result!r} {failure!r}"
            )
        else:
            assert sources and isinstance(failure, DatabaseRetirementError), (
                f"CLI retirement failure became success/ordinary result: "
                f"{stage} {kind} {result!r} {failure!r}"
            )
            assert failure.retirement_owner is owners[0]
            assert sources[0] in tuple(_explicit_control_chain(failure.cleanup_error))
            if kind == "native-control":
                assert is_execution_control_error(failure)
                if stage == "body-close":
                    assert body_sources and failure.operation_error is body_sources[0]
        # The CLI never mutates an existing corpus just by auditing it.
        with closing(sqlite3.connect(data_dir / "kestrel_prime.db")) as observer:
            assert (
                observer.execute("SELECT count(*) FROM saved_items").fetchone()[0] == 0
            )
        if stage in {"close", "body-close"}:
            assert physical_close, "the fault must follow actual native close"
    finally:
        # Refused close admission intentionally leaves the original DB live.
        # Retire that SAME original owner in a bounded test-owned event loop;
        # no successful Task/result is substituted for its failed receipt.
        for db in owners:
            try:
                await asyncio.to_thread(
                    lambda db=db: asyncio.run(actual_closes[id(db)]())
                )
            except BaseException as error:
                if not is_execution_control_error(error) and kind != "ordinary":
                    raise
        assert all(
            not _aiosqlite_worker_is_alive(native) for native in native_connections
        )
