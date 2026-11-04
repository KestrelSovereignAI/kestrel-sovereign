"""Observe physical retirement of real, module-local SQLite test connections."""

import sqlite3
from contextlib import ExitStack
from types import SimpleNamespace

import pytest


@pytest.fixture
def owned_sqlite_observers(request, monkeypatch):
    """Keep original connections alive until their test must have closed them.

    Only the requesting test module's sqlite3 binding is replaced. Production
    adapters and aiosqlite keep the real sqlite3 module and their own owners.
    Every returned connection is real; holding it prevents GC from hiding a
    missing close. ExitStack retires originals even when the assertion fails.
    """
    connections = []
    connect = sqlite3.connect
    with ExitStack() as cleanup:
        def observed_connect(*args, **kwargs):
            connection = connect(*args, **kwargs)
            cleanup.callback(connection.close)
            connections.append(connection)
            return connection

        monkeypatch.setattr(
            request.module, "sqlite3", SimpleNamespace(connect=observed_connect)
        )
        yield
        for connection in connections:
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                connection.execute("SELECT 1")
