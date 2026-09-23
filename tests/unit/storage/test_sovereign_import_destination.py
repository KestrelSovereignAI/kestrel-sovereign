"""#2525: ``target_db_path`` is refused before the bound database is touched.

Backend-independent half of the regression (the SQLite two-database proof
lives in ``tests/integration/test_sovereign_import_destination.py``). The
adapter here is bound to a database that fails on ANY attribute access —
``backend_type`` included — so the refusal is shown to precede any read,
write, transaction, or backend inspection: the value is never interpreted as
a SQLite path or a PostgreSQL DSN, whatever store the adapter is bound to.
"""

from __future__ import annotations

from typing import List

import pytest

from kestrel_sovereign.storage.sovereign_adapter import (
    TARGET_DB_PATH_REMOVAL_RELEASE,
    SovereignStorageAdapter,
    UnsupportedImportDestinationError,
)


class _UntouchableDatabase:
    """Records every attribute access; any access is a test failure."""

    def __init__(self) -> None:
        object.__setattr__(self, "accessed", [])

    def __getattr__(self, name: str):
        self.accessed.append(name)
        raise AssertionError(f"bound database touched: {name}")


class _NoFetch:
    """Storage transport that must never be asked for the package."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def __getattr__(self, name: str):
        self.calls.append(name)
        raise AssertionError(f"package transport touched: {name}")


@pytest.mark.parametrize(
    "target",
    [
        "/tmp/other-agent/kestrel_prime.db",
        "postgresql://kestrel:s3cret@db.internal:5432/other",
        "",
    ],
)
async def test_non_none_target_db_path_is_refused_before_any_io(target):
    db = _UntouchableDatabase()
    transport = _NoFetch()
    adapter = SovereignStorageAdapter(
        db, user_secret="secret", filecoin_adapter=transport, agent_id="a",
    )

    with pytest.warns(DeprecationWarning, match=TARGET_DB_PATH_REMOVAL_RELEASE):
        with pytest.raises(UnsupportedImportDestinationError) as excinfo:
            await adapter.import_agent(
                "bafy-not-fetched", target_db_path=target,
            )

    assert db.accessed == []
    assert transport.calls == []
    # A DSN may carry credentials; the refusal never echoes the value.
    if target:
        assert target not in str(excinfo.value)
    assert "s3cret" not in str(excinfo.value)


async def test_omitted_keyword_does_not_warn(recwarn):
    transport = _NoFetch()
    adapter = SovereignStorageAdapter(
        _UntouchableDatabase(), user_secret="secret",
        filecoin_adapter=transport, agent_id="a",
    )

    # An ordinary call proceeds to the package fetch, its first I/O.
    with pytest.raises(RuntimeError, match="Failed to fetch"):
        await adapter.import_agent("bafy-fetch-attempted")

    assert transport.calls
    assert not [w for w in recwarn if "target_db_path" in str(w.message)]


def test_unsupported_destination_is_a_value_error():
    assert issubclass(UnsupportedImportDestinationError, ValueError)
