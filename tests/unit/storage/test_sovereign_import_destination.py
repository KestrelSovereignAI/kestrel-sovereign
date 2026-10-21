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

from kestrel_sovereign.storage.async_database import AsyncDatabase
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


async def test_omitted_keyword_does_not_warn(recwarn, tmp_path):
    # The omitted keyword admits the ordinary import path. Unlike an explicit
    # unsupported destination, it owns an actual audit write even on rejection;
    # an untouchable database would be a second, unrelated failure, not a
    # transport/refusal proof. Use real SQLite and archive validation instead.
    db = await AsyncDatabase.sqlite(str(tmp_path / "omitted-destination.db"))
    try:
        adapter = SovereignStorageAdapter(
            db, user_secret="secret", filecoin_adapter=object(), agent_id="a",
        )
        result = await adapter.import_agent(b"owned invalid CAR fixture")
        assert not result.success and result.status == "rejected"
        rows = await adapter.get_import_log()
        assert len(rows) == 1 and rows[0]["status"] == "rejected"
    finally:
        await db.close()
    assert not [w for w in recwarn if "target_db_path" in str(w.message)]


def test_unsupported_destination_is_a_value_error():
    assert issubclass(UnsupportedImportDestinationError, ValueError)
