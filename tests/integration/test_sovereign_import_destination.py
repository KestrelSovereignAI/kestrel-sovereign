"""#2525: ``import_agent`` has exactly one destination — its bound database.

``SovereignStorageAdapter.import_agent`` accepted ``target_db_path`` but
never read it: verification, audit, conversation restore, and asset
restorers all went through the adapter's bound ``self.db``. A caller asking
for database B got a nominal result while database A was mutated.

These tests bind an adapter to SQLite database A, keep an independent
database B, and prove:

* the supported call (no keyword) restores into A and never touches B —
  which is exactly why a ``target_db_path=B`` call used to mutate A;
* any non-``None`` ``target_db_path`` fails closed before the package is
  fetched and before any continuity audit, ``agent_import_log`` write,
  conversation mutation, or asset-restorer call, on every path the import
  could otherwise take (accepted, errored, continuity-rejected,
  consent-rejected) — neither A nor B changes by a single row;
* an explicit ``target_db_path=None`` is deprecated but still imports.

The PostgreSQL leg proves the value is never reinterpreted against a
non-SQLite backend. Run it with::

    TEST_POSTGRES_URL=postgresql://u:p@127.0.0.1:5432/db uv run pytest \\
        tests/integration/test_sovereign_import_destination.py

NO MOCKS: real SQLite files, real CAR export, real AES-GCM.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from kestrel_sovereign.filecoin_adapter import StorageTier
from kestrel_sovereign.identity.access_grant import DataAccessGrant
from kestrel_sovereign.storage import Storage
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.sovereign_adapter import (
    AssetCollector,
    AssetDescriptor,
    AssetMetadata,
    AssetRestorer,
    SovereignStorageAdapter,
    UnsupportedImportDestinationError,
)

pytestmark = pytest.mark.integration

POSTGRES_URL = (
    os.environ.get("TEST_POSTGRES_URL")
    or os.environ.get("KESTREL_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
)

AGENT_ID = "test:import-destination"
AGENT_DID = "did:pkh:eip155:1:0x25" + "25" * 19
HOST_DID = "did:pkh:eip155:1:0x26" + "26" * 19
SECRET = "2525-owner-secret"


class _StaticAssetCollector(AssetCollector):
    def __init__(self, assets: List[AssetDescriptor]) -> None:
        self._assets = assets

    async def collect_assets(self, agent_did: str) -> List[AssetDescriptor]:
        return list(self._assets)


class _CapturingRestorer(AssetRestorer):
    def __init__(self) -> None:
        self.calls: List[Tuple[str, List[Tuple[AssetMetadata, bytes]]]] = []

    @property
    def asset_types(self) -> List[str]:
        return ["avatar"]

    async def restore_assets(
        self, agent_did: str, assets: List[Tuple[AssetMetadata, bytes]],
    ) -> int:
        self.calls.append((agent_did, list(assets)))
        return len(assets)


def _dump(path: Path) -> List[str]:
    """Every committed schema object and row, via an independent connection."""
    conn = sqlite3.connect(str(path))
    try:
        return list(conn.iterdump())
    finally:
        conn.close()


@dataclass
class _Databases:
    a: Storage
    a_path: Path
    b: Storage
    b_path: Path
    cid: str
    car_bytes: bytes


@pytest.fixture
async def databases(tmp_path: Path):
    """A holds an exported, then wiped, history; B is an independent store."""
    a_path = tmp_path / "a.db"
    b_path = tmp_path / "b.db"
    async with Storage(db_path=str(a_path), agent_id=AGENT_ID) as a, Storage(
        db_path=str(b_path), agent_id=AGENT_ID,
    ) as b:
        await a.add_conversation(
            "user", "A_EXPORTED_MARKER",
            metadata={"timestamp": "2026-09-01T10:00:00Z"},
        )
        await b.add_conversation(
            "user", "B_RESIDENT_MARKER",
            metadata={"timestamp": "2026-09-02T10:00:00Z"},
        )
        exporter = SovereignStorageAdapter(
            a.db, user_secret=SECRET, agent_id=AGENT_ID,
        )
        cid = await exporter.export_agent(
            AGENT_DID,
            storage_tier=StorageTier.LOCAL_ONLY,
            asset_collector=_StaticAssetCollector([
                AssetDescriptor(
                    asset_type="avatar", asset_key="main",
                    content_hash="c0ffee", size_bytes=4, data=b"PNG!",
                ),
            ]),
        )
        car_bytes = await exporter._download_content(cid)
        # A restore into A is now observable as reappearing rows.
        await a.db.execute_commit("DELETE FROM conversation_history")
        yield _Databases(
            a=a, a_path=a_path, b=b, b_path=b_path,
            cid=cid, car_bytes=car_bytes,
        )


async def _contents(storage: Storage) -> List[str]:
    return [m["content"] for m in await storage.get_conversation_history()]


async def test_bound_database_is_the_only_destination(databases):
    """The supported call restores into A and leaves B untouched.

    This is the hazard in the other direction: because A is the only
    database the adapter can write, a caller that asked for B used to
    receive an ``imported`` result while A — not B — was rewritten.
    """
    b_before = _dump(databases.b_path)
    restorer = _CapturingRestorer()
    adapter = SovereignStorageAdapter(
        databases.a.db, user_secret=SECRET, agent_id=AGENT_ID,
    )

    result = await adapter.import_agent(
        databases.car_bytes, asset_restorers=[restorer],
    )

    assert result.status == "imported"
    assert await _contents(databases.a) == ["A_EXPORTED_MARKER"]
    assert [row["status"] for row in await adapter.get_import_log()] == [
        "imported",
    ]
    assert len(restorer.calls) == 1
    assert _dump(databases.b_path) == b_before
    assert await _contents(databases.b) == ["B_RESIDENT_MARKER"]


def _accepted(dbs: _Databases) -> Tuple[SovereignStorageAdapter, Any, Dict]:
    adapter = SovereignStorageAdapter(dbs.a.db, user_secret=SECRET, agent_id=AGENT_ID)
    return adapter, dbs.car_bytes, {}


def _errored(dbs: _Databases) -> Tuple[SovereignStorageAdapter, Any, Dict]:
    # An unfetchable CID used to append an ``error`` audit row to A.
    adapter = SovereignStorageAdapter(dbs.a.db, user_secret=SECRET, agent_id=AGENT_ID)
    return adapter, f"local:missing-{uuid.uuid4().hex}", {}


def _continuity_rejected(dbs: _Databases) -> Tuple[SovereignStorageAdapter, Any, Dict]:
    # The wrong owner secret cannot open the keyring: a ``rejected`` row.
    adapter = SovereignStorageAdapter(
        dbs.a.db, user_secret="not-the-owner", agent_id=AGENT_ID,
    )
    return adapter, dbs.car_bytes, {}


def _consent_rejected(dbs: _Databases) -> Tuple[SovereignStorageAdapter, Any, Dict]:
    # An unsigned grant fails consent verification: a ``rejected`` row.
    adapter = SovereignStorageAdapter(dbs.a.db, user_secret=SECRET, agent_id=AGENT_ID)
    grant = DataAccessGrant(
        owner_did=AGENT_DID,
        source_did=AGENT_DID,
        host_did=HOST_DID,
        issued_at="2026-09-22T00:00:00+00:00",
        purpose="2525 unsigned grant",
        owner_verification_methods=[],
    )
    return adapter, dbs.car_bytes, {"grant": grant, "host_did": HOST_DID}


_PATHS = {
    "accepted": _accepted,
    "errored": _errored,
    "continuity_rejected": _continuity_rejected,
    "consent_rejected": _consent_rejected,
}


@pytest.mark.parametrize("path", sorted(_PATHS))
@pytest.mark.parametrize("target", ["b", "a"])
async def test_target_db_path_fails_closed_without_mutating_either_database(
    databases, path, target,
):
    """Every import path is refused before its first side effect.

    ``target="a"`` covers a caller naming the very database the adapter is
    bound to: the keyword is still refused rather than trusted, because it
    has never been an input to where the import writes.
    """
    adapter, package, kwargs = _PATHS[path](databases)
    target_path = databases.b_path if target == "b" else databases.a_path
    a_before = _dump(databases.a_path)
    b_before = _dump(databases.b_path)
    restorer = _CapturingRestorer()

    with pytest.warns(DeprecationWarning, match="target_db_path"):
        with pytest.raises(UnsupportedImportDestinationError) as excinfo:
            await adapter.import_agent(
                package,
                target_db_path=str(target_path),
                asset_restorers=[restorer],
                **kwargs,
            )

    assert str(target_path) not in str(excinfo.value)
    assert restorer.calls == []
    assert _dump(databases.a_path) == a_before
    assert _dump(databases.b_path) == b_before
    assert await _contents(databases.a) == []
    assert await _contents(databases.b) == ["B_RESIDENT_MARKER"]


async def test_explicit_none_is_deprecated_but_imports_into_bound_database(
    databases,
):
    b_before = _dump(databases.b_path)
    adapter = SovereignStorageAdapter(
        databases.a.db, user_secret=SECRET, agent_id=AGENT_ID,
    )

    with pytest.warns(DeprecationWarning, match="target_db_path"):
        result = await adapter.import_agent(
            databases.car_bytes, target_db_path=None,
        )

    assert result.status == "imported"
    assert await _contents(databases.a) == ["A_EXPORTED_MARKER"]
    assert _dump(databases.b_path) == b_before


async def test_postgres_bound_adapter_never_reinterprets_target_db_path(
    databases,
):
    """A PostgreSQL-bound adapter refuses the keyword without writing.

    Neither a filesystem path nor a DSN is ever opened: the refused call
    appends no audit row to the bound PostgreSQL database.
    """
    if not POSTGRES_URL:  # pragma: no cover - environment gate
        pytest.skip("TEST_POSTGRES_URL / KESTREL_DATABASE_URL / DATABASE_URL required")

    db = await AsyncDatabase.postgres(POSTGRES_URL)
    try:
        adapter = SovereignStorageAdapter(
            db, user_secret=SECRET, agent_id=f"test:2525:{uuid.uuid4().hex}",
        )
        # Creates the audit table, so the count below is a real observation.
        assert await adapter.get_import_log(agent_did=AGENT_DID) is not None
        before = await db.fetchall(
            "SELECT COUNT(*) FROM agent_import_log WHERE host_agent_id = ?",
            (adapter.agent_id,),
        )
        restorer = _CapturingRestorer()

        for target in (str(databases.b_path), POSTGRES_URL):
            with pytest.warns(DeprecationWarning, match="target_db_path"):
                with pytest.raises(UnsupportedImportDestinationError) as excinfo:
                    await adapter.import_agent(
                        databases.car_bytes,
                        target_db_path=target,
                        asset_restorers=[restorer],
                    )
            assert target not in str(excinfo.value)

        after = await db.fetchall(
            "SELECT COUNT(*) FROM agent_import_log WHERE host_agent_id = ?",
            (adapter.agent_id,),
        )
        assert after == before
        assert restorer.calls == []
    finally:
        await db.close()
