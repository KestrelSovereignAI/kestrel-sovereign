"""Contract suite for the principal master-key stores (#2527).

``HostKeyStorage``, ``UserMasterKeyStorage`` and ``SponsorKeyStorage`` are thin
facades over ``PrincipalMasterKeyStore``. Every behaviour they promise is
asserted here once, parameterized over all three facades, so a policy change
cannot land in one copy and drift from the others.

The suite runs on SQLite always and on a real PostgreSQL when
``TEST_POSTGRES_URL`` (or ``KESTREL_DATABASE_URL`` / ``DATABASE_URL``) is set.
``TestAsyncpgRowMaterialization`` covers the PostgreSQL row shape without a
server: the real ``PostgresBackend`` over a pool that returns asyncpg-shaped
records carrying native ``datetime`` values.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator, Optional

import pytest
import pytest_asyncio
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from kestrel_sdk.security.encryption import get_agent_key

from kestrel_sovereign.security.agent_encryption import encrypt
from kestrel_sovereign.security.exceptions import (
    DecryptionError,
    KeyNotConfiguredError,
)
from kestrel_sovereign.security.host_key_storage import HostKeyInfo, HostKeyStorage
from kestrel_sovereign.security.legacy_decrypt import decrypt_with_legacy_fallback
from kestrel_sovereign.security.principal_master_key_store import (
    MasterKeyRecord,
    MasterKeyTable,
    PrincipalMasterKeyScope,
    PrincipalMasterKeyStore,
)
from kestrel_sovereign.security.sponsor_key_storage import (
    SponsorKeyInfo,
    SponsorKeyStorage,
)
from kestrel_sovereign.security.user_master_key_storage import (
    UserMasterKeyInfo,
    UserMasterKeyStorage,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from kestrel_sovereign.storage.timestamps import (
    timestamp_column_value,
    utc_timestamp_parameter,
)

PURPOSE = "service-keys"
PROVIDER_PREFIX = "pmk2527-"

# Longer than the 30-character log/error label, so a test can tell a
# truncated principal from a leaked one.
USER_DID = "did:test:user-master-funding-principal-alice-0001"
OTHER_USER_DID = "did:test:user-master-funding-principal-bob-00002"
SPONSOR_DID = "did:test:sponsor-funding-principal-organisation-1"
OTHER_SPONSOR_DID = "did:test:sponsor-funding-principal-organisation-2"

POSTGRES_URL = (
    os.environ.get("TEST_POSTGRES_URL")
    or os.environ.get("KESTREL_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
)


@dataclass(frozen=True)
class Facade:
    name: str
    table: str
    principal_column: Optional[str]
    principal: Optional[str]
    other_principal: Optional[str]
    identity: str
    kind: str
    info_type: type
    info_principal_field: Optional[str]
    build: Callable[[Any, Optional[str]], Any]

    def make(self, db: Any, principal: Optional[str] = None) -> Any:
        return self.build(db, principal or self.principal)

    def where(self, principal: Optional[str] = None) -> tuple[str, tuple]:
        if self.principal_column is None:
            return "provider_id = ?", ()
        return (
            f"{self.principal_column} = ? AND provider_id = ?",
            (principal or self.principal,),
        )


FACADES = [
    Facade(
        name="host",
        table="host_service_keys",
        principal_column=None,
        principal=None,
        other_principal=None,
        identity="host",
        kind="host",
        info_type=HostKeyInfo,
        info_principal_field=None,
        build=lambda db, _principal: HostKeyStorage(db),
    ),
    Facade(
        name="user_master",
        table="user_master_service_keys",
        principal_column="master_did",
        principal=USER_DID,
        other_principal=OTHER_USER_DID,
        identity=USER_DID,
        kind="user",
        info_type=UserMasterKeyInfo,
        info_principal_field="master_did",
        build=lambda db, principal: UserMasterKeyStorage(db, principal),
    ),
    Facade(
        name="sponsor",
        table="sponsor_master_service_keys",
        principal_column="master_did",
        principal=SPONSOR_DID,
        other_principal=OTHER_SPONSOR_DID,
        identity=SPONSOR_DID,
        kind="sponsor",
        info_type=SponsorKeyInfo,
        info_principal_field="sponsor_did",
        build=lambda db, principal: SponsorKeyStorage(db, principal),
    ),
]
SCOPED = [f for f in FACADES if f.principal_column is not None]


def _provider() -> str:
    """A provider id no other test (or real host row) shares."""
    return f"{PROVIDER_PREFIX}{uuid.uuid4().hex[:12]}"


@pytest.fixture(autouse=True)
def _kestrel_data_key(monkeypatch) -> Iterator[None]:
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-32-bytes-fixed--")
    yield


@pytest_asyncio.fixture(params=["sqlite", "postgres"])
async def db(request, tmp_path):
    if request.param == "sqlite":
        database = await AsyncDatabase.sqlite(str(tmp_path / "test.db"))
    else:
        if not POSTGRES_URL:
            pytest.skip("TEST_POSTGRES_URL / KESTREL_DATABASE_URL / DATABASE_URL")
        database = await AsyncDatabase.postgres(POSTGRES_URL, max_pool_size=4)
    try:
        yield database
    finally:
        pattern = f"{PROVIDER_PREFIX}%"
        for facade in FACADES:
            await database.execute(
                f"DELETE FROM {facade.table} WHERE provider_id LIKE ?", (pattern,)
            )
        await database.execute(
            "DELETE FROM service_providers WHERE id LIKE ?", (pattern,)
        )
        await database.close()


@pytest.fixture(params=FACADES, ids=lambda f: f.name)
def facade(request) -> Facade:
    return request.param


@pytest.fixture(params=SCOPED, ids=lambda f: f.name)
def scoped_facade(request) -> Facade:
    return request.param


async def _rows(db, facade: Facade, provider: str, principal=None) -> list:
    where, params = facade.where(principal)
    return await db.fetchall(
        f"SELECT id, encrypted_key, key_hash, is_active FROM {facade.table} "
        f"WHERE {where}",
        params + (provider,),
    )


async def _set_encrypted(db, facade: Facade, provider: str, value: str) -> None:
    where, params = facade.where()
    await db.execute(
        f"UPDATE {facade.table} SET encrypted_key = ? WHERE {where}",
        (value,) + params + (provider,),
    )


def _db_timestamp(db, instant: datetime) -> Any:
    """Bind ``instant`` the way each backend's CURRENT_TIMESTAMP stores it."""
    if db.backend_type == "sqlite":
        return instant.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return utc_timestamp_parameter(db.backend_type, instant)


async def _insert_raw_row(
    db, facade: Facade, provider: str, encrypted_b64: str, *, is_active: int = 1
) -> str:
    """Write a row the way a previous release (or another tool) left it."""
    row_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO service_providers (id, name, supports_sub_accounts) "
        "VALUES (?, ?, 0)",
        (provider, provider),
    )
    if facade.principal_column is None:
        await db.execute(
            f"INSERT INTO {facade.table} "
            "(id, provider_id, encrypted_key, key_hash, is_active, created_at) "
            "VALUES (?, ?, ?, 'legacy-hash', ?, CURRENT_TIMESTAMP)",
            (row_id, provider, encrypted_b64, is_active),
        )
    else:
        await db.execute(
            f"INSERT INTO {facade.table} "
            f"(id, {facade.principal_column}, provider_id, encrypted_key, "
            "key_hash, is_active, created_at) "
            "VALUES (?, ?, ?, ?, 'legacy-hash', ?, CURRENT_TIMESTAMP)",
            (row_id, facade.principal, provider, encrypted_b64, is_active),
        )
    return row_id


# =============================================================================
# Round trip, replacement, ciphertext format
# =============================================================================


class TestRoundTrip:
    async def test_store_then_get_returns_plaintext(self, db, facade) -> None:
        store = facade.make(db)
        provider = _provider()
        secret = "sk-or-v1-" + secrets.token_hex(16)

        key_id = await store.store_key(provider, secret)

        assert await store.get_key(provider) == secret
        assert await store.has_key(provider) is True
        rows = await _rows(db, facade, provider)
        assert [r[0] for r in rows] == [key_id]
        assert rows[0][2] == hashlib.sha256(secret.encode()).hexdigest()[:32]
        assert bool(rows[0][3]) is True

    async def test_ciphertext_format_and_identity_are_unchanged(
        self, db, facade
    ) -> None:
        """The stored value is base64 of the SDK's v2 envelope under this
        facade's identity and the ``service-keys`` purpose, readable by the
        SDK directly and by no other principal's identity."""
        store = facade.make(db)
        provider = _provider()
        secret = "sk-or-v1-format-" + secrets.token_hex(8)
        await store.store_key(provider, secret)

        stored = base64.b64decode((await _rows(db, facade, provider))[0][1])
        assert stored.startswith(b"KSAv2:")
        assert decrypt_with_legacy_fallback(facade.identity, PURPOSE, stored) == (
            secret.encode()
        )
        for other in FACADES:
            if other.identity == facade.identity:
                continue
            with pytest.raises(DecryptionError):
                decrypt_with_legacy_fallback(other.identity, PURPOSE, stored)
        with pytest.raises(DecryptionError):
            decrypt_with_legacy_fallback(facade.identity, "wallet", stored)

    async def test_store_replaces_existing_key(self, db, facade) -> None:
        store = facade.make(db)
        provider = _provider()
        await store.store_key(provider, "first-key")
        second_id = await store.store_key(provider, "second-key")

        assert await store.get_key(provider) == "second-key"
        rows = await _rows(db, facade, provider)
        assert len(rows) == 1
        assert rows[0][0] == second_id
        assert rows[0][2] == hashlib.sha256(b"second-key").hexdigest()[:32]

    async def test_store_reactivates_an_inactive_key(self, db, facade) -> None:
        store = facade.make(db)
        provider = _provider()
        await store.store_key(provider, "old")
        where, params = facade.where()
        await db.execute(
            f"UPDATE {facade.table} SET is_active = 0 WHERE {where}",
            params + (provider,),
        )

        await store.store_key(provider, "new")

        assert await store.has_key(provider) is True
        assert await store.get_key(provider) == "new"

    async def test_concurrent_replacements_leave_exactly_one_row(
        self, db, facade
    ) -> None:
        store = facade.make(db)
        provider = _provider()
        values = [f"key-{i}" for i in range(6)]

        await asyncio.gather(*(store.store_key(provider, v) for v in values))

        rows = await _rows(db, facade, provider)
        assert len(rows) == 1
        assert await store.get_key(provider) in values


# =============================================================================
# Principal isolation
# =============================================================================


class TestPrincipalIsolation:
    async def test_facades_do_not_share_rows(self, db) -> None:
        provider = _provider()
        for f in FACADES:
            await f.make(db).store_key(provider, f"secret-for-{f.name}")

        for f in FACADES:
            store = f.make(db)
            assert await store.get_key(provider) == f"secret-for-{f.name}"
            assert [k.provider_id for k in await store.list_keys()].count(provider) == 1

        assert await FACADES[0].make(db).delete_key(provider) is True
        for f in FACADES[1:]:
            assert await f.make(db).get_key(provider) == f"secret-for-{f.name}"

    async def test_principals_in_one_table_are_isolated(
        self, db, scoped_facade
    ) -> None:
        f = scoped_facade
        provider = _provider()
        mine, theirs = f.make(db), f.make(db, f.other_principal)
        await mine.store_key(provider, "mine")
        await theirs.store_key(provider, "theirs")

        assert await mine.get_key(provider) == "mine"
        assert await theirs.get_key(provider) == "theirs"
        assert {k.provider_id for k in await mine.list_keys()} == {provider}
        assert await f.make(db, "did:test:nobody").has_key(provider) is False

        assert await mine.delete_key(provider) is True
        assert await theirs.get_key(provider) == "theirs"

    async def test_ciphertext_moved_to_another_principal_does_not_decrypt(
        self, db, scoped_facade
    ) -> None:
        f = scoped_facade
        provider = _provider()
        await f.make(db).store_key(provider, "only-mine")
        blob = (await _rows(db, f, provider))[0][1]

        await f.make(db, f.other_principal).store_key(provider, "placeholder")
        where, params = f.where(f.other_principal)
        await db.execute(
            f"UPDATE {f.table} SET encrypted_key = ? WHERE {where}",
            (blob,) + params + (provider,),
        )

        with pytest.raises(DecryptionError):
            await f.make(db, f.other_principal).get_key(provider)

    @pytest.mark.parametrize("value", ["", None])
    def test_scoped_facades_require_a_principal(self, scoped_facade, value) -> None:
        with pytest.raises(ValueError):
            scoped_facade.build(object(), value)


# =============================================================================
# Missing and inactive keys
# =============================================================================


class TestMissingAndInactive:
    async def test_missing_key(self, db, facade) -> None:
        store = facade.make(db)
        provider = _provider()

        with pytest.raises(KeyNotConfiguredError) as excinfo:
            await store.get_key(provider)
        assert f"No {facade.kind} master key configured" in str(excinfo.value)
        assert provider in str(excinfo.value)
        assert await store.has_key(provider) is False
        assert await store.delete_key(provider) is False

    async def test_inactive_key_is_invisible_and_not_deleted(self, db, facade) -> None:
        store = facade.make(db)
        provider = _provider()
        await store.store_key(provider, "dormant")
        where, params = facade.where()
        await db.execute(
            f"UPDATE {facade.table} SET is_active = 0 WHERE {where}",
            params + (provider,),
        )

        with pytest.raises(KeyNotConfiguredError):
            await store.get_key(provider)
        assert await store.has_key(provider) is False
        assert await store.delete_key(provider) is False
        assert len(await _rows(db, facade, provider)) == 1
        [info] = [k for k in await store.list_keys() if k.provider_id == provider]
        assert info.is_active is False

    async def test_empty_ciphertext_reads_as_not_configured(self, db, facade) -> None:
        store = facade.make(db)
        provider = _provider()
        await store.store_key(provider, "will-be-blanked")
        await _set_encrypted(db, facade, provider, "")

        with pytest.raises(KeyNotConfiguredError):
            await store.get_key(provider)


# =============================================================================
# Listing
# =============================================================================


class TestListing:
    async def test_list_is_newest_first_with_provider_tiebreak(
        self, db, facade
    ) -> None:
        store = facade.make(db)
        base = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        providers = sorted(_provider() for _ in range(4))
        stamps = {
            providers[0]: base,
            providers[1]: base + timedelta(hours=2),
            providers[2]: base + timedelta(hours=1),
            providers[3]: base + timedelta(hours=1),
        }
        # Reverse insertion order, so the equal-timestamp pair only lists in
        # provider order if the query breaks the tie.
        for provider in reversed(providers):
            await store.store_key(provider, f"secret-{provider}")
            where, params = facade.where()
            await db.execute(
                f"UPDATE {facade.table} SET created_at = ? WHERE {where}",
                (_db_timestamp(db, stamps[provider]),) + params + (provider,),
            )
        if facade.principal_column is not None:
            await facade.make(db, facade.other_principal).store_key(
                _provider(), "not-listed"
            )

        listed = [k for k in await store.list_keys() if k.provider_id in stamps]

        assert [k.provider_id for k in listed] == [
            providers[1],
            providers[2],
            providers[3],
            providers[0],
        ]
        for info in listed:
            assert type(info) is facade.info_type
            assert info.created_at == stamps[info.provider_id].replace(tzinfo=None)
            assert info.is_active is True
            if facade.info_principal_field is not None:
                assert getattr(info, facade.info_principal_field) == facade.principal
            assert not hasattr(info, "api_key")
            assert not hasattr(info, "encrypted_key")
            assert not hasattr(info, "key_hash")
        if facade.principal_column is not None:
            assert {k.provider_id for k in await store.list_keys()} == set(stamps)

    async def test_info_record_fields_are_unchanged(self, facade) -> None:
        expected = ["id", "provider_id", "is_active", "created_at"]
        if facade.info_principal_field is not None:
            expected.insert(1, facade.info_principal_field)
        assert list(facade.info_type.__dataclass_fields__) == expected


# =============================================================================
# Delete truthfulness
# =============================================================================


class _LockstepDatabase:
    """Forces two callers to reach every database call together.

    A delete implemented as SELECT-then-DELETE lets both callers pass the
    SELECT before either deletes, so both would report success. This makes
    that interleaving certain instead of scheduler-dependent.
    """

    def __init__(self, db: AsyncDatabase, parties: int) -> None:
        self._db = db
        self._barrier = asyncio.Barrier(parties)

    async def _rendezvous(self) -> None:
        await asyncio.wait_for(self._barrier.wait(), timeout=10)

    async def execute(self, sql: str, params: tuple = ()) -> int:
        await self._rendezvous()
        return await self._db.execute(sql, params)

    async def fetchone(self, sql: str, params: tuple = ()):
        await self._rendezvous()
        return await self._db.fetchone(sql, params)

    async def fetchall(self, sql: str, params: tuple = ()):
        await self._rendezvous()
        return await self._db.fetchall(sql, params)


class TestDelete:
    async def test_delete_reports_removal_once(self, db, facade) -> None:
        store = facade.make(db)
        provider = _provider()
        await store.store_key(provider, "to-delete")

        assert await store.delete_key(provider) is True
        assert await store.has_key(provider) is False
        assert await _rows(db, facade, provider) == []
        assert await store.delete_key(provider) is False

    async def test_lockstep_concurrent_deletes_report_one_success(
        self, db, facade
    ) -> None:
        provider = _provider()
        await facade.make(db).store_key(provider, "contested")
        lockstep = _LockstepDatabase(db, parties=2)
        first, second = facade.make(lockstep), facade.make(lockstep)

        results = await asyncio.gather(
            first.delete_key(provider), second.delete_key(provider)
        )

        assert sorted(results) == [False, True]
        assert await _rows(db, facade, provider) == []

    async def test_many_concurrent_deletes_report_one_success(self, db, facade) -> None:
        provider = _provider()
        await facade.make(db).store_key(provider, "contested")
        stores = [facade.make(db) for _ in range(8)]

        results = await asyncio.gather(*(s.delete_key(provider) for s in stores))

        assert results.count(True) == 1


# =============================================================================
# Malformed and legacy ciphertext
# =============================================================================


class TestCiphertextCompatibility:
    @pytest.mark.parametrize(
        "stored",
        [
            "!!!not base64",
            base64.b64encode(b"\x00" * 40).decode(),
            base64.b64encode(b"KSAv2:" + b"\x01" * 40).decode(),
        ],
        ids=["bad-base64", "random-bytes", "corrupt-v2"],
    )
    async def test_malformed_ciphertext_raises_decryption_error(
        self, db, facade, stored
    ) -> None:
        store = facade.make(db)
        provider = _provider()
        await store.store_key(provider, "valid-first")
        await _set_encrypted(db, facade, provider, stored)

        with pytest.raises(DecryptionError) as excinfo:
            await store.get_key(provider)
        assert stored not in str(excinfo.value)
        assert "valid-first" not in str(excinfo.value)

    async def test_non_utf8_plaintext_error_does_not_quote_plaintext(
        self, db, facade
    ) -> None:
        store = facade.make(db)
        provider = _provider()
        await store.store_key(provider, "valid-first")
        blob = base64.b64encode(encrypt(facade.identity, PURPOSE, b"\xffsecret"))
        await _set_encrypted(db, facade, provider, blob.decode())

        with pytest.raises(DecryptionError) as excinfo:
            await store.get_key(provider)
        assert "0xff" not in str(excinfo.value)
        assert excinfo.value.__cause__ is None
        assert excinfo.value.__suppress_context__ is True

    async def test_pre_v2_raw_aes_gcm_row_still_decrypts(self, db, facade) -> None:
        secret = "sk-or-v1-legacy-" + secrets.token_hex(8)
        nonce = os.urandom(12)
        key = get_agent_key(facade.identity, PURPOSE)
        legacy = nonce + AESGCM(key).encrypt(nonce, secret.encode(), None)
        provider = _provider()
        await _insert_raw_row(db, facade, provider, base64.b64encode(legacy).decode())

        store = facade.make(db)
        assert await store.get_key(provider) == secret
        assert await store.has_key(provider) is True

    async def test_pre_existing_row_is_listed_replaced_and_deleted(
        self, db, facade
    ) -> None:
        provider = _provider()
        blob = base64.b64encode(encrypt(facade.identity, PURPOSE, b"old-release"))
        legacy_id = await _insert_raw_row(db, facade, provider, blob.decode())
        store = facade.make(db)

        assert await store.get_key(provider) == "old-release"
        assert [k.id for k in await store.list_keys() if k.provider_id == provider] == [
            legacy_id
        ]
        new_id = await store.store_key(provider, "new-release")
        assert new_id != legacy_id
        assert [r[0] for r in await _rows(db, facade, provider)] == [new_id]
        assert await store.delete_key(provider) is True


# =============================================================================
# Secrets stay out of logs, errors and reprs
# =============================================================================


class TestRedaction:
    async def test_lifecycle_logs_never_carry_secret_ciphertext_or_full_did(
        self, db, facade, caplog
    ) -> None:
        caplog.set_level(logging.DEBUG)
        store = facade.make(db)
        provider = _provider()
        secret = "sk-or-v1-redact-" + secrets.token_hex(16)

        await store.store_key(provider, secret)
        ciphertext = (await _rows(db, facade, provider))[0][1]
        assert await store.get_key(provider) == secret
        await store.list_keys()
        await store.delete_key(provider)
        with pytest.raises(KeyNotConfiguredError) as missing:
            await store.get_key(provider)

        # Kestrel's own log lines only: aiosqlite's DEBUG logger echoes every
        # bound parameter for every store, which no store can redact.
        logged = "\n".join(
            record.getMessage()
            for record in caplog.records
            if record.name.startswith("kestrel_sovereign")
        )
        text = logged + str(missing.value) + repr(store) + repr(store._store)
        assert secret not in text
        assert ciphertext not in text
        assert f"Stored {facade.kind} master key" in logged
        assert f"Deleted {facade.kind} master key" in logged
        if facade.principal is not None:
            assert facade.principal not in text
            label = f"{facade.principal[:30]}..."
            assert f"{facade.kind}={label}, provider={provider}" in logged
            assert f"and {facade.kind} '{label}'" in str(missing.value)


# =============================================================================
# Typed configuration
# =============================================================================


class TestScopeConfiguration:
    def test_tables_are_a_closed_set_of_valid_identifiers(self) -> None:
        assert {(t.table_name, t.principal_column) for t in MasterKeyTable} == {
            ("host_service_keys", None),
            ("user_master_service_keys", "master_did"),
            ("sponsor_master_service_keys", "master_did"),
        }

    def test_scope_rejects_a_table_name_string(self) -> None:
        with pytest.raises(TypeError):
            PrincipalMasterKeyScope(
                table="host_service_keys; DROP TABLE x",  # type: ignore[arg-type]
                principal=None,
                encryption_identity="host",
                kind="host",
                project_info=lambda r: r,
            )

    @pytest.mark.parametrize(
        ("table", "principal"),
        [(MasterKeyTable.HOST, "did:test:x"), (MasterKeyTable.USER_MASTER, None)],
    )
    def test_scope_rejects_a_principal_mismatch(self, table, principal) -> None:
        with pytest.raises(ValueError):
            PrincipalMasterKeyScope(
                table=table,
                principal=principal,
                encryption_identity="x",
                kind="k",
                project_info=lambda r: r,
            )

    @pytest.mark.parametrize("field", ["encryption_identity", "kind"])
    def test_scope_requires_identity_and_kind(self, field) -> None:
        values = dict(
            table=MasterKeyTable.HOST,
            principal=None,
            encryption_identity="host",
            kind="host",
            project_info=lambda r: r,
        )
        values[field] = ""
        with pytest.raises(ValueError):
            PrincipalMasterKeyScope(**values)


# =============================================================================
# PostgreSQL row shape without a server
# =============================================================================


class _Record:
    """The part of ``asyncpg.Record`` that ``PostgresBackend`` reads."""

    def __init__(self, *values: Any) -> None:
        self._values = values

    def values(self):
        return iter(self._values)


class _AsyncpgShapedPool:
    """Stands in for ``asyncpg.Pool``: answers with native Python values."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, tuple]] = []
        self.fetch_rows: list[_Record] = []
        self.status = "DELETE 1"

    async def execute(self, query: str, *args: Any) -> str:
        self.queries.append((query, args))
        return self.status

    async def fetch(self, query: str, *args: Any) -> list:
        self.queries.append((query, args))
        return list(self.fetch_rows)

    async def fetchrow(self, query: str, *args: Any):
        self.queries.append((query, args))
        return self.fetch_rows[0] if self.fetch_rows else None


def _postgres_db() -> tuple[AsyncDatabase, _AsyncpgShapedPool]:
    backend = PostgresBackend("postgresql://unused@localhost/unused")
    pool = _AsyncpgShapedPool()
    backend._pool = pool
    return AsyncDatabase(backend), pool


class TestAsyncpgRowMaterialization:
    async def test_list_accepts_native_datetime(self, facade) -> None:
        db, pool = _postgres_db()
        created = datetime(2026, 9, 1, 12, 30, 45, 123456)
        principal = facade.principal
        pool.fetch_rows = [_Record("row-1", principal, "openrouter", 1, created)]

        [info] = await facade.make(db).list_keys()

        assert type(info) is facade.info_type
        assert info.id == "row-1"
        assert info.provider_id == "openrouter"
        assert info.is_active is True
        assert info.created_at == created
        if facade.info_principal_field is not None:
            assert getattr(info, facade.info_principal_field) == principal
        query, args = pool.queries[-1]
        assert "?" not in query
        assert f"FROM {facade.table}" in query
        assert "ORDER BY created_at DESC, provider_id ASC" in query
        assert args == (() if principal is None else (principal,))

    async def test_upsert_is_native_on_conflict_with_real_target(self, facade) -> None:
        db, pool = _postgres_db()
        pool.status = "INSERT 0 1"

        await facade.make(db).store_key("openrouter", "sk-pg")

        upsert = next(q for q, _ in pool.queries if facade.table in q)
        target = (
            "provider_id"
            if facade.principal_column is None
            else (f"{facade.principal_column}, provider_id")
        )
        assert f"ON CONFLICT ({target}) DO UPDATE SET" in upsert
        assert upsert.count("ON CONFLICT") == 1
        assert "OR REPLACE" not in upsert.upper()

    @pytest.mark.parametrize(
        ("status", "expected"), [("DELETE 1", True), ("DELETE 0", False)]
    )
    async def test_delete_reads_the_command_tag(self, facade, status, expected) -> None:
        db, pool = _postgres_db()
        pool.status = status

        assert await facade.make(db).delete_key("openrouter") is expected
        [(query, _args)] = pool.queries
        assert query.lstrip().startswith("DELETE FROM")
        assert "is_active = 1" in query


# =============================================================================
# Timestamp column helper
# =============================================================================


class TestTimestampColumnValue:
    def test_sqlite_current_timestamp_text(self) -> None:
        assert timestamp_column_value("2026-09-26 08:15:30") == datetime(
            2026, 9, 26, 8, 15, 30
        )

    def test_native_datetime_passes_through(self) -> None:
        value = datetime(2026, 9, 26, 8, 15, 30, 5)
        assert timestamp_column_value(value) is value

    def test_aware_datetime_is_not_altered(self) -> None:
        value = datetime(2026, 9, 26, tzinfo=timezone(timedelta(hours=2)))
        assert timestamp_column_value(value) is value

    def test_null_is_not_a_timestamp(self) -> None:
        with pytest.raises(TypeError):
            timestamp_column_value(None)

    @pytest.mark.parametrize("value", [1727338530, b"2026-09-26"])
    def test_other_types_are_refused_without_echoing_the_value(self, value) -> None:
        with pytest.raises(TypeError) as excinfo:
            timestamp_column_value(value)
        assert str(value) not in str(excinfo.value)

    def test_malformed_text_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            timestamp_column_value("not a timestamp")


def test_primitive_is_generic_over_the_info_projection() -> None:
    record = MasterKeyRecord(
        id="i",
        principal=None,
        provider_id="p",
        is_active=True,
        created_at=datetime(2026, 1, 1),
    )
    scope = PrincipalMasterKeyScope(
        table=MasterKeyTable.HOST,
        principal=None,
        encryption_identity="host",
        kind="host",
        project_info=lambda r: ("projected", r.provider_id),
    )
    store = PrincipalMasterKeyStore(object(), scope)  # type: ignore[arg-type]
    assert scope.project_info(record) == ("projected", "p")
    assert "host_service_keys" in repr(store)
