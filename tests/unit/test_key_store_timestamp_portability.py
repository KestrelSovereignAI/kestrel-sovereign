"""Backend-portable timestamp parsing for the master-credential key stores (#2391).

``HostKeyStorage``, ``SponsorKeyStorage``, ``UserMasterKeyStorage`` and
``UserBYOKKeyStorage`` all expose ``list_keys()``, which reads a ``created_at``
column. SQLite returns ISO-format **strings** for ``CURRENT_TIMESTAMP``, but
PostgreSQL (asyncpg) returns native ``datetime`` objects for ``TIMESTAMP``.
The stores previously called ``datetime.fromisoformat(row[n])`` unconditionally,
which raises ``TypeError: fromisoformat: argument must be str`` on Postgres —
a failure SQLite-only unit tests never surface.

These tests drive each store's ``list_keys()`` through a stub database that
returns each backend's row shape (ISO string *and* native datetime), proving
the shared ``_as_datetime`` coercion handles both without raising, and that a
NULL timestamp degrades to a real ``datetime`` instead of ``None``.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from kestrel_sovereign.security.host_key_storage import HostKeyStorage, HostKeyInfo
from kestrel_sovereign.security.sponsor_key_storage import (
    SponsorKeyStorage,
    SponsorKeyInfo,
)
from kestrel_sovereign.security.user_master_key_storage import (
    UserMasterKeyStorage,
    UserMasterKeyInfo,
)
from kestrel_sovereign.security.user_byok_key_storage import (
    UserBYOKKeyStorage,
    UserBYOKKeyInfo,
)


class _StubDB:
    """AsyncDatabase stand-in whose ``fetchall`` returns canned rows.

    ``list_keys`` only calls ``fetchall``; supplying the row shape each store's
    SELECT produces lets us simulate SQLite (ISO strings) vs asyncpg (native
    ``datetime``) without a live Postgres.
    """

    def __init__(self, rows):
        self._rows = rows

    async def fetchall(self, sql, params=()):
        return self._rows


# A native datetime (asyncpg) and an ISO string (SQLite) for the same instant.
_NATIVE = datetime(2026, 1, 2, 3, 4, 5)
_ISO = "2026-01-02T03:04:05"


@pytest.mark.asyncio
@pytest.mark.parametrize("created_at", [_NATIVE, _ISO, None])
async def test_host_list_keys_parses_both_backends(created_at):
    # host SELECT: id, provider_id, is_active, created_at
    db = _StubDB([("row-id", "openrouter", 1, created_at)])
    keys = await HostKeyStorage(db).list_keys()

    assert len(keys) == 1
    assert isinstance(keys[0], HostKeyInfo)
    assert isinstance(keys[0].created_at, datetime)
    if created_at is not None:
        assert keys[0].created_at == _NATIVE


@pytest.mark.asyncio
@pytest.mark.parametrize("created_at", [_NATIVE, _ISO, None])
async def test_sponsor_list_keys_parses_both_backends(created_at):
    # sponsor SELECT: id, master_did, provider_id, is_active, created_at
    db = _StubDB([("row-id", "did:sponsor", "openrouter", 1, created_at)])
    keys = await SponsorKeyStorage(db, "did:sponsor").list_keys()

    assert len(keys) == 1
    assert isinstance(keys[0], SponsorKeyInfo)
    assert isinstance(keys[0].created_at, datetime)
    if created_at is not None:
        assert keys[0].created_at == _NATIVE


@pytest.mark.asyncio
@pytest.mark.parametrize("created_at", [_NATIVE, _ISO, None])
async def test_user_master_list_keys_parses_both_backends(created_at):
    # user-master SELECT: id, master_did, provider_id, is_active, created_at
    db = _StubDB([("row-id", "did:user", "openrouter", 1, created_at)])
    keys = await UserMasterKeyStorage(db, "did:user").list_keys()

    assert len(keys) == 1
    assert isinstance(keys[0], UserMasterKeyInfo)
    assert isinstance(keys[0].created_at, datetime)
    if created_at is not None:
        assert keys[0].created_at == _NATIVE


@pytest.mark.asyncio
@pytest.mark.parametrize("created_at", [_NATIVE, _ISO, None])
async def test_user_byok_list_keys_parses_both_backends(created_at):
    # user-byok SELECT: id, agent_did, provider_id, is_active, created_at
    db = _StubDB([("row-id", "did:agent", "openrouter", 1, created_at)])
    keys = await UserBYOKKeyStorage(db, "did:agent").list_keys()

    assert len(keys) == 1
    assert isinstance(keys[0], UserBYOKKeyInfo)
    assert isinstance(keys[0].created_at, datetime)
    if created_at is not None:
        assert keys[0].created_at == _NATIVE
