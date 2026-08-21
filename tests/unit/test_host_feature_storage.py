"""Private custody tests for the fleet/host feature SQLite database (#2610)."""

from __future__ import annotations

import errno
import os
import sqlite3
import sys
from pathlib import Path

import pytest

from kestrel_sovereign.host_features.context import build_host_context
from kestrel_sovereign.host_features.storage import (
    HOST_DB_PATH_ENV,
    HOST_FEATURE_DB_FILENAME,
    HostStorageError,
    host_database_path,
    prepare_host_database,
)


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


async def _close_context(ctx) -> None:
    if ctx.session_factory is not None:
        await ctx.session_factory.close()
    if ctx.db is not None:
        await ctx.db.close()


def _create_legacy_sqlite(path: Path, value: str = "history") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("CREATE TABLE legacy_probe (value TEXT NOT NULL)")
        connection.execute("INSERT INTO legacy_probe VALUES (?)", (value,))
        connection.commit()
    finally:
        connection.close()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX custody contract")
async def test_default_host_database_is_private_at_creation_under_umask_zero(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source-checkout"
    (source / "kestrel_sovereign").mkdir(parents=True)
    (source / "kestrel_sovereign" / "__init__.py").write_text("")
    operator_home = tmp_path / "operator-home"
    operator_home.mkdir()
    monkeypatch.chdir(source)
    monkeypatch.setenv("HOME", str(operator_home))
    monkeypatch.delenv("KESTREL_HOME", raising=False)
    monkeypatch.delenv(HOST_DB_PATH_ENV, raising=False)

    previous_umask = os.umask(0)
    try:
        ctx = await build_host_context()
    finally:
        os.umask(previous_umask)

    assert ctx.db is not None
    db_path = operator_home / ".kestrel" / "host-data" / HOST_FEATURE_DB_FILENAME
    try:
        await ctx.db.execute("CREATE TABLE custody_probe (value TEXT)")
        assert source not in db_path.parents
        assert _mode(operator_home / ".kestrel") == 0o700
        assert _mode(db_path.parent) == 0o700
        assert _mode(db_path) == 0o600
        assert _mode(Path(f"{db_path}-wal")) == 0o600
        assert _mode(Path(f"{db_path}-shm")) == 0o600
    finally:
        await _close_context(ctx)

    reopened = await build_host_context()
    try:
        assert reopened.db is not None
        assert await reopened.db.table_exists("custody_probe") is True
        assert _mode(db_path) == 0o600
    finally:
        await _close_context(reopened)


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX custody contract")
async def test_custom_env_path_is_supported_hardened_and_reopened(
    tmp_path, monkeypatch,
):
    private_parent = tmp_path / "host-volume"
    private_parent.mkdir(mode=0o700)
    db_path = private_parent / "custom.db"
    db_path.write_bytes(b"")
    db_path.chmod(0o666)
    monkeypatch.setenv(HOST_DB_PATH_ENV, str(db_path))

    previous_umask = os.umask(0)
    try:
        ctx = await build_host_context()
    finally:
        os.umask(previous_umask)
    try:
        assert ctx.db is not None
        await ctx.db.execute("CREATE TABLE custom_probe (value TEXT)")
        assert ctx.db.backend.db_path == str(db_path)
        assert _mode(private_parent) == 0o700
        assert all(
            _mode(path) == 0o600
            for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm"))
        )
    finally:
        await _close_context(ctx)

    reopened = await build_host_context()
    try:
        assert reopened.db is not None
        assert await reopened.db.table_exists("custom_probe") is True
    finally:
        await _close_context(reopened)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX custody contract")
def test_custom_path_refuses_shared_parent_without_chmod(tmp_path):
    shared_parent = tmp_path / "shared"
    shared_parent.mkdir()
    # chmod, not mkdir(mode=...): mkdir's mode is masked by the process
    # umask, so under a 0o077 umask this "shared" parent was created 0o700
    # and the custody guard correctly did not fire -- the test then failed
    # against its own setup rather than against the contract.
    shared_parent.chmod(0o755)

    with pytest.raises(HostStorageError, match="must have mode 0700"):
        prepare_host_database(str(shared_parent / "host.db"))

    assert _mode(shared_parent) == 0o755
    assert not (shared_parent / "host.db").exists()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX custody contract")
async def test_context_disables_store_when_custom_parent_is_not_private(
    tmp_path, monkeypatch,
):
    shared_parent = tmp_path / "shared"
    shared_parent.mkdir()
    # chmod, not mkdir(mode=...): mkdir's mode is masked by the process
    # umask, so under a 0o077 umask this "shared" parent was created 0o700
    # and the custody guard correctly did not fire -- the test then failed
    # against its own setup rather than against the contract.
    shared_parent.chmod(0o755)
    monkeypatch.setenv(HOST_DB_PATH_ENV, str(shared_parent / "host.db"))

    ctx = await build_host_context()

    assert ctx.db is None
    assert ctx.session_factory is None
    assert _mode(shared_parent) == 0o755


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX custody contract")
def test_custom_path_creates_missing_dedicated_parent_privately(tmp_path):
    parent = tmp_path / "new-private-volume"
    db_path = parent / "host.db"

    assert prepare_host_database(str(db_path)) == db_path

    assert _mode(parent) == 0o700
    assert _mode(db_path) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX link contract")
def test_custom_path_rejects_symbolic_link_parent(tmp_path):
    real_parent = tmp_path / "real-private"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked-private"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(HostStorageError, match="real directory"):
        prepare_host_database(str(linked_parent / "host.db"))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX link contract")
@pytest.mark.parametrize("link_target", ["database", "sidecar"])
def test_custom_path_rejects_symbolic_links(tmp_path, link_target):
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    db_path = private_parent / "host.db"
    outside = tmp_path / "outside"
    outside.write_text("sensitive")
    if link_target == "database":
        db_path.symlink_to(outside)
    else:
        db_path.write_bytes(b"")
        Path(f"{db_path}-wal").symlink_to(outside)

    with pytest.raises(HostStorageError, match="not regular|cannot open"):
        prepare_host_database(str(db_path))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX hard-link contract")
def test_custom_path_rejects_multiply_linked_database(tmp_path):
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    db_path = private_parent / "host.db"
    db_path.write_bytes(b"")
    os.link(db_path, tmp_path / "second-name")

    with pytest.raises(HostStorageError, match="hard links"):
        prepare_host_database(str(db_path))


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX migration contract")
async def test_default_migrates_and_hardens_stopped_legacy_database(
    tmp_path, monkeypatch,
):
    home = tmp_path / "kestrel-home"
    legacy = home / "kestrel_host.db"
    _create_legacy_sqlite(legacy)
    legacy.chmod(0o644)
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.delenv(HOST_DB_PATH_ENV, raising=False)

    ctx = await build_host_context()
    destination = home / "host-data" / HOST_FEATURE_DB_FILENAME
    try:
        assert ctx.db is not None
        assert not legacy.exists()
        assert destination.exists()
        assert _mode(destination.parent) == 0o700
        assert _mode(destination) == 0o600
        assert await ctx.db.fetchval("SELECT value FROM legacy_probe") == "history"
    finally:
        await _close_context(ctx)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX migration contract")
def test_cross_filesystem_migration_uses_private_staging(tmp_path, monkeypatch):
    home = tmp_path / "kestrel-home"
    legacy = home / "kestrel_host.db"
    _create_legacy_sqlite(legacy, value="cross-device")
    legacy.chmod(0o644)
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.delenv(HOST_DB_PATH_ENV, raising=False)

    from kestrel_sovereign.host_features import storage

    real_replace = storage.os.replace

    def _replace_with_one_cross_device_failure(source, destination):
        if Path(source) == legacy:
            raise OSError(errno.EXDEV, "cross-device link")
        return real_replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", _replace_with_one_cross_device_failure)
    destination = prepare_host_database()

    assert not legacy.exists()
    assert _mode(destination) == 0o600
    assert not list(destination.parent.glob(".host-features-migrate-*"))
    connection = sqlite3.connect(destination)
    try:
        assert connection.execute("SELECT value FROM legacy_probe").fetchone() == (
            "cross-device",
        )
    finally:
        connection.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX migration contract")
def test_live_legacy_sidecars_are_contained_and_fail_closed(tmp_path, monkeypatch):
    home = tmp_path / "kestrel-home"
    legacy = home / "kestrel_host.db"
    _create_legacy_sqlite(legacy)
    sidecars = [Path(f"{legacy}-wal"), Path(f"{legacy}-shm")]
    for path in (legacy, *sidecars):
        if path != legacy:
            path.write_text("possibly live")
        path.chmod(0o644)
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.delenv(HOST_DB_PATH_ENV, raising=False)

    with pytest.raises(HostStorageError, match="another Kestrel process"):
        prepare_host_database()

    assert legacy.exists()
    assert all(_mode(path) == 0o600 for path in (legacy, *sidecars))
    assert not (home / "host-data" / HOST_FEATURE_DB_FILENAME).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX migration contract")
def test_dual_legacy_and_destination_stores_are_contained_and_rejected(
    tmp_path, monkeypatch,
):
    home = tmp_path / "kestrel-home"
    legacy = home / "kestrel_host.db"
    destination = home / "host-data" / HOST_FEATURE_DB_FILENAME
    _create_legacy_sqlite(legacy, value="legacy")
    _create_legacy_sqlite(destination, value="destination")
    legacy.chmod(0o644)
    destination.chmod(0o644)
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.delenv(HOST_DB_PATH_ENV, raising=False)

    with pytest.raises(HostStorageError, match="both legacy host database"):
        prepare_host_database()

    assert _mode(legacy) == 0o600
    assert _mode(destination) == 0o600
    assert legacy.exists() and destination.exists()


def test_host_database_path_distinguishes_explicit_override(tmp_path, monkeypatch):
    override = tmp_path / "private" / "custom.db"
    monkeypatch.setenv(HOST_DB_PATH_ENV, str(override))
    assert host_database_path() == (override.absolute(), False)
