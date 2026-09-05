"""Private custody tests for the fleet/host feature SQLite database (#2610)."""

from __future__ import annotations

import errno
import os
import sqlite3
import sys
import threading
from pathlib import Path
from uuid import uuid4

import pytest

from kestrel_sovereign.host_features.context import build_host_context
from kestrel_sovereign.host_features.storage import (
    DERIVED_HOST_DB_PATH_ENV,
    HOST_DB_PATH_ENV,
    HOST_FEATURE_DB_FILENAME,
    HostStorageError,
    host_database_path,
    prepare_host_database,
    validate_host_database_migration_readiness,
)

# These tests are *about* default host-database resolution, so they set
# HOME / KESTREL_HOME / KESTREL_HOST_DB_PATH themselves (or pass an explicit
# path) and opt out of the suite-wide isolation in tests/unit/conftest.py
# (#3087). Every test below must keep doing so: without the fixture's
# override, a test that forgot would resolve the operator's real database.
pytestmark = pytest.mark.owns_host_paths


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
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)

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
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)

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
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)

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
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)

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
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)

    with pytest.raises(HostStorageError, match="both legacy host database"):
        prepare_host_database()

    assert _mode(legacy) == 0o600
    assert _mode(destination) == 0o600
    assert legacy.exists() and destination.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX migration contract")
def test_destination_hold_remnant_refuses_migration_before_source_move(
    tmp_path,
    monkeypatch,
):
    """A damaged destination cannot consume the only intact prior database."""

    from kestrel_sovereign.hold.state import hold_initialization_witness_path

    home = tmp_path / "kestrel-home"
    legacy = home / "kestrel_host.db"
    destination = home / "host-data" / HOST_FEATURE_DB_FILENAME
    _create_legacy_sqlite(legacy, value="authoritative-source")
    witness = hold_initialization_witness_path(destination)
    witness.parent.mkdir(parents=True, mode=0o700)
    witness.write_text("damaged-destination-custody")
    legacy.chmod(0o644)
    witness.chmod(0o600)
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.delenv(HOST_DB_PATH_ENV, raising=False)
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)

    with pytest.raises(HostStorageError, match="destination.*custody remnants"):
        prepare_host_database()

    assert legacy.exists()
    assert not destination.exists()
    assert witness.read_text() == "damaged-destination-custody"
    assert _mode(legacy) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX migration contract")
def test_migration_skips_source_alias_of_destination(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    source = real_parent / HOST_FEATURE_DB_FILENAME
    destination = alias_parent / HOST_FEATURE_DB_FILENAME
    _create_legacy_sqlite(source)

    selected = validate_host_database_migration_readiness(
        destination,
        (("legacy host database", source),),
    )

    assert selected is None


def test_host_database_path_distinguishes_explicit_override(tmp_path, monkeypatch):
    override = tmp_path / "private" / "custom.db"
    monkeypatch.setenv(HOST_DB_PATH_ENV, str(override))
    assert host_database_path() == (override.absolute(), False)


def test_host_database_path_follows_agent_data_root_without_override(
    tmp_path,
    monkeypatch,
):
    """One Docker data-root override moves agent and Hold custody together."""

    data_root = tmp_path / "mounted-data"
    monkeypatch.delenv(HOST_DB_PATH_ENV, raising=False)
    monkeypatch.setenv("KESTREL_DB_PATH", str(data_root))

    assert host_database_path() == (
        data_root / "host-data" / HOST_FEATURE_DB_FILENAME,
        False,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX migration contract")
def test_agent_data_root_migrates_previous_default_host_database(
    tmp_path,
    monkeypatch,
):
    """Changing the implicit root must not hide the pre-upgrade host store."""

    home = tmp_path / "kestrel-home"
    previous = home / "host-data" / HOST_FEATURE_DB_FILENAME
    data_root = tmp_path / "mounted-data"
    destination = data_root / "host-data" / HOST_FEATURE_DB_FILENAME
    _create_legacy_sqlite(previous, value="pre-upgrade")
    previous.chmod(0o644)
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.setenv("KESTREL_DB_PATH", str(data_root))
    monkeypatch.delenv(HOST_DB_PATH_ENV, raising=False)

    assert prepare_host_database() == destination
    assert not previous.exists()
    assert _mode(destination) == 0o600
    with sqlite3.connect(destination) as connection:
        assert connection.execute("SELECT value FROM legacy_probe").fetchone() == (
            "pre-upgrade",
        )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX migration interleaving")
def test_concurrent_host_database_migration_adopts_one_completed_move(
    tmp_path,
    monkeypatch,
):
    """Sibling fleet boots serialize the one-time move instead of aborting."""

    from kestrel_sovereign.host_features import storage as storage_module

    source = tmp_path / "previous" / HOST_FEATURE_DB_FILENAME
    destination = tmp_path / "current" / HOST_FEATURE_DB_FILENAME
    destination.parent.mkdir(mode=0o700)
    _create_legacy_sqlite(source, value="shared-upgrade-history")
    source.chmod(0o600)
    first_replace_entered = threading.Event()
    release_first_replace = threading.Event()
    replace_calls = 0
    replace_calls_lock = threading.Lock()
    outcomes: list[BaseException | None] = []
    real_replace = storage_module.os.replace

    def pause_first_replace(old, new):
        nonlocal replace_calls
        with replace_calls_lock:
            replace_calls += 1
            call_number = replace_calls
        if call_number == 1:
            first_replace_entered.set()
            assert release_first_replace.wait(5)
        return real_replace(old, new)

    monkeypatch.setattr(storage_module.os, "replace", pause_first_replace)

    def migrate() -> None:
        failure = None
        try:
            storage_module._migrate_prior_database(
                destination,
                (("previous default host database", source),),
            )
        except BaseException as exc:  # captured for the parent test thread
            failure = exc
        outcomes.append(failure)

    first = threading.Thread(target=migrate, daemon=True)
    second = threading.Thread(target=migrate, daemon=True)
    first.start()
    assert first_replace_entered.wait(5)
    second.start()
    second.join(0.1)
    assert second.is_alive()
    release_first_replace.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert outcomes == [None, None]
    assert replace_calls == 1
    assert not source.exists()
    with sqlite3.connect(destination) as connection:
        assert connection.execute("SELECT value FROM legacy_probe").fetchone() == (
            "shared-upgrade-history",
        )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX migration contract")
def test_agent_data_root_refuses_migration_with_backend_custody_binding(
    tmp_path,
    monkeypatch,
):
    """An implicit root change cannot strand a PostgreSQL Hold selection."""

    from kestrel_sovereign.hold.state import (
        claim_hold_backend_custody,
        hold_backend_binding_path,
    )

    home = tmp_path / "kestrel-home"
    previous = home / "host-data" / HOST_FEATURE_DB_FILENAME
    data_root = tmp_path / "mounted-data"
    destination = data_root / "host-data" / HOST_FEATURE_DB_FILENAME
    _create_legacy_sqlite(previous, value="postgres-host-features")
    claim_hold_backend_custody(
        previous,
        "postgres",
        postgres_pair_id=uuid4(),
        postgres_primary_cluster_identity="primary-cluster",
        postgres_evidence_cluster_identity="evidence-cluster",
    )
    binding = hold_backend_binding_path(previous)
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.setenv("KESTREL_DB_PATH", str(data_root))
    monkeypatch.delenv(HOST_DB_PATH_ENV, raising=False)

    with pytest.raises(HostStorageError, match="Hold custody evidence"):
        prepare_host_database()

    assert previous.exists()
    assert binding.exists()
    assert not destination.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX migration contract")
def test_launcher_derived_host_path_keeps_implicit_migration_semantics(
    tmp_path,
    monkeypatch,
):
    """Publishing one fleet path to a child must not strand prior state."""

    home = tmp_path / "kestrel-home"
    previous = home / "host-data" / HOST_FEATURE_DB_FILENAME
    fleet_root = tmp_path / "mounted-data"
    destination = fleet_root / "host-data" / HOST_FEATURE_DB_FILENAME
    _create_legacy_sqlite(previous, value="pre-launcher-upgrade")
    previous.chmod(0o644)
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path / "agent_data" / "alice"))
    monkeypatch.setenv(HOST_DB_PATH_ENV, str(destination))
    monkeypatch.setenv(DERIVED_HOST_DB_PATH_ENV, str(destination))

    assert prepare_host_database() == destination
    assert not previous.exists()
    with sqlite3.connect(destination) as connection:
        assert connection.execute("SELECT value FROM legacy_probe").fetchone() == (
            "pre-launcher-upgrade",
        )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX migration contract")
def test_agent_data_root_rejects_previous_and_destination_histories(
    tmp_path,
    monkeypatch,
):
    """An upgrade refuses two histories instead of selecting the blanker one."""

    home = tmp_path / "kestrel-home"
    previous = home / "host-data" / HOST_FEATURE_DB_FILENAME
    data_root = tmp_path / "mounted-data"
    destination = data_root / "host-data" / HOST_FEATURE_DB_FILENAME
    _create_legacy_sqlite(previous, value="pre-upgrade")
    _create_legacy_sqlite(destination, value="new-root")
    destination.parent.chmod(0o700)
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.setenv("KESTREL_DB_PATH", str(data_root))
    monkeypatch.delenv(HOST_DB_PATH_ENV, raising=False)

    with pytest.raises(HostStorageError, match="both previous default"):
        prepare_host_database()

    assert previous.exists() and destination.exists()
