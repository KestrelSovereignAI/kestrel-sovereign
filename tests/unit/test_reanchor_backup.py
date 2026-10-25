"""Where a reanchor's database copy goes, and how it is written (#3785).

Reanchor copies piled up beside the agent databases (18.6 GB in 24 files on
the production fleet). They now go under the configured backup root, and
each is published under its final name only once it is complete. Nothing
here deletes a copy: that is the Sovereign's call.
"""

from __future__ import annotations

import errno
import os
import shutil
import sqlite3
import stat
from contextlib import closing
from pathlib import Path

import pytest

from kestrel_sovereign import paths
from kestrel_sovereign.storage import db_backups
from kestrel_sovereign.storage.db_backups import (
    reanchor_backup_path,
    write_reanchor_backup,
)

DB_NAME = "kestrel_prime.db"


def _database(path: Path, rows: int) -> Path:
    """A real SQLite file whose bytes depend on *rows*."""
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE t (v TEXT)")
        connection.executemany(
            "INSERT INTO t VALUES (?)", [("x" * 200 + str(i),) for i in range(rows)]
        )
        connection.commit()
    return path


@pytest.fixture
def agent_dir(tmp_path):
    directory = tmp_path / "agent_data" / "meridian"
    directory.mkdir(parents=True)
    _database(directory / DB_NAME, 50)
    return directory


@pytest.fixture
def backup_root(tmp_path):
    return tmp_path / "backups"


def _files_below(root: Path) -> list:
    return sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    )


def test_a_copy_lands_under_the_backup_root_not_beside_the_database(
    agent_dir, backup_root,
):
    from kestrel_sovereign.setup.constitution_reanchor import _backup_db

    db = agent_dir / DB_NAME
    copy = _backup_db(db, backup_root)

    event, agent, name = copy.relative_to(backup_root).parts
    assert event.endswith("-reanchor")
    assert agent == "meridian"
    assert name.startswith(f"{DB_NAME}.backup-")
    assert copy == reanchor_backup_path(db, backup_root).with_name(name)
    assert copy.read_bytes() == db.read_bytes()
    assert copy.stat().st_mtime_ns == db.stat().st_mtime_ns, (
        "the copy keeps the database's mtime, as shutil.copy2 did"
    )
    assert copy.stat().st_nlink == 1, (
        "the staging name is gone, so the copy is one exclusively held file"
    )
    assert [p.name for p in agent_dir.iterdir()] == [DB_NAME], (
        "nothing is written beside the database any more"
    )
    assert _files_below(backup_root) == [str(copy.relative_to(backup_root))]


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory modes")
def test_the_directories_it_creates_are_private(agent_dir, backup_root):
    copy = write_reanchor_backup(agent_dir / DB_NAME, backup_root)
    for directory in (backup_root, copy.parent.parent, copy.parent):
        assert stat.S_IMODE(directory.stat().st_mode) & 0o077 == 0, directory


def test_an_existing_backup_root_behind_a_symlink_is_used(
    agent_dir, tmp_path,
):
    """The production layout: ``<project>/backups`` links to a backup volume."""
    volume = tmp_path / "volume" / "kestrel"
    volume.mkdir(parents=True)
    (volume / "MANIFEST.tsv").write_text("path\tbytes\n")
    link = tmp_path / "project" / "backups"
    link.parent.mkdir()
    link.symlink_to(volume, target_is_directory=True)

    copy = write_reanchor_backup(agent_dir / DB_NAME, link)

    assert (volume / copy.relative_to(link)).read_bytes() == (
        agent_dir / DB_NAME
    ).read_bytes()
    assert (volume / "MANIFEST.tsv").read_text() == "path\tbytes\n"


def test_a_failed_copy_leaves_nothing_under_any_name(
    agent_dir, backup_root, monkeypatch,
):
    """``shutil.copy2`` used to write straight to the final name, so a copy
    that failed part-way left a truncated file that looked like a backup."""
    db = agent_dir / DB_NAME

    def fail_half_way(source, target, length=0):
        target.write(source.read(db.stat().st_size // 2))
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(shutil, "copyfileobj", fail_half_way)
    with pytest.raises(OSError, match="No space left"):
        write_reanchor_backup(db, backup_root)

    assert _files_below(backup_root) == [], (
        "neither a partial copy nor its staging file is left behind"
    )
    assert [p.name for p in agent_dir.iterdir()] == [DB_NAME]


def test_a_copy_in_the_same_second_never_overwrites_the_first(
    agent_dir, backup_root, monkeypatch,
):
    """Two forced reanchors of one agent can run within a second (a repair,
    then the reanchor). ``shutil.copy2`` silently replaced the first copy."""
    db = agent_dir / DB_NAME
    first = write_reanchor_backup(db, backup_root)
    before = first.read_bytes()
    with closing(sqlite3.connect(db)) as connection:
        connection.execute("INSERT INTO t VALUES ('written after the copy')")
        connection.commit()
    monkeypatch.setattr(
        db_backups, "reanchor_backup_path", lambda _db, _root: first
    )

    second = write_reanchor_backup(db, backup_root)
    third = write_reanchor_backup(db, backup_root)

    assert (second.name, third.name) == (f"{first.name}-1", f"{first.name}-2")
    assert first.read_bytes() == before
    assert second.read_bytes() == db.read_bytes() != before
    assert _files_below(backup_root) == sorted(
        str(path.relative_to(backup_root)) for path in (first, second, third)
    )


def test_a_copy_gives_up_rather_than_overwrite(agent_dir, backup_root, monkeypatch):
    db = agent_dir / DB_NAME
    first = write_reanchor_backup(db, backup_root)
    monkeypatch.setattr(db_backups, "_MAX_COPIES_PER_SECOND", 2)
    monkeypatch.setattr(
        db_backups, "reanchor_backup_path", lambda _db, _root: first
    )
    write_reanchor_backup(db, backup_root)
    before = {path: path.read_bytes() for path in first.parent.iterdir()}

    with pytest.raises(FileExistsError, match="none is ever overwritten"):
        write_reanchor_backup(db, backup_root)

    assert {path: path.read_bytes() for path in first.parent.iterdir()} == before


def test_a_root_whose_parent_is_missing_is_refused_not_created(
    agent_dir, tmp_path,
):
    """A root on an unmounted volume must not be recreated on the boot disk."""
    unmounted = tmp_path / "Volumes" / "data2" / "backups" / "kestrel"

    with pytest.raises(FileNotFoundError, match="KESTREL_BACKUP_DIR"):
        write_reanchor_backup(agent_dir / DB_NAME, unmounted)

    assert not (tmp_path / "Volumes").exists()


def test_a_root_linking_to_a_missing_directory_is_refused(agent_dir, tmp_path):
    link = tmp_path / "backups"
    link.symlink_to(tmp_path / "unmounted" / "kestrel", target_is_directory=True)

    with pytest.raises(NotADirectoryError, match="does not exist"):
        write_reanchor_backup(agent_dir / DB_NAME, link)

    assert not (tmp_path / "unmounted").exists()


def test_a_root_that_is_a_file_is_refused(agent_dir, tmp_path):
    root = tmp_path / "backups"
    root.write_text("not a directory")

    with pytest.raises(NotADirectoryError, match="not a directory"):
        write_reanchor_backup(agent_dir / DB_NAME, root)

    assert root.read_text() == "not a directory"


# --- resolving the backup root -----------------------------------------------


def test_the_backup_root_defaults_below_the_project(tmp_path):
    assert paths.backup_dir(tmp_path, environ={}) == tmp_path / "backups"


def test_the_backup_root_follows_the_environment_it_is_given(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.BACKUP_DIR_ENV, str(tmp_path / "from-os-environ"))
    environ = {paths.BACKUP_DIR_ENV: str(tmp_path / "offsite")}

    assert paths.backup_dir(tmp_path, environ=environ) == tmp_path / "offsite"
    assert paths.backup_dir(tmp_path) == tmp_path / "from-os-environ"


def test_a_relative_backup_root_is_relative_to_the_project(tmp_path):
    environ = {paths.BACKUP_DIR_ENV: "copies/kestrel"}

    assert paths.backup_dir(tmp_path, environ=environ) == (
        tmp_path / "copies" / "kestrel"
    )


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_backup_root_is_refused_not_defaulted(tmp_path, blank):
    with pytest.raises(ValueError, match="set but empty"):
        paths.backup_dir(tmp_path, environ={paths.BACKUP_DIR_ENV: blank})


def test_the_backup_root_is_a_guarded_runtime_path(tmp_path, storage_path_refusals):
    """Under the test harness a root outside the temporary roots is refused,
    so no test can write a database copy into an operator's backup volume."""
    with pytest.raises(paths.StoragePathOutsideTestRootsError):
        paths.backup_dir(
            tmp_path, environ={paths.BACKUP_DIR_ENV: "/nonexistent-operator-volume"}
        )
    assert [r.source for r in storage_path_refusals.drain()] == [
        paths.BACKUP_DIR_ENV
    ]
