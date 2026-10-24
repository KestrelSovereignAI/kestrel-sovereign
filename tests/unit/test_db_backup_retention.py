"""Retention for reanchor copies of an agent database (#3785).

Reanchor copies piled up beside the agent databases (18.6 GB in 24 files on
the production fleet, two of them identical). The policy keeps the newest
distinct copies, removes nothing without ``--apply``, and never touches a
file it does not manage: forensic copies and inception's lifetime archives in
particular.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import sqlite3
from argparse import Namespace
from contextlib import closing
from pathlib import Path

import pytest

from kestrel_sovereign.storage import db_backups
from kestrel_sovereign.storage.db_backups import (
    DEFAULT_KEEP,
    apply_backup_retention,
    plan_backup_retention,
    reanchor_backup_path,
)

DB_NAME = "kestrel_prime.db"


def _database(path: Path, rows: int) -> Path:
    """A real SQLite file whose bytes depend on *rows*."""
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE t (v TEXT)")
        connection.executemany(
            "INSERT INTO t VALUES (?)", [("x" * 200 + str(i),) for i in range(rows)]
        )
        connection.commit()
    finally:
        connection.close()
    return path


def _copy(db: Path, stamp: str, source: Path | None = None) -> Path:
    target = db.with_name(f"{db.name}.backup-{stamp}")
    target.write_bytes((source or db).read_bytes())
    return target


def _actions(plan) -> dict:
    return {d.copy.path.name: (d.keep, d.reason) for d in plan.decisions}


@pytest.fixture
def agent_dir(tmp_path):
    directory = tmp_path / "agent_data" / "meridian"
    directory.mkdir(parents=True)
    _database(directory / DB_NAME, 50)
    return directory


def _quick_check(path: Path) -> list:
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        return connection.execute("PRAGMA quick_check").fetchall()


def test_reanchor_writer_names_its_copies_the_way_the_policy_reads_them(
    agent_dir,
):
    from kestrel_sovereign.setup.constitution_reanchor import _backup_db

    db = agent_dir / DB_NAME
    copy = _backup_db(db)
    assert copy == reanchor_backup_path(db).with_name(copy.name)
    assert copy.read_bytes() == db.read_bytes()
    assert copy.stat().st_mtime_ns == db.stat().st_mtime_ns, (
        "the copy keeps the database's mtime, as shutil.copy2 did"
    )
    assert sorted(p.name for p in agent_dir.iterdir()) == sorted([DB_NAME, copy.name])
    plan = plan_backup_retention(db)
    assert [d.copy.path for d in plan.decisions] == [copy]


def test_a_failed_reanchor_copy_leaves_nothing_the_policy_counts(
    agent_dir, tmp_path, monkeypatch,
):
    """A copy that fails part-way must not displace a usable older copy.

    ``shutil.copy2`` used to write straight to the final name, so a failed
    copy left a truncated file there that the policy counted as the newest.
    """
    from kestrel_sovereign.setup.constitution_reanchor import _backup_db

    db = agent_dir / DB_NAME
    older = _copy(db, "20261006-120000", _database(tmp_path / "older.db", 80))

    def fail_half_way(source, target, length=0):
        target.write(source.read(db.stat().st_size // 2))
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(shutil, "copyfileobj", fail_half_way)
    with pytest.raises(OSError, match="No space left"):
        _backup_db(db)
    monkeypatch.undo()

    assert sorted(p.name for p in agent_dir.iterdir()) == sorted([DB_NAME, older.name]), (
        "neither a partial copy nor its staging file is left behind"
    )
    plan = plan_backup_retention(db, keep=1)
    assert [(d.copy.path, d.keep, d.counted) for d in plan.decisions] == [
        (older, True, True),
    ]
    assert apply_backup_retention(plan) == []
    assert _quick_check(older) == [("ok",)]


def test_a_reanchor_copy_is_never_overwritten(agent_dir, monkeypatch):
    from kestrel_sovereign.setup.constitution_reanchor import _backup_db

    db = agent_dir / DB_NAME
    first = _backup_db(db)
    before = first.read_bytes()
    with closing(sqlite3.connect(db)) as connection:
        connection.execute("INSERT INTO t VALUES ('written after the copy')")
        connection.commit()
    monkeypatch.setattr(db_backups, "reanchor_backup_path", lambda _db: first)

    with pytest.raises(FileExistsError, match="never overwritten"):
        _backup_db(db)

    assert first.read_bytes() == before
    assert not list(agent_dir.glob("*.partial-backup-*"))


def test_identical_copies_collapse_to_the_newest(agent_dir):
    db = agent_dir / DB_NAME
    earlier = _copy(db, "20261006-120000")
    later = _copy(db, "20261006-120001")

    plan = plan_backup_retention(db)

    assert _actions(plan) == {
        later.name: (True, f"one of the newest {DEFAULT_KEEP}"),
        earlier.name: (False, f"identical to {later.name}"),
    }
    assert plan.reclaimable_bytes == earlier.stat().st_size
    decision = next(d for d in plan.decisions if not d.keep)
    assert decision.copy.sha256 == next(
        d for d in plan.decisions if d.keep
    ).copy.sha256


def test_only_the_newest_distinct_copies_are_kept(agent_dir, tmp_path):
    db = agent_dir / DB_NAME
    stamps = ["20260718-095147", "20260808-061008", "20261006-120000", "20261020-080000"]
    for index, stamp in enumerate(stamps):
        _copy(db, stamp, _database(tmp_path / f"src-{index}.db", 50 * (index + 2)))

    plan = plan_backup_retention(db, keep=2)

    kept = [d.copy.stamp for d in plan.decisions if d.keep]
    removed = [d.copy.stamp for d in plan.decisions if not d.keep]
    assert kept == ["20261020-080000", "20261006-120000"]
    assert removed == ["20260808-061008", "20260718-095147"]
    assert all(d.copy.sha256 is None for d in plan.decisions), (
        "copies of distinct sizes are never hashed"
    )


def test_a_duplicate_of_an_expired_copy_is_expired_too(agent_dir, tmp_path):
    db = agent_dir / DB_NAME
    _copy(db, "20261020-080000", _database(tmp_path / "newest.db", 400))
    old = _database(tmp_path / "old.db", 300)
    _copy(db, "20260101-000001", old)
    _copy(db, "20260101-000000", old)

    plan = plan_backup_retention(db, keep=1)

    assert {name: keep for name, (keep, _) in _actions(plan).items()} == {
        f"{DB_NAME}.backup-20261020-080000": True,
        f"{DB_NAME}.backup-20260101-000001": False,
        f"{DB_NAME}.backup-20260101-000000": False,
    }
    assert all(d.duplicate_of is None for d in plan.removals)


def _truncate(path: Path, keep_bytes: int) -> Path:
    with open(path, "r+b") as handle:
        handle.truncate(keep_bytes)
    return path


def test_a_truncated_copy_is_kept_but_never_counted(agent_dir, tmp_path):
    """The reproduced failure: a newer truncated copy beside a usable one.

    With ``keep=1`` the truncated copy used to be "one of the newest 1" and
    the usable copy was removed, leaving a file SQLite cannot open.
    """
    db = agent_dir / DB_NAME
    usable = _copy(db, "20261006-120000", _database(tmp_path / "usable.db", 80))
    truncated = _copy(db, "20261021-090000", _database(tmp_path / "newer.db", 200))
    _truncate(truncated, truncated.stat().st_size // 2 + 7)

    plan = plan_backup_retention(db, keep=1)

    decisions = {d.copy.path: d for d in plan.decisions}
    assert decisions[truncated].keep and not decisions[truncated].counted
    assert "not counted toward the newest 1: incomplete" in decisions[truncated].reason
    assert decisions[usable].keep and decisions[usable].counted
    assert plan.removals == ()
    assert apply_backup_retention(plan) == []
    assert _quick_check(usable) == [("ok",)]
    assert truncated.exists(), "an unreadable copy is reported, never removed"


def test_an_unreadable_copy_does_not_displace_the_next_usable_one(agent_dir, tmp_path):
    db = agent_dir / DB_NAME
    oldest = _copy(db, "20260718-095147", _database(tmp_path / "a.db", 60))
    middle = _copy(db, "20260808-061008", _database(tmp_path / "b.db", 120))
    newest = _copy(db, "20261021-090000", _database(tmp_path / "c.db", 240))
    _truncate(newest, 4096 * 3)

    plan = plan_backup_retention(db, keep=1)

    assert [(d.copy.path, d.keep, d.counted) for d in plan.decisions] == [
        (newest, True, False),
        (middle, True, True),
        (oldest, False, False),
    ]
    assert [(r.path, r.removed) for r in apply_backup_retention(plan)] == [(oldest, True)]


def _corrupt_a_page(path: Path) -> None:
    page_size = 4096
    with open(path, "r+b") as handle:
        handle.seek(page_size * 10)
        handle.write(b"\xff" * page_size)


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        # SQLite itself reads an empty file as an empty database.
        pytest.param(lambda p: _truncate(p, 0), "not a SQLite database", id="empty"),
        pytest.param(lambda p: _truncate(p, 50), "not a SQLite database", id="partial-header"),
        pytest.param(
            lambda p: _truncate(p, 4096 * 5), "incomplete: the header names",
            id="whole-pages-missing",
        ),
        pytest.param(
            _corrupt_a_page, "SQLite (cannot read it|reports damage)", id="damaged-page",
        ),
        pytest.param(
            lambda p: p.write_bytes(b"not a database" * 500), "not a SQLite database",
            id="not-sqlite",
        ),
    ],
)
def test_a_damaged_copy_is_not_counted(agent_dir, tmp_path, damage, reason):
    db = agent_dir / DB_NAME
    usable = _copy(db, "20261006-120000", _database(tmp_path / "usable.db", 80))
    damaged = _copy(db, "20261021-090000", _database(tmp_path / "newer.db", 400))
    damage(damaged)

    plan = plan_backup_retention(db, keep=1)

    assert [(d.copy.path, d.keep, d.counted) for d in plan.decisions] == [
        (damaged, True, False),
        (usable, True, True),
    ]
    assert re.search(f"not counted toward the newest 1: {reason}", plan.decisions[0].reason)


def test_copies_are_read_only_when_something_would_be_removed(
    agent_dir, tmp_path, monkeypatch,
):
    db = agent_dir / DB_NAME
    for index, stamp in enumerate(["20261006-120000", "20261021-090000"]):
        _copy(db, stamp, _database(tmp_path / f"src-{index}.db", 50 * (index + 2)))

    def unexpected(path):
        raise AssertionError(f"{path.name} was read with nothing to remove")

    monkeypatch.setattr(db_backups, "_backup_defect", unexpected)

    plan = plan_backup_retention(db, keep=2)

    assert plan.removals == ()
    assert all(d.counted for d in plan.decisions)


def test_files_the_policy_does_not_manage_are_never_planned(agent_dir):
    db = agent_dir / DB_NAME
    (agent_dir / f"{DB_NAME}.damaged-20260424_112745").write_bytes(db.read_bytes())
    (agent_dir / f"{DB_NAME}.before-meridian-recovery-20260429_103429").write_bytes(
        db.read_bytes()
    )
    # A forced inception's archive is constitutional-lifetime evidence.
    (agent_dir / f"{DB_NAME}.backup-20260501-101010-0a1b2c3d").write_bytes(
        db.read_bytes()
    )
    (agent_dir / f"{DB_NAME}.sync").write_text("{}")
    (agent_dir / f"{DB_NAME}-wal").write_bytes(b"")

    plan = plan_backup_retention(db)

    assert plan.decisions == ()
    assert sorted(path.name for path in plan.preserved) == [
        f"{DB_NAME}.backup-20260501-101010-0a1b2c3d",
        f"{DB_NAME}.before-meridian-recovery-20260429_103429",
        f"{DB_NAME}.damaged-20260424_112745",
    ]


def test_a_hard_linked_copy_is_not_managed(agent_dir):
    db = agent_dir / DB_NAME
    copy = _copy(db, "20261006-120000")
    os.link(copy, agent_dir.parent / "elsewhere.db")

    plan = plan_backup_retention(db)

    assert plan.decisions == ()
    assert plan.preserved == (copy,)


def test_without_the_live_database_every_copy_is_kept(agent_dir):
    db = agent_dir / DB_NAME
    first = _copy(db, "20261006-120000")
    _copy(db, "20261006-120001")
    _copy(db, "20261005-120001", first)
    db.unlink()

    plan = plan_backup_retention(db, keep=1)

    assert plan.database_bytes is None
    assert plan.removals == ()
    assert all("missing" in d.reason for d in plan.decisions)


def test_keep_must_leave_at_least_one_copy(agent_dir):
    with pytest.raises(ValueError):
        plan_backup_retention(agent_dir / DB_NAME, keep=0)


def test_apply_removes_exactly_the_planned_copies(agent_dir):
    db = agent_dir / DB_NAME
    earlier = _copy(db, "20261006-120000")
    later = _copy(db, "20261006-120001")
    forensic = agent_dir / f"{DB_NAME}.damaged-20260424"
    forensic.write_bytes(db.read_bytes())

    results = apply_backup_retention(plan_backup_retention(db))

    assert [(r.path, r.removed) for r in results] == [(earlier, True)]
    assert sorted(p.name for p in agent_dir.iterdir()) == sorted(
        [DB_NAME, later.name, forensic.name]
    )


def test_apply_skips_a_copy_that_changed_after_planning(agent_dir):
    db = agent_dir / DB_NAME
    earlier = _copy(db, "20261006-120000")
    _copy(db, "20261006-120001")
    plan = plan_backup_retention(db)
    with open(earlier, "ab") as handle:
        handle.write(b"later write")

    results = apply_backup_retention(plan)

    assert [(r.removed, r.reason) for r in results] == [
        (False, "changed since it was planned"),
    ]
    assert earlier.exists()


def test_apply_keeps_a_duplicate_whose_original_changed(agent_dir):
    db = agent_dir / DB_NAME
    earlier = _copy(db, "20261006-120000")
    later = _copy(db, "20261006-120001")
    plan = plan_backup_retention(db)
    later.unlink()

    results = apply_backup_retention(plan)

    assert results[0].removed is False
    assert later.name in results[0].reason
    assert earlier.exists(), "the only copy left of that content survives"


def test_apply_keeps_older_copies_when_a_counted_copy_changed(agent_dir, tmp_path):
    db = agent_dir / DB_NAME
    older = _copy(db, "20261006-120000", _database(tmp_path / "older.db", 80))
    newer = _copy(db, "20261021-090000", _database(tmp_path / "newer.db", 200))
    plan = plan_backup_retention(db, keep=1)
    assert [d.copy.path for d in plan.removals] == [older]
    _truncate(newer, 4096)

    results = apply_backup_retention(plan)

    assert [(r.path, r.removed, r.reason) for r in results] == [
        (older, False, f"{newer.name} changed since it was planned"),
    ]
    assert older.exists()


# --- kestrel storage backups -------------------------------------------------


def _args(**overrides) -> Namespace:
    values = dict(agent=None, db=None, keep=DEFAULT_KEEP, apply=False, json=False)
    values.update(overrides)
    return Namespace(**values)


def test_cli_reports_without_removing(agent_dir, capsys):
    from kestrel_sovereign.cli import cmd_storage_backups

    db = agent_dir / DB_NAME
    earlier = _copy(db, "20261006-120000")
    _copy(db, "20261006-120001")

    assert cmd_storage_backups(_args(db=str(db))) == 0

    out = capsys.readouterr().out
    assert f"remove  {earlier.name}" in out
    assert "Nothing was removed" in out
    assert earlier.exists()


def test_cli_apply_removes_and_reports_json(agent_dir, capsys):
    from kestrel_sovereign.cli import cmd_storage_backups

    db = agent_dir / DB_NAME
    earlier = _copy(db, "20261006-120000")
    _copy(db, "20261006-120001")

    assert cmd_storage_backups(_args(db=str(db), apply=True, json=True)) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["applied"] is True
    assert report["agents"][str(db)]["removed"] == [str(earlier)]
    assert not earlier.exists()


def test_cli_walks_every_local_agent(tmp_path, monkeypatch, capsys):
    from kestrel_sovereign import cli

    home = tmp_path / "home"
    (home / "agent_data" / "emma").mkdir(parents=True)
    (home / "agent_data" / "claw").mkdir(parents=True)
    for name in ("emma", "claw"):
        _database(home / "agent_data" / name / DB_NAME, 10)
    (home / "multi_agent.toml").write_text(
        '[agents.Emma]\ndata_dir = "agent_data/emma"\nport = 8801\n\n'
        '[agents.Claw]\ndata_dir = "agent_data/claw"\nport = 8802\n'
    )
    monkeypatch.setattr(cli, "_get_project_dir", lambda: home)

    assert cli.cmd_storage_backups(_args(json=True)) == 0
    agents = json.loads(capsys.readouterr().out)["agents"]
    assert sorted(agents) == ["Claw", "Emma"]

    assert cli.cmd_storage_backups(_args(agent="Nellie")) == 2
    assert "not in multi_agent" in capsys.readouterr().err


def test_cli_refuses_keep_below_one(capsys):
    from kestrel_sovereign.cli import cmd_storage_backups

    assert cmd_storage_backups(_args(keep=0)) == 2
    assert "--keep" in capsys.readouterr().err
