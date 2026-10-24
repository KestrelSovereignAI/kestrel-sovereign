"""Retention for the database copies a reanchor leaves beside an agent (#3785).

``kestrel constitution reanchor --force`` copies the agent database to
``kestrel_prime.db.backup-<YYYYmmdd-HHMMSS>`` before it writes, and nothing
ever removed one. On the production fleet they reached 18.6 GB in 24 files
beside databases of 1-2 GB, two of them written a second apart.

:func:`write_reanchor_backup` writes each copy under a staging name and
publishes it under its final name only once it is complete and on disk, so a
copy that fails part-way never looks like one of the copies the policy counts.

The policy keeps the newest :data:`DEFAULT_KEEP` distinct copies of each
database. Copies with identical bytes (same size, then same SHA-256) count as
one, and only the newest file of each content is kept. A copy counts toward
that number only once SQLite has read every page of it: a truncated or
damaged copy, such as a failed copy written before staging, is kept and
reported but never counted, so it cannot displace a usable older copy.
Nothing is deleted on its own: :func:`plan_backup_retention` only reports,
and :func:`apply_backup_retention` removes exactly what a plan names, when
the Sovereign runs ``kestrel storage backups --apply``.

Only reanchor copies are managed. Every other file beside the database is
left alone, in particular:

* a forced inception's archive, ``kestrel_prime.db.backup-<stamp>-<hex>``,
  which inception reads as evidence of an earlier constitutional lifetime;
* hand-made forensic copies such as ``.damaged-<date>`` or
  ``.before-<event>-<date>``, which the plan lists as preserved;
* a staging file a crashed reanchor left, ``<name>.partial-backup-<random>``,
  listed as preserved too.

When the live database is missing, its copies may be all that is left of
the agent, and nothing is planned for removal.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: Distinct copies kept per database.
DEFAULT_KEEP = 3

_STAMP_FORMAT = "%Y%m%d-%H%M%S"
# Never ``.backup-``: inception reads every ``kestrel_prime.db.backup-*``
# file as an archived database.
_STAGING_INFIX = ".partial-backup-"
_SQLITE_HEADER = b"SQLite format 3\x00"
_SQLITE_HEADER_BYTES = 100
_HASH_CHUNK_BYTES = 1 << 20


def reanchor_backup_path(db_path: Path) -> Path:
    """Where a reanchor copies *db_path* now: ``<name>.backup-<local time>``."""
    stamp = time.strftime(_STAMP_FORMAT)
    return db_path.with_name(f"{db_path.name}.backup-{stamp}")


def _fsync_directory(path: Path) -> None:
    """Make a new directory entry in *path* durable (POSIX only)."""
    if os.name == "nt":  # pragma: no cover - directory fsync is POSIX-specific
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_reanchor_backup(db_path: Path) -> Path:
    """Copy *db_path* to a new reanchor copy and return the copy's path.

    The bytes go to a staging file beside the database and are flushed to
    disk; only then is the file linked under its final name, which is never
    overwritten. A copy that fails part-way leaves nothing under that name.
    Like ``shutil.copy2``, the copy keeps the database's mode and mtime.

    Raises ``FileExistsError`` when a copy with this second's stamp exists.
    """
    db_path = Path(db_path)
    final = reanchor_backup_path(db_path)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f"{db_path.name}{_STAGING_INFIX}", dir=db_path.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            with open(db_path, "rb") as source:
                shutil.copyfileobj(source, target, _HASH_CHUNK_BYTES)
            target.flush()
            os.fsync(target.fileno())
        shutil.copystat(db_path, staging)
        try:
            os.link(staging, final, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(
                errno.EEXIST,
                "a reanchor copy with this stamp already exists and is never "
                "overwritten; retry in a second",
                str(final),
            ) from None
        _fsync_directory(final.parent)
    finally:
        staging.unlink(missing_ok=True)
    return final


def _reanchor_backup_stamp(db_path: Path, candidate: str) -> Optional[str]:
    """The stamp of a reanchor copy of *db_path* named *candidate*, else None."""
    match = re.fullmatch(
        re.escape(db_path.name) + r"\.backup-(\d{8}-\d{6})", candidate
    )
    return match.group(1) if match else None


@dataclass(frozen=True)
class BackupCopy:
    """One reanchor copy, as it was when the plan looked at it."""

    path: Path
    stamp: str
    size: int
    mtime_ns: int
    sha256: Optional[str] = None


@dataclass(frozen=True)
class BackupDecision:
    copy: BackupCopy
    keep: bool
    reason: str
    duplicate_of: Optional[Path] = None
    #: True for a kept copy that counts toward the newest ``keep``. Only these
    #: copies justify removing an older one.
    counted: bool = False


@dataclass(frozen=True)
class BackupRetentionPlan:
    """What the policy would keep and remove beside one database."""

    database: Path
    database_bytes: Optional[int]
    keep: int
    decisions: Tuple[BackupDecision, ...] = field(default_factory=tuple)
    preserved: Tuple[Path, ...] = field(default_factory=tuple)

    @property
    def removals(self) -> Tuple[BackupDecision, ...]:
        return tuple(decision for decision in self.decisions if not decision.keep)

    @property
    def reclaimable_bytes(self) -> int:
        return sum(decision.copy.size for decision in self.removals)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "database": str(self.database),
            "database_bytes": self.database_bytes,
            "keep": self.keep,
            "copies": [
                {
                    "path": str(decision.copy.path),
                    "bytes": decision.copy.size,
                    "sha256": decision.copy.sha256,
                    "action": "keep" if decision.keep else "remove",
                    "counted": decision.counted,
                    "reason": decision.reason,
                }
                for decision in self.decisions
            ],
            "preserved": [str(path) for path in self.preserved],
            "reclaimable_bytes": self.reclaimable_bytes,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sqlite_file(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(len(_SQLITE_HEADER)) == _SQLITE_HEADER
    except OSError:
        return False


def _backup_defect(path: Path) -> Optional[str]:
    """Why *path* is not a complete SQLite database, or None when it is.

    The header must name a whole number of pages the file holds; SQLite then
    reads every page (``PRAGMA quick_check``) through a read-only, immutable
    connection that takes no lock and creates no sidecar file. The header
    check matters on its own: SQLite reads an empty file as an empty database.
    """
    try:
        with open(path, "rb") as handle:
            header = handle.read(_SQLITE_HEADER_BYTES)
            size = os.fstat(handle.fileno()).st_size
    except OSError as exc:
        return f"cannot be read: {exc.strerror or exc}"
    if len(header) < _SQLITE_HEADER_BYTES or not header.startswith(_SQLITE_HEADER):
        return "not a SQLite database"
    page_size = int.from_bytes(header[16:18], "big")
    if page_size == 1:
        page_size = 65536
    if page_size < 512 or page_size & (page_size - 1):
        return f"the header names an invalid page size ({page_size})"
    if size % page_size:
        return f"incomplete: {size} bytes is not a whole number of {page_size}-byte pages"
    pages = int.from_bytes(header[28:32], "big")
    # The in-header page count is valid only while the change counter matches
    # the version-valid-for number (SQLite file format, section 1.3.8).
    if pages and header[24:28] == header[92:96] and pages * page_size > size:
        return f"incomplete: the header names {pages * page_size} bytes, the file has {size}"
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            rows = connection.execute("PRAGMA quick_check").fetchall()
    except sqlite3.Error as exc:
        return f"SQLite cannot read it: {exc}"
    if rows != [("ok",)]:
        first = str(rows[0][0]) if rows else "no result"
        return f"SQLite reports damage: {' '.join(first.split())}"
    return None


def plan_backup_retention(db_path: Path, *, keep: int = DEFAULT_KEEP) -> BackupRetentionPlan:
    """Decide which reanchor copies of *db_path* to keep; remove nothing.

    Copies sharing a size are hashed; identical ones collapse to their newest
    file. The newest *keep* distinct contents SQLite can read in full are
    kept and counted, the rest marked for removal. A content SQLite cannot
    read is kept without being counted. Copies are read only when there are
    more distinct contents than *keep*, since only then is anything removed.
    Database copies the policy does not manage are listed as ``preserved``.
    """
    if keep < 1:
        raise ValueError("keep must be at least 1")
    db_path = Path(db_path)
    try:
        live = db_path.lstat()
        database_bytes = live.st_size if stat.S_ISREG(live.st_mode) else None
    except FileNotFoundError:
        database_bytes = None

    copies: List[BackupCopy] = []
    preserved: List[Path] = []
    if db_path.parent.is_dir():
        for entry in sorted(db_path.parent.iterdir()):
            if not entry.name.startswith(f"{db_path.name}."):
                continue
            info = entry.lstat()
            if not stat.S_ISREG(info.st_mode):
                continue
            stamp = _reanchor_backup_stamp(db_path, entry.name)
            if stamp is None or info.st_nlink != 1:
                if _is_sqlite_file(entry):
                    preserved.append(entry)
                continue
            copies.append(BackupCopy(entry, stamp, info.st_size, info.st_mtime_ns))

    by_size: Dict[int, List[BackupCopy]] = {}
    for copy in copies:
        by_size.setdefault(copy.size, []).append(copy)
    hashed: List[BackupCopy] = []
    for group in by_size.values():
        if len(group) == 1:
            hashed.extend(group)
        else:
            hashed.extend(
                BackupCopy(c.path, c.stamp, c.size, c.mtime_ns, _sha256(c.path))
                for c in group
            )

    newest_first = sorted(
        hashed, key=lambda c: (c.stamp, c.mtime_ns, c.path.name), reverse=True
    )
    if database_bytes is None:
        decisions = [
            BackupDecision(copy, True, "the live database is missing; every copy is kept")
            for copy in newest_first
        ]
    else:
        # One entry per distinct content: its newest file, then its older
        # byte-identical files.
        contents: List[Tuple[BackupCopy, List[BackupCopy]]] = []
        by_content: Dict[Tuple[int, str], List[BackupCopy]] = {}
        for copy in newest_first:
            content = (copy.size, copy.sha256) if copy.sha256 is not None else None
            if content is not None and content in by_content:
                by_content[content].append(copy)
                continue
            duplicates: List[BackupCopy] = []
            contents.append((copy, duplicates))
            if content is not None:
                by_content[content] = duplicates

        verify = len(contents) > keep
        decisions = []
        counted = 0
        for newest, duplicates in contents:
            if counted >= keep:
                decisions.extend(
                    BackupDecision(copy, False, f"older than the newest {keep}")
                    for copy in (newest, *duplicates)
                )
                continue
            defect = _backup_defect(newest.path) if verify else None
            if defect is None:
                counted += 1
                decisions.append(BackupDecision(
                    newest, True, f"one of the newest {keep}", counted=True,
                ))
            else:
                decisions.append(BackupDecision(
                    newest, True, f"not counted toward the newest {keep}: {defect}",
                ))
            decisions.extend(
                BackupDecision(
                    copy, False, f"identical to {newest.path.name}",
                    duplicate_of=newest.path,
                )
                for copy in duplicates
            )
        order = {copy.path: index for index, copy in enumerate(newest_first)}
        decisions.sort(key=lambda decision: order[decision.copy.path])

    return BackupRetentionPlan(
        database=db_path,
        database_bytes=database_bytes,
        keep=keep,
        decisions=tuple(decisions),
        preserved=tuple(preserved),
    )


@dataclass(frozen=True)
class BackupRemoval:
    path: Path
    removed: bool
    reason: Optional[str] = None


def _unchanged(copy: BackupCopy) -> bool:
    try:
        info = copy.path.lstat()
    except FileNotFoundError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and info.st_size == copy.size
        and info.st_mtime_ns == copy.mtime_ns
    )


def apply_backup_retention(plan: BackupRetentionPlan) -> List[BackupRemoval]:
    """Remove the copies *plan* marks for removal, and nothing else.

    A copy that changed since it was planned is skipped. A duplicate is
    removed only while the copy it duplicates is still there unchanged, and
    an older copy only while every counted copy is.
    """
    kept = {decision.copy.path: decision.copy for decision in plan.decisions if decision.keep}
    counted = [decision.copy for decision in plan.decisions if decision.counted]
    results: List[BackupRemoval] = []
    for decision in plan.removals:
        copy = decision.copy
        if not _unchanged(copy):
            results.append(BackupRemoval(copy.path, False, "changed since it was planned"))
            continue
        if decision.duplicate_of is not None:
            original = kept.get(decision.duplicate_of)
            if original is None or not _unchanged(original):
                results.append(BackupRemoval(
                    copy.path, False,
                    f"{decision.duplicate_of.name} changed since it was planned",
                ))
                continue
        else:
            changed = next((c for c in counted if not _unchanged(c)), None)
            if changed is not None:
                results.append(BackupRemoval(
                    copy.path, False,
                    f"{changed.path.name} changed since it was planned",
                ))
                continue
        try:
            os.unlink(copy.path)
        except OSError as exc:
            results.append(BackupRemoval(copy.path, False, str(exc)))
            continue
        results.append(BackupRemoval(copy.path, True))
    return results
