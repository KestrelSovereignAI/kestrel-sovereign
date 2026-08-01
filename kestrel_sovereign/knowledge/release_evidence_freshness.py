"""Verifier-owned durable replay protection for external release evidence."""

from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

from .release_evidence_models import ExternalCapabilityReport, ReleaseEvidenceError


class ExternalFreshnessLedger:
    """Consume each valid external-CI freshness receipt exactly once.

    The ledger path is an explicit dependency of the independent verifier.
    Report authors and the public structural CLI never receive this object.
    SQLite's unique constraint and ``BEGIN IMMEDIATE`` transaction make the
    claim durable across verifier instances and cooperating processes.
    """

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ReleaseEvidenceError("external freshness ledger path must be absolute and verifier-owned")
        self._path = path
        self._prepare_path()

    @property
    def path(self) -> Path:
        """Return the verifier-selected ledger location without exposing entries."""
        return self._path

    def _prepare_path(self) -> tuple[int, int]:
        """Create and validate a private regular ledger file, returning its inode."""
        parent = self._path.parent
        parent_status = self._validate_directory_components(parent)
        if parent_status.st_uid != os.geteuid() or parent_status.st_mode & 0o077:
            raise ReleaseEvidenceError(
                "external freshness ledger parent must be verifier-owned with no group/other access"
            )

        try:
            path_status = self._path.lstat()
        except FileNotFoundError:
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(self._path, flags, 0o600)
            except OSError as exc:
                raise ReleaseEvidenceError("cannot create external freshness ledger") from exc
            try:
                path_status = os.fstat(descriptor)
                return self._validate_file_status(path_status)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise ReleaseEvidenceError("cannot inspect external freshness ledger") from exc
        return self._validate_file_status(path_status)

    @staticmethod
    def _validate_directory_components(parent: Path) -> os.stat_result:
        """Reject symlink/non-directory path components before opening SQLite."""
        components = tuple(reversed((parent, *parent.parents)))
        final_status: os.stat_result | None = None
        for component in components:
            try:
                status = component.lstat()
            except OSError as exc:
                raise ReleaseEvidenceError("external freshness ledger parent must already exist") from exc
            if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
                raise ReleaseEvidenceError("external freshness ledger parent has a symlink or non-directory component")
            if component == parent:
                final_status = status
        assert final_status is not None
        return final_status

    @staticmethod
    def _validate_file_status(path_status: os.stat_result) -> tuple[int, int]:
        if stat.S_ISLNK(path_status.st_mode) or not stat.S_ISREG(path_status.st_mode):
            raise ReleaseEvidenceError("external freshness ledger must be a regular non-symlink file")
        if path_status.st_uid != os.geteuid() or path_status.st_mode & 0o077:
            raise ReleaseEvidenceError("external freshness ledger must be verifier-owned with owner-only access")
        return (path_status.st_dev, path_status.st_ino)

    def consume(self, report: ExternalCapabilityReport) -> None:
        """Persist the receipt claim, rejecting an already consumed external run."""
        if not isinstance(report, ExternalCapabilityReport):
            raise ReleaseEvidenceError("external freshness ledger requires ExternalCapabilityReport")
        expected_identity = self._prepare_path()
        try:
            with sqlite3.connect(f"{self._path.as_uri()}?mode=rw", uri=True, isolation_level=None) as connection:
                if self._prepare_path() != expected_identity:
                    raise ReleaseEvidenceError("external freshness ledger changed while opening")
                connection.execute("PRAGMA journal_mode=DELETE")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS consumed_external_freshness (
                        freshness_receipt TEXT PRIMARY KEY,
                        attestation_digest TEXT NOT NULL
                    )
                    """
                )
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(
                        "INSERT INTO consumed_external_freshness "
                        "(freshness_receipt, attestation_digest) VALUES (?, ?)",
                        (report.freshness_receipt, report.attestation_digest),
                    )
                except sqlite3.IntegrityError as exc:
                    connection.execute("ROLLBACK")
                    raise ReleaseEvidenceError(
                        "external freshness receipt was already consumed by this verifier"
                    ) from exc
                if self._prepare_path() != expected_identity:
                    connection.execute("ROLLBACK")
                    raise ReleaseEvidenceError("external freshness ledger changed during receipt consumption")
                connection.execute("COMMIT")
        except ReleaseEvidenceError:
            raise
        except sqlite3.Error as exc:
            raise ReleaseEvidenceError("external freshness ledger cannot durably consume receipt") from exc
