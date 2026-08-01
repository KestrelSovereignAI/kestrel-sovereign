"""Verifier-owned durable replay protection for external release evidence."""

from __future__ import annotations

import os
import secrets
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

    def __init__(self, path: Path, *, trusted_root: Path) -> None:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ReleaseEvidenceError("external freshness ledger path must be absolute")
        if not isinstance(trusted_root, Path) or not trusted_root.is_absolute():
            raise ReleaseEvidenceError("external freshness ledger trusted_root must be absolute")
        try:
            relative_path = path.relative_to(trusted_root)
        except ValueError as exc:
            raise ReleaseEvidenceError("external freshness ledger must live below its trusted_root") from exc
        if relative_path == Path("."):
            raise ReleaseEvidenceError("external freshness ledger path must be a file below its trusted_root")
        self._path = path
        self._trusted_root = trusted_root
        self._prepare_path()

    @property
    def path(self) -> Path:
        """Return the verifier-selected ledger location without exposing entries."""
        return self._path

    @property
    def trusted_root(self) -> Path:
        """Return the explicit private root selected by the independent verifier."""
        return self._trusted_root

    def _prepare_path(self) -> tuple[int, int]:
        """Create and validate a private regular ledger file, returning its inode."""
        self._validate_directory_components(self._path.parent)

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

    def _validate_directory_components(self, parent: Path) -> None:
        """Require a private verifier-owned root and private descendants only.

        Components above ``trusted_root`` are deliberately not trusted: a
        secure owner-only root may be placed beneath a system-managed (even
        writable) ancestor.  The root itself must be a resolved real directory
        so that a shared ancestor cannot be substituted after this check.
        """
        try:
            resolved_root = self._trusted_root.resolve(strict=True)
        except OSError as exc:
            raise ReleaseEvidenceError("external freshness ledger trusted_root must already exist") from exc
        if self._trusted_root != resolved_root:
            raise ReleaseEvidenceError("external freshness ledger trusted_root must be resolved, not a symlink path")
        try:
            relative_parent = parent.relative_to(self._trusted_root)
        except ValueError as exc:
            raise ReleaseEvidenceError("external freshness ledger parent escapes its trusted_root") from exc
        components = [self._trusted_root]
        current = self._trusted_root
        for part in relative_parent.parts:
            current = current / part
            components.append(current)
        for component in components:
            try:
                status = component.lstat()
            except OSError as exc:
                raise ReleaseEvidenceError("external freshness ledger trusted_root and parent must already exist") from exc
            if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
                raise ReleaseEvidenceError("external freshness ledger trusted_root has a symlink or non-directory component")
            if status.st_uid != os.geteuid() or status.st_mode & 0o077:
                raise ReleaseEvidenceError(
                    "external freshness ledger trusted_root and descendants must be verifier-owned with no group/other access"
                )

    @staticmethod
    def _validate_file_status(path_status: os.stat_result) -> tuple[int, int]:
        if stat.S_ISLNK(path_status.st_mode) or not stat.S_ISREG(path_status.st_mode):
            raise ReleaseEvidenceError("external freshness ledger must be a regular non-symlink file")
        if path_status.st_uid != os.geteuid() or path_status.st_mode & 0o077:
            raise ReleaseEvidenceError("external freshness ledger must be verifier-owned with owner-only access")
        return (path_status.st_dev, path_status.st_ino)

    def _initialize_connection(
        self,
        connection: sqlite3.Connection,
        expected_identity: tuple[int, int],
    ) -> None:
        if self._prepare_path() != expected_identity:
            raise ReleaseEvidenceError("external freshness ledger changed while opening")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS external_freshness_challenges (
                run_nonce TEXT PRIMARY KEY,
                state TEXT NOT NULL CHECK(state IN ('pending', 'consumed'))
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS consumed_external_freshness (
                freshness_receipt TEXT PRIMARY KEY,
                attestation_digest TEXT NOT NULL
            )
            """
        )

    def issue_challenge(self) -> str:
        """Persist and return the verifier's one-time nonce for an external run."""
        run_nonce = secrets.token_hex(32)
        expected_identity = self._prepare_path()
        try:
            with sqlite3.connect(f"{self._path.as_uri()}?mode=rw", uri=True, isolation_level=None) as connection:
                self._initialize_connection(connection, expected_identity)
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO external_freshness_challenges (run_nonce, state) VALUES (?, 'pending')",
                    (run_nonce,),
                )
                if self._prepare_path() != expected_identity:
                    connection.execute("ROLLBACK")
                    raise ReleaseEvidenceError("external freshness ledger changed while issuing challenge")
                connection.execute("COMMIT")
                return run_nonce
        except ReleaseEvidenceError:
            raise
        except sqlite3.Error as exc:
            raise ReleaseEvidenceError("external freshness ledger cannot issue challenge") from exc

    def consume(self, report: ExternalCapabilityReport) -> None:
        """Persist the receipt claim, rejecting an already consumed external run."""
        if not isinstance(report, ExternalCapabilityReport):
            raise ReleaseEvidenceError("external freshness ledger requires ExternalCapabilityReport")
        expected_identity = self._prepare_path()
        try:
            with sqlite3.connect(f"{self._path.as_uri()}?mode=rw", uri=True, isolation_level=None) as connection:
                self._initialize_connection(connection, expected_identity)
                connection.execute("BEGIN IMMEDIATE")
                try:
                    challenge = connection.execute(
                        "UPDATE external_freshness_challenges SET state = 'consumed' "
                        "WHERE run_nonce = ? AND state = 'pending'",
                        (report.run_nonce,),
                    )
                    if challenge.rowcount != 1:
                        connection.execute("ROLLBACK")
                        existing = connection.execute(
                            "SELECT state FROM external_freshness_challenges WHERE run_nonce = ?",
                            (report.run_nonce,),
                        ).fetchone()
                        if existing == ("consumed",):
                            raise ReleaseEvidenceError(
                                "external freshness nonce was already consumed by this verifier"
                            )
                        raise ReleaseEvidenceError(
                            "external freshness nonce was not an issued pending verifier challenge"
                        )
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
