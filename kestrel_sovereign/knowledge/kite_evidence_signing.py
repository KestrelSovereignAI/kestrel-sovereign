"""Private signing and replay state for an isolated Kite evidence run.

This module is deliberately usable only when the launch environment opts into
the Kite release-evidence harness.  Its Ed25519 private key and nonce ledger
live below an explicitly selected, private trusted root.  That root may sit
below a writable system-managed ancestor: its own inode, every descendant,
and every opened leaf are checked so a path replacement fails closed.  The
private key is never returned by an HTTP route; the loopback bootstrap
handshake receives only its public half.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
import stat
import weakref

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


_OPT_IN_ENV = "KESTREL_KITE_RELEASE_EVIDENCE"
_TRUSTED_ROOT_ENV = "KESTREL_KITE_RELEASE_EVIDENCE_ROOT"
_KEY_FILE = ".kite-evidence-ed25519.key"
_NONCE_LEDGER = ".kite-evidence-nonces.sqlite3"


class KiteEvidenceSigningError(RuntimeError):
    """The isolated server cannot safely sign a Kite evidence response."""


class KiteEvidenceNonceReplay(KiteEvidenceSigningError):
    """A request nonce has already been consumed by this isolated evidence home."""


class KiteEvidenceNonceReceipt:
    """Opaque proof that one exact Kite nonce committed to the local ledger.

    The typed evidence endpoint is the only code that receives this object.
    Callers cannot construct or copy it: its nonce remains in this module's
    private, one-shot registry until an operation-specific authority claims it.
    """

    __slots__ = ("__weakref__",)

    def __new__(cls, *args, **kwargs):
        raise TypeError("Kite evidence nonce receipts are issued after durable consumption")


_nonce_receipts: weakref.WeakKeyDictionary[KiteEvidenceNonceReceipt, str] = (
    weakref.WeakKeyDictionary()
)


def _issued_nonce_receipt(nonce: str) -> KiteEvidenceNonceReceipt:
    """Create the unforgeable receipt only after the ledger transaction commits."""
    receipt = object.__new__(KiteEvidenceNonceReceipt)
    _nonce_receipts[receipt] = nonce
    return receipt


def claim_kite_evidence_nonce_receipt(receipt: object) -> str:
    """Consume a committed nonce receipt for one typed internal authority.

    This deliberately returns the nonce only to another internal authority;
    it is not an HTTP or storage operation and cannot itself produce evidence.
    """
    if type(receipt) is not KiteEvidenceNonceReceipt:
        raise KiteEvidenceSigningError("Kite evidence nonce receipt is invalid")
    try:
        return _nonce_receipts.pop(receipt)
    except KeyError as error:
        raise KiteEvidenceSigningError("Kite evidence nonce receipt was already claimed") from error


@dataclass(frozen=True, slots=True)
class _TrustedHome:
    """A checked Kite home plus stable identities for its trusted path."""

    root: Path
    home: Path
    root_identity: tuple[int, int]
    home_identity: tuple[int, int]


def _enabled() -> bool:
    return os.environ.get(_OPT_IN_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _identity(status: os.stat_result) -> tuple[int, int]:
    return (status.st_dev, status.st_ino)


def _private_directory_status(path: Path, *, label: str) -> os.stat_result:
    try:
        status = path.lstat()
    except OSError as error:
        raise KiteEvidenceSigningError(f"{label} cannot be inspected") from error
    if (
        stat.S_ISLNK(status.st_mode)
        or not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or status.st_mode & 0o077
    ):
        raise KiteEvidenceSigningError(f"{label} must be a private verifier-owned directory")
    return status


def _private_file_status(path: Path, *, label: str) -> os.stat_result:
    try:
        status = path.lstat()
    except OSError as error:
        raise KiteEvidenceSigningError(f"{label} cannot be inspected") from error
    if (
        stat.S_ISLNK(status.st_mode)
        or not stat.S_ISREG(status.st_mode)
        or status.st_uid != os.geteuid()
        or status.st_mode & 0o077
    ):
        raise KiteEvidenceSigningError(f"{label} must be a private regular file")
    return status


def _reject_parent_traversal(path: Path, *, label: str) -> None:
    """Keep lexical ``..`` out before any normalization can hide it."""
    if ".." in path.parts:
        raise KiteEvidenceSigningError(f"{label} cannot contain parent traversal")


def _assert_original_directory_components(path: Path, *, label: str) -> None:
    """Reject a symlink in any caller-supplied path component.

    Ownership/mode requirements begin at the explicit trusted root.  Its
    system-managed ancestors may be writable, but they still must be the
    original directory chain, never a symlink that ``resolve()`` would erase.
    """
    _reject_parent_traversal(path, label=label)
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            status = current.lstat()
        except OSError as error:
            raise KiteEvidenceSigningError(f"{label} cannot be inspected") from error
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise KiteEvidenceSigningError(f"{label} has a symlink or non-directory component")


def _normalized_trusted_paths() -> tuple[Path, Path]:
    """Validate original absolute paths, then resolve their non-symlink form."""
    home_value = os.environ.get("KESTREL_HOME")
    root_value = os.environ.get(_TRUSTED_ROOT_ENV)
    if not home_value or not root_value:
        raise KiteEvidenceSigningError("Kite evidence trusted root and home are required")
    raw_home = Path(home_value).expanduser()
    raw_root = Path(root_value).expanduser()
    if not raw_home.is_absolute() or not raw_root.is_absolute():
        raise KiteEvidenceSigningError("Kite evidence trusted root and home must be absolute")
    _assert_original_directory_components(raw_root, label="Kite evidence trusted root")
    _assert_original_directory_components(raw_home, label="Kite evidence home")
    try:
        root = raw_root.resolve(strict=True)
        home = raw_home.resolve(strict=True)
    except OSError as error:
        raise KiteEvidenceSigningError("Kite evidence trusted root and home cannot be resolved") from error
    try:
        home.relative_to(root)
    except ValueError as error:
        raise KiteEvidenceSigningError("Kite evidence home must live below its trusted root") from error
    return root, home


def _trusted_home() -> _TrustedHome:
    """Validate the explicit root and every private component to KESTREL_HOME.

    The root is the trust anchor, not its parent.  This deliberately accepts a
    private isolated home below a writable temporary-directory ancestor while
    retaining inode checks that detect a root/home rename or replacement.
    """
    if not _enabled():
        raise KiteEvidenceSigningError("Kite evidence signing is not enabled")
    root, home = _normalized_trusted_paths()
    try:
        relative_home = home.relative_to(root)
    except ValueError as error:
        raise KiteEvidenceSigningError("Kite evidence home must live below its trusted root") from error
    root_status = _private_directory_status(root, label="Kite evidence trusted root")
    root_identity = _identity(root_status)
    current = root
    home_status = root_status
    for component in relative_home.parts:
        current = current / component
        home_status = _private_directory_status(current, label="Kite evidence home component")
    return _TrustedHome(
        root=root,
        home=home,
        root_identity=root_identity,
        home_identity=_identity(home_status),
    )


def _assert_trusted_home_unchanged(trusted_home: _TrustedHome) -> None:
    """Reject a root/home replacement that happened after the initial check."""
    root, home = _normalized_trusted_paths()
    if root != trusted_home.root or home != trusted_home.home:
        raise KiteEvidenceSigningError("Kite evidence trusted root or home changed while opening")
    root_status = _private_directory_status(
        trusted_home.root,
        label="Kite evidence trusted root",
    )
    if _identity(root_status) != trusted_home.root_identity:
        raise KiteEvidenceSigningError("Kite evidence trusted root changed while opening")
    try:
        relative_home = trusted_home.home.relative_to(trusted_home.root)
    except ValueError as error:  # defensive: _TrustedHome is private and immutable
        raise KiteEvidenceSigningError("Kite evidence home escaped its trusted root") from error
    current = trusted_home.root
    home_status = root_status
    for component in relative_home.parts:
        current = current / component
        home_status = _private_directory_status(current, label="Kite evidence home component")
    if _identity(home_status) != trusted_home.home_identity:
        raise KiteEvidenceSigningError("Kite evidence home changed while opening")


def _read_private_key(path: Path, trusted_home: _TrustedHome) -> bytes:
    before = _private_file_status(path, label="Kite evidence signing key")
    _assert_trusted_home_unchanged(trusted_home)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise KiteEvidenceSigningError("Kite evidence signing key cannot be opened") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_mode & 0o077
            or _identity(opened) != _identity(before)
        ):
            raise KiteEvidenceSigningError("Kite evidence signing key changed while opening")
        material = os.read(descriptor, 33)
        _assert_trusted_home_unchanged(trusted_home)
        if _identity(_private_file_status(path, label="Kite evidence signing key")) != _identity(opened):
            raise KiteEvidenceSigningError("Kite evidence signing key changed while opening")
        return material
    finally:
        os.close(descriptor)


def _create_private_key(path: Path, trusted_home: _TrustedHome) -> bytes:
    generated = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    _assert_trusted_home_unchanged(trusted_home)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        return _read_private_key(path, trusted_home)
    except OSError as error:
        raise KiteEvidenceSigningError("Kite evidence signing key cannot be created") from error
    try:
        os.fchmod(descriptor, 0o600)
        if os.write(descriptor, generated) != len(generated):
            raise KiteEvidenceSigningError("Kite evidence signing key could not be fully written")
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_mode & 0o077
        ):
            raise KiteEvidenceSigningError("Kite evidence signing key cannot be secured")
        _assert_trusted_home_unchanged(trusted_home)
        if _identity(_private_file_status(path, label="Kite evidence signing key")) != _identity(opened):
            raise KiteEvidenceSigningError("Kite evidence signing key changed while creating")
        return generated
    finally:
        os.close(descriptor)


def _private_key() -> Ed25519PrivateKey:
    trusted_home = _trusted_home()
    path = trusted_home.home / _KEY_FILE
    try:
        material = _read_private_key(path, trusted_home)
    except KiteEvidenceSigningError as signing_error:
        try:
            path.lstat()
        except FileNotFoundError:
            material = _create_private_key(path, trusted_home)
        except OSError as os_error:
            raise KiteEvidenceSigningError("Kite evidence signing key cannot be inspected") from os_error
        else:
            raise signing_error
    if len(material) != 32:
        raise KiteEvidenceSigningError("Kite evidence signing key has invalid material")
    try:
        return Ed25519PrivateKey.from_private_bytes(material)
    except ValueError as error:
        raise KiteEvidenceSigningError("Kite evidence signing key has invalid material") from error


def kite_evidence_public_key() -> str:
    """Return only the key pinned by the loopback launch handshake."""
    return _private_key().public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()


def sign_kite_evidence(payload: bytes) -> str:
    """Sign already-canonical, content-free response bytes with the private key."""
    if not isinstance(payload, bytes):
        raise KiteEvidenceSigningError("Kite evidence signing payload must be bytes")
    return _private_key().sign(payload).hex()


def _prepare_nonce_ledger(path: Path, trusted_home: _TrustedHome) -> tuple[int, int]:
    """Open/create only a private nonce ledger and return its checked inode."""
    _assert_trusted_home_unchanged(trusted_home)
    try:
        status = _private_file_status(path, label="Kite evidence nonce ledger")
    except KiteEvidenceSigningError as signing_error:
        try:
            path.lstat()
        except FileNotFoundError:
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(path, flags, 0o600)
            except FileExistsError:
                return _prepare_nonce_ledger(path, trusted_home)
            except OSError as error:
                raise KiteEvidenceSigningError("Kite evidence nonce ledger cannot be created") from error
            try:
                os.fchmod(descriptor, 0o600)
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_uid != os.geteuid()
                    or opened.st_mode & 0o077
                ):
                    raise KiteEvidenceSigningError("Kite evidence nonce ledger cannot be secured")
                status = opened
            finally:
                os.close(descriptor)
        except OSError as os_error:
            raise KiteEvidenceSigningError("Kite evidence nonce ledger cannot be inspected") from os_error
        else:
            raise signing_error
    _assert_trusted_home_unchanged(trusted_home)
    if _identity(_private_file_status(path, label="Kite evidence nonce ledger")) != _identity(status):
        raise KiteEvidenceSigningError("Kite evidence nonce ledger changed while opening")
    return _identity(status)


def _assert_nonce_ledger_unchanged(
    path: Path,
    trusted_home: _TrustedHome,
    expected_identity: tuple[int, int],
) -> None:
    _assert_trusted_home_unchanged(trusted_home)
    if _identity(_private_file_status(path, label="Kite evidence nonce ledger")) != expected_identity:
        raise KiteEvidenceSigningError("Kite evidence nonce ledger changed while opening")


def consume_kite_evidence_nonce(
    nonce: str, *, issue_receipt: bool = False,
) -> KiteEvidenceNonceReceipt | None:
    """Durably reserve ``nonce`` before operation execution, including restarts."""
    if not isinstance(nonce, str):
        raise KiteEvidenceSigningError("Kite evidence nonce must be text")
    trusted_home = _trusted_home()
    path = trusted_home.home / _NONCE_LEDGER
    expected_identity = _prepare_nonce_ledger(path, trusted_home)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=rw", uri=True, isolation_level=None)
        _assert_nonce_ledger_unchanged(path, trusted_home, expected_identity)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS consumed_kite_evidence_nonce (nonce TEXT PRIMARY KEY)"
        )
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO consumed_kite_evidence_nonce (nonce) VALUES (?)",
                (nonce,),
            )
        except sqlite3.IntegrityError as error:
            connection.execute("ROLLBACK")
            raise KiteEvidenceNonceReplay("Kite evidence nonce was already consumed") from error
        _assert_nonce_ledger_unchanged(path, trusted_home, expected_identity)
        connection.execute("COMMIT")
        _assert_nonce_ledger_unchanged(path, trusted_home, expected_identity)
    except KiteEvidenceSigningError:
        raise
    except sqlite3.Error as error:
        raise KiteEvidenceSigningError("Kite evidence nonce ledger is unavailable") from error
    finally:
        if connection is not None:
            connection.close()
    # The ordinary typed probes need only durable replay protection.  Retain a
    # live receipt solely for the core-erasure route, which immediately hands
    # it to the endpoint-scoped authority.  This keeps ignored/non-erasure
    # requests from accumulating in the private weak registry.
    return _issued_nonce_receipt(nonce) if issue_receipt else None
