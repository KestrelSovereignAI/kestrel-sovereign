"""Encrypted identity custody bundles for disposable container runtimes.

The bundle contains only public identity documents and Kestrel's existing
``*.enc`` private-key artifacts.  It never contains a database or plaintext
private key.  A compact JSON representation is suitable for one pinned Google
Secret Manager version; Cloud Run injects it into the bootstrap process, which
restores the files into the instance-local identity directory and then removes
the bundle from the uvicorn child environment.

The PostgreSQL database remains the authoritative durable state store.  This
module is deliberately only the cryptographic-custody half of that contract.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Optional

from kestrel_sovereign.identity.runtime_identity import load_agent_identity


SCHEMA_VERSION = 1
IDENTITY_BUNDLE_ENV = "KESTREL_IDENTITY_BUNDLE"
# Secret Manager accepts 64 KiB payloads, but Cloud Run environment values are
# limited to 32 KiB.  The bundle is injected only for bootstrap, so enforce the
# lower runtime limit before an operator can publish an undeployable version.
MAX_BUNDLE_BYTES = 32 * 1024
MAX_IDENTITY_FILES = 32
MAX_IDENTITY_FILE_BYTES = 48 * 1024


class IdentityCustodyError(RuntimeError):
    """Raised when a custody bundle is unsafe, corrupt, or mis-bound."""


_ROOT_FILE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"kestrel_0x[0-9A-Fa-f]+\.json",
        r"kestrel_0x[0-9A-Fa-f]+\.key\.enc",
        r"[A-Za-z0-9._-]+_did\.json",
        r"[A-Za-z0-9._-]+_ed25519\.key\.enc",
        r"[A-Za-z0-9._-]+_mldsa65\.bytes\.enc",
        r"[A-Za-z0-9._-]+_archival_slhdsa\.bytes\.enc",
        r"[A-Za-z0-9._-]+_archival_slhdsa_pub\.bytes\.enc",
        r"[A-Za-z0-9._-]+_x25519\.key\.enc",
        r"[A-Za-z0-9._-]+_mlkem768\.bytes\.enc",
        r"[A-Za-z0-9._-]+_mlkem768_pub\.bytes\.enc",
    )
)
_SUCCESSION_FILE = re.compile(r"[A-Za-z0-9._-]+\.json")


def _is_allowed_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        return False
    if len(path.parts) == 1:
        return any(pattern.fullmatch(path.name) for pattern in _ROOT_FILE_PATTERNS)
    return (
        len(path.parts) == 2
        and path.parts[0] == "successions"
        and _SUCCESSION_FILE.fullmatch(path.parts[1]) is not None
    )


def _identity_files(agent_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not agent_dir.is_dir():
        return files
    for path in sorted(agent_dir.rglob("*")):
        relative = path.relative_to(agent_dir).as_posix()
        if _is_allowed_relative_path(relative):
            if path.is_symlink():
                raise IdentityCustodyError(
                    f"identity artifact {relative!r} is a symlink; refusing custody export"
                )
            if path.is_file():
                files.append(path)
    return files


def _load_bound_identity(agent_dir: Path, expected_did: str):
    legacy_docs = sorted(agent_dir.glob("kestrel_0x*.json"))
    born_docs = sorted(agent_dir.glob("*_did.json"))
    if len(legacy_docs) > 1 or len(born_docs) > 1 or (legacy_docs and born_docs):
        raise IdentityCustodyError(
            "identity directory has ambiguous DID documents; refusing custody operation"
        )
    legacy_key_id = legacy_docs[0].stem if legacy_docs else None
    try:
        identity = load_agent_identity(legacy_key_id, storage_dir=agent_dir)
    except Exception as exc:
        raise IdentityCustodyError(
            "identity custody validation failed; restore the complete encrypted "
            f"identity set ({type(exc).__name__})"
        ) from exc
    if identity.signing_did != expected_did:
        raise IdentityCustodyError(
            "active signing identity does not bind to KESTREL_EXPECTED_DID"
        )
    return identity


def _encoded_file(path: Path, root: Path) -> dict[str, str]:
    relative = path.relative_to(root).as_posix()
    payload = path.read_bytes()
    if len(payload) > MAX_IDENTITY_FILE_BYTES:
        raise IdentityCustodyError(
            f"identity artifact {relative!r} exceeds the custody file limit"
        )
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "content_b64": base64.b64encode(payload).decode("ascii"),
    }


def create_identity_bundle(agent_dir: Path | str, *, expected_did: str) -> str:
    """Create a compact Secret Manager payload from encrypted identity files."""
    root = Path(agent_dir).expanduser().resolve()
    expected_did = str(expected_did or "").strip()
    if not expected_did.startswith("did:"):
        raise IdentityCustodyError("expected_did must be an explicit DID")
    plaintext_keys = [
        path
        for pattern in (
            "kestrel_0x*.pem",
            "*_ed25519.key",
            "*_mldsa65.bytes",
            "*_archival_slhdsa.bytes",
            "*_x25519.key",
            "*_mlkem768.bytes",
        )
        for path in root.glob(pattern)
    ]
    if plaintext_keys:
        raise IdentityCustodyError(
            "plaintext private-key material cannot enter a custody bundle; "
            "encrypt it with KESTREL_DATA_KEY first"
        )

    files = _identity_files(root)
    if not files:
        raise IdentityCustodyError("no encrypted identity artifacts found")
    if len(files) > MAX_IDENTITY_FILES:
        raise IdentityCustodyError("identity custody bundle has too many files")
    identity = _load_bound_identity(root, expected_did)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "expected_did": expected_did,
        "signing_did": identity.signing_did,
        "files": [_encoded_file(path, root) for path in files],
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_BUNDLE_BYTES:
        raise IdentityCustodyError(
            "identity custody bundle exceeds Cloud Run's 32 KiB environment limit"
        )
    return encoded


def _decode_manifest(bundle: str | bytes, expected_did: str) -> list[tuple[str, bytes]]:
    raw = bundle.encode("utf-8") if isinstance(bundle, str) else bytes(bundle)
    if not raw or len(raw) > MAX_BUNDLE_BYTES:
        raise IdentityCustodyError("identity custody bundle is empty or oversized")
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IdentityCustodyError("identity custody bundle is not valid JSON") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise IdentityCustodyError("unsupported identity custody bundle schema")
    if manifest.get("expected_did") != expected_did:
        raise IdentityCustodyError(
            "identity custody bundle does not match KESTREL_EXPECTED_DID"
        )
    if manifest.get("signing_did") != expected_did:
        raise IdentityCustodyError(
            "identity custody manifest does not bind the active signing DID"
        )
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not 1 <= len(raw_files) <= MAX_IDENTITY_FILES:
        raise IdentityCustodyError("identity custody bundle has an invalid file list")

    decoded: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for item in raw_files:
        if not isinstance(item, Mapping):
            raise IdentityCustodyError("identity custody file entry is not an object")
        relative = item.get("path")
        if not isinstance(relative, str) or not _is_allowed_relative_path(relative):
            raise IdentityCustodyError("identity custody bundle contains an unsafe path")
        if relative in seen:
            raise IdentityCustodyError("identity custody bundle contains duplicate paths")
        seen.add(relative)
        content = item.get("content_b64")
        digest = item.get("sha256")
        if not isinstance(content, str) or not isinstance(digest, str):
            raise IdentityCustodyError("identity custody file entry is incomplete")
        try:
            payload = base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError, TypeError) as exc:
            raise IdentityCustodyError("identity custody file is not valid base64") from exc
        if len(payload) > MAX_IDENTITY_FILE_BYTES:
            raise IdentityCustodyError("identity custody file is oversized")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise IdentityCustodyError("identity custody file digest mismatch")
        decoded.append((relative, payload))
    return decoded


def _existing_files_match(
    target: Path, decoded: Iterable[tuple[str, bytes]]
) -> bool:
    decoded_map = dict(decoded)
    existing = _identity_files(target)
    if {path.relative_to(target).as_posix() for path in existing} != set(decoded_map):
        return False
    return all(
        path.read_bytes() == decoded_map[path.relative_to(target).as_posix()]
        for path in existing
    )


def _unexpected_restore_entries(target: Path) -> list[str]:
    """Return local entries that are not part of a custody identity set."""
    unexpected: list[str] = []
    for path in target.rglob("*"):
        relative = path.relative_to(target).as_posix()
        if path.is_symlink():
            unexpected.append(relative)
        elif path.is_dir() and relative == "successions":
            continue
        elif path.is_file() and _is_allowed_relative_path(relative):
            continue
        else:
            unexpected.append(relative)
    return unexpected


def restore_identity_bundle(
    bundle: str | bytes,
    agent_dir: Path | str,
    *,
    expected_did: str,
) -> str:
    """Restore and cryptographically validate one pinned custody bundle.

    Existing identity material is never overwritten.  An exact already-present
    bundle is accepted idempotently; any drift fails startup.
    """
    expected_did = str(expected_did or "").strip()
    if not expected_did.startswith("did:"):
        raise IdentityCustodyError("KESTREL_EXPECTED_DID must be an explicit DID")
    decoded = _decode_manifest(bundle, expected_did)
    target = Path(agent_dir).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise IdentityCustodyError("identity restore target is not a safe directory")
    if target.exists():
        unexpected = _unexpected_restore_entries(target)
        if unexpected:
            raise IdentityCustodyError(
                "identity restore target contains unexpected local state"
            )
    existing = _identity_files(target) if target.exists() else []
    if existing:
        if not _existing_files_match(target, decoded):
            raise IdentityCustodyError(
                "existing identity material differs from the pinned custody bundle"
            )
        _load_bound_identity(target, expected_did)
        return expected_did

    # Durable Cloud Run has no legitimate local state before restore: its
    # coherent state is PostgreSQL.  Requiring an absent/empty target lets us
    # install the entire validated directory with one atomic rename instead
    # of exposing a partially-restored identity if startup races or crashes.
    if target.exists():
        # Any unexpected entry was rejected above, and an allowed identity
        # entry would have populated ``existing``. This is therefore empty.
        target.rmdir()

    stage = Path(tempfile.mkdtemp(prefix=".identity-restore-", dir=target.parent))
    try:
        os.chmod(stage, 0o700)
        for relative, payload in decoded:
            destination = stage / PurePosixPath(relative)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            destination.write_bytes(payload)
            os.chmod(destination, 0o600)
        _load_bound_identity(stage, expected_did)

        try:
            os.replace(stage, target)
        except OSError as exc:
            raise IdentityCustodyError(
                "identity restore target changed during atomic installation"
            ) from exc
        _load_bound_identity(target, expected_did)
        return expected_did
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def restore_identity_bundle_from_env(
    agent_dir: Path | str,
    *,
    expected_did: str,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """Restore the bundle supplied by Secret Manager as an environment value."""
    environ = os.environ if env is None else env
    bundle = environ.get(IDENTITY_BUNDLE_ENV)
    if not bundle:
        raise IdentityCustodyError(
            "durable identity custody is unavailable; refusing to re-incept"
        )
    return restore_identity_bundle(bundle, agent_dir, expected_did=expected_did)


def write_identity_bundle(path: Path | str, bundle: str) -> Path:
    """Create a mode-0600 bundle file without overwriting prior custody data."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(destination, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(bundle)
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage encrypted identity custody bundles")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create a Secret Manager payload")
    create.add_argument("--agent-dir", required=True)
    create.add_argument("--expected-did", required=True)
    create.add_argument("--output", required=True)

    restore = subparsers.add_parser("restore-env", help="restore the injected bundle")
    restore.add_argument("--agent-dir", required=True)
    restore.add_argument("--expected-did", required=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "create":
        bundle = create_identity_bundle(
            args.agent_dir,
            expected_did=args.expected_did,
        )
        destination = write_identity_bundle(args.output, bundle)
        print(f"Identity custody bundle written securely to {destination}")
        return 0
    restored = restore_identity_bundle_from_env(
        args.agent_dir,
        expected_did=args.expected_did,
    )
    print(f"Restored and verified identity custody for {restored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IDENTITY_BUNDLE_ENV",
    "IdentityCustodyError",
    "create_identity_bundle",
    "restore_identity_bundle",
    "restore_identity_bundle_from_env",
    "write_identity_bundle",
]
