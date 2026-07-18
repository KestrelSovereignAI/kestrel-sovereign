"""Bounded, no-follow intake for portable identity packages.

Identity import and verification accept either a local package path or an IPFS
CID.  Both public tools use this module so their I/O security contract cannot
drift.  Local reads are descriptor-based and run off the event loop; CID reads
ask the storage adapter to enforce the same decoded-output ceiling throughout
cache, network, and decompression processing.
"""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path
from typing import Optional

# Continuity packages can legitimately contain a substantial memory/personality
# snapshot.  64 MiB is intentionally much larger than normal exports while
# still placing a firm ceiling on untrusted local and remote inputs.
MAX_IDENTITY_PACKAGE_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024


class IdentityPackageIntakeError(ValueError):
    """An identity package source failed the bounded intake contract."""


def _is_cid(source: str) -> bool:
    return source.startswith("Qm") or source.startswith("bafy")


def _owned_by_operator(metadata: os.stat_result) -> bool:
    get_uid = getattr(os, "geteuid", None) or getattr(os, "getuid", None)
    return get_uid is None or metadata.st_uid == get_uid()


def _read_local_identity_package(path: Path) -> str:
    """Read one regular local package without following its final path entry."""

    try:
        before = os.lstat(path)
    except FileNotFoundError as exc:
        raise IdentityPackageIntakeError(
            "local identity package was not found"
        ) from exc
    except OSError as exc:
        raise IdentityPackageIntakeError(
            f"local identity package metadata is unavailable ({type(exc).__name__})"
        ) from exc

    if not stat.S_ISREG(before.st_mode):
        raise IdentityPackageIntakeError(
            "local identity package must be a regular file, not a link or special entry"
        )

    nofollow = getattr(os, "O_NOFOLLOW", None)
    nonblock = getattr(os, "O_NONBLOCK", None)
    if nofollow is None or nonblock is None:
        raise IdentityPackageIntakeError(
            "secure local identity-package reads are unsupported on this platform"
        )
    flags = os.O_RDONLY | nofollow | nonblock | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise IdentityPackageIntakeError("local identity package was not found") from exc
    except OSError as exc:
        raise IdentityPackageIntakeError(
            f"local identity package could not be opened securely ({type(exc).__name__})"
        ) from exc

    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise IdentityPackageIntakeError(
                "local identity package changed or is not a regular file"
            )
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise IdentityPackageIntakeError(
                "local identity package changed before it could be opened"
            )
        if not _owned_by_operator(opened):
            raise IdentityPackageIntakeError(
                "local identity package is not owned by the current operator"
            )
        if stat.S_IMODE(opened.st_mode) & 0o022:
            raise IdentityPackageIntakeError(
                "local identity package is writable by group or other users"
            )
        if opened.st_size > MAX_IDENTITY_PACKAGE_BYTES:
            raise IdentityPackageIntakeError(
                f"identity package exceeds the {MAX_IDENTITY_PACKAGE_BYTES}-byte limit"
            )

        chunks: list[bytes] = []
        total = 0
        while total <= MAX_IDENTITY_PACKAGE_BYTES:
            chunk = os.read(
                descriptor,
                min(
                    _READ_CHUNK_BYTES,
                    MAX_IDENTITY_PACKAGE_BYTES + 1 - total,
                ),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > MAX_IDENTITY_PACKAGE_BYTES:
            raise IdentityPackageIntakeError(
                f"identity package exceeds the {MAX_IDENTITY_PACKAGE_BYTES}-byte limit"
            )
        raw = b"".join(chunks)
    except IdentityPackageIntakeError:
        raise
    except OSError as exc:
        raise IdentityPackageIntakeError(
            f"local identity package could not be read securely ({type(exc).__name__})"
        ) from exc
    finally:
        os.close(descriptor)

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IdentityPackageIntakeError(
            "identity package is not valid UTF-8 JSON text"
        ) from exc


def _retrieve_cid_identity_package(source: str, key_hash: Optional[str]) -> str:
    from kestrel_sovereign.filecoin_adapter import (
        ContentRetrievalLimitError,
        FilecoinAdapter,
    )

    try:
        content = FilecoinAdapter().retrieve_content(
            source,
            ipfs_cid=source,
            key_hash=key_hash,
            max_output_bytes=MAX_IDENTITY_PACKAGE_BYTES,
        )
    except ContentRetrievalLimitError as exc:
        raise IdentityPackageIntakeError(
            f"identity package exceeds the {MAX_IDENTITY_PACKAGE_BYTES}-byte limit"
        ) from exc
    except Exception as exc:
        raise IdentityPackageIntakeError(
            f"identity CID retrieval failed ({type(exc).__name__})"
        ) from exc
    if not isinstance(content, bytes):
        raise IdentityPackageIntakeError("identity CID retrieval returned invalid data")
    if len(content) > MAX_IDENTITY_PACKAGE_BYTES:
        # Defense in depth for adapters/test doubles that ignore the bound.
        raise IdentityPackageIntakeError(
            f"identity package exceeds the {MAX_IDENTITY_PACKAGE_BYTES}-byte limit"
        )
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IdentityPackageIntakeError(
            "identity package is not valid UTF-8 JSON text"
        ) from exc


async def load_identity_package_source(
    source: str,
    *,
    key_hash: Optional[str] = None,
) -> str:
    """Return bounded UTF-8 package text without blocking the event loop."""

    if not isinstance(source, str) or not source:
        raise IdentityPackageIntakeError("identity package source must be a string")
    if _is_cid(source):
        return await asyncio.to_thread(_retrieve_cid_identity_package, source, key_hash)
    return await asyncio.to_thread(_read_local_identity_package, Path(source))


__all__ = [
    "IdentityPackageIntakeError",
    "MAX_IDENTITY_PACKAGE_BYTES",
    "load_identity_package_source",
]
