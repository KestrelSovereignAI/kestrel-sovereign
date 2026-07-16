"""Dirfd- and inode-backed private workspaces for Local MPS generation."""

from __future__ import annotations

import asyncio
import os
import secrets
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from ._owned_async_task import (
    await_owned_task,
    raise_owned_outcome,
    run_blocking_operation,
)


GENERATION_WORKSPACE_PREFIX = ".generation-"
_ARTIFACT_IO_CHUNK_BYTES = 1024 * 1024
_WORKSPACE_CREATION_ATTEMPTS = 128

_FileIdentity = tuple[int, int]


class GenerationWorkspaceIdentityError(RuntimeError):
    """A generation workspace no longer has its creation-time identity."""


@dataclass(slots=True, eq=False)
class GenerationWorkspaceLease:
    """Creation-time ownership of one private generation workspace."""

    root_path: Path
    workspace_name: str
    root_fd: int
    workspace_fd: int
    root_identity: _FileIdentity
    workspace_identity: _FileIdentity
    _artifact_fds: set[int] = field(default_factory=set, repr=False)
    _closed: bool = field(default=False, repr=False)
    _preserve_reason: BaseException | None = field(default=None, repr=False)

    @property
    def path(self) -> Path:
        """Return the original lexical workspace path for diagnostics only."""
        return self.root_path / self.workspace_name

    def preserve(self, reason: BaseException) -> None:
        """Record why this lease must never remove the workspace."""
        if self._preserve_reason is None:
            self._preserve_reason = reason


@dataclass(frozen=True, slots=True)
class GenerationArtifactLease:
    """An open private artifact inode owned by a workspace lease."""

    workspace: GenerationWorkspaceLease
    name: str
    fd: int
    identity: _FileIdentity

    @property
    def path(self) -> Path:
        """Return the original lexical path for diagnostics only."""
        return self.workspace.path / self.name


def _identity(stat_result: os.stat_result) -> _FileIdentity:
    return stat_result.st_dev, stat_result.st_ino


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    return flags


def _artifact_open_flags() -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    return flags


def _working_root(working_dir: Path) -> Path:
    """Resolve the operator-configured root, which must already exist."""
    return working_dir.resolve(strict=True)


def _assert_open_lease(lease: GenerationWorkspaceLease) -> None:
    if lease._closed:
        raise RuntimeError("Generation workspace lease is already closed")


def _validate_generation_workspace(lease: GenerationWorkspaceLease) -> None:
    """Prove the retained fds and their current path entries still agree."""
    _assert_open_lease(lease)
    if lease._preserve_reason is not None:
        raise RuntimeError("Generation workspace is marked for preservation") from (
            lease._preserve_reason
        )

    if _identity(os.fstat(lease.root_fd)) != lease.root_identity:
        raise GenerationWorkspaceIdentityError(
            "Generation working-root descriptor changed identity"
        )
    if _identity(os.fstat(lease.workspace_fd)) != lease.workspace_identity:
        raise GenerationWorkspaceIdentityError(
            "Generation workspace descriptor changed identity"
        )

    try:
        current_root = os.stat(lease.root_path, follow_symlinks=False)
    except FileNotFoundError as error:
        raise GenerationWorkspaceIdentityError(
            "Generation working root disappeared; workspace preserved"
        ) from error
    if not stat.S_ISDIR(current_root.st_mode) or _identity(current_root) != (
        lease.root_identity
    ):
        raise GenerationWorkspaceIdentityError(
            "Generation working root was replaced; workspace preserved"
        )

    try:
        current_workspace = os.stat(
            lease.workspace_name,
            dir_fd=lease.root_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as error:
        raise GenerationWorkspaceIdentityError(
            "Generation workspace path disappeared; workspace preserved"
        ) from error
    if not stat.S_ISDIR(current_workspace.st_mode) or _identity(current_workspace) != (
        lease.workspace_identity
    ):
        raise GenerationWorkspaceIdentityError(
            "Generation workspace path was replaced; workspace preserved"
        )


def _close_generation_workspace(lease: GenerationWorkspaceLease) -> None:
    """Close retained descriptors without removing any pathname."""
    if lease._closed:
        return
    lease._closed = True
    for artifact_fd in tuple(lease._artifact_fds):
        try:
            os.close(artifact_fd)
        except OSError:
            pass
        lease._artifact_fds.discard(artifact_fd)
    for directory_fd in (lease.workspace_fd, lease.root_fd):
        try:
            os.close(directory_fd)
        except OSError:
            pass


def _create_generation_workspace(working_dir: Path) -> GenerationWorkspaceLease:
    """Create a private workspace and retain its root and leaf identities."""
    if os.name != "posix":
        raise RuntimeError("Local MPS generation requires POSIX dirfd semantics")

    working_root = _working_root(working_dir)
    root_fd = os.open(working_root, _directory_open_flags())
    workspace_fd: int | None = None
    workspace_name: str | None = None
    created_workspace_identity: _FileIdentity | None = None
    try:
        root_identity = _identity(os.fstat(root_fd))
        current_root = os.stat(working_root, follow_symlinks=False)
        if not stat.S_ISDIR(current_root.st_mode) or _identity(current_root) != (
            root_identity
        ):
            raise GenerationWorkspaceIdentityError(
                "Generation working root changed during workspace creation"
            )

        for _attempt in range(_WORKSPACE_CREATION_ATTEMPTS):
            candidate = f"{GENERATION_WORKSPACE_PREFIX}{secrets.token_hex(16)}"
            try:
                os.mkdir(candidate, mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                continue
            workspace_name = candidate
            created_stat = os.stat(
                workspace_name,
                dir_fd=root_fd,
                follow_symlinks=False,
            )
            created_workspace_identity = _identity(created_stat)
            break
        if workspace_name is None:
            raise RuntimeError("Could not allocate a unique generation workspace")

        workspace_fd = os.open(
            workspace_name,
            _directory_open_flags(),
            dir_fd=root_fd,
        )
        workspace_stat = os.fstat(workspace_fd)
        workspace_identity = _identity(workspace_stat)
        if not stat.S_ISDIR(workspace_stat.st_mode):
            raise RuntimeError("Generation workspace is not a directory")
        if workspace_identity != created_workspace_identity:
            raise GenerationWorkspaceIdentityError(
                "Generation workspace changed identity while it was opened"
            )

        lease = GenerationWorkspaceLease(
            root_path=working_root,
            workspace_name=workspace_name,
            root_fd=root_fd,
            workspace_fd=workspace_fd,
            root_identity=root_identity,
            workspace_identity=workspace_identity,
        )
        _validate_generation_workspace(lease)
        return lease
    except BaseException:
        if workspace_fd is not None:
            try:
                os.close(workspace_fd)
            except OSError:
                pass
        if workspace_name is not None and created_workspace_identity is not None:
            try:
                current_stat = os.stat(
                    workspace_name,
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
                if _identity(current_stat) == created_workspace_identity:
                    os.rmdir(workspace_name, dir_fd=root_fd)
            except OSError:
                pass
        os.close(root_fd)
        raise


def _cleanup_generation_workspace(lease: GenerationWorkspaceLease) -> None:
    """Remove only the workspace whose root and leaf identities still match."""
    try:
        _validate_generation_workspace(lease)
        for artifact_fd in tuple(lease._artifact_fds):
            os.close(artifact_fd)
            lease._artifact_fds.discard(artifact_fd)

        # Clear the retained workspace inode, not whatever a mutable pathname
        # might select after validation.  Each descendant directory is opened
        # no-follow and identity-checked before recursion.
        _clear_directory_contents(lease.workspace_fd)

        # The leaf must still name the creation-time inode after the clear.
        # A replacement is preserved rather than recursively removed.
        _validate_generation_workspace(lease)
        os.rmdir(lease.workspace_name, dir_fd=lease.root_fd)
    finally:
        _close_generation_workspace(lease)


def _clear_directory_contents(directory_fd: int) -> None:
    """Delete entries through a retained directory fd without following links."""
    with os.scandir(directory_fd) as entries:
        entry_names = [entry.name for entry in entries]

    for entry_name in entry_names:
        entry_stat = os.stat(entry_name, dir_fd=directory_fd, follow_symlinks=False)
        entry_identity = _identity(entry_stat)
        if stat.S_ISDIR(entry_stat.st_mode):
            child_fd = os.open(
                entry_name,
                _directory_open_flags(),
                dir_fd=directory_fd,
            )
            try:
                if _identity(os.fstat(child_fd)) != entry_identity:
                    raise GenerationWorkspaceIdentityError(
                        "Generation cleanup descendant changed identity"
                    )
                _clear_directory_contents(child_fd)
                current_stat = os.stat(
                    entry_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if _identity(current_stat) != entry_identity:
                    raise GenerationWorkspaceIdentityError(
                        "Generation cleanup descendant was replaced"
                    )
                os.rmdir(entry_name, dir_fd=directory_fd)
            finally:
                os.close(child_fd)
            continue

        current_stat = os.stat(
            entry_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if _identity(current_stat) != entry_identity:
            raise GenerationWorkspaceIdentityError(
                "Generation cleanup artifact was replaced"
            )
        os.unlink(entry_name, dir_fd=directory_fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        written = os.pwrite(
            fd, view[offset : offset + _ARTIFACT_IO_CHUNK_BYTES], offset
        )
        if written <= 0:
            raise OSError("Short write while creating generation artifact")
        offset += written


def _create_generation_artifact(
    lease: GenerationWorkspaceLease,
    name: str,
    data: bytes | None,
) -> GenerationArtifactLease:
    """Create one no-follow artifact relative to the retained workspace fd."""
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Generation artifact name must be one plain path component")
    _validate_generation_workspace(lease)

    artifact_fd = os.open(
        name,
        _artifact_open_flags(),
        0o600,
        dir_fd=lease.workspace_fd,
    )
    try:
        artifact_stat = os.fstat(artifact_fd)
        if not stat.S_ISREG(artifact_stat.st_mode):
            raise RuntimeError("Generation artifact is not a regular file")
        if data is not None:
            _write_all(artifact_fd, data)
            os.fsync(artifact_fd)
        artifact = GenerationArtifactLease(
            workspace=lease,
            name=name,
            fd=artifact_fd,
            identity=_identity(artifact_stat),
        )
        lease._artifact_fds.add(artifact_fd)
        return artifact
    except BaseException:
        os.close(artifact_fd)
        raise


def _read_generation_artifact(artifact: GenerationArtifactLease) -> bytes:
    """Read the retained artifact inode without reopening its mutable name."""
    lease = artifact.workspace
    _validate_generation_workspace(lease)
    artifact_stat = os.fstat(artifact.fd)
    if not stat.S_ISREG(artifact_stat.st_mode) or _identity(artifact_stat) != (
        artifact.identity
    ):
        raise RuntimeError("Generation artifact descriptor changed identity")

    chunks: list[bytes] = []
    offset = 0
    while offset < artifact_stat.st_size:
        chunk = os.pread(
            artifact.fd,
            min(_ARTIFACT_IO_CHUNK_BYTES, artifact_stat.st_size - offset),
            offset,
        )
        if not chunk:
            raise OSError("Short read while loading generation artifact")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


async def create_generation_workspace(
    working_dir: Path,
) -> GenerationWorkspaceLease:
    """Create a workspace without leaking it when the caller is cancelled."""
    creation_task = asyncio.create_task(
        asyncio.to_thread(_create_generation_workspace, working_dir)
    )
    creation = await await_owned_task(creation_task)
    if creation.cancellation is None or creation.error is not None:
        return raise_owned_outcome(creation, operation="Workspace creation")

    lease = cast(GenerationWorkspaceLease, creation.result)
    cleanup_task = asyncio.create_task(
        asyncio.to_thread(_cleanup_generation_workspace, lease)
    )
    cleanup = await await_owned_task(cleanup_task, creation.cancellation)
    return raise_owned_outcome(cleanup, operation="Cancelled workspace cleanup")


async def create_generation_artifact(
    lease: GenerationWorkspaceLease,
    name: str,
    data: bytes | None = None,
) -> GenerationArtifactLease:
    """Create a private no-follow artifact without cancellation races."""
    return await run_blocking_operation(
        _create_generation_artifact,
        lease,
        name,
        data,
    )


async def read_generation_artifact(artifact: GenerationArtifactLease) -> bytes:
    """Read a private artifact by retained inode rather than pathname."""
    return await run_blocking_operation(_read_generation_artifact, artifact)


async def validate_generation_workspace(lease: GenerationWorkspaceLease) -> None:
    """Offload creation-time root/workspace identity validation."""
    await run_blocking_operation(_validate_generation_workspace, lease)
