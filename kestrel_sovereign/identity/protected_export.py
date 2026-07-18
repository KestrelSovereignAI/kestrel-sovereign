"""Private, atomic publication for plaintext identity export packages.

Identity exports contain an agent's portable continuity state.  They therefore
need stronger filesystem semantics than a normal JSON configuration file:

* every path component is opened without following links;
* the operator-owned export directory is mode ``0700``;
* payloads are written and fsynced through a new mode-``0600`` inode;
* new exports are published atomically without replacing an existing name; and
* the exceptional replacement path is explicit and limited to an existing,
  operator-owned ``identity_*.json`` below a configured data root.

The legacy audit and hardening helpers below inspect directory entries and
metadata only.  They never open or parse package contents.
"""

from __future__ import annotations

import fnmatch
import os
import secrets
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

IDENTITY_EXPORT_PATTERN = "identity_*.json"
IDENTITY_EXPORT_DIR_ENV = "KESTREL_IDENTITY_EXPORT_DIR"
_TEMP_PREFIX = ".identity-export-"
_TEMP_SUFFIX = ".tmp"


class IdentityExportSecurityError(OSError):
    """Raised when an export path cannot satisfy the custody contract."""


@dataclass(frozen=True, slots=True)
class LegacyIdentityExportFinding:
    """One metadata-only finding for a legacy local export."""

    root: Path
    entry_name: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class LegacyIdentityExportHardeningResult:
    """Counts from a metadata-only legacy permission hardening pass."""

    hardened: int = 0
    already_private: int = 0
    refused: int = 0
    missing_roots: int = 0


def identity_export_directory(
    *,
    agent_data_dir: Path | str | None = None,
    per_agent_override: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Return the absolute, agent-bound local identity export directory.

    ``per_agent_override`` is the strongest signal because an in-process host
    cannot represent distinct agent settings with one process environment.
    ``KESTREL_IDENTITY_EXPORT_DIR`` carries a process-managed child's resolved
    binding without overloading unrelated data-placement settings.
    ``KESTREL_DATA_DIR`` remains the intentional standalone legacy override.
    Without an override, exports follow the active agent's data root instead of
    falling into a process-CWD-relative shared directory.

    ``KESTREL_DB_PATH`` is retained as the direct single-agent fallback for
    callers that do not have an agent object.  The historical ``agent_data``
    default is used only when no runtime identity/storage binding exists.
    """

    environ = os.environ if env is None else env
    candidate = (
        per_agent_override
        or environ.get(IDENTITY_EXPORT_DIR_ENV)
        or environ.get("KESTREL_DATA_DIR")
        or agent_data_dir
        or environ.get("KESTREL_DB_PATH")
        or "agent_data"
    )
    return _absolute_path(candidate)


def configured_identity_export_roots(
    project_dir: Path | str,
    *,
    env: Mapping[str, str] | None = None,
    additional_roots: Iterable[Path | str] = (),
) -> tuple[Path, ...]:
    """Resolve and deduplicate roots authorized to hold identity exports.

    Relative configuration is resolved against ``project_dir`` so doctor and
    the CLI inspect the same deployment tree that runtime-relative paths name.
    ``AGENT_DATA_DIR`` remains a supported legacy configuration source.
    """

    base = _absolute_path(project_dir)
    environ = os.environ if env is None else env
    candidates: list[Path | str] = []
    candidates.append("agent_data")
    if environ.get(IDENTITY_EXPORT_DIR_ENV):
        candidates.append(environ[IDENTITY_EXPORT_DIR_ENV])
    if environ.get("KESTREL_DATA_DIR"):
        candidates.append(environ["KESTREL_DATA_DIR"])
    if environ.get("AGENT_DATA_DIR"):
        candidates.append(environ["AGENT_DATA_DIR"])
    candidates.extend(_configured_agent_export_roots(base))
    candidates.extend(additional_roots)

    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = base / path
        absolute = _absolute_path(path)
        key = os.path.normcase(os.fspath(absolute))
        if key not in seen:
            roots.append(absolute)
            seen.add(key)
    return tuple(roots)


def _configured_agent_export_roots(project_dir: Path) -> tuple[Path, ...]:
    """Return agent-bound roots declared by a valid multi-agent registry."""

    from kestrel_sovereign.multi_agent.config import (
        MULTI_AGENT_CONFIG_FILENAME,
        MultiAgentConfig,
    )

    config_path = project_dir / MULTI_AGENT_CONFIG_FILENAME
    if not config_path.is_file():
        return ()
    try:
        config = MultiAgentConfig.from_file(config_path)
    except (OSError, ValueError):
        # Doctor reports an invalid registry through its dedicated config
        # check. Custody remediation must not guess roots from malformed data.
        return ()

    roots: list[Path] = []
    for agent in config.get_local_agents().values():
        roots.append(agent.resolve_data_dir(project_dir))
        override = agent.resolve_identity_export_dir(project_dir)
        if override is not None:
            roots.append(override)
    return tuple(roots)


def write_protected_identity_export(
    destination: Path | str,
    payload: str | bytes,
    *,
    replace_existing: bool = False,
    allowed_destination_roots: Sequence[Path | str] = (),
    allowed_replacement_roots: Sequence[Path | str] = (),
) -> Path:
    """Write one private identity package and atomically publish it.

    New destinations are never clobbered.  ``replace_existing=True`` is an
    explicit compatibility seam for :func:`sign_and_export`; it accepts only a
    regular, operator-owned ``identity_*.json`` below one of
    ``allowed_replacement_roots``.  Replacement is staged and atomic, and the
    prior inode is retained as a private hard-link until the new directory
    entry is durable so a failed replace can be rolled back.
    """

    _require_secure_platform()
    output = _absolute_path(destination)
    if not output.name or output.name in {".", ".."}:
        raise IdentityExportSecurityError("identity export requires a file name")
    if output.parent == Path(output.anchor):
        raise IdentityExportSecurityError(
            "refusing to publish identity data directly in a filesystem root"
        )
    if allowed_destination_roots and not _path_within_roots(
        output, allowed_destination_roots
    ):
        raise IdentityExportSecurityError(
            "identity export destination is outside configured data roots"
        )
    if replace_existing:
        _validate_replacement_scope(output, allowed_replacement_roots)

    encoded = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    directory_fd = _open_private_directory(output.parent, create=True)
    temp_name = f"{_TEMP_PREFIX}{secrets.token_hex(16)}{_TEMP_SUFFIX}"
    temp_fd: int | None = None
    temp_present = False
    try:
        _assert_directory_binding(directory_fd, output.parent)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        temp_fd = os.open(temp_name, flags, 0o600, dir_fd=directory_fd)
        temp_present = True
        os.fchmod(temp_fd, 0o600)
        _write_payload(temp_fd, encoded)
        os.fsync(temp_fd)
        temp_stat = os.fstat(temp_fd)
        if not stat.S_ISREG(temp_stat.st_mode):
            raise IdentityExportSecurityError(
                "identity export staging inode is not a regular file"
            )
        if stat.S_IMODE(temp_stat.st_mode) != 0o600:
            raise IdentityExportSecurityError(
                "identity export staging inode is not mode 0600"
            )
        os.close(temp_fd)
        temp_fd = None

        if replace_existing:
            _replace_existing_export(
                directory_fd,
                temp_name,
                output,
                temp_stat,
            )
            temp_present = False
        else:
            _publish_new_export(
                directory_fd,
                temp_name,
                output.name,
                output.parent,
                temp_stat,
            )
            temp_present = False
        return output
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        if temp_present:
            _unlink_if_present(temp_name, directory_fd)
        os.close(directory_fd)


def audit_legacy_identity_exports(
    roots: Iterable[Path | str],
) -> list[LegacyIdentityExportFinding]:
    """Inspect legacy export metadata without opening package contents."""

    findings: list[LegacyIdentityExportFinding] = []
    for configured_root in roots:
        root = _absolute_path(configured_root)
        try:
            directory_fd = _open_existing_directory(root)
        except FileNotFoundError:
            continue
        except (OSError, IdentityExportSecurityError) as exc:
            findings.append(
                LegacyIdentityExportFinding(root, None, f"unsafe export root: {exc}")
            )
            continue
        try:
            try:
                names = _identity_export_names(directory_fd)
            except OSError as exc:
                findings.append(
                    LegacyIdentityExportFinding(
                        root,
                        None,
                        f"export metadata scan unavailable: {exc}",
                    )
                )
                continue
            if not names:
                continue
            root_stat = os.fstat(directory_fd)
            if not _owned_by_operator(root_stat):
                findings.append(
                    LegacyIdentityExportFinding(
                        root,
                        None,
                        "export root is not owned by the current operator",
                    )
                )
            if stat.S_IMODE(root_stat.st_mode) != 0o700:
                findings.append(
                    LegacyIdentityExportFinding(root, None, "export root is not mode 0700")
                )
            for name in names:
                try:
                    entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError as exc:
                    findings.append(
                        LegacyIdentityExportFinding(root, name, f"metadata unavailable: {exc}")
                    )
                    continue
                if stat.S_ISLNK(entry.st_mode):
                    reason = "entry is a symbolic link"
                elif not stat.S_ISREG(entry.st_mode):
                    reason = "entry is not a regular file"
                elif not _owned_by_operator(entry):
                    reason = "entry is not owned by the current operator"
                elif stat.S_IMODE(entry.st_mode) != 0o600:
                    reason = "entry is not mode 0600"
                else:
                    continue
                findings.append(LegacyIdentityExportFinding(root, name, reason))
        finally:
            os.close(directory_fd)
    return findings


def harden_legacy_identity_exports(
    roots: Iterable[Path | str],
) -> LegacyIdentityExportHardeningResult:
    """Privatize eligible legacy exports without reading their contents.

    Only direct ``identity_*.json`` children of the supplied configured roots
    are eligible.  Links, non-regular entries, foreign-owned roots/files, and
    entries that change identity during inspection are refused.
    """

    _require_secure_platform()
    hardened = already_private = refused = missing_roots = 0
    for configured_root in roots:
        root = _absolute_path(configured_root)
        try:
            directory_fd = _open_existing_directory(root)
        except FileNotFoundError:
            missing_roots += 1
            continue
        except (OSError, IdentityExportSecurityError):
            refused += 1
            continue
        try:
            root_stat = os.fstat(directory_fd)
            if not _owned_by_operator(root_stat):
                refused += 1
                continue
            try:
                names = _identity_export_names(directory_fd)
            except OSError:
                refused += 1
                continue
            if not names:
                continue
            try:
                os.fchmod(directory_fd, 0o700)
            except OSError:
                refused += 1
                continue
            for name in names:
                try:
                    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if not stat.S_ISREG(before.st_mode) or not _owned_by_operator(before):
                        refused += 1
                        continue
                    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
                    descriptor = os.open(name, flags, dir_fd=directory_fd)
                except OSError:
                    refused += 1
                    continue
                try:
                    opened = os.fstat(descriptor)
                    if not _same_identity(before, opened) or not _owned_by_operator(opened):
                        refused += 1
                        continue
                    if stat.S_IMODE(opened.st_mode) == 0o600:
                        already_private += 1
                        continue
                    os.fchmod(descriptor, 0o600)
                    os.fsync(descriptor)
                    hardened += 1
                except OSError:
                    refused += 1
                finally:
                    os.close(descriptor)
            try:
                os.fsync(directory_fd)
            except OSError:
                # Every eligible inode was fsynced immediately after chmod.
                # Directory fsync support varies, and a failure here cannot
                # safely roll those metadata restrictions back.
                pass
        finally:
            os.close(directory_fd)
    return LegacyIdentityExportHardeningResult(
        hardened=hardened,
        already_private=already_private,
        refused=refused,
        missing_roots=missing_roots,
    )


def _absolute_path(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _require_secure_platform() -> None:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if os.name != "posix" or any(not hasattr(os, name) for name in required):
        raise IdentityExportSecurityError(
            "secure identity export requires POSIX no-follow directory semantics"
        )
    for function in (os.mkdir, os.open, os.stat, os.unlink, os.link):
        if function not in os.supports_dir_fd:
            raise IdentityExportSecurityError(
                "secure identity export requires descriptor-relative filesystem operations"
            )


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_existing_directory(path: Path) -> int:
    _require_secure_platform()
    return _open_directory_chain(path, create=False)


def _open_private_directory(path: Path, *, create: bool) -> int:
    if path == Path(path.anchor):
        raise IdentityExportSecurityError(
            "refusing to use a filesystem root as an identity export directory"
        )
    system_temp = _absolute_path(tempfile.gettempdir())
    if path == system_temp:
        raise IdentityExportSecurityError(
            "refusing to privatize the shared system temporary directory"
        )
    descriptor = _open_directory_chain(path, create=create)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise IdentityExportSecurityError("identity export root is not a directory")
        if not _owned_by_operator(metadata):
            raise IdentityExportSecurityError(
                "identity export root is not owned by the current operator"
            )
        os.fchmod(descriptor, 0o700)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o700:
            raise IdentityExportSecurityError(
                "identity export root could not be made mode 0700"
            )
        _assert_directory_binding(descriptor, path)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_chain(path: Path, *, create: bool) -> int:
    absolute = _absolute_path(path)
    descriptor = os.open(absolute.anchor, _directory_flags())
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _assert_directory_binding(descriptor: int, path: Path) -> None:
    try:
        lexical = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise IdentityExportSecurityError(
            "identity export directory changed during publication"
        ) from exc
    if not stat.S_ISDIR(lexical.st_mode) or not _same_identity(
        lexical, os.fstat(descriptor)
    ):
        raise IdentityExportSecurityError(
            "identity export directory changed during publication"
        )


def _write_payload(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("identity export write made no progress")
        view = view[written:]


def _publish_new_export(
    directory_fd: int,
    temp_name: str,
    final_name: str,
    directory_path: Path,
    expected: os.stat_result,
) -> None:
    linked = False
    try:
        os.link(
            temp_name,
            final_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        linked = True
        os.unlink(temp_name, dir_fd=directory_fd)
        _validate_published_entry(
            directory_fd,
            directory_path,
            final_name,
            expected,
        )
        os.fsync(directory_fd)
    except BaseException:
        if linked:
            _unlink_if_same_identity(final_name, directory_fd, expected)
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
        raise


def _replace_existing_export(
    directory_fd: int,
    temp_name: str,
    output: Path,
    expected: os.stat_result,
) -> None:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        before = os.stat(output.name, dir_fd=directory_fd, follow_symlinks=False)
        existing_fd = os.open(output.name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        _publish_new_export(
            directory_fd,
            temp_name,
            output.name,
            output.parent,
            expected,
        )
        return
    except OSError as exc:
        raise IdentityExportSecurityError(
            "existing identity destination is unsafe"
        ) from exc

    backup_name = f"{_TEMP_PREFIX}{secrets.token_hex(16)}.rollback"
    backup_present = False
    replaced = False
    try:
        opened = os.fstat(existing_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or not _same_identity(before, opened)
            or not _owned_by_operator(opened)
        ):
            raise IdentityExportSecurityError(
                "existing identity destination is not an operator-owned regular file"
            )
        os.fchmod(existing_fd, 0o600)
        os.fsync(existing_fd)
        current = os.stat(output.name, dir_fd=directory_fd, follow_symlinks=False)
        if not _same_identity(opened, current):
            raise IdentityExportSecurityError(
                "existing identity destination changed before replacement"
            )
        os.link(
            output.name,
            backup_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        backup_present = True
        os.replace(
            temp_name,
            output.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        replaced = True
        _validate_published_entry(
            directory_fd,
            output.parent,
            output.name,
            expected,
        )
        os.fsync(directory_fd)
        os.unlink(backup_name, dir_fd=directory_fd)
        backup_present = False
        try:
            # The new final entry was already fsynced while the private rollback
            # link existed. A failure to persist only the rollback-link removal
            # must not turn a successful publication into a reported failure
            # after rollback has become impossible.
            os.fsync(directory_fd)
        except OSError:
            pass
    except BaseException:
        if replaced and backup_present:
            try:
                os.replace(
                    backup_name,
                    output.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
                backup_present = False
                os.fsync(directory_fd)
            except OSError as rollback_error:
                raise IdentityExportSecurityError(
                    "identity replacement failed and its private rollback "
                    "entry could not be restored"
                ) from rollback_error
        raise
    finally:
        os.close(existing_fd)
        if backup_present and not replaced:
            _unlink_if_present(backup_name, directory_fd)


def _validate_replacement_scope(
    output: Path,
    allowed_roots: Sequence[Path | str],
) -> None:
    if not fnmatch.fnmatchcase(output.name, IDENTITY_EXPORT_PATTERN):
        raise IdentityExportSecurityError(
            "existing identity replacement requires an identity_*.json destination"
        )
    if not _path_within_roots(output, allowed_roots):
        raise IdentityExportSecurityError(
            "existing identity replacement is outside configured data roots"
        )


def _validate_published_entry(
    directory_fd: int,
    directory_path: Path,
    final_name: str,
    expected: os.stat_result,
) -> None:
    _assert_directory_binding(directory_fd, directory_path)
    final_stat = os.stat(final_name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(final_stat.st_mode):
        raise IdentityExportSecurityError(
            "published identity export is not a regular file"
        )
    if not _same_identity(final_stat, expected) or not _owned_by_operator(final_stat):
        raise IdentityExportSecurityError(
            "published identity export changed during publication"
        )
    if stat.S_IMODE(final_stat.st_mode) != 0o600:
        raise IdentityExportSecurityError(
            "published identity export is not mode 0600"
        )


def _path_within_roots(path: Path, roots: Sequence[Path | str]) -> bool:
    for root_value in roots:
        root = _absolute_path(root_value)
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if relative.parts:
            return True
    return False


def _identity_export_names(directory_fd: int) -> list[str]:
    return sorted(
        name
        for name in os.listdir(directory_fd)
        if fnmatch.fnmatchcase(name, IDENTITY_EXPORT_PATTERN)
    )


def _owned_by_operator(metadata: os.stat_result) -> bool:
    get_uid = getattr(os, "geteuid", None) or getattr(os, "getuid", None)
    return get_uid is None or metadata.st_uid == get_uid()


def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _unlink_if_present(name: str, directory_fd: int) -> None:
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        pass


def _unlink_if_same_identity(
    name: str,
    directory_fd: int,
    expected: os.stat_result,
) -> None:
    """Remove our failed publication without deleting a raced-in entry."""

    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _same_identity(current, expected):
            os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        pass


__all__ = [
    "IDENTITY_EXPORT_PATTERN",
    "IdentityExportSecurityError",
    "IDENTITY_EXPORT_DIR_ENV",
    "LegacyIdentityExportFinding",
    "LegacyIdentityExportHardeningResult",
    "audit_legacy_identity_exports",
    "configured_identity_export_roots",
    "harden_legacy_identity_exports",
    "identity_export_directory",
    "write_protected_identity_export",
]
